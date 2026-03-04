//! Phase 8: Provider abstraction.
//!
//! Abstracts over LLM providers (OpenAI, Anthropic, local models)
//! with config-driven routing. Uses rig under the hood.

use serde::{Deserialize, Serialize};

use super::error::{AgentError, Result};
use super::history::ConversationHistory;

/// LLM provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Provider name (e.g. "openai", "anthropic", "local").
    pub name: String,
    /// Model identifier (e.g. "gpt-4o", "claude-sonnet-4-20250514", "qwen/qwen3-8b").
    pub model: String,
    /// API base URL.
    pub base_url: String,
    /// API key (can be empty for local models).
    pub api_key: String,
    /// System prompt / preamble.
    pub preamble: Option<String>,
    /// Fallback model to try if primary model retries are exhausted.
    /// Uses same provider/base_url/api_key.
    pub fallback_model: Option<String>,
    /// Max retry attempts for transient errors (429, 500, 502, 503). Default: 3.
    pub max_retries: usize,
}

impl ProviderConfig {
    /// Create config for a local LM Studio model.
    pub fn local(model: impl Into<String>) -> Self {
        Self {
            name: "local".to_string(),
            model: model.into(),
            base_url: "http://127.0.0.1:1234/v1".to_string(),
            api_key: String::new(),
            preamble: None,
            fallback_model: None,
            max_retries: 3,
        }
    }

    /// Create config for OpenAI.
    pub fn openai(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            name: "openai".to_string(),
            model: model.into(),
            base_url: "https://api.openai.com/v1".to_string(),
            api_key: api_key.into(),
            preamble: None,
            fallback_model: None,
            max_retries: 3,
        }
    }

    /// Create config for any OpenAI-compatible endpoint.
    pub fn openai_compatible(
        name: impl Into<String>,
        model: impl Into<String>,
        base_url: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            model: model.into(),
            base_url: base_url.into(),
            api_key: api_key.into(),
            preamble: None,
            fallback_model: None,
            max_retries: 3,
        }
    }

    /// Set the system preamble.
    pub fn with_preamble(mut self, preamble: impl Into<String>) -> Self {
        self.preamble = Some(preamble.into());
        self
    }

    /// Set a fallback model to try if primary retries are exhausted.
    pub fn with_fallback(mut self, model: impl Into<String>) -> Self {
        self.fallback_model = Some(model.into());
        self
    }

    /// Set max retry attempts for transient errors.
    pub fn with_max_retries(mut self, n: usize) -> Self {
        self.max_retries = n;
        self
    }
}

/// Provider-agnostic LLM interface.
///
/// Wraps rig's Agent behind a simple chat interface. The provider
/// is selected at construction time via `ProviderConfig`.
pub struct LlmProvider {
    config: ProviderConfig,
    /// Shared HTTP client — reuses TCP connection pool across calls.
    http: reqwest::Client,
}

impl LlmProvider {
    pub fn new(config: ProviderConfig) -> Self {
        Self {
            config,
            http: reqwest::Client::new(),
        }
    }

    /// Send a chat completion request.
    ///
    /// Takes the current conversation history and returns the model's
    /// response text. Records an OTEL span with GenAI semantic conventions.
    pub async fn chat(
        &self,
        history: &ConversationHistory,
        trace_ctx: &super::otel::TraceContext,
    ) -> Result<(String, super::otel::TokenUsage)> {
        use rig::client::CompletionClient;
        use rig::completion::CompletionModel as _;

        let start = std::time::Instant::now();
        eprintln!(
            "🔗 [provider] chat() → calling {}/{}",
            self.config.name, self.config.model
        );

        let client = rig::providers::openai::Client::<reqwest::Client>::builder()
            .base_url(&self.config.base_url)
            .api_key(&self.config.api_key)
            .http_client(self.http.clone())
            .build()
            .map_err(|e| AgentError::Inference(e.to_string()))?;

        let (prompt, chat_history) = history.to_rig_prompt();
        let prompt_snapshot = prompt.clone();

        // ── Retry loop with exponential backoff ──────────────────────
        // Retries on transient errors: 429, 500, 502, 503, timeouts.
        // Non-retryable errors (400, 401, 403) fail immediately.
        let max_attempts = self.config.max_retries.max(1);
        let models_to_try: Vec<&str> = {
            let mut m = vec![self.config.model.as_str()];
            if let Some(ref fallback) = self.config.fallback_model {
                m.push(fallback.as_str());
            }
            m
        };

        let mut last_err = None;

        for model_id in models_to_try {
            let model = client.completion_model(model_id).completions_api();

            for attempt in 0..max_attempts {
                // Build request fresh each attempt
                let mut req_builder = model
                    .completion_request(prompt.clone())
                    .messages(chat_history.clone());
                if let Some(ref preamble) = self.config.preamble {
                    req_builder = req_builder.preamble(preamble.clone());
                }
                let request = req_builder.build();

                match model.completion(request).await {
                    Ok(resp) => {
                        let duration = start.elapsed();
                        use rig::message::AssistantContent;
                        let text = match resp.choice.first() {
                            AssistantContent::Text(t) => t.text.clone(),
                            _ => String::new(),
                        };
                        let usage = super::otel::TokenUsage {
                            input_tokens: Some(resp.usage.input_tokens as i64),
                            output_tokens: Some(resp.usage.output_tokens as i64),
                        };

                        eprintln!(
                            "🔗 [provider] chat() ← {:.1}s, success=true, tokens={}in/{}out{}",
                            duration.as_secs_f64(),
                            usage.input_tokens.unwrap_or(0),
                            usage.output_tokens.unwrap_or(0),
                            if attempt > 0 {
                                format!(" (retry #{attempt})")
                            } else {
                                String::new()
                            },
                        );

                        super::otel::record_llm_span(
                            "chat",
                            &self.config.name,
                            model_id,
                            &usage,
                            duration,
                            true,
                            trace_ctx,
                            Some(&prompt_snapshot),
                            Some(&text),
                        );

                        return Ok((text, usage));
                    }
                    Err(e) => {
                        let err_str = e.to_string();
                        let is_retryable = err_str.contains("429")
                            || err_str.contains("500")
                            || err_str.contains("502")
                            || err_str.contains("503")
                            || err_str.contains("timeout")
                            || err_str.contains("connection")
                            || err_str.contains("rate limit");

                        if !is_retryable || attempt + 1 >= max_attempts {
                            eprintln!(
                                "🔗 [provider] chat() ← {:.1}s, FAILED (attempt {}/{max_attempts}, model={model_id}): {err_str}",
                                start.elapsed().as_secs_f64(),
                                attempt + 1,
                            );
                            last_err = Some(AgentError::Inference(err_str));
                            break; // try fallback model
                        }

                        // Exponential backoff: 1s, 2s, 4s, ...
                        let backoff = std::time::Duration::from_secs(1 << attempt);
                        eprintln!(
                            "🔗 [provider] chat() ⚠ attempt {}/{max_attempts} failed (model={model_id}): {err_str}, retrying in {:.0}s",
                            attempt + 1,
                            backoff.as_secs_f64(),
                        );
                        tokio::time::sleep(backoff).await;
                    }
                }
            }
        }

        let duration = start.elapsed();
        super::otel::record_llm_span(
            "chat",
            &self.config.name,
            &self.config.model,
            &super::otel::TokenUsage::default(),
            duration,
            false,
            trace_ctx,
            Some(&prompt_snapshot),
            None,
        );

        Err(last_err.unwrap_or_else(|| AgentError::Inference("all retries exhausted".into())))
    }

    /// Single-prompt completion — convenience for sub-LLM bridging.
    ///
    /// Used by `llm_query()` in the sandbox: generated code calls
    /// `llm_query("summarize this")` → routes here → returns LLM text.
    pub async fn complete(&self, prompt: &str) -> Result<String> {
        let mut history = ConversationHistory::new();
        history.push(super::history::HistoryMessage {
            role: super::action::Role::User,
            content: std::borrow::Cow::Owned(prompt.to_string()),
            attachments: vec![],
        });
        self.chat(&history, &super::otel::TraceContext::new())
            .await
            .map(|(text, _usage)| text)
    }

    /// Get the model name.
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Get the provider name.
    pub fn provider_name(&self) -> &str {
        &self.config.name
    }

    /// Generate an embedding vector for the given text.
    ///
    /// Uses the OpenAI-compatible `/embeddings` endpoint (works with OpenRouter).
    /// Default model: `qwen/qwen3-embedding-8b` (4096 dimensions).
    ///
    /// Falls back to a deterministic hash-based pseudo-embedding if the API call fails.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        use serde_json::json;

        let embed_model = std::env::var("RLM_EMBEDDING_MODEL")
            .unwrap_or_else(|_| "qwen/qwen3-embedding-8b".to_string());
        let url = format!("{}/embeddings", self.config.base_url.trim_end_matches('/'));

        let body = json!({
            "model": embed_model,
            "input": text,
        });

        let response = self
            .http
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await;

        match response {
            Ok(resp) if resp.status().is_success() => {
                let json: serde_json::Value = resp
                    .json()
                    .await
                    .map_err(|e| AgentError::Internal(format!("embed parse: {e}")))?;

                // OpenAI format: data[0].embedding = [f32, ...]
                let embedding = json["data"][0]["embedding"]
                    .as_array()
                    .ok_or_else(|| {
                        AgentError::Internal("missing embedding in response".to_string())
                    })?
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect::<Vec<f32>>();

                Ok(embedding)
            }
            Ok(resp) => {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                eprintln!("⚠️ embed API returned {status}: {body}. Using hash fallback.");
                Ok(hash_embedding(text, HASH_EMBED_DIM))
            }
            Err(e) => {
                eprintln!("⚠️ embed API error: {e}. Using hash fallback.");
                Ok(hash_embedding(text, HASH_EMBED_DIM))
            }
        }
    }
}

/// Dimension for hash-based fallback embeddings.
const HASH_EMBED_DIM: usize = 4096;

/// Deterministic hash-based pseudo-embedding fallback.
///
/// NOT semantically meaningful — only provides basic lexical similarity.
/// Used when no embedding API is available.
fn hash_embedding(text: &str, dim: usize) -> Vec<f32> {
    use std::hash::{Hash, Hasher};

    let mut embedding = vec![0.0f32; dim];

    // Hash each word and distribute into the embedding dimensions
    for (i, word) in text.split_whitespace().enumerate() {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        word.to_lowercase().hash(&mut hasher);
        let hash = hasher.finish();

        // Use different bits of the hash to set multiple dims
        for offset in 0..4 {
            let idx = ((hash >> (offset * 16)) as usize + i) % dim;
            let val = ((hash >> (offset * 8)) & 0xFF) as f32 / 255.0 - 0.5;
            embedding[idx] += val;
        }
    }

    // L2-normalize
    let norm: f32 = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut embedding {
            *v /= norm;
        }
    }

    embedding
}
