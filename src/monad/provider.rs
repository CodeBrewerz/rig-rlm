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
        }
    }

    /// Set the system preamble.
    pub fn with_preamble(mut self, preamble: impl Into<String>) -> Self {
        self.preamble = Some(preamble.into());
        self
    }
}

/// Provider-agnostic LLM interface.
///
/// Wraps rig's Agent behind a simple chat interface. The provider
/// is selected at construction time via `ProviderConfig`.
pub struct LlmProvider {
    config: ProviderConfig,
}

impl LlmProvider {
    pub fn new(config: ProviderConfig) -> Self {
        Self { config }
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
            .http_client(reqwest::Client::new())
            .build()
            .map_err(|e| AgentError::Inference(e.to_string()))?;

        let model = client
            .completion_model(&self.config.model)
            .completions_api();

        let (prompt, chat_history) = history.to_rig_prompt();
        let prompt_snapshot = prompt.clone();

        // Build completion request with chat history
        let mut req_builder = model.completion_request(prompt).messages(chat_history);
        if let Some(ref preamble) = self.config.preamble {
            req_builder = req_builder.preamble(preamble.clone());
        }
        let request = req_builder.build();

        // Use low-level completion API to get token usage
        let completion_result = model
            .completion(request)
            .await
            .map_err(|e: rig::completion::CompletionError| AgentError::Inference(e.to_string()));

        let duration = start.elapsed();
        let success = completion_result.is_ok();

        // Extract response text and usage
        let (response_text, usage): (Result<String>, _) = match completion_result {
            Ok(resp) => {
                // Extract text from the first assistant content choice
                use rig::message::AssistantContent;
                let text = match resp.choice.first() {
                    AssistantContent::Text(t) => t.text.clone(),
                    _ => String::new(),
                };
                let usage = super::otel::TokenUsage {
                    input_tokens: Some(resp.usage.input_tokens as i64),
                    output_tokens: Some(resp.usage.output_tokens as i64),
                };
                (Ok(text), usage)
            }
            Err(e) => (Err(e), super::otel::TokenUsage::default()),
        };

        eprintln!(
            "🔗 [provider] chat() ← {:.1}s, success={success}, tokens={}in/{}out",
            duration.as_secs_f64(),
            usage.input_tokens.unwrap_or(0),
            usage.output_tokens.unwrap_or(0),
        );

        // Record OTEL span with GenAI + LangFuse attributes (including token usage)
        let output_str = response_text.as_ref().ok().map(|s| s.as_str());
        super::otel::record_llm_span(
            "chat",
            &self.config.name,
            &self.config.model,
            &usage,
            duration,
            success,
            trace_ctx,
            Some(&prompt_snapshot),
            output_str,
        );

        response_text.map(|text| (text, usage))
    }

    /// Single-prompt completion — convenience for sub-LLM bridging.
    ///
    /// Used by `llm_query()` in the sandbox: generated code calls
    /// `llm_query("summarize this")` → routes here → returns LLM text.
    pub async fn complete(&self, prompt: &str) -> Result<String> {
        let mut history = ConversationHistory::new();
        history.push(super::history::HistoryMessage {
            role: super::action::Role::User,
            content: prompt.to_string(),
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
}
