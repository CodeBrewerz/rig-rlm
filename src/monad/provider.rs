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
    /// response text.
    pub async fn chat(&self, history: &ConversationHistory) -> Result<String> {
        use rig::client::CompletionClient;
        use rig::completion::Chat;

        let client = rig::providers::openai::Client::<reqwest::Client>::builder()
            .base_url(&self.config.base_url)
            .api_key(&self.config.api_key)
            .http_client(reqwest::Client::new())
            .build()
            .map_err(|e| AgentError::Inference(e.to_string()))?;

        let mut builder = client
            .completion_model(&self.config.model)
            .completions_api()
            .into_agent_builder();

        if let Some(ref preamble) = self.config.preamble {
            builder = builder.preamble(preamble);
        }

        let agent = builder.build();

        let (prompt, chat_history) = history.to_rig_prompt();

        let response: String = agent
            .chat(prompt, chat_history)
            .await
            .map_err(|e: rig::completion::PromptError| {
                AgentError::Inference(e.to_string())
            })?;

        Ok(response)
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
        self.chat(&history).await
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
