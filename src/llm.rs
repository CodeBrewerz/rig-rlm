//! The LLM module — λ-RLM entry point.
//!
//! Replaces the open-ended REPL loop (standard RLM) with the λ-RLM
//! typed functional runtime. The base model M is invoked only at
//! bounded leaf subproblems; all control flow is symbolic and
//! deterministic.
//!
//! See: "The Y-Combinator for LLMs" (Roy et al., arXiv:2603.20105v1)

use std::sync::Arc;

use crate::lambda::{self, LambdaConfig};
use crate::monad::provider::{LlmProvider, ProviderConfig};

/// λ-RLM agent: long-context reasoning via typed combinators.
///
/// Replaces the old `RigRlm` which used an open-ended REPL loop
/// where the LLM generated arbitrary code. λ-RLM instead:
///
/// 1. Detects the task type (1 LLM call)
/// 2. Computes optimal partition (k*=2, pure math)
/// 3. Executes a pre-built combinator chain Φ (no code generation)
///
/// Result: +21.9pp accuracy, 3.3–4.1× latency reduction.
pub struct RigRlm {
    provider: Arc<LlmProvider>,
    config: LambdaConfig,
}

impl RigRlm {
    /// Create with a local LM Studio model.
    pub fn new_local() -> Self {
        let provider = LlmProvider::new(ProviderConfig::local("qwen/qwen3-8b"));
        Self {
            provider: Arc::new(provider),
            config: LambdaConfig::default(),
        }
    }

    /// Create with OpenAI.
    pub fn new() -> Self {
        let provider = LlmProvider::new(
            ProviderConfig::openai("gpt-5.2", std::env::var("OPENAI_API_KEY").unwrap_or_default())
                .with_preamble(PREAMBLE),
        );
        Self {
            provider: Arc::new(provider),
            config: LambdaConfig::default().with_context_window(128_000),
        }
    }

    /// Create with a specific provider and config.
    pub fn with_provider(provider: LlmProvider, config: LambdaConfig) -> Self {
        Self {
            provider: Arc::new(provider),
            config,
        }
    }

    /// Query using λ-RLM.
    ///
    /// This replaces the old open-ended REPL loop. The execution path is:
    ///
    /// ```text
    /// λ-RLM ≡ fix λf. λP. if |P| ≤ τ* then M(P)
    ///          else REDUCE(⊕, MAP(λpi. f(pi), SPLIT(P, k*)))
    /// ```
    ///
    /// - If the prompt fits in the context window → single M call
    /// - Otherwise → recursive decomposition with bounded depth d
    pub async fn query(&self, input: &str) -> Result<String, Box<dyn std::error::Error>> {
        // The prompt is stored externally (prompt-as-environment).
        // The query is the user's question; the input is the full context.
        let result = lambda::lambda_rlm(
            input,
            input, // user_query = input for simple queries
            self.provider.clone(),
            self.config.clone(),
        )
        .await
        .map_err(|e| -> Box<dyn std::error::Error> { Box::new(e) })?;

        Ok(result.trim().to_string())
    }

    /// Query with separate context and question.
    ///
    /// Use this when you have a long document (context) and a
    /// short question about it. The context is the prompt P that
    /// gets decomposed; the question is used in leaf templates.
    pub async fn query_with_context(
        &self,
        context: &str,
        question: &str,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let result = lambda::lambda_rlm(
            context,
            question,
            self.provider.clone(),
            self.config.clone(),
        )
        .await
        .map_err(|e| -> Box<dyn std::error::Error> { Box::new(e) })?;

        Ok(result.trim().to_string())
    }
}

/// System preamble for leaf sub-calls.
///
/// This is injected into the LLM provider's system prompt.
/// It's much simpler than the old RLM preamble because the LLM
/// no longer needs to generate code or manage recursion — it just
/// answers bounded sub-questions.
const PREAMBLE: &str = r#"You are a precise and thorough assistant. You receive a specific sub-task as part of a larger analysis pipeline. Focus on answering the sub-task accurately using only the provided text. Be concise and factual."#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rig_rlm_creation() {
        // Just verify construction doesn't panic
        let _rlm = RigRlm::new_local();
    }
}
