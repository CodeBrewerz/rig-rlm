//! λ-RLM: Typed functional runtime for long-context reasoning.
//!
//! Replaces the open-ended REPL loop (standard RLM) with a deterministic
//! combinator chain grounded in λ-calculus. The base model M is invoked
//! ONLY at bounded leaf subproblems; all control flow is symbolic.
//!
//! **Paper**: "The Y-Combinator for LLMs: Solving Long-Context Rot with
//! λ-Calculus" — Roy et al., arXiv:2603.20105v1 (Mar 2026)
//!
//! **Key results**: +21.9pp accuracy, 3.3–4.1× latency reduction, wins
//! 29/36 model-task comparisons vs standard RLM.
//!
//! # Architecture
//!
//! ```text
//! Phase 1: REPL Init          — store P externally, register combinators
//! Phase 2: Task Detection     — 1 LLM call (menu selection)
//! Phase 3: Planning           — 0 LLM calls (pure math: k*, τ*, d)
//! Phase 4: Cost Estimation    — deterministic, pre-execution
//! Phase 5: Execute Φ(P)       — single recursive execution
//! ```

pub mod combinators;
pub mod executor;
pub mod planner;
pub mod templates;
pub mod yoneda;
pub mod effects;

#[cfg(test)]

mod live_tests;

use std::sync::Arc;

use crate::monad::provider::LlmProvider;

pub use executor::{ExecutionMetrics, LambdaExecutor};
pub use planner::{CostParams, ExecutionPlan, TaskType};
pub use yoneda::YonedaContext;

// ─── Configuration ──────────────────────────────────────────────────

/// Configuration for the λ-RLM runtime.
#[derive(Debug, Clone)]
pub struct LambdaConfig {
    /// Model context window K (in tokens).
    pub context_window: usize,
    /// Accuracy target α ∈ (0, 1].
    pub accuracy_target: f64,
    /// Cost parameters for optimal k* computation.
    pub cost_params: CostParams,
}

impl Default for LambdaConfig {
    fn default() -> Self {
        Self {
            context_window: 32_000,  // 32K default (Qwen3-8B)
            accuracy_target: 0.80,   // α = 0.80 (paper default)
            cost_params: CostParams::default(),
        }
    }
}

impl LambdaConfig {
    /// Create config for a specific context window size.
    pub fn with_context_window(mut self, k: usize) -> Self {
        self.context_window = k;
        self
    }

    /// Set accuracy target.
    pub fn with_accuracy_target(mut self, alpha: f64) -> Self {
        self.accuracy_target = alpha.clamp(0.01, 1.0);
        self
    }
}

// ─── Top-Level Entry Point (Algorithm 1) ────────────────────────────

/// The complete λ-RLM system — Algorithm 1.
///
/// Replaces `RigRlm::query()` as the primary long-context reasoning path.
/// Executes 5 deterministic phases:
///
/// 1. Peek at prompt preview (symbolic, free)
/// 2. Detect task type (1 LLM call)
/// 3. Compute optimal plan (0 LLM calls, pure math)
/// 4. Estimate cost (deterministic)
/// 5. Execute pre-built Φ (single recursive run)
///
/// The open-ended REPL while-loop is eliminated entirely.
/// Recursion is bounded: depth d = ⌈log_k*(n/τ*)⌉.
pub async fn lambda_rlm(
    prompt: &str,
    user_query: &str,
    provider: Arc<LlmProvider>,
    config: LambdaConfig,
) -> crate::monad::error::Result<String> {
    let n = combinators::token_count(prompt);

    eprintln!("⚡ [λ-RLM] Starting — {} tokens, K={}", n, config.context_window);

    // ── Phase 2: Task Detection (1 LLM call) ──
    let preview = combinators::peek(prompt, 0, 500);
    let task_type = planner::detect_task_type(&preview, n, &provider).await?;

    eprintln!("⚡ [λ-RLM] Task detected: {task_type}");

    // ── Phase 3: Short-circuit if prompt fits in context window ──
    if n <= config.context_window {
        eprintln!("⚡ [λ-RLM] Prompt fits in window, direct call");
        return provider.complete(prompt).await;
    }

    // ── Phase 4: Optimal Planning (0 LLM calls) ──
    let plan = planner::plan(
        n,
        config.context_window,
        config.accuracy_target,
        task_type,
        &config.cost_params,
    );

    eprintln!("⚡ [λ-RLM] {}", plan.summary());

    // ── Phase 5: Build and Execute Φ ──
    let executor = LambdaExecutor::new(plan, provider, user_query.to_string());
    let result = executor.execute(prompt).await?;

    eprintln!("⚡ [λ-RLM] Complete — {} chars output", result.len());

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lambda_config_defaults() {
        let config = LambdaConfig::default();
        assert_eq!(config.context_window, 32_000);
        assert!((config.accuracy_target - 0.80).abs() < f64::EPSILON);
    }

    #[test]
    fn test_lambda_config_builder() {
        let config = LambdaConfig::default()
            .with_context_window(128_000)
            .with_accuracy_target(0.90);
        assert_eq!(config.context_window, 128_000);
        assert!((config.accuracy_target - 0.90).abs() < f64::EPSILON);
    }

    #[test]
    fn test_accuracy_target_clamped() {
        let config = LambdaConfig::default().with_accuracy_target(2.0);
        assert!((config.accuracy_target - 1.0).abs() < f64::EPSILON);

        let config = LambdaConfig::default().with_accuracy_target(-0.5);
        assert!((config.accuracy_target - 0.01).abs() < f64::EPSILON);
    }
}
