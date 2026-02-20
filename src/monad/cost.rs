//! Per-call and per-session cost tracking with budget limits.
//!
//! Tracks token usage and calculates costs using a model pricing table.
//! Integrates with `AgentContext` for budget enforcement and with
//! Turso for persistence.

use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::LazyLock;

/// Cost per 1 million tokens (USD).
#[derive(Debug, Clone, Copy)]
pub struct ModelPricing {
    /// USD per 1M input tokens.
    pub input_per_1m: f64,
    /// USD per 1M output tokens.
    pub output_per_1m: f64,
}

impl ModelPricing {
    pub const fn new(input_per_1m: f64, output_per_1m: f64) -> Self {
        Self {
            input_per_1m,
            output_per_1m,
        }
    }

    /// Calculate cost for a given number of input/output tokens.
    pub fn cost(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        (input_tokens as f64 * self.input_per_1m / 1_000_000.0)
            + (output_tokens as f64 * self.output_per_1m / 1_000_000.0)
    }
}

/// Known model pricing (USD per 1M tokens).
///
/// Sources: OpenRouter pricing page, OpenAI/Anthropic pricing pages.
/// Free models have $0 pricing. Unknown models default to $0.
static PRICING_TABLE: LazyLock<HashMap<&'static str, ModelPricing>> = LazyLock::new(|| {
    let mut m = HashMap::new();

    // ── OpenRouter free models ──────────────────────────────────
    m.insert(
        "arcee-ai/trinity-large-preview:free",
        ModelPricing::new(0.0, 0.0),
    );
    m.insert(
        "meta-llama/llama-3.1-8b-instruct:free",
        ModelPricing::new(0.0, 0.0),
    );
    m.insert("google/gemma-2-9b-it:free", ModelPricing::new(0.0, 0.0));
    m.insert(
        "mistralai/mistral-7b-instruct:free",
        ModelPricing::new(0.0, 0.0),
    );

    // ── OpenAI ──────────────────────────────────────────────────
    m.insert("gpt-4o", ModelPricing::new(2.50, 10.00));
    m.insert("gpt-4o-mini", ModelPricing::new(0.15, 0.60));
    m.insert("gpt-4-turbo", ModelPricing::new(10.00, 30.00));
    m.insert("gpt-4", ModelPricing::new(30.00, 60.00));
    m.insert("gpt-3.5-turbo", ModelPricing::new(0.50, 1.50));
    m.insert("o1", ModelPricing::new(15.00, 60.00));
    m.insert("o1-mini", ModelPricing::new(3.00, 12.00));
    m.insert("o3-mini", ModelPricing::new(1.10, 4.40));

    // ── Anthropic ───────────────────────────────────────────────
    m.insert("claude-sonnet-4-20250514", ModelPricing::new(3.00, 15.00));
    m.insert("claude-3-5-sonnet-20241022", ModelPricing::new(3.00, 15.00));
    m.insert("claude-3-opus-20240229", ModelPricing::new(15.00, 75.00));
    m.insert("claude-3-haiku-20240307", ModelPricing::new(0.25, 1.25));

    // ── Google ──────────────────────────────────────────────────
    m.insert("gemini-2.0-flash", ModelPricing::new(0.10, 0.40));
    m.insert("gemini-1.5-pro", ModelPricing::new(1.25, 5.00));
    m.insert("gemini-1.5-flash", ModelPricing::new(0.075, 0.30));

    // ── OpenRouter paid models ──────────────────────────────────
    m.insert(
        "meta-llama/llama-3.1-70b-instruct",
        ModelPricing::new(0.52, 0.75),
    );
    m.insert(
        "meta-llama/llama-3.1-405b-instruct",
        ModelPricing::new(2.70, 2.70),
    );
    m.insert("deepseek/deepseek-r1", ModelPricing::new(0.55, 2.19));
    m.insert("qwen/qwen-2.5-72b-instruct", ModelPricing::new(0.36, 0.40));

    m
});

/// Look up pricing for a model. Returns zero-cost pricing for unknown models.
pub fn get_pricing(model: &str) -> ModelPricing {
    PRICING_TABLE
        .get(model)
        .copied()
        .unwrap_or(ModelPricing::new(0.0, 0.0))
}

/// Cost record for a single LLM call.
#[derive(Debug, Clone)]
pub struct CallCost {
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_usd: f64,
    pub timestamp: DateTime<Utc>,
}

/// Cumulative cost tracker for an agent session.
///
/// Records every LLM call's cost and enforces budget limits.
#[derive(Debug, Clone, Default)]
pub struct CostTracker {
    /// Total input tokens across all calls.
    pub total_input_tokens: u64,
    /// Total output tokens across all calls.
    pub total_output_tokens: u64,
    /// Total cost in USD.
    pub total_cost_usd: f64,
    /// Per-call breakdown.
    pub calls: Vec<CallCost>,
}

impl CostTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a completed LLM call and update cumulative totals.
    pub fn record(&mut self, model: &str, input_tokens: u64, output_tokens: u64) -> f64 {
        let pricing = get_pricing(model);
        let cost = pricing.cost(input_tokens, output_tokens);

        self.total_input_tokens += input_tokens;
        self.total_output_tokens += output_tokens;
        self.total_cost_usd += cost;

        self.calls.push(CallCost {
            model: model.to_string(),
            input_tokens,
            output_tokens,
            cost_usd: cost,
            timestamp: Utc::now(),
        });

        cost
    }

    /// Check if spending a call would exceed the budget.
    ///
    /// Returns `Err` with spent/limit if budget would be exceeded.
    pub fn check_budget(&self, limit: Option<f64>) -> Result<(), (f64, f64)> {
        if let Some(max) = limit {
            if self.total_cost_usd >= max {
                return Err((self.total_cost_usd, max));
            }
        }
        Ok(())
    }

    /// Total tokens (input + output).
    pub fn total_tokens(&self) -> u64 {
        self.total_input_tokens + self.total_output_tokens
    }

    /// Number of LLM calls recorded.
    pub fn call_count(&self) -> usize {
        self.calls.len()
    }

    /// Human-readable cost summary.
    pub fn summary(&self) -> String {
        format!(
            "${:.6} ({} calls, {}in/{}out tokens)",
            self.total_cost_usd,
            self.calls.len(),
            self.total_input_tokens,
            self.total_output_tokens,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pricing_lookup_known_model() {
        let p = get_pricing("gpt-4o");
        assert_eq!(p.input_per_1m, 2.50);
        assert_eq!(p.output_per_1m, 10.00);
    }

    #[test]
    fn pricing_lookup_unknown_model() {
        let p = get_pricing("some-unknown-model");
        assert_eq!(p.input_per_1m, 0.0);
        assert_eq!(p.output_per_1m, 0.0);
    }

    #[test]
    fn pricing_cost_calculation() {
        let p = ModelPricing::new(2.50, 10.00); // gpt-4o pricing
        // 1000 input + 500 output
        let cost = p.cost(1000, 500);
        // 1000 * 2.50 / 1M + 500 * 10.00 / 1M = 0.0025 + 0.005 = 0.0075
        assert!((cost - 0.0075).abs() < 1e-10);
    }

    #[test]
    fn cost_tracker_accumulation() {
        let mut t = CostTracker::new();
        t.record("gpt-4o", 1000, 500);
        t.record("gpt-4o", 2000, 300);

        assert_eq!(t.total_input_tokens, 3000);
        assert_eq!(t.total_output_tokens, 800);
        assert_eq!(t.call_count(), 2);
        assert!(t.total_cost_usd > 0.0);
    }

    #[test]
    fn cost_tracker_free_model() {
        let mut t = CostTracker::new();
        t.record("arcee-ai/trinity-large-preview:free", 1000, 500);
        assert_eq!(t.total_cost_usd, 0.0);
    }

    #[test]
    fn budget_check_unlimited() {
        let t = CostTracker::new();
        assert!(t.check_budget(None).is_ok());
    }

    #[test]
    fn budget_check_within_limit() {
        let mut t = CostTracker::new();
        t.record("gpt-4o", 1000, 500); // ~$0.0075
        assert!(t.check_budget(Some(1.0)).is_ok());
    }

    #[test]
    fn budget_check_exceeded() {
        let mut t = CostTracker::new();
        t.total_cost_usd = 5.0; // simulate spending
        let result = t.check_budget(Some(1.0));
        assert!(result.is_err());
        let (spent, limit) = result.unwrap_err();
        assert_eq!(spent, 5.0);
        assert_eq!(limit, 1.0);
    }
}
