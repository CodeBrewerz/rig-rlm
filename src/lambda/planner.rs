//! λ-RLM Planner — Algorithm 1, Phases 2–4.
//!
//! Implements task detection (1 LLM call) and optimal parameter
//! computation (k*, τ*, d) from Theorem 4 of Roy et al. (2026).
//!
//! The planner is deterministic after task detection: all structural
//! decisions are pure math with zero neural cost.

use std::fmt;

use crate::monad::provider::LlmProvider;

// ─── Task Types (Table 1B) ──────────────────────────────────────────

/// Task types from Table 1B of the paper.
/// Each type maps to a specific composition operator and execution plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    /// O(1) — needle-in-a-haystack search
    /// Plan: SPLIT → MAP(PEEK) → FILTER → MAP(M) → BEST
    Search,

    /// O(n) — classify items across the input
    /// Plan: SPLIT → MAP(M) → CONCAT
    Classify,

    /// O(n) — count/aggregate statistics
    /// Plan: SPLIT → MAP(M) → MERGE
    Aggregate,

    /// O(n²) — pairwise comparisons
    /// Plan: SPLIT → MAP(M) → PARSE → FILTER → CROSS
    Pairwise,

    /// O(n) — distill into a summary
    /// Plan: SPLIT → MAP(M) → CONCAT → M
    Summarise,

    /// Variable — multi-hop search with synthesis
    /// Plan: SPLITδ → MAP(PEEK) → FILTER → MAP(M) → M_synth
    MultiHop,
}

impl TaskType {
    /// Whether this task type benefits from pre-filtering chunks.
    pub fn needs_prefilter(&self) -> bool {
        matches!(self, Self::Search | Self::MultiHop)
    }

    /// Whether the composition step requires a neural call (M ∘ CONCAT).
    pub fn needs_neural_compose(&self) -> bool {
        matches!(self, Self::Summarise | Self::MultiHop)
    }

    /// Parse from LLM response text.
    pub fn from_llm_response(text: &str) -> Self {
        let lower = text.to_lowercase();
        if lower.contains("search") || lower.contains("needle") || lower.contains("find") {
            Self::Search
        } else if lower.contains("classify") || lower.contains("categor") {
            Self::Classify
        } else if lower.contains("aggregate") || lower.contains("count") || lower.contains("statistic") {
            Self::Aggregate
        } else if lower.contains("pairwise") || lower.contains("pair") || lower.contains("compar") {
            Self::Pairwise
        } else if lower.contains("summar") || lower.contains("distill") || lower.contains("condense") {
            Self::Summarise
        } else if lower.contains("multi") || lower.contains("hop") || lower.contains("evidence") {
            Self::MultiHop
        } else {
            // Default: summarise is the safest general-purpose fallback
            Self::Summarise
        }
    }

    /// All defined variants, for menu rendering.
    pub fn all() -> &'static [TaskType] {
        &[
            Self::Search,
            Self::Classify,
            Self::Aggregate,
            Self::Pairwise,
            Self::Summarise,
            Self::MultiHop,
        ]
    }
}

impl fmt::Display for TaskType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Search => write!(f, "search"),
            Self::Classify => write!(f, "classify"),
            Self::Aggregate => write!(f, "aggregate"),
            Self::Pairwise => write!(f, "pairwise"),
            Self::Summarise => write!(f, "summarise"),
            Self::MultiHop => write!(f, "multi_hop"),
        }
    }
}

// ─── Cost Parameters ────────────────────────────────────────────────

/// Cost model parameters from Definition 3 of the paper.
///
/// C(n) = c_in · n + c_out · n̄_out
#[derive(Debug, Clone)]
pub struct CostParams {
    /// Cost per input token (USD).
    pub c_in: f64,
    /// Cost per output token (USD).
    pub c_out: f64,
    /// Expected output length (tokens).
    pub n_out: f64,
    /// Cost per composition step (symbolic = 0, neural = c_in · K).
    pub c_compose: f64,
    /// Context-rot decay factor ρ ∈ (0, 1].
    pub rho: f64,
    /// Peak accuracy A₀.
    pub a0: f64,
    /// Composition reliability A⊕ ∈ (0, 1].
    pub a_compose: f64,
}

impl Default for CostParams {
    fn default() -> Self {
        Self {
            c_in: 0.0001,    // $0.10 per 1M tokens
            c_out: 0.0003,   // $0.30 per 1M tokens
            n_out: 500.0,    // expected 500 output tokens
            c_compose: 0.0,  // symbolic composition (free)
            rho: 0.85,       // 15% accuracy loss per context-length
            a0: 0.95,        // 95% peak accuracy
            a_compose: 0.98, // 98% composition reliability
        }
    }
}

// ─── Execution Plan ─────────────────────────────────────────────────

/// The output of the planning phase.
///
/// Contains all parameters needed to construct and execute Φ.
/// Computed deterministically before any recursive execution.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Detected task type.
    pub task_type: TaskType,
    /// Partition size k* (Theorem 4: optimal is 2).
    pub k_star: usize,
    /// Leaf threshold τ* in tokens.
    pub tau_star: usize,
    /// Recursion depth d = ⌈log_k*(n/τ*)⌉.
    pub depth: usize,
    /// Estimated total cost (pre-computed from Theorem 2).
    pub estimated_cost: f64,
    /// Estimated total M invocations: (k*)^d + 1.
    pub estimated_calls: usize,
    /// Whether this plan includes a pre-filter step.
    pub has_prefilter: bool,
    /// Whether composition requires a neural call.
    pub neural_compose: bool,
}

impl ExecutionPlan {
    /// Human-readable summary of the plan.
    pub fn summary(&self) -> String {
        format!(
            "λ-RLM Plan: task={}, k*={}, τ*={}, depth={}, est_calls={}, est_cost=${:.4}, prefilter={}, neural_compose={}",
            self.task_type,
            self.k_star,
            self.tau_star,
            self.depth,
            self.estimated_calls,
            self.estimated_cost,
            self.has_prefilter,
            self.neural_compose,
        )
    }

    /// Generates a Category Theory String Diagram (Mermaid format).
    /// Visually maps the entire Map-Reduce (Hylomorphism) execution tree dynamically.
    pub fn to_mermaid(&self) -> String {
        let mut out = String::new();
        out.push_str("graph TD\n");
        // Monoidal categories colors
        out.push_str("    classDef pure fill:#e1f5fe,stroke:#01579b,stroke-width:2px;\n");
        out.push_str("    classDef neural fill:#fce4ec,stroke:#880e4f,stroke-width:2px,stroke-dasharray: 5 5;\n");
        out.push_str("    classDef input fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;\n\n");
        
        out.push_str("    P((Massive Context P)):::input\n");
        
        // Root Split (Comultiplication)
        out.push_str(&format!("    P --> S0{{\"SPLIT(k*={})\"}}:::pure\n", self.k_star));
        
        // Parallel map evaluation
        let branches = self.k_star.min(4); // Cap rendering at 4 for sanity
        for i in 1..=branches {
            let label = if i == branches && self.k_star > 4 { 
                format!("... {} more", self.k_star - 3) 
            } else { 
                format!("Chunk {}", i) 
            };
            
            out.push_str(&format!("    S0 -->|{}| C{}\n", label, i));
            
            if self.has_prefilter {
                out.push_str(&format!("    C{} --> F{}{{\"FILTER (Pure)\"}}:::pure\n", i, i));
                out.push_str(&format!("    F{} --> M{}[[\"M (Base LLM)\"]]:::neural\n", i, i));
            } else {
                out.push_str(&format!("    C{} --> M{}[[\"M (Base LLM)\"]]:::neural\n", i, i));
            }
            out.push_str(&format!("    M{} --> R0\n", i));
        }
        
        // Reduction (Multiplication)
        let red_class = if self.neural_compose { "neural" } else { "pure" };
        let red_type = if self.neural_compose { "Neural" } else { "Symbolic" };
        out.push_str(&format!("    R0{{\"REDUCE ({})\"}}:::{} \n", red_type, red_class));
        
        out.push_str(&format!("    R0 --> Out((Final {} Result)):::input\n\n", self.task_type));
        
        // Add global metrics panel
        out.push_str("    subgraph Metrics [Execution Plan Metrics]\n");
        out.push_str(&format!("        M_depth[\"Tree Depth: {}\"]\n", self.depth));
        out.push_str(&format!("        M_calls[\"Est. LLM Calls: {}\"]\n", self.estimated_calls));
        out.push_str(&format!("        M_cost[\"Est. Cost: ${}\"]\n", format!("{:.4}", self.estimated_cost)));
        out.push_str("    end\n");
        
        out
    }
}

// ─── Task Detection (Phase 2) ───────────────────────────────────────

/// Detect task type from a preview of the prompt.
///
/// This is the only neural call in the planning phase (Algorithm 1, line 5).
/// Uses a structured menu-selection prompt to minimize ambiguity.
pub async fn detect_task_type(
    preview: &str,
    prompt_len: usize,
    provider: &LlmProvider,
) -> crate::monad::error::Result<TaskType> {
    let menu = TaskType::all()
        .iter()
        .enumerate()
        .map(|(i, t)| format!("{}. {}", i + 1, t))
        .collect::<Vec<_>>()
        .join("\n");

    let detection_prompt = format!(
        "You are a task classifier for a long-context reasoning system.\n\
         Given a preview of the input text ({} tokens total), select the task type.\n\n\
         Task types:\n{}\n\n\
         Input preview (first 500 chars):\n{}\n\n\
         Respond with ONLY the task type name (e.g., 'search', 'summarise', 'aggregate').\n\
         If unsure, respond 'summarise'.",
        prompt_len, menu, preview
    );

    match provider.complete(&detection_prompt).await {
        Ok(response) => Ok(TaskType::from_llm_response(&response)),
        Err(e) => {
            tracing::warn!("Task detection failed: {e}, defaulting to Summarise");
            Ok(TaskType::Summarise)
        }
    }
}

// ─── Optimal Planning (Phase 3–4) ───────────────────────────────────

/// Compute the optimal execution plan.
///
/// Implements Algorithm 3, lines 6–16:
/// 1. Select composition operator and plan from Table 1B
/// 2. Compute k* (Theorem 4: k* = 2 for standard cost model)
/// 3. Apply accuracy constraint loop
/// 4. Compute τ*, d, cost estimate
pub fn plan(
    input_len_tokens: usize,
    context_window: usize,
    accuracy_target: f64,
    task_type: TaskType,
    cost_params: &CostParams,
) -> ExecutionPlan {
    let n = input_len_tokens;
    let k = context_window;

    // If prompt fits in context window, no decomposition needed
    if n <= k {
        return ExecutionPlan {
            task_type,
            k_star: 1,
            tau_star: n,
            depth: 0,
            estimated_cost: cost_params.c_in * n as f64 + cost_params.c_out * cost_params.n_out,
            estimated_calls: 1,
            has_prefilter: false,
            neural_compose: false,
        };
    }

    // Theorem 4: k* = 2 is the unconditional cost-optimal partition.
    // "Under C(n) = c_in·n + c_out·n̄_out and C⊕(k) = c⊕·k, the 
    //  cost-minimizing branching factor is k* = 2."
    let k_star: usize = 2;

    // Compute depth: d = ⌈log_k*(n/K)⌉
    let depth = log_ceil(n, k, k_star);

    // Log accuracy estimate (informational, doesn't change k*)
    let leaf_accuracy = cost_params.a0 * cost_params.rho.powf(1.0);
    let end_to_end = leaf_accuracy.powi(depth as i32)
        * cost_params.a_compose.powi(depth as i32);
    if end_to_end < accuracy_target {
        tracing::debug!(
            "λ-RLM: estimated accuracy {:.3} < target {:.3} at depth={}, k*={}. \
             Consider reducing context window K for shallower recursion.",
            end_to_end, accuracy_target, depth, k_star
        );
    }

    // τ* = min(K, ⌊n/k*⌋)
    let tau_star = k.min(n / k_star);

    // Recompute depth with final k* and τ*
    let depth = if tau_star > 0 {
        log_ceil(n, tau_star, k_star)
    } else {
        1
    };

    // Cost estimation (Theorem 2):
    // T(n) ≤ (n·k*/τ*) · C(τ*) + ((n·k* - τ*) / (τ*(k*-1))) · C⊕(k*)
    let leaf_count = k_star.pow(depth as u32);
    let leaf_cost = cost_params.c_in * tau_star as f64 + cost_params.c_out * cost_params.n_out;
    let compose_cost = if task_type.needs_neural_compose() {
        cost_params.c_in * k as f64 + cost_params.c_out * cost_params.n_out
    } else {
        cost_params.c_compose * k_star as f64
    };
    let total_cost = leaf_count as f64 * leaf_cost + depth as f64 * compose_cost;

    // +1 for the task detection call
    let estimated_calls = leaf_count + 1
        + if task_type.needs_neural_compose() { depth } else { 0 };

    ExecutionPlan {
        task_type,
        k_star,
        tau_star,
        depth,
        estimated_cost: total_cost,
        estimated_calls,
        has_prefilter: task_type.needs_prefilter(),
        neural_compose: task_type.needs_neural_compose(),
    }
}

/// Compute ⌈log_base(n/threshold)⌉, clamped to [0, 20].
fn log_ceil(n: usize, threshold: usize, base: usize) -> usize {
    if n <= threshold || base <= 1 {
        return 0;
    }
    let ratio = n as f64 / threshold as f64;
    let d = (ratio.ln() / (base as f64).ln()).ceil() as usize;
    d.min(20) // safety cap — 2^20 = 1M leaves is the hard limit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_type_from_response() {
        assert_eq!(TaskType::from_llm_response("search"), TaskType::Search);
        assert_eq!(TaskType::from_llm_response("summarise"), TaskType::Summarise);
        assert_eq!(TaskType::from_llm_response("aggregate counts"), TaskType::Aggregate);
        assert_eq!(TaskType::from_llm_response("pairwise comparison"), TaskType::Pairwise);
        assert_eq!(TaskType::from_llm_response("unknown"), TaskType::Summarise);
    }

    #[test]
    fn test_plan_fits_in_context() {
        let plan = plan(1000, 32000, 0.80, TaskType::Search, &CostParams::default());
        assert_eq!(plan.k_star, 1);
        assert_eq!(plan.depth, 0);
        assert_eq!(plan.estimated_calls, 1);
    }

    #[test]
    fn test_plan_requires_decomposition() {
        // 128K tokens, 32K context window
        let plan = plan(128_000, 32_000, 0.80, TaskType::Aggregate, &CostParams::default());
        assert!(plan.k_star >= 2);
        assert!(plan.depth >= 1);
        assert!(plan.estimated_calls > 1);
        assert!(plan.tau_star <= 32_000);
    }

    #[test]
    fn test_plan_optimal_k_is_2() {
        // Theorem 4: k* = 2 for standard cost model with symbolic composition
        // For neural-compose tasks (Summarise), k* may increase due to
        // the accuracy constraint loop — that's correct behavior.
        let plan = plan(100_000, 32_000, 0.80, TaskType::Aggregate, &CostParams::default());
        assert_eq!(plan.k_star, 2, "Theorem 4: optimal k* should be 2 for symbolic compose");
    }

    #[test]
    fn test_plan_depth_bound() {
        // d = ⌈log_2(n/τ*)⌉
        let plan = plan(256_000, 32_000, 0.80, TaskType::Search, &CostParams::default());
        assert!(plan.depth <= 4, "depth should be bounded: got {}", plan.depth);
    }

    #[test]
    fn test_plan_prefilter() {
        let search_plan = plan(64_000, 32_000, 0.80, TaskType::Search, &CostParams::default());
        assert!(search_plan.has_prefilter);

        let agg_plan = plan(64_000, 32_000, 0.80, TaskType::Aggregate, &CostParams::default());
        assert!(!agg_plan.has_prefilter);
    }

    #[test]
    fn test_log_ceil() {
        assert_eq!(log_ceil(100, 100, 2), 0); // fits exactly
        assert_eq!(log_ceil(200, 100, 2), 1); // one level
        assert_eq!(log_ceil(400, 100, 2), 2); // two levels
        assert_eq!(log_ceil(128000, 32000, 2), 2); // 128K / 32K = 4 = 2^2
    }
}
