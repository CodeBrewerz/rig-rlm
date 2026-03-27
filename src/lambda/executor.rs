//! λ-RLM Recursive Executor Φ — Algorithm 4.
//!
//! The "Y-combinator for LLMs": recursion expressed as a fixed-point
//! over deterministic combinators. The base model M is invoked ONLY
//! at bounded leaf subproblems. All control flow is symbolic.
//!
//! Key properties (proven in the paper):
//! - **Termination** (Theorem 1): rank strictly decreases at each level  
//! - **Cost bound** (Theorem 2): T(n) ≤ (n·k*/τ*) · C(τ*)
//! - **Accuracy** (Theorem 3): power-law decay, not exponential

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::combinators;
use super::planner::ExecutionPlan;
use super::templates;
use crate::monad::provider::LlmProvider;

/// The recursive executor Φ from Algorithm 4.
///
/// Constructed by the planner with fixed parameters (k*, τ*, ⊕, π).
/// Execution is a single recursive call — no open-ended loop.
pub struct LambdaExecutor {
    /// The pre-computed execution plan.
    plan: ExecutionPlan,
    /// The base language model M (used only at leaves).
    provider: Arc<LlmProvider>,
    /// The original user query (for leaf template formatting).
    user_query: String,
}

/// Metrics collected during execution (atomic for parallel MAP).
#[derive(Debug, Clone, Default)]
pub struct ExecutionMetrics {
    /// Total M invocations (leaf β-reductions).
    pub leaf_calls: usize,
    /// Total composition steps.
    pub compose_steps: usize,
    /// Maximum depth reached.
    pub max_depth_reached: usize,
    /// Chunks that were pre-filtered out.
    pub filtered_chunks: usize,
}

/// Thread-safe atomic metrics for parallel MAP execution.
struct AtomicMetrics {
    leaf_calls: AtomicUsize,
    compose_steps: AtomicUsize,
    max_depth_reached: AtomicUsize,
    filtered_chunks: AtomicUsize,
}

impl AtomicMetrics {
    fn new() -> Self {
        Self {
            leaf_calls: AtomicUsize::new(0),
            compose_steps: AtomicUsize::new(0),
            max_depth_reached: AtomicUsize::new(0),
            filtered_chunks: AtomicUsize::new(0),
        }
    }

    fn snapshot(&self) -> ExecutionMetrics {
        ExecutionMetrics {
            leaf_calls: self.leaf_calls.load(Ordering::Relaxed),
            compose_steps: self.compose_steps.load(Ordering::Relaxed),
            max_depth_reached: self.max_depth_reached.load(Ordering::Relaxed),
            filtered_chunks: self.filtered_chunks.load(Ordering::Relaxed),
        }
    }
}

impl LambdaExecutor {
    /// Create a new executor from a plan and provider.
    pub fn new(plan: ExecutionPlan, provider: Arc<LlmProvider>, user_query: String) -> Self {
        Self {
            plan,
            provider,
            user_query,
        }
    }

    /// Execute Φ(P) — the complete recursive decomposition.
    ///
    /// This is a single invocation that recursively decomposes the prompt,
    /// processes leaves with M, and composes results with ⊕.
    /// MAP branches run in parallel via `join_all`.
    pub async fn execute(&self, prompt: &str) -> crate::monad::error::Result<String> {
        let metrics = AtomicMetrics::new();
        
        // Print the String Diagram before execution
        println!("\n{}\n🔮 λ-RLM Execution String Diagram\n{}\n{}\n{}\n", 
            "═".repeat(60),
            "═".repeat(60),
            self.plan.to_mermaid(),
            "═".repeat(60)
        );

        let result = self.phi(prompt, 0, &metrics).await?;

        let snapshot = metrics.snapshot();

        tracing::info!(
            leaf_calls = snapshot.leaf_calls,
            compose_steps = snapshot.compose_steps,
            max_depth = snapshot.max_depth_reached,
            filtered = snapshot.filtered_chunks,
            "λ-RLM execution complete"
        );

        // For neural-compose tasks (Summarise, MultiHop), do a final synthesis
        if self.plan.neural_compose {
            let synthesis_prompt = templates::format_synthesis(
                &result,
                self.plan.task_type,
                &self.user_query,
            );
            let final_result = self.provider.complete(&synthesis_prompt).await?;
            return Ok(final_result);
        }

        Ok(result)
    }

    /// The recursive function Φ(P) from Equation (4) / Algorithm 4.
    ///
    /// ```text
    /// Φ(P) = if |P| ≤ τ* then M(P)
    ///        else REDUCE(⊕, MAP(λpi. Φ(pi), SPLIT(P, k*)))
    /// ```
    ///
    /// `depth` tracks current recursion level for metrics and safety.
    /// MAP branches execute in **parallel** via `join_all` (paper §4.1).
    fn phi<'a>(
        &'a self,
        prompt: &'a str,
        depth: usize,
        metrics: &'a AtomicMetrics,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::monad::error::Result<String>> + Send + 'a>>
    {
        Box::pin(async move {
            metrics.max_depth_reached.fetch_max(depth, Ordering::Relaxed);

            let token_len = combinators::token_count(prompt);

            // ── Base case: |P| ≤ τ* → leaf β-reduction ──
            if token_len <= self.plan.tau_star {
                let leaf_prompt = templates::format_leaf(
                    prompt,
                    self.plan.task_type,
                    &self.user_query,
                );
                metrics.leaf_calls.fetch_add(1, Ordering::Relaxed);

                tracing::debug!(
                    depth,
                    tokens = token_len,
                    "λ-RLM leaf: invoking M"
                );

                return self.provider.complete(&leaf_prompt).await;
            }

            // ── Guard: depth exceeded (defense-in-depth) ──
            if depth > self.plan.depth + 2 {
                tracing::warn!(
                    depth,
                    max = self.plan.depth,
                    "λ-RLM: depth exceeded plan, falling back to direct call"
                );
                let leaf_prompt = templates::format_leaf(
                    prompt,
                    self.plan.task_type,
                    &self.user_query,
                );
                metrics.leaf_calls.fetch_add(1, Ordering::Relaxed);
                return self.provider.complete(&leaf_prompt).await;
            }

            // ── Recursive case: SPLIT → [FILTER] → MAP(Φ) → REDUCE(⊕) ──

            // SPLIT(P, k*): deterministic, pre-verified
            let chunks = combinators::split(prompt, self.plan.k_star);

            tracing::debug!(
                depth,
                tokens = token_len,
                chunks = chunks.len(),
                k_star = self.plan.k_star,
                "λ-RLM: splitting"
            );

            // Optional pre-filter (for Search and MultiHop tasks)
            let chunks = if self.plan.has_prefilter {
                let preview_len = self.plan.tau_star / 10;
                let filtered: Vec<String> = chunks
                    .into_iter()
                    .filter(|chunk| {
                        let preview = combinators::peek(chunk, 0, preview_len);
                        let is_relevant = self.is_relevant_preview(&preview);
                        if !is_relevant {
                            metrics.filtered_chunks.fetch_add(1, Ordering::Relaxed);
                        }
                        is_relevant
                    })
                    .collect();

                // Safety: if everything was filtered, keep at least 1 chunk
                if filtered.is_empty() {
                    tracing::warn!(depth, "λ-RLM: all chunks filtered, keeping original");
                    combinators::split(prompt, self.plan.k_star)
                } else {
                    filtered
                }
            } else {
                chunks
            };

            // MAP(λpi. Φ(pi)): parallel recursive sub-calls via join_all
            let futures: Vec<_> = chunks
                .iter()
                .map(|chunk| self.phi(chunk, depth + 1, metrics))
                .collect();

            let results_vec = futures_util::future::join_all(futures).await;
            let mut results = Vec::with_capacity(results_vec.len());
            for r in results_vec {
                results.push(r?);
            }

            // REDUCE(⊕): deterministic composition
            metrics.compose_steps.fetch_add(1, Ordering::Relaxed);
            let composed = templates::compose_symbolic(results, self.plan.task_type);

            Ok(composed)
        })
    }

    /// Symbolic relevance check for pre-filtering.
    ///
    /// Checks if a preview contains keywords from the user's query.
    /// This is a zero-cost symbolic operation (no neural call).
    fn is_relevant_preview(&self, preview: &str) -> bool {
        let query_lower = self.user_query.to_lowercase();
        let preview_lower = preview.to_lowercase();

        // Extract significant words from query (≥3 chars, not stop words)
        let stop_words = [
            "the", "is", "at", "which", "on", "a", "an", "and", "or",
            "but", "in", "with", "to", "for", "of", "it", "by", "from",
            "how", "what", "when", "where", "who", "why", "are", "was",
            "were", "been", "being", "have", "has", "had", "do", "does",
            "did", "will", "would", "could", "should", "may", "might",
            "can", "this", "that", "these", "those", "not", "all", "any",
        ];
        let query_words: Vec<String> = query_lower
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| w.len() >= 3 && !stop_words.contains(&w.as_str()))
            .collect();

        if query_words.is_empty() {
            return true; // no filter if query has no significant words
        }

        // Keep chunk if ≥1 query keyword appears in preview
        query_words.iter().any(|w| preview_lower.contains(w.as_str()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relevance_check() {
        let executor = LambdaExecutor {
            plan: ExecutionPlan {
                task_type: super::super::planner::TaskType::Search,
                k_star: 2,
                tau_star: 1000,
                depth: 2,
                estimated_cost: 0.0,
                estimated_calls: 5,
                has_prefilter: true,
                neural_compose: false,
            },
            provider: Arc::new(LlmProvider::new(
                crate::monad::provider::ProviderConfig::local("test-model"),
            )),
            user_query: "What is the magic number?".to_string(),
        };

        assert!(executor.is_relevant_preview("The magic number is 42"));
        assert!(executor.is_relevant_preview("There is a number here"));
        assert!(!executor.is_relevant_preview("Lorem ipsum dolor sit amet"));
    }
}
