//! Router Projectors for Memory Sparse Attention.
//!
//! Implements the Router K and Router Q projectors (Eq. 1 of MSA paper):
//!   K^R_{i,h} = H_i · W^h_{KR}
//!   Q^R_{q,h} = H_q · W^h_{QR}
//!
//! These specialized projectors generate routing-specific representations
//! that are separate from the standard K/Q/V projections used in attention.
//! The routing projections are lower-dimensional latent spaces optimized
//! specifically for document-level retrieval relevance scoring.

use burn::nn;
use burn::prelude::*;

/// Configuration for the Router Projector.
#[derive(Debug, Clone)]
pub struct RouterProjectorConfig {
    /// Input hidden dimension (model hidden size).
    pub hidden_dim: usize,
    /// Router projection dimension (can differ from hidden_dim for efficiency).
    pub router_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
}

/// Router Projector: produces routing keys (K^R) or routing queries (Q^R).
///
/// In the MSA architecture, each attention head has a separate routing
/// projection space. The router projector is a simple linear layer that
/// maps hidden states to routing representations.
///
/// Paper §3.2.1: "We introduce a Router K Projector, parameterized by W^h_{KR},
/// to generate a specialized routing key matrix K^R_{i,h}"
#[derive(Module, Debug)]
pub struct RouterProjector<B: Backend> {
    /// Linear projection: hidden_dim → router_dim * num_heads
    pub proj: nn::Linear<B>,
    /// Number of attention heads
    #[module(skip)]
    pub num_heads: usize,
    /// Dimension per head in routing space
    #[module(skip)]
    pub head_dim: usize,
}

impl<B: Backend> RouterProjector<B> {
    /// Create a new Router Projector.
    ///
    /// # Arguments
    /// * `config` - Router configuration
    /// * `device` - Computation device
    pub fn new(config: &RouterProjectorConfig, device: &B::Device) -> Self {
        let total_dim = config.router_dim * config.num_heads;
        let proj = nn::LinearConfig::new(config.hidden_dim, total_dim)
            .with_bias(false)
            .init(device);

        Self {
            proj,
            num_heads: config.num_heads,
            head_dim: config.router_dim,
        }
    }

    /// Project hidden states into routing space.
    ///
    /// # Arguments
    /// * `hidden` - Input hidden states [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
    ///
    /// # Returns
    /// * Routing representations [batch, seq_len, num_heads, head_dim] or [seq_len, num_heads, head_dim]
    pub fn forward(&self, hidden: Tensor<B, 2>) -> Tensor<B, 3> {
        let [seq_len, _hidden_dim] = hidden.dims();
        let projected = self.proj.forward(hidden); // [seq_len, num_heads * head_dim]
        projected.reshape([seq_len, self.num_heads, self.head_dim])
    }

    /// Project hidden states and return flat representation (for pooling).
    ///
    /// # Returns
    /// * [seq_len, num_heads * head_dim]
    pub fn forward_flat(&self, hidden: Tensor<B, 2>) -> Tensor<B, 2> {
        self.proj.forward(hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn test_router_projector_shapes() {
        let device = <B as Backend>::Device::default();
        let config = RouterProjectorConfig {
            hidden_dim: 64,
            router_dim: 16,
            num_heads: 4,
        };
        let router = RouterProjector::<B>::new(&config, &device);

        // Simulate 10 tokens with 64-dim hidden states
        let hidden = Tensor::<B, 2>::random(
            [10, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let out = router.forward(hidden);
        assert_eq!(out.dims(), [10, 4, 16], "Output shape should be [seq, heads, head_dim]");

        println!("✅ RouterProjector: [10, 64] → [10, 4, 16]");
    }

    #[test]
    fn test_router_projector_flat() {
        let device = <B as Backend>::Device::default();
        let config = RouterProjectorConfig {
            hidden_dim: 128,
            router_dim: 32,
            num_heads: 8,
        };
        let router = RouterProjector::<B>::new(&config, &device);

        let hidden = Tensor::<B, 2>::random(
            [20, 128],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let out = router.forward_flat(hidden);
        assert_eq!(out.dims(), [20, 256], "Flat output: [seq, num_heads * head_dim]");

        println!("✅ RouterProjector flat: [20, 128] → [20, 256]");
    }
}

// ═══════════════════════════════════════════════════════════════
// HyperRouter — Metacognitive parameter co-evolution for routing
// ═══════════════════════════════════════════════════════════════

/// Per-expert performance metrics for metacognitive routing.
#[derive(Debug, Clone, Default)]
pub struct ExpertMetrics {
    /// How many times this expert was selected (top-K gating).
    pub selections: u64,
    /// Sum of gating weights assigned to this expert.
    pub total_weight: f64,
    /// Sum of downstream quality signals (if available).
    pub quality_sum: f64,
    /// Count of quality observations.
    pub quality_count: u64,
}

impl ExpertMetrics {
    /// Average gating weight when selected.
    pub fn avg_weight(&self) -> f64 {
        if self.selections == 0 { 0.0 } else { self.total_weight / self.selections as f64 }
    }

    /// Average quality score.
    pub fn avg_quality(&self) -> f64 {
        if self.quality_count == 0 { 0.0 } else { self.quality_sum / self.quality_count as f64 }
    }
}

/// Recommended action from HyperRouter analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouterAction {
    /// No change needed.
    NoOp,
    /// Increase top_k (experts are underutilized).
    IncreaseTopK { from: usize, to: usize },
    /// Decrease top_k (too much compute for little benefit).
    DecreaseTopK { from: usize, to: usize },
    /// Expert is dead (never selected) — recommend merging.
    MergeExpert { expert_id: usize },
    /// Expert is overloaded — recommend splitting.
    SplitExpert { expert_id: usize },
}

/// HyperRouter: metacognitive parameter co-evolution for MSA routing.
///
/// Tracks per-expert utilization and quality, then recommends
/// structural changes (top_k tuning, expert merge/split) based on
/// observed routing patterns. This applies the HyperAgent
/// Parameter Co-Evolution pattern to the MoE routing layer.
///
/// ## Architecture
/// ```text
/// RouterProjector → top-K gating → experts
///       ↓               ↓              ↓
///   HyperRouter tracks:
///     - per-expert hit rates
///     - per-expert quality scores
///     - top_k utilization efficiency
///       ↓
///   Recommends: IncreaseTopK / DecreaseTopK / MergeExpert / SplitExpert
/// ```
#[derive(Debug, Clone)]
pub struct HyperRouter {
    /// Per-expert metrics.
    pub expert_metrics: Vec<ExpertMetrics>,
    /// Current top_k value.
    pub top_k: usize,
    /// Number of experts.
    pub num_experts: usize,
    /// Total routing decisions observed.
    pub total_routes: u64,
    /// Number of structural changes applied.
    pub evolution_count: u32,

    // ── Config ──

    /// Minimum routes before analysis triggers.
    pub min_routes_before_analysis: u64,
    /// Expert is "dead" if selection rate < this threshold.
    pub dead_threshold: f64,
    /// Expert is "overloaded" if selection rate > this threshold.
    pub overload_threshold: f64,
}

impl HyperRouter {
    /// Create a new HyperRouter for the given number of experts.
    pub fn new(num_experts: usize, top_k: usize) -> Self {
        Self {
            expert_metrics: (0..num_experts).map(|_| ExpertMetrics::default()).collect(),
            top_k,
            num_experts,
            total_routes: 0,
            evolution_count: 0,
            min_routes_before_analysis: 100,
            dead_threshold: 0.05,
            overload_threshold: 0.80,
        }
    }

    /// Record a routing decision.
    ///
    /// `selected_experts` is the list of (expert_id, gating_weight) pairs
    /// chosen by the top-K gating for this token/query.
    pub fn record_route(&mut self, selected_experts: &[(usize, f64)]) {
        self.total_routes += 1;
        for &(expert_id, weight) in selected_experts {
            if expert_id < self.expert_metrics.len() {
                let m = &mut self.expert_metrics[expert_id];
                m.selections += 1;
                m.total_weight += weight;
            }
        }
    }

    /// Record a quality signal for an expert.
    pub fn record_quality(&mut self, expert_id: usize, quality: f64) {
        if expert_id < self.expert_metrics.len() {
            let m = &mut self.expert_metrics[expert_id];
            m.quality_sum += quality;
            m.quality_count += 1;
        }
    }

    /// Per-expert selection rates.
    pub fn selection_rates(&self) -> Vec<f64> {
        if self.total_routes == 0 {
            return vec![0.0; self.num_experts];
        }
        self.expert_metrics
            .iter()
            .map(|m| m.selections as f64 / self.total_routes as f64)
            .collect()
    }

    /// Analyze routing patterns and recommend structural changes.
    pub fn analyze(&self) -> Vec<RouterAction> {
        if self.total_routes < self.min_routes_before_analysis {
            return vec![RouterAction::NoOp];
        }

        let rates = self.selection_rates();
        let mut actions = Vec::new();

        // Check for dead experts
        for (i, &rate) in rates.iter().enumerate() {
            if rate < self.dead_threshold {
                actions.push(RouterAction::MergeExpert { expert_id: i });
            }
        }

        // Check for overloaded experts
        for (i, &rate) in rates.iter().enumerate() {
            if rate > self.overload_threshold {
                actions.push(RouterAction::SplitExpert { expert_id: i });
            }
        }

        // Check if top_k is too small (most experts never activated)
        let active_count = rates.iter().filter(|&&r| r >= self.dead_threshold).count();
        if active_count < self.top_k && self.top_k < self.num_experts {
            actions.push(RouterAction::IncreaseTopK {
                from: self.top_k,
                to: (self.top_k + 1).min(self.num_experts),
            });
        }

        // Check if top_k is too large (diminishing returns)
        // All experts have similar selection rates = top_k wastes compute
        let rate_std = self.rate_std(&rates);
        if rate_std < 0.02 && self.top_k > 1 {
            actions.push(RouterAction::DecreaseTopK {
                from: self.top_k,
                to: self.top_k - 1,
            });
        }

        if actions.is_empty() {
            actions.push(RouterAction::NoOp);
        }

        actions
    }

    /// Standard deviation of selection rates.
    fn rate_std(&self, rates: &[f64]) -> f64 {
        let n = rates.len() as f64;
        if n <= 1.0 { return 0.0; }
        let mean = rates.iter().sum::<f64>() / n;
        let variance = rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        variance.sqrt()
    }

    /// Apply a recommended top_k change.
    pub fn apply_top_k_change(&mut self, new_top_k: usize) {
        self.top_k = new_top_k;
        self.evolution_count += 1;
        // Reset metrics to measure the new config fairly
        self.total_routes = 0;
        for m in &mut self.expert_metrics {
            *m = ExpertMetrics::default();
        }
    }

    /// Summary for diagnostics.
    pub fn summary(&self) -> String {
        let rates = self.selection_rates();
        let mut s = format!(
            "HyperRouter: top_k={}, experts={}, routes={}, evolutions={}\n",
            self.top_k, self.num_experts, self.total_routes, self.evolution_count
        );
        for (i, rate) in rates.iter().enumerate() {
            let m = &self.expert_metrics[i];
            s.push_str(&format!(
                "  expert_{}: sel_rate={:.1}%, avg_weight={:.3}, avg_quality={:.3}\n",
                i, rate * 100.0, m.avg_weight(), m.avg_quality()
            ));
        }
        s
    }
}

#[cfg(test)]
mod hyper_router_tests {
    use super::*;

    #[test]
    fn test_hyper_router_tracks_selections() {
        let mut router = HyperRouter::new(4, 2);

        // Route 200 tokens, always selecting experts 0 and 1
        for _ in 0..200 {
            router.record_route(&[(0, 0.6), (1, 0.4)]);
        }

        assert_eq!(router.total_routes, 200);
        let rates = router.selection_rates();
        assert!((rates[0] - 1.0).abs() < 0.01); // Expert 0: 100% selection rate
        assert!((rates[1] - 1.0).abs() < 0.01); // Expert 1: 100%
        assert!((rates[2] - 0.0).abs() < 0.01); // Expert 2: never selected
        assert!((rates[3] - 0.0).abs() < 0.01); // Expert 3: never selected
    }

    #[test]
    fn test_hyper_router_detects_dead_experts() {
        let mut router = HyperRouter::new(4, 2);
        router.min_routes_before_analysis = 50;

        // Route — only uses experts 0 and 1, ignoring 2 and 3
        for _ in 0..100 {
            router.record_route(&[(0, 0.7), (1, 0.3)]);
        }

        let actions = router.analyze();
        // Should recommend merging experts 2 and 3 (dead)
        let merge_actions: Vec<_> = actions.iter()
            .filter(|a| matches!(a, RouterAction::MergeExpert { .. }))
            .collect();
        assert!(merge_actions.len() >= 2, "Should recommend merging dead experts");
    }

    #[test]
    fn test_hyper_router_detects_overloaded_expert() {
        let mut router = HyperRouter::new(4, 1);
        router.min_routes_before_analysis = 50;

        // Route — always picks expert 0
        for _ in 0..100 {
            router.record_route(&[(0, 1.0)]);
        }

        let actions = router.analyze();
        let split_actions: Vec<_> = actions.iter()
            .filter(|a| matches!(a, RouterAction::SplitExpert { .. }))
            .collect();
        assert!(!split_actions.is_empty(), "Should recommend splitting overloaded expert");
    }

    #[test]
    fn test_hyper_router_no_action_when_insufficient_data() {
        let router = HyperRouter::new(4, 2);
        let actions = router.analyze();
        assert_eq!(actions, vec![RouterAction::NoOp]);
    }

    #[test]
    fn test_hyper_router_top_k_update() {
        let mut router = HyperRouter::new(4, 2);
        for _ in 0..50 {
            router.record_route(&[(0, 0.5), (1, 0.5)]);
        }
        assert_eq!(router.total_routes, 50);

        router.apply_top_k_change(3);
        assert_eq!(router.top_k, 3);
        assert_eq!(router.total_routes, 0); // Reset
        assert_eq!(router.evolution_count, 1);
    }
}
