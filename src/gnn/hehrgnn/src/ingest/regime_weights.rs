//! Regime-Switching Temporal Weights
//!
//! From the queue-reactive LOB paper (Huang et al. 2014):
//! cross-queue dependencies become stronger during volatile regimes
//! (when queues deplete frequently and ρ(n) fluctuates).
//!
//! This module implements adaptive edge weight scaling based on
//! volatility regime detection:
//! - High-volatility regime → upweight cross-dependency edges
//! - Low-volatility regime → use standard weights
//!
//! Integrates with the graph builder and temporal selector.

use std::collections::HashMap;

/// Volatility regime classification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VolatilityRegime {
    /// Low volatility: stable flow patterns, ρ(n) near 1.
    Low,
    /// Medium volatility: moderate flow fluctuations.
    Medium,
    /// High volatility: rapid queue depletion, ρ(n) highly variable.
    High,
}

impl VolatilityRegime {
    /// Classify volatility regime from the coefficient of variation of ρ values.
    ///
    /// CV = std(ρ) / mean(ρ) captures how much the flow ratios fluctuate.
    /// - CV < 0.3 → Low volatility
    /// - CV < 0.7 → Medium volatility
    /// - CV ≥ 0.7 → High volatility
    pub fn classify(rho_values: &[f32]) -> Self {
        if rho_values.len() < 2 {
            return VolatilityRegime::Low;
        }

        let mean = rho_values.iter().sum::<f32>() / rho_values.len() as f32;
        if mean.abs() < 1e-8 {
            return VolatilityRegime::Low;
        }

        let variance =
            rho_values.iter().map(|&r| (r - mean).powi(2)).sum::<f32>() / rho_values.len() as f32;
        let cv = variance.sqrt() / mean.abs();

        if cv < 0.3 {
            VolatilityRegime::Low
        } else if cv < 0.7 {
            VolatilityRegime::Medium
        } else {
            VolatilityRegime::High
        }
    }

    /// Get the edge weight scaling factor for cross-dependency edges.
    ///
    /// From the paper: cross-queue effects are strongest during volatile periods.
    /// We upweight cross_dep edges by this factor during message passing.
    pub fn cross_dep_scale(&self) -> f32 {
        match self {
            VolatilityRegime::Low => 1.0,    // No upweighting
            VolatilityRegime::Medium => 1.5, // 50% boost
            VolatilityRegime::High => 2.5,   // 150% boost
        }
    }
}

/// Compute per-node-type volatility regime from flow ratio features.
///
/// Takes the ρ(n) values computed by `compute_flow_ratio_edges` and
/// classifies the overall volatility regime for each node type.
pub fn detect_volatility_regimes(
    node_rho: &HashMap<String, Vec<f32>>,
) -> HashMap<String, VolatilityRegime> {
    let mut regimes = HashMap::new();
    for (node_type, rho_values) in node_rho {
        regimes.insert(node_type.clone(), VolatilityRegime::classify(rho_values));
    }
    regimes
}

/// Apply regime-based temporal weighting to edge features.
///
/// For cross_dep edges: multiply edge features by the volatility scaling factor.
/// This implements the paper's insight that cross-queue dependencies
/// are more informative during volatile regimes.
pub fn apply_regime_weights<B: burn::prelude::Backend>(
    graph: &mut crate::data::hetero_graph::HeteroGraph<B>,
    regimes: &HashMap<String, VolatilityRegime>,
    device: &B::Device,
) {
    use burn::prelude::*;

    let edge_types: Vec<_> = graph.edge_features.keys().cloned().collect();

    for et in edge_types {
        let (src_type, relation, _dst_type) = &et;

        // Only scale cross_dep edges based on regime
        if relation != "cross_dep" {
            continue;
        }

        let regime = regimes
            .get(src_type)
            .copied()
            .unwrap_or(VolatilityRegime::Low);
        let scale = regime.cross_dep_scale();

        if (scale - 1.0).abs() < 1e-6 {
            continue; // No scaling needed
        }

        if let Some(features) = graph.edge_features.get(&et) {
            let scaled = features.clone().mul_scalar(scale);
            graph.edge_features.insert(et.clone(), scaled);
        }
    }
}

/// Compute ρ(n) values per node type from graph degree statistics.
///
/// Used by `detect_volatility_regimes()` to classify the current regime.
pub fn compute_node_rho<B: burn::prelude::Backend>(
    graph: &crate::data::hetero_graph::HeteroGraph<B>,
) -> HashMap<String, Vec<f32>> {
    let mut in_degree: HashMap<String, Vec<f32>> = HashMap::new();
    let mut out_degree: HashMap<String, Vec<f32>> = HashMap::new();

    for (nt, &count) in &graph.node_counts {
        in_degree.insert(nt.clone(), vec![0.0; count]);
        out_degree.insert(nt.clone(), vec![0.0; count]);
    }

    for (et, _) in &graph.edge_index {
        let (src_type, _, dst_type) = et;
        if let Some((src_vec, dst_vec)) = graph.edges_as_vecs(et) {
            if let Some(out_deg) = out_degree.get_mut(src_type) {
                for &s in &src_vec {
                    let s = s as usize;
                    if s < out_deg.len() {
                        out_deg[s] += 1.0;
                    }
                }
            }
            if let Some(in_deg) = in_degree.get_mut(dst_type) {
                for &d in &dst_vec {
                    let d = d as usize;
                    if d < in_deg.len() {
                        in_deg[d] += 1.0;
                    }
                }
            }
        }
    }

    let mut rho = HashMap::new();
    for (nt, in_deg) in &in_degree {
        let out_deg = out_degree.get(nt).unwrap();
        let node_rho: Vec<f32> = in_deg
            .iter()
            .zip(out_deg.iter())
            .map(|(&i, &o)| i / (o + 1.0))
            .collect();
        rho.insert(nt.clone(), node_rho);
    }

    rho
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volatility_classification() {
        // Stable: all ρ ≈ 1.0
        let stable = vec![1.0, 1.1, 0.9, 1.05, 0.95];
        assert_eq!(VolatilityRegime::classify(&stable), VolatilityRegime::Low);

        // Volatile: ρ varies widely
        let volatile = vec![0.1, 3.5, 0.2, 5.0, 0.05, 4.0];
        assert_eq!(
            VolatilityRegime::classify(&volatile),
            VolatilityRegime::High
        );

        // Medium
        let medium = vec![0.5, 1.5, 0.8, 1.2, 0.6, 1.4];
        let regime = VolatilityRegime::classify(&medium);
        assert!(
            regime == VolatilityRegime::Medium || regime == VolatilityRegime::Low,
            "Expected Medium or Low, got {:?}",
            regime
        );
    }

    #[test]
    fn test_cross_dep_scaling() {
        assert!((VolatilityRegime::Low.cross_dep_scale() - 1.0).abs() < 1e-6);
        assert!(VolatilityRegime::Medium.cross_dep_scale() > 1.0);
        assert!(
            VolatilityRegime::High.cross_dep_scale() > VolatilityRegime::Medium.cross_dep_scale()
        );
    }

    #[test]
    fn test_detect_regimes() {
        let mut rho = HashMap::new();
        rho.insert("user".to_string(), vec![1.0, 1.1, 0.9]);
        rho.insert("merchant".to_string(), vec![0.1, 5.0, 0.2, 4.0]);

        let regimes = detect_volatility_regimes(&rho);
        assert_eq!(regimes["user"], VolatilityRegime::Low);
        assert_eq!(regimes["merchant"], VolatilityRegime::High);
    }
}
