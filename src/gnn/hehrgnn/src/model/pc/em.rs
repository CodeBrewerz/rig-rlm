//! Expectation-Maximization parameter learning for Probabilistic Circuits.
//!
//! - **E-step**: Forward + backward pass → accumulate parameter flows
//! - **M-step**: Normalize flows → update sum weights and input distributions
//!
//! Supports mini-batch EM with step_size blending and Laplacian pseudocount.

use std::collections::HashMap;

use super::circuit::CompiledCircuit;
use super::distribution::Distribution;
use super::node::{NodeId, NodeKind};

/// EM training configuration.
#[derive(Debug, Clone)]
pub struct EmConfig {
    /// Step size for mini-batch EM blending (0..1).
    pub step_size: f64,
    /// Laplacian pseudocount for smoothing.
    pub pseudocount: f64,
    /// Number of EM epochs.
    pub epochs: usize,
}

impl Default for EmConfig {
    fn default() -> Self {
        EmConfig {
            step_size: 0.05,
            pseudocount: 0.01,
            epochs: 20,
        }
    }
}

/// Report from EM training.
#[derive(Debug, Clone)]
pub struct EmReport {
    pub epochs_trained: usize,
    pub log_likelihoods: Vec<f64>,
    pub final_ll: f64,
}

/// Accumulated flows for a batch of data.
struct FlowAccumulator {
    /// Sum node → accumulated edge flows.
    sum_flows: HashMap<NodeId, Vec<f64>>,
    /// Input node → accumulated per-category counts.
    input_flows: HashMap<NodeId, Vec<f64>>,
    /// Number of samples accumulated.
    n_samples: usize,
}

impl FlowAccumulator {
    fn new(circuit: &CompiledCircuit) -> Self {
        let mut sum_flows = HashMap::new();
        let mut input_flows = HashMap::new();

        for node in &circuit.nodes {
            match &node.kind {
                NodeKind::Sum { log_weights } => {
                    sum_flows.insert(node.id, vec![0.0; log_weights.len()]);
                }
                NodeKind::Input { dist, .. } => {
                    let k = dist.num_categories().max(1);
                    input_flows.insert(node.id, vec![0.0; k]);
                }
                _ => {}
            }
        }

        FlowAccumulator {
            sum_flows,
            input_flows,
            n_samples: 0,
        }
    }

    fn accumulate(&mut self, circuit: &CompiledCircuit, evidence: &super::circuit::Evidence) {
        // Sum nodes: accumulate edge flows
        let sum_ids: Vec<NodeId> = self.sum_flows.keys().copied().collect();
        for sum_id in &sum_ids {
            let edge_flows = circuit.sum_edge_flows(*sum_id);
            if let Some(acc) = self.sum_flows.get_mut(sum_id) {
                let n = acc.len().min(edge_flows.len());
                for i in 0..n {
                    acc[i] += edge_flows[i];
                }
            }
        }

        // Input nodes: accumulate observed category counts
        for (&input_id, acc) in self.input_flows.iter_mut() {
            let flow = circuit.flow(input_id);
            if let NodeKind::Input { var, .. } = &circuit.nodes[input_id].kind {
                if let Some(Some(x)) = evidence.get(*var) {
                    if *x < acc.len() {
                        acc[*x] += flow;
                    }
                }
            }
        }

        self.n_samples += 1;
    }
}

/// Train the circuit parameters using mini-batch EM.
///
/// `data` is a slice of evidence vectors (one per sample).
pub fn train_em(
    circuit: &mut CompiledCircuit,
    data: &[super::circuit::Evidence],
    config: &EmConfig,
) -> EmReport {
    let mut log_likelihoods = Vec::new();

    for epoch in 0..config.epochs {
        let mut acc = FlowAccumulator::new(circuit);
        let mut epoch_ll = 0.0;

        // E-step: forward + backward on each sample
        for sample in data {
            let ll = circuit.forward(sample);
            epoch_ll += ll;
            circuit.backward();
            acc.accumulate(circuit, sample);
        }

        let avg_ll = epoch_ll / data.len().max(1) as f64;
        log_likelihoods.push(avg_ll);

        // M-step: update parameters from accumulated flows

        // Update sum node weights
        for (&sum_id, flows) in &acc.sum_flows {
            let total: f64 = flows.iter().sum::<f64>() + config.pseudocount * flows.len() as f64;
            if total > 0.0 {
                let new_log_weights: Vec<f64> = flows
                    .iter()
                    .map(|f| ((f + config.pseudocount) / total).ln())
                    .collect();

                // Blend with old weights
                if let NodeKind::Sum { log_weights } = &circuit.nodes[sum_id].kind {
                    let blended: Vec<f64> = log_weights
                        .iter()
                        .zip(new_log_weights.iter())
                        .map(|(&old, &new)| (1.0 - config.step_size) * old + config.step_size * new)
                        .collect();
                    // Renormalize blended weights
                    let log_total = super::circuit::log_sum_exp(&blended);
                    let normalized: Vec<f64> = blended.iter().map(|w| w - log_total).collect();
                    circuit.set_sum_weights(sum_id, normalized);
                }
            }
        }

        // Update input node distributions (categorical only)
        for (&input_id, flows) in &acc.input_flows {
            if let NodeKind::Input { dist, .. } = &circuit.nodes[input_id].kind {
                if let Distribution::Categorical { log_probs } = dist {
                    let total: f64 =
                        flows.iter().sum::<f64>() + config.pseudocount * flows.len() as f64;
                    if total > 0.0 {
                        let new_log_probs: Vec<f64> = flows
                            .iter()
                            .map(|f| ((f + config.pseudocount) / total).ln())
                            .collect();

                        // Blend
                        let blended: Vec<f64> = log_probs
                            .iter()
                            .zip(new_log_probs.iter())
                            .map(|(&old, &new)| {
                                (1.0 - config.step_size) * old + config.step_size * new
                            })
                            .collect();
                        let log_total = super::circuit::log_sum_exp(&blended);
                        let normalized: Vec<f64> = blended.iter().map(|w| w - log_total).collect();
                        circuit.set_input_dist(
                            input_id,
                            Distribution::Categorical {
                                log_probs: normalized,
                            },
                        );
                    }
                }
            }
        }
    }

    let final_ll = *log_likelihoods.last().unwrap_or(&f64::NEG_INFINITY);
    EmReport {
        epochs_trained: config.epochs,
        log_likelihoods,
        final_ll,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::pc::circuit::CompiledCircuit;
    use crate::model::pc::distribution::Distribution;
    use crate::model::pc::node::CircuitBuilder;

    #[test]
    fn test_em_learns_mixture() {
        // True distribution: P(X=0) = 0.7, P(X=1) = 0.3
        // Model: mixture of two components (start uniform)
        let mut b = CircuitBuilder::new();
        let x0a = b.add_input(0, Distribution::uniform_categorical(2));
        let x0b = b.add_input(0, Distribution::uniform_categorical(2));
        let _s = b.add_sum(vec![x0a, x0b]); // uniform mixture

        let mut circuit = CompiledCircuit::compile(&b);

        // Generate data: 70% zeros, 30% ones
        let data: Vec<Vec<Option<usize>>> = (0..100)
            .map(|i| vec![Some(if i < 70 { 0 } else { 1 })])
            .collect();

        let config = EmConfig {
            step_size: 0.3,
            pseudocount: 0.001,
            epochs: 50,
        };
        let report = train_em(&mut circuit, &data, &config);

        // After training, P(X=0) should be ≈ 0.7
        let ll0 = circuit.forward(&vec![Some(0)]);
        let ll1 = circuit.forward(&vec![Some(1)]);
        let p0 = ll0.exp();
        let p1 = ll1.exp();
        println!("Learned: P(X=0) = {:.4}, P(X=1) = {:.4}", p0, p1);
        println!("Final avg LL: {:.4}", report.final_ll);

        assert!(
            (p0 - 0.7).abs() < 0.15,
            "Expected P(X=0) ≈ 0.7, got {:.4}",
            p0
        );
        assert!(
            (p1 - 0.3).abs() < 0.15,
            "Expected P(X=1) ≈ 0.3, got {:.4}",
            p1
        );
    }
}
