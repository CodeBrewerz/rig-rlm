//! Execution Probability Prediction Head
//!
//! From the queue-reactive LOB paper (Huang et al. 2014):
//! models the probability that an order at position n in queue of size Q
//! will execute before the queue depletes.
//!
//! In the hehrgnn financial graph context, this translates to predicting
//! P(action succeeds | graph state) — e.g., will a transaction go through,
//! will a goal be met, will a fiduciary recommendation be accepted.
//!
//! Architecture: 2-layer MLP with sigmoid output
//!   embed_dim → hidden → 1 (sigmoid)
//!
//! Used as an auxiliary prediction task during training for multi-task
//! regularization, which forces embeddings to encode actionability signals.

use std::collections::HashMap;

/// Configuration for the execution probability head.
#[derive(Debug, Clone)]
pub struct ExecProbConfig {
    /// Input embedding dimension (must match GNN hidden_dim).
    pub embed_dim: usize,
    /// Hidden layer dimension.
    pub hidden_dim: usize,
}

impl Default for ExecProbConfig {
    fn default() -> Self {
        Self {
            embed_dim: 32,
            hidden_dim: 16,
        }
    }
}

/// Execution probability prediction head.
///
/// A lightweight 2-layer MLP that maps node embeddings to fill probabilities.
/// Weights are stored as plain f32 vectors for compatibility with the
/// perturbation-based training (no autograd needed).
pub struct ExecProbHead {
    /// First layer weights: [hidden_dim × embed_dim]
    w1: Vec<Vec<f32>>,
    /// First layer bias: [hidden_dim]
    b1: Vec<f32>,
    /// Second layer weights: [1 × hidden_dim]
    w2: Vec<f32>,
    /// Second layer bias: scalar
    b2: f32,
    pub config: ExecProbConfig,
}

impl ExecProbHead {
    /// Create a new execution probability head with Xavier initialization.
    pub fn new(config: ExecProbConfig) -> Self {
        let mut seed: u64 = 42;
        let scale1 = (2.0 / (config.embed_dim + config.hidden_dim) as f64).sqrt() as f32;
        let scale2 = (2.0 / (config.hidden_dim + 1) as f64).sqrt() as f32;

        let w1: Vec<Vec<f32>> = (0..config.hidden_dim)
            .map(|_| {
                (0..config.embed_dim)
                    .map(|_| next_f32(&mut seed) * scale1)
                    .collect()
            })
            .collect();

        let b1 = vec![0.0; config.hidden_dim];

        let w2: Vec<f32> = (0..config.hidden_dim)
            .map(|_| next_f32(&mut seed) * scale2)
            .collect();

        let b2 = 0.0;

        Self {
            w1,
            b1,
            w2,
            b2,
            config,
        }
    }

    /// Predict execution probability for a single node embedding.
    ///
    /// Returns P(execution) ∈ [0, 1].
    pub fn predict(&self, embedding: &[f32]) -> f32 {
        // Layer 1: hidden = ReLU(W1 · x + b1)
        let hidden: Vec<f32> = self
            .w1
            .iter()
            .zip(self.b1.iter())
            .map(|(w_row, &bias)| {
                let z: f32 = w_row
                    .iter()
                    .zip(embedding.iter())
                    .map(|(&w, &x)| w * x)
                    .sum::<f32>()
                    + bias;
                z.max(0.0) // ReLU
            })
            .collect();

        // Layer 2: logit = W2 · hidden + b2
        let logit: f32 = self
            .w2
            .iter()
            .zip(hidden.iter())
            .map(|(&w, &h)| w * h)
            .sum::<f32>()
            + self.b2;

        // Sigmoid
        sigmoid(logit)
    }

    /// Predict execution probabilities for all nodes of all types.
    ///
    /// Returns: HashMap<node_type, Vec<f32>> where each f32 is P(exec) for that node.
    pub fn predict_all(
        &self,
        embeddings: &HashMap<String, Vec<Vec<f32>>>,
    ) -> HashMap<String, Vec<f32>> {
        let mut result = HashMap::new();
        for (node_type, node_embs) in embeddings {
            let probs: Vec<f32> = node_embs.iter().map(|emb| self.predict(emb)).collect();
            result.insert(node_type.clone(), probs);
        }
        result
    }

    /// Compute binary cross-entropy loss for execution probability prediction.
    ///
    /// Labels are generated from graph structure:
    /// - Nodes with many connections (high degree) → high exec probability
    /// - Isolated nodes → low exec probability
    ///
    /// This is a self-supervised signal: we don't need ground-truth labels.
    pub fn compute_loss(
        &self,
        embeddings: &HashMap<String, Vec<Vec<f32>>>,
        node_degrees: &HashMap<String, Vec<f32>>,
    ) -> f32 {
        let mut total_loss = 0.0f32;
        let mut count = 0usize;

        for (node_type, node_embs) in embeddings {
            let degrees = match node_degrees.get(node_type) {
                Some(d) => d,
                None => continue,
            };

            // Compute max degree for normalization
            let max_deg = degrees.iter().copied().fold(0.0f32, f32::max).max(1.0);

            for (i, emb) in node_embs.iter().enumerate() {
                if i >= degrees.len() {
                    break;
                }

                let pred = self.predict(emb);
                // Label: normalized degree as execution probability proxy
                // High-degree nodes are more "connected" and thus more likely
                // to have their actions execute (analogous to queue position).
                let label = (degrees[i] / max_deg).clamp(0.01, 0.99);

                // Binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
                let bce =
                    -(label * pred.max(1e-7).ln() + (1.0 - label) * (1.0 - pred).max(1e-7).ln());
                total_loss += bce;
                count += 1;
            }
        }

        if count > 0 {
            total_loss / count as f32
        } else {
            0.0
        }
    }

    /// Get total number of parameters (for perturbation training).
    pub fn num_params(&self) -> usize {
        self.config.hidden_dim * self.config.embed_dim // w1
            + self.config.hidden_dim                   // b1
            + self.config.hidden_dim                   // w2
            + 1 // b2
    }

    /// Perturb all weights by a small delta (for SPSA gradient estimation).
    pub fn perturb(&mut self, delta: f32, seed: &mut u64) {
        for row in &mut self.w1 {
            for w in row.iter_mut() {
                *w += if next_u64(seed) % 2 == 0 {
                    delta
                } else {
                    -delta
                };
            }
        }
        for b in &mut self.b1 {
            *b += if next_u64(seed) % 2 == 0 {
                delta
            } else {
                -delta
            };
        }
        for w in &mut self.w2 {
            *w += if next_u64(seed) % 2 == 0 {
                delta
            } else {
                -delta
            };
        }
        self.b2 += if next_u64(seed) % 2 == 0 {
            delta
        } else {
            -delta
        };
    }

    /// Clone weights from another head (for restoring after perturbation).
    pub fn copy_weights_from(&mut self, other: &ExecProbHead) {
        self.w1 = other.w1.clone();
        self.b1 = other.b1.clone();
        self.w2 = other.w2.clone();
        self.b2 = other.b2;
    }

    /// Compute node degrees from embeddings hash (by counting edge appearances).
    /// This is a utility to generate self-supervised labels.
    pub fn compute_degrees_from_graph<B: burn::prelude::Backend>(
        graph: &crate::data::hetero_graph::HeteroGraph<B>,
    ) -> HashMap<String, Vec<f32>> {
        let mut degrees: HashMap<String, Vec<f32>> = HashMap::new();

        for (nt, &count) in &graph.node_counts {
            degrees.insert(nt.clone(), vec![0.0; count]);
        }

        for (et, _) in &graph.edge_index {
            let (src_type, _, dst_type) = et;
            if let Some((src_vec, dst_vec)) = graph.edges_as_vecs(et) {
                if let Some(deg) = degrees.get_mut(src_type) {
                    for &s in &src_vec {
                        let s = s as usize;
                        if s < deg.len() {
                            deg[s] += 1.0;
                        }
                    }
                }
                if let Some(deg) = degrees.get_mut(dst_type) {
                    for &d in &dst_vec {
                        let d = d as usize;
                        if d < deg.len() {
                            deg[d] += 1.0;
                        }
                    }
                }
            }
        }

        degrees
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn next_u64(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    *seed
}

fn next_f32(seed: &mut u64) -> f32 {
    let u = next_u64(seed);
    // Convert to [-1, 1] range
    (u as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exec_prob_head_basic() {
        let config = ExecProbConfig {
            embed_dim: 8,
            hidden_dim: 4,
        };
        let head = ExecProbHead::new(config);

        // Test prediction is in [0, 1]
        let embedding = vec![0.1, 0.2, -0.3, 0.4, 0.5, -0.1, 0.3, 0.2];
        let prob = head.predict(&embedding);
        assert!(
            prob >= 0.0 && prob <= 1.0,
            "Prob should be in [0,1], got {}",
            prob
        );
    }

    #[test]
    fn test_exec_prob_loss() {
        let config = ExecProbConfig {
            embed_dim: 4,
            hidden_dim: 2,
        };
        let head = ExecProbHead::new(config);

        let mut embeddings = HashMap::new();
        embeddings.insert(
            "user".to_string(),
            vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
            ],
        );

        let mut degrees = HashMap::new();
        degrees.insert("user".to_string(), vec![5.0, 2.0, 1.0]);

        let loss = head.compute_loss(&embeddings, &degrees);
        assert!(loss.is_finite(), "Loss should be finite, got {}", loss);
        assert!(loss >= 0.0, "BCE loss should be non-negative, got {}", loss);
    }

    #[test]
    fn test_exec_prob_perturbation() {
        let config = ExecProbConfig {
            embed_dim: 4,
            hidden_dim: 2,
        };
        let original = ExecProbHead::new(config.clone());
        let mut perturbed = ExecProbHead::new(config);
        perturbed.copy_weights_from(&original);

        let emb = vec![0.5, 0.5, 0.5, 0.5];
        let before = perturbed.predict(&emb);

        let mut seed = 123u64;
        perturbed.perturb(0.01, &mut seed);
        let after = perturbed.predict(&emb);

        // Predictions should differ after perturbation
        assert!(
            (before - after).abs() > 1e-6,
            "Perturbation should change predictions"
        );
    }
}
