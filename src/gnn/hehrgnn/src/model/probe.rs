//! RLFR-inspired Embedding Quality Probes.
//!
//! Implements the "Features as Rewards" concept from Goodfire's research:
//! train lightweight probes on frozen model activations, then use probe
//! scores as auxiliary reward signals during SPSA training.
//!
//! Probe types:
//! - NodeTypeProbe: classifies embeddings by node type (measures discriminability)
//! - ClusterSeparationProbe: measures inter-type vs intra-type distances
//!
//! Reference: Goodfire RLFR (arXiv 2602.10067)

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════
// Node Type Classification Probe
// ═══════════════════════════════════════════════════════════════

/// A simple linear probe that classifies embeddings by node type.
///
/// From RLFR: "the model's internals carry important signals that don't
/// appear to be properly used." By probing whether embeddings encode
/// node type, we measure embedding quality without external labels.
///
/// Architecture: single linear layer (hidden_dim → num_types) + softmax.
/// Trained on frozen pre-training embeddings, then used as reward signal.
#[derive(Debug, Clone)]
pub struct NodeTypeProbe {
    /// Weight matrix: [num_types × hidden_dim]
    pub weights: Vec<Vec<f32>>,
    /// Bias vector: [num_types]
    pub bias: Vec<f32>,
    /// Type name → index mapping
    pub type_to_idx: HashMap<String, usize>,
    /// Number of node types
    pub num_types: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
}

impl NodeTypeProbe {
    /// Create a new probe for the given node types and embedding dimension.
    pub fn new(node_types: &[String], hidden_dim: usize) -> Self {
        let num_types = node_types.len();
        let mut type_to_idx = HashMap::new();
        for (i, t) in node_types.iter().enumerate() {
            type_to_idx.insert(t.clone(), i);
        }

        // Xavier initialization
        let scale = (2.0 / (hidden_dim + num_types) as f64).sqrt() as f32;
        let mut seed: u64 = 12345;
        let mut weights = vec![vec![0.0f32; hidden_dim]; num_types];
        for row in &mut weights {
            for w in row.iter_mut() {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u = (seed >> 33) as f32 / (u32::MAX as f32);
                *w = (u - 0.5) * 2.0 * scale;
            }
        }
        let bias = vec![0.0f32; num_types];

        NodeTypeProbe {
            weights,
            bias,
            type_to_idx,
            num_types,
            hidden_dim,
        }
    }

    /// Train the probe on frozen embeddings (from RLFR: probes are trained
    /// on a frozen copy of the model).
    ///
    /// embeddings: node_type → Vec<Vec<f32>> (per-node embedding vectors)
    pub fn train_on_frozen(
        &mut self,
        embeddings: &HashMap<String, Vec<Vec<f32>>>,
        epochs: usize,
        lr: f32,
    ) {
        for _epoch in 0..epochs {
            for (node_type, vecs) in embeddings {
                let target_idx = match self.type_to_idx.get(node_type) {
                    Some(&idx) => idx,
                    None => continue,
                };

                for emb in vecs {
                    if emb.len() != self.hidden_dim {
                        continue;
                    }

                    // Forward: logits = W @ emb + b
                    let mut logits = self.bias.clone();
                    for (c, w_row) in self.weights.iter().enumerate() {
                        for (j, &e) in emb.iter().enumerate() {
                            logits[c] += w_row[j] * e;
                        }
                    }

                    // Softmax
                    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
                    let sum_exp: f32 = exp.iter().sum();
                    let probs: Vec<f32> = exp.iter().map(|e| e / sum_exp).collect();

                    // Cross-entropy gradient: dL/d_logits = probs - one_hot(target)
                    let mut grad_logits = probs.clone();
                    grad_logits[target_idx] -= 1.0;

                    // Update weights: W -= lr * grad_logits ⊗ emb
                    for (c, gl) in grad_logits.iter().enumerate() {
                        for (j, &e) in emb.iter().enumerate() {
                            self.weights[c][j] -= lr * gl * e;
                        }
                        self.bias[c] -= lr * gl;
                    }
                }
            }
        }
    }

    /// Score embeddings: what fraction are correctly classified by node type?
    /// This is the "probe reward" — higher = better embedding quality.
    ///
    /// From RLFR: "probe scores become the reward signal for RL training"
    pub fn score(&self, embeddings: &HashMap<String, Vec<Vec<f32>>>) -> f32 {
        let mut correct = 0usize;
        let mut total = 0usize;

        for (node_type, vecs) in embeddings {
            let target_idx = match self.type_to_idx.get(node_type) {
                Some(&idx) => idx,
                None => continue,
            };

            for emb in vecs {
                if emb.len() != self.hidden_dim {
                    continue;
                }

                // Forward: logits = W @ emb + b
                let mut logits = self.bias.clone();
                for (c, w_row) in self.weights.iter().enumerate() {
                    for (j, &e) in emb.iter().enumerate() {
                        logits[c] += w_row[j] * e;
                    }
                }

                // Argmax = predicted class
                let predicted = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                if predicted == target_idx {
                    correct += 1;
                }
                total += 1;
            }
        }

        if total > 0 {
            correct as f32 / total as f32
        } else {
            0.0
        }
    }

    /// Compute per-type accuracy breakdown.
    pub fn per_type_accuracy(
        &self,
        embeddings: &HashMap<String, Vec<Vec<f32>>>,
    ) -> HashMap<String, f32> {
        let mut result = HashMap::new();

        for (node_type, vecs) in embeddings {
            let target_idx = match self.type_to_idx.get(node_type) {
                Some(&idx) => idx,
                None => continue,
            };

            let mut correct = 0usize;
            let mut total = 0usize;

            for emb in vecs {
                if emb.len() != self.hidden_dim {
                    continue;
                }

                let mut logits = self.bias.clone();
                for (c, w_row) in self.weights.iter().enumerate() {
                    for (j, &e) in emb.iter().enumerate() {
                        logits[c] += w_row[j] * e;
                    }
                }

                let predicted = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                if predicted == target_idx {
                    correct += 1;
                }
                total += 1;
            }

            result.insert(
                node_type.clone(),
                if total > 0 {
                    correct as f32 / total as f32
                } else {
                    0.0
                },
            );
        }

        result
    }
}

// ═══════════════════════════════════════════════════════════════
// Cluster Separation Probe (no training needed)
// ═══════════════════════════════════════════════════════════════

/// Measures how well-separated node type clusters are in embedding space.
///
/// cluster_separation = mean_inter_type_distance / mean_intra_type_distance
/// Higher = embeddings form tighter, more separated clusters = better quality.
pub fn cluster_separation_score(embeddings: &HashMap<String, Vec<Vec<f32>>>) -> f32 {
    let types: Vec<&String> = embeddings.keys().collect();
    if types.len() < 2 {
        return 0.0;
    }

    // Compute centroid per type
    let mut centroids: HashMap<&String, Vec<f32>> = HashMap::new();
    for (nt, vecs) in embeddings {
        if vecs.is_empty() {
            continue;
        }
        let dim = vecs[0].len();
        let mut centroid = vec![0.0f32; dim];
        for v in vecs {
            for (j, &x) in v.iter().enumerate() {
                centroid[j] += x;
            }
        }
        let n = vecs.len() as f32;
        for c in centroid.iter_mut() {
            *c /= n;
        }
        centroids.insert(nt, centroid);
    }

    // Mean intra-type distance (avg distance of each node to its centroid)
    let mut intra_total = 0.0f32;
    let mut intra_count = 0usize;
    for (nt, vecs) in embeddings {
        if let Some(centroid) = centroids.get(nt) {
            for v in vecs {
                let dist: f32 = v
                    .iter()
                    .zip(centroid)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                intra_total += dist;
                intra_count += 1;
            }
        }
    }
    let mean_intra = if intra_count > 0 {
        intra_total / intra_count as f32
    } else {
        1.0
    };

    // Mean inter-type distance (avg distance between centroids)
    let mut inter_total = 0.0f32;
    let mut inter_count = 0usize;
    let centroid_list: Vec<(&&String, &Vec<f32>)> = centroids.iter().collect();
    for i in 0..centroid_list.len() {
        for j in (i + 1)..centroid_list.len() {
            let dist: f32 = centroid_list[i]
                .1
                .iter()
                .zip(centroid_list[j].1.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            inter_total += dist;
            inter_count += 1;
        }
    }
    let mean_inter = if inter_count > 0 {
        inter_total / inter_count as f32
    } else {
        0.0
    };

    // Separation ratio: higher = better
    if mean_intra > 1e-8 {
        mean_inter / mean_intra
    } else {
        mean_inter * 1000.0 // if intra=0, everything is perfectly clustered
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_embeddings() -> HashMap<String, Vec<Vec<f32>>> {
        let mut emb = HashMap::new();
        // Each type has distinct patterns
        emb.insert(
            "user".to_string(),
            vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.9, 0.1, 0.0, 0.0],
                vec![0.8, 0.2, 0.0, 0.0],
            ],
        );
        emb.insert(
            "account".to_string(),
            vec![
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.9, 0.1, 0.0],
                vec![0.0, 0.8, 0.2, 0.0],
            ],
        );
        emb.insert(
            "tx".to_string(),
            vec![
                vec![0.0, 0.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.9, 0.1],
                vec![0.0, 0.0, 0.8, 0.2],
            ],
        );
        emb
    }

    #[test]
    fn test_probe_trains_and_scores() {
        let emb = make_test_embeddings();
        let types: Vec<String> = emb.keys().cloned().collect();

        let mut probe = NodeTypeProbe::new(&types, 4);

        // Before training: should be ~33% (random)
        let score_before = probe.score(&emb);
        println!(
            "  Probe score before training: {:.2}%",
            score_before * 100.0
        );

        // Train on frozen embeddings
        probe.train_on_frozen(&emb, 50, 0.1);

        // After training: should be high (embeddings are separable)
        let score_after = probe.score(&emb);
        println!("  Probe score after training:  {:.2}%", score_after * 100.0);

        assert!(
            score_after > score_before,
            "Probe should improve: {} > {}",
            score_after,
            score_before
        );
        assert!(score_after > 0.8, "Should achieve >80% on separable data");

        // Per-type breakdown
        let per_type = probe.per_type_accuracy(&emb);
        for (t, acc) in &per_type {
            println!("    {}: {:.0}%", t, acc * 100.0);
        }
    }

    #[test]
    fn test_cluster_separation() {
        let emb = make_test_embeddings();
        let sep = cluster_separation_score(&emb);
        println!("  Cluster separation: {:.4}", sep);
        assert!(sep > 1.0, "Well-separated clusters should have ratio > 1");

        // Make bad embeddings (all same)
        let mut bad_emb = HashMap::new();
        bad_emb.insert("a".to_string(), vec![vec![0.5, 0.5, 0.5, 0.5]]);
        bad_emb.insert("b".to_string(), vec![vec![0.5, 0.5, 0.5, 0.5]]);
        let bad_sep = cluster_separation_score(&bad_emb);
        println!("  Bad cluster separation: {:.4}", bad_sep);
        assert!(bad_sep < sep, "Bad embeddings should have lower separation");
    }
}
