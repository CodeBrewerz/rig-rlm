//! Hierarchical Multi-Scale JEPA (Gap #7)
//!
//! From jepa-rs `jepa-world/hierarchy.rs` and LeCun's position paper:
//! stacks multiple JEPA levels at different temporal strides.
//!
//! In the hehrgnn financial graph context:
//! - Level 0 (stride 1): raw transaction-level patterns
//! - Level 1 (stride 6): daily/weekly aggregates
//! - Level 2 (stride 24): monthly/quarterly trends
//!
//! Each level produces representations at a different temporal granularity.
//! Higher levels capture longer-horizon patterns but at lower resolution.

use std::collections::HashMap;

/// A single level in the embedding hierarchy.
#[derive(Debug, Clone)]
pub struct HierarchyLevel {
    /// Temporal stride: how many base-level steps per this level's step.
    pub temporal_stride: usize,
    /// Embedding dimension at this level (can differ from base).
    pub embed_dim: usize,
    /// Aggregation method for pooling from finer to coarser level.
    pub aggregation: AggregationType,
}

/// How to aggregate from one hierarchy level to the next.
#[derive(Debug, Clone, Copy)]
pub enum AggregationType {
    /// Mean of stride-many consecutive embeddings.
    Mean,
    /// Max over stride-many consecutive embeddings.
    Max,
    /// Weighted exponential decay (recent weighted more).
    ExponentialDecay { decay_rate: f32 },
}

/// Hierarchical embedding stack.
#[derive(Debug, Clone)]
pub struct HierarchicalJepa {
    /// Levels from finest (index 0) to coarsest.
    pub levels: Vec<HierarchyLevel>,
}

impl HierarchicalJepa {
    /// Create a 3-level hierarchy with default strides.
    pub fn default_3_level(base_dim: usize) -> Self {
        Self {
            levels: vec![
                HierarchyLevel {
                    temporal_stride: 1,
                    embed_dim: base_dim,
                    aggregation: AggregationType::Mean,
                },
                HierarchyLevel {
                    temporal_stride: 6,
                    embed_dim: base_dim,
                    aggregation: AggregationType::Mean,
                },
                HierarchyLevel {
                    temporal_stride: 24,
                    embed_dim: base_dim,
                    aggregation: AggregationType::ExponentialDecay { decay_rate: 0.9 },
                },
            ],
        }
    }

    /// Create from custom specifications.
    pub fn new(levels: Vec<HierarchyLevel>) -> Self {
        Self { levels }
    }

    /// Number of hierarchy levels.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Effective temporal stride at a given level.
    /// Product of all strides up to and including this level.
    pub fn effective_stride(&self, level_idx: usize) -> usize {
        self.levels[..=level_idx.min(self.levels.len() - 1)]
            .iter()
            .map(|l| l.temporal_stride)
            .product()
    }

    /// Encode a sequence of embeddings through all hierarchy levels.
    ///
    /// Input: sequence of embedding snapshots (one per base timestep).
    /// Output: one representation vector per level.
    pub fn encode_multi_scale(
        &self,
        sequence: &[HashMap<String, Vec<Vec<f32>>>],
    ) -> Vec<HashMap<String, Vec<f32>>> {
        let mut level_outputs = Vec::with_capacity(self.levels.len());

        // Level 0: pool all timesteps at stride 1
        if !self.levels.is_empty() && !sequence.is_empty() {
            level_outputs.push(self.aggregate_level(sequence, &self.levels[0]));
        }

        // Higher levels: aggregate over stride-many chunks
        for level in self.levels.iter().skip(1) {
            if sequence.len() < level.temporal_stride {
                // Not enough data for this stride — use latest
                if let Some(last) = sequence.last() {
                    let pooled = self.pool_single(last);
                    level_outputs.push(pooled);
                }
            } else {
                // Take the last stride-many snapshots
                let start = sequence.len().saturating_sub(level.temporal_stride);
                let chunk = &sequence[start..];
                level_outputs.push(self.aggregate_level(chunk, level));
            }
        }

        level_outputs
    }

    fn aggregate_level(
        &self,
        snapshots: &[HashMap<String, Vec<Vec<f32>>>],
        level: &HierarchyLevel,
    ) -> HashMap<String, Vec<f32>> {
        if snapshots.is_empty() {
            return HashMap::new();
        }

        // Collect all node types
        let node_types: Vec<String> = snapshots[0].keys().cloned().collect();
        let mut result = HashMap::new();

        for nt in &node_types {
            // Collect all embeddings for this node type across snapshots
            let mut all_embs: Vec<Vec<f32>> = Vec::new();
            for snapshot in snapshots {
                if let Some(vecs) = snapshot.get(nt) {
                    for v in vecs {
                        all_embs.push(v.clone());
                    }
                }
            }

            if all_embs.is_empty() || all_embs[0].is_empty() {
                continue;
            }

            let d = all_embs[0].len();
            let aggregated = match level.aggregation {
                AggregationType::Mean => {
                    let mut mean = vec![0.0f32; d];
                    for emb in &all_embs {
                        for (j, &val) in emb.iter().enumerate() {
                            if j < d {
                                mean[j] += val;
                            }
                        }
                    }
                    let n = all_embs.len() as f32;
                    for v in mean.iter_mut() {
                        *v /= n;
                    }
                    mean
                }
                AggregationType::Max => {
                    let mut max_vals = vec![f32::NEG_INFINITY; d];
                    for emb in &all_embs {
                        for (j, &val) in emb.iter().enumerate() {
                            if j < d && val > max_vals[j] {
                                max_vals[j] = val;
                            }
                        }
                    }
                    max_vals
                }
                AggregationType::ExponentialDecay { decay_rate } => {
                    let n = all_embs.len();
                    let mut weighted = vec![0.0f32; d];
                    let mut weight_sum = 0.0f32;
                    for (i, emb) in all_embs.iter().enumerate() {
                        let w = decay_rate.powi((n - 1 - i) as i32);
                        weight_sum += w;
                        for (j, &val) in emb.iter().enumerate() {
                            if j < d {
                                weighted[j] += w * val;
                            }
                        }
                    }
                    if weight_sum > 1e-8 {
                        for v in weighted.iter_mut() {
                            *v /= weight_sum;
                        }
                    }
                    weighted
                }
            };

            result.insert(nt.clone(), aggregated);
        }

        result
    }

    fn pool_single(&self, snapshot: &HashMap<String, Vec<Vec<f32>>>) -> HashMap<String, Vec<f32>> {
        let mut result = HashMap::new();
        for (nt, vecs) in snapshot {
            if vecs.is_empty() || vecs[0].is_empty() {
                continue;
            }
            let d = vecs[0].len();
            let mut mean = vec![0.0f32; d];
            for v in vecs {
                for (j, &val) in v.iter().enumerate() {
                    if j < d {
                        mean[j] += val;
                    }
                }
            }
            let n = vecs.len() as f32;
            for v in mean.iter_mut() {
                *v /= n;
            }
            result.insert(nt.clone(), mean);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot(val: f32) -> HashMap<String, Vec<Vec<f32>>> {
        let mut m = HashMap::new();
        m.insert("user".to_string(), vec![vec![val, val * 2.0]]);
        m.insert("tx".to_string(), vec![vec![val * 0.5, val * 1.5]]);
        m
    }

    #[test]
    fn test_default_3_level() {
        let h = HierarchicalJepa::default_3_level(64);
        assert_eq!(h.num_levels(), 3);
        assert_eq!(h.effective_stride(0), 1);
        assert_eq!(h.effective_stride(1), 6);
        assert_eq!(h.effective_stride(2), 144); // 1*6*24
    }

    #[test]
    fn test_encode_multi_scale() {
        let h = HierarchicalJepa::default_3_level(2);
        let sequence: Vec<_> = (0..10).map(|i| make_snapshot(i as f32)).collect();

        let outputs = h.encode_multi_scale(&sequence);
        assert_eq!(outputs.len(), 3, "Should have 3 level outputs");

        // Each level should have user and tx types
        for (i, out) in outputs.iter().enumerate() {
            assert!(out.contains_key("user"), "Level {} missing user", i);
            assert!(out.contains_key("tx"), "Level {} missing tx", i);
        }
    }

    #[test]
    fn test_effective_stride_single_level() {
        let h = HierarchicalJepa::new(vec![HierarchyLevel {
            temporal_stride: 5,
            embed_dim: 32,
            aggregation: AggregationType::Mean,
        }]);
        assert_eq!(h.effective_stride(0), 5);
    }

    #[test]
    fn test_mean_aggregation() {
        let h = HierarchicalJepa::new(vec![HierarchyLevel {
            temporal_stride: 1,
            embed_dim: 2,
            aggregation: AggregationType::Mean,
        }]);

        let s1 = make_snapshot(0.0); // user: [0, 0]
        let s2 = make_snapshot(2.0); // user: [2, 4]
        let outputs = h.encode_multi_scale(&[s1, s2]);

        // Mean of [0,0] and [2,4] = [1,2]
        let user_out = &outputs[0]["user"];
        assert!(
            (user_out[0] - 1.0).abs() < 0.01,
            "Mean[0] = {}",
            user_out[0]
        );
        assert!(
            (user_out[1] - 2.0).abs() < 0.01,
            "Mean[1] = {}",
            user_out[1]
        );
    }
}
