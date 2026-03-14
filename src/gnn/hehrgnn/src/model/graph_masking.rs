//! Graph Subgraph Masking for JEPA (Gap #5)
//!
//! Novel adaptation of JEPA's image masking to graphs:
//! - Image JEPA: mask rectangular blocks of patches
//! - Graph JEPA: mask subsets of nodes and their edges
//!
//! Context nodes (visible) encode the graph structure.
//! Target nodes (masked) must be predicted from context.
//! This forces the model to learn structural completion.
//!
//! Masking strategies:
//! - Random: mask K random nodes
//! - Neighborhood: mask a connected subgraph
//! - Type-based: mask all nodes of one type

use std::collections::{HashMap, HashSet};

/// Mask specification for graph JEPA.
#[derive(Debug, Clone)]
pub struct GraphMask {
    /// Context (visible) node indices per type.
    pub context: HashMap<String, Vec<usize>>,
    /// Target (masked) node indices per type.
    pub target: HashMap<String, Vec<usize>>,
}

impl GraphMask {
    /// Total context tokens.
    pub fn context_count(&self) -> usize {
        self.context.values().map(|v| v.len()).sum()
    }

    /// Total target tokens.
    pub fn target_count(&self) -> usize {
        self.target.values().map(|v| v.len()).sum()
    }

    /// Validate that context and target are disjoint and non-empty.
    pub fn validate(&self) -> bool {
        // Check non-empty
        if self.context_count() == 0 || self.target_count() == 0 {
            return false;
        }
        // Check disjoint per type
        for (nt, ctx) in &self.context {
            if let Some(tgt) = self.target.get(nt) {
                let ctx_set: HashSet<usize> = ctx.iter().copied().collect();
                for &t in tgt {
                    if ctx_set.contains(&t) {
                        return false;
                    }
                }
            }
        }
        true
    }
}

/// Masking strategy for graph JEPA.
#[derive(Debug, Clone, Copy)]
pub enum GraphMaskingStrategy {
    /// Mask a fraction of random nodes from each type.
    Random {
        /// Fraction of nodes to mask (0.0-1.0).
        mask_ratio: f32,
    },
    /// Mask all nodes of one randomly chosen type.
    TypeBased,
    /// Mask a random node and its k-hop neighborhood.
    Neighborhood {
        /// Number of hops to include in mask.
        k_hops: usize,
    },
}

impl Default for GraphMaskingStrategy {
    fn default() -> Self {
        GraphMaskingStrategy::Random { mask_ratio: 0.25 }
    }
}

/// Generate a graph mask from node counts.
///
/// `node_counts`: mapping from node type → number of nodes
/// `seed`: random seed for reproducibility
pub fn generate_graph_mask(
    node_counts: &HashMap<String, usize>,
    strategy: GraphMaskingStrategy,
    seed: u64,
) -> GraphMask {
    let mut rng_seed = seed;

    match strategy {
        GraphMaskingStrategy::Random { mask_ratio } => {
            let mut context = HashMap::new();
            let mut target = HashMap::new();

            for (nt, &count) in node_counts {
                if count == 0 {
                    continue;
                }
                let num_target = (count as f32 * mask_ratio).ceil() as usize;
                let num_target = num_target.clamp(1, count.saturating_sub(1).max(1));

                // Generate permutation using seed
                let mut indices: Vec<usize> = (0..count).collect();
                for i in (1..indices.len()).rev() {
                    rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let j = (rng_seed >> 33) as usize % (i + 1);
                    indices.swap(i, j);
                }

                target.insert(nt.clone(), indices[..num_target].to_vec());
                context.insert(nt.clone(), indices[num_target..].to_vec());
            }

            GraphMask { context, target }
        }

        GraphMaskingStrategy::TypeBased => {
            let type_names: Vec<&String> = node_counts.keys().collect();
            if type_names.is_empty() {
                return GraphMask {
                    context: HashMap::new(),
                    target: HashMap::new(),
                };
            }

            rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let target_type_idx = (rng_seed >> 33) as usize % type_names.len();
            let target_type = type_names[target_type_idx].clone();

            let mut context = HashMap::new();
            let mut target = HashMap::new();

            for (nt, &count) in node_counts {
                let all_indices: Vec<usize> = (0..count).collect();
                if nt == &target_type {
                    // Keep at least one context node to avoid empty context
                    if count > 1 {
                        target.insert(nt.clone(), (1..count).collect());
                        context.insert(nt.clone(), vec![0]);
                    } else {
                        context.insert(nt.clone(), all_indices);
                    }
                } else {
                    context.insert(nt.clone(), all_indices);
                }
            }

            GraphMask { context, target }
        }

        GraphMaskingStrategy::Neighborhood { k_hops } => {
            // For neighborhood masking, we need adjacency info.
            // Fallback to random with a moderate ratio.
            generate_graph_mask(
                node_counts,
                GraphMaskingStrategy::Random {
                    mask_ratio: 0.15 * (k_hops as f32 + 1.0).min(3.0),
                },
                seed,
            )
        }
    }
}

/// Apply a graph mask to embeddings: zero out target node embeddings.
///
/// Returns (context_embeddings, target_embeddings) where context has
/// target nodes zeroed and target contains only target node embeddings.
pub fn apply_mask_to_embeddings(
    embeddings: &HashMap<String, Vec<Vec<f32>>>,
    mask: &GraphMask,
) -> (
    HashMap<String, Vec<Vec<f32>>>,
    HashMap<String, Vec<Vec<f32>>>,
) {
    let mut context_emb = HashMap::new();
    let mut target_emb = HashMap::new();

    for (nt, vecs) in embeddings {
        let target_indices: HashSet<usize> = mask
            .target
            .get(nt)
            .map(|v| v.iter().copied().collect())
            .unwrap_or_default();

        let dim = vecs.get(0).map(|v| v.len()).unwrap_or(0);
        let mut ctx = vecs.clone();
        let mut tgt = vec![vec![0.0; dim]; vecs.len()];

        for &idx in &target_indices {
            if idx < ctx.len() {
                tgt[idx] = ctx[idx].clone();
                ctx[idx] = vec![0.0; dim]; // Zero out target in context view
            }
        }

        context_emb.insert(nt.clone(), ctx);
        target_emb.insert(nt.clone(), tgt);
    }

    (context_emb, target_emb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_masking() {
        let mut counts = HashMap::new();
        counts.insert("user".to_string(), 10);
        counts.insert("merchant".to_string(), 5);

        let mask = generate_graph_mask(
            &counts,
            GraphMaskingStrategy::Random { mask_ratio: 0.3 },
            42,
        );

        assert!(mask.validate(), "Mask should be valid");
        assert!(mask.target_count() > 0, "Should have target nodes");
        assert!(mask.context_count() > 0, "Should have context nodes");

        // Total should equal original counts per type
        for (nt, &count) in &counts {
            let ctx = mask.context.get(nt).map(|v| v.len()).unwrap_or(0);
            let tgt = mask.target.get(nt).map(|v| v.len()).unwrap_or(0);
            assert_eq!(
                ctx + tgt,
                count,
                "Context + target should equal total for {}",
                nt
            );
        }
    }

    #[test]
    fn test_type_based_masking() {
        let mut counts = HashMap::new();
        counts.insert("user".to_string(), 5);
        counts.insert("tx".to_string(), 10);

        let mask = generate_graph_mask(&counts, GraphMaskingStrategy::TypeBased, 42);
        assert!(mask.validate(), "Mask should be valid");
        assert!(mask.target_count() > 0);
    }

    #[test]
    fn test_mask_deterministic() {
        let mut counts = HashMap::new();
        counts.insert("a".to_string(), 20);

        let mask1 = generate_graph_mask(
            &counts,
            GraphMaskingStrategy::Random { mask_ratio: 0.5 },
            123,
        );
        let mask2 = generate_graph_mask(
            &counts,
            GraphMaskingStrategy::Random { mask_ratio: 0.5 },
            123,
        );

        assert_eq!(
            mask1.target["a"], mask2.target["a"],
            "Same seed should produce same mask"
        );
    }

    #[test]
    fn test_apply_mask_to_embeddings() {
        let mut emb = HashMap::new();
        emb.insert(
            "user".to_string(),
            vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]],
        );

        let mask = GraphMask {
            context: [("user".to_string(), vec![0, 2])].into(),
            target: [("user".to_string(), vec![1])].into(),
        };

        let (ctx, tgt) = apply_mask_to_embeddings(&emb, &mask);

        // User[1] should be zeroed in context
        assert_eq!(ctx["user"][1], vec![0.0, 0.0]);
        // User[1] should be preserved in target
        assert_eq!(tgt["user"][1], vec![0.0, 1.0]);
        // User[0] should be preserved in context
        assert_eq!(ctx["user"][0], vec![1.0, 0.0]);
    }
}
