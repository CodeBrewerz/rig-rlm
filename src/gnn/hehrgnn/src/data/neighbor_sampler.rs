//! Neighbor sampling for mini-batch GNN training.
//!
//! Implements layer-wise typed neighbor sampling (like PyG's `NeighborLoader`).
//! For each seed node, samples a fixed number of neighbors per edge type per
//! GNN layer, building a subgraph suitable for message passing.

use rand::seq::SliceRandom;
use std::collections::{HashMap, HashSet};

use burn::prelude::*;

use super::hetero_graph::{EdgeType, HeteroGraph, MiniBatchGraph, NodeType};

/// Configuration for neighbor sampling.
#[derive(Debug, Clone)]
pub struct NeighborSampleConfig {
    /// Number of neighbors to sample per edge type, per layer.
    /// Length = number of GNN layers.
    /// Each entry maps edge_type → max neighbors.
    /// If an edge type is not in the map, all neighbors are used.
    pub fanout_per_layer: Vec<HashMap<EdgeType, usize>>,

    /// Default fanout if edge type not specified in the map.
    pub default_fanout: usize,
}

impl Default for NeighborSampleConfig {
    fn default() -> Self {
        Self {
            fanout_per_layer: vec![HashMap::new(), HashMap::new()], // 2 layers
            default_fanout: 10,
        }
    }
}

impl NeighborSampleConfig {
    /// Create a uniform config: same fanout for all edge types across layers.
    pub fn uniform(num_layers: usize, fanout: usize) -> Self {
        Self {
            fanout_per_layer: vec![HashMap::new(); num_layers],
            default_fanout: fanout,
        }
    }
}

/// Neighbor sampler for heterogeneous graphs.
pub struct NeighborSampler;

impl NeighborSampler {
    /// Sample a mini-batch subgraph from a heterogeneous graph.
    ///
    /// Starting from `seed_nodes` (per type), expands outward layer by layer,
    /// sampling up to `fanout` neighbors per edge type at each layer.
    ///
    /// Returns a `MiniBatchGraph` with:
    /// - Re-indexed subgraph
    /// - Mapping from local → original node IDs
    /// - Seed node mask
    pub fn sample<B: Backend>(
        graph: &HeteroGraph<B>,
        seed_nodes: &HashMap<NodeType, Vec<usize>>,
        config: &NeighborSampleConfig,
        device: &B::Device,
    ) -> MiniBatchGraph<B> {
        let mut rng = rand::rng();

        // Track all nodes that need to be included, per type
        let mut included_nodes: HashMap<NodeType, HashSet<usize>> = HashMap::new();

        // Initialize with seed nodes
        for (node_type, seeds) in seed_nodes {
            let set = included_nodes.entry(node_type.clone()).or_default();
            for &s in seeds {
                set.insert(s);
            }
        }

        // Pre-build adjacency lists for efficient lookup
        let mut adj: HashMap<EdgeType, HashMap<usize, Vec<usize>>> = HashMap::new();
        for et in graph.edge_types() {
            if let Some((src_vec, dst_vec)) = graph.edges_as_vecs(et) {
                let map = adj.entry(et.clone()).or_default();
                for (s, d) in src_vec.iter().zip(dst_vec.iter()) {
                    map.entry(*d as usize).or_default().push(*s as usize);
                }
            }
        }

        // Layer-wise expansion (from seed layer outward)
        for layer_idx in (0..config.fanout_per_layer.len()).rev() {
            let layer_config = &config.fanout_per_layer[layer_idx];

            // Collect current frontier nodes (nodes that need neighbors)
            let frontier: HashMap<NodeType, Vec<usize>> = included_nodes
                .iter()
                .map(|(nt, set)| (nt.clone(), set.iter().copied().collect()))
                .collect();

            // For each edge type, sample neighbors of frontier dst nodes
            for et in graph.edge_types() {
                let dst_type = &et.2;
                let src_type = &et.0;

                let fanout = layer_config
                    .get(et)
                    .copied()
                    .unwrap_or(config.default_fanout);

                if let Some(adj_map) = adj.get(et) {
                    if let Some(frontier_nodes) = frontier.get(dst_type) {
                        let src_set = included_nodes.entry(src_type.clone()).or_default();

                        for &node_idx in frontier_nodes {
                            if let Some(neighbors) = adj_map.get(&node_idx) {
                                // Sample up to `fanout` neighbors
                                let sampled: Vec<usize> = if neighbors.len() <= fanout {
                                    neighbors.clone()
                                } else {
                                    let mut shuffled = neighbors.clone();
                                    shuffled.shuffle(&mut rng);
                                    shuffled.truncate(fanout);
                                    shuffled
                                };

                                for s in sampled {
                                    src_set.insert(s);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Build the subgraph with re-indexed nodes
        Self::build_subgraph(graph, &included_nodes, seed_nodes, device)
    }

    /// Build a MiniBatchGraph from the included nodes.
    fn build_subgraph<B: Backend>(
        graph: &HeteroGraph<B>,
        included_nodes: &HashMap<NodeType, HashSet<usize>>,
        seed_nodes: &HashMap<NodeType, Vec<usize>>,
        device: &B::Device,
    ) -> MiniBatchGraph<B> {
        let mut sub_graph = HeteroGraph::new();
        let mut original_ids: HashMap<NodeType, Vec<usize>> = HashMap::new();
        let mut seed_mask: HashMap<NodeType, Vec<bool>> = HashMap::new();

        // Re-index nodes: original_idx → local_idx
        let mut reindex: HashMap<NodeType, HashMap<usize, usize>> = HashMap::new();

        for (node_type, node_set) in included_nodes {
            let mut sorted: Vec<usize> = node_set.iter().copied().collect();
            sorted.sort();

            let local_map: HashMap<usize, usize> = sorted
                .iter()
                .enumerate()
                .map(|(local, &orig)| (orig, local))
                .collect();

            // Extract features for included nodes
            if let Some(full_features) = graph.node_features.get(node_type) {
                let indices: Vec<i64> = sorted.iter().map(|&i| i as i64).collect();
                let idx_tensor = Tensor::<B, 1, Int>::from_data(indices.as_slice(), device);
                let sub_features = full_features.clone().select(0, idx_tensor);
                sub_graph.add_node_type(node_type, sub_features);
            }

            // Build seed mask
            let seeds = seed_nodes.get(node_type);
            let mask: Vec<bool> = sorted
                .iter()
                .map(|orig| seeds.map(|s| s.contains(orig)).unwrap_or(false))
                .collect();

            original_ids.insert(node_type.clone(), sorted);
            seed_mask.insert(node_type.clone(), mask);
            reindex.insert(node_type.clone(), local_map);
        }

        // Re-index edges
        for et in graph.edge_types() {
            let src_type = &et.0;
            let dst_type = &et.2;

            if let (Some(src_map), Some(dst_map)) = (reindex.get(src_type), reindex.get(dst_type)) {
                if let Some((orig_src, orig_dst)) = graph.edges_as_vecs(et) {
                    let mut new_src = Vec::new();
                    let mut new_dst = Vec::new();

                    for (s, d) in orig_src.iter().zip(orig_dst.iter()) {
                        if let (Some(&ls), Some(&ld)) =
                            (src_map.get(&(*s as usize)), dst_map.get(&(*d as usize)))
                        {
                            new_src.push(ls as i64);
                            new_dst.push(ld as i64);
                        }
                    }

                    if !new_src.is_empty() {
                        let num_edges = new_src.len();
                        let mut flat = Vec::with_capacity(2 * num_edges);
                        flat.extend_from_slice(&new_src);
                        flat.extend_from_slice(&new_dst);

                        let edge_idx = Tensor::<B, 1, Int>::from_data(flat.as_slice(), device)
                            .reshape([2, num_edges]);

                        sub_graph.add_edge_type(et.clone(), edge_idx);
                    }
                }
            }
        }

        MiniBatchGraph {
            graph: sub_graph,
            original_ids,
            seed_mask,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::graph_builder::{GraphBuildConfig, GraphFact, build_hetero_graph};
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    fn build_test_graph() -> HeteroGraph<TestBackend> {
        let device = <TestBackend as Backend>::Device::default();

        let facts = vec![
            // user0 → acct0, user0 → acct1
            GraphFact {
                src: ("user".into(), "u0".into()),
                relation: "owns".into(),
                dst: ("account".into(), "a0".into()),
            },
            GraphFact {
                src: ("user".into(), "u0".into()),
                relation: "owns".into(),
                dst: ("account".into(), "a1".into()),
            },
            // user1 → acct2
            GraphFact {
                src: ("user".into(), "u1".into()),
                relation: "owns".into(),
                dst: ("account".into(), "a2".into()),
            },
            // tx0 → acct0, tx0 → merchant0
            GraphFact {
                src: ("tx".into(), "t0".into()),
                relation: "posted".into(),
                dst: ("account".into(), "a0".into()),
            },
            GraphFact {
                src: ("tx".into(), "t0".into()),
                relation: "at_merchant".into(),
                dst: ("merchant".into(), "m0".into()),
            },
            // tx1 → acct1, tx1 → merchant1
            GraphFact {
                src: ("tx".into(), "t1".into()),
                relation: "posted".into(),
                dst: ("account".into(), "a1".into()),
            },
            GraphFact {
                src: ("tx".into(), "t1".into()),
                relation: "at_merchant".into(),
                dst: ("merchant".into(), "m1".into()),
            },
            // tx2 → acct2, tx2 → merchant0
            GraphFact {
                src: ("tx".into(), "t2".into()),
                relation: "posted".into(),
                dst: ("account".into(), "a2".into()),
            },
            GraphFact {
                src: ("tx".into(), "t2".into()),
                relation: "at_merchant".into(),
                dst: ("merchant".into(), "m0".into()),
            },
        ];

        let config = GraphBuildConfig {
            node_feat_dim: 4,
            add_reverse_edges: true,
            add_self_loops: false,
            add_positional_encoding: true,
            add_cross_dependency_edges: true,
        };

        build_hetero_graph(&facts, &config, &device)
    }

    #[test]
    fn test_neighbor_sampling_basic() {
        let graph = build_test_graph();
        let device = <TestBackend as Backend>::Device::default();

        // Seed: just tx node t0
        let mut seed_nodes = HashMap::new();
        seed_nodes.insert("tx".to_string(), vec![0]); // t0

        let config = NeighborSampleConfig::uniform(1, 5);
        let batch = NeighborSampler::sample(&graph, &seed_nodes, &config, &device);

        // t0 is included
        assert!(batch.original_ids.contains_key("tx"));
        assert!(batch.original_ids["tx"].contains(&0));

        // seed mask should mark t0 as seed
        assert!(batch.seed_mask["tx"][0]);

        println!(
            "Sampled subgraph: {} nodes, {} edges",
            batch.graph.total_nodes(),
            batch.graph.total_edges()
        );
    }

    #[test]
    fn test_neighbor_sampling_preserves_features() {
        let graph = build_test_graph();
        let device = <TestBackend as Backend>::Device::default();

        let mut seed_nodes = HashMap::new();
        seed_nodes.insert("account".to_string(), vec![0, 1]);

        let config = NeighborSampleConfig::uniform(2, 10);
        let batch = NeighborSampler::sample(&graph, &seed_nodes, &config, &device);

        // Check that feature dimensions are preserved
        for (nt, feat) in &batch.graph.node_features {
            let feat_dim = feat.dims()[1];
            let expected = graph.node_feat_dims[nt];
            assert_eq!(feat_dim, expected, "Feature dim mismatch for {}", nt);
        }
    }
}
