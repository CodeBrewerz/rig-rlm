//! Graph builder: constructs HeteroGraph from TQL schema and facts.
//!
//! The key insight is the **transaction-as-node** pattern: each hyperedge
//! (a transaction touching multiple entities) becomes a node connected
//! to its participants via typed edges. This turns the hypergraph into
//! a standard heterogeneous graph.

use burn::prelude::*;
use std::collections::HashMap;

use super::hetero_graph::{EdgeType, HeteroGraph, NodeType};
use super::synthetic::TqlSchema;

/// Configuration for building a HeteroGraph from schema + data.
#[derive(Debug, Clone)]
pub struct GraphBuildConfig {
    /// Feature dimension for all node types.
    pub node_feat_dim: usize,
    /// Whether to add reverse edges (for undirected message passing).
    pub add_reverse_edges: bool,
    /// Whether to add self-loops for each node type.
    pub add_self_loops: bool,
    /// Whether to add structural positional encoding (node-type ID, in/out degree).
    /// Adds 3 extra dimensions to node features. (KumoRFM §2.3)
    pub add_positional_encoding: bool,
}

impl Default for GraphBuildConfig {
    fn default() -> Self {
        Self {
            node_feat_dim: 16,
            add_reverse_edges: true,
            add_self_loops: true,
            add_positional_encoding: true,
        }
    }
}

/// A fact suitable for graph building — connects entities via a named relation.
#[derive(Debug, Clone)]
pub struct GraphFact {
    /// Source entity: (type, instance_name)
    pub src: (String, String),
    /// Relation name
    pub relation: String,
    /// Destination entity: (type, instance_name)
    pub dst: (String, String),
}

/// Builds a `HeteroGraph` from a set of graph facts.
///
/// Steps:
/// 1. Collect all unique (type, instance) pairs → assign local indices per type
/// 2. Initialize node features (random or one-hot encoded)
/// 3. Build edge index tensors per (src_type, rel, dst_type)
/// 4. Optionally add reverse edges and self-loops
pub fn build_hetero_graph<B: Backend>(
    facts: &[GraphFact],
    config: &GraphBuildConfig,
    device: &B::Device,
) -> HeteroGraph<B> {
    let mut graph = HeteroGraph::new();

    // Step 1: Collect unique entities per type
    let mut entity_to_local: HashMap<NodeType, HashMap<String, usize>> = HashMap::new();

    for fact in facts {
        let src_map = entity_to_local.entry(fact.src.0.clone()).or_default();
        let src_len = src_map.len();
        src_map.entry(fact.src.1.clone()).or_insert(src_len);

        let dst_map = entity_to_local.entry(fact.dst.0.clone()).or_default();
        let dst_len = dst_map.len();
        dst_map.entry(fact.dst.1.clone()).or_insert(dst_len);
    }

    // Step 2: Initialize node features (random initialization)
    for (node_type, instances) in &entity_to_local {
        let num_nodes = instances.len();
        let features = Tensor::random(
            [num_nodes, config.node_feat_dim],
            burn::tensor::Distribution::Default,
            device,
        );
        graph.add_node_type(node_type, features);
    }

    // Step 3: Build edge indices per edge type
    let mut edge_accumulator: HashMap<EdgeType, (Vec<i64>, Vec<i64>)> = HashMap::new();

    for fact in facts {
        let src_idx = entity_to_local[&fact.src.0][&fact.src.1] as i64;
        let dst_idx = entity_to_local[&fact.dst.0][&fact.dst.1] as i64;

        let et: EdgeType = (
            fact.src.0.clone(),
            fact.relation.clone(),
            fact.dst.0.clone(),
        );
        let entry = edge_accumulator
            .entry(et)
            .or_insert_with(|| (Vec::new(), Vec::new()));
        entry.0.push(src_idx);
        entry.1.push(dst_idx);
    }

    // Convert accumulated edges to tensors
    let mut reverse_edges: Vec<(EdgeType, Vec<i64>, Vec<i64>)> = Vec::new();

    for (et, (src_vec, dst_vec)) in &edge_accumulator {
        let num_edges = src_vec.len();
        let mut flat = Vec::with_capacity(2 * num_edges);
        flat.extend_from_slice(src_vec);
        flat.extend_from_slice(dst_vec);

        let edge_idx =
            Tensor::<B, 1, Int>::from_data(flat.as_slice(), device).reshape([2, num_edges]);

        graph.add_edge_type(et.clone(), edge_idx);

        // Collect reverse edges
        if config.add_reverse_edges {
            let rev_et = (et.2.clone(), format!("rev_{}", et.1), et.0.clone());
            reverse_edges.push((rev_et, dst_vec.clone(), src_vec.clone()));
        }
    }

    // Step 4a: Add reverse edges
    for (rev_et, src_vec, dst_vec) in reverse_edges {
        let num_edges = src_vec.len();
        let mut flat = Vec::with_capacity(2 * num_edges);
        flat.extend_from_slice(&src_vec);
        flat.extend_from_slice(&dst_vec);

        let edge_idx =
            Tensor::<B, 1, Int>::from_data(flat.as_slice(), device).reshape([2, num_edges]);

        graph.add_edge_type(rev_et, edge_idx);
    }

    // Step 4b: Add self-loops
    if config.add_self_loops {
        for (node_type, count) in &graph.node_counts.clone() {
            let self_loop_et: EdgeType = (node_type.clone(), "self_loop".into(), node_type.clone());
            let indices: Vec<i64> = (0..*count as i64).collect();
            let mut flat = Vec::with_capacity(2 * *count);
            flat.extend_from_slice(&indices);
            flat.extend_from_slice(&indices);

            let edge_idx =
                Tensor::<B, 1, Int>::from_data(flat.as_slice(), device).reshape([2, *count]);

            graph.add_edge_type(self_loop_et, edge_idx);
        }
    }

    // Step 5: Add structural positional encoding (KumoRFM §2.3)
    // Concatenate [node_type_id, log(in_degree+1), log(out_degree+1)] to node features
    if config.add_positional_encoding {
        // Assign a type index to each node type (sorted alphabetically for determinism)
        let mut sorted_types: Vec<&NodeType> = graph.node_counts.keys().collect();
        sorted_types.sort();
        let num_types = sorted_types.len().max(1) as f32;
        let type_idx: HashMap<&str, f32> = sorted_types
            .iter()
            .enumerate()
            .map(|(i, nt)| (nt.as_str(), i as f32 / num_types))
            .collect();

        // Compute in-degree and out-degree per (node_type, node_idx)
        let mut in_degree: HashMap<String, Vec<f32>> = HashMap::new();
        let mut out_degree: HashMap<String, Vec<f32>> = HashMap::new();

        for (nt, &count) in &graph.node_counts {
            in_degree.insert(nt.clone(), vec![0.0; count]);
            out_degree.insert(nt.clone(), vec![0.0; count]);
        }

        for (et, edge_idx) in &graph.edge_index {
            let (src_type, _, dst_type) = et;
            if let Some((src_vec, dst_vec)) = graph.edges_as_vecs(et) {
                // Out-degree for source nodes
                if let Some(out_deg) = out_degree.get_mut(src_type) {
                    for &s in &src_vec {
                        let s = s as usize;
                        if s < out_deg.len() {
                            out_deg[s] += 1.0;
                        }
                    }
                }
                // In-degree for destination nodes
                if let Some(in_deg) = in_degree.get_mut(dst_type) {
                    for &d in &dst_vec {
                        let d = d as usize;
                        if d < in_deg.len() {
                            in_deg[d] += 1.0;
                        }
                    }
                }
                let _ = edge_idx; // suppress unused warning
            }
        }

        // Concatenate positional features to existing node features
        for (nt, feat) in &graph.node_features.clone() {
            let count = graph.node_counts[nt];
            let tid = *type_idx.get(nt.as_str()).unwrap_or(&0.0);
            let in_deg = in_degree
                .get(nt)
                .cloned()
                .unwrap_or_else(|| vec![0.0; count]);
            let out_deg = out_degree
                .get(nt)
                .cloned()
                .unwrap_or_else(|| vec![0.0; count]);

            // Build positional features: [type_id, log(in_deg+1), log(out_deg+1)]
            let mut pe_data = Vec::with_capacity(count * 3);
            for i in 0..count {
                pe_data.push(tid);
                pe_data.push((in_deg[i] + 1.0).ln());
                pe_data.push((out_deg[i] + 1.0).ln());
            }

            let pe_tensor: Tensor<B, 2> =
                Tensor::<B, 1>::from_data(pe_data.as_slice(), device).reshape([count, 3]);

            // Concatenate: [original_features | positional_encoding]
            let new_feat = Tensor::cat(vec![feat.clone(), pe_tensor], 1);
            let new_dim = config.node_feat_dim + 3;

            graph.node_features.insert(nt.clone(), new_feat);
            graph.node_feat_dims.insert(nt.clone(), new_dim);
        }
    }

    graph
}

/// Convert synthetic raw facts from TQL into GraphFacts using the
/// transaction-as-node pattern.
///
/// Each raw fact `(head, relation, tail, qualifiers)` becomes:
/// - A "transaction" node (synthesized)
/// - Edges: transaction --[rel_from]--> head, transaction --[rel_to]--> tail
/// - For each qualifier: transaction --[qual_rel]--> qual_entity
pub fn raw_facts_to_graph_facts(raw_facts: &[crate::data::fact::RawFact]) -> Vec<GraphFact> {
    let mut graph_facts = Vec::new();

    for (i, fact) in raw_facts.iter().enumerate() {
        let tx_type = "transaction".to_string();
        let tx_instance = format!("tx_{}", i);

        // Transaction → head entity
        graph_facts.push(GraphFact {
            src: (tx_type.clone(), tx_instance.clone()),
            relation: format!("{}_from", fact.relation),
            dst: (entity_type_from_name(&fact.head), fact.head.clone()),
        });

        // Transaction → tail entity
        graph_facts.push(GraphFact {
            src: (tx_type.clone(), tx_instance.clone()),
            relation: format!("{}_to", fact.relation),
            dst: (entity_type_from_name(&fact.tail), fact.tail.clone()),
        });

        // Transaction → qualifier entities
        for qual in &fact.qualifiers {
            graph_facts.push(GraphFact {
                src: (tx_type.clone(), tx_instance.clone()),
                relation: qual.relation.clone(),
                dst: (entity_type_from_name(&qual.entity), qual.entity.clone()),
            });
        }
    }

    graph_facts
}

/// Extract the entity type from a synthetic entity name.
///
/// Synthetic entity names follow the pattern "type_N" (e.g., "account_0", "user_1").
/// We extract everything before the last underscore as the type.
fn entity_type_from_name(name: &str) -> String {
    if let Some(pos) = name.rfind('_') {
        // Check if what's after the underscore is a number
        if name[pos + 1..].parse::<usize>().is_ok() {
            return name[..pos].to_string();
        }
    }
    // Fallback: use the whole name as the type
    name.to_string()
}

/// Build a HeteroGraph from the TQL schema using synthetic data.
///
/// High-level convenience function that:
/// 1. Generates synthetic facts from the schema
/// 2. Converts to graph facts using transaction-as-node pattern
/// 3. Builds the HeteroGraph
pub fn build_from_schema<B: Backend>(
    schema: &TqlSchema,
    syn_config: &crate::data::synthetic::SyntheticDataConfig,
    graph_config: &GraphBuildConfig,
    device: &B::Device,
) -> HeteroGraph<B> {
    let raw_facts = crate::data::synthetic::generate_synthetic_facts(schema, syn_config);
    let graph_facts = raw_facts_to_graph_facts(&raw_facts);
    build_hetero_graph(&graph_facts, graph_config, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_build_hetero_graph_simple() {
        let device = <TestBackend as Backend>::Device::default();

        let facts = vec![
            GraphFact {
                src: ("user".into(), "alice".into()),
                relation: "owns".into(),
                dst: ("account".into(), "checking_1".into()),
            },
            GraphFact {
                src: ("user".into(), "alice".into()),
                relation: "owns".into(),
                dst: ("account".into(), "savings_1".into()),
            },
            GraphFact {
                src: ("transaction".into(), "tx_1".into()),
                relation: "posted_to".into(),
                dst: ("account".into(), "checking_1".into()),
            },
            GraphFact {
                src: ("transaction".into(), "tx_1".into()),
                relation: "from_merchant".into(),
                dst: ("merchant".into(), "starbucks".into()),
            },
        ];

        let config = GraphBuildConfig {
            node_feat_dim: 8,
            add_reverse_edges: true,
            add_self_loops: false,
            add_positional_encoding: true,
        };

        let graph = build_hetero_graph::<TestBackend>(&facts, &config, &device);

        // node types: user(1), account(2), transaction(1), merchant(1)
        assert_eq!(graph.node_counts["user"], 1);
        assert_eq!(graph.node_counts["account"], 2);
        assert_eq!(graph.node_counts["transaction"], 1);
        assert_eq!(graph.node_counts["merchant"], 1);
        assert_eq!(graph.total_nodes(), 5);

        // 4 forward edges + 4 reverse edges = 8 total edge types
        // (some edge types may have multiple edges)
        assert!(graph.total_edges() >= 8);
    }

    #[test]
    fn test_entity_type_from_name() {
        assert_eq!(entity_type_from_name("account_0"), "account");
        assert_eq!(
            entity_type_from_name("user-merchant-unit_3"),
            "user-merchant-unit"
        );
        assert_eq!(entity_type_from_name("starbucks"), "starbucks");
        assert_eq!(entity_type_from_name("main-account_12"), "main-account");
    }

    #[test]
    fn test_raw_facts_to_graph_facts() {
        use crate::data::fact::{RawFact, RawQualifier};

        let raw = vec![RawFact {
            head: "user_0".into(),
            relation: "owns_account".into(),
            tail: "account_1".into(),
            qualifiers: vec![RawQualifier {
                relation: "via_provider".into(),
                entity: "feed-provider_0".into(),
            }],
        }];

        let graph_facts = raw_facts_to_graph_facts(&raw);

        // 1 fact → 1 tx_from + 1 tx_to + 1 qualifier = 3 edges
        assert_eq!(graph_facts.len(), 3);
        assert_eq!(graph_facts[0].src.0, "transaction");
        assert_eq!(graph_facts[0].relation, "owns_account_from");
        assert_eq!(graph_facts[1].relation, "owns_account_to");
        assert_eq!(graph_facts[2].relation, "via_provider");
    }

    #[test]
    fn test_build_from_schema() {
        let tql = r#"define
entity user,
 owns attribute instance_guid,
 plays user-owns-account:owner;
entity account,
 owns attribute instance_guid,
 plays user-owns-account:owned-account;
relation user-owns-account,
 relates owner,
 relates owned-account;
"#;

        let schema = TqlSchema::parse(tql);
        let syn_config = crate::data::synthetic::SyntheticDataConfig {
            instances_per_type: 3,
            num_facts: 20,
            max_qualifiers: 1,
            seed: 42,
        };
        let graph_config = GraphBuildConfig {
            node_feat_dim: 8,
            add_reverse_edges: true,
            add_self_loops: true,
            add_positional_encoding: true,
        };

        let device = <TestBackend as Backend>::Device::default();
        let graph = build_from_schema::<TestBackend>(&schema, &syn_config, &graph_config, &device);

        assert!(graph.total_nodes() > 0);
        assert!(graph.total_edges() > 0);
        println!(
            "Built graph: {} nodes, {} edges, {} node types, {} edge types",
            graph.total_nodes(),
            graph.total_edges(),
            graph.node_types().len(),
            graph.edge_types().len()
        );
    }
}
