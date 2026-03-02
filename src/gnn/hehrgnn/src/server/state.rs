//! Server state: pre-computed embeddings stored as plain Vecs.
//!
//! Since Burn's `Tensor<NdArray>` doesn't implement `Send`, we pre-compute
//! all embeddings and model outputs at startup and store them as plain `Vec<f32>`.
//! This makes the state `Send + Sync` as required by axum.

use std::collections::HashMap;
use std::sync::Arc;

use burn::backend::NdArray;
use burn::prelude::*;

use crate::data::hetero_graph::{EdgeType, HeteroGraph};
use crate::model::graphsage::GraphSageModelConfig;

/// The inference backend.
type B = NdArray;

/// Pre-computed node embeddings stored as plain vectors.
#[derive(Debug, Clone)]
pub struct PlainEmbeddings {
    /// Per-node-type embeddings: `embeddings[node_type][node_idx] = Vec<f32>`.
    pub data: HashMap<String, Vec<Vec<f32>>>,
    pub hidden_dim: usize,
}

impl PlainEmbeddings {
    /// Extract from Burn NodeEmbeddings (generic over any backend).
    pub fn from_burn<BK: Backend>(emb: &crate::model::backbone::NodeEmbeddings<BK>) -> Self {
        let mut data = HashMap::new();
        let mut hidden_dim = 0;

        for (nt, tensor) in &emb.embeddings {
            let dims = tensor.dims();
            let num_nodes = dims[0];
            hidden_dim = dims[1];

            let flat = tensor.clone().reshape([num_nodes * hidden_dim]).into_data();
            let values: Vec<f32> = flat.as_slice::<f32>().unwrap().to_vec();

            let mut per_node = Vec::with_capacity(num_nodes);
            for i in 0..num_nodes {
                per_node.push(values[i * hidden_dim..(i + 1) * hidden_dim].to_vec());
            }
            data.insert(nt.clone(), per_node);
        }

        PlainEmbeddings { data, hidden_dim }
    }

    /// Cosine similarity between two embedding vectors.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a < 1e-8 || norm_b < 1e-8 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// L2 distance between two embedding vectors.
    pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Dot-product score between two embedding vectors (simple link prediction).
    pub fn dot_score(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

/// Graph metadata (Send-safe, no Burn tensors).
#[derive(Debug, Clone)]
pub struct GraphMeta {
    pub node_types: Vec<String>,
    pub node_counts: HashMap<String, usize>,
    pub edge_types: Vec<(String, String, String)>,
    pub edge_counts: HashMap<(String, String, String), usize>,
    pub total_nodes: usize,
    pub total_edges: usize,
}

/// Shared application state — all plain data, `Send + Sync` safe.
pub struct AppState {
    /// Pre-computed GNN embeddings.
    pub embeddings: PlainEmbeddings,

    /// Graph metadata.
    pub graph_meta: GraphMeta,

    /// Node type → list of instance names.
    pub node_names: HashMap<String, Vec<String>>,

    /// Model configuration metadata.
    pub hidden_dim: usize,
    pub num_classes: usize,
}

/// Configuration for initializing the server state.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub hidden_dim: usize,
    pub num_gnn_layers: usize,
    pub num_classes: usize,
    pub schema_path: Option<String>,
    pub num_facts: usize,
    pub instances_per_type: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 32,
            num_gnn_layers: 2,
            num_classes: 5,
            schema_path: None,
            num_facts: 200,
            instances_per_type: 5,
        }
    }
}

impl AppState {
    /// Initialize: build graph → run GNN → extract plain embeddings.
    pub fn init(config: &ServerConfig) -> Arc<Self> {
        use crate::data::graph_builder::{build_from_schema, GraphBuildConfig};
        use crate::data::synthetic::{SyntheticDataConfig, TqlSchema};

        let device = <B as Backend>::Device::default();
        let feat_dim = 16;

        // Load schema
        let schema = if let Some(ref path) = config.schema_path {
            let tql = std::fs::read_to_string(path).expect("Failed to read schema file");
            TqlSchema::parse(&tql)
        } else {
            TqlSchema::parse(DEFAULT_SCHEMA)
        };

        let syn_config = SyntheticDataConfig {
            instances_per_type: config.instances_per_type,
            num_facts: config.num_facts,
            max_qualifiers: 2,
            seed: 42,
        };

        let graph_config = GraphBuildConfig {
            node_feat_dim: feat_dim,
            add_reverse_edges: true,
            add_self_loops: true,
        };

        // Build graph
        let graph: HeteroGraph<B> = build_from_schema(&schema, &syn_config, &graph_config, &device);

        println!(
            "  Graph: {} nodes, {} edges, {} node types, {} edge types",
            graph.total_nodes(),
            graph.total_edges(),
            graph.node_types().len(),
            graph.edge_types().len()
        );

        // Run GraphSAGE to compute embeddings
        let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
        let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();

        let sage_config = GraphSageModelConfig {
            in_dim: feat_dim,
            hidden_dim: config.hidden_dim,
            num_layers: config.num_gnn_layers,
            dropout: 0.0,
        };

        let model = sage_config.init::<B>(&node_types, &edge_types, &device);
        let burn_embeddings = model.forward(&graph);

        // Convert to plain embeddings (no more Burn tensors)
        let embeddings = PlainEmbeddings::from_burn(&burn_embeddings);

        // Build graph metadata
        let graph_meta = GraphMeta {
            node_types: node_types.clone(),
            node_counts: graph.node_counts.clone(),
            edge_types: graph
                .edge_types()
                .iter()
                .map(|et| (et.0.clone(), et.1.clone(), et.2.clone()))
                .collect(),
            edge_counts: graph
                .edge_counts
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
            total_nodes: graph.total_nodes(),
            total_edges: graph.total_edges(),
        };

        // Build node names
        let mut node_names = HashMap::new();
        for nt in &node_types {
            let count = graph.node_counts.get(nt).copied().unwrap_or(0);
            let names: Vec<String> = (0..count).map(|i| format!("{}_{}", nt, i)).collect();
            node_names.insert(nt.clone(), names);
        }

        Arc::new(Self {
            embeddings,
            graph_meta,
            node_names,
            hidden_dim: config.hidden_dim,
            num_classes: config.num_classes,
        })
    }
}

const DEFAULT_SCHEMA: &str = r#"define
entity user,
 plays user-owns-account:owner;
entity account,
 plays user-owns-account:owned-account,
 plays transaction-posted-to:target-account;
entity transaction,
 plays transaction-posted-to:posted-tx,
 plays transaction-at-merchant:spending-tx;
entity merchant,
 plays transaction-at-merchant:merchant;
entity category,
 plays transaction-has-category:assigned-category;
relation user-owns-account,
 relates owner,
 relates owned-account;
relation transaction-posted-to,
 relates posted-tx,
 relates target-account;
relation transaction-at-merchant,
 relates spending-tx,
 relates merchant;
relation transaction-has-category,
 relates categorized-tx,
 relates assigned-category;
"#;
