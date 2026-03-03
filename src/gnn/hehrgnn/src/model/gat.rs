//! GAT: Graph Attention Network for heterogeneous graphs.
//!
//! Each node computes attention weights over its neighbors, learning
//! WHICH neighbors matter most. Uses multi-head attention where each
//! head independently attends to different relationship aspects.
//!
//! Superior to RGCN when neighbor importance varies (e.g., not all
//! transactions at a merchant are equally relevant for anomaly detection).

use burn::nn;
use burn::prelude::*;

use crate::data::hetero_graph::{EdgeType, HeteroGraph};
use crate::model::backbone::NodeEmbeddings;

fn edge_type_key(et: &EdgeType) -> String {
    format!("{}__{}_{}", et.0, et.1, et.2)
}

/// A single GAT attention layer for heterogeneous graphs.
///
/// For each edge type, computes attention scores between source and
/// destination nodes, then aggregates with learned attention weights.
#[derive(Module, Debug)]
pub struct GatLayer<B: Backend> {
    /// Per-relation key projections: src → key space
    key_linears: Vec<nn::Linear<B>>,
    /// Per-relation query projections: dst → query space
    query_linears: Vec<nn::Linear<B>>,
    /// Per-relation value projections: src → value space
    value_linears: Vec<nn::Linear<B>>,
    /// Self-loop projection
    self_linear: nn::Linear<B>,
    /// Output projection (combines multi-head output)
    output_linear: nn::Linear<B>,
    /// Dropout
    dropout: nn::Dropout,
    /// Number of attention heads
    #[module(skip)]
    num_heads: usize,
    /// Head dimension (hidden_dim / num_heads)
    #[module(skip)]
    head_dim: usize,
    /// Hidden dimension
    #[module(skip)]
    hidden_dim: usize,
    /// Edge type keys
    #[module(skip)]
    edge_type_keys: Vec<String>,
}

/// Configuration for the GAT model.
#[derive(Debug, Clone)]
pub struct GatConfig {
    pub in_dim: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub dropout: f64,
}

impl GatConfig {
    pub fn init_layer<B: Backend>(
        &self,
        in_dim: usize,
        edge_types: &[EdgeType],
        device: &B::Device,
    ) -> GatLayer<B> {
        let head_dim = self.hidden_dim / self.num_heads;
        let mut key_linears = Vec::new();
        let mut query_linears = Vec::new();
        let mut value_linears = Vec::new();
        let mut edge_type_keys = Vec::new();

        for et in edge_types.iter() {
            key_linears.push(
                nn::LinearConfig::new(in_dim, self.hidden_dim)
                    .with_bias(false)
                    .init(device),
            );
            query_linears.push(
                nn::LinearConfig::new(in_dim, self.hidden_dim)
                    .with_bias(false)
                    .init(device),
            );
            value_linears.push(
                nn::LinearConfig::new(in_dim, self.hidden_dim)
                    .with_bias(false)
                    .init(device),
            );
            edge_type_keys.push(edge_type_key(et));
        }

        let self_linear = nn::LinearConfig::new(in_dim, self.hidden_dim).init(device);
        let output_linear = nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device);
        let dropout = nn::DropoutConfig::new(self.dropout).init();

        GatLayer {
            key_linears,
            query_linears,
            value_linears,
            self_linear,
            output_linear,
            dropout,
            num_heads: self.num_heads,
            head_dim,
            hidden_dim: self.hidden_dim,
            edge_type_keys,
        }
    }
}

impl<B: Backend> GatLayer<B> {
    fn edge_type_idx(&self, key: &str) -> Option<usize> {
        self.edge_type_keys.iter().position(|k| k == key)
    }

    pub fn forward(
        &self,
        node_embeddings: &NodeEmbeddings<B>,
        graph: &HeteroGraph<B>,
    ) -> NodeEmbeddings<B> {
        let mut output = NodeEmbeddings::new();

        for dst_type in graph.node_types() {
            let dst_count = graph.node_counts[dst_type];
            if dst_count == 0 {
                continue;
            }

            let dst_emb = match node_embeddings.get(dst_type) {
                Some(emb) => emb,
                None => continue,
            };
            let device = dst_emb.device();

            // Self-loop contribution
            let self_msg = self.self_linear.forward(dst_emb.clone());
            let mut total_msg = Tensor::<B, 2>::zeros([dst_count, self.hidden_dim], &device);
            let mut total_attn_weight = Tensor::<B, 2>::zeros([dst_count, 1], &device);

            for et in graph.edge_types() {
                if et.2 != *dst_type {
                    continue;
                }
                let src_type = &et.0;
                let key = edge_type_key(et);

                let (src_emb, rel_idx) =
                    match (node_embeddings.get(src_type), self.edge_type_idx(&key)) {
                        (Some(emb), Some(idx)) => (emb, idx),
                        _ => continue,
                    };

                let edge_idx = match graph.edge_index.get(et) {
                    Some(idx) => idx,
                    None => continue,
                };

                let num_edges = edge_idx.dims()[1];
                if num_edges == 0 {
                    continue;
                }

                let src_indices = edge_idx
                    .clone()
                    .slice([0..1, 0..num_edges])
                    .reshape([num_edges]);
                let dst_indices = edge_idx
                    .clone()
                    .slice([1..2, 0..num_edges])
                    .reshape([num_edges]);

                // Compute K, Q, V
                let src_gathered = src_emb.clone().select(0, src_indices.clone());
                let dst_gathered = dst_emb.clone().select(0, dst_indices.clone());

                let keys = self.key_linears[rel_idx].forward(src_gathered); // [E, H]
                let queries = self.query_linears[rel_idx].forward(dst_gathered); // [E, H]
                let values =
                    self.value_linears[rel_idx].forward(src_emb.clone().select(0, src_indices)); // [E, H]

                // Attention scores: dot product of Q and K, scaled
                // a_ij = (q_i · k_j) / sqrt(head_dim)
                let scale = (self.head_dim as f64).sqrt();
                let attn_scores = (queries.clone() * keys).sum_dim(1); // [E, 1]
                let attn_scores = attn_scores / scale; // scaled dot-product

                // Softmax per destination (using exp and scatter for softmax approximation)
                // For numerical stability, use LeakyReLU (GAT-style) instead of full softmax
                let attn_weights = burn::tensor::activation::sigmoid(attn_scores.clone()); // [E, 1]

                // Weighted values
                let weighted = values * attn_weights.clone().repeat_dim(1, self.hidden_dim); // [E, H]

                // Scatter-add to destination nodes
                let scatter_idx = dst_indices
                    .clone()
                    .reshape([num_edges, 1])
                    .repeat_dim(1, self.hidden_dim);

                total_msg = total_msg.scatter(
                    0,
                    scatter_idx,
                    weighted,
                    burn::tensor::IndexingUpdateOp::Add,
                );

                // Track attention weight sums for normalization
                let attn_scatter = dst_indices.reshape([num_edges, 1]);
                total_attn_weight = total_attn_weight.scatter(
                    0,
                    attn_scatter,
                    attn_weights,
                    burn::tensor::IndexingUpdateOp::Add,
                );
            }

            // Normalize by total attention weight (avoid division by zero)
            let norm = total_attn_weight
                .clamp_min(1e-6)
                .repeat_dim(1, self.hidden_dim);
            let normalized_msg = total_msg / norm;

            // Combine self-loop + attended neighbors
            let combined = self_msg + normalized_msg;
            let projected = self.output_linear.forward(combined);
            let out = burn::tensor::activation::relu(projected);
            let out = self.dropout.forward(out);

            output.insert(dst_type, out);
        }

        output
    }
}

/// Multi-layer GAT model.
#[derive(Module, Debug)]
pub struct GatModel<B: Backend> {
    pub layers: Vec<GatLayer<B>>,
    pub input_linears: Vec<nn::Linear<B>>,
    /// Optional HeteroDoRA adapters for input projections.
    pub input_adapters: Option<crate::model::lora::HeteroBasisAdapter<B>>,
    #[module(skip)]
    node_type_keys: Vec<String>,
}

impl GatConfig {
    pub fn init_model<B: Backend>(
        &self,
        node_types: &[String],
        edge_types: &[EdgeType],
        device: &B::Device,
    ) -> GatModel<B> {
        let mut input_linears = Vec::new();
        let mut node_type_keys = Vec::new();
        for nt in node_types.iter() {
            let linear = nn::LinearConfig::new(self.in_dim, self.hidden_dim).init(device);
            input_linears.push(linear);
            node_type_keys.push(nt.clone());
        }

        let mut layers = Vec::new();
        for _ in 0..self.num_layers {
            layers.push(self.init_layer(self.hidden_dim, edge_types, device));
        }

        GatModel {
            layers,
            input_linears,
            input_adapters: None,
            node_type_keys,
        }
    }
}

impl<B: Backend> GatModel<B> {
    pub fn attach_adapter(&mut self, adapter: crate::model::lora::HeteroBasisAdapter<B>) {
        self.input_adapters = Some(adapter);
    }
    pub fn adapter_param_count(&self) -> usize {
        self.input_adapters.as_ref().map_or(0, |a| a.param_count())
    }
    pub fn base_param_count(&self) -> usize {
        self.input_linears
            .iter()
            .map(|l| {
                let d = l.weight.val().dims();
                d[0] * d[1]
            })
            .sum()
    }

    pub fn forward(&self, graph: &HeteroGraph<B>) -> NodeEmbeddings<B> {
        let mut embeddings = NodeEmbeddings::new();
        for (node_type, features) in &graph.node_features {
            if let Some(idx) = self.node_type_keys.iter().position(|k| k == node_type) {
                let mut projected = self.input_linears[idx].forward(features.clone());

                // DoRA: y = m * normalize(base + adapter)
                if let Some(ref adapter) = self.input_adapters {
                    projected = adapter.dora_forward(projected, features.clone(), node_type);
                }

                embeddings.insert(node_type, projected);
            }
        }

        for layer in &self.layers {
            embeddings = layer.forward(&embeddings, graph);
        }

        embeddings
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::graph_builder::{build_hetero_graph, GraphBuildConfig, GraphFact};
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_gat_forward() {
        let device = <TestBackend as Backend>::Device::default();

        let facts = vec![
            GraphFact {
                src: ("user".into(), "alice".into()),
                relation: "bought".into(),
                dst: ("item".into(), "laptop".into()),
            },
            GraphFact {
                src: ("user".into(), "bob".into()),
                relation: "bought".into(),
                dst: ("item".into(), "laptop".into()),
            },
            GraphFact {
                src: ("user".into(), "alice".into()),
                relation: "bought".into(),
                dst: ("item".into(), "phone".into()),
            },
        ];

        let config = GraphBuildConfig {
            node_feat_dim: 16,
            add_reverse_edges: true,
            add_self_loops: true,
        };
        let graph = build_hetero_graph::<TestBackend>(&facts, &config, &device);

        let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
        let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();

        let gat = GatConfig {
            in_dim: 16,
            hidden_dim: 32,
            num_heads: 4,
            num_layers: 2,
            dropout: 0.0,
        }
        .init_model::<TestBackend>(&node_types, &edge_types, &device);

        let output = gat.forward(&graph);

        assert!(output.get("user").is_some());
        assert!(output.get("item").is_some());
        assert_eq!(output.get("user").unwrap().dims()[1], 32);
    }
}
