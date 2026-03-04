//! GraphSAGE: inductive, scalable GNN for heterogeneous graphs.
//!
//! Implements the GraphSAGE message-passing scheme adapted for heterogeneous
//! graphs. Each layer:
//! 1. Aggregates neighbor embeddings per edge type (mean/max)
//! 2. Concatenates self-embedding with aggregated neighbor message
//! 3. Applies linear transform + ReLU
//!
//! Best for: transaction categorization, merchant similarity, fast embeddings.

use burn::module::Param;
use burn::nn;
use burn::prelude::*;

use crate::data::hetero_graph::{EdgeType, HeteroGraph};
use crate::model::backbone::NodeEmbeddings;

fn edge_type_key(et: &EdgeType) -> String {
    format!("{}__{}_{}", et.0, et.1, et.2)
}

/// A single GraphSAGE message-passing layer for heterogeneous graphs.
///
/// For each destination node type, messages from each edge type are
/// aggregated separately, then combined with a Linear + ReLU.
#[derive(Module, Debug)]
pub struct GraphSageLayer<B: Backend> {
    /// Per-edge-type message transform: maps `in_dim` → `hidden_dim`.
    message_linears: Vec<nn::Linear<B>>,

    /// Self-transform: maps `in_dim` → `hidden_dim`.
    self_linear: nn::Linear<B>,

    /// Output projection: combines self + aggregated → `hidden_dim`.
    output_linear: nn::Linear<B>,

    /// Dropout.
    dropout: nn::Dropout,

    /// Hidden dimension.
    #[module(skip)]
    hidden_dim: usize,

    /// Edge type keys corresponding to message_linears indices.
    #[module(skip)]
    edge_type_keys: Vec<String>,
}

/// Configuration for the GraphSAGE model.
#[derive(Config, Debug)]
pub struct GraphSageConfig {
    /// Input feature dimension (from node features).
    pub in_dim: usize,
    /// Hidden dimension for intermediate layers.
    pub hidden_dim: usize,
    /// Number of GraphSAGE layers.
    #[config(default = "2")]
    pub num_layers: usize,
    /// Dropout rate.
    #[config(default = "0.1")]
    pub dropout: f64,
}

impl GraphSageConfig {
    /// Initialize the GraphSAGE layer for the given edge types.
    pub fn init_layer<B: Backend>(
        &self,
        in_dim: usize,
        edge_types: &[EdgeType],
        device: &B::Device,
    ) -> GraphSageLayer<B> {
        let mut message_linears = Vec::new();
        let mut edge_type_keys = Vec::new();

        for et in edge_types.iter() {
            let linear = nn::LinearConfig::new(in_dim, self.hidden_dim).init(device);
            message_linears.push(linear);
            edge_type_keys.push(edge_type_key(et));
        }

        let self_linear = nn::LinearConfig::new(in_dim, self.hidden_dim).init(device);
        let output_linear =
            nn::LinearConfig::new(self.hidden_dim * 2, self.hidden_dim).init(device);
        let dropout = nn::DropoutConfig::new(self.dropout).init();

        GraphSageLayer {
            message_linears,
            self_linear,
            output_linear,
            dropout,
            hidden_dim: self.hidden_dim,
            edge_type_keys,
        }
    }
}

impl<B: Backend> GraphSageLayer<B> {
    fn edge_type_idx(&self, key: &str) -> Option<usize> {
        self.edge_type_keys.iter().position(|k| k == key)
    }

    /// Forward pass through one GraphSAGE layer.
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

            let self_msg = self.self_linear.forward(dst_emb.clone());

            let mut msg_sum = Tensor::<B, 2>::zeros([dst_count, self.hidden_dim], &device);
            let mut msg_count = 0;

            for et in graph.edge_types() {
                if et.2 != *dst_type {
                    continue;
                }

                let src_type = &et.0;
                let key = edge_type_key(et);

                let (src_emb, linear_idx) =
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

                let gathered_src = src_emb.clone().select(0, src_indices);
                let transformed = self.message_linears[linear_idx].forward(gathered_src);

                let scatter_idx = dst_indices
                    .reshape([num_edges, 1])
                    .repeat_dim(1, self.hidden_dim);

                msg_sum = msg_sum.scatter(
                    0,
                    scatter_idx,
                    transformed,
                    burn::tensor::IndexingUpdateOp::Add,
                );

                msg_count += 1;
            }

            if msg_count > 0 {
                msg_sum = msg_sum / (msg_count as f32);
            }

            let combined = Tensor::cat(vec![self_msg, msg_sum], 1);
            let out = self.output_linear.forward(combined);
            let out = burn::tensor::activation::relu(out);
            let out = self.dropout.forward(out);

            output.insert(dst_type, out);
        }

        output
    }
}

/// Multi-layer GraphSAGE model for heterogeneous graphs.
#[derive(Module, Debug)]
pub struct GraphSageModel<B: Backend> {
    pub layers: Vec<GraphSageLayer<B>>,
    pub input_linears: Vec<nn::Linear<B>>,
    /// Optional HeteroDoRA adapters for input projections.
    /// When present: y = input_linear(x) + adapter(x)
    pub input_adapters: Option<crate::model::lora::HeteroBasisAdapter<B>>,
    /// Learnable node-type embedding (KumoRFM §2.3)
    type_embeddings: Vec<Param<Tensor<B, 2>>>,
    #[module(skip)]
    node_type_keys: Vec<String>,
}

/// Configuration for the full GraphSAGE model.
#[derive(Debug, Clone)]
pub struct GraphSageModelConfig {
    pub in_dim: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub dropout: f64,
}

impl GraphSageModelConfig {
    pub fn init<B: Backend>(
        &self,
        node_types: &[String],
        edge_types: &[EdgeType],
        device: &B::Device,
    ) -> GraphSageModel<B> {
        let mut input_linears = Vec::new();
        let mut node_type_keys = Vec::new();
        let mut type_embeddings = Vec::new();
        for nt in node_types.iter() {
            let linear = nn::LinearConfig::new(self.in_dim, self.hidden_dim).init(device);
            input_linears.push(linear);
            node_type_keys.push(nt.clone());
            let emb = Tensor::<B, 2>::random(
                [1, self.hidden_dim],
                burn::tensor::Distribution::Uniform(-0.1, 0.1),
                device,
            );
            type_embeddings.push(Param::from_tensor(emb));
        }

        let sage_config = GraphSageConfig {
            in_dim: self.in_dim,
            hidden_dim: self.hidden_dim,
            num_layers: self.num_layers,
            dropout: self.dropout,
        };

        let mut layers = Vec::new();
        for _ in 0..self.num_layers {
            layers.push(sage_config.init_layer(self.hidden_dim, edge_types, device));
        }

        GraphSageModel {
            layers,
            input_linears,
            input_adapters: None,
            type_embeddings,
            node_type_keys,
        }
    }
}

impl<B: Backend> GraphSageModel<B> {
    fn align_feature_dim(features: Tensor<B, 2>, expected_dim: usize) -> Tensor<B, 2> {
        let [num_nodes, in_dim] = features.dims();
        if in_dim == expected_dim {
            return features;
        }

        if in_dim > expected_dim {
            features.slice([0..num_nodes, 0..expected_dim])
        } else {
            let device = features.device();
            let pad = Tensor::<B, 2>::zeros([num_nodes, expected_dim - in_dim], &device);
            Tensor::cat(vec![features, pad], 1)
        }
    }

    /// Attach a HeteroDoRA adapter for input projections.
    pub fn attach_adapter(&mut self, adapter: crate::model::lora::HeteroBasisAdapter<B>) {
        self.input_adapters = Some(adapter);
    }

    /// Count adapter parameters (0 if no adapter attached).
    pub fn adapter_param_count(&self) -> usize {
        self.input_adapters.as_ref().map_or(0, |a| a.param_count())
    }

    /// Count base model parameters (input linears only for comparison).
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
                let expected_in = self.input_linears[idx].weight.val().dims()[0];
                let aligned_features = Self::align_feature_dim(features.clone(), expected_in);
                let mut projected = self.input_linears[idx].forward(aligned_features.clone());

                // DoRA: y = m * normalize(base + adapter)
                if let Some(ref adapter) = self.input_adapters {
                    projected = adapter.dora_forward(projected, aligned_features, node_type);
                }

                // Add learnable node-type embedding (KumoRFM §2.3)
                if idx < self.type_embeddings.len() {
                    projected = projected + self.type_embeddings[idx].val();
                }

                embeddings.insert(node_type, projected);
            }
        }

        for layer in &self.layers {
            embeddings = layer.forward(&embeddings, graph);
        }

        embeddings
    }

    /// Forward pass that also captures per-layer activations.
    /// Returns (final_embeddings, vec_of_per_layer_embeddings).
    /// Layer 0 = input projection, Layer 1..N = after each GNN layer.
    pub fn forward_with_activations(
        &self,
        graph: &HeteroGraph<B>,
    ) -> (NodeEmbeddings<B>, Vec<NodeEmbeddings<B>>) {
        let mut embeddings = NodeEmbeddings::new();
        for (node_type, features) in &graph.node_features {
            if let Some(idx) = self.node_type_keys.iter().position(|k| k == node_type) {
                let expected_in = self.input_linears[idx].weight.val().dims()[0];
                let aligned_features = Self::align_feature_dim(features.clone(), expected_in);
                let projected = self.input_linears[idx].forward(aligned_features);
                embeddings.insert(node_type, projected);
            }
        }

        let mut layer_activations: Vec<NodeEmbeddings<B>> = Vec::new();
        // Capture input projection as layer 0
        layer_activations.push(embeddings.clone());

        for layer in &self.layers {
            embeddings = layer.forward(&embeddings, graph);
            layer_activations.push(embeddings.clone());
        }

        (embeddings, layer_activations)
    }
}

impl<B: Backend> crate::model::trainer::JepaTrainable<B> for GraphSageModel<B> {
    fn forward_embeddings(&self, graph: &HeteroGraph<B>) -> NodeEmbeddings<B> {
        self.forward(graph)
    }

    fn num_input_weights(&self) -> usize {
        self.input_linears.len()
    }

    fn get_input_weight(&self, idx: usize) -> Tensor<B, 2> {
        self.input_linears[idx].weight.val()
    }

    fn set_input_weight(&mut self, idx: usize, weight: Tensor<B, 2>) {
        self.input_linears[idx].weight = self.input_linears[idx].weight.clone().map(|_| weight);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::graph_builder::{GraphBuildConfig, GraphFact, build_hetero_graph};
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_graphsage_forward() {
        let device = <TestBackend as Backend>::Device::default();

        let facts = vec![
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
            GraphFact {
                src: ("tx".into(), "t0".into()),
                relation: "posted".into(),
                dst: ("account".into(), "a0".into()),
            },
            GraphFact {
                src: ("tx".into(), "t1".into()),
                relation: "posted".into(),
                dst: ("account".into(), "a1".into()),
            },
        ];

        let graph_config = GraphBuildConfig {
            node_feat_dim: 8,
            add_reverse_edges: true,
            add_self_loops: true,
            add_positional_encoding: true,
        };
        let graph = build_hetero_graph::<TestBackend>(&facts, &graph_config, &device);

        let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
        let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();

        let in_dim = graph
            .node_features
            .values()
            .next()
            .map(|t| t.dims()[1])
            .unwrap_or(8);

        let model_config = GraphSageModelConfig {
            in_dim,
            hidden_dim: 16,
            num_layers: 2,
            dropout: 0.0,
        };
        let model = model_config.init::<TestBackend>(&node_types, &edge_types, &device);

        let embeddings = model.forward(&graph);

        for nt in &node_types {
            let emb = embeddings
                .get(nt)
                .expect(&format!("Missing embedding for {}", nt));
            let dims = emb.dims();
            assert_eq!(
                dims[0], graph.node_counts[nt],
                "Wrong node count for {}",
                nt
            );
            assert_eq!(dims[1], 16, "Wrong hidden dim for {}", nt);
        }
    }
}
