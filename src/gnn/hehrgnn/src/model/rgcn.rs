//! RGCN: Relational Graph Convolutional Network for heterogeneous graphs.
//!
//! Implements relation-typed message passing where each edge type has its own
//! weight matrix. Strong baseline when the schema has many relation types.
//!
//! Best for: match scoring, GL allocation conditioned on relation types.

use burn::nn;
use burn::prelude::*;

use crate::data::hetero_graph::{EdgeType, HeteroGraph};
use crate::model::backbone::NodeEmbeddings;

fn edge_type_key(et: &EdgeType) -> String {
    format!("{}__{}_{}", et.0, et.1, et.2)
}

/// A single RGCN message-passing layer.
#[derive(Module, Debug)]
pub struct RgcnLayer<B: Backend> {
    /// Per-relation weight matrices.
    relation_weights: Vec<nn::Linear<B>>,
    /// Self-loop weight.
    self_weight: nn::Linear<B>,
    /// Dropout.
    dropout: nn::Dropout,
    /// Hidden dimension.
    #[module(skip)]
    hidden_dim: usize,
    /// Edge type keys for index lookup.
    #[module(skip)]
    edge_type_keys: Vec<String>,
}

/// Configuration for the RGCN model.
#[derive(Debug, Clone)]
pub struct RgcnConfig {
    pub in_dim: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_bases: usize,
    pub dropout: f64,
}

impl RgcnConfig {
    pub fn init_layer<B: Backend>(
        &self,
        in_dim: usize,
        edge_types: &[EdgeType],
        device: &B::Device,
    ) -> RgcnLayer<B> {
        let mut relation_weights = Vec::new();
        let mut edge_type_keys = Vec::new();

        for et in edge_types.iter() {
            let linear = nn::LinearConfig::new(in_dim, self.hidden_dim)
                .with_bias(false)
                .init(device);
            relation_weights.push(linear);
            edge_type_keys.push(edge_type_key(et));
        }

        let self_weight = nn::LinearConfig::new(in_dim, self.hidden_dim).init(device);
        let dropout = nn::DropoutConfig::new(self.dropout).init();

        RgcnLayer {
            relation_weights,
            self_weight,
            dropout,
            hidden_dim: self.hidden_dim,
            edge_type_keys,
        }
    }
}

impl<B: Backend> RgcnLayer<B> {
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

            let self_msg = self.self_weight.forward(dst_emb.clone());
            let mut total_msg = Tensor::<B, 2>::zeros([dst_count, self.hidden_dim], &device);

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

                let gathered = src_emb.clone().select(0, src_indices);
                let transformed = self.relation_weights[rel_idx].forward(gathered);

                let scatter_idx = dst_indices
                    .reshape([num_edges, 1])
                    .repeat_dim(1, self.hidden_dim);

                total_msg = total_msg.scatter(
                    0,
                    scatter_idx,
                    transformed,
                    burn::tensor::IndexingUpdateOp::Add,
                );
            }

            let combined = self_msg + total_msg;
            let out = burn::tensor::activation::relu(combined);
            let out = self.dropout.forward(out);

            output.insert(dst_type, out);
        }

        output
    }
}

/// Multi-layer RGCN model.
#[derive(Module, Debug)]
pub struct RgcnModel<B: Backend> {
    pub layers: Vec<RgcnLayer<B>>,
    pub input_linears: Vec<nn::Linear<B>>,
    /// Optional HeteroDoRA adapters for input projections.
    pub input_adapters: Option<crate::model::lora::HeteroBasisAdapter<B>>,
    #[module(skip)]
    node_type_keys: Vec<String>,
}

impl RgcnConfig {
    pub fn init_model<B: Backend>(
        &self,
        node_types: &[String],
        edge_types: &[EdgeType],
        device: &B::Device,
    ) -> RgcnModel<B> {
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

        RgcnModel {
            layers,
            input_linears,
            input_adapters: None,
            node_type_keys,
        }
    }
}

impl<B: Backend> RgcnModel<B> {
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
    fn test_rgcn_forward() {
        let device = <TestBackend as Backend>::Device::default();

        let facts = vec![
            GraphFact {
                src: ("user".into(), "u0".into()),
                relation: "owns".into(),
                dst: ("acct".into(), "a0".into()),
            },
            GraphFact {
                src: ("user".into(), "u1".into()),
                relation: "owns".into(),
                dst: ("acct".into(), "a1".into()),
            },
            GraphFact {
                src: ("tx".into(), "t0".into()),
                relation: "posted".into(),
                dst: ("acct".into(), "a0".into()),
            },
            GraphFact {
                src: ("tx".into(), "t0".into()),
                relation: "at".into(),
                dst: ("merchant".into(), "m0".into()),
            },
        ];

        let graph = build_hetero_graph::<TestBackend>(
            &facts,
            &GraphBuildConfig {
                node_feat_dim: 8,
                add_reverse_edges: true,
                add_self_loops: true, add_positional_encoding: true,
            },
            &device,
        );

        let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
        let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();

        let config = RgcnConfig {
            in_dim: 8,
            hidden_dim: 16,
            num_layers: 2,
            num_bases: 0,
            dropout: 0.0,
        };

        let model = config.init_model::<TestBackend>(&node_types, &edge_types, &device);
        let embeddings = model.forward(&graph);

        for nt in &node_types {
            let emb = embeddings.get(nt).expect(&format!("Missing {}", nt));
            assert_eq!(emb.dims()[0], graph.node_counts[nt]);
            assert_eq!(emb.dims()[1], 16);
        }
    }
}
