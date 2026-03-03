//! Graph Transformer (GPS-style) for heterogeneous graphs.
//!
//! Implements the GraphGPS architecture:
//! - Local MPNN: relation-typed message passing (like RGCN) for local context
//! - Global Attention: multi-head self-attention over all nodes of each type
//! - FFN + Residual: feed-forward network with skip connections
//!
//! Reference: Rampášek et al. "Recipe for a General, Powerful, Scalable
//! Graph Transformer" (NeurIPS 2022)

use burn::nn;
use burn::prelude::*;
use std::collections::HashMap;

use crate::data::hetero_graph::{EdgeType, HeteroGraph};
use crate::model::backbone::NodeEmbeddings;

fn edge_type_key(et: &EdgeType) -> String {
    format!("{}_{}_{}", et.0, et.1, et.2)
}

/// Configuration for Graph Transformer.
#[derive(Debug, Clone)]
pub struct GraphTransformerConfig {
    pub in_dim: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub ffn_ratio: usize, // FFN hidden = hidden_dim * ffn_ratio (typically 2 or 4)
    pub dropout: f64,
}

/// A single GPS layer: Local MPNN + Global Attention + FFN
#[derive(Module, Debug)]
pub struct GpsLayer<B: Backend> {
    // Local MPNN: per-edge-type linear transforms
    mpnn_linears: Vec<nn::Linear<B>>,
    mpnn_bias: nn::Linear<B>, // hidden → hidden (bias-only via identity+bias)

    // Global multi-head self-attention projections
    attn_q: nn::Linear<B>,
    attn_k: nn::Linear<B>,
    attn_v: nn::Linear<B>,
    attn_out: nn::Linear<B>,

    // FFN
    ffn1: nn::Linear<B>,
    ffn2: nn::Linear<B>,

    // Non-module metadata
    #[module(skip)]
    num_heads: usize,
    #[module(skip)]
    head_dim: usize,
    #[module(skip)]
    edge_type_keys: Vec<String>,
}

impl<B: Backend> GpsLayer<B> {
    pub fn new(
        hidden_dim: usize,
        num_heads: usize,
        ffn_ratio: usize,
        edge_types: &[EdgeType],
        device: &B::Device,
    ) -> Self {
        let head_dim = hidden_dim / num_heads;
        let ffn_hidden = hidden_dim * ffn_ratio;

        // Local MPNN: per-relation linear transforms (no bias, like RGCN)
        let mut mpnn_linears = Vec::new();
        let mut edge_type_keys = Vec::new();
        for et in edge_types {
            let linear = nn::LinearConfig::new(hidden_dim, hidden_dim)
                .with_bias(false)
                .init(device);
            mpnn_linears.push(linear);
            edge_type_keys.push(edge_type_key(et));
        }
        // MPNN output bias (identity transform + bias)
        let mpnn_bias = nn::LinearConfig::new(hidden_dim, hidden_dim)
            .with_bias(true)
            .init(device);

        // Global self-attention projections
        let attn_q = nn::LinearConfig::new(hidden_dim, hidden_dim)
            .with_bias(false)
            .init(device);
        let attn_k = nn::LinearConfig::new(hidden_dim, hidden_dim)
            .with_bias(false)
            .init(device);
        let attn_v = nn::LinearConfig::new(hidden_dim, hidden_dim)
            .with_bias(false)
            .init(device);
        let attn_out = nn::LinearConfig::new(hidden_dim, hidden_dim)
            .with_bias(false)
            .init(device);

        // FFN layers
        let ffn1 = nn::LinearConfig::new(hidden_dim, ffn_hidden)
            .with_bias(true)
            .init(device);
        let ffn2 = nn::LinearConfig::new(ffn_hidden, hidden_dim)
            .with_bias(true)
            .init(device);

        Self {
            mpnn_linears,
            mpnn_bias,
            attn_q,
            attn_k,
            attn_v,
            attn_out,
            num_heads,
            head_dim,
            ffn1,
            ffn2,
            edge_type_keys,
        }
    }

    fn edge_type_idx(&self, key: &str) -> Option<usize> {
        self.edge_type_keys.iter().position(|k| k == key)
    }

    /// Forward pass for one GPS layer.
    pub fn forward(
        &self,
        embeddings: &NodeEmbeddings<B>,
        graph: &HeteroGraph<B>,
    ) -> NodeEmbeddings<B> {
        let mut output = NodeEmbeddings::new();

        // ── Step 1: Local MPNN (aggregate neighbor messages per relation) ──
        let mut mpnn_out: HashMap<String, Tensor<B, 2>> = HashMap::new();

        for et in graph.edge_types() {
            let key = edge_type_key(et);
            let linear_idx = match self.edge_type_idx(&key) {
                Some(idx) => idx,
                None => continue,
            };
            let linear = &self.mpnn_linears[linear_idx];

            let src_emb: &Tensor<B, 2> = match embeddings.get(&et.0) {
                Some(e) => e,
                None => continue,
            };

            let edge_key = (et.0.clone(), et.1.clone(), et.2.clone());
            let edge_index: &Tensor<B, 2, Int> = match graph.edge_index.get(&edge_key) {
                Some(idx) => idx,
                None => continue,
            };

            let num_edges = edge_index.dims()[1];
            if num_edges == 0 {
                continue;
            }

            let dst_count = *graph.node_counts.get(&et.2).unwrap_or(&0);
            if dst_count == 0 {
                continue;
            }

            // Transform source embeddings via nn::Linear
            let transformed = linear.forward(src_emb.clone()); // [num_src, hidden]
            let hidden_dim = transformed.dims()[1];

            // Gather source node features by edge source indices
            let src_indices = edge_index
                .clone()
                .slice([0..1, 0..num_edges])
                .reshape([num_edges]);
            let msg = transformed
                .select(0, src_indices.clone())
                .reshape([num_edges, hidden_dim]);

            // Scatter-add to destination nodes
            let dst_indices = edge_index
                .clone()
                .slice([1..2, 0..num_edges])
                .reshape([num_edges]);

            // Manual scatter-add (autodiff-compatible: only reads data for indices)
            let msg_data: Vec<f32> = msg.clone().into_data().as_slice::<f32>().unwrap().to_vec();
            let dst_data: Vec<i32> = {
                let data = dst_indices.into_data();
                if let Ok(slice) = data.as_slice::<i32>() {
                    slice.to_vec()
                } else if let Ok(slice) = data.as_slice::<i64>() {
                    slice.iter().map(|&v| v as i32).collect()
                } else {
                    panic!("Edge index tensor has unsupported integer type");
                }
            };

            let mut agg = vec![0.0f32; dst_count * hidden_dim];
            let mut counts = vec![0.0f32; dst_count];

            for (edge_idx, &dst) in dst_data.iter().enumerate() {
                let d = dst as usize;
                if d < dst_count {
                    for f in 0..hidden_dim {
                        agg[d * hidden_dim + f] += msg_data[edge_idx * hidden_dim + f];
                    }
                    counts[d] += 1.0;
                }
            }

            // Normalize by degree
            for d in 0..dst_count {
                if counts[d] > 0.0 {
                    for f in 0..hidden_dim {
                        agg[d * hidden_dim + f] /= counts[d];
                    }
                }
            }

            let device = src_emb.device();
            let agg_tensor: Tensor<B, 2> =
                Tensor::<B, 1>::from_data(agg.as_slice(), &device).reshape([dst_count, hidden_dim]);

            // Accumulate into mpnn_out for dst_type
            mpnn_out
                .entry(et.2.clone())
                .and_modify(|existing| {
                    *existing = existing.clone() + agg_tensor.clone();
                })
                .or_insert(agg_tensor);
        }

        // ── Step 2: Global Self-Attention (per node type) ──
        for nt in graph.node_types() {
            let x = match embeddings.get(nt) {
                Some(e) => e,
                None => continue,
            };

            let n = x.dims()[0];
            let d = x.dims()[1];

            let residual = x.clone();

            // Local MPNN result + bias + ReLU
            let local_out = match mpnn_out.get(nt) {
                Some(m) => {
                    // Apply MPNN bias through the linear layer (acts as bias transform)
                    let biased = self.mpnn_bias.forward(m.clone());
                    biased.clamp_min(0.0) // ReLU
                }
                None => Tensor::zeros([n, d], &x.device()),
            };

            // Global self-attention
            let max_attn_nodes = 5000;
            let global_out = if n <= 1 {
                x.clone()
            } else if n > max_attn_nodes {
                eprintln!(
                    "      [GT] Skipping global attn for {} ({} nodes > {})",
                    nt, n, max_attn_nodes
                );
                x.clone()
            } else {
                let scale = (self.head_dim as f64).sqrt();
                let q = self.attn_q.forward(x.clone());
                let k = self.attn_k.forward(x.clone());
                let v = self.attn_v.forward(x.clone());
                let scores = q.matmul(k.transpose()) / scale;
                let attn_weights = burn::tensor::activation::softmax(scores, 1);
                let attn_out = attn_weights.matmul(v);
                self.attn_out.forward(attn_out)
            };

            // ── Step 3: Combine local + global, then FFN with residual ──
            let combined = local_out + global_out;
            let h1 = combined + residual;

            // FFN: Linear → ReLU → Linear
            let ffn_out = self.ffn1.forward(h1.clone());
            let ffn_out = ffn_out.clamp_min(0.0); // ReLU
            let ffn_out = self.ffn2.forward(ffn_out);

            // Residual connection 2
            let final_out = ffn_out + h1;

            output.insert(nt, final_out);
        }

        output
    }
}

/// Multi-layer Graph Transformer model.
#[derive(Module, Debug)]
pub struct GraphTransformerModel<B: Backend> {
    /// Input projection per node type
    pub input_projs: Vec<nn::Linear<B>>,
    /// GPS layers
    layers: Vec<GpsLayer<B>>,
    /// Optional HeteroDoRA adapters for input projections.
    pub input_adapters: Option<crate::model::lora::HeteroBasisAdapter<B>>,
    /// Node type keys for input projection lookup
    #[module(skip)]
    node_type_keys: Vec<String>,
}

impl<B: Backend> GraphTransformerModel<B> {
    pub fn attach_adapter(&mut self, adapter: crate::model::lora::HeteroBasisAdapter<B>) {
        self.input_adapters = Some(adapter);
    }
    pub fn adapter_param_count(&self) -> usize {
        self.input_adapters.as_ref().map_or(0, |a| a.param_count())
    }
    pub fn base_param_count(&self) -> usize {
        self.input_projs
            .iter()
            .map(|l| {
                let d = l.weight.val().dims();
                d[0] * d[1]
            })
            .sum()
    }

    pub fn forward(&self, graph: &HeteroGraph<B>) -> NodeEmbeddings<B> {
        let mut embeddings = NodeEmbeddings::new();
        for (i, nt) in self.node_type_keys.iter().enumerate() {
            if let Some(feat) = graph.node_features.get(nt) {
                let mut projected = self.input_projs[i].forward(feat.clone());

                // DoRA: y = m * normalize(base + adapter)
                if let Some(ref adapter) = self.input_adapters {
                    projected = adapter.dora_forward(projected, feat.clone(), nt);
                }

                embeddings.insert(nt, projected);
            }
        }

        for layer in &self.layers {
            embeddings = layer.forward(&embeddings, graph);
        }

        embeddings
    }
}

impl GraphTransformerConfig {
    pub fn init_model<B: Backend>(
        &self,
        node_types: &[String],
        edge_types: &[EdgeType],
        device: &B::Device,
    ) -> GraphTransformerModel<B> {
        // Input projections per node type (using nn::Linear)
        let mut input_projs = Vec::new();
        let mut node_type_keys = Vec::new();
        for nt in node_types {
            let proj = nn::LinearConfig::new(self.in_dim, self.hidden_dim)
                .with_bias(false)
                .init(device);
            input_projs.push(proj);
            node_type_keys.push(nt.clone());
        }

        // GPS layers
        let mut layers = Vec::new();
        for _ in 0..self.num_layers {
            layers.push(GpsLayer::new(
                self.hidden_dim,
                self.num_heads,
                self.ffn_ratio,
                edge_types,
                device,
            ));
        }

        GraphTransformerModel {
            input_projs,
            layers,
            input_adapters: None,
            node_type_keys,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::graph_builder::{build_hetero_graph, GraphBuildConfig, GraphFact};
    use burn::backend::NdArray;

    type B = NdArray;

    fn gf(ht: &str, h: &str, r: &str, tt: &str, t: &str) -> GraphFact {
        GraphFact {
            src: (ht.to_string(), h.to_string()),
            relation: r.to_string(),
            dst: (tt.to_string(), t.to_string()),
        }
    }

    #[test]
    fn test_graph_transformer_forward() {
        let facts = vec![
            gf("user", "alice", "purchased", "item", "phone"),
            gf("user", "bob", "purchased", "item", "laptop"),
            gf("user", "alice", "reviewed", "item", "laptop"),
            gf("item", "phone", "in_category", "category", "electronics"),
            gf("item", "laptop", "in_category", "category", "electronics"),
        ];

        let config = GraphBuildConfig {
            node_feat_dim: 16,
            add_reverse_edges: true,
            add_self_loops: true,
        };
        let device = <B as Backend>::Device::default();
        let graph = build_hetero_graph::<B>(&facts, &config, &device);

        let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
        let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();

        let model = GraphTransformerConfig {
            in_dim: 16,
            hidden_dim: 32,
            num_heads: 4,
            num_layers: 2,
            ffn_ratio: 2,
            dropout: 0.0,
        }
        .init_model::<B>(&node_types, &edge_types, &device);

        let out = model.forward(&graph);

        // Check output has all node types
        for nt in &node_types {
            assert!(out.embeddings.contains_key(nt), "Missing node type: {}", nt);
            let emb = &out.embeddings[nt];
            assert_eq!(emb.dims()[1], 32, "Wrong hidden dim for {}", nt);
        }

        println!(
            "✅ Graph Transformer forward pass: {} node types",
            node_types.len()
        );
        for nt in &node_types {
            println!(
                "   {}: {} nodes × {} dim",
                nt,
                out.embeddings[nt].dims()[0],
                out.embeddings[nt].dims()[1]
            );
        }
    }
}
