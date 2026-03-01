//! Graph Transformer (GPS-style) for heterogeneous graphs.
//!
//! Implements the GraphGPS architecture:
//! - Local MPNN: relation-typed message passing (like RGCN) for local context
//! - Global Attention: multi-head self-attention over all nodes of each type
//! - FFN + Residual: feed-forward network with skip connections
//!
//! Reference: Rampášek et al. "Recipe for a General, Powerful, Scalable
//! Graph Transformer" (NeurIPS 2022)

use burn::prelude::*;
use std::collections::HashMap;

use crate::data::hetero_graph::{EdgeType, HeteroGraph};
use crate::model::backbone::NodeEmbeddings;

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
#[derive(Debug)]
pub struct GpsLayer<B: Backend> {
    // Local MPNN weights (per edge type, like RGCN)
    mpnn_weights: HashMap<String, Tensor<B, 2>>,
    mpnn_bias: Tensor<B, 1>,

    // Global multi-head self-attention (per node type)
    // Q, K, V projections
    attn_q: Tensor<B, 2>,
    attn_k: Tensor<B, 2>,
    attn_v: Tensor<B, 2>,
    attn_out: Tensor<B, 2>,
    num_heads: usize,
    head_dim: usize,

    // FFN
    ffn_w1: Tensor<B, 2>,
    ffn_b1: Tensor<B, 1>,
    ffn_w2: Tensor<B, 2>,
    ffn_b2: Tensor<B, 1>,
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

        // Local MPNN: per-relation weight matrices
        let mut mpnn_weights = HashMap::new();
        for et in edge_types {
            let key = format!("{}_{}_{}", et.0, et.1, et.2);
            let w: Tensor<B, 2> = Tensor::random(
                [hidden_dim, hidden_dim],
                burn::tensor::Distribution::Uniform(
                    -(1.0 / (hidden_dim as f64).sqrt()),
                    1.0 / (hidden_dim as f64).sqrt(),
                ),
                device,
            );
            mpnn_weights.insert(key, w);
        }
        let mpnn_bias: Tensor<B, 1> = Tensor::zeros([hidden_dim], device);

        // Global self-attention projections
        let scale = 1.0 / (hidden_dim as f64).sqrt();
        let attn_q: Tensor<B, 2> = Tensor::random(
            [hidden_dim, hidden_dim],
            burn::tensor::Distribution::Uniform(-scale, scale),
            device,
        );
        let attn_k = Tensor::random(
            [hidden_dim, hidden_dim],
            burn::tensor::Distribution::Uniform(-scale, scale),
            device,
        );
        let attn_v = Tensor::random(
            [hidden_dim, hidden_dim],
            burn::tensor::Distribution::Uniform(-scale, scale),
            device,
        );
        let attn_out = Tensor::random(
            [hidden_dim, hidden_dim],
            burn::tensor::Distribution::Uniform(-scale, scale),
            device,
        );

        // FFN layers
        let ffn_scale = 1.0 / (ffn_hidden as f64).sqrt();
        let ffn_w1: Tensor<B, 2> = Tensor::random(
            [hidden_dim, ffn_hidden],
            burn::tensor::Distribution::Uniform(-scale, scale),
            device,
        );
        let ffn_b1: Tensor<B, 1> = Tensor::zeros([ffn_hidden], device);
        let ffn_w2: Tensor<B, 2> = Tensor::random(
            [ffn_hidden, hidden_dim],
            burn::tensor::Distribution::Uniform(-ffn_scale, ffn_scale),
            device,
        );
        let ffn_b2: Tensor<B, 1> = Tensor::zeros([hidden_dim], device);

        Self {
            mpnn_weights,
            mpnn_bias,
            attn_q,
            attn_k,
            attn_v,
            attn_out,
            num_heads,
            head_dim,
            ffn_w1,
            ffn_b1,
            ffn_w2,
            ffn_b2,
        }
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
            let key = format!("{}_{}_{}", et.0, et.1, et.2);
            let weight = match self.mpnn_weights.get(&key) {
                Some(w) => w,
                None => continue,
            };

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

            // Transform source embeddings: src_emb @ weight
            let transformed = src_emb.clone().matmul(weight.clone()); // [num_src, hidden]
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
            let msg_expanded = msg.clone();

            // Manual scatter-add
            let msg_data: Vec<f32> = msg_expanded
                .clone()
                .into_data()
                .as_slice::<f32>()
                .unwrap()
                .to_vec();
            let dst_data: Vec<i64> = dst_indices.into_data().as_slice::<i64>().unwrap().to_vec();

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
        // For each node type, apply multi-head self-attention
        for nt in graph.node_types() {
            let x = match embeddings.get(nt) {
                Some(e) => e,
                None => continue,
            };

            let n = x.dims()[0];
            let d = x.dims()[1];

            // Residual connection input = x
            let residual = x.clone();

            // Local MPNN result + bias + ReLU
            let local_out = match mpnn_out.get(nt) {
                Some(m) => {
                    let biased =
                        m.clone() + self.mpnn_bias.clone().unsqueeze_dim::<2>(0).expand([n, d]);
                    biased.clamp_min(0.0) // ReLU
                }
                None => Tensor::zeros([n, d], &x.device()),
            };

            // Global self-attention over all nodes of this type
            let global_out = if n <= 1 {
                x.clone()
            } else {
                // Q, K, V projections: [n, d]
                let q = x.clone().matmul(self.attn_q.clone()); // [n, d]
                let k = x.clone().matmul(self.attn_k.clone()); // [n, d]
                let v = x.clone().matmul(self.attn_v.clone()); // [n, d]

                // For efficiency with large node counts, use chunked attention
                if n > 2000 {
                    // Chunked: process in blocks of 2000 to avoid O(N²) memory
                    self.chunked_attention(&q, &k, &v, n, d, &x.device())
                } else {
                    // Full attention: [n, n] attention matrix
                    let scale = (self.head_dim as f64).sqrt();
                    let scores = q.clone().matmul(k.clone().transpose()); // [n, n]
                    let scores = scores / scale;
                    let attn_weights = burn::tensor::activation::softmax(scores, 1); // [n, n]
                    let attn_out = attn_weights.matmul(v); // [n, d]
                    attn_out.matmul(self.attn_out.clone()) // [n, d]
                }
            };

            // ── Step 3: Combine local + global, then FFN with residual ──
            // GPS formula: out = local + global (sum)
            let combined = local_out + global_out;

            // Residual connection 1
            let h1 = combined + residual;

            // FFN: Linear → ReLU → Linear
            let ffn_out = h1.clone().matmul(self.ffn_w1.clone())
                + self
                    .ffn_b1
                    .clone()
                    .unsqueeze_dim::<2>(0)
                    .expand([n, self.ffn_w1.dims()[1]]);
            let ffn_out = ffn_out.clamp_min(0.0); // ReLU
            let ffn_out = ffn_out.matmul(self.ffn_w2.clone())
                + self.ffn_b2.clone().unsqueeze_dim::<2>(0).expand([n, d]);

            // Residual connection 2
            let final_out = ffn_out + h1;

            output.insert(nt, final_out);
        }

        output
    }

    /// Chunked attention for large graphs (avoids O(N²) memory)
    fn chunked_attention(
        &self,
        q: &Tensor<B, 2>,
        k: &Tensor<B, 2>,
        v: &Tensor<B, 2>,
        n: usize,
        d: usize,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let chunk_size = 2000;
        let scale = (self.head_dim as f64).sqrt();
        let mut result_data = vec![0.0f32; n * d];

        let k_t = k.clone().transpose(); // [d, n]

        for start in (0..n).step_by(chunk_size) {
            let end = (start + chunk_size).min(n);
            let chunk_n = end - start;

            // q_chunk: [chunk_n, d]
            let q_chunk = q.clone().slice([start..end, 0..d]);
            let scores = q_chunk.matmul(k_t.clone()) / scale; // [chunk_n, n]
            let attn = burn::tensor::activation::softmax(scores, 1);
            let out_chunk = attn.matmul(v.clone()); // [chunk_n, d]
            let out_proj = out_chunk.matmul(self.attn_out.clone()); // [chunk_n, d]

            let chunk_data: Vec<f32> = out_proj.into_data().as_slice::<f32>().unwrap().to_vec();

            for i in 0..chunk_n {
                for j in 0..d {
                    result_data[(start + i) * d + j] = chunk_data[i * d + j];
                }
            }
        }

        Tensor::<B, 1>::from_data(result_data.as_slice(), device).reshape([n, d])
    }
}

/// Multi-layer Graph Transformer model.
#[derive(Debug)]
pub struct GraphTransformerModel<B: Backend> {
    /// Input projection per node type
    input_proj: HashMap<String, Tensor<B, 2>>,
    /// GPS layers
    layers: Vec<GpsLayer<B>>,
}

impl<B: Backend> GraphTransformerModel<B> {
    pub fn forward(&self, graph: &HeteroGraph<B>) -> NodeEmbeddings<B> {
        // Project input features to hidden dim
        let mut embeddings = NodeEmbeddings::new();
        for nt in graph.node_types() {
            if let (Some(feat), Some(proj)) = (graph.node_features.get(nt), self.input_proj.get(nt))
            {
                let projected = feat.clone().matmul(proj.clone());
                embeddings.insert(nt, projected);
            }
        }

        // Apply GPS layers
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
        let scale = 1.0 / (self.in_dim as f64).sqrt();

        // Input projections per node type
        let mut input_proj = HashMap::new();
        for nt in node_types {
            let proj: Tensor<B, 2> = Tensor::random(
                [self.in_dim, self.hidden_dim],
                burn::tensor::Distribution::Uniform(-scale, scale),
                device,
            );
            input_proj.insert(nt.clone(), proj);
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

        GraphTransformerModel { input_proj, layers }
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
            assert!(out.data.contains_key(nt), "Missing node type: {}", nt);
            let emb = &out.data[nt];
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
                out.data[nt].dims()[0],
                out.data[nt].dims()[1]
            );
        }
    }
}
