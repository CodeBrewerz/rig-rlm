//! mHC-GNN: Manifold-Constrained Hyper-Connections for Graph Neural Networks.
//!
//! Adapts the mHC framework (Xie et al., 2025 / Mishra, 2026) to our
//! heterogeneous GNN models. Each node maintains `n` parallel representation
//! streams mixed through doubly stochastic matrices (Sinkhorn-normalized).
//!
//! Key benefits:
//! - Exponentially slower over-smoothing: rate (1 - γ)^(L/n) vs (1 - γ)^L
//! - Enables deep GNNs (8-128 layers) without feature collapse
//! - Architecture-agnostic wrapper around existing GNN layers
//!
//! Reference: arXiv 2601.02451v1

use burn::module::Param;
use burn::nn;
use burn::prelude::*;

use crate::data::hetero_graph::{EdgeType, HeteroGraph};
use crate::model::backbone::NodeEmbeddings;
use crate::model::graphsage::{GraphSageConfig, GraphSageLayer};

// ═══════════════════════════════════════════════════════════════
// Sinkhorn-Knopp Normalization
// ═══════════════════════════════════════════════════════════════

/// Project a matrix onto the Birkhoff polytope (set of doubly stochastic
/// matrices) via Sinkhorn-Knopp iterations.
///
/// A doubly stochastic matrix has all rows and columns summing to 1.
/// This guarantees feature mean conservation and bounded signal propagation.
pub fn sinkhorn_normalize<B: Backend>(raw: Tensor<B, 2>, iterations: usize) -> Tensor<B, 2> {
    // Start with exp(raw) to ensure positivity
    let mut m = raw.exp();

    for _ in 0..iterations {
        // Row normalization: each row sums to 1
        let row_sum = m.clone().sum_dim(1); // [n, 1]
        let row_sum_safe = row_sum.clamp_min(1e-8);
        let dims = m.dims();
        m = m / row_sum_safe.expand([dims[0], dims[1]]);

        // Column normalization: each column sums to 1
        let col_sum = m.clone().sum_dim(0); // [1, n]
        let col_sum_safe = col_sum.clamp_min(1e-8);
        m = m / col_sum_safe.expand([dims[0], dims[1]]);
    }

    m
}

// ═══════════════════════════════════════════════════════════════
// mHC-GNN Layer
// ═══════════════════════════════════════════════════════════════

/// Per-layer mHC connection matrices.
///
/// For each GNN layer l, we have three learnable matrices:
/// - H_res: n×n stream mixing for residual path (Sinkhorn-normalized)
/// - H_pre: 1×n stream aggregation for GNN input
/// - H_post: 1×n stream expansion for GNN output
#[derive(Module, Debug)]
pub struct MhcConnection<B: Backend> {
    /// Raw parameters for H_res [n × n] (will be Sinkhorn-normalized).
    pub h_res_raw: Param<Tensor<B, 2>>,
    /// Raw parameters for H_pre [1 × n] (softmax-normalized).
    pub h_pre_raw: Param<Tensor<B, 2>>,
    /// Raw parameters for H_post [1 × n] (softmax-normalized).
    pub h_post_raw: Param<Tensor<B, 2>>,
    /// Number of parallel streams.
    #[module(skip)]
    pub n_streams: usize,
}

/// Initialize mHC connection matrices.
pub fn init_mhc_connection<B: Backend>(n_streams: usize, device: &B::Device) -> MhcConnection<B> {
    // Initialize H_res near identity (diagonal dominant)
    let mut h_res_data = vec![0.0f32; n_streams * n_streams];
    for i in 0..n_streams {
        h_res_data[i * n_streams + i] = 1.0; // diagonal = 1, off-diagonal = 0
    }
    let h_res =
        Tensor::<B, 1>::from_data(h_res_data.as_slice(), device).reshape([n_streams, n_streams]);

    // Initialize H_pre and H_post to uniform (equal stream contribution)
    let h_pre = Tensor::<B, 2>::ones([1, n_streams], device);
    let h_post = Tensor::<B, 2>::ones([1, n_streams], device);

    MhcConnection {
        h_res_raw: Param::from_tensor(h_res),
        h_pre_raw: Param::from_tensor(h_pre),
        h_post_raw: Param::from_tensor(h_post),
        n_streams,
    }
}

impl<B: Backend> MhcConnection<B> {
    /// Get the Sinkhorn-normalized residual mixing matrix.
    pub fn h_res(&self) -> Tensor<B, 2> {
        sinkhorn_normalize(self.h_res_raw.val(), 5)
    }

    /// Get the softmax-normalized pre-aggregation weights.
    pub fn h_pre(&self) -> Tensor<B, 2> {
        burn::tensor::activation::softmax(self.h_pre_raw.val(), 1) // [1, n]
    }

    /// Get the softmax-normalized post-expansion weights.
    pub fn h_post(&self) -> Tensor<B, 2> {
        burn::tensor::activation::softmax(self.h_post_raw.val(), 1) // [1, n]
    }
}

// ═══════════════════════════════════════════════════════════════
// mHC GraphSAGE Model
// ═══════════════════════════════════════════════════════════════

/// Configuration for mHC-enhanced GraphSAGE model.
#[derive(Debug, Clone)]
pub struct MhcGraphSageConfig {
    pub in_dim: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub n_streams: usize,
    pub dropout: f64,
}

/// mHC-enhanced GraphSAGE model.
///
/// Each node maintains `n_streams` parallel representation streams.
/// Between layers, streams are mixed via Sinkhorn-normalized matrices,
/// preventing over-smoothing even at 8+ layers.
///
/// Layer update:
///   x^(l+1) = H_res · x^(l) + H_post^T · GNN(H_pre · x^(l))
#[derive(Module, Debug)]
pub struct MhcGraphSageModel<B: Backend> {
    /// GNN layers (standard GraphSAGE message passing).
    pub layers: Vec<GraphSageLayer<B>>,
    /// Per-layer mHC connection matrices.
    pub connections: Vec<MhcConnection<B>>,
    /// Input projection per node type.
    pub input_linears: Vec<nn::Linear<B>>,
    /// Output aggregation: project n*hidden → hidden.
    pub output_linear: nn::Linear<B>,
    /// Learnable node-type embedding (KumoRFM §2.3)
    type_embeddings: Vec<Param<Tensor<B, 2>>>,
    /// Node type key mapping.
    #[module(skip)]
    node_type_keys: Vec<String>,
    /// Number of parallel streams.
    #[module(skip)]
    pub n_streams: usize,
    /// Hidden dimension per stream.
    #[module(skip)]
    pub hidden_dim: usize,
}

impl MhcGraphSageConfig {
    pub fn init<B: Backend>(
        &self,
        node_types: &[String],
        edge_types: &[EdgeType],
        device: &B::Device,
    ) -> MhcGraphSageModel<B> {
        // Input projections per node type
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

        // GNN layers (one per depth level)
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

        // mHC connection matrices (one per layer)
        let mut connections = Vec::new();
        for _ in 0..self.num_layers {
            connections.push(init_mhc_connection(self.n_streams, device));
        }

        // Output: aggregate n streams → single embedding
        let output_linear =
            nn::LinearConfig::new(self.hidden_dim * self.n_streams, self.hidden_dim).init(device);

        MhcGraphSageModel {
            layers,
            connections,
            input_linears,
            output_linear,
            type_embeddings,
            node_type_keys,
            n_streams: self.n_streams,
            hidden_dim: self.hidden_dim,
        }
    }
}

impl<B: Backend> MhcGraphSageModel<B> {
    /// Forward pass with multi-stream representations.
    ///
    /// 1. Project inputs → hidden_dim
    /// 2. Expand to n_streams copies
    /// 3. For each layer: mix streams (H_res), aggregate (H_pre) → GNN → expand (H_post)
    /// 4. Concatenate streams → output projection
    pub fn forward(&self, graph: &HeteroGraph<B>) -> NodeEmbeddings<B> {
        // Step 1: Input projection (same as standard GraphSAGE)
        let mut stream_embeddings: Vec<NodeEmbeddings<B>> = Vec::new();

        // Initialize n_streams copies of input embeddings
        let mut base_embeddings = NodeEmbeddings::new();
        for (node_type, features) in &graph.node_features {
            if let Some(idx) = self.node_type_keys.iter().position(|k| k == node_type) {
                let mut projected = self.input_linears[idx].forward(features.clone());
                // Add learnable node-type embedding (KumoRFM §2.3)
                if idx < self.type_embeddings.len() {
                    projected = projected + self.type_embeddings[idx].val();
                }
                base_embeddings.insert(node_type, projected);
            }
        }

        // Step 2: Expand to n parallel streams (initially identical)
        for _ in 0..self.n_streams {
            stream_embeddings.push(base_embeddings.clone());
        }

        // Step 3: Apply each GNN layer with mHC mixing
        for (l, layer) in self.layers.iter().enumerate() {
            let conn = &self.connections[l];
            let h_res = conn.h_res(); // [n × n] doubly stochastic
            let h_pre = conn.h_pre(); // [1 × n] softmax weights
            let h_post = conn.h_post(); // [1 × n] softmax weights

            // 3a: Aggregate streams into single GNN input using H_pre weights
            let mut gnn_input = NodeEmbeddings::new();
            for node_type in graph.node_types() {
                let device = graph.node_features[node_type].device();
                let num_nodes = graph.node_counts[node_type];
                let mut aggregated = Tensor::<B, 2>::zeros([num_nodes, self.hidden_dim], &device);

                for s in 0..self.n_streams {
                    if let Some(emb) = stream_embeddings[s].get(node_type) {
                        let w = h_pre
                            .clone()
                            .slice([0..1, s..s + 1])
                            .reshape([1, 1])
                            .expand([num_nodes, self.hidden_dim]);
                        aggregated = aggregated + emb.clone() * w;
                    }
                }
                gnn_input.insert(node_type, aggregated);
            }

            // 3b: Run standard GNN layer on aggregated input
            let gnn_output = layer.forward(&gnn_input, graph);

            // 3c: Update streams with mHC formula:
            //   x_s^(l+1) = Σ_t H_res[s,t] * x_t^(l) + H_post[s] * GNN(H_pre · x^(l))
            let mut new_streams: Vec<NodeEmbeddings<B>> = Vec::new();
            for s in 0..self.n_streams {
                let mut new_emb = NodeEmbeddings::new();
                for node_type in graph.node_types() {
                    let device = graph.node_features[node_type].device();
                    let num_nodes = graph.node_counts[node_type];
                    let mut mixed = Tensor::<B, 2>::zeros([num_nodes, self.hidden_dim], &device);

                    // Residual mixing: Σ_t H_res[s,t] * x_t^(l)
                    for t in 0..self.n_streams {
                        if let Some(emb) = stream_embeddings[t].get(node_type) {
                            let w = h_res
                                .clone()
                                .slice([s..s + 1, t..t + 1])
                                .reshape([1, 1])
                                .expand([num_nodes, self.hidden_dim]);
                            mixed = mixed + emb.clone() * w;
                        }
                    }

                    // GNN contribution: H_post[s] * GNN_output
                    if let Some(gnn_out) = gnn_output.get(node_type) {
                        let w = h_post
                            .clone()
                            .slice([0..1, s..s + 1])
                            .reshape([1, 1])
                            .expand([num_nodes, self.hidden_dim]);
                        mixed = mixed + gnn_out.clone() * w;
                    }

                    new_emb.insert(node_type, mixed);
                }
                new_streams.push(new_emb);
            }
            stream_embeddings = new_streams;
        }

        // Step 4: Concatenate all streams and project to final embedding
        let mut result = NodeEmbeddings::new();
        for node_type in graph.node_types() {
            let mut parts: Vec<Tensor<B, 2>> = Vec::new();
            for s in 0..self.n_streams {
                if let Some(emb) = stream_embeddings[s].get(node_type) {
                    parts.push(emb.clone());
                }
            }
            if !parts.is_empty() {
                // Concatenate streams: [N, n*hidden_dim]
                let concatenated = Tensor::cat(parts, 1);
                // Project to final dim: [N, hidden_dim]
                let projected = self.output_linear.forward(concatenated);
                result.insert(node_type, projected);
            }
        }

        result
    }

    /// Measure embedding variance across node types.
    /// Higher variance = less over-smoothing.
    pub fn embedding_variance(&self, graph: &HeteroGraph<B>) -> f32 {
        let emb = self.forward(graph);
        let mut total_var = 0.0f32;
        let mut count = 0;
        for (_nt, tensor) in &emb.embeddings {
            let dims = tensor.dims();
            if dims[0] < 2 {
                continue;
            }
            // Variance along feature dimension per node, then mean
            let mean = tensor.clone().mean_dim(0); // [1, d]
            let diff = tensor.clone() - mean.expand(dims);
            let var = (diff.clone() * diff).mean();
            let var_val: f32 = var.into_data().as_slice::<f32>().unwrap()[0];
            total_var += var_val;
            count += 1;
        }
        if count > 0 {
            total_var / count as f32
        } else {
            0.0
        }
    }

    /// Count total trainable parameters (mHC overhead only; GNN layer params excluded).
    pub fn param_count(&self) -> usize {
        let input_params: usize = self
            .input_linears
            .iter()
            .map(|l| {
                let d = l.weight.val().dims();
                d[0] * d[1]
            })
            .sum();

        // Each connection: n*n (h_res) + n (h_pre) + n (h_post)
        let conn_params =
            self.connections.len() * (self.n_streams * self.n_streams + 2 * self.n_streams);

        // Output linear
        let out_dims = self.output_linear.weight.val().dims();
        let output_params = out_dims[0] * out_dims[1];

        input_params + conn_params + output_params
    }
}

// ═══════════════════════════════════════════════════════════════
// mHC RGCN Model
// ═══════════════════════════════════════════════════════════════

/// Configuration for mHC-enhanced RGCN model.
#[derive(Debug, Clone)]
pub struct MhcRgcnConfig {
    pub in_dim: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_bases: usize,
    pub n_streams: usize,
    pub dropout: f64,
}

/// mHC-enhanced RGCN model.
#[derive(Module, Debug)]
pub struct MhcRgcnModel<B: Backend> {
    pub layers: Vec<super::rgcn::RgcnLayer<B>>,
    pub connections: Vec<MhcConnection<B>>,
    pub input_linears: Vec<nn::Linear<B>>,
    pub output_linear: nn::Linear<B>,
    /// Learnable node-type embedding (KumoRFM §2.3)
    type_embeddings: Vec<Param<Tensor<B, 2>>>,
    #[module(skip)]
    node_type_keys: Vec<String>,
    #[module(skip)]
    pub n_streams: usize,
    #[module(skip)]
    pub hidden_dim: usize,
}

impl MhcRgcnConfig {
    pub fn init<B: Backend>(
        &self,
        node_types: &[String],
        edge_types: &[EdgeType],
        device: &B::Device,
    ) -> MhcRgcnModel<B> {
        let mut input_linears = Vec::new();
        let mut node_type_keys = Vec::new();
        let mut type_embeddings = Vec::new();
        for nt in node_types.iter() {
            input_linears.push(nn::LinearConfig::new(self.in_dim, self.hidden_dim).init(device));
            node_type_keys.push(nt.clone());
            let emb = Tensor::<B, 2>::random(
                [1, self.hidden_dim],
                burn::tensor::Distribution::Uniform(-0.1, 0.1),
                device,
            );
            type_embeddings.push(Param::from_tensor(emb));
        }

        let rgcn_cfg = super::rgcn::RgcnConfig {
            in_dim: self.in_dim,
            hidden_dim: self.hidden_dim,
            num_layers: self.num_layers,
            num_bases: self.num_bases,
            dropout: self.dropout,
        };
        let mut layers = Vec::new();
        let mut connections = Vec::new();
        for _ in 0..self.num_layers {
            layers.push(rgcn_cfg.init_layer(self.hidden_dim, edge_types, device));
            connections.push(init_mhc_connection(self.n_streams, device));
        }

        let output_linear =
            nn::LinearConfig::new(self.hidden_dim * self.n_streams, self.hidden_dim).init(device);

        MhcRgcnModel {
            layers,
            connections,
            input_linears,
            output_linear,
            type_embeddings,
            node_type_keys,
            n_streams: self.n_streams,
            hidden_dim: self.hidden_dim,
        }
    }
}

impl<B: Backend> MhcRgcnModel<B> {
    pub fn forward(&self, graph: &HeteroGraph<B>) -> NodeEmbeddings<B> {
        mhc_forward_generic(
            graph,
            &self.input_linears,
            &self.node_type_keys,
            &self.connections,
            &self.output_linear,
            self.n_streams,
            self.hidden_dim,
            Some(&self.type_embeddings),
            |layer_idx, input_emb, g| self.layers[layer_idx].forward(input_emb, g),
        )
    }
}

// ═══════════════════════════════════════════════════════════════
// mHC GAT Model
// ═══════════════════════════════════════════════════════════════

/// Configuration for mHC-enhanced GAT model.
#[derive(Debug, Clone)]
pub struct MhcGatConfig {
    pub in_dim: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub n_streams: usize,
    pub dropout: f64,
}

/// mHC-enhanced GAT model.
#[derive(Module, Debug)]
pub struct MhcGatModel<B: Backend> {
    pub layers: Vec<super::gat::GatLayer<B>>,
    pub connections: Vec<MhcConnection<B>>,
    pub input_linears: Vec<nn::Linear<B>>,
    pub output_linear: nn::Linear<B>,
    /// Learnable node-type embedding (KumoRFM §2.3)
    type_embeddings: Vec<Param<Tensor<B, 2>>>,
    #[module(skip)]
    node_type_keys: Vec<String>,
    #[module(skip)]
    pub n_streams: usize,
    #[module(skip)]
    pub hidden_dim: usize,
}

impl MhcGatConfig {
    pub fn init<B: Backend>(
        &self,
        node_types: &[String],
        edge_types: &[EdgeType],
        device: &B::Device,
    ) -> MhcGatModel<B> {
        let mut input_linears = Vec::new();
        let mut node_type_keys = Vec::new();
        let mut type_embeddings = Vec::new();
        for nt in node_types.iter() {
            input_linears.push(nn::LinearConfig::new(self.in_dim, self.hidden_dim).init(device));
            node_type_keys.push(nt.clone());
            let emb = Tensor::<B, 2>::random(
                [1, self.hidden_dim],
                burn::tensor::Distribution::Uniform(-0.1, 0.1),
                device,
            );
            type_embeddings.push(Param::from_tensor(emb));
        }

        let gat_cfg = super::gat::GatConfig {
            in_dim: self.in_dim,
            hidden_dim: self.hidden_dim,
            num_heads: self.num_heads,
            num_layers: self.num_layers,
            dropout: self.dropout,
        };
        let mut layers = Vec::new();
        let mut connections = Vec::new();
        for _ in 0..self.num_layers {
            layers.push(gat_cfg.init_layer(self.hidden_dim, edge_types, device));
            connections.push(init_mhc_connection(self.n_streams, device));
        }

        let output_linear =
            nn::LinearConfig::new(self.hidden_dim * self.n_streams, self.hidden_dim).init(device);

        MhcGatModel {
            layers,
            connections,
            input_linears,
            output_linear,
            type_embeddings,
            node_type_keys,
            n_streams: self.n_streams,
            hidden_dim: self.hidden_dim,
        }
    }
}

impl<B: Backend> MhcGatModel<B> {
    pub fn forward(&self, graph: &HeteroGraph<B>) -> NodeEmbeddings<B> {
        mhc_forward_generic(
            graph,
            &self.input_linears,
            &self.node_type_keys,
            &self.connections,
            &self.output_linear,
            self.n_streams,
            self.hidden_dim,
            Some(&self.type_embeddings),
            |layer_idx, input_emb, g| self.layers[layer_idx].forward(input_emb, g),
        )
    }
}

// ═══════════════════════════════════════════════════════════════
// mHC GPS Transformer Model
// ═══════════════════════════════════════════════════════════════

/// Configuration for mHC-enhanced GPS Transformer model.
#[derive(Debug, Clone)]
pub struct MhcGpsConfig {
    pub in_dim: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub ffn_ratio: usize,
    pub n_streams: usize,
    pub dropout: f64,
}

/// mHC-enhanced GPS Transformer model.
#[derive(Module, Debug)]
pub struct MhcGpsModel<B: Backend> {
    pub layers: Vec<super::graph_transformer::GpsLayer<B>>,
    pub connections: Vec<MhcConnection<B>>,
    pub input_linears: Vec<nn::Linear<B>>,
    pub output_linear: nn::Linear<B>,
    /// Learnable node-type embedding (KumoRFM §2.3)
    type_embeddings: Vec<Param<Tensor<B, 2>>>,
    #[module(skip)]
    node_type_keys: Vec<String>,
    #[module(skip)]
    pub n_streams: usize,
    #[module(skip)]
    pub hidden_dim: usize,
}

impl MhcGpsConfig {
    pub fn init<B: Backend>(
        &self,
        node_types: &[String],
        edge_types: &[EdgeType],
        device: &B::Device,
    ) -> MhcGpsModel<B> {
        let mut input_linears = Vec::new();
        let mut node_type_keys = Vec::new();
        let mut type_embeddings = Vec::new();
        for nt in node_types.iter() {
            input_linears.push(nn::LinearConfig::new(self.in_dim, self.hidden_dim).init(device));
            node_type_keys.push(nt.clone());
            let emb = Tensor::<B, 2>::random(
                [1, self.hidden_dim],
                burn::tensor::Distribution::Uniform(-0.1, 0.1),
                device,
            );
            type_embeddings.push(Param::from_tensor(emb));
        }

        let gps_cfg = super::graph_transformer::GraphTransformerConfig {
            in_dim: self.in_dim,
            hidden_dim: self.hidden_dim,
            num_heads: self.num_heads,
            num_layers: self.num_layers,
            ffn_ratio: self.ffn_ratio,
            dropout: self.dropout,
        };
        let mut layers = Vec::new();
        let mut connections = Vec::new();
        for _ in 0..self.num_layers {
            layers.push(super::graph_transformer::GpsLayer::new(
                self.hidden_dim,
                self.num_heads,
                self.ffn_ratio,
                edge_types,
                device,
            ));
            connections.push(init_mhc_connection(self.n_streams, device));
        }

        let output_linear =
            nn::LinearConfig::new(self.hidden_dim * self.n_streams, self.hidden_dim).init(device);

        MhcGpsModel {
            layers,
            connections,
            input_linears,
            output_linear,
            type_embeddings,
            node_type_keys,
            n_streams: self.n_streams,
            hidden_dim: self.hidden_dim,
        }
    }
}

impl<B: Backend> MhcGpsModel<B> {
    pub fn forward(&self, graph: &HeteroGraph<B>) -> NodeEmbeddings<B> {
        mhc_forward_generic(
            graph,
            &self.input_linears,
            &self.node_type_keys,
            &self.connections,
            &self.output_linear,
            self.n_streams,
            self.hidden_dim,
            Some(&self.type_embeddings),
            |layer_idx, input_emb, g| self.layers[layer_idx].forward(input_emb, g),
        )
    }
}

// ═══════════════════════════════════════════════════════════════
// Generic mHC Forward Pass
// ═══════════════════════════════════════════════════════════════

/// Generic mHC forward pass that works with any GNN layer.
///
/// The `layer_forward` closure takes (layer_index, input_embeddings, graph)
/// and returns output embeddings — making this work for any GNN type.
fn mhc_forward_generic<B: Backend, F>(
    graph: &HeteroGraph<B>,
    input_linears: &[nn::Linear<B>],
    node_type_keys: &[String],
    connections: &[MhcConnection<B>],
    output_linear: &nn::Linear<B>,
    n_streams: usize,
    hidden_dim: usize,
    type_embeddings: Option<&Vec<Param<Tensor<B, 2>>>>,
    layer_forward: F,
) -> NodeEmbeddings<B>
where
    F: Fn(usize, &NodeEmbeddings<B>, &HeteroGraph<B>) -> NodeEmbeddings<B>,
{
    // Step 1: Input projection
    let mut base_embeddings = NodeEmbeddings::new();
    for (node_type, features) in &graph.node_features {
        if let Some(idx) = node_type_keys.iter().position(|k| k == node_type) {
            let mut projected = input_linears[idx].forward(features.clone());
            // Add learnable node-type embedding (KumoRFM §2.3)
            if let Some(te) = type_embeddings {
                if idx < te.len() {
                    projected = projected + te[idx].val();
                }
            }
            base_embeddings.insert(node_type, projected);
        }
    }

    // Step 2: Expand to n parallel streams
    let mut stream_embeddings: Vec<NodeEmbeddings<B>> = Vec::new();
    for _ in 0..n_streams {
        stream_embeddings.push(base_embeddings.clone());
    }

    // Step 3: Apply each GNN layer with mHC mixing
    for (l, conn) in connections.iter().enumerate() {
        let h_res = conn.h_res();
        let h_pre = conn.h_pre();
        let h_post = conn.h_post();

        // 3a: Aggregate streams → single GNN input
        let mut gnn_input = NodeEmbeddings::new();
        for node_type in graph.node_types() {
            let device = graph.node_features[node_type].device();
            let num_nodes = graph.node_counts[node_type];
            let mut aggregated = Tensor::<B, 2>::zeros([num_nodes, hidden_dim], &device);

            for s in 0..n_streams {
                if let Some(emb) = stream_embeddings[s].get(node_type) {
                    let w = h_pre
                        .clone()
                        .slice([0..1, s..s + 1])
                        .reshape([1, 1])
                        .expand([num_nodes, hidden_dim]);
                    aggregated = aggregated + emb.clone() * w;
                }
            }
            gnn_input.insert(node_type, aggregated);
        }

        // 3b: Run GNN layer
        let gnn_output = layer_forward(l, &gnn_input, graph);

        // 3c: mHC stream update
        let mut new_streams: Vec<NodeEmbeddings<B>> = Vec::new();
        for s in 0..n_streams {
            let mut new_emb = NodeEmbeddings::new();
            for node_type in graph.node_types() {
                let device = graph.node_features[node_type].device();
                let num_nodes = graph.node_counts[node_type];
                let mut mixed = Tensor::<B, 2>::zeros([num_nodes, hidden_dim], &device);

                for t in 0..n_streams {
                    if let Some(emb) = stream_embeddings[t].get(node_type) {
                        let w = h_res
                            .clone()
                            .slice([s..s + 1, t..t + 1])
                            .reshape([1, 1])
                            .expand([num_nodes, hidden_dim]);
                        mixed = mixed + emb.clone() * w;
                    }
                }

                if let Some(gnn_out) = gnn_output.get(node_type) {
                    let w = h_post
                        .clone()
                        .slice([0..1, s..s + 1])
                        .reshape([1, 1])
                        .expand([num_nodes, hidden_dim]);
                    mixed = mixed + gnn_out.clone() * w;
                }

                new_emb.insert(node_type, mixed);
            }
            new_streams.push(new_emb);
        }
        stream_embeddings = new_streams;
    }

    // Step 4: Concatenate streams → output projection
    let mut result = NodeEmbeddings::new();
    for node_type in graph.node_types() {
        let mut parts: Vec<Tensor<B, 2>> = Vec::new();
        for s in 0..n_streams {
            if let Some(emb) = stream_embeddings[s].get(node_type) {
                parts.push(emb.clone());
            }
        }
        if !parts.is_empty() {
            let concatenated = Tensor::cat(parts, 1);
            let projected = output_linear.forward(concatenated);
            result.insert(node_type, projected);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::graph_builder::{build_hetero_graph, GraphBuildConfig, GraphFact};
    use burn::backend::NdArray;

    type B = NdArray;

    fn test_graph() -> HeteroGraph<B> {
        let device = <B as Backend>::Device::default();
        let facts = vec![
            GraphFact {
                src: ("user".into(), "alice".into()),
                relation: "owns".into(),
                dst: ("account".into(), "acc1".into()),
            },
            GraphFact {
                src: ("user".into(), "bob".into()),
                relation: "owns".into(),
                dst: ("account".into(), "acc2".into()),
            },
            GraphFact {
                src: ("tx".into(), "tx1".into()),
                relation: "posted_to".into(),
                dst: ("account".into(), "acc1".into()),
            },
            GraphFact {
                src: ("tx".into(), "tx2".into()),
                relation: "posted_to".into(),
                dst: ("account".into(), "acc2".into()),
            },
            GraphFact {
                src: ("tx".into(), "tx1".into()),
                relation: "at_merchant".into(),
                dst: ("merchant".into(), "walmart".into()),
            },
        ];

        build_hetero_graph::<B>(
            &facts,
            &GraphBuildConfig {
                node_feat_dim: 8,
                add_reverse_edges: true,
                add_self_loops: true,
                add_positional_encoding: true,
            },
            &device,
        )
    }

    #[test]
    fn test_sinkhorn_produces_doubly_stochastic() {
        let device = <B as Backend>::Device::default();
        let raw = Tensor::<B, 2>::random(
            [4, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let ds = sinkhorn_normalize(raw, 10);

        let data: Vec<f32> = ds.clone().into_data().as_slice::<f32>().unwrap().to_vec();

        // Check row sums ≈ 1
        for r in 0..4 {
            let row_sum: f32 = (0..4).map(|c| data[r * 4 + c]).sum();
            assert!((row_sum - 1.0).abs() < 0.01, "Row {} sum = {}", r, row_sum);
        }
        // Check column sums ≈ 1
        for c in 0..4 {
            let col_sum: f32 = (0..4).map(|r| data[r * 4 + c]).sum();
            assert!((col_sum - 1.0).abs() < 0.01, "Col {} sum = {}", c, col_sum);
        }
        // Check all entries >= 0
        for v in &data {
            assert!(*v >= 0.0, "Negative entry: {}", v);
        }
    }

    #[test]
    fn test_mhc_graphsage_forward() {
        let device = <B as Backend>::Device::default();
        let graph = test_graph();
        let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
        let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();

        let in_dim = graph
            .node_features
            .values()
            .next()
            .map(|t| t.dims()[1])
            .unwrap_or(8);

        let config = MhcGraphSageConfig {
            in_dim,
            hidden_dim: 16,
            num_layers: 8,
            n_streams: 2,
            dropout: 0.0,
        };

        let model = config.init::<B>(&node_types, &edge_types, &device);
        let embeddings = model.forward(&graph);

        // Verify all node types have embeddings with correct dims
        for nt in &node_types {
            let emb = embeddings.get(nt).expect(&format!("Missing {}", nt));
            assert_eq!(emb.dims()[0], graph.node_counts[nt]);
            assert_eq!(emb.dims()[1], 16, "Output dim should be hidden_dim");
        }

        println!("  ✅ mHC-GraphSAGE 8-layer forward pass works");
        println!("    Params: {}", model.param_count());
        println!("    Streams: {}", model.n_streams);
        println!("    Layers: {}", model.layers.len());
    }
}
