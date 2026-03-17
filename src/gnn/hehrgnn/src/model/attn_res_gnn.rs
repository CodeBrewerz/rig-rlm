//! AttnRes-GNN: Attention Residuals for Graph Neural Networks.
//!
//! Integration of the `attnres` crate (MoonshotAI/Kimi) with our GNN backbone.
//! Replaces fixed residual connections with learned, input-dependent attention
//! over depth, enabling selective information routing across layers.
//!
//! Three integration levels:
//! 1. `AttnResGraphSage` — AttnRes replaces mHC stream mixing  
//! 2. `AttnResEmbeddingPolicy` — AttnRes-enhanced RL policy network  
//! 3. Standalone `AttnResOp` benchmark — pure depth-attention test

use attnres::{AttnResConfig, AttnResOp};
use burn::module::Param;
use burn::nn;
use burn::prelude::*;

use crate::data::hetero_graph::{EdgeType, HeteroGraph};
use crate::model::backbone::NodeEmbeddings;
use crate::model::graphsage::{GraphSageConfig, GraphSageLayer};

// ═══════════════════════════════════════════════════════════════
// Generic Depth-Attention Wrapper (used by ALL GNN models)
// ═══════════════════════════════════════════════════════════════

/// Reusable AttnRes depth-attention for ANY multi-layer GNN.
///
/// Instead of simple sequential: `for layer in layers { emb = layer(emb, g); }`
/// this wraps each layer with AttnRes attention over all previous layer outputs,
/// preventing over-smoothing and enabling selective information routing.
///
/// Usage in any model:
/// ```ignore
/// let wrapper = DepthAttnWrapper::new(num_layers, hidden_dim, &device);
/// let result = wrapper.forward_with_layers(input_embeddings, graph, |emb, g, l| {
///     layers[l].forward(emb, g)
/// });
/// ```
#[derive(Module, Debug)]
pub struct DepthAttnWrapper<B: Backend> {
    /// Per-layer AttnRes operators for depth-attention
    pub attn_ops: Vec<AttnResOp<B>>,
    #[module(skip)]
    pub hidden_dim: usize,
    #[module(skip)]
    pub num_layers: usize,
    #[module(skip)]
    pub block_size: usize,
}

impl<B: Backend> DepthAttnWrapper<B> {
    /// Create a new depth-attention wrapper.
    pub fn new(
        num_layers: usize,
        hidden_dim: usize,
        num_blocks: usize,
        device: &B::Device,
    ) -> Self {
        let config =
            AttnResConfig::new(hidden_dim, num_layers * 2, num_blocks.max(1)).with_num_heads(4);

        let mut attn_ops = Vec::new();
        for _ in 0..num_layers {
            attn_ops.push(config.init_op(device));
        }

        let block_size = if num_blocks > 0 {
            (num_layers / num_blocks).max(1)
        } else {
            1
        };

        Self {
            attn_ops,
            hidden_dim,
            num_layers,
            block_size,
        }
    }

    /// Run forward pass with depth-attention between layers.
    ///
    /// `layer_fn(embeddings, graph, layer_idx)` is your GNN layer's forward fn.
    pub fn forward_with_layers<F>(
        &self,
        initial_embeddings: NodeEmbeddings<B>,
        graph: &HeteroGraph<B>,
        num_layers: usize,
        layer_fn: F,
    ) -> NodeEmbeddings<B>
    where
        F: Fn(&NodeEmbeddings<B>, &HeteroGraph<B>, usize) -> NodeEmbeddings<B>,
    {
        let node_types: Vec<String> = graph.node_types().iter().map(|s| (*s).clone()).collect();
        let mut current = initial_embeddings;

        // Track completed blocks per node type for Block AttnRes
        let mut completed_blocks: std::collections::HashMap<String, Vec<Tensor<B, 3>>> =
            std::collections::HashMap::new();
        for nt in &node_types {
            completed_blocks.insert(nt.clone(), Vec::<Tensor<B, 3>>::new());
        }

        for l in 0..num_layers.min(self.attn_ops.len()) {
            // Step 1: AttnRes — attend over completed blocks
            if l > 0 {
                let mut attn_enhanced = NodeEmbeddings::new();
                for nt in &node_types {
                    let blocks = completed_blocks.get(nt).unwrap();
                    if let Some(current_emb) = current.get(nt) {
                        if blocks.is_empty() {
                            attn_enhanced.insert(nt, current_emb.clone());
                        } else {
                            let [num_nodes, dim] = current_emb.dims();
                            let partial = current_emb.clone().reshape([1, num_nodes, dim]);
                            let h = self.attn_ops[l].forward(blocks, &partial);
                            let [_, n, d] = h.dims();
                            attn_enhanced.insert(nt, h.reshape([n, d]));
                        }
                    }
                }
                current = attn_enhanced;
            }

            // Step 2: Run GNN layer
            let gnn_output = layer_fn(&current, graph, l);

            // Step 3: Update block boundaries
            if self.block_size > 0 && (l + 1) % self.block_size == 0 {
                for nt in &node_types {
                    if let Some(emb) = gnn_output.get(nt) {
                        let [n, d] = emb.dims();
                        completed_blocks
                            .get_mut(nt)
                            .unwrap()
                            .push(emb.clone().reshape([1, n, d]));
                    }
                }
            }

            current = gnn_output;
        }

        current
    }

    /// Parameter count for the depth-attention ops.
    pub fn param_count(&self) -> usize {
        // Each AttnResOp has pseudo_query (d) + RmsNorm (d)
        self.attn_ops.len() * self.hidden_dim * 2
    }
}

// ═══════════════════════════════════════════════════════════════
// Integration 1: AttnRes-Enhanced GraphSAGE Model
// ═══════════════════════════════════════════════════════════════

/// Configuration for AttnRes-enhanced GraphSAGE model.
#[derive(Debug, Clone)]
pub struct AttnResGraphSageConfig {
    pub in_dim: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_blocks: usize, // Block AttnRes: how many blocks to partition layers into
    pub dropout: f64,
}

/// AttnRes-enhanced GraphSAGE model.
///
/// Instead of mHC's fixed Sinkhorn matrices, each layer uses a learned
/// pseudo-query vector to selectively attend over ALL previous layer outputs.
///
/// Layer update:
///   h_l = AttnRes(h_0, h_1, ..., h_{l-1}) → GNN → h_l
#[derive(Module, Debug)]
pub struct AttnResGraphSageModel<B: Backend> {
    /// GNN layers.
    pub layers: Vec<GraphSageLayer<B>>,
    /// Per-layer AttnRes operators (one per sublayer = 2 × num_layers).
    pub attn_res_ops: Vec<AttnResOp<B>>,
    /// Input projection per node type.
    pub input_linears: Vec<nn::Linear<B>>,
    /// Output projection.
    pub output_linear: nn::Linear<B>,
    /// Learnable node-type embeddings.
    type_embeddings: Vec<Param<Tensor<B, 2>>>,
    #[module(skip)]
    node_type_keys: Vec<String>,
    #[module(skip)]
    pub hidden_dim: usize,
    #[module(skip)]
    pub num_layers: usize,
    #[module(skip)]
    pub block_size: usize, // layers per block
}

fn align_feature_dim<B: Backend>(features: Tensor<B, 2>, expected_dim: usize) -> Tensor<B, 2> {
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

impl AttnResGraphSageConfig {
    pub fn init<B: Backend>(
        &self,
        node_types: &[String],
        edge_types: &[EdgeType],
        device: &B::Device,
    ) -> AttnResGraphSageModel<B> {
        // Input projections
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

        // GNN layers
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

        // AttnRes operators: one per layer (pre-GNN attention over depth)
        let attn_res_config = AttnResConfig::new(
            self.hidden_dim,
            self.num_layers * 2, // sublayers
            self.num_blocks,
        )
        .with_num_heads(4);

        let mut attn_res_ops = Vec::new();
        for _ in 0..self.num_layers {
            attn_res_ops.push(attn_res_config.init_op(device));
        }

        let output_linear = nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device);

        let block_size = if self.num_blocks > 0 {
            self.num_layers / self.num_blocks
        } else {
            1
        };

        AttnResGraphSageModel {
            layers,
            attn_res_ops,
            input_linears,
            output_linear,
            type_embeddings,
            node_type_keys,
            hidden_dim: self.hidden_dim,
            num_layers: self.num_layers,
            block_size: block_size.max(1),
        }
    }
}

impl<B: Backend> AttnResGraphSageModel<B> {
    /// Forward pass with Attention Residual connections.
    ///
    /// Instead of fixed residuals, each layer attends over all previous outputs.
    pub fn forward(&self, graph: &HeteroGraph<B>) -> NodeEmbeddings<B> {
        // Step 1: Input projection + type embeddings (same as mHC)
        let mut base_embeddings = NodeEmbeddings::new();
        for (node_type, features) in &graph.node_features {
            if let Some(idx) = self.node_type_keys.iter().position(|k| k == node_type) {
                let expected_in = self.input_linears[idx].weight.val().dims()[0];
                let aligned = align_feature_dim(features.clone(), expected_in);
                let mut projected = self.input_linears[idx].forward(aligned);
                if idx < self.type_embeddings.len() {
                    projected = projected + self.type_embeddings[idx].val();
                }
                base_embeddings.insert(node_type, projected);
            }
        }

        // Step 2: Layer-by-layer with AttnRes depth attention
        // Track all layer outputs per node type for AttnRes attention
        let node_types: Vec<String> = graph.node_types().iter().map(|s| (*s).clone()).collect();
        let mut all_layer_outputs: Vec<NodeEmbeddings<B>> = vec![base_embeddings.clone()];
        let mut current = base_embeddings;

        // Block tracking
        let mut completed_blocks: std::collections::HashMap<String, Vec<Tensor<B, 3>>> =
            std::collections::HashMap::new();
        for nt in &node_types {
            completed_blocks.insert(nt.clone(), Vec::<Tensor<B, 3>>::new());
        }

        for (l, layer) in self.layers.iter().enumerate() {
            // Step 2a: AttnRes — attend over all previous layer outputs
            let mut attn_input = NodeEmbeddings::new();
            for nt in &node_types {
                let blocks = completed_blocks.get(nt).unwrap();

                if let Some(current_emb) = current.get(nt) {
                    // Reshape to [1, N, D] for AttnRes (batch=1, seq=nodes, dim=hidden)
                    let [num_nodes, dim] = current_emb.dims();
                    let partial = current_emb.clone().reshape([1, num_nodes, dim]);

                    // Convert block tensors to references
                    if blocks.is_empty() {
                        // First layer - just use input directly
                        attn_input.insert(nt, current_emb.clone());
                    } else {
                        let block_refs: Vec<Tensor<B, 3>> = blocks.clone();
                        let h = self.attn_res_ops[l].forward(&block_refs, &partial);
                        let [_, n, d] = h.dims();
                        attn_input.insert(nt, h.reshape([n, d]));
                    }
                }
            }

            // Step 2b: Run GNN layer
            let gnn_output = layer.forward(&attn_input, graph);

            // Step 2c: Track outputs for future AttnRes attention
            // Update block boundaries
            if self.block_size > 0 && (l + 1) % self.block_size == 0 {
                for nt in &node_types {
                    if let Some(emb) = gnn_output.get(nt) {
                        let [n, d] = emb.dims();
                        let block_repr = emb.clone().reshape([1, n, d]);
                        completed_blocks.get_mut(nt).unwrap().push(block_repr);
                    }
                }
            }

            all_layer_outputs.push(gnn_output.clone());
            current = gnn_output;
        }

        // Step 3: Output projection
        let mut result = NodeEmbeddings::new();
        for nt in &node_types {
            if let Some(emb) = current.get(nt) {
                result.insert(nt, self.output_linear.forward(emb.clone()));
            }
        }

        result
    }

    /// Measure embedding variance (same interface as mHC for fair comparison).
    pub fn embedding_variance(&self, graph: &HeteroGraph<B>) -> f32 {
        let emb = self.forward(graph);
        let mut total_var = 0.0f32;
        let mut count = 0;
        for (_nt, tensor) in &emb.embeddings {
            let dims = tensor.dims();
            if dims[0] < 2 {
                continue;
            }
            let mean = tensor.clone().mean_dim(0);
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

    /// Count trainable parameters (AttnRes overhead).
    pub fn param_count(&self) -> usize {
        let input_params: usize = self
            .input_linears
            .iter()
            .map(|l| {
                let d = l.weight.val().dims();
                d[0] * d[1]
            })
            .sum();

        // Each AttnResOp has: pseudo_query (d) + norm (d)
        let attn_res_params = self.attn_res_ops.len() * self.hidden_dim * 2;

        let out_dims = self.output_linear.weight.val().dims();
        let output_params = out_dims[0] * out_dims[1];

        input_params + attn_res_params + output_params
    }
}

// ═══════════════════════════════════════════════════════════════
// Integration 3: AttnRes-Enhanced RL Policy
// ═══════════════════════════════════════════════════════════════

/// An RL policy that uses attention-over-depth instead of fixed residuals.
///
/// Instead of 2-layer MLP with simple residual, this uses N layers with
/// AttnRes connections — allowing each decision layer to selectively attend
/// to all previous feature transformations.
pub struct AttnResEmbeddingPolicy {
    state_dim: usize,
    hidden_dim: usize,
    num_actions: usize,
    num_layers: usize,

    // Layer weights: [num_layers][hidden_dim][input_dim]
    weights: Vec<Vec<Vec<f32>>>,
    biases: Vec<Vec<f32>>,

    // AttnRes pseudo-queries: [num_layers][hidden_dim]
    // (zero-initialized per paper)
    pseudo_queries: Vec<Vec<f32>>,

    // Output layer
    w_out: Vec<Vec<f32>>,
    b_out: Vec<f32>,

    temperature: f64,
    baseline: f64,
    lr: f64,
    seed: u64,
}

impl AttnResEmbeddingPolicy {
    pub fn new(num_actions: usize, state_dim: usize) -> Self {
        let hidden_dim = 64;
        // Same depth as baseline MLP, but with AttnRes replacing fixed residuals
        let num_layers = 2;
        let mut seed = 7919u64;

        let mut next_randn = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = (seed >> 33) as f64 / (1u64 << 31) as f64;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (seed >> 33) as f64 / (1u64 << 31) as f64;
            let r = (-2.0 * (u1 + 1e-8).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            (r * 0.1) as f32
        };

        // Initialize layer weights
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut pseudo_queries = Vec::new();

        for l in 0..num_layers {
            let in_d = if l == 0 { state_dim } else { hidden_dim };
            let mut w = vec![vec![0.0; in_d]; hidden_dim];
            let b = vec![0.0; hidden_dim];
            for i in 0..hidden_dim {
                for j in 0..in_d {
                    w[i][j] = next_randn();
                }
            }
            weights.push(w);
            biases.push(b);

            // Zero-initialized pseudo-queries (paper: starts as uniform averaging)
            pseudo_queries.push(vec![0.0; hidden_dim]);
        }

        // Output layer
        let mut w_out = vec![vec![0.0; hidden_dim]; num_actions];
        let b_out = vec![0.0; num_actions];
        for i in 0..num_actions {
            for j in 0..hidden_dim {
                w_out[i][j] = next_randn();
            }
        }

        Self {
            state_dim,
            hidden_dim,
            num_actions,
            num_layers,
            weights,
            biases,
            pseudo_queries,
            w_out,
            b_out,
            temperature: 1.0,
            baseline: 0.0,
            lr: 0.01,
            seed,
        }
    }

    fn next_rng(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.seed >> 33) as f64 / (1u64 << 31) as f64
    }

    /// Forward pass with AttnRes depth attention.
    /// Returns (all_hidden_states, logits)  
    fn forward(&self, state: &[f32]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let mut layer_outputs: Vec<Vec<f32>> = Vec::new();
        let mut current = state.to_vec();

        for l in 0..self.num_layers {
            // 1. Apply AttnRes: attend over all previous outputs
            if l > 0 && !layer_outputs.is_empty() {
                current = self.attn_res_aggregate(&layer_outputs, &current, l);
            }

            // 2. Apply linear transform + ReLU
            let in_d = current.len();
            let mut h = vec![0.0f32; self.hidden_dim];
            for i in 0..self.hidden_dim {
                let mut sum = self.biases[l][i];
                for j in 0..in_d.min(self.weights[l][i].len()) {
                    sum += self.weights[l][i][j] * current[j];
                }
                h[i] = if sum > 0.0 { sum } else { 0.0 }; // ReLU
                h[i] = h[i].clamp(-10.0, 10.0);
            }

            layer_outputs.push(h.clone());
            current = h;
        }

        // Output layer
        let mut logits = vec![0.0f32; self.num_actions];
        for i in 0..self.num_actions {
            let mut sum = self.b_out[i];
            for j in 0..self.hidden_dim {
                sum += self.w_out[i][j] * current[j];
            }
            logits[i] = sum.clamp(-10.0, 10.0);
        }

        (layer_outputs, logits)
    }

    /// AttnRes aggregation: softmax attention over previous layer outputs
    /// using the learned pseudo-query for this layer.
    fn attn_res_aggregate(
        &self,
        prev_outputs: &[Vec<f32>],
        current_partial: &[f32],
        layer_idx: usize,
    ) -> Vec<f32> {
        let query = &self.pseudo_queries[layer_idx];
        let n_sources = prev_outputs.len() + 1; // prev layers + current partial

        // Compute attention logits: dot(query, normalized_source)
        let mut logits = Vec::new();
        for prev in prev_outputs {
            let dot: f32 = query.iter().zip(prev.iter()).map(|(q, v)| q * v).sum();
            logits.push(dot);
        }
        // Current partial
        let dot: f32 = query
            .iter()
            .zip(current_partial.iter())
            .take(self.hidden_dim.min(current_partial.len()))
            .map(|(q, v)| q * v)
            .sum();
        logits.push(dot);

        // Softmax
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut weights: Vec<f32> = logits.iter().map(|l| (l - max_l).exp()).collect();
        let sum: f32 = weights.iter().sum();
        for w in &mut weights {
            *w /= sum + 1e-8;
        }

        // Weighted combination
        let mut result = vec![0.0f32; self.hidden_dim];
        for (idx, prev) in prev_outputs.iter().enumerate() {
            for d in 0..self.hidden_dim.min(prev.len()) {
                result[d] += weights[idx] * prev[d];
            }
        }
        // Add current partial contribution
        let last_w = weights[prev_outputs.len()];
        for d in 0..self.hidden_dim.min(current_partial.len()) {
            result[d] += last_w * current_partial[d];
        }

        result
    }

    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<f32> = logits
            .iter()
            .map(|l| ((l - max_l) / self.temperature as f32).exp())
            .collect();
        let sum: f32 = exps.iter().sum();
        for e in &mut exps {
            *e /= sum;
        }
        exps
    }

    pub fn reinforce_update(&mut self, action: usize, state: &[f32], reward: f64) {
        let advantage = (reward - self.baseline) as f32;
        let (layer_outputs, logits) = self.forward(state);
        let probs = self.softmax(&logits);

        let mut d_logits = vec![0.0f32; self.num_actions];
        for i in 0..self.num_actions {
            let target = if i == action { 1.0 } else { 0.0 };
            d_logits[i] = (target - probs[i]) * self.lr as f32 * advantage;
            d_logits[i] = d_logits[i].clamp(-1.0, 1.0);
        }

        // Backprop through output layer
        let last_h = &layer_outputs[self.num_layers - 1];
        let mut d_h = vec![0.0f32; self.hidden_dim];
        for i in 0..self.num_actions {
            self.b_out[i] += d_logits[i];
            for j in 0..self.hidden_dim {
                d_h[j] += d_logits[i] * self.w_out[i][j];
                self.w_out[i][j] += d_logits[i] * last_h[j];
            }
        }

        // Backprop through each layer (simplified — gradient flows through last path)
        for l in (0..self.num_layers).rev() {
            // ReLU gradient
            for j in 0..self.hidden_dim {
                if layer_outputs[l][j] <= 0.0 {
                    d_h[j] = 0.0;
                }
            }

            // Weight gradient
            let input = if l == 0 {
                state.to_vec()
            } else {
                layer_outputs[l - 1].clone()
            };
            let in_d = input.len().min(self.weights[l][0].len());
            let mut d_input = vec![0.0f32; in_d];
            for j in 0..self.hidden_dim {
                self.biases[l][j] += d_h[j];
                for k in 0..in_d {
                    d_input[k] += d_h[j] * self.weights[l][j][k];
                    self.weights[l][j][k] += d_h[j] * input[k];
                }
            }

            // Pseudo-query gradient: learn to attend to previous layers
            // Use the reward-weighted direction of the attended layer
            if l > 0 {
                let pq_lr = self.lr as f32 * advantage * 0.1;
                for d in 0..self.hidden_dim {
                    // Gradient of attention logit w.r.t. pseudo-query = source value
                    self.pseudo_queries[l][d] += pq_lr * layer_outputs[l - 1][d];
                    self.pseudo_queries[l][d] = self.pseudo_queries[l][d].clamp(-2.0, 2.0);
                }
            }

            d_h = d_input;
            d_h.resize(self.hidden_dim, 0.0);
        }

        self.baseline = 0.95 * self.baseline + 0.05 * reward;
    }

    pub fn train_from_buffer(&mut self, states: &[Vec<f32>], actions: &[usize], rewards: &[f32]) {
        let n = states.len().min(actions.len()).min(rewards.len());
        for i in 0..n {
            self.reinforce_update(actions[i], &states[i], rewards[i] as f64);
        }
    }
}

impl crate::eval::rl_policy::Policy for AttnResEmbeddingPolicy {
    fn select_action(&mut self, state: &[f32], available_actions: usize) -> usize {
        if available_actions == 0 {
            return 0;
        }

        let (_, logits) = self.forward(state);
        let probs = self.softmax(&logits);

        // Epsilon greedy
        if self.next_rng() < 0.1 {
            return (self.next_rng() * available_actions as f64).floor() as usize;
        }

        let u = self.next_rng() as f32;
        let mut cum = 0.0;
        for i in 0..available_actions.min(self.num_actions) {
            cum += probs[i];
            if u <= cum {
                return i;
            }
        }
        available_actions - 1
    }

    fn name(&self) -> &str {
        "attnres_dnn"
    }
}
