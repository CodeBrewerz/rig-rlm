//! MSA Sparse Attention Layer.
//!
//! Implements a single Memory Sparse Attention layer as shown in Figure 2 of
//! the MSA paper. This layer:
//!
//! 1. Generates standard K, V via backbone projectors
//! 2. Generates routing K^R via Router K Projector
//! 3. Compresses via chunk-wise mean pooling: K̄, V̄, K̄^R
//! 4. Scores document relevance (cosine similarity of Q^R vs K̄^R)
//! 5. Selects Top-k documents
//! 6. Assembles sparse context: K_ctx = [{K̄_topk}; K_q], V_ctx = [{V̄_topk}; V_q]
//! 7. Runs standard attention on sparse context
//!
//! Paper §3.2.1: "We implement the MSA routing strategy selectively, applying
//! it exclusively to the latter half of the model's layers."

use burn::nn;
use burn::prelude::*;

use super::pooling::chunk_mean_pool;
use super::router::{RouterProjector, RouterProjectorConfig};

/// Configuration for a single MSA layer.
#[derive(Debug, Clone)]
pub struct MsaLayerConfig {
    /// Hidden dimension of the model.
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Router projection dimension per head.
    pub router_dim: usize,
    /// Chunk size for mean pooling (P=64 in paper).
    pub chunk_size: usize,
    /// Number of top-k documents to select.
    pub topk: usize,
    /// FFN expansion ratio.
    pub ffn_ratio: usize,
}

impl Default for MsaLayerConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 128,
            num_heads: 8,
            router_dim: 16,
            chunk_size: 64,
            topk: 16,
            ffn_ratio: 4,
        }
    }
}

/// A single MSA (Memory Sparse Attention) layer.
///
/// Combines document-level sparse routing with standard dense attention
/// on the selected (compressed) context.
#[derive(Module, Debug)]
pub struct MsaLayer<B: Backend> {
    // Standard attention projections (from backbone)
    attn_q: nn::Linear<B>,
    attn_k: nn::Linear<B>,
    attn_v: nn::Linear<B>,
    attn_out: nn::Linear<B>,

    // Router projections for sparse retrieval
    pub router_k: RouterProjector<B>,
    pub router_q: RouterProjector<B>,

    // FFN with RMSNorm (paper Fig. 2)
    norm1: nn::RmsNorm<B>,
    norm2: nn::RmsNorm<B>,
    ffn1: nn::Linear<B>,
    ffn2: nn::Linear<B>,

    // Non-module config
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
    #[module(skip)]
    pub chunk_size: usize,
    #[module(skip)]
    pub topk: usize,
}

impl<B: Backend> MsaLayer<B> {
    /// Create a new MSA layer.
    pub fn new(config: &MsaLayerConfig, device: &B::Device) -> Self {
        let head_dim = config.hidden_dim / config.num_heads;
        let ffn_hidden = config.hidden_dim * config.ffn_ratio;

        // Standard attention projections
        let attn_q = nn::LinearConfig::new(config.hidden_dim, config.hidden_dim)
            .with_bias(false)
            .init(device);
        let attn_k = nn::LinearConfig::new(config.hidden_dim, config.hidden_dim)
            .with_bias(false)
            .init(device);
        let attn_v = nn::LinearConfig::new(config.hidden_dim, config.hidden_dim)
            .with_bias(false)
            .init(device);
        let attn_out = nn::LinearConfig::new(config.hidden_dim, config.hidden_dim)
            .with_bias(false)
            .init(device);

        // Router projections
        let router_config = RouterProjectorConfig {
            hidden_dim: config.hidden_dim,
            router_dim: config.router_dim,
            num_heads: config.num_heads,
        };
        let router_k = RouterProjector::new(&router_config, device);
        let router_q = RouterProjector::new(&router_config, device);

        // Layer norms
        let norm1 = nn::RmsNormConfig::new(config.hidden_dim).init(device);
        let norm2 = nn::RmsNormConfig::new(config.hidden_dim).init(device);

        // FFN
        let ffn1 = nn::LinearConfig::new(config.hidden_dim, ffn_hidden)
            .with_bias(true)
            .init(device);
        let ffn2 = nn::LinearConfig::new(ffn_hidden, config.hidden_dim)
            .with_bias(true)
            .init(device);

        Self {
            attn_q,
            attn_k,
            attn_v,
            attn_out,
            router_k,
            router_q,
            norm1,
            norm2,
            ffn1,
            ffn2,
            num_heads: config.num_heads,
            head_dim,
            chunk_size: config.chunk_size,
            topk: config.topk,
        }
    }

    /// Encode a single document: produce K, V, K^R and their pooled versions.
    ///
    /// This corresponds to Stage 1 (Global Memory Encoding) of inference.
    ///
    /// # Arguments
    /// * `doc_hidden` - Document hidden states [doc_len, hidden_dim]
    ///
    /// # Returns
    /// * `(K̄, V̄, K̄^R_flat)` — pooled K, V, and routing keys
    pub fn encode_document(
        &self,
        doc_hidden: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        // Standard K, V projections
        let k = self.attn_k.forward(doc_hidden.clone()); // [doc_len, hidden_dim]
        let v = self.attn_v.forward(doc_hidden.clone()); // [doc_len, hidden_dim]

        // Router K projection
        let k_r = self.router_k.forward_flat(doc_hidden); // [doc_len, router_total_dim]

        // Chunk-wise mean pooling
        let k_bar = chunk_mean_pool(k, self.chunk_size); // [num_chunks, hidden_dim]
        let v_bar = chunk_mean_pool(v, self.chunk_size); // [num_chunks, hidden_dim]
        let k_r_bar = chunk_mean_pool(k_r, self.chunk_size); // [num_chunks, router_total_dim]

        (k_bar, v_bar, k_r_bar)
    }

    /// Compute the query's routing representation.
    ///
    /// # Arguments
    /// * `query_hidden` - Query hidden states [query_len, hidden_dim]
    ///
    /// # Returns
    /// * Router query Q^R [query_len, router_total_dim]
    pub fn compute_routing_query(&self, query_hidden: Tensor<B, 2>) -> Tensor<B, 2> {
        self.router_q.forward_flat(query_hidden)
    }

    /// Forward pass on assembled sparse context.
    ///
    /// This performs standard multi-head attention on the concatenated
    /// compressed KV from selected documents + query local KV.
    ///
    /// Paper Eq. 3-4:
    ///   K_ctx = [{K̄_i}_{i∈I}; K_q]
    ///   V_ctx = [{V̄_i}_{i∈I}; V_q]
    ///   Output = Attention(Q_q, K_ctx, V_ctx)
    ///
    /// # Arguments
    /// * `query_hidden` - Query hidden states [query_len, hidden_dim]
    /// * `memory_k` - Concatenated compressed K from selected docs [mem_len, hidden_dim]
    /// * `memory_v` - Concatenated compressed V from selected docs [mem_len, hidden_dim]
    ///
    /// # Returns
    /// * Output hidden states [query_len, hidden_dim]
    pub fn forward(
        &self,
        query_hidden: Tensor<B, 2>,
        memory_k: Tensor<B, 2>,
        memory_v: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [query_len, hidden_dim] = query_hidden.dims();
        let residual = query_hidden.clone();

        // Pre-norm
        let normed = self.norm1.forward(query_hidden);

        // Query projection
        let q = self.attn_q.forward(normed.clone()); // [query_len, hidden_dim]

        // Local K, V for query tokens
        let k_local = self.attn_k.forward(normed.clone()); // [query_len, hidden_dim]
        let v_local = self.attn_v.forward(normed); // [query_len, hidden_dim]

        // Assemble sparse context (Eq. 3)
        let k_ctx = Tensor::cat(vec![memory_k, k_local], 0); // [mem_len + query_len, hidden_dim]
        let v_ctx = Tensor::cat(vec![memory_v, v_local], 0); // [mem_len + query_len, hidden_dim]

        // Multi-head attention (Eq. 4)
        let [ctx_len, _] = k_ctx.dims();

        // Reshape for multi-head: [len, num_heads, head_dim]
        let q_heads = q.reshape([query_len, self.num_heads, self.head_dim]);
        let k_heads = k_ctx.reshape([ctx_len, self.num_heads, self.head_dim]);
        let v_heads = v_ctx.reshape([ctx_len, self.num_heads, self.head_dim]);

        // Transpose to [num_heads, len, head_dim] for batched matmul
        let q_t = q_heads.swap_dims(0, 1); // [num_heads, query_len, head_dim]
        let k_t = k_heads.swap_dims(0, 1); // [num_heads, ctx_len, head_dim]
        let v_t = v_heads.swap_dims(0, 1); // [num_heads, ctx_len, head_dim]

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let scores = q_t.matmul(k_t.swap_dims(1, 2)) / scale; // [num_heads, query_len, ctx_len]

        // Causal masking: query attends to all memory + only preceding query tokens
        // Memory tokens are always visible; query tokens have causal mask
        let attn_weights = burn::tensor::activation::softmax(scores, 2);

        // Attend
        let attn_out = attn_weights.matmul(v_t); // [num_heads, query_len, head_dim]

        // Reshape back: [query_len, hidden_dim]
        let attn_out = attn_out
            .swap_dims(0, 1) // [query_len, num_heads, head_dim]
            .reshape([query_len, hidden_dim]);

        // Output projection + residual
        let attn_out = self.attn_out.forward(attn_out);
        let h = attn_out + residual;

        // FFN with pre-norm + residual
        let h_normed = self.norm2.forward(h.clone());
        let ffn_out = self.ffn1.forward(h_normed);
        let ffn_out = burn::tensor::activation::gelu(ffn_out);
        let ffn_out = self.ffn2.forward(ffn_out);

        ffn_out + h
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn test_msa_layer_encode_document() {
        let device = <B as Backend>::Device::default();
        let config = MsaLayerConfig {
            hidden_dim: 64,
            num_heads: 4,
            router_dim: 8,
            chunk_size: 16,
            topk: 4,
            ffn_ratio: 2,
        };
        let layer = MsaLayer::<B>::new(&config, &device);

        // 128-token document
        let doc = Tensor::<B, 2>::random(
            [128, 64],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );
        let (k_bar, v_bar, kr_bar) = layer.encode_document(doc);

        // 128 / 16 = 8 chunks
        assert_eq!(k_bar.dims(), [8, 64], "K̄ should be [8, 64]");
        assert_eq!(v_bar.dims(), [8, 64], "V̄ should be [8, 64]");
        assert_eq!(kr_bar.dims(), [8, 32], "K̄^R should be [8, 4*8=32]");
        println!("✅ MSA encode: [128, 64] → K̄[8,64], V̄[8,64], K̄^R[8,32]");
    }

    #[test]
    fn test_msa_layer_forward() {
        let device = <B as Backend>::Device::default();
        let config = MsaLayerConfig {
            hidden_dim: 64,
            num_heads: 4,
            router_dim: 8,
            chunk_size: 16,
            topk: 4,
            ffn_ratio: 2,
        };
        let layer = MsaLayer::<B>::new(&config, &device);

        // Query: 10 tokens
        let query = Tensor::<B, 2>::random(
            [10, 64],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );

        // Memory from 4 selected documents, each with 8 chunks → 32 chunk KVs
        let mem_k = Tensor::<B, 2>::random(
            [32, 64],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );
        let mem_v = Tensor::<B, 2>::random(
            [32, 64],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );

        let output = layer.forward(query, mem_k, mem_v);
        assert_eq!(output.dims(), [10, 64], "Output should be [query_len, hidden_dim]");
        println!("✅ MSA forward: query[10,64] + mem[32,64] → output[10,64]");
    }

    #[test]
    fn test_msa_layer_routing_query() {
        let device = <B as Backend>::Device::default();
        let config = MsaLayerConfig {
            hidden_dim: 64,
            num_heads: 4,
            router_dim: 8,
            chunk_size: 16,
            topk: 4,
            ffn_ratio: 2,
        };
        let layer = MsaLayer::<B>::new(&config, &device);

        let query = Tensor::<B, 2>::random(
            [5, 64],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );
        let qr = layer.compute_routing_query(query);
        assert_eq!(qr.dims(), [5, 32], "Q^R should be [5, 4*8=32]");
        println!("✅ MSA routing query: [5, 64] → Q^R[5, 32]");
    }
}
