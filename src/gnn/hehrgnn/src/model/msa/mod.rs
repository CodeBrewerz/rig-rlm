//! # Memory Sparse Attention (MSA)
//!
//! Implementation of "MSA: Memory Sparse Attention for Efficient End-to-End
//! Memory Model Scaling to 100M Tokens" (arXiv:2603.23516v1, NeurIPS 2026).
//!
//! ## Architecture Overview
//!
//! MSA is a scalable sparse attention mechanism for lifelong memory contexts:
//!
//! 1. **Router Projectors** (`router`): Specialized W_KR / W_QR projections
//!    for document-level routing, separate from standard K/Q/V.
//!
//! 2. **Chunk-wise Pooling** (`pooling`): φ(·) compresses KV caches by
//!    averaging over fixed-length chunks (P=64), achieving 64× compression.
//!
//! 3. **Cosine Scoring + Top-k** (`scoring`): Eq. 2 scores query-document
//!    relevance via cosine similarity, aggregated across heads then tokens.
//!
//! 4. **Document-wise RoPE** (`rope`): Independent positional encoding per
//!    document, decoupling position from document count for extrapolation.
//!
//! 5. **MSA Layer** (`sparse_attn`): Core attention layer that routes, selects,
//!    and attends to sparse compressed context.
//!
//! 6. **Memory Bank** (`memory_bank`): Three-stage inference pipeline —
//!    offline encoding, online routing, online generation.
//!
//! 7. **Auxiliary Loss** (`loss`): Supervised contrastive loss (Eq. 5) for
//!    router training with two-phase schedule.
//!
//! 8. **Memory Interleave** (`interleave`): Iterative multi-hop reasoning
//!    via repeated route→retrieve→expand cycles.
//!
//! ## Key Properties
//!
//! - **Linear complexity**: O(L) in both training and inference
//! - **Minimal degradation**: <9% from 16K to 100M tokens
//! - **End-to-end trainable**: Fully differentiable routing via soft selection
//! - **Hardware efficient**: 100M tokens on 2×A800 via tiered storage

pub mod interleave;
pub mod loss;
pub mod memory_bank;
pub mod pooling;
pub mod rope;
pub mod router;
pub mod scoring;
pub mod sparse_attn;

use burn::prelude::*;

use memory_bank::MemoryBank;
use sparse_attn::{MsaLayer, MsaLayerConfig};
use rope::{RoPEConfig, RoPETable};
use interleave::{InterleaveConfig, MemoryInterleave, InterleaveStep};

/// Full MSA model configuration.
#[derive(Debug, Clone)]
pub struct MsaConfig {
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Router projection dimension per head.
    pub router_dim: usize,
    /// Chunk size for KV compression (P=64 in paper).
    pub chunk_size: usize,
    /// Number of total layers.
    pub num_layers: usize,
    /// FFN expansion ratio.
    pub ffn_ratio: usize,
    /// Top-k documents for retrieval.
    pub topk: usize,
    /// RoPE base frequency.
    pub rope_base: f32,
    /// Maximum sequence length for RoPE.
    pub rope_max_len: usize,
    /// Memory interleave configuration.
    pub interleave: InterleaveConfig,
}

impl Default for MsaConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 128,
            num_heads: 8,
            router_dim: 16,
            chunk_size: 64,
            num_layers: 4,
            ffn_ratio: 4,
            topk: 16,
            rope_base: 10000.0,
            rope_max_len: 8192,
            interleave: InterleaveConfig::default(),
        }
    }
}

/// Full MSA Model.
///
/// Paper §3.2.1: "We implement the MSA routing strategy selectively, applying
/// it exclusively to the latter half of the model's layers."
///
/// Architecture:
/// - Lower layers: standard self-attention (no routing)
/// - Upper layers: MSA sparse attention with routing
#[derive(Module, Debug)]
pub struct MsaModel<B: Backend> {
    /// MSA layers (applied to latter half of model)
    msa_layers: Vec<MsaLayer<B>>,

    /// Number of standard layers (no routing, lower half)
    pub num_standard_layers: usize,
    
    // Config values
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub router_dim: usize,
    pub ffn_ratio: usize,
    pub topk: usize,
    pub rope_base: f32,
    pub rope_max_len: usize,
    pub interleave_max_rounds: usize,
    pub interleave_topk_per_round: usize,
    pub interleave_min_score_threshold: f32,
}

impl<B: Backend> MsaModel<B> {
    /// Create a new MSA model.
    pub fn new(config: &MsaConfig, device: &B::Device) -> Self {
        let num_msa_layers = config.num_layers / 2;
        let num_standard_layers = config.num_layers - num_msa_layers;

        let layer_config = MsaLayerConfig {
            hidden_dim: config.hidden_dim,
            num_heads: config.num_heads,
            router_dim: config.router_dim,
            chunk_size: config.chunk_size,
            topk: config.topk,
            ffn_ratio: config.ffn_ratio,
        };

        let msa_layers: Vec<MsaLayer<B>> = (0..num_msa_layers)
            .map(|_| MsaLayer::new(&layer_config, device))
            .collect();

        Self {
            msa_layers,
            num_standard_layers,
            hidden_dim: config.hidden_dim,
            num_heads: config.num_heads,
            router_dim: config.router_dim,
            ffn_ratio: config.ffn_ratio,
            topk: config.topk,
            rope_base: config.rope_base,
            rope_max_len: config.rope_max_len,
            interleave_max_rounds: config.interleave.max_rounds,
            interleave_topk_per_round: config.interleave.topk_per_round,
            interleave_min_score_threshold: config.interleave.min_score_threshold,
        }
    }

    /// Create a RoPE table for this model.
    pub fn create_rope_table(&self, device: &B::Device) -> RoPETable<B> {
        RoPETable::new(
            &RoPEConfig {
                dim: self.hidden_dim,
                base: self.rope_base,
                max_len: self.rope_max_len,
            },
            device,
        )
    }

    /// Create a Memory Bank for this model.
    pub fn create_memory_bank(&self) -> MemoryBank<B> {
        MemoryBank::new(self.num_heads, self.topk)
    }

    /// Create a Memory Interleave mechanism for this model.
    pub fn create_interleave(&self) -> MemoryInterleave {
        MemoryInterleave::new(InterleaveConfig {
            max_rounds: self.interleave_max_rounds,
            topk_per_round: self.interleave_topk_per_round,
            min_score_threshold: self.interleave_min_score_threshold,
        })
    }

    /// Get the MSA layer at a given index.
    pub fn get_msa_layer(&self, idx: usize) -> Option<&MsaLayer<B>> {
        self.msa_layers.get(idx)
    }

    /// Encode a document through all MSA layers.
    ///
    /// Returns per-layer encoded caches.
    pub fn encode_document_all_layers(
        &self,
        doc_hidden: Tensor<B, 2>,
    ) -> Vec<(Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>)> {
        self.msa_layers
            .iter()
            .map(|layer| layer.encode_document(doc_hidden.clone()))
            .collect()
    }

    /// Run full MSA forward through all MSA layers.
    ///
    /// # Arguments
    /// * `query_hidden` - Query hidden states [query_len, hidden_dim]
    /// * `memory_k_per_layer` - Per-layer memory K from selected documents
    /// * `memory_v_per_layer` - Per-layer memory V from selected documents
    ///
    /// # Returns
    /// * Final hidden states [query_len, hidden_dim]
    pub fn forward(
        &self,
        query_hidden: Tensor<B, 2>,
        memory_k_per_layer: &[Tensor<B, 2>],
        memory_v_per_layer: &[Tensor<B, 2>],
    ) -> Tensor<B, 2> {
        let mut h = query_hidden;

        for (i, layer) in self.msa_layers.iter().enumerate() {
            let mem_k = &memory_k_per_layer[i];
            let mem_v = &memory_v_per_layer[i];
            h = layer.forward(h, mem_k.clone(), mem_v.clone());
        }

        h
    }

    /// End-to-end inference: encode → route → retrieve → generate.
    ///
    /// This is the simplified single-shot pipeline.
    ///
    /// # Arguments
    /// * `query_hidden` - Query hidden states [query_len, hidden_dim]
    /// * `bank` - Memory bank with pre-encoded documents (first MSA layer)
    ///
    /// # Returns
    /// * `(output, selected_doc_ids)` — final hidden states and retrieved doc IDs
    pub fn inference(
        &self,
        query_hidden: Tensor<B, 2>,
        bank: &MemoryBank<B>,
    ) -> Option<(Tensor<B, 2>, Vec<usize>)> {
        let layer = &self.msa_layers[0];

        // Stage 2: Route
        let routing_query = layer.compute_routing_query(query_hidden.clone());
        let result = bank.route_and_retrieve(routing_query)?;
        let (mem_k, mem_v, doc_ids) = result;

        // Stage 3: Generate (single layer for simplicity)
        let output = layer.forward(query_hidden, mem_k, mem_v);

        Some((output, doc_ids))
    }

    /// End-to-end inference with Memory Interleave for multi-hop.
    ///
    /// # Arguments
    /// * `query_hidden` - Query hidden states [query_len, hidden_dim]
    /// * `bank` - Memory bank
    ///
    /// # Returns
    /// * `(output, steps)` — final hidden states and interleave retrieval log
    pub fn inference_interleaved(
        &self,
        query_hidden: Tensor<B, 2>,
        bank: &MemoryBank<B>,
    ) -> (Tensor<B, 2>, Vec<InterleaveStep>) {
        let layer = &self.msa_layers[0];
        let interleave = self.create_interleave();

        let (mem_k, mem_v, steps) = interleave.run(query_hidden.clone(), bank, layer);

        let [mem_len, _] = mem_k.dims();
        let output = if mem_len > 0 {
            layer.forward(query_hidden, mem_k, mem_v)
        } else {
            // No documents retrieved — just pass through
            let device = query_hidden.device();
            let [_q_len, h_dim] = query_hidden.dims();
            let empty_k = Tensor::<B, 2>::zeros([0, h_dim], &device);
            let empty_v = Tensor::<B, 2>::zeros([0, h_dim], &device);
            layer.forward(query_hidden, empty_k, empty_v)
        };

        (output, steps)
    }

    /// Get total parameter count.
    pub fn param_count(&self) -> usize {
        let mut count = 0;
        for _layer in &self.msa_layers {
            // Each layer has: Q,K,V,Out projections + router K,Q + FFN + norms
            let h = self.hidden_dim;
            let r = self.router_dim * self.num_heads;
            let ffn = h * self.ffn_ratio;

            count += h * h * 4; // Q, K, V, Out
            count += h * r * 2; // Router K, Router Q
            count += h * ffn + ffn * h; // FFN
            count += h * 2; // RMSNorm weights
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn test_msa_model_construction() {
        let device = <B as Backend>::Device::default();
        let config = MsaConfig {
            hidden_dim: 64,
            num_heads: 4,
            router_dim: 8,
            chunk_size: 16,
            num_layers: 4,
            ffn_ratio: 2,
            topk: 2,
            rope_max_len: 1024,
            ..Default::default()
        };

        let model = MsaModel::<B>::new(&config, &device);

        // 4 total layers → 2 MSA layers (latter half)
        assert_eq!(model.msa_layers.len(), 2);
        assert_eq!(model.num_standard_layers, 2);
        println!(
            "✅ MsaModel: {} MSA layers, {} standard layers, ~{} params",
            model.msa_layers.len(),
            model.num_standard_layers,
            model.param_count()
        );
    }

    #[test]
    fn test_msa_model_end_to_end() {
        let device = <B as Backend>::Device::default();
        let config = MsaConfig {
            hidden_dim: 64,
            num_heads: 4,
            router_dim: 8,
            chunk_size: 16,
            num_layers: 4,
            ffn_ratio: 2,
            topk: 2,
            rope_max_len: 1024,
            ..Default::default()
        };

        let model = MsaModel::<B>::new(&config, &device);
        let mut bank = model.create_memory_bank();
        let layer = model.get_msa_layer(0).unwrap();

        // Encode 5 documents
        for i in 0..5 {
            let doc = Tensor::<B, 2>::random(
                [64, 64],
                burn::tensor::Distribution::Normal(0.0, 0.1),
                &device,
            );
            bank.encode_document(i, doc, layer);
        }

        // Query inference
        let query = Tensor::<B, 2>::random(
            [8, 64],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );

        let result = model.inference(query.clone(), &bank);
        assert!(result.is_some());

        let (output, doc_ids) = result.unwrap();
        assert_eq!(output.dims(), [8, 64]);
        assert_eq!(doc_ids.len(), 2); // topk=2

        println!(
            "✅ MsaModel E2E: query[8,64] → output[{},{}], retrieved docs {:?}",
            output.dims()[0],
            output.dims()[1],
            doc_ids
        );
    }

    #[test]
    fn test_msa_model_interleaved_inference() {
        let device = <B as Backend>::Device::default();
        let config = MsaConfig {
            hidden_dim: 64,
            num_heads: 4,
            router_dim: 8,
            chunk_size: 16,
            num_layers: 4,
            ffn_ratio: 2,
            topk: 2,
            rope_max_len: 1024,
            interleave: InterleaveConfig {
                max_rounds: 3,
                topk_per_round: 2,
                min_score_threshold: -1.0,
            },
            ..Default::default()
        };

        let model = MsaModel::<B>::new(&config, &device);
        let mut bank = model.create_memory_bank();
        let layer = model.get_msa_layer(0).unwrap();

        // Encode 8 documents
        for i in 0..8 {
            let doc = Tensor::<B, 2>::random(
                [64, 64],
                burn::tensor::Distribution::Normal(0.0, 0.1),
                &device,
            );
            bank.encode_document(i, doc, layer);
        }

        let query = Tensor::<B, 2>::random(
            [8, 64],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );

        let (output, steps) = model.inference_interleaved(query, &bank);

        assert_eq!(output.dims()[1], 64);
        assert!(!steps.is_empty());

        println!("✅ MsaModel Interleaved: {} rounds", steps.len());
        for step in &steps {
            println!(
                "   Round {}: docs {:?}, scores {:?}",
                step.step, step.retrieved_doc_ids, step.scores
            );
        }
    }

    #[test]
    fn test_msa_model_multi_layer_forward() {
        let device = <B as Backend>::Device::default();
        let config = MsaConfig {
            hidden_dim: 64,
            num_heads: 4,
            router_dim: 8,
            chunk_size: 16,
            num_layers: 6,
            ffn_ratio: 2,
            topk: 2,
            rope_max_len: 1024,
            ..Default::default()
        };

        let model = MsaModel::<B>::new(&config, &device);
        assert_eq!(model.msa_layers.len(), 3, "6 total → 3 MSA layers");

        // Create per-layer memory KV
        let mem_k: Vec<Tensor<B, 2>> = (0..3)
            .map(|_| {
                Tensor::<B, 2>::random(
                    [16, 64],
                    burn::tensor::Distribution::Normal(0.0, 0.1),
                    &device,
                )
            })
            .collect();
        let mem_v: Vec<Tensor<B, 2>> = (0..3)
            .map(|_| {
                Tensor::<B, 2>::random(
                    [16, 64],
                    burn::tensor::Distribution::Normal(0.0, 0.1),
                    &device,
                )
            })
            .collect();

        let query = Tensor::<B, 2>::random(
            [5, 64],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );

        let output = model.forward(query, &mem_k, &mem_v);
        assert_eq!(output.dims(), [5, 64]);
        println!("✅ MsaModel multi-layer forward: 3 MSA layers → [5, 64]");
    }

    #[test]
    fn test_rope_integration() {
        let device = <B as Backend>::Device::default();
        let config = MsaConfig {
            hidden_dim: 64,
            rope_max_len: 2048,
            ..Default::default()
        };

        let model = MsaModel::<B>::new(&config, &device);
        let rope = model.create_rope_table(&device);

        // Document-wise RoPE
        let doc = Tensor::<B, 2>::ones([100, 64], &device);
        let encoded = rope.apply_docwise(doc);
        assert_eq!(encoded.dims(), [100, 64]);

        // Global RoPE with offset k=16
        let query = Tensor::<B, 2>::ones([20, 64], &device);
        let global = rope.apply_global(query, 16);
        assert_eq!(global.dims(), [20, 64]);

        println!("✅ RoPE integration: docwise + global modes work");
    }
}
