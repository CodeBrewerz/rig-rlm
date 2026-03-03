//! HeteroDoRA: Heterogeneous Graph-Aware Low-Rank Adapters.
//!
//! Combines three ideas:
//! 1. **Basis decomposition**: K shared basis adapters blended per type
//!    (scales to 50-80+ entity/relation types without explosion)
//! 2. **DoRA**: weight magnitude/direction decomposition for better convergence
//! 3. **Structure gate**: adapter output gated by neighbor aggregation
//!
//! Usage:
//!   Freeze base GNN weights → attach adapters → train only adapter params.
//!   Adapter starts at zero contribution (B init to zeros),
//!   so base model behavior is preserved until training.

use burn::module::Param;
use burn::nn;
use burn::prelude::*;
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════
// Core LoRA Adapter
// ═══════════════════════════════════════════════════════════════

/// Configuration for LoRA adapters.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Low-rank dimension (typically 2-8).
    pub rank: usize,
    /// Scaling factor: output = alpha/rank * B(A(x)).
    pub alpha: f32,
    /// Number of shared basis adapters.
    pub num_bases: usize,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 4,
            alpha: 1.0,
            num_bases: 8,
        }
    }
}

/// A single low-rank adapter: output = scaling * B(A(x)).
///
/// A is initialized with small random values (Kaiming uniform).
/// B is initialized to zeros → adapter starts at zero contribution.
#[derive(Module, Debug)]
pub struct LoraAdapter<B: Backend> {
    /// Down projection: d_in → rank.
    pub lora_a: nn::Linear<B>,
    /// Up projection: rank → d_out.
    pub lora_b: nn::Linear<B>,
    /// Scaling = alpha / rank.
    #[module(skip)]
    pub scaling: f32,
    /// Output dimension.
    #[module(skip)]
    pub d_out: usize,
}

/// Initialize a single LoRA adapter.
pub fn init_lora_adapter<B: Backend>(
    d_in: usize,
    d_out: usize,
    rank: usize,
    alpha: f32,
    device: &B::Device,
) -> LoraAdapter<B> {
    let lora_a = nn::LinearConfig::new(d_in, rank)
        .with_bias(false)
        .init(device);
    let mut lora_b = nn::LinearConfig::new(rank, d_out)
        .with_bias(false)
        .init(device);

    // Zero-init B so adapter starts with zero contribution
    let b_dims = lora_b.weight.val().dims();
    let zero_w = Tensor::<B, 2>::zeros([b_dims[0], b_dims[1]], device);
    lora_b.weight = lora_b.weight.map(|_| zero_w);

    LoraAdapter {
        lora_a,
        lora_b,
        scaling: alpha / rank as f32,
        d_out,
    }
}

impl<B: Backend> LoraAdapter<B> {
    /// Forward: scaling * B(A(x)).
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.lora_a.forward(x);
        let out = self.lora_b.forward(h);
        out * self.scaling
    }
}

// ═══════════════════════════════════════════════════════════════
// Heterogeneous Basis Adapter
// ═══════════════════════════════════════════════════════════════

/// Shared basis adapters + per-type blend coefficients + DoRA magnitude.
///
/// Instead of one adapter per type (which would be 50-80+ adapters),
/// we share K basis adapters and learn a blend vector per type.
///
/// DoRA formula: y = m * (V + ΔV) / ‖V + ΔV‖
///   where V = base weight output, ΔV = adapter output, m = magnitude
#[derive(Module, Debug)]
pub struct HeteroBasisAdapter<B: Backend> {
    /// K shared basis adapters.
    pub bases: Vec<LoraAdapter<B>>,
    /// Per-type blend coefficients: [num_types × K].
    pub blend_weights: Param<Tensor<B, 2>>,
    /// DoRA per-type magnitude vectors: [num_types × d_out].
    /// Initialized to 1.0 (preserves original scale at init).
    pub magnitudes: Param<Tensor<B, 2>>,
    /// Type name → index mapping.
    #[module(skip)]
    pub type_keys: Vec<String>,
    /// Number of bases K.
    #[module(skip)]
    pub num_bases: usize,
    /// Output dimension.
    #[module(skip)]
    pub d_out: usize,
}

/// Initialize a heterogeneous basis adapter.
pub fn init_hetero_basis_adapter<B: Backend>(
    d_in: usize,
    d_out: usize,
    config: &LoraConfig,
    type_keys: Vec<String>,
    device: &B::Device,
) -> HeteroBasisAdapter<B> {
    let num_types = type_keys.len();
    let mut bases = Vec::new();
    for _ in 0..config.num_bases {
        bases.push(init_lora_adapter(
            d_in,
            d_out,
            config.rank,
            config.alpha,
            device,
        ));
    }

    // Initialize blend weights to uniform (all bases contribute equally)
    let blend = Tensor::<B, 2>::ones([num_types, config.num_bases], device);
    let blend_weights = Param::from_tensor(blend);

    // DoRA magnitude: init to 1.0 so adapter preserves base model scale at init
    let mag = Tensor::<B, 2>::ones([num_types, d_out], device);
    let magnitudes = Param::from_tensor(mag);

    HeteroBasisAdapter {
        bases,
        blend_weights,
        magnitudes,
        type_keys,
        num_bases: config.num_bases,
        d_out: d_out,
    }
}

impl<B: Backend> HeteroBasisAdapter<B> {
    /// Get the type index for a given type key.
    pub fn type_idx(&self, key: &str) -> Option<usize> {
        self.type_keys.iter().position(|k| k == key)
    }

    /// Forward pass for a specific type.
    /// Blends all basis adapters using learned weights for this type.
    pub fn forward_for_type(&self, x: Tensor<B, 2>, type_key: &str) -> Tensor<B, 2> {
        let type_idx = match self.type_idx(type_key) {
            Some(idx) => idx,
            None => return Tensor::zeros_like(&x), // unknown type → zero adapter
        };

        // Get blend coefficients for this type and softmax normalize
        let blend_row = self
            .blend_weights
            .val()
            .slice([type_idx..type_idx + 1, 0..self.num_bases])
            .reshape([self.num_bases]);
        let blend_normalized = burn::tensor::activation::softmax(blend_row, 0);

        // Weighted sum of basis adapter outputs
        let device = x.device();
        let dims = x.dims();
        let d_out = self.d_out;
        let mut result = Tensor::<B, 2>::zeros([dims[0], d_out], &device);

        for k in 0..self.num_bases {
            let basis_out = self.bases[k].forward(x.clone());
            let weight = blend_normalized.clone().slice([k..k + 1]).reshape([1, 1]);
            result = result + basis_out * weight.expand([dims[0], d_out]);
        }

        result
    }

    /// DoRA forward: applies magnitude/direction decomposition.
    ///
    /// Given base output V and adapter output ΔV:
    ///   y = m * (V + ΔV) / ‖V + ΔV‖
    ///
    /// This decouples scale (magnitude m) from direction (V + ΔV).
    /// Magnitude absorbs quantization/approximation noise,
    /// letting the LoRA adapter focus on semantic direction updates.
    pub fn dora_forward(
        &self,
        base_output: Tensor<B, 2>,
        adapter_input: Tensor<B, 2>,
        type_key: &str,
    ) -> Tensor<B, 2> {
        let type_idx = match self.type_idx(type_key) {
            Some(idx) => idx,
            None => return base_output, // unknown type → passthrough
        };

        // Get adapter contribution
        let adapter_out = self.forward_for_type(adapter_input, type_key);

        // V + ΔV
        let combined = base_output + adapter_out;

        // Normalize along feature dimension (column norm per the QDoRA paper)
        let eps_val = 1e-6;
        let batch_size = combined.dims()[0];
        let norm = combined
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .sqrt()
            .reshape([batch_size, 1]);
        let norm_safe = norm.clamp_min(eps_val);
        let normalized = combined / norm_safe.expand([batch_size, self.d_out]);

        // Apply per-type magnitude: m[type_idx] ∈ R^(d_out)
        let mag = self
            .magnitudes
            .val()
            .slice([type_idx..type_idx + 1, 0..self.d_out])
            .reshape([1, self.d_out]);
        let batch_size = normalized.dims()[0];
        let mag_expanded = mag.expand([batch_size, self.d_out]);

        normalized * mag_expanded
    }

    /// Count trainable parameters (bases + blend + magnitudes).
    pub fn param_count(&self) -> usize {
        let basis_params: usize = self
            .bases
            .iter()
            .map(|b| {
                let a_dims = b.lora_a.weight.val().dims();
                let b_dims = b.lora_b.weight.val().dims();
                a_dims[0] * a_dims[1] + b_dims[0] * b_dims[1]
            })
            .sum();
        let blend_params = self.type_keys.len() * self.num_bases;
        let magnitude_params = self.type_keys.len() * self.d_out;
        basis_params + blend_params + magnitude_params
    }
}

// ═══════════════════════════════════════════════════════════════
// Adapter Evaluation Metrics
// ═══════════════════════════════════════════════════════════════

/// Report comparing full-weight training vs adapter-only training.
#[derive(Debug, Clone)]
pub struct AdapterEvalReport {
    /// Link prediction AUC from full weight training.
    pub auc_full: f32,
    /// Link prediction AUC from adapter-only training.
    pub auc_adapter: f32,
    /// Ratio: adapter_auc / full_auc.
    pub auc_ratio: f32,
    /// KL divergence: KL(full || adapter) on score distributions.
    pub kl_divergence: f32,
    /// Average cosine similarity between embeddings.
    pub avg_cosine_sim: f32,
    /// Trainable params for full weight approach.
    pub params_full: usize,
    /// Trainable params for adapter approach.
    pub params_adapter: usize,
    /// Parameter savings: 1.0 - adapter/full.
    pub param_savings: f32,
    /// Speedup: time_full / time_adapter.
    pub speedup: f32,
}

/// Compute KL divergence: KL(P || Q) = Σ p_i * log(p_i / q_i).
///
/// Input: raw score vectors. They get softmax-normalized internally.
pub fn kl_divergence(scores_p: &[f32], scores_q: &[f32]) -> f32 {
    assert_eq!(scores_p.len(), scores_q.len());
    if scores_p.is_empty() {
        return 0.0;
    }

    // Softmax normalize both
    let p = softmax_vec(scores_p);
    let q = softmax_vec(scores_q);

    let eps = 1e-10;
    p.iter()
        .zip(q.iter())
        .map(|(pi, qi)| {
            if *pi < eps {
                return 0.0;
            }
            pi * ((pi + eps) / (qi + eps)).ln()
        })
        .sum()
}

/// Compute cosine similarity between two embedding vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Average cosine similarity between two sets of embeddings.
pub fn avg_cosine_similarity(
    emb_a: &HashMap<String, Vec<Vec<f32>>>,
    emb_b: &HashMap<String, Vec<Vec<f32>>>,
) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0;
    for (nt, vecs_a) in emb_a {
        if let Some(vecs_b) = emb_b.get(nt) {
            for (va, vb) in vecs_a.iter().zip(vecs_b.iter()) {
                total += cosine_similarity(va, vb);
                count += 1;
            }
        }
    }
    if count == 0 {
        return 0.0;
    }
    total / count as f32
}

/// Compute link prediction scores for KL divergence comparison.
/// Returns dot-product scores for all (src, dst) pairs.
pub fn compute_link_scores(
    embeddings: &HashMap<String, Vec<Vec<f32>>>,
    edges: &[(String, usize, String, usize)],
) -> Vec<f32> {
    edges
        .iter()
        .map(|(src_type, src_idx, dst_type, dst_idx)| {
            let src_emb = embeddings
                .get(src_type)
                .and_then(|v| v.get(*src_idx))
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            let dst_emb = embeddings
                .get(dst_type)
                .and_then(|v| v.get(*dst_idx))
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            if src_emb.is_empty() || dst_emb.is_empty() {
                return 0.0;
            }
            src_emb.iter().zip(dst_emb.iter()).map(|(a, b)| a * b).sum()
        })
        .collect()
}

fn softmax_vec(v: &[f32]) -> Vec<f32> {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}
