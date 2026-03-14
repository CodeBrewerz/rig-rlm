//! Lightweight Document Vision Transformer
//!
//! Adapted from jepa-rs `jepa-vision/vit.rs`.
//!
//! A small ViT encoder (4 layers, 4 heads, 32-dim) for processing
//! document images (receipts, bank statements) and producing embeddings
//! compatible with our GNN hidden dimension.
//!
//! Pipeline:
//!   Image → PatchEmbed → DroPE PE → TransformerBlocks → LayerNorm → Embeddings
//!
//! Uses DroPE (paper 2512.12167) instead of RoPE:
//!   Train phase: sinusoidal 2D PE helps convergence
//!   Inference: PE dropped — works on any document size

use super::document_drope::{DroPEController, DroPEPhase};
use super::document_patch::{DocumentPatchConfig, DocumentPatchEmbedding};

/// Configuration for the Document ViT.
#[derive(Debug, Clone)]
pub struct DocumentVitConfig {
    /// Patch embedding config.
    pub patch_config: DocumentPatchConfig,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// MLP expansion ratio (hidden = embed_dim * mlp_ratio).
    pub mlp_ratio: usize,
    /// Dropout rate.
    pub dropout: f32,
    /// Total training steps for DroPE scheduling.
    pub drope_train_steps: usize,
    /// DroPE recalibration fraction (typically 0.02-0.05).
    pub drope_recal_fraction: f32,
}

impl Default for DocumentVitConfig {
    fn default() -> Self {
        Self {
            patch_config: DocumentPatchConfig::default(),
            num_layers: 4,
            num_heads: 4,
            mlp_ratio: 4,
            dropout: 0.0,
            drope_train_steps: 40,
            drope_recal_fraction: 0.05,
        }
    }
}

impl DocumentVitConfig {
    /// Tiny config for testing.
    pub fn tiny_test() -> Self {
        Self {
            patch_config: DocumentPatchConfig {
                in_channels: 1,
                patch_h: 4,
                patch_w: 4,
                embed_dim: 8,
            },
            num_layers: 2,
            num_heads: 2,
            mlp_ratio: 2,
            dropout: 0.0,
            drope_train_steps: 10,
            drope_recal_fraction: 0.1,
        }
    }
}

/// A single self-attention + MLP transformer block.
///
/// Uses simplified multi-head self-attention with pre-norm.
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    /// Query/Key/Value projection: [embed_dim, 3 * embed_dim].
    pub qkv_weights: Vec<Vec<f32>>,
    /// Output projection: [embed_dim, embed_dim].
    pub out_weights: Vec<Vec<f32>>,
    /// MLP layer 1: [embed_dim, hidden_dim].
    pub mlp1_weights: Vec<Vec<f32>>,
    /// MLP layer 1 bias.
    pub mlp1_bias: Vec<f32>,
    /// MLP layer 2: [hidden_dim, embed_dim].
    pub mlp2_weights: Vec<Vec<f32>>,
    /// MLP layer 2 bias.
    pub mlp2_bias: Vec<f32>,
    /// LayerNorm 1 gamma/beta.
    pub ln1_gamma: Vec<f32>,
    pub ln1_beta: Vec<f32>,
    /// LayerNorm 2 gamma/beta.
    pub ln2_gamma: Vec<f32>,
    pub ln2_beta: Vec<f32>,
    /// Config.
    pub embed_dim: usize,
    pub num_heads: usize,
}

impl TransformerBlock {
    /// Initialize with random weights (Xavier init).
    pub fn new(embed_dim: usize, num_heads: usize, mlp_ratio: usize) -> Self {
        let hidden_dim = embed_dim * mlp_ratio;
        let mut seed = 137u64;

        let init = |rows: usize, cols: usize, seed: &mut u64| -> Vec<Vec<f32>> {
            let scale = (2.0 / (rows + cols) as f64).sqrt() as f32;
            (0..rows)
                .map(|_| {
                    (0..cols)
                        .map(|_| {
                            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                            let u = (*seed >> 33) as f32 / (1u64 << 31) as f32;
                            (u - 0.5) * 2.0 * scale
                        })
                        .collect()
                })
                .collect()
        };

        Self {
            qkv_weights: init(embed_dim, 3 * embed_dim, &mut seed),
            out_weights: init(embed_dim, embed_dim, &mut seed),
            mlp1_weights: init(embed_dim, hidden_dim, &mut seed),
            mlp1_bias: vec![0.0; hidden_dim],
            mlp2_weights: init(hidden_dim, embed_dim, &mut seed),
            mlp2_bias: vec![0.0; embed_dim],
            ln1_gamma: vec![1.0; embed_dim],
            ln1_beta: vec![0.0; embed_dim],
            ln2_gamma: vec![1.0; embed_dim],
            ln2_beta: vec![0.0; embed_dim],
            embed_dim,
            num_heads,
        }
    }

    /// Forward pass through one transformer block.
    ///
    /// Pre-norm: x = x + Attn(LN(x)); x = x + MLP(LN(x))
    pub fn forward(&self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let seq_len = input.len();
        let d = self.embed_dim;

        // LayerNorm 1
        let normed1 = self.layer_norm(input, &self.ln1_gamma, &self.ln1_beta);

        // Self-attention
        let attn_out = self.self_attention(&normed1);

        // Residual 1
        let mut residual1: Vec<Vec<f32>> = input
            .iter()
            .zip(attn_out.iter())
            .map(|(x, a)| x.iter().zip(a.iter()).map(|(&xv, &av)| xv + av).collect())
            .collect();

        // LayerNorm 2
        let normed2 = self.layer_norm(&residual1, &self.ln2_gamma, &self.ln2_beta);

        // MLP
        let mlp_out = self.mlp(&normed2);

        // Residual 2
        residual1
            .iter()
            .zip(mlp_out.iter())
            .map(|(r, m)| r.iter().zip(m.iter()).map(|(&rv, &mv)| rv + mv).collect())
            .collect()
    }

    fn layer_norm(&self, x: &[Vec<f32>], gamma: &[f32], beta: &[f32]) -> Vec<Vec<f32>> {
        x.iter()
            .map(|v| {
                let mean = v.iter().sum::<f32>() / v.len() as f32;
                let var = v.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
                let std = (var + 1e-6).sqrt();
                v.iter()
                    .enumerate()
                    .map(|(i, &val)| {
                        let g = if i < gamma.len() { gamma[i] } else { 1.0 };
                        let b = if i < beta.len() { beta[i] } else { 0.0 };
                        (val - mean) / std * g + b
                    })
                    .collect()
            })
            .collect()
    }

    fn self_attention(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let seq_len = x.len();
        let d = self.embed_dim;
        let head_dim = d / self.num_heads;

        // Compute Q, K, V
        let mut qkv: Vec<Vec<f32>> = x
            .iter()
            .map(|token| {
                let mut out = vec![0.0f32; 3 * d];
                for (i, &val) in token.iter().enumerate() {
                    if i < self.qkv_weights.len() {
                        for (j, &w) in self.qkv_weights[i].iter().enumerate() {
                            out[j] += val * w;
                        }
                    }
                }
                out
            })
            .collect();

        // Split into Q, K, V and compute attention per head
        let mut output = vec![vec![0.0f32; d]; seq_len];

        for h in 0..self.num_heads {
            let h_start = h * head_dim;
            let h_end = h_start + head_dim;

            // Compute attention scores
            let scale = (head_dim as f32).sqrt();
            for i in 0..seq_len {
                let mut weights = Vec::with_capacity(seq_len);
                for j in 0..seq_len {
                    let mut dot = 0.0f32;
                    for k in h_start..h_end {
                        let q = qkv[i][k]; // Q
                        let kk = qkv[j][d + k]; // K
                        dot += q * kk;
                    }
                    weights.push(dot / scale);
                }

                // Softmax
                let max_w = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_w: Vec<f32> = weights.iter().map(|&w| (w - max_w).exp()).collect();
                let sum_exp: f32 = exp_w.iter().sum();
                let attn: Vec<f32> = exp_w.iter().map(|&e| e / (sum_exp + 1e-8)).collect();

                // Weighted sum of V
                for j in 0..seq_len {
                    for k in h_start..h_end {
                        output[i][k] += attn[j] * qkv[j][2 * d + k]; // V
                    }
                }
            }
        }

        // Output projection
        output
            .iter()
            .map(|token| {
                let mut out = vec![0.0f32; d];
                for (i, &val) in token.iter().enumerate() {
                    if i < self.out_weights.len() {
                        for (j, &w) in self.out_weights[i].iter().enumerate() {
                            out[j] += val * w;
                        }
                    }
                }
                out
            })
            .collect()
    }

    fn mlp(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        x.iter()
            .map(|token| {
                // Layer 1: embed_dim → hidden_dim + GELU
                let mut hidden = self.mlp1_bias.clone();
                for (i, &val) in token.iter().enumerate() {
                    if i < self.mlp1_weights.len() {
                        for (j, &w) in self.mlp1_weights[i].iter().enumerate() {
                            hidden[j] += val * w;
                        }
                    }
                }
                // GELU activation
                for v in hidden.iter_mut() {
                    *v = *v * 0.5 * (1.0 + (*v * 0.7978846 * (1.0 + 0.044715 * *v * *v)).tanh());
                }
                // Layer 2: hidden_dim → embed_dim
                let mut out = self.mlp2_bias.clone();
                for (i, &val) in hidden.iter().enumerate() {
                    if i < self.mlp2_weights.len() {
                        for (j, &w) in self.mlp2_weights[i].iter().enumerate() {
                            out[j] += val * w;
                        }
                    }
                }
                out
            })
            .collect()
    }
}

/// Lightweight Document Vision Transformer.
///
/// Processes document images into patch-level embeddings,
/// using DroPE for position-invariant generalization.
#[derive(Debug, Clone)]
pub struct DocumentVit {
    /// Patch embedding.
    pub patch_embed: DocumentPatchEmbedding,
    /// Transformer blocks.
    pub blocks: Vec<TransformerBlock>,
    /// Final layer norm.
    pub final_ln_gamma: Vec<f32>,
    pub final_ln_beta: Vec<f32>,
    /// DroPE controller.
    pub drope: DroPEController,
    /// Config.
    pub config: DocumentVitConfig,
}

impl DocumentVit {
    /// Initialize a new Document ViT.
    pub fn new(config: DocumentVitConfig) -> Self {
        let embed_dim = config.patch_config.embed_dim;
        let patch_embed = DocumentPatchEmbedding::new(config.patch_config.clone());

        let blocks: Vec<TransformerBlock> = (0..config.num_layers)
            .map(|_| TransformerBlock::new(embed_dim, config.num_heads, config.mlp_ratio))
            .collect();

        // Default grid for DroPE (will work with various sizes after recalibration)
        let drope = DroPEController::new(
            16,
            16, // Default 16×16 patch grid (256×256 image with 16×16 patches)
            embed_dim,
            config.drope_train_steps,
            config.drope_recal_fraction,
        );

        Self {
            patch_embed,
            blocks,
            final_ln_gamma: vec![1.0; embed_dim],
            final_ln_beta: vec![0.0; embed_dim],
            drope,
            config,
        }
    }

    /// Encode a document image into patch-level embeddings.
    ///
    /// Input: image pixels [C, H, W], height, width.
    /// Output: Vec of embed_dim vectors (one per patch).
    pub fn encode(&mut self, image: &[f32], height: usize, width: usize) -> Vec<Vec<f32>> {
        // Step 1: Patchify + project
        let mut tokens = self.patch_embed.forward(image, height, width);

        // Step 2: DroPE positional encoding (only during training)
        tokens = self.drope.apply(&tokens);

        // Step 3: Transformer blocks
        for block in &self.blocks {
            tokens = block.forward(&tokens);
        }

        // Step 4: Final layer norm
        let d = self.config.patch_config.embed_dim;
        tokens
            .iter()
            .map(|v| {
                let mean = v.iter().sum::<f32>() / v.len() as f32;
                let var = v.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
                let std = (var + 1e-6).sqrt();
                v.iter()
                    .enumerate()
                    .map(|(i, &val)| {
                        let g = if i < d { self.final_ln_gamma[i] } else { 1.0 };
                        let b = if i < d { self.final_ln_beta[i] } else { 0.0 };
                        (val - mean) / std * g + b
                    })
                    .collect()
            })
            .collect()
    }

    /// Encode and pool to a single document embedding.
    ///
    /// Mean-pools all patch embeddings to produce one vector per document.
    pub fn encode_pooled(&mut self, image: &[f32], height: usize, width: usize) -> Vec<f32> {
        let patch_embs = self.encode(image, height, width);
        if patch_embs.is_empty() {
            return vec![0.0; self.config.patch_config.embed_dim];
        }

        let d = patch_embs[0].len();
        let n = patch_embs.len() as f32;
        let mut pooled = vec![0.0f32; d];
        for emb in &patch_embs {
            for (j, &val) in emb.iter().enumerate() {
                if j < d {
                    pooled[j] += val;
                }
            }
        }
        for v in pooled.iter_mut() {
            *v /= n;
        }
        pooled
    }

    /// Advance DroPE by one training step.
    pub fn training_step(&mut self) {
        self.drope.step();
    }

    /// Get current DroPE phase.
    pub fn drope_phase(&self) -> DroPEPhase {
        self.drope.phase
    }

    /// Force inference mode (skip recalibration).
    pub fn set_inference(&mut self) {
        self.drope.force_inference();
    }

    /// Get embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.config.patch_config.embed_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_vit_default() {
        let config = DocumentVitConfig::default();
        let vit = DocumentVit::new(config);
        assert_eq!(vit.blocks.len(), 4);
        assert_eq!(vit.embed_dim(), 32);
    }

    #[test]
    fn test_document_vit_tiny_encode() {
        let config = DocumentVitConfig::tiny_test();
        let mut vit = DocumentVit::new(config);
        // 8×8 image with 4×4 patches = 2×2 grid = 4 patches
        let image = vec![0.5f32; 1 * 8 * 8];
        let embeddings = vit.encode(&image, 8, 8);
        assert_eq!(embeddings.len(), 4, "2×2 grid = 4 patches");
        assert_eq!(embeddings[0].len(), 8, "embed_dim = 8");
    }

    #[test]
    fn test_document_vit_pooled() {
        let config = DocumentVitConfig::tiny_test();
        let mut vit = DocumentVit::new(config);
        let image = vec![0.5f32; 1 * 8 * 8];
        let pooled = vit.encode_pooled(&image, 8, 8);
        assert_eq!(pooled.len(), 8, "Pooled to single embed_dim vector");
        assert!(pooled.iter().all(|v| v.is_finite()), "All values finite");
    }

    #[test]
    fn test_drope_phases_in_vit() {
        let config = DocumentVitConfig::tiny_test();
        let mut vit = DocumentVit::new(config);
        assert_eq!(vit.drope_phase(), DroPEPhase::TrainWithPE);

        // Train for 10 steps
        for _ in 0..10 {
            vit.training_step();
        }
        assert_eq!(vit.drope_phase(), DroPEPhase::Recalibrate);

        // Recalibrate (1 step = 10 * 0.1)
        vit.training_step();
        assert_eq!(vit.drope_phase(), DroPEPhase::Inference);
    }

    #[test]
    fn test_different_image_sizes() {
        let config = DocumentVitConfig::tiny_test();
        let mut vit = DocumentVit::new(config);
        vit.set_inference(); // No PE = works on any size

        // 8×8 image
        let img1 = vec![0.5f32; 1 * 8 * 8];
        let emb1 = vit.encode(&img1, 8, 8);
        assert_eq!(emb1.len(), 4); // 2×2

        // 16×8 image (different aspect ratio)
        let img2 = vec![0.5f32; 1 * 16 * 8];
        let emb2 = vit.encode(&img2, 16, 8);
        assert_eq!(emb2.len(), 8); // 4×2

        // Both produce same embed_dim
        assert_eq!(emb1[0].len(), emb2[0].len());
    }

    #[test]
    fn test_pooled_same_dim_any_size() {
        let config = DocumentVitConfig::tiny_test();
        let mut vit = DocumentVit::new(config);
        vit.set_inference();

        let p1 = vit.encode_pooled(&vec![0.5f32; 64], 8, 8);
        let p2 = vit.encode_pooled(&vec![0.5f32; 128], 16, 8);
        let p3 = vit.encode_pooled(&vec![0.5f32; 256], 16, 16);

        assert_eq!(p1.len(), p2.len());
        assert_eq!(p2.len(), p3.len());
    }

    #[test]
    fn test_transformer_block() {
        let block = TransformerBlock::new(8, 2, 4);
        let input = vec![vec![1.0f32; 8]; 4]; // 4 tokens, 8-dim
        let output = block.forward(&input);
        assert_eq!(output.len(), 4);
        assert_eq!(output[0].len(), 8);
        assert!(output.iter().flat_map(|v| v.iter()).all(|v| v.is_finite()));
    }
}
