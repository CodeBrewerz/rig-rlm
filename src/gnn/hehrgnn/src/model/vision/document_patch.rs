//! Document Patch Embedding
//!
//! Adapted from jepa-rs `jepa-vision/patch.rs`.
//!
//! Splits document images (receipts, bank statements) into non-overlapping
//! patches and projects each through a linear layer to produce patch embeddings.
//!
//! ```text
//! [B, C, H, W] → reshape → [B, grid_h·grid_w, C·patch_h·patch_w] → linear → [B, S, D]
//! ```
//!
//! For documents:
//! - Each patch captures a table cell, line item, logo, or handwritten note
//! - Default: grayscale (1 channel) or RGB (3 channels)
//! - Patch size 16×16 optimized for document resolution

use std::collections::HashMap;

/// Configuration for document patch embedding.
#[derive(Debug, Clone)]
pub struct DocumentPatchConfig {
    /// Number of input channels (1=grayscale, 3=RGB).
    pub in_channels: usize,
    /// Patch height in pixels.
    pub patch_h: usize,
    /// Patch width in pixels.
    pub patch_w: usize,
    /// Output embedding dimension.
    pub embed_dim: usize,
}

impl Default for DocumentPatchConfig {
    fn default() -> Self {
        Self {
            in_channels: 1, // Grayscale for scanned documents
            patch_h: 16,
            patch_w: 16,
            embed_dim: 32, // Matches our GNN hidden_dim
        }
    }
}

impl DocumentPatchConfig {
    /// RGB config for color receipts/photos.
    pub fn rgb(embed_dim: usize) -> Self {
        Self {
            in_channels: 3,
            patch_h: 16,
            patch_w: 16,
            embed_dim,
        }
    }

    /// Small patches for higher resolution on dense documents.
    pub fn high_res(in_channels: usize, embed_dim: usize) -> Self {
        Self {
            in_channels,
            patch_h: 8,
            patch_w: 8,
            embed_dim,
        }
    }

    /// Compute the number of patches for a given image size.
    pub fn num_patches(&self, height: usize, width: usize) -> usize {
        (height / self.patch_h) * (width / self.patch_w)
    }

    /// Compute grid dimensions.
    pub fn grid_size(&self, height: usize, width: usize) -> (usize, usize) {
        (height / self.patch_h, width / self.patch_w)
    }

    /// Dimension of each flattened patch.
    pub fn patch_dim(&self) -> usize {
        self.in_channels * self.patch_h * self.patch_w
    }
}

/// Document patch embedding module.
///
/// Splits an image into patches and projects through a weight matrix.
/// Uses our SPSA-compatible weight format (Vec<Vec<f32>>) for integration
/// with the existing hehrgnn training loop.
#[derive(Debug, Clone)]
pub struct DocumentPatchEmbedding {
    /// Projection weights: [patch_dim, embed_dim].
    pub weights: Vec<Vec<f32>>,
    /// Bias: [embed_dim].
    pub bias: Vec<f32>,
    /// Config.
    pub config: DocumentPatchConfig,
}

impl DocumentPatchEmbedding {
    /// Initialize with random weights (Xavier init).
    pub fn new(config: DocumentPatchConfig) -> Self {
        let patch_dim = config.patch_dim();
        let embed_dim = config.embed_dim;
        let scale = (2.0 / (patch_dim + embed_dim) as f64).sqrt() as f32;
        let mut seed = 42u64;

        let weights: Vec<Vec<f32>> = (0..patch_dim)
            .map(|_| {
                (0..embed_dim)
                    .map(|_| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let u = (seed >> 33) as f32 / (1u64 << 31) as f32;
                        (u - 0.5) * 2.0 * scale
                    })
                    .collect()
            })
            .collect();

        let bias = vec![0.0f32; embed_dim];

        Self {
            weights,
            bias,
            config,
        }
    }

    /// Patchify an image into non-overlapping patches.
    ///
    /// Input: image pixels as flat Vec<f32>, shape [C, H, W] in row-major order.
    /// Output: Vec of patch vectors, each of length patch_dim.
    pub fn patchify(&self, image: &[f32], height: usize, width: usize) -> Vec<Vec<f32>> {
        let c = self.config.in_channels;
        let ph = self.config.patch_h;
        let pw = self.config.patch_w;
        let grid_h = height / ph;
        let grid_w = width / pw;
        let mut patches = Vec::with_capacity(grid_h * grid_w);

        for gh in 0..grid_h {
            for gw in 0..grid_w {
                let mut patch = Vec::with_capacity(c * ph * pw);
                for ch in 0..c {
                    for py in 0..ph {
                        for px in 0..pw {
                            let y = gh * ph + py;
                            let x = gw * pw + px;
                            let idx = ch * height * width + y * width + x;
                            patch.push(if idx < image.len() { image[idx] } else { 0.0 });
                        }
                    }
                }
                patches.push(patch);
            }
        }

        patches
    }

    /// Project patches through the linear layer.
    ///
    /// Input: Vec of patch vectors (from patchify).
    /// Output: Vec of embed_dim vectors.
    pub fn embed(&self, patches: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let embed_dim = self.config.embed_dim;
        patches
            .iter()
            .map(|patch| {
                let mut out = self.bias.clone();
                for (i, &val) in patch.iter().enumerate() {
                    if i < self.weights.len() {
                        for (j, &w) in self.weights[i].iter().enumerate() {
                            if j < embed_dim {
                                out[j] += val * w;
                            }
                        }
                    }
                }
                out
            })
            .collect()
    }

    /// Full forward pass: image → patch embeddings.
    ///
    /// Input: image pixels [C, H, W], height, width.
    /// Output: Vec of embed_dim vectors (one per patch).
    pub fn forward(&self, image: &[f32], height: usize, width: usize) -> Vec<Vec<f32>> {
        let patches = self.patchify(image, height, width);
        self.embed(&patches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DocumentPatchConfig::default();
        assert_eq!(config.in_channels, 1);
        assert_eq!(config.patch_h, 16);
        assert_eq!(config.embed_dim, 32);
        assert_eq!(config.patch_dim(), 256); // 1 * 16 * 16
    }

    #[test]
    fn test_num_patches() {
        let config = DocumentPatchConfig::default();
        assert_eq!(config.num_patches(64, 64), 16); // 4×4 grid
        assert_eq!(config.num_patches(128, 64), 32); // 8×4 grid
        assert_eq!(config.num_patches(256, 256), 256); // 16×16 grid
    }

    #[test]
    fn test_patchify_shape() {
        let config = DocumentPatchConfig::default();
        let pe = DocumentPatchEmbedding::new(config);
        // 64×64 grayscale image
        let image = vec![0.5f32; 1 * 64 * 64];
        let patches = pe.patchify(&image, 64, 64);
        assert_eq!(patches.len(), 16, "4×4 grid = 16 patches");
        assert_eq!(patches[0].len(), 256, "Each patch = 1×16×16 = 256");
    }

    #[test]
    fn test_forward_shape() {
        let config = DocumentPatchConfig::default();
        let pe = DocumentPatchEmbedding::new(config);
        let image = vec![0.5f32; 1 * 64 * 64];
        let embeddings = pe.forward(&image, 64, 64);
        assert_eq!(embeddings.len(), 16);
        assert_eq!(embeddings[0].len(), 32);
    }

    #[test]
    fn test_rgb_config() {
        let config = DocumentPatchConfig::rgb(64);
        assert_eq!(config.in_channels, 3);
        assert_eq!(config.patch_dim(), 768); // 3 * 16 * 16
    }

    #[test]
    fn test_high_res_config() {
        let config = DocumentPatchConfig::high_res(1, 32);
        assert_eq!(config.patch_h, 8);
        assert_eq!(config.num_patches(64, 64), 64); // 8×8 grid
    }

    #[test]
    fn test_embed_nonzero() {
        let config = DocumentPatchConfig::default();
        let pe = DocumentPatchEmbedding::new(config);
        let image = vec![1.0f32; 1 * 64 * 64]; // All ones
        let embeddings = pe.forward(&image, 64, 64);
        // Output should not be all zeros (random weights)
        let sum: f32 = embeddings.iter().flat_map(|v| v.iter()).sum::<f32>();
        assert!(sum.abs() > 0.01, "Embeddings should be non-trivial");
    }
}
