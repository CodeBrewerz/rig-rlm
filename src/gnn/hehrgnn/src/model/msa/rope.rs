//! Rotary Position Embedding (RoPE) for MSA.
//!
//! MSA Paper §3.2.2: "MSA employs independent RoPE for each document."
//!
//! Two modes:
//! 1. **Document-wise RoPE**: Each document gets position IDs starting from 0.
//!    This decouples positional semantics from the total number of documents,
//!    enabling train-on-short, infer-on-long extrapolation.
//!
//! 2. **Global RoPE**: Applied to the query/generation context. Position IDs
//!    are offset by k (number of retrieved documents) so the model perceives
//!    the active context as a continuation of the retrieved background info.

use burn::prelude::*;
use std::f32::consts::PI;

/// RoPE configuration.
#[derive(Debug, Clone)]
pub struct RoPEConfig {
    /// Dimension of the embedding (must be even).
    pub dim: usize,
    /// Base frequency (typically 10000.0, or 1_000_000 for extended context).
    pub base: f32,
    /// Maximum sequence length for precomputation.
    pub max_len: usize,
}

impl Default for RoPEConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            base: 10000.0,
            max_len: 8192,
        }
    }
}

/// Precomputed RoPE frequency table.
///
/// Stores cos and sin values for all positions up to `max_len`.
pub struct RoPETable<B: Backend> {
    /// Cosine values: [max_len, dim/2]
    cos_table: Tensor<B, 2>,
    /// Sine values: [max_len, dim/2]
    sin_table: Tensor<B, 2>,
    /// Configuration
    config: RoPEConfig,
}

impl<B: Backend> RoPETable<B> {
    /// Create a new RoPE table with precomputed frequencies.
    pub fn new(config: &RoPEConfig, device: &B::Device) -> Self {
        let half_dim = config.dim / 2;
        let mut cos_data = vec![0.0f32; config.max_len * half_dim];
        let mut sin_data = vec![0.0f32; config.max_len * half_dim];

        for pos in 0..config.max_len {
            for i in 0..half_dim {
                let freq = 1.0 / config.base.powf(2.0 * i as f32 / config.dim as f32);
                let angle = pos as f32 * freq;
                cos_data[pos * half_dim + i] = angle.cos();
                sin_data[pos * half_dim + i] = angle.sin();
            }
        }

        let cos_table = Tensor::<B, 1>::from_data(cos_data.as_slice(), device)
            .reshape([config.max_len, half_dim]);
        let sin_table = Tensor::<B, 1>::from_data(sin_data.as_slice(), device)
            .reshape([config.max_len, half_dim]);

        Self {
            cos_table,
            sin_table,
            config: config.clone(),
        }
    }

    /// Apply RoPE to a tensor with given position offset.
    ///
    /// # Arguments
    /// * `x` - Input tensor [seq_len, dim]
    /// * `offset` - Position offset (0 for doc-wise, k for global query)
    ///
    /// # Returns
    /// * RoPE-encoded tensor [seq_len, dim]
    pub fn apply(&self, x: Tensor<B, 2>, offset: usize) -> Tensor<B, 2> {
        let [seq_len, dim] = x.dims();
        let half_dim = dim / 2;

        assert!(
            offset + seq_len <= self.config.max_len,
            "Position {} + seq_len {} exceeds max_len {}",
            offset,
            seq_len,
            self.config.max_len
        );
        assert_eq!(dim, self.config.dim, "Dimension mismatch");

        // Slice cos/sin for the position range [offset, offset+seq_len)
        let cos = self
            .cos_table
            .clone()
            .slice([offset..offset + seq_len, 0..half_dim]); // [seq_len, half_dim]
        let sin = self
            .sin_table
            .clone()
            .slice([offset..offset + seq_len, 0..half_dim]); // [seq_len, half_dim]

        // Split x into two halves: x1 = x[..., :half], x2 = x[..., half:]
        let x1 = x.clone().slice([0..seq_len, 0..half_dim]);
        let x2 = x.clone().slice([0..seq_len, half_dim..dim]);

        // Apply rotation:
        //   out1 = x1 * cos - x2 * sin
        //   out2 = x2 * cos + x1 * sin
        let out1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
        let out2 = x2 * cos + x1 * sin;

        Tensor::cat(vec![out1, out2], 1) // [seq_len, dim]
    }

    /// Apply Document-wise RoPE (position starts from 0).
    ///
    /// Paper §3.2.2: "By assigning independent position IDs (starting from 0)
    /// to each document, MSA decouples the positional semantics from the total
    /// number of documents in memory."
    pub fn apply_docwise(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.apply(x, 0)
    }

    /// Apply Global RoPE with offset k (number of retrieved documents).
    ///
    /// Paper §3.2.2: "The position indices for the query initiate from k
    /// (corresponding to the Top-k retrieved compressed KVs)."
    pub fn apply_global(&self, x: Tensor<B, 2>, topk: usize) -> Tensor<B, 2> {
        self.apply(x, topk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn test_rope_basic() {
        let device = <B as Backend>::Device::default();
        let config = RoPEConfig {
            dim: 8,
            base: 10000.0,
            max_len: 1024,
        };
        let rope = RoPETable::<B>::new(&config, &device);

        let x = Tensor::<B, 2>::ones([4, 8], &device);
        let encoded = rope.apply_docwise(x.clone());
        assert_eq!(encoded.dims(), [4, 8]);

        // Position 0 should preserve the original (cos=1, sin=0 at pos=0)
        let first_row: Vec<f32> = encoded
            .clone()
            .slice([0..1, 0..8])
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        // At pos=0, angle=0 for all dims, so cos=1, sin=0
        // out1 = 1*1 - 1*0 = 1, out2 = 1*1 + 1*0 = 1
        for v in &first_row {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "Position 0 should preserve values, got {}",
                v
            );
        }

        println!("✅ RoPE basic: shape preserved, pos-0 identity verified");
    }

    #[test]
    fn test_rope_global_offset() {
        let device = <B as Backend>::Device::default();
        let config = RoPEConfig {
            dim: 8,
            base: 10000.0,
            max_len: 1024,
        };
        let rope = RoPETable::<B>::new(&config, &device);

        let x = Tensor::<B, 2>::ones([4, 8], &device);

        // Doc-wise (offset=0) vs Global (offset=16)
        let docwise = rope.apply_docwise(x.clone());
        let global = rope.apply_global(x, 16);

        // They should differ because different positions
        let d: Vec<f32> = docwise.into_data().as_slice::<f32>().unwrap().to_vec();
        let g: Vec<f32> = global.into_data().as_slice::<f32>().unwrap().to_vec();

        let mut differs = false;
        for i in 0..d.len() {
            if (d[i] - g[i]).abs() > 1e-5 {
                differs = true;
                break;
            }
        }
        assert!(differs, "Document-wise and global RoPE should produce different encodings");
        println!("✅ RoPE: doc-wise ≠ global (offset=16) verified");
    }

    #[test]
    fn test_rope_different_positions() {
        let device = <B as Backend>::Device::default();
        let config = RoPEConfig {
            dim: 8,
            base: 10000.0,
            max_len: 1024,
        };
        let rope = RoPETable::<B>::new(&config, &device);

        let x = Tensor::<B, 2>::ones([2, 8], &device);
        let encoded = rope.apply_docwise(x);

        let row0: Vec<f32> = encoded
            .clone()
            .slice([0..1, 0..8])
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let row1: Vec<f32> = encoded
            .slice([1..2, 0..8])
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();

        let mut differs = false;
        for i in 0..row0.len() {
            if (row0[i] - row1[i]).abs() > 1e-5 {
                differs = true;
                break;
            }
        }
        assert!(differs, "Different positions should have different encodings");
        println!("✅ RoPE: different positions produce different encodings");
    }
}
