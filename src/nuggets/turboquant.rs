//! TurboQuant (MSE + Prod) for HRR memory backends.
//! Based on TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
//!
//! Two quantizers:
//!   - `quantize_mse_4bit` / `dequantize_mse_4bit` — for memory bank compression (§1)
//!   - `QjlCleanupNetwork` — binary sign-sketch for fast inner-product cleanup (§2)

use super::core::ComplexVector;
use super::core::Mulberry32;

// ═══════════════════════════════════════════════════════════════════
// §1  TurboQuant-MSE — 4-bit scalar quantizer for memory banks
// ═══════════════════════════════════════════════════════════════════

const CENTROIDS_4BIT: [f64; 16] = [
    -2.733, -2.069, -1.618, -1.256, -0.942, -0.657, -0.388, -0.128,
     0.128,  0.388,  0.657,  0.942,  1.256,  1.618,  2.069,  2.733,
];

const BOUNDARIES_4BIT: [f64; 15] = [
    -2.401, -1.8435, -1.437, -1.099, -0.7995, -0.5225, -0.258, 0.0,
     0.258,  0.5225,  0.7995,  1.099,  1.437,  1.8435,  2.401,
];

#[inline(always)]
fn quantize_f64(val: f64) -> u8 {
    for (i, &b) in BOUNDARIES_4BIT.iter().enumerate() {
        if val <= b { return i as u8; }
    }
    15
}

/// 4-bit quantization of a ComplexVector. Packing 2 coords into 1 byte.
pub fn quantize_mse_4bit(v: &ComplexVector) -> Vec<u8> {
    let d = v.dim();
    let mut packed = vec![0u8; d];
    let norm_factor = std::f64::consts::SQRT_2;

    for i in 0..(d / 2) {
        let r1 = quantize_f64(v.re[i*2] * norm_factor);
        let r2 = quantize_f64(v.re[i*2 + 1] * norm_factor);
        packed[i] = (r1 << 4) | r2;

        let i1 = quantize_f64(v.im[i*2] * norm_factor);
        let i2 = quantize_f64(v.im[i*2 + 1] * norm_factor);
        packed[d/2 + i] = (i1 << 4) | i2;
    }
    packed
}

pub fn dequantize_mse_4bit(packed: &[u8]) -> ComplexVector {
    let d = packed.len();
    let denorm_factor = std::f64::consts::FRAC_1_SQRT_2;
    let mut re = vec![0.0; d];
    let mut im = vec![0.0; d];

    for i in 0..(d / 2) {
        let b_re = packed[i];
        re[i*2] = CENTROIDS_4BIT[(b_re >> 4) as usize] * denorm_factor;
        re[i*2 + 1] = CENTROIDS_4BIT[(b_re & 0x0F) as usize] * denorm_factor;

        let b_im = packed[d/2 + i];
        im[i*2] = CENTROIDS_4BIT[(b_im >> 4) as usize] * denorm_factor;
        im[i*2 + 1] = CENTROIDS_4BIT[(b_im & 0x0F) as usize] * denorm_factor;
    }
    ComplexVector { re, im }
}

// ═══════════════════════════════════════════════════════════════════
// §2  Binary Sign-Sketch Cleanup Network (SimHash-style)
// ═══════════════════════════════════════════════════════════════════
//
// Instead of f64 dot products over 2D real vectors, we:
//   1. For each codebook vector, take random projections (R · v) and store sign bits
//   2. For a query, compute the same random projections and store sign bits
//   3. Similarity ≈ cos(π · hamming_distance / num_projections)
//
// The key optimization: the random projection matrix R is stored as
// PACKED SIGN BITS. Each row of R is a {+1,-1} vector packed into u64s.
// The dot product sign(R_row · v) becomes: POPCNT(R_row_bits XOR sign_bits(v))
// which tells us how many components disagree in sign.
//
// But even this requires knowing sign_bits(v) first. The REAL trick for speed:
// we store the projection matrix as f64 for the one-time codebook precomputation
// (which happens at build_hrr time), but at query time we use a 2-stage approach:
//
//   Stage 1 (FAST): Use raw coordinate sign-bits directly (zero projection cost)
//            This is O(1) to compute and gives a coarse ranking via Hamming.
//   Stage 2 (EXACT): Re-rank the top-K candidates using exact f64 dot product.
//
// This is the classic "coarse filter + re-rank" pattern used in FAISS/ScaNN.

/// A cleanup network using sign-bit sketches for fast coarse filtering,
/// with exact f64 re-ranking of top candidates.
#[derive(Debug, Clone)]
pub struct QjlCleanupNetwork {
    /// Pre-computed unit-normed 2D representations (for exact re-ranking).
    codebook_norm: Vec<Vec<f64>>,
    /// Sign-bit sketches for each codebook entry.
    sketches: Vec<Vec<u64>>,
    /// Number of u64 words per sketch.
    num_words: usize,
    /// 2D dimension.
    dim_2d: usize,
    /// Number of top candidates to re-rank with exact f64.
    top_k: usize,
}

impl QjlCleanupNetwork {
    pub fn new(codebook: &[ComplexVector], top_k: usize) -> Self {
        if codebook.is_empty() {
            return Self {
                codebook_norm: vec![],
                sketches: vec![],
                num_words: 0,
                dim_2d: 0,
                top_k,
            };
        }

        let d = codebook[0].dim();
        let dim_2d = d * 2;
        let num_words = (dim_2d + 63) / 64;

        let mut codebook_norm_vecs = Vec::with_capacity(codebook.len());
        let mut sketches = Vec::with_capacity(codebook.len());

        for v in codebook {
            let mut flat = vec![0.0; dim_2d];
            flat[..d].copy_from_slice(&v.re);
            flat[d..dim_2d].copy_from_slice(&v.im);
            let norm: f64 = flat.iter().map(|x| x * x).sum::<f64>().sqrt() + 1e-12;
            for x in flat.iter_mut() {
                *x /= norm;
            }
            codebook_norm_vecs.push(flat.clone());
            sketches.push(Self::sign_sketch(&flat, num_words));
        }

        Self {
            codebook_norm: codebook_norm_vecs,
            sketches,
            num_words,
            dim_2d,
            top_k,
        }
    }

    #[inline]
    fn sign_sketch(v: &[f64], num_words: usize) -> Vec<u64> {
        let mut sketch = vec![0u64; num_words];
        for (j, &val) in v.iter().enumerate() {
            if val >= 0.0 {
                sketch[j / 64] |= 1u64 << (j % 64);
            }
        }
        sketch
    }

    #[inline]
    fn hamming_distance(a: &[u64], b: &[u64]) -> u32 {
        let mut dist = 0u32;
        for i in 0..a.len() {
            dist += (a[i] ^ b[i]).count_ones();
        }
        dist
    }

    #[inline]
    fn dot_product(a: &[f64], b: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        sum
    }

    /// Number of codebook entries.
    pub fn vocab_size(&self) -> usize {
        self.codebook_norm.len()
    }

    /// Clean up a noisy signal using 2-stage: Hamming coarse → f64 re-rank.
    /// Returns (best_index, similarity, empty_vector).
    pub fn cleanup(&self, noisy: &ComplexVector) -> (usize, f64, ComplexVector) {
        if self.codebook_norm.is_empty() {
            return (0, 0.0, noisy.clone());
        }

        let d = noisy.dim();
        let mut query = vec![0.0; self.dim_2d];
        query[..d].copy_from_slice(&noisy.re);
        query[d..self.dim_2d].copy_from_slice(&noisy.im);
        let norm: f64 = query.iter().map(|x| x * x).sum::<f64>().sqrt() + 1e-12;
        let inv_norm = 1.0 / norm;
        for x in query.iter_mut() {
            *x *= inv_norm;
        }

        let query_sketch = Self::sign_sketch(&query, self.num_words);
        let v = self.sketches.len();
        let actual_top_k = self.top_k.min(v);

        if actual_top_k >= v {
            let mut best_idx = 0;
            let mut best_sim = f64::NEG_INFINITY;
            for (i, row) in self.codebook_norm.iter().enumerate() {
                let sim = Self::dot_product(row, &query);
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = i;
                }
            }
            return (best_idx, best_sim, ComplexVector::zeros(d));
        }

        let mut candidates: Vec<(u32, usize)> = self.sketches.iter()
            .enumerate()
            .map(|(i, sketch)| (Self::hamming_distance(sketch, &query_sketch), i))
            .collect();

        candidates.select_nth_unstable(actual_top_k - 1);
        let candidates = &candidates[..actual_top_k];

        let mut best_idx = candidates[0].1;
        let mut best_sim = f64::NEG_INFINITY;
        for &(_, idx) in candidates {
            let sim = Self::dot_product(&self.codebook_norm[idx], &query);
            if sim > best_sim {
                best_sim = sim;
                best_idx = idx;
            }
        }

        (best_idx, best_sim, ComplexVector::zeros(d))
    }
}
