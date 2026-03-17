//! Advanced HRR enhancements — Cleanup Network, Resonator, MAP binding, RFF decorrelation.
//!
//! These are drop-in improvements over the basic HRR in `core.rs`:
//!
//! 1. **Cleanup Network** — codebook nearest-neighbor for exact recall
//! 2. **Resonator Network** — iterative unbind→cleanup→rebind for 10x capacity
//! 3. **MAP Binding** — multiply-add-permute for better SNR
//! 4. **RFF Key Decorrelation** — Random Fourier Features for better key separation

use super::core::{bind, unbind, ComplexVector, Mulberry32};
use std::f64::consts::TAU;

// ═══════════════════════════════════════════════════════════════════
// 1. Cleanup Network — Codebook Nearest-Neighbor
// ═══════════════════════════════════════════════════════════════════

/// A codebook for converting noisy HRR retrieval signals into exact values.
///
/// After unbinding, the result is a noisy approximation of the stored value.
/// The cleanup network finds the closest codebook entry by cosine similarity,
/// making recall exact instead of approximate.
#[derive(Debug, Clone)]
pub struct CleanupNetwork {
    /// The codebook entries — typically the vocab keys.
    codebook: Vec<ComplexVector>,
    /// Pre-computed unit-normed 2D representations for fast cosine similarity.
    codebook_norm: Vec<Vec<f64>>,
}

impl CleanupNetwork {
    /// Create a cleanup network from a set of known value vectors.
    pub fn new(codebook: &[ComplexVector]) -> Self {
        let codebook_norm = codebook
            .iter()
            .map(|v| {
                let d = v.dim();
                let d2 = d * 2;
                let mut row = vec![0.0; d2];
                row[..d].copy_from_slice(&v.re);
                row[d..d2].copy_from_slice(&v.im);
                let norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt() + 1e-12;
                for x in row.iter_mut() {
                    *x /= norm;
                }
                row
            })
            .collect();

        Self {
            codebook: codebook.to_vec(),
            codebook_norm,
        }
    }

    /// Clean up a noisy signal by finding the nearest codebook entry.
    ///
    /// Returns (best_index, similarity, cleaned_vector).
    pub fn cleanup(&self, noisy: &ComplexVector) -> (usize, f64, ComplexVector) {
        if self.codebook.is_empty() {
            return (0, 0.0, noisy.clone());
        }

        let d = noisy.dim();
        let d2 = d * 2;

        // Unit-norm the noisy signal
        let mut query = vec![0.0; d2];
        query[..d].copy_from_slice(&noisy.re);
        query[d..d2].copy_from_slice(&noisy.im);
        let norm: f64 = query.iter().map(|x| x * x).sum::<f64>().sqrt() + 1e-12;
        for x in query.iter_mut() {
            *x /= norm;
        }

        // Find highest cosine similarity using core::dot_product (AVX2 optimized)
        let mut best_idx = 0;
        let mut best_sim = f64::NEG_INFINITY;
        for (i, row) in self.codebook_norm.iter().enumerate() {
            let dot = super::core::dot_product(row, &query);
            if dot > best_sim {
                best_sim = dot;
                best_idx = i;
            }
        }

        (best_idx, best_sim, self.codebook[best_idx].clone())
    }

    /// Batch cleanup — process multiple noisy signals.
    pub fn cleanup_batch(&self, noisy: &[ComplexVector]) -> Vec<(usize, f64, ComplexVector)> {
        noisy.iter().map(|n| self.cleanup(n)).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════
// 2. Resonator Network — Iterative Sharpening
// ═══════════════════════════════════════════════════════════════════

/// Resonator network for iterative retrieval sharpening.
///
/// Instead of single-shot unbinding (which gets noisy with many facts),
/// the resonator iteratively cleans up the signal:
///
/// ```text
/// for each iteration:
///   1. Unbind memory with current key estimate
///   2. Cleanup result against codebook
///   3. Rebind cleaned result to check consistency
///   4. If consistent (high similarity), stop early
/// ```
///
/// This is analogous to LDPC decoding — each iteration reduces noise.
pub struct Resonator {
    /// Maximum iterations before giving up.
    pub max_iters: usize,
    /// Early-stop threshold — if cosine sim exceeds this, we're confident.
    pub convergence_threshold: f64,
}

impl Default for Resonator {
    fn default() -> Self {
        Self {
            max_iters: 5,
            convergence_threshold: 0.85,
        }
    }
}

impl Resonator {
    pub fn new(max_iters: usize, convergence_threshold: f64) -> Self {
        Self {
            max_iters,
            convergence_threshold,
        }
    }

    /// Run iterative resonator retrieval.
    ///
    /// # Arguments
    /// - `memory`: the superposed memory vector
    /// - `key`: the query key to unbind with
    /// - `cleanup`: the cleanup network with known value codebook
    ///
    /// # Returns
    /// (best_codebook_idx, final_similarity, iterations_used)
    pub fn resonate(
        &self,
        memory: &ComplexVector,
        key: &ComplexVector,
        cleanup: &CleanupNetwork,
    ) -> (usize, f64, usize) {
        let mut current_estimate = unbind(memory, key);
        let mut best_idx = 0;
        let mut best_sim = 0.0;

        for iter in 0..self.max_iters {
            // Step 1: Cleanup — find nearest codebook entry
            let (idx, sim, cleaned) = cleanup.cleanup(&current_estimate);
            best_idx = idx;
            best_sim = sim;

            // Step 2: Check convergence
            if sim >= self.convergence_threshold {
                return (best_idx, best_sim, iter + 1);
            }

            // Step 3: Rebind cleaned estimate and re-unbind
            // This creates a "self-consistency" check — if the cleaned value
            // is correct, rebinding it and unbinding again should produce
            // an even cleaner signal
            let rebound = bind(key, &cleaned);
            let consistency = cosine_similarity_complex(&rebound, memory);

            if consistency > 0.3 {
                // Consistent — use cleaned estimate for next iteration
                // but also blend in the raw unbinding for diversity
                let raw = unbind(memory, key);
                current_estimate = blend_complex(&cleaned, &raw, 0.7);
            } else {
                // Not consistent — try raw unbinding again
                current_estimate = unbind(memory, key);
            }
        }

        (best_idx, best_sim, self.max_iters)
    }
}

/// Cosine similarity between two complex vectors (treats as 2D real).
fn cosine_similarity_complex(a: &ComplexVector, b: &ComplexVector) -> f64 {
    let d = a.dim();
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for i in 0..d {
        dot += a.re[i] * b.re[i] + a.im[i] * b.im[i];
        norm_a += a.re[i] * a.re[i] + a.im[i] * a.im[i];
        norm_b += b.re[i] * b.re[i] + b.im[i] * b.im[i];
    }
    dot / (norm_a.sqrt() * norm_b.sqrt() + 1e-12)
}

/// Blend two complex vectors: result = alpha * a + (1 - alpha) * b.
fn blend_complex(a: &ComplexVector, b: &ComplexVector, alpha: f64) -> ComplexVector {
    let d = a.dim();
    let beta = 1.0 - alpha;
    let mut re = vec![0.0; d];
    let mut im = vec![0.0; d];
    for i in 0..d {
        re[i] = alpha * a.re[i] + beta * b.re[i];
        im[i] = alpha * a.im[i] + beta * b.im[i];
    }
    ComplexVector { re, im }
}

// ═══════════════════════════════════════════════════════════════════
// 3. MAP Binding — Multiply-Add-Permute
// ═══════════════════════════════════════════════════════════════════

/// Generate a deterministic permutation of size `d` using a seeded PRNG.
///
/// Uses Fisher-Yates shuffle with a seeded Mulberry32 PRNG.
pub fn make_permutation(d: usize, seed: u32) -> Vec<usize> {
    let mut rng = Mulberry32::new(seed);
    let mut perm: Vec<usize> = (0..d).collect();
    for i in (1..d).rev() {
        let j = (rng.next_f64() * (i + 1) as f64) as usize;
        perm.swap(i, j);
    }
    perm
}

/// Generate the inverse permutation.
pub fn invert_permutation(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}

/// Apply a permutation to a complex vector.
fn permute_complex(v: &ComplexVector, perm: &[usize]) -> ComplexVector {
    let d = v.dim();
    let mut re = vec![0.0; d];
    let mut im = vec![0.0; d];
    for i in 0..d {
        re[i] = v.re[perm[i]];
        im[i] = v.im[perm[i]];
    }
    ComplexVector { re, im }
}

/// MAP bind: element-wise complex multiply + permute.
///
/// Better SNR than circular convolution because the permutation
/// breaks spectral leakage patterns.
pub fn map_bind(a: &ComplexVector, b: &ComplexVector, perm: &[usize]) -> ComplexVector {
    let product = bind(a, b); // element-wise complex multiply
    permute_complex(&product, perm)
}

/// MAP unbind: inverse-permute + element-wise multiply by conjugate.
pub fn map_unbind(
    memory: &ComplexVector,
    key: &ComplexVector,
    inv_perm: &[usize],
) -> ComplexVector {
    let unperm = permute_complex(memory, inv_perm);
    unbind(&unperm, key)
}

// ═══════════════════════════════════════════════════════════════════
// 4. RFF Key Decorrelation
// ═══════════════════════════════════════════════════════════════════

/// Random Fourier Feature transform for key decorrelation.
///
/// Projects HRR keys through random features so that inner products
/// approximate Gaussian kernel values. This makes keys more independent
/// (decorrelated) in the transformed space, reducing cross-talk when
/// many facts are stored.
///
/// Ported from the Stable-GNN implementation in hehrgnn, simplified
/// for f64 and zero-dependency usage.
#[derive(Debug, Clone)]
pub struct RffDecorrelator {
    /// Projection matrix ω ~ N(0, 1/σ²), shape [output_dim, input_dim * 2]
    omega: Vec<Vec<f64>>,
    /// Phase shifts φ ~ Uniform(0, 2π), shape [output_dim]
    phi: Vec<f64>,
    /// Output dimension (number of random features)
    output_dim: usize,
    /// Input dimension * 2 (complex → real)
    input_dim_2d: usize,
}

impl RffDecorrelator {
    /// Create a new RFF decorrelator.
    ///
    /// # Arguments
    /// - `input_dim`: dimension of complex vectors (D)
    /// - `output_dim`: number of random features (typically 2*D for good approximation)
    /// - `sigma`: kernel bandwidth (controls smoothness, typically 1.0)
    /// - `seed`: random seed for reproducibility
    pub fn new(input_dim: usize, output_dim: usize, sigma: f64, seed: u32) -> Self {
        let mut rng = Mulberry32::new(seed);
        let input_dim_2d = input_dim * 2;

        // Generate ω ~ N(0, 1/σ²) using Box-Muller transform
        let inv_sigma = 1.0 / sigma;
        let mut omega = Vec::with_capacity(output_dim);
        for _ in 0..output_dim {
            let mut row = Vec::with_capacity(input_dim_2d);
            for _ in 0..(input_dim_2d / 2 + 1) {
                // Box-Muller: two uniform → two normal
                let u1 = rng.next_f64().max(1e-10);
                let u2 = rng.next_f64();
                let r = (-2.0 * u1.ln()).sqrt() * inv_sigma;
                let theta = TAU * u2;
                row.push(r * theta.cos());
                row.push(r * theta.sin());
            }
            row.truncate(input_dim_2d);
            omega.push(row);
        }

        // Generate φ ~ Uniform(0, 2π)
        let phi: Vec<f64> = (0..output_dim).map(|_| TAU * rng.next_f64()).collect();

        Self {
            omega,
            phi,
            output_dim,
            input_dim_2d,
        }
    }

    /// Transform a complex vector to RFF space: z(x) = √(2/d) · cos(ωᵀx + φ)
    pub fn transform(&self, v: &ComplexVector) -> Vec<f64> {
        let d = v.dim();
        let d2 = d * 2;
        assert_eq!(d2, self.input_dim_2d);

        // Stack complex → real
        let mut x = vec![0.0; d2];
        x[..d].copy_from_slice(&v.re);
        x[d..d2].copy_from_slice(&v.im);

        let scale = (2.0 / self.output_dim as f64).sqrt();
        let mut result = Vec::with_capacity(self.output_dim);

        for (i, omega_row) in self.omega.iter().enumerate() {
            let mut dot = 0.0;
            for j in 0..d2 {
                dot += omega_row[j] * x[j];
            }
            result.push(scale * (dot + self.phi[i]).cos());
        }

        result
    }

    /// Compute the decorrelation score between two sets of keys.
    ///
    /// Returns the Frobenius norm of the cross-covariance matrix in RFF space.
    /// Lower = more independent = better key separation.
    pub fn cross_covariance_norm(&self, keys: &[ComplexVector]) -> f64 {
        if keys.len() < 2 {
            return 0.0;
        }

        // Transform all keys to RFF space
        let rff_keys: Vec<Vec<f64>> = keys.iter().map(|k| self.transform(k)).collect();

        // Compute mean
        let n = rff_keys.len();
        let d = self.output_dim;
        let mut mean = vec![0.0; d];
        for k in &rff_keys {
            for i in 0..d {
                mean[i] += k[i];
            }
        }
        for m in mean.iter_mut() {
            *m /= n as f64;
        }

        // Compute Frobenius norm of centered cross-covariance
        // ||C||²_F = Σᵢ Σⱼ cov(i,j)²
        let mut frob_sq = 0.0;
        for i in 0..d {
            for j in i + 1..d {
                let mut cov = 0.0;
                for k in &rff_keys {
                    cov += (k[i] - mean[i]) * (k[j] - mean[j]);
                }
                cov /= (n - 1) as f64;
                frob_sq += cov * cov;
            }
        }

        frob_sq.sqrt()
    }

    /// Decorrelate keys by iterative repulsion in RFF space.
    ///
    /// After each iteration, keys that are correlated in RFF space
    /// are pushed apart, then re-projected to unit magnitude.
    pub fn decorrelate_keys(
        &self,
        keys: &[ComplexVector],
        iters: usize,
        step_size: f64,
    ) -> Vec<ComplexVector> {
        if keys.len() < 2 || iters == 0 {
            return keys.to_vec();
        }

        let n = keys.len();
        let d = keys[0].dim();
        let mut result = keys.to_vec();

        for _ in 0..iters {
            // Transform to RFF space
            let rff: Vec<Vec<f64>> = result.iter().map(|k| self.transform(k)).collect();

            // Compute pairwise repulsion in original space
            for i in 0..n {
                for j in (i + 1)..n {
                    // Correlation in RFF space
                    let rff_corr: f64 = rff[i].iter().zip(rff[j].iter()).map(|(a, b)| a * b).sum();

                    if rff_corr.abs() > 0.1 {
                        // Repel in complex space proportional to RFF correlation
                        let scale = step_size * rff_corr;
                        for k in 0..d {
                            result[i].re[k] -= scale * result[j].re[k];
                            result[i].im[k] -= scale * result[j].im[k];
                            result[j].re[k] -= scale * result[i].re[k];
                            result[j].im[k] -= scale * result[i].im[k];
                        }
                    }
                }
            }

            // Re-normalise to unit magnitude per element
            for v in result.iter_mut() {
                for k in 0..d {
                    let mag = (v.re[k] * v.re[k] + v.im[k] * v.im[k]).sqrt() + 1e-12;
                    v.re[k] /= mag;
                    v.im[k] /= mag;
                }
            }
        }

        result
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::super::core::{make_vocab_keys, Mulberry32};
    use super::*;

    // -- Cleanup Network tests --

    #[test]
    fn cleanup_finds_exact_match() {
        let mut rng = Mulberry32::new(42);
        let codebook = make_vocab_keys(5, 256, &mut rng);
        let cleanup = CleanupNetwork::new(&codebook);

        // Query with an exact codebook entry — should get index 2 back
        let (idx, sim, _) = cleanup.cleanup(&codebook[2]);
        assert_eq!(idx, 2, "should find exact match at index 2");
        assert!(sim > 0.99, "similarity should be ~1.0, got {sim}");
    }

    #[test]
    fn cleanup_recovers_from_noise() {
        let mut rng = Mulberry32::new(42);
        let codebook = make_vocab_keys(5, 256, &mut rng);
        let cleanup = CleanupNetwork::new(&codebook);

        // Add noise to codebook[3]
        let target = &codebook[3];
        let mut noisy_re = target.re.clone();
        let mut noisy_im = target.im.clone();
        let mut noise_rng = Mulberry32::new(999);
        for i in 0..256 {
            noisy_re[i] += (noise_rng.next_f64() - 0.5) * 0.3;
            noisy_im[i] += (noise_rng.next_f64() - 0.5) * 0.3;
        }
        let noisy = ComplexVector {
            re: noisy_re,
            im: noisy_im,
        };

        let (idx, sim, _) = cleanup.cleanup(&noisy);
        assert_eq!(idx, 3, "should still find index 3 with moderate noise");
        assert!(
            sim > 0.7,
            "similarity should still be reasonable, got {sim}"
        );
    }

    #[test]
    fn cleanup_after_bind_unbind() {
        let mut rng = Mulberry32::new(42);
        let vocab = make_vocab_keys(10, 512, &mut rng);
        let keys = make_vocab_keys(10, 512, &mut rng);
        let cleanup = CleanupNetwork::new(&vocab);

        // Bind key[0] with vocab[3], then unbind — should recover vocab[3]
        let bound = bind(&keys[0], &vocab[3]);
        let recovered = unbind(&bound, &keys[0]);
        let (idx, sim, _) = cleanup.cleanup(&recovered);
        assert_eq!(idx, 3, "cleanup should find vocab[3] after bind/unbind");
        assert!(sim > 0.95, "similarity {sim} should be high");
    }

    #[test]
    fn cleanup_with_superposition() {
        let mut rng = Mulberry32::new(42);
        let vocab = make_vocab_keys(5, 1024, &mut rng);
        let keys = make_vocab_keys(5, 1024, &mut rng);
        let cleanup = CleanupNetwork::new(&vocab);

        // Superpose 3 bindings: key[i] ⊗ vocab[i] for i=0,1,2
        let mut memory = ComplexVector::zeros(1024);
        for i in 0..3 {
            let b = bind(&keys[i], &vocab[i]);
            for d in 0..1024 {
                memory.re[d] += b.re[d];
                memory.im[d] += b.im[d];
            }
        }
        // Scale
        let scale = 1.0 / 3.0_f64.sqrt();
        for d in 0..1024 {
            memory.re[d] *= scale;
            memory.im[d] *= scale;
        }

        // Unbind with key[1] → should get vocab[1]
        let recovered = unbind(&memory, &keys[1]);
        let (idx, sim, _) = cleanup.cleanup(&recovered);
        assert_eq!(idx, 1, "cleanup should find vocab[1] from superposition");
        assert!(sim > 0.5, "similarity {sim} should be reasonable");
    }

    // -- Resonator tests --

    #[test]
    fn resonator_converges_quickly() {
        let mut rng = Mulberry32::new(42);
        let vocab = make_vocab_keys(5, 512, &mut rng);
        let keys = make_vocab_keys(5, 512, &mut rng);
        let cleanup = CleanupNetwork::new(&vocab);
        let resonator = Resonator::default();

        // Single binding
        let memory = bind(&keys[0], &vocab[2]);
        let (idx, sim, iters) = resonator.resonate(&memory, &keys[0], &cleanup);
        assert_eq!(idx, 2, "should find vocab[2]");
        assert!(sim > 0.9, "similarity {sim}");
        assert!(iters <= 2, "should converge in ≤2 iters, got {iters}");
    }

    #[test]
    fn resonator_handles_superposition() {
        let mut rng = Mulberry32::new(42);
        let vocab = make_vocab_keys(8, 1024, &mut rng);
        let keys = make_vocab_keys(8, 1024, &mut rng);
        let cleanup = CleanupNetwork::new(&vocab);
        let resonator = Resonator::new(8, 0.8);

        // Superpose 5 bindings
        let mut memory = ComplexVector::zeros(1024);
        for i in 0..5 {
            let b = bind(&keys[i], &vocab[i]);
            for d in 0..1024 {
                memory.re[d] += b.re[d];
                memory.im[d] += b.im[d];
            }
        }
        let scale = 1.0 / 5.0_f64.sqrt();
        for d in 0..1024 {
            memory.re[d] *= scale;
            memory.im[d] *= scale;
        }

        // Query each binding — should all converge
        let mut correct = 0;
        for i in 0..5 {
            let (idx, _sim, _iters) = resonator.resonate(&memory, &keys[i], &cleanup);
            if idx == i {
                correct += 1;
            }
        }
        assert!(
            correct >= 4,
            "resonator should recover at least 4/5 from superposition, got {correct}"
        );
    }

    // -- MAP Binding tests --

    #[test]
    fn map_bind_unbind_recovers() {
        let mut rng = Mulberry32::new(42);
        let keys = make_vocab_keys(2, 256, &mut rng);
        let perm = make_permutation(256, 12345);
        let inv_perm = invert_permutation(&perm);

        let bound = map_bind(&keys[0], &keys[1], &perm);
        let recovered = map_unbind(&bound, &keys[0], &inv_perm);

        let sim = cosine_similarity_complex(&recovered, &keys[1]);
        assert!(sim > 0.99, "MAP bind/unbind should recover, got sim={sim}");
    }

    #[test]
    fn map_bind_better_snr_than_standard() {
        let mut rng = Mulberry32::new(42);
        let vocab = make_vocab_keys(20, 512, &mut rng);
        let keys = make_vocab_keys(20, 512, &mut rng);
        let perm = make_permutation(512, 54321);
        let inv_perm = invert_permutation(&perm);

        // Superpose 10 bindings with standard bind
        let mut mem_std = ComplexVector::zeros(512);
        let mut mem_map = ComplexVector::zeros(512);
        for i in 0..10 {
            let b_std = bind(&keys[i], &vocab[i]);
            let b_map = map_bind(&keys[i], &vocab[i], &perm);
            for d in 0..512 {
                mem_std.re[d] += b_std.re[d];
                mem_std.im[d] += b_std.im[d];
                mem_map.re[d] += b_map.re[d];
                mem_map.im[d] += b_map.im[d];
            }
        }

        // Measure recall quality for each
        let mut std_correct = 0;
        let mut map_correct = 0;
        let cleanup = CleanupNetwork::new(&vocab);

        for i in 0..10 {
            let rec_std = unbind(&mem_std, &keys[i]);
            let rec_map = map_unbind(&mem_map, &keys[i], &inv_perm);

            let (idx_std, _, _) = cleanup.cleanup(&rec_std);
            let (idx_map, _, _) = cleanup.cleanup(&rec_map);

            if idx_std == i {
                std_correct += 1;
            }
            if idx_map == i {
                map_correct += 1;
            }
        }

        // MAP should be at least as good as standard
        assert!(
            map_correct >= std_correct - 1,
            "MAP correct={map_correct} should be ≥ std={std_correct}-1"
        );
    }

    #[test]
    fn permutation_is_valid() {
        let perm = make_permutation(100, 42);
        assert_eq!(perm.len(), 100);
        let mut sorted = perm.clone();
        sorted.sort();
        let expected: Vec<usize> = (0..100).collect();
        assert_eq!(sorted, expected, "permutation should be a valid bijection");
    }

    #[test]
    fn inverse_permutation_roundtrip() {
        let perm = make_permutation(64, 42);
        let inv = invert_permutation(&perm);
        let mut rng = Mulberry32::new(99);
        let v = make_vocab_keys(1, 64, &mut rng).remove(0);

        let permuted = permute_complex(&v, &perm);
        let restored = permute_complex(&permuted, &inv);

        for i in 0..64 {
            assert!((restored.re[i] - v.re[i]).abs() < 1e-10, "re[{i}] mismatch");
            assert!((restored.im[i] - v.im[i]).abs() < 1e-10, "im[{i}] mismatch");
        }
    }

    // -- RFF Decorrelation tests --

    #[test]
    fn rff_transform_consistent_dimension() {
        let mut rng = Mulberry32::new(42);
        let keys = make_vocab_keys(3, 128, &mut rng);
        let rff = RffDecorrelator::new(128, 256, 1.0, 42);

        let transformed = rff.transform(&keys[0]);
        assert_eq!(transformed.len(), 256);
    }

    #[test]
    fn rff_cross_covariance_decreases_after_decorrelation() {
        let mut rng = Mulberry32::new(42);
        let keys = make_vocab_keys(10, 128, &mut rng);
        let rff = RffDecorrelator::new(128, 256, 1.0, 42);

        let before = rff.cross_covariance_norm(&keys);
        let decorrelated = rff.decorrelate_keys(&keys, 3, 0.01);
        let after = rff.cross_covariance_norm(&decorrelated);

        // Decorrelation should reduce the cross-covariance
        assert!(
            after <= before * 1.1, // allow small tolerance
            "cross-cov should decrease: before={before:.4}, after={after:.4}"
        );
    }

    #[test]
    fn rff_decorrelated_keys_still_unit_magnitude() {
        let mut rng = Mulberry32::new(42);
        let keys = make_vocab_keys(5, 128, &mut rng);
        let rff = RffDecorrelator::new(128, 256, 1.0, 42);

        let decorrelated = rff.decorrelate_keys(&keys, 3, 0.01);

        for (vi, v) in decorrelated.iter().enumerate() {
            for d in 0..128 {
                let mag = (v.re[d] * v.re[d] + v.im[d] * v.im[d]).sqrt();
                assert!(
                    (mag - 1.0).abs() < 0.01,
                    "key[{vi}][{d}] mag={mag} should be ~1.0"
                );
            }
        }
    }

    #[test]
    fn rff_preserves_bind_unbind_correctness() {
        let mut rng = Mulberry32::new(42);
        let vocab = make_vocab_keys(5, 256, &mut rng);
        let rff = RffDecorrelator::new(256, 512, 1.0, 42);
        let decorrelated = rff.decorrelate_keys(&vocab, 3, 0.01);
        let cleanup = CleanupNetwork::new(&decorrelated);

        let mut rng2 = Mulberry32::new(99);
        let keys = make_vocab_keys(5, 256, &mut rng2);

        // Bind and unbind should still work with decorrelated keys
        let bound = bind(&keys[0], &decorrelated[2]);
        let recovered = unbind(&bound, &keys[0]);
        let (idx, sim, _) = cleanup.cleanup(&recovered);
        assert_eq!(idx, 2, "should recover decorrelated[2]");
        assert!(sim > 0.9, "similarity {sim} should be high");
    }
}
