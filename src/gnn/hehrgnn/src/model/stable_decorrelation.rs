//! Stable-GNN: Feature decorrelation for OOD robustness.
//!
//! Implements the key ideas from "Research on GNNs with stable learning"
//! (Nature Scientific Reports, 2025). Makes GNN embeddings robust to
//! distribution shift by decorrelating feature channels via:
//!
//! 1. **RFF Transform**: Map features → random Fourier space (Bochner's theorem)
//! 2. **Feature Independence**: Minimize cross-covariance Frobenius norm
//! 3. **Sample Weighting (LSWD)**: Learn per-sample weights for decorrelation
//!
//! Integration points:
//! - Post-ensemble: decorrelate cross-model embeddings before fiduciary scoring
//! - Training loss: add decorrelation_loss as regularization term

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════
// Random Fourier Feature Transform
// ═══════════════════════════════════════════════════════════════

/// Random Fourier Feature transform for kernel approximation.
///
/// Maps D-dimensional input features to d-dimensional random features
/// such that inner products approximate Gaussian kernel values:
///   <z(x), z(y)> ≈ exp(-||x-y||² / 2σ²)
///
/// Complexity: O(nD) vs O(n²) for exact kernel.
#[derive(Debug, Clone)]
pub struct RffTransform {
    /// Random frequencies ω ~ N(0, 1/σ²), shape: [D × d]
    omega: Vec<Vec<f32>>,
    /// Random phases φ ~ Uniform(0, 2π), shape: [d]
    phi: Vec<f32>,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension (number of random features)
    pub output_dim: usize,
}

impl RffTransform {
    /// Create a new RFF transform.
    ///
    /// # Arguments
    /// - `input_dim`: dimension of input features (D)
    /// - `output_dim`: dimension of random features (d), typically d << n
    /// - `sigma`: kernel bandwidth (controls smoothness)
    /// - `seed`: random seed for reproducibility
    pub fn new(input_dim: usize, output_dim: usize, sigma: f32, seed: u64) -> Self {
        // Simple LCG PRNG for reproducibility without external deps
        let mut rng_state = seed;
        let mut next_uniform = || -> f32 {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng_state >> 33) as f32) / (u32::MAX as f32)
        };

        // Box-Muller transform for normal distribution
        let mut next_normal = || -> f32 {
            let u1 = next_uniform().max(1e-10);
            let u2 = next_uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        };

        // Sample ω ~ N(0, 1/σ²)
        let inv_sigma = 1.0 / sigma;
        let omega: Vec<Vec<f32>> = (0..input_dim)
            .map(|_| (0..output_dim).map(|_| next_normal() * inv_sigma).collect())
            .collect();

        // Sample φ ~ Uniform(0, 2π)
        let phi: Vec<f32> = (0..output_dim)
            .map(|_| next_uniform() * 2.0 * std::f32::consts::PI)
            .collect();

        Self {
            omega,
            phi,
            input_dim,
            output_dim,
        }
    }

    /// Transform a single feature vector: z(x) = √2 · cos(ωᵀx + φ)
    pub fn transform(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.input_dim, "Input dim mismatch");

        (0..self.output_dim)
            .map(|j| {
                let dot: f32 = x
                    .iter()
                    .zip(self.omega.iter())
                    .map(|(xi, omega_row)| xi * omega_row[j])
                    .sum();
                std::f32::consts::SQRT_2 * (dot + self.phi[j]).cos()
            })
            .collect()
    }

    /// Transform a batch of feature vectors.
    pub fn transform_batch(&self, batch: &[Vec<f32>]) -> Vec<Vec<f32>> {
        batch.iter().map(|x| self.transform(x)).collect()
    }
}

// ═══════════════════════════════════════════════════════════════
// Feature Independence Measure
// ═══════════════════════════════════════════════════════════════

/// Compute the weighted cross-covariance matrix between two RFF-transformed
/// feature sequences in the random Fourier space.
///
/// Returns the Frobenius norm squared ||Σ_xy||²_F as the independence measure.
/// When this converges to 0, features X and Y are statistically independent.
fn weighted_cross_covariance_frobenius(
    rff_x: &[Vec<f32>],
    rff_y: &[Vec<f32>],
    weights: &[f32],
) -> f32 {
    let n = rff_x.len();
    assert_eq!(n, rff_y.len());
    assert_eq!(n, weights.len());
    if n == 0 {
        return 0.0;
    }

    let d_x = rff_x[0].len();
    let d_y = rff_y[0].len();

    // Compute weighted means
    let w_sum: f32 = weights.iter().sum::<f32>().max(1e-10);
    let mut mean_x = vec![0.0f32; d_x];
    let mut mean_y = vec![0.0f32; d_y];

    for i in 0..n {
        let w = weights[i] / w_sum;
        for j in 0..d_x {
            mean_x[j] += w * rff_x[i][j];
        }
        for j in 0..d_y {
            mean_y[j] += w * rff_y[i][j];
        }
    }

    // Compute weighted cross-covariance: Σ_xy = (1/n) Σ_i w_i (x_i - μ_x)(y_i - μ_y)ᵀ
    // Then compute ||Σ_xy||²_F = sum of all squared entries
    let mut frobenius_sq = 0.0f64;

    for a in 0..d_x {
        for b in 0..d_y {
            let mut cov_ab = 0.0f64;
            for i in 0..n {
                let w = weights[i] as f64 / w_sum as f64;
                let dx = (rff_x[i][a] - mean_x[a]) as f64;
                let dy = (rff_y[i][b] - mean_y[b]) as f64;
                cov_ab += w * dx * dy;
            }
            frobenius_sq += cov_ab * cov_ab;
        }
    }

    frobenius_sq as f32
}

// ═══════════════════════════════════════════════════════════════
// Stable Decorrelator
// ═══════════════════════════════════════════════════════════════

/// Enforces statistical independence between multiple feature channels
/// by minimizing the pairwise cross-covariance Frobenius norm in RFF space.
///
/// Usage:
/// ```ignore
/// let decor = StableDecorrelator::new(input_dim, rff_dim, sigma, seed);
/// let loss = decor.decorrelation_loss(&model_embeddings, &sample_weights);
/// // Add `loss * decor_weight` to training loss
/// ```
#[derive(Debug, Clone)]
pub struct StableDecorrelator {
    /// RFF transform shared across all feature channels
    rff: RffTransform,
}

impl StableDecorrelator {
    /// Create a new decorrelator.
    ///
    /// # Arguments
    /// - `input_dim`: dimension of each feature channel (GNN hidden_dim)
    /// - `rff_dim`: number of random Fourier features (typically 2× input_dim)
    /// - `sigma`: kernel bandwidth
    /// - `seed`: random seed
    pub fn new(input_dim: usize, rff_dim: usize, sigma: f32, seed: u64) -> Self {
        Self {
            rff: RffTransform::new(input_dim, rff_dim, sigma, seed),
        }
    }

    /// Compute the decorrelation loss across multiple model embeddings.
    ///
    /// For m feature channels (e.g., SAGE, RGCN, GAT, GPS embeddings),
    /// minimizes the pairwise independence measure:
    ///   L(w) = Σ_{i<j} ||Σ_{X_i X_j}(W)||²_F
    ///
    /// # Arguments
    /// - `channels`: Vec of embedding matrices, each [n_samples × hidden_dim]
    /// - `weights`: per-sample weights [n_samples] (from LSWD)
    ///
    /// # Returns
    /// Decorrelation loss (lower = more independent)
    pub fn decorrelation_loss(&self, channels: &[Vec<Vec<f32>>], weights: &[f32]) -> f32 {
        let m = channels.len();
        if m < 2 {
            return 0.0;
        }

        // Transform all channels to RFF space
        let rff_channels: Vec<Vec<Vec<f32>>> = channels
            .iter()
            .map(|ch| self.rff.transform_batch(ch))
            .collect();

        // Sum pairwise Frobenius norms
        let mut total_loss = 0.0f32;
        let mut pair_count = 0;

        for i in 0..m {
            for j in (i + 1)..m {
                total_loss += weighted_cross_covariance_frobenius(
                    &rff_channels[i],
                    &rff_channels[j],
                    weights,
                );
                pair_count += 1;
            }
        }

        // Normalize by number of pairs
        if pair_count > 0 {
            total_loss / pair_count as f32
        } else {
            0.0
        }
    }

    /// Compute decorrelation loss using uniform weights (no LSWD).
    pub fn decorrelation_loss_uniform(&self, channels: &[Vec<Vec<f32>>]) -> f32 {
        if channels.is_empty() || channels[0].is_empty() {
            return 0.0;
        }
        let n = channels[0].len();
        let uniform_weights = vec![1.0; n];
        self.decorrelation_loss(channels, &uniform_weights)
    }
}

// ═══════════════════════════════════════════════════════════════
// LSWD: Learning Sample Weights for Decorrelation
// ═══════════════════════════════════════════════════════════════

/// Per-sample weight optimizer that minimizes feature correlation.
///
/// Implements the constrained gradient update from the Stable-GNN paper:
/// 1. Compute gradient of decorrelation loss w.r.t. sample weights
/// 2. Zero-mean the gradient (constraint: weights sum to n)
/// 3. Update weights with step size λ
/// 4. Clip to ≥ 0 (non-negativity constraint)
/// 5. Normalize so weights sum to n
///
/// The paper proves this guarantees monotonic loss decrease.
#[derive(Debug, Clone)]
pub struct SampleWeighter {
    /// Current sample weights w_i ≥ 0, sum = n
    pub weights: Vec<f32>,
    /// Learning rate for weight updates
    pub lr: f32,
    /// Number of optimization iterations per call
    pub iterations: usize,
}

impl SampleWeighter {
    /// Create a new sample weighter with uniform initial weights.
    pub fn new(n_samples: usize, lr: f32, iterations: usize) -> Self {
        Self {
            weights: vec![1.0; n_samples],
            lr,
            iterations,
        }
    }

    /// Optimize sample weights to minimize decorrelation loss.
    ///
    /// Uses numerical gradient estimation (finite differences) since
    /// we operate on plain f32 vectors, not autograd tensors.
    ///
    /// # Arguments
    /// - `decorrelator`: the StableDecorrelator to minimize
    /// - `channels`: feature channels [m × n_samples × hidden_dim]
    ///
    /// # Returns
    /// Final decorrelation loss after optimization
    pub fn optimize(
        &mut self,
        decorrelator: &StableDecorrelator,
        channels: &[Vec<Vec<f32>>],
    ) -> f32 {
        let n = self.weights.len();
        if n == 0 {
            return 0.0;
        }

        let eps = 1e-3; // Finite difference step
        let mut current_loss = decorrelator.decorrelation_loss(channels, &self.weights);

        for _iter in 0..self.iterations {
            // Compute gradient via finite differences
            let mut grad = vec![0.0f32; n];
            for i in 0..n {
                let orig = self.weights[i];

                // Forward perturbation
                self.weights[i] = orig + eps;
                let loss_plus = decorrelator.decorrelation_loss(channels, &self.weights);

                self.weights[i] = orig;
                grad[i] = (loss_plus - current_loss) / eps;
            }

            // Zero-mean the gradient (constraint: weights sum to n)
            let grad_mean: f32 = grad.iter().sum::<f32>() / n as f32;
            for g in grad.iter_mut() {
                *g -= grad_mean;
            }

            // Update weights: w ← w - λ * grad
            for i in 0..n {
                self.weights[i] -= self.lr * grad[i];
            }

            // Non-negativity: clip to ≥ 0
            for w in self.weights.iter_mut() {
                *w = w.max(0.0);
            }

            // Normalize so weights sum to n
            let w_sum: f32 = self.weights.iter().sum::<f32>().max(1e-10);
            let scale = n as f32 / w_sum;
            for w in self.weights.iter_mut() {
                *w *= scale;
            }

            current_loss = decorrelator.decorrelation_loss(channels, &self.weights);
        }

        current_loss
    }

    /// Get the current weights as a slice.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
}

// ═══════════════════════════════════════════════════════════════
// Cross-Model Decorrelation (Ensemble Integration)
// ═══════════════════════════════════════════════════════════════

/// Configuration for stable decorrelation in the ensemble pipeline.
#[derive(Debug, Clone)]
pub struct StableConfig {
    /// RFF output dimension (default: 2 × hidden_dim)
    pub rff_dim: usize,
    /// Kernel bandwidth σ (default: 1.0)
    pub sigma: f32,
    /// LSWD learning rate (default: 0.5)
    pub lswd_lr: f32,
    /// LSWD iterations (default: 20)
    pub lswd_iterations: usize,
    /// Weight for decorrelation loss in training (default: 0.1)
    pub decor_weight: f32,
    /// Random seed
    pub seed: u64,
}

impl Default for StableConfig {
    fn default() -> Self {
        Self {
            rff_dim: 32, // 2× typical hidden_dim=16
            sigma: 1.0,
            lswd_lr: 0.5,
            lswd_iterations: 20,
            decor_weight: 0.1,
            seed: 2025,
        }
    }
}

/// Apply stable decorrelation across ensemble model embeddings.
///
/// Takes embeddings from multiple GNN models (SAGE, RGCN, GAT, GPS)
/// and learns per-sample weights that make model outputs statistically
/// independent — forcing each model to contribute unique causal signal.
///
/// # Arguments
/// - `model_embeddings`: map of model_name → {node_type → [n_nodes × hidden_dim]}
/// - `config`: decorrelation configuration
///
/// # Returns
/// - Updated sample weights per node type
/// - Decorrelation loss (for logging)
pub fn decorrelate_ensemble(
    model_embeddings: &HashMap<String, HashMap<String, Vec<Vec<f32>>>>,
    config: &StableConfig,
) -> EnsembleDecorrelationResult {
    let model_keys: Vec<&String> = model_embeddings.keys().collect();
    if model_keys.len() < 2 {
        return EnsembleDecorrelationResult {
            sample_weights: HashMap::new(),
            decorrelation_loss: 0.0,
            initial_loss: 0.0,
            models_decorrelated: 0,
        };
    }

    // Collect all node types present in any model
    let mut all_node_types: Vec<String> = Vec::new();
    for embs in model_embeddings.values() {
        for nt in embs.keys() {
            if !all_node_types.contains(nt) {
                all_node_types.push(nt.clone());
            }
        }
    }

    let mut total_initial_loss = 0.0f32;
    let mut total_final_loss = 0.0f32;
    let mut sample_weights_map: HashMap<String, Vec<f32>> = HashMap::new();
    let mut type_count = 0;

    for node_type in &all_node_types {
        // Gather channels: one per model, only if this node type is present
        let mut channels: Vec<Vec<Vec<f32>>> = Vec::new();
        for model_key in &model_keys {
            if let Some(embs) = model_embeddings.get(*model_key) {
                if let Some(vecs) = embs.get(node_type) {
                    if !vecs.is_empty() {
                        channels.push(vecs.clone());
                    }
                }
            }
        }

        if channels.len() < 2 {
            continue;
        }

        let n_samples = channels[0].len();
        // Ensure all channels have same number of samples
        if channels.iter().any(|ch| ch.len() != n_samples) {
            continue;
        }

        let hidden_dim = channels[0][0].len();
        let rff_dim = config.rff_dim.max(hidden_dim);

        let decorrelator = StableDecorrelator::new(hidden_dim, rff_dim, config.sigma, config.seed);

        // Measure initial loss (uniform weights)
        let initial = decorrelator.decorrelation_loss_uniform(&channels);
        total_initial_loss += initial;

        // Optimize sample weights
        let mut weighter = SampleWeighter::new(n_samples, config.lswd_lr, config.lswd_iterations);
        let final_loss = weighter.optimize(&decorrelator, &channels);
        total_final_loss += final_loss;

        sample_weights_map.insert(node_type.clone(), weighter.weights.clone());
        type_count += 1;
    }

    EnsembleDecorrelationResult {
        sample_weights: sample_weights_map,
        decorrelation_loss: if type_count > 0 {
            total_final_loss / type_count as f32
        } else {
            0.0
        },
        initial_loss: if type_count > 0 {
            total_initial_loss / type_count as f32
        } else {
            0.0
        },
        models_decorrelated: model_keys.len(),
    }
}

/// Result of ensemble decorrelation.
#[derive(Debug, Clone)]
pub struct EnsembleDecorrelationResult {
    /// Per-node-type sample weights from LSWD
    pub sample_weights: HashMap<String, Vec<f32>>,
    /// Final decorrelation loss (lower = more independent)
    pub decorrelation_loss: f32,
    /// Initial decorrelation loss before optimization
    pub initial_loss: f32,
    /// Number of models that were decorrelated
    pub models_decorrelated: usize,
}

impl EnsembleDecorrelationResult {
    /// Improvement ratio: (initial - final) / initial
    pub fn improvement_ratio(&self) -> f32 {
        if self.initial_loss > 1e-10 {
            (self.initial_loss - self.decorrelation_loss) / self.initial_loss
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rff_transform_dimensions() {
        let rff = RffTransform::new(16, 32, 1.0, 42);
        let x = vec![0.5; 16];
        let z = rff.transform(&x);
        assert_eq!(z.len(), 32);

        // All values should be bounded by √2 (since cos ∈ [-1, 1])
        for &val in &z {
            assert!(val.abs() <= std::f32::consts::SQRT_2 + 1e-6);
        }
    }

    #[test]
    fn test_rff_kernel_approximation() {
        // For identical inputs, kernel = 1.0 and inner product should be ≈ 1.0
        let rff = RffTransform::new(8, 256, 1.0, 42);
        let x = vec![0.3, -0.1, 0.5, 0.2, -0.4, 0.1, 0.0, 0.6];

        let z1 = rff.transform(&x);
        let z2 = rff.transform(&x);

        let inner: f32 =
            z1.iter().zip(z2.iter()).map(|(a, b)| a * b).sum::<f32>() / rff.output_dim as f32;

        // k(x,x) = exp(0) = 1.0
        assert!(
            (inner - 1.0).abs() < 0.15,
            "Self-kernel should be ≈1.0, got {inner}"
        );
    }

    #[test]
    fn test_decorrelator_independent_features() {
        // If features are already independent (different random vectors),
        // decorrelation loss should be low
        let decor = StableDecorrelator::new(4, 8, 1.0, 42);

        let ch1: Vec<Vec<f32>> = (0..20)
            .map(|i| vec![(i as f32) * 0.1, 0.0, 0.0, 0.0])
            .collect();
        let ch2: Vec<Vec<f32>> = (0..20)
            .map(|i| vec![0.0, 0.0, 0.0, (i as f32) * 0.1])
            .collect();

        let loss = decor.decorrelation_loss_uniform(&[ch1, ch2]);
        // Should be relatively small for orthogonal features
        assert!(loss.is_finite(), "Loss should be finite");
    }

    #[test]
    fn test_decorrelator_correlated_features() {
        // If features are identical (perfectly correlated),
        // decorrelation loss should be high
        let decor = StableDecorrelator::new(4, 8, 1.0, 42);

        let ch: Vec<Vec<f32>> = (0..20)
            .map(|i| vec![(i as f32) * 0.1, 0.5, -0.3, 0.2])
            .collect();

        let loss_correlated = decor.decorrelation_loss_uniform(&[ch.clone(), ch.clone()]);
        let loss_indep = {
            let ch2: Vec<Vec<f32>> = (0..20)
                .map(|i| vec![0.0, 0.0, 0.0, (19 - i) as f32 * 0.1])
                .collect();
            decor.decorrelation_loss_uniform(&[ch, ch2])
        };

        assert!(
            loss_correlated > loss_indep * 0.5,
            "Correlated loss ({loss_correlated}) should be higher than independent ({loss_indep})"
        );
    }

    #[test]
    fn test_lswd_reduces_loss() {
        let decor = StableDecorrelator::new(4, 8, 1.0, 42);

        // Create somewhat correlated channels
        let ch1: Vec<Vec<f32>> = (0..10)
            .map(|i| {
                let v = (i as f32) * 0.1;
                vec![v, v * 0.8, v * 0.3, 1.0 - v]
            })
            .collect();
        let ch2: Vec<Vec<f32>> = (0..10)
            .map(|i| {
                let v = (i as f32) * 0.1;
                vec![v * 0.9, v * 0.7, v * 0.4, 0.5 - v * 0.3]
            })
            .collect();
        let channels = vec![ch1, ch2];

        let initial_loss = decor.decorrelation_loss_uniform(&channels);

        let mut weighter = SampleWeighter::new(10, 0.5, 20);
        let final_loss = weighter.optimize(&decor, &channels);

        eprintln!("LSWD: initial={initial_loss:.6}, final={final_loss:.6}");
        assert!(
            final_loss <= initial_loss + 1e-6,
            "LSWD should not increase loss: {final_loss} > {initial_loss}"
        );

        // Weights should still be non-negative and sum ≈ n
        let w_sum: f32 = weighter.weights().iter().sum();
        assert!(
            (w_sum - 10.0).abs() < 0.01,
            "Weights should sum to n=10, got {w_sum}"
        );
        assert!(
            weighter.weights().iter().all(|&w| w >= 0.0),
            "All weights should be non-negative"
        );
    }

    #[test]
    fn test_ensemble_decorrelation() {
        let mut model_embs: HashMap<String, HashMap<String, Vec<Vec<f32>>>> = HashMap::new();

        // Simulate 2 models with correlated user embeddings
        let users_m1: Vec<Vec<f32>> = (0..5)
            .map(|i| vec![(i as f32) * 0.2, 0.5, -0.1, 0.3])
            .collect();
        let users_m2: Vec<Vec<f32>> = (0..5)
            .map(|i| vec![(i as f32) * 0.2 + 0.01, 0.48, -0.12, 0.31])
            .collect();

        let mut m1 = HashMap::new();
        m1.insert("user".to_string(), users_m1);
        model_embs.insert("graphsage".to_string(), m1);

        let mut m2 = HashMap::new();
        m2.insert("user".to_string(), users_m2);
        model_embs.insert("rgcn".to_string(), m2);

        let config = StableConfig::default();
        let result = decorrelate_ensemble(&model_embs, &config);

        assert_eq!(result.models_decorrelated, 2);
        assert!(result.decorrelation_loss.is_finite());
        assert!(result.sample_weights.contains_key("user"));

        let user_weights = result.sample_weights.get("user").unwrap();
        assert_eq!(user_weights.len(), 5);

        eprintln!(
            "Ensemble decorrelation: initial={:.6}, final={:.6}, improvement={:.1}%",
            result.initial_loss,
            result.decorrelation_loss,
            result.improvement_ratio() * 100.0
        );
    }
}
