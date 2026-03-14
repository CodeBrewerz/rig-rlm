//! VICReg Collapse Prevention (from jepa-rs)
//!
//! Implements Variance-Invariance-Covariance Regularization:
//! 1. **Variance**: each embedding dimension has high variance across nodes
//! 2. **Invariance**: paired representations (positive edges) are similar
//! 3. **Covariance**: off-diagonal covariance → 0 (dimensions decorrelated)
//!
//! This is stronger than our existing uniformity-only loss because it
//! explicitly prevents three failure modes of representation collapse.
//!
//! Reference: Zbontar et al. (2021), "Barlow Twins" + Bardes et al. (2022), "VICReg"

use std::collections::HashMap;

/// VICReg configuration.
#[derive(Debug, Clone)]
pub struct VICRegConfig {
    /// Weight for variance term (default: 25.0).
    pub variance_weight: f32,
    /// Weight for invariance term (default: 25.0).
    pub invariance_weight: f32,
    /// Weight for covariance term (default: 1.0).
    pub covariance_weight: f32,
    /// Target standard deviation for variance hinge (default: 1.0).
    pub target_std: f32,
}

impl Default for VICRegConfig {
    fn default() -> Self {
        Self {
            variance_weight: 25.0,
            invariance_weight: 25.0,
            covariance_weight: 1.0,
            target_std: 1.0,
        }
    }
}

/// Decomposed VICReg loss for inspection.
#[derive(Debug, Clone)]
pub struct VICRegLoss {
    pub variance_loss: f32,
    pub invariance_loss: f32,
    pub covariance_loss: f32,
    pub total: f32,
}

/// Compute VICReg loss on node embeddings.
///
/// `z_a` and `z_b` are embeddings of positive-edge endpoints.
pub fn compute_vicreg_loss(
    embeddings: &HashMap<String, Vec<Vec<f32>>>,
    positive: &[(String, usize, String, usize)],
    config: &VICRegConfig,
) -> VICRegLoss {
    // Collect paired embeddings from positive edges
    let mut z_a: Vec<Vec<f32>> = Vec::new();
    let mut z_b: Vec<Vec<f32>> = Vec::new();

    for (src_type, src_idx, dst_type, dst_idx) in positive {
        let a = embeddings.get(src_type).and_then(|v| v.get(*src_idx));
        let b = embeddings.get(dst_type).and_then(|v| v.get(*dst_idx));
        if let (Some(a), Some(b)) = (a, b) {
            z_a.push(a.clone());
            z_b.push(b.clone());
        }
    }

    if z_a.is_empty() || z_a[0].is_empty() {
        return VICRegLoss {
            variance_loss: 0.0,
            invariance_loss: 0.0,
            covariance_loss: 0.0,
            total: 0.0,
        };
    }

    let n = z_a.len() as f32;
    let d = z_a[0].len();

    // 1. Invariance: MSE between paired representations
    let invariance_loss = {
        let mut total = 0.0f32;
        for (a, b) in z_a.iter().zip(z_b.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                total += (x - y).powi(2);
            }
        }
        total / n
    };

    // 2. Variance: hinge loss on per-dimension std
    // var_loss = max(0, target_std - std(z_j)) for each dimension j
    let variance_loss = {
        let var_a = compute_variance_hinge(&z_a, config.target_std);
        let var_b = compute_variance_hinge(&z_b, config.target_std);
        (var_a + var_b) / 2.0
    };

    // 3. Covariance: off-diagonal elements of covariance matrix → 0
    let covariance_loss = {
        let cov_a = compute_covariance_loss(&z_a);
        let cov_b = compute_covariance_loss(&z_b);
        (cov_a + cov_b) / 2.0
    };

    let total = config.variance_weight * variance_loss
        + config.invariance_weight * invariance_loss
        + config.covariance_weight * covariance_loss;

    VICRegLoss {
        variance_loss,
        invariance_loss,
        covariance_loss,
        total,
    }
}

/// Variance hinge loss: Σ max(0, target_std - std(z_j))
fn compute_variance_hinge(z: &[Vec<f32>], target_std: f32) -> f32 {
    if z.is_empty() || z[0].is_empty() {
        return 0.0;
    }
    let n = z.len() as f32;
    let d = z[0].len();

    let mut total = 0.0f32;
    for j in 0..d {
        let mean: f32 = z.iter().map(|v| v[j]).sum::<f32>() / n;
        let variance: f32 = z.iter().map(|v| (v[j] - mean).powi(2)).sum::<f32>() / n;
        let std = variance.sqrt();
        let hinge = (target_std - std).max(0.0);
        total += hinge;
    }
    total / d as f32
}

/// Covariance loss: sum of squared off-diagonal covariance matrix elements.
fn compute_covariance_loss(z: &[Vec<f32>]) -> f32 {
    if z.is_empty() || z[0].is_empty() {
        return 0.0;
    }
    let n = z.len() as f32;
    let d = z[0].len();

    // Center embeddings
    let mut means = vec![0.0f32; d];
    for v in z {
        for (j, &val) in v.iter().enumerate() {
            means[j] += val;
        }
    }
    for m in means.iter_mut() {
        *m /= n;
    }

    // Compute covariance matrix (only off-diagonal elements)
    let mut total = 0.0f32;
    for i in 0..d {
        for j in (i + 1)..d {
            let cov: f32 = z
                .iter()
                .map(|v| (v[i] - means[i]) * (v[j] - means[j]))
                .sum::<f32>()
                / n;
            total += cov.powi(2);
        }
    }
    total / d as f32
}

/// Convenience: compute VICReg on all node embeddings (not just pairs).
///
/// Useful for adding as a general regularizer to the training loss.
pub fn compute_vicreg_regularizer(
    embeddings: &HashMap<String, Vec<Vec<f32>>>,
    config: &VICRegConfig,
) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0;
    for (_nt, vecs) in embeddings {
        if vecs.len() < 2 || vecs[0].is_empty() {
            continue;
        }
        let var = compute_variance_hinge(vecs, config.target_std);
        let cov = compute_covariance_loss(vecs);
        total += config.variance_weight * var + config.covariance_weight * cov;
        count += 1;
    }
    if count > 0 {
        total / count as f32
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vicreg_identical_pairs() {
        let mut embeddings = HashMap::new();
        embeddings.insert(
            "user".to_string(),
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        );

        // Same type, same index → invariance loss = 0
        let positive = vec![
            ("user".to_string(), 0, "user".to_string(), 0),
            ("user".to_string(), 1, "user".to_string(), 1),
        ];

        let config = VICRegConfig::default();
        let loss = compute_vicreg_loss(&embeddings, &positive, &config);
        assert!(
            loss.invariance_loss < 1e-6,
            "Invariance should be ~0 for identical pairs, got {}",
            loss.invariance_loss
        );
    }

    #[test]
    fn test_vicreg_collapsed_has_high_variance_loss() {
        let mut embeddings = HashMap::new();
        // All embeddings are identical → collapsed representation
        embeddings.insert("user".to_string(), vec![vec![1.0, 1.0, 1.0]; 5]);

        let positive = vec![
            ("user".to_string(), 0, "user".to_string(), 1),
            ("user".to_string(), 2, "user".to_string(), 3),
        ];

        let config = VICRegConfig::default();
        let loss = compute_vicreg_loss(&embeddings, &positive, &config);
        // Collapsed embeddings have zero variance → high variance hinge loss
        assert!(
            loss.variance_loss > 0.5,
            "Collapsed embeddings should have high variance loss, got {}",
            loss.variance_loss
        );
    }

    #[test]
    fn test_vicreg_decorrelated_has_low_covariance() {
        let mut embeddings = HashMap::new();
        embeddings.insert(
            "user".to_string(),
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![-1.0, 0.0, 0.0],
                vec![0.0, -1.0, 0.0],
            ],
        );

        let positive = vec![("user".to_string(), 0, "user".to_string(), 1)];

        let config = VICRegConfig::default();
        let loss = compute_vicreg_loss(&embeddings, &positive, &config);
        assert!(
            loss.covariance_loss < 0.1,
            "Decorrelated embeddings should have low covariance loss, got {}",
            loss.covariance_loss
        );
    }

    #[test]
    fn test_vicreg_regularizer() {
        let mut embeddings = HashMap::new();
        embeddings.insert(
            "user".to_string(),
            vec![vec![1.0, 1.0]; 5], // Collapsed → high loss
        );

        let config = VICRegConfig::default();
        let reg = compute_vicreg_regularizer(&embeddings, &config);
        assert!(
            reg > 0.0,
            "Regularizer should be positive for collapsed embeddings"
        );
    }
}
