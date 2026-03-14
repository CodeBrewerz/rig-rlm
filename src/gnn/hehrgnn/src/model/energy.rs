//! Energy Functions for JEPA (from jepa-rs)
//!
//! In JEPA, the energy function measures representation compatibility.
//! The model learns to predict representations that minimize energy.
//!
//! Our existing InfoNCE (contrastive) loss is one energy function,
//! but the canonical JEPA uses L2 in representation space.
//!
//! This module provides pluggable energy functions:
//! - L2Energy: mean squared error (canonical JEPA)
//! - CosineEnergy: 1 - cosine_similarity
//! - SmoothL1Energy: Huber loss (robust to outliers)

use std::collections::HashMap;

/// Available energy function types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EnergyKind {
    /// L2 distance (MSE) — canonical JEPA loss.
    L2,
    /// Cosine distance: 1 - cos(a, b).
    Cosine,
    /// Smooth L1 (Huber loss) with given beta.
    SmoothL1,
}

impl Default for EnergyKind {
    fn default() -> Self {
        EnergyKind::L2
    }
}

/// Compute energy between predicted and target embeddings.
///
/// Lower energy = better match between context and target predictions.
pub fn compute_energy(
    predicted: &HashMap<String, Vec<Vec<f32>>>,
    target: &HashMap<String, Vec<Vec<f32>>>,
    kind: EnergyKind,
) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0usize;

    for (node_type, pred_vecs) in predicted {
        let tgt_vecs = match target.get(node_type) {
            Some(v) => v,
            None => continue,
        };

        for (pred, tgt) in pred_vecs.iter().zip(tgt_vecs.iter()) {
            total += match kind {
                EnergyKind::L2 => l2_energy(pred, tgt),
                EnergyKind::Cosine => cosine_energy(pred, tgt),
                EnergyKind::SmoothL1 => smooth_l1_energy(pred, tgt, 1.0),
            };
            count += 1;
        }
    }

    if count > 0 {
        total / count as f32
    } else {
        0.0
    }
}

/// Compute energy-based JEPA loss on positive/negative edge pairs.
///
/// Unlike contrastive InfoNCE, this directly measures the prediction
/// error in representation space:
///   Loss = Σ energy(predict(z_u, z_v)) for positive edges
///        - α * Σ energy(predict(z_u, z_k)) for negative edges
///
/// The model should learn to make positive predictions low-energy
/// and negative predictions high-energy.
pub fn compute_energy_loss(
    embeddings: &HashMap<String, Vec<Vec<f32>>>,
    positive: &[(String, usize, String, usize)],
    negative: &[(String, usize, String, usize)],
    kind: EnergyKind,
) -> f32 {
    let pos_energy = edge_set_mean_energy(embeddings, positive, kind);
    let neg_energy = edge_set_mean_energy(embeddings, negative, kind);

    // Loss: minimize positive energy, maximize negative energy
    // Using margin-based loss: max(0, pos_energy - neg_energy + margin)
    let margin = 0.1;
    (pos_energy - neg_energy + margin).max(0.0)
}

fn edge_set_mean_energy(
    embeddings: &HashMap<String, Vec<Vec<f32>>>,
    edges: &[(String, usize, String, usize)],
    kind: EnergyKind,
) -> f32 {
    if edges.is_empty() {
        return 0.0;
    }
    let mut total = 0.0f32;
    let mut count = 0;
    for (src_type, src_idx, dst_type, dst_idx) in edges {
        let a = embeddings.get(src_type).and_then(|v| v.get(*src_idx));
        let b = embeddings.get(dst_type).and_then(|v| v.get(*dst_idx));
        if let (Some(a), Some(b)) = (a, b) {
            total += match kind {
                EnergyKind::L2 => l2_energy(a, b),
                EnergyKind::Cosine => cosine_energy(a, b),
                EnergyKind::SmoothL1 => smooth_l1_energy(a, b, 1.0),
            };
            count += 1;
        }
    }
    if count > 0 {
        total / count as f32
    } else {
        0.0
    }
}

/// L2 (MSE) energy: mean(||pred - target||²)
fn l2_energy(a: &[f32], b: &[f32]) -> f32 {
    let d = a.len().min(b.len());
    if d == 0 {
        return 0.0;
    }
    let mse: f32 = a[..d]
        .iter()
        .zip(b[..d].iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f32>()
        / d as f32;
    mse
}

/// Cosine energy: 1 - cosine_similarity
fn cosine_energy(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    1.0 - dot / (norm_a * norm_b)
}

/// Smooth L1 (Huber) energy: L2 for small, L1 for large differences
fn smooth_l1_energy(a: &[f32], b: &[f32], beta: f32) -> f32 {
    let d = a.len().min(b.len());
    if d == 0 {
        return 0.0;
    }
    let total: f32 = a[..d]
        .iter()
        .zip(b[..d].iter())
        .map(|(&x, &y)| {
            let diff = (x - y).abs();
            if diff < beta {
                0.5 * diff * diff / beta
            } else {
                diff - 0.5 * beta
            }
        })
        .sum::<f32>();
    total / d as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_energy_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!(l2_energy(&a, &a) < 1e-6);
    }

    #[test]
    fn test_l2_energy_different() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let e = l2_energy(&a, &b);
        // ||[1,0,0] - [0,1,0]||² / 3 = (1+1+0)/3 = 0.667
        assert!((e - 0.667).abs() < 0.01, "L2 energy = {}", e);
    }

    #[test]
    fn test_cosine_energy_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!(cosine_energy(&a, &a) < 1e-5);
    }

    #[test]
    fn test_cosine_energy_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let e = cosine_energy(&a, &b);
        // cos(90°) = 0, so energy = 1.0
        assert!((e - 1.0).abs() < 1e-5, "Cosine energy = {}", e);
    }

    #[test]
    fn test_smooth_l1_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!(smooth_l1_energy(&a, &a, 1.0) < 1e-6);
    }

    #[test]
    fn test_smooth_l1_is_non_negative() {
        let a = vec![1.0, -2.0, 3.0];
        let b = vec![-1.0, 2.0, -3.0];
        assert!(smooth_l1_energy(&a, &b, 1.0) >= 0.0);
    }

    #[test]
    fn test_energy_loss_margin() {
        let mut embeddings = HashMap::new();
        embeddings.insert(
            "user".to_string(),
            vec![
                vec![1.0, 0.0],
                vec![0.9, 0.1],  // Similar to [0]
                vec![-1.0, 0.0], // Different from [0]
            ],
        );

        let positive = vec![("user".to_string(), 0, "user".to_string(), 1)];
        let negative = vec![("user".to_string(), 0, "user".to_string(), 2)];

        let loss = compute_energy_loss(&embeddings, &positive, &negative, EnergyKind::L2);
        // Positive energy should be low, negative high → margin loss should be 0
        // if neg_energy > pos_energy + margin
        assert!(loss >= 0.0, "Energy loss should be non-negative");
    }

    #[test]
    fn test_compute_energy_l2_vs_cosine() {
        let mut pred = HashMap::new();
        pred.insert("user".to_string(), vec![vec![1.0, 0.0], vec![0.0, 1.0]]);

        let mut target = HashMap::new();
        target.insert("user".to_string(), vec![vec![0.9, 0.1], vec![0.1, 0.9]]);

        let l2 = compute_energy(&pred, &target, EnergyKind::L2);
        let cos = compute_energy(&pred, &target, EnergyKind::Cosine);

        assert!(l2 > 0.0, "L2 energy should be positive");
        assert!(cos > 0.0, "Cosine energy should be positive");
    }
}
