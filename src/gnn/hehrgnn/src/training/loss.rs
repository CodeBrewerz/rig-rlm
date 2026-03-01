//! Loss functions for knowledge graph embedding training.
//!
//! Implements the margin ranking loss used to push positive fact scores
//! above negative fact scores by a configurable margin γ.

use burn::prelude::*;

/// Margin ranking loss for contrastive training.
///
/// For each positive-negative pair:
/// `loss = max(0, γ - score(positive) + score(negative))`
///
/// # Arguments
/// - `positive_scores`: `[batch_size]` — scores of positive (true) facts
/// - `negative_scores`: `[batch_size]` — scores of negative (corrupted) facts
/// - `margin`: γ — the desired margin between positive and negative scores
///
/// # Returns
/// Scalar loss tensor.
pub fn margin_ranking_loss<B: Backend>(
    positive_scores: Tensor<B, 1>,
    negative_scores: Tensor<B, 1>,
    margin: f64,
) -> Tensor<B, 1> {
    // γ - score(f) + score(f')
    let margin_tensor = negative_scores - positive_scores + margin;

    // max(0, ...)
    let zero = Tensor::zeros_like(&margin_tensor);
    let clamped = margin_tensor.max_pair(zero);

    // Mean over batch
    clamped.mean().unsqueeze()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_margin_loss_perfect_separation() {
        let device = <TestBackend as Backend>::Device::default();

        // Positive scores much higher than negative → loss should be ~0
        let pos = Tensor::<TestBackend, 1>::from_data([5.0, 6.0, 7.0], &device);
        let neg = Tensor::<TestBackend, 1>::from_data([1.0, 1.0, 1.0], &device);
        let loss = margin_ranking_loss(pos, neg, 1.0);
        let loss_val: f32 = loss.into_scalar().elem();
        assert!(
            loss_val < 0.01,
            "Loss should be near zero, got {}",
            loss_val
        );
    }

    #[test]
    fn test_margin_loss_violation() {
        let device = <TestBackend as Backend>::Device::default();

        // Negative scores higher than positive → should produce positive loss
        let pos = Tensor::<TestBackend, 1>::from_data([1.0, 1.0], &device);
        let neg = Tensor::<TestBackend, 1>::from_data([3.0, 4.0], &device);
        let loss = margin_ranking_loss(pos, neg, 1.0);
        let loss_val: f32 = loss.into_scalar().elem();
        assert!(loss_val > 0.0, "Loss should be positive, got {}", loss_val);
    }
}
