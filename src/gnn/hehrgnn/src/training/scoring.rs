//! Scoring functions for knowledge graph link prediction.
//!
//! A scorer takes (head, relation, tail) embeddings and produces a scalar
//! plausibility score for each triple in the batch.

use burn::prelude::*;

/// Trait for scoring functions that evaluate triple plausibility.
pub trait Scorer<B: Backend> {
    /// Score a batch of triples.
    ///
    /// # Arguments
    /// - `head`: `[batch_size, hidden_dim]`
    /// - `relation`: `[batch_size, hidden_dim]`
    /// - `tail`: `[batch_size, hidden_dim]`
    ///
    /// # Returns
    /// Scores: `[batch_size]` — higher means more plausible.
    fn score(&self, head: Tensor<B, 2>, relation: Tensor<B, 2>, tail: Tensor<B, 2>)
    -> Tensor<B, 1>;
}

/// TransE-style scoring: `score = -||h + r - t||_p`.
///
/// Based on the translational assumption: `h + r ≈ t` for valid facts.
/// The negative L_p norm is used so that higher scores = more plausible.
#[derive(Debug, Clone)]
pub struct TransEScorer {
    /// L_p norm order (typically 1 or 2).
    pub p_norm: f32,
}

impl TransEScorer {
    pub fn new(p_norm: f32) -> Self {
        Self { p_norm }
    }

    /// Default TransE with L2 norm.
    pub fn l2() -> Self {
        Self::new(2.0)
    }
}

impl<B: Backend> Scorer<B> for TransEScorer {
    fn score(
        &self,
        head: Tensor<B, 2>,
        relation: Tensor<B, 2>,
        tail: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        // h + r - t
        let batch_size = head.dims()[0];
        let diff = head + relation - tail;

        if (self.p_norm - 1.0).abs() < f32::EPSILON {
            // L1 norm
            let norm: Tensor<B, 1> = diff.abs().sum_dim(1).reshape([batch_size]); // [batch_size]
            norm.neg()
        } else {
            // L2 norm (default)
            let norm: Tensor<B, 1> = diff
                .powf_scalar(2.0)
                .sum_dim(1)
                .reshape([batch_size])
                .sqrt(); // [batch_size]
            norm.neg()
        }
    }
}

/// DistMult-style scoring: `score = <h, r, t>` (element-wise product summed).
///
/// Based on semantic matching: valid triples have high inner product.
#[derive(Debug, Clone)]
pub struct DistMultScorer;

impl DistMultScorer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DistMultScorer {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Scorer<B> for DistMultScorer {
    fn score(
        &self,
        head: Tensor<B, 2>,
        relation: Tensor<B, 2>,
        tail: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        // Element-wise product h * r * t, summed over hidden dim
        let batch_size = head.dims()[0];
        let result: Tensor<B, 1> = (head * relation * tail).sum_dim(1).reshape([batch_size]);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_transe_scorer() {
        let device = <TestBackend as Backend>::Device::default();
        let batch_size = 4;
        let hidden_dim = 8;

        let head: Tensor<TestBackend, 2> = Tensor::random(
            [batch_size, hidden_dim],
            burn::tensor::Distribution::Default,
            &device,
        );
        let rel: Tensor<TestBackend, 2> = Tensor::random(
            [batch_size, hidden_dim],
            burn::tensor::Distribution::Default,
            &device,
        );
        let tail: Tensor<TestBackend, 2> = Tensor::random(
            [batch_size, hidden_dim],
            burn::tensor::Distribution::Default,
            &device,
        );

        let scorer = TransEScorer::l2();
        let scores = scorer.score(head, rel, tail);
        assert_eq!(scores.dims(), [batch_size]);
        // TransE scores should be negative (negative norm)
    }

    #[test]
    fn test_distmult_scorer() {
        let device = <TestBackend as Backend>::Device::default();
        let batch_size = 4;
        let hidden_dim = 8;

        let head: Tensor<TestBackend, 2> = Tensor::random(
            [batch_size, hidden_dim],
            burn::tensor::Distribution::Default,
            &device,
        );
        let rel: Tensor<TestBackend, 2> = Tensor::random(
            [batch_size, hidden_dim],
            burn::tensor::Distribution::Default,
            &device,
        );
        let tail: Tensor<TestBackend, 2> = Tensor::random(
            [batch_size, hidden_dim],
            burn::tensor::Distribution::Default,
            &device,
        );

        let scorer = DistMultScorer::new();
        let scores = scorer.score(head, rel, tail);
        assert_eq!(scores.dims(), [batch_size]);
    }
}
