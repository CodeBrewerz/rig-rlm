//! Link prediction task head for matching and ranking.
//!
//! Scores (source, relation, destination) tuples and ranks candidates.
//! Key use cases:
//! - `score_match(statement_line, case)` → probability of match
//! - `rank_matches(statement_line)` → top-k case candidates
//! - `rank_receipt_links(receipt)` → top-k transaction candidates

use burn::nn;
use burn::prelude::*;

use crate::model::backbone::NodeEmbeddings;

/// Link predictor: scores (src, dst) pairs using their GNN embeddings.
///
/// Architecture: concatenate [src_emb, dst_emb] → MLP → scalar score.
#[derive(Module, Debug)]
pub struct LinkPredictor<B: Backend> {
    linear1: nn::Linear<B>,
    linear2: nn::Linear<B>,
    output: nn::Linear<B>,
    dropout: nn::Dropout,
}

/// Configuration for the link predictor.
#[derive(Debug, Clone)]
pub struct LinkPredictorConfig {
    /// Hidden dimension of node embeddings from the GNN backbone.
    pub hidden_dim: usize,
    /// Internal MLP dimension.
    pub mlp_dim: usize,
    /// Dropout rate.
    pub dropout: f64,
}

impl LinkPredictorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LinkPredictor<B> {
        // Input: concat of src + dst embeddings = 2 * hidden_dim
        let linear1 = nn::LinearConfig::new(self.hidden_dim * 2, self.mlp_dim).init(device);
        let linear2 = nn::LinearConfig::new(self.mlp_dim, self.mlp_dim / 2).init(device);
        let output = nn::LinearConfig::new(self.mlp_dim / 2, 1).init(device);
        let dropout = nn::DropoutConfig::new(self.dropout).init();

        LinkPredictor {
            linear1,
            linear2,
            output,
            dropout,
        }
    }
}

impl<B: Backend> LinkPredictor<B> {
    /// Score pairs of (source, destination) embeddings.
    ///
    /// # Arguments
    /// - `src_emb`: `[batch_size, hidden_dim]`
    /// - `dst_emb`: `[batch_size, hidden_dim]`
    ///
    /// # Returns
    /// Scores: `[batch_size]` — higher = more likely to be a valid link.
    pub fn score(&self, src_emb: Tensor<B, 2>, dst_emb: Tensor<B, 2>) -> Tensor<B, 1> {
        let batch_size = src_emb.dims()[0];

        // Concat [src_emb, dst_emb]
        let concat = Tensor::cat(vec![src_emb, dst_emb], 1);

        // MLP
        let h = self.linear1.forward(concat);
        let h = burn::tensor::activation::relu(h);
        let h = self.dropout.forward(h);

        let h = self.linear2.forward(h);
        let h = burn::tensor::activation::relu(h);

        let out = self.output.forward(h);
        out.reshape([batch_size])
    }

    /// Score a single source against multiple destination candidates.
    ///
    /// # Arguments
    /// - `src_emb`: `[1, hidden_dim]` — the query entity
    /// - `candidate_embs`: `[num_candidates, hidden_dim]` — potential matches
    ///
    /// # Returns
    /// Scores: `[num_candidates]` — higher = more likely match.
    pub fn rank_candidates(
        &self,
        src_emb: Tensor<B, 2>,
        candidate_embs: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let num_candidates = candidate_embs.dims()[0];

        // Repeat src_emb for each candidate
        let src_repeated = src_emb.repeat_dim(0, num_candidates);

        self.score(src_repeated, candidate_embs)
    }

    /// Score a source-destination pair from NodeEmbeddings by type and index.
    pub fn score_from_embeddings(
        &self,
        embeddings: &NodeEmbeddings<B>,
        src_type: &str,
        src_indices: Tensor<B, 1, Int>,
        dst_type: &str,
        dst_indices: Tensor<B, 1, Int>,
    ) -> Option<Tensor<B, 1>> {
        let src_emb = embeddings.select(src_type, src_indices)?;
        let dst_emb = embeddings.select(dst_type, dst_indices)?;
        Some(self.score(src_emb, dst_emb))
    }
}

/// BPR (Bayesian Personalized Ranking) loss for link prediction.
///
/// Pushes scores of positive links above negative links:
/// L = -log(σ(score_pos - score_neg))
pub fn bpr_loss<B: Backend>(pos_scores: Tensor<B, 1>, neg_scores: Tensor<B, 1>) -> Tensor<B, 1> {
    let diff = pos_scores - neg_scores;
    // log(sigmoid(x)) = x - softplus(x) = x - log(1 + exp(x))
    // For numerical stability, use: -softplus(-x) = -log(1 + exp(-x))
    let sigmoid_diff = burn::tensor::activation::sigmoid(diff);
    // Clamp to avoid log(0)
    let log_sig = sigmoid_diff.clamp_min(1e-7).log();
    log_sig.neg().mean().unsqueeze()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_link_predictor_score() {
        let device = <TestBackend as Backend>::Device::default();
        let config = LinkPredictorConfig {
            hidden_dim: 16,
            mlp_dim: 32,
            dropout: 0.0,
        };
        let predictor = config.init::<TestBackend>(&device);

        let src = Tensor::random([4, 16], burn::tensor::Distribution::Default, &device);
        let dst = Tensor::random([4, 16], burn::tensor::Distribution::Default, &device);

        let scores = predictor.score(src, dst);
        assert_eq!(scores.dims(), [4]);
    }

    #[test]
    fn test_link_predictor_rank_candidates() {
        let device = <TestBackend as Backend>::Device::default();
        let config = LinkPredictorConfig {
            hidden_dim: 8,
            mlp_dim: 16,
            dropout: 0.0,
        };
        let predictor = config.init::<TestBackend>(&device);

        let query = Tensor::random([1, 8], burn::tensor::Distribution::Default, &device);
        let candidates = Tensor::random([10, 8], burn::tensor::Distribution::Default, &device);

        let scores = predictor.rank_candidates(query, candidates);
        assert_eq!(scores.dims(), [10]); // One score per candidate
    }

    #[test]
    fn test_bpr_loss() {
        let device = <TestBackend as Backend>::Device::default();
        let pos = Tensor::<TestBackend, 1>::from_data([2.0, 3.0, 4.0], &device);
        let neg = Tensor::<TestBackend, 1>::from_data([1.0, 1.0, 1.0], &device);

        let loss = bpr_loss(pos, neg);
        let loss_val: f32 = loss.into_data().as_slice::<f32>().unwrap()[0];

        // Since pos > neg, loss should be small
        assert!(loss_val > 0.0);
        assert!(loss_val < 1.0);
    }
}
