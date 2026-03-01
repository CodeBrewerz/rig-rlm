//! Anomaly scoring task head.
//!
//! Reconstruction-based anomaly detection: encodes a node, predicts its
//! neighborhood, and scores divergence from actual neighbors.
//!
//! Key use cases:
//! - Unusual transaction amount vs merchant history
//! - Missing evidence for a transaction (structural anomaly)
//! - Vendor connecting to unexpected GL accounts
//! - Tax mapping gaps

use burn::nn;
use burn::prelude::*;

/// Anomaly scorer: reconstruction-based anomaly detection.
///
/// Given a node embedding, predicts what its neighbor embeddings "should"
/// look like. Large reconstruction error = anomaly.
#[derive(Module, Debug)]
pub struct AnomalyScorer<B: Backend> {
    /// Encoder: compress embedding.
    encoder: nn::Linear<B>,
    /// Decoder: reconstruct expected neighborhood.
    decoder: nn::Linear<B>,
    /// Hidden dim.
    #[module(skip)]
    hidden_dim: usize,
}

/// Configuration for the anomaly scorer.
#[derive(Debug, Clone)]
pub struct AnomalyScorerConfig {
    /// Node embedding dimension.
    pub hidden_dim: usize,
    /// Bottleneck dimension for autoencoder.
    pub bottleneck_dim: usize,
}

impl AnomalyScorerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AnomalyScorer<B> {
        let encoder = nn::LinearConfig::new(self.hidden_dim, self.bottleneck_dim).init(device);
        let decoder = nn::LinearConfig::new(self.bottleneck_dim, self.hidden_dim).init(device);

        AnomalyScorer {
            encoder,
            decoder,
            hidden_dim: self.hidden_dim,
        }
    }
}

impl<B: Backend> AnomalyScorer<B> {
    /// Compute anomaly scores for a batch of node embeddings.
    ///
    /// Score = reconstruction error (L2 norm of difference between
    /// original embedding and reconstructed embedding).
    ///
    /// # Arguments
    /// - `node_emb`: `[batch_size, hidden_dim]`
    ///
    /// # Returns
    /// Anomaly scores: `[batch_size]` — higher = more anomalous.
    pub fn score(&self, node_emb: Tensor<B, 2>) -> Tensor<B, 1> {
        let batch_size = node_emb.dims()[0];

        let encoded = self.encoder.forward(node_emb.clone());
        let encoded = burn::tensor::activation::relu(encoded);
        let decoded = self.decoder.forward(encoded);

        // Reconstruction error: L2 norm per sample
        let diff = node_emb - decoded;
        let sq_err = diff.powf_scalar(2.0);
        sq_err.sum_dim(1).reshape([batch_size]).sqrt()
    }

    /// Reconstruction loss (for training the autoencoder).
    pub fn reconstruction_loss(&self, node_emb: Tensor<B, 2>) -> Tensor<B, 1> {
        let encoded = self.encoder.forward(node_emb.clone());
        let encoded = burn::tensor::activation::relu(encoded);
        let decoded = self.decoder.forward(encoded);

        let diff = node_emb - decoded;
        diff.powf_scalar(2.0).mean().unsqueeze()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_anomaly_scorer() {
        let device = <TestBackend as Backend>::Device::default();
        let config = AnomalyScorerConfig {
            hidden_dim: 16,
            bottleneck_dim: 4,
        };
        let scorer = config.init::<TestBackend>(&device);

        let emb = Tensor::random([5, 16], burn::tensor::Distribution::Default, &device);

        let scores = scorer.score(emb.clone());
        assert_eq!(scores.dims(), [5]);

        let loss = scorer.reconstruction_loss(emb);
        let loss_val: f32 = loss.into_data().as_slice::<f32>().unwrap()[0];
        assert!(loss_val >= 0.0);
    }
}
