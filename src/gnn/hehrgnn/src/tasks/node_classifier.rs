//! Node classification task head.
//!
//! MLP on node embeddings → class logits.
//! Key use cases:
//! - `predict_category(tx)` → transaction category + confidence
//! - `predict_tax_code(case)` → tax code + confidence
//! - `predict_case_status(case)` → needs_review / auto_book / needs_evidence

use burn::nn;
use burn::prelude::*;

/// Node classifier: maps node embeddings to class predictions.
#[derive(Module, Debug)]
pub struct NodeClassifier<B: Backend> {
    linear1: nn::Linear<B>,
    linear2: nn::Linear<B>,
    output: nn::Linear<B>,
    dropout: nn::Dropout,
}

/// Configuration for the node classifier.
#[derive(Debug, Clone)]
pub struct NodeClassifierConfig {
    /// Hidden dimension of node embeddings.
    pub hidden_dim: usize,
    /// Internal MLP dimension.
    pub mlp_dim: usize,
    /// Number of output classes.
    pub num_classes: usize,
    /// Dropout rate.
    pub dropout: f64,
}

impl NodeClassifierConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> NodeClassifier<B> {
        let linear1 = nn::LinearConfig::new(self.hidden_dim, self.mlp_dim).init(device);
        let linear2 = nn::LinearConfig::new(self.mlp_dim, self.mlp_dim / 2).init(device);
        let output = nn::LinearConfig::new(self.mlp_dim / 2, self.num_classes).init(device);
        let dropout = nn::DropoutConfig::new(self.dropout).init();

        NodeClassifier {
            linear1,
            linear2,
            output,
            dropout,
        }
    }
}

impl<B: Backend> NodeClassifier<B> {
    /// Classify nodes given their embeddings.
    ///
    /// # Arguments
    /// - `node_emb`: `[batch_size, hidden_dim]`
    ///
    /// # Returns
    /// Logits: `[batch_size, num_classes]`
    pub fn forward(&self, node_emb: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.linear1.forward(node_emb);
        let h = burn::tensor::activation::relu(h);
        let h = self.dropout.forward(h);

        let h = self.linear2.forward(h);
        let h = burn::tensor::activation::relu(h);

        self.output.forward(h)
    }

    /// Get predicted class indices and confidence scores.
    ///
    /// # Returns
    /// `(predicted_classes, confidences)`:
    /// - `predicted_classes`: `[batch_size]` of class indices
    /// - `confidences`: `[batch_size]` of softmax probabilities for predicted class
    pub fn predict(&self, node_emb: Tensor<B, 2>) -> (Tensor<B, 1, Int>, Tensor<B, 1>) {
        let logits = self.forward(node_emb);
        let probs = burn::tensor::activation::softmax(logits, 1);

        let batch_size = probs.dims()[0];
        let predicted = probs.clone().argmax(1).reshape([batch_size]);
        let confidence = probs.max_dim(1).reshape([batch_size]);

        (predicted, confidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_node_classifier() {
        let device = <TestBackend as Backend>::Device::default();
        let config = NodeClassifierConfig {
            hidden_dim: 16,
            mlp_dim: 32,
            num_classes: 5,
            dropout: 0.0,
        };
        let classifier = config.init::<TestBackend>(&device);

        let emb = Tensor::random([8, 16], burn::tensor::Distribution::Default, &device);

        let logits = classifier.forward(emb.clone());
        assert_eq!(logits.dims(), [8, 5]);

        let (classes, confidences) = classifier.predict(emb);
        assert_eq!(classes.dims(), [8]);
        assert_eq!(confidences.dims(), [8]);
    }
}
