//! Router Projectors for Memory Sparse Attention.
//!
//! Implements the Router K and Router Q projectors (Eq. 1 of MSA paper):
//!   K^R_{i,h} = H_i · W^h_{KR}
//!   Q^R_{q,h} = H_q · W^h_{QR}
//!
//! These specialized projectors generate routing-specific representations
//! that are separate from the standard K/Q/V projections used in attention.
//! The routing projections are lower-dimensional latent spaces optimized
//! specifically for document-level retrieval relevance scoring.

use burn::nn;
use burn::prelude::*;

/// Configuration for the Router Projector.
#[derive(Debug, Clone)]
pub struct RouterProjectorConfig {
    /// Input hidden dimension (model hidden size).
    pub hidden_dim: usize,
    /// Router projection dimension (can differ from hidden_dim for efficiency).
    pub router_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
}

/// Router Projector: produces routing keys (K^R) or routing queries (Q^R).
///
/// In the MSA architecture, each attention head has a separate routing
/// projection space. The router projector is a simple linear layer that
/// maps hidden states to routing representations.
///
/// Paper §3.2.1: "We introduce a Router K Projector, parameterized by W^h_{KR},
/// to generate a specialized routing key matrix K^R_{i,h}"
#[derive(Module, Debug)]
pub struct RouterProjector<B: Backend> {
    /// Linear projection: hidden_dim → router_dim * num_heads
    pub proj: nn::Linear<B>,
    /// Number of attention heads
    #[module(skip)]
    pub num_heads: usize,
    /// Dimension per head in routing space
    #[module(skip)]
    pub head_dim: usize,
}

impl<B: Backend> RouterProjector<B> {
    /// Create a new Router Projector.
    ///
    /// # Arguments
    /// * `config` - Router configuration
    /// * `device` - Computation device
    pub fn new(config: &RouterProjectorConfig, device: &B::Device) -> Self {
        let total_dim = config.router_dim * config.num_heads;
        let proj = nn::LinearConfig::new(config.hidden_dim, total_dim)
            .with_bias(false)
            .init(device);

        Self {
            proj,
            num_heads: config.num_heads,
            head_dim: config.router_dim,
        }
    }

    /// Project hidden states into routing space.
    ///
    /// # Arguments
    /// * `hidden` - Input hidden states [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
    ///
    /// # Returns
    /// * Routing representations [batch, seq_len, num_heads, head_dim] or [seq_len, num_heads, head_dim]
    pub fn forward(&self, hidden: Tensor<B, 2>) -> Tensor<B, 3> {
        let [seq_len, _hidden_dim] = hidden.dims();
        let projected = self.proj.forward(hidden); // [seq_len, num_heads * head_dim]
        projected.reshape([seq_len, self.num_heads, self.head_dim])
    }

    /// Project hidden states and return flat representation (for pooling).
    ///
    /// # Returns
    /// * [seq_len, num_heads * head_dim]
    pub fn forward_flat(&self, hidden: Tensor<B, 2>) -> Tensor<B, 2> {
        self.proj.forward(hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn test_router_projector_shapes() {
        let device = <B as Backend>::Device::default();
        let config = RouterProjectorConfig {
            hidden_dim: 64,
            router_dim: 16,
            num_heads: 4,
        };
        let router = RouterProjector::<B>::new(&config, &device);

        // Simulate 10 tokens with 64-dim hidden states
        let hidden = Tensor::<B, 2>::random(
            [10, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let out = router.forward(hidden);
        assert_eq!(out.dims(), [10, 4, 16], "Output shape should be [seq, heads, head_dim]");

        println!("✅ RouterProjector: [10, 64] → [10, 4, 16]");
    }

    #[test]
    fn test_router_projector_flat() {
        let device = <B as Backend>::Device::default();
        let config = RouterProjectorConfig {
            hidden_dim: 128,
            router_dim: 32,
            num_heads: 8,
        };
        let router = RouterProjector::<B>::new(&config, &device);

        let hidden = Tensor::<B, 2>::random(
            [20, 128],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let out = router.forward_flat(hidden);
        assert_eq!(out.dims(), [20, 256], "Flat output: [seq, num_heads * head_dim]");

        println!("✅ RouterProjector flat: [20, 128] → [20, 256]");
    }
}
