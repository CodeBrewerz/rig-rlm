//! Leaf distributions for Probabilistic Circuit input nodes.
//!
//! Each distribution computes log P(x | params) for a single variable.
//! All computation is in log-space for numerical stability.

use serde::{Deserialize, Serialize};

/// A leaf distribution for an input node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Distribution {
    /// Categorical distribution over k classes.
    /// `log_probs[i]` = log P(X = i).
    Categorical { log_probs: Vec<f64> },

    /// Gaussian distribution with mean and log-variance.
    Gaussian { mean: f64, log_var: f64 },

    /// Bernoulli distribution parametrized by logit.
    /// P(X=1) = sigmoid(logit).
    Bernoulli { logit: f64 },

    /// Indicator: P(X=val) = 1, P(X≠val) = 0.
    Indicator { value: usize },
}

impl Distribution {
    /// Create a uniform categorical over `k` categories.
    pub fn uniform_categorical(k: usize) -> Self {
        let log_p = -(k as f64).ln();
        Distribution::Categorical {
            log_probs: vec![log_p; k],
        }
    }

    /// Create a categorical from raw (unnormalized) probabilities.
    pub fn categorical_from_probs(probs: &[f64]) -> Self {
        let sum: f64 = probs.iter().sum();
        let log_probs: Vec<f64> = probs.iter().map(|p| (p / sum).ln()).collect();
        Distribution::Categorical { log_probs }
    }

    /// Create a standard Gaussian.
    pub fn standard_gaussian() -> Self {
        Distribution::Gaussian {
            mean: 0.0,
            log_var: 0.0,
        }
    }

    /// Create a Bernoulli with probability p.
    pub fn bernoulli(p: f64) -> Self {
        let p = p.clamp(1e-10, 1.0 - 1e-10);
        Distribution::Bernoulli {
            logit: (p / (1.0 - p)).ln(),
        }
    }

    /// Number of categories (for categorical distributions).
    pub fn num_categories(&self) -> usize {
        match self {
            Distribution::Categorical { log_probs } => log_probs.len(),
            Distribution::Bernoulli { .. } => 2,
            Distribution::Indicator { .. } => 1,
            Distribution::Gaussian { .. } => 0, // continuous
        }
    }

    /// Compute log P(X = x) under this distribution.
    ///
    /// For continuous distributions, `x` is discretized to the nearest integer
    /// bin or treated as a real value depending on context.
    pub fn log_prob(&self, x: usize) -> f64 {
        match self {
            Distribution::Categorical { log_probs } => {
                if x < log_probs.len() {
                    log_probs[x]
                } else {
                    f64::NEG_INFINITY
                }
            }
            Distribution::Bernoulli { logit } => {
                if x == 1 {
                    // log sigmoid(logit) = -log(1 + exp(-logit))
                    -softplus(-*logit)
                } else {
                    // log(1 - sigmoid(logit)) = -log(1 + exp(logit))
                    -softplus(*logit)
                }
            }
            Distribution::Indicator { value } => {
                if x == *value {
                    0.0
                } else {
                    f64::NEG_INFINITY
                }
            }
            Distribution::Gaussian { mean, log_var } => {
                // Treat x as a real value (cast from usize for discrete case)
                let xf = x as f64;
                let var = log_var.exp();
                -0.5 * ((xf - mean).powi(2) / var + *log_var + (2.0 * std::f64::consts::PI).ln())
            }
        }
    }

    /// Compute log P(X = x) for continuous-valued x.
    pub fn log_prob_f64(&self, x: f64) -> f64 {
        match self {
            Distribution::Categorical { log_probs } => {
                let idx = x.round() as usize;
                if idx < log_probs.len() {
                    log_probs[idx]
                } else {
                    f64::NEG_INFINITY
                }
            }
            Distribution::Gaussian { mean, log_var } => {
                let var = log_var.exp();
                -0.5 * ((x - mean).powi(2) / var + *log_var + (2.0 * std::f64::consts::PI).ln())
            }
            Distribution::Bernoulli { logit } => {
                if x > 0.5 {
                    -softplus(-*logit)
                } else {
                    -softplus(*logit)
                }
            }
            Distribution::Indicator { value } => {
                if (x - *value as f64).abs() < 1e-6 {
                    0.0
                } else {
                    f64::NEG_INFINITY
                }
            }
        }
    }

    /// Number of learnable parameters.
    pub fn num_params(&self) -> usize {
        match self {
            Distribution::Categorical { log_probs } => log_probs.len(),
            Distribution::Gaussian { .. } => 2,
            Distribution::Bernoulli { .. } => 1,
            Distribution::Indicator { .. } => 0,
        }
    }
}

/// Numerically stable log(1 + exp(x)).
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_categorical_sums_to_one() {
        let d = Distribution::uniform_categorical(4);
        let total: f64 = (0..4).map(|x| d.log_prob(x).exp()).sum();
        assert!((total - 1.0).abs() < 1e-10, "Sum = {}", total);
    }

    #[test]
    fn test_categorical_from_probs() {
        let d = Distribution::categorical_from_probs(&[0.3, 0.7]);
        let p0 = d.log_prob(0).exp();
        let p1 = d.log_prob(1).exp();
        assert!((p0 - 0.3).abs() < 1e-10);
        assert!((p1 - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_bernoulli() {
        let d = Distribution::bernoulli(0.6);
        let p0 = d.log_prob(0).exp();
        let p1 = d.log_prob(1).exp();
        assert!((p0 - 0.4).abs() < 1e-6, "P(0) = {}", p0);
        assert!((p1 - 0.6).abs() < 1e-6, "P(1) = {}", p1);
        assert!((p0 + p1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian() {
        let d = Distribution::standard_gaussian();
        let lp = d.log_prob_f64(0.0);
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!((lp - expected).abs() < 1e-10);
    }

    #[test]
    fn test_indicator() {
        let d = Distribution::Indicator { value: 3 };
        assert_eq!(d.log_prob(3), 0.0);
        assert_eq!(d.log_prob(0), f64::NEG_INFINITY);
    }
}
