//! Auxiliary Routing Contrastive Loss for MSA.
//!
//! MSA Paper §3.3.1, Eq. 5: Supervised contrastive loss for router training.
//!
//! For a query q with positive documents P and negatives N = D \ P:
//!   L_aux = -(1/|P|) Σ_i log( exp(s⁺_i/τ) / (exp(s⁺_i/τ) + Σ_j exp(s⁻_{i,j}/τ)) )
//!
//! This explicitly enforces separation between relevant and irrelevant
//! document chunks in the latent routing space.

use burn::prelude::*;

/// Compute the auxiliary routing contrastive loss.
///
/// # Arguments
/// * `positive_scores` - Relevance scores for positive (relevant) document pairs [num_positive]
/// * `negative_scores` - Relevance scores for negative document pairs [num_positive, num_negative]
///   Each row i contains the negative scores paired with positive i.
/// * `temperature` - Temperature parameter τ
///
/// # Returns
/// * Scalar loss value
pub fn aux_routing_loss<B: Backend>(
    positive_scores: Tensor<B, 1>,
    negative_scores: Tensor<B, 2>,
    temperature: f32,
) -> Tensor<B, 1> {
    let [num_positive] = positive_scores.dims();
    let [_np, num_negative] = negative_scores.dims();

    // Scale by temperature
    let pos_scaled = positive_scores / temperature; // [num_positive]
    let neg_scaled = negative_scores / temperature; // [num_positive, num_negative]

    // Compute exp
    let pos_exp = pos_scaled.clone().exp(); // [num_positive]
    let neg_exp = neg_scaled.exp(); // [num_positive, num_negative]

    // Sum of negative exponentials per positive
    let neg_sum = neg_exp.sum_dim(1).reshape([num_positive]); // [num_positive]

    // Denominator: exp(s+/τ) + Σ exp(s-/τ)
    let denom = pos_exp.clone() + neg_sum;

    // Loss: -mean( log( exp(s+/τ) / denom ) )
    //      = -mean( s+/τ - log(denom) )
    let log_ratio = pos_scaled - denom.log();
    let loss = -log_ratio.mean(); // scalar

    // Return as 1D tensor for compatibility
    loss.unsqueeze()
}

/// Compute the combined MSA training loss.
///
/// Paper §3.3.1:
///   Warmup:  L = 0.1 * L_LLM + L_aux
///   Main:    L = L_LLM + 0.1 * L_aux
///
/// # Arguments
/// * `llm_loss` - Language model generation loss (scalar)
/// * `aux_loss` - Auxiliary routing loss (scalar)
/// * `is_warmup` - Whether in warmup phase
///
/// # Returns
/// * Combined loss (scalar tensor)
pub fn combined_loss<B: Backend>(
    llm_loss: Tensor<B, 1>,
    aux_loss: Tensor<B, 1>,
    is_warmup: bool,
) -> Tensor<B, 1> {
    if is_warmup {
        // Warmup: L = 0.1 * L_LLM + L_aux
        llm_loss * 0.1 + aux_loss
    } else {
        // Main: L = L_LLM + 0.1 * L_aux
        llm_loss + aux_loss * 0.1
    }
}

/// Training schedule configuration.
#[derive(Debug, Clone)]
pub struct TrainingSchedule {
    /// Warmup learning rate (1e-4 in paper)
    pub warmup_lr: f32,
    /// Main phase learning rate (6e-6 in paper)
    pub main_lr: f32,
    /// Number of warmup steps
    pub warmup_steps: usize,
    /// Total training steps
    pub total_steps: usize,
    /// Temperature for contrastive loss
    pub temperature: f32,
}

impl Default for TrainingSchedule {
    fn default() -> Self {
        Self {
            warmup_lr: 1e-4,
            main_lr: 6e-6,
            warmup_steps: 5000,
            total_steps: 100000,
            temperature: 0.07,
        }
    }
}

impl TrainingSchedule {
    /// Check if the current step is in the warmup phase.
    pub fn is_warmup(&self, step: usize) -> bool {
        step < self.warmup_steps
    }

    /// Get the learning rate for the current step.
    pub fn lr_at(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            self.warmup_lr
        } else {
            self.main_lr
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn test_aux_loss_basic() {
        let device = <B as Backend>::Device::default();

        // 2 positive documents, 3 negatives each
        let pos = Tensor::<B, 1>::from_data([0.9f32, 0.8], &device);
        let neg = Tensor::<B, 2>::from_data([[0.1f32, 0.2, 0.15], [0.05, 0.1, 0.3]], &device);

        let loss = aux_routing_loss(pos, neg, 0.07);
        let loss_val: f32 = loss.into_data().as_slice::<f32>().unwrap()[0];

        assert!(loss_val.is_finite(), "Loss should be finite");
        assert!(loss_val >= 0.0, "Loss should be non-negative (got {})", loss_val);
        println!("✅ Aux routing loss: {:.4}", loss_val);
    }

    #[test]
    fn test_aux_loss_perfect_separation() {
        let device = <B as Backend>::Device::default();

        // Perfect separation: high positive, low negative
        let pos = Tensor::<B, 1>::from_data([5.0f32, 5.0], &device);
        let neg = Tensor::<B, 2>::from_data([[-5.0f32, -5.0], [-5.0, -5.0]], &device);

        let loss = aux_routing_loss(pos, neg, 0.07);
        let loss_val: f32 = loss.into_data().as_slice::<f32>().unwrap()[0];

        // With perfect separation, loss should be very small
        assert!(loss_val < 0.1, "Perfect separation should give near-zero loss, got {}", loss_val);
        println!("✅ Perfect separation loss: {:.6}", loss_val);
    }

    #[test]
    fn test_combined_loss_warmup_vs_main() {
        let device = <B as Backend>::Device::default();

        let llm = Tensor::<B, 1>::from_data([2.0f32], &device);
        let aux = Tensor::<B, 1>::from_data([1.0f32], &device);

        let warmup = combined_loss(llm.clone(), aux.clone(), true);
        let main = combined_loss(llm, aux, false);

        let w: f32 = warmup.into_data().as_slice::<f32>().unwrap()[0];
        let m: f32 = main.into_data().as_slice::<f32>().unwrap()[0];

        // Warmup: 0.1*2 + 1 = 1.2
        assert!((w - 1.2).abs() < 1e-5, "Warmup loss should be 1.2, got {}", w);
        // Main: 2 + 0.1*1 = 2.1
        assert!((m - 2.1).abs() < 1e-5, "Main loss should be 2.1, got {}", m);

        println!("✅ Combined loss: warmup={:.1}, main={:.1}", w, m);
    }

    #[test]
    fn test_training_schedule() {
        let schedule = TrainingSchedule::default();
        assert!(schedule.is_warmup(0));
        assert!(schedule.is_warmup(4999));
        assert!(!schedule.is_warmup(5000));
        assert_eq!(schedule.lr_at(0), 1e-4);
        assert_eq!(schedule.lr_at(5000), 6e-6);
        println!("✅ Training schedule: warmup/main phase transitions correct");
    }
}
