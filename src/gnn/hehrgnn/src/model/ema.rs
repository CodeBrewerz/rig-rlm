//! Exponential Moving Average (EMA) for JEPA target encoder.
//!
//! From the JEPA paper (LeCun 2022) and jepa-rs implementation:
//! the target encoder is NOT trained by gradient descent. Instead,
//! it is an EMA copy of the context encoder:
//!
//!   target_w = momentum * target_w + (1 - momentum) * context_w
//!
//! This provides a stable prediction target that changes slowly,
//! preventing representation collapse.
//!
//! Supports cosine momentum schedule (V-JEPA 2): momentum ramps
//! from base (0.996) to 1.0 over training, making the target
//! increasingly stable.

use std::collections::HashMap;

/// EMA weight updater for target encoder.
#[derive(Debug, Clone)]
pub struct Ema {
    /// Base momentum (e.g., 0.996).
    pub base_momentum: f64,
    /// Optional cosine schedule total steps.
    /// If set, momentum ramps from base to 1.0 over total_steps.
    pub cosine_total_steps: Option<usize>,
}

impl Ema {
    /// Create a constant-momentum EMA.
    pub fn new(momentum: f64) -> Self {
        Self {
            base_momentum: momentum.clamp(0.0, 1.0),
            cosine_total_steps: None,
        }
    }

    /// Create an EMA with cosine momentum schedule (V-JEPA 2 style).
    ///
    /// Momentum ramps from `base_momentum` to 1.0 via:
    ///   m(t) = 1.0 - (1.0 - base) * (1 + cos(π * t / T)) / 2
    pub fn with_cosine_schedule(base_momentum: f64, total_steps: usize) -> Self {
        Self {
            base_momentum: base_momentum.clamp(0.0, 1.0),
            cosine_total_steps: Some(total_steps.max(1)),
        }
    }

    /// Get the effective momentum at a given training step.
    pub fn get_momentum(&self, step: usize) -> f64 {
        match self.cosine_total_steps {
            Some(total) => {
                let t = (step as f64 / total as f64).min(1.0);
                1.0 - (1.0 - self.base_momentum) * (1.0 + (std::f64::consts::PI * t).cos()) / 2.0
            }
            None => self.base_momentum,
        }
    }

    /// Perform EMA update on scalar values (for testing).
    pub fn step(&self, target: f64, online: f64, step: usize) -> f64 {
        let m = self.get_momentum(step);
        m * target + (1.0 - m) * online
    }

    /// Perform EMA update on flat weight vectors.
    ///
    /// target_w = momentum * target_w + (1 - momentum) * online_w
    pub fn update_weights(&self, target: &mut [f32], online: &[f32], step: usize) {
        let m = self.get_momentum(step) as f32;
        let one_minus_m = 1.0 - m;
        for (t, &o) in target.iter_mut().zip(online.iter()) {
            *t = m * *t + one_minus_m * o;
        }
    }

    /// Perform EMA update on a full set of model weights.
    ///
    /// Takes two sets of weights indexed by layer index → flat Vec<f32>.
    pub fn update_model_weights(
        &self,
        target_weights: &mut Vec<Vec<f32>>,
        online_weights: &[Vec<f32>],
        step: usize,
    ) {
        let m = self.get_momentum(step) as f32;
        let one_minus_m = 1.0 - m;
        for (target_layer, online_layer) in target_weights.iter_mut().zip(online_weights.iter()) {
            for (t, &o) in target_layer.iter_mut().zip(online_layer.iter()) {
                *t = m * *t + one_minus_m * o;
            }
        }
    }
}

/// Stores EMA target weights for use as a stable prediction target.
///
/// In the context of hehrgnn's perturbation-based training:
/// - Online weights = model.get_input_weight(i) — updated by SPSA
/// - Target weights = EMA copy — updated by momentum averaging
/// - Target embeddings = forward pass with target weights
#[derive(Debug, Clone)]
pub struct EmaTargetEncoder {
    /// EMA updater configuration.
    pub ema: Ema,
    /// Target weights: one Vec<f32> per input weight layer.
    pub target_weights: Vec<Vec<f32>>,
    /// Current EMA step counter.
    pub step: usize,
}

impl EmaTargetEncoder {
    /// Initialize from a model's current weights (deep copy).
    pub fn from_model_weights(ema: Ema, weights: Vec<Vec<f32>>) -> Self {
        Self {
            ema,
            target_weights: weights,
            step: 0,
        }
    }

    /// Update target weights from online (context) model weights.
    pub fn update(&mut self, online_weights: &[Vec<f32>]) {
        self.ema
            .update_model_weights(&mut self.target_weights, online_weights, self.step);
        self.step += 1;
    }

    /// Get current momentum value.
    pub fn current_momentum(&self) -> f64 {
        self.ema.get_momentum(self.step)
    }

    /// Apply target weights to a model temporarily for forward pass.
    ///
    /// Returns the original weights so they can be restored.
    pub fn apply_to_model<B: burn::prelude::Backend, M: crate::model::trainer::JepaTrainable<B>>(
        &self,
        model: &mut M,
    ) -> Vec<Vec<f32>> {
        let mut original_weights = Vec::with_capacity(self.target_weights.len());
        for (i, target_w) in self.target_weights.iter().enumerate() {
            if i >= model.num_input_weights() {
                break;
            }
            // Save original
            let orig = model.get_input_weight(i);
            let orig_data: Vec<f32> = orig.clone().into_data().as_slice::<f32>().unwrap().to_vec();
            let dims = orig.dims();
            original_weights.push(orig_data);

            // Set target weights
            let device = orig.device();
            let target_tensor =
                burn::tensor::Tensor::<B, 1>::from_data(target_w.as_slice(), &device)
                    .reshape([dims[0], dims[1]]);
            model.set_input_weight(i, target_tensor);
        }
        original_weights
    }

    /// Restore original weights after a target forward pass.
    pub fn restore_model<B: burn::prelude::Backend, M: crate::model::trainer::JepaTrainable<B>>(
        &self,
        model: &mut M,
        original_weights: Vec<Vec<f32>>,
    ) {
        for (i, orig_w) in original_weights.iter().enumerate() {
            if i >= model.num_input_weights() {
                break;
            }
            let w = model.get_input_weight(i);
            let dims = w.dims();
            let device = w.device();
            let orig_tensor = burn::tensor::Tensor::<B, 1>::from_data(orig_w.as_slice(), &device)
                .reshape([dims[0], dims[1]]);
            model.set_input_weight(i, orig_tensor);
        }
    }

    /// Extract current model weights as Vec<Vec<f32>>.
    pub fn extract_model_weights<
        B: burn::prelude::Backend,
        M: crate::model::trainer::JepaTrainable<B>,
    >(
        model: &M,
    ) -> Vec<Vec<f32>> {
        let mut weights = Vec::with_capacity(model.num_input_weights());
        for i in 0..model.num_input_weights() {
            let w = model.get_input_weight(i);
            let data: Vec<f32> = w.into_data().as_slice::<f32>().unwrap().to_vec();
            weights.push(data);
        }
        weights
    }
}

/// Warmup + cosine decay learning rate schedule (Gap #2).
///
/// From jepa-rs `jepa-train/schedule.rs` and I-JEPA/V-JEPA papers:
/// 1. Linear warmup from ~0 to peak_lr over warmup_epochs
/// 2. Cosine decay from peak_lr to end_lr over remaining epochs
#[derive(Debug, Clone)]
pub struct WarmupCosineSchedule {
    /// Peak learning rate (at end of warmup).
    pub peak_lr: f64,
    /// Final learning rate (at end of training).
    pub end_lr: f64,
    /// Number of warmup epochs.
    pub warmup_epochs: usize,
    /// Total number of epochs.
    pub total_epochs: usize,
}

impl WarmupCosineSchedule {
    /// Create a standard JEPA LR schedule.
    pub fn new(peak_lr: f64, warmup_epochs: usize, total_epochs: usize) -> Self {
        Self {
            peak_lr,
            end_lr: peak_lr * 0.001, // 1000x decay
            warmup_epochs,
            total_epochs: total_epochs.max(1),
        }
    }

    /// Get learning rate at a given epoch.
    pub fn get_lr(&self, epoch: usize) -> f64 {
        if epoch < self.warmup_epochs {
            // Linear warmup
            let progress = epoch as f64 / self.warmup_epochs.max(1) as f64;
            self.end_lr + (self.peak_lr - self.end_lr) * progress
        } else {
            // Cosine decay
            let decay_epochs = self.total_epochs.saturating_sub(self.warmup_epochs);
            if decay_epochs == 0 {
                return self.end_lr;
            }
            let progress = (epoch - self.warmup_epochs) as f64 / decay_epochs as f64;
            let progress = progress.min(1.0);
            self.end_lr
                + (self.peak_lr - self.end_lr) * (1.0 + (progress * std::f64::consts::PI).cos())
                    / 2.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_constant_momentum() {
        let ema = Ema::new(0.996);
        assert!((ema.get_momentum(0) - 0.996).abs() < 1e-6);
        assert!((ema.get_momentum(100) - 0.996).abs() < 1e-6);
    }

    #[test]
    fn test_ema_cosine_schedule() {
        let ema = Ema::with_cosine_schedule(0.996, 1000);
        // At start: momentum ≈ base (0.996)
        assert!((ema.get_momentum(0) - 0.996).abs() < 1e-3);
        // At end: momentum → 1.0
        assert!((ema.get_momentum(1000) - 1.0).abs() < 1e-3);
        // Midpoint should be between base and 1.0
        let mid = ema.get_momentum(500);
        assert!(mid > 0.996 && mid < 1.0, "Mid momentum = {}", mid);
    }

    #[test]
    fn test_ema_step_converges() {
        let ema = Ema::new(0.99);
        let mut target = 0.0;
        for step in 0..500 {
            target = ema.step(target, 1.0, step);
        }
        // Should converge toward online value (1.0)
        assert!(target > 0.99, "EMA should converge, got {}", target);
    }

    #[test]
    fn test_ema_weight_update() {
        let ema = Ema::new(0.5);
        let mut target = vec![0.0, 0.0, 0.0];
        let online = vec![1.0, 2.0, 3.0];
        ema.update_weights(&mut target, &online, 0);
        assert!((target[0] - 0.5).abs() < 1e-6);
        assert!((target[1] - 1.0).abs() < 1e-6);
        assert!((target[2] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_ema_momentum_1_preserves_target() {
        let ema = Ema::new(1.0);
        let result = ema.step(5.0, 100.0, 0);
        assert!((result - 5.0).abs() < 1e-6, "m=1 should keep target");
    }

    #[test]
    fn test_ema_momentum_0_copies_online() {
        let ema = Ema::new(0.0);
        let result = ema.step(5.0, 100.0, 0);
        assert!((result - 100.0).abs() < 1e-6, "m=0 should copy online");
    }

    #[test]
    fn test_warmup_cosine_lr() {
        let schedule = WarmupCosineSchedule::new(0.01, 5, 40);
        // At start: LR ≈ end_lr (near 0)
        let lr0 = schedule.get_lr(0);
        assert!(lr0 < 0.002, "Start LR should be small, got {}", lr0);
        // At warmup end: LR ≈ peak_lr
        let lr5 = schedule.get_lr(5);
        assert!(
            (lr5 - 0.01).abs() < 0.001,
            "LR at warmup end should be ~peak, got {}",
            lr5
        );
        // At end: LR ≈ end_lr
        let lr40 = schedule.get_lr(40);
        assert!(lr40 < 0.001, "End LR should be small, got {}", lr40);
        // Monotonic during warmup
        for e in 1..5 {
            assert!(
                schedule.get_lr(e) >= schedule.get_lr(e - 1) - 1e-10,
                "LR should increase during warmup at epoch {}",
                e
            );
        }
    }

    #[test]
    fn test_ema_target_encoder() {
        let weights = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0]];
        let ema = Ema::new(0.5);
        let mut target = EmaTargetEncoder::from_model_weights(ema, weights);

        let online = vec![vec![10.0, 20.0, 30.0], vec![40.0, 50.0]];
        target.update(&online);

        // After one update with m=0.5:
        // target = 0.5 * [1,2,3] + 0.5 * [10,20,30] = [5.5, 11.0, 16.5]
        assert!((target.target_weights[0][0] - 5.5).abs() < 1e-4);
        assert!((target.target_weights[0][1] - 11.0).abs() < 1e-4);
        assert!((target.target_weights[1][0] - 22.0).abs() < 1e-4);
        assert_eq!(target.step, 1);
    }
}
