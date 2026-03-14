//! DroPE: Dropping Positional Embeddings after training.
//!
//! From paper: "Extending the Context of Pretrained LLMs by Dropping
//! Their Positional Embeddings" (arXiv:2512.12167, Gelberg et al.)
//!
//! Key insight: Positional embeddings (PE) accelerate training convergence
//! but limit generalization to unseen sequence lengths. DroPE trains with
//! sinusoidal PE, then drops them for inference after a short recalibration
//! phase — enabling zero-shot generalization to any document size.
//!
//! For document ViT:
//! - Phase 1 (train): Add 2D sinusoidal PE to patch embeddings
//! - Phase 2 (recalibrate): Fine-tune briefly without PE to adapt
//! - Phase 3 (inference): Run without PE → works on any document size
//!
//! This replaces RoPE with a simpler, more generalizable approach.

/// DroPE phase during the document ViT lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DroPEPhase {
    /// Phase 1: Training with positional embeddings active.
    TrainWithPE,
    /// Phase 2: Recalibration — PE removed, short adaptation phase.
    Recalibrate,
    /// Phase 3: Inference — PE dropped, full generalization.
    Inference,
}

/// 2D Sinusoidal Position Encoding for document patches.
///
/// Used during TrainWithPE phase only. Generates position vectors
/// encoding (row, col) spatial position of each patch in the document grid.
///
/// Based on the standard sinusoidal encoding from Vaswani et al. (2017),
/// extended to 2D for document layout understanding.
#[derive(Debug, Clone)]
pub struct Sinusoidal2DPositionEncoding {
    /// Precomputed position encodings: [max_patches, embed_dim].
    pub encodings: Vec<Vec<f32>>,
    /// Grid height (rows of patches).
    pub grid_h: usize,
    /// Grid width (columns of patches).
    pub grid_w: usize,
    /// Embedding dimension.
    pub embed_dim: usize,
}

impl Sinusoidal2DPositionEncoding {
    /// Create 2D sinusoidal position encodings.
    ///
    /// Half the dimensions encode the row position, half encode column position.
    pub fn new(grid_h: usize, grid_w: usize, embed_dim: usize) -> Self {
        let half_dim = embed_dim / 2;
        let num_patches = grid_h * grid_w;
        let mut encodings = Vec::with_capacity(num_patches);

        for row in 0..grid_h {
            for col in 0..grid_w {
                let mut enc = Vec::with_capacity(embed_dim);

                // First half: encode row position
                for i in 0..half_dim {
                    let freq = 1.0 / (10000.0f32).powf(2.0 * i as f32 / half_dim as f32);
                    if i % 2 == 0 {
                        enc.push((row as f32 * freq).sin());
                    } else {
                        enc.push((row as f32 * freq).cos());
                    }
                }

                // Second half: encode column position
                for i in 0..half_dim {
                    let freq = 1.0 / (10000.0f32).powf(2.0 * i as f32 / half_dim as f32);
                    if i % 2 == 0 {
                        enc.push((col as f32 * freq).sin());
                    } else {
                        enc.push((col as f32 * freq).cos());
                    }
                }

                // Handle odd embed_dim
                while enc.len() < embed_dim {
                    enc.push(0.0);
                }

                encodings.push(enc);
            }
        }

        Self {
            encodings,
            grid_h,
            grid_w,
            embed_dim,
        }
    }

    /// Apply position encoding to patch embeddings.
    ///
    /// Adds the sinusoidal encoding vectors element-wise.
    pub fn apply(&self, embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
        embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| {
                if i < self.encodings.len() {
                    emb.iter()
                        .zip(self.encodings[i].iter())
                        .map(|(&e, &p)| e + p)
                        .collect()
                } else {
                    emb.clone()
                }
            })
            .collect()
    }
}

/// DroPE controller for the document ViT.
///
/// Manages the lifecycle: TrainWithPE → Recalibrate → Inference.
///
/// Algorithm (from paper):
/// 1. Train normally with 2D sinusoidal PE for T_train steps
/// 2. Remove PE, recalibrate for T_recal steps (typically 1-5% of training)
/// 3. Inference: model works on any document size without PE
#[derive(Debug, Clone)]
pub struct DroPEController {
    /// Current phase.
    pub phase: DroPEPhase,
    /// Position encoding (used only during TrainWithPE).
    pub pe: Option<Sinusoidal2DPositionEncoding>,
    /// Total training steps before dropping PE.
    pub train_steps: usize,
    /// Recalibration steps after dropping PE.
    pub recal_steps: usize,
    /// Current step counter.
    pub current_step: usize,
}

impl DroPEController {
    /// Create a new DroPE controller.
    pub fn new(
        grid_h: usize,
        grid_w: usize,
        embed_dim: usize,
        train_steps: usize,
        recal_fraction: f32,
    ) -> Self {
        let pe = Sinusoidal2DPositionEncoding::new(grid_h, grid_w, embed_dim);
        let recal_steps = (train_steps as f32 * recal_fraction) as usize;
        Self {
            phase: DroPEPhase::TrainWithPE,
            pe: Some(pe),
            train_steps,
            recal_steps,
            current_step: 0,
        }
    }

    /// Advance one step. Automatically transitions between phases.
    pub fn step(&mut self) {
        self.current_step += 1;
        match self.phase {
            DroPEPhase::TrainWithPE => {
                if self.current_step >= self.train_steps {
                    self.phase = DroPEPhase::Recalibrate;
                    self.current_step = 0;
                    // PE is kept but no longer applied
                }
            }
            DroPEPhase::Recalibrate => {
                if self.current_step >= self.recal_steps {
                    self.phase = DroPEPhase::Inference;
                    self.pe = None; // Drop PE entirely
                }
            }
            DroPEPhase::Inference => {
                // No-op, model runs without PE
            }
        }
    }

    /// Apply position encoding if in TrainWithPE phase.
    /// During Recalibrate and Inference, returns embeddings unchanged.
    pub fn apply(&self, embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
        match self.phase {
            DroPEPhase::TrainWithPE => {
                if let Some(ref pe) = self.pe {
                    pe.apply(embeddings)
                } else {
                    embeddings.to_vec()
                }
            }
            DroPEPhase::Recalibrate | DroPEPhase::Inference => embeddings.to_vec(),
        }
    }

    /// Force transition to inference mode (skip recalibration).
    pub fn force_inference(&mut self) {
        self.phase = DroPEPhase::Inference;
        self.pe = None;
    }

    /// Get the current learning rate multiplier for the phase.
    ///
    /// During recalibration, the paper suggests using a lower LR
    /// (typically 10% of peak LR for 1-5% of training steps).
    pub fn lr_multiplier(&self) -> f32 {
        match self.phase {
            DroPEPhase::TrainWithPE => 1.0,
            DroPEPhase::Recalibrate => 0.1, // 10% of peak LR
            DroPEPhase::Inference => 0.0,   // No training
        }
    }

    /// Whether position encoding is currently active.
    pub fn pe_active(&self) -> bool {
        matches!(self.phase, DroPEPhase::TrainWithPE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinusoidal_2d_shape() {
        let pe = Sinusoidal2DPositionEncoding::new(4, 4, 32);
        assert_eq!(pe.encodings.len(), 16, "4×4 grid = 16 positions");
        assert_eq!(pe.encodings[0].len(), 32, "embed_dim = 32");
    }

    #[test]
    fn test_sinusoidal_2d_different_positions() {
        let pe = Sinusoidal2DPositionEncoding::new(4, 4, 32);
        // Position (0,0) and (3,3) should have different encodings
        let diff: f32 = pe.encodings[0]
            .iter()
            .zip(pe.encodings[15].iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.1,
            "Different positions should have different encodings"
        );
    }

    #[test]
    fn test_sinusoidal_2d_row_similarity() {
        let pe = Sinusoidal2DPositionEncoding::new(4, 4, 32);
        // Same row, adjacent columns: (0,0) and (0,1)
        let sim_same_row: f32 = pe.encodings[0]
            .iter()
            .zip(pe.encodings[1].iter())
            .map(|(&a, &b)| a * b)
            .sum();
        // Different row, same column: (0,0) and (1,0)
        let sim_diff_row: f32 = pe.encodings[0]
            .iter()
            .zip(pe.encodings[4].iter())
            .map(|(&a, &b)| a * b)
            .sum();
        // Both should have some similarity but be distinguishable
        assert!(sim_same_row.is_finite());
        assert!(sim_diff_row.is_finite());
    }

    #[test]
    fn test_apply_pe() {
        let pe = Sinusoidal2DPositionEncoding::new(2, 2, 4);
        let embeddings = vec![
            vec![1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 1.0],
        ];
        let with_pe = pe.apply(&embeddings);
        assert_eq!(with_pe.len(), 4);
        // Values should be different from input (PE added)
        assert_ne!(with_pe[0], embeddings[0]);
    }

    #[test]
    fn test_drope_lifecycle() {
        let mut ctrl = DroPEController::new(2, 2, 8, 10, 0.2);
        assert_eq!(ctrl.phase, DroPEPhase::TrainWithPE);
        assert!(ctrl.pe_active());
        assert_eq!(ctrl.lr_multiplier(), 1.0);

        // Train for 10 steps
        for _ in 0..10 {
            ctrl.step();
        }
        assert_eq!(ctrl.phase, DroPEPhase::Recalibrate);
        assert!(!ctrl.pe_active());
        assert_eq!(ctrl.lr_multiplier(), 0.1);

        // Recalibrate for 2 steps (10 * 0.2 = 2)
        for _ in 0..2 {
            ctrl.step();
        }
        assert_eq!(ctrl.phase, DroPEPhase::Inference);
        assert!(ctrl.pe.is_none(), "PE should be dropped");
        assert_eq!(ctrl.lr_multiplier(), 0.0);
    }

    #[test]
    fn test_drope_apply_phases() {
        let mut ctrl = DroPEController::new(2, 2, 4, 5, 0.2);
        let embeddings = vec![
            vec![1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 1.0],
        ];

        // Phase 1: PE applied
        let with_pe = ctrl.apply(&embeddings);
        assert_ne!(with_pe[0], embeddings[0], "PE should modify embeddings");

        // Advance to recalibrate
        for _ in 0..5 {
            ctrl.step();
        }
        let without_pe = ctrl.apply(&embeddings);
        assert_eq!(without_pe, embeddings, "Recalibrate: no PE applied");
    }

    #[test]
    fn test_force_inference() {
        let mut ctrl = DroPEController::new(2, 2, 8, 100, 0.1);
        assert_eq!(ctrl.phase, DroPEPhase::TrainWithPE);
        ctrl.force_inference();
        assert_eq!(ctrl.phase, DroPEPhase::Inference);
        assert!(ctrl.pe.is_none());
    }
}
