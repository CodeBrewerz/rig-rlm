//! Learnable Fiduciary Scoring Head.
//!
//! Replaces hand-coded `score_action()` formulas with a learnable MLP that:
//! 1. Initializes from expert rules via knowledge distillation
//! 2. Improves recursively through user feedback (accept/reject rewards)
//! 3. Learns the 6-axis scoring weights, conflict patterns, and decision boundaries
//!
//! Architecture:
//!   Input:  [user_emb ∥ target_emb ∥ action_one_hot ∥ context_features]
//!   Hidden: 2 layers with ReLU
//!   Anomaly Gate: h2_gated = h2 ⊙ σ(w_anomaly · anomaly + b_anomaly)
//!   Output: 6 fiduciary axes + 1 recommend/don't-recommend logit
//!
//! The anomaly gate gives anomaly_score a privileged role: instead of being
//! just 1 of 89 input features, it directly modulates every hidden neuron,
//! making anomaly sensitivity structural rather than learned from scratch.
//!
//! Training:
//!   Phase 1 (distillation): MSE loss + auto-generated anomaly-gradient examples
//!   Phase 2 (reward):       reward signal from user feedback
//!   Phase 3 (recursive):    self-improve by replaying successful predictions

use super::fiduciary::{FiduciaryActionType, FiduciaryAxes};
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════
// Learnable Scorer MLP
// ═══════════════════════════════════════════════════════════════

/// A small MLP that learns to score fiduciary actions.
///
/// Input features:
///   - user_emb (hidden_dim)
///   - target_emb (hidden_dim)
///   - action_one_hot (18 action types)
///   - context: [anomaly_score, embedding_affinity, degree_ratio,
///               debt_ratio, goal_progress, has_tax, has_recurring]
///   Total: 2*hidden_dim + 18 + 7
///
/// Output:
///   - 6 fiduciary axes: [cost, risk, goal, urgency, conflict, reversibility]
///   - 1 should_recommend logit
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnableScorer {
    /// input_dim → hidden1
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    /// hidden1 → hidden2
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
    /// hidden2 → output (7: 6 axes + 1 recommend logit)
    w3: Vec<Vec<f32>>,
    b3: Vec<f32>,

    /// Anomaly gate weights: one per hidden2 neuron.
    /// gate_i = σ(w_anomaly_gate[i] * anomaly_score + b_anomaly_gate[i])
    /// h2_gated = h2 ⊙ gate
    w_anomaly_gate: Vec<f32>,
    b_anomaly_gate: Vec<f32>,

    /// Learnable axis weights for combining 6 axes into final score.
    pub axis_weights: [f32; 6],

    /// Learnable conflict matrix: action_i conflicts with action_j.
    /// Shape: [18 × 18], values in [0, 1] where 1 = full conflict.
    pub conflict_matrix: Vec<Vec<f32>>,

    /// Learning rate for gradient updates.
    lr: f32,
    /// Number of training samples seen (for logging).
    pub samples_seen: usize,
    /// Input dimensionality.
    input_dim: usize,
    hidden_dim1: usize,
    hidden_dim2: usize,
}

/// Configuration for the learnable scorer.
#[derive(Debug, Clone)]
pub struct ScorerConfig {
    pub embedding_dim: usize,
    pub hidden1: usize,
    pub hidden2: usize,
    pub lr: f32,
}

impl Default for ScorerConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 32,
            hidden1: 64,
            hidden2: 32,
            lr: 0.001,
        }
    }
}

/// A single training example for the scorer.
#[derive(Debug, Clone)]
pub struct ScorerExample {
    /// User embedding.
    pub user_emb: Vec<f32>,
    /// Target entity embedding.
    pub target_emb: Vec<f32>,
    /// Which action type.
    pub action_type: FiduciaryActionType,
    /// Target's anomaly score.
    pub anomaly_score: f32,
    /// Embedding affinity (cosine sim) between user and target.
    pub embedding_affinity: f32,
    /// Context features: [degree_ratio, debt_ratio, goal_progress, has_tax, has_recurring]
    pub context: [f32; 5],
}

/// Ground truth label from expert rules or user feedback.
#[derive(Debug, Clone)]
pub struct ScorerLabel {
    /// Expert-generated 6-axis scores.
    pub axes: FiduciaryAxes,
    /// Should this action be recommended?
    pub should_recommend: bool,
}

/// User feedback for reward-based learning.
#[derive(Debug, Clone)]
pub struct RewardSignal {
    /// The action that was recommended.
    pub action_type: FiduciaryActionType,
    /// User accepted (true) or rejected (false) the recommendation.
    pub accepted: bool,
    /// Optional: how helpful was this? (1-5 scale, normalized to 0-1)
    pub helpfulness: Option<f32>,
    /// The input that generated this recommendation.
    pub example: ScorerExample,
}

// ═══════════════════════════════════════════════════════════════
// Implementation
// ═══════════════════════════════════════════════════════════════

const NUM_ACTIONS: usize = 18;
const NUM_CONTEXT: usize = 7; // anomaly + affinity + 5 context
const NUM_AXES: usize = 6;
const OUTPUT_DIM: usize = NUM_AXES + 1; // 6 axes + 1 recommend logit

impl LearnableScorer {
    /// Create a new scorer, randomly initialized.
    pub fn new(config: &ScorerConfig) -> Self {
        let input_dim = config.embedding_dim * 2 + NUM_ACTIONS + NUM_CONTEXT;
        let h1 = config.hidden1;
        let h2 = config.hidden2;

        // Initialize anomaly gate: small positive weights so gate starts near 0.5
        // (neutral — doesn't amplify or suppress until trained)
        let w_anomaly_gate: Vec<f32> = (0..h2)
            .map(|i| {
                let seed = (i * 131 + 7) as f32;
                (seed.sin() * 43758.5453).fract() * 0.1
            })
            .collect();
        let b_anomaly_gate = vec![0.0; h2];

        Self {
            w1: random_matrix(h1, input_dim, input_dim),
            b1: vec![0.0; h1],
            w2: random_matrix(h2, h1, h1),
            b2: vec![0.0; h2],
            w3: random_matrix(OUTPUT_DIM, h2, h2),
            b3: vec![0.0; OUTPUT_DIM],
            w_anomaly_gate,
            b_anomaly_gate,
            axis_weights: [0.25, 0.25, 0.15, 0.15, 0.10, 0.10],
            conflict_matrix: vec![vec![0.0; NUM_ACTIONS]; NUM_ACTIONS],
            lr: config.lr,
            samples_seen: 0,
            input_dim,
            hidden_dim1: h1,
            hidden_dim2: h2,
        }
    }

    /// Build input feature vector from example.
    fn build_input(&self, example: &ScorerExample) -> Vec<f32> {
        let mut input = Vec::with_capacity(self.input_dim);
        // User embedding (padded/truncated to fit)
        let emb_dim = (self.input_dim - NUM_ACTIONS - NUM_CONTEXT) / 2;
        for i in 0..emb_dim {
            input.push(example.user_emb.get(i).copied().unwrap_or(0.0));
        }
        // Target embedding
        for i in 0..emb_dim {
            input.push(example.target_emb.get(i).copied().unwrap_or(0.0));
        }
        // Action one-hot
        let action_idx = action_to_index(example.action_type);
        for i in 0..NUM_ACTIONS {
            input.push(if i == action_idx { 1.0 } else { 0.0 });
        }
        // Context features
        input.push(example.anomaly_score);
        input.push(example.embedding_affinity);
        for &c in &example.context {
            input.push(c);
        }
        input
    }

    /// Forward pass: input → anomaly-gated hidden → (6 axes, recommend_logit).
    pub fn forward(&self, example: &ScorerExample) -> (FiduciaryAxes, f32) {
        let input = self.build_input(example);

        // Layer 1: ReLU
        let h1 = matmul_bias_relu(&self.w1, &self.b1, &input);
        // Layer 2: ReLU
        let h2 = matmul_bias_relu(&self.w2, &self.b2, &h1);

        // Anomaly gate: modulate h2 based on anomaly score
        let gate = anomaly_gate(
            &self.w_anomaly_gate,
            &self.b_anomaly_gate,
            example.anomaly_score,
        );
        let h2_gated: Vec<f32> = h2.iter().zip(gate.iter()).map(|(h, g)| h * g).collect();

        // Output: sigmoid for axes (0-1), raw logit for recommend
        let output = matmul_bias(&self.w3, &self.b3, &h2_gated);

        let axes = FiduciaryAxes {
            cost_reduction: sigmoid(output[0]),
            risk_reduction: sigmoid(output[1]),
            goal_alignment: sigmoid(output[2]),
            urgency: sigmoid(output[3]),
            conflict_freedom: sigmoid(output[4]),
            reversibility: sigmoid(output[5]),
        };
        let recommend_logit = output[6];

        (axes, recommend_logit)
    }

    /// Score an action using learned weights (replaces hand-coded score_action).
    pub fn score(&self, axes: &FiduciaryAxes) -> f32 {
        let values = [
            axes.cost_reduction,
            axes.risk_reduction,
            axes.goal_alignment,
            axes.urgency,
            axes.conflict_freedom,
            axes.reversibility,
        ];
        values
            .iter()
            .zip(self.axis_weights.iter())
            .map(|(v, w)| v * w)
            .sum()
    }

    /// Check if action_a conflicts with action_b using learned conflict matrix.
    pub fn conflicts(&self, action_a: FiduciaryActionType, action_b: FiduciaryActionType) -> f32 {
        let a = action_to_index(action_a);
        let b = action_to_index(action_b);
        self.conflict_matrix[a][b]
    }

    // ─── TRAINING ───────────────────────────────────────────────

    /// Phase 1: Knowledge distillation from expert rules.
    ///
    /// Takes examples paired with expert-generated labels and trains
    /// the MLP to replicate the expert's scoring behavior.
    pub fn distill(&mut self, examples: &[ScorerExample], labels: &[ScorerLabel], epochs: usize) {
        assert_eq!(examples.len(), labels.len());

        // Auto-generate anomaly-gradient distillation examples:
        // investigate(high anomaly) → recommend, investigate(low anomaly) → don't
        let emb_dim = (self.input_dim - NUM_ACTIONS - NUM_CONTEXT) / 2;
        let mut aug_examples = examples.to_vec();
        let mut aug_labels = labels.to_vec();
        for anomaly in [0.7, 0.8, 0.9] {
            let ue: Vec<f32> = (0..emb_dim).map(|d| (d as f32 * 0.1).sin() * 0.5).collect();
            let te = ue.clone();
            aug_examples.push(ScorerExample {
                user_emb: ue,
                target_emb: te,
                action_type: FiduciaryActionType::ShouldInvestigate,
                anomaly_score: anomaly,
                embedding_affinity: 0.3,
                context: [0.5, 0.5, anomaly, 0.0, 0.0],
            });
            aug_labels.push(ScorerLabel {
                axes: FiduciaryAxes {
                    cost_reduction: 0.2,
                    risk_reduction: anomaly,
                    goal_alignment: 0.4,
                    urgency: anomaly,
                    conflict_freedom: 0.5,
                    reversibility: 1.0,
                },
                should_recommend: true,
            });
        }
        for anomaly in [0.02, 0.05, 0.1] {
            let ue: Vec<f32> = (0..emb_dim).map(|d| (d as f32 * 0.1).sin() * 0.5).collect();
            let te = ue.clone();
            aug_examples.push(ScorerExample {
                user_emb: ue,
                target_emb: te,
                action_type: FiduciaryActionType::ShouldInvestigate,
                anomaly_score: anomaly,
                embedding_affinity: 0.8,
                context: [0.5, 0.5, anomaly, 0.0, 0.0],
            });
            aug_labels.push(ScorerLabel {
                axes: FiduciaryAxes {
                    cost_reduction: 0.2,
                    risk_reduction: 0.1,
                    goal_alignment: 0.4,
                    urgency: 0.1,
                    conflict_freedom: 0.9,
                    reversibility: 1.0,
                },
                should_recommend: false,
            });
        }

        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for (example, label) in aug_examples.iter().zip(aug_labels.iter()) {
                let input = self.build_input(example);

                // Forward
                let h1 = matmul_bias_relu(&self.w1, &self.b1, &input);
                let h2 = matmul_bias_relu(&self.w2, &self.b2, &h1);
                let gate = anomaly_gate(
                    &self.w_anomaly_gate,
                    &self.b_anomaly_gate,
                    example.anomaly_score,
                );
                let h2_gated: Vec<f32> = h2.iter().zip(gate.iter()).map(|(h, g)| h * g).collect();
                let output = matmul_bias(&self.w3, &self.b3, &h2_gated);

                // Target values
                let target = [
                    label.axes.cost_reduction,
                    label.axes.risk_reduction,
                    label.axes.goal_alignment,
                    label.axes.urgency,
                    label.axes.conflict_freedom,
                    label.axes.reversibility,
                    if label.should_recommend { 2.0 } else { -2.0 },
                ];

                // MSE loss per output
                let mut grad_output = vec![0.0f32; OUTPUT_DIM];
                for i in 0..OUTPUT_DIM {
                    let pred = if i < NUM_AXES {
                        sigmoid(output[i])
                    } else {
                        output[i]
                    };
                    let err = pred - target[i];
                    total_loss += err * err;

                    if i < NUM_AXES {
                        let s = sigmoid(output[i]);
                        grad_output[i] = err * s * (1.0 - s);
                    } else {
                        grad_output[i] = err;
                    }
                }

                // Backprop: output → h2_gated
                let grad_h2_gated = backprop_linear(&self.w3, &grad_output, &h2_gated);

                // Backprop through anomaly gate: grad_h2 = grad_h2_gated ⊙ gate
                // grad_gate = grad_h2_gated ⊙ h2
                let grad_h2: Vec<f32> = grad_h2_gated
                    .iter()
                    .zip(gate.iter())
                    .map(|(g, gt)| g * gt)
                    .collect();
                let grad_gate: Vec<f32> = grad_h2_gated
                    .iter()
                    .zip(h2.iter())
                    .map(|(g, h)| g * h)
                    .collect();

                // Update anomaly gate weights
                for i in 0..self.hidden_dim2 {
                    let g_i = gate[i];
                    let dgate_dw = g_i * (1.0 - g_i) * example.anomaly_score;
                    let dgate_db = g_i * (1.0 - g_i);
                    self.w_anomaly_gate[i] -= self.lr * grad_gate[i] * dgate_dw;
                    self.b_anomaly_gate[i] -= self.lr * grad_gate[i] * dgate_db;
                }

                // Backprop through ReLU layers
                let grad_h1 = backprop_layer(&self.w2, &grad_h2, &h1);

                // Update MLP weights
                update_weights(&mut self.w3, &mut self.b3, &h2_gated, &grad_output, self.lr);
                update_weights(&mut self.w2, &mut self.b2, &h1, &grad_h2, self.lr);
                update_weights(&mut self.w1, &mut self.b1, &input, &grad_h1, self.lr);

                self.samples_seen += 1;
            }

            self.learn_axis_weights(&aug_examples, &aug_labels);

            if epoch == 0 || epoch == epochs - 1 {
                let avg = total_loss / aug_examples.len() as f32 / OUTPUT_DIM as f32;
                eprintln!("  [distill] epoch {}: avg_loss={:.6}", epoch, avg);
            }
        }
    }

    /// Learn axis weights by minimizing score prediction error.
    fn learn_axis_weights(&mut self, examples: &[ScorerExample], labels: &[ScorerLabel]) {
        let mut weight_grad = [0.0f32; 6];

        for (example, label) in examples.iter().zip(labels.iter()) {
            let (pred_axes, _) = self.forward(example);
            let pred_score = self.score(&pred_axes);
            let target_score = label.axes.score();
            let err = pred_score - target_score;

            let axis_values = [
                pred_axes.cost_reduction,
                pred_axes.risk_reduction,
                pred_axes.goal_alignment,
                pred_axes.urgency,
                pred_axes.conflict_freedom,
                pred_axes.reversibility,
            ];
            for i in 0..6 {
                weight_grad[i] += err * axis_values[i];
            }
        }

        // SGD update + normalize
        let n = examples.len() as f32;
        for i in 0..6 {
            self.axis_weights[i] -= self.lr * weight_grad[i] / n;
            self.axis_weights[i] = self.axis_weights[i].max(0.01); // keep positive
        }
        // Normalize to sum to 1
        let sum: f32 = self.axis_weights.iter().sum();
        for w in &mut self.axis_weights {
            *w /= sum;
        }
    }

    /// Phase 2: Reward-based fine-tuning from user feedback.
    ///
    /// When a user accepts a recommendation → positive reward → strengthen.
    /// When a user rejects → negative reward → weaken.
    pub fn apply_reward(&mut self, reward: &RewardSignal) {
        let input = self.build_input(&reward.example);

        // Forward pass with anomaly gate
        let h1 = matmul_bias_relu(&self.w1, &self.b1, &input);
        let h2 = matmul_bias_relu(&self.w2, &self.b2, &h1);
        let gate = anomaly_gate(
            &self.w_anomaly_gate,
            &self.b_anomaly_gate,
            reward.example.anomaly_score,
        );
        let h2_gated: Vec<f32> = h2.iter().zip(gate.iter()).map(|(h, g)| h * g).collect();
        let output = matmul_bias(&self.w3, &self.b3, &h2_gated);

        // Reward signal: +1 for accepted, -1 for rejected
        let base_reward = if reward.accepted { 1.0 } else { -1.0 };
        let reward_strength = reward.helpfulness.unwrap_or(0.5) * base_reward;

        let mut grad_output = vec![0.0f32; OUTPUT_DIM];
        grad_output[NUM_AXES] = -reward_strength;

        for i in 0..NUM_AXES {
            let s = sigmoid(output[i]);
            grad_output[i] = -reward_strength * s * (1.0 - s) * 0.1;
        }

        // Backprop through gated path
        let grad_h2_gated = backprop_linear(&self.w3, &grad_output, &h2_gated);

        // Gate gradients
        let grad_h2: Vec<f32> = grad_h2_gated
            .iter()
            .zip(gate.iter())
            .map(|(g, gt)| g * gt)
            .collect();
        let grad_gate: Vec<f32> = grad_h2_gated
            .iter()
            .zip(h2.iter())
            .map(|(g, h)| g * h)
            .collect();

        let reward_lr = self.lr * 0.1;

        // Update anomaly gate
        for i in 0..self.hidden_dim2 {
            let g_i = gate[i];
            let dgate_dw = g_i * (1.0 - g_i) * reward.example.anomaly_score;
            let dgate_db = g_i * (1.0 - g_i);
            self.w_anomaly_gate[i] -= reward_lr * grad_gate[i] * dgate_dw;
            self.b_anomaly_gate[i] -= reward_lr * grad_gate[i] * dgate_db;
        }

        // Backprop through ReLU layers
        let grad_h1 = backprop_layer(&self.w2, &grad_h2, &h1);

        update_weights(
            &mut self.w3,
            &mut self.b3,
            &h2_gated,
            &grad_output,
            reward_lr,
        );
        update_weights(&mut self.w2, &mut self.b2, &h1, &grad_h2, reward_lr);
        update_weights(&mut self.w1, &mut self.b1, &input, &grad_h1, reward_lr);

        self.samples_seen += 1;

        if !reward.accepted {
            let action_idx = action_to_index(reward.example.action_type);
            self.conflict_matrix[action_idx][action_idx] += 0.01;
            self.conflict_matrix[action_idx][action_idx] =
                self.conflict_matrix[action_idx][action_idx].min(1.0);
        }
    }

    /// Phase 3: Recursive self-improvement.
    ///
    /// Replays successful predictions (accepted by users) with boosted
    /// reward to compound learning. Also identifies weak areas.
    pub fn recursive_improve(
        &mut self,
        replay_buffer: &[RewardSignal],
        rounds: usize,
    ) -> RecursiveImprovementReport {
        let mut report = RecursiveImprovementReport {
            rounds_completed: 0,
            initial_accuracy: 0.0,
            final_accuracy: 0.0,
            axis_weight_drift: [0.0; 6],
            conflict_patterns_learned: 0,
        };

        let initial_weights = self.axis_weights;

        // Measure initial accuracy on buffer
        let initial_correct = replay_buffer
            .iter()
            .filter(|r| {
                let (_, logit) = self.forward(&r.example);
                (logit > 0.0) == r.accepted
            })
            .count();
        report.initial_accuracy = initial_correct as f32 / replay_buffer.len().max(1) as f32;

        for round in 0..rounds {
            // Replay successful predictions with boosted reward
            for signal in replay_buffer {
                if signal.accepted {
                    let boosted = RewardSignal {
                        helpfulness: Some(signal.helpfulness.unwrap_or(0.5) * 1.2),
                        ..signal.clone()
                    };
                    self.apply_reward(&boosted);
                } else {
                    // Replay failures with stronger negative signal
                    let boosted = RewardSignal {
                        helpfulness: Some(signal.helpfulness.unwrap_or(0.5) * 1.5),
                        ..signal.clone()
                    };
                    self.apply_reward(&boosted);
                }
            }
            report.rounds_completed = round + 1;
        }

        // Final accuracy
        let final_correct = replay_buffer
            .iter()
            .filter(|r| {
                let (_, logit) = self.forward(&r.example);
                (logit > 0.0) == r.accepted
            })
            .count();
        report.final_accuracy = final_correct as f32 / replay_buffer.len().max(1) as f32;

        // Axis weight drift
        for i in 0..6 {
            report.axis_weight_drift[i] = (self.axis_weights[i] - initial_weights[i]).abs();
        }

        // Count learned conflict patterns
        report.conflict_patterns_learned = self
            .conflict_matrix
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&v| v > 0.05)
            .count();

        report
    }
}

/// Report from recursive self-improvement.
#[derive(Debug, Clone)]
pub struct RecursiveImprovementReport {
    pub rounds_completed: usize,
    pub initial_accuracy: f32,
    pub final_accuracy: f32,
    pub axis_weight_drift: [f32; 6],
    pub conflict_patterns_learned: usize,
}

// ═══════════════════════════════════════════════════════════════
// Helper functions (plain Rust linear algebra, no deps)
// ═══════════════════════════════════════════════════════════════

fn random_matrix(rows: usize, cols: usize, fan_in: usize) -> Vec<Vec<f32>> {
    // Xavier initialization
    let scale = (2.0 / fan_in as f32).sqrt();
    (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| {
                    let seed = (r * 997 + c * 131 + 7) as f32;
                    (seed.sin() * 43758.5453).fract() * scale
                })
                .collect()
        })
        .collect()
}

fn matmul_bias(w: &[Vec<f32>], b: &[f32], x: &[f32]) -> Vec<f32> {
    w.iter()
        .zip(b.iter())
        .map(|(row, &bias)| row.iter().zip(x.iter()).map(|(&w, &x)| w * x).sum::<f32>() + bias)
        .collect()
}

fn matmul_bias_relu(w: &[Vec<f32>], b: &[f32], x: &[f32]) -> Vec<f32> {
    matmul_bias(w, b, x)
        .into_iter()
        .map(|v| v.max(0.0))
        .collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-20.0, 20.0)).exp())
}

/// Anomaly gate: produces a per-neuron gating signal from the anomaly score.
/// gate_i = sigmoid(w_i * anomaly + b_i)
fn anomaly_gate(w: &[f32], b: &[f32], anomaly_score: f32) -> Vec<f32> {
    w.iter()
        .zip(b.iter())
        .map(|(&wi, &bi)| sigmoid(wi * anomaly_score + bi))
        .collect()
}

/// Backprop through a linear layer (no ReLU activation — for gated paths).
fn backprop_linear(w: &[Vec<f32>], grad_out: &[f32], _input: &[f32]) -> Vec<f32> {
    let in_dim = w[0].len();
    let mut grad_in = vec![0.0f32; in_dim];
    for (i, row) in w.iter().enumerate() {
        let g = grad_out[i];
        for j in 0..in_dim {
            grad_in[j] += g * row[j];
        }
    }
    grad_in
}

fn backprop_layer(w: &[Vec<f32>], grad_out: &[f32], activation: &[f32]) -> Vec<f32> {
    let in_dim = w[0].len();
    let mut grad_in = vec![0.0f32; in_dim];

    for (i, row) in w.iter().enumerate() {
        let g = grad_out[i];
        // ReLU derivative: pass gradient only if activation > 0
        for j in 0..in_dim {
            if activation[j] > 0.0 {
                grad_in[j] += g * row[j];
            }
        }
    }
    grad_in
}

fn update_weights(w: &mut [Vec<f32>], b: &mut [f32], input: &[f32], grad: &[f32], lr: f32) {
    for (i, row) in w.iter_mut().enumerate() {
        let g = grad[i];
        for (j, w_ij) in row.iter_mut().enumerate() {
            if j < input.len() {
                *w_ij -= lr * g * input[j];
            }
        }
        b[i] -= lr * g;
    }
}

fn action_to_index(action: FiduciaryActionType) -> usize {
    match action {
        FiduciaryActionType::ShouldPay => 0,
        FiduciaryActionType::ShouldCancel => 1,
        FiduciaryActionType::ShouldTransfer => 2,
        FiduciaryActionType::ShouldConsolidate => 3,
        FiduciaryActionType::ShouldAvoid => 4,
        FiduciaryActionType::ShouldInvestigate => 5,
        FiduciaryActionType::ShouldRefinance => 6,
        FiduciaryActionType::ShouldPayDownLien => 7,
        FiduciaryActionType::ShouldDispute => 8,
        FiduciaryActionType::ShouldFundGoal => 9,
        FiduciaryActionType::ShouldAdjustBudget => 10,
        FiduciaryActionType::ShouldPrepareTax => 11,
        FiduciaryActionType::ShouldFundTaxSinking => 12,
        FiduciaryActionType::ShouldClaimExemption => 13,
        FiduciaryActionType::ShouldRunTaxScenario => 14,
        FiduciaryActionType::ShouldReconcile => 15,
        FiduciaryActionType::ShouldReviewRecurring => 16,
        FiduciaryActionType::ShouldRevalueAsset => 17,
    }
}
