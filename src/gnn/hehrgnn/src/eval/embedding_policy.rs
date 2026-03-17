//! GNN-Embedding RL Policy (Upgraded to Deep Neural Network).
//!
//! Replaces the old linear dot-product with a "real" 2-Layer Perceptron (MLP)
//! with ReLU activations. Uses policy gradient (REINFORCE) to learn complex
//! non-linear behaviors in the simulation environments.

use crate::eval::rl_policy::Policy;
use std::collections::HashMap;

// ──────────────────────────────────────────────────────
// Embedding Policy (DNN)
// ──────────────────────────────────────────────────────

/// Deep Neural Network policy using REINFORCE.
pub struct EmbeddingPolicy {
    state_dim: usize,
    hidden_dim: usize,
    num_actions: usize,

    // Layer 1
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,

    // Layer 2
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,

    // AttnRes pseudo-query for depth-attention between layers
    attn_query: Vec<f32>,

    temperature: f64,
    baseline: f64,
    lr: f64,
    seed: u64,
    policy_name: String,
}

impl EmbeddingPolicy {
    pub fn new(num_actions: usize, state_dim: usize) -> Self {
        let hidden_dim = 64; // "Real" capacity
        let mut seed = 7919u64;

        // Helper to initialize weights safely ~ N(0, 0.1)
        let mut next_randn = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = (seed >> 33) as f64 / (1u64 << 31) as f64;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (seed >> 33) as f64 / (1u64 << 31) as f64;
            let r = (-2.0 * (u1 + 1e-8).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            (r * 0.1) as f32
        };

        let mut w1 = vec![vec![0.0; state_dim]; hidden_dim];
        let mut b1 = vec![0.0; hidden_dim];
        for i in 0..hidden_dim {
            b1[i] = 0.0;
            for j in 0..state_dim {
                w1[i][j] = next_randn();
            }
        }

        let mut w2 = vec![vec![0.0; hidden_dim]; num_actions];
        let mut b2 = vec![0.0; num_actions];
        for i in 0..num_actions {
            b2[i] = 0.0;
            for j in 0..hidden_dim {
                w2[i][j] = next_randn();
            }
        }

        Self {
            state_dim,
            hidden_dim,
            num_actions,
            w1,
            b1,
            w2,
            b2,
            attn_query: vec![0.0; hidden_dim], // zero-init per AttnRes paper
            temperature: 1.0,
            baseline: 0.0,
            lr: 0.01, // Higher LR to escape initialization bias
            seed,
            policy_name: "dnn_embedding".to_string(),
        }
    }

    pub fn from_embeddings(
        _embeddings: &HashMap<String, Vec<Vec<f32>>>,
        num_actions: usize,
        hidden_dim: usize,
    ) -> Self {
        // Fallback for compatibility
        Self::new(num_actions, hidden_dim)
    }

    fn next_rng(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.seed >> 33) as f64 / (1u64 << 31) as f64
    }

    fn forward(&self, state: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut h = vec![0.0f32; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let mut sum = self.b1[i];
            for j in 0..self.state_dim {
                sum += self.w1[i][j] * state[j];
            }
            h[i] = if sum > 0.0 { sum } else { 0.0 }; // ReLU
            h[i] = h[i].clamp(-10.0, 10.0); // Prevent explosion
        }

        // AttnRes depth-attention: blend input projection with hidden
        let input_proj: Vec<f32> = (0..self.hidden_dim)
            .map(|i| if i < self.state_dim { state[i] } else { 0.0 })
            .collect();
        let dot_h: f32 = self
            .attn_query
            .iter()
            .zip(h.iter())
            .map(|(q, v)| q * v)
            .sum();
        let dot_i: f32 = self
            .attn_query
            .iter()
            .zip(input_proj.iter())
            .map(|(q, v)| q * v)
            .sum();
        let max_d = dot_h.max(dot_i);
        let w_h = (dot_h - max_d).exp();
        let w_i = (dot_i - max_d).exp();
        let w_sum = w_h + w_i + 1e-8;
        let alpha_h = w_h / w_sum;
        let alpha_i = w_i / w_sum;
        let blended: Vec<f32> = h
            .iter()
            .zip(input_proj.iter())
            .map(|(hv, iv)| alpha_h * hv + alpha_i * iv)
            .collect();

        let mut logits = vec![0.0f32; self.num_actions];
        for i in 0..self.num_actions {
            let mut sum = self.b2[i];
            for j in 0..self.hidden_dim {
                sum += self.w2[i][j] * blended[j];
            }
            logits[i] = sum.clamp(-10.0, 10.0);
        }

        (blended, logits)
    }

    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exps = vec![0.0; logits.len()];
        let mut sum = 0.0;
        for i in 0..logits.len() {
            exps[i] = ((logits[i] - max_l) / self.temperature as f32).exp();
            sum += exps[i];
        }
        for i in 0..logits.len() {
            exps[i] /= sum;
        }
        exps
    }

    pub fn reinforce_update(&mut self, action: usize, state: &[f32], reward: f64) {
        self.reinforce_update_with_baseline(action, state, reward, self.baseline);
        self.baseline = 0.95 * self.baseline + 0.05 * reward;
    }

    pub fn reinforce_update_with_baseline(
        &mut self,
        action: usize,
        state: &[f32],
        reward: f64,
        baseline: f64,
    ) {
        let advantage = (reward - baseline) as f32;
        let (h, logits) = self.forward(state);
        let probs = self.softmax(&logits);

        // \nabla_\theta \log \pi(a|s) -> dLogit
        let mut d_logits = vec![0.0f32; self.num_actions];
        for i in 0..self.num_actions {
            let target = if i == action { 1.0 } else { 0.0 };
            d_logits[i] = target - probs[i];
        }

        // Apply advantage and learning rate
        let factor = self.lr as f32 * advantage;
        for i in 0..self.num_actions {
            d_logits[i] *= factor;
            d_logits[i] = d_logits[i].clamp(-1.0, 1.0); // gradient clipping
        }

        // Backprop through Layer 2
        let mut d_h = vec![0.0f32; self.hidden_dim];
        for i in 0..self.num_actions {
            let dl = d_logits[i];
            self.b2[i] += dl;
            for j in 0..self.hidden_dim {
                d_h[j] += dl * self.w2[i][j];
                self.w2[i][j] += dl * h[j];
            }
        }

        // Backprop through ReLU
        for j in 0..self.hidden_dim {
            if h[j] <= 0.0 {
                d_h[j] = 0.0;
            }
        }

        // Backprop through Layer 1 + AttnRes
        for j in 0..self.hidden_dim {
            let dh = d_h[j];
            self.b1[j] += dh;
            for k in 0..self.state_dim {
                self.w1[j][k] += dh * state[k];
            }
            // Update AttnRes pseudo-query via gradient
            self.attn_query[j] += (self.lr as f32 * 0.1) * dh * h[j];
        }
    }

    pub fn train_from_buffer(&mut self, states: &[Vec<f32>], actions: &[usize], rewards: &[f32]) {
        let n = states.len().min(actions.len()).min(rewards.len());
        if n == 0 {
            return;
        }

        // Use the current baseline for the entire batch
        let current_baseline = self.baseline;
        let mut sum_reward = 0.0;

        for i in 0..n {
            sum_reward += rewards[i] as f64;
            self.reinforce_update_with_baseline(
                actions[i],
                &states[i],
                rewards[i] as f64,
                current_baseline,
            );
        }

        // Update baseline once for the batch
        let avg_reward = sum_reward / n as f64;
        self.baseline = 0.95 * self.baseline + 0.05 * avg_reward;
    }

    pub fn temperature(&self) -> f64 {
        self.temperature
    }
    pub fn set_temperature(&mut self, t: f64) {
        self.temperature = t.max(0.01);
    }
    pub fn baseline(&self) -> f64 {
        self.baseline
    }

    // Stub to prevent compilation breakages in older test logic
    pub fn action_prototypes(&self) -> &[Vec<f32>] {
        &self.w2
    }
}

impl Policy for EmbeddingPolicy {
    fn select_action(&mut self, state: &[f32], available_actions: usize) -> usize {
        if available_actions == 0 {
            return 0;
        }

        let (_, logits) = self.forward(state);
        let probs = self.softmax(&logits);

        // Epsilon greedy safety net
        if self.next_rng() < 0.1 {
            return (self.next_rng() * available_actions as f64).floor() as usize;
        }

        // Categorical sampling
        let u = self.next_rng() as f32;
        let mut cumulative = 0.0;
        for i in 0..available_actions.min(self.num_actions) {
            cumulative += probs[i];
            if u <= cumulative {
                return i;
            }
        }
        available_actions - 1
    }

    fn name(&self) -> &str {
        &self.policy_name
    }
}
