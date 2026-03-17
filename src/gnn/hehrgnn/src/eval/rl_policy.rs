//! RL Policies for environment interaction.
//!
//! Random, RuleBased, and EpsilonGreedy policies that work with both
//! agent and fiduciary environments via action index abstraction.

// ──────────────────────────────────────────────────────
// Policy trait
// ──────────────────────────────────────────────────────

/// RL policy: maps state features → action index.
pub trait Policy {
    /// Select an action index given state features.
    fn select_action(&mut self, state: &[f32], available_actions: usize) -> usize;

    /// Policy name for logging.
    fn name(&self) -> &str;
}

// ──────────────────────────────────────────────────────
// Random policy (baseline)
// ──────────────────────────────────────────────────────

/// Random baseline policy — picks uniformly from available actions.
pub struct RandomPolicy {
    seed: u64,
}

impl RandomPolicy {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    fn next_rng(&mut self) -> u64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.seed >> 33
    }
}

impl Policy for RandomPolicy {
    fn select_action(&mut self, _state: &[f32], available_actions: usize) -> usize {
        if available_actions == 0 {
            return 0;
        }
        self.next_rng() as usize % available_actions
    }

    fn name(&self) -> &str {
        "random"
    }
}

// ──────────────────────────────────────────────────────
// Rule-based policy
// ──────────────────────────────────────────────────────

/// Simple rule-based policy using threshold heuristics.
pub struct RuleBasedPolicy {
    /// Feature index to check.
    key_feature_idx: usize,
    /// Threshold for switching actions.
    threshold: f32,
    /// Action to take when feature < threshold.
    below_action: usize,
    /// Action to take when feature >= threshold.
    above_action: usize,
}

impl RuleBasedPolicy {
    pub fn new(key_feature_idx: usize, threshold: f32, below: usize, above: usize) -> Self {
        Self {
            key_feature_idx: key_feature_idx,
            threshold,
            below_action: below,
            above_action: above,
        }
    }

    /// Conservative fiduciary rule: hold if drawdown is high, rebalance otherwise.
    pub fn conservative_fiduciary() -> Self {
        Self::new(
            2,    // Feature index 2 = max_drawdown in FiduciaryState
            0.10, // 10% drawdown threshold
            1,    // Below: rebalance (index 1 in available_actions)
            0,    // Above: hold (index 0 in available_actions)
        )
    }

    /// Active agent rule: train if loss is high, evaluate otherwise.
    pub fn active_agent() -> Self {
        Self::new(
            8,   // Feature index 8 = model_loss in AgentState
            0.5, // Loss threshold
            6,   // Below: evaluate metric
            3,   // Above: train epoch
        )
    }
}

impl Policy for RuleBasedPolicy {
    fn select_action(&mut self, state: &[f32], available_actions: usize) -> usize {
        if available_actions == 0 {
            return 0;
        }
        let feature_val = state.get(self.key_feature_idx).copied().unwrap_or(0.0);
        let action = if feature_val >= self.threshold {
            self.above_action
        } else {
            self.below_action
        };
        action.min(available_actions - 1)
    }

    fn name(&self) -> &str {
        "rule_based"
    }
}

// ──────────────────────────────────────────────────────
// Epsilon-greedy policy
// ──────────────────────────────────────────────────────

/// Epsilon-greedy policy: with probability ε pick random, otherwise best.
pub struct EpsilonGreedyPolicy {
    /// Exploration rate.
    epsilon: f64,
    /// Decay rate per step.
    decay: f64,
    /// Minimum epsilon.
    min_epsilon: f64,
    /// Learned Q-values: q_table[action] = estimated value.
    q_table: Vec<f64>,
    /// Visit counts per action.
    visit_counts: Vec<usize>,
    seed: u64,
}

impl EpsilonGreedyPolicy {
    pub fn new(num_actions: usize, epsilon: f64, decay: f64) -> Self {
        Self {
            epsilon,
            decay,
            min_epsilon: 0.01,
            q_table: vec![0.0; num_actions],
            visit_counts: vec![0; num_actions],
            seed: 12345,
        }
    }

    fn next_rng(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.seed >> 33) as f64 / (1u64 << 31) as f64
    }

    /// Update Q-value for an action after observing reward.
    pub fn update(&mut self, action: usize, reward: f64) {
        if action < self.q_table.len() {
            self.visit_counts[action] += 1;
            let alpha = 1.0 / self.visit_counts[action] as f64; // Decreasing LR
            self.q_table[action] += alpha * (reward - self.q_table[action]);
        }
        self.epsilon = (self.epsilon * self.decay).max(self.min_epsilon);
    }

    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    pub fn q_values(&self) -> &[f64] {
        &self.q_table
    }
}

impl Policy for EpsilonGreedyPolicy {
    fn select_action(&mut self, _state: &[f32], available_actions: usize) -> usize {
        if available_actions == 0 {
            return 0;
        }

        if self.next_rng() < self.epsilon {
            // Explore: random action
            self.next_rng() as usize % available_actions
        } else {
            // Exploit: best Q-value
            let mut best_a = 0;
            let mut best_q = f64::NEG_INFINITY;
            for a in 0..available_actions.min(self.q_table.len()) {
                if self.q_table[a] > best_q {
                    best_q = self.q_table[a];
                    best_a = a;
                }
            }
            best_a
        }
    }

    fn name(&self) -> &str {
        "epsilon_greedy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_policy() {
        let mut p = RandomPolicy::new(42);
        let state = vec![0.0; 8];
        let mut actions = vec![0usize; 7];
        for _ in 0..100 {
            let a = p.select_action(&state, 7);
            assert!(a < 7);
            actions[a] += 1;
        }
        // Should be roughly uniform
        assert!(
            actions.iter().all(|&c| c > 0),
            "All actions should be taken"
        );
    }

    #[test]
    fn test_rule_based_policy() {
        let mut p = RuleBasedPolicy::new(0, 0.5, 1, 2);

        let low = vec![0.3f32; 5];
        assert_eq!(p.select_action(&low, 5), 1);

        let high = vec![0.8f32; 5];
        assert_eq!(p.select_action(&high, 5), 2);
    }

    #[test]
    fn test_epsilon_greedy() {
        let mut p = EpsilonGreedyPolicy::new(5, 1.0, 0.99);
        let state = vec![0.0; 8];

        // Initially ε=1.0, should explore randomly
        let a1 = p.select_action(&state, 5);
        assert!(a1 < 5);

        // Train action 2 with high rewards
        for _ in 0..50 {
            p.update(2, 10.0);
        }

        // ε should have decayed
        assert!(p.epsilon() < 1.0);
        assert!(p.q_values()[2] > p.q_values()[0]);
    }

    #[test]
    fn test_epsilon_decay() {
        let mut p = EpsilonGreedyPolicy::new(3, 1.0, 0.95);
        for _ in 0..100 {
            p.update(0, 1.0);
        }
        assert!(p.epsilon() < 0.1);
        assert!(p.epsilon() >= 0.01); // min_epsilon
    }
}
