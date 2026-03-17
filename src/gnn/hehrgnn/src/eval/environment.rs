//! RL Environment traits adapted from burn-rl.
//!
//! Provides `Environment`, `StepResult`, and `RewardSource` — the shared
//! foundation for both the coding-agent (RFT) and fiduciary environments.

use std::collections::HashMap;

// ──────────────────────────────────────────────────────
// Reward source classification
// ──────────────────────────────────────────────────────

/// Where a reward signal came from.
#[derive(Debug, Clone, PartialEq)]
pub enum RewardSource {
    /// Binary pass/fail (compile, test).
    Verifiable,
    /// Measured numeric delta (loss change, AUC change).
    MetricDelta,
    /// Rubric-scored (RLER evolving rubrics).
    Rubric { score: f64, rubric_version: u32 },
    /// Combined from multiple sources.
    Composite(Vec<(RewardSource, f64)>),
}

// ──────────────────────────────────────────────────────
// Step result
// ──────────────────────────────────────────────────────

/// Result of one step in the environment (adapted from burn-rl `StepResult`).
#[derive(Debug, Clone)]
pub struct StepResult<S> {
    /// Updated state after the action.
    pub next_state: S,
    /// Scalar reward for this step.
    pub reward: f64,
    /// Whether the episode has terminated (natural end).
    pub done: bool,
    /// Whether the episode was truncated (hit MAX_STEPS).
    pub truncated: bool,
    /// Which mechanism produced the reward.
    pub reward_source: RewardSource,
    /// Extra per-step info (metrics, diagnostics).
    pub info: HashMap<String, f64>,
    /// Constraint cost for this step (for CMDP safety).
    pub constraint_cost: f64,
}

// ──────────────────────────────────────────────────────
// Environment trait (from burn-rl)
// ──────────────────────────────────────────────────────

/// RL environment interface — adapted from burn-rl's `Environment` trait.
pub trait Environment {
    /// State representation.
    type State: Clone;
    /// Action space.
    type Action: Clone;

    /// Maximum steps per episode.
    const MAX_STEPS: usize;

    /// Current observation.
    fn state(&self) -> Self::State;
    /// Take an action, get the result.
    fn step(&mut self, action: Self::Action) -> StepResult<Self::State>;
    /// Reset to initial state for a new episode.
    fn reset(&mut self);
    /// Current step number within the episode.
    fn current_step(&self) -> usize;
    /// Available actions from the current state.
    fn available_actions(&self) -> Vec<Self::Action>;
}

// ──────────────────────────────────────────────────────
// Environment initialization
// ──────────────────────────────────────────────────────

/// Factory for creating environments (from burn-rl).
pub trait EnvironmentInit<E: Environment>: Clone {
    fn init(&self) -> E;
}

impl<F, E> EnvironmentInit<E> for F
where
    F: Fn() -> E + Clone,
    E: Environment,
{
    fn init(&self) -> E {
        (self)()
    }
}

// ──────────────────────────────────────────────────────
// Reward verifier
// ──────────────────────────────────────────────────────

/// Verifiable reward: produces ground-truth reward from an outcome.
pub trait RewardVerifier {
    type State;
    type Action;

    /// Compute verified reward given (before, action, after).
    fn verify(
        &self,
        before: &Self::State,
        action: &Self::Action,
        after: &Self::State,
    ) -> (f64, RewardSource);
}

// ──────────────────────────────────────────────────────
// Episode log
// ──────────────────────────────────────────────────────

/// Full record of an episode for analysis/replay.
#[derive(Debug, Clone)]
pub struct EpisodeLog {
    /// Total reward accumulated.
    pub total_reward: f64,
    /// Number of steps taken.
    pub steps: usize,
    /// Whether the episode terminated naturally.
    pub done: bool,
    /// Per-step rewards.
    pub rewards: Vec<f64>,
    /// Per-step constraint costs.
    pub constraint_costs: Vec<f64>,
    /// Total constraint cost for the episode.
    pub total_constraint_cost: f64,
    /// Whether any safety constraint was violated.
    pub safety_violated: bool,
    /// Aggregated info across the episode.
    pub info: HashMap<String, f64>,
}

impl EpisodeLog {
    pub fn new() -> Self {
        Self {
            total_reward: 0.0,
            steps: 0,
            done: false,
            rewards: Vec::new(),
            constraint_costs: Vec::new(),
            total_constraint_cost: 0.0,
            safety_violated: false,
            info: HashMap::new(),
        }
    }

    pub fn record_step(&mut self, result: &StepResult<impl Clone>, constraint_budget: f64) {
        self.rewards.push(result.reward);
        self.total_reward += result.reward;
        self.constraint_costs.push(result.constraint_cost);
        self.total_constraint_cost += result.constraint_cost;
        if self.total_constraint_cost > constraint_budget {
            self.safety_violated = true;
        }
        self.steps += 1;
        self.done = result.done;
    }

    /// Mean reward per step.
    pub fn mean_reward(&self) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            self.total_reward / self.steps as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_result_creation() {
        let result: StepResult<Vec<f32>> = StepResult {
            next_state: vec![1.0, 2.0, 3.0],
            reward: 1.5,
            done: false,
            truncated: false,
            reward_source: RewardSource::Verifiable,
            info: HashMap::new(),
            constraint_cost: 0.0,
        };
        assert_eq!(result.reward, 1.5);
        assert!(!result.done);
        assert_eq!(result.reward_source, RewardSource::Verifiable);
    }

    #[test]
    fn test_episode_log() {
        let mut log = EpisodeLog::new();
        assert_eq!(log.steps, 0);
        assert_eq!(log.total_reward, 0.0);

        let result = StepResult {
            next_state: vec![1.0f32],
            reward: 2.5,
            done: false,
            truncated: false,
            reward_source: RewardSource::MetricDelta,
            info: HashMap::new(),
            constraint_cost: 0.3,
        };
        log.record_step(&result, 1.0);
        assert_eq!(log.steps, 1);
        assert_eq!(log.total_reward, 2.5);
        assert_eq!(log.total_constraint_cost, 0.3);
        assert!(!log.safety_violated);

        // Exceed constraint budget
        let result2 = StepResult {
            next_state: vec![2.0f32],
            reward: 1.0,
            done: true,
            truncated: false,
            reward_source: RewardSource::Verifiable,
            info: HashMap::new(),
            constraint_cost: 0.8,
        };
        log.record_step(&result2, 1.0);
        assert!(log.safety_violated);
        assert_eq!(log.steps, 2);
    }

    #[test]
    fn test_reward_source_composite() {
        let composite = RewardSource::Composite(vec![
            (RewardSource::Verifiable, 1.0),
            (RewardSource::MetricDelta, 3.0),
            (
                RewardSource::Rubric {
                    score: 0.85,
                    rubric_version: 2,
                },
                2.0,
            ),
        ]);
        match composite {
            RewardSource::Composite(sources) => assert_eq!(sources.len(), 3),
            _ => panic!("Expected composite"),
        }
    }
}
