//! Agent Environment for RFT (Reinforcement Fine-Tuning).
//!
//! The agent takes coding actions (compile, test, edit, train) and receives
//! rewards from 3 tiers: verifiable, metric-delta, and rubric-scored.
//! Safety is enforced via CMDP constraints.

use crate::eval::environment::{Environment, EpisodeLog, RewardSource, StepResult};
use crate::eval::rubric::{Rubric, RubricJudge};
use crate::eval::safety::{CmdpController, CmdpPhase, ConstraintSet};
use std::collections::HashMap;

// ──────────────────────────────────────────────────────
// State
// ──────────────────────────────────────────────────────

/// Observable state for the coding agent.
#[derive(Debug, Clone)]
pub struct AgentState {
    /// Graph embedding summary: mean, std, node count.
    pub graph_summary: Vec<f32>,
    /// Current model metrics.
    pub model_loss: f64,
    pub model_auc: f64,
    pub embedding_norm: f64,
    /// Result of last action.
    pub last_action_success: bool,
    pub last_reward: f64,
    /// Budget tracking.
    pub step_count: usize,
    pub budget_remaining: usize,
    /// Compilation state.
    pub compiles: bool,
    pub test_pass_rate: f64,
    pub tests_total: usize,
    pub tests_passed: usize,
}

impl AgentState {
    pub fn initial() -> Self {
        Self {
            graph_summary: vec![0.0; 8],
            model_loss: 1.0,
            model_auc: 0.5,
            embedding_norm: 1.0,
            last_action_success: true,
            last_reward: 0.0,
            step_count: 0,
            budget_remaining: 50,
            compiles: true,
            test_pass_rate: 1.0,
            tests_total: 10,
            tests_passed: 10,
        }
    }

    /// Convert to feature vector for policy input.
    pub fn to_features(&self) -> Vec<f32> {
        let mut f = self.graph_summary.clone();
        f.push(self.model_loss as f32);
        f.push(self.model_auc as f32);
        f.push(self.embedding_norm as f32);
        f.push(if self.last_action_success { 1.0 } else { 0.0 });
        f.push(self.step_count as f32);
        f.push(self.budget_remaining as f32);
        f.push(if self.compiles { 1.0 } else { 0.0 });
        f.push(self.test_pass_rate as f32);
        f
    }

    /// Convert to metrics map for rubric scoring.
    pub fn to_metrics(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert(
            "compile_success".to_string(),
            if self.compiles { 1.0 } else { 0.0 },
        );
        m.insert("test_pass_rate".to_string(), self.test_pass_rate);
        m.insert("primary_metric".to_string(), self.model_auc);
        m.insert(
            "action_efficiency".to_string(),
            if self.budget_remaining > 0 {
                self.model_auc / (self.step_count.max(1) as f64 / 50.0)
            } else {
                0.0
            },
        );
        m.insert(
            "no_regressions".to_string(),
            if self.test_pass_rate >= 1.0 { 1.0 } else { 0.0 },
        );
        m
    }
}

// ──────────────────────────────────────────────────────
// Actions
// ──────────────────────────────────────────────────────

/// Actions available to the coding agent.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentAction {
    /// Attempt to compile the crate.
    Compile,
    /// Run a specific test.
    RunTest(usize),
    /// Edit a model parameter: (param_index, delta).
    EditModelParam(usize, f32),
    /// Run N training epochs.
    TrainEpoch(usize),
    /// Request a document through the vision pipeline.
    RequestDocument,
    /// Query the graph for information.
    QueryGraph,
    /// Measure a specific metric.
    EvaluateMetric,
}

impl AgentAction {
    /// Estimated constraint costs for this action.
    pub fn estimated_costs(&self) -> Vec<f64> {
        // [compile_failures, test_regressions, action_budget]
        match self {
            AgentAction::Compile => vec![0.0, 0.0, 1.0],
            AgentAction::RunTest(_) => vec![0.0, 0.0, 1.0],
            AgentAction::EditModelParam(_, _) => vec![0.0, 0.0, 1.0],
            AgentAction::TrainEpoch(_) => vec![0.0, 0.0, 2.0],
            AgentAction::RequestDocument => vec![0.0, 0.0, 1.0],
            AgentAction::QueryGraph => vec![0.0, 0.0, 1.0],
            AgentAction::EvaluateMetric => vec![0.0, 0.0, 1.0],
        }
    }

    /// Encode action as index for policy output.
    pub fn to_index(&self) -> usize {
        match self {
            AgentAction::Compile => 0,
            AgentAction::RunTest(_) => 1,
            AgentAction::EditModelParam(_, _) => 2,
            AgentAction::TrainEpoch(_) => 3,
            AgentAction::RequestDocument => 4,
            AgentAction::QueryGraph => 5,
            AgentAction::EvaluateMetric => 6,
        }
    }
}

// ──────────────────────────────────────────────────────
// Agent Environment
// ──────────────────────────────────────────────────────

/// RFT environment for training a coding agent.
pub struct AgentEnv {
    state: AgentState,
    step_num: usize,
    cmdp: CmdpController,
    rubric_judge: RubricJudge,
    rng_seed: u64,
    /// Baseline metrics at episode start (for delta rewards).
    baseline_loss: f64,
    baseline_auc: f64,
}

impl AgentEnv {
    pub fn new() -> Self {
        Self {
            state: AgentState::initial(),
            step_num: 0,
            cmdp: CmdpController::new(ConstraintSet::agent(), 5),
            rubric_judge: RubricJudge::new(Rubric::agent_default()),
            rng_seed: 42,
            baseline_loss: 1.0,
            baseline_auc: 0.5,
        }
    }

    fn next_rng(&mut self) -> f64 {
        self.rng_seed = self
            .rng_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.rng_seed >> 33) as f64 / (1u64 << 31) as f64
    }

    /// Simulate the effect of an action on the environment state.
    fn simulate_action(&mut self, action: &AgentAction) -> (f64, RewardSource, f64) {
        let noise = self.next_rng() * 0.1 - 0.05;

        match action {
            AgentAction::Compile => {
                let success = self.next_rng() > 0.1; // 90% success rate
                self.state.compiles = success;
                let reward = if success { 1.0 } else { -0.5 };
                let cost = if success { 0.0 } else { 1.0 };
                (reward, RewardSource::Verifiable, cost)
            }
            AgentAction::RunTest(idx) => {
                let pass = self.state.compiles && self.next_rng() > 0.15;
                if pass {
                    self.state.tests_passed =
                        (self.state.tests_passed + 1).min(self.state.tests_total);
                } else {
                    self.state.tests_passed = self.state.tests_passed.saturating_sub(1);
                }
                self.state.test_pass_rate =
                    self.state.tests_passed as f64 / self.state.tests_total as f64;
                let reward = if pass { 2.0 } else { -1.0 };
                let cost = if !pass { 1.0 } else { 0.0 };
                (reward, RewardSource::Verifiable, cost)
            }
            AgentAction::EditModelParam(_, delta) => {
                let improvement = *delta as f64 * 0.01 + noise;
                let old_auc = self.state.model_auc;
                self.state.model_auc = (self.state.model_auc + improvement).clamp(0.0, 1.0);
                let delta_auc = self.state.model_auc - old_auc;
                let reward = if delta_auc > 0.0 {
                    5.0 * delta_auc
                } else {
                    -3.0 * delta_auc.abs()
                };
                (reward, RewardSource::MetricDelta, 0.0)
            }
            AgentAction::TrainEpoch(n) => {
                let lr = 0.01 * (1.0 - self.step_num as f64 / 50.0).max(0.1);
                let old_loss = self.state.model_loss;
                self.state.model_loss =
                    (self.state.model_loss - lr * *n as f64 + noise * 0.05).max(0.01);
                self.state.model_auc =
                    (self.state.model_auc + lr * 0.5 * *n as f64 + noise * 0.02).clamp(0.0, 1.0);
                let delta_loss = old_loss - self.state.model_loss;
                let reward = if delta_loss > 0.0 {
                    3.0 * delta_loss
                } else {
                    -1.0
                };
                (reward, RewardSource::MetricDelta, 0.0)
            }
            AgentAction::RequestDocument => {
                self.state.embedding_norm += 0.1 + noise;
                (1.0, RewardSource::MetricDelta, 0.0)
            }
            AgentAction::QueryGraph => {
                // Information gathering — small positive reward
                (0.5, RewardSource::MetricDelta, 0.0)
            }
            AgentAction::EvaluateMetric => (0.5, RewardSource::MetricDelta, 0.0),
        }
    }

    pub fn cmdp(&self) -> &CmdpController {
        &self.cmdp
    }

    pub fn rubric_judge(&self) -> &RubricJudge {
        &self.rubric_judge
    }

    pub fn rubric_judge_mut(&mut self) -> &mut RubricJudge {
        &mut self.rubric_judge
    }
}

impl Environment for AgentEnv {
    type State = AgentState;
    type Action = AgentAction;
    const MAX_STEPS: usize = 50;

    fn state(&self) -> AgentState {
        self.state.clone()
    }

    fn step(&mut self, action: AgentAction) -> StepResult<AgentState> {
        // Check safety before executing
        let costs = action.estimated_costs();
        let safe = self.cmdp.is_action_safe(&costs);

        let (base_reward, source, constraint_cost) = if safe {
            self.simulate_action(&action)
        } else {
            // Unsafe action: skip but penalize
            (-2.0, RewardSource::Verifiable, 0.0)
        };

        // Apply optimistic reward bonus (Phase 2)
        let reward = self.cmdp.reward_bonus(base_reward);

        // Record constraint costs
        let cost_vec = vec![
            if !self.state.compiles {
                constraint_cost
            } else {
                0.0
            },
            if self.state.test_pass_rate < 1.0 {
                constraint_cost * 0.5
            } else {
                0.0
            },
            1.0, // action budget cost
        ];
        self.cmdp.constraints_mut().record_costs(&cost_vec);

        self.step_num += 1;
        self.state.step_count = self.step_num;
        self.state.budget_remaining = Self::MAX_STEPS.saturating_sub(self.step_num);
        self.state.last_action_success = base_reward > 0.0;
        self.state.last_reward = reward;

        let done = self.step_num >= Self::MAX_STEPS || self.state.budget_remaining == 0;
        let truncated = self.step_num >= Self::MAX_STEPS;

        let mut info = self.state.to_metrics();
        info.insert("loss".to_string(), self.state.model_loss);
        info.insert("auc".to_string(), self.state.model_auc);
        info.insert(
            "loss_delta".to_string(),
            self.baseline_loss - self.state.model_loss,
        );
        info.insert(
            "auc_delta".to_string(),
            self.state.model_auc - self.baseline_auc,
        );

        StepResult {
            next_state: self.state.clone(),
            reward,
            done,
            truncated,
            reward_source: source,
            info,
            constraint_cost,
        }
    }

    fn reset(&mut self) {
        let violated = !self.cmdp.constraints().all_satisfied();
        self.cmdp.end_episode(violated);
        self.state = AgentState::initial();
        self.step_num = 0;
        self.baseline_loss = self.state.model_loss;
        self.baseline_auc = self.state.model_auc;
    }

    fn current_step(&self) -> usize {
        self.step_num
    }

    fn available_actions(&self) -> Vec<AgentAction> {
        let mut actions = vec![
            AgentAction::Compile,
            AgentAction::RunTest(0),
            AgentAction::EditModelParam(0, 1.0),
            AgentAction::TrainEpoch(1),
            AgentAction::QueryGraph,
            AgentAction::EvaluateMetric,
        ];
        if self.step_num < Self::MAX_STEPS / 2 {
            actions.push(AgentAction::RequestDocument);
        }
        actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_env_basic() {
        let mut env = AgentEnv::new();
        let state = env.state();
        assert_eq!(state.step_count, 0);
        assert_eq!(state.budget_remaining, 50);
        assert!(state.compiles);
    }

    #[test]
    fn test_agent_env_step() {
        let mut env = AgentEnv::new();
        let result = env.step(AgentAction::Compile);
        assert!(result.reward.is_finite());
        assert_eq!(result.next_state.step_count, 1);
        assert!(!result.done);
    }

    #[test]
    fn test_agent_env_episode() {
        let mut env = AgentEnv::new();
        let mut total_reward = 0.0;
        let mut steps = 0;

        loop {
            let actions = env.available_actions();
            let action = actions[steps % actions.len()].clone();
            let result = env.step(action);
            total_reward += result.reward;
            steps += 1;
            if result.done {
                break;
            }
        }

        assert_eq!(steps, AgentEnv::MAX_STEPS);
        assert!(total_reward.is_finite());
        println!(
            "  Agent episode: {} steps, total_reward={:.2}",
            steps, total_reward
        );

        env.reset();
        assert_eq!(env.current_step(), 0);
    }

    #[test]
    fn test_agent_state_to_features() {
        let state = AgentState::initial();
        let features = state.to_features();
        assert_eq!(features.len(), 16); // 8 graph + 8 scalars
        assert!(features.iter().all(|f| f.is_finite()));
    }

    #[test]
    fn test_cmdp_phase_integration() {
        let mut env = AgentEnv::new();

        // Phase 1: safe exploration
        assert!(matches!(
            env.cmdp().phase(),
            CmdpPhase::SafeExploration { .. }
        ));

        // Run 5 safe episodes to trigger phase 2
        for _ in 0..5 {
            for _ in 0..10 {
                env.step(AgentAction::QueryGraph);
            }
            env.reset();
        }

        assert_eq!(env.cmdp().phase(), CmdpPhase::OptimisticExploration);
    }
}
