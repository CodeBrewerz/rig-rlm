//! Episode simulator: runs environments with policies, collects data,
//! evolves rubrics, and compares policy performance.

use crate::eval::environment::{Environment, EpisodeLog};
use crate::eval::rl_policy::Policy;
use crate::eval::transition_buffer::{Transition, TransitionBuffer};
use std::collections::HashMap;

// ──────────────────────────────────────────────────────
// Simulation report
// ──────────────────────────────────────────────────────

/// Results from a batch of episodes.
#[derive(Debug, Clone)]
pub struct SimulationReport {
    pub policy_name: String,
    pub num_episodes: usize,
    pub mean_reward: f64,
    pub std_reward: f64,
    pub mean_steps: f64,
    pub mean_constraint_cost: f64,
    pub violation_count: usize,
    pub per_episode: Vec<EpisodeSummary>,
}

/// Per-episode summary.
#[derive(Debug, Clone)]
pub struct EpisodeSummary {
    pub total_reward: f64,
    pub steps: usize,
    pub total_constraint_cost: f64,
    pub safety_violated: bool,
    pub info: HashMap<String, f64>,
}

impl SimulationReport {
    pub fn from_logs(policy_name: &str, logs: &[EpisodeLog]) -> Self {
        let n = logs.len() as f64;
        let mean_reward = logs.iter().map(|l| l.total_reward).sum::<f64>() / n;
        let var = logs
            .iter()
            .map(|l| (l.total_reward - mean_reward).powi(2))
            .sum::<f64>()
            / n;
        let std_reward = var.sqrt();
        let mean_steps = logs.iter().map(|l| l.steps as f64).sum::<f64>() / n;
        let mean_cost = logs.iter().map(|l| l.total_constraint_cost).sum::<f64>() / n;
        let violations = logs.iter().filter(|l| l.safety_violated).count();

        let per_episode = logs
            .iter()
            .map(|l| EpisodeSummary {
                total_reward: l.total_reward,
                steps: l.steps,
                total_constraint_cost: l.total_constraint_cost,
                safety_violated: l.safety_violated,
                info: l.info.clone(),
            })
            .collect();

        Self {
            policy_name: policy_name.to_string(),
            num_episodes: logs.len(),
            mean_reward,
            std_reward,
            mean_steps,
            mean_constraint_cost: mean_cost,
            violation_count: violations,
            per_episode,
        }
    }
}

// ──────────────────────────────────────────────────────
// Episode runner
// ──────────────────────────────────────────────────────

/// Run a single episode, returning the log and filling the buffer.
pub fn run_episode<E, P>(
    env: &mut E,
    policy: &mut P,
    mut buffer: Option<&mut TransitionBuffer>,
    constraint_budget: f64,
) -> EpisodeLog
where
    E: Environment,
    E::State: StateToFeatures,
    P: Policy,
{
    env.reset();
    let mut log = EpisodeLog::new();
    let mut prev_features = env.state().to_features();

    loop {
        let available = env.available_actions();
        let num_available = available.len();
        let action_idx = policy.select_action(&prev_features, num_available);
        let action = available.into_iter().nth(action_idx.min(num_available - 1));

        let result = match action {
            Some(a) => env.step(a),
            None => break,
        };

        let next_features = result.next_state.to_features();
        log.record_step(&result, constraint_budget);

        for (k, v) in &result.info {
            log.info.insert(k.clone(), *v);
        }

        if let Some(ref mut buf) = buffer {
            buf.push(Transition {
                state: prev_features.clone(),
                action_id: action_idx,
                reward: result.reward as f32,
                next_state: next_features.clone(),
                done: result.done,
                constraint_cost: result.constraint_cost as f32,
            });
        }

        prev_features = next_features;

        if result.done || result.truncated {
            break;
        }
    }

    log
}

/// Run N episodes.
pub fn run_batch<E, P>(
    env: &mut E,
    policy: &mut P,
    num_episodes: usize,
    mut buffer: Option<&mut TransitionBuffer>,
    constraint_budget: f64,
) -> SimulationReport
where
    E: Environment,
    E::State: StateToFeatures,
    P: Policy,
{
    let mut logs = Vec::with_capacity(num_episodes);

    for _ in 0..num_episodes {
        let log = run_episode(env, policy, buffer.as_deref_mut(), constraint_budget);
        logs.push(log);
    }

    SimulationReport::from_logs(policy.name(), &logs)
}

// ──────────────────────────────────────────────────────
// Policy comparison
// ──────────────────────────────────────────────────────

/// Compare two policies A/B style.
#[derive(Debug)]
pub struct PolicyComparison {
    pub report_a: SimulationReport,
    pub report_b: SimulationReport,
    pub reward_delta: f64,
    pub a_is_better: bool,
}

impl PolicyComparison {
    pub fn compare(a: SimulationReport, b: SimulationReport) -> Self {
        let delta = a.mean_reward - b.mean_reward;
        Self {
            reward_delta: delta,
            a_is_better: delta > 0.0,
            report_a: a,
            report_b: b,
        }
    }
}

// ──────────────────────────────────────────────────────
// State → features conversion
// ──────────────────────────────────────────────────────

/// Convert environment state to feature vector for policy input.
pub trait StateToFeatures: Clone {
    fn to_features(&self) -> Vec<f32>;
}

// Implement for agent state
impl StateToFeatures for crate::eval::agent_env::AgentState {
    fn to_features(&self) -> Vec<f32> {
        self.to_features()
    }
}

// Implement for fiduciary state
impl StateToFeatures for crate::eval::fiduciary_env::FiduciaryState {
    fn to_features(&self) -> Vec<f32> {
        self.to_features()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::agent_env::AgentEnv;
    use crate::eval::fiduciary_env::FiduciaryEnv;
    use crate::eval::rl_policy::{EpsilonGreedyPolicy, RandomPolicy, RuleBasedPolicy};

    #[test]
    fn test_agent_episode() {
        let mut env = AgentEnv::new();
        let mut policy = RandomPolicy::new(42);
        let log = run_episode(&mut env, &mut policy, None, 50.0);

        println!(
            "  Agent episode: {} steps, reward={:.2}",
            log.steps, log.total_reward
        );
        assert!(log.steps > 0);
        assert!(log.total_reward.is_finite());
    }

    #[test]
    fn test_fiduciary_episode() {
        let mut env = FiduciaryEnv::new(4);
        let mut policy = RandomPolicy::new(42);
        let log = run_episode(&mut env, &mut policy, None, 1.0);

        println!(
            "  Fiduciary episode: {} steps, reward={:.2}",
            log.steps, log.total_reward
        );
        assert!(log.steps > 0);
    }

    #[test]
    fn test_agent_batch() {
        let mut env = AgentEnv::new();
        let mut policy = RandomPolicy::new(42);
        let mut buffer = TransitionBuffer::new(1000);

        let report = run_batch(&mut env, &mut policy, 5, Some(&mut buffer), 50.0);

        println!("  Agent batch ({} episodes):", report.num_episodes);
        println!(
            "    Mean reward: {:.2} ± {:.2}",
            report.mean_reward, report.std_reward
        );
        println!("    Mean steps: {:.0}", report.mean_steps);
        println!("    Violations: {}", report.violation_count);
        println!("    Buffer size: {}", buffer.len());

        assert_eq!(report.num_episodes, 5);
        assert!(buffer.len() > 0);
    }

    #[test]
    fn test_fiduciary_batch() {
        let mut env = FiduciaryEnv::new(4);
        let mut policy = RandomPolicy::new(42);

        let report = run_batch(&mut env, &mut policy, 3, None, 1.0);

        println!("  Fiduciary batch ({} episodes):", report.num_episodes);
        println!(
            "    Mean reward: {:.2} ± {:.2}",
            report.mean_reward, report.std_reward
        );
        println!("    Mean steps: {:.0}", report.mean_steps);

        assert_eq!(report.num_episodes, 3);
    }

    #[test]
    fn test_policy_comparison_agent() {
        let mut env_a = AgentEnv::new();
        let mut env_b = AgentEnv::new();
        let mut random = RandomPolicy::new(42);
        let mut rule = RuleBasedPolicy::active_agent();

        let report_a = run_batch(&mut env_a, &mut random, 5, None, 50.0);
        let report_b = run_batch(&mut env_b, &mut rule, 5, None, 50.0);

        let comparison = PolicyComparison::compare(report_a, report_b);
        println!("\n  Agent Policy Comparison:");
        println!(
            "    {} mean_reward={:.2}",
            comparison.report_a.policy_name, comparison.report_a.mean_reward
        );
        println!(
            "    {} mean_reward={:.2}",
            comparison.report_b.policy_name, comparison.report_b.mean_reward
        );
        println!(
            "    Delta: {:.2}, {} is better",
            comparison.reward_delta,
            if comparison.a_is_better {
                &comparison.report_a.policy_name
            } else {
                &comparison.report_b.policy_name
            }
        );
    }

    #[test]
    fn test_policy_comparison_fiduciary() {
        let mut env_a = FiduciaryEnv::new(4);
        let mut env_b = FiduciaryEnv::new(4);
        let mut random = RandomPolicy::new(42);
        let mut conservative = RuleBasedPolicy::conservative_fiduciary();

        let report_a = run_batch(&mut env_a, &mut random, 3, None, 1.0);
        let report_b = run_batch(&mut env_b, &mut conservative, 3, None, 1.0);

        let comparison = PolicyComparison::compare(report_a, report_b);
        println!("\n  Fiduciary Policy Comparison:");
        println!(
            "    {} mean_reward={:.2}",
            comparison.report_a.policy_name, comparison.report_a.mean_reward
        );
        println!(
            "    {} mean_reward={:.2}",
            comparison.report_b.policy_name, comparison.report_b.mean_reward
        );
        println!("    Delta: {:.2}", comparison.reward_delta);
    }

    #[test]
    fn test_transition_buffer_fills() {
        let mut env = AgentEnv::new();
        let mut policy = RandomPolicy::new(42);
        let mut buffer = TransitionBuffer::new(500);

        run_episode(&mut env, &mut policy, Some(&mut buffer), 50.0);

        assert!(buffer.len() > 0);
        let batch = buffer.sample(5);
        assert_eq!(batch.len(), 5);
        assert!(batch.mean_reward().is_finite());
    }
}
