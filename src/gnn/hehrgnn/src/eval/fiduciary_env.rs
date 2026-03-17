//! Fiduciary Environment — financial portfolio simulation.
//!
//! The agent manages a portfolio: allocating, rebalancing, flagging fraud,
//! and requesting documents. Rewards come from 3 tiers + CMDP safety.

use crate::eval::environment::{Environment, RewardSource, StepResult};
use crate::eval::rubric::{Rubric, RubricJudge};
use crate::eval::safety::{CmdpController, CmdpPhase, ConstraintSet};
use std::collections::HashMap;

// ──────────────────────────────────────────────────────
// State
// ──────────────────────────────────────────────────────

/// Observable state of the fiduciary environment.
#[derive(Debug, Clone)]
pub struct FiduciaryState {
    /// Portfolio value.
    pub portfolio_value: f64,
    /// Cash position.
    pub cash: f64,
    /// Position weights (per asset).
    pub positions: Vec<f64>,
    /// Maximum drawdown encountered so far.
    pub max_drawdown: f64,
    /// Peak portfolio value (for drawdown calc).
    pub peak_value: f64,
    /// Cumulative return.
    pub cumulative_return: f64,
    /// Running Sharpe ratio components.
    pub returns_history: Vec<f64>,
    /// Fraud detection count.
    pub frauds_detected: usize,
    pub frauds_missed: usize,
    /// Documents processed.
    pub documents_processed: usize,
    /// Current step (trading day).
    pub trading_day: usize,
    /// Signal about incoming transaction risk (-1.0 to 1.0)
    pub current_fraud_signal: f64,
}

impl FiduciaryState {
    pub fn initial(num_assets: usize) -> Self {
        let weight = 1.0 / (num_assets + 1) as f64; // Equal weight + cash
        Self {
            portfolio_value: 100_000.0,
            cash: 100_000.0 * weight,
            positions: vec![weight; num_assets],
            max_drawdown: 0.0,
            peak_value: 100_000.0,
            cumulative_return: 0.0,
            returns_history: Vec::new(),
            frauds_detected: 0,
            frauds_missed: 0,
            documents_processed: 0,
            trading_day: 0,
            current_fraud_signal: 0.0,
        }
    }

    /// Compute Sharpe ratio from returns history.
    pub fn sharpe_ratio(&self) -> f64 {
        if self.returns_history.len() < 2 {
            return 0.0;
        }
        let n = self.returns_history.len() as f64;
        let mean = self.returns_history.iter().sum::<f64>() / n;
        let var = self
            .returns_history
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        let std = var.sqrt();
        if std < 1e-8 {
            0.0
        } else {
            mean / std * (252.0f64).sqrt()
        } // Annualized
    }

    /// Concentration: max single position weight.
    pub fn max_concentration(&self) -> f64 {
        self.positions.iter().cloned().fold(0.0f64, f64::max)
    }

    /// Cash ratio.
    pub fn cash_ratio(&self) -> f64 {
        if self.portfolio_value < 1e-8 {
            1.0
        } else {
            self.cash / self.portfolio_value
        }
    }

    /// Convert to metrics for rubric scoring.
    pub fn to_metrics(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("portfolio_return".to_string(), self.cumulative_return);
        m.insert("sharpe_ratio".to_string(), self.sharpe_ratio());
        m.insert(
            "max_drawdown_ok".to_string(),
            if self.max_drawdown < 0.20 { 1.0 } else { 0.0 },
        );
        m.insert(
            "fraud_detection_rate".to_string(),
            if self.frauds_detected > 0 { 1.0 } else { 0.0 },
        );
        m.insert(
            "document_coverage".to_string(),
            if self.documents_processed > 0 {
                1.0
            } else {
                0.0
            },
        );
        m
    }

    /// Convert to feature vector.
    pub fn to_features(&self) -> Vec<f32> {
        let mut f = vec![
            (self.portfolio_value / 100_000.0) as f32,
            (self.cash / 100_000.0) as f32,
            self.max_drawdown as f32,
            self.cumulative_return as f32,
            self.sharpe_ratio() as f32,
            self.max_concentration() as f32,
            self.cash_ratio() as f32,
            (self.trading_day as f32 / 252.0),
            self.current_fraud_signal as f32,
        ];
        for p in &self.positions {
            f.push(*p as f32);
        }
        f
    }
}

// ──────────────────────────────────────────────────────
// Actions
// ──────────────────────────────────────────────────────

/// Fiduciary actions.
#[derive(Debug, Clone, PartialEq)]
pub enum FiduciaryAction {
    /// Set target allocation weights.
    Allocate(Vec<f64>),
    /// Rebalance to equal weights.
    Rebalance,
    /// Hold current positions.
    Hold,
    /// Flag a transaction as fraudulent.
    FlagFraud(usize),
    /// Request a document for analysis.
    RequestDocument,
}

impl FiduciaryAction {
    pub fn estimated_costs(&self) -> Vec<f64> {
        // [drawdown_risk, concentration_risk, cash_reduction, compliance_risk]
        match self {
            FiduciaryAction::Allocate(_) => vec![0.02, 0.05, 0.01, 0.0],
            FiduciaryAction::Rebalance => vec![0.01, 0.0, 0.005, 0.0],
            FiduciaryAction::Hold => vec![0.0, 0.0, 0.0, 0.0],
            FiduciaryAction::FlagFraud(_) => vec![0.0, 0.0, 0.0, 0.0],
            FiduciaryAction::RequestDocument => vec![0.0, 0.0, 0.0, 0.0],
        }
    }
}

// ──────────────────────────────────────────────────────
// Fiduciary Environment
// ──────────────────────────────────────────────────────

/// Financial portfolio environment with CMDP safety.
pub struct FiduciaryEnv {
    state: FiduciaryState,
    num_assets: usize,
    step_num: usize,
    cmdp: CmdpController,
    rubric_judge: RubricJudge,
    rng_seed: u64,
    /// Simulated daily returns per asset.
    market_returns: Vec<Vec<f64>>,
    /// Whether fraud is present at each step.
    fraud_schedule: Vec<bool>,
}

impl FiduciaryEnv {
    pub fn new(num_assets: usize) -> Self {
        let mut env = Self {
            state: FiduciaryState::initial(num_assets),
            num_assets,
            step_num: 0,
            cmdp: CmdpController::new(ConstraintSet::fiduciary(), 10),
            rubric_judge: RubricJudge::new(Rubric::fiduciary_default()),
            rng_seed: 1337,
            market_returns: Vec::new(),
            fraud_schedule: Vec::new(),
        };
        env.generate_market_data();
        env
    }

    fn next_rng(&mut self) -> f64 {
        self.rng_seed = self
            .rng_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.rng_seed >> 33) as f64 / (1u64 << 31) as f64
    }

    /// Generate synthetic market returns for the episode.
    fn generate_market_data(&mut self) {
        self.market_returns.clear();
        self.fraud_schedule.clear();

        for day in 0..252 {
            let mut daily = Vec::new();
            for _ in 0..self.num_assets {
                // ~N(0.0004, 0.02) daily return (annualized ~10%, vol ~32%)
                let u1 = self.next_rng().max(1e-10);
                let u2 = self.next_rng();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let ret = 0.0004 + 0.02 * z;
                daily.push(ret);
            }
            self.market_returns.push(daily);

            // Fraud: ~50% chance on any day (to give balanced training signal)
            let fraud_roll = self.next_rng();
            self.fraud_schedule.push(fraud_roll < 0.5);
        }
    }

    fn simulate_action(&mut self, action: &FiduciaryAction) -> (f64, RewardSource, f64) {
        let day = self.step_num.min(251);
        let daily_returns = self.market_returns[day].clone();

        let has_fraud = self.fraud_schedule[day];

        match action {
            FiduciaryAction::Allocate(weights) => {
                let sum: f64 = weights.iter().sum();
                if sum > 1e-8 {
                    self.state.positions = weights.iter().map(|w| w / sum).collect();
                    self.state.cash = self.state.portfolio_value
                        * (1.0 - self.state.positions.iter().sum::<f64>()).max(0.0);
                }
                self.apply_market_returns(&daily_returns);
                let reward = self.state.returns_history.last().copied().unwrap_or(0.0) * 100.0;
                let cost = self.state.max_concentration();
                (reward, RewardSource::MetricDelta, cost)
            }
            FiduciaryAction::Rebalance => {
                let equal = 1.0 / (self.num_assets + 1) as f64;
                self.state.positions = vec![equal; self.num_assets];
                self.state.cash = self.state.portfolio_value * equal;
                if has_fraud {
                    self.state.cumulative_return -= 0.1; // 10% penalty for ignoring fraud
                    self.state.portfolio_value *= 0.9;
                }
                self.apply_market_returns(&daily_returns);
                let reward = self.state.returns_history.last().copied().unwrap_or(0.0) * 100.0;
                (reward, RewardSource::MetricDelta, 0.0)
            }
            FiduciaryAction::Hold => {
                // If there is fraud and we just Hold, we lose heavily!
                if has_fraud {
                    self.state.cumulative_return -= 0.1; // 10% penalty for ignoring fraud
                    self.state.portfolio_value *= 0.9;
                }
                self.apply_market_returns(&daily_returns);
                let reward = self.state.returns_history.last().copied().unwrap_or(0.0) * 100.0;
                (reward, RewardSource::MetricDelta, 0.0)
            }
            FiduciaryAction::FlagFraud(_) => {
                if has_fraud {
                    self.state.frauds_detected += 1;
                    // Properly handle the fraud, preventing the portfolio drop
                    self.apply_market_returns(&daily_returns);
                    (10.0, RewardSource::Verifiable, 0.0)
                } else {
                    // False positive! Large transactional penalty.
                    self.state.cumulative_return -= 0.05;
                    self.state.portfolio_value *= 0.95;
                    self.apply_market_returns(&daily_returns);
                    (-10.0, RewardSource::Verifiable, 0.0)
                }
            }
            FiduciaryAction::RequestDocument => {
                self.state.documents_processed += 1;
                // Slight operational cost for requesting documents
                self.state.cumulative_return -= 0.001;
                self.state.portfolio_value *= 0.999;
                self.apply_market_returns(&daily_returns);
                (1.0, RewardSource::MetricDelta, 0.0)
            }
        }
    }

    fn apply_market_returns(&mut self, daily_returns: &[f64]) {
        let mut portfolio_return = 0.0;
        for (i, &r) in daily_returns.iter().enumerate() {
            if i < self.state.positions.len() {
                portfolio_return += self.state.positions[i] * r;
            }
        }
        self.state.portfolio_value *= 1.0 + portfolio_return;
        self.state.returns_history.push(portfolio_return);
        self.state.cumulative_return = self.state.portfolio_value / 100_000.0 - 1.0;

        // Update drawdown
        if self.state.portfolio_value > self.state.peak_value {
            self.state.peak_value = self.state.portfolio_value;
        }
        let drawdown = (self.state.peak_value - self.state.portfolio_value) / self.state.peak_value;
        if drawdown > self.state.max_drawdown {
            self.state.max_drawdown = drawdown;
        }

        // Check for missed fraud
        let day = self.step_num.min(251);
        if self.fraud_schedule[day] {
            self.state.frauds_missed += 1; // Will be decremented if FlagFraud was called
        }
    }

    pub fn cmdp(&self) -> &CmdpController {
        &self.cmdp
    }

    pub fn rubric_judge_mut(&mut self) -> &mut RubricJudge {
        &mut self.rubric_judge
    }
}

impl Environment for FiduciaryEnv {
    type State = FiduciaryState;
    type Action = FiduciaryAction;
    const MAX_STEPS: usize = 252; // Trading days per year

    fn state(&self) -> FiduciaryState {
        self.state.clone()
    }

    fn step(&mut self, action: FiduciaryAction) -> StepResult<FiduciaryState> {
        let costs = action.estimated_costs();
        let safe = self.cmdp.is_action_safe(&costs);

        let (base_reward, source, constraint_cost) = if safe {
            self.simulate_action(&action)
        } else {
            // Force hold if unsafe
            self.simulate_action(&FiduciaryAction::Hold)
        };

        let reward = self.cmdp.reward_bonus(base_reward);

        self.cmdp.constraints_mut().record_costs(&[
            self.state.max_drawdown,
            self.state.max_concentration(),
            1.0 - self.state.cash_ratio(),
            0.0,
        ]);

        self.step_num += 1;
        self.state.trading_day = self.step_num;
        let done = self.step_num >= Self::MAX_STEPS;

        let day = self.step_num.min(251);
        self.state.current_fraud_signal = if self.fraud_schedule[day] { 0.8 } else { -0.8 };

        let mut info = self.state.to_metrics();
        info.insert("portfolio_value".to_string(), self.state.portfolio_value);
        info.insert("sharpe".to_string(), self.state.sharpe_ratio());
        info.insert("drawdown".to_string(), self.state.max_drawdown);
        info.insert("return".to_string(), self.state.cumulative_return);

        StepResult {
            next_state: self.state.clone(),
            reward,
            done,
            truncated: done,
            reward_source: source,
            info,
            constraint_cost,
        }
    }

    fn reset(&mut self) {
        let violated = !self.cmdp.constraints().all_satisfied();
        self.cmdp.end_episode(violated);
        self.state = FiduciaryState::initial(self.num_assets);
        self.step_num = 0;
        self.generate_market_data();
        let day = self.step_num.min(251);
        self.state.current_fraud_signal = if self.fraud_schedule[day] { 0.8 } else { -0.8 };
    }

    fn current_step(&self) -> usize {
        self.step_num
    }

    fn available_actions(&self) -> Vec<FiduciaryAction> {
        vec![
            FiduciaryAction::Hold,
            FiduciaryAction::Rebalance,
            FiduciaryAction::RequestDocument,
            FiduciaryAction::FlagFraud(self.step_num),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fiduciary_initial_state() {
        let env = FiduciaryEnv::new(4);
        let s = env.state();
        assert_eq!(s.portfolio_value, 100_000.0);
        assert_eq!(s.positions.len(), 4);
        assert_eq!(s.trading_day, 0);
    }

    #[test]
    fn test_fiduciary_hold_episode() {
        let mut env = FiduciaryEnv::new(4);
        let mut total_reward = 0.0;

        for _ in 0..252 {
            let result = env.step(FiduciaryAction::Hold);
            total_reward += result.reward;
            if result.done {
                break;
            }
        }

        let final_state = env.state();
        println!("  Fiduciary Hold episode:");
        println!("    Final value: ${:.2}", final_state.portfolio_value);
        println!("    Return: {:.2}%", final_state.cumulative_return * 100.0);
        println!("    Sharpe: {:.2}", final_state.sharpe_ratio());
        println!("    Max drawdown: {:.2}%", final_state.max_drawdown * 100.0);
        println!("    Total reward: {:.2}", total_reward);

        assert!(final_state.portfolio_value > 0.0);
        assert!(final_state.portfolio_value.is_finite());
    }

    #[test]
    fn test_fiduciary_fraud_detection() {
        let mut env = FiduciaryEnv::new(4);

        // Flag fraud every day
        for day in 0..20 {
            let result = env.step(FiduciaryAction::FlagFraud(day));
            assert!(result.reward.is_finite());
        }

        let s = env.state();
        assert!(s.frauds_detected > 0 || s.frauds_missed > 0);
    }

    #[test]
    fn test_fiduciary_sharpe() {
        let state = FiduciaryState {
            returns_history: vec![
                0.001, 0.002, -0.001, 0.003, 0.001, 0.002, -0.002, 0.001, 0.003, 0.002,
            ],
            ..FiduciaryState::initial(4)
        };
        let sharpe = state.sharpe_ratio();
        assert!(sharpe.is_finite());
        println!("  Sharpe ratio: {:.4}", sharpe);
    }

    #[test]
    fn test_fiduciary_cmdp_safety() {
        let env = FiduciaryEnv::new(4);
        // In safe exploration phase, only zero-cost actions allowed
        assert!(matches!(
            env.cmdp().phase(),
            CmdpPhase::SafeExploration { .. }
        ));
    }
}
