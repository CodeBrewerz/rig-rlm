//! AttnRes Benchmark: Empirical comparison of all 3 integration levels.
//!
//! Tests:
//! 1. mHC vs AttnRes for GNN depth-attention (embedding variance)
//! 2. Standard MLP vs AttnRes RL policy (learning speed)
//! 3. Pure AttnResOp depth-attention benchmark

use hehrgnn::eval::embedding_policy::EmbeddingPolicy;
use hehrgnn::eval::environment::{Environment, RewardSource, StepResult};
use hehrgnn::eval::rl_policy::Policy;
use hehrgnn::eval::rubric::{CriterionScorer, Rubric, RubricCriterion, RubricJudge};
use hehrgnn::eval::simulator::{run_episode, StateToFeatures};
use hehrgnn::eval::transition_buffer::TransitionBuffer;
use hehrgnn::model::attn_res_gnn::AttnResEmbeddingPolicy;
use std::collections::HashMap;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════
// DummyEnv: Transparent test environment
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct DummyState {
    pub target_action: usize,
    pub step: usize,
}

impl StateToFeatures for DummyState {
    fn to_features(&self) -> Vec<f32> {
        // Feature encodes the target action + noise
        vec![self.target_action as f32, 1.0 - self.target_action as f32]
    }
}

pub struct DummyEnv {
    state: DummyState,
    rng_seed: u64,
}

impl DummyEnv {
    pub fn new() -> Self {
        Self {
            state: DummyState {
                target_action: 0,
                step: 0,
            },
            rng_seed: 42,
        }
    }
    fn next_rng(&mut self) -> f64 {
        self.rng_seed = self
            .rng_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.rng_seed >> 33) as f64 / (1u64 << 31) as f64
    }
}

impl Environment for DummyEnv {
    type State = DummyState;
    type Action = usize;
    const MAX_STEPS: usize = 1;
    fn state(&self) -> Self::State {
        self.state.clone()
    }
    fn step(&mut self, action: Self::Action) -> StepResult<Self::State> {
        let mut info = HashMap::new();
        let score = if action == self.state.target_action {
            1.0
        } else {
            0.0
        };
        info.insert("score".to_string(), score);
        self.state.step += 1;
        StepResult {
            next_state: self.state.clone(),
            reward: score,
            done: true,
            truncated: true,
            reward_source: RewardSource::Verifiable,
            info,
            constraint_cost: 0.0,
        }
    }
    fn reset(&mut self) {
        self.state.step = 0;
        self.state.target_action = if self.next_rng() > 0.5 { 1 } else { 0 };
    }
    fn current_step(&self) -> usize {
        self.state.step
    }
    fn available_actions(&self) -> Vec<Self::Action> {
        vec![0, 1]
    }
}

// ═══════════════════════════════════════════════════════════════
// HarderEnv: Multi-feature pattern matching (harder task)
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct HarderState {
    pub features: Vec<f32>,
    pub target: usize,
    pub step: usize,
}

impl StateToFeatures for HarderState {
    fn to_features(&self) -> Vec<f32> {
        self.features.clone()
    }
}

pub struct HarderEnv {
    state: HarderState,
    rng_seed: u64,
    num_actions: usize,
    feature_dim: usize,
}

impl HarderEnv {
    pub fn new(num_actions: usize, feature_dim: usize) -> Self {
        Self {
            state: HarderState {
                features: vec![0.0; feature_dim],
                target: 0,
                step: 0,
            },
            rng_seed: 12345,
            num_actions,
            feature_dim,
        }
    }
    fn next_rng(&mut self) -> f64 {
        self.rng_seed = self
            .rng_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.rng_seed >> 33) as f64 / (1u64 << 31) as f64
    }
}

impl Environment for HarderEnv {
    type State = HarderState;
    type Action = usize;
    const MAX_STEPS: usize = 1;
    fn state(&self) -> Self::State {
        self.state.clone()
    }
    fn step(&mut self, action: Self::Action) -> StepResult<Self::State> {
        let mut info = HashMap::new();
        let score = if action == self.state.target {
            1.0
        } else {
            0.0
        };
        info.insert("score".to_string(), score);
        self.state.step += 1;
        StepResult {
            next_state: self.state.clone(),
            reward: score,
            done: true,
            truncated: true,
            reward_source: RewardSource::Verifiable,
            info,
            constraint_cost: 0.0,
        }
    }
    fn reset(&mut self) {
        self.state.step = 0;
        // Target is a nonlinear function of features
        self.state.target = (self.next_rng() * self.num_actions as f64).floor() as usize;
        self.state.target = self.state.target.min(self.num_actions - 1);

        // Generate features that encode the target through a pattern
        self.state.features = vec![0.0; self.feature_dim];
        for i in 0..self.feature_dim {
            self.state.features[i] = if i == self.state.target {
                0.8 + 0.2 * self.next_rng() as f32
            } else if i < self.num_actions {
                0.2 * self.next_rng() as f32
            } else {
                self.next_rng() as f32 * 0.5 // noise dimensions
            };
        }
    }
    fn current_step(&self) -> usize {
        self.state.step
    }
    fn available_actions(&self) -> Vec<Self::Action> {
        (0..self.num_actions).collect()
    }
}

// ═══════════════════════════════════════════════════════════════
// Benchmark Runner
// ═══════════════════════════════════════════════════════════════

struct BenchResult {
    name: String,
    zero_shot: f64,
    trained: f64,
    improvement: f64,
    training_time_ms: f64,
    rolling_scores: Vec<f64>,
}

fn bench_policy<E, P>(
    env: &mut E,
    policy: &mut P,
    name: &str,
    train_epochs: usize,
    eval_episodes: usize,
    train_fn: impl Fn(&mut P, usize, &[f32], f64), // (policy, action, state, reward)
) -> BenchResult
where
    E: Environment,
    E::State: StateToFeatures + Clone + std::fmt::Debug,
    E::Action: Into<usize> + From<usize> + Clone + std::fmt::Debug,
    P: Policy,
{
    // Zero-shot
    let mut untrained_sum = 0.0;
    for _ in 0..eval_episodes {
        let log = run_episode(env, policy, None, 1.0);
        untrained_sum += *log.info.get("score").unwrap_or(&0.0);
    }
    let zero_shot = untrained_sum / eval_episodes as f64;

    // Train
    let mut buffer = TransitionBuffer::new(10000);
    let mut rolling = zero_shot;
    let mut rolling_scores = Vec::new();
    let start = Instant::now();

    for ep in 1..=train_epochs {
        let log = run_episode(env, policy, Some(&mut buffer), 1.0);
        let score = *log.info.get("score").unwrap_or(&0.0);
        rolling = 0.95 * rolling + 0.05 * score;

        if buffer.len() >= log.steps {
            let batch = buffer.pop_last(log.steps);
            for i in 0..batch.states.len() {
                train_fn(policy, batch.action_ids[i], &batch.states[i], score);
            }
        }

        if ep % 500 == 0 {
            rolling_scores.push(rolling);
        }
    }
    let training_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Evaluate trained
    let mut trained_sum = 0.0;
    for _ in 0..eval_episodes {
        let log = run_episode(env, policy, None, 1.0);
        trained_sum += *log.info.get("score").unwrap_or(&0.0);
    }
    let trained = trained_sum / eval_episodes as f64;

    BenchResult {
        name: name.to_string(),
        zero_shot,
        trained,
        improvement: (trained - zero_shot) * 100.0,
        training_time_ms,
        rolling_scores,
    }
}

fn print_result(r: &BenchResult) {
    let status = if r.improvement > 5.0 {
        "✅"
    } else if r.improvement > 0.0 {
        "🟡"
    } else {
        "❌"
    };
    println!(
        "  {:30} │ {:6.1}% → {:6.1}% │ {:+6.1}% {} │ {:.0}ms",
        r.name,
        r.zero_shot * 100.0,
        r.trained * 100.0,
        r.improvement,
        status,
        r.training_time_ms
    );

    if !r.rolling_scores.is_empty() {
        let curve: Vec<String> = r
            .rolling_scores
            .iter()
            .map(|s| format!("{:.0}%", s * 100.0))
            .collect();
        println!("    └ curve: [{}]", curve.join(" → "));
    }
}

fn main() {
    println!("╔═════════════════════════════════════════════════════════════════════╗");
    println!("║  🧪 AttnRes Benchmark: Empirical Comparison (3 Integrations)      ║");
    println!("╚═════════════════════════════════════════════════════════════════════╝");

    let train_epochs = 3000;
    let eval_eps = 200;

    // ═══════════════════════════════════════════════════════════
    // TEST 1: Simple Binary Classification (2 actions, 2 features)
    // ═══════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  TEST 1: Simple Binary Task (2 actions, 2 features)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!(
        "  {:30} │ {:16} │ {:12} │ Time",
        "Model", "Zero → Trained", "Improvement"
    );
    println!("  ──────────────────────────────┼──────────────────┼──────────────┼──────");

    // Baseline: Standard 2-layer MLP
    {
        let mut env = DummyEnv::new();
        let mut policy = EmbeddingPolicy::new(2, 2);
        let r = bench_policy(
            &mut env,
            &mut policy,
            "Standard MLP (2-layer)",
            train_epochs,
            eval_eps,
            |p, action, state, reward| {
                p.reinforce_update(action, state, reward);
            },
        );
        print_result(&r);
    }

    // Integration 3: AttnRes RL Policy
    {
        let mut env = DummyEnv::new();
        let mut policy = AttnResEmbeddingPolicy::new(2, 2);
        let r = bench_policy(
            &mut env,
            &mut policy,
            "AttnRes Policy (2-layer+attn)",
            train_epochs,
            eval_eps,
            |p, action, state, reward| {
                p.reinforce_update(action, state, reward);
            },
        );
        print_result(&r);
    }

    // ═══════════════════════════════════════════════════════════
    // TEST 2: Harder Multi-Class Task (4 actions, 8 features)
    // ═══════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  TEST 2: Multi-Class Task (4 actions, 8 features)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!(
        "  {:30} │ {:16} │ {:12} │ Time",
        "Model", "Zero → Trained", "Improvement"
    );
    println!("  ──────────────────────────────┼──────────────────┼──────────────┼──────");

    {
        let mut env = HarderEnv::new(4, 8);
        let mut policy = EmbeddingPolicy::new(4, 8);
        let r = bench_policy(
            &mut env,
            &mut policy,
            "Standard MLP (2-layer)",
            train_epochs * 2,
            eval_eps,
            |p, action, state, reward| {
                p.reinforce_update(action, state, reward);
            },
        );
        print_result(&r);
    }

    {
        let mut env = HarderEnv::new(4, 8);
        let mut policy = AttnResEmbeddingPolicy::new(4, 8);
        let r = bench_policy(
            &mut env,
            &mut policy,
            "AttnRes Policy (2L+attn)",
            train_epochs * 2,
            eval_eps,
            |p, action, state, reward| {
                p.reinforce_update(action, state, reward);
            },
        );
        print_result(&r);
    }

    // ═══════════════════════════════════════════════════════════
    // TEST 3: Very Hard Task (8 actions, 16 features with noise)
    // ═══════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  TEST 3: Hard Task (8 actions, 16 features + noise)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!(
        "  {:30} │ {:16} │ {:12} │ Time",
        "Model", "Zero → Trained", "Improvement"
    );
    println!("  ──────────────────────────────┼──────────────────┼──────────────┼──────");

    {
        let mut env = HarderEnv::new(8, 16);
        let mut policy = EmbeddingPolicy::new(8, 16);
        let r = bench_policy(
            &mut env,
            &mut policy,
            "Standard MLP (2-layer)",
            train_epochs * 3,
            eval_eps,
            |p, action, state, reward| {
                p.reinforce_update(action, state, reward);
            },
        );
        print_result(&r);
    }

    {
        let mut env = HarderEnv::new(8, 16);
        let mut policy = AttnResEmbeddingPolicy::new(8, 16);
        let r = bench_policy(
            &mut env,
            &mut policy,
            "AttnRes Policy (2L+attn)",
            train_epochs * 3,
            eval_eps,
            |p, action, state, reward| {
                p.reinforce_update(action, state, reward);
            },
        );
        print_result(&r);
    }

    // ═══════════════════════════════════════════════════════════
    // TEST 4: GNN-level AttnRes (burn tensor benchmark)
    // ═══════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  TEST 4: AttnResOp Burn Tensor Benchmark (Integration 1 & 2)");
    println!("═══════════════════════════════════════════════════════════════════");

    test_burn_attnres_op();

    println!("\n╔═════════════════════════════════════════════════════════════════════╗");
    println!("║  📊 Benchmark Complete                                            ║");
    println!("╚═════════════════════════════════════════════════════════════════════╝");
}

fn test_burn_attnres_op() {
    use burn::backend::NdArray;
    type B = NdArray;
    let device = Default::default();

    // Test pure AttnResOp forward pass
    let config = attnres::AttnResConfig::new(64, 8, 2).with_num_heads(4);

    // Initialize AttnResOp
    let op: attnres::AttnResOp<B> = config.init_op(&device);

    // Create dummy blocks: 2 completed blocks of shape [batch=1, nodes=32, dim=64]
    let block1 = burn::tensor::Tensor::<B, 3>::random(
        [1, 32, 64],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let block2 = burn::tensor::Tensor::<B, 3>::random(
        [1, 32, 64],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let partial = burn::tensor::Tensor::<B, 3>::random(
        [1, 32, 64],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );

    // Benchmark forward pass
    let blocks = vec![block1.clone(), block2.clone()];

    let start = Instant::now();
    let n_iters = 1000;
    for _ in 0..n_iters {
        let _h = op.forward(&blocks, &partial);
    }
    let elapsed = start.elapsed();
    let per_iter_us = elapsed.as_micros() as f64 / n_iters as f64;

    println!("  AttnResOp forward (1×32×64, 2 blocks):");
    println!(
        "    {:.1} μs/iter ({} iters in {:.2}s)",
        per_iter_us,
        n_iters,
        elapsed.as_secs_f64()
    );

    // Compare with equivalent mHC Sinkhorn operation
    let sinkhorn_start = Instant::now();
    for _ in 0..n_iters {
        let raw = burn::tensor::Tensor::<B, 2>::random(
            [4, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let _normalized = hehrgnn::model::mhc::sinkhorn_normalize(raw, 5);
    }
    let sinkhorn_elapsed = sinkhorn_start.elapsed();
    let sinkhorn_us = sinkhorn_elapsed.as_micros() as f64 / n_iters as f64;

    println!("  mHC Sinkhorn (4×4, 5 iters):");
    println!(
        "    {:.1} μs/iter ({} iters in {:.2}s)",
        sinkhorn_us,
        n_iters,
        sinkhorn_elapsed.as_secs_f64()
    );

    // Output quality comparison: variance
    let h_attnres = op.forward(&blocks, &partial);
    let attnres_var: f32 = {
        let mean = h_attnres.clone().mean_dim(1); // mean across nodes
        let diff = h_attnres.clone() - mean;
        let var = (diff.clone() * diff).mean();
        var.into_data().as_slice::<f32>().unwrap()[0]
    };

    println!("\n  Output Signal Quality:");
    println!("    AttnRes output variance:  {:.6}", attnres_var);
    println!("    (Higher = richer representation, less over-smoothing)");

    // Test scaling: more blocks
    for n_blocks in [2, 4, 8, 16] {
        let test_blocks: Vec<_> = (0..n_blocks)
            .map(|_| {
                burn::tensor::Tensor::<B, 3>::random(
                    [1, 32, 64],
                    burn::tensor::Distribution::Normal(0.0, 1.0),
                    &device,
                )
            })
            .collect();

        let start = Instant::now();
        for _ in 0..100 {
            let _h = op.forward(&test_blocks, &partial);
        }
        let elapsed = start.elapsed();
        println!(
            "    {} blocks: {:.1} μs/iter",
            n_blocks,
            elapsed.as_micros() as f64 / 100.0
        );
    }
}
