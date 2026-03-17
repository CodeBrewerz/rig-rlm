use hehrgnn::eval::embedding_policy::EmbeddingPolicy;
use hehrgnn::eval::environment::{Environment, RewardSource, StepResult};
use hehrgnn::eval::fiduciary_env::FiduciaryEnv;
use hehrgnn::eval::rubric::{Rubric, RubricJudge};
use hehrgnn::eval::simulator::{run_episode, StateToFeatures};
use hehrgnn::eval::transition_buffer::TransitionBuffer;
use std::collections::HashMap;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct DummyState {
    pub target_action: usize,
    pub step: usize,
}

impl StateToFeatures for DummyState {
    fn to_features(&self) -> Vec<f32> {
        vec![self.target_action as f32] // single feature indicating the target action (0 or 1)
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
        // Give 1.0 if it guessed the target action, else 0.0
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
        vec![0, 1] // exactly 2 actions
    }
}

fn main() {
    println!("=======================================================================");
    println!(" 🧪 RL PROOF OF CONCEPT: Does the EmbeddingPolicy actually learn? ");
    println!("=======================================================================");

    let mut env = DummyEnv::new();
    let state_dim = env.state().to_features().len();
    let mut policy = EmbeddingPolicy::new(env.available_actions().len(), state_dim);

    // Use a real Rubric with a single criterion matching our dummy env's "score"
    let dummy_rubric = hehrgnn::eval::rubric::Rubric {
        criteria: vec![hehrgnn::eval::rubric::RubricCriterion {
            id: "score".to_string(),
            description: "Did the agent pick the correct action?".to_string(),
            weight: 1.0,
            pass_count: 0,
            eval_count: 0,
            scorer: hehrgnn::eval::rubric::CriterionScorer::Threshold {
                metric_key: "score".to_string(),
                threshold: 0.9,
            },
        }],
        version: 1,
        created_at_episode: 0,
    };
    let mut judge = RubricJudge::new(dummy_rubric);

    // ==========================================
    // Phase 1: Zero-Shot
    // ==========================================
    println!("\n▶ PHASE 1: Zero-Shot Evaluation (Untrained Policy)");

    let zero_eval_eps = 100;
    let mut untrained_score_sum = 0.0;
    for _ in 0..zero_eval_eps {
        let log = run_episode(&mut env, &mut policy, None, 1.0);
        untrained_score_sum += judge.score(&log.info);
    }
    let zero_shot_mean = untrained_score_sum / zero_eval_eps as f64;

    println!(
        "   Average Rubric Score (Untrained): {:.2}%",
        zero_shot_mean * 100.0
    );

    // --- PHASE 2: TRAINING (RLVR via REINFORCE) ---
    println!("\n▶ PHASE 2: Training agent using REINFORCE on Rubric Scores...");
    let train_eps = 3000;
    let mut rolling_score = zero_shot_mean;
    let mut buffer = TransitionBuffer::new(10000);

    let start = Instant::now();
    for ep in 1..=train_eps {
        let log = run_episode(&mut env, &mut policy, Some(&mut buffer), 1.0);
        let score = judge.score(&log.info);

        rolling_score = 0.95 * rolling_score + 0.05 * score;

        // Train from buffer using the Rubric Score as the target shaped reward for the full trajectory
        if buffer.len() >= log.steps {
            let batch = buffer.pop_last(log.steps);
            let mut shaped = Vec::new();
            for _ in &batch.rewards {
                shaped.push((score as f32 - 0.5));
            }
            policy.train_from_buffer(&batch.states, &batch.action_ids, &shaped);
        }

        if ep % 300 == 0 {
            println!(
                "   - Epoch {:4} | Rolling Score: {:6.2}%",
                ep,
                rolling_score * 100.0
            );
        }
    }
    println!(
        "   Training finished in {:.2}s\n",
        start.elapsed().as_secs_f64()
    );

    // --- PHASE 3: POST-TRAINING EVALUATION ---
    println!("▶ PHASE 3: Post-Training Evaluation (Trained Policy)");
    let mut trained_score_sum = 0.0;
    for _ in 0..zero_eval_eps {
        let log = run_episode(&mut env, &mut policy, None, 1.0);
        trained_score_sum += judge.score(&log.info);
    }
    let trained_mean = trained_score_sum / zero_eval_eps as f64;

    println!(
        "   Average Rubric Score (Trained): {:.2}%\n",
        trained_mean * 100.0
    );

    // --- PHASE 4: CONCLUSION ---
    println!("=======================================================================");
    println!(" 📈 FINAL VERDICT ");
    println!("=======================================================================");
    println!(" Baseline (Untrained): {:.2}%", zero_shot_mean * 100.0);
    println!(" Final    (Trained):   {:.2}%", trained_mean * 100.0);

    let improvement = (trained_mean - zero_shot_mean) * 100.0;
    if improvement > 5.0 {
        println!(" Improvement:          +{:.2}% ✅", improvement);
        println!("\n CONCLUSION: The model SUCCESSFULLY learned to optimize the reward!");
    } else {
        println!(" Improvement:          {:.2}% ❌", improvement);
        println!("\n CONCLUSION: The model did not learn effectively in this simulated run.");
    }
    println!("=======================================================================");
}
