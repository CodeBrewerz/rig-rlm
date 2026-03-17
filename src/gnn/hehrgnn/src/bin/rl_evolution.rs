use hehrgnn::eval::agent_env::AgentEnv;
use hehrgnn::eval::embedding_policy::EmbeddingPolicy;
use hehrgnn::eval::environment::Environment;
use hehrgnn::eval::rubric::{Rubric, RubricEvolver, RubricJudge};
use hehrgnn::eval::simulator::run_episode;
use hehrgnn::eval::transition_buffer::TransitionBuffer;
use std::time::Instant;

fn main() {
    println!("============================================================");
    println!(" Starting RL Evolution Loop (RLER + RLVR + Embedding Policy)");
    println!("============================================================");

    // Initialize environment and policy
    let mut env = AgentEnv::new(); // Safe RL testbed
    let state_dim = 16;
    let mut policy = EmbeddingPolicy::new(env.available_actions().len(), state_dim);
    let mut buffer = TransitionBuffer::new(10000);
    
    // Initialize Rubric (RLVR) and Evolver (RLER)
    let mut judge = RubricJudge::new(Rubric::agent_default());
    let mut evolver = RubricEvolver::new(100, 0.05, 50); // evolve every 100 episodes
    
    let total_episodes = 2500;
    
    println!("\n[Initial Rubric v{}]", judge.rubric_version());
    for c in &judge.rubric().criteria {
        println!("  - {} (weight={:.2})", c.description, c.weight);
    }
    println!("\nTraining Begins...\n");
    
    let start_time = Instant::now();
    let mut rolling_reward = 0.0;
    let mut rolling_rubric = 0.0;
    
    for ep in 1..=total_episodes {
        // 1. Run episode in the environment
        let log = run_episode(&mut env, &mut policy, Some(&mut buffer), 50.0);
        
        // 2. Score with RLVR (Verifiable Reward mapped to [0,1])
        let rubric_score: f64 = judge.score(&log.info);
        
        // 3. Optional: Map composite reward = Env Reward + scaled rubric
        let composite_reward = log.total_reward + (rubric_score - 0.5) * 10.0;
        
        // 4. Train via Replay Buffer (REINFORCE over Prototypes)
        if buffer.len() >= 64 {
            let batch = buffer.sample(64);
            
            // Apply shaped rewards to the batch to teach the policy RLVR constraints
            let mut shaped_rewards = Vec::new();
            for r in &batch.rewards {
                shaped_rewards.push(*r + ((rubric_score as f32 - 0.5) * 10.0) / 64.0);
            }
            
            policy.train_from_buffer(&batch.states, &batch.action_ids, &shaped_rewards);
        }
        
        evolver.record_episode();
        rolling_reward = 0.95 * rolling_reward + 0.05 * log.total_reward;
        rolling_rubric = 0.95 * rolling_rubric + 0.05 * rubric_score;
        
        // 5. Check if we need to evolve the rubric
        if evolver.should_evolve(judge.num_scored()) {
            println!("------------------------------------------------------------");
            println!("Episode {} | Evaluating Rubric Evolution...", ep);
            
            let new_rubric = evolver.evolve(&judge);
            
            if new_rubric.version > judge.rubric_version() {
                println!(">> Rubric Evolved -> v{}", new_rubric.version);
                println!(">> Remaining Criteria & New Auto-Discovered Constraints:");
                for c in &new_rubric.criteria {
                    println!("   - {} [{}] (weight: {:.3})", c.description, c.id, c.weight);
                }
                judge.update_rubric(new_rubric);
            } else {
                println!(">> No evolution criteria met. Saturated rules kept.");
            }
            println!(">> Current Policy Baseline: {:.3}", policy.baseline());
            println!("------------------------------------------------------------\n");
        }
        
        if ep % 200 == 0 {
            println!("Ep {:4} | Rolling Env Reward: {:6.2} | Rolling Rubric: {:.3} | BufLen: {}", 
                ep, rolling_reward, rolling_rubric, buffer.len());
        }
    }
    
    println!("\n============================================================");
    println!("Training Finished: {} episodes in {:.2?}", total_episodes, start_time.elapsed());
    println!("Final Policy Baseline: {:.3}", policy.baseline());
    println!("Final Rubric Version: v{}", judge.rubric_version());
    println!("============================================================");
}
