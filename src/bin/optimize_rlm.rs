use hehrgnn::optimizer::gepa::{Candidate, LlmMutator, OptimizeConfig, optimize_async};
use rig_rlm::lambda::gepa_rlm::{LambdaExecutorEvaluator, trinity_provider};

#[tokio::main]
async fn main() {
    println!("🚀 Starting GEPA-RLM Persistent Optimization Daemon");

    let provider = trinity_provider();
    
    // Construct a massive generic dataset for benchmarking
    let mut massive_context = String::new();
    for _ in 0..100 {
        massive_context.push_str("Filler document full of useless financial data. Total noise. Nothing to see here.\n");
    }
    massive_context.push_str("\n\nThe secret anomaly is located in Account x-999.\n\n");
    for _ in 0..100 {
        massive_context.push_str("More filler data. Ignore it. Completely irrelevant.\n");
    }

    let evaluator = LambdaExecutorEvaluator {
        provider,
        query: "Identify the secret anomaly account.".to_string(),
        massive_context,
        expected_keyword: "x-999".to_string(),
    };

    // The seed candidate (current hardcoded defaults)
    let seed = Candidate::seed(vec![
        ("k_star", "2.0"),
        ("tau_star", "1500.0"),
    ]);

    // Initialize the LlmMutator (Trinity via OpenRouter)
    let mutator = LlmMutator::from_env(
        "Optimize the λ-RLM topological execution parameters (k_star and tau_star). \
         You want to maximize 'accuracy' (finding the secret phrase) while maximizing 'efficiency' \
         by minimizing the depth of the Map-Reduce String Diagram log."
    ).expect("Failed to initialize LlmMutator from environment vars");

    let config = OptimizeConfig {
        max_evals: 10, // Small bound for safety
        max_frontier_size: 5,
        log_every: 1,
        objective: "Optimize λ-RLM hyperparameters".to_string(),
    };

    println!("🔮 Evolving String Diagram Topology... (Press Ctrl+C to abort)");
    
    // Run the Pareto optimization async loop!
    let result = optimize_async(seed, &evaluator, &mutator, config).await;

    println!("\n✅ Optimization Complete!");
    println!("🏆 Best Candidate Score: {:.4}", result.best_score);
    println!("🎯 Optimal Parameters found:\n{}", result.best_candidate.to_text());
}
