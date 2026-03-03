//! GEPA Optimizer — GNN Training Hyperparameter Optimization
//!
//! Uses GEPA + Trinity to find optimal training hyperparameters for GraphSAGE:
//! - lr (learning rate)
//! - weight_decay (grokking cleanup phase)
//! - neg_ratio (negative samples per edge)
//! - perturb_frac (fraction of weights perturbed per epoch)
//!
//! Evaluator: trains a fresh GraphSAGE model with candidate params and scores by final AUC.

use burn::backend::NdArray;
use std::collections::HashMap;

use hehrgnn::data::graph_builder::{build_hetero_graph, GraphBuildConfig, GraphFact};
use hehrgnn::data::hetero_graph::EdgeType;
use hehrgnn::model::graphsage::GraphSageModelConfig;
use hehrgnn::model::trainer::*;
use hehrgnn::optimizer::gepa::*;

type B = NdArray;

fn gf(ht: &str, h: &str, r: &str, tt: &str, t: &str) -> GraphFact {
    GraphFact {
        src: (ht.to_string(), h.to_string()),
        relation: r.to_string(),
        dst: (tt.to_string(), t.to_string()),
    }
}

/// Build a financial graph for training evaluation.
fn build_training_graph() -> (Vec<GraphFact>, GraphBuildConfig) {
    let facts = vec![
        // User Alice with accounts
        gf("user", "alice", "has_account", "account", "checking"),
        gf("user", "alice", "has_account", "account", "savings"),
        gf("user", "alice", "has_account", "account", "credit_card"),
        gf("user", "alice", "has_account", "account", "investment"),
        // Transactions
        gf("account", "checking", "has_txn", "transaction", "t1"),
        gf("account", "checking", "has_txn", "transaction", "t2"),
        gf("account", "checking", "has_txn", "transaction", "t3"),
        gf("account", "credit_card", "has_txn", "transaction", "t4"),
        gf("account", "credit_card", "has_txn", "transaction", "t5"),
        gf("account", "savings", "has_txn", "transaction", "t6"),
        gf("account", "investment", "has_txn", "transaction", "t7"),
        gf("account", "investment", "has_txn", "transaction", "t8"),
        // Merchants
        gf("transaction", "t1", "at_merchant", "merchant", "grocery"),
        gf("transaction", "t2", "at_merchant", "merchant", "gas"),
        gf("transaction", "t3", "at_merchant", "merchant", "grocery"),
        gf("transaction", "t4", "at_merchant", "merchant", "restaurant"),
        gf("transaction", "t5", "at_merchant", "merchant", "online"),
        gf("transaction", "t6", "at_merchant", "merchant", "transfer"),
        gf("transaction", "t7", "at_merchant", "merchant", "brokerage"),
        gf("transaction", "t8", "at_merchant", "merchant", "brokerage"),
        // Categories
        gf("merchant", "grocery", "in_category", "category", "essentials"),
        gf("merchant", "gas", "in_category", "category", "essentials"),
        gf("merchant", "restaurant", "in_category", "category", "dining"),
        gf("merchant", "online", "in_category", "category", "shopping"),
        gf("merchant", "brokerage", "in_category", "category", "investing"),
        // Obligations
        gf("user", "alice", "has_obligation", "obligation", "mortgage"),
        gf("user", "alice", "has_obligation", "obligation", "car_loan"),
        gf("obligation", "mortgage", "linked_to", "account", "checking"),
        gf("obligation", "car_loan", "linked_to", "account", "checking"),
        // Goals
        gf("user", "alice", "has_goal", "goal", "retirement"),
        gf("user", "alice", "has_goal", "goal", "emergency_fund"),
        gf("user", "alice", "has_goal", "goal", "college_fund"),
        gf("goal", "retirement", "funded_by", "account", "savings"),
        gf("goal", "college_fund", "funded_by", "account", "investment"),
    ];
    let config = GraphBuildConfig {
        node_feat_dim: 16,
        add_reverse_edges: true,
        add_self_loops: true,
    };
    (facts, config)
}

// ═══════════════════════════════════════════════════════════════
// Training Hyperparameter Evaluator
// ═══════════════════════════════════════════════════════════════

/// Evaluates training hyperparameters by training a fresh GraphSAGE model
/// and scoring by final link prediction AUC.
///
/// Candidate parameters:
/// - lr: learning rate (0.001 to 0.3)
/// - weight_decay: regularization (0.001 to 0.1)
/// - neg_ratio: negative samples per positive edge (1 to 6, rounded)
/// - perturb_frac: fraction of weights perturbed (0.1 to 1.0)
struct TrainHyperEvaluator {
    facts: Vec<GraphFact>,
    build_config: GraphBuildConfig,
}

impl TrainHyperEvaluator {
    fn new() -> Self {
        let (facts, build_config) = build_training_graph();
        Self { facts, build_config }
    }
}

impl Evaluator for TrainHyperEvaluator {
    fn evaluate(&self, candidate: &Candidate) -> EvalResult {
        let device = <B as burn::prelude::Backend>::Device::default();
        let graph = build_hetero_graph::<B>(&self.facts, &self.build_config, &device);
        let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
        let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();

        // Parse candidate hyperparameters
        let lr = candidate.get_f32("lr", 0.05) as f64;
        let weight_decay = candidate.get_f32("weight_decay", 0.01) as f64;
        let neg_ratio = candidate.get_f32("neg_ratio", 3.0).round() as usize;
        let perturb_frac = candidate.get_f32("perturb_frac", 0.3) as f64;

        let config = TrainConfig {
            epochs: 30, // Fixed — we're optimizing the other params
            lr,
            neg_ratio: neg_ratio.max(1).min(8),
            patience: 10,
            perturb_frac,
            mode: TrainMode::Fast,
            weight_decay,
        };

        // Train a fresh model
        let mut model = GraphSageModelConfig {
            in_dim: 16,
            hidden_dim: 16,
            num_layers: 2,
            dropout: 0.0,
        }
        .init::<B>(&node_types, &edge_types, &device);

        let report = train_graphsage(&mut model, &graph, &config);

        // Score: final AUC is primary metric
        // Also track loss improvement and grokking progress
        let auc_score = report.final_auc as f64;
        let loss_improvement = (report.initial_loss - report.final_loss).max(0.0) as f64;
        let combined = auc_score * 0.7 + loss_improvement * 0.3;

        let mut side_info = SideInfo::new();
        side_info.score("final_auc", auc_score);
        side_info.score("loss_improvement", loss_improvement);
        side_info.score("weight_norm_sq", report.weight_norm_sq as f64);
        side_info.log(format!(
            "lr={:.4}, wd={:.4}, neg={}, pf={:.2} → AUC={:.4}, loss={:.4}→{:.4}, epochs={}",
            lr, weight_decay, neg_ratio, perturb_frac,
            report.final_auc, report.initial_loss, report.final_loss, report.epochs_trained
        ));

        EvalResult { score: combined, side_info }
    }
}

// ═══════════════════════════════════════════════════════════════
// Sync Test — NumericMutator baseline
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_gepa_optimize_training_hyperparams() {
    println!("\n  ╔══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("  ║  GEPA — GNN Training Hyperparameter Optimization                                 ║");
    println!("  ╠══════════════════════════════════════════════════════════════════════════════════════╣");

    let evaluator = TrainHyperEvaluator::new();

    let seed = Candidate::seed(vec![
        ("lr", "0.0500"),
        ("weight_decay", "0.0100"),
        ("neg_ratio", "3.0000"),
        ("perturb_frac", "0.3000"),
    ]);

    let seed_eval = evaluator.evaluate(&seed);
    println!("  ║  Seed: lr=0.05, wd=0.01, neg=3, pf=0.3");
    println!("  ║  Seed score: {:.6} (AUC={:.4})",
        seed_eval.score,
        seed_eval.side_info.scores.get("final_auc").unwrap_or(&0.0));
    println!("  ╠══════════════════════════════════════════════════════════════════════════════════════╣");

    let mutator = NumericMutator::new(0.3, 42);
    let config = OptimizeConfig {
        max_evals: 20,
        max_frontier_size: 8,
        log_every: 5,
        objective: "Optimize GNN training hyperparameters to maximize link prediction AUC".into(),
    };

    let result = optimize(seed, &evaluator, &mutator, config);

    println!("  ╠══════════════════════════════════════════════════════════════════════════════════════╣");
    println!("  ║  Best score: {:.6}  ({} evals, frontier={})", result.best_score, result.total_evals, result.frontier_size);
    println!("  ║  Best hyperparams:");
    let mut params: Vec<_> = result.best_candidate.params.iter().collect();
    params.sort_by_key(|(k, _)| k.clone());
    for (key, val) in &params {
        println!("  ║    {:16}: {}", key, val);
    }

    let improvement = result.best_score - seed_eval.score;
    if improvement > 0.0 {
        println!("  ║  ✅ Improved by {:.6} ({:.1}%)", improvement, improvement / seed_eval.score.abs().max(0.001) * 100.0);
    } else {
        println!("  ║  ℹ️  No improvement (defaults were near-optimal)");
    }
    println!("  ╚══════════════════════════════════════════════════════════════════════════════════════╝");

    assert!(result.total_evals >= 15);
    assert!(result.best_score.is_finite());
}

// ═══════════════════════════════════════════════════════════════
// Live LLM Test — Trinity-guided with persistence
// ═══════════════════════════════════════════════════════════════

/// Uses Trinity model to intelligently search training hyperparameter space.
///
/// Run: `cargo test -p hehrgnn --test gepa_training_test test_gepa_llm_training -- --ignored --nocapture`
#[tokio::test]
#[ignore]
async fn test_gepa_llm_training_hyperparams_with_trinity() {
    let weights_path = "/tmp/gepa_train_config.json";

    println!("\n  ╔══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("  ║  GEPA + TRINITY — GNN Training Hyperparameter Optimization (feedback loop)        ║");
    println!("  ╠══════════════════════════════════════════════════════════════════════════════════════╣");

    let objective = "Optimize GNN training hyperparameters for a financial heterogeneous graph. \
        Parameters: lr (learning rate, 0.001-0.3), weight_decay (regularization, 0.001-0.1), \
        neg_ratio (negative samples per positive edge, 1-6), perturb_frac (SPSA weight perturbation \
        fraction, 0.1-1.0). The model uses BPR link prediction loss with SPSA gradient estimation. \
        Goal: maximize link prediction AUC (how well the model predicts graph edges) and minimize \
        training loss. Higher lr → faster learning but risk divergence. Higher weight_decay → \
        stronger regularization (grokking cleanup). Higher neg_ratio → harder task. \
        Higher perturb_frac → more thorough gradient estimation per epoch.";

    let llm_mutator = match LlmMutator::from_env(objective) {
        Ok(m) => m,
        Err(e) => {
            println!("  ║  ⚠️  Skipping: {}", e);
            println!("  ╚══════════════════════════════════════════════════════════════════════════════════════╝");
            return;
        }
    };

    // Feedback loop: load previous best or defaults
    let prev_weights = OptimizedWeights::load_or_default(weights_path);
    let seed = if prev_weights.total_evals > 0 {
        println!("  ║  📂 Loaded previous best from {} (score={:.6}, evals={})",
            weights_path, prev_weights.score, prev_weights.total_evals);
        prev_weights.to_candidate()
    } else {
        println!("  ║  🆕 Starting from default hyperparameters");
        Candidate::seed(vec![
            ("lr", "0.0500"),
            ("weight_decay", "0.0100"),
            ("neg_ratio", "3.0000"),
            ("perturb_frac", "0.3000"),
        ])
    };

    let evaluator = TrainHyperEvaluator::new();
    let seed_eval = evaluator.evaluate(&seed);
    println!("  ║  Seed score: {:.6} (AUC={:.4})",
        seed_eval.score, seed_eval.side_info.scores.get("final_auc").unwrap_or(&0.0));
    println!("  ╠══════════════════════════════════════════════════════════════════════════════════════╣");

    let config = OptimizeConfig {
        max_evals: 10,  // Each eval trains a model (~1s), so 10 evals × ~4s per Trinity call ≈ 40s
        max_frontier_size: 5,
        log_every: 1,
        objective: objective.into(),
    };

    let result = optimize_async(seed, &evaluator, &llm_mutator, config).await;

    // Save best hyperparams
    let mut best_weights = OptimizedWeights::from_candidate(&result.best_candidate, result.best_score);
    best_weights.total_evals = prev_weights.total_evals + result.total_evals;
    match best_weights.save(weights_path) {
        Ok(()) => println!("  ║  💾 Saved to {} (cumulative evals={})", weights_path, best_weights.total_evals),
        Err(e) => println!("  ║  ⚠️  Save failed: {}", e),
    }

    println!("  ╠══════════════════════════════════════════════════════════════════════════════════════╣");
    println!("  ║  Best score: {:.6}  ({} evals, {} cumulative)",
        result.best_score, result.total_evals, best_weights.total_evals);
    println!("  ║  Best hyperparams (discovered by Trinity):");
    let mut params: Vec<_> = result.best_candidate.params.iter().collect();
    params.sort_by_key(|(k, _)| k.clone());
    for (key, val) in &params {
        println!("  ║    {:16}: {}", key, val);
    }

    let improvement = result.best_score - seed_eval.score;
    if improvement > 0.0 {
        println!("  ║  ✅ Trinity improved by {:.6} ({:.1}%)", improvement, improvement / seed_eval.score.abs().max(0.001) * 100.0);
    } else {
        println!("  ║  ℹ️  No improvement this run");
    }
    println!("  ║  🔄 Run again to continue optimizing from checkpoint!");
    println!("  ╚══════════════════════════════════════════════════════════════════════════════════════╝");

    assert!(result.total_evals >= 5);
    assert!(result.best_score.is_finite());
}
