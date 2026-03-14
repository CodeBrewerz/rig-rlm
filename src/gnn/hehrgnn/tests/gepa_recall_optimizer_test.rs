//! GEPA Recall Optimizer — automatically finds optimal fiduciary priority weights.
//!
//! Uses GEPA's `optimize_anything` pattern to search for per-action priority
//! weights that maximize archetype recall across the 1000-user generalization
//! benchmark. Follows GEPA best practices:
//!
//! 1. Rich SideInfo with per-archetype recall diagnostics
//! 2. Multi-objective Pareto: mean_recall, min_recall, zero_misalignment
//! 3. Two phases: NumericMutator (fast, no API) + LlmMutator (ignored, needs key)

use std::collections::HashMap;

use hehrgnn::eval::fiduciary::*;
use hehrgnn::optimizer::gepa::*;

// ═══════════════════════════════════════════════════════════════
// Archetype definitions (same as fiduciary_generalization_test)
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy)]
enum UserArchetype {
    DrowningInDebt,
    DebtPaydownActive,
    NewlyDebtFree,
    GoalBuilder,
    TaxOptimizer,
    WellManaged,
    HighNetWorth,
    FinanciallyFree,
    FraudVictim,
    SubscriptionCreep,
}

impl UserArchetype {
    fn all() -> Vec<Self> {
        vec![
            Self::DrowningInDebt,
            Self::DebtPaydownActive,
            Self::NewlyDebtFree,
            Self::GoalBuilder,
            Self::TaxOptimizer,
            Self::WellManaged,
            Self::HighNetWorth,
            Self::FinanciallyFree,
            Self::FraudVictim,
            Self::SubscriptionCreep,
        ]
    }

    fn name(&self) -> &'static str {
        match self {
            Self::DrowningInDebt => "DrowningInDebt",
            Self::DebtPaydownActive => "DebtPaydownActive",
            Self::NewlyDebtFree => "NewlyDebtFree",
            Self::GoalBuilder => "GoalBuilder",
            Self::TaxOptimizer => "TaxOptimizer",
            Self::WellManaged => "WellManaged",
            Self::HighNetWorth => "HighNetWorth",
            Self::FinanciallyFree => "FinanciallyFree",
            Self::FraudVictim => "FraudVictim",
            Self::SubscriptionCreep => "SubscriptionCreep",
        }
    }

    fn required_action_types(&self) -> Vec<&'static str> {
        match self {
            Self::DrowningInDebt => vec!["should_refinance", "should_avoid"],
            Self::DebtPaydownActive => vec!["should_refinance"],
            Self::NewlyDebtFree => vec!["should_fund_goal"],
            Self::GoalBuilder => vec!["should_fund_goal"],
            Self::TaxOptimizer => vec!["should_prepare_tax", "should_claim_exemption"],
            Self::WellManaged => vec!["should_reconcile"],
            Self::HighNetWorth => vec!["should_revalue_asset", "should_pay_down_lien"],
            Self::FinanciallyFree => vec![],
            Self::FraudVictim => vec!["should_investigate", "should_avoid"],
            Self::SubscriptionCreep => vec!["should_cancel", "should_review_recurring"],
        }
    }

    fn forbidden_action_types(&self) -> Vec<&'static str> {
        match self {
            Self::DrowningInDebt => vec!["should_fund_goal"],
            Self::FinanciallyFree => vec!["should_investigate"],
            _ => vec![],
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Per-user evaluation
// ═══════════════════════════════════════════════════════════════

struct ArchetypeResult {
    recall: f32,
    misaligned: bool,
    missing_actions: Vec<String>,
}

/// Evaluate one user with a given config.
fn evaluate_user(
    archetype: UserArchetype,
    user_id: usize,
    dim: usize,
    config: &RecommendConfig,
) -> ArchetypeResult {
    let seed = user_id;

    // Generate embedding with archetype signal
    let mut user_emb: Vec<f32> = (0..dim)
        .map(|d| ((seed * 7 + d * 3) as f32 * 0.1).sin() * 0.3)
        .collect();

    match archetype {
        UserArchetype::DrowningInDebt => {
            user_emb[0] += 2.5;
            user_emb[1] += 2.0;
            user_emb[2] += 1.5;
        }
        UserArchetype::DebtPaydownActive => {
            user_emb[0] += 1.5;
            user_emb[1] += 1.0;
            user_emb[4] += 0.5;
        }
        UserArchetype::NewlyDebtFree => {
            user_emb[4] += 2.0;
            user_emb[5] += 1.5;
        }
        UserArchetype::GoalBuilder => {
            user_emb[4] += 2.5;
            user_emb[5] += 2.0;
            user_emb[6] += 1.5;
        }
        UserArchetype::TaxOptimizer => {
            user_emb[8] += 2.5;
            user_emb[9] += 2.0;
            user_emb[10] += 1.5;
        }
        UserArchetype::WellManaged => {
            for d in 0..dim {
                user_emb[d] += 0.5;
            }
        }
        UserArchetype::HighNetWorth => {
            user_emb[14] += 2.5;
            user_emb[15] += 2.0;
        }
        UserArchetype::FinanciallyFree => {
            for d in 0..dim {
                user_emb[d] += 0.8;
            }
        }
        UserArchetype::FraudVictim => {
            user_emb[12] += 2.0;
            user_emb[13] += 1.5;
        }
        UserArchetype::SubscriptionCreep => {
            user_emb[12] += 1.5;
            user_emb[13] += 1.0;
        }
    }

    // Build graph context using REAL TQL relations
    let mut embeddings: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    embeddings.insert("user".into(), vec![user_emb.clone()]);
    let mut node_names: HashMap<String, Vec<String>> = HashMap::new();
    node_names.insert("user".into(), vec![format!("{}_{}", archetype.name(), user_id)]);
    let mut node_counts: HashMap<String, usize> = HashMap::new();
    node_counts.insert("user".into(), 1);
    let mut edges: HashMap<(String, String, String), Vec<(usize, usize)>> = HashMap::new();
    let mut anomaly_map: HashMap<String, HashMap<String, Vec<f32>>> = HashMap::new();
    anomaly_map.insert("SAGE".into(), HashMap::new());

    let mut add = |node_type: &str, name: &str, relation: &str, anomaly: f32| {
        let node_id = node_counts.get(node_type).copied().unwrap_or(0);
        let emb: Vec<f32> = (0..dim)
            .map(|d| ((node_id * 11 + d * 5 + seed) as f32 * 0.13 + anomaly).sin())
            .collect();
        embeddings.entry(node_type.into()).or_default().push(emb);
        node_names.entry(node_type.into()).or_default().push(name.into());
        *node_counts.entry(node_type.into()).or_insert(0) += 1;
        anomaly_map
            .get_mut("SAGE")
            .unwrap()
            .entry(node_type.into())
            .or_default()
            .push(anomaly);
        edges
            .entry(("user".into(), relation.into(), node_type.into()))
            .or_default()
            .push((0, node_id));
    };

    // Add entities per archetype (identical to fiduciary_generalization_test)
    match archetype {
        UserArchetype::DrowningInDebt => {
            let rate = 0.18 + (user_id % 10) as f32 * 0.04;
            add("obligation", &format!("CC_{}APR_{}", (rate * 100.0) as u32, user_id), "obligation-has-interest-term", 0.5 + rate);
            if user_id % 3 == 0 {
                add("obligation", &format!("PersonalLoan_{}", user_id), "obligation-has-interest-term", 0.4);
            }
            add("instrument", &format!("Checking_{}", user_id), "user-has-instrument", 0.05);
        }
        UserArchetype::DebtPaydownActive => {
            add("obligation", &format!("CC_Paydown_{}", user_id), "obligation-has-interest-term", 0.35);
            add("goal", &format!("DebtFree_Goal_{}", user_id), "subledger-holds-goal-funds", 0.05);
            add("instrument", &format!("Savings_{}", user_id), "user-has-instrument", 0.05);
        }
        UserArchetype::NewlyDebtFree => {
            add("goal", &format!("EmergencyFund_{}", user_id), "subledger-holds-goal-funds", 0.1);
            add("instrument", &format!("Checking_{}", user_id), "user-has-instrument", 0.03);
            add("instrument", &format!("Savings_{}", user_id), "user-has-instrument", 0.03);
        }
        UserArchetype::GoalBuilder => {
            add("goal", &format!("Retirement_{}", user_id), "subledger-holds-goal-funds", 0.05);
            add("goal", &format!("HouseDown_{}", user_id), "subledger-holds-goal-funds", 0.05);
            add("budget-estimation", &format!("Budget_{}", user_id), "records-budget-estimation", 0.05);
            add("instrument", &format!("Investment_{}", user_id), "user-has-instrument", 0.03);
        }
        UserArchetype::TaxOptimizer => {
            add("tax-due-event", &format!("TaxDue_{}", user_id), "tax-liability-has-due-event", 0.2);
            add("tax-sinking-fund", &format!("TaxReserve_{}", user_id), "tax-sinking-fund-backed-by-account", 0.1);
            add("tax-exemption-certificate", &format!("Exemption_{}", user_id), "tax-party-has-exemption-certificate", 0.05);
            add("instrument", &format!("Checking_{}", user_id), "user-has-instrument", 0.03);
        }
        UserArchetype::WellManaged => {
            add("reconciliation-case", &format!("Recon_{}", user_id), "reconciliation-for-instrument", 0.15);
            add("budget-estimation", &format!("Budget_{}", user_id), "records-budget-estimation", 0.05);
            add("instrument", &format!("Checking_{}", user_id), "user-has-instrument", 0.03);
            add("goal", &format!("VacationFund_{}", user_id), "subledger-holds-goal-funds", 0.03);
        }
        UserArchetype::HighNetWorth => {
            add("asset", &format!("House_{}", user_id), "lien-on-asset", 0.05);
            add("asset-valuation", &format!("HouseVal_{}", user_id), "asset-has-valuation", 0.1);
            add("asset-valuation", &format!("PortfolioVal_{}", user_id), "asset-has-valuation", 0.08);
            add("instrument", &format!("Brokerage_{}", user_id), "user-has-instrument", 0.03);
        }
        UserArchetype::FinanciallyFree => {
            add("instrument", &format!("Checking_{}", user_id), "user-has-instrument", 0.02);
            add("instrument", &format!("Savings_{}", user_id), "user-has-instrument", 0.02);
            add("goal", &format!("Charity_{}", user_id), "subledger-holds-goal-funds", 0.02);
        }
        UserArchetype::FraudVictim => {
            let fraud_anomaly = 0.75 + (user_id % 5) as f32 * 0.05;
            add("user-merchant-unit", &format!("FraudMerchant_{}", user_id), "case-has-counterparty", fraud_anomaly);
            add("obligation", &format!("SuspiciousCharge_{}", user_id), "obligation-between-parties", 0.6);
            add("instrument", &format!("Checking_{}", user_id), "user-has-instrument", 0.05);
        }
        UserArchetype::SubscriptionCreep => {
            let n_subs = 3 + (user_id % 4);
            for s in 0..n_subs {
                let anomaly = if s < 2 { 0.35 } else { 0.08 };
                add("recurring-pattern", &format!("Sub_{}_{}", s, user_id), "pattern-owned-by", anomaly);
            }
            add("goal", &format!("EmergencyFund_{}", user_id), "subledger-holds-goal-funds", 0.1);
        }
    }

    // Evaluate with GEPA config overrides
    let ctx = FiduciaryContext {
        user_emb: &user_emb,
        embeddings: &embeddings,
        anomaly_scores: &anomaly_map,
        edges: &edges,
        node_names: &node_names,
        node_counts: &node_counts,
        user_type: "user".into(),
        user_id: 0,
        hidden_dim: dim,
    };

    let response = recommend_with_config(&ctx, None, Some(config));

    let recommended_types: Vec<String> = response
        .recommendations
        .iter()
        .filter(|r| r.is_recommended)
        .map(|r| r.action_type.clone())
        .collect();

    // Check required actions
    let mut required_found = 0;
    let mut missing = Vec::new();
    for req in archetype.required_action_types() {
        if recommended_types.iter().any(|a| a == req) {
            required_found += 1;
        } else {
            missing.push(req.to_string());
        }
    }
    let recall = if archetype.required_action_types().is_empty() {
        1.0
    } else {
        required_found as f32 / archetype.required_action_types().len() as f32
    };

    let misaligned = archetype
        .forbidden_action_types()
        .iter()
        .any(|forbidden| recommended_types.iter().any(|a| a == forbidden));

    ArchetypeResult {
        recall,
        misaligned,
        missing_actions: missing,
    }
}

// ═══════════════════════════════════════════════════════════════
// RecallEvaluator — GEPA evaluator with rich SideInfo
// ═══════════════════════════════════════════════════════════════

/// Evaluates a candidate weight configuration by running the fiduciary
/// recommendation system across all 10 archetypes × N users per archetype.
///
/// Follows GEPA best practices:
/// - Decompose outcomes into per-archetype recall + misalignment
/// - Expose trajectories: which specific actions are missing
/// - Rich textual SideInfo for LLM reflection
struct RecallEvaluator {
    users_per_archetype: usize,
    dim: usize,
}

impl RecallEvaluator {
    fn new(users_per_archetype: usize) -> Self {
        Self {
            users_per_archetype,
            dim: 16,
        }
    }

    /// Build a RecommendConfig from GEPA candidate.
    fn candidate_to_config(candidate: &Candidate) -> RecommendConfig {
        let axes = [
            candidate.get_f32("cost_weight", 0.25),
            candidate.get_f32("risk_weight", 0.25),
            candidate.get_f32("goal_weight", 0.15),
            candidate.get_f32("urgency_weight", 0.15),
            // conflict and reversibility stay at defaults for now
            0.10,
            0.10,
        ];
        // Normalize axes to sum to 1
        let sum: f32 = axes.iter().sum();
        let axes_normed = if sum > 1e-8 {
            [
                axes[0] / sum, axes[1] / sum, axes[2] / sum,
                axes[3] / sum, axes[4] / sum, axes[5] / sum,
            ]
        } else {
            [0.25, 0.25, 0.15, 0.15, 0.10, 0.10]
        };

        let mut priority_overrides = HashMap::new();
        for action in FiduciaryActionType::all() {
            let key = format!("prio_{}", action.name());
            let val = candidate.get_f32(&key, action.priority_weight());
            priority_overrides.insert(action.name().to_string(), val);
        }

        RecommendConfig {
            axes_weights: Some(axes_normed),
            priority_overrides,
        }
    }
}

impl Evaluator for RecallEvaluator {
    fn evaluate(&self, candidate: &Candidate) -> EvalResult {
        let config = Self::candidate_to_config(candidate);

        let mut per_archetype_recall: Vec<(String, f32)> = Vec::new();
        let mut per_archetype_missing: Vec<(String, Vec<String>)> = Vec::new();
        let mut total_misaligned = 0usize;
        let mut total_recall = 0.0f32;

        for archetype in UserArchetype::all() {
            let mut arch_recall_sum = 0.0f32;
            let mut arch_misaligned = 0usize;
            let mut arch_missing_all: Vec<String> = Vec::new();

            for user_idx in 0..self.users_per_archetype {
                let user_id = user_idx * 10 + 1; // spread across seed space
                let result = evaluate_user(archetype, user_id, self.dim, &config);
                arch_recall_sum += result.recall;
                if result.misaligned {
                    arch_misaligned += 1;
                }
                for m in &result.missing_actions {
                    if !arch_missing_all.contains(m) {
                        arch_missing_all.push(m.clone());
                    }
                }
            }

            let mean_recall = arch_recall_sum / self.users_per_archetype as f32;
            per_archetype_recall.push((archetype.name().to_string(), mean_recall));
            if !arch_missing_all.is_empty() {
                per_archetype_missing.push((archetype.name().to_string(), arch_missing_all));
            }
            total_misaligned += arch_misaligned;
            total_recall += mean_recall;
        }

        let num_archetypes = UserArchetype::all().len() as f32;
        let mean_recall = total_recall / num_archetypes;
        let min_recall = per_archetype_recall
            .iter()
            .map(|(_, r)| *r)
            .fold(f32::INFINITY, f32::min);
        let zero_misalign = if total_misaligned == 0 { 1.0 } else { 0.0 };

        // Combined score: weighted blend of objectives
        let score = mean_recall as f64 * 0.40
            + min_recall as f64 * 0.30
            + zero_misalign * 0.20
            + mean_recall as f64 * 0.10; // bonus for mean

        // ── Rich SideInfo — the key to GEPA's reflective mutations ──
        let mut side_info = SideInfo::new();
        side_info.score("mean_recall", mean_recall as f64);
        side_info.score("min_recall", min_recall as f64);
        side_info.score("zero_misalignment", zero_misalign);

        // Per-archetype scores for Pareto
        for (name, recall) in &per_archetype_recall {
            side_info.score(&format!("recall_{}", name), *recall as f64);
        }

        // Diagnostic logs (the "Actionable Side Information")
        side_info.log(format!(
            "MEAN RECALL: {:.1}%  MIN RECALL: {:.1}%  MISALIGNED: {}",
            mean_recall * 100.0,
            min_recall * 100.0,
            total_misaligned
        ));

        // Per-archetype breakdown — this is what the LLM needs to reason
        for (name, recall) in &per_archetype_recall {
            let status = if *recall >= 0.75 { "✅" } else { "❌" };
            side_info.log(format!(
                "  {} {} recall={:.0}%",
                status, name, recall * 100.0
            ));
        }

        // Actionable guidance: tell the LLM exactly what to fix
        for (name, missing) in &per_archetype_missing {
            side_info.log(format!(
                "  ACTION NEEDED: {} is missing [{}]",
                name,
                missing.join(", ")
            ));
            for m in missing {
                let prio_key = format!("prio_{}", m);
                let current_prio = candidate.get_f32(&prio_key, 0.0);
                side_info.log(format!(
                    "    → prio_{} is currently {:.4} — try raising it",
                    m, current_prio
                ));
            }
        }

        EvalResult { score, side_info }
    }
}

// ═══════════════════════════════════════════════════════════════
// Test 1: NumericMutator — fast, no API key needed
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_gepa_numeric_recall_optimizer() {
    println!(
        "\n  ╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "  ║  GEPA RECALL OPTIMIZER — Fiduciary Priority Weight Search (Numeric)                ║"
    );
    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════╣"
    );

    // Use 2 users per archetype (20 total) — with full PC circuit
    let evaluator = RecallEvaluator::new(2);

    // Seed: current defaults
    let seed = OptimizedWeights::default().to_candidate();
    let seed_eval = evaluator.evaluate(&seed);
    println!(
        "  ║  Seed score: {:.6} (mean_recall={:.1}%, min_recall={:.1}%)",
        seed_eval.score,
        seed_eval.side_info.scores.get("mean_recall").unwrap_or(&0.0) * 100.0,
        seed_eval.side_info.scores.get("min_recall").unwrap_or(&0.0) * 100.0,
    );
    for log in &seed_eval.side_info.logs {
        println!("  ║  {}", log);
    }
    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════╣"
    );

    // Run GEPA optimization
    let mutator = NumericMutator::new(0.20, 42);
    let config = OptimizeConfig {
        max_evals: 25,
        max_frontier_size: 8,
        log_every: 5,
        objective: "Optimize per-action priority weights to maximize recall across all 10 \
                    fiduciary user archetypes. Each action has a prio_<name> weight (0.01-1.0) \
                    that scales the fiduciary score. Goal: all archetypes ≥75% recall with 0 misalignment."
            .into(),
    };

    let result = optimize(seed, &evaluator, &mutator, config);

    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "  ║  Best score: {:.6}  (after {} evaluations, frontier={})",
        result.best_score, result.total_evals, result.frontier_size
    );

    // Show final evaluation with best weights
    let best_eval = evaluator.evaluate(&result.best_candidate);
    for log in &best_eval.side_info.logs {
        println!("  ║  {}", log);
    }

    // Show changed priority weights
    println!("  ║");
    println!("  ║  Changed priority weights (vs defaults):");
    let defaults = OptimizedWeights::default().to_candidate();
    let mut changes = Vec::new();
    for action in FiduciaryActionType::all() {
        let key = format!("prio_{}", action.name());
        let default_val = defaults.get_f32(&key, action.priority_weight());
        let best_val = result.best_candidate.get_f32(&key, action.priority_weight());
        if (best_val - default_val).abs() > 0.01 {
            changes.push((action.name(), default_val, best_val));
        }
    }
    for (name, default, best) in &changes {
        let dir = if best > default { "↑" } else { "↓" };
        println!(
            "  ║    prio_{:25}: {:.4} → {:.4} {}",
            name, default, best, dir
        );
    }

    // Improvement summary
    let improvement = result.best_score - seed_eval.score;
    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════╣"
    );
    if improvement > 0.0 {
        println!(
            "  ║  ✅ GEPA improved score by {:.6} ({:.1}%)",
            improvement,
            improvement / seed_eval.score.abs().max(0.001) * 100.0
        );
    } else {
        println!(
            "  ║  ℹ️  Seed weights were already near-optimal (Δ={:.6})",
            improvement
        );
    }

    // Save optimized weights
    let weights_path = "/tmp/gepa_recall_weights.json";
    let mut best_weights =
        OptimizedWeights::from_candidate(&result.best_candidate, result.best_score);
    best_weights.total_evals = result.total_evals;
    match best_weights.save(weights_path) {
        Ok(()) => println!("  ║  💾 Saved optimized weights to {}", weights_path),
        Err(e) => println!("  ║  ⚠️  Save failed: {}", e),
    }

    println!(
        "  ╚══════════════════════════════════════════════════════════════════════════════════════╝"
    );

    // Assertions
    assert!(result.total_evals >= 20, "Should complete at least 20 evaluations");
    assert!(result.best_score.is_finite(), "Best score should be finite");
    let final_mean = *best_eval.side_info.scores.get("mean_recall").unwrap_or(&0.0);
    let final_min = *best_eval.side_info.scores.get("min_recall").unwrap_or(&0.0);
    assert!(
        final_mean >= 0.65,
        "Mean recall should be ≥65% after GEPA optimization, got {:.1}%",
        final_mean * 100.0
    );
    println!(
        "\n  Final: mean_recall={:.1}%, min_recall={:.1}%",
        final_mean * 100.0,
        final_min * 100.0
    );
}

// ═══════════════════════════════════════════════════════════════
// Test 2: LlmMutator — uses Trinity for reflective mutations
// ═══════════════════════════════════════════════════════════════

#[tokio::test]
#[ignore]
async fn test_gepa_llm_recall_optimizer() {
    let weights_path = "/tmp/gepa_recall_weights.json";

    println!(
        "\n  ╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "  ║  GEPA + TRINITY — LLM-Guided Recall Optimization (with rich SideInfo)              ║"
    );
    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════╣"
    );

    let objective = "Optimize fiduciary priority weights. Each prio_<action> controls how \
        important that action type is in the final recommendation score. The system serves \
        10 user archetypes (DrowningInDebt, DebtPaydownActive, NewlyDebtFree, GoalBuilder, \
        TaxOptimizer, WellManaged, HighNetWorth, FinanciallyFree, FraudVictim, SubscriptionCreep). \
        Each archetype has required actions that MUST appear in recommendations. \
        Goal: ALL archetypes ≥75% recall, 0 misalignment. \
        Focus on archetypes with ❌ marks — they need their required action priorities raised.";

    let llm_mutator = match LlmMutator::from_env(objective) {
        Ok(m) => m,
        Err(e) => {
            println!("  ║  ⚠️  Skipping: {}", e);
            println!(
                "  ╚══════════════════════════════════════════════════════════════════════════════════════╝"
            );
            return;
        }
    };

    // Load previous best or use defaults
    let prev_weights = OptimizedWeights::load_or_default(weights_path);
    let seed = if prev_weights.total_evals > 0 {
        println!(
            "  ║  📂 Loaded previous best from {} (score={:.6}, evals={})",
            weights_path, prev_weights.score, prev_weights.total_evals
        );
        prev_weights.to_candidate()
    } else {
        println!("  ║  🆕 No previous weights found — starting from defaults");
        OptimizedWeights::default().to_candidate()
    };

    let evaluator = RecallEvaluator::new(5);
    let seed_eval = evaluator.evaluate(&seed);
    println!("  ║  Seed score: {:.6}", seed_eval.score);
    for log in &seed_eval.side_info.logs {
        println!("  ║  {}", log);
    }
    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════╣"
    );

    let config = OptimizeConfig {
        max_evals: 15,
        max_frontier_size: 8,
        log_every: 1,
        objective: objective.into(),
    };

    let result = optimize_async(seed, &evaluator, &llm_mutator, config).await;

    // Save best weights for next run (feedback loop)
    let mut best_weights =
        OptimizedWeights::from_candidate(&result.best_candidate, result.best_score);
    best_weights.total_evals = prev_weights.total_evals + result.total_evals;
    match best_weights.save(weights_path) {
        Ok(()) => println!(
            "  ║  💾 Saved best weights to {} (cumulative evals={})",
            weights_path, best_weights.total_evals
        ),
        Err(e) => println!("  ║  ⚠️  Save failed: {}", e),
    }

    // Final evaluation
    let best_eval = evaluator.evaluate(&result.best_candidate);
    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "  ║  Best score: {:.6}  (after {} LLM-guided evals, {} cumulative)",
        result.best_score, result.total_evals, best_weights.total_evals
    );
    for log in &best_eval.side_info.logs {
        println!("  ║  {}", log);
    }

    println!("  ║");
    println!("  ║  🔄 Run again to continue optimizing from this checkpoint!");
    println!(
        "  ╚══════════════════════════════════════════════════════════════════════════════════════╝"
    );

    assert!(result.total_evals >= 5, "Should complete at least 5 LLM-guided evaluations");
    assert!(result.best_score.is_finite(), "Best score should be finite");
}
