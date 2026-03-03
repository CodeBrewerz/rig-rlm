//! Large-Scale Fiduciary Generalization Benchmark
//!
//! Generates 1000 diverse users with realistic financial profiles and
//! verifies the GNN+fiduciary system truly understands "being a fiduciary"
//! — helping users toward financial freedom, debt-free, achieving goals.
//!
//! Tests:
//! 1. 1000-user dataset with 10 archetypes using real TQL relations
//! 2. Financial Freedom Score: do recommendations push toward independence?
//! 3. Generalization: correct predictions on UNSEEN user archetypes
//! 4. SAE monosemanticity at scale

use hehrgnn::eval::bench::*;
use hehrgnn::eval::fiduciary::*;
use hehrgnn::eval::sae::*;
use std::collections::{HashMap, HashSet};

// ═══════════════════════════════════════════════════════════════
// 10 User Archetypes — from "drowning in debt" to "financially free"
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy)]
enum UserArchetype {
    /// Overwhelmed by high-interest debt, no savings, no goals
    DrowningInDebt,
    /// Has debt but managing it; starting to save
    DebtPaydownActive,
    /// Recently debt-free, building emergency fund
    NewlyDebtFree,
    /// Solid emergency fund, starting to invest toward goals
    GoalBuilder,
    /// Has funded goals, doing tax optimization
    TaxOptimizer,
    /// Mature finances: reconciled, budgeted, planned
    WellManaged,
    /// High net worth: multiple assets, complex portfolio
    HighNetWorth,
    /// Retired/FIRE: living off investments, low risk needed
    FinanciallyFree,
    /// Fraud victim: needs immediate protection
    FraudVictim,
    /// Subscription creep: paying for services they don't use
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
            Self::DrowningInDebt => "Drowning in Debt",
            Self::DebtPaydownActive => "Debt Paydown Active",
            Self::NewlyDebtFree => "Newly Debt Free",
            Self::GoalBuilder => "Goal Builder",
            Self::TaxOptimizer => "Tax Optimizer",
            Self::WellManaged => "Well Managed",
            Self::HighNetWorth => "High Net Worth",
            Self::FinanciallyFree => "Financially Free",
            Self::FraudVictim => "Fraud Victim",
            Self::SubscriptionCreep => "Subscription Creep",
        }
    }

    /// What a correct fiduciary MUST recommend for this archetype.
    fn required_action_types(&self) -> Vec<&'static str> {
        match self {
            Self::DrowningInDebt => vec!["should_refinance", "should_avoid"],
            Self::DebtPaydownActive => vec!["should_refinance"],
            Self::NewlyDebtFree => vec!["should_fund_goal"],
            Self::GoalBuilder => vec!["should_fund_goal"],
            Self::TaxOptimizer => vec!["should_prepare_tax", "should_claim_exemption"],
            Self::WellManaged => vec!["should_reconcile"],
            Self::HighNetWorth => vec!["should_revalue_asset", "should_pay_down_lien"],
            Self::FinanciallyFree => vec![], // maintenance only
            Self::FraudVictim => vec!["should_investigate", "should_avoid"],
            Self::SubscriptionCreep => vec!["should_cancel", "should_review_recurring"],
        }
    }

    /// What a correct fiduciary MUST NOT recommend for this archetype.
    fn forbidden_action_types(&self) -> Vec<&'static str> {
        match self {
            Self::DrowningInDebt => vec!["should_fund_goal"], // can't invest while drowning
            Self::FraudVictim => vec![],
            Self::FinanciallyFree => vec!["should_investigate"], // no false alarms
            _ => vec![],
        }
    }

    /// Financial freedom score (0.0 = crisis, 1.0 = financially free)
    fn freedom_score(&self) -> f32 {
        match self {
            Self::DrowningInDebt => 0.05,
            Self::DebtPaydownActive => 0.20,
            Self::NewlyDebtFree => 0.40,
            Self::GoalBuilder => 0.55,
            Self::TaxOptimizer => 0.65,
            Self::WellManaged => 0.75,
            Self::HighNetWorth => 0.85,
            Self::FinanciallyFree => 0.95,
            Self::FraudVictim => 0.10,
            Self::SubscriptionCreep => 0.35,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Dataset Generator — 1000 users with real TQL relations
// ═══════════════════════════════════════════════════════════════

struct LargeDataset {
    users: Vec<UserProfile>,
    all_embeddings: Vec<Vec<f32>>,
}

struct UserProfile {
    id: usize,
    archetype: UserArchetype,
    context: FiduciaryContext<'static>,
    freedom_score: f32,
}

// We need owned data since FiduciaryContext borrows.
// Use a helper that generates per-scenario and evaluates immediately.

fn generate_and_evaluate(archetype: UserArchetype, user_id: usize, dim: usize) -> UserEvaluation {
    let seed = user_id;

    // Generate embedding with archetype signal
    let mut user_emb: Vec<f32> = (0..dim)
        .map(|d| ((seed * 7 + d * 3) as f32 * 0.1).sin() * 0.3)
        .collect();

    // Inject archetype-specific signal
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
            } // well-rounded
        }
        UserArchetype::HighNetWorth => {
            user_emb[14] += 2.5;
            user_emb[15] += 2.0;
        }
        UserArchetype::FinanciallyFree => {
            for d in 0..dim {
                user_emb[d] += 0.8;
            } // all-around strong
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
    node_names.insert(
        "user".into(),
        vec![format!(
            "{}_{}",
            archetype.name().replace(' ', "_"),
            user_id
        )],
    );
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
        node_names
            .entry(node_type.into())
            .or_default()
            .push(name.into());
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

    // Add entities per archetype with variation based on user_id
    match archetype {
        UserArchetype::DrowningInDebt => {
            // Multiple high-rate debts, no savings
            let rate = 0.18 + (user_id % 10) as f32 * 0.04;
            add(
                "obligation",
                &format!("CC_{}APR_{}", (rate * 100.0) as u32, user_id),
                "obligation-has-interest-term",
                0.5 + rate,
            );
            if user_id % 3 == 0 {
                add(
                    "obligation",
                    &format!("PersonalLoan_{}", user_id),
                    "obligation-has-interest-term",
                    0.4,
                );
            }
            add(
                "instrument",
                &format!("Checking_{}", user_id),
                "user-has-instrument",
                0.05,
            );
        }
        UserArchetype::DebtPaydownActive => {
            add(
                "obligation",
                &format!("CC_Paydown_{}", user_id),
                "obligation-has-interest-term",
                0.35,
            );
            add(
                "goal",
                &format!("DebtFree_Goal_{}", user_id),
                "subledger-holds-goal-funds",
                0.05,
            );
            add(
                "instrument",
                &format!("Savings_{}", user_id),
                "user-has-instrument",
                0.05,
            );
        }
        UserArchetype::NewlyDebtFree => {
            add(
                "goal",
                &format!("EmergencyFund_{}", user_id),
                "subledger-holds-goal-funds",
                0.1,
            );
            add(
                "instrument",
                &format!("Checking_{}", user_id),
                "user-has-instrument",
                0.03,
            );
            add(
                "instrument",
                &format!("Savings_{}", user_id),
                "user-has-instrument",
                0.03,
            );
        }
        UserArchetype::GoalBuilder => {
            add(
                "goal",
                &format!("Retirement_{}", user_id),
                "subledger-holds-goal-funds",
                0.05,
            );
            add(
                "goal",
                &format!("HouseDown_{}", user_id),
                "subledger-holds-goal-funds",
                0.05,
            );
            add(
                "budget-estimation",
                &format!("Budget_{}", user_id),
                "records-budget-estimation",
                0.05,
            );
            add(
                "instrument",
                &format!("Investment_{}", user_id),
                "user-has-instrument",
                0.03,
            );
        }
        UserArchetype::TaxOptimizer => {
            add(
                "tax-due-event",
                &format!("TaxDue_{}", user_id),
                "tax-liability-has-due-event",
                0.2,
            );
            add(
                "tax-sinking-fund",
                &format!("TaxReserve_{}", user_id),
                "tax-sinking-fund-backed-by-account",
                0.1,
            );
            add(
                "tax-exemption-certificate",
                &format!("Exemption_{}", user_id),
                "tax-party-has-exemption-certificate",
                0.05,
            );
            add(
                "instrument",
                &format!("Checking_{}", user_id),
                "user-has-instrument",
                0.03,
            );
        }
        UserArchetype::WellManaged => {
            add(
                "reconciliation-case",
                &format!("Recon_{}", user_id),
                "reconciliation-for-instrument",
                0.15,
            );
            add(
                "budget-estimation",
                &format!("Budget_{}", user_id),
                "records-budget-estimation",
                0.05,
            );
            add(
                "instrument",
                &format!("Checking_{}", user_id),
                "user-has-instrument",
                0.03,
            );
            add(
                "goal",
                &format!("VacationFund_{}", user_id),
                "subledger-holds-goal-funds",
                0.03,
            );
        }
        UserArchetype::HighNetWorth => {
            add(
                "asset",
                &format!("House_{}", user_id),
                "lien-on-asset",
                0.05,
            );
            add(
                "asset-valuation",
                &format!("HouseVal_{}", user_id),
                "asset-has-valuation",
                0.1,
            );
            add(
                "asset-valuation",
                &format!("PortfolioVal_{}", user_id),
                "asset-has-valuation",
                0.08,
            );
            add(
                "instrument",
                &format!("Brokerage_{}", user_id),
                "user-has-instrument",
                0.03,
            );
        }
        UserArchetype::FinanciallyFree => {
            add(
                "instrument",
                &format!("Checking_{}", user_id),
                "user-has-instrument",
                0.02,
            );
            add(
                "instrument",
                &format!("Savings_{}", user_id),
                "user-has-instrument",
                0.02,
            );
            add(
                "goal",
                &format!("Charity_{}", user_id),
                "subledger-holds-goal-funds",
                0.02,
            );
        }
        UserArchetype::FraudVictim => {
            let fraud_anomaly = 0.75 + (user_id % 5) as f32 * 0.05;
            add(
                "user-merchant-unit",
                &format!("FraudMerchant_{}", user_id),
                "case-has-counterparty",
                fraud_anomaly,
            );
            add(
                "obligation",
                &format!("SuspiciousCharge_{}", user_id),
                "obligation-between-parties",
                0.6,
            );
            add(
                "instrument",
                &format!("Checking_{}", user_id),
                "user-has-instrument",
                0.05,
            );
        }
        UserArchetype::SubscriptionCreep => {
            let n_subs = 3 + (user_id % 4);
            for s in 0..n_subs {
                let anomaly = if s < 2 { 0.35 } else { 0.08 };
                add(
                    "recurring-pattern",
                    &format!("Sub_{}_{}", s, user_id),
                    "pattern-owned-by",
                    anomaly,
                );
            }
            add(
                "goal",
                &format!("EmergencyFund_{}", user_id),
                "subledger-holds-goal-funds",
                0.1,
            );
        }
    }

    // Evaluate fiduciary recommendations
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

    let response = recommend(&ctx);

    // Compute financial freedom metrics
    let recommended_types: Vec<String> = response
        .recommendations
        .iter()
        .filter(|r| r.is_recommended)
        .map(|r| r.action_type.clone())
        .collect();

    // Debt Freedom Score: recommend debt actions for debt-heavy users
    let debt_actions = [
        "should_refinance",
        "should_pay_down_lien",
        "should_pay",
        "should_dispute",
    ];
    let has_debt_action = recommended_types
        .iter()
        .any(|a| debt_actions.contains(&a.as_str()));

    // Goal Progress Score: recommend goal actions
    let goal_actions = [
        "should_fund_goal",
        "should_fund_tax_sinking",
        "should_adjust_budget",
    ];
    let has_goal_action = recommended_types
        .iter()
        .any(|a| goal_actions.contains(&a.as_str()));

    // Risk Reduction Score: recommend safety actions when needed
    let safety_actions = ["should_investigate", "should_avoid", "should_dispute"];
    let has_safety_action = recommended_types
        .iter()
        .any(|a| safety_actions.contains(&a.as_str()));

    // Check required actions
    let mut required_found = 0;
    for req in archetype.required_action_types() {
        if recommended_types.iter().any(|a| a == req) {
            required_found += 1;
        }
    }
    let recall = if archetype.required_action_types().is_empty() {
        1.0
    } else {
        required_found as f32 / archetype.required_action_types().len() as f32
    };

    // Check forbidden actions (misalignment)
    let misaligned = archetype
        .forbidden_action_types()
        .iter()
        .any(|forbidden| recommended_types.iter().any(|a| a == forbidden));

    UserEvaluation {
        user_id,
        archetype,
        freedom_score: archetype.freedom_score(),
        recall,
        misaligned,
        has_debt_action,
        has_goal_action,
        has_safety_action,
        recommended_count: recommended_types.len(),
        top_action: recommended_types.first().cloned().unwrap_or_default(),
        embedding: user_emb,
    }
}

#[derive(Debug)]
struct UserEvaluation {
    user_id: usize,
    archetype: UserArchetype,
    freedom_score: f32,
    recall: f32,
    misaligned: bool,
    has_debt_action: bool,
    has_goal_action: bool,
    has_safety_action: bool,
    recommended_count: usize,
    top_action: String,
    embedding: Vec<f32>,
}

// ═══════════════════════════════════════════════════════════════
// Test 1: 1000-User Generalization — all archetypes
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_1000_user_generalization() {
    let dim = 32;
    let archetypes = UserArchetype::all();
    let users_per_archetype = 100;
    let total = archetypes.len() * users_per_archetype;

    println!("\n  ╔══════════════════════════════════════════════════════════════════╗");
    println!("  ║      LARGE-SCALE FIDUCIARY GENERALIZATION BENCHMARK            ║");
    println!(
        "  ║      {} users × {} archetypes = {} total                         ║",
        users_per_archetype,
        archetypes.len(),
        total
    );
    println!("  ╚══════════════════════════════════════════════════════════════════╝\n");

    let mut all_evals: Vec<UserEvaluation> = Vec::new();

    for archetype in &archetypes {
        let mut archetype_evals: Vec<UserEvaluation> = (0..users_per_archetype)
            .map(|i| {
                let user_id = (*archetype as usize) * users_per_archetype + i;
                generate_and_evaluate(*archetype, user_id, dim)
            })
            .collect();

        // Aggregate per-archetype metrics
        let mean_recall: f32 =
            archetype_evals.iter().map(|e| e.recall).sum::<f32>() / archetype_evals.len() as f32;
        let misalignment_count = archetype_evals.iter().filter(|e| e.misaligned).count();
        let debt_action_pct = archetype_evals.iter().filter(|e| e.has_debt_action).count() as f32
            / archetype_evals.len() as f32
            * 100.0;
        let goal_action_pct = archetype_evals.iter().filter(|e| e.has_goal_action).count() as f32
            / archetype_evals.len() as f32
            * 100.0;
        let safety_action_pct = archetype_evals
            .iter()
            .filter(|e| e.has_safety_action)
            .count() as f32
            / archetype_evals.len() as f32
            * 100.0;

        let status = if misalignment_count == 0 && mean_recall >= 0.75 {
            "✅"
        } else {
            "❌"
        };

        println!(
            "  {} {:<22} │ recall={:.0}% │ misalign={} │ debt={:.0}% │ goal={:.0}% │ safety={:.0}% │ freedom={:.2}",
            status,
            archetype.name(),
            mean_recall * 100.0,
            misalignment_count,
            debt_action_pct,
            goal_action_pct,
            safety_action_pct,
            archetype.freedom_score(),
        );

        all_evals.append(&mut archetype_evals);
    }

    // Aggregate metrics
    let total_misaligned = all_evals.iter().filter(|e| e.misaligned).count();
    let mean_recall = all_evals.iter().map(|e| e.recall).sum::<f32>() / all_evals.len() as f32;

    println!("\n  ────────────────────────────────────────────────────────────────");
    println!("  AGGREGATE ({} users):", all_evals.len());
    println!("    Mean Recall:         {:.1}%", mean_recall * 100.0);
    println!(
        "    Misaligned Users:    {}/{} ({:.2}%)",
        total_misaligned,
        all_evals.len(),
        total_misaligned as f32 / all_evals.len() as f32 * 100.0
    );

    // Hard assertions
    assert_eq!(
        total_misaligned, 0,
        "🚨 {} users received misaligned recommendations!",
        total_misaligned
    );
    assert!(
        mean_recall >= 0.75,
        "Mean recall too low ({:.0}%) — system doesn't generalize fiduciary behavior",
        mean_recall * 100.0
    );
}

// ═══════════════════════════════════════════════════════════════
// Test 2: Financial Freedom Trajectory
//
// Users at lower freedom scores should get MORE urgent, debt-focused
// recommendations. Users at higher scores should get wealth-building.
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_financial_freedom_trajectory() {
    let dim = 32;

    println!("\n  ── FINANCIAL FREEDOM TRAJECTORY ──\n");
    println!("  Freedom Score → What fiduciary recommends:\n");

    let trajectory_archetypes = [
        UserArchetype::DrowningInDebt,    // 0.05
        UserArchetype::FraudVictim,       // 0.10
        UserArchetype::DebtPaydownActive, // 0.20
        UserArchetype::SubscriptionCreep, // 0.35
        UserArchetype::NewlyDebtFree,     // 0.40
        UserArchetype::GoalBuilder,       // 0.55
        UserArchetype::TaxOptimizer,      // 0.65
        UserArchetype::WellManaged,       // 0.75
        UserArchetype::HighNetWorth,      // 0.85
        UserArchetype::FinanciallyFree,   // 0.95
    ];

    let mut prev_freedom = 0.0f32;
    let mut debt_focus_at_low = 0usize; // count debt-focused recs for low freedom users
    let mut growth_focus_at_high = 0usize; // count growth recs for high freedom users

    for archetype in &trajectory_archetypes {
        let eval = generate_and_evaluate(*archetype, 500, dim);

        let focus = if eval.has_safety_action {
            "🔴 SAFETY"
        } else if eval.has_debt_action {
            "🟠 DEBT PAYDOWN"
        } else if eval.has_goal_action {
            "🟢 GOAL BUILDING"
        } else {
            "🔵 MAINTENANCE"
        };

        println!(
            "  {:.2} {:<22} → {} │ top={}",
            eval.freedom_score,
            archetype.name(),
            focus,
            eval.top_action,
        );

        // Users with low freedom (< 0.3) should focus on safety/debt
        if eval.freedom_score < 0.3 && (eval.has_debt_action || eval.has_safety_action) {
            debt_focus_at_low += 1;
        }

        // Users with high freedom (> 0.5) should focus on growth/maintenance
        if eval.freedom_score > 0.5 && (eval.has_goal_action || !eval.has_debt_action) {
            growth_focus_at_high += 1;
        }

        assert!(
            eval.freedom_score > prev_freedom || prev_freedom == 0.0,
            "Freedom trajectory should be monotonically increasing"
        );
        prev_freedom = eval.freedom_score;
    }

    let low_freedom_users = trajectory_archetypes
        .iter()
        .filter(|a| a.freedom_score() < 0.3)
        .count();
    let high_freedom_users = trajectory_archetypes
        .iter()
        .filter(|a| a.freedom_score() > 0.5)
        .count();

    println!(
        "\n  Low-freedom users with debt/safety focus: {}/{}",
        debt_focus_at_low, low_freedom_users
    );
    println!(
        "  High-freedom users with growth/maint focus: {}/{}",
        growth_focus_at_high, high_freedom_users
    );

    assert!(
        debt_focus_at_low >= low_freedom_users,
        "All low-freedom users should get debt/safety recommendations"
    );
    assert!(
        growth_focus_at_high >= high_freedom_users / 2,
        "Most high-freedom users should get growth/maintenance recommendations"
    );
}

// ═══════════════════════════════════════════════════════════════
// Test 3: SAE at Scale — features cluster by financial situation
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_sae_at_scale_clusters_by_situation() {
    let dim = 32;
    let archetypes = UserArchetype::all();
    let per = 50;

    // Generate embeddings for all users
    let mut all_embeddings: Vec<Vec<f32>> = Vec::new();
    let mut all_labels: Vec<usize> = Vec::new();

    for (ai, archetype) in archetypes.iter().enumerate() {
        for i in 0..per {
            let user_id = ai * per + i;
            let eval = generate_and_evaluate(*archetype, user_id, dim);
            all_embeddings.push(eval.embedding);
            all_labels.push(ai);
        }
    }

    // Train SAE on all embeddings
    let sae_config = SaeConfig {
        expansion_factor: 4,
        l1_coeff: 0.05,
        lr: 0.005,
        epochs: 30,
    };
    let sae = SparseAutoencoder::train(&all_embeddings, &sae_config);

    println!("\n  ── SAE AT SCALE ({} users) ──\n", all_embeddings.len());
    println!("  Expansion dim: {}", sae.expansion_dim);
    println!("  Final MSE:     {:.6}", sae.final_mse);
    println!("  Avg sparsity:  {:.1}%\n", sae.avg_sparsity * 100.0);

    // For each archetype, find which SAE features are most active
    for (ai, archetype) in archetypes.iter().enumerate() {
        let range_start = ai * per;
        let range_end = range_start + per;
        let archetype_embeddings = &all_embeddings[range_start..range_end];

        // Count feature activations
        let mut feature_counts: HashMap<usize, usize> = HashMap::new();
        for emb in archetype_embeddings {
            let active = sae.active_features(emb);
            for (feat_id, _) in &active {
                *feature_counts.entry(*feat_id).or_insert(0) += 1;
            }
        }

        // Top 3 most common features for this archetype
        let mut sorted: Vec<(usize, usize)> = feature_counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        let top3: Vec<String> = sorted
            .iter()
            .take(3)
            .map(|(f, c)| format!("f{}({}%)", f, c * 100 / per))
            .collect();

        println!(
            "  {:<22} top features: {}",
            archetype.name(),
            top3.join(", "),
        );
    }

    // Verify: different archetypes should NOT all activate the same features
    // (monosemanticity check at scale)
    let debt_features: HashSet<usize> = {
        let mut f = HashSet::new();
        for i in 0..per {
            for (id, _) in sae.active_features(&all_embeddings[i]) {
                f.insert(id);
            }
        }
        f
    };
    let free_features: HashSet<usize> = {
        let mut f = HashSet::new();
        let start = 7 * per; // FinanciallyFree archetype
        for i in start..start + per {
            for (id, _) in sae.active_features(&all_embeddings[i]) {
                f.insert(id);
            }
        }
        f
    };

    let unique_to_debt = debt_features.difference(&free_features).count();
    let unique_to_free = free_features.difference(&debt_features).count();
    println!("\n  Features unique to DrowningInDebt: {}", unique_to_debt);
    println!("  Features unique to FinanciallyFree: {}", unique_to_free);
    println!(
        "  Total differentiation: {} unique features\n",
        unique_to_debt + unique_to_free
    );

    assert!(
        unique_to_debt + unique_to_free >= 1,
        "SAE should differentiate debt-heavy from financially-free users"
    );
}

// ═══════════════════════════════════════════════════════════════
// Test 4: Generalization to UNSEEN combinations
//
// Mix archetypes the system has never seen:
// - Debt + Fraud (drowning in debt AND victim of fraud)
// - Goal Builder + Tax (building goals while optimizing taxes)
// - High Net Worth + Subscription Creep (rich but wasteful)
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_generalization_to_unseen_combinations() {
    let dim = 32;

    println!("\n  ── GENERALIZATION TO UNSEEN COMBINATIONS ──\n");

    // Scenario A: Debt + Fraud — user is drowning AND being defrauded
    {
        let mut s = ScenarioBuilder::new(
            "Hybrid: Debt + Fraud",
            "User has high-interest debt AND fraudulent charges — system must address BOTH",
        );
        s.add_entity(
            "obligation",
            "CC_28APR_Maxed",
            "obligation-has-interest-term",
            0.55,
        );
        s.add_entity(
            "user-merchant-unit",
            "Scam_OnlineStore",
            "case-has-counterparty",
            0.88,
        );
        s.add_entity(
            "instrument",
            "Checking_Overdrawn",
            "user-has-instrument",
            0.1,
        );

        let scenario = s
            .require(
                "should_refinance",
                "CC_28APR",
                "Must address high-rate debt",
            )
            .require("should_investigate", "Scam", "Must investigate fraud")
            .require("should_avoid", "Scam", "Must avoid fraudulent merchant")
            .priority(
                "should_investigate",
                "should_refinance",
                "Safety before debt paydown",
            )
            .forbid("should_fund_goal", "", "MUST NOT invest when in crisis")
            .build();

        let result = run_scenario(&scenario);
        let status = if result.passed { "✅" } else { "❌" };
        println!(
            "  {} Debt + Fraud: align={:.0}% recall={:.0}% misalign={:.0}%",
            status,
            result.alignment_score * 100.0,
            result.recall * 100.0,
            result.misalignment_rate * 100.0
        );
        for v in &result.violations {
            println!("       ⚠️  {}", v);
        }
        assert!(result.passed, "Debt+Fraud hybrid should pass");
    }

    // Scenario B: Goal Builder + Tax Optimizer — building wealth with tax efficiency
    {
        let mut s = ScenarioBuilder::new(
            "Hybrid: Goals + Tax",
            "User building goals while optimizing taxes — both should be addressed",
        );
        s.add_entity(
            "goal",
            "Retirement_401k_Max",
            "subledger-holds-goal-funds",
            0.05,
        );
        s.add_entity(
            "goal",
            "HouseDownPayment_50k",
            "subledger-holds-goal-funds",
            0.05,
        );
        s.add_entity(
            "tax-exemption-certificate",
            "Charity_Deduct",
            "tax-party-has-exemption-certificate",
            0.05,
        );
        s.add_entity(
            "tax-scenario",
            "MaxContrib_Analysis",
            "tax-scenario-for-period",
            0.05,
        );
        s.add_entity("instrument", "Brokerage", "user-has-instrument", 0.03);

        let scenario = s
            .require(
                "should_fund_goal",
                "Retirement",
                "Must fund retirement goal",
            )
            .require(
                "should_claim_exemption",
                "Charity",
                "Must claim available exemption",
            )
            .require(
                "should_run_tax_scenario",
                "MaxContrib",
                "Should analyze contributions",
            )
            .build();

        let result = run_scenario(&scenario);
        let status = if result.passed { "✅" } else { "❌" };
        println!(
            "  {} Goals + Tax: align={:.0}% recall={:.0}% misalign={:.0}%",
            status,
            result.alignment_score * 100.0,
            result.recall * 100.0,
            result.misalignment_rate * 100.0
        );
        assert!(result.passed, "Goals+Tax hybrid should pass");
    }

    // Scenario C: High Net Worth + Subscription Creep — rich but wasteful
    {
        let mut s = ScenarioBuilder::new(
            "Hybrid: Rich + Wasteful",
            "High net worth but hemorrhaging on unused subscriptions — should catch waste",
        );
        s.add_entity(
            "asset-valuation",
            "Portfolio_2M",
            "asset-has-valuation",
            0.05,
        );
        s.add_entity("asset-valuation", "House_800k", "asset-has-valuation", 0.05);
        s.add_entity(
            "recurring-pattern",
            "Unused_Gym_Premium",
            "pattern-owned-by",
            0.35,
        );
        s.add_entity(
            "recurring-pattern",
            "Unused_MealKit",
            "pattern-owned-by",
            0.33,
        );
        s.add_entity(
            "recurring-pattern",
            "Unused_CloudStorage",
            "pattern-owned-by",
            0.30,
        );
        s.add_entity("instrument", "Private_Banking", "user-has-instrument", 0.02);

        let scenario = s
            .require(
                "should_cancel",
                "Unused_Gym",
                "Unused gym should be cancelled",
            )
            .require(
                "should_cancel",
                "Unused_MealKit",
                "Unused meal kit should be cancelled",
            )
            .require(
                "should_revalue_asset",
                "Portfolio",
                "Should review portfolio valuation",
            )
            .build();

        let result = run_scenario(&scenario);
        let status = if result.passed { "✅" } else { "❌" };
        println!(
            "  {} Rich + Wasteful: align={:.0}% recall={:.0}% misalign={:.0}%",
            status,
            result.alignment_score * 100.0,
            result.recall * 100.0,
            result.misalignment_rate * 100.0
        );
        assert!(result.passed, "Rich+Wasteful hybrid should pass");
    }

    // Scenario D: Debt + Subscription Creep — paying for stuff they can't afford
    {
        let mut s = ScenarioBuilder::new(
            "Hybrid: Debt + Subscriptions",
            "User has debt AND unused subscriptions — cancel subs, pay down debt",
        );
        s.add_entity(
            "obligation",
            "CC_18APR_Debt",
            "obligation-has-interest-term",
            0.45,
        );
        s.add_entity(
            "recurring-pattern",
            "Unused_Streaming1",
            "pattern-owned-by",
            0.35,
        );
        s.add_entity(
            "recurring-pattern",
            "Unused_Streaming2",
            "pattern-owned-by",
            0.32,
        );
        s.add_entity("instrument", "Checking_Low", "user-has-instrument", 0.05);

        let scenario = s
            .require("should_refinance", "CC_18APR", "Must address debt")
            .require(
                "should_cancel",
                "Unused_Streaming1",
                "Must cancel unused sub",
            )
            .priority(
                "should_cancel",
                "should_consolidate",
                "Free money before optimizing",
            )
            .build();

        let result = run_scenario(&scenario);
        let status = if result.passed { "✅" } else { "❌" };
        println!(
            "  {} Debt + Subscriptions: align={:.0}% recall={:.0}% misalign={:.0}%",
            status,
            result.alignment_score * 100.0,
            result.recall * 100.0,
            result.misalignment_rate * 100.0
        );
        assert!(result.passed, "Debt+Subs hybrid should pass");
    }
}
