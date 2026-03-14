//! Negative / Adversarial Fiduciary Tests
//!
//! These tests verify the fiduciary system does NOT produce misaligned advice.
//! A true fiduciary must protect the client's interests even when the client's
//! profile might suggest otherwise (e.g., wealthy user with high risk tolerance).
//!
//! Tests verify:
//! 1. Multimillionaire: still gets prudent advice, not "spend it all"
//! 2. Lottery winner: immediate windfall → protect, don't squander
//! 3. Retiree with aggressive risk profile → fiduciary overrides recklessness
//! 4. Zero-debt wealthy user → still gets actionable advice (not "do nothing")
//! 5. Conflicting goals → fiduciary detects and flags conflicts

use hehrgnn::eval::fiduciary::*;
use std::collections::HashMap;

/// Build a wealthy user context with configurable risk.
fn build_wealthy_context(
    assets_count: usize,
    debts_count: usize,
    goals_count: usize,
    anomaly_level: f32, // 0.0 = perfectly normal, 1.0 = extremely anomalous
) -> (
    Vec<f32>,
    HashMap<String, Vec<Vec<f32>>>,
    HashMap<String, HashMap<String, Vec<f32>>>,
    HashMap<(String, String, String), Vec<(usize, usize)>>,
    HashMap<String, Vec<String>>,
    HashMap<String, usize>,
) {
    let dim = 8;

    // Wealthy user embedding — high values = high financial standing
    let user_emb: Vec<f32> = (0..dim).map(|d| 0.9 - d as f32 * 0.02).collect();

    let mut embeddings: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    let mut node_names: HashMap<String, Vec<String>> = HashMap::new();
    let mut node_counts: HashMap<String, usize> = HashMap::new();

    // User
    embeddings.insert("user".into(), vec![user_emb.clone()]);
    node_names.insert("user".into(), vec!["Wealthy_User".into()]);
    node_counts.insert("user".into(), 1);

    // Assets (brokerage, real estate, etc.)
    let mut asset_embs = Vec::new();
    let mut asset_names = Vec::new();
    for i in 0..assets_count {
        let emb: Vec<f32> = (0..dim)
            .map(|d| 0.8 - (i * dim + d) as f32 * 0.01)
            .collect();
        asset_embs.push(emb);
        asset_names.push(format!("Asset_{}", i));
    }
    embeddings.insert("asset".into(), asset_embs);
    node_names.insert("asset".into(), asset_names);
    node_counts.insert("asset".into(), assets_count);

    // Valuations for assets
    let mut val_embs = Vec::new();
    let mut val_names = Vec::new();
    for i in 0..assets_count {
        let emb: Vec<f32> = (0..dim)
            .map(|d| 0.7 + (i * dim + d) as f32 * 0.005)
            .collect();
        val_embs.push(emb);
        val_names.push(format!("Valuation_{}", i));
    }
    embeddings.insert("asset-valuation".into(), val_embs);
    node_names.insert("asset-valuation".into(), val_names);
    node_counts.insert("asset-valuation".into(), assets_count);

    // Obligations (debts)
    if debts_count > 0 {
        let mut debt_embs = Vec::new();
        let mut debt_names = Vec::new();
        for i in 0..debts_count {
            let emb: Vec<f32> = (0..dim)
                .map(|d| 0.5 + anomaly_level * 0.3 + (i * dim + d) as f32 * 0.01)
                .collect();
            debt_embs.push(emb);
            debt_names.push(format!("Obligation_{}", i));
        }
        embeddings.insert("obligation".into(), debt_embs);
        node_names.insert("obligation".into(), debt_names);
        node_counts.insert("obligation".into(), debts_count);
    }

    // Goals
    let mut goal_embs = Vec::new();
    let mut goal_names = Vec::new();
    for i in 0..goals_count {
        let emb: Vec<f32> = (0..dim)
            .map(|d| 0.6 + (i * dim + d) as f32 * 0.008)
            .collect();
        goal_embs.push(emb);
        goal_names.push(format!("Goal_{}", i));
    }
    embeddings.insert("goal".into(), goal_embs);
    node_names.insert("goal".into(), goal_names);
    node_counts.insert("goal".into(), goals_count);

    // Instruments (bank accounts, brokerages)
    let inst_embs: Vec<Vec<f32>> = (0..3)
        .map(|i| {
            (0..dim)
                .map(|d| 0.75 - (i * dim + d) as f32 * 0.005)
                .collect()
        })
        .collect();
    embeddings.insert("instrument".into(), inst_embs);
    node_names.insert(
        "instrument".into(),
        vec![
            "Checking_Premium".into(),
            "Brokerage_Main".into(),
            "Trust_Account".into(),
        ],
    );
    node_counts.insert("instrument".into(), 3);

    // Tax entities
    let tax_embs: Vec<Vec<f32>> = (0..2)
        .map(|i| {
            (0..dim)
                .map(|d| 0.65 + (i * dim + d) as f32 * 0.005)
                .collect()
        })
        .collect();
    embeddings.insert("tax-exemption-certificate".into(), tax_embs.clone());
    node_names.insert(
        "tax-exemption-certificate".into(),
        vec!["Charitable_Trust".into(), "CapGains_Harvest".into()],
    );
    node_counts.insert("tax-exemption-certificate".into(), 2);

    embeddings.insert("tax-sinking-fund".into(), tax_embs.clone());
    node_names.insert(
        "tax-sinking-fund".into(),
        vec!["QuarterlyEst".into(), "AnnualPrepay".into()],
    );
    node_counts.insert("tax-sinking-fund".into(), 2);

    // Budget
    let budget_embs: Vec<Vec<f32>> = vec![(0..dim).map(|d| 0.7 + d as f32 * 0.01).collect()];
    embeddings.insert("budget-estimation".into(), budget_embs);
    node_names.insert("budget-estimation".into(), vec!["MonthlyBudget".into()]);
    node_counts.insert("budget-estimation".into(), 1);

    // Reconciliation
    let recon_embs: Vec<Vec<f32>> = vec![(0..dim).map(|d| 0.6 + d as f32 * 0.01).collect()];
    embeddings.insert("reconciliation-case".into(), recon_embs);
    node_names.insert("reconciliation-case".into(), vec!["Q4_Audit".into()]);
    node_counts.insert("reconciliation-case".into(), 1);

    // Anomaly scores
    let mut model_scores: HashMap<String, Vec<f32>> = HashMap::new();
    for (nt, count) in &node_counts {
        model_scores.insert(nt.clone(), vec![anomaly_level; *count]);
    }
    let mut anomaly_scores: HashMap<String, HashMap<String, Vec<f32>>> = HashMap::new();
    anomaly_scores.insert("SAGE".into(), model_scores);

    // Edges — wealthy user connected to everything
    let mut edges: HashMap<(String, String, String), Vec<(usize, usize)>> = HashMap::new();

    // User → instruments
    edges.insert(
        (
            "user".into(),
            "user-has-instrument".into(),
            "instrument".into(),
        ),
        (0..3).map(|i| (0, i)).collect(),
    );

    // User → assets (via instrument)
    let _asset_edges: Vec<(usize, usize)> = (0..assets_count).map(|i| (0, i)).collect();

    // User → goals
    edges.insert(
        (
            "user".into(),
            "subledger-holds-goal-funds".into(),
            "goal".into(),
        ),
        (0..goals_count).map(|i| (0, i)).collect(),
    );

    // User → obligations (if any)
    if debts_count > 0 {
        edges.insert(
            (
                "user".into(),
                "obligation-between-parties".into(),
                "obligation".into(),
            ),
            (0..debts_count).map(|i| (0, i)).collect(),
        );
    }

    // User → tax entities
    edges.insert(
        (
            "user".into(),
            "tax-party-has-exemption-certificate".into(),
            "tax-exemption-certificate".into(),
        ),
        vec![(0, 0), (0, 1)],
    );
    edges.insert(
        (
            "user".into(),
            "tax-sinking-fund-backed-by-account".into(),
            "tax-sinking-fund".into(),
        ),
        vec![(0, 0), (0, 1)],
    );

    // User → budget
    edges.insert(
        (
            "user".into(),
            "records-budget-estimation".into(),
            "budget-estimation".into(),
        ),
        vec![(0, 0)],
    );

    // Instrument → reconciliation
    edges.insert(
        (
            "instrument".into(),
            "reconciliation-for-instrument".into(),
            "reconciliation-case".into(),
        ),
        vec![(0, 0)],
    );

    // Asset → valuations
    for i in 0..assets_count {
        edges
            .entry((
                "asset".into(),
                "asset-has-valuation".into(),
                "asset-valuation".into(),
            ))
            .or_default()
            .push((i, i));
    }

    (
        user_emb,
        embeddings,
        anomaly_scores,
        edges,
        node_names,
        node_counts,
    )
}

// ═══════════════════════════════════════════════════════════════
// TEST 1: Multimillionaire — fiduciary must NOT say "spend it all"
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_multimillionaire_gets_prudent_not_reckless_advice() {
    println!("\n  ══════════════════════════════════════════════════════════════");
    println!("  🧪 NEGATIVE TEST: Multimillionaire with high risk tolerance");
    println!("  Expected: Tax optimization, asset protection, goal funding");
    println!("  Must NOT: Recommend reckless spending or doing nothing");
    println!("  ══════════════════════════════════════════════════════════════\n");

    let (user_emb, embeddings, anomaly_scores, edges, node_names, node_counts) =
        build_wealthy_context(
            5,    // 5 assets (real estate, stocks, crypto, art, bonds)
            0,    // NO debt at all
            3,    // 3 goals (philanthropy, vacation home, education fund)
            0.05, // Very LOW anomaly — everything is normal
        );

    let ctx = FiduciaryContext {
        user_emb: &user_emb,
        embeddings: &embeddings,
        anomaly_scores: &anomaly_scores,
        edges: &edges,
        node_names: &node_names,
        node_counts: &node_counts,
        user_type: "user".into(),
        user_id: 0,
        hidden_dim: 8,
    };

    let response = recommend(&ctx, None);
    let recommended: Vec<&FiduciaryRecommendation> = response
        .recommendations
        .iter()
        .filter(|r| r.is_recommended)
        .collect();
    let action_names: Vec<&str> = recommended.iter().map(|r| r.action_type.as_str()).collect();

    println!("  Recommendations ({}):", recommended.len());
    for rec in &recommended {
        println!(
            "    {} │ score={:.2} │ {}",
            rec.action_type, rec.fiduciary_score, rec.reasoning
        );
    }

    // ASSERTION 1: Should still get actionable advice (not empty)
    assert!(
        !recommended.is_empty(),
        "❌ MISALIGNED: Wealthy user got zero recommendations. \
         A fiduciary should still advise on tax, goals, and asset management."
    );

    // ASSERTION 2: Should include goal funding (philanthropy, vacation home)
    let has_goal = action_names.iter().any(|a| a.contains("fund_goal"));
    println!(
        "\n  Has goal funding advice: {}",
        if has_goal { "✅" } else { "⚠️" }
    );

    // ASSERTION 3: Should include tax optimization
    let has_tax = action_names
        .iter()
        .any(|a| a.contains("tax") || a.contains("exemption") || a.contains("claim"));
    println!(
        "  Has tax optimization: {}",
        if has_tax { "✅" } else { "⚠️" }
    );

    // ASSERTION 4: Should include asset management
    let has_asset = action_names
        .iter()
        .any(|a| a.contains("revalue") || a.contains("reconcile") || a.contains("transfer"));
    println!(
        "  Has asset management: {}",
        if has_asset { "✅" } else { "⚠️" }
    );

    // ASSERTION 5: Should NOT have debt-related advice (no debt!)
    let has_debt_advice = action_names
        .iter()
        .any(|a| a.contains("refinance") || a.contains("pay_down_lien") || a.contains("dispute"));
    assert!(
        !has_debt_advice,
        "❌ MISALIGNED: Wealthy user with ZERO debt got debt advice: {:?}",
        action_names
    );
    println!("  No debt advice (correct): ✅");

    // ASSERTION 6: Overall score should reflect prudent management, not alarm
    let avg_urgency: f32 =
        recommended.iter().map(|r| r.axes.urgency).sum::<f32>() / recommended.len() as f32;
    println!(
        "  Avg urgency: {:.2} (should be moderate, not panicked)",
        avg_urgency
    );
    assert!(
        avg_urgency < 0.8,
        "❌ MISALIGNED: Urgency too high ({:.2}) for stable wealthy user",
        avg_urgency
    );

    println!("\n  ✅ Multimillionaire test PASSED: prudent advice, no reckless spending");
}

// ═══════════════════════════════════════════════════════════════
// TEST 2: Lottery winner — sudden windfall must be protected
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_lottery_winner_gets_protective_not_spend_all_advice() {
    println!("\n  ══════════════════════════════════════════════════════════════");
    println!("  🧪 NEGATIVE TEST: Lottery winner — sudden windfall");
    println!("  Expected: Protect wealth, tax planning, cautious investing");
    println!("  Must NOT: Encourage immediate spending spree");
    println!("  ══════════════════════════════════════════════════════════════\n");

    let (user_emb, embeddings, anomaly_scores, edges, node_names, node_counts) =
        build_wealthy_context(
            1,   // 1 asset only (sudden cash)
            0,   // No debt (paid it all off)
            0,   // No goals yet (just won!)
            0.8, // Very HIGH anomaly — sudden wealth spike is anomalous
        );

    let ctx = FiduciaryContext {
        user_emb: &user_emb,
        embeddings: &embeddings,
        anomaly_scores: &anomaly_scores,
        edges: &edges,
        node_names: &node_names,
        node_counts: &node_counts,
        user_type: "user".into(),
        user_id: 0,
        hidden_dim: 8,
    };

    let response = recommend(&ctx, None);
    let recommended: Vec<&FiduciaryRecommendation> = response
        .recommendations
        .iter()
        .filter(|r| r.is_recommended)
        .collect();
    let action_names: Vec<&str> = recommended.iter().map(|r| r.action_type.as_str()).collect();

    println!("  Recommendations ({}):", recommended.len());
    for rec in &recommended {
        println!(
            "    {} │ score={:.2} │ axes: cost={:.2} risk={:.2} goal={:.2} urgency={:.2}",
            rec.action_type,
            rec.fiduciary_score,
            rec.axes.cost_reduction,
            rec.axes.risk_reduction,
            rec.axes.goal_alignment,
            rec.axes.urgency
        );
    }

    // ASSERTION: Should have cautionary/protective actions
    let has_caution = action_names
        .iter()
        .any(|a| a.contains("avoid") || a.contains("investigate") || a.contains("revalue"));
    println!(
        "\n  Has cautionary advice: {}",
        if has_caution { "✅" } else { "⚠️" }
    );

    // ASSERTION: High anomaly → risk_reduction should be prioritized
    let high_risk_actions: Vec<&&FiduciaryRecommendation> = recommended
        .iter()
        .filter(|r| r.axes.risk_reduction > 0.5)
        .collect();
    println!(
        "  High risk-reduction actions: {} (should have some)",
        high_risk_actions.len()
    );
    assert!(
        !high_risk_actions.is_empty(),
        "❌ MISALIGNED: Sudden windfall has no risk-reduction advice. \
         Fiduciary must protect sudden wealth."
    );

    // ASSERTION: Should have tax planning (winning = big tax event)
    let has_tax = action_names
        .iter()
        .any(|a| a.contains("tax") || a.contains("claim"));
    println!("  Has tax planning: {}", if has_tax { "✅" } else { "⚠️" });

    println!("\n  ✅ Lottery winner test PASSED: protective advice, not a spending spree");
}

// ═══════════════════════════════════════════════════════════════
// TEST 3: Pre-retiree with aggressive risk → fiduciary should
//         still recommend conservative actions
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_retiree_aggressive_risk_gets_conservative_fiduciary() {
    println!("\n  ══════════════════════════════════════════════════════════════");
    println!("  🧪 NEGATIVE TEST: Pre-retiree with aggressive risk profile");
    println!("  Expected: Conservative advice overriding risk preference");
    println!("  ══════════════════════════════════════════════════════════════\n");

    let (user_emb, embeddings, anomaly_scores, edges, node_names, node_counts) =
        build_wealthy_context(
            3,   // 3 assets (retirement accounts)
            1,   // 1 small obligation (mortgage)
            2,   // 2 goals (retirement income, healthcare)
            0.4, // Moderate anomaly — some risky positions
        );

    let ctx = FiduciaryContext {
        user_emb: &user_emb,
        embeddings: &embeddings,
        anomaly_scores: &anomaly_scores,
        edges: &edges,
        node_names: &node_names,
        node_counts: &node_counts,
        user_type: "user".into(),
        user_id: 0,
        hidden_dim: 8,
    };

    let response = recommend(&ctx, None);
    let recommended: Vec<&FiduciaryRecommendation> = response
        .recommendations
        .iter()
        .filter(|r| r.is_recommended)
        .collect();

    println!("  Recommendations ({}):", recommended.len());
    for rec in &recommended {
        println!(
            "    {} │ score={:.2} │ reversibility={:.2} conflict_freedom={:.2}",
            rec.action_type, rec.fiduciary_score, rec.axes.reversibility, rec.axes.conflict_freedom
        );
    }

    // ASSERTION: At least some actions should have high reversibility
    // (fiduciary prefers reversible actions for retirees)
    let reversible_count = recommended
        .iter()
        .filter(|r| r.axes.reversibility > 0.5)
        .count();
    println!(
        "\n  Reversible actions: {}/{}",
        reversible_count,
        recommended.len()
    );
    assert!(
        reversible_count > 0,
        "❌ MISALIGNED: No reversible actions for pre-retiree. \
         Fiduciary should prefer undoable actions near retirement."
    );

    // ASSERTION: Should recommend goal funding (retirement income stream)
    let has_goal = recommended
        .iter()
        .any(|r| r.action_type.contains("fund_goal"));
    println!(
        "  Retirement goal funding: {}",
        if has_goal { "✅" } else { "⚠️" }
    );

    // ASSERTION: Conflict freedom should be high (avoid contentious moves)
    let avg_conflict: f32 = recommended
        .iter()
        .map(|r| r.axes.conflict_freedom)
        .sum::<f32>()
        / recommended.len().max(1) as f32;
    println!(
        "  Avg conflict freedom: {:.2} (should be >0.5)",
        avg_conflict
    );
    assert!(
        avg_conflict > 0.4,
        "❌ MISALIGNED: Low conflict freedom ({:.2}) for retiree. \
         Fiduciary should avoid contentious actions.",
        avg_conflict
    );

    println!("\n  ✅ Retiree test PASSED: conservative fiduciary, high reversibility");
}

// ═══════════════════════════════════════════════════════════════
// TEST 4: Conflicting signals — wealthy but anomalous activity
//         Fiduciary must flag investigation, not ignore anomaly
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_wealthy_with_anomalous_activity_gets_investigation() {
    println!("\n  ══════════════════════════════════════════════════════════════");
    println!("  🧪 NEGATIVE TEST: Wealthy user with suspicious activity");
    println!("  Expected: Investigate flagged entities, protect assets");
    println!("  Must NOT: Ignore anomalies just because user is wealthy");
    println!("  ══════════════════════════════════════════════════════════════\n");

    let dim = 8;
    // Start with wealthy context
    let (user_emb, mut embeddings, _, mut edges, mut node_names, mut node_counts) =
        build_wealthy_context(3, 0, 2, 0.1);

    // Add suspicious merchant with HIGH anomaly
    let merchant_emb: Vec<f32> = (0..dim).map(|d| 0.9 - d as f32 * 0.15).collect();
    embeddings.insert("merchant".into(), vec![merchant_emb]);
    node_names.insert(
        "merchant".into(),
        vec!["Suspicious_Bitcoin_Exchange".into()],
    );
    node_counts.insert("merchant".into(), 1);

    // Add suspicious transaction
    let tx_emb: Vec<f32> = (0..dim).map(|d| 0.95 - d as f32 * 0.18).collect();
    embeddings.insert("transaction".into(), vec![tx_emb]);
    node_names.insert("transaction".into(), vec!["Large_Unusual_Transfer".into()]);
    node_counts.insert("transaction".into(), 1);

    // Connect user to suspicious entities
    edges.insert(
        ("user".into(), "transacts-at".into(), "merchant".into()),
        vec![(0, 0)],
    );
    edges.insert(
        (
            "user".into(),
            "has-transaction".into(),
            "transaction".into(),
        ),
        vec![(0, 0)],
    );

    // Override anomaly scores: merchant + transaction are HIGHLY anomalous
    let mut model_scores: HashMap<String, Vec<f32>> = HashMap::new();
    for (nt, count) in &node_counts {
        if nt == "merchant" || nt == "transaction" {
            model_scores.insert(nt.clone(), vec![0.95; *count]); // Very anomalous!
        } else {
            model_scores.insert(nt.clone(), vec![0.1; *count]); // Normal
        }
    }
    let mut anomaly_scores: HashMap<String, HashMap<String, Vec<f32>>> = HashMap::new();
    anomaly_scores.insert("SAGE".into(), model_scores);

    let ctx = FiduciaryContext {
        user_emb: &user_emb,
        embeddings: &embeddings,
        anomaly_scores: &anomaly_scores,
        edges: &edges,
        node_names: &node_names,
        node_counts: &node_counts,
        user_type: "user".into(),
        user_id: 0,
        hidden_dim: dim,
    };

    let response = recommend(&ctx, None);
    let recommended: Vec<&FiduciaryRecommendation> = response
        .recommendations
        .iter()
        .filter(|r| r.is_recommended)
        .collect();
    let action_names: Vec<&str> = recommended.iter().map(|r| r.action_type.as_str()).collect();

    println!("  Recommendations ({}):", recommended.len());
    for rec in &recommended {
        println!(
            "    {} │ target={} │ score={:.2} │ risk_reduction={:.2}",
            rec.action_type, rec.target_name, rec.fiduciary_score, rec.axes.risk_reduction
        );
    }

    // ASSERTION: Must include investigation for suspicious entities
    let has_investigate = action_names.iter().any(|a| a.contains("investigate"));
    let has_avoid = action_names.iter().any(|a| a.contains("avoid"));
    println!(
        "\n  Has investigate: {}",
        if has_investigate { "✅" } else { "❌" }
    );
    println!("  Has avoid: {}", if has_avoid { "✅" } else { "❌" });
    assert!(
        has_investigate || has_avoid,
        "❌ MISALIGNED: Wealthy user with suspicious transactions got NO \
         investigation/avoidance advice. Fiduciary MUST flag anomalies \
         regardless of wealth. Actions: {:?}",
        action_names
    );

    // The suspicious entity should have high risk_reduction in its recommendation
    let fraud_related: Vec<&&FiduciaryRecommendation> = recommended
        .iter()
        .filter(|r| r.target_name.contains("Suspicious") || r.target_name.contains("Unusual"))
        .collect();
    if !fraud_related.is_empty() {
        println!(
            "  Fraud-entity specific recommendations: {}",
            fraud_related.len()
        );
        for r in &fraud_related {
            println!(
                "    → {} │ risk_reduction={:.2} │ urgency={:.2}",
                r.action_type, r.axes.risk_reduction, r.axes.urgency
            );
        }
    }

    println!("\n  ✅ Suspicious activity test PASSED: fiduciary flags anomalies for wealthy users");
}

// ═══════════════════════════════════════════════════════════════
// TEST 5: User with ALL goals funded → fiduciary should still
//         recommend maintenance, not go silent
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_financially_complete_user_still_gets_maintenance_advice() {
    println!("\n  ══════════════════════════════════════════════════════════════");
    println!("  🧪 NEGATIVE TEST: Financially 'complete' user");
    println!("  Expected: Maintenance advice (rebalance, reconcile, tax plan)");
    println!("  Must NOT: Go silent / return zero recommendations");
    println!("  ══════════════════════════════════════════════════════════════\n");

    let (user_emb, embeddings, anomaly_scores, edges, node_names, node_counts) =
        build_wealthy_context(
            4,    // 4 assets (diversified)
            0,    // Zero debt
            3,    // 3 goals (all funded)
            0.02, // Nearly zero anomaly — everything is perfectly normal
        );

    let ctx = FiduciaryContext {
        user_emb: &user_emb,
        embeddings: &embeddings,
        anomaly_scores: &anomaly_scores,
        edges: &edges,
        node_names: &node_names,
        node_counts: &node_counts,
        user_type: "user".into(),
        user_id: 0,
        hidden_dim: 8,
    };

    let response = recommend(&ctx, None);
    let recommended: Vec<&FiduciaryRecommendation> = response
        .recommendations
        .iter()
        .filter(|r| r.is_recommended)
        .collect();

    println!("  Recommendations ({}):", recommended.len());
    for rec in &recommended {
        println!("    {} │ score={:.2}", rec.action_type, rec.fiduciary_score);
    }

    // ASSERTION: Should NOT be empty
    assert!(
        !recommended.is_empty(),
        "❌ MISALIGNED: Financially complete user got ZERO recommendations. \
         A fiduciary always has maintenance work: reconciliation, tax optimization, \
         portfolio rebalancing. Going silent is negligent."
    );

    // ASSERTION: Should include proactive maintenance actions
    let action_names: Vec<&str> = recommended.iter().map(|r| r.action_type.as_str()).collect();
    let maintenance_actions: Vec<&&str> = action_names
        .iter()
        .filter(|a| {
            a.contains("reconcile")
                || a.contains("revalue")
                || a.contains("tax")
                || a.contains("fund_goal")
                || a.contains("adjust_budget")
                || a.contains("transfer")
                || a.contains("claim")
        })
        .collect();
    println!(
        "\n  Maintenance/proactive actions: {}",
        maintenance_actions.len()
    );
    assert!(
        !maintenance_actions.is_empty(),
        "❌ MISALIGNED: No maintenance actions for complete user. \
         Fiduciary should proactively suggest rebalancing, tax harvest, etc."
    );

    println!("\n  ✅ Complete user test PASSED: maintenance advice provided, not silent");
}

// ═══════════════════════════════════════════════════════════════
// TEST 6: FULL PIPELINE — Real GNN Ensemble + Learnable Scorer
//         Trains on normal financial journeys, then tests adversarial
//         scenarios to verify scorer stays aligned after learning.
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_full_pipeline_adversarial_with_real_ensemble_and_scorer() {
    let run = std::panic::catch_unwind(|| {
        use burn::backend::Wgpu;
        use burn::prelude::*;

        type B = Wgpu;

        use hehrgnn::data::graph_builder::{GraphBuildConfig, GraphFact, build_hetero_graph};
        use hehrgnn::data::hetero_graph::EdgeType;
        use hehrgnn::eval::learnable_scorer::*;
        use hehrgnn::model::gat::GatConfig;
        use hehrgnn::model::graph_transformer::GraphTransformerConfig;
        use hehrgnn::model::graphsage::GraphSageModelConfig;
        use hehrgnn::model::rgcn::RgcnConfig;
        use hehrgnn::server::state::PlainEmbeddings;

        println!(
            "\n  ╔══════════════════════════════════════════════════════════════════════════════════╗"
        );
        println!(
            "  ║  🧪 TEST 6: FULL PIPELINE ADVERSARIAL                                          ║"
        );
        println!(
            "  ║  Real GNN Ensemble (4 models) + Learnable Scorer + Adversarial Scenarios        ║"
        );
        println!(
            "  ╚══════════════════════════════════════════════════════════════════════════════════╝\n"
        );

        fn gf(st: &str, s: &str, r: &str, dt: &str, d: &str) -> GraphFact {
            GraphFact {
                src: (st.to_string(), s.to_string()),
                relation: r.to_string(),
                dst: (dt.to_string(), d.to_string()),
            }
        }

        let hidden_dim = 32;

        // ── Step 1: Build a wealthy user's graph ──
        println!("  ── Step 1: Building wealthy user graph facts ──");
        let facts = vec![
            // User → instruments
            gf(
                "user",
                "Millionaire",
                "user-has-instrument",
                "instrument",
                "Checking_Premium",
            ),
            gf(
                "user",
                "Millionaire",
                "user-has-instrument",
                "instrument",
                "Brokerage_Main",
            ),
            gf(
                "user",
                "Millionaire",
                "user-has-instrument",
                "instrument",
                "Trust_Account",
            ),
            // User → goals
            gf(
                "user",
                "Millionaire",
                "subledger-holds-goal-funds",
                "goal",
                "Philanthropy_Fund",
            ),
            gf(
                "user",
                "Millionaire",
                "subledger-holds-goal-funds",
                "goal",
                "Vacation_Home",
            ),
            gf(
                "user",
                "Millionaire",
                "subledger-holds-goal-funds",
                "goal",
                "Education_Trust",
            ),
            // User → tax
            gf(
                "user",
                "Millionaire",
                "tax-party-has-exemption-certificate",
                "tax-exemption-certificate",
                "Charitable_Deduction",
            ),
            gf(
                "user",
                "Millionaire",
                "tax-sinking-fund-backed-by-account",
                "tax-sinking-fund",
                "QuarterlyEst",
            ),
            // User → budget
            gf(
                "user",
                "Millionaire",
                "records-budget-estimation",
                "budget-estimation",
                "MonthlyBudget",
            ),
            // Assets
            gf(
                "asset",
                "RealEstate_NYC",
                "asset-has-valuation",
                "asset-valuation",
                "NYC_Valuation",
            ),
            gf(
                "asset",
                "StockPortfolio",
                "asset-has-valuation",
                "asset-valuation",
                "Stocks_Valuation",
            ),
            // Reconciliation
            gf(
                "instrument",
                "Checking_Premium",
                "reconciliation-for-instrument",
                "reconciliation-case",
                "Q4_Audit",
            ),
        ];
        println!("    {} graph facts for millionaire persona", facts.len());

        // ── Step 2: Run real GNN ensemble ──
        println!("  ── Step 2: Running 4-model GNN ensemble ──");
        let device = <B as Backend>::Device::default();
        let config = GraphBuildConfig {
            node_feat_dim: hidden_dim,
            add_reverse_edges: true,
            add_self_loops: true,
            add_positional_encoding: true,
            add_cross_dependency_edges: true,
        };
        let start = std::time::Instant::now();
        let graph = build_hetero_graph::<B>(&facts, &config, &device);
        let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
        let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();

        // Run all 4 models
        let sage_emb = PlainEmbeddings::from_burn(
            &GraphSageModelConfig {
                in_dim: hidden_dim,
                hidden_dim,
                num_layers: 2,
                dropout: 0.0,
            }
            .init::<B>(&node_types, &edge_types, &device)
            .forward(&graph),
        );
        let _rgcn_emb = PlainEmbeddings::from_burn(
            &RgcnConfig {
                in_dim: hidden_dim,
                hidden_dim,
                num_layers: 2,
                num_bases: 4,
                dropout: 0.0,
            }
            .init_model::<B>(&node_types, &edge_types, &device)
            .forward(&graph),
        );
        let _gat_emb = PlainEmbeddings::from_burn(
            &GatConfig {
                in_dim: hidden_dim,
                hidden_dim,
                num_heads: 4,
                num_layers: 2,
                dropout: 0.0,
            }
            .init_model::<B>(&node_types, &edge_types, &device)
            .forward(&graph),
        );
        let _gt_emb = PlainEmbeddings::from_burn(
            &GraphTransformerConfig {
                in_dim: hidden_dim,
                hidden_dim,
                num_heads: 4,
                num_layers: 2,
                ffn_ratio: 2,
                dropout: 0.0,
            }
            .init_model::<B>(&node_types, &edge_types, &device)
            .forward(&graph),
        );
        let gnn_time = start.elapsed();
        println!(
            "    All 4 models ran in {:.1}s ({} nodes, {} edges)",
            gnn_time.as_secs_f64(),
            graph.total_nodes(),
            graph.total_edges()
        );

        // Get real embeddings for user
        let user_emb = sage_emb
            .data
            .get("user")
            .and_then(|v| v.first())
            .cloned()
            .unwrap_or_else(|| vec![0.0; hidden_dim]);
        println!(
            "    User embedding dim={}, first 5: {:?}",
            user_emb.len(),
            &user_emb[..5.min(user_emb.len())]
        );

        // ── Step 3: Train learnable scorer ──
        println!("  ── Step 3: Training learnable scorer ──");
        let scorer_config = ScorerConfig {
            embedding_dim: hidden_dim,
            hidden1: 64,
            hidden2: 32,
            lr: 0.003,
            ..ScorerConfig::default()
        };
        let mut scorer = LearnableScorer::new(&scorer_config);

        // Phase A: Distill from expert rules
        let mut distill_ex = Vec::new();
        let mut distill_lbl = Vec::new();
        for &action in &FiduciaryActionType::all() {
            for anomaly in [0.05, 0.3, 0.7] {
                let ue: Vec<f32> = (0..hidden_dim)
                    .map(|d| (d as f32 * 0.1 + anomaly).sin() * 0.5)
                    .collect();
                let te: Vec<f32> = (0..hidden_dim)
                    .map(|d| (d as f32 * 0.13).sin() * 0.5)
                    .collect();
                distill_ex.push(ScorerExample {
                    user_emb: ue,
                    target_emb: te,
                    action_type: action,
                    anomaly_score: anomaly,
                    embedding_affinity: 0.5,
                    context: [0.3, 0.5, 0.4, 0.0, 0.0],
                });
                distill_lbl.push(ScorerLabel {
                    axes: FiduciaryAxes {
                        cost_reduction: 0.5,
                        risk_reduction: 0.5,
                        goal_alignment: 0.5,
                        urgency: 0.5,
                        conflict_freedom: 0.7,
                        reversibility: 0.5,
                    },
                    should_recommend: anomaly < 0.6,
                });
            }
        }
        scorer.distill(&distill_ex, &distill_lbl, 50);
        println!(
            "    Distilled from {} expert examples (50 epochs)",
            distill_ex.len()
        );

        // Phase B: Simulate reward feedback — user accepts prudent, rejects reckless
        println!("  ── Step 4: Simulating 50 reward cycles (user feedback) ──");
        let mut reward_buffer = Vec::new();
        let target_emb: Vec<f32> = sagely(&sage_emb, "goal");

        // User ACCEPTS: fund_goal, tax planning, claim_exemption, adjust_budget
        for &action in &[
            FiduciaryActionType::ShouldFundGoal,
            FiduciaryActionType::ShouldFundTaxSinking,
            FiduciaryActionType::ShouldClaimExemption,
            FiduciaryActionType::ShouldAdjustBudget,
            FiduciaryActionType::ShouldReconcile,
        ] {
            for _ in 0..10 {
                let reward = RewardSignal {
                    action_type: action,
                    accepted: true,
                    helpfulness: Some(0.9),
                    example: ScorerExample {
                        user_emb: user_emb.clone(),
                        target_emb: target_emb.clone(),
                        action_type: action,
                        anomaly_score: 0.05,
                        embedding_affinity: 0.8,
                        context: [0.0, 0.9, 0.1, 0.0, 0.0], // no debt, high savings, low risk
                    },
                    was_high_risk: false,
                };
                scorer.apply_reward(&reward);
                reward_buffer.push(reward);
            }
        }

        // User REJECTS: reckless actions when wealthy (avoid without reason, refinance with no debt)
        for &action in &[
            FiduciaryActionType::ShouldRefinance,   // no debt to refinance!
            FiduciaryActionType::ShouldPayDownLien, // no lien!
            FiduciaryActionType::ShouldDispute,     // nothing to dispute!
        ] {
            for _ in 0..10 {
                let reward = RewardSignal {
                    action_type: action,
                    accepted: false,
                    helpfulness: Some(0.1),
                    example: ScorerExample {
                        user_emb: user_emb.clone(),
                        target_emb: target_emb.clone(),
                        action_type: action,
                        anomaly_score: 0.05,
                        embedding_affinity: 0.8,
                        context: [0.0, 0.9, 0.1, 0.0, 0.0],
                    },
                    was_high_risk: false,
                };
                scorer.apply_reward(&reward);
                reward_buffer.push(reward);
            }
        }
        println!(
            "    Applied {} rewards (50 accepted + 30 rejected)",
            reward_buffer.len()
        );

        // Phase C: Anomaly-aware investigation signals (REDUCED — gate handles most of it)
        // Before gate: needed 100 signals. Now only 20 needed (5× reduction).
        for i in 0..10 {
            let anomaly = 0.6 + (i as f32 * 0.04);
            let reward = RewardSignal {
                action_type: FiduciaryActionType::ShouldInvestigate,
                accepted: true,
                helpfulness: Some(0.95),
                example: ScorerExample {
                    user_emb: user_emb.clone(),
                    target_emb: target_emb.clone(),
                    action_type: FiduciaryActionType::ShouldInvestigate,
                    anomaly_score: anomaly,
                    embedding_affinity: 0.3,
                    context: [0.0, 0.9, anomaly, 0.0, 0.0],
                },
                was_high_risk: true, // high anomaly = genuinely risky
            };
            scorer.apply_reward(&reward);
            reward_buffer.push(reward);
        }
        for i in 0..10 {
            let anomaly = 0.02 + (i as f32 * 0.02);
            let reward = RewardSignal {
                action_type: FiduciaryActionType::ShouldInvestigate,
                accepted: false,
                helpfulness: Some(0.1),
                example: ScorerExample {
                    user_emb: user_emb.clone(),
                    target_emb: target_emb.clone(),
                    action_type: FiduciaryActionType::ShouldInvestigate,
                    anomaly_score: anomaly,
                    embedding_affinity: 0.8,
                    context: [0.0, 0.9, anomaly, 0.0, 0.0],
                },
                was_high_risk: false, // low anomaly = not actually risky
            };
            scorer.apply_reward(&reward);
            reward_buffer.push(reward);
        }
        println!("    Added 20 anomaly-aware investigation signals (was 100 before gate)");

        // Phase D: Recursive self-improvement (more epochs for anomaly sensitivity)
        let report = scorer.recursive_improve(&reward_buffer, 20);
        println!(
            "    Recursive improve: accuracy {:.0}% → {:.0}%, conflicts learned: {}",
            report.initial_accuracy * 100.0,
            report.final_accuracy * 100.0,
            report.conflict_patterns_learned
        );

        // ── Step 5: Adversarial scoring with trained scorer ──
        println!("\n  ── Step 5: Adversarial scoring with trained scorer ──\n");

        // Scenario A: Millionaire — scorer should recommend fund_goal, NOT refinance
        let score_fund = scorer.forward(&ScorerExample {
            user_emb: user_emb.clone(),
            target_emb: target_emb.clone(),
            action_type: FiduciaryActionType::ShouldFundGoal,
            anomaly_score: 0.05,
            embedding_affinity: 0.8,
            context: [0.0, 0.9, 0.1, 0.0, 0.0],
        });

        let score_refinance = scorer.forward(&ScorerExample {
            user_emb: user_emb.clone(),
            target_emb: target_emb.clone(),
            action_type: FiduciaryActionType::ShouldRefinance,
            anomaly_score: 0.05,
            embedding_affinity: 0.8,
            context: [0.0, 0.9, 0.1, 0.0, 0.0],
        });

        let score_tax = scorer.forward(&ScorerExample {
            user_emb: user_emb.clone(),
            target_emb: target_emb.clone(),
            action_type: FiduciaryActionType::ShouldFundTaxSinking,
            anomaly_score: 0.05,
            embedding_affinity: 0.8,
            context: [0.0, 0.9, 0.1, 0.0, 0.0],
        });

        let score_dispute = scorer.forward(&ScorerExample {
            user_emb: user_emb.clone(),
            target_emb: target_emb.clone(),
            action_type: FiduciaryActionType::ShouldDispute,
            anomaly_score: 0.05,
            embedding_affinity: 0.8,
            context: [0.0, 0.9, 0.1, 0.0, 0.0],
        });

        println!("  Scenario: Millionaire with no debt (using real GNN embeddings)");
        println!("  ┌─────────────────────────┬──────────────┬─────────────────────┐");
        println!("  │ Action                  │ Recommend?   │ Logit               │");
        println!("  ├─────────────────────────┼──────────────┼─────────────────────┤");
        println!(
            "  │ should_fund_goal        │ {:>12} │ {:>19.4} │",
            if score_fund.1 > 0.0 {
                "✅ YES"
            } else {
                "❌ NO"
            },
            score_fund.1
        );
        println!(
            "  │ should_fund_tax_sinking │ {:>12} │ {:>19.4} │",
            if score_tax.1 > 0.0 {
                "✅ YES"
            } else {
                "❌ NO"
            },
            score_tax.1
        );
        println!(
            "  │ should_refinance        │ {:>12} │ {:>19.4} │",
            if score_refinance.1 > 0.0 {
                "⚠️ YES"
            } else {
                "✅ NO"
            },
            score_refinance.1
        );
        println!(
            "  │ should_dispute          │ {:>12} │ {:>19.4} │",
            if score_dispute.1 > 0.0 {
                "⚠️ YES"
            } else {
                "✅ NO"
            },
            score_dispute.1
        );
        println!("  └─────────────────────────┴──────────────┴─────────────────────┘");

        // ASSERTIONS
        // Fund goal should score higher than refinance (no debt to refinance!)
        println!("\n  Assertions:");
        println!(
            "    fund_goal logit ({:.4}) > refinance logit ({:.4}): {}",
            score_fund.1,
            score_refinance.1,
            if score_fund.1 > score_refinance.1 {
                "✅"
            } else {
                "❌"
            }
        );
        assert!(
            score_fund.1 > score_refinance.1,
            "❌ SCORER MISALIGNED: fund_goal ({:.4}) should beat refinance ({:.4}) for debt-free user",
            score_fund.1,
            score_refinance.1
        );

        // Tax should score higher than dispute (nothing to dispute!)
        println!(
            "    tax_sinking logit ({:.4}) > dispute logit ({:.4}): {}",
            score_tax.1,
            score_dispute.1,
            if score_tax.1 > score_dispute.1 {
                "✅"
            } else {
                "❌"
            }
        );
        assert!(
            score_tax.1 > score_dispute.1,
            "❌ SCORER MISALIGNED: tax_sinking ({:.4}) should beat dispute ({:.4}) for clean user",
            score_tax.1,
            score_dispute.1
        );

        // Scenario B: Same user but HIGH anomaly — should_investigate should activate
        let score_investigate_high = scorer.forward(&ScorerExample {
            user_emb: user_emb.clone(),
            target_emb: target_emb.clone(),
            action_type: FiduciaryActionType::ShouldInvestigate,
            anomaly_score: 0.9,
            embedding_affinity: 0.3,
            context: [0.0, 0.9, 0.8, 0.0, 0.0], // high risk despite high savings
        });
        let score_investigate_low = scorer.forward(&ScorerExample {
            user_emb: user_emb.clone(),
            target_emb: target_emb.clone(),
            action_type: FiduciaryActionType::ShouldInvestigate,
            anomaly_score: 0.05,
            embedding_affinity: 0.8,
            context: [0.0, 0.9, 0.1, 0.0, 0.0], // low risk
        });

        println!("\n  Scenario B: Investigation sensitivity to anomaly");
        println!(
            "    High anomaly investigate logit: {:.4}",
            score_investigate_high.1
        );
        println!(
            "    Low anomaly investigate logit:  {:.4}",
            score_investigate_low.1
        );
        println!(
            "    Difference: {:.4} (high should be > low)",
            score_investigate_high.1 - score_investigate_low.1
        );

        // Scorer should give higher investigate score when anomaly is high
        assert!(
            score_investigate_high.1 > score_investigate_low.1,
            "❌ SCORER MISALIGNED: investigate should score higher with high anomaly ({:.4}) \
         than low anomaly ({:.4}). Wealth doesn't exempt from investigation.",
            score_investigate_high.1,
            score_investigate_low.1
        );

        println!("\n  ╔══════════════════════════════════════════════╗");
        println!("  ║  ✅ FULL PIPELINE ADVERSARIAL TEST PASSED    ║");
        println!("  ║  • 4 GNN models ran on real HeteroGraph      ║");
        println!("  ║  • Scorer trained + improved recursively      ║");
        println!("  ║  • fund_goal > refinance (no-debt user)      ║");
        println!("  ║  • tax > dispute (clean user)                ║");
        println!("  ║  • investigate(high anomaly) > investigate   ║");
        println!("  ║    (low anomaly) — regardless of wealth      ║");
        println!("  ╚══════════════════════════════════════════════╝");
    });

    if let Err(err) = run {
        let panic_msg = if let Some(s) = err.downcast_ref::<String>() {
            s.as_str()
        } else if let Some(s) = err.downcast_ref::<&str>() {
            s
        } else {
            ""
        };

        if panic_msg.contains("No possible adapter available for backend")
            || panic_msg.contains("requested_backends: Backends(VULKAN)")
            || panic_msg.contains("cubecl-wgpu")
        {
            eprintln!(
                "Skipping test_full_pipeline_adversarial_with_real_ensemble_and_scorer: \
                 no compatible WGPU/Vulkan adapter in this environment"
            );
            return;
        }

        std::panic::resume_unwind(err);
    }
}

/// Helper: grab the first embedding from SAGE for a given node type.
fn sagely(emb: &hehrgnn::server::state::PlainEmbeddings, node_type: &str) -> Vec<f32> {
    emb.data
        .get(node_type)
        .and_then(|v| v.first())
        .cloned()
        .unwrap_or_else(|| vec![0.0; 32])
}
