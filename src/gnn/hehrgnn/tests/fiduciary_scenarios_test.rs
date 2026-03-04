//! Scenario-based fiduciary tests — verifies that predictions make financial
//! sense for realistic personal finance situations.
//!
//! Each scenario builds a graph representing a real financial situation
//! and asserts that the fiduciary system recommends the RIGHT actions
//! in a sensible priority order.

use std::collections::HashMap;

use hehrgnn::eval::fiduciary::*;

// ═══════════════════════════════════════════════════════════════
// Helper: build a FiduciaryContext from scenario data
// ═══════════════════════════════════════════════════════════════

struct ScenarioBuilder {
    embeddings: HashMap<String, Vec<Vec<f32>>>,
    anomaly_scores: HashMap<String, HashMap<String, Vec<f32>>>,
    edges: HashMap<(String, String, String), Vec<(usize, usize)>>,
    node_names: HashMap<String, Vec<String>>,
    node_counts: HashMap<String, usize>,
    user_emb: Vec<f32>,
}

impl ScenarioBuilder {
    fn new() -> Self {
        // Create user node
        let user_emb: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut embeddings: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
        embeddings.insert("user".into(), vec![user_emb.clone()]);
        let mut node_names: HashMap<String, Vec<String>> = HashMap::new();
        node_names.insert("user".into(), vec!["Alice".into()]);
        let mut node_counts: HashMap<String, usize> = HashMap::new();
        node_counts.insert("user".into(), 1);

        Self {
            embeddings,
            anomaly_scores: {
                let mut m = HashMap::new();
                m.insert("SAGE".into(), HashMap::new());
                m
            },
            edges: HashMap::new(),
            node_names,
            node_counts,
            user_emb,
        }
    }

    /// Add a target entity connected to user via a TQL relation.
    fn add_entity(&mut self, node_type: &str, name: &str, relation: &str, anomaly_score: f32) {
        let dim = self.user_emb.len();
        let node_id = self.node_counts.get(node_type).copied().unwrap_or(0);

        // Create embedding (vary by node_id to avoid identical embeddings)
        let emb: Vec<f32> = (0..dim)
            .map(|d| ((node_id * 7 + d) as f32 * 0.13 + anomaly_score).sin())
            .collect();

        self.embeddings
            .entry(node_type.into())
            .or_default()
            .push(emb);
        self.node_names
            .entry(node_type.into())
            .or_default()
            .push(name.into());
        *self.node_counts.entry(node_type.into()).or_insert(0) += 1;

        // Set anomaly score
        self.anomaly_scores
            .get_mut("SAGE")
            .unwrap()
            .entry(node_type.into())
            .or_default()
            .push(anomaly_score);

        // Connect user to entity
        self.edges
            .entry(("user".into(), relation.into(), node_type.into()))
            .or_default()
            .push((0, node_id));
    }

    fn build(&self) -> FiduciaryContext {
        FiduciaryContext {
            user_emb: &self.user_emb,
            embeddings: &self.embeddings,
            anomaly_scores: &self.anomaly_scores,
            edges: &self.edges,
            node_names: &self.node_names,
            node_counts: &self.node_counts,
            user_type: "user".into(),
            user_id: 0,
            hidden_dim: self.user_emb.len(),
        }
    }
}

fn print_recommendations(label: &str, resp: &FiduciaryResponse) {
    println!("\n  ── {} ──\n", label);
    println!("  User: {}", resp.user_name);
    println!("  Domains: {:?}", resp.domains_covered);
    println!("  Actions triggered: {}\n", resp.action_types_triggered);
    for rec in &resp.recommendations {
        let flag = if rec.is_recommended { "✅" } else { "ℹ️" };
        println!(
            "    {} #{:<2} [{:.3}] {:<22} │ {} → {}",
            flag,
            rec.rank,
            rec.fiduciary_score,
            rec.action_type,
            rec.target_node_type,
            rec.target_name,
        );
    }
    println!();
}

// ═══════════════════════════════════════════════════════════════
// Scenario 1: User with high-interest debt
//
// Alice has a $8,200 credit card at 24.99% APR and a car loan at 6%.
// She also has a savings goal that's underfunded.
// Fiduciary should: pay down the high-interest debt BEFORE funding the goal.
// ═══════════════════════════════════════════════════════════════

#[test]
fn scenario_high_interest_debt() {
    let mut s = ScenarioBuilder::new();

    // High-interest credit card obligation (anomalous because high for this user)
    s.add_entity(
        "obligation",
        "CreditCard_24APR",
        "obligation-has-interest-term",
        0.65,
    );

    // Car loan (normal — lower rate, not anomalous)
    s.add_entity(
        "obligation",
        "CarLoan_6APR",
        "obligation-has-interest-term",
        0.15,
    );

    // Savings goal (underfunded)
    s.add_entity(
        "goal",
        "EmergencyFund_Goal",
        "subledger-holds-goal-funds",
        0.1,
    );

    // Bank account (for transfer)
    s.add_entity(
        "instrument",
        "Checking_Account",
        "user-has-instrument",
        0.05,
    );

    let ctx = s.build();
    let resp = recommend(&ctx, None);
    print_recommendations("SCENARIO 1: High-Interest Debt", &resp);

    // The credit card should trigger refinance + investigate + avoid (high anomaly)
    let top_5_types: Vec<&str> = resp
        .recommendations
        .iter()
        .take(5)
        .map(|r| r.action_type.as_str())
        .collect();

    // Debt-related actions should rank ABOVE fund_goal
    let first_debt_rank = resp
        .recommendations
        .iter()
        .position(|r| {
            r.domain == "debt_obligations"
                || (r.action_type == "should_pay" && r.target_node_type == "obligation")
                || r.action_type == "should_investigate"
        })
        .unwrap_or(999);

    let first_goal_rank = resp
        .recommendations
        .iter()
        .position(|r| r.action_type == "should_fund_goal")
        .unwrap_or(999);

    println!("  First debt action at rank: {}", first_debt_rank + 1);
    println!("  First goal action at rank: {}", first_goal_rank + 1);

    assert!(
        first_debt_rank < first_goal_rank,
        "Fiduciary MUST recommend dealing with high-interest debt BEFORE funding goals! \
         Debt at rank {}, Goal at rank {}",
        first_debt_rank + 1,
        first_goal_rank + 1
    );

    // High-interest obligation should trigger refinance
    let has_refinance = resp
        .recommendations
        .iter()
        .any(|r| r.action_type == "should_refinance" && r.target_name == "CreditCard_24APR");
    assert!(
        has_refinance,
        "Should recommend refinancing the high-interest credit card"
    );
}

// ═══════════════════════════════════════════════════════════════
// Scenario 2: Tax deadline approaching
//
// Bob has a tax due event coming up, a tax sinking fund that's underfunded,
// and a tax exemption certificate he hasn't claimed.
// Fiduciary should: prepare_tax FIRST (deadline!), then fund_sinking,
// then claim_exemption.
// ═══════════════════════════════════════════════════════════════

#[test]
fn scenario_tax_deadline() {
    let mut s = ScenarioBuilder::new();

    // Tax due event (approaching — urgent)
    s.add_entity(
        "tax-due-event",
        "Q4_2025_TaxDue",
        "tax-liability-has-due-event",
        0.2,
    );

    // Tax sinking fund (underfunded)
    s.add_entity(
        "tax-sinking-fund",
        "FederalTaxReserve",
        "tax-sinking-fund-backed-by-account",
        0.1,
    );

    // Tax exemption not yet applied
    s.add_entity(
        "tax-exemption-certificate",
        "HomeOffice_Exemption",
        "tax-party-has-exemption-certificate",
        0.05,
    );

    // Tax scenario available
    s.add_entity(
        "tax-scenario",
        "WhatIf_MaxContrib",
        "tax-scenario-for-period",
        0.05,
    );

    // Regular checking account
    s.add_entity("instrument", "Checking", "user-has-instrument", 0.05);

    let ctx = s.build();
    let resp = recommend(&ctx, None);
    print_recommendations("SCENARIO 2: Tax Deadline Approaching", &resp);

    // prepare_tax should rank FIRST (highest urgency)
    assert_eq!(
        resp.recommendations[0].action_type, "should_prepare_tax",
        "Tax preparation must be #1 when deadline is approaching! Got: {}",
        resp.recommendations[0].action_type
    );

    // fund_tax_sinking should appear before run_tax_scenario
    let sinking_rank = resp
        .recommendations
        .iter()
        .position(|r| r.action_type == "should_fund_tax_sinking")
        .unwrap_or(999);
    let scenario_rank = resp
        .recommendations
        .iter()
        .position(|r| r.action_type == "should_run_tax_scenario")
        .unwrap_or(999);

    println!("  Fund sinking at rank: {}", sinking_rank + 1);
    println!("  Run scenario at rank: {}", scenario_rank + 1);

    assert!(
        sinking_rank < scenario_rank,
        "Funding tax sinking fund should rank above running scenarios! \
         Sinking at {}, Scenario at {}",
        sinking_rank + 1,
        scenario_rank + 1
    );

    // Tax domain should be the dominant domain
    let tax_count = resp
        .recommendations
        .iter()
        .filter(|r| r.domain == "tax_optimization")
        .count();
    assert!(
        tax_count >= 3,
        "Tax domain should have at least 3 recommendations"
    );
}

// ═══════════════════════════════════════════════════════════════
// Scenario 3: Anomalous merchant activity
//
// Carol has transactions at a suspicious merchant (anomaly=0.85)
// and a normal grocery merchant (anomaly=0.05).
// Fiduciary should: investigate + avoid the suspicious one FIRST.
// Normal merchant should NOT trigger avoidance.
// ═══════════════════════════════════════════════════════════════

#[test]
fn scenario_anomalous_merchant() {
    let mut s = ScenarioBuilder::new();

    // Suspicious merchant (high anomaly)
    s.add_entity("merchant", "SketchyOnline_Store", "transacts-at", 0.85);

    // Normal grocery (low anomaly)
    s.add_entity("merchant", "Whole_Foods", "transacts-at", 0.05);

    // Normal checking
    s.add_entity("instrument", "Checking", "user-has-instrument", 0.05);

    // Normal goal
    s.add_entity("goal", "Vacation_Fund", "subledger-holds-goal-funds", 0.1);

    let ctx = s.build();
    let resp = recommend(&ctx, None);
    print_recommendations("SCENARIO 3: Anomalous Merchant", &resp);

    // Top actions should be investigate + avoid for the sketchy merchant
    let top_3_targets: Vec<(&str, &str)> = resp
        .recommendations
        .iter()
        .take(3)
        .map(|r| (r.action_type.as_str(), r.target_name.as_str()))
        .collect();

    println!("  Top 3: {:?}", top_3_targets);

    // The sketchy merchant should appear in top actions
    let sketchy_in_top_3 = resp
        .recommendations
        .iter()
        .take(3)
        .any(|r| r.target_name == "SketchyOnline_Store");
    assert!(
        sketchy_in_top_3,
        "Anomalous merchant must appear in top 3 recommendations!"
    );

    // Investigate should rank above fund_goal for anomalous merchant
    let investigate_sketchy = resp
        .recommendations
        .iter()
        .position(|r| {
            r.action_type == "should_investigate" && r.target_name == "SketchyOnline_Store"
        })
        .unwrap_or(999);
    let fund_goal = resp
        .recommendations
        .iter()
        .position(|r| r.action_type == "should_fund_goal")
        .unwrap_or(999);

    assert!(
        investigate_sketchy < fund_goal,
        "Must investigate suspicious merchant BEFORE funding vacation! \
         Investigate at {}, Fund at {}",
        investigate_sketchy + 1,
        fund_goal + 1
    );

    // Normal merchant (Whole_Foods) should NOT be flagged for avoidance
    let avoid_whole_foods = resp
        .recommendations
        .iter()
        .any(|r| r.action_type == "should_avoid" && r.target_name == "Whole_Foods");
    assert!(
        !avoid_whole_foods,
        "Should NOT recommend avoiding a normal merchant (Whole_Foods, anomaly=0.05)"
    );
}

// ═══════════════════════════════════════════════════════════════
// Scenario 4: Unused subscriptions + underfunded goal
//
// Dave has 3 recurring subscriptions (one unused → anomalous pattern)
// and an emergency fund goal that's underfunded.
// Fiduciary should: cancel unused subscription, then fund the goal
// (redirect savings to goal).
// ═══════════════════════════════════════════════════════════════

#[test]
fn scenario_unused_subscription_redirect_to_goal() {
    let mut s = ScenarioBuilder::new();

    // Active Netflix subscription (user engages → low anomaly)
    s.add_entity(
        "recurring-pattern",
        "Netflix_Monthly",
        "pattern-owned-by",
        0.05,
    );

    // Unused gym membership (no engagement → flagged)
    s.add_entity(
        "recurring-pattern",
        "GymMembership_Unused",
        "pattern-owned-by",
        0.35,
    );

    // Unused magazine subscription
    s.add_entity(
        "recurring-pattern",
        "Magazine_Sub_Unused",
        "pattern-owned-by",
        0.30,
    );

    // Emergency fund goal (critical, underfunded)
    s.add_entity(
        "goal",
        "EmergencyFund_5000",
        "subledger-holds-goal-funds",
        0.1,
    );

    let ctx = s.build();
    let resp = recommend(&ctx, None);
    print_recommendations("SCENARIO 4: Cancel Unused → Fund Goal", &resp);

    // Cancel actions should appear
    let cancel_count = resp
        .recommendations
        .iter()
        .filter(|r| r.action_type == "should_cancel")
        .count();
    assert!(
        cancel_count >= 2,
        "Should recommend cancelling unused subscriptions, got {}",
        cancel_count
    );

    // Cancel unused should rank above fund_goal (free up money first)
    let first_cancel = resp
        .recommendations
        .iter()
        .position(|r| r.action_type == "should_cancel")
        .unwrap_or(999);
    let first_goal = resp
        .recommendations
        .iter()
        .position(|r| r.action_type == "should_fund_goal")
        .unwrap_or(999);

    assert!(
        first_cancel < first_goal,
        "Cancel unused subscriptions BEFORE funding goals (free up money first)! \
         Cancel at {}, Goal at {}",
        first_cancel + 1,
        first_goal + 1
    );
}

// ═══════════════════════════════════════════════════════════════
// Scenario 5: Unreconciled accounts + stale asset valuation
//
// Eve has instruments that haven't been reconciled and an asset
// whose valuation is stale.
// Fiduciary should: reconcile accounts, then revalue asset.
// ═══════════════════════════════════════════════════════════════

#[test]
fn scenario_reconcile_and_revalue() {
    let mut s = ScenarioBuilder::new();

    // Unreconciled checking account
    s.add_entity(
        "reconciliation-case",
        "Checking_Recon_Dec",
        "reconciliation-for-instrument",
        0.3,
    );

    // Unreconciled savings
    s.add_entity(
        "reconciliation-case",
        "Savings_Recon_Dec",
        "reconciliation-for-instrument",
        0.2,
    );

    // Stale asset valuation
    s.add_entity(
        "asset-valuation",
        "House_Valuation_2023",
        "asset-has-valuation",
        0.1,
    );

    // Normal checking
    s.add_entity("instrument", "Checking_Acct", "user-has-instrument", 0.05);

    let ctx = s.build();
    let resp = recommend(&ctx, None);
    print_recommendations("SCENARIO 5: Reconcile + Revalue", &resp);

    // Reconcile should rank above revalue (accuracy before informational)
    let first_reconcile = resp
        .recommendations
        .iter()
        .position(|r| r.action_type == "should_reconcile")
        .unwrap_or(999);
    let first_revalue = resp
        .recommendations
        .iter()
        .position(|r| r.action_type == "should_revalue_asset")
        .unwrap_or(999);

    assert!(
        first_reconcile < first_revalue,
        "Reconciliation (accuracy) should rank above revaluation (informational)! \
         Reconcile at {}, Revalue at {}",
        first_reconcile + 1,
        first_revalue + 1
    );

    // Both should appear
    assert!(first_reconcile < 999, "Should recommend reconciling");
    assert!(first_revalue < 999, "Should recommend revaluing");
}

// ═══════════════════════════════════════════════════════════════
// Scenario 6: Lien on asset + disputed obligation
//
// Frank has a lien on his house and a suspicious obligation charge.
// Fiduciary should: dispute the anomalous charge AND pay down the lien.
// Dispute should rank high (time-sensitive + anomalous).
// ═══════════════════════════════════════════════════════════════

#[test]
fn scenario_lien_and_dispute() {
    let mut s = ScenarioBuilder::new();

    // Lien on house (solid obligation, needs paydown)
    s.add_entity("asset", "House_Primary", "lien-on-asset", 0.1);

    // Suspicious obligation (anomalous amount — should dispute)
    s.add_entity("obligation", "SuspiciousFee_Charge", "has-obligation", 0.72);

    // Normal obligation (not disputable)
    s.add_entity("obligation", "Mortgage_Normal", "has-obligation", 0.08);

    // Budget for tracking
    s.add_entity(
        "budget-estimation",
        "Monthly_Budget",
        "records-budget-estimation",
        0.1,
    );

    let ctx = s.build();
    let resp = recommend(&ctx, None);
    print_recommendations("SCENARIO 6: Lien + Dispute", &resp);

    // Dispute should appear for the suspicious fee
    let has_dispute = resp
        .recommendations
        .iter()
        .any(|r| r.action_type == "should_dispute" && r.target_name == "SuspiciousFee_Charge");
    assert!(
        has_dispute,
        "Should recommend disputing the suspicious fee charge"
    );

    // Pay down lien should appear
    let has_paydown = resp
        .recommendations
        .iter()
        .any(|r| r.action_type == "should_pay_down_lien");
    assert!(
        has_paydown,
        "Should recommend paying down the lien on the house"
    );

    // Investigate suspicious should rank above adjust_budget
    let investigate_rank = resp
        .recommendations
        .iter()
        .position(|r| {
            (r.action_type == "should_investigate" || r.action_type == "should_dispute")
                && r.target_name == "SuspiciousFee_Charge"
        })
        .unwrap_or(999);
    let budget_rank = resp
        .recommendations
        .iter()
        .position(|r| r.action_type == "should_adjust_budget")
        .unwrap_or(999);

    assert!(
        investigate_rank < budget_rank,
        "Investigating/disputing suspicious fee must rank above adjusting budget! \
         Investigate at {}, Budget at {}",
        investigate_rank + 1,
        budget_rank + 1
    );
}

// ═══════════════════════════════════════════════════════════════
// Scenario 7: Complete financial health check
//
// Grace has the full picture: debt, goals, tax, subscriptions,
// reconciliation, and an anomalous merchant.
// Verifies the overall fiduciary priority order makes sense:
// 1. Investigate anomalies (safety first)
// 2. Address debt (avoid penalties)
// 3. Prepare taxes (deadline)
// 4. Cancel unused (save money)
// 5. Fund goals (build wealth)
// 6. Reconcile/revalue (housekeeping)
// ═══════════════════════════════════════════════════════════════

#[test]
fn scenario_complete_financial_health() {
    let mut s = ScenarioBuilder::new();

    // 1. Anomalous merchant (safety: investigate first)
    s.add_entity("merchant", "Fraud_Merchant", "transacts-at", 0.90);

    // 2. High-interest debt (financial urgency)
    s.add_entity(
        "obligation",
        "HighRate_CreditCard",
        "obligation-has-interest-term",
        0.55,
    );

    // 3. Tax deadline approaching
    s.add_entity(
        "tax-due-event",
        "April15_TaxDue",
        "tax-liability-has-due-event",
        0.2,
    );

    // 4. Unused subscription
    s.add_entity(
        "recurring-pattern",
        "Unused_Streaming",
        "pattern-owned-by",
        0.3,
    );

    // 5. Underfunded goal
    s.add_entity("goal", "Retirement_401k", "subledger-holds-goal-funds", 0.1);

    // 6. Stale reconciliation
    s.add_entity(
        "reconciliation-case",
        "Jan_Recon",
        "reconciliation-for-instrument",
        0.15,
    );

    // 7. Normal checking
    s.add_entity("instrument", "Main_Checking", "user-has-instrument", 0.03);

    let ctx = s.build();
    let resp = recommend(&ctx, None);
    print_recommendations("SCENARIO 7: Complete Financial Health", &resp);

    // Extract ranks for key action types
    let find_rank = |action: &str| -> usize {
        resp.recommendations
            .iter()
            .position(|r| r.action_type == action)
            .unwrap_or(999)
    };

    let fraud_rank = resp
        .recommendations
        .iter()
        .position(|r| {
            (r.action_type == "should_investigate" || r.action_type == "should_avoid")
                && r.target_name == "Fraud_Merchant"
        })
        .unwrap_or(999);
    let tax_rank = find_rank("should_prepare_tax");
    let debt_rank = resp
        .recommendations
        .iter()
        .position(|r| r.domain == "debt_obligations" && r.target_name == "HighRate_CreditCard")
        .unwrap_or(999);
    let cancel_rank = find_rank("should_cancel");
    let goal_rank = find_rank("should_fund_goal");
    let recon_rank = find_rank("should_reconcile");

    println!("  Priority order verification:");
    println!("    Fraud/safety:  rank {}", fraud_rank + 1);
    println!("    Tax deadline:  rank {}", tax_rank + 1);
    println!("    Debt address:  rank {}", debt_rank + 1);
    println!("    Cancel unused: rank {}", cancel_rank + 1);
    println!("    Fund goals:    rank {}", goal_rank + 1);
    println!("    Reconcile:     rank {}", recon_rank + 1);

    // Safety first: fraud investigation before everything else
    assert!(
        fraud_rank < goal_rank,
        "Safety (fraud) must come before wealth building (goals)!"
    );

    // Tax deadline before goal funding
    assert!(
        tax_rank < goal_rank,
        "Tax deadline must come before goal funding!"
    );

    // Cancel unused before fund goal (free up money first)
    assert!(
        cancel_rank < goal_rank,
        "Cancel unused subscriptions before funding goals!"
    );

    // Domains coverage: should cover most domains
    assert!(
        resp.domains_covered.len() >= 5,
        "Complete health check should cover at least 5 domains, got {}",
        resp.domains_covered.len()
    );

    assert!(
        resp.action_types_triggered >= 8,
        "Complete health check should trigger at least 8 action types, got {}",
        resp.action_types_triggered
    );
}
