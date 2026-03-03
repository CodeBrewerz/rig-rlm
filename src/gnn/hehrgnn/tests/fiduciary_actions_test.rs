//! Test for all 18 ontology-driven fiduciary action types.
//!
//! Creates a graph with entities and relations from the TQL schema
//! to trigger every fiduciary action type and verify scoring.

use std::collections::HashMap;

use hehrgnn::eval::fiduciary::*;

/// Build a minimal graph that triggers all 18 fiduciary action types.
fn build_test_context() -> (
    Vec<f32>,                                               // user_emb
    HashMap<String, Vec<Vec<f32>>>,                         // embeddings
    HashMap<String, HashMap<String, Vec<f32>>>,             // anomaly_scores
    HashMap<(String, String, String), Vec<(usize, usize)>>, // edges
    HashMap<String, Vec<String>>,                           // node_names
    HashMap<String, usize>,                                 // node_counts
) {
    let hidden_dim = 8;

    // Node types from TQL ontology
    let node_types = vec![
        "user",
        "instrument",
        "obligation",
        "asset",
        "asset-valuation",
        "goal",
        "budget-estimation",
        "tax-due-event",
        "tax-sinking-fund",
        "tax-exemption-certificate",
        "tax-scenario",
        "reconciliation-case",
        "recurring-pattern",
        "recurring-missing-alert",
        "merchant",
        "transaction",
        "rate-observation",
    ];

    let mut embeddings: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    let mut node_names: HashMap<String, Vec<String>> = HashMap::new();
    let mut node_counts: HashMap<String, usize> = HashMap::new();

    // Create 2 nodes per type with deterministic embeddings
    for (ti, nt) in node_types.iter().enumerate() {
        let mut vecs = Vec::new();
        let mut names = Vec::new();
        for ni in 0..2 {
            let emb: Vec<f32> = (0..hidden_dim)
                .map(|d| ((ti * 2 + ni + d) as f32 * 0.1).sin())
                .collect();
            vecs.push(emb);
            names.push(format!("{}_{}", nt, ni));
        }
        embeddings.insert(nt.to_string(), vecs);
        node_names.insert(nt.to_string(), names);
        node_counts.insert(nt.to_string(), 2);
    }

    let user_emb = embeddings["user"][0].clone();

    // Anomaly scores — make some targets anomalous
    let mut model_scores: HashMap<String, Vec<f32>> = HashMap::new();
    for nt in &node_types {
        let scores = if *nt == "merchant" || *nt == "transaction" {
            vec![0.7, 0.2] // First merchant/tx is anomalous
        } else if *nt == "obligation" {
            vec![0.6, 0.1] // First obligation is anomalous → dispute
        } else {
            vec![0.1, 0.1]
        };
        model_scores.insert(nt.to_string(), scores);
    }
    let mut anomaly_scores: HashMap<String, HashMap<String, Vec<f32>>> = HashMap::new();
    anomaly_scores.insert("SAGE".into(), model_scores);

    // Edges: connect user (node 0) to entities via TQL relations
    let mut edges: HashMap<(String, String, String), Vec<(usize, usize)>> = HashMap::new();

    // ── Core triggers ──
    // user → instrument (owns → transfer + consolidate)
    edges.insert(
        (
            "user".into(),
            "user-has-instrument".into(),
            "instrument".into(),
        ),
        vec![(0, 0), (0, 1)],
    );

    // user → merchant (high anomaly → investigate + avoid)
    edges.insert(
        ("user".into(), "transacts-at".into(), "merchant".into()),
        vec![(0, 0)], // Merchant 0 is anomalous (0.7)
    );

    // user → transaction (high anomaly → investigate)
    edges.insert(
        (
            "user".into(),
            "has-transaction".into(),
            "transaction".into(),
        ),
        vec![(0, 0)], // Tx 0 is anomalous (0.7)
    );

    // recurring pattern (subscribe → cancel)
    edges.insert(
        (
            "user".into(),
            "pattern-owned-by".into(),
            "recurring-pattern".into(),
        ),
        vec![(0, 0)],
    );

    // ── Debt & Obligations ──
    // user → obligation via interest term → refinance
    edges.insert(
        (
            "user".into(),
            "obligation-has-interest-term".into(),
            "obligation".into(),
        ),
        vec![(0, 0)],
    );
    // Make obligation connected to user (also triggers dispute via anomaly 0.6)
    edges.insert(
        ("user".into(), "has-obligation".into(), "obligation".into()),
        vec![(0, 0), (0, 1)],
    );
    // user → payment relation → should_pay
    edges.insert(
        (
            "user".into(),
            "settlement-for-shares".into(),
            "obligation".into(),
        ),
        vec![(0, 1)], // Non-anomalous obligation triggers pay
    );

    // user → asset via lien → pay_down_lien
    edges.insert(
        ("user".into(), "lien-on-asset".into(), "asset".into()),
        vec![(0, 0)],
    );

    // anomalous obligation → dispute (obligation 0 has anomaly 0.6)
    // (triggered by anomaly check in infer_actions when ttype contains "obligation")

    // ── Goals & Budgets ──
    // goal → fund_goal
    edges.insert(
        (
            "user".into(),
            "subledger-holds-goal-funds".into(),
            "goal".into(),
        ),
        vec![(0, 0)],
    );

    // budget → adjust_budget
    edges.insert(
        (
            "user".into(),
            "records-budget-estimation".into(),
            "budget-estimation".into(),
        ),
        vec![(0, 0)],
    );

    // ── Tax Optimization ──
    // tax due event → prepare_tax
    edges.insert(
        (
            "user".into(),
            "tax-liability-has-due-event".into(),
            "tax-due-event".into(),
        ),
        vec![(0, 0)],
    );

    // tax sinking fund → fund_tax_sinking
    edges.insert(
        (
            "user".into(),
            "tax-sinking-fund-backed-by-account".into(),
            "tax-sinking-fund".into(),
        ),
        vec![(0, 0)],
    );

    // tax exemption → claim_exemption
    edges.insert(
        (
            "user".into(),
            "tax-party-has-exemption-certificate".into(),
            "tax-exemption-certificate".into(),
        ),
        vec![(0, 0)],
    );

    // tax scenario → run_tax_scenario
    edges.insert(
        (
            "user".into(),
            "tax-scenario-for-period".into(),
            "tax-scenario".into(),
        ),
        vec![(0, 0)],
    );

    // ── Reconciliation ──
    edges.insert(
        (
            "user".into(),
            "reconciliation-for-instrument".into(),
            "reconciliation-case".into(),
        ),
        vec![(0, 0)],
    );

    // ── Recurring Patterns ──
    edges.insert(
        (
            "user".into(),
            "pattern-has-recurring-alert".into(),
            "recurring-missing-alert".into(),
        ),
        vec![(0, 0)],
    );

    // ── Asset Management ──
    edges.insert(
        (
            "user".into(),
            "asset-has-valuation".into(),
            "asset-valuation".into(),
        ),
        vec![(0, 0)],
    );

    (
        user_emb,
        embeddings,
        anomaly_scores,
        edges,
        node_names,
        node_counts,
    )
}

#[test]
fn test_all_18_action_types_exist() {
    let all = FiduciaryActionType::all();
    assert_eq!(all.len(), 18, "Expected 18 fiduciary action types");

    // Verify names are unique
    let names: Vec<&str> = all.iter().map(|a| a.name()).collect();
    let unique: std::collections::HashSet<&&str> = names.iter().collect();
    assert_eq!(unique.len(), 18, "Action type names must be unique");

    // Verify domains cover 7 domains
    let domains: std::collections::HashSet<&str> = all.iter().map(|a| a.domain()).collect();
    assert_eq!(domains.len(), 7, "Expected 7 ontology domains");
    assert!(domains.contains("core"));
    assert!(domains.contains("debt_obligations"));
    assert!(domains.contains("goals_budgets"));
    assert!(domains.contains("tax_optimization"));
    assert!(domains.contains("reconciliation"));
    assert!(domains.contains("recurring_patterns"));
    assert!(domains.contains("asset_management"));
}

#[test]
fn test_priority_weights_ordered() {
    let all = FiduciaryActionType::all();
    for action in &all {
        let w = action.priority_weight();
        assert!(
            w > 0.0 && w <= 1.0,
            "{} has invalid weight {}",
            action.name(),
            w
        );
    }
    // Investigate should be highest
    assert_eq!(
        FiduciaryActionType::ShouldInvestigate.priority_weight(),
        1.0
    );
}

#[test]
fn test_fiduciary_scoring_all_axes() {
    let (user_emb, embeddings, anomaly_scores, edges, node_names, node_counts) =
        build_test_context();

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

    // Score every action type against instrument 0
    for action in FiduciaryActionType::all() {
        let axes = score_action(action, "instrument", 0, &ctx);
        let score = axes.score();

        // All axes must be in [0, 1]
        assert!(
            axes.cost_reduction >= 0.0 && axes.cost_reduction <= 1.0,
            "{}: cost_reduction={}",
            action.name(),
            axes.cost_reduction
        );
        assert!(
            axes.risk_reduction >= 0.0 && axes.risk_reduction <= 1.0,
            "{}: risk_reduction={}",
            action.name(),
            axes.risk_reduction
        );
        assert!(
            axes.goal_alignment >= 0.0 && axes.goal_alignment <= 1.0,
            "{}: goal_alignment={}",
            action.name(),
            axes.goal_alignment
        );
        assert!(
            axes.urgency >= 0.0 && axes.urgency <= 1.0,
            "{}: urgency={}",
            action.name(),
            axes.urgency
        );
        assert!(
            axes.conflict_freedom >= 0.0 && axes.conflict_freedom <= 1.0,
            "{}: conflict_freedom={}",
            action.name(),
            axes.conflict_freedom
        );
        assert!(
            axes.reversibility >= 0.0 && axes.reversibility <= 1.0,
            "{}: reversibility={}",
            action.name(),
            axes.reversibility
        );

        // Weighted score must be positive
        assert!(score > 0.0, "{}: weighted score={}", action.name(), score);
    }
}

#[test]
fn test_candidate_generation_triggers_all_domains() {
    let (user_emb, embeddings, anomaly_scores, edges, node_names, node_counts) =
        build_test_context();

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

    let candidates = generate_candidates(&ctx);

    // Collect triggered action types
    let triggered: std::collections::HashSet<String> = candidates
        .iter()
        .map(|(action, _, _)| action.name().to_string())
        .collect();

    println!("\n  ── FIDUCIARY CANDIDATES ({}) ──\n", candidates.len());
    for (action, target_type, target_id) in &candidates {
        println!(
            "    {} → {} ({}:{})",
            action.name(),
            action.verb(),
            target_type,
            target_id
        );
    }
    println!("\n  Triggered {} unique action types:", triggered.len());
    let mut sorted: Vec<&String> = triggered.iter().collect();
    sorted.sort();
    for t in &sorted {
        println!("    ✅ {}", t);
    }

    // Verify core actions triggered
    assert!(
        triggered.contains("should_transfer"),
        "Missing: should_transfer"
    );
    assert!(
        triggered.contains("should_consolidate"),
        "Missing: should_consolidate"
    );
    assert!(
        triggered.contains("should_investigate"),
        "Missing: should_investigate"
    );
    assert!(triggered.contains("should_avoid"), "Missing: should_avoid");
    assert!(
        triggered.contains("should_cancel"),
        "Missing: should_cancel"
    );
    assert!(triggered.contains("should_pay"), "Missing: should_pay");

    // Verify debt/obligation actions
    assert!(
        triggered.contains("should_refinance"),
        "Missing: should_refinance"
    );
    assert!(
        triggered.contains("should_pay_down_lien"),
        "Missing: should_pay_down_lien"
    );

    // Verify goal/budget actions
    assert!(
        triggered.contains("should_fund_goal"),
        "Missing: should_fund_goal"
    );
    assert!(
        triggered.contains("should_adjust_budget"),
        "Missing: should_adjust_budget"
    );

    // Verify tax actions
    assert!(
        triggered.contains("should_prepare_tax"),
        "Missing: should_prepare_tax"
    );
    assert!(
        triggered.contains("should_fund_tax_sinking"),
        "Missing: should_fund_tax_sinking"
    );
    assert!(
        triggered.contains("should_claim_exemption"),
        "Missing: should_claim_exemption"
    );
    assert!(
        triggered.contains("should_run_tax_scenario"),
        "Missing: should_run_tax_scenario"
    );

    // Verify reconciliation
    assert!(
        triggered.contains("should_reconcile"),
        "Missing: should_reconcile"
    );

    // Verify recurring patterns
    assert!(
        triggered.contains("should_review_recurring"),
        "Missing: should_review_recurring"
    );

    // Verify asset management
    assert!(
        triggered.contains("should_revalue_asset"),
        "Missing: should_revalue_asset"
    );
}

#[test]
fn test_full_recommendation_pipeline() {
    let (user_emb, embeddings, anomaly_scores, edges, node_names, node_counts) =
        build_test_context();

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

    println!("\n  ── FIDUCIARY RECOMMENDATIONS ──\n");
    println!(
        "  User: {} ({}:{})",
        response.user_name, response.user_node_type, response.user_node_id
    );
    println!(
        "  Action types triggered: {}",
        response.action_types_triggered
    );
    println!("  Domains covered: {:?}", response.domains_covered);
    println!("  Recommendations: {}\n", response.recommendations.len());

    for rec in &response.recommendations {
        println!(
            "    #{} [{:.3}] {} │ {} │ {} → {} │ anomaly={:.2}",
            rec.rank,
            rec.fiduciary_score,
            rec.domain,
            rec.action_type,
            rec.target_node_type,
            rec.target_name,
            rec.target_anomaly_score,
        );
    }

    println!("\n  Assessment: {}", response.assessment);

    // Verify response structure
    assert!(
        response.recommendations.len() > 0,
        "Should have recommendations"
    );
    assert!(
        response.action_types_triggered >= 10,
        "Should trigger many action types"
    );
    assert!(
        response.domains_covered.len() >= 5,
        "Should cover multiple domains"
    );

    // Verify ranking is sorted by fiduciary score
    for i in 1..response.recommendations.len() {
        assert!(
            response.recommendations[i - 1].fiduciary_score
                >= response.recommendations[i].fiduciary_score,
            "Recommendations not sorted at position {}",
            i
        );
    }

    // Verify ranks are sequential
    for (i, rec) in response.recommendations.iter().enumerate() {
        assert_eq!(rec.rank, i + 1, "Rank mismatch at position {}", i);
    }

    // Verify all recommendations have reasoning
    for rec in &response.recommendations {
        assert!(
            !rec.reasoning.is_empty(),
            "Missing reasoning for {}",
            rec.action_type
        );
    }

    // Verify investigation/avoidance appears for anomalous merchant
    let investigate_merchant = response
        .recommendations
        .iter()
        .any(|r| r.action_type == "should_investigate" && r.target_node_type == "merchant");
    assert!(
        investigate_merchant,
        "Should recommend investigating anomalous merchant"
    );

    // Verify models used
    assert_eq!(response.models_used.len(), 4, "Should list 4 models");
}

#[test]
fn test_each_action_has_correct_metadata() {
    for action in FiduciaryActionType::all() {
        // Every action must have non-empty name, verb, reasoning, domain
        assert!(!action.name().is_empty(), "Empty name for {:?}", action);
        assert!(!action.verb().is_empty(), "Empty verb for {:?}", action);
        assert!(
            !action.reasoning_suffix().is_empty(),
            "Empty reasoning for {:?}",
            action
        );
        assert!(!action.domain().is_empty(), "Empty domain for {:?}", action);

        // Priority weight in (0, 1]
        let w = action.priority_weight();
        assert!(w > 0.0 && w <= 1.0, "Invalid weight {} for {:?}", w, action);
    }
}
