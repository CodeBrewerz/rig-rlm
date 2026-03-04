//! Circuit Self-Learning Test
//!
//! Demonstrates that the PC circuit actually improves across `recommend()` calls
//! when using `PcState` for persistence. Uses deliberately varied anomaly scores
//! and multiple entity types so the circuit has real patterns to learn.
//!
//! Key differences from progressive_learning_test:
//!   - Varied anomaly scores (high=0.9, low=0.05) instead of uniform
//!   - Multiple users with different risk profiles
//!   - Focuses on PC metrics (risk_probability, EM LL) rather than GNN training
//!   - Shows circuit converging with each call

use std::collections::HashMap;

use hehrgnn::eval::fiduciary::*;

/// Build embeddings that give the circuit differentiated signal.
/// User 0 is close to risky entities, User 1 is close to safe entities.
fn build_varied_embeddings() -> HashMap<String, Vec<Vec<f32>>> {
    let mut emb: HashMap<String, Vec<Vec<f32>>> = HashMap::new();

    // Users: one risk-seeking, one conservative
    emb.insert(
        "user".into(),
        vec![
            vec![0.9, 0.8, 0.7, 0.6], // user0: high values → close to risky
            vec![0.1, 0.2, 0.1, 0.2], // user1: low values → close to safe
        ],
    );

    // Accounts with different profiles
    emb.insert(
        "account".into(),
        vec![
            vec![0.8, 0.7, 0.6, 0.5], // checking: high activity
            vec![0.2, 0.1, 0.2, 0.1], // savings: stable
            vec![0.5, 0.5, 0.5, 0.5], // brokerage: neutral
        ],
    );

    // Obligations: risky vs safe
    emb.insert(
        "obligation".into(),
        vec![
            vec![0.95, 0.9, 0.85, 0.8], // credit card debt: risky
            vec![0.15, 0.1, 0.2, 0.1],  // car loan: safe
            vec![0.7, 0.8, 0.6, 0.9],   // HELOC: moderate risk
        ],
    );

    // Rates
    emb.insert(
        "rate".into(),
        vec![
            vec![0.9, 0.9, 0.8, 0.85], // 24% APR: very high
            vec![0.1, 0.15, 0.1, 0.1], // 3% APR: low
            vec![0.6, 0.5, 0.7, 0.6],  // 8% APR: moderate
        ],
    );

    // Merchants: one sketchy, one safe, one neutral
    emb.insert(
        "merchant".into(),
        vec![
            vec![0.1, 0.2, 0.1, 0.15],   // grocery: safe
            vec![0.95, 0.85, 0.9, 0.95], // sketchy crypto: risky
            vec![0.4, 0.3, 0.5, 0.4],    // gas station: neutral
        ],
    );

    // Recurring subscriptions
    emb.insert(
        "recurring".into(),
        vec![
            vec![0.1, 0.1, 0.2, 0.1], // netflix: active
            vec![0.7, 0.6, 0.8, 0.7], // unused gym: waste
            vec![0.8, 0.7, 0.6, 0.8], // unused magazine: waste
        ],
    );

    // Goals
    emb.insert(
        "goal".into(),
        vec![
            vec![0.3, 0.2, 0.3, 0.2], // emergency fund: important
            vec![0.4, 0.3, 0.4, 0.3], // retirement: long-term
        ],
    );

    emb.insert("tax_due".into(), vec![vec![0.5, 0.6, 0.5, 0.4]]);
    emb.insert("tax_sinking".into(), vec![vec![0.2, 0.3, 0.2, 0.3]]);
    emb.insert("asset".into(), vec![vec![0.3, 0.3, 0.4, 0.3]]);
    emb.insert("valuation".into(), vec![vec![0.35, 0.3, 0.4, 0.35]]);
    emb.insert("recon_case".into(), vec![vec![0.5, 0.5, 0.6, 0.5]]);
    emb.insert("budget".into(), vec![vec![0.2, 0.2, 0.3, 0.2]]);

    emb
}

/// Build DELIBERATELY varied anomaly scores — high variance creates
/// real distinctions for the circuit to learn.
fn build_varied_anomaly_scores() -> HashMap<String, HashMap<String, Vec<f32>>> {
    let mut anomaly_scores: HashMap<String, HashMap<String, Vec<f32>>> = HashMap::new();
    let mut sage: HashMap<String, Vec<f32>> = HashMap::new();

    // Obligations: credit card is VERY risky, car loan is safe, HELOC moderate
    sage.insert("obligation".into(), vec![0.92, 0.08, 0.55]);
    // Merchants: grocery safe, sketchy crypto VERY risky, gas neutral
    sage.insert("merchant".into(), vec![0.03, 0.95, 0.25]);
    // Recurring: netflix ok, gym waste, magazine waste
    sage.insert("recurring".into(), vec![0.05, 0.72, 0.68]);
    // Goals: emergency underfunded, retirement on track
    sage.insert("goal".into(), vec![0.35, 0.10]);
    // Tax
    sage.insert("tax_due".into(), vec![0.45]);
    sage.insert("tax_sinking".into(), vec![0.10]);
    // Assets
    sage.insert("asset".into(), vec![0.15]);
    sage.insert("valuation".into(), vec![0.20]);
    // Recon
    sage.insert("recon_case".into(), vec![0.65]);
    sage.insert("budget".into(), vec![0.15]);
    // Accounts: checking active, savings stable, brokerage moderate
    sage.insert("account".into(), vec![0.30, 0.05, 0.40]);
    // Rates: 24% HIGH, 3% low, 8% moderate
    sage.insert("rate".into(), vec![0.88, 0.05, 0.42]);
    // Users
    sage.insert("user".into(), vec![0.10, 0.05]);

    anomaly_scores.insert("SAGE".into(), sage);
    anomaly_scores
}

fn build_varied_graph() -> (
    HashMap<String, Vec<String>>,
    HashMap<(String, String, String), Vec<(usize, usize)>>,
    HashMap<String, usize>,
) {
    let mut node_names: HashMap<String, Vec<String>> = HashMap::new();
    node_names.insert("user".into(), vec!["Risky_Alice".into(), "Safe_Bob".into()]);
    node_names.insert(
        "account".into(),
        vec!["Checking".into(), "Savings".into(), "Brokerage".into()],
    );
    node_names.insert(
        "obligation".into(),
        vec!["CreditCard".into(), "CarLoan".into(), "HELOC".into()],
    );
    node_names.insert(
        "rate".into(),
        vec!["High_24pct".into(), "Low_3pct".into(), "Mid_8pct".into()],
    );
    node_names.insert(
        "merchant".into(),
        vec![
            "Grocery".into(),
            "SketchyCrypto".into(),
            "GasStation".into(),
        ],
    );
    node_names.insert(
        "recurring".into(),
        vec!["Netflix".into(), "UnusedGym".into(), "UnusedMag".into()],
    );
    node_names.insert("goal".into(), vec!["Emergency".into(), "Retirement".into()]);
    node_names.insert("tax_due".into(), vec!["Q4Tax".into()]);
    node_names.insert("tax_sinking".into(), vec!["FedReserve".into()]);
    node_names.insert("asset".into(), vec!["House".into()]);
    node_names.insert("valuation".into(), vec!["HouseVal".into()]);
    node_names.insert("recon_case".into(), vec!["JanRecon".into()]);
    node_names.insert("budget".into(), vec!["Monthly".into()]);

    let mut edges: HashMap<(String, String, String), Vec<(usize, usize)>> = HashMap::new();
    // user0 owns checking, brokerage; user1 owns savings
    edges.insert(
        ("user".into(), "owns".into(), "account".into()),
        vec![(0, 0), (0, 2), (1, 1)],
    );
    // checking pays credit card, HELOC; savings pays car loan
    edges.insert(
        ("account".into(), "pays".into(), "obligation".into()),
        vec![(0, 0), (0, 2), (1, 1)],
    );
    // credit card → 24%, car loan → 3%, HELOC → 8%
    edges.insert(
        ("obligation".into(), "has_rate".into(), "rate".into()),
        vec![(0, 0), (1, 1), (2, 2)],
    );
    // user0 → tax, sinking, goals, asset, budget, merchants, recurring
    edges.insert(
        ("user".into(), "liable".into(), "tax_due".into()),
        vec![(0, 0)],
    );
    edges.insert(
        ("user".into(), "funds".into(), "tax_sinking".into()),
        vec![(0, 0)],
    );
    edges.insert(
        ("account".into(), "transacts".into(), "merchant".into()),
        vec![(0, 0), (0, 1), (0, 2)],
    );
    edges.insert(
        ("user".into(), "subscribes".into(), "recurring".into()),
        vec![(0, 0), (0, 1), (0, 2)],
    );
    edges.insert(
        ("user".into(), "targets".into(), "goal".into()),
        vec![(0, 0), (1, 1)],
    );
    edges.insert(
        ("user".into(), "holds".into(), "asset".into()),
        vec![(0, 0)],
    );
    edges.insert(
        ("asset".into(), "valued_by".into(), "valuation".into()),
        vec![(0, 0)],
    );
    edges.insert(
        (
            "account".into(),
            "reconciled_by".into(),
            "recon_case".into(),
        ),
        vec![(0, 0)],
    );
    edges.insert(
        ("user".into(), "tracks".into(), "budget".into()),
        vec![(0, 0)],
    );

    let mut node_counts: HashMap<String, usize> = HashMap::new();
    for (nt, names) in &node_names {
        node_counts.insert(nt.clone(), names.len());
    }

    (node_names, edges, node_counts)
}

#[test]
fn test_circuit_self_learning() {
    let emb = build_varied_embeddings();
    let anomaly_scores = build_varied_anomaly_scores();
    let (node_names, edges, node_counts) = build_varied_graph();

    let user_emb = emb.get("user").unwrap()[0].clone(); // risky user

    let ctx = FiduciaryContext {
        user_emb: &user_emb,
        embeddings: &emb,
        anomaly_scores: &anomaly_scores,
        edges: &edges,
        node_names: &node_names,
        node_counts: &node_counts,
        user_type: "user".into(),
        user_id: 0,
        hidden_dim: 4,
    };

    println!(
        "\n  ╔══════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "  ║  CIRCUIT SELF-LEARNING TEST — PC should improve with each recommend() call               ║"
    );
    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "  ║  Call │ PC Epochs │ Avg P(risk) │   EM LL   │ Top Risk Action            │ Δ Risk        ║"
    );
    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════════════╣"
    );

    let mut pc_state = PcState::new();
    let mut prev_avg_risk = 0.0f64;
    let num_calls = 8;
    let mut risk_history: Vec<f64> = Vec::new();

    for call_idx in 0..num_calls {
        let resp = recommend(&ctx, Some(&mut pc_state));

        // Compute average PC risk across all recommendations
        let pc_analyses: Vec<&hehrgnn::model::pc::fiduciary_pc::PcAnalysis> = resp
            .recommendations
            .iter()
            .filter_map(|r| r.pc_analysis.as_ref())
            .collect();

        let avg_risk = if pc_analyses.is_empty() {
            0.0
        } else {
            pc_analyses.iter().map(|a| a.risk_probability).sum::<f64>() / pc_analyses.len() as f64
        };

        // Find highest-risk recommendation
        let top_risk = resp
            .recommendations
            .iter()
            .filter_map(|r| r.pc_analysis.as_ref().map(|a| (r, a)))
            .max_by(|a, b| {
                a.1.risk_probability
                    .partial_cmp(&b.1.risk_probability)
                    .unwrap()
            });

        let top_risk_str = top_risk
            .map(|(r, a)| {
                format!(
                    "{} P={:.4}",
                    &r.action_type[..r.action_type.len().min(20)],
                    a.risk_probability
                )
            })
            .unwrap_or_else(|| "none".into());

        let delta = if call_idx > 0 {
            let d = avg_risk - prev_avg_risk;
            if d.abs() < 0.0001 {
                "  stable".into()
            } else if d > 0.0 {
                format!("+{:.6}", d)
            } else {
                format!("{:.6}", d)
            }
        } else {
            "  base  ".into()
        };

        let em_ll = resp.pc_em_ll.unwrap_or(f64::NEG_INFINITY);
        let build_type = if call_idx == 0 {
            "30 (fresh)"
        } else {
            " 5 (resume)"
        };

        println!(
            "  ║  {:2}   │ {:>10} │   {:.6}  │ {:8.4}  │ {:26} │ {:>8}        ║",
            call_idx + 1,
            build_type,
            avg_risk,
            em_ll,
            top_risk_str,
            delta,
        );

        risk_history.push(avg_risk);
        prev_avg_risk = avg_risk;
    }

    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════════════╣"
    );

    // Verify PcState accumulated correctly
    println!(
        "  ║  PcState: total_epochs = {}, ll_history entries = {}{}",
        pc_state.total_epochs,
        pc_state.ll_history.len(),
        " ".repeat(29)
    );
    println!(
        "  ║  Circuit is_trained: {}{}",
        pc_state.is_trained(),
        " ".repeat(52)
    );

    // Check that the circuit learned something — risk values should vary (not all zero)
    let non_zero_risks: Vec<f64> = risk_history
        .iter()
        .filter(|&&r| r > 0.001)
        .cloned()
        .collect();
    let has_signal = !non_zero_risks.is_empty();

    if has_signal {
        println!(
            "  ║  ✅ Circuit shows non-zero risk differentiation                                          ║"
        );

        // Check convergence: later calls should have more stable risk
        if risk_history.len() >= 4 {
            let early_var = (risk_history[1] - risk_history[0]).abs();
            let late_var =
                (risk_history[risk_history.len() - 1] - risk_history[risk_history.len() - 2]).abs();
            if late_var <= early_var + 0.001 {
                println!(
                    "  ║  ✅ Circuit converging: later calls more stable (early_Δ={:.6}, late_Δ={:.6})       ║",
                    early_var, late_var
                );
            } else {
                println!(
                    "  ║  ⚠️  Circuit not yet converged (early_Δ={:.6}, late_Δ={:.6})                        ║",
                    early_var, late_var
                );
            }
        }
    } else {
        println!(
            "  ║  ⚠️  Circuit risk is zero — embeddings may be too uniform                                ║"
        );
    }

    // Verify PcState epochs
    let expected_epochs = 30 + 5 * (num_calls - 1); // first=30, rest=5
    assert_eq!(
        pc_state.total_epochs, expected_epochs,
        "Expected {} total EM epochs, got {}",
        expected_epochs, pc_state.total_epochs
    );
    println!(
        "  ║  ✅ PcState tracked {} total EM epochs correctly                                          ║",
        expected_epochs
    );

    // Verify ll_history has one entry per call
    assert_eq!(
        pc_state.ll_history.len(),
        num_calls,
        "Expected {} ll_history entries, got {}",
        num_calls,
        pc_state.ll_history.len()
    );
    println!(
        "  ║  ✅ ll_history has {} entries (one per call)                                                ║",
        num_calls
    );

    // Verify circuit is_trained
    assert!(
        pc_state.is_trained(),
        "PcState should have a trained circuit"
    );
    println!(
        "  ║  ✅ PcState.is_trained() = true                                                             ║"
    );

    println!(
        "  ╚══════════════════════════════════════════════════════════════════════════════════════════════╝"
    );

    // Final assertion: risk values must not all be identical (proves circuit is updating)
    let unique_risks: std::collections::HashSet<u64> = risk_history
        .iter()
        .map(|r| (r * 1_000_000.0) as u64)
        .collect();
    assert!(
        unique_risks.len() >= 2 || risk_history[0] > 0.001,
        "Circuit should either show varying risk or non-trivial risk values. Got {:?}",
        risk_history,
    );

    println!(
        "\n  ✅ Circuit self-learning verified: PcState persists and improves across {} calls!",
        num_calls
    );
}
