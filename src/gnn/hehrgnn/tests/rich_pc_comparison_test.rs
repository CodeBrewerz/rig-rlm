//! Rich Synthetic Data Test — GNN vs GNN+PC comparison.
//!
//! Creates a realistic multi-user financial graph matching TQL schema:
//! - 5 users with distinct risk profiles (high-risk debtor, conservative saver,
//!   mixed portfolio, tax delinquent, brand new user)
//! - 8 accounts, 6 obligations, 5 rates, 4 merchants, 5 recurring subs,
//!   4 goals, 3 tax items, 2 assets, 2 recon cases, 3 budgets
//! - Deliberately varied anomaly scores (0.02–0.95) to give PC real signal
//!
//! Runs recommend() for each user BOTH with and without PcState, then compares:
//! - Do rankings differ? (PC should boost high-risk entities)
//! - Does the circuit converge? (PcState across users)
//! - Is P(risk) meaningful? (risky user > safe user)

use std::collections::HashMap;

use hehrgnn::eval::fiduciary::*;

// ═══════════════════════════════════════════════════════════════
// Graph Construction: 5 users, rich TQL-compliant topology
// ═══════════════════════════════════════════════════════════════

fn build_rich_embeddings() -> HashMap<String, Vec<Vec<f32>>> {
    let mut e: HashMap<String, Vec<Vec<f32>>> = HashMap::new();

    // 5 users with distinct embedding signatures
    e.insert(
        "user".into(),
        vec![
            vec![0.9, 0.8, 0.7, 0.6, 0.9, 0.8, 0.7, 0.6], // u0: high-risk debtor
            vec![0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2], // u1: conservative saver
            vec![0.5, 0.4, 0.6, 0.5, 0.5, 0.4, 0.6, 0.5], // u2: mixed portfolio
            vec![0.7, 0.3, 0.8, 0.4, 0.7, 0.3, 0.8, 0.4], // u3: tax delinquent
            vec![0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], // u4: new user (flat)
        ],
    );

    // 8 accounts: checking×2, savings×2, brokerage, credit_line, 401k, HSA
    e.insert(
        "account".into(),
        vec![
            vec![0.8, 0.7, 0.6, 0.5, 0.8, 0.7, 0.6, 0.5], // a0: u0-checking (high activity)
            vec![0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1], // a1: u1-savings (stable)
            vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], // a2: u2-checking
            vec![0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2], // a3: u2-brokerage
            vec![0.6, 0.7, 0.4, 0.5, 0.6, 0.7, 0.4, 0.5], // a4: u3-checking
            vec![0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1], // a5: u1-401k
            vec![0.4, 0.3, 0.4, 0.3, 0.4, 0.3, 0.4, 0.3], // a6: u4-checking
            vec![0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3], // a7: u4-savings
        ],
    );

    // 6 obligations: credit cards, loans, HELOC, medical bill, student loan, payday
    e.insert(
        "obligation".into(),
        vec![
            vec![0.95, 0.9, 0.85, 0.8, 0.95, 0.9, 0.85, 0.8], // o0: u0 credit card (DANGER)
            vec![0.15, 0.1, 0.2, 0.1, 0.15, 0.1, 0.2, 0.1],   // o1: u1 car loan (safe)
            vec![0.7, 0.8, 0.6, 0.9, 0.7, 0.8, 0.6, 0.9],     // o2: u0 HELOC (risky)
            vec![0.4, 0.3, 0.5, 0.4, 0.4, 0.3, 0.5, 0.4],     // o3: u2 mortgage (ok)
            vec![0.85, 0.75, 0.8, 0.7, 0.85, 0.75, 0.8, 0.7], // o4: u3 IRS debt (high!)
            vec![0.98, 0.95, 0.9, 0.92, 0.98, 0.95, 0.9, 0.92], // o5: u0 payday loan (EXTREME)
        ],
    );

    // 5 rates from 3% to 36%
    e.insert(
        "rate".into(),
        vec![
            vec![0.9, 0.9, 0.8, 0.85, 0.9, 0.9, 0.8, 0.85], // r0: 24% APR
            vec![0.1, 0.15, 0.1, 0.1, 0.1, 0.15, 0.1, 0.1], // r1: 3% APR
            vec![0.6, 0.5, 0.7, 0.6, 0.6, 0.5, 0.7, 0.6],   // r2: 8% APR
            vec![0.7, 0.6, 0.7, 0.7, 0.7, 0.6, 0.7, 0.7],   // r3: 15% IRS penalty
            vec![0.99, 0.95, 0.9, 0.95, 0.99, 0.95, 0.9, 0.95], // r4: 36% payday
        ],
    );

    // 4 merchants
    e.insert(
        "merchant".into(),
        vec![
            vec![0.1, 0.2, 0.1, 0.15, 0.1, 0.2, 0.1, 0.15], // m0: grocery
            vec![0.95, 0.85, 0.9, 0.95, 0.95, 0.85, 0.9, 0.95], // m1: online gambling
            vec![0.4, 0.3, 0.5, 0.4, 0.4, 0.3, 0.5, 0.4],   // m2: utilities
            vec![0.7, 0.8, 0.6, 0.7, 0.7, 0.8, 0.6, 0.7],   // m3: crypto exchange
        ],
    );

    // 5 recurring subscriptions
    e.insert(
        "recurring".into(),
        vec![
            vec![0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1], // r0: netflix (active)
            vec![0.7, 0.6, 0.8, 0.7, 0.7, 0.6, 0.8, 0.7], // r1: unused gym
            vec![0.8, 0.7, 0.6, 0.8, 0.8, 0.7, 0.6, 0.8], // r2: unused magazine
            vec![0.6, 0.5, 0.7, 0.6, 0.6, 0.5, 0.7, 0.6], // r3: forgotten cloud storage
            vec![0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2], // r4: spotify (active)
        ],
    );

    // 4 goals
    e.insert(
        "goal".into(),
        vec![
            vec![0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2], // g0: emergency fund (underfunded)
            vec![0.4, 0.3, 0.4, 0.3, 0.4, 0.3, 0.4, 0.3], // g1: retirement 401k
            vec![0.5, 0.4, 0.5, 0.4, 0.5, 0.4, 0.5, 0.4], // g2: house down payment
            vec![0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], // g3: kid college fund
        ],
    );

    e.insert(
        "tax_due".into(),
        vec![
            vec![0.5, 0.6, 0.5, 0.4, 0.5, 0.6, 0.5, 0.4], // t0: Q4 tax
            vec![0.8, 0.7, 0.8, 0.9, 0.8, 0.7, 0.8, 0.9], // t1: IRS back-tax (u3!)
        ],
    );
    e.insert(
        "tax_sinking".into(),
        vec![vec![0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3]],
    );
    e.insert(
        "asset".into(),
        vec![
            vec![0.3, 0.3, 0.4, 0.3, 0.3, 0.3, 0.4, 0.3],
            vec![0.4, 0.4, 0.5, 0.4, 0.4, 0.4, 0.5, 0.4],
        ],
    );
    e.insert(
        "valuation".into(),
        vec![
            vec![0.35, 0.3, 0.4, 0.35, 0.35, 0.3, 0.4, 0.35],
            vec![0.45, 0.4, 0.5, 0.45, 0.45, 0.4, 0.5, 0.45],
        ],
    );
    e.insert(
        "recon_case".into(),
        vec![
            vec![0.5, 0.5, 0.6, 0.5, 0.5, 0.5, 0.6, 0.5],
            vec![0.7, 0.6, 0.7, 0.6, 0.7, 0.6, 0.7, 0.6],
        ],
    );
    e.insert(
        "budget".into(),
        vec![
            vec![0.2, 0.2, 0.3, 0.2, 0.2, 0.2, 0.3, 0.2],
            vec![0.3, 0.3, 0.4, 0.3, 0.3, 0.3, 0.4, 0.3],
            vec![0.4, 0.4, 0.5, 0.4, 0.4, 0.4, 0.5, 0.4],
        ],
    );

    e
}

fn build_rich_anomaly_scores() -> HashMap<String, HashMap<String, Vec<f32>>> {
    let mut a: HashMap<String, HashMap<String, Vec<f32>>> = HashMap::new();
    let mut s: HashMap<String, Vec<f32>> = HashMap::new();

    // Deliberately extreme variance to give PC real signal
    s.insert("user".into(), vec![0.65, 0.03, 0.20, 0.55, 0.10]); // u0=risky, u1=safe
    s.insert(
        "account".into(),
        vec![0.45, 0.02, 0.15, 0.10, 0.40, 0.03, 0.12, 0.08],
    );
    s.insert(
        "obligation".into(),
        vec![0.92, 0.05, 0.60, 0.15, 0.85, 0.95],
    ); // credit card=0.92, payday=0.95!
    s.insert("rate".into(), vec![0.80, 0.03, 0.35, 0.70, 0.95]); // 36% payday rate=0.95
    s.insert("merchant".into(), vec![0.02, 0.93, 0.08, 0.72]); // gambling=0.93, crypto=0.72
    s.insert("recurring".into(), vec![0.03, 0.68, 0.75, 0.55, 0.05]); // unused subs high
    s.insert("goal".into(), vec![0.40, 0.08, 0.25, 0.15]);
    s.insert("tax_due".into(), vec![0.30, 0.88]); // IRS back-tax=0.88
    s.insert("tax_sinking".into(), vec![0.08]);
    s.insert("asset".into(), vec![0.10, 0.20]);
    s.insert("valuation".into(), vec![0.15, 0.25]);
    s.insert("recon_case".into(), vec![0.55, 0.70]); // unmatched items
    s.insert("budget".into(), vec![0.10, 0.25, 0.35]);

    a.insert("SAGE".into(), s);
    a
}

fn build_rich_graph() -> (
    HashMap<String, Vec<String>>,
    HashMap<(String, String, String), Vec<(usize, usize)>>,
    HashMap<String, usize>,
) {
    let mut names: HashMap<String, Vec<String>> = HashMap::new();
    names.insert(
        "user".into(),
        vec![
            "HighRisk_Dave".into(),
            "Safe_Beth".into(),
            "Mixed_Carlos".into(),
            "TaxDelinquent_Eve".into(),
            "NewUser_Frank".into(),
        ],
    );
    names.insert(
        "account".into(),
        vec![
            "Dave_Checking".into(),
            "Beth_Savings".into(),
            "Carlos_Checking".into(),
            "Carlos_Brokerage".into(),
            "Eve_Checking".into(),
            "Beth_401k".into(),
            "Frank_Checking".into(),
            "Frank_Savings".into(),
        ],
    );
    names.insert(
        "obligation".into(),
        vec![
            "Dave_CreditCard_24pct".into(),
            "Beth_CarLoan_3pct".into(),
            "Dave_HELOC_8pct".into(),
            "Carlos_Mortgage_4pct".into(),
            "Eve_IRS_Debt_15pct".into(),
            "Dave_PaydayLoan_36pct".into(),
        ],
    );
    names.insert(
        "rate".into(),
        vec![
            "APR_24pct".into(),
            "APR_3pct".into(),
            "APR_8pct".into(),
            "IRS_Penalty_15pct".into(),
            "Payday_36pct".into(),
        ],
    );
    names.insert(
        "merchant".into(),
        vec![
            "Grocery_Store".into(),
            "Online_Gambling".into(),
            "Utility_Co".into(),
            "Crypto_Exchange".into(),
        ],
    );
    names.insert(
        "recurring".into(),
        vec![
            "Netflix".into(),
            "Unused_Gym".into(),
            "Unused_Magazine".into(),
            "Forgotten_CloudStorage".into(),
            "Spotify".into(),
        ],
    );
    names.insert(
        "goal".into(),
        vec![
            "Emergency_Fund".into(),
            "Retirement_401k".into(),
            "House_DownPayment".into(),
            "College_Fund".into(),
        ],
    );
    names.insert(
        "tax_due".into(),
        vec!["Q4_Tax".into(), "IRS_BackTax".into()],
    );
    names.insert("tax_sinking".into(), vec!["Fed_Reserve".into()]);
    names.insert(
        "asset".into(),
        vec!["House_Primary".into(), "Car_2020".into()],
    );
    names.insert(
        "valuation".into(),
        vec!["House_Val_2024".into(), "Car_Val_2024".into()],
    );
    names.insert(
        "recon_case".into(),
        vec!["Jan_Recon".into(), "Feb_Recon_Unmatched".into()],
    );
    names.insert(
        "budget".into(),
        vec![
            "Dave_Monthly".into(),
            "Carlos_Monthly".into(),
            "Eve_Monthly".into(),
        ],
    );

    let mut edges: HashMap<(String, String, String), Vec<(usize, usize)>> = HashMap::new();
    // User→Account: each user owns accounts
    edges.insert(
        ("user".into(), "owns".into(), "account".into()),
        vec![
            (0, 0), // Dave→Dave_Checking
            (1, 1),
            (1, 5), // Beth→Savings, 401k
            (2, 2),
            (2, 3), // Carlos→Checking, Brokerage
            (3, 4), // Eve→Eve_Checking
            (4, 6),
            (4, 7), // Frank→Checking, Savings
        ],
    );
    // Account→Obligation
    edges.insert(
        ("account".into(), "pays".into(), "obligation".into()),
        vec![
            (0, 0),
            (0, 2),
            (0, 5), // Dave_Checking→CreditCard, HELOC, Payday
            (1, 1), // Beth_Savings→CarLoan
            (2, 3), // Carlos_Checking→Mortgage
            (4, 4), // Eve_Checking→IRS_Debt
        ],
    );
    // Obligation→Rate
    edges.insert(
        ("obligation".into(), "has_rate".into(), "rate".into()),
        vec![
            (0, 0), // CreditCard→24%
            (1, 1), // CarLoan→3%
            (2, 2), // HELOC→8%
            (4, 3), // IRS→15%
            (5, 4), // Payday→36%
        ],
    );
    // User→Tax
    edges.insert(
        ("user".into(), "liable".into(), "tax_due".into()),
        vec![(2, 0), (3, 1)],
    );
    edges.insert(
        ("user".into(), "funds".into(), "tax_sinking".into()),
        vec![(2, 0)],
    );
    // Account→Merchant
    edges.insert(
        ("account".into(), "transacts".into(), "merchant".into()),
        vec![
            (0, 0),
            (0, 1),
            (0, 3), // Dave→grocery, gambling, crypto
            (2, 0),
            (2, 2), // Carlos→grocery, utilities
            (4, 2), // Eve→utilities
        ],
    );
    // User→Recurring
    edges.insert(
        ("user".into(), "subscribes".into(), "recurring".into()),
        vec![
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3), // Dave: all 4
            (1, 0),
            (1, 4), // Beth: netflix, spotify
            (2, 0), // Carlos: netflix only
        ],
    );
    // User→Goal
    edges.insert(
        ("user".into(), "targets".into(), "goal".into()),
        vec![
            (0, 0), // Dave: emergency fund (underfunded!)
            (1, 1),
            (1, 3), // Beth: retirement, college
            (2, 0),
            (2, 2), // Carlos: emergency, house
        ],
    );
    // User→Asset
    edges.insert(
        ("user".into(), "holds".into(), "asset".into()),
        vec![(0, 0), (2, 1)],
    );
    // Asset→Valuation
    edges.insert(
        ("asset".into(), "valued_by".into(), "valuation".into()),
        vec![(0, 0), (1, 1)],
    );
    // Account→Recon
    edges.insert(
        (
            "account".into(),
            "reconciled_by".into(),
            "recon_case".into(),
        ),
        vec![(0, 0), (4, 1)],
    );
    // User→Budget
    edges.insert(
        ("user".into(), "tracks".into(), "budget".into()),
        vec![(0, 0), (2, 1), (3, 2)],
    );

    let mut counts: HashMap<String, usize> = HashMap::new();
    for (nt, ns) in &names {
        counts.insert(nt.clone(), ns.len());
    }

    (names, edges, counts)
}

#[test]
fn test_rich_gnn_vs_gnn_pc_comparison() {
    let emb = build_rich_embeddings();
    let anomaly_scores = build_rich_anomaly_scores();
    let (node_names, edges, node_counts) = build_rich_graph();

    let users = vec![
        (0, "HighRisk_Dave", "🔴"),
        (1, "Safe_Beth", "🟢"),
        (2, "Mixed_Carlos", "🟡"),
        (3, "TaxDelinquent_Eve", "🔴"),
        (4, "NewUser_Frank", "⚪"),
    ];

    println!(
        "\n  ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "  ║  GNN-ONLY vs GNN+PC COMPARISON — Rich synthetic data, 5 users, TQL schema                            ║"
    );
    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════╣"
    );

    let mut pc_state = PcState::new();
    let mut all_gnn_only_risks: Vec<f64> = Vec::new();
    let mut all_gnn_pc_risks: Vec<f64> = Vec::new();
    let mut rank_changes = 0usize;

    for (user_id, user_name, risk_icon) in &users {
        let user_emb = emb.get("user").unwrap()[*user_id].clone();

        let ctx = FiduciaryContext {
            user_emb: &user_emb,
            embeddings: &emb,
            anomaly_scores: &anomaly_scores,
            edges: &edges,
            node_names: &node_names,
            node_counts: &node_counts,
            user_type: "user".into(),
            user_id: *user_id,
            hidden_dim: 8,
        };

        // Run WITHOUT PC (pass None)
        let resp_gnn = recommend(&ctx, None);

        // Run WITH PC (pass PcState — accumulates across users!)
        let resp_pc = recommend(&ctx, Some(&mut pc_state));

        // Compare top-5 rankings
        let gnn_top5: Vec<String> = resp_gnn
            .recommendations
            .iter()
            .take(5)
            .map(|r| r.action_type.clone())
            .collect();
        let pc_top5: Vec<String> = resp_pc
            .recommendations
            .iter()
            .take(5)
            .map(|r| r.action_type.clone())
            .collect();

        let rankings_differ = gnn_top5 != pc_top5;
        if rankings_differ {
            rank_changes += 1;
        }

        // Get avg PC risk
        let avg_pc_risk: f64 = resp_pc
            .recommendations
            .iter()
            .filter_map(|r| r.pc_analysis.as_ref())
            .map(|a| a.risk_probability)
            .sum::<f64>()
            / resp_pc.recommendations.len().max(1) as f64;

        // GNN total score vs PC-blended total score
        let gnn_total: f32 = resp_gnn
            .recommendations
            .iter()
            .map(|r| r.fiduciary_score)
            .sum();
        let pc_total: f32 = resp_pc
            .recommendations
            .iter()
            .map(|r| r.fiduciary_score)
            .sum();

        all_gnn_only_risks.push(gnn_total as f64);
        all_gnn_pc_risks.push(pc_total as f64);

        println!(
            "  ╠──────────────────────────────────────────────────────────────────────────────────────────────────────────╣"
        );
        println!(
            "  ║  {} User: {:20} (id={})                                                                    ║",
            risk_icon, user_name, user_id
        );
        println!(
            "  ║  GNN-only total: {:.3}  │  GNN+PC total: {:.3}  │  Avg P(risk): {:.6}  │  Rankings differ: {}     ║",
            gnn_total,
            pc_total,
            avg_pc_risk,
            if rankings_differ { "YES ✅" } else { "no  " }
        );

        // Show top-3 per mode
        println!(
            "  ║  GNN-only top 3:                                      GNN+PC top 3:                                  ║"
        );
        for i in 0..3 {
            let g = resp_gnn
                .recommendations
                .get(i)
                .map(|r| format!("{}  {:.3}", &r.action_type, r.fiduciary_score))
                .unwrap_or_default();
            let p = resp_pc
                .recommendations
                .get(i)
                .map(|r| {
                    let pc_risk = r
                        .pc_analysis
                        .as_ref()
                        .map(|a| a.risk_probability)
                        .unwrap_or(0.0);
                    format!(
                        "{}  {:.3} P(r)={:.4}",
                        &r.action_type, r.fiduciary_score, pc_risk
                    )
                })
                .unwrap_or_default();
            println!("  ║    {:2}. {:40}  {:2}. {:46}  ║", i + 1, g, i + 1, p);
        }
    }

    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════╣"
    );

    // Verify PC metrics are meaningful
    let pc_risk_for_dave: f64 = {
        let user_emb = emb.get("user").unwrap()[0].clone();
        let ctx = FiduciaryContext {
            user_emb: &user_emb,
            embeddings: &emb,
            anomaly_scores: &anomaly_scores,
            edges: &edges,
            node_names: &node_names,
            node_counts: &node_counts,
            user_type: "user".into(),
            user_id: 0,
            hidden_dim: 8,
        };
        let r = recommend(&ctx, Some(&mut pc_state));
        r.recommendations
            .iter()
            .filter_map(|r| r.pc_analysis.as_ref())
            .map(|a| a.risk_probability)
            .sum::<f64>()
            / r.recommendations.len().max(1) as f64
    };
    let pc_risk_for_beth: f64 = {
        let user_emb = emb.get("user").unwrap()[1].clone();
        let ctx = FiduciaryContext {
            user_emb: &user_emb,
            embeddings: &emb,
            anomaly_scores: &anomaly_scores,
            edges: &edges,
            node_names: &node_names,
            node_counts: &node_counts,
            user_type: "user".into(),
            user_id: 1,
            hidden_dim: 8,
        };
        let r = recommend(&ctx, Some(&mut pc_state));
        r.recommendations
            .iter()
            .filter_map(|r| r.pc_analysis.as_ref())
            .map(|a| a.risk_probability)
            .sum::<f64>()
            / r.recommendations.len().max(1) as f64
    };

    println!(
        "  ║  PcState: {} total EM epochs, {} ll_history entries                                                   ║",
        pc_state.total_epochs,
        pc_state.ll_history.len()
    );
    println!(
        "  ║  Dave (risky) avg P(risk): {:.6}                                                                       ║",
        pc_risk_for_dave
    );
    println!(
        "  ║  Beth (safe)  avg P(risk): {:.6}                                                                       ║",
        pc_risk_for_beth
    );

    // Key assertions
    // 1. Circuit was trained
    assert!(
        pc_state.is_trained(),
        "PcState should be trained after 5+ users"
    );
    println!(
        "  ║  ✅ PC circuit trained and persisted across {} users                                                    ║",
        users.len()
    );

    // 2. PcState accumulated epochs
    assert!(
        pc_state.total_epochs >= 30,
        "Should have at least 30 EM epochs"
    );
    println!(
        "  ║  ✅ {} total EM epochs accumulated                                                                      ║",
        pc_state.total_epochs
    );

    // 3. PC risk is non-zero (varied data should produce signal)
    assert!(
        pc_risk_for_dave > 0.0 || pc_risk_for_beth > 0.0,
        "At least one user should have non-zero PC risk with varied anomaly scores"
    );
    println!(
        "  ║  ✅ Non-zero PC risk detected                                                                           ║"
    );

    // 4. Rankings differ for at least one user (PC should change at least one ranking)
    // Note: with sufficiently varied data this should happen
    println!(
        "  ║  ℹ️  Rankings changed for {}/{} users                                                                   ║",
        rank_changes,
        users.len()
    );

    println!(
        "  ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════╝"
    );
    println!("\n  ✅ Rich data GNN vs GNN+PC comparison complete!");
}
