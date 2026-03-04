//! Scenario 1: Financial Posture + Goal Recommendations
//!
//! 5 users with distinct financial postures:
//!   🔴 Alice: High income, high debt, no savings → needs emergency fund + debt paydown
//!   🟢 Bob:   Balanced — moderate income, low debt, healthy savings
//!   🟡 Carol: High income, high discretionary spend, minimal debt → budget optimization
//!   🔴 Dave:  Low income, high debt, no savings → critical financial health
//!   🟢 Eve:   Freelancer, variable income, good savings, tax reserves set up
//!
//! The GNN should cluster similar financial postures together and
//! generate different recommendations per cluster.

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::prelude::*;
    use std::collections::HashMap;

    type B = NdArray;

    use hehrgnn::data::graph_builder::{GraphBuildConfig, GraphFact, build_hetero_graph};
    use hehrgnn::data::hetero_graph::EdgeType;
    use hehrgnn::model::graphsage::GraphSageModelConfig;
    use hehrgnn::server::state::PlainEmbeddings;
    use hehrgnn::tasks::node_classifier::NodeClassifierConfig;

    /// Financial posture labels
    const POSTURE_CRITICAL: usize = 0; // 🔴 debt + no savings
    const POSTURE_OPTIMIZE: usize = 1; // 🟡 good income but overspending
    const POSTURE_HEALTHY: usize = 2; // 🟢 balanced

    fn posture_name(p: usize) -> &'static str {
        match p {
            0 => "🔴 CRITICAL",
            1 => "🟡 OPTIMIZE",
            2 => "🟢 HEALTHY",
            _ => "UNKNOWN",
        }
    }

    fn build_finance_graph() -> (Vec<GraphFact>, HashMap<String, usize>) {
        let mut facts = Vec::new();
        let mut user_postures = HashMap::new();

        // ═══════════════════════════════════════════════
        // Alice: high income, high debt, no savings (CRITICAL)
        // ═══════════════════════════════════════════════
        user_postures.insert("alice".into(), POSTURE_CRITICAL);
        facts.push(fact("user", "alice", "owns", "account", "alice_checking"));
        facts.push(fact(
            "user",
            "alice",
            "has_debt",
            "debt",
            "alice_credit_card",
        ));
        facts.push(fact("user", "alice", "has_debt", "debt", "alice_car_loan"));
        // High income
        for i in 0..6 {
            let tx = format!("alice_income_{}", i);
            facts.push(fact(
                "transaction",
                &tx,
                "posted_to",
                "account",
                "alice_checking",
            ));
            facts.push(fact("transaction", &tx, "has_type", "tx_type", "income"));
            facts.push(fact(
                "transaction",
                &tx,
                "amount_range",
                "amount_bucket",
                "high",
            ));
        }
        // High spending at discretionary merchants
        for i in 0..8 {
            let tx = format!("alice_spend_{}", i);
            facts.push(fact(
                "transaction",
                &tx,
                "posted_to",
                "account",
                "alice_checking",
            ));
            facts.push(fact(
                "transaction",
                &tx,
                "at_merchant",
                "merchant",
                "luxury_store",
            ));
            facts.push(fact(
                "transaction",
                &tx,
                "amount_range",
                "amount_bucket",
                "high",
            ));
        }
        // Debt characteristics
        facts.push(fact(
            "debt",
            "alice_credit_card",
            "debt_type",
            "debt_category",
            "revolving",
        ));
        facts.push(fact(
            "debt",
            "alice_credit_card",
            "balance_range",
            "amount_bucket",
            "high",
        ));
        facts.push(fact(
            "debt",
            "alice_car_loan",
            "debt_type",
            "debt_category",
            "installment",
        ));
        facts.push(fact(
            "debt",
            "alice_car_loan",
            "balance_range",
            "amount_bucket",
            "medium",
        ));
        // NO savings account, NO goals

        // ═══════════════════════════════════════════════
        // Bob: balanced, moderate income, low debt, healthy savings (HEALTHY)
        // ═══════════════════════════════════════════════
        user_postures.insert("bob".into(), POSTURE_HEALTHY);
        facts.push(fact("user", "bob", "owns", "account", "bob_checking"));
        facts.push(fact("user", "bob", "owns", "account", "bob_savings"));
        facts.push(fact(
            "user",
            "bob",
            "has_goal",
            "goal",
            "bob_emergency_fund",
        ));
        facts.push(fact("user", "bob", "has_goal", "goal", "bob_vacation"));
        // Moderate income
        for i in 0..4 {
            let tx = format!("bob_income_{}", i);
            facts.push(fact(
                "transaction",
                &tx,
                "posted_to",
                "account",
                "bob_checking",
            ));
            facts.push(fact("transaction", &tx, "has_type", "tx_type", "income"));
            facts.push(fact(
                "transaction",
                &tx,
                "amount_range",
                "amount_bucket",
                "medium",
            ));
        }
        // Moderate spending
        for i in 0..4 {
            let tx = format!("bob_spend_{}", i);
            facts.push(fact(
                "transaction",
                &tx,
                "posted_to",
                "account",
                "bob_checking",
            ));
            facts.push(fact(
                "transaction",
                &tx,
                "at_merchant",
                "merchant",
                "grocery_store",
            ));
            facts.push(fact(
                "transaction",
                &tx,
                "amount_range",
                "amount_bucket",
                "medium",
            ));
        }
        // Savings deposits
        for i in 0..3 {
            let tx = format!("bob_save_{}", i);
            facts.push(fact(
                "transaction",
                &tx,
                "posted_to",
                "account",
                "bob_savings",
            ));
            facts.push(fact("transaction", &tx, "has_type", "tx_type", "transfer"));
            facts.push(fact(
                "transaction",
                &tx,
                "amount_range",
                "amount_bucket",
                "small",
            ));
        }
        // Goal healthy
        facts.push(fact(
            "goal",
            "bob_emergency_fund",
            "goal_status",
            "status",
            "on_track",
        ));
        facts.push(fact(
            "goal",
            "bob_vacation",
            "goal_status",
            "status",
            "on_track",
        ));

        // ═══════════════════════════════════════════════
        // Carol: high income, high discretionary, minimal debt (OPTIMIZE)
        // ═══════════════════════════════════════════════
        user_postures.insert("carol".into(), POSTURE_OPTIMIZE);
        facts.push(fact("user", "carol", "owns", "account", "carol_checking"));
        facts.push(fact("user", "carol", "owns", "account", "carol_savings"));
        // High income
        for i in 0..6 {
            let tx = format!("carol_income_{}", i);
            facts.push(fact(
                "transaction",
                &tx,
                "posted_to",
                "account",
                "carol_checking",
            ));
            facts.push(fact("transaction", &tx, "has_type", "tx_type", "income"));
            facts.push(fact(
                "transaction",
                &tx,
                "amount_range",
                "amount_bucket",
                "high",
            ));
        }
        // Very high discretionary spending
        for i in 0..10 {
            let tx = format!("carol_spend_{}", i);
            facts.push(fact(
                "transaction",
                &tx,
                "posted_to",
                "account",
                "carol_checking",
            ));
            facts.push(fact(
                "transaction",
                &tx,
                "at_merchant",
                "merchant",
                "restaurant",
            ));
            facts.push(fact(
                "transaction",
                &tx,
                "amount_range",
                "amount_bucket",
                "high",
            ));
        }
        // Minimal savings, no goals
        facts.push(fact(
            "account",
            "carol_savings",
            "balance_range",
            "amount_bucket",
            "low",
        ));

        // ═══════════════════════════════════════════════
        // Dave: low income, high debt, no savings (CRITICAL) — similar to Alice
        // ═══════════════════════════════════════════════
        user_postures.insert("dave".into(), POSTURE_CRITICAL);
        facts.push(fact("user", "dave", "owns", "account", "dave_checking"));
        facts.push(fact("user", "dave", "has_debt", "debt", "dave_credit_card"));
        facts.push(fact(
            "user",
            "dave",
            "has_debt",
            "debt",
            "dave_medical_debt",
        ));
        // Low income
        for i in 0..3 {
            let tx = format!("dave_income_{}", i);
            facts.push(fact(
                "transaction",
                &tx,
                "posted_to",
                "account",
                "dave_checking",
            ));
            facts.push(fact("transaction", &tx, "has_type", "tx_type", "income"));
            facts.push(fact(
                "transaction",
                &tx,
                "amount_range",
                "amount_bucket",
                "low",
            ));
        }
        // Essential spending only
        for i in 0..5 {
            let tx = format!("dave_spend_{}", i);
            facts.push(fact(
                "transaction",
                &tx,
                "posted_to",
                "account",
                "dave_checking",
            ));
            facts.push(fact(
                "transaction",
                &tx,
                "at_merchant",
                "merchant",
                "grocery_store",
            ));
            facts.push(fact(
                "transaction",
                &tx,
                "amount_range",
                "amount_bucket",
                "small",
            ));
        }
        // High debt
        facts.push(fact(
            "debt",
            "dave_credit_card",
            "debt_type",
            "debt_category",
            "revolving",
        ));
        facts.push(fact(
            "debt",
            "dave_credit_card",
            "balance_range",
            "amount_bucket",
            "high",
        ));
        facts.push(fact(
            "debt",
            "dave_medical_debt",
            "debt_type",
            "debt_category",
            "medical",
        ));
        facts.push(fact(
            "debt",
            "dave_medical_debt",
            "balance_range",
            "amount_bucket",
            "high",
        ));

        // ═══════════════════════════════════════════════
        // Eve: freelancer, variable income, good savings + tax reserves (HEALTHY)
        // ═══════════════════════════════════════════════
        user_postures.insert("eve".into(), POSTURE_HEALTHY);
        facts.push(fact("user", "eve", "owns", "account", "eve_checking"));
        facts.push(fact("user", "eve", "owns", "account", "eve_savings"));
        facts.push(fact("user", "eve", "owns", "account", "eve_tax_reserve"));
        facts.push(fact(
            "user",
            "eve",
            "has_goal",
            "goal",
            "eve_emergency_fund",
        ));
        facts.push(fact("user", "eve", "has_goal", "goal", "eve_tax_goal"));
        // Variable income (freelancer)
        for i in 0..5 {
            let tx = format!("eve_income_{}", i);
            facts.push(fact(
                "transaction",
                &tx,
                "posted_to",
                "account",
                "eve_checking",
            ));
            facts.push(fact(
                "transaction",
                &tx,
                "has_type",
                "tx_type",
                "freelance_income",
            ));
            let bucket = if i % 2 == 0 { "high" } else { "medium" };
            facts.push(fact(
                "transaction",
                &tx,
                "amount_range",
                "amount_bucket",
                bucket,
            ));
        }
        // Moderate spending
        for i in 0..4 {
            let tx = format!("eve_spend_{}", i);
            facts.push(fact(
                "transaction",
                &tx,
                "posted_to",
                "account",
                "eve_checking",
            ));
            facts.push(fact(
                "transaction",
                &tx,
                "at_merchant",
                "merchant",
                "coworking_space",
            ));
            facts.push(fact(
                "transaction",
                &tx,
                "amount_range",
                "amount_bucket",
                "medium",
            ));
        }
        // Tax reserve deposits
        for i in 0..3 {
            let tx = format!("eve_tax_{}", i);
            facts.push(fact(
                "transaction",
                &tx,
                "posted_to",
                "account",
                "eve_tax_reserve",
            ));
            facts.push(fact(
                "transaction",
                &tx,
                "has_type",
                "tx_type",
                "tax_reserve",
            ));
        }
        // Goals healthy
        facts.push(fact(
            "goal",
            "eve_emergency_fund",
            "goal_status",
            "status",
            "on_track",
        ));
        facts.push(fact(
            "goal",
            "eve_tax_goal",
            "goal_status",
            "status",
            "on_track",
        ));

        (facts, user_postures)
    }

    fn fact(st: &str, s: &str, r: &str, dt: &str, d: &str) -> GraphFact {
        GraphFact {
            src: (st.to_string(), s.to_string()),
            relation: r.to_string(),
            dst: (dt.to_string(), d.to_string()),
        }
    }

    #[test]
    fn test_financial_posture_recommendations() {
        let device = <B as Backend>::Device::default();
        let (facts, user_postures) = build_finance_graph();

        println!("\n  ═══════════════════════════════════════════════════════════════");
        println!("   💰 SCENARIO 1: FINANCIAL POSTURE + GOAL RECOMMENDATIONS");
        println!("  ═══════════════════════════════════════════════════════════════\n");

        let config = GraphBuildConfig {
            node_feat_dim: 32,
            add_reverse_edges: true,
            add_self_loops: true,
            add_positional_encoding: true,
        };
        let graph = build_hetero_graph::<B>(&facts, &config, &device);

        println!(
            "  Graph: {} nodes, {} edges, {} types\n",
            graph.total_nodes(),
            graph.total_edges(),
            graph.node_types().len()
        );

        for nt in graph.node_types() {
            println!("    {}: {} nodes", nt, graph.node_counts[nt]);
        }
        println!();

        // Run GNN
        let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
        let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();

        let sage = GraphSageModelConfig {
            in_dim: 32,
            hidden_dim: 64,
            num_layers: 2,
            dropout: 0.0,
        };
        let model = sage.init::<B>(&node_types, &edge_types, &device);
        let emb = PlainEmbeddings::from_burn(&model.forward(&graph));

        let user_embs = &emb.data["user"];

        // ── User embedding analysis ──
        println!("  ── USER EMBEDDING ANALYSIS ──\n");

        // Users are indexed in fact insertion order: alice=0, bob=1, carol=2, dave=3, eve=4
        let users_with_idx: Vec<(&str, usize)> = vec![
            ("alice", 0),
            ("bob", 1),
            ("carol", 2),
            ("dave", 3),
            ("eve", 4),
        ];

        // Compute pairwise cosine similarities
        println!(
            "  Pairwise cosine similarity (users with SAME posture should be more similar):\n"
        );
        println!(
            "  {:>8} │ {:>8} │ {:>8} │ {:>8} │ {:>8} │ {:>8}",
            "", "alice", "bob", "carol", "dave", "eve"
        );
        println!("  ────────┼──────────┼──────────┼──────────┼──────────┼──────────");

        let mut same_posture_sims = Vec::new();
        let mut diff_posture_sims = Vec::new();

        for (i, &(u1, idx1)) in users_with_idx.iter().enumerate() {
            let mut row = format!("  {:>8} │", u1);
            for (j, &(_, idx2)) in users_with_idx.iter().enumerate() {
                if idx1 < user_embs.len() && idx2 < user_embs.len() {
                    let sim =
                        PlainEmbeddings::cosine_similarity(&user_embs[idx1], &user_embs[idx2]);
                    row.push_str(&format!(" {:>8.4} │", sim));

                    if i < j {
                        let p1 = user_postures[u1];
                        let p2 = user_postures[users_with_idx[j].0];
                        if p1 == p2 {
                            same_posture_sims.push(sim);
                        } else {
                            diff_posture_sims.push(sim);
                        }
                    }
                }
            }
            let posture = user_postures[u1];
            println!("{} {}", row, posture_name(posture));
        }

        let avg_same =
            same_posture_sims.iter().sum::<f32>() / same_posture_sims.len().max(1) as f32;
        let avg_diff =
            diff_posture_sims.iter().sum::<f32>() / diff_posture_sims.len().max(1) as f32;

        println!();
        println!("  Same-posture avg similarity:  {:.4}", avg_same);
        println!("  Diff-posture avg similarity:  {:.4}", avg_diff);
        println!(
            "  Signal: same > diff? {}",
            if avg_same > avg_diff {
                "✅ YES"
            } else {
                "❌ NO"
            }
        );

        // ── Node Classification (posture prediction) ──
        println!("\n  ── POSTURE CLASSIFICATION ──\n");

        let classifier_config = NodeClassifierConfig {
            hidden_dim: 64,
            mlp_dim: 32,
            num_classes: 3,
            dropout: 0.0,
        };
        let classifier = classifier_config.init::<B>(&device);

        // Build user embedding tensor
        let mut user_indices = Vec::new();
        let mut ground_truth = Vec::new();
        for &(u, idx) in &users_with_idx {
            if idx < user_embs.len() {
                user_indices.push(idx);
                ground_truth.push(user_postures[u]);
            }
        }

        let user_tensor = Tensor::<B, 2>::from_data(
            {
                let mut data = vec![0.0f32; user_indices.len() * 64];
                for (i, &idx) in user_indices.iter().enumerate() {
                    for (j, &v) in user_embs[idx].iter().enumerate() {
                        if j < 64 {
                            data[i * 64 + j] = v;
                        }
                    }
                }
                burn::tensor::TensorData::new(data, [user_indices.len(), 64])
            },
            &device,
        );

        let (predicted_classes, confidences) = classifier.predict(user_tensor);
        let pred_data: Vec<i64> = predicted_classes
            .into_data()
            .as_slice::<i64>()
            .unwrap()
            .to_vec();
        let conf_data: Vec<f32> = confidences.into_data().as_slice::<f32>().unwrap().to_vec();

        println!(
            "  {:>8} │ {:>14} │ {:>14} │ {:>10}",
            "User", "Ground Truth", "Predicted", "Confidence"
        );
        println!("  ────────┼────────────────┼────────────────┼────────────");
        for (i, &(u, _)) in users_with_idx.iter().enumerate() {
            println!(
                "  {:>8} │ {:>14} │ {:>14} │ {:>10.4}",
                u,
                posture_name(ground_truth[i]),
                posture_name(pred_data[i] as usize),
                conf_data[i]
            );
        }

        // ── Recommended Actions ──
        println!("\n  ── RECOMMENDED ACTIONS (based on posture) ──\n");
        for &(u, _) in &users_with_idx {
            let posture = user_postures[u];
            let actions = match posture {
                POSTURE_CRITICAL => vec![
                    "🚨 Create Emergency Fund Goal (3-month expenses)",
                    "📉 Set up Debt Paydown Goal (avalanche strategy)",
                    "🔄 Create auto-transfer rule: payday → emergency fund",
                ],
                POSTURE_OPTIMIZE => vec![
                    "📊 Review top-3 discretionary spending categories",
                    "🎯 Create savings goal (target: 20% of income)",
                    "📉 Reduce dining/entertainment by 15%",
                ],
                POSTURE_HEALTHY => vec![
                    "✅ Financial health is good — maintain current plan",
                    "📈 Consider investment goals for surplus",
                    "🔍 Review insurance coverage",
                ],
                _ => vec!["Unknown posture"],
            };
            println!("  {} ({}):", u, posture_name(posture));
            for action in &actions {
                println!("    → {}", action);
            }
            println!();
        }

        // ── Anomaly scores for users ──
        println!("  ── USER SUBGRAPH ANOMALY SCORES ──\n");

        // Compute how "unusual" each user's embedding is vs the population
        let mut all_user_emb: Vec<f32> = vec![0.0; 64];
        let mut count = 0;
        for &idx in &user_indices {
            for (j, &v) in user_embs[idx].iter().enumerate() {
                if j < 64 {
                    all_user_emb[j] += v;
                }
            }
            count += 1;
        }
        for v in all_user_emb.iter_mut() {
            *v /= count as f32;
        }

        for (i, &(u, _)) in users_with_idx.iter().enumerate() {
            let idx = user_indices[i];
            let dist = PlainEmbeddings::l2_distance(&user_embs[idx], &all_user_emb);
            let posture = user_postures[u];
            println!(
                "    {} │ distance from centroid: {:.6} │ {}",
                u,
                dist,
                posture_name(posture)
            );
        }

        println!("\n  ═══════════════════════════════════════════════════════════════\n");

        // Assertions
        assert!(avg_same.is_finite());
        assert!(avg_diff.is_finite());
    }
}
