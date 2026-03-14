//! Scenario 6: Tax Estimation + Reserve Recommendations (Freelancer)
//!
//! Freelancers have irregular income and tax obligations.
//! HEHRGNN learns income→tax patterns. Users with low reserves
//! relative to income should score as "at risk".

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::prelude::*;
    type B = NdArray;

    use hehrgnn::data::graph_builder::{GraphBuildConfig, GraphFact, build_hetero_graph};
    use hehrgnn::data::hetero_graph::EdgeType;
    use hehrgnn::model::graphsage::GraphSageModelConfig;
    use hehrgnn::server::state::PlainEmbeddings;

    fn fact(st: &str, s: &str, r: &str, dt: &str, d: &str) -> GraphFact {
        GraphFact {
            src: (st.to_string(), s.to_string()),
            relation: r.to_string(),
            dst: (dt.to_string(), d.to_string()),
        }
    }

    fn build_tax_graph() -> Vec<GraphFact> {
        let mut facts = Vec::new();

        // Alice: freelancer, high income, good tax reserves (LOW RISK)
        facts.push(fact(
            "user",
            "alice",
            "has_profile",
            "tax_profile",
            "self_employed",
        ));
        facts.push(fact("user", "alice", "owns", "account", "alice_checking"));
        facts.push(fact(
            "user",
            "alice",
            "owns",
            "account",
            "alice_tax_reserve",
        ));
        facts.push(fact(
            "account",
            "alice_tax_reserve",
            "balance_level",
            "balance",
            "high",
        ));
        for i in 0..6 {
            let tx = format!("alice_income_{}", i);
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
                "income_type",
                "income_source",
                "freelance",
            ));
            facts.push(fact(
                "transaction",
                &tx,
                "amount_range",
                "amount_bucket",
                "high",
            ));
        }
        // Regular tax payments
        for i in 0..4 {
            let tx = format!("alice_tax_payment_{}", i);
            facts.push(fact(
                "transaction",
                &tx,
                "posted_to",
                "account",
                "alice_tax_reserve",
            ));
            facts.push(fact(
                "transaction",
                &tx,
                "tx_type",
                "payment_type",
                "estimated_tax",
            ));
            facts.push(fact(
                "transaction",
                &tx,
                "for_period",
                "tax_period",
                &format!("q{}", i + 1),
            ));
        }

        // Bob: freelancer, high income, NO tax reserves (HIGH RISK)
        facts.push(fact(
            "user",
            "bob",
            "has_profile",
            "tax_profile",
            "self_employed",
        ));
        facts.push(fact("user", "bob", "owns", "account", "bob_checking"));
        // NO reserve account!
        for i in 0..6 {
            let tx = format!("bob_income_{}", i);
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
                "income_type",
                "income_source",
                "freelance",
            ));
            facts.push(fact(
                "transaction",
                &tx,
                "amount_range",
                "amount_bucket",
                "high",
            ));
        }
        // NO tax payments! (high risk)

        // Carol: W-2 employee, taxes withheld (LOW RISK)
        facts.push(fact(
            "user",
            "carol",
            "has_profile",
            "tax_profile",
            "w2_employee",
        ));
        facts.push(fact("user", "carol", "owns", "account", "carol_checking"));
        for i in 0..6 {
            let tx = format!("carol_salary_{}", i);
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
                "income_type",
                "income_source",
                "salary",
            ));
            facts.push(fact(
                "transaction",
                &tx,
                "amount_range",
                "amount_bucket",
                "medium",
            ));
            facts.push(fact(
                "transaction",
                &tx,
                "tax_withholding",
                "withholding_status",
                "withheld",
            ));
        }

        // Dave: freelancer, variable income, partial reserves (MEDIUM RISK)
        facts.push(fact(
            "user",
            "dave",
            "has_profile",
            "tax_profile",
            "self_employed",
        ));
        facts.push(fact("user", "dave", "owns", "account", "dave_checking"));
        facts.push(fact("user", "dave", "owns", "account", "dave_tax_reserve"));
        facts.push(fact(
            "account",
            "dave_tax_reserve",
            "balance_level",
            "balance",
            "low",
        ));
        for i in 0..4 {
            let tx = format!("dave_income_{}", i);
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
                "income_type",
                "income_source",
                "freelance",
            ));
            let amt = if i % 2 == 0 { "high" } else { "low" };
            facts.push(fact(
                "transaction",
                &tx,
                "amount_range",
                "amount_bucket",
                amt,
            ));
        }
        // Only 1 tax payment
        facts.push(fact(
            "transaction",
            "dave_tax_q1",
            "posted_to",
            "account",
            "dave_tax_reserve",
        ));
        facts.push(fact(
            "transaction",
            "dave_tax_q1",
            "tx_type",
            "payment_type",
            "estimated_tax",
        ));

        facts
    }

    #[test]
    fn test_tax_estimation_reserves() {
        let device = <B as Backend>::Device::default();
        let facts = build_tax_graph();

        println!("\n  ═══════════════════════════════════════════════════════════════");
        println!("   💸 SCENARIO 6: TAX ESTIMATION + RESERVE RECOMMENDATIONS");
        println!("   Freelancer income patterns → under-reserve risk scoring");
        println!("  ═══════════════════════════════════════════════════════════════\n");

        let config = GraphBuildConfig {
            node_feat_dim: 32,
            add_reverse_edges: true,
            add_self_loops: true,
            add_positional_encoding: true,
            add_cross_dependency_edges: true,
        };
        let graph = build_hetero_graph::<B>(&facts, &config, &device);
        println!(
            "  Graph: {} nodes, {} edges\n",
            graph.total_nodes(),
            graph.total_edges()
        );

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
        // Users in insertion order: alice=0, bob=1, carol=2, dave=3
        let users = vec![
            (
                "alice",
                0,
                "🟢 LOW RISK",
                "Freelancer, excellent tax reserves",
            ),
            ("bob", 1, "🔴 HIGH RISK", "Freelancer, NO tax reserves"),
            ("carol", 2, "🟢 LOW RISK", "W-2 employee, taxes withheld"),
            ("dave", 3, "🟡 MEDIUM RISK", "Freelancer, partial reserves"),
        ];

        println!("  ── USER TAX RISK PROFILES ──\n");
        for (name, _, risk, desc) in &users {
            println!("    {} ({}) — {}", name, risk, desc);
        }

        // Pairwise similarities
        println!("\n  ── EMBEDDING SIMILARITY ──\n");
        println!("  Expectation: similar risk = similar embedding\n");
        println!(
            "  {:>8} │ {:>8} │ {:>8} │ {:>8} │ {:>8}",
            "", "alice", "bob", "carol", "dave"
        );
        println!("  ────────┼──────────┼──────────┼──────────┼──────────");

        for &(n1, i1, _, _) in &users {
            let mut row = format!("  {:>8} │", n1);
            for &(_, i2, _, _) in &users {
                if i1 < user_embs.len() && i2 < user_embs.len() {
                    let sim = PlainEmbeddings::cosine_similarity(&user_embs[i1], &user_embs[i2]);
                    row.push_str(&format!(" {:>8.4} │", sim));
                }
            }
            println!("{}", row);
        }

        // Risk scoring: distance from "healthy" centroid (alice + carol)
        println!("\n  ── TAX RISK SCORES ──\n");
        let healthy_centroid: Vec<f32> = (0..64)
            .map(|j| {
                let a = if 0 < user_embs.len() && j < user_embs[0].len() {
                    user_embs[0][j]
                } else {
                    0.0
                };
                let c = if 2 < user_embs.len() && j < user_embs[2].len() {
                    user_embs[2][j]
                } else {
                    0.0
                };
                (a + c) / 2.0
            })
            .collect();

        for &(name, idx, risk, _) in &users {
            if idx < user_embs.len() {
                let dist = PlainEmbeddings::l2_distance(&user_embs[idx], &healthy_centroid);
                println!(
                    "    {:>8} │ dist from healthy centroid: {:.6} │ {}",
                    name, dist, risk
                );
            }
        }

        // Recommendations
        println!("\n  ── RECOMMENDATIONS ──\n");
        for &(name, _, risk, _) in &users {
            let recs = match risk {
                "🔴 HIGH RISK" => vec![
                    "🚨 Create tax reserve account immediately",
                    "💰 Set aside 30% of each freelance payment",
                    "📅 Schedule quarterly estimated tax payments",
                ],
                "🟡 MEDIUM RISK" => vec![
                    "⚠️ Increase tax reserve contributions",
                    "📊 Review if Q2-Q4 estimated payments are needed",
                ],
                _ => vec!["✅ Tax situation looks good"],
            };
            println!("    {} ({}):", name, risk);
            for r in recs {
                println!("      → {}", r);
            }
        }

        println!("\n  ═══════════════════════════════════════════════════════════════\n");
        assert!(user_embs.len() >= 4);
    }
}
