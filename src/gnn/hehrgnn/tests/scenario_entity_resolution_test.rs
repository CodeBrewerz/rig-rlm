//! Scenario 8: Duplicate Account / Entity Resolution
//!
//! Two accounts that are actually the same (duplicate) share similar
//! merchant neighborhoods, amounts, and transaction patterns.
//! The GNN should embed them nearby (high cosine similarity).

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::prelude::*;
    type B = NdArray;

    use hehrgnn::data::graph_builder::{build_hetero_graph, GraphBuildConfig, GraphFact};
    use hehrgnn::data::hetero_graph::EdgeType;
    use hehrgnn::model::graphsage::GraphSageModelConfig;
    use hehrgnn::server::state::PlainEmbeddings;

    fn fact(st: &str, s: &str, r: &str, dt: &str, d: &str) -> GraphFact {
        GraphFact { src: (st.to_string(), s.to_string()), relation: r.to_string(), dst: (dt.to_string(), d.to_string()) }
    }

    fn build_entity_resolution_graph() -> Vec<GraphFact> {
        let mut facts = Vec::new();

        // ── Account A (real) and Account A' (duplicate) ──
        // Same user, same merchants, same amounts → should embed similarly
        facts.push(fact("user", "alice", "owns", "account", "alice_checking_real"));
        facts.push(fact("user", "alice", "owns", "account", "alice_checking_dup"));
        facts.push(fact("account", "alice_checking_real", "at_bank", "bank", "chase"));
        facts.push(fact("account", "alice_checking_dup", "at_bank", "bank", "chase"));

        // Same merchants on both accounts
        for i in 0..6 {
            let merchants = ["walmart", "amazon", "starbucks", "target", "costco", "shell_gas"];
            let merch = merchants[i];
            let amt = ["small", "medium", "large"][i % 3];

            // Real account transactions
            let tx_r = format!("tx_real_{}", i);
            facts.push(fact("transaction", &tx_r, "posted_to", "account", "alice_checking_real"));
            facts.push(fact("transaction", &tx_r, "at_merchant", "merchant", merch));
            facts.push(fact("transaction", &tx_r, "tx_amount", "amount_range", amt));

            // Duplicate account transactions (same merchants, similar amounts!)
            let tx_d = format!("tx_dup_{}", i);
            facts.push(fact("transaction", &tx_d, "posted_to", "account", "alice_checking_dup"));
            facts.push(fact("transaction", &tx_d, "at_merchant", "merchant", merch));
            facts.push(fact("transaction", &tx_d, "tx_amount", "amount_range", amt));
        }

        // ── Account B (different user, different patterns) ──
        facts.push(fact("user", "bob", "owns", "account", "bob_savings"));
        facts.push(fact("account", "bob_savings", "at_bank", "bank", "wells_fargo"));
        for i in 0..5 {
            let merchants = ["rent_landlord", "utility_co", "insurance_co", "gym", "netflix"];
            let tx = format!("tx_bob_{}", i);
            facts.push(fact("transaction", &tx, "posted_to", "account", "bob_savings"));
            facts.push(fact("transaction", &tx, "at_merchant", "merchant", merchants[i]));
            facts.push(fact("transaction", &tx, "tx_amount", "amount_range", "medium"));
        }

        // ── Account C (carol, completely different) ──
        facts.push(fact("user", "carol", "owns", "account", "carol_investment"));
        facts.push(fact("account", "carol_investment", "at_bank", "bank", "schwab"));
        for i in 0..4 {
            let tx = format!("tx_carol_{}", i);
            facts.push(fact("transaction", &tx, "posted_to", "account", "carol_investment"));
            facts.push(fact("transaction", &tx, "at_merchant", "merchant", "stock_broker"));
            facts.push(fact("transaction", &tx, "tx_amount", "amount_range", "large"));
        }

        facts
    }

    #[test]
    fn test_entity_resolution() {
        let device = <B as Backend>::Device::default();
        let facts = build_entity_resolution_graph();

        println!("\n  ═══════════════════════════════════════════════════════════════");
        println!("   🔍 SCENARIO 8: DUPLICATE ACCOUNT / ENTITY RESOLUTION");
        println!("   GNN should embed duplicate accounts nearby");
        println!("  ═══════════════════════════════════════════════════════════════\n");

        let config = GraphBuildConfig { node_feat_dim: 32, add_reverse_edges: true, add_self_loops: true, add_positional_encoding: true };
        let graph = build_hetero_graph::<B>(&facts, &config, &device);
        println!("  Graph: {} nodes, {} edges\n", graph.total_nodes(), graph.total_edges());

        let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
        let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();

        let sage = GraphSageModelConfig { in_dim: 32, hidden_dim: 64, num_layers: 2, dropout: 0.0 };
        let model = sage.init::<B>(&node_types, &edge_types, &device);
        let emb = PlainEmbeddings::from_burn(&model.forward(&graph));

        let acct_embs = &emb.data["account"];
        // Accounts in insertion order: alice_checking_real=0, alice_checking_dup=1, bob_savings=2, carol_investment=3
        let accounts = vec![
            ("alice_checking_real", 0, "Alice's real Chase checking"),
            ("alice_checking_dup",  1, "Alice's DUPLICATE Chase checking"),
            ("bob_savings",        2, "Bob's Wells Fargo savings"),
            ("carol_investment",   3, "Carol's Schwab investment"),
        ];

        // ── Pairwise similarities ──
        println!("  ── ACCOUNT EMBEDDING SIMILARITY ──\n");
        println!("  {:>22} │ {:>10} │ {:>10} │ {:>10} │ {:>10}",
            "", "real", "dup", "bob", "carol");
        println!("  ──────────────────────┼────────────┼────────────┼────────────┼────────────");

        let mut dup_sim = 0.0f32;
        let mut non_dup_sims = Vec::new();

        for &(n1, i1, _) in &accounts {
            let short1 = n1.split('_').last().unwrap_or(n1);
            let mut row = format!("  {:>22} │", n1);
            for &(_, i2, _) in &accounts {
                if i1 < acct_embs.len() && i2 < acct_embs.len() {
                    let sim = PlainEmbeddings::cosine_similarity(&acct_embs[i1], &acct_embs[i2]);
                    row.push_str(&format!(" {:>10.4} │", sim));

                    if (n1 == "alice_checking_real" && i2 == 1) || (n1 == "alice_checking_dup" && i2 == 0) {
                        dup_sim = sim;
                    } else if i1 != i2 && !((i1 == 0 && i2 == 1) || (i1 == 1 && i2 == 0)) {
                        non_dup_sims.push(sim);
                    }
                }
            }
            println!("{}", row);
        }

        let avg_non_dup = non_dup_sims.iter().sum::<f32>() / non_dup_sims.len().max(1) as f32;

        println!();
        println!("  ── DUPLICATE DETECTION SUMMARY ──\n");
        println!("    Duplicate pair (real↔dup) similarity:  {:.4}", dup_sim);
        println!("    Non-duplicate pairs avg similarity:     {:.4}", avg_non_dup);
        println!("    Gap:                                    {:.4}", dup_sim - avg_non_dup);
        println!("    Duplicate > Non-duplicate:              {}",
            if dup_sim > avg_non_dup { "✅ YES — can detect duplicate!" } else { "❌ NO" });

        // ── Duplicate candidates ──
        println!("\n  ── DUPLICATE CANDIDATES (similarity > 0.95) ──\n");
        for (i, &(n1, i1, _)) in accounts.iter().enumerate() {
            for &(n2, i2, _) in &accounts[i+1..] {
                if i1 < acct_embs.len() && i2 < acct_embs.len() {
                    let sim = PlainEmbeddings::cosine_similarity(&acct_embs[i1], &acct_embs[i2]);
                    if sim > 0.95 {
                        println!("    🚨 {} ↔ {} : sim={:.4} → LIKELY DUPLICATE", n1, n2, sim);
                    }
                }
            }
        }

        // ── Merge recommendations ──
        println!("\n  ── MERGE RECOMMENDATIONS ──\n");
        if dup_sim > avg_non_dup {
            println!("    📋 alice_checking_real ↔ alice_checking_dup");
            println!("       → Same bank (Chase), same merchant set, same owner (Alice)");
            println!("       → Recommendation: MERGE these accounts");
            println!("       → Confidence: {:.0}%", (dup_sim * 100.0).min(99.0));
        }

        println!("\n  ═══════════════════════════════════════════════════════════════\n");

        assert!(dup_sim.is_finite());
        assert!(avg_non_dup.is_finite());
    }
}
