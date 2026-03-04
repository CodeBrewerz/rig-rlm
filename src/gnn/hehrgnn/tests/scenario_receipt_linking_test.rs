//! Scenario 4: Receipt Auto-Linking to Transactions
//!
//! Receipts arrive with merchant/amount data. The GNN embeds both
//! receipts and transactions, and the LinkPredictor ranks which
//! transaction a receipt belongs to.

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::prelude::*;

    type B = NdArray;

    use hehrgnn::data::graph_builder::{GraphBuildConfig, GraphFact, build_hetero_graph};
    use hehrgnn::data::hetero_graph::EdgeType;
    use hehrgnn::model::graphsage::GraphSageModelConfig;
    use hehrgnn::server::state::PlainEmbeddings;
    use hehrgnn::tasks::link_predictor::{LinkPredictor, LinkPredictorConfig};

    fn fact(st: &str, s: &str, r: &str, dt: &str, d: &str) -> GraphFact {
        GraphFact {
            src: (st.to_string(), s.to_string()),
            relation: r.to_string(),
            dst: (dt.to_string(), d.to_string()),
        }
    }

    fn build_receipt_graph() -> (Vec<GraphFact>, Vec<(String, String)>) {
        let mut facts = Vec::new();
        let mut gt_links: Vec<(String, String)> = Vec::new();

        let merchants = ["walmart", "costco", "target", "home_depot", "amazon"];

        // 20 transactions with known receipts
        for i in 0..20 {
            let tx = format!("tx_{}", i);
            let receipt = format!("receipt_{}", i);
            let merch = merchants[i % merchants.len()];
            let amt = if i % 3 == 0 {
                "large"
            } else if i % 3 == 1 {
                "medium"
            } else {
                "small"
            };
            let acct = if i < 10 { "checking" } else { "credit_card" };

            // Transaction
            facts.push(fact("transaction", &tx, "at_merchant", "merchant", merch));
            facts.push(fact("transaction", &tx, "tx_amount", "amount_range", amt));
            facts.push(fact("transaction", &tx, "posted_to", "account", acct));

            // Receipt (same merchant + amount = should match)
            facts.push(fact(
                "receipt",
                &receipt,
                "receipt_merchant",
                "merchant",
                merch,
            ));
            facts.push(fact(
                "receipt",
                &receipt,
                "receipt_amount",
                "amount_range",
                amt,
            ));

            // Known link
            facts.push(fact(
                "receipt",
                &receipt,
                "evidence_for",
                "transaction",
                &tx,
            ));
            gt_links.push((receipt, tx));
        }

        // 5 orphan receipts (no matching transaction)
        for i in 20..25 {
            let receipt = format!("receipt_{}", i);
            facts.push(fact(
                "receipt",
                &receipt,
                "receipt_merchant",
                "merchant",
                "unknown_vendor",
            ));
            facts.push(fact(
                "receipt",
                &receipt,
                "receipt_amount",
                "amount_range",
                "small",
            ));
        }

        // Account ownership
        facts.push(fact("user", "alice", "owns", "account", "checking"));
        facts.push(fact("user", "alice", "owns", "account", "credit_card"));

        // Merchant categories
        for m in &merchants {
            facts.push(fact("merchant", m, "in_category", "category", "retail"));
        }

        (facts, gt_links)
    }

    #[test]
    fn test_receipt_auto_linking() {
        let device = <B as Backend>::Device::default();
        let (facts, gt_links) = build_receipt_graph();

        println!("\n  ═══════════════════════════════════════════════════════════════");
        println!("   🧾 SCENARIO 4: RECEIPT AUTO-LINKING TO TRANSACTIONS");
        println!("  ═══════════════════════════════════════════════════════════════\n");

        let config = GraphBuildConfig {
            node_feat_dim: 32,
            add_reverse_edges: true,
            add_self_loops: true,
            add_positional_encoding: true,
        };
        let graph = build_hetero_graph::<B>(&facts, &config, &device);
        println!(
            "  Graph: {} nodes, {} edges",
            graph.total_nodes(),
            graph.total_edges()
        );

        let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
        let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();

        let in_dim = graph
            .node_features
            .values()
            .next()
            .map(|t| t.dims()[1])
            .unwrap_or(32);
        let sage = GraphSageModelConfig {
            in_dim,
            hidden_dim: 64,
            num_layers: 2,
            dropout: 0.0,
        };
        let model = sage.init::<B>(&node_types, &edge_types, &device);
        let emb = PlainEmbeddings::from_burn(&model.forward(&graph));

        let receipt_embs = &emb.data["receipt"];
        let tx_embs = &emb.data["transaction"];

        let lp = LinkPredictor::new(LinkPredictorConfig {
            top_k: tx_embs.len(),
            ..Default::default()
        });
        let tx_targets: Vec<(String, usize, String, Vec<f32>)> = tx_embs
            .iter()
            .enumerate()
            .map(|(idx, v)| ("transaction".into(), idx, format!("tx_{}", idx), v.clone()))
            .collect();

        // Score each receipt against all transactions
        println!("\n  ── RECEIPT ↔ TRANSACTION LINKING ──\n");
        let mut ranks = Vec::new();

        for (receipt, tx) in &gt_links {
            let r_num: usize = receipt.strip_prefix("receipt_").unwrap().parse().unwrap();
            let t_num: usize = tx.strip_prefix("tx_").unwrap().parse().unwrap();
            if r_num >= receipt_embs.len() || t_num >= tx_embs.len() {
                continue;
            }

            let result = lp.predict(
                &receipt_embs[r_num],
                &tx_targets,
                None,
                None,
                "receipt",
                r_num,
            );
            let rank = result
                .predictions
                .iter()
                .position(|p| p.target_id == t_num)
                .unwrap_or(tx_embs.len())
                + 1;
            ranks.push(rank);
        }

        let hit1 = ranks.iter().filter(|&&r| r == 1).count();
        let hit3 = ranks.iter().filter(|&&r| r <= 3).count();
        let hit5 = ranks.iter().filter(|&&r| r <= 5).count();
        let mrr: f64 = ranks.iter().map(|&r| 1.0 / r as f64).sum::<f64>() / ranks.len() as f64;

        println!("    Total receipt-tx pairs: {}", ranks.len());
        println!(
            "    Hit@1:  {}/{} ({:.0}%)",
            hit1,
            ranks.len(),
            hit1 as f64 / ranks.len() as f64 * 100.0
        );
        println!(
            "    Hit@3:  {}/{} ({:.0}%)",
            hit3,
            ranks.len(),
            hit3 as f64 / ranks.len() as f64 * 100.0
        );
        println!(
            "    Hit@5:  {}/{} ({:.0}%)",
            hit5,
            ranks.len(),
            hit5 as f64 / ranks.len() as f64 * 100.0
        );
        println!("    MRR:    {:.4}", mrr);

        // Embedding similarity
        let mut matched_sims = Vec::new();
        for (receipt, tx) in &gt_links {
            let r: usize = receipt.strip_prefix("receipt_").unwrap().parse().unwrap();
            let t: usize = tx.strip_prefix("tx_").unwrap().parse().unwrap();
            if r < receipt_embs.len() && t < tx_embs.len() {
                matched_sims.push(PlainEmbeddings::cosine_similarity(
                    &receipt_embs[r],
                    &tx_embs[t],
                ));
            }
        }
        let avg_sim = matched_sims.iter().sum::<f32>() / matched_sims.len().max(1) as f32;
        println!("    Matched pair avg similarity: {:.4}\n", avg_sim);
        println!("  ═══════════════════════════════════════════════════════════════\n");

        assert!(mrr.is_finite());
    }
}
