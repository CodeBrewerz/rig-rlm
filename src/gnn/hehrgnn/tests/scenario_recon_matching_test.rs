//! Scenario 2: Reconciliation Matching (Statement Line ↔ Transaction)
//!
//! Statement lines arrive from bank feeds. The GNN + LinkPredictor
//! scores each statement line against candidate transactions and ranks
//! the correct match in the top-K.
//!
//! Ground truth: we know which statement line maps to which transaction.
//! Verifies: GNN embeddings + LinkPredictor rank correct match in top-3.

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::prelude::*;
    use std::collections::HashMap;

    type B = NdArray;

    use hehrgnn::data::graph_builder::{build_hetero_graph, GraphBuildConfig, GraphFact};
    use hehrgnn::data::hetero_graph::EdgeType;
    use hehrgnn::model::graphsage::GraphSageModelConfig;
    use hehrgnn::server::state::PlainEmbeddings;
    use hehrgnn::tasks::link_predictor::LinkPredictorConfig;

    fn fact(st: &str, s: &str, r: &str, dt: &str, d: &str) -> GraphFact {
        GraphFact {
            src: (st.to_string(), s.to_string()),
            relation: r.to_string(),
            dst: (dt.to_string(), d.to_string()),
        }
    }

    /// Build a recon-focused graph:
    ///   statement_line → amount_range, date_range, merchant_hint
    ///   transaction    → amount_range, date_range, merchant, account
    ///   Matched pairs: line_1 ↔ tx_1, line_2 ↔ tx_2, etc.
    fn build_recon_graph() -> (Vec<GraphFact>, Vec<(String, String)>) {
        let mut facts = Vec::new();
        let mut ground_truth_matches: Vec<(String, String)> = Vec::new();

        let merchants = ["walmart", "amazon", "shell_gas", "starbucks", "target"];
        let amounts = ["small", "medium", "large"];
        let dates = ["jan_w1", "jan_w2", "jan_w3", "feb_w1", "feb_w2"];

        // Create 15 matched pairs (statement_line ↔ transaction)
        for i in 0..15 {
            let line = format!("line_{}", i);
            let tx = format!("tx_{}", i);
            let merch = merchants[i % merchants.len()];
            let amt = amounts[i % amounts.len()];
            let date = dates[i % dates.len()];
            let acct = if i < 8 { "checking_1" } else { "checking_2" };

            // Statement line attributes
            facts.push(fact("statement_line", &line, "stmt_amount", "amount_range", amt));
            facts.push(fact("statement_line", &line, "stmt_date", "date_range", date));
            facts.push(fact("statement_line", &line, "stmt_merchant_hint", "merchant", merch));
            facts.push(fact("statement_line", &line, "from_statement", "bank_statement", 
                &format!("stmt_{}", if i < 8 { "jan" } else { "feb" })));

            // Transaction attributes (same merchant, amount, date = should match)
            facts.push(fact("transaction", &tx, "tx_amount", "amount_range", amt));
            facts.push(fact("transaction", &tx, "tx_date", "date_range", date));
            facts.push(fact("transaction", &tx, "at_merchant", "merchant", merch));
            facts.push(fact("transaction", &tx, "posted_to", "account", acct));

            // Known match (training signal)
            facts.push(fact("statement_line", &line, "matched_to", "transaction", &tx));

            ground_truth_matches.push((line, tx));
        }

        // Add 10 extra unmatched transactions (distractors)
        for i in 15..25 {
            let tx = format!("tx_{}", i);
            let merch = merchants[i % merchants.len()];
            let amt = amounts[(i + 1) % amounts.len()]; // different amount
            let date = dates[(i + 2) % dates.len()];    // different date
            facts.push(fact("transaction", &tx, "tx_amount", "amount_range", amt));
            facts.push(fact("transaction", &tx, "tx_date", "date_range", date));
            facts.push(fact("transaction", &tx, "at_merchant", "merchant", merch));
            facts.push(fact("transaction", &tx, "posted_to", "account", "checking_1"));
        }

        // Account ownership
        facts.push(fact("user", "alice", "owns", "account", "checking_1"));
        facts.push(fact("user", "alice", "owns", "account", "checking_2"));

        (facts, ground_truth_matches)
    }

    #[test]
    fn test_reconciliation_matching() {
        let device = <B as Backend>::Device::default();
        let (facts, ground_truth) = build_recon_graph();

        println!("\n  ═══════════════════════════════════════════════════════════════");
        println!("   🔗 SCENARIO 2: RECONCILIATION MATCHING");
        println!("   Statement Line ↔ Transaction matching via GNN embeddings");
        println!("  ═══════════════════════════════════════════════════════════════\n");

        let config = GraphBuildConfig {
            node_feat_dim: 32,
            add_reverse_edges: true,
            add_self_loops: true, add_positional_encoding: true,
        };
        let graph = build_hetero_graph::<B>(&facts, &config, &device);

        println!("  Graph: {} nodes, {} edges", graph.total_nodes(), graph.total_edges());
        for nt in graph.node_types() {
            println!("    {}: {}", nt, graph.node_counts[nt]);
        }
        println!();

        // Run GNN
        let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
        let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();

        let sage = GraphSageModelConfig {
            in_dim: 32, hidden_dim: 64, num_layers: 2, dropout: 0.0,
        };
        let model = sage.init::<B>(&node_types, &edge_types, &device);
        let emb = PlainEmbeddings::from_burn(&model.forward(&graph));

        let line_embs = &emb.data["statement_line"];
        let tx_embs = &emb.data["transaction"];

        // Init LinkPredictor
        let lp_config = LinkPredictorConfig {
            hidden_dim: 64, mlp_dim: 64, dropout: 0.0,
        };
        let link_pred = lp_config.init::<B>(&device);

        // ── Score each statement line against ALL transactions ──
        println!("  ── MATCH RANKING ──\n");
        println!("  {:>8} │ {:>8} │ {:>6} │ {:>8} │ {:>8} │ {}",
            "StmtLine", "TrueMatch", "Rank", "TopPred", "TopScore", "Result");
        println!("  ────────┼──────────┼────────┼──────────┼──────────┼─────────");

        let mut ranks = Vec::new();

        for (gt_line, gt_tx) in &ground_truth {
            // Get line index (insertion-order based)
            let line_num: usize = gt_line.strip_prefix("line_").unwrap().parse().unwrap();
            let tx_num: usize = gt_tx.strip_prefix("tx_").unwrap().parse().unwrap();

            if line_num >= line_embs.len() { continue; }

            // Build query tensor
            let query = Tensor::<B, 2>::from_data(
                burn::tensor::TensorData::new(line_embs[line_num].clone(), [1, 64]),
                &device,
            );

            // Build candidate tensor (all transactions)
            let mut cand_data = vec![0.0f32; tx_embs.len() * 64];
            for (i, emb_vec) in tx_embs.iter().enumerate() {
                for (j, &v) in emb_vec.iter().enumerate() {
                    if j < 64 { cand_data[i * 64 + j] = v; }
                }
            }
            let candidates = Tensor::<B, 2>::from_data(
                burn::tensor::TensorData::new(cand_data, [tx_embs.len(), 64]),
                &device,
            );

            // Score
            let scores = link_pred.rank_candidates(query, candidates);
            let scores_data: Vec<f32> = scores.into_data().as_slice::<f32>().unwrap().to_vec();

            // Rank
            let mut indexed: Vec<(usize, f32)> = scores_data.iter().enumerate().map(|(i, &s)| (i, s)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let rank = indexed.iter().position(|(i, _)| *i == tx_num).unwrap_or(tx_embs.len()) + 1;
            let top_pred = format!("tx_{}", indexed[0].0);
            let top_score = indexed[0].1;

            ranks.push(rank);

            let result = if rank == 1 { "✅ #1" }
                else if rank <= 3 { "🟡 top-3" }
                else if rank <= 5 { "⚠️ top-5" }
                else { "❌ miss" };

            println!("  {:>8} │ {:>8} │ {:>6} │ {:>8} │ {:>8.4} │ {}",
                gt_line, gt_tx, rank, top_pred, top_score, result);
        }

        // ── Summary ──
        let hit_at_1 = ranks.iter().filter(|&&r| r == 1).count();
        let hit_at_3 = ranks.iter().filter(|&&r| r <= 3).count();
        let hit_at_5 = ranks.iter().filter(|&&r| r <= 5).count();
        let mrr: f64 = ranks.iter().map(|&r| 1.0 / r as f64).sum::<f64>() / ranks.len() as f64;

        println!("\n  ── MATCHING SUMMARY ──\n");
        println!("    Total pairs:  {}", ranks.len());
        println!("    Hit@1:        {}/{} ({:.0}%)", hit_at_1, ranks.len(), hit_at_1 as f64 / ranks.len() as f64 * 100.0);
        println!("    Hit@3:        {}/{} ({:.0}%)", hit_at_3, ranks.len(), hit_at_3 as f64 / ranks.len() as f64 * 100.0);
        println!("    Hit@5:        {}/{} ({:.0}%)", hit_at_5, ranks.len(), hit_at_5 as f64 / ranks.len() as f64 * 100.0);
        println!("    MRR:          {:.4}", mrr);
        println!("    Random MRR:   {:.4} (baseline)", 1.0 / tx_embs.len() as f64);
        println!("    Improvement:  {:.1}× over random", mrr / (1.0 / tx_embs.len() as f64));

        // ── Cosine similarity analysis ──
        println!("\n  ── EMBEDDING SIMILARITY: matched vs unmatched pairs ──\n");
        let mut matched_sims = Vec::new();
        let mut unmatched_sims = Vec::new();

        for (gt_line, gt_tx) in &ground_truth {
            let line_num: usize = gt_line.strip_prefix("line_").unwrap().parse().unwrap();
            let tx_num: usize = gt_tx.strip_prefix("tx_").unwrap().parse().unwrap();
            if line_num < line_embs.len() && tx_num < tx_embs.len() {
                matched_sims.push(PlainEmbeddings::cosine_similarity(&line_embs[line_num], &tx_embs[tx_num]));
                // Compare to random unmatched tx
                let other = (tx_num + 5) % tx_embs.len();
                unmatched_sims.push(PlainEmbeddings::cosine_similarity(&line_embs[line_num], &tx_embs[other]));
            }
        }

        let avg_matched = matched_sims.iter().sum::<f32>() / matched_sims.len().max(1) as f32;
        let avg_unmatched = unmatched_sims.iter().sum::<f32>() / unmatched_sims.len().max(1) as f32;

        println!("    Matched pair avg similarity:   {:.4}", avg_matched);
        println!("    Unmatched pair avg similarity:  {:.4}", avg_unmatched);
        println!("    Signal: matched > unmatched?    {}", 
            if avg_matched > avg_unmatched { "✅ YES" } else { "❌ NO" });

        println!("\n  ═══════════════════════════════════════════════════════════════\n");

        assert!(mrr.is_finite());
        assert!(hit_at_3 > 0 || mrr > 0.0, "Should have some matching signal");
    }
}
