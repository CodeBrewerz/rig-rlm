//! Verbose multi-hop predictions: shows each individual prediction,
//! the GNN's answer, the ground truth, and whether it was correct.

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

    /// Build a small graph where we can trace every prediction.
    /// 10 users, 4 tx each = 40 tx, 10 merchants, 5 categories.
    fn build_traceable_graph() -> (
        Vec<GraphFact>,
        HashMap<usize, usize>, // tx → account
        HashMap<usize, usize>, // tx → user
        HashMap<usize, usize>, // tx → category (via merchant)
        HashMap<usize, usize>, // user → merchant
        HashMap<usize, usize>, // user → category
    ) {
        let mut facts = Vec::new();
        let mut tx_acct = HashMap::new();
        let mut tx_user = HashMap::new();
        let mut tx_cat = HashMap::new();
        let mut u_merch = HashMap::new();
        let mut u_cat = HashMap::new();

        for uid in 0..10 {
            let user = format!("user_{}", uid);
            let acct = format!("account_{}", uid);
            let merch = format!("merchant_{}", uid);
            let cat_id = uid % 5;
            let cat = format!("category_{}", cat_id);

            facts.push(GraphFact {
                src: ("user".into(), user.clone()),
                relation: "owns".into(),
                dst: ("account".into(), acct.clone()),
            });
            facts.push(GraphFact {
                src: ("merchant".into(), merch.clone()),
                relation: "in_category".into(),
                dst: ("category".into(), cat.clone()),
            });

            u_merch.insert(uid, uid);
            u_cat.insert(uid, cat_id);

            for t in 0..4 {
                let tx_id = uid * 4 + t;
                let tx = format!("tx_{}", tx_id);

                facts.push(GraphFact {
                    src: ("tx".into(), tx.clone()),
                    relation: "posted_to".into(),
                    dst: ("account".into(), acct.clone()),
                });
                facts.push(GraphFact {
                    src: ("tx".into(), tx.clone()),
                    relation: "at_merchant".into(),
                    dst: ("merchant".into(), merch.clone()),
                });

                tx_acct.insert(tx_id, uid);
                tx_user.insert(tx_id, uid);
                tx_cat.insert(tx_id, cat_id);
            }
        }

        (facts, tx_acct, tx_user, tx_cat, u_merch, u_cat)
    }

    fn top_k_predictions(src_emb: &[f32], dst_embs: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
        let mut scores: Vec<(usize, f32)> = dst_embs
            .iter()
            .enumerate()
            .map(|(id, emb)| (id, PlainEmbeddings::cosine_similarity(src_emb, emb)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }

    #[test]
    fn test_verbose_multihop_predictions() {
        let device = <B as Backend>::Device::default();
        let (facts, tx_acct, tx_user, tx_cat, u_merch, u_cat) = build_traceable_graph();

        let config = GraphBuildConfig {
            node_feat_dim: 16,
            add_reverse_edges: true,
            add_self_loops: true,
            add_positional_encoding: true,
        };
        let graph = build_hetero_graph::<B>(&facts, &config, &device);

        let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
        let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();

        let sage = GraphSageModelConfig {
            in_dim: 16,
            hidden_dim: 64,
            num_layers: 2,
            dropout: 0.0,
        };
        let model = sage.init::<B>(&node_types, &edge_types, &device);
        let emb = PlainEmbeddings::from_burn(&model.forward(&graph));

        let tx_e = &emb.data["tx"];
        let acct_e = &emb.data["account"];
        let user_e = &emb.data["user"];
        let cat_e = &emb.data["category"];
        let merch_e = &emb.data["merchant"];

        // ═══════════════════════════════════════════════════
        println!("\n  ═══════════════════════════════════════════════════════════════");
        println!("   VERBOSE MULTI-HOP PREDICTIONS — 10 users, 40 tx, 5 categories");
        println!("  ═══════════════════════════════════════════════════════════════\n");

        // ───────────────────────────────────────────────────
        // 1-HOP: tx → account (direct edge)
        // ───────────────────────────────────────────────────
        println!("  ── 1-HOP: Which account does this tx belong to? ──");
        println!("  (Path: tx ──posted_to──▶ account)\n");
        println!(
            "  {:>6} │ {:>10} │ {:>14} │ {:>10} │ {}",
            "TX", "Predicted", "Ground Truth", "Rank", "Correct?"
        );
        println!("  ──────┼────────────┼────────────────┼────────────┼─────────");

        let mut h1_correct = 0;
        let mut h1_top3 = 0;
        for tx_id in [0, 4, 8, 12, 16, 20, 24, 28, 32, 36] {
            if tx_id >= tx_e.len() {
                continue;
            }
            let gt = tx_acct[&tx_id];
            let preds = top_k_predictions(&tx_e[tx_id], acct_e, 10);
            let predicted_id = preds[0].0;
            let rank = preds
                .iter()
                .position(|(id, _)| *id == gt)
                .map(|r| r + 1)
                .unwrap_or(99);
            let correct = predicted_id == gt;
            if correct {
                h1_correct += 1;
            }
            if rank <= 3 {
                h1_top3 += 1;
            }

            println!(
                "  tx_{:>3} │ account_{:<2} │ account_{:<2}     │ {:>10} │ {}",
                tx_id,
                predicted_id,
                gt,
                rank,
                if correct {
                    "✅ YES"
                } else if rank <= 3 {
                    "⚠️  top-3"
                } else {
                    "❌ NO"
                }
            );
        }
        println!(
            "\n  1-HOP Score: {}/10 exact, {}/10 in top-3\n",
            h1_correct, h1_top3
        );

        // ───────────────────────────────────────────────────
        // 2-HOP: tx → user (tx→account→user)
        // ───────────────────────────────────────────────────
        println!("  ── 2-HOP: Which user made this transaction? ──");
        println!("  (Path: tx ──posted_to──▶ account ◀──owns── user)\n");
        println!(
            "  {:>6} │ {:>10} │ {:>14} │ {:>10} │ {}",
            "TX", "Predicted", "Ground Truth", "Rank", "Correct?"
        );
        println!("  ──────┼────────────┼────────────────┼────────────┼─────────");

        let mut h2_correct = 0;
        let mut h2_top3 = 0;
        for tx_id in [0, 4, 8, 12, 16, 20, 24, 28, 32, 36] {
            if tx_id >= tx_e.len() {
                continue;
            }
            let gt = tx_user[&tx_id];
            let preds = top_k_predictions(&tx_e[tx_id], user_e, 10);
            let predicted_id = preds[0].0;
            let rank = preds
                .iter()
                .position(|(id, _)| *id == gt)
                .map(|r| r + 1)
                .unwrap_or(99);
            let correct = predicted_id == gt;
            if correct {
                h2_correct += 1;
            }
            if rank <= 3 {
                h2_top3 += 1;
            }

            println!(
                "  tx_{:>3} │ user_{:<5}  │ user_{:<5}      │ {:>10} │ {}",
                tx_id,
                predicted_id,
                gt,
                rank,
                if correct {
                    "✅ YES"
                } else if rank <= 3 {
                    "⚠️  top-3"
                } else {
                    "❌ NO"
                }
            );
        }
        println!(
            "\n  2-HOP Score: {}/10 exact, {}/10 in top-3\n",
            h2_correct, h2_top3
        );

        // ───────────────────────────────────────────────────
        // 3-HOP: tx → category (tx→merchant→category)
        // ───────────────────────────────────────────────────
        println!("  ── 3-HOP: What spending category is this transaction? ──");
        println!("  (Path: tx ──at_merchant──▶ merchant ──in_category──▶ category)\n");
        println!(
            "  {:>6} │ {:>12} │ {:>14} │ {:>10} │ {}",
            "TX", "Predicted", "Ground Truth", "Rank", "Correct?"
        );
        println!("  ──────┼──────────────┼────────────────┼────────────┼─────────");

        let mut h3_correct = 0;
        let mut h3_top3 = 0;
        for tx_id in [0, 4, 8, 12, 16, 20, 24, 28, 32, 36] {
            if tx_id >= tx_e.len() {
                continue;
            }
            let gt = tx_cat[&tx_id];
            let preds = top_k_predictions(&tx_e[tx_id], cat_e, 5);
            let predicted_id = preds[0].0;
            let rank = preds
                .iter()
                .position(|(id, _)| *id == gt)
                .map(|r| r + 1)
                .unwrap_or(99);
            let correct = predicted_id == gt;
            if correct {
                h3_correct += 1;
            }
            if rank <= 3 {
                h3_top3 += 1;
            }

            println!(
                "  tx_{:>3} │ category_{:<2} │ category_{:<2}    │ {:>10} │ {}",
                tx_id,
                predicted_id,
                gt,
                rank,
                if correct {
                    "✅ YES"
                } else if rank <= 3 {
                    "⚠️  top-3"
                } else {
                    "❌ NO"
                }
            );
        }
        println!(
            "\n  3-HOP Score: {}/10 exact, {}/10 in top-3 (out of 5 categories)\n",
            h3_correct, h3_top3
        );

        // ───────────────────────────────────────────────────
        // 3-HOP: user → merchant (user→account→tx→merchant)
        // ───────────────────────────────────────────────────
        println!("  ── 3-HOP: Which merchant does this user frequent? ──");
        println!("  (Path: user ──owns──▶ account ◀──posted_to── tx ──at_merchant──▶ merchant)\n");
        println!(
            "  {:>7} │ {:>12} │ {:>14} │ {:>10} │ {}",
            "User", "Predicted", "Ground Truth", "Rank", "Correct?"
        );
        println!("  ───────┼──────────────┼────────────────┼────────────┼─────────");

        let mut h3b_correct = 0;
        let mut h3b_top3 = 0;
        for uid in 0..10 {
            if uid >= user_e.len() {
                continue;
            }
            let gt = u_merch[&uid];
            let preds = top_k_predictions(&user_e[uid], merch_e, 10);
            let predicted_id = preds[0].0;
            let rank = preds
                .iter()
                .position(|(id, _)| *id == gt)
                .map(|r| r + 1)
                .unwrap_or(99);
            let correct = predicted_id == gt;
            if correct {
                h3b_correct += 1;
            }
            if rank <= 3 {
                h3b_top3 += 1;
            }

            println!(
                "  user_{} │ merchant_{:<2} │ merchant_{:<2}    │ {:>10} │ {}",
                uid,
                predicted_id,
                gt,
                rank,
                if correct {
                    "✅ YES"
                } else if rank <= 3 {
                    "⚠️  top-3"
                } else {
                    "❌ NO"
                }
            );
        }
        println!(
            "\n  3-HOP Score: {}/10 exact, {}/10 in top-3\n",
            h3b_correct, h3b_top3
        );

        // ───────────────────────────────────────────────────
        // 4-HOP: user → category (user→account→tx→merchant→category)
        // ───────────────────────────────────────────────────
        println!("  ── 4-HOP: What spending category does this user belong to? ──");
        println!(
            "  (Path: user ──owns──▶ account ◀──posted_to── tx ──at_merchant──▶ merchant ──in_category──▶ category)\n"
        );
        println!(
            "  {:>7} │ {:>12} │ {:>14} │ {:>10} │ {}",
            "User", "Predicted", "Ground Truth", "Rank", "Correct?"
        );
        println!("  ───────┼──────────────┼────────────────┼────────────┼─────────");

        let mut h4_correct = 0;
        let mut h4_top3 = 0;
        for uid in 0..10 {
            if uid >= user_e.len() {
                continue;
            }
            let gt = u_cat[&uid];
            let preds = top_k_predictions(&user_e[uid], cat_e, 5);
            let predicted_id = preds[0].0;
            let rank = preds
                .iter()
                .position(|(id, _)| *id == gt)
                .map(|r| r + 1)
                .unwrap_or(99);
            let correct = predicted_id == gt;
            if correct {
                h4_correct += 1;
            }
            if rank <= 3 {
                h4_top3 += 1;
            }

            println!(
                "  user_{} │ category_{:<2} │ category_{:<2}    │ {:>10} │ {}",
                uid,
                predicted_id,
                gt,
                rank,
                if correct {
                    "✅ YES"
                } else if rank <= 3 {
                    "⚠️  top-3"
                } else {
                    "❌ NO"
                }
            );
        }
        println!(
            "\n  4-HOP Score: {}/10 exact, {}/10 in top-3 (out of 5 categories)\n",
            h4_correct, h4_top3
        );

        // FINAL SUMMARY
        println!("  ═══════════════════════════════════════════════════════════════");
        println!("   FINAL SCORECARD (untrained GNN, structural signals only)");
        println!("  ───────────────────────────────────────────────────────────────");
        println!("   Hops │ Query                        │ Exact │ Top-3 │ Candidates");
        println!("  ──────┼──────────────────────────────┼───────┼───────┼───────────");
        println!(
            "    1   │ tx → account                  │ {}/10  │ {}/10  │ 10",
            h1_correct, h1_top3
        );
        println!(
            "    2   │ tx → user                     │ {}/10  │ {}/10  │ 10",
            h2_correct, h2_top3
        );
        println!(
            "    3   │ tx → category                 │ {}/10  │ {}/10  │ 5",
            h3_correct, h3_top3
        );
        println!(
            "    3   │ user → merchant               │ {}/10  │ {}/10  │ 10",
            h3b_correct, h3b_top3
        );
        println!(
            "    4   │ user → category               │ {}/10  │ {}/10  │ 5",
            h4_correct, h4_top3
        );
        println!("  ═══════════════════════════════════════════════════════════════\n");
    }
}
