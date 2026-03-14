//! Multi-hop reasoning test: tests GNN's ability to answer queries
//! that require traversing multiple edges in the graph.
//!
//! Ground truth graph:
//!
//!   user ──owns──► account ──posted_to◄── tx ──at──► merchant ──in_cat──► category
//!                                                       │
//!                                                   has_receipt──► receipt
//!
//! Hop difficulty:
//!   1-hop: tx → account (direct edge)
//!   2-hop: tx → user    (tx→account→user)
//!   3-hop: tx → category (tx→merchant→category)
//!   3-hop: user → merchant (user→account→tx→merchant)
//!   4-hop: user → category (user→account→tx→merchant→category)
//!
//! For each hop level, we measure Hit@K — how often the GNN
//! ranks the correct answer in the top-K.

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

    const NUM_USERS: usize = 100;
    const TX_PER_USER: usize = 4;
    const NUM_CATEGORIES: usize = 10;

    /// Build a graph with known multi-hop paths.
    ///
    /// Each user has:
    ///  - 1 account
    ///  - 4 transactions (each posted to their account)
    ///  - Transactions go to 1 merchant
    ///  - Each merchant belongs to 1 of 10 categories
    ///  - Each transaction has 1 receipt
    struct GroundTruth {
        facts: Vec<GraphFact>,
        tx_to_account: HashMap<usize, usize>,    // 1-hop
        tx_to_user: HashMap<usize, usize>,       // 2-hop
        tx_to_category: HashMap<usize, usize>,   // 3-hop (tx→merchant→category)
        user_to_merchant: HashMap<usize, usize>, // 3-hop (user→account→tx→merchant)
        user_to_category: HashMap<usize, usize>, // 4-hop
    }

    fn build_multihop_graph() -> GroundTruth {
        let mut facts = Vec::new();
        let mut tx_to_account = HashMap::new();
        let mut tx_to_user = HashMap::new();
        let mut tx_to_category = HashMap::new();
        let mut user_to_merchant = HashMap::new();
        let mut user_to_category = HashMap::new();

        for user_id in 0..NUM_USERS {
            let user = format!("user_{}", user_id);
            let account = format!("account_{}", user_id);
            let merchant = format!("merchant_{}", user_id);
            let category_id = user_id % NUM_CATEGORIES;
            let category = format!("category_{}", category_id);

            // user → account (owns)
            facts.push(GraphFact {
                src: ("user".into(), user.clone()),
                relation: "owns".into(),
                dst: ("account".into(), account.clone()),
            });

            // merchant → category
            facts.push(GraphFact {
                src: ("merchant".into(), merchant.clone()),
                relation: "in_category".into(),
                dst: ("category".into(), category.clone()),
            });

            user_to_merchant.insert(user_id, user_id);
            user_to_category.insert(user_id, category_id);

            for tx_off in 0..TX_PER_USER {
                let tx_id = user_id * TX_PER_USER + tx_off;
                let tx = format!("tx_{}", tx_id);
                let receipt = format!("receipt_{}", tx_id);

                // tx → account (posted_to)
                facts.push(GraphFact {
                    src: ("tx".into(), tx.clone()),
                    relation: "posted_to".into(),
                    dst: ("account".into(), account.clone()),
                });

                // tx → merchant (at)
                facts.push(GraphFact {
                    src: ("tx".into(), tx.clone()),
                    relation: "at_merchant".into(),
                    dst: ("merchant".into(), merchant.clone()),
                });

                // tx → receipt (has_receipt)
                facts.push(GraphFact {
                    src: ("tx".into(), tx.clone()),
                    relation: "has_receipt".into(),
                    dst: ("receipt".into(), receipt.clone()),
                });

                tx_to_account.insert(tx_id, user_id);
                tx_to_user.insert(tx_id, user_id);
                tx_to_category.insert(tx_id, category_id);
            }
        }

        GroundTruth {
            facts,
            tx_to_account,
            tx_to_user,
            tx_to_category,
            user_to_merchant,
            user_to_category,
        }
    }

    /// Evaluate Hit@K for a set of queries.
    ///
    /// For each (source_id, ground_truth_target_id), score source
    /// against all candidates of the target type, and check if GT
    /// is in top-K.
    fn evaluate_hop(
        src_embs: &[Vec<f32>],
        dst_embs: &[Vec<f32>],
        ground_truth: &HashMap<usize, usize>,
        sample_size: usize,
    ) -> HopResults {
        let mut hit_at_1 = 0;
        let mut hit_at_3 = 0;
        let mut hit_at_5 = 0;
        let mut hit_at_10 = 0;
        let mut mrr_sum = 0.0f64;
        let mut count = 0;
        let total = ground_truth.len();
        let step = if total <= sample_size {
            1
        } else {
            total / sample_size
        };

        for (&src_id, &gt_dst_id) in ground_truth.iter() {
            if count >= sample_size {
                break;
            }
            if src_id % step != 0 {
                continue;
            }
            if src_id >= src_embs.len() || gt_dst_id >= dst_embs.len() {
                continue;
            }

            let src_emb = &src_embs[src_id];

            // Score against all destinations
            let mut scores: Vec<(usize, f32)> = dst_embs
                .iter()
                .enumerate()
                .map(|(id, emb)| (id, PlainEmbeddings::cosine_similarity(src_emb, emb)))
                .collect();

            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let rank = scores
                .iter()
                .position(|(id, _)| *id == gt_dst_id)
                .unwrap_or(999)
                + 1;

            if rank == 1 {
                hit_at_1 += 1;
            }
            if rank <= 3 {
                hit_at_3 += 1;
            }
            if rank <= 5 {
                hit_at_5 += 1;
            }
            if rank <= 10 {
                hit_at_10 += 1;
            }
            mrr_sum += 1.0 / rank as f64;
            count += 1;
        }

        let n = count.max(1) as f64;
        HopResults {
            queries: count,
            candidates: dst_embs.len(),
            hit_at_1: hit_at_1 as f64 / n,
            hit_at_3: hit_at_3 as f64 / n,
            hit_at_5: hit_at_5 as f64 / n,
            hit_at_10: hit_at_10 as f64 / n,
            mrr: mrr_sum / n,
        }
    }

    struct HopResults {
        queries: usize,
        candidates: usize,
        hit_at_1: f64,
        hit_at_3: f64,
        hit_at_5: f64,
        hit_at_10: f64,
        mrr: f64,
    }

    impl HopResults {
        fn print(&self, label: &str) {
            let random = 1.0 / self.candidates.max(1) as f64;
            println!(
                "    {} ({} queries, {} candidates, random={:.1}%)",
                label,
                self.queries,
                self.candidates,
                random * 100.0
            );
            println!(
                "      Hit@1:  {:.1}%  ({:.1}× random)",
                self.hit_at_1 * 100.0,
                self.hit_at_1 / random.max(1e-8)
            );
            println!(
                "      Hit@3:  {:.1}%  ({:.1}× random)",
                self.hit_at_3 * 100.0,
                self.hit_at_3 / random.max(1e-8)
            );
            println!(
                "      Hit@5:  {:.1}%  ({:.1}× random)",
                self.hit_at_5 * 100.0,
                self.hit_at_5 / random.max(1e-8)
            );
            println!(
                "      Hit@10: {:.1}%  ({:.1}× random)",
                self.hit_at_10 * 100.0,
                self.hit_at_10 / random.max(1e-8)
            );
            println!("      MRR:    {:.4}", self.mrr);
        }
    }

    #[test]
    fn test_multihop_reasoning() {
        let device = <B as Backend>::Device::default();
        let gt = build_multihop_graph();

        println!("\n  ════════════════════════════════════════════════");
        println!("   MULTI-HOP REASONING TEST");
        println!(
            "   {} users, {} tx, {} categories, {} receipts",
            NUM_USERS,
            NUM_USERS * TX_PER_USER,
            NUM_CATEGORIES,
            NUM_USERS * TX_PER_USER
        );
        println!("  ════════════════════════════════════════════════\n");

        // Build graph with 2-layer GNN (reaches 2-hop neighbors)
        let config = GraphBuildConfig {
            node_feat_dim: 16,
            add_reverse_edges: true,
            add_self_loops: true,
            add_positional_encoding: true,
            add_cross_dependency_edges: true,
        };
        let graph = build_hetero_graph::<B>(&gt.facts, &config, &device);

        println!(
            "  Graph: {} nodes, {} edges, {} types",
            graph.total_nodes(),
            graph.total_edges(),
            graph.node_types().len()
        );
        for nt in graph.node_types() {
            println!("    {}: {} nodes", nt, graph.node_counts[nt]);
        }

        let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
        let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();

        // Test with 2-layer GNN
        println!("\n  ── GNN with 2 layers (receptive field = 2 hops) ──\n");

        let sage_2layer = GraphSageModelConfig {
            in_dim: 16,
            hidden_dim: 64,
            num_layers: 2,
            dropout: 0.0,
        };
        let model_2 = sage_2layer.init::<B>(&node_types, &edge_types, &device);
        let emb_2 = PlainEmbeddings::from_burn(&model_2.forward(&graph));

        let tx_e = &emb_2.data["tx"];
        let acct_e = &emb_2.data["account"];
        let user_e = &emb_2.data["user"];
        let cat_e = &emb_2.data["category"];
        let merch_e = &emb_2.data["merchant"];

        let sample = 100;

        // 1-hop: tx → account (direct edge in graph)
        let r1 = evaluate_hop(tx_e, acct_e, &gt.tx_to_account, sample);
        r1.print("1-HOP: tx → account");

        // 2-hop: tx → user (tx→account→user, needs info from 2 edges)
        let r2 = evaluate_hop(tx_e, user_e, &gt.tx_to_user, sample);
        r2.print("2-HOP: tx → user");

        // 3-hop: tx → category (tx→merchant→category)
        let r3a = evaluate_hop(tx_e, cat_e, &gt.tx_to_category, sample);
        r3a.print("3-HOP: tx → category (via merchant)");

        // 3-hop: user → merchant (user→account→tx→merchant)
        let r3b = evaluate_hop(user_e, merch_e, &gt.user_to_merchant, sample);
        r3b.print("3-HOP: user → merchant (via account→tx)");

        // 4-hop: user → category (user→account→tx→merchant→category)
        let r4 = evaluate_hop(user_e, cat_e, &gt.user_to_category, sample);
        r4.print("4-HOP: user → category (via account→tx→merchant)");

        // Now test with 3-layer GNN (receptive field = 3 hops)
        println!("\n  ── GNN with 3 layers (receptive field = 3 hops) ──\n");

        let sage_3layer = GraphSageModelConfig {
            in_dim: 16,
            hidden_dim: 64,
            num_layers: 3,
            dropout: 0.0,
        };
        let model_3 = sage_3layer.init::<B>(&node_types, &edge_types, &device);
        let emb_3 = PlainEmbeddings::from_burn(&model_3.forward(&graph));

        let tx_e3 = &emb_3.data["tx"];
        let acct_e3 = &emb_3.data["account"];
        let user_e3 = &emb_3.data["user"];
        let cat_e3 = &emb_3.data["category"];
        let merch_e3 = &emb_3.data["merchant"];

        let r1_3 = evaluate_hop(tx_e3, acct_e3, &gt.tx_to_account, sample);
        r1_3.print("1-HOP: tx → account");

        let r2_3 = evaluate_hop(tx_e3, user_e3, &gt.tx_to_user, sample);
        r2_3.print("2-HOP: tx → user");

        let r3a_3 = evaluate_hop(tx_e3, cat_e3, &gt.tx_to_category, sample);
        r3a_3.print("3-HOP: tx → category");

        let r3b_3 = evaluate_hop(user_e3, merch_e3, &gt.user_to_merchant, sample);
        r3b_3.print("3-HOP: user → merchant");

        let r4_3 = evaluate_hop(user_e3, cat_e3, &gt.user_to_category, sample);
        r4_3.print("4-HOP: user → category");

        // Summary comparison
        println!("\n  ═══════════════════════════════════════════════════════════");
        println!("   SUMMARY: Hit@10 by Hop Distance");
        println!("  ───────────────────────────────────────────────────────────");
        println!("   Hops │ Query                  │ 2-Layer │ 3-Layer │ Harder?");
        println!("  ──────┼────────────────────────┼─────────┼─────────┼────────");
        println!(
            "    1   │ tx → account            │ {:>5.1}%  │ {:>5.1}%  │",
            r1.hit_at_10 * 100.0,
            r1_3.hit_at_10 * 100.0
        );
        println!(
            "    2   │ tx → user               │ {:>5.1}%  │ {:>5.1}%  │",
            r2.hit_at_10 * 100.0,
            r2_3.hit_at_10 * 100.0
        );
        println!(
            "    3   │ tx → category            │ {:>5.1}%  │ {:>5.1}%  │ ← concept",
            r3a.hit_at_10 * 100.0,
            r3a_3.hit_at_10 * 100.0
        );
        println!(
            "    3   │ user → merchant          │ {:>5.1}%  │ {:>5.1}%  │ ← cross-type",
            r3b.hit_at_10 * 100.0,
            r3b_3.hit_at_10 * 100.0
        );
        println!(
            "    4   │ user → category          │ {:>5.1}%  │ {:>5.1}%  │ ← hardest",
            r4.hit_at_10 * 100.0,
            r4_3.hit_at_10 * 100.0
        );
        println!("  ═══════════════════════════════════════════════════════════\n");

        // Assertions
        assert!(r1.mrr.is_finite());
        assert!(r4.mrr.is_finite());
    }
}
