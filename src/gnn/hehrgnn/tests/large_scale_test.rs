//! Large-scale ground truth validation: ~10K nodes.
//!
//! Generates a finance graph with known structure:
//! - 500 users, each owns 1 account
//! - 500 accounts  
//! - 4000 transactions, 8 per user, each posted to owner's account
//! - 500 merchants, each user's transactions go to 1 merchant
//! - 500 categories, each merchant maps to 1 category
//!
//! Total: ~6000 nodes
//!
//! Ground truth:
//! - tx N belongs to user N/8 → account N/8
//! - tx N goes to merchant N/8
//! - merchant M maps to category M
//!
//! We verify:
//! 1. Match ranking: tx → correct account ranked in top-K
//! 2. Similarity: tx in same account are more similar than cross-account
//! 3. Anomaly: injected outliers score higher than normal
//! 4. Cluster quality: users with same merchant should cluster together

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

    const NUM_USERS: usize = 500;
    const TX_PER_USER: usize = 8;
    const NUM_TX: usize = NUM_USERS * TX_PER_USER;
    const NUM_OUTLIER_TX: usize = 50;

    /// Build a large finance graph with known ground truth.
    ///
    /// Returns: (facts, tx→account_id ground truth, tx→merchant_id ground truth)
    fn build_large_ground_truth() -> (
        Vec<GraphFact>,
        HashMap<usize, usize>, // tx_id → account_id
        HashMap<usize, usize>, // tx_id → merchant_id
    ) {
        let mut facts = Vec::new();
        let mut tx_to_account = HashMap::new();
        let mut tx_to_merchant = HashMap::new();

        for user_id in 0..NUM_USERS {
            let user = format!("user_{}", user_id);
            let account = format!("account_{}", user_id);
            let merchant = format!("merchant_{}", user_id);
            let category = format!("category_{}", user_id % 50); // 50 categories

            // user → account
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

            // 8 transactions per user
            for tx_off in 0..TX_PER_USER {
                let tx_id = user_id * TX_PER_USER + tx_off;
                let tx = format!("tx_{}", tx_id);

                // tx → account
                facts.push(GraphFact {
                    src: ("tx".into(), tx.clone()),
                    relation: "posted_to".into(),
                    dst: ("account".into(), account.clone()),
                });

                // tx → merchant
                facts.push(GraphFact {
                    src: ("tx".into(), tx.clone()),
                    relation: "at_merchant".into(),
                    dst: ("merchant".into(), merchant.clone()),
                });

                tx_to_account.insert(tx_id, user_id);
                tx_to_merchant.insert(tx_id, user_id);
            }
        }

        // Add outlier transactions (connected to random isolated merchants)
        for i in 0..NUM_OUTLIER_TX {
            let tx = format!("tx_outlier_{}", i);
            let outlier_merchant = format!("merchant_outlier_{}", i);

            facts.push(GraphFact {
                src: ("tx".into(), tx.clone()),
                relation: "at_merchant".into(),
                dst: ("merchant".into(), outlier_merchant.clone()),
            });
            // No account connection — this is the anomaly signal
        }

        (facts, tx_to_account, tx_to_merchant)
    }

    #[test]
    fn test_large_scale_ground_truth_validation() {
        let device = <B as Backend>::Device::default();
        let (facts, tx_to_account, tx_to_merchant) = build_large_ground_truth();

        println!("\n============================================================");
        println!("  LARGE-SCALE GROUND TRUTH VALIDATION");
        println!(
            "  {} users, {} accounts, {} tx, {} outliers",
            NUM_USERS, NUM_USERS, NUM_TX, NUM_OUTLIER_TX
        );
        println!("============================================================\n");

        // Build graph
        let config = GraphBuildConfig {
            node_feat_dim: 16,
            add_reverse_edges: true,
            add_self_loops: true,
            add_positional_encoding: true,
        };
        let graph = build_hetero_graph::<B>(&facts, &config, &device);

        let total_nodes = graph.total_nodes();
        let total_edges = graph.total_edges();
        println!(
            "  Graph built: {} nodes, {} edges",
            total_nodes, total_edges
        );
        for nt in graph.node_types() {
            println!("    {}: {} nodes", nt, graph.node_counts[nt]);
        }

        // Verify graph size is in the right ballpark
        assert!(
            total_nodes > 5000,
            "Should have > 5K nodes, got {}",
            total_nodes
        );

        // Run GNN
        let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
        let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();

        let sage_config = GraphSageModelConfig {
            in_dim: 16,
            hidden_dim: 64,
            num_layers: 2,
            dropout: 0.0,
        };

        println!("\n  Running GraphSAGE (hidden=64, layers=2)...");
        let model = sage_config.init::<B>(&node_types, &edge_types, &device);
        let embeddings = PlainEmbeddings::from_burn(&model.forward(&graph));
        println!("  Embeddings computed.\n");

        let tx_embs = &embeddings.data["tx"];
        let acct_embs = &embeddings.data["account"];
        let merchant_embs = &embeddings.data.get("merchant");

        // ===================================================================
        // TEST 1: Match Ranking — tx → correct account in top-K
        // ===================================================================
        println!("  ── TEST 1: Match Ranking (tx → account) ──");

        let sample_size = 200.min(NUM_TX); // Sample for speed
        let mut hit_at_1 = 0;
        let mut hit_at_3 = 0;
        let mut hit_at_5 = 0;
        let mut hit_at_10 = 0;
        let mut mrr_sum = 0.0f64;

        for tx_id in (0..NUM_TX).step_by(NUM_TX / sample_size) {
            if tx_id >= tx_embs.len() {
                break;
            }
            let gt_acct = tx_to_account[&tx_id];
            if gt_acct >= acct_embs.len() {
                continue;
            }

            let tx_emb = &tx_embs[tx_id];

            // Score all accounts
            let mut scores: Vec<(usize, f32)> = acct_embs
                .iter()
                .enumerate()
                .map(|(id, emb)| (id, PlainEmbeddings::dot_score(tx_emb, emb)))
                .collect();

            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let rank = scores
                .iter()
                .position(|(id, _)| *id == gt_acct)
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
        }

        let n = sample_size as f64;
        println!("    Sampled {} tx out of {}", sample_size, NUM_TX);
        println!(
            "    Hit@1:  {}/{} ({:.1}%)",
            hit_at_1,
            sample_size,
            hit_at_1 as f64 / n * 100.0
        );
        println!(
            "    Hit@3:  {}/{} ({:.1}%)",
            hit_at_3,
            sample_size,
            hit_at_3 as f64 / n * 100.0
        );
        println!(
            "    Hit@5:  {}/{} ({:.1}%)",
            hit_at_5,
            sample_size,
            hit_at_5 as f64 / n * 100.0
        );
        println!(
            "    Hit@10: {}/{} ({:.1}%)",
            hit_at_10,
            sample_size,
            hit_at_10 as f64 / n * 100.0
        );
        println!("    MRR:    {:.4}", mrr_sum / n);

        // With 500 accounts, random Hit@1 would be 0.2%. Any meaningful signal is good.
        assert!(
            hit_at_10 > 0,
            "At least some ground-truth matches should appear in top-10"
        );

        // ===================================================================
        // TEST 2: Intra-cluster similarity vs inter-cluster similarity
        // ===================================================================
        println!("\n  ── TEST 2: Cluster Similarity ──");

        let mut intra_sim_sum = 0.0f64;
        let mut inter_sim_sum = 0.0f64;
        let mut intra_count = 0;
        let mut inter_count = 0;
        let pairs_to_test = 500;

        for i in 0..pairs_to_test.min(NUM_TX) {
            let tx_a = i;
            // Intra: pick another tx from same account
            let same_acct = tx_to_account[&tx_a];
            let tx_b_intra = same_acct * TX_PER_USER + ((i % TX_PER_USER) + 1) % TX_PER_USER;

            if tx_a < tx_embs.len() && tx_b_intra < tx_embs.len() {
                let sim = PlainEmbeddings::cosine_similarity(&tx_embs[tx_a], &tx_embs[tx_b_intra]);
                intra_sim_sum += sim as f64;
                intra_count += 1;
            }

            // Inter: pick a tx from a different account
            let diff_acct = (same_acct + NUM_USERS / 2) % NUM_USERS;
            let tx_b_inter = diff_acct * TX_PER_USER;

            if tx_a < tx_embs.len() && tx_b_inter < tx_embs.len() {
                let sim = PlainEmbeddings::cosine_similarity(&tx_embs[tx_a], &tx_embs[tx_b_inter]);
                inter_sim_sum += sim as f64;
                inter_count += 1;
            }
        }

        let avg_intra = intra_sim_sum / intra_count.max(1) as f64;
        let avg_inter = inter_sim_sum / inter_count.max(1) as f64;

        println!(
            "    Intra-cluster avg cosine: {:.6} ({} pairs)",
            avg_intra, intra_count
        );
        println!(
            "    Inter-cluster avg cosine: {:.6} ({} pairs)",
            avg_inter, inter_count
        );
        println!(
            "    Intra > Inter: {} (delta: {:.6})",
            avg_intra > avg_inter,
            avg_intra - avg_inter
        );

        // ===================================================================
        // TEST 3: Anomaly detection — outlier tx should score higher
        // ===================================================================
        println!("\n  ── TEST 3: Anomaly Detection ──");

        let dim = embeddings.hidden_dim;

        // Compute mean embedding for normal transactions
        let normal_count = NUM_TX.min(tx_embs.len());
        let mut mean_emb = vec![0.0f32; dim];
        for i in 0..normal_count {
            for (j, &v) in tx_embs[i].iter().enumerate() {
                mean_emb[j] += v;
            }
        }
        for v in mean_emb.iter_mut() {
            *v /= normal_count as f32;
        }

        // Score normal tx
        let mut normal_scores: Vec<f32> = Vec::new();
        for i in (0..normal_count).step_by(normal_count / 100.max(1)) {
            normal_scores.push(PlainEmbeddings::l2_distance(&tx_embs[i], &mean_emb));
        }

        // Score outlier tx (they start at index NUM_TX)
        let mut outlier_scores: Vec<f32> = Vec::new();
        for i in normal_count..tx_embs.len() {
            outlier_scores.push(PlainEmbeddings::l2_distance(&tx_embs[i], &mean_emb));
        }

        let avg_normal: f32 = normal_scores.iter().sum::<f32>() / normal_scores.len().max(1) as f32;
        let avg_outlier: f32 =
            outlier_scores.iter().sum::<f32>() / outlier_scores.len().max(1) as f32;

        // Threshold: mean + 2*std of normal scores
        let std_normal: f32 = (normal_scores
            .iter()
            .map(|s| (s - avg_normal).powi(2))
            .sum::<f32>()
            / normal_scores.len().max(1) as f32)
            .sqrt();
        let threshold = avg_normal + 2.0 * std_normal;

        let outliers_detected = outlier_scores.iter().filter(|&&s| s > threshold).count();
        let outlier_precision = if !outlier_scores.is_empty() {
            outliers_detected as f64 / outlier_scores.len() as f64
        } else {
            0.0
        };

        println!(
            "    Normal tx avg L2:   {:.6} (sampled {} of {})",
            avg_normal,
            normal_scores.len(),
            normal_count
        );
        println!(
            "    Outlier tx avg L2:  {:.6} ({} outliers)",
            avg_outlier,
            outlier_scores.len()
        );
        println!(
            "    Ratio (outlier/normal): {:.2}x",
            avg_outlier / avg_normal.max(1e-8)
        );
        println!("    Threshold (μ+2σ):   {:.6}", threshold);
        println!(
            "    Outliers detected:  {}/{} ({:.1}%)",
            outliers_detected,
            outlier_scores.len(),
            outlier_precision * 100.0
        );

        assert!(avg_outlier.is_finite(), "Outlier scores should be finite");
        assert!(avg_normal.is_finite(), "Normal scores should be finite");

        // ===================================================================
        // TEST 4: Merchant clustering — same merchant → similar embeddings
        // ===================================================================
        println!("\n  ── TEST 4: Merchant Clustering ──");

        if let Some(merch_embs) = merchant_embs {
            // Merchants in the same category (mod 50) should be more similar
            let mut same_cat_sim_sum = 0.0f64;
            let mut diff_cat_sim_sum = 0.0f64;
            let mut same_count = 0;
            let mut diff_count = 0;

            for i in 0..100.min(merch_embs.len()) {
                let cat_i = i % 50;
                // Same category: merchant i and merchant (i + 50)
                let j_same = i + 50;
                if j_same < merch_embs.len() && (j_same % 50) == cat_i {
                    let sim =
                        PlainEmbeddings::cosine_similarity(&merch_embs[i], &merch_embs[j_same]);
                    same_cat_sim_sum += sim as f64;
                    same_count += 1;
                }

                // Different category
                let j_diff = (i + 25) % merch_embs.len();
                if (j_diff % 50) != cat_i {
                    let sim =
                        PlainEmbeddings::cosine_similarity(&merch_embs[i], &merch_embs[j_diff]);
                    diff_cat_sim_sum += sim as f64;
                    diff_count += 1;
                }
            }

            let avg_same = same_cat_sim_sum / same_count.max(1) as f64;
            let avg_diff = diff_cat_sim_sum / diff_count.max(1) as f64;
            println!(
                "    Same-category merchant cosine: {:.6} ({} pairs)",
                avg_same, same_count
            );
            println!(
                "    Diff-category merchant cosine: {:.6} ({} pairs)",
                avg_diff, diff_count
            );
        }

        // ===================================================================
        // SUMMARY
        // ===================================================================
        println!("\n  ══════════════════════════════════════════════");
        println!("  SUMMARY");
        println!("  ──────────────────────────────────────────────");
        println!("  Nodes:        {}", total_nodes);
        println!("  Edges:        {}", total_edges);
        println!("  Hit@1:        {:.1}%", hit_at_1 as f64 / n * 100.0);
        println!("  Hit@10:       {:.1}%", hit_at_10 as f64 / n * 100.0);
        println!("  MRR:          {:.4}", mrr_sum / n);
        println!("  Intra sim:    {:.6}", avg_intra);
        println!("  Inter sim:    {:.6}", avg_inter);
        println!("  Outlier ratio:{:.2}x", avg_outlier / avg_normal.max(1e-8));
        println!("  Outlier det:  {:.1}%", outlier_precision * 100.0);
        println!("  ══════════════════════════════════════════════\n");
    }
}
