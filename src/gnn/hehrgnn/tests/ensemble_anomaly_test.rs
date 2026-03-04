//! Ensemble Multi-Model Score Combiner
//!
//! Runs the SAME financial anomaly scenario through 3 different GNN architectures:
//!   1. GraphSAGE (inductive) — structural embedding distance
//!   2. RGCN (relation-typed) — relation-aware embedding distance
//!   3. HEHRGNN (transductive) — fact plausibility scoring (DistMult)
//!
//! Then combines all scores via:
//!   - Min-max normalization to [0,1]
//!   - Weighted ensemble: final = w1×sage + w2×rgcn + w3×hehrgnn + w4×amount_zscore
//!   - Comparison: individual models vs ensemble

#[cfg(test)]
mod tests {
    use burn::backend::{Autodiff, NdArray};
    use burn::prelude::*;
    use std::collections::HashMap;

    type B = NdArray;
    type TrainB = Autodiff<NdArray>;
    type InferB = NdArray;

    use burn::data::dataloader::batcher::Batcher;
    use hehrgnn::data::batcher::{HehrBatch, HehrBatcher, HehrFactItem};
    use hehrgnn::data::fact::{HehrFact, RawFact};
    use hehrgnn::data::graph_builder::{build_hetero_graph, GraphBuildConfig, GraphFact};
    use hehrgnn::data::hetero_graph::EdgeType;
    use hehrgnn::data::vocab::KgVocabulary;
    use hehrgnn::model::graphsage::GraphSageModelConfig;
    use hehrgnn::model::rgcn::RgcnConfig;
    use hehrgnn::server::state::PlainEmbeddings;
    use hehrgnn::training::scoring::DistMultScorer;
    use hehrgnn::training::train::{train, TrainConfig};

    fn gfact(st: &str, s: &str, r: &str, dt: &str, d: &str) -> GraphFact {
        GraphFact {
            src: (st.to_string(), s.to_string()),
            relation: r.to_string(),
            dst: (dt.to_string(), d.to_string()),
        }
    }

    /// Merchant spending profiles for amount z-score calculation
    struct MerchantProfile {
        mean: f64,
        std: f64,
    }

    /// Normalize a slice of scores to [0, 1] (higher = more anomalous)
    fn min_max_normalize(scores: &[f64]) -> Vec<f64> {
        let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = (max - min).max(1e-10);
        scores.iter().map(|s| (s - min) / range).collect()
    }

    /// Transaction data for the scenario
    struct TxData {
        id: String,
        user: String,
        merchant: String,
        amount: f64,
        is_anomaly: bool,
        label: String,
    }

    fn build_scenario() -> (Vec<GraphFact>, Vec<RawFact>, Vec<TxData>) {
        let mut graph_facts = Vec::new();
        let mut kg_facts = Vec::new();
        let mut txs = Vec::new();

        // Merchant profiles
        let merchants: HashMap<&str, MerchantProfile> = [
            (
                "mcdonalds",
                MerchantProfile {
                    mean: 15.0,
                    std: 8.0,
                },
            ),
            (
                "starbucks",
                MerchantProfile {
                    mean: 6.0,
                    std: 3.0,
                },
            ),
            (
                "amazon",
                MerchantProfile {
                    mean: 50.0,
                    std: 40.0,
                },
            ),
            (
                "grocery_store",
                MerchantProfile {
                    mean: 80.0,
                    std: 30.0,
                },
            ),
            (
                "gas_station",
                MerchantProfile {
                    mean: 45.0,
                    std: 15.0,
                },
            ),
        ]
        .into_iter()
        .collect();

        let users = ["alice", "bob", "carol"];

        // ── Normal transactions (60 total) ──
        let mut tx_idx = 0;
        for user in &users {
            // Each user has typical patterns
            let user_merchants = match *user {
                "alice" => vec!["mcdonalds", "starbucks", "grocery_store", "gas_station"],
                "bob" => vec!["amazon", "grocery_store", "gas_station", "mcdonalds"],
                _ => vec!["starbucks", "grocery_store", "amazon", "gas_station"],
            };

            for merch in &user_merchants {
                let profile = &merchants[merch];
                // 5 normal transactions per user-merchant pair
                for _ in 0..5 {
                    let amount = profile.mean + (tx_idx as f64 % 3.0 - 1.0) * profile.std * 0.5;
                    let amount = amount.max(1.0);
                    let tx_id = format!("tx_{}", tx_idx);
                    let amt_bucket = if amount < 20.0 {
                        "small"
                    } else if amount < 60.0 {
                        "medium"
                    } else {
                        "large"
                    };

                    graph_facts.push(gfact(
                        "transaction",
                        &tx_id,
                        "posted_to",
                        "account",
                        &format!("{}_checking", user),
                    ));
                    graph_facts.push(gfact(
                        "transaction",
                        &tx_id,
                        "at_merchant",
                        "merchant",
                        merch,
                    ));
                    graph_facts.push(gfact(
                        "transaction",
                        &tx_id,
                        "tx_amount",
                        "amount_range",
                        amt_bucket,
                    ));

                    kg_facts.push(RawFact {
                        head: user.to_string(),
                        relation: "transacts_at".into(),
                        tail: merch.to_string(),
                        qualifiers: vec![],
                    });
                    kg_facts.push(RawFact {
                        head: tx_id.clone(),
                        relation: "amount_range".into(),
                        tail: amt_bucket.to_string(),
                        qualifiers: vec![],
                    });

                    txs.push(TxData {
                        id: tx_id,
                        user: user.to_string(),
                        merchant: merch.to_string(),
                        amount,
                        is_anomaly: false,
                        label: format!("✅ {} at {} ${:.0}", user, merch, amount),
                    });
                    tx_idx += 1;
                }
            }

            // Account ownership
            graph_facts.push(gfact(
                "user",
                user,
                "owns",
                "account",
                &format!("{}_checking", user),
            ));
        }

        // ── ANOMALOUS transactions ──

        // ANOM1: Alice at McDonalds $1000 (avg is $15)
        let tx_id = format!("tx_{}", tx_idx);
        tx_idx += 1;
        graph_facts.push(gfact(
            "transaction",
            &tx_id,
            "posted_to",
            "account",
            "alice_checking",
        ));
        graph_facts.push(gfact(
            "transaction",
            &tx_id,
            "at_merchant",
            "merchant",
            "mcdonalds",
        ));
        graph_facts.push(gfact(
            "transaction",
            &tx_id,
            "tx_amount",
            "amount_range",
            "huge",
        ));
        kg_facts.push(RawFact {
            head: "alice".into(),
            relation: "transacts_at".into(),
            tail: "mcdonalds".into(),
            qualifiers: vec![],
        });
        kg_facts.push(RawFact {
            head: tx_id.clone(),
            relation: "amount_range".into(),
            tail: "huge".into(),
            qualifiers: vec![],
        });
        txs.push(TxData {
            id: tx_id,
            user: "alice".into(),
            merchant: "mcdonalds".into(),
            amount: 1000.0,
            is_anomaly: true,
            label: "🚨 alice→mcdonalds $1000 (avg $15)".into(),
        });

        // ANOM2: Bob at unknown merchant "crypto_exchange"
        let tx_id = format!("tx_{}", tx_idx);
        tx_idx += 1;
        graph_facts.push(gfact(
            "transaction",
            &tx_id,
            "posted_to",
            "account",
            "bob_checking",
        ));
        graph_facts.push(gfact(
            "transaction",
            &tx_id,
            "at_merchant",
            "merchant",
            "crypto_exchange",
        ));
        graph_facts.push(gfact(
            "transaction",
            &tx_id,
            "tx_amount",
            "amount_range",
            "huge",
        ));
        kg_facts.push(RawFact {
            head: "bob".into(),
            relation: "transacts_at".into(),
            tail: "crypto_exchange".into(),
            qualifiers: vec![],
        });
        kg_facts.push(RawFact {
            head: tx_id.clone(),
            relation: "amount_range".into(),
            tail: "huge".into(),
            qualifiers: vec![],
        });
        txs.push(TxData {
            id: tx_id,
            user: "bob".into(),
            merchant: "crypto_exchange".into(),
            amount: 5000.0,
            is_anomaly: true,
            label: "🚨 bob→crypto_exchange $5000 (NEW merchant)".into(),
        });

        // ANOM3: Carol at gas station $500 (avg $45)
        let tx_id = format!("tx_{}", tx_idx);
        tx_idx += 1;
        graph_facts.push(gfact(
            "transaction",
            &tx_id,
            "posted_to",
            "account",
            "carol_checking",
        ));
        graph_facts.push(gfact(
            "transaction",
            &tx_id,
            "at_merchant",
            "merchant",
            "gas_station",
        ));
        graph_facts.push(gfact(
            "transaction",
            &tx_id,
            "tx_amount",
            "amount_range",
            "huge",
        ));
        kg_facts.push(RawFact {
            head: "carol".into(),
            relation: "transacts_at".into(),
            tail: "gas_station".into(),
            qualifiers: vec![],
        });
        kg_facts.push(RawFact {
            head: tx_id.clone(),
            relation: "amount_range".into(),
            tail: "huge".into(),
            qualifiers: vec![],
        });
        txs.push(TxData {
            id: tx_id,
            user: "carol".into(),
            merchant: "gas_station".into(),
            amount: 500.0,
            is_anomaly: true,
            label: "🚨 carol→gas_station $500 (avg $45)".into(),
        });

        // ANOM4: Unknown user "hacker" at amazon
        let tx_id = format!("tx_{}", tx_idx);
        tx_idx += 1;
        graph_facts.push(gfact(
            "transaction",
            &tx_id,
            "posted_to",
            "account",
            "hacker_checking",
        ));
        graph_facts.push(gfact(
            "transaction",
            &tx_id,
            "at_merchant",
            "merchant",
            "amazon",
        ));
        graph_facts.push(gfact(
            "transaction",
            &tx_id,
            "tx_amount",
            "amount_range",
            "huge",
        ));
        graph_facts.push(gfact(
            "user",
            "hacker",
            "owns",
            "account",
            "hacker_checking",
        ));
        kg_facts.push(RawFact {
            head: "hacker".into(),
            relation: "transacts_at".into(),
            tail: "amazon".into(),
            qualifiers: vec![],
        });
        kg_facts.push(RawFact {
            head: tx_id.clone(),
            relation: "amount_range".into(),
            tail: "huge".into(),
            qualifiers: vec![],
        });
        txs.push(TxData {
            id: tx_id,
            user: "hacker".into(),
            merchant: "amazon".into(),
            amount: 9999.0,
            is_anomaly: true,
            label: "🚨 hacker→amazon $9999 (UNKNOWN user)".into(),
        });

        // ANOM5: Alice at starbucks $500 (avg $6)
        let tx_id = format!("tx_{}", tx_idx);
        graph_facts.push(gfact(
            "transaction",
            &tx_id,
            "posted_to",
            "account",
            "alice_checking",
        ));
        graph_facts.push(gfact(
            "transaction",
            &tx_id,
            "at_merchant",
            "merchant",
            "starbucks",
        ));
        graph_facts.push(gfact(
            "transaction",
            &tx_id,
            "tx_amount",
            "amount_range",
            "huge",
        ));
        kg_facts.push(RawFact {
            head: "alice".into(),
            relation: "transacts_at".into(),
            tail: "starbucks".into(),
            qualifiers: vec![],
        });
        kg_facts.push(RawFact {
            head: tx_id.clone(),
            relation: "amount_range".into(),
            tail: "huge".into(),
            qualifiers: vec![],
        });
        txs.push(TxData {
            id: tx_id,
            user: "alice".into(),
            merchant: "starbucks".into(),
            amount: 500.0,
            is_anomaly: true,
            label: "🚨 alice→starbucks $500 (avg $6)".into(),
        });

        (graph_facts, kg_facts, txs)
    }

    #[test]
    fn test_ensemble_anomaly_detection() {
        let device_b = <B as Backend>::Device::default();
        let device_t = <TrainB as Backend>::Device::default();
        let (graph_facts, kg_facts, txs) = build_scenario();

        println!("\n  ═══════════════════════════════════════════════════════════════");
        println!("   🏗️  ENSEMBLE MULTI-MODEL ANOMALY DETECTION");
        println!("   GraphSAGE + RGCN + HEHRGNN + Amount Z-Score → Combined");
        println!("  ═══════════════════════════════════════════════════════════════\n");

        let num_normal = txs.iter().filter(|t| !t.is_anomaly).count();
        let num_anomaly = txs.iter().filter(|t| t.is_anomaly).count();
        println!(
            "  Dataset: {} transactions ({} normal, {} anomalous)\n",
            txs.len(),
            num_normal,
            num_anomaly
        );

        // ═══════════════════════════════════════════════
        // MODEL 1: GraphSAGE
        // ═══════════════════════════════════════════════
        println!("  ── MODEL 1: GraphSAGE (inductive, structural) ──\n");

        let config = GraphBuildConfig {
            node_feat_dim: 32,
            add_reverse_edges: true,
            add_self_loops: true, add_positional_encoding: true,
        };
        let graph = build_hetero_graph::<B>(&graph_facts, &config, &device_b);

        let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
        let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();

        let sage_model = GraphSageModelConfig {
            in_dim: 32,
            hidden_dim: 64,
            num_layers: 2,
            dropout: 0.0,
        }
        .init::<B>(&node_types, &edge_types, &device_b);
        let sage_emb = PlainEmbeddings::from_burn(&sage_model.forward(&graph));

        // Compute per-merchant centroids
        let tx_embs = &sage_emb.data["transaction"];
        let merchant_embs = &sage_emb.data.get("merchant");

        // GraphSAGE score: L2 distance of each tx from its merchant centroid
        let mut sage_scores: Vec<f64> = Vec::new();
        let global_centroid: Vec<f32> = (0..64)
            .map(|j| {
                tx_embs
                    .iter()
                    .map(|e| if j < e.len() { e[j] } else { 0.0 })
                    .sum::<f32>()
                    / tx_embs.len() as f32
            })
            .collect();

        for (i, tx) in txs.iter().enumerate() {
            if i < tx_embs.len() {
                let dist = PlainEmbeddings::l2_distance(&tx_embs[i], &global_centroid) as f64;
                sage_scores.push(dist);
            } else {
                sage_scores.push(0.0);
            }
        }
        println!("    ✅ GraphSAGE: {} scores computed", sage_scores.len());

        // ═══════════════════════════════════════════════
        // MODEL 2: RGCN
        // ═══════════════════════════════════════════════
        println!("  ── MODEL 2: RGCN (relation-typed message passing) ──\n");

        let rgcn_model = RgcnConfig {
            in_dim: 32,
            hidden_dim: 64,
            num_layers: 2,
            num_bases: 4,
            dropout: 0.0,
        }
        .init_model::<B>(&node_types, &edge_types, &device_b);
        let rgcn_emb = PlainEmbeddings::from_burn(&rgcn_model.forward(&graph));

        let rgcn_tx_embs = &rgcn_emb.data["transaction"];
        let rgcn_centroid: Vec<f32> = (0..64)
            .map(|j| {
                rgcn_tx_embs
                    .iter()
                    .map(|e| if j < e.len() { e[j] } else { 0.0 })
                    .sum::<f32>()
                    / rgcn_tx_embs.len() as f32
            })
            .collect();

        let mut rgcn_scores: Vec<f64> = Vec::new();
        for (i, _tx) in txs.iter().enumerate() {
            if i < rgcn_tx_embs.len() {
                let dist = PlainEmbeddings::l2_distance(&rgcn_tx_embs[i], &rgcn_centroid) as f64;
                rgcn_scores.push(dist);
            } else {
                rgcn_scores.push(0.0);
            }
        }
        println!("    ✅ RGCN: {} scores computed", rgcn_scores.len());

        // ═══════════════════════════════════════════════
        // MODEL 3: HEHRGNN (DistMult fact plausibility)
        // ═══════════════════════════════════════════════
        println!("  ── MODEL 3: HEHRGNN DistMult (transductive, fact plausibility) ──\n");

        let vocab = KgVocabulary::from_facts(&kg_facts);
        let kg_idx: Vec<HehrFact> = kg_facts
            .iter()
            .filter_map(|f| HehrFact::from_raw(f, &vocab))
            .collect();
        let split = (kg_idx.len() as f64 * 0.8) as usize;

        let result = train::<TrainB>(
            &TrainConfig {
                epochs: 20,
                lr: 0.005,
                margin: 1.0,
                batch_size: 32,
                negatives_per_positive: 5,
                hidden_dim: 32,
                num_layers: 2,
                dropout: 0.1,
                eval_every: 20,
                scorer_type: "distmult".into(),
                output_dir: "/tmp/hehrgnn_ensemble".into(),
            },
            &kg_idx[..split],
            &kg_idx[split..],
            vocab.num_entities(),
            vocab.num_relations(),
            &device_t,
        );

        let hehrgnn_model = result.model;
        let infer_device = hehrgnn_model
            .embeddings
            .entity_embedding
            .weight
            .val()
            .device();
        let scorer = DistMultScorer::new();
        let batcher = HehrBatcher::new();

        // Score the user→merchant facts (normal = high score, anomalous = low score)
        // We invert: low plausibility → high anomaly score
        let mut hehrgnn_raw_scores: Vec<f64> = Vec::new();
        for tx in &txs {
            let raw = RawFact {
                head: tx.user.clone(),
                relation: "transacts_at".into(),
                tail: tx.merchant.clone(),
                qualifiers: vec![],
            };
            if let Some(fact) = HehrFact::from_raw(&raw, &vocab) {
                let item = HehrFactItem { fact, label: 1.0 };
                let batch: HehrBatch<InferB> = batcher.batch(vec![item], &infer_device);
                let score = hehrgnn_model.score_batch(&batch, &scorer);
                let val: f64 = score.into_data().as_slice::<f32>().unwrap()[0] as f64;
                // INVERT: high plausibility = low anomaly, so negate
                hehrgnn_raw_scores.push(-val);
            } else {
                // Entity not in vocab → definitely anomalous
                hehrgnn_raw_scores.push(100.0);
            }
        }
        println!(
            "    ✅ HEHRGNN: {} scores computed\n",
            hehrgnn_raw_scores.len()
        );

        // ═══════════════════════════════════════════════
        // SIGNAL 4: Amount Z-Score
        // ═══════════════════════════════════════════════
        println!("  ── SIGNAL 4: Amount Z-Score ──\n");

        let merchant_stats: HashMap<&str, (f64, f64)> = [
            ("mcdonalds", (15.0, 8.0)),
            ("starbucks", (6.0, 3.0)),
            ("amazon", (50.0, 40.0)),
            ("grocery_store", (80.0, 30.0)),
            ("gas_station", (45.0, 15.0)),
            ("crypto_exchange", (0.0, 1.0)), // unknown → extreme z-score
        ]
        .into_iter()
        .collect();

        let mut zscore_scores: Vec<f64> = Vec::new();
        for tx in &txs {
            let (mean, std) = merchant_stats
                .get(tx.merchant.as_str())
                .copied()
                .unwrap_or((50.0, 50.0));
            let z = ((tx.amount - mean) / std.max(1.0)).abs();
            zscore_scores.push(z);
        }
        println!("    ✅ Z-scores: {} computed\n", zscore_scores.len());

        // ═══════════════════════════════════════════════
        // ENSEMBLE: Normalize + Weight + Combine
        // ═══════════════════════════════════════════════
        println!("  ═══════════════════════════════════════════════════════════════");
        println!("   🎯 SCORE NORMALIZATION + WEIGHTED ENSEMBLE");
        println!("  ═══════════════════════════════════════════════════════════════\n");

        let sage_norm = min_max_normalize(&sage_scores);
        let rgcn_norm = min_max_normalize(&rgcn_scores);
        let hehrgnn_norm = min_max_normalize(&hehrgnn_raw_scores);
        let zscore_norm = min_max_normalize(&zscore_scores);

        // Weights (tunable)
        let w_sage = 0.20;
        let w_rgcn = 0.20;
        let w_hehrgnn = 0.30;
        let w_zscore = 0.30;

        println!(
            "  Weights: GraphSAGE={:.0}%, RGCN={:.0}%, HEHRGNN={:.0}%, Z-Score={:.0}%\n",
            w_sage * 100.0,
            w_rgcn * 100.0,
            w_hehrgnn * 100.0,
            w_zscore * 100.0
        );

        let mut ensemble_scores: Vec<f64> = Vec::new();
        for i in 0..txs.len() {
            let combined = w_sage * sage_norm[i]
                + w_rgcn * rgcn_norm[i]
                + w_hehrgnn * hehrgnn_norm[i]
                + w_zscore * zscore_norm[i];
            ensemble_scores.push(combined);
        }

        // ═══════════════════════════════════════════════
        // RESULTS TABLE
        // ═══════════════════════════════════════════════
        println!("  ── DETAILED SCORES (showing anomalies + 5 sample normals) ──\n");
        println!(
            "  {:>6} │ {:>6} │ {:>6} │ {:>6} │ {:>6} │ {:>8} │ {}",
            "SAGE", "RGCN", "HEHRGNN", "Z-Score", "ENSEM", "Anomaly?", "Description"
        );
        println!(
            "  ──────┼────────┼────────┼────────┼────────┼──────────┼──────────────────────────"
        );

        // Show all anomalies
        for (i, tx) in txs.iter().enumerate() {
            if tx.is_anomaly {
                println!(
                    "  {:>6.3} │ {:>6.3} │ {:>7.3} │ {:>7.3} │ {:>6.3} │ {:>8} │ {}",
                    sage_norm[i],
                    rgcn_norm[i],
                    hehrgnn_norm[i],
                    zscore_norm[i],
                    ensemble_scores[i],
                    "🚨 YES",
                    tx.label
                );
            }
        }
        println!(
            "  ------┼--------┼--------┼--------┼--------┼----------┼--------------------------"
        );
        // Show 5 sample normals
        let mut shown = 0;
        for (i, tx) in txs.iter().enumerate() {
            if !tx.is_anomaly && shown < 5 {
                println!(
                    "  {:>6.3} │ {:>6.3} │ {:>7.3} │ {:>7.3} │ {:>6.3} │ {:>8} │ {}",
                    sage_norm[i],
                    rgcn_norm[i],
                    hehrgnn_norm[i],
                    zscore_norm[i],
                    ensemble_scores[i],
                    "✅ no",
                    tx.label
                );
                shown += 1;
            }
        }

        // ── Model comparison: which threshold catches all anomalies? ──
        println!("\n  ── MODEL COMPARISON: DETECTION AT VARIOUS THRESHOLDS ──\n");

        let anomaly_idx: Vec<usize> = txs
            .iter()
            .enumerate()
            .filter(|(_, t)| t.is_anomaly)
            .map(|(i, _)| i)
            .collect();
        let normal_idx: Vec<usize> = txs
            .iter()
            .enumerate()
            .filter(|(_, t)| !t.is_anomaly)
            .map(|(i, _)| i)
            .collect();

        for (name, scores) in &[
            ("GraphSAGE", &sage_norm),
            ("RGCN", &rgcn_norm),
            ("HEHRGNN", &hehrgnn_norm),
            ("Z-Score", &zscore_norm),
            ("✨ ENSEMBLE", &min_max_normalize(&ensemble_scores)),
        ] {
            // Find threshold that catches all anomalies
            let min_anomaly = anomaly_idx
                .iter()
                .map(|&i| scores[i])
                .fold(f64::INFINITY, f64::min);
            let max_normal = normal_idx
                .iter()
                .map(|&i| scores[i])
                .fold(f64::NEG_INFINITY, f64::max);

            // Count true positives and false positives at threshold = 0.5
            let threshold = 0.5;
            let tp = anomaly_idx
                .iter()
                .filter(|&&i| scores[i] >= threshold)
                .count();
            let fp = normal_idx
                .iter()
                .filter(|&&i| scores[i] >= threshold)
                .count();
            let fn_ = anomaly_idx.len() - tp;

            let precision = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                0.0
            };
            let recall = tp as f64 / anomaly_idx.len() as f64;
            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };

            let separable = min_anomaly > max_normal;

            println!("    {:>12} │ TP: {}/{} │ FP: {:>2} │ Prec: {:.2} │ Recall: {:.2} │ F1: {:.2} │ Sep: {}",
                name, tp, anomaly_idx.len(), fp, precision, recall, f1,
                if separable { "✅" } else { "❌" });
        }

        // ── Per-anomaly: which model catches which? ──
        println!("\n  ── PER-ANOMALY DETECTION MATRIX ──\n");
        println!("  {:>45} │ SAGE │ RGCN │ HEHR │ ZSCR │ ENSM", "Anomaly");
        println!(
            "  ─────────────────────────────────────────────┼──────┼──────┼──────┼──────┼──────"
        );

        let ens_norm = min_max_normalize(&ensemble_scores);
        for &i in &anomaly_idx {
            let sage_catch = if sage_norm[i] > 0.5 { "✅" } else { "❌" };
            let rgcn_catch = if rgcn_norm[i] > 0.5 { "✅" } else { "❌" };
            let hehr_catch = if hehrgnn_norm[i] > 0.5 { "✅" } else { "❌" };
            let z_catch = if zscore_norm[i] > 0.5 { "✅" } else { "❌" };
            let ens_catch = if ens_norm[i] > 0.5 { "✅" } else { "❌" };

            println!(
                "  {:>45} │  {}  │  {}  │  {}  │  {}  │  {}",
                &txs[i].label[..txs[i].label.len().min(45)],
                sage_catch,
                rgcn_catch,
                hehr_catch,
                z_catch,
                ens_catch
            );
        }

        // ── Summary ──
        let ens_tp = anomaly_idx.iter().filter(|&&i| ens_norm[i] > 0.5).count();
        let ens_fp = normal_idx.iter().filter(|&&i| ens_norm[i] > 0.5).count();

        println!("\n  ── ENSEMBLE RESULT ──\n");
        println!("    True Positives:   {}/{}", ens_tp, anomaly_idx.len());
        println!("    False Positives:  {}/{}", ens_fp, normal_idx.len());
        println!(
            "    Precision:        {:.2}",
            if ens_tp + ens_fp > 0 {
                ens_tp as f64 / (ens_tp + ens_fp) as f64
            } else {
                0.0
            }
        );
        println!(
            "    Recall:           {:.2}",
            ens_tp as f64 / anomaly_idx.len() as f64
        );

        println!("\n  ═══════════════════════════════════════════════════════════════\n");

        assert_eq!(txs.len(), sage_scores.len());
        assert_eq!(txs.len(), rgcn_scores.len());
        assert_eq!(txs.len(), hehrgnn_raw_scores.len());
        assert_eq!(txs.len(), ensemble_scores.len());
    }
}
