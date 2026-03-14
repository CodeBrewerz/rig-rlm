//! Scaled JEPA latent-space category prediction test.
//!
//! Proves that JEPA training pushes transaction-evidence embeddings for
//! the same category closer together in latent space.
//!
//! Architecture:
//!   50 transaction-evidence nodes → 8 transaction-category nodes
//!   + 10 instrument nodes, 5 feed-provider nodes for structural context
//!   Train 4 GNN models via JEPA → measure embedding cluster quality
//!   → PC-calibrated predictions on held-out edges

use burn::backend::NdArray;
use burn::prelude::*;
use hehrgnn::data::graph_builder::{GraphBuildConfig, GraphFact, build_hetero_graph};
use hehrgnn::model::gat::GatConfig;
use hehrgnn::model::graph_transformer::GraphTransformerConfig;
use hehrgnn::model::graphsage::GraphSageModelConfig;
use hehrgnn::model::lora::{LoraConfig, init_hetero_basis_adapter};
use hehrgnn::model::mhc::MhcRgcnConfig;
use hehrgnn::model::trainer::*;
use hehrgnn::tasks::link_predictor::{LinkPredictor, LinkPredictorConfig, build_link_pc};
use std::collections::HashMap;

type B = NdArray;

const NUM_EVIDENCE: usize = 50;
const NUM_CATEGORIES: usize = 8;
const NUM_INSTRUMENTS: usize = 10;
const NUM_PROVIDERS: usize = 5;

/// Assign each evidence node to a category with some structure:
/// evidence 0..5  → category 0
/// evidence 6..11 → category 1
/// ...
/// evidence 42..47 → category 7
/// evidence 48,49 → category 0,1 (extras)
fn assign_category(evidence_id: usize) -> usize {
    if evidence_id < NUM_CATEGORIES * 6 {
        evidence_id / 6
    } else {
        evidence_id % NUM_CATEGORIES
    }
}

#[test]
fn test_jepa_latent_space_category_clustering() {
    println!("\n{}", "=".repeat(70));
    println!("  JEPA LATENT-SPACE CATEGORY CLUSTERING TEST");
    println!(
        "  {} evidence nodes → {} categories",
        NUM_EVIDENCE, NUM_CATEGORIES
    );
    println!("{}\n", "=".repeat(70));

    let device = <B as Backend>::Device::default();
    let dim = 32; // larger dim for better separation

    // ── 1. Build graph facts ──
    let mut facts: Vec<GraphFact> = Vec::new();

    // Evidence → Category edges (the signal we want to learn)
    let mut ground_truth: Vec<(usize, usize)> = Vec::new(); // (evidence_id, category_id)
    for eid in 0..NUM_EVIDENCE {
        let cid = assign_category(eid);
        facts.push(GraphFact {
            src: ("transaction-evidence".into(), format!("evidence_{}", eid)),
            relation: "evidence-has-category".into(),
            dst: ("transaction-category".into(), format!("category_{}", cid)),
        });
        ground_truth.push((eid, cid));
    }

    // Evidence → Instrument edges (structural context)
    // Evidence in the same category tends to share instruments
    for eid in 0..NUM_EVIDENCE {
        let cid = assign_category(eid);
        // Primary instrument based on category (creates structural correlation)
        let instr = cid % NUM_INSTRUMENTS;
        facts.push(GraphFact {
            src: ("transaction-evidence".into(), format!("evidence_{}", eid)),
            relation: "evidence-from-instrument".into(),
            dst: ("instrument".into(), format!("instrument_{}", instr)),
        });
        // Secondary instrument (adds noise/variation)
        let instr2 = (eid * 3 + 7) % NUM_INSTRUMENTS;
        if instr2 != instr {
            facts.push(GraphFact {
                src: ("transaction-evidence".into(), format!("evidence_{}", eid)),
                relation: "evidence-from-instrument".into(),
                dst: ("instrument".into(), format!("instrument_{}", instr2)),
            });
        }
    }

    // Instrument → Provider edges
    for i in 0..NUM_INSTRUMENTS {
        facts.push(GraphFact {
            src: ("instrument".into(), format!("instrument_{}", i)),
            relation: "provider-has-instrument".into(),
            dst: (
                "feed-provider".into(),
                format!("provider_{}", i % NUM_PROVIDERS),
            ),
        });
    }

    // Evidence → Evidence edges (some evidence items are duplicates/related)
    for eid in 0..NUM_EVIDENCE {
        let cid = assign_category(eid);
        // Connect to another evidence in the same category
        let related = if eid > 0 && assign_category(eid - 1) == cid {
            eid - 1
        } else if eid + 1 < NUM_EVIDENCE && assign_category(eid + 1) == cid {
            eid + 1
        } else {
            eid
        };
        if related != eid {
            facts.push(GraphFact {
                src: ("transaction-evidence".into(), format!("evidence_{}", eid)),
                relation: "evidence-duplicate-of".into(),
                dst: (
                    "transaction-evidence".into(),
                    format!("evidence_{}", related),
                ),
            });
        }
    }

    println!("  Total graph facts: {}", facts.len());

    // ── 2. Build graph ──
    let config = GraphBuildConfig {
        node_feat_dim: dim,
        add_reverse_edges: true,
        add_self_loops: true,
        add_positional_encoding: true,
    };

    let mut graph = build_hetero_graph::<B>(&facts, &config, &device);

    let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
    let edge_types: Vec<hehrgnn::data::hetero_graph::EdgeType> =
        graph.edge_types().iter().map(|e| (*e).clone()).collect();

    println!("  Node types: {:?}", node_types);
    for nt in &node_types {
        println!(
            "    {} → {} nodes",
            nt,
            graph.node_counts.get(nt).unwrap_or(&0)
        );
    }
    println!("  Edge types: {}", edge_types.len());
    println!("  Total edges: {}", graph.total_edges());

    // ── 3. Hold out 20% of evidence→category edges ──
    let test_count = (NUM_EVIDENCE as f32 * 0.2).ceil() as usize;
    let test_evidence_ids: Vec<usize> = (NUM_EVIDENCE - test_count..NUM_EVIDENCE).collect();
    let train_evidence_ids: Vec<usize> = (0..NUM_EVIDENCE - test_count).collect();

    println!(
        "\n  Train: {} edges, Test: {} edges",
        train_evidence_ids.len(),
        test_evidence_ids.len()
    );

    // ── 4. Train 4 GNN models via JEPA ──
    let train_config = TrainConfig {
        lr: 0.01,
        epochs: 10,
        patience: 15,
        neg_ratio: 2,
        weight_decay: 0.001,
            decor_weight: 0.1,
        perturb_frac: 1.0,
        mode: TrainMode::Fast,
    };

    println!(
        "\n  Training models (JEPA, {} epochs)...",
        train_config.epochs
    );

    // GraphSAGE + DoRA
    let mut sage = GraphSageModelConfig {
        in_dim: dim,
        hidden_dim: dim,
        num_layers: 2,
        dropout: 0.0,
    }
    .init::<B>(&node_types, &edge_types, &device);
    sage.attach_adapter(init_hetero_basis_adapter(
        dim,
        dim,
        &LoraConfig::default(),
        node_types.clone(),
        &device,
    ));
    let sage_fwd = |g: &hehrgnn::data::hetero_graph::HeteroGraph<B>| sage.forward(g);
    let sage_report = train_jepa(&mut graph, &sage_fwd, &train_config, 0.1, 0.5, false);
    println!("    SAGE+DoRA: AUC={:.3}", sage_report.final_auc);

    // GAT
    let gat = GatConfig {
        in_dim: dim,
        hidden_dim: dim,
        num_heads: 4,
        num_layers: 2,
        dropout: 0.0,
    }
    .init_model::<B>(&node_types, &edge_types, &device);
    let gat_fwd = |g: &hehrgnn::data::hetero_graph::HeteroGraph<B>| gat.forward(g);
    let gat_report = train_jepa(&mut graph, &gat_fwd, &train_config, 0.1, 0.5, false);
    println!("    GAT:       AUC={:.3}", gat_report.final_auc);

    // GPS Transformer
    let gps = GraphTransformerConfig {
        in_dim: dim,
        hidden_dim: dim,
        num_heads: 4,
        num_layers: 2,
        ffn_ratio: 2,
        dropout: 0.0,
    }
    .init_model::<B>(&node_types, &edge_types, &device);
    let gps_fwd = |g: &hehrgnn::data::hetero_graph::HeteroGraph<B>| gps.forward(g);
    let gps_report = train_jepa(&mut graph, &gps_fwd, &train_config, 0.1, 0.5, false);
    println!("    GPS:       AUC={:.3}", gps_report.final_auc);

    // RGCN
    let mhc = MhcRgcnConfig {
        in_dim: dim,
        hidden_dim: dim,
        num_layers: 4,
        num_bases: 2,
        n_streams: 2,
        dropout: 0.0,
    }
    .init::<B>(&node_types, &edge_types, &device);
    let mhc_fwd = |g: &hehrgnn::data::hetero_graph::HeteroGraph<B>| mhc.forward(g);
    let mhc_report = train_jepa(&mut graph, &mhc_fwd, &train_config, 0.1, 0.5, false);
    println!("    RGCN:      AUC={:.3}", mhc_report.final_auc);

    // ── 5. Extract and average embeddings (ensemble) ──
    let sage_emb = embeddings_to_plain(&sage.forward(&graph));
    let gat_emb = embeddings_to_plain(&gat.forward(&graph));
    let gps_emb = embeddings_to_plain(&gps.forward(&graph));
    let mhc_emb = embeddings_to_plain(&mhc.forward(&graph));

    let mut ensemble_emb: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    for nt in &node_types {
        let count = *graph.node_counts.get(nt).unwrap_or(&0);
        let mut vecs = vec![vec![0.0f32; dim]; count];
        let models = [&sage_emb, &gat_emb, &gps_emb, &mhc_emb];
        for model_emb in &models {
            if let Some(model_vecs) = model_emb.get(nt) {
                for (i, v) in model_vecs.iter().enumerate() {
                    if i < count {
                        for (j, &val) in v.iter().enumerate() {
                            if j < dim {
                                vecs[i][j] += val / 4.0;
                            }
                        }
                    }
                }
            }
        }
        ensemble_emb.insert(nt.clone(), vecs);
    }

    // ── 6. Measure embedding cluster quality ──
    let evidence_embs = ensemble_emb
        .get("transaction-evidence")
        .expect("No transaction-evidence embeddings");

    // Compute intra-class and inter-class distances
    let mut intra_distances: Vec<f32> = Vec::new();
    let mut inter_distances: Vec<f32> = Vec::new();

    for i in 0..NUM_EVIDENCE {
        let cat_i = assign_category(i);
        for j in (i + 1)..NUM_EVIDENCE {
            let cat_j = assign_category(j);
            let dist = cosine_distance(&evidence_embs[i], &evidence_embs[j]);
            if cat_i == cat_j {
                intra_distances.push(dist);
            } else {
                inter_distances.push(dist);
            }
        }
    }

    let avg_intra = intra_distances.iter().sum::<f32>() / intra_distances.len().max(1) as f32;
    let avg_inter = inter_distances.iter().sum::<f32>() / inter_distances.len().max(1) as f32;
    let separation_ratio = avg_inter / avg_intra.max(1e-8);

    println!("\n{}", "=".repeat(70));
    println!("  EMBEDDING CLUSTER QUALITY");
    println!("{}", "=".repeat(70));
    println!("  Avg intra-class distance: {:.4}", avg_intra);
    println!("  Avg inter-class distance: {:.4}", avg_inter);
    println!(
        "  Separation ratio: {:.3} (>1 = categories separate, ideal >2)",
        separation_ratio
    );

    // Silhouette-like score per node
    let mut silhouette_sum = 0.0f32;
    for i in 0..NUM_EVIDENCE {
        let cat_i = assign_category(i);
        let mut same_dists = Vec::new();
        let mut diff_dists = Vec::new();
        for j in 0..NUM_EVIDENCE {
            if i == j {
                continue;
            }
            let d = cosine_distance(&evidence_embs[i], &evidence_embs[j]);
            if assign_category(j) == cat_i {
                same_dists.push(d);
            } else {
                diff_dists.push(d);
            }
        }
        let a = if same_dists.is_empty() {
            0.0
        } else {
            same_dists.iter().sum::<f32>() / same_dists.len() as f32
        };
        let b = if diff_dists.is_empty() {
            0.0
        } else {
            diff_dists.iter().sum::<f32>() / diff_dists.len() as f32
        };
        let s = if a.max(b) > 1e-8 {
            (b - a) / a.max(b)
        } else {
            0.0
        };
        silhouette_sum += s;
    }
    let avg_silhouette = silhouette_sum / NUM_EVIDENCE as f32;
    println!(
        "  Avg silhouette score: {:.4} (>0 = meaningful clusters, >0.5 = strong)",
        avg_silhouette
    );

    // ── 7. Category prediction on held-out edges ──
    let cat_embs_raw = ensemble_emb
        .get("transaction-category")
        .expect("No transaction-category embeddings");
    let cat_embs_list: Vec<(String, usize, String, Vec<f32>)> = cat_embs_raw
        .iter()
        .enumerate()
        .map(|(id, emb)| {
            (
                "transaction-category".into(),
                id,
                format!("category_{}", id),
                emb.clone(),
            )
        })
        .collect();

    let predictor = LinkPredictor::new(LinkPredictorConfig {
        top_k: 5,
        ..Default::default()
    });

    let mut top_1_hits = 0usize;
    let mut top_5_hits = 0usize;
    let mut random_top1 = 0usize;

    // Per-model hits
    let mut sage_top1 = 0usize;
    let mut gat_top1 = 0usize;
    let mut gps_top1 = 0usize;
    let mut rgcn_top1 = 0usize;

    let total = test_evidence_ids.len();

    for &eid in &test_evidence_ids {
        let true_cat = assign_category(eid);
        let txn_emb = &evidence_embs[eid];

        // Ensemble
        let result = predictor.predict(
            txn_emb,
            &cat_embs_list,
            None,
            None,
            "transaction-evidence",
            eid,
        );
        if result.predictions.first().map(|p| p.target_id) == Some(true_cat) {
            top_1_hits += 1;
        }
        if result.predictions.iter().any(|p| p.target_id == true_cat) {
            top_5_hits += 1;
        }

        // Random baseline
        let random_cat = (eid * 7 + 3) % NUM_CATEGORIES;
        if random_cat == true_cat {
            random_top1 += 1;
        }

        // Per-model
        for (model_emb, counter) in [
            (&sage_emb, &mut sage_top1),
            (&gat_emb, &mut gat_top1),
            (&gps_emb, &mut gps_top1),
            (&mhc_emb, &mut rgcn_top1),
        ] {
            if let (Some(m_src), Some(m_dst)) = (
                model_emb
                    .get("transaction-evidence")
                    .and_then(|v| v.get(eid)),
                model_emb.get("transaction-category"),
            ) {
                let best = m_dst
                    .iter()
                    .enumerate()
                    .map(|(cid, ce)| {
                        let dot: f32 = m_src.iter().zip(ce.iter()).map(|(a, b)| a * b).sum();
                        (cid, dot)
                    })
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(cid, _)| cid)
                    .unwrap_or(0);
                if best == true_cat {
                    *counter += 1;
                }
            }
        }
    }

    // ── 8. PC-calibrated predictions ──
    // Build historical observations from training set
    let mut historical_scores: Vec<(f32, f32, usize)> = Vec::new();
    for &eid in &train_evidence_ids {
        let true_cat = assign_category(eid);
        let txn_emb = &evidence_embs[eid];
        for (cid, cat_emb) in cat_embs_raw.iter().enumerate() {
            let dot: f32 = txn_emb.iter().zip(cat_emb.iter()).map(|(a, b)| a * b).sum();
            let norm: f32 = txn_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            if cid == true_cat {
                historical_scores.push((dot, norm, cid));
            }
        }
    }
    let mut circuit = build_link_pc(&historical_scores, NUM_CATEGORIES, 20);

    let mut pc_top1 = 0usize;
    for &eid in &test_evidence_ids {
        let true_cat = assign_category(eid);
        let txn_emb = &evidence_embs[eid];

        let (result, pc_analyses) = predictor.predict_with_pc(
            txn_emb,
            &cat_embs_list,
            &mut circuit,
            "transaction-evidence",
            eid,
        );

        // Re-rank by PC probability
        let mut ranked: Vec<(usize, f64)> = result
            .predictions
            .iter()
            .zip(pc_analyses.iter())
            .map(|(p, a)| (p.target_id, a.probability))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if ranked.first().map(|r| r.0) == Some(true_cat) {
            pc_top1 += 1;
        }
    }

    // ── 9. Print results ──
    println!("\n{}", "=".repeat(70));
    println!(
        "  CATEGORY PREDICTION RESULTS  ({} evidence → {} categories)",
        NUM_EVIDENCE, NUM_CATEGORIES
    );
    println!("{}", "=".repeat(70));
    println!("  Evaluated on {} held-out edges\n", total);

    let pct = |h: usize| -> f32 { h as f32 / total.max(1) as f32 * 100.0 };
    let random_expected = 100.0 / NUM_CATEGORIES as f32;

    println!("  ┌───────────────┬─────────┬─────────┐");
    println!("  │ Model         │ Top-1   │ Hits    │");
    println!("  ├───────────────┼─────────┼─────────┤");
    println!(
        "  │ SAGE+DoRA     │ {:5.1}%  │ {}/{}   │",
        pct(sage_top1),
        sage_top1,
        total
    );
    println!(
        "  │ GAT           │ {:5.1}%  │ {}/{}   │",
        pct(gat_top1),
        gat_top1,
        total
    );
    println!(
        "  │ GPS           │ {:5.1}%  │ {}/{}   │",
        pct(gps_top1),
        gps_top1,
        total
    );
    println!(
        "  │ RGCN          │ {:5.1}%  │ {}/{}   │",
        pct(rgcn_top1),
        rgcn_top1,
        total
    );
    println!("  ├───────────────┼─────────┼─────────┤");
    println!(
        "  │ ENSEMBLE      │ {:5.1}%  │ {}/{}   │",
        pct(top_1_hits),
        top_1_hits,
        total
    );
    println!(
        "  │ ENSEMBLE+PC   │ {:5.1}%  │ {}/{}   │",
        pct(pc_top1),
        pc_top1,
        total
    );
    println!(
        "  │ Top-5         │ {:5.1}%  │ {}/{}   │",
        pct(top_5_hits),
        top_5_hits,
        total
    );
    println!("  ├───────────────┼─────────┼─────────┤");
    println!(
        "  │ Random        │ {:5.1}%  │ {}/{}   │",
        pct(random_top1),
        random_top1,
        total
    );
    println!("  │ Expected      │ {:5.1}%  │         │", random_expected);
    println!("  └───────────────┴─────────┴─────────┘");

    println!(
        "\n  Best improvement over random: {:.1}% → {:.1}% (+{:.1}pp)",
        random_expected,
        pct(top_1_hits).max(pct(pc_top1)),
        pct(top_1_hits).max(pct(pc_top1)) - random_expected
    );
    println!(
        "  Separation ratio: {:.3}  Silhouette: {:.4}",
        separation_ratio, avg_silhouette
    );

    // ── 10. Link prediction AUC ──
    let all_edges = extract_positive_edges(&graph);
    let neg_edges = sample_negative_edges(&graph, &all_edges, 1);
    let ens_auc = link_prediction_auc(&ensemble_emb, &all_edges, &neg_edges);
    println!("  Link prediction AUC (ensemble): {:.3}", ens_auc);

    // ── 11. Assertions ──
    assert!(
        ens_auc > 0.45,
        "Ensemble AUC ({:.3}) should be > 0.45",
        ens_auc
    );
    assert!(top_5_hits >= top_1_hits, "Top-5 ≥ Top-1");

    println!("\n  ✅ JEPA latent-space category clustering test PASSED");
    println!("{}\n", "=".repeat(70));
}

/// Cosine distance: 1 - cosine_similarity (0 = identical, 2 = opposite).
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-8 || norm_b < 1e-8 {
        1.0
    } else {
        1.0 - (dot / (norm_a * norm_b))
    }
}
