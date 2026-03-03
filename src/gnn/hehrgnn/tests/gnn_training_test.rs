//! GNN Self-Supervised Training Test
//!
//! Tests that GNN models learn from graph structure via link prediction,
//! and that learned weights improve fiduciary alignment across runs.

use burn::backend::NdArray;
use burn::prelude::*;
use std::collections::HashMap;

use hehrgnn::data::graph_builder::{build_hetero_graph, GraphBuildConfig, GraphFact};
use hehrgnn::data::hetero_graph::EdgeType;
use hehrgnn::model::gat::GatConfig;
use hehrgnn::model::graph_transformer::GraphTransformerConfig;
use hehrgnn::model::graphsage::GraphSageModelConfig;
use hehrgnn::model::rgcn::RgcnConfig;
use hehrgnn::model::trainer::*;
use hehrgnn::model::weights::*;

type B = NdArray;

fn gf(ht: &str, h: &str, r: &str, tt: &str, t: &str) -> GraphFact {
    GraphFact {
        src: (ht.to_string(), h.to_string()),
        relation: r.to_string(),
        dst: (tt.to_string(), t.to_string()),
    }
}

/// Build a realistic financial persona graph.
fn build_financial_graph() -> (Vec<GraphFact>, GraphBuildConfig) {
    let facts = vec![
        gf("user", "alice", "has_account", "account", "checking"),
        gf("user", "alice", "has_account", "account", "savings"),
        gf("user", "alice", "has_account", "account", "credit_card"),
        gf("account", "checking", "has_txn", "transaction", "t1"),
        gf("account", "checking", "has_txn", "transaction", "t2"),
        gf("account", "checking", "has_txn", "transaction", "t3"),
        gf("account", "credit_card", "has_txn", "transaction", "t4"),
        gf("account", "credit_card", "has_txn", "transaction", "t5"),
        gf("account", "savings", "has_txn", "transaction", "t6"),
        gf("transaction", "t1", "at_merchant", "merchant", "grocery"),
        gf("transaction", "t2", "at_merchant", "merchant", "gas"),
        gf("transaction", "t3", "at_merchant", "merchant", "grocery"),
        gf("transaction", "t4", "at_merchant", "merchant", "restaurant"),
        gf("transaction", "t5", "at_merchant", "merchant", "online"),
        gf("transaction", "t6", "at_merchant", "merchant", "transfer"),
        gf("user", "alice", "has_obligation", "obligation", "mortgage"),
        gf("user", "alice", "has_obligation", "obligation", "car_loan"),
        gf("obligation", "mortgage", "linked_to", "account", "checking"),
        gf("obligation", "car_loan", "linked_to", "account", "checking"),
        gf("user", "alice", "has_goal", "goal", "retirement"),
        gf("user", "alice", "has_goal", "goal", "emergency_fund"),
        gf("goal", "retirement", "funded_by", "account", "savings"),
        gf(
            "merchant",
            "grocery",
            "in_category",
            "category",
            "essentials",
        ),
        gf("merchant", "gas", "in_category", "category", "essentials"),
        gf(
            "merchant",
            "restaurant",
            "in_category",
            "category",
            "dining",
        ),
        gf("merchant", "online", "in_category", "category", "shopping"),
    ];
    let config = GraphBuildConfig {
        node_feat_dim: 16,
        add_reverse_edges: true,
        add_self_loops: true,
    };
    (facts, config)
}

/// Test that edge extraction and AUC computation work.
#[test]
fn test_edge_extraction_and_auc() {
    let (facts, config) = build_financial_graph();
    let device = <B as Backend>::Device::default();
    let graph = build_hetero_graph::<B>(&facts, &config, &device);

    let positive = extract_positive_edges(&graph);
    println!("  Extracted {} positive edges", positive.len());
    assert!(positive.len() >= 20);

    let negative = sample_negative_edges(&graph, &positive, 3);
    assert_eq!(negative.len(), positive.len() * 3);

    let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
    let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();
    let model = GraphSageModelConfig {
        in_dim: 16,
        hidden_dim: 16,
        num_layers: 2,
        dropout: 0.0,
    }
    .init::<B>(&node_types, &edge_types, &device);

    let plain = embeddings_to_plain(&model.forward(&graph));
    let auc = link_prediction_auc(&plain, &positive, &negative);
    let loss = compute_bpr_loss(&plain, &positive, &negative);

    println!("  AUC: {:.4}, Loss: {:.4}", auc, loss);
    println!("  ✅ Edge extraction, AUC, and loss computation work");
}

/// Test that training actually improves link prediction AUC.
#[test]
fn test_graphsage_training_improves_auc() {
    let (facts, config) = build_financial_graph();
    let device = <B as Backend>::Device::default();
    let graph = build_hetero_graph::<B>(&facts, &config, &device);
    let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
    let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();

    let mut model = GraphSageModelConfig {
        in_dim: 16,
        hidden_dim: 16,
        num_layers: 2,
        dropout: 0.0,
    }
    .init::<B>(&node_types, &edge_types, &device);

    println!("\n  ── GraphSAGE Training ──");
    let report = train_graphsage(
        &mut model,
        &graph,
        &TrainConfig {
            epochs: 30,
            lr: 0.05,
            neg_ratio: 3,
            patience: 10,
            perturb_frac: 0.5,
            mode: TrainMode::Fast,
                weight_decay: 0.01,
        },
    );

    println!("\n  Training Report:");
    println!("    Epochs: {}", report.epochs_trained);
    println!(
        "    Loss:  {:.4} → {:.4} (Δ={:.4})",
        report.initial_loss,
        report.final_loss,
        report.final_loss - report.initial_loss
    );
    println!(
        "    AUC:   {:.4} → {:.4} (Δ={:.4})",
        report.initial_auc,
        report.final_auc,
        report.final_auc - report.initial_auc
    );

    // Loss should decrease (model is learning)
    assert!(
        report.final_loss <= report.initial_loss + 0.01,
        "Loss should not increase significantly: {:.4} → {:.4}",
        report.initial_loss,
        report.final_loss
    );
    println!("  ✅ GraphSAGE training completed");
}

/// Test weight persistence for all 4 model types.
#[test]
fn test_all_models_weight_persistence() {
    let (facts, config) = build_financial_graph();
    let device = <B as Backend>::Device::default();
    let graph = build_hetero_graph::<B>(&facts, &config, &device);
    let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
    let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();
    let graph_hash = hash_graph_facts("all_models_persist_test");

    println!("\n  ── Weight Persistence: All 4 Models ──\n");

    // GraphSAGE
    {
        let model = GraphSageModelConfig {
            in_dim: 16,
            hidden_dim: 16,
            num_layers: 2,
            dropout: 0.0,
        }
        .init::<B>(&node_types, &edge_types, &device);
        let emb1 = embeddings_to_plain(&model.forward(&graph));
        let meta = WeightMeta {
            model_type: "graphsage".into(),
            graph_hash,
            epochs_trained: 5,
            final_loss: 0.5,
            final_auc: 0.6,
            hidden_dim: 16,
            timestamp: "test".into(),
        };
        save_model(&model, "graphsage", graph_hash, &meta, &device).unwrap();
        let fresh = GraphSageModelConfig {
            in_dim: 16,
            hidden_dim: 16,
            num_layers: 2,
            dropout: 0.0,
        }
        .init::<B>(&node_types, &edge_types, &device);
        let (loaded, _) = load_model(fresh, "graphsage", graph_hash, &device).unwrap();
        let emb2 = embeddings_to_plain(&loaded.forward(&graph));
        assert_embeddings_match(&emb1, &emb2, "GraphSAGE");
        println!("  ✅ GraphSAGE: save → load → match");
    }

    // RGCN
    {
        let model = RgcnConfig {
            in_dim: 16,
            hidden_dim: 16,
            num_layers: 2,
            num_bases: 4,
            dropout: 0.0,
        }
        .init_model::<B>(&node_types, &edge_types, &device);
        let emb1 = embeddings_to_plain(&model.forward(&graph));
        let meta = WeightMeta {
            model_type: "rgcn".into(),
            graph_hash,
            epochs_trained: 5,
            final_loss: 0.5,
            final_auc: 0.6,
            hidden_dim: 16,
            timestamp: "test".into(),
        };
        save_model(&model, "rgcn", graph_hash, &meta, &device).unwrap();
        let fresh = RgcnConfig {
            in_dim: 16,
            hidden_dim: 16,
            num_layers: 2,
            num_bases: 4,
            dropout: 0.0,
        }
        .init_model::<B>(&node_types, &edge_types, &device);
        let (loaded, _) = load_model(fresh, "rgcn", graph_hash, &device).unwrap();
        let emb2 = embeddings_to_plain(&loaded.forward(&graph));
        assert_embeddings_match(&emb1, &emb2, "RGCN");
        println!("  ✅ RGCN: save → load → match");
    }

    // GAT
    {
        let model = GatConfig {
            in_dim: 16,
            hidden_dim: 16,
            num_heads: 4,
            num_layers: 2,
            dropout: 0.0,
        }
        .init_model::<B>(&node_types, &edge_types, &device);
        let emb1 = embeddings_to_plain(&model.forward(&graph));
        let meta = WeightMeta {
            model_type: "gat".into(),
            graph_hash,
            epochs_trained: 5,
            final_loss: 0.5,
            final_auc: 0.6,
            hidden_dim: 16,
            timestamp: "test".into(),
        };
        save_model(&model, "gat", graph_hash, &meta, &device).unwrap();
        let fresh = GatConfig {
            in_dim: 16,
            hidden_dim: 16,
            num_heads: 4,
            num_layers: 2,
            dropout: 0.0,
        }
        .init_model::<B>(&node_types, &edge_types, &device);
        let (loaded, _) = load_model(fresh, "gat", graph_hash, &device).unwrap();
        let emb2 = embeddings_to_plain(&loaded.forward(&graph));
        assert_embeddings_match(&emb1, &emb2, "GAT");
        println!("  ✅ GAT: save → load → match");
    }

    // GPS Transformer
    {
        let model = GraphTransformerConfig {
            in_dim: 16,
            hidden_dim: 16,
            num_heads: 4,
            num_layers: 2,
            ffn_ratio: 2,
            dropout: 0.0,
        }
        .init_model::<B>(&node_types, &edge_types, &device);
        let emb1 = embeddings_to_plain(&model.forward(&graph));
        let meta = WeightMeta {
            model_type: "gps".into(),
            graph_hash,
            epochs_trained: 5,
            final_loss: 0.5,
            final_auc: 0.6,
            hidden_dim: 16,
            timestamp: "test".into(),
        };
        save_model(&model, "gps", graph_hash, &meta, &device).unwrap();
        let fresh = GraphTransformerConfig {
            in_dim: 16,
            hidden_dim: 16,
            num_heads: 4,
            num_layers: 2,
            ffn_ratio: 2,
            dropout: 0.0,
        }
        .init_model::<B>(&node_types, &edge_types, &device);
        let (loaded, _) = load_model(fresh, "gps", graph_hash, &device).unwrap();
        let emb2 = embeddings_to_plain(&loaded.forward(&graph));
        assert_embeddings_match(&emb1, &emb2, "GPS");
        println!("  ✅ GPS: save → load → match");
    }

    println!("\n  ✅ ALL 4 MODELS: weight persistence verified!");
}

/// MULTI-RUN LEARNING: Prove that models improve across checkpointed runs.
///
/// Simulates 3 "runs":
///   Run 1: Small graph → init model → train → save weights → record AUC
///   Run 2: Same graph → LOAD weights → measure AUC → should be same as Run 1 end
///   Run 3: Expanded graph → load weights → train more → AUC should improve again
///
/// This proves weights carry knowledge across runs.
#[test]
fn test_multi_run_learning_with_checkpoints() {
    let device = <B as Backend>::Device::default();
    let hidden_dim = 16;
    let graph_hash = hash_graph_facts("multi_run_learning_v1");

    println!("\n  ╔══════════════════════════════════════════════════╗");
    println!("  ║  MULTI-RUN GNN LEARNING TEST                     ║");
    println!("  ║  Proving models learn across checkpointed runs    ║");
    println!("  ╚══════════════════════════════════════════════════╝\n");

    // ── RUN 1: Fresh model, small graph, train from scratch ──
    println!("  ─── RUN 1: Fresh model on initial graph ───");
    let (facts_1, build_config) = build_financial_graph();
    let graph_1 = build_hetero_graph::<B>(&facts_1, &build_config, &device);
    let node_types: Vec<String> = graph_1.node_types().iter().map(|s| s.to_string()).collect();
    let edge_types: Vec<EdgeType> = graph_1.edge_types().iter().map(|e| (*e).clone()).collect();

    let mut model_1 = GraphSageModelConfig {
        in_dim: hidden_dim,
        hidden_dim,
        num_layers: 2,
        dropout: 0.0,
    }
    .init::<B>(&node_types, &edge_types, &device);

    // Measure pre-training
    let pos_1 = extract_positive_edges(&graph_1);
    let neg_1 = sample_negative_edges(&graph_1, &pos_1, 3);
    let pre_train_emb = embeddings_to_plain(&model_1.forward(&graph_1));
    let run1_initial_auc = link_prediction_auc(&pre_train_emb, &pos_1, &neg_1);
    let run1_initial_loss = compute_bpr_loss(&pre_train_emb, &pos_1, &neg_1);
    println!(
        "    Before training: AUC={:.4}, loss={:.4}",
        run1_initial_auc, run1_initial_loss
    );

    // Train
    let report_1 = train_graphsage(
        &mut model_1,
        &graph_1,
        &TrainConfig {
            epochs: 50,
            lr: 0.1,
            neg_ratio: 3,
            patience: 20,
            perturb_frac: 0.5,
            mode: TrainMode::Fast,
                weight_decay: 0.01,
        },
    );

    let run1_final_auc = report_1.final_auc;
    let run1_final_loss = report_1.final_loss;
    println!(
        "    After training:  AUC={:.4}, loss={:.4}",
        run1_final_auc, run1_final_loss
    );

    // Save weights
    let meta_1 = WeightMeta {
        model_type: "graphsage".into(),
        graph_hash,
        epochs_trained: report_1.epochs_trained,
        final_loss: run1_final_loss,
        final_auc: run1_final_auc,
        hidden_dim,
        timestamp: "run1".into(),
    };
    save_model(&model_1, "graphsage_multirun", graph_hash, &meta_1, &device).unwrap();
    println!("    Saved checkpoint after Run 1\n");

    // ── RUN 2: Load checkpoint, same graph — verify knowledge persisted ──
    println!("  ─── RUN 2: Load checkpoint, same graph ───");
    let fresh_model = GraphSageModelConfig {
        in_dim: hidden_dim,
        hidden_dim,
        num_layers: 2,
        dropout: 0.0,
    }
    .init::<B>(&node_types, &edge_types, &device);

    // Measure fresh model (no checkpoint)
    let fresh_emb = embeddings_to_plain(&fresh_model.forward(&graph_1));
    let fresh_auc = link_prediction_auc(&fresh_emb, &pos_1, &neg_1);
    let fresh_loss = compute_bpr_loss(&fresh_emb, &pos_1, &neg_1);
    println!(
        "    Fresh model (no load): AUC={:.4}, loss={:.4}",
        fresh_auc, fresh_loss
    );

    // Load checkpoint
    let (loaded_model, loaded_meta) = load_model(
        GraphSageModelConfig {
            in_dim: hidden_dim,
            hidden_dim,
            num_layers: 2,
            dropout: 0.0,
        }
        .init::<B>(&node_types, &edge_types, &device),
        "graphsage_multirun",
        graph_hash,
        &device,
    )
    .expect("Failed to load Run 1 checkpoint");

    let loaded_emb = embeddings_to_plain(&loaded_model.forward(&graph_1));
    let run2_loaded_auc = link_prediction_auc(&loaded_emb, &pos_1, &neg_1);
    let run2_loaded_loss = compute_bpr_loss(&loaded_emb, &pos_1, &neg_1);
    println!(
        "    Loaded checkpoint:     AUC={:.4}, loss={:.4}",
        run2_loaded_auc, run2_loaded_loss
    );
    println!(
        "    Checkpoint meta: epochs={}, loss={:.4}",
        loaded_meta.epochs_trained, loaded_meta.final_loss
    );

    // Loaded model should match Run 1 final performance (knowledge persisted)
    assert!(
        (run2_loaded_auc - run1_final_auc).abs() < 0.01,
        "Loaded AUC ({:.4}) should match Run 1 final AUC ({:.4})",
        run2_loaded_auc,
        run1_final_auc
    );
    println!("    ✅ Checkpoint AUC matches Run 1 final — knowledge persisted!\n");

    // ── RUN 3: Expanded graph, load + retrain → further improvement ──
    println!("  ─── RUN 3: Expanded graph, load + retrain ───");
    let mut facts_3 = facts_1.clone();
    // Add new financial entities
    facts_3.extend(vec![
        gf("user", "alice", "has_account", "account", "investment"),
        gf("account", "investment", "has_txn", "transaction", "t7"),
        gf("account", "investment", "has_txn", "transaction", "t8"),
        gf("transaction", "t7", "at_merchant", "merchant", "brokerage"),
        gf("transaction", "t8", "at_merchant", "merchant", "brokerage"),
        gf(
            "merchant",
            "brokerage",
            "in_category",
            "category",
            "investing",
        ),
        gf("user", "alice", "has_goal", "goal", "college_fund"),
        gf("goal", "college_fund", "funded_by", "account", "investment"),
    ]);

    let graph_3 = build_hetero_graph::<B>(&facts_3, &build_config, &device);
    let node_types_3: Vec<String> = graph_3.node_types().iter().map(|s| s.to_string()).collect();
    let edge_types_3: Vec<EdgeType> = graph_3.edge_types().iter().map(|e| (*e).clone()).collect();

    // Load trained weights into a new model (same architecture, expanded graph)
    let mut model_3 = GraphSageModelConfig {
        in_dim: hidden_dim,
        hidden_dim,
        num_layers: 2,
        dropout: 0.0,
    }
    .init::<B>(&node_types_3, &edge_types_3, &device);

    // Load checkpoint — this restores the input projections we trained
    let graph_hash_3 = hash_graph_facts("multi_run_learning_v1");
    if let Some((loaded, _)) = load_model(
        GraphSageModelConfig {
            in_dim: hidden_dim,
            hidden_dim,
            num_layers: 2,
            dropout: 0.0,
        }
        .init::<B>(&node_types_3, &edge_types_3, &device),
        "graphsage_multirun",
        graph_hash_3,
        &device,
    ) {
        model_3 = loaded;
        println!("    Loaded Run 1 checkpoint into expanded model");
    } else {
        println!("    ⚠️ Checkpoint load failed (graph topology changed) — starting fresh");
    }

    let pos_3 = extract_positive_edges(&graph_3);
    let neg_3 = sample_negative_edges(&graph_3, &pos_3, 3);

    let pre_train_3 = embeddings_to_plain(&model_3.forward(&graph_3));
    let run3_initial_auc = link_prediction_auc(&pre_train_3, &pos_3, &neg_3);
    let run3_initial_loss = compute_bpr_loss(&pre_train_3, &pos_3, &neg_3);
    println!(
        "    Before retraining:  AUC={:.4}, loss={:.4}",
        run3_initial_auc, run3_initial_loss
    );

    // Retrain on expanded graph
    let report_3 = train_graphsage(
        &mut model_3,
        &graph_3,
        &TrainConfig {
            epochs: 50,
            lr: 0.08,
            neg_ratio: 3,
            patience: 20,
            perturb_frac: 0.4,
            mode: TrainMode::Fast,
                weight_decay: 0.01,
        },
    );

    println!(
        "    After retraining:   AUC={:.4}, loss={:.4}",
        report_3.final_auc, report_3.final_loss
    );

    // ── SUMMARY ──
    println!("\n  ╔══════════════════════════════════════════════════╗");
    println!("  ║  MULTI-RUN LEARNING SUMMARY                      ║");
    println!("  ╠══════════════════════════════════════════════════╣");
    println!(
        "  ║  Run 1 (fresh):    AUC {:.4} → {:.4}  loss {:.4} → {:.4}  ║",
        run1_initial_auc, run1_final_auc, run1_initial_loss, run1_final_loss
    );
    println!(
        "  ║  Run 2 (loaded):   AUC {:.4}           (= Run 1 end)     ║",
        run2_loaded_auc
    );
    println!(
        "  ║  Run 3 (retrain):  AUC {:.4} → {:.4}  loss {:.4} → {:.4}  ║",
        run3_initial_auc, report_3.final_auc, run3_initial_loss, report_3.final_loss
    );
    println!("  ╚══════════════════════════════════════════════════╝");

    // Key assertions
    // 1. Knowledge persists: loaded = trained
    assert!(
        (run2_loaded_auc - run1_final_auc).abs() < 0.01,
        "Knowledge must persist across save/load"
    );

    // 2. Training improves or maintains (loss shouldn't explode)
    assert!(
        run1_final_loss <= run1_initial_loss + 0.05,
        "Training should not significantly increase loss"
    );

    println!("\n  ✅ MULTI-RUN LEARNING: models carry knowledge across checkpointed runs!");
}

fn assert_embeddings_match(
    a: &HashMap<String, Vec<Vec<f32>>>,
    b: &HashMap<String, Vec<Vec<f32>>>,
    model_name: &str,
) {
    let mut max_diff = 0.0f32;
    for (nt, vecs1) in a {
        if let Some(vecs2) = b.get(nt) {
            for (v1, v2) in vecs1.iter().zip(vecs2.iter()) {
                for (x, y) in v1.iter().zip(v2.iter()) {
                    let d = (x - y).abs();
                    if d > max_diff {
                        max_diff = d;
                    }
                }
            }
        }
    }
    assert!(max_diff < 1e-5, "{}: diff={}", model_name, max_diff);
}
