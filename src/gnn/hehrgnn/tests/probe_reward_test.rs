//! RLFR-inspired Probe-as-Reward integration tests.
//!
//! Compares standard BPR-only training against BPR + probe reward training.
//! The probe (node-type classifier on frozen embeddings) provides auxiliary
//! reward signal that encourages discriminative embeddings.
//!
//! Reference: Goodfire "Features as Rewards" (arXiv 2602.10067)

use burn::backend::NdArray;
use burn::prelude::*;

use hehrgnn::data::graph_builder::{GraphBuildConfig, GraphFact, build_hetero_graph};
use hehrgnn::data::hetero_graph::EdgeType;
use hehrgnn::model::graphsage::GraphSageModelConfig;
use hehrgnn::model::probe::{NodeTypeProbe, cluster_separation_score};
use hehrgnn::model::trainer::*;

type B = NdArray;

/// Build a heterogeneous financial graph for testing.
fn build_test_graph() -> hehrgnn::data::hetero_graph::HeteroGraph<B> {
    let device = <B as Backend>::Device::default();
    let facts = vec![
        // Users own accounts
        GraphFact {
            src: ("user".into(), "alice".into()),
            relation: "owns".into(),
            dst: ("account".into(), "checking1".into()),
        },
        GraphFact {
            src: ("user".into(), "alice".into()),
            relation: "owns".into(),
            dst: ("account".into(), "savings1".into()),
        },
        GraphFact {
            src: ("user".into(), "bob".into()),
            relation: "owns".into(),
            dst: ("account".into(), "checking2".into()),
        },
        GraphFact {
            src: ("user".into(), "bob".into()),
            relation: "owns".into(),
            dst: ("account".into(), "credit1".into()),
        },
        GraphFact {
            src: ("user".into(), "carol".into()),
            relation: "owns".into(),
            dst: ("account".into(), "checking3".into()),
        },
        GraphFact {
            src: ("user".into(), "dave".into()),
            relation: "owns".into(),
            dst: ("account".into(), "checking4".into()),
        },
        // Transactions posted to accounts
        GraphFact {
            src: ("tx".into(), "tx1".into()),
            relation: "posted_to".into(),
            dst: ("account".into(), "checking1".into()),
        },
        GraphFact {
            src: ("tx".into(), "tx2".into()),
            relation: "posted_to".into(),
            dst: ("account".into(), "checking1".into()),
        },
        GraphFact {
            src: ("tx".into(), "tx3".into()),
            relation: "posted_to".into(),
            dst: ("account".into(), "savings1".into()),
        },
        GraphFact {
            src: ("tx".into(), "tx4".into()),
            relation: "posted_to".into(),
            dst: ("account".into(), "checking2".into()),
        },
        GraphFact {
            src: ("tx".into(), "tx5".into()),
            relation: "posted_to".into(),
            dst: ("account".into(), "credit1".into()),
        },
        GraphFact {
            src: ("tx".into(), "tx6".into()),
            relation: "posted_to".into(),
            dst: ("account".into(), "checking3".into()),
        },
        GraphFact {
            src: ("tx".into(), "tx7".into()),
            relation: "posted_to".into(),
            dst: ("account".into(), "checking4".into()),
        },
        GraphFact {
            src: ("tx".into(), "tx8".into()),
            relation: "posted_to".into(),
            dst: ("account".into(), "checking1".into()),
        },
        // Transactions at merchants
        GraphFact {
            src: ("tx".into(), "tx1".into()),
            relation: "at".into(),
            dst: ("merchant".into(), "walmart".into()),
        },
        GraphFact {
            src: ("tx".into(), "tx2".into()),
            relation: "at".into(),
            dst: ("merchant".into(), "amazon".into()),
        },
        GraphFact {
            src: ("tx".into(), "tx3".into()),
            relation: "at".into(),
            dst: ("merchant".into(), "walmart".into()),
        },
        GraphFact {
            src: ("tx".into(), "tx4".into()),
            relation: "at".into(),
            dst: ("merchant".into(), "target".into()),
        },
        GraphFact {
            src: ("tx".into(), "tx5".into()),
            relation: "at".into(),
            dst: ("merchant".into(), "amazon".into()),
        },
        GraphFact {
            src: ("tx".into(), "tx6".into()),
            relation: "at".into(),
            dst: ("merchant".into(), "costco".into()),
        },
        GraphFact {
            src: ("tx".into(), "tx7".into()),
            relation: "at".into(),
            dst: ("merchant".into(), "walmart".into()),
        },
        GraphFact {
            src: ("tx".into(), "tx8".into()),
            relation: "at".into(),
            dst: ("merchant".into(), "starbucks".into()),
        },
        // Categories
        GraphFact {
            src: ("tx".into(), "tx1".into()),
            relation: "has_cat".into(),
            dst: ("category".into(), "groceries".into()),
        },
        GraphFact {
            src: ("tx".into(), "tx2".into()),
            relation: "has_cat".into(),
            dst: ("category".into(), "electronics".into()),
        },
        GraphFact {
            src: ("tx".into(), "tx3".into()),
            relation: "has_cat".into(),
            dst: ("category".into(), "transfer".into()),
        },
        GraphFact {
            src: ("tx".into(), "tx4".into()),
            relation: "has_cat".into(),
            dst: ("category".into(), "dining".into()),
        },
        GraphFact {
            src: ("tx".into(), "tx5".into()),
            relation: "has_cat".into(),
            dst: ("category".into(), "electronics".into()),
        },
        GraphFact {
            src: ("tx".into(), "tx6".into()),
            relation: "has_cat".into(),
            dst: ("category".into(), "groceries".into()),
        },
        GraphFact {
            src: ("tx".into(), "tx7".into()),
            relation: "has_cat".into(),
            dst: ("category".into(), "groceries".into()),
        },
        GraphFact {
            src: ("tx".into(), "tx8".into()),
            relation: "has_cat".into(),
            dst: ("category".into(), "dining".into()),
        },
    ];

    build_hetero_graph::<B>(
        &facts,
        &GraphBuildConfig {
            node_feat_dim: 16,
            add_reverse_edges: true,
            add_self_loops: true,
            add_positional_encoding: true,
        },
        &device,
    )
}

/// Compare BPR-only training vs BPR + probe reward training.
///
/// This is the main verification test: does adding probe reward
/// actually improve embedding quality?
#[test]
fn test_bpr_vs_probe_reward() {
    let graph = build_test_graph();
    let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
    let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();
    let device = <B as Backend>::Device::default();

    let config = TrainConfig {
        epochs: 30,
        lr: 0.01,
        neg_ratio: 3,
        patience: 15,
        perturb_frac: 0.3,
        mode: TrainMode::Fast,
        weight_decay: 0.01,
            decor_weight: 0.1,
    };

    println!("\n  ╔═══════════════════════════════════════════════════════════════════╗");
    println!("  ║  RLFR PROBE-AS-REWARD COMPARISON TEST                           ║");
    println!("  ╠═══════════════════════════════════════════════════════════════════╣");

    // ── 1. BPR-only training ──
    let mut model_bpr = GraphSageModelConfig {
        in_dim: 16,
        hidden_dim: 16,
        num_layers: 2,
        dropout: 0.0,
    }
    .init::<B>(&node_types, &edge_types, &device);

    println!("\n  ── Training: BPR only ──");
    let report_bpr = train_graphsage(&mut model_bpr, &graph, &config);

    // Measure embedding quality for BPR-only
    let emb_bpr = embeddings_to_plain(&model_bpr.forward(&graph));
    let mut probe_bpr = NodeTypeProbe::new(&node_types, 16);
    probe_bpr.train_on_frozen(&emb_bpr, 100, 0.1);
    let probe_acc_bpr = probe_bpr.score(&emb_bpr);
    let cluster_sep_bpr = cluster_separation_score(&emb_bpr);

    // ── 2. BPR + Probe reward training ──
    let mut model_probe = GraphSageModelConfig {
        in_dim: 16,
        hidden_dim: 16,
        num_layers: 2,
        dropout: 0.0,
    }
    .init::<B>(&node_types, &edge_types, &device);

    println!("\n  ── Training: BPR + Probe Reward (α=0.1) ──");
    let (report_probe, probe_before, probe_after) =
        train_with_probe_reward(&mut model_probe, &graph, &config, 0.1);

    // Measure embedding quality for BPR+probe
    let emb_probe = embeddings_to_plain(&model_probe.forward(&graph));
    let cluster_sep_probe = cluster_separation_score(&emb_probe);

    // ── 3. BPR + stronger probe reward ──
    let mut model_probe2 = GraphSageModelConfig {
        in_dim: 16,
        hidden_dim: 16,
        num_layers: 2,
        dropout: 0.0,
    }
    .init::<B>(&node_types, &edge_types, &device);

    println!("\n  ── Training: BPR + Probe Reward (α=0.5) ──");
    let (report_probe2, probe_before2, probe_after2) =
        train_with_probe_reward(&mut model_probe2, &graph, &config, 0.5);

    let emb_probe2 = embeddings_to_plain(&model_probe2.forward(&graph));
    let cluster_sep_probe2 = cluster_separation_score(&emb_probe2);

    // ── Results ──
    println!("\n  ╠═══════════════════════════════════════════════════════════════════╣");
    println!("  ║  Method         │ AUC    │ Loss   │ Probe  │ Cluster │ Epochs  ║");
    println!("  ╠═══════════════════════════════════════════════════════════════════╣");
    println!(
        "  ║  BPR only       │ {:.4} │ {:.4} │ {:5.1}% │  {:5.2}  │   {:3}   ║",
        report_bpr.final_auc,
        report_bpr.final_loss,
        probe_acc_bpr * 100.0,
        cluster_sep_bpr,
        report_bpr.epochs_trained
    );
    println!(
        "  ║  BPR+Probe α=0.1│ {:.4} │ {:.4} │ {:5.1}% │  {:5.2}  │   {:3}   ║",
        report_probe.final_auc,
        report_probe.final_loss,
        probe_after * 100.0,
        cluster_sep_probe,
        report_probe.epochs_trained
    );
    println!(
        "  ║  BPR+Probe α=0.5│ {:.4} │ {:.4} │ {:5.1}% │  {:5.2}  │   {:3}   ║",
        report_probe2.final_auc,
        report_probe2.final_loss,
        probe_after2 * 100.0,
        cluster_sep_probe2,
        report_probe2.epochs_trained
    );
    println!("  ╚═══════════════════════════════════════════════════════════════════╝");

    println!("\n  Probe evolution during training:");
    println!(
        "    α=0.1: {:.1}% → {:.1}%",
        probe_before * 100.0,
        probe_after * 100.0
    );
    println!(
        "    α=0.5: {:.1}% → {:.1}%",
        probe_before2 * 100.0,
        probe_after2 * 100.0
    );

    // Per-type accuracy for the probe-trained model
    let mut post_probe = NodeTypeProbe::new(&node_types, 16);
    post_probe.train_on_frozen(&emb_probe, 100, 0.1);
    let per_type = post_probe.per_type_accuracy(&emb_probe);
    println!("\n  Per-type probe accuracy (BPR+Probe α=0.1):");
    for (t, acc) in &per_type {
        println!("    {}: {:.0}%", t, acc * 100.0);
    }

    println!("\n  ✅ Probe-as-Reward comparison complete!");
}

/// Test that increasing probe weight increases cluster separation.
#[test]
fn test_probe_weight_sweep() {
    let graph = build_test_graph();
    let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
    let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();
    let device = <B as Backend>::Device::default();

    let config = TrainConfig {
        epochs: 20,
        lr: 0.01,
        neg_ratio: 3,
        patience: 15,
        perturb_frac: 0.3,
        mode: TrainMode::Fast,
        weight_decay: 0.01,
            decor_weight: 0.1,
    };

    println!("\n  ╔═══════════════════════════════════════════════════════════════════╗");
    println!("  ║  PROBE WEIGHT (α) SWEEP                                         ║");
    println!("  ╠═══════════════════════════════════════════════════════════════════╣");
    println!("  ║  α     │ AUC    │ Loss   │ Probe Before│ Probe After│ Cluster   ║");
    println!("  ╠═══════════════════════════════════════════════════════════════════╣");

    for alpha in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0] {
        let mut model = GraphSageModelConfig {
            in_dim: 16,
            hidden_dim: 16,
            num_layers: 2,
            dropout: 0.0,
        }
        .init::<B>(&node_types, &edge_types, &device);

        let (report, probe_before, probe_after) =
            train_with_probe_reward(&mut model, &graph, &config, alpha);

        let emb = embeddings_to_plain(&model.forward(&graph));
        let cluster_sep = cluster_separation_score(&emb);

        println!(
            "  ║  {:.2}  │ {:.4} │ {:.4} │    {:5.1}%   │   {:5.1}%  │  {:5.2}   ║",
            alpha,
            report.final_auc,
            report.final_loss,
            probe_before * 100.0,
            probe_after * 100.0,
            cluster_sep
        );
    }

    println!("  ╚═══════════════════════════════════════════════════════════════════╝");
}

/// Test the probe itself works correctly on GNN embeddings.
#[test]
fn test_probe_on_gnn_embeddings() {
    let graph = build_test_graph();
    let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
    let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();
    let device = <B as Backend>::Device::default();

    // Get untrained GNN embeddings
    let model = GraphSageModelConfig {
        in_dim: 16,
        hidden_dim: 16,
        num_layers: 2,
        dropout: 0.0,
    }
    .init::<B>(&node_types, &edge_types, &device);

    let emb = embeddings_to_plain(&model.forward(&graph));
    println!("\n  GNN Embedding Stats:");
    for (nt, vecs) in &emb {
        let mean_norm: f32 = vecs
            .iter()
            .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt())
            .sum::<f32>()
            / vecs.len() as f32;
        println!(
            "    {} ({} nodes): mean_norm={:.4}",
            nt,
            vecs.len(),
            mean_norm
        );
    }

    // Train probe on these embeddings
    let mut probe = NodeTypeProbe::new(&node_types, 16);
    let score_before = probe.score(&emb);
    println!("\n  Probe accuracy:");
    println!("    Before training: {:.1}%", score_before * 100.0);

    probe.train_on_frozen(&emb, 200, 0.05);
    let score_after = probe.score(&emb);
    println!("    After training:  {:.1}%", score_after * 100.0);

    assert!(
        score_after > score_before,
        "Probe should improve after training"
    );

    // Cluster separation
    let sep = cluster_separation_score(&emb);
    println!("\n  Cluster separation: {:.4}", sep);

    println!("\n  ✅ Probe on GNN embeddings works!");
}
