//! Grokking-inspired tuning test.
//!
//! Compares training results across different hyperparameter settings:
//! - Weight decay: 0.0, 0.005, 0.01, 0.02
//! - Patience: 5, 10, 20
//! - Epochs: 20, 40, 80
//!
//! Measures: final AUC, final loss, w_norm², emb_norm, epochs actually trained.

use burn::backend::NdArray;
use burn::prelude::*;

use hehrgnn::data::graph_builder::{GraphBuildConfig, GraphFact, build_hetero_graph};
use hehrgnn::data::hetero_graph::EdgeType;
use hehrgnn::model::graphsage::GraphSageModelConfig;
use hehrgnn::model::trainer::*;

type B = NdArray;

/// Build a richer graph to make tuning differences visible.
fn build_tuning_graph() -> hehrgnn::data::hetero_graph::HeteroGraph<B> {
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

#[test]
fn test_weight_decay_sweep() {
    let graph = build_tuning_graph();
    let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
    let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();
    let device = <B as Backend>::Device::default();

    println!("\n  ╔═════════════════════════════════════════════════════════════════╗");
    println!("  ║  WEIGHT DECAY SWEEP                                           ║");
    println!("  ╠═════════════════════════════════════════════════════════════════╣");
    println!("  ║  wd     │ epochs │ AUC    │ Loss   │ w_norm² │ emb_norm │ early║");
    println!("  ╠═════════════════════════════════════════════════════════════════╣");

    for wd in [0.0, 0.005, 0.01, 0.02, 0.05] {
        let mut model = GraphSageModelConfig {
            in_dim: 16,
            hidden_dim: 16,
            num_layers: 2,
            dropout: 0.0,
        }
        .init::<B>(&node_types, &edge_types, &device);

        let config = TrainConfig {
            epochs: 40,
            lr: 0.01,
            neg_ratio: 3,
            patience: 20, // generous patience to see full effect
            perturb_frac: 0.3,
            mode: TrainMode::Fast,
            weight_decay: wd,
            decor_weight: 0.1,
        };

        let report = train_graphsage(&mut model, &graph, &config);
        println!(
            "  ║  {:.3} │   {:3}  │ {:.4} │ {:.4} │ {:7.2} │ {:.4}   │ {:5}║",
            wd,
            report.epochs_trained,
            report.final_auc,
            report.final_loss,
            report.weight_norm_sq,
            report.mean_emb_norm,
            report.early_stopped
        );
    }

    println!("  ╚═════════════════════════════════════════════════════════════════╝");
}

#[test]
fn test_patience_sweep() {
    let graph = build_tuning_graph();
    let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
    let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();
    let device = <B as Backend>::Device::default();

    println!("\n  ╔═════════════════════════════════════════════════════════════════╗");
    println!("  ║  PATIENCE SWEEP (epochs=80, wd=0.01)                          ║");
    println!("  ╠═════════════════════════════════════════════════════════════════╣");
    println!("  ║  pat │ epochs │ AUC    │ Loss   │ w_norm² │ emb_norm │ early  ║");
    println!("  ╠═════════════════════════════════════════════════════════════════╣");

    for patience in [3, 5, 10, 15, 20] {
        let mut model = GraphSageModelConfig {
            in_dim: 16,
            hidden_dim: 16,
            num_layers: 2,
            dropout: 0.0,
        }
        .init::<B>(&node_types, &edge_types, &device);

        let config = TrainConfig {
            epochs: 80,
            lr: 0.01,
            neg_ratio: 3,
            patience,
            perturb_frac: 0.3,
            mode: TrainMode::Fast,
            weight_decay: 0.01,
            decor_weight: 0.1,
        };

        let report = train_graphsage(&mut model, &graph, &config);
        println!(
            "  ║  {:3} │   {:3}  │ {:.4} │ {:.4} │ {:7.2} │ {:.4}   │ {:5}  ║",
            patience,
            report.epochs_trained,
            report.final_auc,
            report.final_loss,
            report.weight_norm_sq,
            report.mean_emb_norm,
            report.early_stopped
        );
    }

    println!("  ╚═════════════════════════════════════════════════════════════════╝");
}

#[test]
fn test_epoch_scaling() {
    let graph = build_tuning_graph();
    let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
    let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();
    let device = <B as Backend>::Device::default();

    println!("\n  ╔═══════════════════════════════════════════════════════════════════╗");
    println!("  ║  EPOCH SCALING (patience=∞ i.e. no early stop, wd=0.01)         ║");
    println!("  ╠═══════════════════════════════════════════════════════════════════╣");
    println!("  ║  epochs │ AUC    │ Loss   │ w_norm² │ emb_norm │ Δloss vs 20    ║");
    println!("  ╠═══════════════════════════════════════════════════════════════════╣");

    let mut baseline_loss = 0.0f32;
    for max_epochs in [20, 40, 80, 120] {
        let mut model = GraphSageModelConfig {
            in_dim: 16,
            hidden_dim: 16,
            num_layers: 2,
            dropout: 0.0,
        }
        .init::<B>(&node_types, &edge_types, &device);

        let config = TrainConfig {
            epochs: max_epochs,
            lr: 0.01,
            neg_ratio: 3,
            patience: 999, // no early stopping
            perturb_frac: 0.3,
            mode: TrainMode::Fast,
            weight_decay: 0.01,
            decor_weight: 0.1,
        };

        let report = train_graphsage(&mut model, &graph, &config);
        if max_epochs == 20 {
            baseline_loss = report.final_loss;
        }
        let delta = if baseline_loss > 0.0 {
            ((report.final_loss - baseline_loss) / baseline_loss * 100.0)
        } else {
            0.0
        };

        println!(
            "  ║    {:3}  │ {:.4} │ {:.4} │ {:7.2} │ {:.4}   │ {:+6.1}%         ║",
            max_epochs,
            report.final_auc,
            report.final_loss,
            report.weight_norm_sq,
            report.mean_emb_norm,
            delta
        );
    }

    println!("  ╚═══════════════════════════════════════════════════════════════════╝");
    println!("\n  Key: Negative Δloss = better. Lower w_norm² = more cleanup.");
}

#[test]
fn test_lr_vs_weight_decay() {
    let graph = build_tuning_graph();
    let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
    let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();
    let device = <B as Backend>::Device::default();

    println!("\n  ╔═══════════════════════════════════════════════════════════════════╗");
    println!("  ║  LR × WEIGHT_DECAY GRID (epochs=40, patience=20)                ║");
    println!("  ╠═══════════════════════════════════════════════════════════════════╣");
    println!("  ║  lr     │ wd    │ AUC    │ Loss   │ w_norm² │ emb_norm          ║");
    println!("  ╠═══════════════════════════════════════════════════════════════════╣");

    for lr in [0.005, 0.01, 0.05] {
        for wd in [0.0, 0.01, 0.02] {
            let mut model = GraphSageModelConfig {
                in_dim: 16,
                hidden_dim: 16,
                num_layers: 2,
                dropout: 0.0,
            }
            .init::<B>(&node_types, &edge_types, &device);

            let config = TrainConfig {
                epochs: 40,
                lr,
                neg_ratio: 3,
                patience: 20,
                perturb_frac: 0.3,
                mode: TrainMode::Fast,
                weight_decay: wd,
            decor_weight: 0.1,
            };

            let report = train_graphsage(&mut model, &graph, &config);
            println!(
                "  ║  {:.3} │ {:.2}  │ {:.4} │ {:.4} │ {:7.2} │ {:.4}             ║",
                lr,
                wd,
                report.final_auc,
                report.final_loss,
                report.weight_norm_sq,
                report.mean_emb_norm
            );
        }
    }

    println!("  ╚═══════════════════════════════════════════════════════════════════╝");
}
