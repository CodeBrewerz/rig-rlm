//! Per-model probe weight (α) sweep.
//!
//! Finds the optimal α for each GNN model type individually.
//! Each model gets its own sweep to identify where probe reward helps most.

use burn::backend::NdArray;
use burn::prelude::*;

use hehrgnn::data::graph_builder::{GraphBuildConfig, GraphFact, build_hetero_graph};
use hehrgnn::data::hetero_graph::EdgeType;
use hehrgnn::model::backbone::NodeEmbeddings;
use hehrgnn::model::gat::GatConfig;
use hehrgnn::model::graph_transformer::GraphTransformerConfig;
use hehrgnn::model::graphsage::GraphSageModelConfig;
use hehrgnn::model::probe::cluster_separation_score;
use hehrgnn::model::rgcn::RgcnConfig;
use hehrgnn::model::trainer::*;

type B = NdArray;

fn build_test_graph() -> hehrgnn::data::hetero_graph::HeteroGraph<B> {
    let device = <B as Backend>::Device::default();
    let facts = vec![
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
fn test_per_model_alpha_sweep() {
    let device = <B as Backend>::Device::default();
    let alphas = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5];

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

    let model_names = ["GraphSAGE", "RGCN", "GAT", "GPS"];

    for model_name in &model_names {
        println!("\n  ╔═══════════════════════════════════════════════════════════════════╗");
        println!(
            "  ║  {} α SWEEP                                          ║",
            model_name
        );
        println!("  ╠═══════════════════════════════════════════════════════════════════╣");
        println!("  ║   α    │ AUC    │ Loss   │ Probe% │ Cluster │ Δ vs α=0          ║");
        println!("  ╠═══════════════════════════════════════════════════════════════════╣");

        let mut baseline_auc = 0.0f32;

        for &alpha in &alphas {
            let mut graph = build_test_graph();
            let node_types: Vec<String> =
                graph.node_types().iter().map(|s| s.to_string()).collect();
            let edge_types: Vec<EdgeType> =
                graph.edge_types().iter().map(|e| (*e).clone()).collect();

            let (report, _probe_before, probe_after) = match *model_name {
                "GraphSAGE" => {
                    let model = GraphSageModelConfig {
                        in_dim: 16,
                        hidden_dim: 16,
                        num_layers: 2,
                        dropout: 0.0,
                    }
                    .init::<B>(&node_types, &edge_types, &device);
                    let fwd = move |g: &hehrgnn::data::hetero_graph::HeteroGraph<B>| -> NodeEmbeddings<B> {
                        model.forward(g)
                    };
                    train_features_with_probe(&mut graph, &fwd, &config, alpha)
                }
                "RGCN" => {
                    let model = RgcnConfig {
                        in_dim: 16,
                        hidden_dim: 16,
                        num_layers: 2,
                        num_bases: 4,
                        dropout: 0.0,
                    }
                    .init_model::<B>(&node_types, &edge_types, &device);
                    let fwd = move |g: &hehrgnn::data::hetero_graph::HeteroGraph<B>| -> NodeEmbeddings<B> {
                        model.forward(g)
                    };
                    train_features_with_probe(&mut graph, &fwd, &config, alpha)
                }
                "GAT" => {
                    let model = GatConfig {
                        in_dim: 16,
                        hidden_dim: 16,
                        num_heads: 4,
                        num_layers: 2,
                        dropout: 0.0,
                    }
                    .init_model::<B>(&node_types, &edge_types, &device);
                    let fwd = move |g: &hehrgnn::data::hetero_graph::HeteroGraph<B>| -> NodeEmbeddings<B> {
                        model.forward(g)
                    };
                    train_features_with_probe(&mut graph, &fwd, &config, alpha)
                }
                "GPS" => {
                    let model = GraphTransformerConfig {
                        in_dim: 16,
                        hidden_dim: 16,
                        num_heads: 4,
                        num_layers: 2,
                        ffn_ratio: 2,
                        dropout: 0.0,
                    }
                    .init_model::<B>(&node_types, &edge_types, &device);
                    let fwd = move |g: &hehrgnn::data::hetero_graph::HeteroGraph<B>| -> NodeEmbeddings<B> {
                        model.forward(g)
                    };
                    train_features_with_probe(&mut graph, &fwd, &config, alpha)
                }
                _ => unreachable!(),
            };

            // Compute cluster sep using a fresh model on refined features
            let cluster = {
                let fwd2: Box<
                    dyn Fn(&hehrgnn::data::hetero_graph::HeteroGraph<B>) -> NodeEmbeddings<B>,
                > = match *model_name {
                    "GraphSAGE" => {
                        let m = GraphSageModelConfig {
                            in_dim: 16,
                            hidden_dim: 16,
                            num_layers: 2,
                            dropout: 0.0,
                        }
                        .init::<B>(&node_types, &edge_types, &device);
                        Box::new(move |g| m.forward(g))
                    }
                    "RGCN" => {
                        let m = RgcnConfig {
                            in_dim: 16,
                            hidden_dim: 16,
                            num_layers: 2,
                            num_bases: 4,
                            dropout: 0.0,
                        }
                        .init_model::<B>(&node_types, &edge_types, &device);
                        Box::new(move |g| m.forward(g))
                    }
                    "GAT" => {
                        let m = GatConfig {
                            in_dim: 16,
                            hidden_dim: 16,
                            num_heads: 4,
                            num_layers: 2,
                            dropout: 0.0,
                        }
                        .init_model::<B>(&node_types, &edge_types, &device);
                        Box::new(move |g| m.forward(g))
                    }
                    "GPS" => {
                        let m = GraphTransformerConfig {
                            in_dim: 16,
                            hidden_dim: 16,
                            num_heads: 4,
                            num_layers: 2,
                            ffn_ratio: 2,
                            dropout: 0.0,
                        }
                        .init_model::<B>(&node_types, &edge_types, &device);
                        Box::new(move |g| m.forward(g))
                    }
                    _ => unreachable!(),
                };
                let emb = embeddings_to_plain(&fwd2(&graph));
                cluster_separation_score(&emb)
            };

            if alpha == 0.0 {
                baseline_auc = report.final_auc;
            }

            let delta = if baseline_auc > 0.001 {
                ((report.final_auc - baseline_auc) / baseline_auc) * 100.0
            } else {
                0.0
            };

            let marker = if report.final_auc > baseline_auc && alpha > 0.0 {
                " ★"
            } else {
                ""
            };

            println!(
                "  ║  {:.2}  │ {:.4} │ {:.4} │ {:5.1}% │  {:5.2}  │ {:+5.1}%{}          ║",
                alpha,
                report.final_auc,
                report.final_loss,
                probe_after * 100.0,
                cluster,
                delta,
                marker
            );
        }

        // Find best α
        println!("  ╚═══════════════════════════════════════════════════════════════════╝");
    }

    println!("\n  ✅ Per-model α sweep complete!");
}
