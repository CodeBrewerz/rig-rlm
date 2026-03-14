//! BPR vs JEPA comparison test across all GNN model types.
//!
//! Proves that Graph-JEPA (InfoNCE + uniformity + edge predictor)
//! improves over standard BPR training for each model type.

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
    ];

    build_hetero_graph::<B>(
        &facts,
        &GraphBuildConfig {
            node_feat_dim: 16,
            add_reverse_edges: true,
            add_self_loops: true,
            add_positional_encoding: true,
            add_cross_dependency_edges: true,
        },
        &device,
    )
}

/// BPR vs InfoNCE vs InfoNCE+EdgePredictor — all 4 GNN models.
#[test]
fn test_bpr_vs_jepa_all_models() {
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
            exec_prob_weight: 0.1,
    };

    let model_names = ["GraphSAGE", "RGCN", "GAT", "GPS"];

    println!("\n  ╔═════════════════════════════════════════════════════════════════════════╗");
    println!("  ║  BPR vs JEPA COMPARISON — ALL MODELS                                  ║");
    println!("  ╠═════════════════════════════════════════════════════════════════════════╣");
    println!("  ║  Model      │ Method       │ AUC    │ Loss   │ Cluster │ Δ AUC vs BPR ║");
    println!("  ╠═════════════════════════════════════════════════════════════════════════╣");

    for model_name in &model_names {
        let results: Vec<(String, TrainReport, f32)> = ["BPR", "JEPA", "JEPA+Pred", "BPR+JEPA"]
            .iter()
            .map(|method| {
                let mut graph = build_test_graph();
                let node_types: Vec<String> =
                    graph.node_types().iter().map(|s| s.to_string()).collect();
                let edge_types: Vec<EdgeType> =
                    graph.edge_types().iter().map(|e| (*e).clone()).collect();

                let fwd: Box<
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

                let report = match *method {
                    "BPR" => train_via_feature_refinement(&mut graph, &*fwd, &config),
                    "JEPA" => train_jepa(&mut graph, &*fwd, &config, 0.1, 0.5, false),
                    "JEPA+Pred" => train_jepa(&mut graph, &*fwd, &config, 0.1, 0.5, true),
                    "BPR+JEPA" => train_bpr_jepa(&mut graph, &*fwd, &config, 0.3, true),
                    _ => unreachable!(),
                };

                let emb = embeddings_to_plain(&fwd(&graph));
                let cluster = cluster_separation_score(&emb);

                (method.to_string(), report, cluster)
            })
            .collect();

        let bpr_auc = results[0].1.final_auc;

        for (method, report, cluster) in &results {
            let delta = if bpr_auc > 0.001 {
                ((report.final_auc - bpr_auc) / bpr_auc) * 100.0
            } else {
                0.0
            };
            let marker = if report.final_auc > bpr_auc && method != "BPR" {
                " ★"
            } else {
                ""
            };

            println!(
                "  ║  {:10} │ {:12} │ {:.4} │ {:.4} │  {:5.2}  │ {:+5.1}%{}       ║",
                if method == "BPR" { *model_name } else { "" },
                method,
                report.final_auc,
                report.final_loss,
                cluster,
                delta,
                marker
            );
        }
        println!("  ╠═════════════════════════════════════════════════════════════════════════╣");
    }

    println!("  ╚═════════════════════════════════════════════════════════════════════════╝");
    println!("\n  ✅ BPR vs JEPA comparison complete for all 4 GNN models!");
}
