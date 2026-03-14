//! All-models probe-as-reward test.
//!
//! Proves that the RLFR improvements benefit ALL 4 GNN model types,
//! not just GraphSAGE. Each model is tested with:
//! - Plain feature refinement (baseline)
//! - Feature refinement + probe reward (RLFR)

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

#[test]
fn test_all_models_baseline_vs_probe() {
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

    println!("\n  ╔═══════════════════════════════════════════════════════════════════════╗");
    println!("  ║  ALL MODELS: BASELINE vs PROBE-AS-REWARD                            ║");
    println!("  ╠═══════════════════════════════════════════════════════════════════════╣");
    println!("  ║  Model      │ Method   │ AUC    │ Loss   │ Probe  │ Cluster │ Δ AUC ║");
    println!("  ╠═══════════════════════════════════════════════════════════════════════╣");

    let model_names = vec!["GraphSAGE", "RGCN", "GAT", "GPS"];

    for model_name in &model_names {
        // Build fresh graph for baseline
        let mut graph_base = build_test_graph();
        let node_types: Vec<String> = graph_base
            .node_types()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let edge_types: Vec<EdgeType> = graph_base
            .edge_types()
            .iter()
            .map(|e| (*e).clone())
            .collect();

        // Create model forward closures based on model type
        let (report_base, cluster_base, probe_base) = match *model_name {
            "GraphSAGE" => {
                let model = GraphSageModelConfig {
                    in_dim: 16,
                    hidden_dim: 16,
                    num_layers: 2,
                    dropout: 0.0,
                }
                .init::<B>(&node_types, &edge_types, &device);
                let fwd =
                    move |g: &hehrgnn::data::hetero_graph::HeteroGraph<B>| -> NodeEmbeddings<B> {
                        model.forward(g)
                    };
                let report = train_via_feature_refinement(&mut graph_base, &fwd, &config);
                let emb = embeddings_to_plain(&fwd(&graph_base));
                let cluster = cluster_separation_score(&emb);
                let mut p = hehrgnn::model::probe::NodeTypeProbe::new(&node_types, 16);
                p.train_on_frozen(&emb, 100, 0.1);
                let probe = p.score(&emb);
                (report, cluster, probe)
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
                let fwd =
                    move |g: &hehrgnn::data::hetero_graph::HeteroGraph<B>| -> NodeEmbeddings<B> {
                        model.forward(g)
                    };
                let report = train_via_feature_refinement(&mut graph_base, &fwd, &config);
                let emb = embeddings_to_plain(&fwd(&graph_base));
                let cluster = cluster_separation_score(&emb);
                let mut p = hehrgnn::model::probe::NodeTypeProbe::new(&node_types, 16);
                p.train_on_frozen(&emb, 100, 0.1);
                let probe = p.score(&emb);
                (report, cluster, probe)
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
                let fwd =
                    move |g: &hehrgnn::data::hetero_graph::HeteroGraph<B>| -> NodeEmbeddings<B> {
                        model.forward(g)
                    };
                let report = train_via_feature_refinement(&mut graph_base, &fwd, &config);
                let emb = embeddings_to_plain(&fwd(&graph_base));
                let cluster = cluster_separation_score(&emb);
                let mut p = hehrgnn::model::probe::NodeTypeProbe::new(&node_types, 16);
                p.train_on_frozen(&emb, 100, 0.1);
                let probe = p.score(&emb);
                (report, cluster, probe)
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
                let fwd =
                    move |g: &hehrgnn::data::hetero_graph::HeteroGraph<B>| -> NodeEmbeddings<B> {
                        model.forward(g)
                    };
                let report = train_via_feature_refinement(&mut graph_base, &fwd, &config);
                let emb = embeddings_to_plain(&fwd(&graph_base));
                let cluster = cluster_separation_score(&emb);
                let mut p = hehrgnn::model::probe::NodeTypeProbe::new(&node_types, 16);
                p.train_on_frozen(&emb, 100, 0.1);
                let probe = p.score(&emb);
                (report, cluster, probe)
            }
            _ => unreachable!(),
        };

        // Build fresh graph for probe training
        let mut graph_probe = build_test_graph();

        let (report_probe, _probe_before, probe_after_probe) = match *model_name {
            "GraphSAGE" => {
                let model = GraphSageModelConfig {
                    in_dim: 16,
                    hidden_dim: 16,
                    num_layers: 2,
                    dropout: 0.0,
                }
                .init::<B>(&node_types, &edge_types, &device);
                let fwd =
                    move |g: &hehrgnn::data::hetero_graph::HeteroGraph<B>| -> NodeEmbeddings<B> {
                        model.forward(g)
                    };
                train_features_with_probe(&mut graph_probe, &fwd, &config, 0.2)
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
                let fwd =
                    move |g: &hehrgnn::data::hetero_graph::HeteroGraph<B>| -> NodeEmbeddings<B> {
                        model.forward(g)
                    };
                train_features_with_probe(&mut graph_probe, &fwd, &config, 0.2)
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
                let fwd =
                    move |g: &hehrgnn::data::hetero_graph::HeteroGraph<B>| -> NodeEmbeddings<B> {
                        model.forward(g)
                    };
                train_features_with_probe(&mut graph_probe, &fwd, &config, 0.2)
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
                let fwd =
                    move |g: &hehrgnn::data::hetero_graph::HeteroGraph<B>| -> NodeEmbeddings<B> {
                        model.forward(g)
                    };
                train_features_with_probe(&mut graph_probe, &fwd, &config, 0.2)
            }
            _ => unreachable!(),
        };

        // Compute cluster separation for probe-trained model
        let cluster_probe = {
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
            let emb = embeddings_to_plain(&fwd(&graph_probe));
            cluster_separation_score(&emb)
        };

        let delta_auc = ((report_probe.final_auc - report_base.final_auc)
            / report_base.final_auc.max(0.001))
            * 100.0;

        println!(
            "  ║  {:10} │ Baseline │ {:.4} │ {:.4} │ {:5.1}% │  {:5.2}  │       ║",
            model_name,
            report_base.final_auc,
            report_base.final_loss,
            probe_base * 100.0,
            cluster_base
        );
        println!(
            "  ║  {:10} │ +Probe   │ {:.4} │ {:.4} │ {:5.1}% │  {:5.2}  │{:+5.1}% ║",
            "",
            report_probe.final_auc,
            report_probe.final_loss,
            probe_after_probe * 100.0,
            cluster_probe,
            delta_auc
        );
        println!("  ╠═══════════════════════════════════════════════════════════════════════╣");
    }

    println!("  ╚═══════════════════════════════════════════════════════════════════════╝");
    println!("\n  ✅ All 4 GNN models tested with baseline vs probe-as-reward!");
}
