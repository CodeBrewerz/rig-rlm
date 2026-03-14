//! Comprehensive feature comparison test for all HeteroGraph GNN models.
//!
//! Tests 4 configurations × 4 models = 16 combinations:
//! - Baseline (plain model)
//! - +JEPA (InfoNCE + uniformity training)
//! - +DoRA (HeteroDoRA adapter)
//! - +mHC (multi-stream, 8 layers)

use burn::backend::NdArray;
use burn::prelude::*;

use hehrgnn::data::graph_builder::{GraphBuildConfig, GraphFact, build_hetero_graph};
use hehrgnn::data::hetero_graph::{EdgeType, HeteroGraph};
use hehrgnn::model::backbone::NodeEmbeddings;
use hehrgnn::model::gat::GatConfig;
use hehrgnn::model::graph_transformer::GraphTransformerConfig;
use hehrgnn::model::graphsage::GraphSageModelConfig;
use hehrgnn::model::lora::{LoraConfig, init_hetero_basis_adapter};
use hehrgnn::model::mhc::{MhcGatConfig, MhcGpsConfig, MhcGraphSageConfig, MhcRgcnConfig};
use hehrgnn::model::rgcn::RgcnConfig;
use hehrgnn::model::trainer::*;

type B = NdArray;

fn gf(ht: &str, h: &str, r: &str, tt: &str, t: &str) -> GraphFact {
    GraphFact {
        src: (ht.into(), h.into()),
        relation: r.into(),
        dst: (tt.into(), t.into()),
    }
}

fn build_test_graph() -> HeteroGraph<B> {
    let device = <B as Backend>::Device::default();
    let facts: Vec<GraphFact> = vec![
        gf("user", "alice", "owns", "account", "checking"),
        gf("user", "alice", "owns", "account", "savings"),
        gf("user", "bob", "owns", "account", "credit"),
        gf("user", "carol", "owns", "account", "brokerage"),
        gf("account", "checking", "posted", "tx", "tx1"),
        gf("account", "checking", "posted", "tx", "tx2"),
        gf("account", "savings", "posted", "tx", "tx3"),
        gf("account", "credit", "posted", "tx", "tx4"),
        gf("tx", "tx1", "at", "merchant", "grocery"),
        gf("tx", "tx2", "at", "merchant", "gas"),
        gf("tx", "tx3", "at", "merchant", "grocery"),
        gf("tx", "tx4", "at", "merchant", "online"),
        gf("user", "alice", "linked", "user", "bob"),
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

fn test_config() -> TrainConfig {
    TrainConfig {
        lr: 0.01,
        epochs: 15,
        patience: 20,
        neg_ratio: 2,
        weight_decay: 0.001,
            decor_weight: 0.1,
            exec_prob_weight: 0.1,
        perturb_frac: 1.0,
        mode: TrainMode::Fast,
    }
}

/// Run training and return (AUC, embedding_variance).
fn train_and_measure(
    graph: &mut HeteroGraph<B>,
    fwd: &dyn Fn(&HeteroGraph<B>) -> NodeEmbeddings<B>,
    config: &TrainConfig,
    method: &str,
) -> (f32, f32) {
    let report = match method {
        "baseline" | "dora" | "mhc" => train_via_feature_refinement(graph, fwd, config),
        "jepa" => train_jepa(graph, fwd, config, 0.1, 0.5, false),
        _ => unreachable!(),
    };

    // Compute embedding variance (higher = less over-smoothing)
    let emb = embeddings_to_plain(&fwd(graph));
    let mut total_var = 0.0f32;
    let mut count = 0;
    for vecs in emb.values() {
        if vecs.len() < 2 {
            continue;
        }
        let dim = vecs[0].len();
        for d in 0..dim {
            let vals: Vec<f32> = vecs.iter().map(|v| v[d]).collect();
            let mean = vals.iter().sum::<f32>() / vals.len() as f32;
            let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
            total_var += var;
            count += 1;
        }
    }
    let avg_var = if count > 0 {
        total_var / count as f32
    } else {
        0.0
    };

    (report.final_auc, avg_var)
}

#[test]
fn test_all_features_all_models() {
    let device = <B as Backend>::Device::default();
    let config = test_config();

    let model_names = ["GraphSAGE", "RGCN", "GAT", "GPS"];
    let methods = ["baseline", "jepa", "dora", "mhc"];

    println!(
        "\n  ╔══════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!("  ║  ALL FEATURES × ALL MODELS COMPARISON                                         ║");
    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "  ║  Model      │ Config     │ AUC    │ EmbVar │ Layers │ Δ AUC vs Base            ║"
    );
    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════╣"
    );

    for model_name in &model_names {
        let mut baseline_auc = 0.0f32;

        for method in &methods {
            let mut graph = build_test_graph();
            let node_types: Vec<String> =
                graph.node_types().iter().map(|s| s.to_string()).collect();
            let edge_types: Vec<EdgeType> =
                graph.edge_types().iter().map(|e| (*e).clone()).collect();

            let (auc, emb_var, layers) = match (*model_name, *method) {
                // ── GraphSAGE ──
                ("GraphSAGE", "baseline") | ("GraphSAGE", "jepa") => {
                    let m = GraphSageModelConfig {
                        in_dim: 16,
                        hidden_dim: 16,
                        num_layers: 2,
                        dropout: 0.0,
                    }
                    .init::<B>(&node_types, &edge_types, &device);
                    let fwd = |g: &HeteroGraph<B>| m.forward(g);
                    let (a, v) = train_and_measure(&mut graph, &fwd, &config, method);
                    (a, v, 2)
                }
                ("GraphSAGE", "dora") => {
                    let mut m = GraphSageModelConfig {
                        in_dim: 16,
                        hidden_dim: 16,
                        num_layers: 2,
                        dropout: 0.0,
                    }
                    .init::<B>(&node_types, &edge_types, &device);
                    let adapter = init_hetero_basis_adapter(
                        16,
                        16,
                        &LoraConfig::default(),
                        node_types.clone(),
                        &device,
                    );
                    m.attach_adapter(adapter);
                    let fwd = |g: &HeteroGraph<B>| m.forward(g);
                    let (a, v) = train_and_measure(&mut graph, &fwd, &config, method);
                    (a, v, 2)
                }
                ("GraphSAGE", "mhc") => {
                    let m = MhcGraphSageConfig {
                        in_dim: 16,
                        hidden_dim: 16,
                        num_layers: 8,
                        n_streams: 4,
                        dropout: 0.0,
                    }
                    .init::<B>(&node_types, &edge_types, &device);
                    let fwd = |g: &HeteroGraph<B>| m.forward(g);
                    let (a, v) = train_and_measure(&mut graph, &fwd, &config, method);
                    (a, v, 8)
                }

                // ── RGCN ──
                ("RGCN", "baseline") | ("RGCN", "jepa") => {
                    let m = RgcnConfig {
                        in_dim: 16,
                        hidden_dim: 16,
                        num_layers: 2,
                        num_bases: 4,
                        dropout: 0.0,
                    }
                    .init_model::<B>(&node_types, &edge_types, &device);
                    let fwd = |g: &HeteroGraph<B>| m.forward(g);
                    let (a, v) = train_and_measure(&mut graph, &fwd, &config, method);
                    (a, v, 2)
                }
                ("RGCN", "dora") => {
                    let mut m = RgcnConfig {
                        in_dim: 16,
                        hidden_dim: 16,
                        num_layers: 2,
                        num_bases: 4,
                        dropout: 0.0,
                    }
                    .init_model::<B>(&node_types, &edge_types, &device);
                    let adapter = init_hetero_basis_adapter(
                        16,
                        16,
                        &LoraConfig::default(),
                        node_types.clone(),
                        &device,
                    );
                    m.attach_adapter(adapter);
                    let fwd = |g: &HeteroGraph<B>| m.forward(g);
                    let (a, v) = train_and_measure(&mut graph, &fwd, &config, method);
                    (a, v, 2)
                }
                ("RGCN", "mhc") => {
                    let m = MhcRgcnConfig {
                        in_dim: 16,
                        hidden_dim: 16,
                        num_layers: 8,
                        num_bases: 4,
                        n_streams: 4,
                        dropout: 0.0,
                    }
                    .init::<B>(&node_types, &edge_types, &device);
                    let fwd = |g: &HeteroGraph<B>| m.forward(g);
                    let (a, v) = train_and_measure(&mut graph, &fwd, &config, method);
                    (a, v, 8)
                }

                // ── GAT ──
                ("GAT", "baseline") | ("GAT", "jepa") => {
                    let m = GatConfig {
                        in_dim: 16,
                        hidden_dim: 16,
                        num_heads: 4,
                        num_layers: 2,
                        dropout: 0.0,
                    }
                    .init_model::<B>(&node_types, &edge_types, &device);
                    let fwd = |g: &HeteroGraph<B>| m.forward(g);
                    let (a, v) = train_and_measure(&mut graph, &fwd, &config, method);
                    (a, v, 2)
                }
                ("GAT", "dora") => {
                    let mut m = GatConfig {
                        in_dim: 16,
                        hidden_dim: 16,
                        num_heads: 4,
                        num_layers: 2,
                        dropout: 0.0,
                    }
                    .init_model::<B>(&node_types, &edge_types, &device);
                    let adapter = init_hetero_basis_adapter(
                        16,
                        16,
                        &LoraConfig::default(),
                        node_types.clone(),
                        &device,
                    );
                    m.attach_adapter(adapter);
                    let fwd = |g: &HeteroGraph<B>| m.forward(g);
                    let (a, v) = train_and_measure(&mut graph, &fwd, &config, method);
                    (a, v, 2)
                }
                ("GAT", "mhc") => {
                    let m = MhcGatConfig {
                        in_dim: 16,
                        hidden_dim: 16,
                        num_heads: 4,
                        num_layers: 8,
                        n_streams: 4,
                        dropout: 0.0,
                    }
                    .init::<B>(&node_types, &edge_types, &device);
                    let fwd = |g: &HeteroGraph<B>| m.forward(g);
                    let (a, v) = train_and_measure(&mut graph, &fwd, &config, method);
                    (a, v, 8)
                }

                // ── GPS ──
                ("GPS", "baseline") | ("GPS", "jepa") => {
                    let m = GraphTransformerConfig {
                        in_dim: 16,
                        hidden_dim: 16,
                        num_heads: 4,
                        num_layers: 2,
                        ffn_ratio: 2,
                        dropout: 0.0,
                    }
                    .init_model::<B>(&node_types, &edge_types, &device);
                    let fwd = |g: &HeteroGraph<B>| m.forward(g);
                    let (a, v) = train_and_measure(&mut graph, &fwd, &config, method);
                    (a, v, 2)
                }
                ("GPS", "dora") => {
                    let mut m = GraphTransformerConfig {
                        in_dim: 16,
                        hidden_dim: 16,
                        num_heads: 4,
                        num_layers: 2,
                        ffn_ratio: 2,
                        dropout: 0.0,
                    }
                    .init_model::<B>(&node_types, &edge_types, &device);
                    let adapter = init_hetero_basis_adapter(
                        16,
                        16,
                        &LoraConfig::default(),
                        node_types.clone(),
                        &device,
                    );
                    m.attach_adapter(adapter);
                    let fwd = |g: &HeteroGraph<B>| m.forward(g);
                    let (a, v) = train_and_measure(&mut graph, &fwd, &config, method);
                    (a, v, 2)
                }
                ("GPS", "mhc") => {
                    let m = MhcGpsConfig {
                        in_dim: 16,
                        hidden_dim: 16,
                        num_heads: 4,
                        num_layers: 8,
                        ffn_ratio: 2,
                        n_streams: 4,
                        dropout: 0.0,
                    }
                    .init::<B>(&node_types, &edge_types, &device);
                    let fwd = |g: &HeteroGraph<B>| m.forward(g);
                    let (a, v) = train_and_measure(&mut graph, &fwd, &config, method);
                    (a, v, 8)
                }

                _ => unreachable!(),
            };

            if *method == "baseline" {
                baseline_auc = auc;
            }
            let delta = if baseline_auc > 0.0 {
                ((auc - baseline_auc) / baseline_auc) * 100.0
            } else {
                0.0
            };
            let star = if delta > 0.5 { " ★" } else { "" };

            let pad_model = if *method == "baseline" {
                format!("  {:10}", model_name)
            } else {
                "            ".to_string()
            };
            println!(
                "  ║{}│ {:10} │ {:.4} │ {:.4} │   {:2}   │ {:+.1}%{}{}",
                pad_model,
                method,
                auc,
                emb_var,
                layers,
                delta,
                star,
                " ".repeat(20 - star.len() - format!("{:+.1}", delta).len())
            );
        }
        println!(
            "  ╠══════════════════════════════════════════════════════════════════════════════════╣"
        );
    }
    println!(
        "  ╚══════════════════════════════════════════════════════════════════════════════════╝"
    );
    println!("\n  ✅ All features × all models comparison complete!");
}
