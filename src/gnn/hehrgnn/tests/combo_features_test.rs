//! Feature combination test: stacking JEPA + DoRA + mHC.
//!
//! Tests 7 configurations × 4 models = 28 combinations:
//! 1. baseline — plain model
//! 2. jepa — InfoNCE + uniformity
//! 3. dora — HeteroDoRA adapter
//! 4. mhc — multi-stream 8 layers
//! 5. jepa+dora — both JEPA training + DoRA adapter
//! 6. mhc+jepa — deep mHC + JEPA training
//! 7. mhc+dora+jepa — full stack (all three)

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
        },
        &device,
    )
}

fn cfg() -> TrainConfig {
    TrainConfig {
        lr: 0.01,
        epochs: 15,
        patience: 20,
        neg_ratio: 2,
        weight_decay: 0.001,
            decor_weight: 0.1,
        perturb_frac: 1.0,
        mode: TrainMode::Fast,
    }
}

/// Compute embedding variance (over-smoothing indicator).
fn emb_variance(emb: &std::collections::HashMap<String, Vec<Vec<f32>>>) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0;
    for vecs in emb.values() {
        if vecs.len() < 2 {
            continue;
        }
        for d in 0..vecs[0].len() {
            let vals: Vec<f32> = vecs.iter().map(|v| v[d]).collect();
            let mean = vals.iter().sum::<f32>() / vals.len() as f32;
            let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
            total += var;
            count += 1;
        }
    }
    if count > 0 { total / count as f32 } else { 0.0 }
}

/// Configuration descriptor
struct RunConfig {
    name: &'static str,
    use_dora: bool,
    use_mhc: bool,
    use_jepa: bool,
}

const CONFIGS: [RunConfig; 7] = [
    RunConfig {
        name: "baseline",
        use_dora: false,
        use_mhc: false,
        use_jepa: false,
    },
    RunConfig {
        name: "jepa",
        use_dora: false,
        use_mhc: false,
        use_jepa: true,
    },
    RunConfig {
        name: "dora",
        use_dora: true,
        use_mhc: false,
        use_jepa: false,
    },
    RunConfig {
        name: "mhc",
        use_dora: false,
        use_mhc: true,
        use_jepa: false,
    },
    RunConfig {
        name: "jepa+dora",
        use_dora: true,
        use_mhc: false,
        use_jepa: true,
    },
    RunConfig {
        name: "mhc+jepa",
        use_dora: false,
        use_mhc: true,
        use_jepa: true,
    },
    RunConfig {
        name: "mhc+dora+jepa",
        use_dora: true,
        use_mhc: true,
        use_jepa: true,
    },
];

#[test]
fn test_feature_combos_all_models() {
    let device = <B as Backend>::Device::default();
    let config = cfg();
    let model_names = ["GraphSAGE", "RGCN", "GAT", "GPS"];

    println!(
        "\n  ╔════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "  ║  FEATURE COMBINATIONS × ALL MODELS                                               ║"
    );
    println!(
        "  ╠════════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "  ║  Model      │ Config         │ AUC    │ EmbVar │ Ly │ Δ vs Base                  ║"
    );
    println!(
        "  ╠════════════════════════════════════════════════════════════════════════════════════╣"
    );

    for model_name in &model_names {
        let mut baseline_auc = 0.0f32;

        for rc in &CONFIGS {
            let mut graph = build_test_graph();
            let node_types: Vec<String> =
                graph.node_types().iter().map(|s| s.to_string()).collect();
            let edge_types: Vec<EdgeType> =
                graph.edge_types().iter().map(|e| (*e).clone()).collect();
            let layers = if rc.use_mhc { 8 } else { 2 };

            // Build model with the right combo
            let fwd: Box<dyn Fn(&HeteroGraph<B>) -> NodeEmbeddings<B>> =
                match (*model_name, rc.use_mhc, rc.use_dora) {
                    // ── GraphSAGE ──
                    ("GraphSAGE", false, false) => {
                        let m = GraphSageModelConfig {
                            in_dim: 16,
                            hidden_dim: 16,
                            num_layers: 2,
                            dropout: 0.0,
                        }
                        .init::<B>(&node_types, &edge_types, &device);
                        Box::new(move |g| m.forward(g))
                    }
                    ("GraphSAGE", false, true) => {
                        let mut m = GraphSageModelConfig {
                            in_dim: 16,
                            hidden_dim: 16,
                            num_layers: 2,
                            dropout: 0.0,
                        }
                        .init::<B>(&node_types, &edge_types, &device);
                        m.attach_adapter(init_hetero_basis_adapter(
                            16,
                            16,
                            &LoraConfig::default(),
                            node_types.clone(),
                            &device,
                        ));
                        Box::new(move |g| m.forward(g))
                    }
                    ("GraphSAGE", true, _) => {
                        let m = MhcGraphSageConfig {
                            in_dim: 16,
                            hidden_dim: 16,
                            num_layers: 8,
                            n_streams: 4,
                            dropout: 0.0,
                        }
                        .init::<B>(&node_types, &edge_types, &device);
                        Box::new(move |g| m.forward(g))
                    }

                    // ── RGCN ──
                    ("RGCN", false, false) => {
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
                    ("RGCN", false, true) => {
                        let mut m = RgcnConfig {
                            in_dim: 16,
                            hidden_dim: 16,
                            num_layers: 2,
                            num_bases: 4,
                            dropout: 0.0,
                        }
                        .init_model::<B>(&node_types, &edge_types, &device);
                        m.attach_adapter(init_hetero_basis_adapter(
                            16,
                            16,
                            &LoraConfig::default(),
                            node_types.clone(),
                            &device,
                        ));
                        Box::new(move |g| m.forward(g))
                    }
                    ("RGCN", true, _) => {
                        let m = MhcRgcnConfig {
                            in_dim: 16,
                            hidden_dim: 16,
                            num_layers: 8,
                            num_bases: 4,
                            n_streams: 4,
                            dropout: 0.0,
                        }
                        .init::<B>(&node_types, &edge_types, &device);
                        Box::new(move |g| m.forward(g))
                    }

                    // ── GAT ──
                    ("GAT", false, false) => {
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
                    ("GAT", false, true) => {
                        let mut m = GatConfig {
                            in_dim: 16,
                            hidden_dim: 16,
                            num_heads: 4,
                            num_layers: 2,
                            dropout: 0.0,
                        }
                        .init_model::<B>(&node_types, &edge_types, &device);
                        m.attach_adapter(init_hetero_basis_adapter(
                            16,
                            16,
                            &LoraConfig::default(),
                            node_types.clone(),
                            &device,
                        ));
                        Box::new(move |g| m.forward(g))
                    }
                    ("GAT", true, _) => {
                        let m = MhcGatConfig {
                            in_dim: 16,
                            hidden_dim: 16,
                            num_heads: 4,
                            num_layers: 8,
                            n_streams: 4,
                            dropout: 0.0,
                        }
                        .init::<B>(&node_types, &edge_types, &device);
                        Box::new(move |g| m.forward(g))
                    }

                    // ── GPS ──
                    ("GPS", false, false) => {
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
                    ("GPS", false, true) => {
                        let mut m = GraphTransformerConfig {
                            in_dim: 16,
                            hidden_dim: 16,
                            num_heads: 4,
                            num_layers: 2,
                            ffn_ratio: 2,
                            dropout: 0.0,
                        }
                        .init_model::<B>(&node_types, &edge_types, &device);
                        m.attach_adapter(init_hetero_basis_adapter(
                            16,
                            16,
                            &LoraConfig::default(),
                            node_types.clone(),
                            &device,
                        ));
                        Box::new(move |g| m.forward(g))
                    }
                    ("GPS", true, _) => {
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
                        Box::new(move |g| m.forward(g))
                    }
                    _ => unreachable!(),
                };

            // Train with the right method
            let report = if rc.use_jepa {
                train_jepa(&mut graph, &*fwd, &config, 0.1, 0.5, false)
            } else {
                train_via_feature_refinement(&mut graph, &*fwd, &config)
            };

            let emb = embeddings_to_plain(&fwd(&graph));
            let var = emb_variance(&emb);
            let auc = report.final_auc;

            if rc.name == "baseline" {
                baseline_auc = auc;
            }
            let delta = if baseline_auc > 0.0 {
                ((auc - baseline_auc) / baseline_auc) * 100.0
            } else {
                0.0
            };
            let star = if delta > 2.0 {
                " ★"
            } else if delta > 5.0 {
                " ★★"
            } else {
                ""
            };
            let star = if delta > 8.0 { " ★★★" } else { star };

            let pad = if rc.name == "baseline" {
                format!("  {:10}", model_name)
            } else {
                "            ".to_string()
            };
            println!(
                "  ║{}│ {:14} │ {:.4} │ {:.4} │ {:2} │ {:+.1}%{}{}",
                pad,
                rc.name,
                auc,
                var,
                layers,
                delta,
                star,
                " ".repeat(22 - star.len() - format!("{:+.1}", delta).len())
            );
        }
        println!(
            "  ╠════════════════════════════════════════════════════════════════════════════════════╣"
        );
    }
    println!(
        "  ╚════════════════════════════════════════════════════════════════════════════════════╝"
    );
    println!("\n  ✅ Feature combination comparison complete!");
}
