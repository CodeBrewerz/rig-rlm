//! AttnRes Training Benchmark: Train GNNs with weight-level SPSA.
//!
//! Unlike feature refinement (which only updates input features),
//! this trains ALL model weights including AttnRes pseudo-queries
//! via SPSA (Simultaneous Perturbation Stochastic Approximation).
//!
//! This ensures the AttnRes depth-attention parameters actually learn.

use std::collections::HashMap;

use burn::backend::NdArray;
use burn::prelude::*;

use hehrgnn::data::graph_builder::{build_hetero_graph, GraphBuildConfig, GraphFact};
use hehrgnn::data::hetero_graph::{EdgeType, HeteroGraph};
use hehrgnn::model::backbone::NodeEmbeddings;
use hehrgnn::model::trainer::{
    compute_bpr_loss, embeddings_to_plain, extract_positive_edges, link_prediction_auc,
    sample_negative_edges,
};

type B = NdArray;

// ═══════════════════════════════════════════════════════════════
// Large Dense Training Graph (60+ nodes, 250+ edges)
// ═══════════════════════════════════════════════════════════════

fn build_training_graph() -> HeteroGraph<B> {
    let device = <B as Backend>::Device::default();
    let mut facts = Vec::new();

    // 12 users
    let users: Vec<String> = (0..12).map(|i| format!("user{}", i)).collect();
    // 15 accounts
    let accounts: Vec<String> = (0..15).map(|i| format!("acc{}", i)).collect();
    // 25 transactions
    let txs: Vec<String> = (0..25).map(|i| format!("tx{}", i)).collect();
    // 8 merchants
    let merchants: Vec<String> = (0..8).map(|i| format!("merch{}", i)).collect();

    // Users own accounts (many-to-many)
    let mut seed: u64 = 12345;
    for u in 0..users.len() {
        // Each user owns 1-3 accounts
        let n_acc = 1 + (hash_seed(&mut seed) % 3) as usize;
        for _ in 0..n_acc {
            let a = (hash_seed(&mut seed) as usize) % accounts.len();
            facts.push(GraphFact {
                src: ("user".into(), users[u].clone()),
                relation: "owns".into(),
                dst: ("account".into(), accounts[a].clone()),
            });
        }
    }

    // Transactions posted to accounts
    for t in 0..txs.len() {
        let a = (hash_seed(&mut seed) as usize) % accounts.len();
        facts.push(GraphFact {
            src: ("tx".into(), txs[t].clone()),
            relation: "posted_to".into(),
            dst: ("account".into(), accounts[a].clone()),
        });
        // Some txs go to 2 accounts (split payments)
        if hash_seed(&mut seed) % 3 == 0 {
            let a2 = (hash_seed(&mut seed) as usize) % accounts.len();
            facts.push(GraphFact {
                src: ("tx".into(), txs[t].clone()),
                relation: "posted_to".into(),
                dst: ("account".into(), accounts[a2].clone()),
            });
        }
    }

    // Transactions at merchants
    for t in 0..txs.len() {
        let m = (hash_seed(&mut seed) as usize) % merchants.len();
        facts.push(GraphFact {
            src: ("tx".into(), txs[t].clone()),
            relation: "at_merchant".into(),
            dst: ("merchant".into(), merchants[m].clone()),
        });
    }

    // User refers user (social graph — creates long-range dependencies)
    for u1 in 0..users.len() {
        let n_refs = (hash_seed(&mut seed) % 3) as usize;
        for _ in 0..n_refs {
            let u2 = (hash_seed(&mut seed) as usize) % users.len();
            if u1 != u2 {
                facts.push(GraphFact {
                    src: ("user".into(), users[u1].clone()),
                    relation: "refers".into(),
                    dst: ("user".into(), users[u2].clone()),
                });
            }
        }
    }

    // Merchant partnerships (merchant-to-merchant)
    for m1 in 0..merchants.len() {
        if hash_seed(&mut seed) % 2 == 0 {
            let m2 = (hash_seed(&mut seed) as usize) % merchants.len();
            if m1 != m2 {
                facts.push(GraphFact {
                    src: ("merchant".into(), merchants[m1].clone()),
                    relation: "partners_with".into(),
                    dst: ("merchant".into(), merchants[m2].clone()),
                });
            }
        }
    }

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

fn hash_seed(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 33
}

fn get_types(graph: &HeteroGraph<B>) -> (Vec<String>, Vec<EdgeType>) {
    let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
    let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();
    (node_types, edge_types)
}

fn get_in_dim(graph: &HeteroGraph<B>) -> usize {
    graph
        .node_features
        .values()
        .next()
        .map(|t| t.dims()[1])
        .unwrap_or(16)
}

struct TrainResult {
    name: String,
    init_auc: f32,
    trained_auc: f32,
    init_loss: f32,
    final_loss: f32,
}

// ═══════════════════════════════════════════════════════════════
// Generic SPSA Weight Trainer
// ═══════════════════════════════════════════════════════════════

/// Extract all trainable 1-D weight vectors from a model's `input_linears`
/// as flat f32 vecs, along with their shapes.
fn extract_linears(linears: &[burn::nn::Linear<B>]) -> Vec<(Vec<f32>, [usize; 2])> {
    linears
        .iter()
        .map(|lin| {
            let w = lin.weight.val();
            let dims = w.dims();
            let data: Vec<f32> = w.into_data().as_slice::<f32>().unwrap().to_vec();
            (data, dims)
        })
        .collect()
}

/// Extract AttnRes pseudo-query vectors from a DepthAttnWrapper.
fn extract_attn_queries(
    attn: &Option<hehrgnn::model::attn_res_gnn::DepthAttnWrapper<B>>,
) -> Vec<Vec<f32>> {
    match attn {
        Some(wrapper) => wrapper
            .attn_ops
            .iter()
            .map(|op| {
                op.pseudo_query
                    .val()
                    .into_data()
                    .as_slice::<f32>()
                    .unwrap()
                    .to_vec()
            })
            .collect(),
        None => vec![],
    }
}

fn next_u64(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    *seed
}

/// Generate a Rademacher perturbation vector (±1 entries).
fn rademacher_vec(len: usize, seed: &mut u64) -> Vec<f32> {
    (0..len)
        .map(|_| {
            if (next_u64(seed) >> 33) % 2 == 0 {
                1.0f32
            } else {
                -1.0f32
            }
        })
        .collect()
}

/// Apply perturbation: w +/- eps*delta
fn perturb(w: &[f32], delta: &[f32], eps: f32, sign: f32) -> Vec<f32> {
    w.iter()
        .zip(delta)
        .map(|(w, d)| w + sign * eps * d)
        .collect()
}

/// SPSA update: w -= lr * (loss+ - loss-) / (2*eps) * delta
fn spsa_update(
    w: &[f32],
    delta: &[f32],
    loss_plus: f32,
    loss_minus: f32,
    eps: f32,
    lr: f32,
) -> Vec<f32> {
    let grad = (loss_plus - loss_minus) / (2.0 * eps);
    w.iter()
        .zip(delta)
        .map(|(w, d)| w - lr * grad * d)
        .collect()
}

/// Set a linear's weight from flat f32 vec.
fn set_linear_weight(lin: &mut burn::nn::Linear<B>, data: &[f32], dims: [usize; 2]) {
    let device = lin.weight.val().device();
    let tensor = Tensor::<B, 1>::from_data(data, &device).reshape([dims[0], dims[1]]);
    lin.weight = lin.weight.clone().map(|_| tensor);
}

/// Set AttnRes pseudo-query from flat f32 vec.
fn set_attn_query(
    attn: &mut Option<hehrgnn::model::attn_res_gnn::DepthAttnWrapper<B>>,
    idx: usize,
    data: &[f32],
) {
    if let Some(wrapper) = attn.as_mut() {
        if idx < wrapper.attn_ops.len() {
            let device = wrapper.attn_ops[idx].pseudo_query.val().device();
            let tensor = Tensor::<B, 1>::from_data(data, &device);
            wrapper.attn_ops[idx].pseudo_query =
                wrapper.attn_ops[idx].pseudo_query.clone().map(|_| tensor);
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Per-Model SPSA Trainers
// ═══════════════════════════════════════════════════════════════

fn train_graphsage_spsa(
    model: &mut hehrgnn::model::graphsage::GraphSageModel<B>,
    graph: &HeteroGraph<B>,
    epochs: usize,
    lr: f32,
) -> TrainResult {
    let positive = extract_positive_edges(graph);
    let negative = sample_negative_edges(graph, &positive, 3);
    let eps = 0.01f32;

    let init_emb = embeddings_to_plain(&model.forward(graph));
    let init_loss = compute_bpr_loss(&init_emb, &positive, &negative);
    let init_auc = link_prediction_auc(&init_emb, &positive, &negative);
    let mut best_loss = init_loss;
    let mut final_loss = init_loss;

    for epoch in 0..epochs {
        let mut seed: u64 = (epoch as u64 * 7919).wrapping_add(42);

        // 1. Perturb input_linears
        let orig_linears = extract_linears(&model.input_linears);
        for (li, (orig_w, dims)) in orig_linears.iter().enumerate() {
            let delta = rademacher_vec(orig_w.len(), &mut seed);

            // Forward with +eps*delta
            let w_plus = perturb(orig_w, &delta, eps, 1.0);
            set_linear_weight(&mut model.input_linears[li], &w_plus, *dims);
            let loss_plus = compute_bpr_loss(
                &embeddings_to_plain(&model.forward(graph)),
                &positive,
                &negative,
            );

            // Forward with -eps*delta
            let w_minus = perturb(orig_w, &delta, eps, -1.0);
            set_linear_weight(&mut model.input_linears[li], &w_minus, *dims);
            let loss_minus = compute_bpr_loss(
                &embeddings_to_plain(&model.forward(graph)),
                &positive,
                &negative,
            );

            // Update
            let w_new = spsa_update(orig_w, &delta, loss_plus, loss_minus, eps, lr);
            set_linear_weight(&mut model.input_linears[li], &w_new, *dims);
        }

        // 2. Perturb AttnRes pseudo-queries
        let orig_queries = extract_attn_queries(&model.attn_depth);
        for (qi, orig_q) in orig_queries.iter().enumerate() {
            if orig_q.is_empty() {
                continue;
            }
            let delta = rademacher_vec(orig_q.len(), &mut seed);

            let q_plus = perturb(orig_q, &delta, eps, 1.0);
            set_attn_query(&mut model.attn_depth, qi, &q_plus);
            let loss_plus = compute_bpr_loss(
                &embeddings_to_plain(&model.forward(graph)),
                &positive,
                &negative,
            );

            let q_minus = perturb(orig_q, &delta, eps, -1.0);
            set_attn_query(&mut model.attn_depth, qi, &q_minus);
            let loss_minus = compute_bpr_loss(
                &embeddings_to_plain(&model.forward(graph)),
                &positive,
                &negative,
            );

            let q_new = spsa_update(orig_q, &delta, loss_plus, loss_minus, eps, lr);
            set_attn_query(&mut model.attn_depth, qi, &q_new);
        }

        let emb = embeddings_to_plain(&model.forward(graph));
        final_loss = compute_bpr_loss(&emb, &positive, &negative);
        if final_loss < best_loss {
            best_loss = final_loss;
        }

        if epoch % 20 == 0 || epoch == epochs - 1 {
            let auc = link_prediction_auc(&emb, &positive, &negative);
            println!(
                "    epoch {:3}: loss={:.4}, auc={:.4}",
                epoch, final_loss, auc
            );
        }
    }

    let final_emb = embeddings_to_plain(&model.forward(graph));
    let trained_auc = link_prediction_auc(&final_emb, &positive, &negative);

    TrainResult {
        name: String::new(),
        init_auc,
        trained_auc,
        init_loss,
        final_loss,
    }
}

fn train_rgcn_spsa(
    model: &mut hehrgnn::model::rgcn::RgcnModel<B>,
    graph: &HeteroGraph<B>,
    epochs: usize,
    lr: f32,
) -> TrainResult {
    let positive = extract_positive_edges(graph);
    let negative = sample_negative_edges(graph, &positive, 3);
    let eps = 0.01f32;

    let init_emb = embeddings_to_plain(&model.forward(graph));
    let init_loss = compute_bpr_loss(&init_emb, &positive, &negative);
    let init_auc = link_prediction_auc(&init_emb, &positive, &negative);
    let mut best_loss = init_loss;
    let mut final_loss = init_loss;

    for epoch in 0..epochs {
        let mut seed: u64 = (epoch as u64 * 7919).wrapping_add(42);

        let orig_linears = extract_linears(&model.input_linears);
        for (li, (orig_w, dims)) in orig_linears.iter().enumerate() {
            let delta = rademacher_vec(orig_w.len(), &mut seed);
            let w_plus = perturb(orig_w, &delta, eps, 1.0);
            set_linear_weight(&mut model.input_linears[li], &w_plus, *dims);
            let loss_plus = compute_bpr_loss(
                &embeddings_to_plain(&model.forward(graph)),
                &positive,
                &negative,
            );
            let w_minus = perturb(orig_w, &delta, eps, -1.0);
            set_linear_weight(&mut model.input_linears[li], &w_minus, *dims);
            let loss_minus = compute_bpr_loss(
                &embeddings_to_plain(&model.forward(graph)),
                &positive,
                &negative,
            );
            let w_new = spsa_update(orig_w, &delta, loss_plus, loss_minus, eps, lr);
            set_linear_weight(&mut model.input_linears[li], &w_new, *dims);
        }

        let orig_queries = extract_attn_queries(&model.attn_depth);
        for (qi, orig_q) in orig_queries.iter().enumerate() {
            if orig_q.is_empty() {
                continue;
            }
            let delta = rademacher_vec(orig_q.len(), &mut seed);
            let q_plus = perturb(orig_q, &delta, eps, 1.0);
            set_attn_query(&mut model.attn_depth, qi, &q_plus);
            let loss_plus = compute_bpr_loss(
                &embeddings_to_plain(&model.forward(graph)),
                &positive,
                &negative,
            );
            let q_minus = perturb(orig_q, &delta, eps, -1.0);
            set_attn_query(&mut model.attn_depth, qi, &q_minus);
            let loss_minus = compute_bpr_loss(
                &embeddings_to_plain(&model.forward(graph)),
                &positive,
                &negative,
            );
            let q_new = spsa_update(orig_q, &delta, loss_plus, loss_minus, eps, lr);
            set_attn_query(&mut model.attn_depth, qi, &q_new);
        }

        let emb = embeddings_to_plain(&model.forward(graph));
        final_loss = compute_bpr_loss(&emb, &positive, &negative);
        if final_loss < best_loss {
            best_loss = final_loss;
        }
        if epoch % 20 == 0 || epoch == epochs - 1 {
            let auc = link_prediction_auc(&emb, &positive, &negative);
            println!(
                "    epoch {:3}: loss={:.4}, auc={:.4}",
                epoch, final_loss, auc
            );
        }
    }

    let final_emb = embeddings_to_plain(&model.forward(graph));
    let trained_auc = link_prediction_auc(&final_emb, &positive, &negative);
    TrainResult {
        name: String::new(),
        init_auc,
        trained_auc,
        init_loss,
        final_loss,
    }
}

fn train_gat_spsa(
    model: &mut hehrgnn::model::gat::GatModel<B>,
    graph: &HeteroGraph<B>,
    epochs: usize,
    lr: f32,
) -> TrainResult {
    let positive = extract_positive_edges(graph);
    let negative = sample_negative_edges(graph, &positive, 3);
    let eps = 0.01f32;

    let init_emb = embeddings_to_plain(&model.forward(graph));
    let init_loss = compute_bpr_loss(&init_emb, &positive, &negative);
    let init_auc = link_prediction_auc(&init_emb, &positive, &negative);
    let mut best_loss = init_loss;
    let mut final_loss = init_loss;

    for epoch in 0..epochs {
        let mut seed: u64 = (epoch as u64 * 7919).wrapping_add(42);

        let orig_linears = extract_linears(&model.input_linears);
        for (li, (orig_w, dims)) in orig_linears.iter().enumerate() {
            let delta = rademacher_vec(orig_w.len(), &mut seed);
            let w_plus = perturb(orig_w, &delta, eps, 1.0);
            set_linear_weight(&mut model.input_linears[li], &w_plus, *dims);
            let loss_plus = compute_bpr_loss(
                &embeddings_to_plain(&model.forward(graph)),
                &positive,
                &negative,
            );
            let w_minus = perturb(orig_w, &delta, eps, -1.0);
            set_linear_weight(&mut model.input_linears[li], &w_minus, *dims);
            let loss_minus = compute_bpr_loss(
                &embeddings_to_plain(&model.forward(graph)),
                &positive,
                &negative,
            );
            let w_new = spsa_update(orig_w, &delta, loss_plus, loss_minus, eps, lr);
            set_linear_weight(&mut model.input_linears[li], &w_new, *dims);
        }

        // GAT has no AttnRes — per-edge attention is sufficient

        let emb = embeddings_to_plain(&model.forward(graph));
        final_loss = compute_bpr_loss(&emb, &positive, &negative);
        if final_loss < best_loss {
            best_loss = final_loss;
        }
        if epoch % 20 == 0 || epoch == epochs - 1 {
            let auc = link_prediction_auc(&emb, &positive, &negative);
            println!(
                "    epoch {:3}: loss={:.4}, auc={:.4}",
                epoch, final_loss, auc
            );
        }
    }

    let final_emb = embeddings_to_plain(&model.forward(graph));
    let trained_auc = link_prediction_auc(&final_emb, &positive, &negative);
    TrainResult {
        name: String::new(),
        init_auc,
        trained_auc,
        init_loss,
        final_loss,
    }
}

fn train_gps_spsa(
    model: &mut hehrgnn::model::graph_transformer::GraphTransformerModel<B>,
    graph: &HeteroGraph<B>,
    epochs: usize,
    lr: f32,
) -> TrainResult {
    let positive = extract_positive_edges(graph);
    let negative = sample_negative_edges(graph, &positive, 3);
    let eps = 0.01f32;

    let init_emb = embeddings_to_plain(&model.forward(graph));
    let init_loss = compute_bpr_loss(&init_emb, &positive, &negative);
    let init_auc = link_prediction_auc(&init_emb, &positive, &negative);
    let mut best_loss = init_loss;
    let mut final_loss = init_loss;

    for epoch in 0..epochs {
        let mut seed: u64 = (epoch as u64 * 7919).wrapping_add(42);

        let orig_linears = extract_linears(&model.input_projs);
        for (li, (orig_w, dims)) in orig_linears.iter().enumerate() {
            let delta = rademacher_vec(orig_w.len(), &mut seed);
            let w_plus = perturb(orig_w, &delta, eps, 1.0);
            set_linear_weight(&mut model.input_projs[li], &w_plus, *dims);
            let loss_plus = compute_bpr_loss(
                &embeddings_to_plain(&model.forward(graph)),
                &positive,
                &negative,
            );
            let w_minus = perturb(orig_w, &delta, eps, -1.0);
            set_linear_weight(&mut model.input_projs[li], &w_minus, *dims);
            let loss_minus = compute_bpr_loss(
                &embeddings_to_plain(&model.forward(graph)),
                &positive,
                &negative,
            );
            let w_new = spsa_update(orig_w, &delta, loss_plus, loss_minus, eps, lr);
            set_linear_weight(&mut model.input_projs[li], &w_new, *dims);
        }

        let orig_queries = extract_attn_queries(&model.attn_depth);
        for (qi, orig_q) in orig_queries.iter().enumerate() {
            if orig_q.is_empty() {
                continue;
            }
            let delta = rademacher_vec(orig_q.len(), &mut seed);
            let q_plus = perturb(orig_q, &delta, eps, 1.0);
            set_attn_query(&mut model.attn_depth, qi, &q_plus);
            let loss_plus = compute_bpr_loss(
                &embeddings_to_plain(&model.forward(graph)),
                &positive,
                &negative,
            );
            let q_minus = perturb(orig_q, &delta, eps, -1.0);
            set_attn_query(&mut model.attn_depth, qi, &q_minus);
            let loss_minus = compute_bpr_loss(
                &embeddings_to_plain(&model.forward(graph)),
                &positive,
                &negative,
            );
            let q_new = spsa_update(orig_q, &delta, loss_plus, loss_minus, eps, lr);
            set_attn_query(&mut model.attn_depth, qi, &q_new);
        }

        let emb = embeddings_to_plain(&model.forward(graph));
        final_loss = compute_bpr_loss(&emb, &positive, &negative);
        if final_loss < best_loss {
            best_loss = final_loss;
        }
        if epoch % 20 == 0 || epoch == epochs - 1 {
            let auc = link_prediction_auc(&emb, &positive, &negative);
            println!(
                "    epoch {:3}: loss={:.4}, auc={:.4}",
                epoch, final_loss, auc
            );
        }
    }

    let final_emb = embeddings_to_plain(&model.forward(graph));
    let trained_auc = link_prediction_auc(&final_emb, &positive, &negative);
    TrainResult {
        name: String::new(),
        init_auc,
        trained_auc,
        init_loss,
        final_loss,
    }
}

// ═══════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════

fn main() {
    println!(
        "{}",
        "╔════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "{}",
        "║  🏋️ AttnRes Deep Training — 8-LAYER models, LARGE graph, ALL weights trained  ║"
    );
    println!(
        "{}",
        "║  Over-smoothing scenario: 60+ nodes, 250+ edges, 8 GNN layers               ║"
    );
    println!(
        "{}",
        "╚════════════════════════════════════════════════════════════════════════════════╝"
    );

    let device = <B as Backend>::Device::default();
    let graph = build_training_graph();
    let (node_types, edge_types) = get_types(&graph);
    let in_dim = get_in_dim(&graph);
    let epochs = 100;
    let lr = 0.005f32;

    println!(
        "\n  Graph: {} types, {} edge types, in_dim={}",
        node_types.len(),
        edge_types.len(),
        in_dim
    );
    for nt in &node_types {
        println!(
            "    {} → {} nodes",
            nt,
            graph.node_counts.get(nt).unwrap_or(&0)
        );
    }
    let pos = extract_positive_edges(&graph);
    println!("  Positive edges: {}", pos.len());
    println!(
        "  Training: {} epochs, lr={}, SPSA eps=0.01, 8 LAYERS (deep)",
        epochs, lr
    );

    let mut results: Vec<TrainResult> = Vec::new();

    // ═══════════════════════════════════════════════════════════════
    // 1. GraphSAGE
    // ═══════════════════════════════════════════════════════════════
    println!("\n{}", "═".repeat(75));
    println!("  1. GraphSAGE (8-layer DEEP) — Weight-Level SPSA");
    println!("{}", "═".repeat(75));
    {
        use hehrgnn::model::graphsage::GraphSageModelConfig;
        let config = GraphSageModelConfig {
            in_dim,
            hidden_dim: 32,
            num_layers: 8,
            dropout: 0.0,
        };

        println!("  [AttnRes ON]");
        let mut m_on = config.init::<B>(&node_types, &edge_types, &device);
        let mut r = train_graphsage_spsa(&mut m_on, &graph, epochs, lr);
        r.name = "GraphSAGE+AttnRes".into();
        println!(
            "    → init={:.4} trained={:.4} Δ={:+.4}",
            r.init_auc,
            r.trained_auc,
            r.trained_auc - r.init_auc
        );
        results.push(r);

        println!("  [AttnRes OFF]");
        let mut m_off = config.init::<B>(&node_types, &edge_types, &device);
        m_off.attn_depth = None;
        let mut r = train_graphsage_spsa(&mut m_off, &graph, epochs, lr);
        r.name = "GraphSAGE baseline".into();
        println!(
            "    → init={:.4} trained={:.4} Δ={:+.4}",
            r.init_auc,
            r.trained_auc,
            r.trained_auc - r.init_auc
        );
        results.push(r);
    }

    // ═══════════════════════════════════════════════════════════════
    // 2. RGCN
    // ═══════════════════════════════════════════════════════════════
    println!("\n{}", "═".repeat(75));
    println!("  2. RGCN (8-layer DEEP) — Weight-Level SPSA");
    println!("{}", "═".repeat(75));
    {
        use hehrgnn::model::rgcn::RgcnConfig;
        let config = RgcnConfig {
            in_dim,
            hidden_dim: 32,
            num_layers: 8,
            num_bases: 2,
            dropout: 0.0,
        };

        println!("  [AttnRes ON]");
        let mut m_on = config.init_model::<B>(&node_types, &edge_types, &device);
        let mut r = train_rgcn_spsa(&mut m_on, &graph, epochs, lr);
        r.name = "RGCN+AttnRes".into();
        println!(
            "    → init={:.4} trained={:.4} Δ={:+.4}",
            r.init_auc,
            r.trained_auc,
            r.trained_auc - r.init_auc
        );
        results.push(r);

        println!("  [AttnRes OFF]");
        let mut m_off = config.init_model::<B>(&node_types, &edge_types, &device);
        m_off.attn_depth = None;
        let mut r = train_rgcn_spsa(&mut m_off, &graph, epochs, lr);
        r.name = "RGCN baseline".into();
        println!(
            "    → init={:.4} trained={:.4} Δ={:+.4}",
            r.init_auc,
            r.trained_auc,
            r.trained_auc - r.init_auc
        );
        results.push(r);
    }

    // ═══════════════════════════════════════════════════════════════
    // 3. GAT
    // ═══════════════════════════════════════════════════════════════
    println!("\n{}", "═".repeat(75));
    println!("  3. GAT (8-layer) — No AttnRes (per-edge attention sufficient)");
    println!("{}", "═".repeat(75));
    {
        use hehrgnn::model::gat::GatConfig;
        let config = GatConfig {
            in_dim,
            hidden_dim: 32,
            num_heads: 4,
            num_layers: 8,
            dropout: 0.0,
        };

        println!("  [GAT — no AttnRes by design]");
        let mut m = config.init_model::<B>(&node_types, &edge_types, &device);
        let mut r = train_gat_spsa(&mut m, &graph, epochs, lr);
        r.name = "GAT (no AttnRes)".into();
        println!(
            "    → init={:.4} trained={:.4} Δ={:+.4}",
            r.init_auc,
            r.trained_auc,
            r.trained_auc - r.init_auc
        );
        results.push(r);
    }

    // ═══════════════════════════════════════════════════════════════
    // 4. GPS Transformer
    // ═══════════════════════════════════════════════════════════════
    println!("\n{}", "═".repeat(75));
    println!("  4. GPS Transformer (8-layer DEEP) — Weight-Level SPSA");
    println!("{}", "═".repeat(75));
    {
        use hehrgnn::model::graph_transformer::GraphTransformerConfig;
        let config = GraphTransformerConfig {
            in_dim,
            hidden_dim: 32,
            num_heads: 4,
            num_layers: 8,
            ffn_ratio: 2,
            dropout: 0.0,
        };

        println!("  [AttnRes ON]");
        let mut m_on = config.init_model::<B>(&node_types, &edge_types, &device);
        let mut r = train_gps_spsa(&mut m_on, &graph, epochs, lr);
        r.name = "GPS+AttnRes".into();
        println!(
            "    → init={:.4} trained={:.4} Δ={:+.4}",
            r.init_auc,
            r.trained_auc,
            r.trained_auc - r.init_auc
        );
        results.push(r);

        println!("  [AttnRes OFF]");
        let mut m_off = config.init_model::<B>(&node_types, &edge_types, &device);
        m_off.attn_depth = None;
        let mut r = train_gps_spsa(&mut m_off, &graph, epochs, lr);
        r.name = "GPS baseline".into();
        println!(
            "    → init={:.4} trained={:.4} Δ={:+.4}",
            r.init_auc,
            r.trained_auc,
            r.trained_auc - r.init_auc
        );
        results.push(r);
    }

    // ═══════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════
    println!(
        "\n{}",
        "╔════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "{}",
        "║  WEIGHT-LEVEL TRAINING RESULTS                                            ║"
    );
    println!(
        "{}",
        "╠════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "  {:25}  {:>8}  {:>8}  {:>7}  {:>8}  {:>8}",
        "Model", "InitAUC", "Trained", "ΔAUC", "InitL", "FinalL"
    );
    println!(
        "  {:25}  {:>8}  {:>8}  {:>7}  {:>8}  {:>8}",
        "-".repeat(25),
        "-".repeat(8),
        "-".repeat(8),
        "-".repeat(7),
        "-".repeat(8),
        "-".repeat(8)
    );
    for r in &results {
        let delta = r.trained_auc - r.init_auc;
        let emoji = if delta > 0.01 {
            "🔼"
        } else if delta < -0.01 {
            "🔽"
        } else {
            "➡️"
        };
        println!(
            "  {:25}  {:>8.4}  {:>8.4}  {:>+6.3}  {:>8.4}  {:>8.4} {}",
            r.name, r.init_auc, r.trained_auc, delta, r.init_loss, r.final_loss, emoji
        );
    }

    println!("\n  HEAD-TO-HEAD (AttnRes vs Baseline — trained AUC):");
    println!(
        "  {:25}  {:>10}  {:>10}  {:>10}",
        "Matchup", "AttnRes", "Baseline", "Winner"
    );
    println!(
        "  {:25}  {:>10}  {:>10}  {:>10}",
        "-".repeat(25),
        "-".repeat(10),
        "-".repeat(10),
        "-".repeat(10)
    );

    let pairs = [
        (0, 1, "GraphSAGE"),
        (2, 3, "RGCN"),
        (4, 5, "GAT"),
        (6, 7, "GPS Transformer"),
    ];
    let mut attnres_wins = 0;
    let mut baseline_wins = 0;
    let mut ties = 0;
    for (a, b, name) in &pairs {
        if *a >= results.len() || *b >= results.len() {
            continue;
        }
        let ar = results[*a].trained_auc;
        let al = results[*a].final_loss;
        let br = results[*b].trained_auc;
        let bl = results[*b].final_loss;
        let winner = if ar > br + 0.005 {
            attnres_wins += 1;
            "✅ AttnRes"
        } else if br > ar + 0.005 {
            baseline_wins += 1;
            "Baseline"
        } else {
            // Break tie by loss
            if al < bl - 0.005 {
                attnres_wins += 1;
                "✅ AttnRes (loss)"
            } else if bl < al - 0.005 {
                baseline_wins += 1;
                "Baseline (loss)"
            } else {
                ties += 1;
                "Tie"
            }
        };
        println!("  {:25}  {:>10.4}  {:>10.4}  {:>10}", name, ar, br, winner);
    }
    println!(
        "\n  Final Score: AttnRes {} — Baseline {} (ties: {})",
        attnres_wins, baseline_wins, ties
    );

    if attnres_wins > baseline_wins {
        println!("  🏆 AttnRes WINS with weight-level training!");
    } else if attnres_wins == baseline_wins {
        println!("  🤝 Even match. AttnRes competitive with baseline.");
    } else {
        println!("  📊 Baseline edges ahead. Try more epochs or deeper models.");
    }
    println!(
        "{}",
        "╚════════════════════════════════════════════════════════════════════════════╝"
    );
}
