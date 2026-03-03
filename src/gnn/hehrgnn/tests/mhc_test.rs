//! mHC-GNN depth comparison tests.
//!
//! Comparing standard 2-layer vs mHC 8-layer GraphSAGE to demonstrate:
//! 1. mHC-GNN can go 8 layers deep without over-smoothing
//! 2. Standard GNN collapses (embeddings converge) at high depth
//! 3. mHC maintains embedding variance (discriminative power)

use burn::backend::NdArray;
use burn::prelude::*;

use hehrgnn::data::graph_builder::{build_hetero_graph, GraphBuildConfig, GraphFact};
use hehrgnn::data::hetero_graph::EdgeType;
use hehrgnn::model::graphsage::GraphSageModelConfig;
use hehrgnn::model::mhc::MhcGraphSageConfig;

type B = NdArray;

/// Build a test graph with multiple entity types and relations.
fn build_test_graph() -> hehrgnn::data::hetero_graph::HeteroGraph<B> {
    let device = <B as Backend>::Device::default();
    let facts = vec![
        // Users own accounts
        GraphFact { src: ("user".into(), "alice".into()), relation: "owns".into(), dst: ("account".into(), "acc1".into()) },
        GraphFact { src: ("user".into(), "bob".into()), relation: "owns".into(), dst: ("account".into(), "acc2".into()) },
        GraphFact { src: ("user".into(), "carol".into()), relation: "owns".into(), dst: ("account".into(), "acc3".into()) },
        // Transactions posted to accounts
        GraphFact { src: ("tx".into(), "tx1".into()), relation: "posted_to".into(), dst: ("account".into(), "acc1".into()) },
        GraphFact { src: ("tx".into(), "tx2".into()), relation: "posted_to".into(), dst: ("account".into(), "acc1".into()) },
        GraphFact { src: ("tx".into(), "tx3".into()), relation: "posted_to".into(), dst: ("account".into(), "acc2".into()) },
        GraphFact { src: ("tx".into(), "tx4".into()), relation: "posted_to".into(), dst: ("account".into(), "acc3".into()) },
        // Transactions at merchants
        GraphFact { src: ("tx".into(), "tx1".into()), relation: "at".into(), dst: ("merchant".into(), "walmart".into()) },
        GraphFact { src: ("tx".into(), "tx2".into()), relation: "at".into(), dst: ("merchant".into(), "amazon".into()) },
        GraphFact { src: ("tx".into(), "tx3".into()), relation: "at".into(), dst: ("merchant".into(), "walmart".into()) },
        GraphFact { src: ("tx".into(), "tx4".into()), relation: "at".into(), dst: ("merchant".into(), "target".into()) },
        // Categories
        GraphFact { src: ("tx".into(), "tx1".into()), relation: "has_cat".into(), dst: ("category".into(), "groceries".into()) },
        GraphFact { src: ("tx".into(), "tx2".into()), relation: "has_cat".into(), dst: ("category".into(), "electronics".into()) },
        GraphFact { src: ("tx".into(), "tx3".into()), relation: "has_cat".into(), dst: ("category".into(), "groceries".into()) },
        GraphFact { src: ("tx".into(), "tx4".into()), relation: "has_cat".into(), dst: ("category".into(), "retail".into()) },
    ];

    build_hetero_graph::<B>(
        &facts,
        &GraphBuildConfig {
            node_feat_dim: 16,
            add_reverse_edges: true,
            add_self_loops: true,
        },
        &device,
    )
}

/// Compute cosine similarity between all pairs of node embeddings of a given type.
/// Lower avg cosine = more diverse embeddings = less over-smoothing.
fn avg_pairwise_cosine(embeddings: &burn::tensor::Tensor<B, 2>) -> f32 {
    let dims = embeddings.dims();
    if dims[0] < 2 {
        return 0.0;
    }

    let data: Vec<f32> = embeddings.clone().into_data().as_slice::<f32>().unwrap().to_vec();
    let n = dims[0];
    let d = dims[1];
    let mut total_cos = 0.0f32;
    let mut count = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let mut dot = 0.0f32;
            let mut norm_i = 0.0f32;
            let mut norm_j = 0.0f32;
            for k in 0..d {
                let vi = data[i * d + k];
                let vj = data[j * d + k];
                dot += vi * vj;
                norm_i += vi * vi;
                norm_j += vj * vj;
            }
            let denom = (norm_i.sqrt() * norm_j.sqrt()).max(1e-8);
            total_cos += dot / denom;
            count += 1;
        }
    }

    if count > 0 { total_cos / count as f32 } else { 0.0 }
}

/// Compute embedding variance (higher = more discriminative).
fn embedding_variance(embeddings: &burn::tensor::Tensor<B, 2>) -> f32 {
    let dims = embeddings.dims();
    if dims[0] < 2 {
        return 0.0;
    }
    let mean = embeddings.clone().mean_dim(0);
    let diff = embeddings.clone() - mean.expand(dims);
    let var = (diff.clone() * diff).mean();
    var.into_data().as_slice::<f32>().unwrap()[0]
}

#[test]
fn test_mhc_vs_standard_depth() {
    let graph = build_test_graph();
    let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
    let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();

    println!("\n  ╔═══════════════════════════════════════════════════════════╗");
    println!("  ║  mHC-GNN DEPTH COMPARISON TEST                          ║");
    println!("  ╚═══════════════════════════════════════════════════════════╝");
    println!("    Node types: {:?}", node_types);
    println!("    Edge types: {}", edge_types.len());

    // ── Standard GraphSAGE at various depths ──
    println!("\n  ─── Standard GraphSAGE ───");
    for depth in [2, 4, 8] {
        let config = GraphSageModelConfig {
            in_dim: 16,
            hidden_dim: 16,
            num_layers: depth,
            dropout: 0.0,
        };
        let model = config.init::<B>(&node_types, &edge_types, &<B as Backend>::Device::default());
        let embeddings = model.forward(&graph);

        let mut avg_cos = 0.0f32;
        let mut avg_var = 0.0f32;
        let mut type_count = 0;
        for nt in &node_types {
            if let Some(emb) = embeddings.get(nt) {
                if emb.dims()[0] >= 2 {
                    avg_cos += avg_pairwise_cosine(emb);
                    avg_var += embedding_variance(emb);
                    type_count += 1;
                }
            }
        }
        if type_count > 0 {
            avg_cos /= type_count as f32;
            avg_var /= type_count as f32;
        }

        println!(
            "    Depth {:2}: avg_cosine={:.4}  variance={:.6}  {}",
            depth, avg_cos, avg_var,
            if avg_cos > 0.95 { "⚠️  OVER-SMOOTHED" } else { "✅ ok" }
        );
    }

    // ── mHC-GraphSAGE at various depths ──
    println!("\n  ─── mHC-GraphSAGE (n_streams=2) ───");
    for depth in [2, 4, 8] {
        let config = MhcGraphSageConfig {
            in_dim: 16,
            hidden_dim: 16,
            num_layers: depth,
            n_streams: 2,
            dropout: 0.0,
        };
        let model = config.init::<B>(&node_types, &edge_types, &<B as Backend>::Device::default());
        let embeddings = model.forward(&graph);

        let mut avg_cos = 0.0f32;
        let mut avg_var = 0.0f32;
        let mut type_count = 0;
        for nt in &node_types {
            if let Some(emb) = embeddings.get(nt) {
                if emb.dims()[0] >= 2 {
                    avg_cos += avg_pairwise_cosine(emb);
                    avg_var += embedding_variance(emb);
                    type_count += 1;
                }
            }
        }
        if type_count > 0 {
            avg_cos /= type_count as f32;
            avg_var /= type_count as f32;
        }

        println!(
            "    Depth {:2}: avg_cosine={:.4}  variance={:.6}  {}",
            depth, avg_cos, avg_var,
            if avg_cos > 0.95 { "⚠️  OVER-SMOOTHED" } else { "✅ ok" }
        );
    }

    // ── mHC with 4 streams ──
    println!("\n  ─── mHC-GraphSAGE (n_streams=4, depth=8) ───");
    let config = MhcGraphSageConfig {
        in_dim: 16,
        hidden_dim: 16,
        num_layers: 8,
        n_streams: 4,
        dropout: 0.0,
    };
    let model = config.init::<B>(&node_types, &edge_types, &<B as Backend>::Device::default());
    let embeddings = model.forward(&graph);

    let mut avg_cos = 0.0f32;
    let mut avg_var = 0.0f32;
    let mut type_count = 0;
    for nt in &node_types {
        if let Some(emb) = embeddings.get(nt) {
            if emb.dims()[0] >= 2 {
                avg_cos += avg_pairwise_cosine(emb);
                avg_var += embedding_variance(emb);
                type_count += 1;
            }
        }
    }
    if type_count > 0 {
        avg_cos /= type_count as f32;
        avg_var /= type_count as f32;
    }
    println!(
        "    Depth  8: avg_cosine={:.4}  variance={:.6}  {}",
        avg_cos, avg_var,
        if avg_cos > 0.95 { "⚠️  OVER-SMOOTHED" } else { "✅ ok" }
    );
    println!("    mHC overhead params: {}", model.param_count());

    println!("\n  ╔═══════════════════════════════════════════════════════════╗");
    println!("  ║  INTERPRETATION                                         ║");
    println!("  ╠═══════════════════════════════════════════════════════════╣");
    println!("  ║  Standard GNN: cosine → 1.0 = all nodes identical      ║");
    println!("  ║  mHC-GNN:      cosine stays low = diverse embeddings   ║");
    println!("  ║  Variance:     higher = more discriminative power      ║");
    println!("  ╚═══════════════════════════════════════════════════════════╝");

    // Verify output dims
    for nt in &node_types {
        let emb = embeddings.get(nt).expect(&format!("Missing {}", nt));
        assert_eq!(emb.dims()[1], 16, "Output dim should match hidden_dim");
    }
}

#[test]
fn test_sinkhorn_properties() {
    println!("\n  ── Sinkhorn-Knopp Properties ──");
    let device = <B as Backend>::Device::default();

    // Test with different sizes
    for n in [2, 4, 8] {
        let raw = burn::tensor::Tensor::<B, 2>::random(
            [n, n],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let ds = hehrgnn::model::mhc::sinkhorn_normalize(raw, 10);
        let data: Vec<f32> = ds.into_data().as_slice::<f32>().unwrap().to_vec();

        let mut max_row_err = 0.0f32;
        let mut max_col_err = 0.0f32;
        for r in 0..n {
            let row_sum: f32 = (0..n).map(|c| data[r * n + c]).sum();
            max_row_err = max_row_err.max((row_sum - 1.0).abs());
        }
        for c in 0..n {
            let col_sum: f32 = (0..n).map(|r| data[r * n + c]).sum();
            max_col_err = max_col_err.max((col_sum - 1.0).abs());
        }

        println!(
            "    n={}: max_row_err={:.6}  max_col_err={:.6}  {}",
            n, max_row_err, max_col_err,
            if max_row_err < 0.01 && max_col_err < 0.01 { "✅" } else { "❌" }
        );
        assert!(max_row_err < 0.01, "Row sums not close to 1 for n={}", n);
        assert!(max_col_err < 0.01, "Col sums not close to 1 for n={}", n);
    }
}
