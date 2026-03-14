//! Negative sampling correctness tests.

use burn::backend::NdArray;
use burn::prelude::*;
use std::collections::{HashMap, HashSet};

use hehrgnn::data::graph_builder::{GraphBuildConfig, GraphFact, build_hetero_graph};
use hehrgnn::model::trainer::{
    extract_positive_edges, sample_negative_edges, sample_temporal_hard_negative_edges,
};

type B = NdArray;

fn make_graph() -> hehrgnn::data::hetero_graph::HeteroGraph<B> {
    let facts = vec![
        GraphFact {
            src: ("user".into(), "u0".into()),
            relation: "owns".into(),
            dst: ("account".into(), "a0".into()),
        },
        GraphFact {
            src: ("user".into(), "u0".into()),
            relation: "owns".into(),
            dst: ("account".into(), "a1".into()),
        },
        GraphFact {
            src: ("user".into(), "u1".into()),
            relation: "owns".into(),
            dst: ("account".into(), "a2".into()),
        },
    ];
    let device = <B as Backend>::Device::default();
    build_hetero_graph::<B>(
        &facts,
        &GraphBuildConfig {
            node_feat_dim: 8,
            add_reverse_edges: true,
            add_self_loops: true,
            add_positional_encoding: true,
            add_cross_dependency_edges: true,
        },
        &device,
    )
}

fn user_account_anchor(
    graph: &hehrgnn::data::hetero_graph::HeteroGraph<B>,
) -> (usize, Vec<usize>, usize) {
    let edges = extract_positive_edges(graph);
    let mut by_src: HashMap<usize, HashSet<usize>> = HashMap::new();
    for (st, si, dt, di) in edges {
        if st == "user" && dt == "account" {
            by_src.entry(si).or_default().insert(di);
        }
    }

    let (src_idx, connected) = by_src
        .into_iter()
        .find(|(_, dsts)| dsts.len() >= 2)
        .expect("expected one user with >=2 account edges");
    let mut connected_vec: Vec<usize> = connected.into_iter().collect();
    connected_vec.sort_unstable();
    let holdout = connected_vec[0];
    (src_idx, connected_vec, holdout)
}

fn synthetic_embeddings(
    graph: &hehrgnn::data::hetero_graph::HeteroGraph<B>,
) -> HashMap<String, Vec<Vec<f32>>> {
    let mut out = HashMap::new();
    for (nt, count) in &graph.node_counts {
        let vecs: Vec<Vec<f32>> = (0..*count)
            .map(|i| vec![i as f32 + 1.0, (i * 3 + 1) as f32, (i * 7 + 3) as f32])
            .collect();
        out.insert(nt.clone(), vecs);
    }
    out
}

#[test]
fn test_negative_sampler_avoids_graph_positives_for_subset() {
    let graph = make_graph();
    let (src_idx, connected, holdout) = user_account_anchor(&graph);

    let subset = vec![("user".to_string(), src_idx, "account".to_string(), holdout)];
    let negatives = sample_negative_edges(&graph, &subset, 6);
    assert!(
        !negatives.is_empty(),
        "expected at least one valid negative"
    );

    for (st, si, dt, di) in negatives {
        assert_eq!(st, "user");
        assert_eq!(dt, "account");
        assert_eq!(si, src_idx);
        assert!(
            !connected.contains(&di),
            "sampled a known positive edge as negative: user:{} -> account:{}",
            si,
            di
        );
    }
}

#[test]
fn test_temporal_hard_negative_sampler_avoids_graph_positives_for_subset() {
    let graph = make_graph();
    let (src_idx, connected, holdout) = user_account_anchor(&graph);
    let embeddings = synthetic_embeddings(&graph);

    let subset = vec![("user".to_string(), src_idx, "account".to_string(), holdout)];
    let negatives =
        sample_temporal_hard_negative_edges(&graph, &subset, &embeddings, 6, 12, 0.7, 0.3);
    assert!(
        !negatives.is_empty(),
        "expected at least one valid hard negative"
    );

    for (st, si, dt, di) in negatives {
        assert_eq!(st, "user");
        assert_eq!(dt, "account");
        assert_eq!(si, src_idx);
        assert!(
            !connected.contains(&di),
            "sampled a known positive edge as hard negative: user:{} -> account:{}",
            si,
            di
        );
    }
}
