//! Temporal validation selector for backbone + objective policy.
//!
//! Provides leakage-safe split by event order and a lightweight
//! model/objective sweep on a warm-start validation set.

use burn::backend::NdArray;
use burn::prelude::*;
use std::collections::HashMap;

use crate::data::graph_builder::{GraphBuildConfig, GraphFact, build_hetero_graph};
use crate::data::hetero_graph::EdgeType;
use crate::model::gat::GatConfig;
use crate::model::graph_transformer::GraphTransformerConfig;
use crate::model::graphsage::GraphSageModelConfig;
use crate::model::lora::{LoraConfig, init_hetero_basis_adapter};
use crate::model::mhc::MhcRgcnConfig;
use crate::model::trainer::{
    JepaTrainable, TrainConfig, TrainReport, embeddings_to_plain, link_prediction_auc,
    sample_negative_edges, train_adapter, train_hybrid_input_weights, train_jepa_input_weights,
};

type B = NdArray;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveKind {
    Jepa,
    Hybrid,
}

impl ObjectiveKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ObjectiveKind::Jepa => "jepa",
            ObjectiveKind::Hybrid => "hybrid",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackboneKind {
    GraphSage,
    RgcnMhc,
    Gat,
    Gps,
}

impl BackboneKind {
    pub fn as_str(self) -> &'static str {
        match self {
            BackboneKind::GraphSage => "graphsage",
            BackboneKind::RgcnMhc => "rgcn_mhc",
            BackboneKind::Gat => "gat",
            BackboneKind::Gps => "gps",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TemporalSelectorConfig {
    /// Fraction of earliest facts used for training.
    pub train_ratio: f32,
    /// Fraction of following facts used for validation.
    pub val_ratio: f32,
    /// Minimum facts required before running selector.
    pub min_facts: usize,
    /// Minimum warm validation edges required.
    pub min_warm_val_edges: usize,
}

impl Default for TemporalSelectorConfig {
    fn default() -> Self {
        Self {
            train_ratio: 0.70,
            val_ratio: 0.15,
            // Allow selector to run on medium-sized graphs instead of defaulting
            // to a fixed objective too often.
            min_facts: 24,
            min_warm_val_edges: 8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SelectorCandidateResult {
    pub backbone: BackboneKind,
    pub objective: ObjectiveKind,
    pub val_auc: f32,
    pub warm_val_edges: usize,
    pub train_report: TrainReport,
}

#[derive(Debug, Clone)]
pub struct PerBackbonePolicy {
    pub backbone: BackboneKind,
    pub objective: ObjectiveKind,
    pub val_auc: f32,
}

#[derive(Debug, Clone)]
pub struct TemporalPolicyReport {
    pub selected_backbone: BackboneKind,
    pub selected_objective: ObjectiveKind,
    pub selected_val_auc: f32,
    pub train_facts: usize,
    pub val_facts: usize,
    pub test_facts: usize,
    pub warm_val_edges: usize,
    pub candidates: Vec<SelectorCandidateResult>,
    pub per_backbone: Vec<PerBackbonePolicy>,
}

impl TemporalPolicyReport {
    pub fn objective_for(&self, backbone: BackboneKind) -> ObjectiveKind {
        self.per_backbone
            .iter()
            .find(|p| p.backbone == backbone)
            .map(|p| p.objective)
            .unwrap_or(self.selected_objective)
    }
}

fn split_facts_by_time<'a>(
    facts: &'a [GraphFact],
    cfg: &TemporalSelectorConfig,
) -> (&'a [GraphFact], &'a [GraphFact], &'a [GraphFact]) {
    let n = facts.len();
    if n == 0 {
        return (&[], &[], &[]);
    }
    if n < 3 {
        let train_end = n.min(1);
        let val_end = n.min(2);
        return (
            &facts[..train_end],
            &facts[train_end..val_end],
            &facts[val_end..],
        );
    }

    let mut train_end = ((n as f32) * cfg.train_ratio).floor() as usize;
    train_end = train_end.clamp(1, n.saturating_sub(1));

    let mut val_end = train_end + ((n as f32) * cfg.val_ratio).floor() as usize;
    val_end = val_end.clamp(train_end + 1, n);

    (
        &facts[..train_end],
        &facts[train_end..val_end],
        &facts[val_end..],
    )
}

fn build_local_index_map(facts: &[GraphFact]) -> HashMap<String, HashMap<String, usize>> {
    let mut map: HashMap<String, HashMap<String, usize>> = HashMap::new();
    for fact in facts {
        let src = map.entry(fact.src.0.clone()).or_default();
        let src_len = src.len();
        src.entry(fact.src.1.clone()).or_insert(src_len);

        let dst = map.entry(fact.dst.0.clone()).or_default();
        let dst_len = dst.len();
        dst.entry(fact.dst.1.clone()).or_insert(dst_len);
    }
    map
}

fn warm_future_edges(
    train_index: &HashMap<String, HashMap<String, usize>>,
    future_facts: &[GraphFact],
) -> Vec<(String, usize, String, usize)> {
    let mut edges = Vec::new();
    for fact in future_facts {
        let Some(src_idx) = train_index
            .get(&fact.src.0)
            .and_then(|m| m.get(&fact.src.1))
            .copied()
        else {
            continue;
        };
        let Some(dst_idx) = train_index
            .get(&fact.dst.0)
            .and_then(|m| m.get(&fact.dst.1))
            .copied()
        else {
            continue;
        };
        edges.push((fact.src.0.clone(), src_idx, fact.dst.0.clone(), dst_idx));
    }
    edges
}

fn evaluate_future_auc<B2: Backend>(
    graph: &crate::data::hetero_graph::HeteroGraph<B2>,
    embeddings: &HashMap<String, Vec<Vec<f32>>>,
    future_edges: &[(String, usize, String, usize)],
    neg_ratio: usize,
) -> Option<f32> {
    if future_edges.is_empty() {
        return None;
    }
    let negatives = sample_negative_edges(graph, future_edges, neg_ratio.max(1));
    if negatives.is_empty() {
        return None;
    }
    Some(link_prediction_auc(embeddings, future_edges, &negatives))
}

fn train_candidate(
    backbone: BackboneKind,
    objective: ObjectiveKind,
    train_graph: &crate::data::hetero_graph::HeteroGraph<B>,
    device: &<B as Backend>::Device,
    node_types: &[String],
    edge_types: &[EdgeType],
    in_dim: usize,
    hidden_dim: usize,
    config: &TrainConfig,
) -> (TrainReport, HashMap<String, Vec<Vec<f32>>>) {
    fn train_for_objective<M: JepaTrainable<B>>(
        model: &mut M,
        train_graph: &crate::data::hetero_graph::HeteroGraph<B>,
        config: &TrainConfig,
        objective: ObjectiveKind,
    ) -> TrainReport {
        match objective {
            ObjectiveKind::Jepa => {
                train_jepa_input_weights(model, train_graph, config, 0.12, 0.30, true)
            }
            ObjectiveKind::Hybrid => {
                train_hybrid_input_weights(model, train_graph, config, 0.12, 0.30, 0.60, true)
            }
        }
    }

    match backbone {
        BackboneKind::GraphSage => {
            let mut model = GraphSageModelConfig {
                in_dim,
                hidden_dim,
                num_layers: 2,
                dropout: 0.0,
            }
            .init::<B>(node_types, edge_types, device);

            let adapter = init_hetero_basis_adapter(
                in_dim,
                hidden_dim,
                &LoraConfig::default(),
                node_types.to_vec(),
                device,
            );
            model.attach_adapter(adapter);
            let _ = train_adapter(&mut model, train_graph, config);
            let report = train_for_objective(&mut model, train_graph, config, objective);
            (report, embeddings_to_plain(&model.forward(train_graph)))
        }
        BackboneKind::RgcnMhc => {
            let mut model = MhcRgcnConfig {
                in_dim,
                hidden_dim,
                num_layers: 8,
                num_bases: 4,
                n_streams: 4,
                dropout: 0.0,
            }
            .init::<B>(node_types, edge_types, device);

            let report = train_for_objective(&mut model, train_graph, config, objective);
            (report, embeddings_to_plain(&model.forward(train_graph)))
        }
        BackboneKind::Gat => {
            let mut model = GatConfig {
                in_dim,
                hidden_dim,
                num_heads: 4,
                num_layers: 2,
                dropout: 0.0,
            }
            .init_model::<B>(node_types, edge_types, device);

            let report = train_for_objective(&mut model, train_graph, config, objective);
            (report, embeddings_to_plain(&model.forward(train_graph)))
        }
        BackboneKind::Gps => {
            let mut model = GraphTransformerConfig {
                in_dim,
                hidden_dim,
                num_heads: 4,
                num_layers: 2,
                ffn_ratio: 2,
                dropout: 0.0,
            }
            .init_model::<B>(node_types, edge_types, device);

            let report = train_for_objective(&mut model, train_graph, config, objective);
            (report, embeddings_to_plain(&model.forward(train_graph)))
        }
    }
}

/// Select objective/backbone policy from leakage-safe temporal validation.
///
/// Returns `None` when not enough data is available.
pub fn select_temporal_policy(
    facts: &[GraphFact],
    hidden_dim: usize,
    base_train_config: &TrainConfig,
) -> Option<TemporalPolicyReport> {
    let cfg = TemporalSelectorConfig::default();
    if facts.len() < cfg.min_facts {
        return None;
    }

    let (train_facts, val_facts, test_facts) = split_facts_by_time(facts, &cfg);
    if train_facts.is_empty() || val_facts.is_empty() {
        return None;
    }

    let device = <B as Backend>::Device::default();
    let build_config = GraphBuildConfig {
        node_feat_dim: hidden_dim,
        add_reverse_edges: true,
        add_self_loops: true,
        add_positional_encoding: true,
    };
    let train_graph = build_hetero_graph::<B>(train_facts, &build_config, &device);
    let node_types: Vec<String> = train_graph
        .node_types()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let edge_types: Vec<EdgeType> = train_graph
        .edge_types()
        .iter()
        .map(|e| (*e).clone())
        .collect();

    if node_types.is_empty() || edge_types.is_empty() {
        return None;
    }

    let train_index = build_local_index_map(train_facts);
    let warm_val = warm_future_edges(&train_index, val_facts);
    // For smaller validation windows, scale the warm-edge requirement down so
    // the selector still runs when there is enough signal.
    let min_required_warm = cfg.min_warm_val_edges.min((val_facts.len() / 3).max(2));
    if warm_val.len() < min_required_warm {
        return None;
    }

    let in_dim = train_graph
        .node_features
        .values()
        .next()
        .map(|t| t.dims()[1])
        .unwrap_or(hidden_dim);

    let quick_config = TrainConfig {
        epochs: base_train_config.epochs.min(4).max(2),
        lr: base_train_config.lr,
        neg_ratio: base_train_config.neg_ratio.max(1),
        patience: base_train_config.patience.min(3).max(2),
        perturb_frac: base_train_config.perturb_frac,
        mode: base_train_config.mode,
        weight_decay: base_train_config.weight_decay,
    };

    let mut candidates = Vec::new();
    let backbones = [
        BackboneKind::GraphSage,
        BackboneKind::RgcnMhc,
        BackboneKind::Gat,
        BackboneKind::Gps,
    ];
    let objectives = [ObjectiveKind::Jepa, ObjectiveKind::Hybrid];

    for backbone in backbones {
        for objective in objectives {
            let (train_report, emb) = train_candidate(
                backbone,
                objective,
                &train_graph,
                &device,
                &node_types,
                &edge_types,
                in_dim,
                hidden_dim,
                &quick_config,
            );
            let Some(val_auc) =
                evaluate_future_auc(&train_graph, &emb, &warm_val, quick_config.neg_ratio.max(1))
            else {
                continue;
            };

            candidates.push(SelectorCandidateResult {
                backbone,
                objective,
                val_auc,
                warm_val_edges: warm_val.len(),
                train_report,
            });
        }
    }

    if candidates.is_empty() {
        return None;
    }

    let mut per_backbone = Vec::new();
    for backbone in backbones {
        let mut best: Option<&SelectorCandidateResult> = None;
        for c in candidates.iter().filter(|c| c.backbone == backbone) {
            best = match best {
                Some(prev) => {
                    if c.val_auc > prev.val_auc
                        || ((c.val_auc - prev.val_auc).abs() < f32::EPSILON
                            && c.train_report.final_loss < prev.train_report.final_loss)
                    {
                        Some(c)
                    } else {
                        Some(prev)
                    }
                }
                None => Some(c),
            };
        }
        if let Some(chosen) = best {
            per_backbone.push(PerBackbonePolicy {
                backbone: chosen.backbone,
                objective: chosen.objective,
                val_auc: chosen.val_auc,
            });
        }
    }

    candidates.sort_by(|a, b| {
        b.val_auc
            .partial_cmp(&a.val_auc)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.train_report
                    .final_loss
                    .partial_cmp(&b.train_report.final_loss)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    let best = candidates[0].clone();

    Some(TemporalPolicyReport {
        selected_backbone: best.backbone,
        selected_objective: best.objective,
        selected_val_auc: best.val_auc,
        train_facts: train_facts.len(),
        val_facts: val_facts.len(),
        test_facts: test_facts.len(),
        warm_val_edges: best.warm_val_edges,
        candidates,
        per_backbone,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gf(ht: &str, h: &str, r: &str, tt: &str, t: &str) -> GraphFact {
        GraphFact {
            src: (ht.to_string(), h.to_string()),
            relation: r.to_string(),
            dst: (tt.to_string(), t.to_string()),
        }
    }

    fn make_facts() -> Vec<GraphFact> {
        let mut out = Vec::new();
        for u in 0..24 {
            let user = format!("u{}", u % 8);
            let acct = format!("a{}", u % 10);
            let tx = format!("t{}", u);
            let merch = format!("m{}", u % 6);
            let cat = format!("c{}", u % 4);
            out.push(gf("user", &user, "has_account", "account", &acct));
            out.push(gf("account", &acct, "has_txn", "transaction", &tx));
            out.push(gf("transaction", &tx, "at_merchant", "merchant", &merch));
            out.push(gf("merchant", &merch, "in_category", "category", &cat));
        }
        out
    }

    #[test]
    fn test_temporal_policy_selector_runs() {
        let facts = make_facts();
        let config = TrainConfig {
            epochs: 3,
            lr: 0.05,
            neg_ratio: 2,
            patience: 2,
            perturb_frac: 0.3,
            mode: crate::model::trainer::TrainMode::Fast,
            weight_decay: 0.01,
        };

        let report = select_temporal_policy(&facts, 16, &config).expect("selector should run");
        assert!(report.selected_val_auc.is_finite());
        assert!(report.warm_val_edges >= 1);
        assert!(!report.candidates.is_empty());
    }
}
