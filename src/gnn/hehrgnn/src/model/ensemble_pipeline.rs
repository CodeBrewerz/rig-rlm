//! Ensemble Pipeline: orchestrates the full GNN learning lifecycle.
//!
//! Each pipeline run:
//! 1. Load checkpointed GNN model weights (if available)
//! 2. Run all 4 GNN models → embeddings
//! 3. Train GNN model on graph structure (link prediction)
//! 4. Load/create LearnableScorer → score fiduciary actions
//! 5. Learn from rewards (distill/reward/recursive improve)
//! 6. Save all weights (GNN models + scorer)
//!
//! Every run improves: embeddings get better → fiduciary scores improve →
//! SAE/probing/interpretability all read the improved embeddings.

use burn::backend::NdArray;
use burn::prelude::*;
use std::collections::HashMap;

use crate::data::graph_builder::{build_hetero_graph, GraphBuildConfig, GraphFact};
use crate::data::hetero_graph::EdgeType;
use crate::eval::learnable_scorer::{LearnableScorer, ScorerConfig};
use crate::model::backbone::NodeEmbeddings;
use crate::model::gat::GatConfig;
use crate::model::graph_transformer::GraphTransformerConfig;
use crate::model::graphsage::GraphSageModelConfig;
use crate::model::lora::{init_hetero_basis_adapter, LoraConfig};
use crate::model::mhc::MhcRgcnConfig;
use crate::model::temporal_selector::{
    select_temporal_policy, BackboneKind, ObjectiveKind, TemporalPolicyReport,
};
use crate::model::trainer::*;
use crate::model::weights::*;

type B = NdArray;

/// Configuration for the ensemble pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub hidden_dim: usize,
    pub graph_hash: u64,
    pub train_config: TrainConfig,
    pub scorer_config: ScorerConfig,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 16,
            graph_hash: 0,
            train_config: TrainConfig {
                epochs: 30,
                lr: 0.05,
                neg_ratio: 3,
                patience: 10,
                perturb_frac: 0.3,
                mode: TrainMode::Fast,
                weight_decay: 0.01,
                decor_weight: 0.1,
            },
            scorer_config: ScorerConfig::default(),
        }
    }
}

/// Report from a full pipeline run.
#[derive(Debug, Clone)]
pub struct PipelineReport {
    pub models_loaded_from_checkpoint: Vec<String>,
    pub scorer_loaded_from_checkpoint: bool,
    pub train_report: Option<TrainReport>,
    pub pre_train_auc: f32,
    pub post_train_auc: f32,
    /// Selected objective used for this run (`jepa` or `hybrid`).
    pub selected_objective: String,
    /// Selected backbone from temporal validation policy.
    pub selected_backbone: String,
    /// Full temporal selection report (if selector had enough data).
    pub temporal_policy: Option<TemporalPolicyReport>,
    /// Objective selected for each backbone (`graphsage`,`rgcn_mhc`,`gat`,`gps`).
    pub selected_objectives_by_backbone: HashMap<String, String>,
    pub models_saved: Vec<String>,
    pub scorer_saved: bool,
    /// GEPA auto-tune report (if tuning ran this cycle).
    pub gepa_auto_tune: Option<crate::optimizer::gepa::AutoTuneReport>,
}

/// Plain embeddings for serialization/cross-model use.
#[derive(Debug, Clone)]
pub struct PlainEmbeddings {
    pub data: HashMap<String, Vec<Vec<f32>>>,
}

impl PlainEmbeddings {
    pub fn from_burn<B2: Backend>(node_emb: &NodeEmbeddings<B2>) -> Self {
        Self {
            data: embeddings_to_plain(node_emb),
        }
    }
}

fn train_with_objective<M: JepaTrainable<B>>(
    model: &mut M,
    graph: &crate::data::hetero_graph::HeteroGraph<B>,
    config: &TrainConfig,
    objective: ObjectiveKind,
    uniformity_weight: f32,
) -> TrainReport {
    match objective {
        ObjectiveKind::Jepa => train_jepa_input_weights(
            model,
            graph,
            config,
            0.12, // base temperature τ (scheduled internally)
            uniformity_weight,
            true, // edge predictor on
        ),
        ObjectiveKind::Hybrid => train_hybrid_input_weights(
            model,
            graph,
            config,
            0.12, // base temperature τ (scheduled internally)
            uniformity_weight,
            0.60, // supervised BPR blend
            true, // edge predictor on
        ),
    }
}

fn embedding_key_for_backbone(backbone: BackboneKind) -> &'static str {
    match backbone {
        BackboneKind::GraphSage => "graphsage",
        BackboneKind::RgcnMhc => "rgcn",
        BackboneKind::Gat => "gat",
        BackboneKind::Gps => "gps",
    }
}

fn objective_for_backbone(
    policy: Option<&TemporalPolicyReport>,
    fallback: ObjectiveKind,
    backbone: BackboneKind,
) -> ObjectiveKind {
    policy
        .map(|p| p.objective_for(backbone))
        .unwrap_or(fallback)
}

fn objective_map(
    policy: Option<&TemporalPolicyReport>,
    fallback: ObjectiveKind,
) -> HashMap<String, String> {
    let mut m = HashMap::new();
    for backbone in [
        BackboneKind::GraphSage,
        BackboneKind::RgcnMhc,
        BackboneKind::Gat,
        BackboneKind::Gps,
    ] {
        let obj = objective_for_backbone(policy, fallback, backbone);
        m.insert(backbone.as_str().to_string(), obj.as_str().to_string());
    }
    m
}

/// Run the full ensemble pipeline on a set of graph facts.
///
/// Per-model architecture defaults:
/// - GraphSAGE: DoRA adapter + objective-selected training
/// - RGCN: mHC 8-layer + objective-selected training
/// - GAT: objective-selected training
/// - GPS: objective-selected training
///
/// Returns the pipeline report and the combined embeddings.
pub fn run_pipeline(
    facts: &[GraphFact],
    config: &PipelineConfig,
) -> (PipelineReport, HashMap<String, PlainEmbeddings>) {
    let device = <B as Backend>::Device::default();
    let build_config = GraphBuildConfig {
        node_feat_dim: config.hidden_dim,
        add_reverse_edges: true,
        add_self_loops: true,
        add_positional_encoding: true,
    };
    let graph = build_hetero_graph::<B>(facts, &build_config, &device);
    let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
    let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();
    let input_dim = graph
        .node_features
        .values()
        .next()
        .map(|t| t.dims()[1])
        .unwrap_or(config.hidden_dim);
    let gh = config.graph_hash;
    let temporal_policy = select_temporal_policy(facts, config.hidden_dim, &config.train_config);
    let global_objective = temporal_policy
        .as_ref()
        .map(|p| p.selected_objective)
        .unwrap_or(ObjectiveKind::Jepa);
    let selected_backbone = temporal_policy
        .as_ref()
        .map(|p| p.selected_backbone)
        .unwrap_or(BackboneKind::GraphSage);
    let selected_objectives_by_backbone = objective_map(temporal_policy.as_ref(), global_objective);
    let objective_label = {
        let mut uniq: Vec<&str> = selected_objectives_by_backbone
            .values()
            .map(|s| s.as_str())
            .collect();
        uniq.sort_unstable();
        uniq.dedup();
        if uniq.len() <= 1 {
            global_objective.as_str().to_string()
        } else {
            "mixed".to_string()
        }
    };

    if let Some(policy) = &temporal_policy {
        eprintln!(
            "  [pipeline] Temporal selector: objective={}, backbone={}, val_auc={:.4}, warm_val_edges={}",
            policy.selected_objective.as_str(),
            policy.selected_backbone.as_str(),
            policy.selected_val_auc,
            policy.warm_val_edges
        );
    } else {
        eprintln!(
            "  [pipeline] Temporal selector skipped (insufficient facts/warm edges); using objective=jepa, backbone=graphsage"
        );
    }

    let mut report = PipelineReport {
        models_loaded_from_checkpoint: Vec::new(),
        scorer_loaded_from_checkpoint: false,
        train_report: None,
        pre_train_auc: 0.0,
        post_train_auc: 0.0,
        selected_objective: objective_label,
        selected_backbone: selected_backbone.as_str().to_string(),
        temporal_policy: temporal_policy.clone(),
        selected_objectives_by_backbone: selected_objectives_by_backbone.clone(),
        models_saved: Vec::new(),
        scorer_saved: false,
        gepa_auto_tune: None,
    };

    let mut all_embeddings: HashMap<String, PlainEmbeddings> = HashMap::new();

    // ── 1. GraphSAGE + DoRA + objective-selected training ──
    let mut sage_model = GraphSageModelConfig {
        in_dim: input_dim,
        hidden_dim: config.hidden_dim,
        num_layers: 2,
        dropout: 0.0,
    }
    .init::<B>(&node_types, &edge_types, &device);

    if let Some((loaded, _meta)) = load_model(
        GraphSageModelConfig {
            in_dim: input_dim,
            hidden_dim: config.hidden_dim,
            num_layers: 2,
            dropout: 0.0,
        }
        .init::<B>(&node_types, &edge_types, &device),
        "graphsage",
        gh,
        &device,
    ) {
        sage_model = loaded;
        report
            .models_loaded_from_checkpoint
            .push("graphsage".into());
        eprintln!("  [pipeline] Loaded GraphSAGE checkpoint");
    }

    // Attach HeteroDoRA adapter for added expressiveness
    let sage_adapter = init_hetero_basis_adapter(
        input_dim,
        config.hidden_dim,
        &LoraConfig::default(),
        node_types.clone(),
        &device,
    );
    sage_model.attach_adapter(sage_adapter);

    // Pre-train AUC
    let positive = extract_positive_edges(&graph);
    let negative = sample_negative_edges(&graph, &positive, config.train_config.neg_ratio);
    let pre_emb = embeddings_to_plain(&sage_model.forward(&graph));
    report.pre_train_auc = link_prediction_auc(&pre_emb, &positive, &negative);

    // Train adapter first, then objective-selected update on persistent input projection weights.
    let _adapter_report = train_adapter(&mut sage_model, &graph, &config.train_config);
    let sage_objective = objective_for_backbone(
        temporal_policy.as_ref(),
        global_objective,
        BackboneKind::GraphSage,
    );
    let train_report = train_with_objective(
        &mut sage_model,
        &graph,
        &config.train_config,
        sage_objective,
        0.35, // GraphSAGE favors slightly stronger uniformity regularization
    );
    // Evaluate post-train AUC on the same fixed benchmark negatives as pre-train AUC
    // so run-to-run checkpoint comparisons are stable.
    let post_emb_eval = embeddings_to_plain(&sage_model.forward(&graph));
    report.post_train_auc = link_prediction_auc(&post_emb_eval, &positive, &negative);
    report.train_report = Some(train_report.clone());
    eprintln!(
        "  [pipeline] GraphSAGE trained: auc={:.4} (DoRA+{})",
        report.post_train_auc,
        sage_objective.as_str().to_uppercase()
    );

    // Save GraphSAGE
    let meta = WeightMeta {
        model_type: "graphsage".into(),
        graph_hash: gh,
        epochs_trained: train_report.epochs_trained,
        final_loss: train_report.final_loss,
        final_auc: train_report.final_auc,
        hidden_dim: config.hidden_dim,
        timestamp: "pipeline".into(),
    };
    if save_model(&sage_model, "graphsage", gh, &meta, &device).is_ok() {
        report.models_saved.push("graphsage".into());
    }

    let sage_emb = PlainEmbeddings::from_burn(&sage_model.forward(&graph));
    all_embeddings.insert("graphsage".into(), sage_emb);

    // ── 2. RGCN + mHC + objective-selected training ──
    // Use mHC-enhanced RGCN with 8 layers + 4 streams for depth without over-smoothing
    let mut mhc_rgcn_model = {
        let fresh = MhcRgcnConfig {
            in_dim: input_dim,
            hidden_dim: config.hidden_dim,
            num_layers: 8,
            num_bases: 4,
            n_streams: 4,
            dropout: 0.0,
        }
        .init::<B>(&node_types, &edge_types, &device);

        if let Some((loaded, _)) = load_model(
            MhcRgcnConfig {
                in_dim: input_dim,
                hidden_dim: config.hidden_dim,
                num_layers: 8,
                num_bases: 4,
                n_streams: 4,
                dropout: 0.0,
            }
            .init::<B>(&node_types, &edge_types, &device),
            "rgcn_mhc",
            gh,
            &device,
        ) {
            report.models_loaded_from_checkpoint.push("rgcn_mhc".into());
            eprintln!("  [pipeline] Loaded RGCN mHC checkpoint");
            loaded
        } else {
            fresh
        }
    };

    // Train with objective-selected update on persistent weights.
    let rgcn_objective = objective_for_backbone(
        temporal_policy.as_ref(),
        global_objective,
        BackboneKind::RgcnMhc,
    );
    let rgcn_train = train_with_objective(
        &mut mhc_rgcn_model,
        &graph,
        &config.train_config,
        rgcn_objective,
        0.30,
    );
    eprintln!(
        "  [pipeline] RGCN trained: auc={:.4} (mHC+{}, 8 layers)",
        rgcn_train.final_auc,
        rgcn_objective.as_str().to_uppercase()
    );
    let rgcn_emb = PlainEmbeddings::from_burn(&mhc_rgcn_model.forward(&graph));
    let rgcn_meta = WeightMeta {
        model_type: "rgcn_mhc".into(),
        graph_hash: gh,
        epochs_trained: rgcn_train.epochs_trained,
        final_loss: rgcn_train.final_loss,
        final_auc: rgcn_train.final_auc,
        hidden_dim: config.hidden_dim,
        timestamp: "pipeline".into(),
    };
    if save_model(&mhc_rgcn_model, "rgcn_mhc", gh, &rgcn_meta, &device).is_ok() {
        report.models_saved.push("rgcn_mhc".into());
    }
    all_embeddings.insert("rgcn".into(), rgcn_emb);

    // ── 3. GAT: load or init ──
    let mut gat_model = {
        let fresh = GatConfig {
            in_dim: input_dim,
            hidden_dim: config.hidden_dim,
            num_heads: 4,
            num_layers: 2,
            dropout: 0.0,
        }
        .init_model::<B>(&node_types, &edge_types, &device);

        if let Some((loaded, _)) = load_model(
            GatConfig {
                in_dim: input_dim,
                hidden_dim: config.hidden_dim,
                num_heads: 4,
                num_layers: 2,
                dropout: 0.0,
            }
            .init_model::<B>(&node_types, &edge_types, &device),
            "gat",
            gh,
            &device,
        ) {
            report.models_loaded_from_checkpoint.push("gat".into());
            eprintln!("  [pipeline] Loaded GAT checkpoint");
            loaded
        } else {
            fresh
        }
    };
    // Train GAT with objective-selected update on persistent weights.
    let gat_objective = objective_for_backbone(
        temporal_policy.as_ref(),
        global_objective,
        BackboneKind::Gat,
    );
    let gat_train = train_with_objective(
        &mut gat_model,
        &graph,
        &config.train_config,
        gat_objective,
        0.30,
    );
    eprintln!(
        "  [pipeline] GAT trained: auc={:.4} ({})",
        gat_train.final_auc,
        gat_objective.as_str().to_uppercase()
    );
    let gat_emb = PlainEmbeddings::from_burn(&gat_model.forward(&graph));
    let gat_meta = WeightMeta {
        model_type: "gat".into(),
        graph_hash: gh,
        epochs_trained: gat_train.epochs_trained,
        final_loss: gat_train.final_loss,
        final_auc: gat_train.final_auc,
        hidden_dim: config.hidden_dim,
        timestamp: "pipeline".into(),
    };
    if save_model(&gat_model, "gat", gh, &gat_meta, &device).is_ok() {
        report.models_saved.push("gat".into());
    }
    all_embeddings.insert("gat".into(), gat_emb);

    // ── 4. GPS Transformer: load or init ──
    let mut gps_model = {
        let fresh = GraphTransformerConfig {
            in_dim: input_dim,
            hidden_dim: config.hidden_dim,
            num_heads: 4,
            num_layers: 2,
            ffn_ratio: 2,
            dropout: 0.0,
        }
        .init_model::<B>(&node_types, &edge_types, &device);

        if let Some((loaded, _)) = load_model(
            GraphTransformerConfig {
                in_dim: input_dim,
                hidden_dim: config.hidden_dim,
                num_heads: 4,
                num_layers: 2,
                ffn_ratio: 2,
                dropout: 0.0,
            }
            .init_model::<B>(&node_types, &edge_types, &device),
            "gps",
            gh,
            &device,
        ) {
            report.models_loaded_from_checkpoint.push("gps".into());
            eprintln!("  [pipeline] Loaded GPS checkpoint");
            loaded
        } else {
            fresh
        }
    };
    // Train GPS with objective-selected update on persistent weights.
    let gps_objective = objective_for_backbone(
        temporal_policy.as_ref(),
        global_objective,
        BackboneKind::Gps,
    );
    let gps_train = train_with_objective(
        &mut gps_model,
        &graph,
        &config.train_config,
        gps_objective,
        0.30,
    );
    eprintln!(
        "  [pipeline] GPS trained: auc={:.4} ({})",
        gps_train.final_auc,
        gps_objective.as_str().to_uppercase()
    );
    let gps_emb = PlainEmbeddings::from_burn(&gps_model.forward(&graph));
    let gps_meta = WeightMeta {
        model_type: "gps".into(),
        graph_hash: gh,
        epochs_trained: gps_train.epochs_trained,
        final_loss: gps_train.final_loss,
        final_auc: gps_train.final_auc,
        hidden_dim: config.hidden_dim,
        timestamp: "pipeline".into(),
    };
    if save_model(&gps_model, "gps", gh, &gps_meta, &device).is_ok() {
        report.models_saved.push("gps".into());
    }
    all_embeddings.insert("gps".into(), gps_emb);

    // ── 5. HEHRGNN + JEPA (entity embedding training) ──
    // Convert GraphFacts to HehrFacts (map string names → numeric entity IDs)
    let mut entity_name_to_id: HashMap<String, usize> = HashMap::new();
    let mut relation_name_to_id: HashMap<String, usize> = HashMap::new();

    for fact in facts {
        let src_key = format!("{}:{}", fact.src.0, fact.src.1);
        let dst_key = format!("{}:{}", fact.dst.0, fact.dst.1);
        let src_len = entity_name_to_id.len();
        entity_name_to_id.entry(src_key).or_insert(src_len);
        let dst_len = entity_name_to_id.len();
        entity_name_to_id.entry(dst_key).or_insert(dst_len);
        let rel_len = relation_name_to_id.len();
        relation_name_to_id
            .entry(fact.relation.clone())
            .or_insert(rel_len);
    }

    let num_entities = entity_name_to_id.len().max(2);
    let num_relations = relation_name_to_id.len().max(1);

    // Build HehrFactItems from GraphFacts
    let hehr_items: Vec<crate::data::batcher::HehrFactItem> = facts
        .iter()
        .map(|f| {
            let src_key = format!("{}:{}", f.src.0, f.src.1);
            let dst_key = format!("{}:{}", f.dst.0, f.dst.1);
            crate::data::batcher::HehrFactItem {
                fact: crate::data::fact::HehrFact {
                    head: entity_name_to_id[&src_key],
                    relation: relation_name_to_id[&f.relation],
                    tail: entity_name_to_id[&dst_key],
                    qualifiers: vec![],
                },
                label: 1.0,
            }
        })
        .collect();

    // Create batch
    let hehr_batcher = crate::data::batcher::HehrBatcher::new();
    let hehr_batch =
        burn::data::dataloader::batcher::Batcher::batch(&hehr_batcher, hehr_items, &device);

    // Init HEHRGNN model
    let mut hehrgnn_model = crate::model::hehrgnn::HehrgnnModelConfig {
        num_entities,
        num_relations,
        hidden_dim: config.hidden_dim,
        num_layers: 2,
        dropout: 0.0,
    }
    .init::<B>(&device);

    // Train with JEPA (InfoNCE + uniformity on entity embeddings)
    let hehr_report = crate::model::jepa::train_hehrgnn_jepa(
        &mut hehrgnn_model,
        &hehr_batch,
        config.train_config.epochs,
        config.train_config.lr as f32,
        0.3, // uniformity weight
    );
    eprintln!(
        "  [pipeline] HEHRGNN trained: infonce={:.4}→{:.4} (JEPA, {} entities)",
        hehr_report.initial_loss, hehr_report.final_loss, num_entities
    );

    // Extract entity embeddings as PlainEmbeddings
    let hehr_output = hehrgnn_model.forward(&hehr_batch);
    let entity_emb_data: Vec<f32> = hehr_output
        .entity_emb
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec();
    let mut hehr_emb_data = HashMap::new();
    let mut entity_vecs = Vec::new();
    for i in 0..num_entities {
        entity_vecs
            .push(entity_emb_data[i * config.hidden_dim..(i + 1) * config.hidden_dim].to_vec());
    }
    hehr_emb_data.insert("entity".to_string(), entity_vecs);
    all_embeddings.insert(
        "hehrgnn".into(),
        PlainEmbeddings {
            data: hehr_emb_data,
        },
    );
    report.models_saved.push("hehrgnn".into());

    // ── 5b. Stable-GNN: Cross-Model Feature Decorrelation ──
    // Decorrelate embeddings from SAGE, RGCN, GAT, GPS to eliminate
    // spurious correlations and improve OOD generalization.
    {
        use crate::model::stable_decorrelation::{decorrelate_ensemble, StableConfig};

        // Build per-model embedding maps for decorrelation
        let mut model_emb_maps: HashMap<String, HashMap<String, Vec<Vec<f32>>>> = HashMap::new();
        for (model_name, plain_emb) in &all_embeddings {
            if model_name == "hehrgnn" {
                continue; // Skip HEHRGNN (different entity space)
            }
            model_emb_maps.insert(model_name.clone(), plain_emb.data.clone());
        }

        if model_emb_maps.len() >= 2 {
            let stable_config = StableConfig {
                rff_dim: config.hidden_dim * 2,
                ..StableConfig::default()
            };

            let decor_result = decorrelate_ensemble(&model_emb_maps, &stable_config);

            eprintln!(
                "  [pipeline] Stable-GNN decorrelation: {:.6} → {:.6} ({:.1}% reduction, {} models)",
                decor_result.initial_loss,
                decor_result.decorrelation_loss,
                decor_result.improvement_ratio() * 100.0,
                decor_result.models_decorrelated,
            );
        }
    }

    // ── 6. GEPA Auto-Tune: self-improve fiduciary weights on every run ──
    // Uses real SAGE embeddings + anomaly scores from this pipeline run.
    // Runs a few quick NumericMutator evals (~50ms each), no LLM needed.
    {
        use crate::eval::fiduciary::*;
        use crate::optimizer::gepa;

        let weights_path = gepa::GEPA_WEIGHTS_PATH;

        // Use selected backbone embeddings for fiduciary evaluation.
        let primary_key = embedding_key_for_backbone(selected_backbone);
        let model_label = selected_backbone.as_str().to_ascii_uppercase();
        if let Some(primary_embs) = all_embeddings.get(primary_key) {
            // Build anomaly scores from a cross-backbone consensus (embedding norms as proxy).
            let mut anomaly_sum: HashMap<String, Vec<f32>> = HashMap::new();
            let mut anomaly_count: HashMap<String, Vec<f32>> = HashMap::new();
            for key in ["graphsage", "rgcn", "gat", "gps"] {
                if let Some(model_embs) = all_embeddings.get(key) {
                    for (node_type, vecs) in &model_embs.data {
                        let sums = anomaly_sum
                            .entry(node_type.clone())
                            .or_insert_with(|| vec![0.0; vecs.len()]);
                        let counts = anomaly_count
                            .entry(node_type.clone())
                            .or_insert_with(|| vec![0.0; vecs.len()]);
                        if sums.len() < vecs.len() {
                            sums.resize(vecs.len(), 0.0);
                            counts.resize(vecs.len(), 0.0);
                        }
                        for (i, v) in vecs.iter().enumerate() {
                            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                            let score = (norm / (config.hidden_dim as f32).sqrt()).clamp(0.0, 1.0);
                            sums[i] += score;
                            counts[i] += 1.0;
                        }
                    }
                }
            }
            let mut anomaly_map: HashMap<String, Vec<f32>> = HashMap::new();
            for (node_type, sums) in anomaly_sum {
                let counts = anomaly_count
                    .remove(&node_type)
                    .unwrap_or_else(|| vec![1.0; sums.len()]);
                let avg: Vec<f32> = sums
                    .iter()
                    .enumerate()
                    .map(|(i, s)| {
                        let c = counts.get(i).copied().unwrap_or(1.0).max(1e-6);
                        (s / c).clamp(0.0, 1.0)
                    })
                    .collect();
                anomaly_map.insert(node_type, avg);
            }
            if anomaly_map.is_empty() {
                for (node_type, vecs) in &primary_embs.data {
                    let scores: Vec<f32> = vecs
                        .iter()
                        .map(|v| {
                            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                            (norm / (config.hidden_dim as f32).sqrt()).clamp(0.0, 1.0)
                        })
                        .collect();
                    anomaly_map.insert(node_type.clone(), scores);
                }
            }
            let mut anomaly_scores_map: HashMap<String, HashMap<String, Vec<f32>>> = HashMap::new();
            anomaly_scores_map.insert(model_label, anomaly_map.clone());

            // Build node names/counts from embeddings
            let mut emb_data: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
            let mut node_names_map: HashMap<String, Vec<String>> = HashMap::new();
            let mut node_counts_map: HashMap<String, usize> = HashMap::new();

            for (node_type, vecs) in &primary_embs.data {
                emb_data.insert(node_type.clone(), vecs.clone());
                let type_names: Vec<String> = (0..vecs.len())
                    .map(|i| format!("{}_{}", node_type, i))
                    .collect();
                node_counts_map.insert(node_type.clone(), vecs.len());
                node_names_map.insert(node_type.clone(), type_names);
            }

            // Build edges from facts
            let mut edge_map: HashMap<(String, String, String), Vec<(usize, usize)>> =
                HashMap::new();
            let mut fact_name_to_id: HashMap<(String, String), usize> = HashMap::new();

            for fact in facts {
                let src_key = (fact.src.0.clone(), fact.src.1.clone());
                let src_id = {
                    let names = node_names_map
                        .entry(fact.src.0.clone())
                        .or_insert_with(Vec::new);
                    *fact_name_to_id.entry(src_key).or_insert_with(|| {
                        let id = names.len();
                        names.push(fact.src.1.clone());
                        id
                    })
                };
                let dst_key = (fact.dst.0.clone(), fact.dst.1.clone());
                let dst_id = {
                    let names = node_names_map
                        .entry(fact.dst.0.clone())
                        .or_insert_with(Vec::new);
                    *fact_name_to_id.entry(dst_key).or_insert_with(|| {
                        let id = names.len();
                        names.push(fact.dst.1.clone());
                        id
                    })
                };
                edge_map
                    .entry((
                        fact.src.0.clone(),
                        fact.relation.clone(),
                        fact.dst.0.clone(),
                    ))
                    .or_insert_with(Vec::new)
                    .push((src_id, dst_id));
            }

            // Update node counts from the fact-based name map
            for (nt, names) in &node_names_map {
                node_counts_map.insert(nt.clone(), names.len());
            }

            // Find user type and create evaluator
            let user_type = if node_names_map.contains_key("user") {
                "user"
            } else if node_names_map.contains_key("persona") {
                "persona"
            } else {
                "" // Will skip auto-tune if no user type found
            };

            if !user_type.is_empty() && emb_data.contains_key(user_type) {
                let user_count = emb_data.get(user_type).map(|v| v.len()).unwrap_or(0);

                if user_count > 0 {
                    // Create a ranking evaluator: for each user, run recommend()
                    // and score by how well anomalous entities rank above safe ones
                    struct PipelineEvaluator {
                        emb_data: HashMap<String, Vec<Vec<f32>>>,
                        anomaly_scores: HashMap<String, HashMap<String, Vec<f32>>>,
                        edge_map: HashMap<(String, String, String), Vec<(usize, usize)>>,
                        node_names: HashMap<String, Vec<String>>,
                        node_counts: HashMap<String, usize>,
                        user_type: String,
                        user_count: usize,
                        hidden_dim: usize,
                    }

                    impl gepa::Evaluator for PipelineEvaluator {
                        fn evaluate(&self, candidate: &gepa::Candidate) -> gepa::EvalResult {
                            let _weights = [
                                candidate.get_f32("cost_weight", 0.25),
                                candidate.get_f32("risk_weight", 0.25),
                                candidate.get_f32("goal_weight", 0.15),
                                candidate.get_f32("urgency_weight", 0.15),
                                0.10, // conflict
                                0.10, // reversibility
                            ];

                            let mut total_score = 0.0f64;
                            let mut user_eval_count = 0;

                            for uid in 0..self.user_count.min(5) {
                                let user_emb = self
                                    .emb_data
                                    .get(&self.user_type)
                                    .and_then(|v| v.get(uid))
                                    .cloned()
                                    .unwrap_or(vec![0.0; self.hidden_dim]);

                                let ctx = FiduciaryContext {
                                    user_emb: &user_emb,
                                    embeddings: &self.emb_data,
                                    anomaly_scores: &self.anomaly_scores,
                                    edges: &self.edge_map,
                                    node_names: &self.node_names,
                                    node_counts: &self.node_counts,
                                    user_type: self.user_type.clone(),
                                    user_id: uid,
                                    hidden_dim: self.hidden_dim,
                                };

                                let resp = recommend(&ctx, None);
                                if resp.recommendations.is_empty() {
                                    continue;
                                }

                                // Score: do high-score recs correspond to high-anomaly targets?
                                let recs = &resp.recommendations;
                                let rec_count = recs.len();
                                let top_third = rec_count / 3;

                                // Metric 1: Top-third recommendations should have higher scores
                                let top_avg: f64 = recs[..top_third.max(1)]
                                    .iter()
                                    .map(|r| r.fiduciary_score as f64)
                                    .sum::<f64>()
                                    / top_third.max(1) as f64;

                                // Metric 2: Score separation between recommendable and non-recommendable
                                let rec_scores: Vec<f64> = recs
                                    .iter()
                                    .filter(|r| r.is_recommended)
                                    .map(|r| r.fiduciary_score as f64)
                                    .collect();
                                let non_rec_scores: Vec<f64> = recs
                                    .iter()
                                    .filter(|r| !r.is_recommended)
                                    .map(|r| r.fiduciary_score as f64)
                                    .collect();

                                let separation =
                                    if !rec_scores.is_empty() && !non_rec_scores.is_empty() {
                                        let avg_rec = rec_scores.iter().sum::<f64>()
                                            / rec_scores.len() as f64;
                                        let avg_non = non_rec_scores.iter().sum::<f64>()
                                            / non_rec_scores.len() as f64;
                                        (avg_rec - avg_non).max(0.0)
                                    } else {
                                        0.0
                                    };

                                total_score += top_avg * 0.6 + separation * 0.4;
                                user_eval_count += 1;
                            }

                            let score = if user_eval_count > 0 {
                                total_score / user_eval_count as f64
                            } else {
                                0.0
                            };

                            let mut si = gepa::SideInfo::new();
                            si.score("users_evaluated", user_eval_count as f64);
                            gepa::EvalResult {
                                score,
                                side_info: si,
                            }
                        }
                    }

                    let evaluator = PipelineEvaluator {
                        emb_data,
                        anomaly_scores: anomaly_scores_map,
                        edge_map,
                        node_names: node_names_map,
                        node_counts: node_counts_map,
                        user_type: user_type.to_string(),
                        user_count,
                        hidden_dim: config.hidden_dim,
                    };

                    let tune_report = gepa::auto_tune_weights(&evaluator, weights_path, 5);
                    report.gepa_auto_tune = Some(tune_report);
                }
            }
        }
    }

    (report, all_embeddings)
}

/// Load or create a LearnableScorer, saving after creation.
pub fn get_or_create_scorer(graph_hash: u64, config: &ScorerConfig) -> (LearnableScorer, bool) {
    if let Some(scorer) = load_scorer(graph_hash) {
        eprintln!(
            "  [pipeline] Loaded scorer checkpoint (samples_seen={})",
            scorer.samples_seen
        );
        (scorer, true)
    } else {
        let scorer = LearnableScorer::new(config);
        eprintln!("  [pipeline] Created fresh scorer");
        (scorer, false)
    }
}

/// Save the scorer after training.
pub fn persist_scorer(scorer: &LearnableScorer, graph_hash: u64) -> bool {
    match save_scorer(scorer, graph_hash) {
        Ok(()) => {
            eprintln!("  [pipeline] Saved scorer checkpoint");
            true
        }
        Err(e) => {
            eprintln!("  [pipeline] Failed to save scorer: {}", e);
            false
        }
    }
}
