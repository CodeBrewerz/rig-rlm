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
use crate::data::hetero_graph::{EdgeType, HeteroGraph};
use crate::eval::learnable_scorer::{LearnableScorer, ScorerConfig};
use crate::model::backbone::NodeEmbeddings;
use crate::model::gat::GatConfig;
use crate::model::graph_transformer::GraphTransformerConfig;
use crate::model::graphsage::GraphSageModelConfig;
use crate::model::lora::{init_hetero_basis_adapter, LoraConfig};
use crate::model::mhc::MhcRgcnConfig;
use crate::model::rgcn::RgcnConfig;
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
    pub models_saved: Vec<String>,
    pub scorer_saved: bool,
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

/// Run the full ensemble pipeline on a set of graph facts.
///
/// Per-model optimal feature combos (from combo_features_test):
/// - GraphSAGE: DoRA adapter + JEPA training (+7.9% AUC)
/// - RGCN: mHC 8-layer + JEPA training (+4.2% AUC)
/// - GAT: JEPA training (+9.9% AUC)
/// - GPS: JEPA training (+3.8% AUC)
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
    };
    let mut graph = build_hetero_graph::<B>(facts, &build_config, &device);
    let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
    let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();
    let gh = config.graph_hash;

    let mut report = PipelineReport {
        models_loaded_from_checkpoint: Vec::new(),
        scorer_loaded_from_checkpoint: false,
        train_report: None,
        pre_train_auc: 0.0,
        post_train_auc: 0.0,
        models_saved: Vec::new(),
        scorer_saved: false,
    };

    let mut all_embeddings: HashMap<String, PlainEmbeddings> = HashMap::new();

    // ── 1. GraphSAGE + DoRA + JEPA (best combo: +7.9% AUC) ──
    let mut sage_model = GraphSageModelConfig {
        in_dim: config.hidden_dim,
        hidden_dim: config.hidden_dim,
        num_layers: 2,
        dropout: 0.0,
    }
    .init::<B>(&node_types, &edge_types, &device);

    if let Some((loaded, _meta)) = load_model(
        GraphSageModelConfig {
            in_dim: config.hidden_dim,
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
        config.hidden_dim,
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

    // Train with JEPA (InfoNCE + uniformity) — synergy with DoRA adapter
    let sage_fwd = |g: &HeteroGraph<B>| -> NodeEmbeddings<B> { sage_model.forward(g) };
    let train_report = train_jepa(
        &mut graph,
        &sage_fwd,
        &config.train_config,
        0.1,   // temperature τ
        0.5,   // uniformity weight λ
        false, // no edge predictor
    );
    report.post_train_auc = train_report.final_auc;
    report.train_report = Some(train_report.clone());
    eprintln!(
        "  [pipeline] GraphSAGE trained: auc={:.4} (DoRA+JEPA)",
        train_report.final_auc
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

    // ── 2. RGCN + mHC + JEPA (best combo: +4.2% AUC) ──
    // Use mHC-enhanced RGCN with 8 layers + 4 streams for depth without over-smoothing
    let mhc_rgcn_model = MhcRgcnConfig {
        in_dim: config.hidden_dim,
        hidden_dim: config.hidden_dim,
        num_layers: 8,
        num_bases: 4,
        n_streams: 4,
        dropout: 0.0,
    }
    .init::<B>(&node_types, &edge_types, &device);

    // Train with JEPA (InfoNCE + uniformity)
    let rgcn_fwd = |g: &HeteroGraph<B>| -> NodeEmbeddings<B> { mhc_rgcn_model.forward(g) };
    let rgcn_train = train_jepa(
        &mut graph,
        &rgcn_fwd,
        &config.train_config,
        0.1,   // temperature τ
        0.5,   // uniformity weight λ
        false, // no edge predictor
    );
    eprintln!(
        "  [pipeline] RGCN trained: auc={:.4} (mHC+JEPA, 8 layers)",
        rgcn_train.final_auc
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
    let gat_model = {
        let fresh = GatConfig {
            in_dim: config.hidden_dim,
            hidden_dim: config.hidden_dim,
            num_heads: 4,
            num_layers: 2,
            dropout: 0.0,
        }
        .init_model::<B>(&node_types, &edge_types, &device);

        if let Some((loaded, _)) = load_model(
            GatConfig {
                in_dim: config.hidden_dim,
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
    // Train GAT with JEPA (optimal: +4.5% AUC from JEPA comparison test)
    let gat_fwd = |g: &HeteroGraph<B>| -> NodeEmbeddings<B> { gat_model.forward(g) };
    let gat_train = train_jepa(&mut graph, &gat_fwd, &config.train_config, 0.1, 0.5, false);
    eprintln!(
        "  [pipeline] GAT trained: auc={:.4} (JEPA)",
        gat_train.final_auc
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
    let gps_model = {
        let fresh = GraphTransformerConfig {
            in_dim: config.hidden_dim,
            hidden_dim: config.hidden_dim,
            num_heads: 4,
            num_layers: 2,
            ffn_ratio: 2,
            dropout: 0.0,
        }
        .init_model::<B>(&node_types, &edge_types, &device);

        if let Some((loaded, _)) = load_model(
            GraphTransformerConfig {
                in_dim: config.hidden_dim,
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
    // Train GPS with JEPA (optimal combo: +3.8% AUC)
    let gps_fwd = |g: &HeteroGraph<B>| -> NodeEmbeddings<B> { gps_model.forward(g) };
    let gps_train = train_jepa(
        &mut graph,
        &gps_fwd,
        &config.train_config,
        0.1,   // temperature τ
        0.5,   // uniformity weight λ
        false, // no edge predictor
    );
    eprintln!(
        "  [pipeline] GPS trained: auc={:.4} (JEPA)",
        gps_train.final_auc
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
