//! HTTP request handlers for the GNN prediction API.
//!
//! All handlers use pre-computed plain embeddings (no Burn tensors).
//! Scoring uses basic vector math (dot product, cosine similarity, L2).

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};

use super::state::{AppState, PlainEmbeddings};

// ---------------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub total_nodes: usize,
    pub total_edges: usize,
    pub node_types: Vec<String>,
    pub hidden_dim: usize,
}

pub async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
        total_nodes: state.graph_meta.total_nodes,
        total_edges: state.graph_meta.total_edges,
        node_types: state.graph_meta.node_types.clone(),
        hidden_dim: state.hidden_dim,
    })
}

// ---------------------------------------------------------------------------
// Get embedding
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct EmbeddingRequest {
    pub node_type: String,
    pub node_id: usize,
}

#[derive(Serialize)]
pub struct EmbeddingResponse {
    pub node_type: String,
    pub node_id: usize,
    pub embedding: Vec<f32>,
    pub dim: usize,
}

pub async fn get_embedding(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, (StatusCode, String)> {
    let nodes = state
        .embeddings
        .data
        .get(&req.node_type)
        .ok_or((StatusCode::NOT_FOUND, format!("Node type '{}' not found", req.node_type)))?;

    let emb = nodes
        .get(req.node_id)
        .ok_or((StatusCode::BAD_REQUEST, format!("node_id {} out of range", req.node_id)))?;

    Ok(Json(EmbeddingResponse {
        node_type: req.node_type,
        node_id: req.node_id,
        embedding: emb.clone(),
        dim: state.hidden_dim,
    }))
}

// ---------------------------------------------------------------------------
// Match ranking (link prediction)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct MatchRankRequest {
    pub src_type: String,
    pub src_id: usize,
    pub dst_type: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

fn default_top_k() -> usize { 5 }

#[derive(Serialize)]
pub struct MatchRankResponse {
    pub src: NodeRef,
    pub matches: Vec<ScoredMatch>,
}

#[derive(Serialize)]
pub struct NodeRef {
    pub node_type: String,
    pub node_id: usize,
    pub name: String,
}

#[derive(Serialize)]
pub struct ScoredMatch {
    pub node_id: usize,
    pub name: String,
    pub score: f32,
    pub rank: usize,
}

pub async fn rank_matches(
    State(state): State<Arc<AppState>>,
    Json(req): Json<MatchRankRequest>,
) -> Result<Json<MatchRankResponse>, (StatusCode, String)> {
    let src_nodes = state
        .embeddings
        .data
        .get(&req.src_type)
        .ok_or((StatusCode::NOT_FOUND, format!("Source type '{}' not found", req.src_type)))?;

    let dst_nodes = state
        .embeddings
        .data
        .get(&req.dst_type)
        .ok_or((StatusCode::NOT_FOUND, format!("Dest type '{}' not found", req.dst_type)))?;

    let src_emb = src_nodes
        .get(req.src_id)
        .ok_or((StatusCode::BAD_REQUEST, "src_id out of range".into()))?;

    // Score all destination candidates using dot product
    let mut scored: Vec<(usize, f32)> = dst_nodes
        .iter()
        .enumerate()
        .map(|(id, dst_emb)| (id, PlainEmbeddings::dot_score(src_emb, dst_emb)))
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_k = req.top_k.min(scored.len());
    let dst_names = state.node_names.get(&req.dst_type);

    let matches: Vec<ScoredMatch> = scored[..top_k]
        .iter()
        .enumerate()
        .map(|(rank, &(id, score))| ScoredMatch {
            node_id: id,
            name: dst_names
                .and_then(|n| n.get(id))
                .cloned()
                .unwrap_or_else(|| format!("{}_{}", req.dst_type, id)),
            score,
            rank: rank + 1,
        })
        .collect();

    let src_name = state
        .node_names
        .get(&req.src_type)
        .and_then(|n| n.get(req.src_id))
        .cloned()
        .unwrap_or_else(|| format!("{}_{}", req.src_type, req.src_id));

    Ok(Json(MatchRankResponse {
        src: NodeRef {
            node_type: req.src_type,
            node_id: req.src_id,
            name: src_name,
        },
        matches,
    }))
}

// ---------------------------------------------------------------------------
// Node classification
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct ClassifyRequest {
    pub node_type: String,
    pub node_ids: Vec<usize>,
}

#[derive(Serialize)]
pub struct ClassifyResponse {
    pub predictions: Vec<ClassPrediction>,
}

#[derive(Serialize)]
pub struct ClassPrediction {
    pub node_id: usize,
    pub predicted_class: usize,
    pub confidence: f32,
}

pub async fn classify_nodes(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ClassifyRequest>,
) -> Result<Json<ClassifyResponse>, (StatusCode, String)> {
    let nodes = state
        .embeddings
        .data
        .get(&req.node_type)
        .ok_or((StatusCode::NOT_FOUND, format!("Node type '{}' not found", req.node_type)))?;

    // Simple classification: hash embedding to class bucket
    let predictions: Vec<ClassPrediction> = req
        .node_ids
        .iter()
        .map(|&node_id| {
            let emb = nodes.get(node_id).map(|e| e.as_slice()).unwrap_or(&[]);

            // Use embedding magnitude to pick a class (placeholder approach)
            let magnitude: f32 = emb.iter().map(|x| x.abs()).sum::<f32>() / emb.len().max(1) as f32;
            let class = ((magnitude * 1000.0) as usize) % state.num_classes;
            let confidence = 1.0 / state.num_classes as f32 + magnitude.fract() * 0.3;

            ClassPrediction {
                node_id,
                predicted_class: class,
                confidence: confidence.min(1.0),
            }
        })
        .collect();

    Ok(Json(ClassifyResponse { predictions }))
}

// ---------------------------------------------------------------------------
// Anomaly scoring
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct AnomalyRequest {
    pub node_type: String,
    pub node_ids: Vec<usize>,
}

#[derive(Serialize)]
pub struct AnomalyResponse {
    pub scores: Vec<AnomalyResult>,
    pub threshold: f32,
}

#[derive(Serialize)]
pub struct AnomalyResult {
    pub node_id: usize,
    pub anomaly_score: f32,
    pub is_anomalous: bool,
}

pub async fn score_anomalies(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AnomalyRequest>,
) -> Result<Json<AnomalyResponse>, (StatusCode, String)> {
    let nodes = state
        .embeddings
        .data
        .get(&req.node_type)
        .ok_or((StatusCode::NOT_FOUND, format!("Node type '{}' not found", req.node_type)))?;

    // Compute mean embedding for this type
    let dim = state.hidden_dim;
    let mut mean_emb = vec![0.0f32; dim];
    for node in nodes.iter() {
        for (i, &v) in node.iter().enumerate() {
            mean_emb[i] += v;
        }
    }
    let n = nodes.len() as f32;
    for v in mean_emb.iter_mut() {
        *v /= n;
    }

    // Anomaly score = L2 distance from mean
    let mut all_scores: Vec<f32> = Vec::new();
    let mut results: Vec<AnomalyResult> = Vec::new();

    for &node_id in &req.node_ids {
        let emb = nodes.get(node_id).map(|e| e.as_slice()).unwrap_or(&[]);
        let score = PlainEmbeddings::l2_distance(emb, &mean_emb);
        all_scores.push(score);
        results.push(AnomalyResult {
            node_id,
            anomaly_score: score,
            is_anomalous: false, // set below
        });
    }

    // Threshold: mean + 2*std
    let mean_score: f32 = all_scores.iter().sum::<f32>() / all_scores.len().max(1) as f32;
    let std_score: f32 = (all_scores
        .iter()
        .map(|s| (s - mean_score).powi(2))
        .sum::<f32>()
        / all_scores.len().max(1) as f32)
        .sqrt();
    let threshold = mean_score + 2.0 * std_score;

    for r in results.iter_mut() {
        r.is_anomalous = r.anomaly_score > threshold;
    }

    Ok(Json(AnomalyResponse {
        scores: results,
        threshold,
    }))
}

// ---------------------------------------------------------------------------
// Similarity search
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct SimilarityRequest {
    pub node_type: String,
    pub node_id: usize,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

#[derive(Serialize)]
pub struct SimilarityResponse {
    pub query: NodeRef,
    pub similar: Vec<ScoredMatch>,
}

pub async fn similarity_search(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SimilarityRequest>,
) -> Result<Json<SimilarityResponse>, (StatusCode, String)> {
    let nodes = state
        .embeddings
        .data
        .get(&req.node_type)
        .ok_or((StatusCode::NOT_FOUND, format!("Node type '{}' not found", req.node_type)))?;

    let query_emb = nodes
        .get(req.node_id)
        .ok_or((StatusCode::BAD_REQUEST, "node_id out of range".into()))?;

    // Cosine similarity against all same-type nodes
    let mut scored: Vec<(usize, f32)> = nodes
        .iter()
        .enumerate()
        .filter(|&(id, _)| id != req.node_id)
        .map(|(id, emb)| (id, PlainEmbeddings::cosine_similarity(query_emb, emb)))
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_k = req.top_k.min(scored.len());
    let names = state.node_names.get(&req.node_type);

    let similar: Vec<ScoredMatch> = scored[..top_k]
        .iter()
        .enumerate()
        .map(|(rank, &(id, score))| ScoredMatch {
            node_id: id,
            name: names
                .and_then(|n| n.get(id))
                .cloned()
                .unwrap_or_else(|| format!("{}_{}", req.node_type, id)),
            score,
            rank: rank + 1,
        })
        .collect();

    let query_name = names
        .and_then(|n| n.get(req.node_id))
        .cloned()
        .unwrap_or_else(|| format!("{}_{}", req.node_type, req.node_id));

    Ok(Json(SimilarityResponse {
        query: NodeRef {
            node_type: req.node_type,
            node_id: req.node_id,
            name: query_name,
        },
        similar,
    }))
}

// ---------------------------------------------------------------------------
// Graph info
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct GraphInfoResponse {
    pub node_types: Vec<NodeTypeInfo>,
    pub edge_types: Vec<EdgeTypeInfo>,
    pub total_nodes: usize,
    pub total_edges: usize,
}

#[derive(Serialize)]
pub struct NodeTypeInfo {
    pub name: String,
    pub count: usize,
}

#[derive(Serialize)]
pub struct EdgeTypeInfo {
    pub src_type: String,
    pub relation: String,
    pub dst_type: String,
    pub count: usize,
}

pub async fn graph_info(State(state): State<Arc<AppState>>) -> Json<GraphInfoResponse> {
    let node_types: Vec<NodeTypeInfo> = state
        .graph_meta
        .node_types
        .iter()
        .map(|nt| NodeTypeInfo {
            name: nt.clone(),
            count: state.graph_meta.node_counts.get(nt).copied().unwrap_or(0),
        })
        .collect();

    let edge_types: Vec<EdgeTypeInfo> = state
        .graph_meta
        .edge_types
        .iter()
        .map(|et| EdgeTypeInfo {
            src_type: et.0.clone(),
            relation: et.1.clone(),
            dst_type: et.2.clone(),
            count: state.graph_meta.edge_counts.get(et).copied().unwrap_or(0),
        })
        .collect();

    Json(GraphInfoResponse {
        total_nodes: state.graph_meta.total_nodes,
        total_edges: state.graph_meta.total_edges,
        node_types,
        edge_types,
    })
}
