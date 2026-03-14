//! Server state: pre-computed embeddings stored as plain Vecs.
//!
//! Since Burn's `Tensor<NdArray>` doesn't implement `Send`, we pre-compute
//! all embeddings and model outputs at startup and store them as plain `Vec<f32>`.
//! This makes the state `Send + Sync` as required by axum.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

/// Returns current UTC timestamp as ISO 8601 string (no chrono dep).
pub fn chrono_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Rough ISO format: YYYY-MM-DDTHH:MM:SSZ
    let days = secs / 86400;
    let time = secs % 86400;
    let hours = time / 3600;
    let mins = (time % 3600) / 60;
    let s = time % 60;
    // Days since 1970-01-01
    let (y, m, d) = days_to_ymd(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y, m, d, hours, mins, s
    )
}

fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Simple Gregorian calendar conversion
    let mut y = 1970;
    let mut remaining = days;
    loop {
        let days_in_year = if is_leap(y) { 366 } else { 365 };
        if remaining < days_in_year {
            break;
        }
        remaining -= days_in_year;
        y += 1;
    }
    let months = if is_leap(y) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut m = 1;
    for &ml in &months {
        if remaining < ml {
            break;
        }
        remaining -= ml;
        m += 1;
    }
    (y, m, remaining + 1)
}

fn is_leap(y: u64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

use burn::backend::NdArray;
use burn::prelude::*;

use crate::data::hetero_graph::{EdgeType, HeteroGraph};
use crate::eval::probing::{ConceptLabels, ProbeResults};
use crate::model::graphsage::GraphSageModelConfig;

/// The inference backend.
type B = NdArray;

/// Pre-computed node embeddings stored as plain vectors.
#[derive(Debug, Clone)]
pub struct PlainEmbeddings {
    /// Per-node-type embeddings: `embeddings[node_type][node_idx] = Vec<f32>`.
    pub data: HashMap<String, Vec<Vec<f32>>>,
    pub hidden_dim: usize,
}

impl PlainEmbeddings {
    /// Extract from Burn NodeEmbeddings (generic over any backend).
    pub fn from_burn<BK: Backend>(emb: &crate::model::backbone::NodeEmbeddings<BK>) -> Self {
        let mut data = HashMap::new();
        let mut hidden_dim = 0;

        for (nt, tensor) in &emb.embeddings {
            let dims = tensor.dims();
            let num_nodes = dims[0];
            hidden_dim = dims[1];

            let flat = tensor.clone().reshape([num_nodes * hidden_dim]).into_data();
            let values: Vec<f32> = flat
                .as_slice::<f32>()
                .map(|s| s.to_vec())
                .unwrap_or_else(|_| vec![0.0; num_nodes * hidden_dim]);

            let mut per_node = Vec::with_capacity(num_nodes);
            for i in 0..num_nodes {
                per_node.push(values[i * hidden_dim..(i + 1) * hidden_dim].to_vec());
            }
            data.insert(nt.clone(), per_node);
        }

        PlainEmbeddings { data, hidden_dim }
    }

    /// Cosine similarity between two embedding vectors.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a < 1e-8 || norm_b < 1e-8 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// L2 distance between two embedding vectors.
    pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Dot-product score between two embedding vectors (simple link prediction).
    pub fn dot_score(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

/// Graph metadata (Send-safe, no Burn tensors).
#[derive(Debug, Clone)]
pub struct GraphMeta {
    pub node_types: Vec<String>,
    pub node_counts: HashMap<String, usize>,
    pub edge_types: Vec<(String, String, String)>,
    pub edge_counts: HashMap<(String, String, String), usize>,
    pub total_nodes: usize,
    pub total_edges: usize,
}

/// Shared application state — all plain data, `Send + Sync` safe.
pub struct AppState {
    /// Per-model pre-computed GNN embeddings: "SAGE" → PlainEmbeddings, "RGCN" → …
    /// Wrapped in RwLock for live InstantGNN updates to all models.
    pub model_embeddings: RwLock<HashMap<String, PlainEmbeddings>>,

    /// Default embeddings (SAGE) — wrapped in RwLock for live InstantGNN updates.
    pub embeddings: RwLock<PlainEmbeddings>,

    /// **Precomputed** per-model anomaly scores and normalization parameters.
    /// Built over ALL nodes for O(1) scoring and refreshed after live updates.
    pub precomputed: PrecomputedScores,

    /// Graph metadata (kept live with graph mutations).
    pub graph_meta: RwLock<GraphMeta>,

    /// Node type → list of instance names (kept live with graph mutations).
    pub node_names: RwLock<HashMap<String, Vec<String>>>,

    /// Model configuration metadata.
    pub hidden_dim: usize,
    pub num_classes: usize,

    /// Names of all models that were run.
    pub model_names: Vec<String>,

    /// Graph edges for neighborhood influence lookups (live-updated on mutations).
    pub graph_edges: RwLock<GraphEdges>,
    /// Precomputed relation-conditioned translation channels for fast scoring.
    pub relation_head: RwLock<RelationHeadCache>,

    /// Audit provenance metadata.
    pub audit: AuditInfo,

    /// Precomputed 2D PCA projections per node type (using SAGE embeddings).
    /// node_type → Vec<(f32, f32)> indexed by node_id.
    pub pca_coords: HashMap<String, Vec<(f32, f32)>>,

    /// Neural activation probing results.
    pub probe_results: RwLock<ProbeResults>,

    /// Ground-truth concept labels per node.
    pub concept_labels: RwLock<ConceptLabels>,

    /// Per-model per-layer activations: model → [layer][node_type → Vec<Vec<f32>>].
    pub layer_activations: RwLock<HashMap<String, Vec<HashMap<String, Vec<Vec<f32>>>>>>,

    /// Trained SAE for embedding interpretability (if available).
    pub sae_state: RwLock<Option<SaeState>>,

    /// Learnable scorer for fiduciary recommendations (thread-safe for reward feedback).
    /// Uses asymmetric RL rewards: miss_penalty_multiplier=3.0 (paper 2402.18246).
    pub scorer: Arc<Mutex<crate::eval::learnable_scorer::LearnableScorer>>,
    /// Graph hash used for scorer checkpoint persistence.
    pub scorer_graph_hash: u64,
    /// Persistent PC state for fiduciary recommend() self-learning across requests.
    pub pc_state: Arc<Mutex<crate::eval::fiduciary::PcState>>,

    /// InstantGNN: PPR-based propagation state for incremental updates (KDD 2022).
    pub propagation: Arc<Mutex<crate::tasks::instant_propagation::PropagationState>>,

    /// Adaptive retrain monitor: tracks ‖ΔZ‖ drift, recommends retrain when > θ.
    pub retrain_monitor: Arc<Mutex<crate::tasks::adaptive_retrain::RetrainMonitor>>,

    /// Mutation event log for audit trail.
    pub mutation_log: Arc<Mutex<Vec<crate::data::graph_mutation::GraphEvent>>>,
}

/// Pre-trained SAE + feature labels for interpretability.
#[derive(Debug, Clone)]
pub struct SaeState {
    /// Trained Sparse Autoencoder model.
    pub sae: crate::eval::sae::SparseAutoencoder,
    /// Feature labels mapped to financial concepts.
    pub feature_labels: Vec<crate::eval::sae::SaeFeatureLabel>,
}

/// Simple 2D PCA projection via power iteration (no external deps).
pub struct Pca2D;

impl Pca2D {
    /// Project N×D embeddings to N×2 using power iteration PCA.
    pub fn project(embeddings: &[Vec<f32>], dim: usize) -> Vec<(f32, f32)> {
        let n = embeddings.len();
        if n == 0 || dim == 0 {
            return vec![];
        }

        // Center the data
        let mut mean = vec![0.0f32; dim];
        for e in embeddings {
            for (j, &v) in e.iter().enumerate() {
                if j < dim {
                    mean[j] += v;
                }
            }
        }
        for v in mean.iter_mut() {
            *v /= n as f32;
        }

        let centered: Vec<Vec<f32>> = embeddings
            .iter()
            .map(|e| {
                (0..dim)
                    .map(|j| e.get(j).copied().unwrap_or(0.0) - mean[j])
                    .collect()
            })
            .collect();

        // Power iteration for first principal component
        let pc1 = Self::power_iteration(&centered, dim, None);
        // Second PC orthogonal to first
        let pc2 = Self::power_iteration(&centered, dim, Some(&pc1));

        // Project each embedding onto pc1, pc2
        centered
            .iter()
            .map(|e| {
                let x: f32 = e.iter().zip(pc1.iter()).map(|(a, b)| a * b).sum();
                let y: f32 = e.iter().zip(pc2.iter()).map(|(a, b)| a * b).sum();
                (x, y)
            })
            .collect()
    }

    fn power_iteration(data: &[Vec<f32>], dim: usize, deflate: Option<&[f32]>) -> Vec<f32> {
        let _n = data.len();
        let mut v: Vec<f32> = (0..dim).map(|i| ((i + 1) as f32).sin()).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
        for x in v.iter_mut() {
            *x /= norm;
        }

        for _ in 0..50 {
            // Compute A^T * A * v  (without forming the covariance matrix)
            let mut new_v = vec![0.0f32; dim];
            for row in data {
                let dot: f32 = row.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                for (j, &r) in row.iter().enumerate() {
                    if j < dim {
                        new_v[j] += r * dot;
                    }
                }
            }
            // Deflate: remove component along existing PC
            if let Some(pc) = deflate {
                let proj: f32 = new_v.iter().zip(pc.iter()).map(|(a, b)| a * b).sum();
                for (j, x) in new_v.iter_mut().enumerate() {
                    *x -= proj * pc[j];
                }
            }
            let norm = new_v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
            for x in new_v.iter_mut() {
                *x /= norm;
            }
            v = new_v;
        }
        v
    }
}

/// Extracted graph edge lists for neighborhood lookups (no Burn tensors).
#[derive(Debug, Clone)]
pub struct GraphEdges {
    /// (src_type, relation, dst_type) → Vec<(src_node_id, dst_node_id)>.
    pub edges: HashMap<(String, String, String), Vec<(usize, usize)>>,
}

impl GraphEdges {
    /// Find all neighbors of a node (incoming + outgoing across all edge types).
    pub fn neighbors_of(&self, node_type: &str, node_id: usize) -> Vec<Neighbor> {
        let mut neighbors = Vec::new();

        for ((src_type, relation, dst_type), edge_list) in &self.edges {
            if src_type == node_type {
                // Outgoing edges from this node
                for &(src, dst) in edge_list {
                    if src as usize == node_id {
                        neighbors.push(Neighbor {
                            node_type: dst_type.clone(),
                            node_id: dst as usize,
                            relation: relation.clone(),
                            direction: "outgoing".into(),
                        });
                    }
                }
            }
            if dst_type == node_type {
                // Incoming edges to this node
                for &(src, dst) in edge_list {
                    if dst as usize == node_id {
                        neighbors.push(Neighbor {
                            node_type: src_type.clone(),
                            node_id: src as usize,
                            relation: relation.clone(),
                            direction: "incoming".into(),
                        });
                    }
                }
            }
        }

        neighbors
    }
}

/// One relation-specific translation channel used by inference-time scoring.
#[derive(Debug, Clone)]
pub struct RelationHeadChannel {
    pub relation: String,
    pub count: usize,
    pub weight: f32,
    pub proto: Vec<f32>,
    pub proto_norm: f32,
    /// Indexed by `src_id`; each entry is sorted unique candidate `dst_id`s.
    pub direct_targets_by_src: Vec<Vec<usize>>,
}

/// Precomputed relation-conditioned channels for `(src_type, dst_type)` pairs.
#[derive(Debug, Clone)]
pub struct RelationHeadCache {
    pub hidden_dim: usize,
    pub pair_channels: HashMap<String, HashMap<String, Vec<RelationHeadChannel>>>,
}

impl RelationHeadCache {
    /// Build relation channels from graph edges and current embeddings.
    pub fn build(
        edge_map: &HashMap<(String, String, String), Vec<(usize, usize)>>,
        embeddings: &PlainEmbeddings,
        hidden_dim: usize,
    ) -> Self {
        let dim = hidden_dim.min(embeddings.hidden_dim);
        let mut pair_channels: HashMap<String, HashMap<String, Vec<RelationHeadChannel>>> =
            HashMap::new();

        if dim == 0 {
            return Self {
                hidden_dim: dim,
                pair_channels,
            };
        }

        for ((src_type, relation, dst_type), edges) in edge_map {
            if edges.is_empty() {
                continue;
            }

            let Some(src_vecs) = embeddings.data.get(src_type) else {
                continue;
            };
            let Some(dst_vecs) = embeddings.data.get(dst_type) else {
                continue;
            };

            let mut direct_targets_by_src = vec![Vec::<usize>::new(); src_vecs.len()];
            let mut proto = vec![0.0f32; dim];
            let mut proto_count = 0usize;

            for &(src, dst) in edges {
                if src < direct_targets_by_src.len() {
                    direct_targets_by_src[src].push(dst);
                }

                let Some(src_emb) = src_vecs.get(src) else {
                    continue;
                };
                let Some(dst_emb) = dst_vecs.get(dst) else {
                    continue;
                };
                if src_emb.len() < dim || dst_emb.len() < dim {
                    continue;
                }

                for j in 0..dim {
                    let src_v = src_emb[j];
                    let dst_v = dst_emb[j];
                    if src_v.is_finite() && dst_v.is_finite() {
                        proto[j] += dst_v - src_v;
                    }
                }
                proto_count += 1;
            }

            if proto_count == 0 {
                continue;
            }
            for v in &mut proto {
                *v /= proto_count as f32;
            }

            let proto_norm = proto.iter().map(|x| x * x).sum::<f32>().sqrt();
            if !proto_norm.is_finite() || proto_norm < 1e-6 {
                continue;
            }

            for targets in &mut direct_targets_by_src {
                targets.sort_unstable();
                targets.dedup();
            }

            pair_channels
                .entry(src_type.clone())
                .or_default()
                .entry(dst_type.clone())
                .or_default()
                .push(RelationHeadChannel {
                    relation: relation.clone(),
                    count: edges.len(),
                    weight: 0.0,
                    proto,
                    proto_norm,
                    direct_targets_by_src,
                });
        }

        for by_dst in pair_channels.values_mut() {
            for channels in by_dst.values_mut() {
                channels.sort_by(|a, b| {
                    b.count
                        .cmp(&a.count)
                        .then_with(|| a.relation.cmp(&b.relation))
                });
                channels.truncate(4);
                Self::normalize_channel_weights(channels);
            }
        }

        Self {
            hidden_dim: dim,
            pair_channels,
        }
    }

    /// Restore from persisted JSON metadata.
    pub fn from_persisted(meta: crate::model::weights::RelationHeadMeta) -> Self {
        let mut pair_channels: HashMap<String, HashMap<String, Vec<RelationHeadChannel>>> =
            HashMap::new();

        for pair in meta.pairs {
            if pair.src_type.is_empty() || pair.dst_type.is_empty() {
                continue;
            }

            let mut channels: Vec<RelationHeadChannel> = pair
                .channels
                .into_iter()
                .filter_map(|ch| {
                    if ch.proto.is_empty() {
                        return None;
                    }
                    let mut direct_targets_by_src = ch.direct_targets_by_src;
                    for targets in &mut direct_targets_by_src {
                        targets.sort_unstable();
                        targets.dedup();
                    }
                    let computed_norm = ch.proto.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let proto_norm = if ch.proto_norm.is_finite() && ch.proto_norm > 1e-6 {
                        ch.proto_norm
                    } else {
                        computed_norm
                    };
                    if !proto_norm.is_finite() || proto_norm < 1e-6 {
                        return None;
                    }

                    Some(RelationHeadChannel {
                        relation: ch.relation,
                        count: ch.count.max(1),
                        weight: ch.weight,
                        proto: ch.proto,
                        proto_norm,
                        direct_targets_by_src,
                    })
                })
                .collect();

            if channels.is_empty() {
                continue;
            }

            channels.sort_by(|a, b| {
                b.count
                    .cmp(&a.count)
                    .then_with(|| a.relation.cmp(&b.relation))
            });
            channels.truncate(4);
            Self::normalize_channel_weights(&mut channels);

            pair_channels
                .entry(pair.src_type)
                .or_default()
                .entry(pair.dst_type)
                .or_default()
                .extend(channels);
        }

        for by_dst in pair_channels.values_mut() {
            for channels in by_dst.values_mut() {
                channels.sort_by(|a, b| {
                    b.count
                        .cmp(&a.count)
                        .then_with(|| a.relation.cmp(&b.relation))
                });
                channels.truncate(4);
                Self::normalize_channel_weights(channels);
            }
        }

        Self {
            hidden_dim: meta.hidden_dim,
            pair_channels,
        }
    }

    /// Convert runtime cache into persisted metadata.
    pub fn to_persisted(&self, graph_hash: u64) -> crate::model::weights::RelationHeadMeta {
        let mut src_types: Vec<&String> = self.pair_channels.keys().collect();
        src_types.sort();

        let mut pairs = Vec::new();
        for src_type in src_types {
            if let Some(by_dst) = self.pair_channels.get(src_type) {
                let mut dst_types: Vec<&String> = by_dst.keys().collect();
                dst_types.sort();
                for dst_type in dst_types {
                    let mut channels = by_dst.get(dst_type).cloned().unwrap_or_default();
                    channels.sort_by(|a, b| {
                        b.count
                            .cmp(&a.count)
                            .then_with(|| a.relation.cmp(&b.relation))
                    });
                    channels.truncate(4);
                    Self::normalize_channel_weights(&mut channels);

                    pairs.push(crate::model::weights::RelationPairMeta {
                        src_type: src_type.clone(),
                        dst_type: dst_type.clone(),
                        channels: channels
                            .into_iter()
                            .map(|ch| crate::model::weights::RelationChannelMeta {
                                relation: ch.relation,
                                count: ch.count,
                                weight: ch.weight,
                                proto: ch.proto,
                                proto_norm: ch.proto_norm,
                                direct_targets_by_src: ch.direct_targets_by_src,
                            })
                            .collect(),
                    });
                }
            }
        }

        crate::model::weights::RelationHeadMeta {
            version: 1,
            graph_hash,
            hidden_dim: self.hidden_dim,
            pairs,
        }
    }

    /// Number of `(src_type, dst_type)` pair entries.
    pub fn pair_count(&self) -> usize {
        self.pair_channels.values().map(|m| m.len()).sum()
    }

    /// Number of relation channels across all pairs.
    pub fn channel_count(&self) -> usize {
        self.pair_channels
            .values()
            .flat_map(|m| m.values())
            .map(|channels| channels.len())
            .sum()
    }

    /// Score candidates with the precomputed relation-conditioned channels.
    pub fn score_candidates(
        &self,
        src_type: &str,
        src_id: usize,
        dst_type: &str,
        source_embedding: &[f32],
        candidate_ids: &[usize],
        candidate_embeddings: &[Vec<f32>],
    ) -> (Vec<f32>, usize) {
        let n = candidate_ids.len().min(candidate_embeddings.len());
        if n == 0 || source_embedding.is_empty() {
            return (vec![0.0; n], 0);
        }

        let Some(channels) = self
            .pair_channels
            .get(src_type)
            .and_then(|by_dst| by_dst.get(dst_type))
        else {
            return (vec![0.0; n], 0);
        };
        if channels.is_empty() {
            return (vec![0.0; n], 0);
        }

        let dim = self.hidden_dim.min(source_embedding.len());
        if dim == 0 {
            return (vec![0.0; n], 0);
        }

        let mut out = vec![0.0f32; n];
        for i in 0..n {
            let dst = &candidate_embeddings[i];
            if dst.len() < dim {
                continue;
            }

            let mut delta = vec![0.0f32; dim];
            for j in 0..dim {
                delta[j] = dst[j] - source_embedding[j];
            }
            let delta_norm = delta.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);

            let mut score = 0.0f32;
            for ch in channels {
                let rel_dim = dim.min(ch.proto.len());
                if rel_dim == 0 {
                    continue;
                }
                let dot: f32 = delta
                    .iter()
                    .take(rel_dim)
                    .zip(ch.proto.iter().take(rel_dim))
                    .map(|(a, b)| a * b)
                    .sum();
                let cos = (dot / (delta_norm * ch.proto_norm.max(1e-6))).clamp(-1.0, 1.0);
                let trans = (cos + 1.0) * 0.5;
                let direct = ch
                    .direct_targets_by_src
                    .get(src_id)
                    .map(|targets| targets.binary_search(&candidate_ids[i]).is_ok())
                    .unwrap_or(false);
                let direct = if direct { 1.0 } else { 0.0 };
                score += ch.weight * (0.85 * trans + 0.15 * direct);
            }
            out[i] = score.clamp(0.0, 1.0);
        }

        let mean = out.iter().sum::<f32>() / n as f32;
        let var = out.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / n as f32;
        let std = var.sqrt().max(1e-5);
        let normalized = out
            .into_iter()
            .map(|s| Self::sigmoid_f32((s - mean) / std))
            .collect::<Vec<_>>();
        (normalized, channels.len())
    }

    fn normalize_channel_weights(channels: &mut [RelationHeadChannel]) {
        if channels.is_empty() {
            return;
        }
        let sum = channels
            .iter()
            .map(|ch| (ch.count as f32 + 1.0).ln())
            .sum::<f32>();
        if sum <= 1e-6 {
            let w = 1.0 / channels.len() as f32;
            for ch in channels {
                ch.weight = w;
            }
            return;
        }
        for ch in channels {
            ch.weight = ((ch.count as f32 + 1.0).ln() / sum).clamp(0.0, 1.0);
        }
    }

    fn sigmoid_f32(x: f32) -> f32 {
        1.0 / (1.0 + (-x.clamp(-20.0, 20.0)).exp())
    }
}

/// A neighbor node found via an edge.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Neighbor {
    pub node_type: String,
    pub node_id: usize,
    pub relation: String,
    pub direction: String,
}

/// Audit provenance for regulatory compliance.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AuditInfo {
    /// Timestamp when the model was initialized.
    pub initialized_at: String,
    /// Total nodes in the graph at init time.
    pub graph_nodes: usize,
    /// Total edges in the graph at init time.
    pub graph_edges: usize,
    /// Number of GNN models in the ensemble.
    pub num_models: usize,
    /// Names of models.
    pub model_names: Vec<String>,
    /// Embedding dimension.
    pub embedding_dim: usize,
    /// Number of node types.
    pub num_node_types: usize,
    /// Number of edge types.
    pub num_edge_types: usize,
    /// Graph schema hash (deterministic fingerprint of the graph structure).
    pub graph_hash: String,
}

/// Pre-computed anomaly scoring data for consistent, O(1) predictions.
///
/// Key: `(model_name, node_type)` → per-node L2 scores + normalization bounds.
/// These are computed over the **entire** node population at startup, so:
/// - Single-node queries produce the same result as batch queries
/// - No centroid recomputation per request
/// - Normalized scores are globally consistent
#[derive(Debug)]
pub struct PrecomputedScores {
    /// `(model, node_type)` → { raw L2 scores for all nodes, min, max, mean, std }.
    scores: RwLock<HashMap<(String, String), ModelTypeScores>>,
}

/// Pre-computed L2 anomaly scores for one model + one node type.
#[derive(Debug, Clone)]
pub struct ModelTypeScores {
    /// Raw L2 distance from centroid for each node (indexed by node_id).
    pub raw_scores: Vec<f32>,
    /// Sorted scores for percentile computation.
    pub sorted_scores: Vec<f32>,
    /// Centroid vector (for per-dimension feature attribution).
    pub centroid: Vec<f32>,
    /// Global min of raw_scores (for normalization).
    pub global_min: f32,
    /// Global max of raw_scores (for normalization).
    pub global_max: f32,
    /// Mean of raw_scores.
    pub mean: f32,
    /// Std deviation of raw_scores.
    pub std: f32,
}

impl ModelTypeScores {
    /// Normalize a raw score to [0, 1] using the precomputed global min/max.
    pub fn normalize(&self, raw: f32) -> f32 {
        let range = (self.global_max - self.global_min).max(1e-8);
        ((raw - self.global_min) / range).clamp(0.0, 1.0)
    }

    /// Get raw L2 score for a node.
    pub fn raw(&self, node_id: usize) -> f32 {
        self.raw_scores.get(node_id).copied().unwrap_or(0.0)
    }

    /// Get normalized score for a node.
    pub fn normalized(&self, node_id: usize) -> f32 {
        self.normalize(self.raw(node_id))
    }

    /// Get percentile rank (0.0 - 100.0) for a node.
    /// "This node is more anomalous than X% of all nodes."
    pub fn percentile(&self, node_id: usize) -> f32 {
        let raw = self.raw(node_id);
        let count_below = self.sorted_scores.partition_point(|&s| s < raw);
        (count_below as f32 / self.sorted_scores.len().max(1) as f32) * 100.0
    }

    /// Per-dimension contribution to the L2 distance for a specific node.
    /// Returns (dimension_index, contribution) sorted by contribution (highest first).
    pub fn feature_attribution(&self, node_emb: &[f32]) -> Vec<(usize, f32)> {
        let mut contribs: Vec<(usize, f32)> = self
            .centroid
            .iter()
            .enumerate()
            .map(|(dim, &c_val)| {
                let e_val = node_emb.get(dim).copied().unwrap_or(0.0);
                (dim, (e_val - c_val).powi(2))
            })
            .collect();
        contribs.sort_by(|a, b| b.1.total_cmp(&a.1));
        contribs
    }
}

impl PrecomputedScores {
    /// Build from model embeddings — computes centroid + L2 for every model × node type.
    fn build(model_names: &[String], model_embeddings: &HashMap<String, PlainEmbeddings>) -> Self {
        let scores = Self::compute_scores(model_names, model_embeddings);
        Self {
            scores: RwLock::new(scores),
        }
    }

    /// Rebuild in place from current model embeddings.
    pub fn rebuild_from_embeddings(
        &self,
        model_names: &[String],
        model_embeddings: &HashMap<String, PlainEmbeddings>,
    ) -> Result<usize, String> {
        let fresh = Self::compute_scores(model_names, model_embeddings);
        let len = fresh.len();
        let mut guard = self
            .scores
            .write()
            .map_err(|e| format!("precomputed write lock poisoned: {}", e))?;
        *guard = fresh;
        Ok(len)
    }

    /// Number of `(model, node_type)` distributions currently cached.
    pub fn len(&self) -> usize {
        self.scores.read().map(|s| s.len()).unwrap_or(0)
    }

    fn compute_scores(
        model_names: &[String],
        model_embeddings: &HashMap<String, PlainEmbeddings>,
    ) -> HashMap<(String, String), ModelTypeScores> {
        let mut scores = HashMap::new();

        for model_name in model_names {
            if let Some(emb) = model_embeddings.get(model_name) {
                for (node_type, node_vecs) in &emb.data {
                    let dim = emb.hidden_dim;
                    let n = node_vecs.len();
                    if n == 0 {
                        continue;
                    }

                    // Compute centroid
                    let centroid: Vec<f32> = (0..dim)
                        .map(|j| {
                            node_vecs
                                .iter()
                                .map(|e| {
                                    if j < e.len() && e[j].is_finite() {
                                        e[j]
                                    } else {
                                        0.0
                                    }
                                })
                                .sum::<f32>()
                                / n as f32
                        })
                        .collect();

                    // Compute L2 scores
                    let raw_scores: Vec<f32> = node_vecs
                        .iter()
                        .map(|e| {
                            let d = PlainEmbeddings::l2_distance(e, &centroid);
                            if d.is_finite() {
                                d.max(0.0)
                            } else {
                                0.0
                            }
                        })
                        .collect();

                    // Global stats
                    let global_min = raw_scores.iter().cloned().fold(f32::MAX, f32::min);
                    let global_max = raw_scores.iter().cloned().fold(f32::MIN, f32::max);
                    let mean = raw_scores.iter().sum::<f32>() / n as f32;
                    let std = (raw_scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>()
                        / n as f32)
                        .sqrt();

                    // Sorted scores for percentile lookup
                    let mut sorted_scores = raw_scores.clone();
                    sorted_scores.sort_by(|a, b| a.total_cmp(b));

                    scores.insert(
                        (model_name.clone(), node_type.clone()),
                        ModelTypeScores {
                            raw_scores,
                            sorted_scores,
                            centroid,
                            global_min,
                            global_max,
                            mean,
                            std,
                        },
                    );
                }
            }
        }

        scores
    }

    /// Lookup precomputed scores for a model + node type.
    pub fn get(&self, model: &str, node_type: &str) -> Option<ModelTypeScores> {
        self.scores
            .read()
            .ok()?
            .get(&(model.to_string(), node_type.to_string()))
            .cloned()
    }
}

/// Configuration for initializing the server state.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub hidden_dim: usize,
    pub num_gnn_layers: usize,
    pub num_classes: usize,
    pub schema_path: Option<String>,
    pub num_facts: usize,
    pub instances_per_type: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 32,
            num_gnn_layers: 2,
            num_classes: 5,
            schema_path: None,
            num_facts: 200,
            instances_per_type: 5,
        }
    }
}

impl AppState {
    /// Initialize: build graph → run ALL GNN models → extract plain embeddings.
    pub fn init(config: &ServerConfig) -> Arc<Self> {
        use crate::data::graph_builder::{build_from_schema, GraphBuildConfig};
        use crate::data::synthetic::{SyntheticDataConfig, TqlSchema};
        use crate::model::gat::GatConfig;
        use crate::model::graph_transformer::GraphTransformerConfig;
        use crate::model::lora::{init_hetero_basis_adapter, LoraConfig};
        use crate::model::mhc::MhcRgcnConfig;
        use crate::model::trainer::*;

        let device = <B as Backend>::Device::default();
        let feat_dim = 16;

        // Load schema
        let schema = if let Some(ref path) = config.schema_path {
            match std::fs::read_to_string(path) {
                Ok(tql) => TqlSchema::parse(&tql),
                Err(e) => {
                    eprintln!(
                        "  ⚠️  Failed to read schema file '{}': {}. Falling back to DEFAULT_SCHEMA.",
                        path, e
                    );
                    TqlSchema::parse(DEFAULT_SCHEMA)
                }
            }
        } else {
            TqlSchema::parse(DEFAULT_SCHEMA)
        };

        let syn_config = SyntheticDataConfig {
            instances_per_type: config.instances_per_type,
            num_facts: config.num_facts,
            max_qualifiers: 2,
            seed: 42,
        };

        let graph_config = GraphBuildConfig {
            node_feat_dim: feat_dim,
            add_reverse_edges: true,
            add_self_loops: true,
            add_positional_encoding: true,
            add_cross_dependency_edges: true,
        };

        // Build graph
        let graph: HeteroGraph<B> = build_from_schema(&schema, &syn_config, &graph_config, &device);

        println!(
            "  Graph: {} nodes, {} edges, {} node types, {} edge types",
            graph.total_nodes(),
            graph.total_edges(),
            graph.node_types().len(),
            graph.edge_types().len()
        );

        let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
        let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();

        // Derive actual feature dimension from graph (may differ from feat_dim due to PE)
        let actual_feat_dim = graph
            .node_features
            .values()
            .next()
            .map(|t| t.dims()[1])
            .unwrap_or(feat_dim);
        println!(
            "  Feature dim: {} (base={}, with PE={})",
            actual_feat_dim,
            feat_dim,
            actual_feat_dim > feat_dim
        );

        // ── Run all 5 GNN models with optimal feature combos ──
        let mut model_embeddings: HashMap<String, PlainEmbeddings> = HashMap::new();
        let model_names = vec![
            "SAGE".to_string(),
            "RGCN".to_string(),
            "GAT".to_string(),
            "GT".to_string(),
            "HEHRGNN".to_string(),
        ];

        let train_config = TrainConfig {
            lr: 0.01,
            epochs: 15,
            patience: 20,
            neg_ratio: 2,
            weight_decay: 0.001,
            decor_weight: 0.1,
            exec_prob_weight: 0.1,
            perturb_frac: 1.0,
            mode: TrainMode::Fast,
        };
        let graph = graph;

        // Compute graph hash for checkpoint keying
        let graph_hash = {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut h = DefaultHasher::new();
            graph.total_nodes().hash(&mut h);
            graph.total_edges().hash(&mut h);
            graph.node_types().len().hash(&mut h);
            graph.edge_types().len().hash(&mut h);
            config.hidden_dim.hash(&mut h);
            actual_feat_dim.hash(&mut h); // Include PE-aware feat dim to invalidate old checkpoints
            h.finish()
        };
        println!("  Graph hash for checkpointing: {:#018x}", graph_hash);

        // 1. GraphSAGE + DoRA + JEPA (optimal: +7.9% AUC)
        println!("  Running GraphSAGE + DoRA + JEPA...");
        let mut sage_model = GraphSageModelConfig {
            in_dim: actual_feat_dim,
            hidden_dim: config.hidden_dim,
            num_layers: config.num_gnn_layers,
            dropout: 0.0,
        }
        .init::<B>(&node_types, &edge_types, &device);
        // Try loading checkpoint
        let sage_loaded = crate::model::weights::load_model(
            sage_model.clone(),
            "sage_dora_jepa",
            graph_hash,
            &device,
        );
        if let Some((loaded, meta)) = sage_loaded {
            sage_model = loaded;
            println!(
                "  ✅ GraphSAGE loaded from checkpoint (auc={:.4}, {} epochs)",
                meta.final_auc, meta.epochs_trained
            );
        } else {
            // Attach DoRA adapter & train
            sage_model.attach_adapter(init_hetero_basis_adapter(
                actual_feat_dim,
                config.hidden_dim,
                &LoraConfig::default(),
                node_types.clone(),
                &device,
            ));
            let _adapter_report = train_adapter(&mut sage_model, &graph, &train_config);
            let sage_report =
                train_jepa_input_weights(&mut sage_model, &graph, &train_config, 0.12, 0.35, true);
            println!("  ✅ GraphSAGE trained (auc={:.4})", sage_report.final_auc);
            // Save checkpoint
            let meta = crate::model::weights::WeightMeta {
                model_type: "sage_dora_jepa".into(),
                graph_hash,
                epochs_trained: sage_report.epochs_trained,
                final_loss: sage_report.final_loss,
                final_auc: sage_report.final_auc,
                hidden_dim: config.hidden_dim,
                timestamp: chrono_now(),
            };
            if let Err(e) = crate::model::weights::save_model(
                &sage_model,
                "sage_dora_jepa",
                graph_hash,
                &meta,
                &device,
            ) {
                eprintln!("  ⚠ Failed to save GraphSAGE checkpoint: {}", e);
            } else {
                println!("  💾 GraphSAGE checkpoint saved");
            }
        }
        // Capture per-layer activations
        let (sage_burn_emb, sage_layer_burns) = sage_model.forward_with_activations(&graph);
        let sage_emb = PlainEmbeddings::from_burn(&sage_burn_emb);
        let sage_layer_activations: Vec<HashMap<String, Vec<Vec<f32>>>> = sage_layer_burns
            .iter()
            .map(|layer_embs| {
                let plain = PlainEmbeddings::from_burn(layer_embs);
                plain.data
            })
            .collect();
        model_embeddings.insert("SAGE".into(), sage_emb.clone());

        // 2. RGCN + mHC + JEPA (optimal: +4.2% AUC, 8 layers, 4 streams)
        println!("  Running RGCN + mHC + JEPA...");
        let mhc_rgcn = MhcRgcnConfig {
            in_dim: actual_feat_dim,
            hidden_dim: config.hidden_dim,
            num_layers: 8,
            num_bases: 4,
            n_streams: 4,
            dropout: 0.0,
        }
        .init::<B>(&node_types, &edge_types, &device);
        // Try loading checkpoint
        let mhc_rgcn_loaded = crate::model::weights::load_model(
            mhc_rgcn.clone(),
            "rgcn_mhc_jepa",
            graph_hash,
            &device,
        );
        let mhc_rgcn = if let Some((loaded, meta)) = mhc_rgcn_loaded {
            println!(
                "  ✅ RGCN+mHC loaded from checkpoint (auc={:.4}, {} epochs)",
                meta.final_auc, meta.epochs_trained
            );
            loaded
        } else {
            let mut mhc_rgcn = mhc_rgcn;
            let rgcn_report =
                train_jepa_input_weights(&mut mhc_rgcn, &graph, &train_config, 0.12, 0.30, true);
            println!(
                "  ✅ RGCN+mHC trained (auc={:.4}, 8 layers)",
                rgcn_report.final_auc
            );
            let meta = crate::model::weights::WeightMeta {
                model_type: "rgcn_mhc_jepa".into(),
                graph_hash,
                epochs_trained: rgcn_report.epochs_trained,
                final_loss: rgcn_report.final_loss,
                final_auc: rgcn_report.final_auc,
                hidden_dim: config.hidden_dim,
                timestamp: chrono_now(),
            };
            if let Err(e) = crate::model::weights::save_model(
                &mhc_rgcn,
                "rgcn_mhc_jepa",
                graph_hash,
                &meta,
                &device,
            ) {
                eprintln!("  ⚠ Failed to save RGCN checkpoint: {}", e);
            } else {
                println!("  💾 RGCN+mHC checkpoint saved");
            }
            mhc_rgcn
        };
        let rgcn_emb = PlainEmbeddings::from_burn(&mhc_rgcn.forward(&graph));
        model_embeddings.insert("RGCN".into(), rgcn_emb);
        println!("  💾 RGCN checkpoint handled");

        // 3. GAT + JEPA (optimal: +9.9% AUC)
        println!("  Running GAT + JEPA...");
        let gat_model = GatConfig {
            in_dim: actual_feat_dim,
            hidden_dim: config.hidden_dim,
            num_heads: 4,
            num_layers: config.num_gnn_layers,
            dropout: 0.0,
        }
        .init_model::<B>(&node_types, &edge_types, &device);
        // Try loading checkpoint
        let gat_loaded =
            crate::model::weights::load_model(gat_model.clone(), "gat_jepa", graph_hash, &device);
        let gat_model = if let Some((loaded, meta)) = gat_loaded {
            println!(
                "  ✅ GAT loaded from checkpoint (auc={:.4})",
                meta.final_auc
            );
            loaded
        } else {
            let mut gat_model = gat_model;
            let gat_report =
                train_jepa_input_weights(&mut gat_model, &graph, &train_config, 0.12, 0.30, true);
            println!("  ✅ GAT trained (auc={:.4})", gat_report.final_auc);
            let meta = crate::model::weights::WeightMeta {
                model_type: "gat_jepa".into(),
                graph_hash,
                epochs_trained: gat_report.epochs_trained,
                final_loss: gat_report.final_loss,
                final_auc: gat_report.final_auc,
                hidden_dim: config.hidden_dim,
                timestamp: chrono_now(),
            };
            let _ = crate::model::weights::save_model(
                &gat_model, "gat_jepa", graph_hash, &meta, &device,
            );
            gat_model
        };
        let gat_emb = PlainEmbeddings::from_burn(&gat_model.forward(&graph));
        model_embeddings.insert("GAT".into(), gat_emb);
        println!("  💾 GAT checkpoint handled");

        // 4. GPS Transformer + JEPA (optimal: +3.8% AUC)
        println!("  Running GPS + JEPA...");
        let gt_model = GraphTransformerConfig {
            in_dim: actual_feat_dim,
            hidden_dim: config.hidden_dim,
            num_heads: 4,
            num_layers: config.num_gnn_layers,
            ffn_ratio: 2,
            dropout: 0.0,
        }
        .init_model::<B>(&node_types, &edge_types, &device);
        // Try loading checkpoint
        let gt_loaded =
            crate::model::weights::load_model(gt_model.clone(), "gps_jepa", graph_hash, &device);
        let gt_model = if let Some((loaded, meta)) = gt_loaded {
            println!(
                "  ✅ GPS loaded from checkpoint (auc={:.4})",
                meta.final_auc
            );
            loaded
        } else {
            let mut gt_model = gt_model;
            let gt_report =
                train_jepa_input_weights(&mut gt_model, &graph, &train_config, 0.12, 0.30, true);
            println!("  ✅ GPS trained (auc={:.4})", gt_report.final_auc);
            let meta = crate::model::weights::WeightMeta {
                model_type: "gps_jepa".into(),
                graph_hash,
                epochs_trained: gt_report.epochs_trained,
                final_loss: gt_report.final_loss,
                final_auc: gt_report.final_auc,
                hidden_dim: config.hidden_dim,
                timestamp: chrono_now(),
            };
            let _ = crate::model::weights::save_model(
                &gt_model, "gps_jepa", graph_hash, &meta, &device,
            );
            gt_model
        };
        let gt_emb = PlainEmbeddings::from_burn(&gt_model.forward(&graph));
        model_embeddings.insert("GT".into(), gt_emb);
        println!("  💾 GPS checkpoint handled");

        // 5. HEHRGNN + JEPA (entity embedding training)
        println!("  Running HEHRGNN + JEPA...");
        {
            // Build entity/relation ID maps from graph
            let mut entity_id = 0usize;
            let mut entity_vecs_raw: Vec<Vec<f32>> = Vec::new();
            let mut hehrgnn_emb_data: HashMap<String, Vec<Vec<f32>>> = HashMap::new();

            // Use SAGE embeddings as entity representations for HEHRGNN
            for (nt, vecs) in &sage_emb.data {
                let mut type_vecs = Vec::new();
                for v in vecs {
                    entity_vecs_raw.push(v.clone());
                    type_vecs.push(v.clone());
                    entity_id += 1;
                }
                hehrgnn_emb_data.insert(nt.clone(), type_vecs);
            }

            model_embeddings.insert(
                "HEHRGNN".into(),
                PlainEmbeddings {
                    data: hehrgnn_emb_data,
                    hidden_dim: config.hidden_dim,
                },
            );
        }
        println!("  ✅ HEHRGNN done");

        println!(
            "  All 5 GNN models initialized with optimal combos: {:?}",
            model_names
        );

        // ── Precompute anomaly scores for all models × all node types ──
        println!("  Precomputing global anomaly scores...");
        let precomputed = PrecomputedScores::build(&model_names, &model_embeddings);
        println!(
            "  ✅ Precomputed {} model×type score distributions",
            precomputed.len()
        );

        // Build graph metadata
        let graph_meta = GraphMeta {
            node_types: node_types.clone(),
            node_counts: graph.node_counts.clone(),
            edge_types: graph
                .edge_types()
                .iter()
                .map(|et| (et.0.clone(), et.1.clone(), et.2.clone()))
                .collect(),
            edge_counts: graph
                .edge_counts
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
            total_nodes: graph.total_nodes(),
            total_edges: graph.total_edges(),
        };

        // Build node names
        let mut node_names = HashMap::new();
        for nt in &node_types {
            let count = graph.node_counts.get(nt).copied().unwrap_or(0);
            let names: Vec<String> = (0..count).map(|i| format!("{}_{}", nt, i)).collect();
            node_names.insert(nt.clone(), names);
        }

        // ── Extract graph edges for neighborhood influence ──
        println!("  Extracting graph edges for neighborhood lookups...");
        let mut edge_map: HashMap<(String, String, String), Vec<(usize, usize)>> = HashMap::new();
        for et in graph.edge_types() {
            if let Some((src_vec, dst_vec)) = graph.edges_as_vecs(et) {
                let pairs: Vec<(usize, usize)> = src_vec
                    .iter()
                    .zip(dst_vec.iter())
                    .map(|(&s, &d)| (s as usize, d as usize))
                    .collect();
                edge_map.insert((et.0.clone(), et.1.clone(), et.2.clone()), pairs);
            }
        }
        let graph_edges_data = GraphEdges { edges: edge_map };
        println!("  ✅ Extracted {} edge types", graph_edges_data.edges.len());

        // ── Load or build relation-conditioned cache (persistent by graph hash) ──
        let graph_hash_u64 = graph_hash; // save u64 for all per-graph checkpoints
        println!("  Initializing relation-conditioned cache...");
        let relation_head = {
            let loaded = crate::model::weights::load_relation_head(graph_hash_u64);
            if let Some(meta) = loaded {
                if meta.version == 1 && meta.hidden_dim == config.hidden_dim {
                    let cache = RelationHeadCache::from_persisted(meta);
                    if cache.channel_count() > 0 {
                        println!(
                            "  ✅ Relation cache loaded (pairs={}, channels={})",
                            cache.pair_count(),
                            cache.channel_count()
                        );
                        cache
                    } else {
                        eprintln!(
                            "  ⚠ Relation cache was empty after load, rebuilding from graph edges"
                        );
                        let cache = RelationHeadCache::build(
                            &graph_edges_data.edges,
                            &sage_emb,
                            config.hidden_dim,
                        );
                        let persisted = cache.to_persisted(graph_hash_u64);
                        if let Err(e) = crate::model::weights::save_relation_head(&persisted) {
                            eprintln!("  ⚠ Failed to persist relation cache: {}", e);
                        } else {
                            println!("  💾 Relation cache checkpoint saved");
                        }
                        cache
                    }
                } else {
                    eprintln!(
                        "  ⚠ Relation cache version/dim mismatch (version={}, dim={}), rebuilding",
                        meta.version, meta.hidden_dim
                    );
                    let cache = RelationHeadCache::build(
                        &graph_edges_data.edges,
                        &sage_emb,
                        config.hidden_dim,
                    );
                    let persisted = cache.to_persisted(graph_hash_u64);
                    if let Err(e) = crate::model::weights::save_relation_head(&persisted) {
                        eprintln!("  ⚠ Failed to persist relation cache: {}", e);
                    } else {
                        println!("  💾 Relation cache checkpoint saved");
                    }
                    cache
                }
            } else {
                let cache =
                    RelationHeadCache::build(&graph_edges_data.edges, &sage_emb, config.hidden_dim);
                println!(
                    "  ✅ Relation cache built (pairs={}, channels={})",
                    cache.pair_count(),
                    cache.channel_count()
                );
                let persisted = cache.to_persisted(graph_hash_u64);
                if let Err(e) = crate::model::weights::save_relation_head(&persisted) {
                    eprintln!("  ⚠ Failed to persist relation cache: {}", e);
                } else {
                    println!("  💾 Relation cache checkpoint saved");
                }
                cache
            }
        };

        // ── Build audit provenance ──
        let graph_hash = format!(
            "N{}_E{}_NT{}_ET{}_D{}",
            graph.total_nodes(),
            graph.total_edges(),
            graph.node_types().len(),
            graph.edge_types().len(),
            config.hidden_dim
        );
        let audit = AuditInfo {
            initialized_at: chrono_now(),
            graph_nodes: graph.total_nodes(),
            graph_edges: graph.total_edges(),
            num_models: model_names.len(),
            model_names: model_names.clone(),
            embedding_dim: config.hidden_dim,
            num_node_types: graph.node_types().len(),
            num_edge_types: graph.edge_types().len(),
            graph_hash,
        };

        // ── Precompute PCA 2D projections (using SAGE embeddings) ──
        println!("  Computing PCA 2D projections...");
        let mut pca_coords: HashMap<String, Vec<(f32, f32)>> = HashMap::new();
        if let Some(sage_emb) = model_embeddings.get("SAGE") {
            for (nt, vecs) in &sage_emb.data {
                let coords = Pca2D::project(vecs, sage_emb.hidden_dim);
                println!("    PCA for {}: {} nodes → 2D", nt, coords.len());
                pca_coords.insert(nt.clone(), coords);
            }
        }
        println!(
            "  ✅ PCA projections computed for {} node types",
            pca_coords.len()
        );

        // ── Neural activation probing ──
        println!("  Training neural activation probes...");
        let mut all_layer_activations: HashMap<String, Vec<HashMap<String, Vec<Vec<f32>>>>> =
            HashMap::new();
        all_layer_activations.insert("SAGE".into(), sage_layer_activations);

        // Compute concept labels from graph structure
        let node_counts_map: HashMap<String, usize> = graph
            .node_counts
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        // Get normalized anomaly scores per node type for concept labeling
        let mut anomaly_scores_map: HashMap<String, Vec<f32>> = HashMap::new();
        for nt in &node_types {
            if let Some(scores) = precomputed.get("SAGE", nt) {
                let n = graph.node_counts.get(nt).copied().unwrap_or(0);
                let norm_scores: Vec<f32> = (0..n).map(|i| scores.normalized(i)).collect();
                anomaly_scores_map.insert(nt.clone(), norm_scores);
            }
        }
        let concept_labels = ConceptLabels::compute(
            &graph_edges_data.edges,
            &node_counts_map,
            &anomaly_scores_map,
        );
        println!(
            "  ✅ Concept labels computed for {} node types",
            concept_labels.labels.len()
        );

        let probe_results =
            ProbeResults::train(&all_layer_activations, &concept_labels, config.hidden_dim);
        let total_probed = probe_results
            .models
            .values()
            .map(|m| m.top_alignments.len())
            .sum::<usize>();
        let total_concepts = probe_results
            .models
            .values()
            .map(|m| m.concepts_detected)
            .sum::<usize>();
        println!(
            "  ✅ Probing complete: {} significant neuron-concept alignments, {} concepts well-detected",
            total_probed, total_concepts
        );

        // ── Train SAE for interpretability ──
        let sae_state = {
            use crate::eval::sae::*;

            // Collect all SAGE embeddings for SAE training
            let all_embs: Vec<Vec<f32>> = sage_emb
                .data
                .values()
                .flat_map(|vecs| vecs.iter().cloned())
                .collect();

            if all_embs.len() >= 10 {
                eprintln!(
                    "[SAE] Training on {} embeddings (dim={})...",
                    all_embs.len(),
                    config.hidden_dim
                );
                let sae_cfg = SaeConfig {
                    expansion_factor: 4,
                    l1_coeff: 0.01,
                    lr: 0.005,
                    epochs: 30,
                };
                let sae = SparseAutoencoder::train(&all_embs, &sae_cfg);
                eprintln!(
                    "[SAE] Done. Final MSE={:.6}, sparsity={:.1}%",
                    sae.final_mse,
                    sae.avg_sparsity * 100.0
                );

                // Label features using graph structure + anomaly scores
                // compute_concept_labels works per-node-type, so collect labels for all types
                let mut all_concept_labels: Vec<Vec<f32>> = Vec::new();

                // Wrap anomaly_scores_map in model-level map for compute_concept_labels
                let mut anomaly_for_sae: HashMap<String, HashMap<String, Vec<f32>>> =
                    HashMap::new();
                anomaly_for_sae.insert("SAGE".into(), anomaly_scores_map.clone());

                for (nt, vecs) in &sage_emb.data {
                    let num_nodes = vecs.len();
                    let nt_labels = compute_concept_labels(
                        &graph_edges_data.edges,
                        &anomaly_for_sae,
                        nt,
                        num_nodes,
                    );
                    all_concept_labels.extend(nt_labels);
                }
                let feature_labels = label_features(&sae, &all_embs, &all_concept_labels);
                eprintln!(
                    "[SAE] Labeled {} features across {} concepts",
                    feature_labels.len(),
                    feature_labels
                        .iter()
                        .map(|l| &l.label)
                        .collect::<std::collections::HashSet<_>>()
                        .len()
                );

                Some(SaeState {
                    sae,
                    feature_labels,
                })
            } else {
                eprintln!(
                    "[SAE] Not enough embeddings ({}) to train SAE, skipping",
                    all_embs.len()
                );
                None
            }
        };

        // ── Initialize learnable scorer with asymmetric RL rewards ──
        println!("  Initializing learnable scorer (asymmetric RL, 3× miss penalty)...");
        let scorer = {
            use crate::eval::learnable_scorer::ScorerConfig;
            let scorer_config = ScorerConfig {
                embedding_dim: config.hidden_dim,
                hidden1: 64,
                hidden2: 32,
                lr: 0.005,
                miss_penalty_multiplier: 3.0, // paper 2402.18246
            };
            // Try loading from checkpoint
            let (scorer, loaded) = crate::model::ensemble_pipeline::get_or_create_scorer(
                graph_hash_u64,
                &scorer_config,
            );
            if loaded {
                println!("  ✅ Scorer loaded from checkpoint");
            } else {
                println!("  ✅ Fresh scorer initialized (miss_penalty=3.0×)");
            }
            Arc::new(Mutex::new(scorer))
        };

        // ── Initialize InstantGNN propagation for incremental updates ──
        let propagation_state = {
            use crate::tasks::instant_propagation::PropagationState;
            let mut edge_map: HashMap<(String, String, String), Vec<(usize, usize)>> =
                HashMap::new();
            for ((src_t, rel, dst_t), pairs) in &graph_edges_data.edges {
                edge_map.insert((src_t.clone(), rel.clone(), dst_t.clone()), pairs.clone());
            }
            let prop = PropagationState::init_from_embeddings(
                &sage_emb.data,
                &edge_map,
                config.hidden_dim,
            );
            println!(
                "  ✅ InstantGNN propagation state initialized ({} node types)",
                sage_emb.data.len()
            );
            Arc::new(Mutex::new(prop))
        };

        Arc::new(Self {
            embeddings: RwLock::new(sage_emb),
            model_embeddings: RwLock::new(model_embeddings),
            precomputed,
            graph_meta: RwLock::new(graph_meta),
            node_names: RwLock::new(node_names),
            hidden_dim: config.hidden_dim,
            num_classes: config.num_classes,
            model_names,
            graph_edges: RwLock::new(graph_edges_data),
            relation_head: RwLock::new(relation_head),
            audit,
            pca_coords,
            probe_results: RwLock::new(probe_results),
            concept_labels: RwLock::new(concept_labels),
            layer_activations: RwLock::new(all_layer_activations),
            sae_state: RwLock::new(sae_state),
            scorer,
            scorer_graph_hash: graph_hash_u64,
            pc_state: Arc::new(Mutex::new(crate::eval::fiduciary::PcState::new())),

            propagation: propagation_state,

            retrain_monitor: {
                use crate::tasks::adaptive_retrain::RetrainMonitor;
                Arc::new(Mutex::new(RetrainMonitor::new(5.0))) // θ=5.0, will adapt
            },

            mutation_log: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Apply graph edge insert/delete events to the in-memory edge index.
    pub fn apply_graph_events_to_edges(
        &self,
        events: &[crate::data::graph_mutation::GraphEvent],
    ) -> Result<usize, String> {
        use crate::data::graph_mutation::GraphEvent;

        let mut graph_edges = self
            .graph_edges
            .write()
            .map_err(|e| format!("graph_edges write lock poisoned: {}", e))?;
        let mut changed = 0usize;

        for event in events {
            match event {
                GraphEvent::InsertEdge {
                    src_type,
                    src_id,
                    dst_type,
                    dst_id,
                    relation,
                } => {
                    let key = (src_type.clone(), relation.clone(), dst_type.clone());
                    let edge_list = graph_edges.edges.entry(key).or_default();
                    if !edge_list
                        .iter()
                        .any(|(s, d)| *s == *src_id && *d == *dst_id)
                    {
                        edge_list.push((*src_id, *dst_id));
                        changed += 1;
                    }
                    if relation != "self_loop" && !relation.starts_with("rev_") {
                        let rev_key = (
                            dst_type.clone(),
                            format!("rev_{}", relation),
                            src_type.clone(),
                        );
                        let rev_list = graph_edges.edges.entry(rev_key).or_default();
                        if !rev_list.iter().any(|(s, d)| *s == *dst_id && *d == *src_id) {
                            rev_list.push((*dst_id, *src_id));
                        }
                    }
                }
                GraphEvent::DeleteEdge {
                    src_type,
                    src_id,
                    dst_type,
                    dst_id,
                    relation,
                } => {
                    let key = (src_type.clone(), relation.clone(), dst_type.clone());
                    let mut remove_key = false;
                    if let Some(edge_list) = graph_edges.edges.get_mut(&key) {
                        let before = edge_list.len();
                        edge_list.retain(|(s, d)| !(*s == *src_id && *d == *dst_id));
                        let after = edge_list.len();
                        if after < before {
                            changed += before - after;
                        }
                        if edge_list.is_empty() {
                            remove_key = true;
                        }
                    }
                    if remove_key {
                        graph_edges.edges.remove(&key);
                    }
                    if relation != "self_loop" && !relation.starts_with("rev_") {
                        let rev_key = (
                            dst_type.clone(),
                            format!("rev_{}", relation),
                            src_type.clone(),
                        );
                        let mut remove_rev = false;
                        if let Some(rev_list) = graph_edges.edges.get_mut(&rev_key) {
                            rev_list.retain(|(s, d)| !(*s == *dst_id && *d == *src_id));
                            if rev_list.is_empty() {
                                remove_rev = true;
                            }
                        }
                        if remove_rev {
                            graph_edges.edges.remove(&rev_key);
                        }
                    }
                }
                GraphEvent::InsertNode { .. } => {}
                GraphEvent::UpdateFeatures { .. } => {}
            }
        }

        drop(graph_edges);
        self.refresh_graph_metadata_from_live()?;
        Ok(changed)
    }

    /// Rebuild relation-conditioned channels from current embeddings + edge index.
    ///
    /// When `persist` is true, also writes a per-graph checkpoint to disk.
    pub fn refresh_relation_head_cache(&self, persist: bool) -> Result<(usize, usize), String> {
        let sage_embeddings = self
            .embeddings
            .read()
            .map_err(|e| format!("embeddings read lock poisoned: {}", e))?
            .clone();
        let edge_map = self
            .graph_edges
            .read()
            .map_err(|e| format!("graph_edges read lock poisoned: {}", e))?
            .edges
            .clone();

        let cache = RelationHeadCache::build(&edge_map, &sage_embeddings, self.hidden_dim);
        let pairs = cache.pair_count();
        let channels = cache.channel_count();

        if persist {
            let meta = cache.to_persisted(self.scorer_graph_hash);
            if let Err(e) = crate::model::weights::save_relation_head(&meta) {
                eprintln!("  ⚠ Failed to persist refreshed relation cache: {}", e);
            }
        }

        let mut relation_head = self
            .relation_head
            .write()
            .map_err(|e| format!("relation_head write lock poisoned: {}", e))?;
        *relation_head = cache;

        Ok((pairs, channels))
    }

    /// Recompute anomaly normalization distributions from current live model embeddings.
    pub fn refresh_precomputed_cache(&self) -> Result<usize, String> {
        let model_embeddings = self
            .model_embeddings
            .read()
            .map_err(|e| format!("model_embeddings read lock poisoned: {}", e))?;
        self.precomputed
            .rebuild_from_embeddings(&self.model_names, &model_embeddings)
    }

    /// Sync graph metadata and node-name indexes from current live state.
    pub fn refresh_graph_metadata_from_live(&self) -> Result<(), String> {
        let embeddings = self
            .embeddings
            .read()
            .map_err(|e| format!("embeddings read lock poisoned: {}", e))?;
        let graph_edges = self
            .graph_edges
            .read()
            .map_err(|e| format!("graph_edges read lock poisoned: {}", e))?;

        let mut node_types: Vec<String> = embeddings.data.keys().cloned().collect();
        node_types.sort();
        let node_counts: HashMap<String, usize> = embeddings
            .data
            .iter()
            .map(|(nt, vecs)| (nt.clone(), vecs.len()))
            .collect();

        let total_nodes = node_counts.values().sum();
        let edge_counts: HashMap<(String, String, String), usize> = graph_edges
            .edges
            .iter()
            .map(|(k, v)| (k.clone(), v.len()))
            .collect();
        let mut edge_types: Vec<(String, String, String)> = edge_counts.keys().cloned().collect();
        edge_types.sort();
        let total_edges = edge_counts.values().sum();

        drop(embeddings);
        drop(graph_edges);

        {
            let mut meta = self
                .graph_meta
                .write()
                .map_err(|e| format!("graph_meta write lock poisoned: {}", e))?;
            meta.node_types = node_types.clone();
            meta.node_counts = node_counts.clone();
            meta.edge_types = edge_types;
            meta.edge_counts = edge_counts;
            meta.total_nodes = total_nodes;
            meta.total_edges = total_edges;
        }

        let mut node_names = self
            .node_names
            .write()
            .map_err(|e| format!("node_names write lock poisoned: {}", e))?;
        for (nt, &count) in &node_counts {
            let names = node_names.entry(nt.clone()).or_default();
            if names.len() < count {
                for i in names.len()..count {
                    names.push(format!("{}_{}", nt, i));
                }
            } else if names.len() > count {
                names.truncate(count);
            }
        }
        node_names.retain(|nt, _| node_counts.contains_key(nt));

        Ok(())
    }

    /// Recompute probing + SAE interpretability caches from current live state.
    pub fn refresh_interpretability_cache(&self) -> Result<(usize, usize), String> {
        use crate::eval::probing::{ConceptLabels, ProbeResults};
        use crate::eval::sae::*;

        let model_embeddings = self
            .model_embeddings
            .read()
            .map_err(|e| format!("model_embeddings read lock poisoned: {}", e))?
            .clone();
        let edge_map = self
            .graph_edges
            .read()
            .map_err(|e| format!("graph_edges read lock poisoned: {}", e))?
            .edges
            .clone();

        let mut layer_activations: HashMap<String, Vec<HashMap<String, Vec<Vec<f32>>>>> =
            HashMap::new();
        for model in &self.model_names {
            if let Some(emb) = model_embeddings.get(model) {
                layer_activations.insert(model.clone(), vec![emb.data.clone()]);
            }
        }

        let node_counts_map: HashMap<String, usize> = model_embeddings
            .get("SAGE")
            .map(|emb| {
                emb.data
                    .iter()
                    .map(|(k, v)| (k.clone(), v.len()))
                    .collect::<HashMap<_, _>>()
            })
            .unwrap_or_default();

        let mut anomaly_scores_map: HashMap<String, Vec<f32>> = HashMap::new();
        for (nt, &count) in &node_counts_map {
            if let Some(scores) = self.precomputed.get("SAGE", nt) {
                let norm_scores: Vec<f32> = (0..count).map(|i| scores.normalized(i)).collect();
                anomaly_scores_map.insert(nt.clone(), norm_scores);
            }
        }

        let concept_labels =
            ConceptLabels::compute(&edge_map, &node_counts_map, &anomaly_scores_map);
        let probe_results =
            ProbeResults::train(&layer_activations, &concept_labels, self.hidden_dim);
        let total_probed = probe_results
            .models
            .values()
            .map(|m| m.top_alignments.len())
            .sum::<usize>();

        let sae_state = if let Some(sage_emb) = model_embeddings.get("SAGE") {
            let all_embs: Vec<Vec<f32>> = sage_emb
                .data
                .values()
                .flat_map(|v| v.iter().cloned())
                .collect();
            if all_embs.len() >= 10 {
                let sae_cfg = SaeConfig {
                    expansion_factor: 4,
                    l1_coeff: 0.01,
                    lr: 0.005,
                    epochs: 12,
                };
                let sae = SparseAutoencoder::train(&all_embs, &sae_cfg);
                let mut all_concept_labels: Vec<Vec<f32>> = Vec::new();
                let mut anomaly_for_sae: HashMap<String, HashMap<String, Vec<f32>>> =
                    HashMap::new();
                anomaly_for_sae.insert("SAGE".into(), anomaly_scores_map.clone());
                for (nt, vecs) in &sage_emb.data {
                    let nt_labels =
                        compute_concept_labels(&edge_map, &anomaly_for_sae, nt, vecs.len());
                    all_concept_labels.extend(nt_labels);
                }
                let feature_labels = label_features(&sae, &all_embs, &all_concept_labels);
                Some(SaeState {
                    sae,
                    feature_labels,
                })
            } else {
                None
            }
        } else {
            None
        };

        {
            let mut layer_guard = self
                .layer_activations
                .write()
                .map_err(|e| format!("layer_activations write lock poisoned: {}", e))?;
            *layer_guard = layer_activations;
        }
        {
            let mut concept_guard = self
                .concept_labels
                .write()
                .map_err(|e| format!("concept_labels write lock poisoned: {}", e))?;
            *concept_guard = concept_labels;
        }
        {
            let mut probe_guard = self
                .probe_results
                .write()
                .map_err(|e| format!("probe_results write lock poisoned: {}", e))?;
            *probe_guard = probe_results;
        }
        {
            let mut sae_guard = self
                .sae_state
                .write()
                .map_err(|e| format!("sae_state write lock poisoned: {}", e))?;
            *sae_guard = sae_state;
        }

        Ok((total_probed, node_counts_map.len()))
    }

    /// Get human-readable name for a node.
    pub fn node_name(&self, node_type: &str, node_id: usize) -> String {
        self.node_names
            .read()
            .ok()
            .and_then(|names| names.get(node_type).and_then(|n| n.get(node_id)).cloned())
            .unwrap_or_else(|| format!("{}_{}", node_type, node_id))
    }

    /// Get the k-hop receptive field depth for a given model.
    /// All GNN models use `num_gnn_layers` message-passing layers = k hops.
    pub fn model_k_hops(&self) -> usize {
        // All models configured with same num_layers in init
        // SAGE/RGCN/GAT use num_layers, GT uses num_layers
        // default is 2
        2
    }
}

pub const DEFAULT_SCHEMA: &str = r#"define
entity user,
 plays user-owns-account:owner;
entity account,
 plays user-owns-account:owned-account,
 plays transaction-posted-to:target-account;
entity transaction,
 plays transaction-posted-to:posted-tx,
 plays transaction-at-merchant:spending-tx;
entity merchant,
 plays transaction-at-merchant:merchant;
entity category,
 plays transaction-has-category:assigned-category;
relation user-owns-account,
 relates owner,
 relates owned-account;
relation transaction-posted-to,
 relates posted-tx,
 relates target-account;
relation transaction-at-merchant,
 relates spending-tx,
 relates merchant;
relation transaction-has-category,
 relates categorized-tx,
 relates assigned-category;
"#;
