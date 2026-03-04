//! Generic link predictor for heterogeneous graphs (paper 2506.09234 + improvements).
//!
//! Predicts which target entity a source entity should link to by treating it as a
//! link prediction problem: score(source_emb, target_emb) → ranking.
//!
//! Works for ANY entity pair + relation:
//!   - transaction-evidence → transaction-category (evidence-has-category)
//!   - reconciliation-case → sub-ledger (case-allocation)
//!   - instrument → account (instrument-mapped-to-ledger)
//!   - ...any schema-defined relation
//!
//! Key features (from Rel-Cat paper, enhanced with our ensemble):
//! - Dot-product link prediction between source and target embeddings
//! - TopK-NN early exit: if historical sources with cosine > 0.8 exist → skip GNN
//! - Weighted negative sampling: proportional to target frequency
//! - Diversity filtering: full-forward partial-backward (40% most diverse)
//! - **Probabilistic Circuit**: calibrated P(target|features) + lift analysis

use std::collections::HashMap;

use crate::model::pc::bridge::{build_fiduciary_pc, NUM_CATEGORIES};
use crate::model::pc::circuit::CompiledCircuit;
use crate::model::pc::query;

/// Configuration for the link predictor.
#[derive(Debug, Clone)]
pub struct LinkPredictorConfig {
    /// Number of top targets to return.
    pub top_k: usize,
    /// Cosine similarity threshold for TopK-NN early exit (paper: 0.8).
    pub nn_sim_threshold: f32,
    /// Minimum distinct targets from TopK-NN to skip GNN (paper: 5).
    pub nn_min_targets: usize,
    /// Target proportion of samples to keep after diversity filtering (paper: 0.4).
    pub diversity_keep_ratio: f32,
}

impl Default for LinkPredictorConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            nn_sim_threshold: 0.8,
            nn_min_targets: 5,
            diversity_keep_ratio: 0.4,
        }
    }
}

/// A predicted link target with score and metadata.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LinkPrediction {
    /// Target node type (e.g. "transaction-category", "sub-ledger").
    pub target_type: String,
    /// Target node ID in the graph.
    pub target_id: usize,
    /// Target name / label.
    pub target_name: String,
    /// Dot-product score (higher = more likely).
    pub score: f32,
    /// Rank (1 = best).
    pub rank: usize,
    /// Source of prediction: "topk_nn", "gnn_ensemble", or "gnn_ensemble+pc".
    pub source: String,
}

/// Full link prediction result.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LinkPredictionResult {
    /// Source node type (e.g. "transaction-evidence", "reconciliation-case").
    pub source_type: String,
    /// Source node ID.
    pub source_id: usize,
    /// Top-K predicted targets, sorted by score.
    pub predictions: Vec<LinkPrediction>,
    /// Whether TopK-NN early exit was used.
    pub used_early_exit: bool,
    /// Number of historical sources matched (for TopK-NN).
    pub nn_matches: usize,
    /// Human-readable explanation.
    pub explanation: String,
}

/// Generic link predictor: dot-product scoring between any two node types.
///
/// Architecture (paper 2506.09234 §3, enhanced):
/// 1. TopK-NN early exit: cosine similarity with historical sources
/// 2. If early exit fails → dot-product scoring with all target embeddings
/// 3. PC calibration: P(target|features) from probabilistic circuit
/// 4. Return top-K targets with calibrated probabilities + lift factors
pub struct LinkPredictor {
    config: LinkPredictorConfig,
}

// ─────────────────────────────────────────────────────────────
// PC-enhanced link prediction
// ─────────────────────────────────────────────────────────────

/// PC analysis attached to a link prediction.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PcLinkAnalysis {
    /// Calibrated probability P(this_target | features).
    pub probability: f64,
    /// Confidence = max_probability / second_best (ratio >2 = confident).
    pub confidence: f64,
    /// Top lift factors: which features most increase this target's probability.
    pub lift_factors: Vec<(String, f64)>,
}

/// PC variables for link prediction:
///   0 = dot_product_score (discretized 0-4)
///   1 = embedding_norm (discretized 0-4)
///   2 = target_id (0..num_targets, capped at NUM_CATEGORIES)
const PC_VAR_SCORE: usize = 0;
const PC_VAR_NORM: usize = 1;
const PC_VAR_TARGET: usize = 2;
#[allow(dead_code)]
const NUM_LINK_PC_VARS: usize = 3;

const LINK_PC_VAR_NAMES: [&str; 3] = ["dot_score", "embedding_norm", "target"];

/// Discretize a dot-product score into 5 bins.
fn discretize_score(score: f32) -> usize {
    if score < -0.5 {
        0
    } else if score < 0.0 {
        1
    } else if score < 0.5 {
        2
    } else if score < 1.0 {
        3
    } else {
        4
    }
}

/// Discretize an embedding L2 norm into 5 bins.
fn discretize_norm(norm: f32) -> usize {
    if norm < 0.5 {
        0
    } else if norm < 1.0 {
        1
    } else if norm < 2.0 {
        2
    } else if norm < 4.0 {
        3
    } else {
        4
    }
}

/// Build a link-prediction PC from historical source→target observations.
///
/// Each observation is: [dot_score_bin, norm_bin, target_id].
/// The PC learns P(target | dot_score, norm) from the joint distribution.
pub fn build_link_pc(
    historical_scores: &[(f32, f32, usize)], // (dot_score, norm, target_id)
    num_targets: usize,
    em_epochs: usize,
) -> CompiledCircuit {
    let mut observations: Vec<Vec<usize>> = historical_scores
        .iter()
        .map(|&(score, norm, target)| {
            vec![
                discretize_score(score),
                discretize_norm(norm),
                target.min(NUM_CATEGORIES - 1),
            ]
        })
        .collect();

    if observations.len() < 10 {
        for score_bin in 0..NUM_CATEGORIES {
            for t in 0..num_targets.min(NUM_CATEGORIES) {
                observations.push(vec![score_bin, 2, t]);
            }
        }
    }

    let (circuit, _report) = build_fiduciary_pc(&observations, em_epochs);
    circuit
}

impl LinkPredictor {
    pub fn new(config: LinkPredictorConfig) -> Self {
        Self { config }
    }

    /// Predict which target a source node should link to.
    ///
    /// Computes dot-product between the source embedding and all target
    /// embeddings, returns the top-K targets.
    pub fn predict(
        &self,
        source_emb: &[f32],
        target_embs: &[(String, usize, String, Vec<f32>)],
        historical_source_embs: Option<&[(Vec<f32>, usize)]>,
        historical_targets: Option<&[(String, usize, String)]>,
        source_type: &str,
        source_id: usize,
    ) -> LinkPredictionResult {
        // ── 1. Try TopK-NN early exit ──
        if let (Some(hist_embs), Some(hist_targets)) = (historical_source_embs, historical_targets)
        {
            if let Some(result) = self.topk_nn_early_exit(
                source_emb,
                hist_embs,
                hist_targets,
                target_embs,
                source_type,
                source_id,
            ) {
                return result;
            }
        }

        // ── 2. Full dot-product scoring ──
        let mut scores: Vec<(usize, f32)> = target_embs
            .iter()
            .enumerate()
            .map(|(idx, (_tt, _tid, _name, t_emb))| (idx, dot_product(source_emb, t_emb)))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k = self.config.top_k.min(scores.len());
        let predictions: Vec<LinkPrediction> = scores[..top_k]
            .iter()
            .enumerate()
            .map(|(rank, &(idx, score))| {
                let (tt, tid, name, _) = &target_embs[idx];
                LinkPrediction {
                    target_type: tt.clone(),
                    target_id: *tid,
                    target_name: name.clone(),
                    score,
                    rank: rank + 1,
                    source: "gnn_ensemble".into(),
                }
            })
            .collect();

        let explanation = format!(
            "Scored {} targets via dot-product link prediction. \
             Top: {} (score={:.3}).",
            target_embs.len(),
            predictions
                .first()
                .map(|p| p.target_name.as_str())
                .unwrap_or("none"),
            predictions.first().map(|p| p.score).unwrap_or(0.0),
        );

        LinkPredictionResult {
            source_type: source_type.into(),
            source_id,
            predictions,
            used_early_exit: false,
            nn_matches: 0,
            explanation,
        }
    }

    /// TopK-NN early exit (paper §3.4.2).
    fn topk_nn_early_exit(
        &self,
        source_emb: &[f32],
        historical_embs: &[(Vec<f32>, usize)],
        historical_targets: &[(String, usize, String)],
        _all_target_embs: &[(String, usize, String, Vec<f32>)],
        source_type: &str,
        source_id: usize,
    ) -> Option<LinkPredictionResult> {
        let mut sim_scores: Vec<(usize, f32, usize)> = historical_embs
            .iter()
            .enumerate()
            .map(|(idx, (hist_emb, target_id))| {
                (idx, cosine_similarity(source_emb, hist_emb), *target_id)
            })
            .filter(|(_idx, sim, _)| *sim >= self.config.nn_sim_threshold)
            .collect();

        if sim_scores.is_empty() {
            return None;
        }

        sim_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut seen: HashMap<usize, f32> = HashMap::new();
        for &(_idx, sim, tid) in &sim_scores {
            seen.entry(tid)
                .and_modify(|s| {
                    if sim > *s {
                        *s = sim;
                    }
                })
                .or_insert(sim);
        }

        if seen.len() < self.config.nn_min_targets && seen.len() < self.config.top_k {
            return None;
        }

        let mut target_scores: Vec<(usize, f32)> = seen.into_iter().collect();
        target_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k = self.config.top_k.min(target_scores.len());
        let predictions: Vec<LinkPrediction> = target_scores[..top_k]
            .iter()
            .enumerate()
            .map(|(rank, &(tid, sim))| {
                let (tt, id, name) = historical_targets
                    .iter()
                    .find(|(_t, id, _n)| *id == tid)
                    .cloned()
                    .unwrap_or_else(|| ("target".into(), tid, format!("target_{}", tid)));
                LinkPrediction {
                    target_type: tt,
                    target_id: id,
                    target_name: name,
                    score: sim,
                    rank: rank + 1,
                    source: "topk_nn".into(),
                }
            })
            .collect();

        let nn_matches = sim_scores.len();
        let explanation = format!(
            "TopK-NN early exit: {} similar sources (cosine ≥ {:.1}). \
             {} distinct targets. Top: {} (sim={:.3}).",
            nn_matches,
            self.config.nn_sim_threshold,
            target_scores.len(),
            predictions
                .first()
                .map(|p| p.target_name.as_str())
                .unwrap_or("none"),
            predictions.first().map(|p| p.score).unwrap_or(0.0),
        );

        Some(LinkPredictionResult {
            source_type: source_type.into(),
            source_id,
            predictions,
            used_early_exit: true,
            nn_matches,
            explanation,
        })
    }

    /// Predict with Probabilistic Circuit calibration.
    pub fn predict_with_pc(
        &self,
        source_emb: &[f32],
        target_embs: &[(String, usize, String, Vec<f32>)],
        circuit: &mut CompiledCircuit,
        source_type: &str,
        source_id: usize,
    ) -> (LinkPredictionResult, Vec<PcLinkAnalysis>) {
        let src_norm: f32 = source_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_bin = discretize_norm(src_norm);

        let mut scores: Vec<(usize, f32)> = target_embs
            .iter()
            .enumerate()
            .map(|(idx, (_tt, _tid, _name, t_emb))| (idx, dot_product(source_emb, t_emb)))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k = self.config.top_k.min(scores.len());
        let mut pc_analyses: Vec<PcLinkAnalysis> = Vec::with_capacity(top_k);

        for &(idx, dot_score) in &scores[..top_k] {
            let score_bin = discretize_score(dot_score);
            let target_bin = idx.min(NUM_CATEGORIES - 1);

            let evidence: Vec<Option<usize>> = vec![Some(score_bin), Some(norm_bin), None];

            let conditionals = query::conditional(circuit, &evidence, &[PC_VAR_TARGET]);
            let target_probs = conditionals
                .get(&PC_VAR_TARGET)
                .cloned()
                .unwrap_or_else(|| vec![1.0 / NUM_CATEGORIES as f64; NUM_CATEGORIES]);

            let prob = if target_bin < target_probs.len() {
                target_probs[target_bin]
            } else {
                1.0 / NUM_CATEGORIES as f64
            };

            let max_other = target_probs
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != target_bin)
                .map(|(_, &p)| p)
                .fold(f64::NEG_INFINITY, f64::max);
            let confidence = if max_other > 1e-10 {
                prob / max_other
            } else {
                prob * 100.0
            };

            let mut lift_factors = Vec::new();
            for var in [PC_VAR_SCORE, PC_VAR_NORM] {
                let var_val = if var == PC_VAR_SCORE {
                    score_bin
                } else {
                    norm_bin
                };
                let lift_val =
                    query::lift(circuit, &evidence, var, var_val, PC_VAR_TARGET, target_bin);
                let var_name = LINK_PC_VAR_NAMES[var];
                if lift_val.abs() > 0.01 && lift_val.is_finite() {
                    lift_factors.push((format!("{}={}", var_name, var_val), lift_val));
                }
            }
            lift_factors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            pc_analyses.push(PcLinkAnalysis {
                probability: prob,
                confidence,
                lift_factors,
            });
        }

        let predictions: Vec<LinkPrediction> = scores[..top_k]
            .iter()
            .enumerate()
            .map(|(rank, &(idx, score))| {
                let (tt, tid, name, _) = &target_embs[idx];
                LinkPrediction {
                    target_type: tt.clone(),
                    target_id: *tid,
                    target_name: name.clone(),
                    score,
                    rank: rank + 1,
                    source: "gnn_ensemble+pc".into(),
                }
            })
            .collect();

        let best_prob = pc_analyses.first().map(|a| a.probability).unwrap_or(0.0);
        let best_conf = pc_analyses.first().map(|a| a.confidence).unwrap_or(0.0);
        let explanation = format!(
            "PC-calibrated: {} targets. Top: {} (dot={:.3}, P={:.3}, conf={:.2}).",
            target_embs.len(),
            predictions
                .first()
                .map(|p| p.target_name.as_str())
                .unwrap_or("none"),
            predictions.first().map(|p| p.score).unwrap_or(0.0),
            best_prob,
            best_conf,
        );

        (
            LinkPredictionResult {
                source_type: source_type.into(),
                source_id,
                predictions,
                used_early_exit: false,
                nn_matches: 0,
                explanation,
            },
            pc_analyses,
        )
    }
}

// ─────────────────────────────────────────────────────────────
// Training utilities (from Rel-Cat paper)
// ─────────────────────────────────────────────────────────────

/// Weighted negative sampling (paper §3.3.1).
pub fn weighted_negative_sample(
    positive_pairs: &[(usize, usize)],
    target_frequencies: &[usize],
    num_negatives_per_positive: usize,
) -> Vec<(usize, usize)> {
    let total_freq: f64 = target_frequencies.iter().sum::<usize>() as f64;
    if total_freq == 0.0 {
        return vec![];
    }

    let weights: Vec<f64> = target_frequencies
        .iter()
        .map(|&f| f as f64 / total_freq)
        .collect();
    let mut cumulative: Vec<f64> = Vec::with_capacity(weights.len());
    let mut acc = 0.0;
    for w in &weights {
        acc += w;
        cumulative.push(acc);
    }

    let num_targets = target_frequencies.len();
    let mut negatives = Vec::with_capacity(positive_pairs.len() * num_negatives_per_positive);

    let mut seed: u64 = 0xDEAD_BEEF_CAFE;
    let next_rand = |s: &mut u64| -> f64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        (*s as f64) / (u64::MAX as f64)
    };

    for &(src_idx, pos_target) in positive_pairs {
        let mut count = 0;
        let mut attempts = 0;
        while count < num_negatives_per_positive && attempts < num_negatives_per_positive * 10 {
            attempts += 1;
            let r = next_rand(&mut seed);
            let neg = match cumulative
                .binary_search_by(|x| x.partial_cmp(&r).unwrap_or(std::cmp::Ordering::Equal))
            {
                Ok(i) => i.min(num_targets - 1),
                Err(i) => i.min(num_targets - 1),
            };
            if neg != pos_target {
                negatives.push((src_idx, neg));
                count += 1;
            }
        }
    }
    negatives
}

/// Diversity filtering (paper §3.3.2).
pub fn diversity_filter(embeddings: &[Vec<f32>], keep_ratio: f32) -> Vec<usize> {
    let n = embeddings.len();
    if n <= 1 || keep_ratio >= 1.0 {
        return (0..n).collect();
    }

    let keep_count = ((n as f32) * keep_ratio).ceil() as usize;
    let keep_count = keep_count.max(1).min(n);

    let mut avg_sims: Vec<(usize, f32)> = (0..n)
        .map(|i| {
            let total_sim: f32 = (0..n)
                .filter(|&j| j != i)
                .map(|j| cosine_similarity(&embeddings[i], &embeddings[j]).abs())
                .sum();
            (i, total_sim / (n - 1).max(1) as f32)
        })
        .collect();

    avg_sims.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    avg_sims[..keep_count].iter().map(|&(idx, _)| idx).collect()
}

// ─────────────────────────────────────────────────────────────
// Helper math
// ─────────────────────────────────────────────────────────────

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-8 || norm_b < 1e-8 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

// ─────────────────────────────────────────────────────────────
// Link prediction registry
// ─────────────────────────────────────────────────────────────

/// Defines a link prediction task: which source and target entity types
/// should be linked, via which relation, and the intermediate path.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LinkPredictionTask {
    /// Unique name for this task (e.g. "categorize", "allocate_subledger").
    pub name: String,
    /// Source entity type (e.g. "transaction-evidence").
    pub source_type: String,
    /// Target entity type (e.g. "transaction-category").
    pub target_type: String,
    /// Relation being predicted (e.g. "evidence-has-category").
    pub relation: String,
    /// Schema path from source to target (for documentation / multi-hop context).
    /// e.g. ["evidence-has-category"] for direct, or
    ///      ["case-has-evidence", "case-has-entry", "allocation-has-journal-entry"]
    pub path: Vec<String>,
    /// Description of what this prediction does.
    pub description: String,
}

/// Registry of link prediction tasks.
///
/// Architecture:
/// - **One shared GNN ensemble** trains on the entire graph → embeddings for ALL node types
/// - **One PC per task** learns P(target | dot_score, norm) from historical data for that task
/// - The `LinkPredictor` is shared across all tasks (same scoring logic)
///
/// Why shared GNN?
/// The GNN sees the FULL heterogeneous graph. A 4-layer RGCN can propagate information
/// across 4 hops, so even source-target pairs that are 3-4 relations apart will have
/// their structural context baked into the embeddings. Training separate GNNs per task
/// would lose this cross-task signal.
pub struct LinkPredictionRegistry {
    /// Registered prediction tasks, keyed by task name.
    pub tasks: HashMap<String, LinkPredictionTask>,
    /// Shared predictor (same config for all tasks).
    pub predictor: LinkPredictor,
    /// Per-task trained PCs: task_name → CompiledCircuit.
    pub circuits: HashMap<String, CompiledCircuit>,
}

impl LinkPredictionRegistry {
    /// Create a new registry with default config.
    pub fn new(config: LinkPredictorConfig) -> Self {
        Self {
            tasks: HashMap::new(),
            predictor: LinkPredictor::new(config),
            circuits: HashMap::new(),
        }
    }

    /// Create a registry pre-loaded with built-in FinCat tasks.
    pub fn with_builtin_tasks(config: LinkPredictorConfig) -> Self {
        let mut registry = Self::new(config);
        for task in builtin_tasks() {
            registry.register(task);
        }
        registry
    }

    /// Register a new prediction task.
    pub fn register(&mut self, task: LinkPredictionTask) {
        self.tasks.insert(task.name.clone(), task);
    }

    /// Train the PC for a specific task from historical data.
    ///
    /// `historical_scores`: (dot_product, embedding_norm, target_id) from past predictions.
    pub fn train_pc(
        &mut self,
        task_name: &str,
        historical_scores: &[(f32, f32, usize)],
        num_targets: usize,
        em_epochs: usize,
    ) {
        let circuit = build_link_pc(historical_scores, num_targets, em_epochs);
        self.circuits.insert(task_name.to_string(), circuit);
    }

    /// Predict for a registered task (dot-product only).
    pub fn predict(
        &self,
        task_name: &str,
        source_emb: &[f32],
        target_embs: &[(String, usize, String, Vec<f32>)],
        source_id: usize,
    ) -> Option<LinkPredictionResult> {
        let task = self.tasks.get(task_name)?;
        Some(self.predictor.predict(
            source_emb,
            target_embs,
            None,
            None,
            &task.source_type,
            source_id,
        ))
    }

    /// Predict with PC calibration for a registered task.
    pub fn predict_with_pc(
        &mut self,
        task_name: &str,
        source_emb: &[f32],
        target_embs: &[(String, usize, String, Vec<f32>)],
        source_id: usize,
    ) -> Option<(LinkPredictionResult, Vec<PcLinkAnalysis>)> {
        let task = self.tasks.get(task_name)?;
        let circuit = self.circuits.get_mut(task_name)?;
        Some(self.predictor.predict_with_pc(
            source_emb,
            target_embs,
            circuit,
            &task.source_type,
            source_id,
        ))
    }

    /// List all registered task names.
    pub fn task_names(&self) -> Vec<&str> {
        self.tasks.keys().map(|s| s.as_str()).collect()
    }

    /// Get a registered task by name.
    pub fn get_task(&self, name: &str) -> Option<&LinkPredictionTask> {
        self.tasks.get(name)
    }

    /// Check if a task has a trained PC.
    pub fn has_trained_pc(&self, task_name: &str) -> bool {
        self.circuits.contains_key(task_name)
    }

    /// Auto-train PC for a task using embeddings and graph structure (KumoRFM §2.2).
    ///
    /// Extracts (dot_score, embedding_norm, target_id) tuples from existing edges
    /// in the graph, then trains the PC. No manual data curation needed.
    ///
    /// # Arguments
    /// - `task_name`: registered task name
    /// - `source_embeddings`: HashMap from source entity name → embedding vector
    /// - `target_embeddings`: Vec of (target_name, target_id, embedding_vector)
    /// - `edges`: Vec of (source_name, target_id) pairs — the known positive edges
    /// - `em_epochs`: number of EM training epochs for the PC
    pub fn auto_train_pc(
        &mut self,
        task_name: &str,
        source_embeddings: &std::collections::HashMap<String, Vec<f32>>,
        target_embeddings: &[(String, usize, Vec<f32>)],
        edges: &[(String, usize)], // (source_name, target_id)
        em_epochs: usize,
    ) -> bool {
        if !self.tasks.contains_key(task_name) {
            return false;
        }

        let num_targets = target_embeddings.len();
        if num_targets == 0 || edges.is_empty() {
            return false;
        }

        // Build training samples: for each edge, compute dot-product and norm
        let mut historical: Vec<(f32, f32, usize)> = Vec::new();

        for (src_name, target_id) in edges {
            if let Some(src_emb) = source_embeddings.get(src_name) {
                if let Some((_, _, tgt_emb)) =
                    target_embeddings.iter().find(|(_, id, _)| id == target_id)
                {
                    // Dot product
                    let dot: f32 = src_emb.iter().zip(tgt_emb.iter()).map(|(a, b)| a * b).sum();
                    // L2 norm of source embedding
                    let norm: f32 = src_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                    historical.push((dot, norm, *target_id));
                }
            }
        }

        if historical.len() < 3 {
            return false; // too few samples for meaningful PC
        }

        self.train_pc(task_name, &historical, num_targets, em_epochs);
        true
    }
}

/// Perturbation-based feature importance for link predictions (KumoRFM §2.4).
///
/// For each feature dimension, perturb by ±σ and measure score change.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FeatureImportance {
    /// Feature index
    pub feature_idx: usize,
    /// Importance score (absolute change in dot-product score)
    pub importance: f32,
    /// Direction: positive means this feature increases the score
    pub direction: f32,
}

/// Compute perturbation-based feature importance for a link prediction.
///
/// Returns top-K most important features sorted by importance.
pub fn explain_prediction(
    source_emb: &[f32],
    target_emb: &[f32],
    top_k: usize,
) -> Vec<FeatureImportance> {
    let dim = source_emb.len().min(target_emb.len());
    if dim == 0 {
        return Vec::new();
    }

    // Base score
    let base_score: f32 = source_emb
        .iter()
        .zip(target_emb.iter())
        .map(|(a, b)| a * b)
        .sum();

    // Compute σ of the source embedding for perturbation scale
    let mean: f32 = source_emb.iter().sum::<f32>() / dim as f32;
    let variance: f32 = source_emb.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / dim as f32;
    let sigma = variance.sqrt().max(0.01); // avoid zero perturbation

    let mut importances: Vec<FeatureImportance> = Vec::with_capacity(dim);

    for i in 0..dim {
        // Perturb feature i by +σ
        let mut perturbed = source_emb.to_vec();
        perturbed[i] += sigma;
        let score_plus: f32 = perturbed
            .iter()
            .zip(target_emb.iter())
            .map(|(a, b)| a * b)
            .sum();

        // Perturb feature i by -σ
        perturbed[i] = source_emb[i] - sigma;
        let score_minus: f32 = perturbed
            .iter()
            .zip(target_emb.iter())
            .map(|(a, b)| a * b)
            .sum();

        let importance = ((score_plus - base_score).abs() + (score_minus - base_score).abs()) / 2.0;
        let direction = score_plus - score_minus; // positive = feature increases score

        importances.push(FeatureImportance {
            feature_idx: i,
            importance,
            direction,
        });
    }

    importances.sort_by(|a, b| {
        b.importance
            .partial_cmp(&a.importance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    importances.truncate(top_k);
    importances
}

/// Built-in prediction tasks for the FinCat schema.
pub fn builtin_tasks() -> Vec<LinkPredictionTask> {
    vec![
        LinkPredictionTask {
            name: "categorize".into(),
            source_type: "transaction-evidence".into(),
            target_type: "transaction-category".into(),
            relation: "evidence-has-category".into(),
            path: vec!["evidence-has-category".into()],
            description: "Predict which category a transaction belongs to".into(),
        },
        LinkPredictionTask {
            name: "allocate_subledger".into(),
            source_type: "reconciliation-case".into(),
            target_type: "sub-ledger".into(),
            relation: "case-allocation".into(),
            path: vec![
                "case-has-entry".into(),
                "allocation-has-journal-entry".into(),
            ],
            description: "Predict which sub-ledger a case should be allocated to \
                           (e.g. Rent, Groceries, Utilities)"
                .into(),
        },
        LinkPredictionTask {
            name: "map_instrument_to_ledger".into(),
            source_type: "instrument".into(),
            target_type: "sub-ledger".into(),
            relation: "instrument-mapped-to-ledger".into(),
            path: vec!["instrument-mapped-to-ledger".into()],
            description: "Predict which ledger an instrument should be mapped to".into(),
        },
        LinkPredictionTask {
            name: "assign_tax_code".into(),
            source_type: "reconciliation-case".into(),
            target_type: "tax-code".into(),
            relation: "tax-code-assigned-to-subject".into(),
            path: vec!["tax-code-assigned-to-subject".into()],
            description: "Predict which tax code should be assigned to a case".into(),
        },
        LinkPredictionTask {
            name: "match_recurring_pattern".into(),
            source_type: "reconciliation-case".into(),
            target_type: "recurring-pattern".into(),
            relation: "pattern-has-case".into(),
            path: vec!["pattern-has-case".into()],
            description: "Predict which recurring pattern a case belongs to".into(),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict_basic() {
        let predictor = LinkPredictor::new(LinkPredictorConfig {
            top_k: 3,
            ..Default::default()
        });
        let src = vec![1.0, 0.0, 0.5, 0.0];
        let targets = vec![
            (
                "sub-ledger".into(),
                0,
                "Utilities".into(),
                vec![0.1, 0.9, 0.0, 0.1],
            ),
            (
                "sub-ledger".into(),
                1,
                "Transportation".into(),
                vec![0.8, 0.0, 0.6, 0.0],
            ),
            (
                "sub-ledger".into(),
                2,
                "Rent".into(),
                vec![0.0, 0.5, 0.5, 0.0],
            ),
            (
                "sub-ledger".into(),
                3,
                "Groceries".into(),
                vec![0.0, 0.0, 0.1, 0.9],
            ),
        ];

        let result = predictor.predict(&src, &targets, None, None, "reconciliation-case", 42);
        assert_eq!(result.predictions.len(), 3);
        assert_eq!(result.predictions[0].target_name, "Transportation");
        println!(
            "✅ Generic link prediction: {} → {}",
            result.source_type, result.predictions[0].target_name
        );
    }

    #[test]
    fn test_topk_nn_early_exit() {
        let predictor = LinkPredictor::new(LinkPredictorConfig {
            top_k: 3,
            nn_sim_threshold: 0.8,
            nn_min_targets: 1,
            ..Default::default()
        });
        let src = vec![1.0, 0.0, 0.5, 0.0];
        let hist = vec![
            (vec![0.99, 0.01, 0.49, 0.01], 1usize),
            (vec![0.98, 0.02, 0.48, 0.02], 2),
            (vec![0.0, 1.0, 0.0, 1.0], 3),
        ];
        let hist_targets = vec![
            ("sub-ledger".into(), 1, "Transportation".into()),
            ("sub-ledger".into(), 2, "Fuel".into()),
            ("sub-ledger".into(), 3, "Marketing".into()),
        ];
        let targets = vec![
            (
                "sub-ledger".into(),
                1,
                "Transportation".into(),
                vec![0.8, 0.0, 0.6, 0.0],
            ),
            (
                "sub-ledger".into(),
                2,
                "Fuel".into(),
                vec![0.7, 0.1, 0.5, 0.0],
            ),
            (
                "sub-ledger".into(),
                3,
                "Marketing".into(),
                vec![0.0, 0.5, 0.5, 0.0],
            ),
        ];

        let result =
            predictor.predict(&src, &targets, Some(&hist), Some(&hist_targets), "case", 42);
        assert!(result.used_early_exit);
        assert!(result.nn_matches >= 2);
        println!("✅ TopK-NN: {} matches", result.nn_matches);
    }

    #[test]
    fn test_weighted_negative_sampling() {
        let freqs = vec![1000, 500, 200, 10];
        let positives = vec![(0, 0), (1, 1), (2, 2), (3, 3), (4, 0)];
        let negatives = weighted_negative_sample(&positives, &freqs, 3);
        assert!(negatives.len() >= 10);
        let mut neg_counts = vec![0u32; 4];
        for &(_s, t) in &negatives {
            neg_counts[t] += 1;
        }
        assert!(neg_counts[0] > neg_counts[3]);
        println!("✅ Weighted negative sampling: {:?}", neg_counts);
    }

    #[test]
    fn test_diversity_filter() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.99, 0.01, 0.0],
            vec![0.98, 0.02, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.5, 0.5, 0.5],
        ];
        let selected = diversity_filter(&embeddings, 0.5);
        assert_eq!(selected.len(), 3);
        assert!(selected.contains(&3) || selected.contains(&4));
        println!("✅ Diversity filter: {:?}", selected);
    }

    #[test]
    fn test_predict_with_pc() {
        let predictor = LinkPredictor::new(LinkPredictorConfig {
            top_k: 3,
            ..Default::default()
        });
        let src = vec![1.0, 0.0, 0.5, 0.0];
        let targets = vec![
            (
                "sub-ledger".into(),
                0,
                "Utilities".into(),
                vec![0.1, 0.9, 0.0, 0.1],
            ),
            (
                "sub-ledger".into(),
                1,
                "Transportation".into(),
                vec![0.8, 0.0, 0.6, 0.0],
            ),
            (
                "sub-ledger".into(),
                2,
                "Rent".into(),
                vec![0.0, 0.5, 0.5, 0.0],
            ),
            (
                "sub-ledger".into(),
                3,
                "Groceries".into(),
                vec![0.0, 0.0, 0.1, 0.9],
            ),
        ];
        let historical: Vec<(f32, f32, usize)> = vec![
            (1.1, 1.2, 1),
            (0.9, 1.1, 1),
            (0.3, 0.8, 0),
            (0.2, 0.7, 0),
            (0.5, 1.0, 2),
            (0.0, 0.9, 3),
            (1.0, 1.3, 1),
            (0.8, 1.0, 1),
            (-0.1, 0.5, 3),
            (0.4, 0.8, 2),
        ];
        let mut circuit = build_link_pc(&historical, 4, 10);
        let (result, pc) = predictor.predict_with_pc(&src, &targets, &mut circuit, "case", 42);
        assert_eq!(result.predictions[0].source, "gnn_ensemble+pc");
        for a in &pc {
            assert!(a.probability >= 0.0 && a.probability <= 1.0);
        }
        println!("✅ PC link prediction: P={:.3}", pc[0].probability);
    }

    #[test]
    fn test_registry_builtin_tasks() {
        let registry = LinkPredictionRegistry::with_builtin_tasks(LinkPredictorConfig::default());
        assert_eq!(registry.tasks.len(), 5);
        assert!(registry.get_task("categorize").is_some());
        assert!(registry.get_task("allocate_subledger").is_some());
        assert!(registry.get_task("map_instrument_to_ledger").is_some());
        assert!(registry.get_task("assign_tax_code").is_some());
        assert!(registry.get_task("match_recurring_pattern").is_some());

        let cat = registry.get_task("categorize").unwrap();
        assert_eq!(cat.source_type, "transaction-evidence");
        assert_eq!(cat.target_type, "transaction-category");

        let sub = registry.get_task("allocate_subledger").unwrap();
        assert_eq!(sub.source_type, "reconciliation-case");
        assert_eq!(sub.path.len(), 2); // multi-hop

        println!(
            "✅ Registry: {} built-in tasks loaded",
            registry.tasks.len()
        );
    }

    #[test]
    fn test_registry_predict() {
        let registry = LinkPredictionRegistry::with_builtin_tasks(LinkPredictorConfig {
            top_k: 2,
            ..Default::default()
        });
        let src = vec![1.0, 0.0, 0.5, 0.0];
        let targets = vec![
            (
                "sub-ledger".into(),
                0,
                "Rent".into(),
                vec![0.8, 0.0, 0.6, 0.0],
            ),
            (
                "sub-ledger".into(),
                1,
                "Groceries".into(),
                vec![0.0, 1.0, 0.0, 0.1],
            ),
        ];

        let result = registry.predict("allocate_subledger", &src, &targets, 42);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.source_type, "reconciliation-case");
        assert_eq!(result.predictions[0].target_name, "Rent");
        println!(
            "✅ Registry predict: {} → {}",
            result.source_type, result.predictions[0].target_name
        );
    }

    #[test]
    fn test_registry_predict_with_pc() {
        let mut registry = LinkPredictionRegistry::with_builtin_tasks(LinkPredictorConfig {
            top_k: 2,
            ..Default::default()
        });

        // Train PC for categorize task
        let historical = vec![
            (1.1, 1.2, 0usize),
            (0.3, 0.8, 1),
            (0.9, 1.1, 0),
            (0.2, 0.7, 1),
            (0.5, 1.0, 0),
            (-0.1, 0.5, 1),
            (1.0, 1.3, 0),
            (0.8, 1.0, 0),
            (0.0, 0.9, 1),
            (0.4, 0.8, 1),
        ];
        registry.train_pc("categorize", &historical, 2, 10);
        assert!(registry.has_trained_pc("categorize"));
        assert!(!registry.has_trained_pc("allocate_subledger"));

        let src = vec![1.0, 0.0, 0.5, 0.0];
        let targets = vec![
            (
                "category".into(),
                0,
                "Utilities".into(),
                vec![0.8, 0.0, 0.6, 0.0],
            ),
            (
                "category".into(),
                1,
                "Rent".into(),
                vec![0.0, 1.0, 0.0, 0.1],
            ),
        ];

        let result = registry.predict_with_pc("categorize", &src, &targets, 42);
        assert!(result.is_some());
        let (result, analyses) = result.unwrap();
        assert_eq!(result.source_type, "transaction-evidence");
        for a in &analyses {
            assert!(a.probability >= 0.0 && a.probability <= 1.0);
        }
        println!("✅ Registry PC predict: P={:.3}", analyses[0].probability);
    }

    #[test]
    fn test_registry_custom_task() {
        let mut registry = LinkPredictionRegistry::new(LinkPredictorConfig::default());
        registry.register(LinkPredictionTask {
            name: "custom_link".into(),
            source_type: "entity-a".into(),
            target_type: "entity-b".into(),
            relation: "a-links-to-b".into(),
            path: vec!["a-through-c".into(), "c-to-b".into()],
            description: "Custom multi-hop prediction".into(),
        });
        assert_eq!(registry.tasks.len(), 1);
        assert!(registry.get_task("custom_link").is_some());
        println!("✅ Custom task registered");
    }

    #[test]
    fn test_auto_train_pc() {
        use std::collections::HashMap;
        let mut registry = LinkPredictionRegistry::with_builtin_tasks(LinkPredictorConfig {
            top_k: 2,
            ..Default::default()
        });

        // Build mock embeddings
        let mut source_embs: HashMap<String, Vec<f32>> = HashMap::new();
        source_embs.insert("ev_0".into(), vec![1.0, 0.0, 0.5, 0.0]);
        source_embs.insert("ev_1".into(), vec![0.0, 1.0, 0.0, 0.5]);
        source_embs.insert("ev_2".into(), vec![0.5, 0.5, 0.3, 0.2]);
        source_embs.insert("ev_3".into(), vec![0.2, 0.8, 0.1, 0.4]);
        source_embs.insert("ev_4".into(), vec![0.9, 0.1, 0.4, 0.0]);

        let target_embs = vec![
            ("Utilities".into(), 0usize, vec![0.8, 0.0, 0.6, 0.0]),
            ("Rent".into(), 1, vec![0.0, 0.9, 0.0, 0.5]),
        ];

        let edges = vec![
            ("ev_0".into(), 0usize),
            ("ev_1".into(), 1),
            ("ev_2".into(), 0),
            ("ev_3".into(), 1),
            ("ev_4".into(), 0),
        ];

        let ok = registry.auto_train_pc("categorize", &source_embs, &target_embs, &edges, 10);
        assert!(ok);
        assert!(registry.has_trained_pc("categorize"));

        // Should fail for non-existent task
        let ok = registry.auto_train_pc("nonexistent", &source_embs, &target_embs, &edges, 10);
        assert!(!ok);

        println!("✅ Auto-train PC: trained from {} edges", edges.len());
    }

    #[test]
    fn test_explain_prediction() {
        let src = vec![1.0, 0.0, 0.5, 0.0];
        let tgt = vec![0.8, 0.0, 0.6, 0.0];

        let importances = super::explain_prediction(&src, &tgt, 3);
        assert_eq!(importances.len(), 3);

        // Feature 0 should be most important (src[0]=1.0, tgt[0]=0.8 → big dot product)
        assert_eq!(importances[0].feature_idx, 0);
        assert!(importances[0].importance > 0.0);
        assert!(importances[0].direction > 0.0); // increases score

        for fi in &importances {
            println!(
                "  feat[{}]: importance={:.4}, direction={:.4}",
                fi.feature_idx, fi.importance, fi.direction
            );
        }
        println!(
            "✅ Feature importance: top feature = {}",
            importances[0].feature_idx
        );
    }
}
