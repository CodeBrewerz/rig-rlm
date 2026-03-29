//! # Evolving Rubric Reward — LLM-as-Judge Scoring with Adaptive Criteria
//!
//! Inspired by DR-Tulu (arXiv:2511.19399), this module implements a
//! **self-evolving rubric system** that discovers task-specific quality
//! dimensions instead of relying on hardcoded keyword/format checks.
//!
//! ## Architecture (3-Layer)
//!
//! ```text
//!   ┌─────────────────────────────────────────── Rubric Buffer ───┐
//!   │                                                              │
//!   │  Persistent Rubrics ─── always scored ──→  Weighted         │
//!   │  (ground truth)                              Reward          │
//!   │                                              │              │
//!   │  Active Adaptive    ─── scored + filtered ──→  ↑            │
//!   │  (LLM-generated)                                            │
//!   │                                                              │
//!   │  Inactive Adaptive  ─── retired (std=0) ────→  ∅            │
//!   │                                                              │
//!   └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Category Theory
//!
//! - **Criterion Category** `Crit`: objects are rubric items
//! - **Judge Functor** `J: Crit → [0,1]`: LLM scores each criterion
//! - **Evolution** `η_t → η_{t+1}`: natural transformation refining criteria
//! - **Retirement** = quotient by `r₁ ~ r₂ iff std(J(r)) = 0`

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::monad::provider::LlmProvider;

// ═════════════════════════════════════════════════════════════════════════════
// Rubric Item — A Single Scoring Criterion
// ═════════════════════════════════════════════════════════════════════════════

/// Whether a rubric is permanent or can be evolved/retired.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RubricType {
    /// Always scored — analogous to DR-Tulu's ground-truth rubrics.
    Persistent,
    /// LLM-generated, can be retired if non-discriminative.
    Adaptive,
}

/// A single scoring criterion.
///
/// Mirrors DR-Tulu's rubric format:
/// ```json
/// { "description": "...", "title": "...", "weight": 1.0 }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RubricItem {
    /// Detailed description of what to evaluate.
    /// Example: "Response provides specific quantitative data with units"
    pub description: String,
    /// Short abstract label (2–5 words).
    /// Example: "Quantitative Evidence"
    pub title: String,
    /// Scoring weight: +1.0 for positive criteria, -1.0 for penalty criteria.
    pub weight: f64,
    /// Lifecycle type.
    pub rubric_type: RubricType,
}

impl RubricItem {
    /// Create a new persistent (always-active) rubric.
    pub fn persistent(title: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            description: description.into(),
            weight: 1.0,
            rubric_type: RubricType::Persistent,
        }
    }

    /// Create a new positive adaptive rubric.
    pub fn adaptive(title: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            description: description.into(),
            weight: 1.0,
            rubric_type: RubricType::Adaptive,
        }
    }

    /// Create a negative (penalty) adaptive rubric.
    pub fn negative(title: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            description: description.into(),
            weight: -1.0,
            rubric_type: RubricType::Adaptive,
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Per-Rubric Score Record — for std tracking
// ═════════════════════════════════════════════════════════════════════════════

/// Tracks score history for a single rubric to compute variance.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct ScoreHistory {
    scores: Vec<f64>,
}

impl ScoreHistory {
    fn push(&mut self, score: f64) {
        self.scores.push(score);
    }

    fn mean(&self) -> f64 {
        if self.scores.is_empty() { return 0.0; }
        self.scores.iter().sum::<f64>() / self.scores.len() as f64
    }

    fn std(&self) -> f64 {
        if self.scores.len() < 2 { return 0.0; }
        let m = self.mean();
        let variance = self.scores.iter()
            .map(|s| (s - m).powi(2))
            .sum::<f64>() / self.scores.len() as f64;
        variance.sqrt()
    }

    fn len(&self) -> usize {
        self.scores.len()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Rubric Buffer — Lifecycle Management (Active/Inactive)
// ═════════════════════════════════════════════════════════════════════════════

/// Per-task rubric lifecycle manager.
///
/// Mirrors DR-Tulu's rubric_buffer pattern:
/// - Persistent rubrics: always scored
/// - Active adaptive: scored + participate in evolution
/// - Inactive adaptive: retired (non-discriminative, std≈0)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RubricBuffer {
    /// Ground-truth criteria — always evaluated.
    pub persistent: Vec<RubricItem>,
    /// Currently active adaptive rubrics.
    pub active: Vec<RubricItem>,
    /// Retired rubrics (non-discriminative).
    pub inactive: Vec<RubricItem>,
    /// Score history per rubric title.
    score_history: HashMap<String, ScoreHistory>,
    /// Maximum active adaptive rubrics (DR-Tulu default: 5).
    pub max_active: usize,
    /// Minimum observations before a rubric can be retired.
    pub min_observations_for_retirement: usize,
}

impl Default for RubricBuffer {
    fn default() -> Self {
        Self {
            persistent: Vec::new(),
            active: Vec::new(),
            inactive: Vec::new(),
            score_history: HashMap::new(),
            max_active: 5,
            min_observations_for_retirement: 3,
        }
    }
}

impl RubricBuffer {
    /// Create a buffer with default persistent rubrics for document QA.
    pub fn for_document_qa() -> Self {
        Self {
            persistent: vec![
                RubricItem::persistent(
                    "Factual Recall",
                    "The response accurately includes specific facts, names, numbers, \
                     or data points that are present in the source document.",
                ),
                RubricItem::persistent(
                    "Answer Relevance",
                    "The response directly addresses the question asked, without \
                     significant tangential or off-topic content.",
                ),
                RubricItem::persistent(
                    "Completeness",
                    "The response covers the key aspects of the question without \
                     major omissions of information available in the document.",
                ),
            ],
            ..Default::default()
        }
    }

    /// All rubrics that should be scored right now (persistent + active).
    pub fn all_active(&self) -> Vec<&RubricItem> {
        self.persistent.iter().chain(self.active.iter()).collect()
    }

    /// Record per-rubric scores from one evaluation.
    pub fn record_scores(&mut self, scores: &HashMap<String, f64>) {
        for (title, &score) in scores {
            self.score_history
                .entry(title.clone())
                .or_default()
                .push(score);
        }
    }

    /// Filter and retire non-discriminative rubrics (DR-Tulu's zero-std filter).
    ///
    /// A rubric is retired if:
    /// 1. It has been scored >= `min_observations_for_retirement` times
    /// 2. Its score standard deviation < 0.01 (all responses score the same)
    ///
    /// Also caps active rubrics at `max_active` by keeping highest-std ones.
    pub fn filter_and_retire(&mut self) -> RetirementReport {
        let mut retired_zero_std = 0;
        let mut retired_cap = 0;

        // Phase 1: Retire zero-std rubrics
        let mut to_retire = Vec::new();
        for (i, rubric) in self.active.iter().enumerate() {
            if let Some(history) = self.score_history.get(&rubric.title) {
                if history.len() >= self.min_observations_for_retirement
                    && history.std() < 0.01
                {
                    to_retire.push(i);
                }
            }
        }

        // Remove in reverse order to preserve indices
        for &i in to_retire.iter().rev() {
            let rubric = self.active.remove(i);
            eprintln!(
                "📊 [Rubric] Retired {:?} (std={:.4}, non-discriminative)",
                rubric.title,
                self.score_history
                    .get(&rubric.title)
                    .map(|h| h.std())
                    .unwrap_or(0.0)
            );
            self.inactive.push(rubric);
            retired_zero_std += 1;
        }

        // Phase 2: Cap at max_active by keeping highest-std rubrics
        if self.active.len() > self.max_active {
            // Sort by std descending
            let mut indexed: Vec<(usize, f64)> = self.active.iter().enumerate()
                .map(|(i, r)| {
                    let std = self.score_history.get(&r.title)
                        .map(|h| h.std())
                        .unwrap_or(0.0);
                    (i, std)
                })
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Keep top max_active, retire the rest
            let to_keep: std::collections::HashSet<usize> =
                indexed.iter().take(self.max_active).map(|&(i, _)| i).collect();

            let mut new_active = Vec::new();
            for (i, rubric) in self.active.drain(..).enumerate() {
                if to_keep.contains(&i) {
                    new_active.push(rubric);
                } else {
                    self.inactive.push(rubric);
                    retired_cap += 1;
                }
            }
            self.active = new_active;
        }

        RetirementReport {
            retired_zero_std,
            retired_cap,
            remaining_active: self.active.len(),
            total_inactive: self.inactive.len(),
        }
    }

    /// Add new adaptive rubrics from LLM generation.
    pub fn add_adaptive(&mut self, rubrics: Vec<RubricItem>) {
        for rubric in rubrics {
            eprintln!(
                "📊 [Rubric] Added adaptive {:?}: {:?} (weight={:.1})",
                rubric.title,
                &rubric.description[..rubric.description.len().min(60)],
                rubric.weight
            );
            self.active.push(rubric);
        }
    }

    /// Statistics summary.
    pub fn summary(&self) -> String {
        let mut lines = vec![
            format!("  persistent: {} rubrics", self.persistent.len()),
            format!("  active:     {} rubrics (max={})", self.active.len(), self.max_active),
            format!("  inactive:   {} rubrics (retired)", self.inactive.len()),
        ];

        // Show per-rubric stats
        for rubric in self.persistent.iter().chain(self.active.iter()) {
            let (mean, std, n) = self.score_history.get(&rubric.title)
                .map(|h| (h.mean(), h.std(), h.len()))
                .unwrap_or((0.0, 0.0, 0));
            let kind = match rubric.rubric_type {
                RubricType::Persistent => "P",
                RubricType::Adaptive => "A",
            };
            lines.push(format!(
                "    [{}] {:25} mean={:.3} std={:.3} n={} w={:+.1}",
                kind, rubric.title, mean, std, n, rubric.weight
            ));
        }

        lines.join("\n")
    }

    // ─── Disk Persistence ───────────────────────────────────────────

    /// Save rubric buffer to a JSON file (atomic write).
    ///
    /// Mirrors DR-Tulu's `save_adaptive_rubric_cache_safe()` — writes to
    /// a temp file first, then renames for atomicity.
    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let path = path.as_ref();
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Atomic write: write to tmp, then rename
        let tmp_path = path.with_extension("tmp");
        std::fs::write(&tmp_path, &json)?;
        std::fs::rename(&tmp_path, path)?;

        eprintln!(
            "💾 [Rubric] Saved buffer to {:?} ({} persistent, {} active, {} inactive, {} bytes)",
            path, self.persistent.len(), self.active.len(), self.inactive.len(), json.len()
        );
        Ok(())
    }

    /// Load rubric buffer from a JSON file.
    ///
    /// Returns `None` if the file doesn't exist (cold start).
    pub fn load(path: impl AsRef<Path>) -> Option<Self> {
        let path = path.as_ref();
        if !path.exists() {
            eprintln!("💾 [Rubric] No saved buffer at {:?}, starting fresh", path);
            return None;
        }

        match std::fs::read_to_string(path) {
            Ok(json) => match serde_json::from_str::<Self>(&json) {
                Ok(buffer) => {
                    eprintln!(
                        "💾 [Rubric] Loaded buffer from {:?} ({} persistent, {} active, {} inactive)",
                        path, buffer.persistent.len(), buffer.active.len(), buffer.inactive.len()
                    );
                    Some(buffer)
                }
                Err(e) => {
                    eprintln!("⚠️ [Rubric] Failed to parse {:?}: {}", path, e);
                    None
                }
            },
            Err(e) => {
                eprintln!("⚠️ [Rubric] Failed to read {:?}: {}", path, e);
                None
            }
        }
    }

    /// Load from disk if available, otherwise create with defaults.
    pub fn load_or_default(path: impl AsRef<Path>) -> Self {
        Self::load(path).unwrap_or_else(Self::for_document_qa)
    }

    // ─── Structured Metrics ─────────────────────────────────────────

    /// Compute structured metrics (DR-Tulu style).
    ///
    /// Mirrors `grpo_fast_rubric.py` metrics:
    /// - `avg_mean`: average of per-rubric means
    /// - `avg_std`: average of per-rubric stds (how discriminative overall)
    /// - `zero_rubrics_ratio`: fraction of rubrics with mean=0 AND std=0
    /// - `discriminative_ratio`: fraction with std > 0.01
    /// - `total_observations`: total scored rubric×response pairs
    pub fn metrics(&self) -> RubricMetrics {
        let active_rubrics = self.all_active();
        if active_rubrics.is_empty() {
            return RubricMetrics::default();
        }

        let mut means = Vec::new();
        let mut stds = Vec::new();
        let mut zero_count = 0;
        let mut discriminative_count = 0;
        let mut total_obs = 0;

        for rubric in &active_rubrics {
            if let Some(history) = self.score_history.get(&rubric.title) {
                let m = history.mean();
                let s = history.std();
                means.push(m);
                stds.push(s);
                total_obs += history.len();

                if m == 0.0 && s == 0.0 { zero_count += 1; }
                if s > 0.01 { discriminative_count += 1; }
            }
        }

        let n = means.len().max(1) as f64;
        RubricMetrics {
            avg_mean: means.iter().sum::<f64>() / n,
            avg_std: stds.iter().sum::<f64>() / n,
            zero_rubrics_ratio: zero_count as f64 / n,
            discriminative_ratio: discriminative_count as f64 / n,
            total_observations: total_obs,
            persistent_count: self.persistent.len(),
            active_count: self.active.len(),
            inactive_count: self.inactive.len(),
        }
    }

    // ─── Per-Rubric Z-Score Normalization ────────────────────────────

    /// Compute z-score normalized advantage (DR-Tulu's `normalize_rubric_scores`).
    ///
    /// For each rubric: `advantage = (score - mean) / std * weight`.
    /// The final score is the weighted average of z-normalized per-rubric scores.
    ///
    /// This prevents high-variance rubrics from dominating and makes each
    /// rubric equally influential in the final score.
    ///
    /// From `grpo_fast_rubric.py:2498-2514`:
    /// ```python
    /// if stats["std"] > 0:
    ///     normalized_score = (score - stats["mean"]) / stats["std"]
    /// else:
    ///     normalized_score = 0.0
    /// weighted_scores.append(normalized_score * weight)
    /// final_advantage = sum(weighted_scores) / max(total_weight, 1.0)
    /// ```
    pub fn z_score_normalize(&self, per_rubric_scores: &HashMap<String, f64>) -> f64 {
        let mut weighted_z_scores = Vec::new();
        let mut total_positive_weight = 0.0;

        for rubric in self.persistent.iter().chain(self.active.iter()) {
            if let Some(&raw_score) = per_rubric_scores.get(&rubric.title) {
                if let Some(history) = self.score_history.get(&rubric.title) {
                    let std = history.std();
                    let z = if std > 0.0 {
                        (raw_score - history.mean()) / std
                    } else {
                        0.0
                    };

                    weighted_z_scores.push(z * rubric.weight);
                    if rubric.weight > 0.0 {
                        total_positive_weight += rubric.weight;
                    }
                }
            }
        }

        if weighted_z_scores.is_empty() || total_positive_weight == 0.0 {
            return 0.0;
        }

        weighted_z_scores.iter().sum::<f64>() / total_positive_weight
    }
}

/// Structured metrics from a rubric evaluation cycle.
///
/// Mirrors DR-Tulu's per-step metrics logged to wandb:
/// - `rubric_keys/avg_mean`
/// - `rubric_keys/avg_std`
/// - `rubric_keys/num_all_zero_rubrics_ratio`
#[derive(Debug, Clone, Default, Serialize)]
pub struct RubricMetrics {
    /// Average of per-rubric mean scores.
    pub avg_mean: f64,
    /// Average of per-rubric std (higher = more discriminative overall).
    pub avg_std: f64,
    /// Fraction of rubrics with mean=0 AND std=0 (all-zero, likely useless).
    pub zero_rubrics_ratio: f64,
    /// Fraction of rubrics with std > 0.01 (actually discriminating).
    pub discriminative_ratio: f64,
    /// Total number of scored rubric×response pairs.
    pub total_observations: usize,
    /// Count of persistent rubrics.
    pub persistent_count: usize,
    /// Count of active adaptive rubrics.
    pub active_count: usize,
    /// Count of inactive (retired) rubrics.
    pub inactive_count: usize,
}

impl std::fmt::Display for RubricMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RubricMetrics {{ avg_mean={:.3}, avg_std={:.3}, zero_ratio={:.2}, \
             discrim_ratio={:.2}, obs={}, P/A/I={}/{}/{} }}",
            self.avg_mean, self.avg_std, self.zero_rubrics_ratio,
            self.discriminative_ratio, self.total_observations,
            self.persistent_count, self.active_count, self.inactive_count,
        )
    }
}

/// Report from a retirement cycle.
#[derive(Debug)]
pub struct RetirementReport {
    pub retired_zero_std: usize,
    pub retired_cap: usize,
    pub remaining_active: usize,
    pub total_inactive: usize,
}

impl std::fmt::Display for RetirementReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RetirementReport: retired_zero_std={}, retired_cap={}, active={}, inactive={}",
            self.retired_zero_std, self.retired_cap, self.remaining_active, self.total_inactive
        )
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// LLM-as-Judge — Score Response Against Rubric
// ═════════════════════════════════════════════════════════════════════════════

/// System prompt for the LLM judge (mirrors DR-Tulu's `_score_property`).
const JUDGE_SYSTEM_PROMPT: &str = r#"You will be given a question someone asked (in <question></question> tags) and the corresponding response (in <response></response> tags) given to them by an assistant. You will then be given a specific criterion of the response to evaluate (in <criterion></criterion> tags).
Return a score on a scale of 0 to 2 indicating how appropriate the response is based on the given criterion. Judge only the specified aspect(s), not any other qualities of the answer.

Score guidelines:
- 0: The response completely fails to meet this criterion
- 1: The response partially meets this criterion
- 2: The response fully meets this criterion

Output ONLY a JSON object in the format: {"score": x}
Do not include any other text."#;

/// Score a single response against a single rubric using LLM-as-Judge.
///
/// Returns a normalized score in [0.0, 1.0].
///
/// This mirrors DR-Tulu's `_score_property_async()` from rubric_utils.py.
pub async fn score_rubric(
    provider: &LlmProvider,
    query: &str,
    response: &str,
    rubric: &RubricItem,
) -> f64 {
    let user_prompt = format!(
        "<question>{}</question>\n<response>{}</response>\n<criterion>{}</criterion>",
        query,
        // Truncate response to avoid blowing context
        &response[..response.len().min(4000)],
        rubric.description,
    );

    let full_prompt = format!("{}\n\n{}", JUDGE_SYSTEM_PROMPT, user_prompt);

    match provider.complete(&full_prompt).await {
        Ok(resp) => parse_judge_score(&resp),
        Err(e) => {
            eprintln!("⚠️ [Judge] Error scoring rubric {:?}: {}", rubric.title, e);
            0.0
        }
    }
}

/// Score a response against ALL active rubrics (persistent + adaptive).
///
/// Returns (weighted_score, per_rubric_scores).
///
/// Rubrics are scored **sequentially** to avoid rate limits on free-tier models.
/// For production use with paid API, switch to `futures::future::join_all`.
pub async fn score_all_rubrics(
    provider: &LlmProvider,
    query: &str,
    response: &str,
    buffer: &RubricBuffer,
) -> (f64, HashMap<String, f64>) {
    let all_rubrics = buffer.all_active();

    if all_rubrics.is_empty() {
        return (0.0, HashMap::new());
    }

    let mut per_rubric = HashMap::new();
    let mut weighted_sum = 0.0;
    let mut total_positive_weight = 0.0;

    for rubric in &all_rubrics {
        let score = score_rubric(provider, query, response, rubric).await;
        per_rubric.insert(rubric.title.clone(), score);
        weighted_sum += score * rubric.weight;
        if rubric.weight > 0.0 {
            total_positive_weight += rubric.weight;
        }
    }

    let overall = weighted_sum / total_positive_weight.max(1.0);

    (overall, per_rubric)
}

/// Parse `{"score": x}` from LLM judge response, normalizing 0–2 → 0.0–1.0.
fn parse_judge_score(response: &str) -> f64 {
    // Try to find JSON in the response
    let json_start = response.find('{');
    let json_end = response.rfind('}');

    if let (Some(start), Some(end)) = (json_start, json_end) {
        if end > start {
            let json_str = &response[start..=end];
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
                if let Some(score) = parsed.get("score").and_then(|v| v.as_f64()) {
                    // Normalize 0–2 → 0.0–1.0, clamped
                    return (score / 2.0).clamp(0.0, 1.0);
                }
            }
        }
    }

    // Fallback: try to find a number directly
    for word in response.split_whitespace() {
        if let Ok(n) = word.trim_matches(|c: char| !c.is_ascii_digit() && c != '.').parse::<f64>() {
            return (n / 2.0).clamp(0.0, 1.0);
        }
    }

    eprintln!("⚠️ [Judge] Could not parse score from: {:?}", &response[..response.len().min(100)]);
    0.0
}

// ═════════════════════════════════════════════════════════════════════════════
// Adaptive Rubric Generation — LLM discovers new criteria
// ═════════════════════════════════════════════════════════════════════════════

/// System prompt for generating adaptive rubrics from response comparison.
/// Mirrors DR-Tulu's `INSTANCE_WISE_RUBRIC_GENERATION_PROMPT`.
const RUBRIC_GEN_PROMPT: &str = r#"You are an expert evaluator generating adaptive rubrics to assess model responses.

## Task
Given a question and multiple responses of varying quality, identify the most discriminative criteria that distinguish high-quality from low-quality answers. Capture subtle quality differences that existing rubrics miss.

## Core Guidelines
1. Focus ONLY on criteria that meaningfully separate quality levels
2. Never duplicate existing rubrics in meaning or scope
3. Avoid mirror rubrics (positive + negative of same criterion)
4. Negative rubrics should identify active mistakes, not absence of excellence
5. Return 1-3 total rubrics (quality > quantity)

## Output Format
Return ONLY a JSON object:
```json
{
  "positive_rubrics": [
    {"description": "<detailed excellence description>", "title": "<abstract label>"}
  ],
  "negative_rubrics": [
    {"description": "<detailed failure description>", "title": "<abstract label>"}
  ]
}
```"#;

/// Generate new adaptive rubrics by comparing multiple responses.
///
/// The LLM examines several responses to the same query and identifies
/// quality dimensions that the existing rubrics don't cover.
///
/// Returns a list of new `RubricItem`s (positive weight = +1.0, negative = -1.0).
pub async fn generate_adaptive_rubrics(
    provider: &LlmProvider,
    query: &str,
    responses: &[String],
    existing_rubrics: &[&RubricItem],
) -> Vec<RubricItem> {
    if responses.len() < 2 {
        // Need at least 2 responses to compare
        return Vec::new();
    }

    let responses_str: String = responses.iter().enumerate()
        .map(|(i, r)| format!("Response {}:\n{}", i + 1, &r[..r.len().min(600)]))
        .collect::<Vec<_>>()
        .join("\n\n");

    let existing_str: String = existing_rubrics.iter()
        .map(|r| format!("- {}: {}", r.title, r.description))
        .collect::<Vec<_>>()
        .join("\n");

    let user_prompt = format!(
        "{}\n\nQuestion: {}\n\n{}\n\nExisting Rubrics:\n{}",
        RUBRIC_GEN_PROMPT, query, responses_str, existing_str
    );

    let result = match provider.complete(&user_prompt).await {
        Ok(resp) => resp,
        Err(e) => {
            eprintln!("⚠️ [RubricGen] Error generating rubrics: {}", e);
            return Vec::new();
        }
    };

    parse_generated_rubrics(&result)
}

/// Parse LLM output into `RubricItem`s.
fn parse_generated_rubrics(response: &str) -> Vec<RubricItem> {
    let json_start = response.find('{');
    let json_end = response.rfind('}');

    let (start, end) = match (json_start, json_end) {
        (Some(s), Some(e)) if e > s => (s, e),
        _ => return Vec::new(),
    };

    let json_str = &response[start..=end];
    let parsed: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let mut rubrics = Vec::new();

    // Parse positive rubrics
    if let Some(positives) = parsed.get("positive_rubrics").and_then(|v| v.as_array()) {
        for item in positives {
            if let (Some(desc), Some(title)) = (
                item.get("description").and_then(|v| v.as_str()),
                item.get("title").and_then(|v| v.as_str()),
            ) {
                rubrics.push(RubricItem::adaptive(title, desc));
            }
        }
    }

    // Parse negative rubrics
    if let Some(negatives) = parsed.get("negative_rubrics").and_then(|v| v.as_array()) {
        for item in negatives {
            if let (Some(desc), Some(title)) = (
                item.get("description").and_then(|v| v.as_str()),
                item.get("title").and_then(|v| v.as_str()),
            ) {
                rubrics.push(RubricItem::negative(title, desc));
            }
        }
    }

    eprintln!(
        "📊 [RubricGen] Generated {} rubrics ({} positive, {} negative)",
        rubrics.len(),
        rubrics.iter().filter(|r| r.weight > 0.0).count(),
        rubrics.iter().filter(|r| r.weight < 0.0).count(),
    );

    rubrics
}

// ═════════════════════════════════════════════════════════════════════════════
// HyperRubricGenerator — Self-Modifying Rubric Generation (DGM-H Level 2)
// ═════════════════════════════════════════════════════════════════════════════

/// A self-modifying rubric generator that can evolve its own generation prompt.
///
/// This is the **metacognitive layer** from HyperAgents (arXiv:2603.19461):
/// instead of a fixed `RUBRIC_GEN_PROMPT`, the prompt itself is tracked and
/// evolved when the generated rubrics prove non-discriminative.
///
/// # Self-Improvement Loop
///
/// ```text
///     ┌──────────── HyperRubricGenerator ────────────┐
///     │                                                │
///     │  generation_prompt ──→ generate_adaptive()     │
///     │       ↑                      │                 │
///     │       │              rubrics produced          │
///     │  evolve_prompt()            │                  │
///     │       ↑              scored → discriminative?  │
///     │       │                      │                 │
///     │  discriminative_ratio < θ ←──┘                 │
///     │  (meta-failure signal)                         │
///     └────────────────────────────────────────────────┘
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperRubricGenerator {
    /// The current rubric generation prompt (starts as RUBRIC_GEN_PROMPT).
    pub generation_prompt: String,
    /// History of (prompt_version, discriminative_ratio) pairs.
    pub prompt_history: Vec<PromptEvolution>,
    /// Current prompt version number (0 = original).
    pub version: usize,
    /// Threshold below which the generator self-modifies (default: 0.5).
    pub discriminative_threshold: f64,
    /// Minimum generations before considering self-modification.
    pub min_generations_before_evolve: usize,
    /// Count of rubric generation calls since last prompt evolution.
    pub calls_since_evolve: usize,
}

/// A record of one prompt evolution event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptEvolution {
    /// The prompt text at this version.
    pub prompt: String,
    /// Discriminative ratio of rubrics generated by this prompt.
    pub discriminative_ratio: f64,
    /// How many rubrics were generated total.
    pub total_generated: usize,
    /// How many were retired as non-discriminative.
    pub total_retired: usize,
    /// Version number.
    pub version: usize,
}

/// System prompt for the meta-level: evolving the rubric generation prompt itself.
const HYPER_EVOLVE_PROMPT: &str = r#"You are a meta-optimizer improving a rubric generation system.

## Context
A rubric generation prompt is used to create quality criteria for evaluating AI responses.
The current prompt is producing rubrics that are NOT discriminative — they score all responses
the same way (std ≈ 0), making them useless for distinguishing good from bad responses.

## Your Task
Rewrite the rubric generation prompt to produce MORE DISCRIMINATIVE criteria.
Discriminative rubrics should:
1. Distinguish between high-quality and low-quality responses with measurable score variance
2. Be specific enough that different responses receive different scores
3. Target observable, concrete properties (not vague "quality" judgments)
4. Cover orthogonal dimensions (don't repeat the same criterion differently)

## Output Format
Return ONLY the improved prompt text. Do not include any explanation before or after.
The prompt should be self-contained — it will replace the existing prompt entirely.
Start directly with "You are an expert evaluator..." or similar."#;

impl HyperRubricGenerator {
    /// Create with the default (v0) rubric generation prompt.
    pub fn new() -> Self {
        Self {
            generation_prompt: RUBRIC_GEN_PROMPT.to_string(),
            prompt_history: vec![PromptEvolution {
                prompt: RUBRIC_GEN_PROMPT.to_string(),
                discriminative_ratio: 1.0, // optimistic initial
                total_generated: 0,
                total_retired: 0,
                version: 0,
            }],
            version: 0,
            discriminative_threshold: 0.5,
            min_generations_before_evolve: 3,
            calls_since_evolve: 0,
        }
    }

    /// Generate adaptive rubrics using the CURRENT (possibly evolved) prompt.
    ///
    /// This replaces the module-level `generate_adaptive_rubrics()` when the
    /// HyperRubricGenerator is in use.
    pub async fn generate(
        &mut self,
        provider: &LlmProvider,
        query: &str,
        responses: &[String],
        existing_rubrics: &[&RubricItem],
    ) -> Vec<RubricItem> {
        if responses.len() < 2 {
            return Vec::new();
        }

        let responses_str: String = responses.iter().enumerate()
            .map(|(i, r)| format!("Response {}:\n{}", i + 1, &r[..r.len().min(600)]))
            .collect::<Vec<_>>()
            .join("\n\n");

        let existing_str: String = existing_rubrics.iter()
            .map(|r| format!("- {}: {}", r.title, r.description))
            .collect::<Vec<_>>()
            .join("\n");

        // Use the EVOLVED prompt (not the constant)
        let user_prompt = format!(
            "{}\n\nQuestion: {}\n\n{}\n\nExisting Rubrics:\n{}",
            self.generation_prompt, query, responses_str, existing_str
        );

        let result = match provider.complete(&user_prompt).await {
            Ok(resp) => resp,
            Err(e) => {
                eprintln!("⚠️ [HyperRubricGen] Error generating rubrics: {}", e);
                return Vec::new();
            }
        };

        self.calls_since_evolve += 1;
        let rubrics = parse_generated_rubrics(&result);

        eprintln!(
            "🧠 [HyperRubricGen] v{} generated {} rubrics (calls_since_evolve={})",
            self.version, rubrics.len(), self.calls_since_evolve
        );

        rubrics
    }

    /// Check if the prompt should self-modify based on rubric buffer metrics.
    ///
    /// Returns `true` if the discriminative_ratio is below threshold AND
    /// enough calls have been made since the last evolution.
    pub fn should_evolve(&self, metrics: &RubricMetrics) -> bool {
        self.calls_since_evolve >= self.min_generations_before_evolve
            && metrics.discriminative_ratio < self.discriminative_threshold
            && metrics.total_observations >= 6 // need enough data
    }

    /// **The metacognitive step**: use the LLM to rewrite the generation prompt.
    ///
    /// This is the core HyperAgents insight applied to rubric generation:
    /// the meta agent doesn't just generate rubrics — it improves HOW it
    /// generates rubrics by analyzing what went wrong with past rubrics.
    pub async fn evolve_prompt(
        &mut self,
        provider: &LlmProvider,
        metrics: &RubricMetrics,
        retired_examples: &[RubricItem],
    ) {
        let retired_str: String = retired_examples.iter().take(5)
            .map(|r| format!("- \"{}\" ({}): {}", r.title, 
                if r.weight > 0.0 { "positive" } else { "negative" },
                r.description
            ))
            .collect::<Vec<_>>()
            .join("\n");

        let evolution_prompt = format!(
            "{}\n\n## Current Prompt (v{}):\n```\n{}\n```\n\n\
             ## Performance Metrics:\n\
             - Discriminative ratio: {:.2} (target: ≥{:.2})\n\
             - Average rubric std: {:.4}\n\
             - Zero-rubrics ratio: {:.2}\n\
             - Total observations: {}\n\n\
             ## Examples of Retired (Non-Discriminative) Rubrics:\n{}\n\n\
             ## What Went Wrong:\n\
             The rubrics above were retired because their score standard deviations \
             were ≈0, meaning all responses scored the same on these criteria. \
             The prompt needs to produce rubrics that create MEANINGFUL VARIANCE \
             across different quality levels of responses.",
            HYPER_EVOLVE_PROMPT,
            self.version,
            self.generation_prompt,
            metrics.discriminative_ratio,
            self.discriminative_threshold,
            metrics.avg_std,
            metrics.zero_rubrics_ratio,
            metrics.total_observations,
            retired_str,
        );

        match provider.complete(&evolution_prompt).await {
            Ok(new_prompt) => {
                let new_prompt = new_prompt.trim().to_string();
                if new_prompt.len() >= 100 {
                    // Record the old prompt's performance
                    if let Some(current) = self.prompt_history.last_mut() {
                        current.discriminative_ratio = metrics.discriminative_ratio;
                    }

                    self.version += 1;
                    self.calls_since_evolve = 0;

                    eprintln!(
                        "🧠 [HyperRubricGen] EVOLVED prompt v{} → v{} \
                         (disc_ratio {:.2} → target ≥{:.2})",
                        self.version - 1, self.version,
                        metrics.discriminative_ratio,
                        self.discriminative_threshold,
                    );
                    eprintln!(
                        "🧠 [HyperRubricGen] New prompt preview: {:?}...",
                        &new_prompt[..new_prompt.len().min(120)]
                    );

                    // Record new version
                    self.prompt_history.push(PromptEvolution {
                        prompt: new_prompt.clone(),
                        discriminative_ratio: 1.0, // optimistic until measured
                        total_generated: 0,
                        total_retired: 0,
                        version: self.version,
                    });

                    self.generation_prompt = new_prompt;
                } else {
                    eprintln!(
                        "⚠️ [HyperRubricGen] Evolved prompt too short ({}), keeping v{}",
                        new_prompt.len(), self.version
                    );
                }
            }
            Err(e) => {
                eprintln!("⚠️ [HyperRubricGen] Failed to evolve prompt: {}", e);
            }
        }
    }

    /// Summary of the hyper generator state.
    pub fn summary(&self) -> String {
        let history_str: String = self.prompt_history.iter()
            .map(|h| format!(
                "    v{}: disc_ratio={:.2}, gen={}, retired={}",
                h.version, h.discriminative_ratio, h.total_generated, h.total_retired
            ))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            "🧠 HyperRubricGenerator v{} (calls={})\n  Evolution History:\n{}",
            self.version, self.calls_since_evolve, history_str
        )
    }
}

impl Default for HyperRubricGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rubric_item_constructors() {
        let p = RubricItem::persistent("Recall", "Contains keywords");
        assert_eq!(p.rubric_type, RubricType::Persistent);
        assert!((p.weight - 1.0).abs() < f64::EPSILON);

        let a = RubricItem::adaptive("Depth", "Provides detail");
        assert_eq!(a.rubric_type, RubricType::Adaptive);

        let n = RubricItem::negative("Hallucination", "Makes up facts");
        assert!((n.weight - (-1.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_score_history_stats() {
        let mut h = ScoreHistory::default();
        assert_eq!(h.mean(), 0.0);
        assert_eq!(h.std(), 0.0);

        h.push(1.0);
        h.push(1.0);
        h.push(1.0);
        assert!((h.mean() - 1.0).abs() < f64::EPSILON);
        assert!(h.std() < 0.01, "All same scores → std ≈ 0");

        let mut h2 = ScoreHistory::default();
        h2.push(0.0);
        h2.push(0.5);
        h2.push(1.0);
        assert!((h2.mean() - 0.5).abs() < f64::EPSILON);
        assert!(h2.std() > 0.3, "Spread scores → std > 0");
    }

    #[test]
    fn test_rubric_buffer_lifecycle() {
        let mut buffer = RubricBuffer::for_document_qa();
        assert_eq!(buffer.persistent.len(), 3);
        assert_eq!(buffer.active.len(), 0);

        // Add some adaptive rubrics
        buffer.add_adaptive(vec![
            RubricItem::adaptive("Conciseness", "Answer is concise"),
            RubricItem::adaptive("Citations", "Cites sources"),
        ]);
        assert_eq!(buffer.active.len(), 2);
        assert_eq!(buffer.all_active().len(), 5);

        // Record uniform scores for "Conciseness" → should get retired
        let mut scores = HashMap::new();
        scores.insert("Conciseness".to_string(), 0.5);
        scores.insert("Citations".to_string(), 0.8);
        buffer.record_scores(&scores);

        scores.insert("Conciseness".to_string(), 0.5);
        scores.insert("Citations".to_string(), 0.2);
        buffer.record_scores(&scores);

        scores.insert("Conciseness".to_string(), 0.5);
        scores.insert("Citations".to_string(), 0.6);
        buffer.record_scores(&scores);

        let report = buffer.filter_and_retire();
        assert_eq!(report.retired_zero_std, 1, "Conciseness (std=0) should be retired");
        assert_eq!(buffer.active.len(), 1, "Only Citations should remain active");
        assert_eq!(buffer.inactive.len(), 1);
    }

    #[test]
    fn test_rubric_buffer_max_active_cap() {
        let mut buffer = RubricBuffer {
            max_active: 2,
            min_observations_for_retirement: 2,
            ..Default::default()
        };

        // Add 4 adaptive rubrics
        buffer.add_adaptive(vec![
            RubricItem::adaptive("A", "desc a"),
            RubricItem::adaptive("B", "desc b"),
            RubricItem::adaptive("C", "desc c"),
            RubricItem::adaptive("D", "desc d"),
        ]);

        // Give them varying stds
        for _ in 0..3 {
            let mut scores = HashMap::new();
            scores.insert("A".to_string(), 0.5); // zero std
            scores.insert("B".to_string(), 0.5); // zero std
            scores.insert("C".to_string(), 0.9); // will have std
            scores.insert("D".to_string(), 0.1); // will have std
            buffer.record_scores(&scores);
        }
        // Add variance to C and D
        let mut scores = HashMap::new();
        scores.insert("A".to_string(), 0.5);
        scores.insert("B".to_string(), 0.5);
        scores.insert("C".to_string(), 0.3);
        scores.insert("D".to_string(), 0.7);
        buffer.record_scores(&scores);

        let report = buffer.filter_and_retire();
        // A and B should be retired (zero std)
        // C and D should remain (but still capped at 2, so both stay)
        assert!(buffer.active.len() <= 2, "Should cap at max_active=2");
        assert!(report.retired_zero_std >= 2, "A and B are zero-std");
    }

    #[test]
    fn test_parse_judge_score() {
        assert!((parse_judge_score(r#"{"score": 2}"#) - 1.0).abs() < f64::EPSILON);
        assert!((parse_judge_score(r#"{"score": 1}"#) - 0.5).abs() < f64::EPSILON);
        assert!((parse_judge_score(r#"{"score": 0}"#) - 0.0).abs() < f64::EPSILON);
        // With surrounding text
        assert!((parse_judge_score(r#"Here is my eval: {"score": 1.5} done"#) - 0.75).abs() < f64::EPSILON);
        // Edge: out of range clamped
        assert!((parse_judge_score(r#"{"score": 5}"#) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_generated_rubrics() {
        let response = r#"Here are rubrics:
        {
            "positive_rubrics": [
                {"description": "Uses specific examples", "title": "Specificity"}
            ],
            "negative_rubrics": [
                {"description": "Contradicts source", "title": "Factual Error"}
            ]
        }"#;

        let rubrics = parse_generated_rubrics(response);
        assert_eq!(rubrics.len(), 2);
        assert!((rubrics[0].weight - 1.0).abs() < f64::EPSILON);
        assert!((rubrics[1].weight - (-1.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rubric_buffer_summary() {
        let buffer = RubricBuffer::for_document_qa();
        let summary = buffer.summary();
        assert!(summary.contains("persistent: 3"));
        assert!(summary.contains("Factual Recall"));
    }

    #[test]
    fn test_rubric_buffer_serialization_roundtrip() {
        let mut buffer = RubricBuffer::for_document_qa();
        buffer.add_adaptive(vec![
            RubricItem::adaptive("Brevity", "Keeps response concise"),
        ]);

        // Record some scores
        let mut scores = HashMap::new();
        scores.insert("Factual Recall".to_string(), 0.8);
        scores.insert("Brevity".to_string(), 0.6);
        buffer.record_scores(&scores);

        // Save
        let tmp_path = std::env::temp_dir().join("rubric_test_save.json");
        buffer.save(&tmp_path).expect("save failed");

        // Load
        let loaded = RubricBuffer::load(&tmp_path).expect("load failed");
        assert_eq!(loaded.persistent.len(), 3);
        assert_eq!(loaded.active.len(), 1);
        assert_eq!(loaded.active[0].title, "Brevity");

        // Score history should be preserved
        let m = loaded.metrics();
        assert!(m.total_observations > 0, "Score history should survive roundtrip");

        // Cleanup
        let _ = std::fs::remove_file(&tmp_path);
    }

    #[test]
    fn test_rubric_buffer_load_missing() {
        let result = RubricBuffer::load("/tmp/nonexistent_rubric_buffer_xyz.json");
        assert!(result.is_none());
    }

    #[test]
    fn test_rubric_buffer_load_or_default() {
        let buf = RubricBuffer::load_or_default("/tmp/nonexistent_rubric_buffer_xyz.json");
        assert_eq!(buf.persistent.len(), 3, "Should fall back to document QA defaults");
    }

    #[test]
    fn test_rubric_metrics() {
        let mut buffer = RubricBuffer::for_document_qa();

        // No observations yet
        let m = buffer.metrics();
        assert_eq!(m.total_observations, 0);

        // Record varied scores
        let mut scores = HashMap::new();
        scores.insert("Factual Recall".to_string(), 0.8);
        scores.insert("Answer Relevance".to_string(), 0.5);
        scores.insert("Completeness".to_string(), 0.0);
        buffer.record_scores(&scores);

        scores.insert("Factual Recall".to_string(), 0.2);
        scores.insert("Answer Relevance".to_string(), 0.5);
        scores.insert("Completeness".to_string(), 0.0);
        buffer.record_scores(&scores);

        let m = buffer.metrics();
        assert_eq!(m.persistent_count, 3);
        assert_eq!(m.active_count, 0);
        assert_eq!(m.total_observations, 6); // 3 rubrics × 2 observations each
        assert!(m.avg_mean > 0.0, "Some rubrics have non-zero scores");
        // Answer Relevance has std=0 (both 0.5), Completeness has std=0 (both 0.0)
        // Only Factual Recall has std > 0
        assert!(m.discriminative_ratio > 0.0, "At least one rubric is discriminative");
    }

    #[test]
    fn test_z_score_normalization() {
        let mut buffer = RubricBuffer::for_document_qa();

        // Build up score history (need 2+ observations for std)
        let mut s1 = HashMap::new();
        s1.insert("Factual Recall".to_string(), 0.8);
        s1.insert("Answer Relevance".to_string(), 0.5);
        s1.insert("Completeness".to_string(), 0.3);
        buffer.record_scores(&s1);

        let mut s2 = HashMap::new();
        s2.insert("Factual Recall".to_string(), 0.2);
        s2.insert("Answer Relevance".to_string(), 0.5);
        s2.insert("Completeness".to_string(), 0.7);
        buffer.record_scores(&s2);

        // Now z-score a new result that's above average on everything
        let mut new_scores = HashMap::new();
        new_scores.insert("Factual Recall".to_string(), 0.9);    // well above mean=0.5
        new_scores.insert("Answer Relevance".to_string(), 0.5);  // at mean (std=0, z=0)
        new_scores.insert("Completeness".to_string(), 0.8);      // above mean=0.5

        let z = buffer.z_score_normalize(&new_scores);
        // Should be positive (above average on non-zero-std rubrics)
        assert!(z > 0.0, "Above-average scores should give positive z-advantage, got {:.3}", z);

        // Test below average
        let mut low_scores = HashMap::new();
        low_scores.insert("Factual Recall".to_string(), 0.1);
        low_scores.insert("Answer Relevance".to_string(), 0.5);
        low_scores.insert("Completeness".to_string(), 0.2);
        let z_low = buffer.z_score_normalize(&low_scores);
        assert!(z_low < 0.0, "Below-average scores should give negative z-advantage, got {:.3}", z_low);
    }
}
