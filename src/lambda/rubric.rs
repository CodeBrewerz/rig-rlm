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
#[derive(Debug, Clone, Default)]
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
#[derive(Debug, Clone)]
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
}
