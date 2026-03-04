//! # GEPA Optimizer — Genetic-Pareto optimization for any measurable parameter.
//!
//! A Rust port of GEPA's `optimize_anything` algorithm. Evolves text parameters
//! (weights, configs, thresholds) using Pareto-efficient evolutionary search.
//!
//! ## Algorithm
//!
//! ```text
//! seed_candidate → evaluate → reflect → mutate → accept/reject → repeat
//!       ↑                                              |
//!       └──────────────────────────────────────────────┘
//! ```
//!
//! ## Components
//!
//! - [`Candidate`] — a set of named text parameters to optimize
//! - [`Evaluator`] — trait for scoring candidates
//! - [`Mutator`] — trait for proposing improved candidates (sync)
//! - [`LlmMutator`] — LLM-guided mutation via Trinity model
//! - [`ParetoFrontier`] — tracks non-dominated candidates
//! - [`optimize`] / [`optimize_async`] — main optimization loops

use std::collections::HashMap;
use std::fmt;

/// Default path for persisted GEPA weights.
pub const GEPA_WEIGHTS_PATH: &str = "gepa_weights.json";

// ═══════════════════════════════════════════════════════════════
// Candidate: the thing we're optimizing
// ═══════════════════════════════════════════════════════════════

/// A candidate is a set of named optimizable parameters (text or numeric).
#[derive(Debug, Clone)]
pub struct Candidate {
    /// Named parameters: "alpha" → "0.7", "risk_weight" → "0.5"
    pub params: HashMap<String, String>,
    /// Generation that produced this candidate (0 = seed).
    pub generation: usize,
    /// Parent candidate index (None for seed).
    pub parent: Option<usize>,
}

impl Candidate {
    /// Create a new seed candidate from key-value pairs.
    pub fn seed(params: Vec<(&str, &str)>) -> Self {
        Self {
            params: params
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
            generation: 0,
            parent: None,
        }
    }

    /// Parse a numeric parameter, returning default if missing or unparseable.
    pub fn get_f32(&self, key: &str, default: f32) -> f32 {
        self.params
            .get(key)
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(default)
    }

    /// Set a parameter.
    pub fn set(&mut self, key: &str, value: &str) {
        self.params.insert(key.into(), value.into());
    }

    /// Format as a human-readable text block (for LLM reflection).
    pub fn to_text(&self) -> String {
        let mut lines: Vec<String> = self
            .params
            .iter()
            .map(|(k, v)| format!("{}: {}", k, v))
            .collect();
        lines.sort();
        lines.join("\n")
    }
}

impl fmt::Display for Candidate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

// ═══════════════════════════════════════════════════════════════
// Side Information (ASI): diagnostic feedback from evaluation
// ═══════════════════════════════════════════════════════════════

/// Actionable Side Information — diagnostic feedback from evaluation.
/// The text-optimization analogue of a gradient: tells the mutator
/// *why* a candidate failed and *how* to fix it.
#[derive(Debug, Clone, Default)]
pub struct SideInfo {
    /// Multi-objective scores: "ranking_accuracy" → 0.85
    pub scores: HashMap<String, f64>,
    /// Diagnostic log lines.
    pub logs: Vec<String>,
}

impl SideInfo {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn log(&mut self, msg: impl Into<String>) {
        self.logs.push(msg.into());
    }

    pub fn score(&mut self, name: impl Into<String>, value: f64) {
        self.scores.insert(name.into(), value);
    }

    /// Primary score (first score, or 0.0 if no scores).
    pub fn primary_score(&self) -> f64 {
        self.scores.values().next().copied().unwrap_or(0.0)
    }
}

// ═══════════════════════════════════════════════════════════════
// Evaluation Result
// ═══════════════════════════════════════════════════════════════

/// Result of evaluating a candidate.
#[derive(Debug, Clone)]
pub struct EvalResult {
    /// Primary score (higher is better).
    pub score: f64,
    /// Side information for reflection.
    pub side_info: SideInfo,
}

// ═══════════════════════════════════════════════════════════════
// Evaluator Trait
// ═══════════════════════════════════════════════════════════════

/// Trait for evaluating candidates. Implement this for your optimization task.
pub trait Evaluator {
    /// Evaluate a candidate and return (score, side_info).
    /// Higher score is better.
    fn evaluate(&self, candidate: &Candidate) -> EvalResult;
}

// ═══════════════════════════════════════════════════════════════
// Mutator Trait (sync, for NumericMutator)
// ═══════════════════════════════════════════════════════════════

/// Trait for proposing improved candidates (synchronous).
pub trait Mutator {
    fn mutate(
        &self,
        parent: &Candidate,
        eval_result: &EvalResult,
        generation: usize,
        parent_idx: usize,
    ) -> Candidate;
}

/// Simple numeric perturbation mutator (no LLM required).
pub struct NumericMutator {
    pub step_size: f32,
    seed: u64,
}

impl NumericMutator {
    pub fn new(step_size: f32, seed: u64) -> Self {
        Self { step_size, seed }
    }

    fn pseudo_random(&self, i: usize) -> f32 {
        let x = self
            .seed
            .wrapping_mul(6364136223846793005u64)
            .wrapping_add(i as u64)
            .wrapping_mul(1442695040888963407u64);
        let frac = ((x >> 48) as f32) / 65535.0;
        frac * 2.0 - 1.0
    }
}

impl Mutator for NumericMutator {
    fn mutate(
        &self,
        parent: &Candidate,
        _eval_result: &EvalResult,
        generation: usize,
        parent_idx: usize,
    ) -> Candidate {
        let mut child = parent.clone();
        child.generation = generation;
        child.parent = Some(parent_idx);
        for (i, (key, value)) in parent.params.iter().enumerate() {
            if let Ok(v) = value.parse::<f32>() {
                let noise = self.pseudo_random(generation * 100 + i + parent_idx * 37);
                let delta = v * self.step_size * noise;
                let new_val = (v + delta).clamp(0.01, 1.0);
                child.set(key, &format!("{:.4}", new_val));
            }
        }
        child
    }
}

// ═══════════════════════════════════════════════════════════════
// LLM Mutator — uses Trinity via OpenRouter for GEPA reflection
// ═══════════════════════════════════════════════════════════════

/// LLM-guided mutator that calls Trinity (`arcee-ai/trinity-large-preview:free`)
/// via OpenRouter's OpenAI-compatible chat completions API.
///
/// Builds a GEPA-style reflection prompt:
/// 1. Shows current parameter values
/// 2. Shows evaluation scores + diagnostic logs (ASI)
/// 3. Asks the LLM to propose improved values
/// 4. Parses the response for numeric updates
pub struct LlmMutator {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
    /// Optimization objective (passed to reflection prompt).
    pub objective: String,
    fallback: NumericMutator,
}

impl LlmMutator {
    /// Create from environment variables (reads .env via dotenvy).
    ///
    /// Requires: `OPENAI_API_KEY`, optionally `OPENAI_BASE_URL` and `RIG_RLM_MODEL`.
    pub fn from_env(objective: &str) -> Result<Self, String> {
        dotenvy::dotenv().ok();
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| "OPENAI_API_KEY not set — needed for Trinity LLM calls".to_string())?;
        let base_url = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());
        let model = std::env::var("RIG_RLM_MODEL")
            .unwrap_or_else(|_| "arcee-ai/trinity-large-preview:free".to_string());

        Ok(Self {
            client: reqwest::Client::new(),
            base_url,
            api_key,
            model,
            objective: objective.to_string(),
            fallback: NumericMutator::new(0.15, 42),
        })
    }

    /// Build the GEPA reflection prompt.
    fn build_reflection_prompt(&self, parent: &Candidate, eval_result: &EvalResult) -> String {
        let mut p = String::new();
        p.push_str("You are an expert optimization assistant for a fiduciary finance system.\n\n");
        p.push_str(&format!("## Objective\n{}\n\n", self.objective));
        p.push_str("## Current Parameter Values\n```\n");
        p.push_str(&parent.to_text());
        p.push_str("\n```\n\n");
        p.push_str(&format!(
            "## Evaluation Score: {:.6} (higher is better)\n\n",
            eval_result.score
        ));

        if !eval_result.side_info.scores.is_empty() {
            p.push_str("## Individual Metrics\n");
            for (name, value) in &eval_result.side_info.scores {
                p.push_str(&format!("- {}: {:.6}\n", name, value));
            }
            p.push('\n');
        }
        if !eval_result.side_info.logs.is_empty() {
            p.push_str("## Diagnostic Feedback\n```\n");
            for log in &eval_result.side_info.logs {
                p.push_str(log);
                p.push('\n');
            }
            p.push_str("```\n\n");
        }

        p.push_str("## Your Task\n\n");
        p.push_str(
            "Analyze the evaluation results. Based on the metrics and diagnostic feedback:\n",
        );
        p.push_str("1. Identify which parameters should increase or decrease\n");
        p.push_str("2. Consider how the parameters interact with each other\n");
        p.push_str("3. Propose improved values that maximize the combined score\n\n");
        p.push_str(
            "Respond with ONLY the improved parameter values, one per line, in the exact format:\n",
        );
        p.push_str("```\nparameter_name: value\n```\n\n");
        p.push_str("Each value must be a decimal number between 0.01 and 1.00.\n");
        p.push_str("Include ALL parameters, even unchanged ones.\n");
        p.push_str("Do NOT include any explanation — only the parameter block.\n");
        p
    }

    /// Parse LLM response to extract parameter values.
    fn parse_response(&self, response: &str, parent: &Candidate) -> Candidate {
        let mut child = parent.clone();
        for line in response.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("```") || line.starts_with('#') {
                continue;
            }
            if let Some((key, value)) = line.split_once(':') {
                let key = key.trim();
                let value = value.trim();
                if parent.params.contains_key(key) {
                    if let Ok(v) = value.parse::<f32>() {
                        child.set(key, &format!("{:.4}", v.clamp(0.01, 1.0)));
                    }
                }
            }
        }
        child
    }

    /// Call the Trinity LLM via OpenRouter and return the response text.
    async fn call_llm(&self, prompt: &str) -> Result<String, String> {
        let url = format!("{}/chat/completions", self.base_url);
        let body = serde_json::json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 512
        });

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("HTTP error: {}", e))?;

        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| format!("Read error: {}", e))?;

        if !status.is_success() {
            return Err(format!(
                "API error {}: {}",
                status,
                &text[..200.min(text.len())]
            ));
        }

        let json: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("JSON parse error: {}", e))?;

        json["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| "No content in response".to_string())
    }

    /// Async mutation: call Trinity for reflection, parse response.
    /// Falls back to NumericMutator on error.
    pub async fn mutate_async(
        &self,
        parent: &Candidate,
        eval_result: &EvalResult,
        generation: usize,
        parent_idx: usize,
    ) -> Candidate {
        let prompt = self.build_reflection_prompt(parent, eval_result);
        match self.call_llm(&prompt).await {
            Ok(response) => {
                println!("  GEPA 🧠 │ Trinity responded ({} chars)", response.len());
                let mut child = self.parse_response(&response, parent);
                child.generation = generation;
                child.parent = Some(parent_idx);
                child
            }
            Err(e) => {
                println!("  GEPA ⚠  │ LLM failed: {} — using numeric fallback", e);
                self.fallback
                    .mutate(parent, eval_result, generation, parent_idx)
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Pareto Frontier
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct FrontierEntry {
    pub candidate: Candidate,
    pub eval: EvalResult,
    pub index: usize,
}

pub struct ParetoFrontier {
    entries: Vec<FrontierEntry>,
    max_size: usize,
}

fn rank_score(score: f64) -> f64 {
    if score.is_finite() {
        score
    } else {
        f64::NEG_INFINITY
    }
}

impl ParetoFrontier {
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_size,
        }
    }

    pub fn try_add(&mut self, candidate: Candidate, eval: EvalResult, index: usize) -> bool {
        let new_entry = FrontierEntry {
            candidate,
            eval,
            index,
        };
        let scores: Vec<(&String, &f64)> = new_entry.eval.side_info.scores.iter().collect();

        if scores.len() <= 1 {
            self.entries.push(new_entry);
            self.entries
                .sort_by(|a, b| rank_score(b.eval.score).total_cmp(&rank_score(a.eval.score)));
            if self.entries.len() > self.max_size {
                self.entries.truncate(self.max_size);
            }
            return self.entries.iter().any(|e| e.index == index);
        }

        self.entries.retain(|existing| {
            !Self::dominates(
                &new_entry.eval.side_info.scores,
                &existing.eval.side_info.scores,
            )
        });
        let is_dominated = self.entries.iter().any(|existing| {
            Self::dominates(
                &existing.eval.side_info.scores,
                &new_entry.eval.side_info.scores,
            )
        });

        if !is_dominated {
            self.entries.push(new_entry);
            if self.entries.len() > self.max_size {
                self.entries
                    .sort_by(|a, b| rank_score(b.eval.score).total_cmp(&rank_score(a.eval.score)));
                self.entries.truncate(self.max_size);
            }
            true
        } else {
            false
        }
    }

    fn dominates(a: &HashMap<String, f64>, b: &HashMap<String, f64>) -> bool {
        if a.is_empty() || b.is_empty() {
            return false;
        }
        let mut all_geq = true;
        let mut any_gt = false;
        for (key, a_val) in a {
            let a_val = rank_score(*a_val);
            let b_val = b.get(key).copied().unwrap_or(0.0);
            let b_val = rank_score(b_val);
            if a_val < b_val {
                all_geq = false;
                break;
            }
            if a_val > b_val {
                any_gt = true;
            }
        }
        all_geq && any_gt
    }

    pub fn select_parent(&self, iteration: usize) -> Option<&FrontierEntry> {
        if self.entries.is_empty() {
            return None;
        }
        Some(&self.entries[iteration % self.entries.len()])
    }

    pub fn best(&self) -> Option<&FrontierEntry> {
        self.entries
            .iter()
            .max_by(|a, b| rank_score(a.eval.score).total_cmp(&rank_score(b.eval.score)))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn entries(&self) -> &[FrontierEntry] {
        &self.entries
    }
}

// ═══════════════════════════════════════════════════════════════
// Optimization Config & Result
// ═══════════════════════════════════════════════════════════════

pub struct OptimizeConfig {
    pub max_evals: usize,
    pub max_frontier_size: usize,
    pub log_every: usize,
    pub objective: String,
}

impl Default for OptimizeConfig {
    fn default() -> Self {
        Self {
            max_evals: 200,
            max_frontier_size: 20,
            log_every: 10,
            objective: String::new(),
        }
    }
}

pub struct OptimizeResult {
    pub best_candidate: Candidate,
    pub best_score: f64,
    pub total_evals: usize,
    pub score_history: Vec<(usize, f64)>,
    pub frontier_size: usize,
}

impl OptimizeResult {
    /// Save the best weights to disk (JSON) for the feedback loop.
    pub fn save_best(&self, path: &str) -> Result<(), String> {
        let weights = OptimizedWeights::from_candidate(&self.best_candidate, self.best_score);
        weights.save(path)
    }
}

// ═══════════════════════════════════════════════════════════════
// Persisted Weights — the feedback loop
// ═══════════════════════════════════════════════════════════════

/// Persisted fiduciary weights optimized by GEPA.
///
/// ## Feedback Loop
/// ```text
/// Run 1: seed (defaults) → evaluate → mutate → best → save to gepa_weights.json
/// Run 2: load gepa_weights.json → seed → evaluate → mutate → better → save
/// Run N: each run starts from the best weights of the previous run
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OptimizedWeights {
    /// GNN blend weight (α): how much to weight GNN anomaly score.
    pub gnn_weight: f32,
    /// PC blend weight (β): how much to weight PC calibrated risk.
    pub pc_weight: f32,
    /// Fiduciary axes weights: [cost, risk, goal, urgency, conflict, reversibility].
    pub cost_weight: f32,
    pub risk_weight: f32,
    pub goal_weight: f32,
    pub urgency_weight: f32,
    pub conflict_weight: f32,
    pub reversibility_weight: f32,
    /// Score achieved by this configuration.
    pub score: f64,
    /// Timestamp of when these weights were optimized.
    pub optimized_at: String,
    /// Number of evaluations that produced these weights.
    pub total_evals: usize,
}

impl Default for OptimizedWeights {
    fn default() -> Self {
        Self {
            gnn_weight: 0.7,
            pc_weight: 0.3,
            cost_weight: 0.25,
            risk_weight: 0.25,
            goal_weight: 0.15,
            urgency_weight: 0.15,
            conflict_weight: 0.10,
            reversibility_weight: 0.10,
            score: 0.0,
            optimized_at: String::new(),
            total_evals: 0,
        }
    }
}

impl OptimizedWeights {
    fn sanitized(mut self) -> Self {
        // Blend weights must be finite, non-negative, and sum to 1.
        self.gnn_weight = if self.gnn_weight.is_finite() {
            self.gnn_weight.max(0.0)
        } else {
            0.7
        };
        self.pc_weight = if self.pc_weight.is_finite() {
            self.pc_weight.max(0.0)
        } else {
            0.3
        };
        let blend_sum = self.gnn_weight + self.pc_weight;
        if blend_sum <= 1e-8 {
            self.gnn_weight = 0.7;
            self.pc_weight = 0.3;
        } else {
            self.gnn_weight /= blend_sum;
            self.pc_weight /= blend_sum;
        }

        // Axes weights must be finite, positive, and normalized.
        let mut axes = [
            if self.cost_weight.is_finite() {
                self.cost_weight
            } else {
                0.25
            },
            if self.risk_weight.is_finite() {
                self.risk_weight
            } else {
                0.25
            },
            if self.goal_weight.is_finite() {
                self.goal_weight
            } else {
                0.15
            },
            if self.urgency_weight.is_finite() {
                self.urgency_weight
            } else {
                0.15
            },
            if self.conflict_weight.is_finite() {
                self.conflict_weight
            } else {
                0.10
            },
            if self.reversibility_weight.is_finite() {
                self.reversibility_weight
            } else {
                0.10
            },
        ];
        for w in &mut axes {
            *w = (*w).max(0.01);
        }
        let axis_sum: f32 = axes.iter().sum();
        if axis_sum <= 1e-8 {
            axes = [0.25, 0.25, 0.15, 0.15, 0.10, 0.10];
        } else {
            for w in &mut axes {
                *w /= axis_sum;
            }
        }
        self.cost_weight = axes[0];
        self.risk_weight = axes[1];
        self.goal_weight = axes[2];
        self.urgency_weight = axes[3];
        self.conflict_weight = axes[4];
        self.reversibility_weight = axes[5];
        self
    }

    /// Save to JSON file.
    pub fn save(&self, path: &str) -> Result<(), String> {
        let json =
            serde_json::to_string_pretty(self).map_err(|e| format!("Serialize error: {}", e))?;
        std::fs::write(path, json).map_err(|e| format!("Write error: {}", e))
    }

    /// Load from JSON file.
    pub fn load(path: &str) -> Result<Self, String> {
        let json = std::fs::read_to_string(path).map_err(|e| format!("Read error: {}", e))?;
        serde_json::from_str(&json).map_err(|e| format!("Parse error: {}", e))
    }

    /// Load from file, falling back to defaults if file doesn't exist.
    pub fn load_or_default(path: &str) -> Self {
        Self::load(path).unwrap_or_default().sanitized()
    }

    /// Convert to a GEPA Candidate for optimization.
    pub fn to_candidate(&self) -> Candidate {
        Candidate::seed(vec![
            ("gnn_weight", &format!("{:.4}", self.gnn_weight)),
            ("pc_weight", &format!("{:.4}", self.pc_weight)),
            ("cost_weight", &format!("{:.4}", self.cost_weight)),
            ("risk_weight", &format!("{:.4}", self.risk_weight)),
            ("goal_weight", &format!("{:.4}", self.goal_weight)),
            ("urgency_weight", &format!("{:.4}", self.urgency_weight)),
        ])
    }

    /// Extract from a GEPA Candidate (after optimization).
    pub fn from_candidate(candidate: &Candidate, score: f64) -> Self {
        Self {
            gnn_weight: candidate.get_f32("gnn_weight", 0.7),
            pc_weight: candidate.get_f32("pc_weight", 0.3),
            cost_weight: candidate.get_f32("cost_weight", 0.25),
            risk_weight: candidate.get_f32("risk_weight", 0.25),
            goal_weight: candidate.get_f32("goal_weight", 0.15),
            urgency_weight: candidate.get_f32("urgency_weight", 0.15),
            conflict_weight: 0.10,
            reversibility_weight: 0.10,
            score,
            optimized_at: chrono::Utc::now().to_rfc3339(),
            total_evals: 0,
        }
        .sanitized()
    }

    /// The 6-element weight array for `FiduciaryAxes::score_with_weights()`.
    pub fn axes_weights(&self) -> [f32; 6] {
        [
            self.cost_weight,
            self.risk_weight,
            self.goal_weight,
            self.urgency_weight,
            self.conflict_weight,
            self.reversibility_weight,
        ]
    }

    /// GNN/PC blend pair.
    pub fn blend_weights(&self) -> (f32, f32) {
        (self.gnn_weight, self.pc_weight)
    }
}

// ═══════════════════════════════════════════════════════════════
// Auto-Tune: production-ready self-improvement on each pipeline run
// ═══════════════════════════════════════════════════════════════

/// Report from an auto-tune pass.
#[derive(Debug, Clone)]
pub struct AutoTuneReport {
    /// Whether weights improved on this run.
    pub improved: bool,
    /// Score before tuning.
    pub score_before: f64,
    /// Score after tuning (may be same as before if no improvement).
    pub score_after: f64,
    /// Number of evaluations performed.
    pub evals: usize,
    /// Cumulative total evaluations across all runs.
    pub cumulative_evals: usize,
    /// Path where weights were saved.
    pub weights_path: String,
}

/// Run a lightweight GEPA auto-tune pass on fiduciary weights.
///
/// This is designed to be called at the end of each pipeline run with real
/// embeddings and anomaly scores. It:
/// 1. Loads previous best weights (or defaults)
/// 2. Runs `max_evals` quick evaluations with NumericMutator (no LLM needed)
/// 3. If improvement found, saves new weights to `weights_path`
///
/// Typical call: `auto_tune_weights(evaluator, weights_path, 5)`
/// Cost: ~50ms per eval (just scoring, no training), negligible overhead.
pub fn auto_tune_weights(
    evaluator: &dyn Evaluator,
    weights_path: &str,
    max_evals: usize,
) -> AutoTuneReport {
    let prev_weights = OptimizedWeights::load_or_default(weights_path);
    let seed = prev_weights.to_candidate();
    let seed_eval = evaluator.evaluate(&seed);
    let score_before = rank_score(seed_eval.score);

    let mutator = NumericMutator::new(0.10, prev_weights.total_evals as u64 + 7);
    let config = OptimizeConfig {
        max_evals,
        max_frontier_size: 5,
        log_every: 0, // Silent — production mode
        objective: "Auto-tune fiduciary weights".into(),
    };

    let result = optimize(seed, evaluator, &mutator, config);

    let improved = result.best_score > score_before;
    let cumulative = prev_weights.total_evals + result.total_evals;

    if improved {
        let mut best = OptimizedWeights::from_candidate(&result.best_candidate, result.best_score);
        best.total_evals = cumulative;
        if let Err(e) = best.save(weights_path) {
            eprintln!("  [gepa] auto-tune save error: {}", e);
        } else {
            eprintln!(
                "  [gepa] auto-tune: improved {:.6} → {:.6} (+{:.4}%), saved to {}",
                score_before,
                result.best_score,
                (result.best_score - score_before) / score_before.abs().max(0.001) * 100.0,
                weights_path
            );
        }
    } else {
        // Still update eval count even if no improvement
        let mut w = prev_weights.clone();
        w.total_evals = cumulative;
        let _ = w.save(weights_path);
        eprintln!(
            "  [gepa] auto-tune: no improvement (score={:.6}, {} cumulative evals)",
            score_before, cumulative
        );
    }

    AutoTuneReport {
        improved,
        score_before,
        score_after: result.best_score,
        evals: result.total_evals,
        cumulative_evals: cumulative,
        weights_path: weights_path.to_string(),
    }
}

// ═══════════════════════════════════════════════════════════════
// Sync Optimization Loop (NumericMutator)
// ═══════════════════════════════════════════════════════════════

pub fn optimize(
    seed: Candidate,
    evaluator: &dyn Evaluator,
    mutator: &dyn Mutator,
    config: OptimizeConfig,
) -> OptimizeResult {
    let mut frontier = ParetoFrontier::new(config.max_frontier_size);
    let mut eval_count = 0;
    let mut score_history = Vec::new();
    let mut best_score = f64::NEG_INFINITY;
    let mut best_candidate = seed.clone();

    let mut seed_eval = evaluator.evaluate(&seed);
    if !seed_eval.score.is_finite() {
        seed_eval
            .side_info
            .log("Seed evaluation produced non-finite score; sanitized to -inf");
        seed_eval.score = f64::NEG_INFINITY;
    }
    if seed_eval.score > best_score {
        best_score = seed_eval.score;
        best_candidate = seed.clone();
    }
    score_history.push((eval_count, seed_eval.score));
    frontier.try_add(seed, seed_eval, eval_count);
    eval_count += 1;

    if config.log_every > 0 {
        println!(
            "  GEPA  │ eval {:3} │ score {:.6} │ frontier {:2} │ SEED",
            eval_count,
            best_score,
            frontier.len()
        );
    }

    while eval_count < config.max_evals {
        let parent_entry = match frontier.select_parent(eval_count) {
            Some(e) => e.clone(),
            None => break,
        };
        let child = mutator.mutate(
            &parent_entry.candidate,
            &parent_entry.eval,
            eval_count,
            parent_entry.index,
        );
        let mut child_eval = evaluator.evaluate(&child);
        if !child_eval.score.is_finite() {
            child_eval
                .side_info
                .log("Child evaluation produced non-finite score; sanitized to -inf");
            child_eval.score = f64::NEG_INFINITY;
        }
        let child_score = child_eval.score;
        score_history.push((eval_count, child_score));
        if child_score > best_score {
            best_score = child_score;
            best_candidate = child.clone();
        }
        let accepted = frontier.try_add(child, child_eval, eval_count);
        eval_count += 1;
        if config.log_every > 0 && eval_count % config.log_every == 0 {
            println!(
                "  GEPA  │ eval {:3} │ score {:.6} │ best {:.6} │ frontier {:2} │ {}",
                eval_count,
                child_score,
                best_score,
                frontier.len(),
                if accepted {
                    "✅ accepted"
                } else {
                    "  rejected"
                }
            );
        }
    }

    OptimizeResult {
        best_candidate,
        best_score,
        total_evals: eval_count,
        score_history,
        frontier_size: frontier.len(),
    }
}

// ═══════════════════════════════════════════════════════════════
// Async Optimization Loop (LlmMutator via Trinity)
// ═══════════════════════════════════════════════════════════════

/// Run the GEPA optimization loop with LLM-guided mutations (async).
/// Uses Trinity model via OpenRouter for reflection-based mutation.
pub async fn optimize_async(
    seed: Candidate,
    evaluator: &dyn Evaluator,
    llm_mutator: &LlmMutator,
    config: OptimizeConfig,
) -> OptimizeResult {
    let mut frontier = ParetoFrontier::new(config.max_frontier_size);
    let mut eval_count = 0;
    let mut score_history = Vec::new();
    let mut best_score = f64::NEG_INFINITY;
    let mut best_candidate = seed.clone();

    let mut seed_eval = evaluator.evaluate(&seed);
    if !seed_eval.score.is_finite() {
        seed_eval
            .side_info
            .log("Seed evaluation produced non-finite score; sanitized to -inf");
        seed_eval.score = f64::NEG_INFINITY;
    }
    if seed_eval.score > best_score {
        best_score = seed_eval.score;
        best_candidate = seed.clone();
    }
    score_history.push((eval_count, seed_eval.score));
    frontier.try_add(seed, seed_eval, eval_count);
    eval_count += 1;

    if config.log_every > 0 {
        println!(
            "  GEPA 🧠│ eval {:3} │ score {:.6} │ frontier {:2} │ SEED",
            eval_count,
            best_score,
            frontier.len()
        );
    }

    while eval_count < config.max_evals {
        let parent_entry = match frontier.select_parent(eval_count) {
            Some(e) => e.clone(),
            None => break,
        };
        let child = llm_mutator
            .mutate_async(
                &parent_entry.candidate,
                &parent_entry.eval,
                eval_count,
                parent_entry.index,
            )
            .await;
        let mut child_eval = evaluator.evaluate(&child);
        if !child_eval.score.is_finite() {
            child_eval
                .side_info
                .log("Child evaluation produced non-finite score; sanitized to -inf");
            child_eval.score = f64::NEG_INFINITY;
        }
        let child_score = child_eval.score;
        score_history.push((eval_count, child_score));
        if child_score > best_score {
            best_score = child_score;
            best_candidate = child.clone();
        }
        let accepted = frontier.try_add(child, child_eval, eval_count);
        eval_count += 1;
        if config.log_every > 0 && eval_count % config.log_every == 0 {
            println!(
                "  GEPA 🧠│ eval {:3} │ score {:.6} │ best {:.6} │ frontier {:2} │ {}",
                eval_count,
                child_score,
                best_score,
                frontier.len(),
                if accepted {
                    "✅ accepted"
                } else {
                    "  rejected"
                }
            );
        }
    }

    OptimizeResult {
        best_candidate,
        best_score,
        total_evals: eval_count,
        score_history,
        frontier_size: frontier.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SumEvaluator;
    impl Evaluator for SumEvaluator {
        fn evaluate(&self, candidate: &Candidate) -> EvalResult {
            let sum: f64 = candidate
                .params
                .values()
                .filter_map(|v| v.parse::<f64>().ok())
                .sum();
            let mut si = SideInfo::new();
            si.score("sum", sum);
            si.log(format!("Sum of params: {:.4}", sum));
            EvalResult {
                score: sum,
                side_info: si,
            }
        }
    }

    #[test]
    fn test_pareto_frontier() {
        let mut frontier = ParetoFrontier::new(5);
        let c1 = Candidate::seed(vec![("a", "0.5")]);
        let e1 = EvalResult {
            score: 0.5,
            side_info: SideInfo::new(),
        };
        assert!(frontier.try_add(c1, e1, 0));
        assert_eq!(frontier.len(), 1);

        let c2 = Candidate::seed(vec![("a", "0.8")]);
        let e2 = EvalResult {
            score: 0.8,
            side_info: SideInfo::new(),
        };
        assert!(frontier.try_add(c2, e2, 1));
        assert!(frontier.best().unwrap().eval.score >= 0.8);
    }

    #[test]
    fn test_optimize_simple() {
        let seed = Candidate::seed(vec![("a", "0.3"), ("b", "0.4")]);
        let config = OptimizeConfig {
            max_evals: 50,
            max_frontier_size: 10,
            log_every: 0,
            objective: "Max sum".into(),
        };
        let result = optimize(seed, &SumEvaluator, &NumericMutator::new(0.2, 42), config);
        println!(
            "  best_score={:.4}, evals={}, frontier={}",
            result.best_score, result.total_evals, result.frontier_size
        );
        assert!(result.best_score >= 0.7, "Should at least match seed score");
    }

    #[test]
    fn test_optimized_weights_are_sanitized() {
        let candidate = Candidate::seed(vec![
            ("gnn_weight", "3.0"),
            ("pc_weight", "1.0"),
            ("cost_weight", "10.0"),
            ("risk_weight", "0.0"),
            ("goal_weight", "-5.0"),
            ("urgency_weight", "2.0"),
        ]);

        let w = OptimizedWeights::from_candidate(&candidate, 0.5);
        let (g, p) = w.blend_weights();
        assert!((g + p - 1.0).abs() < 1e-6, "blend must sum to 1");
        assert!(g >= 0.0 && p >= 0.0, "blend weights must be non-negative");

        let axes = w.axes_weights();
        let sum: f32 = axes.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "axes must sum to 1");
        assert!(
            axes.iter().all(|v| *v > 0.0),
            "axes weights must stay positive"
        );
    }

    #[test]
    fn test_llm_mutator_prompt_building() {
        let mutator = LlmMutator {
            client: reqwest::Client::new(),
            base_url: "https://openrouter.ai/api/v1".into(),
            api_key: "test-key".into(),
            model: "arcee-ai/trinity-large-preview:free".into(),
            objective: "Maximize ranking accuracy for fiduciary recommendations".into(),
            fallback: NumericMutator::new(0.15, 42),
        };
        let candidate = Candidate::seed(vec![("gnn_weight", "0.7000"), ("pc_weight", "0.3000")]);
        let eval = EvalResult {
            score: 0.4686,
            side_info: {
                let mut si = SideInfo::new();
                si.score("kendall_tau", 0.477);
                si.score("top3_anomaly", 0.922);
                si.log("α=0.700, β=0.300 → τ=0.4772, top3=0.9223".to_string());
                si
            },
        };
        let prompt = mutator.build_reflection_prompt(&candidate, &eval);
        assert!(
            prompt.contains("Objective"),
            "Prompt should include objective"
        );
        assert!(
            prompt.contains("gnn_weight: 0.7000"),
            "Prompt should include params"
        );
        assert!(
            prompt.contains("kendall_tau"),
            "Prompt should include metrics"
        );
        println!("  ✅ Reflection prompt ({} chars)", prompt.len());
    }

    #[test]
    fn test_llm_mutator_response_parsing() {
        let mutator = LlmMutator {
            client: reqwest::Client::new(),
            base_url: "test".into(),
            api_key: "test".into(),
            model: "test".into(),
            objective: "test".into(),
            fallback: NumericMutator::new(0.15, 42),
        };
        let parent = Candidate::seed(vec![
            ("gnn_weight", "0.7000"),
            ("pc_weight", "0.3000"),
            ("cost_weight", "0.2500"),
        ]);
        let response = "```\ngnn_weight: 0.85\npc_weight: 0.45\ncost_weight: 0.30\n```";
        let child = mutator.parse_response(response, &parent);
        assert_eq!(child.get_f32("gnn_weight", 0.0), 0.85);
        assert_eq!(child.get_f32("pc_weight", 0.0), 0.45);
        assert_eq!(child.get_f32("cost_weight", 0.0), 0.30);
        println!("  ✅ Parsed LLM response: {}", child);
    }
}
