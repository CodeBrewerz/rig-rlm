//! # Adaptive Yoneda — Self-Learning λ-RLM via GEPA Trajectory Evolution
//!
//! This module closes the loop between the Yoneda representable functor and
//! the GEPA evolutionary optimizer, creating a **self-improving** system:
//!
//! ```text
//!     ┌─────────── GEPA Evolution Loop ─────────────┐
//!     │                                               │
//!     │  Trajectory Store     Morphism Population      │
//!     │  ┌──────────┐       ┌──────────────┐          │
//!     │  │ q₁ → r₁  │       │ morphism_1   │          │
//!     │  │ q₂ → r₂  │  ──→  │ morphism_2   │  ──→ y(P)
//!     │  │ q₃ → r₃  │       │ morphism_3   │          │
//!     │  └──────────┘       └──────────────┘          │
//!     │       ↑                                  │    │
//!     │       └──────── score ← evaluate ←───────┘    │
//!     └───────────────────────────────────────────────┘
//! ```
//!
//! # Category Theory Framing
//!
//! - **Kan Extension**: The system extends its knowledge from observed
//!   (query, result) pairs to unseen queries via the Left Kan Extension
//!   `Lan_J F`: it extrapolates the best morphism for a new query from
//!   the nearest observed trajectories.
//!
//! - **Natural Transformation Evolution**: GEPA evolves the natural
//!   transformation `η: y(P) → F` where `F` is the "optimal response"
//!   functor. Each generation improves `η` by learning from past failures.
//!
//! - **Free Monad over Trajectories**: Each execution trajectory is a
//!   term in the free monad `Free TrajectoryF a`. The GEPA interpret
//!   function folds over past trajectories to compute the next action.

use std::sync::Arc;
use std::collections::HashMap;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use hehrgnn::optimizer::gepa::{
    Candidate, Evaluator, EvalResult, SideInfo, Mutator, NumericMutator,
};
use crate::monad::provider::LlmProvider;
use crate::lambda::{
    lambda_rlm, LambdaConfig,
    yoneda::{YonedaContext, QueryMorphism},
    combinators,
    rubric::{self, RubricBuffer, RubricItem},
};

// ═════════════════════════════════════════════════════════════════════════════
// Trajectory Store — Persistent Memory of Past Executions
// ═════════════════════════════════════════════════════════════════════════════

/// A single execution trajectory — the record of one probe through y(P).
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// The original query string.
    pub query: String,
    /// The morphism name applied (if any).
    pub morphism_name: Option<String>,
    /// The transformed query after morphism application.
    pub effective_query: String,
    /// The raw result from λ-RLM.
    pub result: String,
    /// Score (0.0–1.0) — how good was this result?
    pub score: f64,
    /// Execution latency in seconds.
    pub latency_secs: f64,
    /// Execution parameters used.
    pub k_star: usize,
    pub tau_star: usize,
    pub depth: usize,
    /// Timestamp (unix epoch seconds).
    pub timestamp: u64,
    /// Generation in the GEPA evolution (0 = initial).
    pub generation: usize,
}

/// Persistent store of execution trajectories.
///
/// In the category theory framing, this is the **diagram** over which
/// we compute the Kan Extension: the set of observed (query, result)
/// pairs that inform the extrapolation to unseen queries.
#[derive(Debug, Clone, Default)]
pub struct TrajectoryStore {
    trajectories: Vec<Trajectory>,
    /// Best score seen per query prefix (for fast lookup).
    best_scores: HashMap<String, f64>,
}

impl TrajectoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new trajectory.
    pub fn record(&mut self, trajectory: Trajectory) {
        let key = trajectory.query.chars().take(50).collect::<String>();
        let current_best = self.best_scores.get(&key).copied().unwrap_or(0.0);
        if trajectory.score > current_best {
            self.best_scores.insert(key, trajectory.score);
        }
        self.trajectories.push(trajectory);
    }

    /// Get the best morphism name for a given query type (by prefix match).
    pub fn best_morphism_for(&self, query: &str) -> Option<&str> {
        let prefix: String = query.chars().take(30).collect();
        self.trajectories
            .iter()
            .filter(|t| t.query.starts_with(&prefix) && t.morphism_name.is_some())
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
            .and_then(|t| t.morphism_name.as_deref())
    }

    /// Get all trajectories (for GEPA reflection).
    pub fn all(&self) -> &[Trajectory] {
        &self.trajectories
    }

    /// Get the N most recent trajectories.
    pub fn recent(&self, n: usize) -> Vec<&Trajectory> {
        self.trajectories.iter().rev().take(n).collect()
    }

    /// Mean score across all trajectories.
    pub fn mean_score(&self) -> f64 {
        if self.trajectories.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.trajectories.iter().map(|t| t.score).sum();
        sum / self.trajectories.len() as f64
    }

    /// Improvement rate: mean score of last N vs first N.
    pub fn improvement_rate(&self, window: usize) -> f64 {
        let n = self.trajectories.len();
        if n < window * 2 {
            return 0.0;
        }
        let early_mean: f64 = self.trajectories[..window].iter().map(|t| t.score).sum::<f64>() / window as f64;
        let late_mean: f64 = self.trajectories[n - window..].iter().map(|t| t.score).sum::<f64>() / window as f64;
        late_mean - early_mean
    }

    /// Serialize to JSON for persistence.
    pub fn to_json(&self) -> String {
        let entries: Vec<String> = self.trajectories.iter().map(|t| {
            format!(
                r#"{{"query":"{}","morphism":"{}","score":{},"latency":{},"k_star":{},"tau_star":{},"depth":{},"gen":{}}}"#,
                t.query.replace('"', "\\\"").chars().take(100).collect::<String>(),
                t.morphism_name.as_deref().unwrap_or("none"),
                t.score,
                t.latency_secs,
                t.k_star,
                t.tau_star,
                t.depth,
                t.generation,
            )
        }).collect();
        format!("[{}]", entries.join(",\n"))
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Morphism Population — Evolvable Query Transformations
// ═════════════════════════════════════════════════════════════════════════════

/// A named, scored morphism that GEPA can evolve.
pub struct ScoredMorphism {
    pub name: String,
    /// The prompt prefix/suffix template this morphism applies.
    pub prefix: String,
    pub suffix: String,
    /// Cumulative score from trajectories using this morphism.
    pub total_score: f64,
    pub use_count: usize,
}

impl ScoredMorphism {
    /// Build a QueryMorphism from this scored entry.
    pub fn to_query_morphism(&self) -> QueryMorphism {
        let prefix = self.prefix.clone();
        let suffix = self.suffix.clone();
        QueryMorphism::new(&self.name, move |q: &str| {
            format!("{}{}{}", prefix, q, suffix)
        })
    }

    /// Average score per use.
    pub fn avg_score(&self) -> f64 {
        if self.use_count == 0 { 0.0 } else { self.total_score / self.use_count as f64 }
    }
}

/// A population of evolvable query morphisms, managed by GEPA.
pub struct MorphismPopulation {
    pub morphisms: Vec<ScoredMorphism>,
}

impl MorphismPopulation {
    /// Create the initial population with standard task-type morphisms.
    pub fn seed() -> Self {
        Self {
            morphisms: vec![
                ScoredMorphism {
                    name: "identity".into(),
                    prefix: String::new(),
                    suffix: String::new(),
                    total_score: 0.0,
                    use_count: 0,
                },
                ScoredMorphism {
                    name: "be_concise".into(),
                    prefix: String::new(),
                    suffix: " Be concise and precise. Use bullet points.".into(),
                    total_score: 0.0,
                    use_count: 0,
                },
                ScoredMorphism {
                    name: "step_by_step".into(),
                    prefix: "Think step by step. ".into(),
                    suffix: String::new(),
                    total_score: 0.0,
                    use_count: 0,
                },
                ScoredMorphism {
                    name: "extract_key_facts".into(),
                    prefix: "Extract only the key facts relevant to: ".into(),
                    suffix: " List each fact on its own line.".into(),
                    total_score: 0.0,
                    use_count: 0,
                },
                ScoredMorphism {
                    name: "focus_and_cite".into(),
                    prefix: "Focus precisely on answering: ".into(),
                    suffix: " Cite specific passages from the text as evidence.".into(),
                    total_score: 0.0,
                    use_count: 0,
                },
            ],
        }
    }

    /// Select the best morphism based on past scores (Thompson sampling style).
    pub fn select_best(&self) -> &ScoredMorphism {
        self.morphisms
            .iter()
            .max_by(|a, b| a.avg_score().partial_cmp(&b.avg_score()).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(&self.morphisms[0])
    }

    /// Select with exploration: epsilon-greedy.
    pub fn select_epsilon_greedy(&self, generation: usize) -> &ScoredMorphism {
        // Epsilon decays: 50% exploration initially → 10% by generation 20
        let epsilon = (0.5 * (0.9_f64).powi(generation as i32)).max(0.10);
        let pseudo_random = ((generation as u64).wrapping_mul(2654435761) >> 16) as f64 / 65535.0;

        if pseudo_random < epsilon || self.morphisms.iter().all(|m| m.use_count == 0) {
            // Explore: pick a random morphism
            let idx = generation % self.morphisms.len();
            &self.morphisms[idx]
        } else {
            // Exploit: pick the best
            self.select_best()
        }
    }

    /// Record a score for a morphism (by name).
    pub fn record_score(&mut self, name: &str, score: f64) {
        if let Some(m) = self.morphisms.iter_mut().find(|m| m.name == name) {
            m.total_score += score;
            m.use_count += 1;
        }
    }

    /// Add a new morphism to the population (evolved by GEPA).
    pub fn add(&mut self, morphism: ScoredMorphism) {
        self.morphisms.push(morphism);
    }

    /// Summary of all morphisms and their scores.
    pub fn summary(&self) -> String {
        let mut lines: Vec<String> = self.morphisms.iter().map(|m| {
            format!("  {:25} avg={:.3} uses={}", m.name, m.avg_score(), m.use_count)
        }).collect();
        lines.sort();
        lines.join("\n")
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// GEPA Evaluator for Morphism+Parameter Co-Evolution
// ═════════════════════════════════════════════════════════════════════════════

/// GEPA Evaluator that co-evolves execution parameters (k*, τ*) AND
/// query morphisms. Each candidate encodes both the plan parameters
/// and the morphism index to use.
///
/// This is the **natural transformation evaluator**: it scores how well
/// `η_candidate: y(P) → F` approximates the optimal response functor `F`.
pub struct AdaptiveEvaluator {
    pub provider: Arc<LlmProvider>,
    pub config: LambdaConfig,
    pub document: String,
    pub test_queries: Vec<(String, String)>, // (query, expected_keyword)
    pub morphism_population: Arc<std::sync::Mutex<MorphismPopulation>>,
    pub trajectory_store: Arc<std::sync::Mutex<TrajectoryStore>>,
}

impl Evaluator for AdaptiveEvaluator {
    fn evaluate(&self, candidate: &Candidate) -> EvalResult {
        let k_star = candidate.get_f32("k_star", 2.0).round() as usize;
        let tau_star = candidate.get_f32("tau_star", 1500.0).round() as usize;
        let morphism_idx = candidate.get_f32("morphism_idx", 0.0).round() as usize;
        let generation = candidate.generation;

        // Get the morphism to use
        let population = self.morphism_population.lock().unwrap();
        let morphism_name;
        let query_morphism;
        if morphism_idx < population.morphisms.len() {
            morphism_name = population.morphisms[morphism_idx].name.clone();
            query_morphism = population.morphisms[morphism_idx].to_query_morphism();
        } else {
            morphism_name = "identity".to_string();
            query_morphism = QueryMorphism::identity();
        }
        drop(population);

        let mut side_info = SideInfo::new();
        let mut total_score = 0.0;
        let n_queries = self.test_queries.len().max(1);

        let rt = tokio::runtime::Runtime::new().unwrap();

        for (query, expected) in &self.test_queries {
            // Apply the morphism to the query
            let effective_query = query_morphism.apply(query);

            side_info.log(format!(
                "Query: {:?} → Morphism({}) → {:?}",
                &query[..query.len().min(40)],
                morphism_name,
                &effective_query[..effective_query.len().min(60)]
            ));

            let timer = Instant::now();
            let result = rt.block_on(lambda_rlm(
                &self.document,
                &effective_query,
                Arc::clone(&self.provider),
                self.config.clone(),
            ));
            let latency = timer.elapsed().as_secs_f64();

            match result {
                Ok(output) => {
                    let found = output.to_lowercase().contains(&expected.to_lowercase());
                    let accuracy = if found { 1.0 } else { 0.0 };
                    let efficiency = (30.0 / (latency + 1.0)).min(0.3);
                    let query_score = accuracy * 0.7 + efficiency;

                    total_score += query_score;

                    side_info.log(format!(
                        "  → accuracy={:.1}, latency={:.1}s, found_keyword={}, score={:.3}",
                        accuracy, latency, found, query_score
                    ));

                    // Record trajectory
                    let config = self.config.clone();
                    let mut store = self.trajectory_store.lock().unwrap();
                    store.record(Trajectory {
                        query: query.clone(),
                        morphism_name: Some(morphism_name.clone()),
                        effective_query: effective_query.clone(),
                        result: output.chars().take(200).collect(),
                        score: query_score,
                        latency_secs: latency,
                        k_star,
                        tau_star,
                        depth: 0, // filled by planner in real run
                        timestamp: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                        generation,
                    });
                }
                Err(e) => {
                    side_info.log(format!("  → ERROR: {}", e));
                }
            }
        }

        let avg_score = total_score / n_queries as f64;

        // Update morphism scores
        let mut population = self.morphism_population.lock().unwrap();
        population.record_score(&morphism_name, avg_score);
        drop(population);

        side_info.score("accuracy", avg_score);
        side_info.score("k_star", k_star as f64);
        side_info.score("tau_star", tau_star as f64);
        side_info.score("morphism", morphism_idx as f64);

        // Include trajectory history in side info for GEPA reflection
        let store = self.trajectory_store.lock().unwrap();
        let improvement = store.improvement_rate(3);
        side_info.log(format!(
            "Trajectory stats: {} total, mean={:.3}, improvement_rate={:.3}",
            store.all().len(),
            store.mean_score(),
            improvement,
        ));
        side_info.log(format!("Morphism population:\n{}", {
            let pop = self.morphism_population.lock().unwrap();
            pop.summary()
        }));

        EvalResult {
            score: avg_score,
            side_info,
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Adaptive Yoneda — The Self-Learning Wrapper
// ═════════════════════════════════════════════════════════════════════════════

/// A self-improving Yoneda context that learns from past probe trajectories
/// via GEPA evolutionary optimization.
///
/// # Category Theory
///
/// `AdaptiveYoneda` computes the **Left Kan Extension** of the observed
/// trajectory diagram along the inclusion functor. For unseen queries,
/// it extrapolates the best morphism from the nearest observed trajectories:
///
/// ```text
///     Observed ──J──→ AllQueries
///        │               │
///        F               Lan_J F
///        │               │
///        ↓               ↓
///       Set             Set
/// ```
///
/// Where `F(q) = best_trajectory_result(q)` on observed queries, and
/// `Lan_J F(q') ≈ F(nearest(q'))` for unseen queries.
pub struct AdaptiveYoneda {
    /// The underlying Yoneda representable functor.
    pub yoneda: YonedaContext,
    /// Trajectory memory — learns from every probe.
    pub trajectories: Arc<std::sync::Mutex<TrajectoryStore>>,
    /// Evolvable morphism population.
    pub morphisms: Arc<std::sync::Mutex<MorphismPopulation>>,
    /// Current generation in the GEPA evolution.
    pub generation: usize,
    /// Evolving rubric buffer — LLM-as-judge scoring criteria.
    pub rubric_buffer: Option<RubricBuffer>,
    /// How often to generate new adaptive rubrics (every N probes).
    pub rubric_gen_interval: usize,
}

impl AdaptiveYoneda {
    /// Create a new self-learning Yoneda context.
    pub fn new(document: impl Into<String>, provider: Arc<LlmProvider>, config: LambdaConfig) -> Self {
        Self {
            yoneda: YonedaContext::lift(document, provider, config),
            trajectories: Arc::new(std::sync::Mutex::new(TrajectoryStore::new())),
            morphisms: Arc::new(std::sync::Mutex::new(MorphismPopulation::seed())),
            generation: 0,
            rubric_buffer: None,
            rubric_gen_interval: 5,
        }
    }

    /// Create with the evolving rubric system enabled.
    ///
    /// Uses `RubricBuffer::for_document_qa()` as the default persistent rubric set,
    /// then evolves adaptive rubrics via LLM-as-judge scoring.
    pub fn with_rubrics(document: impl Into<String>, provider: Arc<LlmProvider>, config: LambdaConfig) -> Self {
        Self {
            yoneda: YonedaContext::lift(document, provider, config),
            trajectories: Arc::new(std::sync::Mutex::new(TrajectoryStore::new())),
            morphisms: Arc::new(std::sync::Mutex::new(MorphismPopulation::seed())),
            generation: 0,
            rubric_buffer: Some(RubricBuffer::for_document_qa()),
            rubric_gen_interval: 5,
        }
    }

    /// Create with a custom rubric buffer.
    pub fn with_custom_rubrics(
        document: impl Into<String>,
        provider: Arc<LlmProvider>,
        config: LambdaConfig,
        rubric_buffer: RubricBuffer,
    ) -> Self {
        Self {
            yoneda: YonedaContext::lift(document, provider, config),
            trajectories: Arc::new(std::sync::Mutex::new(TrajectoryStore::new())),
            morphisms: Arc::new(std::sync::Mutex::new(MorphismPopulation::seed())),
            generation: 0,
            rubric_buffer: Some(rubric_buffer),
            rubric_gen_interval: 5,
        }
    }

    /// Probe with **adaptive morphism selection**.
    ///
    /// The system automatically:
    /// 1. Selects the best morphism from the population (epsilon-greedy)
    /// 2. Applies it to the query
    /// 3. Probes via λ-RLM
    /// 4. Scores the result
    /// 5. Records the trajectory for future learning
    pub async fn adaptive_probe(
        &mut self,
        query: &str,
        scorer: impl Fn(&str, &str) -> f64,
    ) -> crate::monad::error::Result<(String, f64)> {
        // 1. Select morphism (exploration vs exploitation)
        let (morphism_name, morphism) = {
            let pop = self.morphisms.lock().unwrap();
            let selected = pop.select_epsilon_greedy(self.generation);
            (selected.name.clone(), selected.to_query_morphism())
        };

        // 2. Apply morphism to query
        let effective_query = morphism.apply(query);

        eprintln!(
            "🧬 [Adaptive] gen={} morphism={:?} query={:?}",
            self.generation,
            morphism_name,
            &effective_query[..effective_query.len().min(60)]
        );

        // 3. Probe via Yoneda / λ-RLM
        let timer = Instant::now();
        let result = self.yoneda.probe(&effective_query).await?;
        let latency = timer.elapsed().as_secs_f64();

        // 4. Score the result
        let score = scorer(query, &result);

        eprintln!(
            "🧬 [Adaptive] score={:.3} latency={:.1}s result_len={}",
            score, latency, result.len()
        );

        // 5. Record trajectory
        {
            let mut store = self.trajectories.lock().unwrap();
            store.record(Trajectory {
                query: query.to_string(),
                morphism_name: Some(morphism_name.clone()),
                effective_query,
                result: result.chars().take(200).collect(),
                score,
                latency_secs: latency,
                k_star: 0,
                tau_star: 0,
                depth: 0,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                generation: self.generation,
            });
        }

        // 6. Update morphism scores
        {
            let mut pop = self.morphisms.lock().unwrap();
            pop.record_score(&morphism_name, score);
        }

        self.generation += 1;

        Ok((result, score))
    }

    /// Probe with **evolving rubric scoring** (DR-Tulu inspired).
    ///
    /// Instead of a user-provided scorer callback, this uses:
    /// 1. LLM-as-Judge to score against all active rubrics
    /// 2. Periodic adaptive rubric generation (every `rubric_gen_interval` probes)
    /// 3. Automatic retirement of non-discriminative rubrics (std ≈ 0)
    ///
    /// Requires rubric_buffer to be initialized (use `with_rubrics()` constructor).
    ///
    /// Returns `(result_text, overall_score, per_rubric_scores)`.
    pub async fn adaptive_probe_with_rubrics(
        &mut self,
        query: &str,
    ) -> crate::monad::error::Result<(String, f64, HashMap<String, f64>)> {
        let buffer = self.rubric_buffer.as_mut().ok_or_else(|| {
            crate::monad::error::AgentError::Inference(
                "RubricBuffer not initialized. Use AdaptiveYoneda::with_rubrics()".into()
            )
        })?;

        // 1. Select morphism (epsilon-greedy)
        let (morphism_name, morphism) = {
            let pop = self.morphisms.lock().unwrap();
            let selected = pop.select_epsilon_greedy(self.generation);
            (selected.name.clone(), selected.to_query_morphism())
        };

        // 2. Apply morphism to query
        let effective_query = morphism.apply(query);

        eprintln!(
            "🧬 [AdaptiveRubric] gen={} morphism={:?} rubrics={} query={:?}",
            self.generation,
            morphism_name,
            buffer.all_active().len(),
            &effective_query[..effective_query.len().min(60)]
        );

        // 3. Probe via Yoneda / λ-RLM
        let timer = Instant::now();
        let result = self.yoneda.probe(&effective_query).await?;
        let latency = timer.elapsed().as_secs_f64();

        // 4. Score with LLM judge against ALL active rubrics
        let provider = self.yoneda.provider();
        let (overall_score, per_rubric_scores) = rubric::score_all_rubrics(
            &provider, query, &result, buffer,
        ).await;

        eprintln!(
            "🧬 [AdaptiveRubric] score={:.3} latency={:.1}s rubric_scores={:?}",
            overall_score, latency, per_rubric_scores
        );

        // 5. Record per-rubric scores for std tracking
        buffer.record_scores(&per_rubric_scores);

        // 6. Record trajectory
        {
            let mut store = self.trajectories.lock().unwrap();
            store.record(Trajectory {
                query: query.to_string(),
                morphism_name: Some(morphism_name.clone()),
                effective_query,
                result: result.chars().take(500).collect(),
                score: overall_score,
                latency_secs: latency,
                k_star: 0,
                tau_star: 0,
                depth: 0,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                generation: self.generation,
            });
        }

        // 7. Update morphism scores
        {
            let mut pop = self.morphisms.lock().unwrap();
            pop.record_score(&morphism_name, overall_score);
        }

        // 8. Periodically generate new adaptive rubrics
        if self.generation > 0
            && self.generation % self.rubric_gen_interval == 0
        {
            let recent_results: Vec<String> = {
                let store = self.trajectories.lock().unwrap();
                store.recent(self.rubric_gen_interval)
                    .iter()
                    .map(|t| t.result.clone())
                    .collect()
            };

            if recent_results.len() >= 2 {
                eprintln!(
                    "🧬 [AdaptiveRubric] Generating new rubrics from {} recent responses...",
                    recent_results.len()
                );

                // Collect existing rubrics for the generation call
                let existing_refs: Vec<&RubricItem> = buffer.persistent.iter()
                    .chain(buffer.active.iter())
                    .collect();

                let new_rubrics = rubric::generate_adaptive_rubrics(
                    &provider, query, &recent_results, &existing_refs,
                ).await;

                if !new_rubrics.is_empty() {
                    buffer.add_adaptive(new_rubrics);
                }
            }
        }

        // 9. Filter and retire non-discriminative rubrics
        if self.generation > 0 && self.generation % 3 == 0 {
            let report = buffer.filter_and_retire();
            if report.retired_zero_std > 0 || report.retired_cap > 0 {
                eprintln!("🧬 [AdaptiveRubric] {}", report);
            }
        }

        self.generation += 1;

        Ok((result, overall_score, per_rubric_scores))
    }

    /// Get the rubric buffer summary (for diagnostics).
    pub fn rubric_summary(&self) -> String {
        match &self.rubric_buffer {
            Some(buf) => format!("📊 Rubric Buffer:\n{}", buf.summary()),
            None => "📊 Rubric Buffer: not initialized".to_string(),
        }
    }

    /// Run a full GEPA evolution loop over the document.
    ///
    /// This is the **main self-learning entry point**. It:
    /// 1. Seeds a GEPA population with (k*, τ*, morphism_idx) candidates
    /// 2. Evaluates each candidate by probing with the selected morphism
    /// 3. Evolves the population via mutation + selection
    /// 4. Records all trajectories for persistent learning
    /// 5. Returns the best-performing configuration
    pub fn evolve(
        &self,
        provider: Arc<LlmProvider>,
        config: LambdaConfig,
        test_queries: Vec<(String, String)>,
        generations: usize,
    ) -> EvolutionResult {
        let document = self.yoneda.document().to_string();
        let n_morphisms = self.morphisms.lock().unwrap().morphisms.len();

        let evaluator = AdaptiveEvaluator {
            provider,
            config,
            document,
            test_queries,
            morphism_population: Arc::clone(&self.morphisms),
            trajectory_store: Arc::clone(&self.trajectories),
        };

        let mutator = NumericMutator::new(0.2, 42);

        // Seed candidate
        let mut best = Candidate::seed(vec![
            ("k_star", "2"),
            ("tau_star", "1500"),
            ("morphism_idx", "0"),
        ]);
        let mut best_result = evaluator.evaluate(&best);

        eprintln!(
            "🧬 [GEPA] Seed: score={:.3} k*={} τ*={} morphism=0",
            best_result.score,
            best.get_f32("k_star", 2.0),
            best.get_f32("tau_star", 1500.0),
        );

        for g_idx in 1..=generations {
            // Mutate
            let mut child = mutator.mutate(&best, &best_result, g_idx, 0);

            // Also mutate morphism index
            let new_morph_idx = (g_idx % n_morphisms) as f32;
            child.set("morphism_idx", &format!("{:.0}", new_morph_idx));

            let child_result = evaluator.evaluate(&child);

            eprintln!(
                "🧬 [GEPA] Gen {}: score={:.3} (best={:.3}) k*={} τ*={} morphism={}",
                g_idx,
                child_result.score,
                best_result.score,
                child.get_f32("k_star", 2.0),
                child.get_f32("tau_star", 1500.0),
                child.get_f32("morphism_idx", 0.0),
            );

            if child_result.score >= best_result.score {
                best = child;
                best_result = child_result;
            }
        }

        let store = self.trajectories.lock().unwrap();
        let pop = self.morphisms.lock().unwrap();

        EvolutionResult {
            best_score: best_result.score,
            best_k_star: best.get_f32("k_star", 2.0).round() as usize,
            best_tau_star: best.get_f32("tau_star", 1500.0).round() as usize,
            best_morphism: pop.morphisms.get(best.get_f32("morphism_idx", 0.0).round() as usize)
                .map(|m| m.name.clone())
                .unwrap_or_else(|| "identity".into()),
            total_trajectories: store.all().len(),
            improvement_rate: store.improvement_rate(3),
            morphism_summary: pop.summary(),
        }
    }
}

/// Result of a GEPA evolution run.
#[derive(Debug)]
pub struct EvolutionResult {
    pub best_score: f64,
    pub best_k_star: usize,
    pub best_tau_star: usize,
    pub best_morphism: String,
    pub total_trajectories: usize,
    pub improvement_rate: f64,
    pub morphism_summary: String,
}

impl std::fmt::Display for EvolutionResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "🧬 Evolution Result:\n  best_score: {:.3}\n  best_k*: {}\n  best_τ*: {}\n  best_morphism: {}\n  trajectories: {}\n  improvement: {:.3}\n\nMorphism Scores:\n{}",
            self.best_score,
            self.best_k_star,
            self.best_tau_star,
            self.best_morphism,
            self.total_trajectories,
            self.improvement_rate,
            self.morphism_summary,
        )
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trajectory_store_basics() {
        let mut store = TrajectoryStore::new();
        assert_eq!(store.mean_score(), 0.0);

        store.record(Trajectory {
            query: "test query".into(),
            morphism_name: Some("identity".into()),
            effective_query: "test query".into(),
            result: "some result".into(),
            score: 0.8,
            latency_secs: 1.5,
            k_star: 2,
            tau_star: 1500,
            depth: 1,
            timestamp: 0,
            generation: 0,
        });

        assert!((store.mean_score() - 0.8).abs() < f64::EPSILON);
        assert_eq!(store.all().len(), 1);
        assert_eq!(store.best_morphism_for("test query"), Some("identity"));
    }

    #[test]
    fn test_trajectory_improvement_rate() {
        let mut store = TrajectoryStore::new();

        // Add 6 trajectories with improving scores
        for i in 0..6 {
            store.record(Trajectory {
                query: format!("q{}", i),
                morphism_name: None,
                effective_query: format!("q{}", i),
                result: "r".into(),
                score: 0.5 + (i as f64) * 0.08, // 0.5, 0.58, 0.66, 0.74, 0.82, 0.90
                latency_secs: 1.0,
                k_star: 2,
                tau_star: 1500,
                depth: 1,
                timestamp: 0,
                generation: i,
            });
        }

        let rate = store.improvement_rate(3);
        assert!(rate > 0.0, "Should show improvement: got {}", rate);
    }

    #[test]
    fn test_morphism_population_seed() {
        let pop = MorphismPopulation::seed();
        assert!(pop.morphisms.len() >= 4, "Should have at least 4 seed morphisms");

        // Identity should be first
        assert_eq!(pop.morphisms[0].name, "identity");
    }

    #[test]
    fn test_morphism_selection_epsilon_greedy() {
        let mut pop = MorphismPopulation::seed();

        // Initially all scores are 0, so should explore
        let _selected = pop.select_epsilon_greedy(0);

        // Record some scores to make one morphism dominate
        pop.record_score("extract_key_facts", 0.9);
        pop.record_score("extract_key_facts", 0.85);
        pop.record_score("identity", 0.3);

        // After enough exploration, should prefer extract_key_facts
        let best = pop.select_best();
        assert_eq!(best.name, "extract_key_facts");
    }

    #[test]
    fn test_trajectory_serialization() {
        let mut store = TrajectoryStore::new();
        store.record(Trajectory {
            query: "what is this about?".into(),
            morphism_name: Some("be_concise".into()),
            effective_query: "what is this about? Be concise.".into(),
            result: "France.".into(),
            score: 0.9,
            latency_secs: 2.1,
            k_star: 2,
            tau_star: 1500,
            depth: 1,
            timestamp: 1000000,
            generation: 3,
        });

        let json = store.to_json();
        assert!(json.contains("what is this about?"));
        assert!(json.contains("be_concise"));
        assert!(json.contains("0.9"));
    }
}
