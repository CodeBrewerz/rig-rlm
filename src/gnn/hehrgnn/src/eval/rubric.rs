//! RLER: Reinforcement Learning with Evolving Rubrics (paper 2511.19399).
//!
//! Rubrics co-evolve with the policy during training:
//! - RubricCriterion: single evaluation criterion with weight
//! - Rubric: versioned collection of criteria
//! - RubricJudge: scores rollouts against the current rubric
//! - RubricEvolver: prunes saturated criteria, discovers new ones

use std::collections::HashMap;

// ──────────────────────────────────────────────────────
// Rubric criteria
// ──────────────────────────────────────────────────────

/// A single evaluation criterion in a rubric.
#[derive(Debug, Clone)]
pub struct RubricCriterion {
    /// Unique identifier.
    pub id: String,
    /// Human-readable description.
    pub description: String,
    /// Relative weight in [0, 1].
    pub weight: f64,
    /// How many times this criterion was satisfied (for evolution tracking).
    pub pass_count: usize,
    /// How many times this criterion was evaluated.
    pub eval_count: usize,
    /// Scoring function type.
    pub scorer: CriterionScorer,
}

/// How a criterion is scored.
#[derive(Debug, Clone)]
pub enum CriterionScorer {
    /// Threshold-based: score 1.0 if metric ≥ threshold.
    Threshold { metric_key: String, threshold: f64 },
    /// Range-based: linear interpolation in [min, max] → [0, 1].
    Range {
        metric_key: String,
        min: f64,
        max: f64,
    },
    /// Delta-based: score 1.0 if metric improved vs baseline.
    Delta { metric_key: String, baseline: f64 },
    /// Boolean: score 1.0 if condition key is present and true.
    BooleanPresence { key: String },
    /// Custom: external scoring function index.
    Custom { fn_id: usize },
}

impl RubricCriterion {
    pub fn threshold(id: &str, desc: &str, weight: f64, key: &str, threshold: f64) -> Self {
        Self {
            id: id.to_string(),
            description: desc.to_string(),
            weight,
            pass_count: 0,
            eval_count: 0,
            scorer: CriterionScorer::Threshold {
                metric_key: key.to_string(),
                threshold,
            },
        }
    }

    pub fn range(id: &str, desc: &str, weight: f64, key: &str, min: f64, max: f64) -> Self {
        Self {
            id: id.to_string(),
            description: desc.to_string(),
            weight,
            pass_count: 0,
            eval_count: 0,
            scorer: CriterionScorer::Range {
                metric_key: key.to_string(),
                min,
                max,
            },
        }
    }

    pub fn delta(id: &str, desc: &str, weight: f64, key: &str, baseline: f64) -> Self {
        Self {
            id: id.to_string(),
            description: desc.to_string(),
            weight,
            pass_count: 0,
            eval_count: 0,
            scorer: CriterionScorer::Delta {
                metric_key: key.to_string(),
                baseline,
            },
        }
    }

    pub fn boolean(id: &str, desc: &str, weight: f64, key: &str) -> Self {
        Self {
            id: id.to_string(),
            description: desc.to_string(),
            weight,
            pass_count: 0,
            eval_count: 0,
            scorer: CriterionScorer::BooleanPresence {
                key: key.to_string(),
            },
        }
    }

    /// Score this criterion against a metrics map. Returns [0, 1].
    pub fn score(&mut self, metrics: &HashMap<String, f64>) -> f64 {
        self.eval_count += 1;
        let s = match &self.scorer {
            CriterionScorer::Threshold {
                metric_key,
                threshold,
            } => metrics
                .get(metric_key)
                .map_or(0.0, |&v| if v >= *threshold { 1.0 } else { 0.0 }),
            CriterionScorer::Range {
                metric_key,
                min,
                max,
            } => metrics
                .get(metric_key)
                .map_or(0.0, |&v| ((v - min) / (max - min)).clamp(0.0, 1.0)),
            CriterionScorer::Delta {
                metric_key,
                baseline,
            } => metrics
                .get(metric_key)
                .map_or(0.0, |&v| if v > *baseline { 1.0 } else { 0.0 }),
            CriterionScorer::BooleanPresence { key } => {
                metrics
                    .get(key)
                    .map_or(0.0, |&v| if v > 0.5 { 1.0 } else { 0.0 })
            }
            CriterionScorer::Custom { .. } => 0.5, // Default for custom
        };
        if s >= 0.5 {
            self.pass_count += 1;
        }
        s
    }

    /// Pass rate for evolution decisions.
    pub fn pass_rate(&self) -> f64 {
        if self.eval_count == 0 {
            0.5
        } else {
            self.pass_count as f64 / self.eval_count as f64
        }
    }

    /// Whether this criterion is saturated (non-discriminative).
    /// A criterion is saturated if nearly all or nearly no rollouts pass it.
    pub fn is_saturated(&self, threshold: f64) -> bool {
        let rate = self.pass_rate();
        rate > (1.0 - threshold) || rate < threshold
    }
}

// ──────────────────────────────────────────────────────
// Rubric (versioned collection)
// ──────────────────────────────────────────────────────

/// A versioned rubric: scored collection of criteria that evolves.
#[derive(Debug, Clone)]
pub struct Rubric {
    pub criteria: Vec<RubricCriterion>,
    pub version: u32,
    pub created_at_episode: usize,
}

impl Rubric {
    pub fn new(criteria: Vec<RubricCriterion>) -> Self {
        Self {
            criteria,
            version: 1,
            created_at_episode: 0,
        }
    }

    /// Score a rollout against all criteria. Returns weighted [0, 1].
    pub fn score(&mut self, metrics: &HashMap<String, f64>) -> f64 {
        let mut total_score = 0.0;
        let mut total_weight = 0.0;
        for criterion in &mut self.criteria {
            let s = criterion.score(metrics);
            total_score += s * criterion.weight;
            total_weight += criterion.weight;
        }
        if total_weight < 1e-8 {
            0.0
        } else {
            total_score / total_weight
        }
    }

    /// Number of criteria.
    pub fn num_criteria(&self) -> usize {
        self.criteria.len()
    }

    /// Default rubric for fiduciary tasks.
    pub fn fiduciary_default() -> Self {
        Self::new(vec![
            RubricCriterion::threshold(
                "positive_return",
                "Portfolio achieves positive return",
                0.3,
                "portfolio_return",
                0.0,
            ),
            RubricCriterion::range(
                "sharpe_ratio",
                "Risk-adjusted return (Sharpe ratio)",
                0.25,
                "sharpe_ratio",
                0.0,
                2.0,
            ),
            RubricCriterion::threshold(
                "drawdown_limit",
                "Drawdown stays within 20%",
                0.2,
                "max_drawdown_ok",
                0.5,
            ),
            RubricCriterion::boolean(
                "fraud_detected",
                "Correctly identified fraudulent transactions",
                0.15,
                "fraud_detection_rate",
            ),
            RubricCriterion::boolean(
                "doc_coverage",
                "Requested relevant documents for decisions",
                0.1,
                "document_coverage",
            ),
        ])
    }

    /// Default rubric for agent/coding tasks.
    pub fn agent_default() -> Self {
        Self::new(vec![
            RubricCriterion::threshold(
                "compile_success",
                "Code compiles without errors",
                0.25,
                "compile_success",
                0.5,
            ),
            RubricCriterion::threshold("tests_pass", "All tests pass", 0.25, "test_pass_rate", 0.9),
            RubricCriterion::delta(
                "metric_improvement",
                "Key metric improved vs baseline",
                0.3,
                "primary_metric",
                0.0,
            ),
            RubricCriterion::range(
                "action_efficiency",
                "Achieved goal in few actions",
                0.1,
                "action_efficiency",
                0.0,
                1.0,
            ),
            RubricCriterion::boolean(
                "no_regressions",
                "No test regressions introduced",
                0.1,
                "no_regressions",
            ),
        ])
    }
}

// ──────────────────────────────────────────────────────
// Rubric judge
// ──────────────────────────────────────────────────────

/// Scores rollouts against the current rubric.
#[derive(Debug, Clone)]
pub struct RubricJudge {
    rubric: Rubric,
    /// History of scores for evolution analysis.
    score_history: Vec<(f64, HashMap<String, f64>)>,
}

impl RubricJudge {
    pub fn new(rubric: Rubric) -> Self {
        Self {
            rubric,
            score_history: Vec::new(),
        }
    }

    /// Score a rollout. Returns [0, 1].
    pub fn score(&mut self, metrics: &HashMap<String, f64>) -> f64 {
        let s = self.rubric.score(metrics);
        self.score_history.push((s, metrics.clone()));
        s
    }

    /// Convert rubric score to reward signal.
    pub fn score_to_reward(&mut self, metrics: &HashMap<String, f64>, scale: f64) -> f64 {
        let s = self.score(metrics);
        // Center around 0: score 0.5 → reward 0, score 1.0 → +scale, score 0.0 → -scale
        (s - 0.5) * 2.0 * scale
    }

    pub fn rubric_version(&self) -> u32 {
        self.rubric.version
    }

    pub fn num_scored(&self) -> usize {
        self.score_history.len()
    }

    /// Replace the rubric with an evolved version.
    pub fn update_rubric(&mut self, new_rubric: Rubric) {
        self.rubric = new_rubric;
        self.score_history.clear();
    }

    pub fn rubric(&self) -> &Rubric {
        &self.rubric
    }

    /// Get top-K and bottom-K rollouts for evolution.
    pub fn top_bottom_k(
        &self,
        k: usize,
    ) -> (Vec<&HashMap<String, f64>>, Vec<&HashMap<String, f64>>) {
        let mut indexed: Vec<(usize, f64)> = self
            .score_history
            .iter()
            .enumerate()
            .map(|(i, (s, _))| (i, *s))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top: Vec<&HashMap<String, f64>> = indexed
            .iter()
            .take(k)
            .map(|(i, _)| &self.score_history[*i].1)
            .collect();
        let bottom: Vec<&HashMap<String, f64>> = indexed
            .iter()
            .rev()
            .take(k)
            .map(|(i, _)| &self.score_history[*i].1)
            .collect();

        (top, bottom)
    }
}

// ──────────────────────────────────────────────────────
// Rubric evolver
// ──────────────────────────────────────────────────────

/// Evolves rubrics based on policy rollouts (core RLER mechanism).
#[derive(Debug, Clone)]
pub struct RubricEvolver {
    /// How often to evolve (every N episodes).
    pub evolution_interval: usize,
    /// Saturation threshold for pruning.
    pub saturation_threshold: f64,
    /// Minimum number of rollouts before evolving.
    pub min_rollouts: usize,
    /// Episodes since last evolution.
    episodes_since_evolution: usize,
}

impl RubricEvolver {
    pub fn new(interval: usize, saturation: f64, min_rollouts: usize) -> Self {
        Self {
            evolution_interval: interval,
            saturation_threshold: saturation,
            min_rollouts,
            episodes_since_evolution: 0,
        }
    }

    pub fn default_config() -> Self {
        Self::new(50, 0.05, 20)
    }

    /// Should we evolve now?
    pub fn should_evolve(&self, num_rollouts: usize) -> bool {
        self.episodes_since_evolution >= self.evolution_interval
            && num_rollouts >= self.min_rollouts
    }

    /// Record an episode was completed.
    pub fn record_episode(&mut self) {
        self.episodes_since_evolution += 1;
    }

    /// Evolve the rubric based on judge's history.
    ///
    /// 1. Prune saturated criteria (non-discriminative).
    /// 2. Discover new criteria from top/bottom rollout differences.
    /// 3. Reweight remaining criteria.
    pub fn evolve(&mut self, judge: &RubricJudge) -> Rubric {
        let old = judge.rubric();
        let mut new_criteria: Vec<RubricCriterion> = Vec::new();

        // Step 1: Keep non-saturated criteria
        for c in &old.criteria {
            if !c.is_saturated(self.saturation_threshold) {
                let mut kept = c.clone();
                kept.pass_count = 0;
                kept.eval_count = 0;
                new_criteria.push(kept);
            }
        }

        // Step 2: Discover new criteria from top/bottom differences
        if judge.num_scored() >= self.min_rollouts {
            let (top, bottom) = judge.top_bottom_k(self.min_rollouts / 4);
            let discovered = self.discover_criteria(&top, &bottom, old);
            new_criteria.extend(discovered);
        }

        // Step 3: Normalize weights
        let total_weight: f64 = new_criteria.iter().map(|c| c.weight).sum();
        if total_weight > 0.0 {
            for c in &mut new_criteria {
                c.weight /= total_weight;
            }
        }

        // Ensure at least one criterion survives
        if new_criteria.is_empty() {
            new_criteria = old.criteria.clone();
            for c in &mut new_criteria {
                c.pass_count = 0;
                c.eval_count = 0;
            }
        }

        self.episodes_since_evolution = 0;

        Rubric {
            criteria: new_criteria,
            version: old.version + 1,
            created_at_episode: 0, // Caller should set this
        }
    }

    /// Discover new criteria by comparing top vs bottom rollouts.
    fn discover_criteria(
        &self,
        top: &[&HashMap<String, f64>],
        bottom: &[&HashMap<String, f64>],
        existing: &Rubric,
    ) -> Vec<RubricCriterion> {
        let mut discovered = Vec::new();
        if top.is_empty() || bottom.is_empty() {
            return discovered;
        }

        // Collect all metric keys seen
        let mut all_keys: Vec<String> = Vec::new();
        for m in top.iter().chain(bottom.iter()) {
            for k in m.keys() {
                if !all_keys.contains(k) {
                    all_keys.push(k.clone());
                }
            }
        }

        // Existing criterion metric keys
        let existing_keys: Vec<String> = existing
            .criteria
            .iter()
            .filter_map(|c| match &c.scorer {
                CriterionScorer::Threshold { metric_key, .. } => Some(metric_key.clone()),
                CriterionScorer::Range { metric_key, .. } => Some(metric_key.clone()),
                CriterionScorer::Delta { metric_key, .. } => Some(metric_key.clone()),
                CriterionScorer::BooleanPresence { key } => Some(key.clone()),
                CriterionScorer::Custom { .. } => None,
            })
            .collect();

        // For each metric not already tracked, check if it discriminates
        for key in &all_keys {
            if existing_keys.contains(key) {
                continue;
            }

            let top_mean: f64 = top.iter().filter_map(|m| m.get(key)).copied().sum::<f64>()
                / top.len().max(1) as f64;

            let bottom_mean: f64 = bottom
                .iter()
                .filter_map(|m| m.get(key))
                .copied()
                .sum::<f64>()
                / bottom.len().max(1) as f64;

            let gap = (top_mean - bottom_mean).abs();

            // If this metric clearly separates top from bottom, add it
            if gap > 0.1 {
                let threshold = (top_mean + bottom_mean) / 2.0;
                discovered.push(RubricCriterion::threshold(
                    &format!("discovered_{}", key),
                    &format!("Auto-discovered: {} > {:.2}", key, threshold),
                    gap.min(1.0),
                    key,
                    threshold,
                ));
            }
        }

        discovered
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_criterion_scoring() {
        let mut c = RubricCriterion::threshold("test", "Test criterion", 1.0, "accuracy", 0.8);

        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.9);
        assert_eq!(c.score(&metrics), 1.0);

        metrics.insert("accuracy".to_string(), 0.5);
        assert_eq!(c.score(&metrics), 0.0);

        assert_eq!(c.eval_count, 2);
        assert_eq!(c.pass_count, 1);
        assert!((c.pass_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_range_scorer() {
        let mut c = RubricCriterion::range("sharpe", "Sharpe ratio", 1.0, "sharpe", 0.0, 2.0);

        let mut m = HashMap::new();
        m.insert("sharpe".to_string(), 1.0);
        assert!((c.score(&m) - 0.5).abs() < 0.01);

        m.insert("sharpe".to_string(), 2.0);
        assert!((c.score(&m) - 1.0).abs() < 0.01);

        m.insert("sharpe".to_string(), -0.5);
        assert_eq!(c.score(&m), 0.0);
    }

    #[test]
    fn test_rubric_scoring() {
        let mut rubric = Rubric::fiduciary_default();
        let mut metrics = HashMap::new();
        metrics.insert("portfolio_return".to_string(), 0.05);
        metrics.insert("sharpe_ratio".to_string(), 1.5);
        metrics.insert("max_drawdown_ok".to_string(), 1.0);
        metrics.insert("fraud_detection_rate".to_string(), 1.0);
        metrics.insert("document_coverage".to_string(), 1.0);

        let score = rubric.score(&metrics);
        assert!(score > 0.7, "Good metrics should score high: {}", score);
    }

    #[test]
    fn test_saturation_detection() {
        let mut c = RubricCriterion::threshold("easy", "Too easy", 1.0, "x", 0.0);

        // Everything passes → saturated
        let m = HashMap::from([("x".to_string(), 1.0)]);
        for _ in 0..20 {
            c.score(&m);
        }
        assert!(c.is_saturated(0.05));

        // Mix of pass/fail → not saturated
        let mut c2 = RubricCriterion::threshold("hard", "Good", 1.0, "x", 0.5);
        let pass = HashMap::from([("x".to_string(), 1.0)]);
        let fail = HashMap::from([("x".to_string(), 0.0)]);
        for _ in 0..10 {
            c2.score(&pass);
            c2.score(&fail);
        }
        assert!(!c2.is_saturated(0.05));
    }

    #[test]
    fn test_rubric_evolution() {
        let rubric = Rubric::new(vec![
            RubricCriterion::threshold("easy", "Everyone passes", 1.0, "x", 0.0),
            RubricCriterion::threshold("useful", "Discriminative", 1.0, "y", 0.5),
        ]);
        let mut judge = RubricJudge::new(rubric);

        // Score rollouts: "easy" always passes, "useful" sometimes
        for i in 0..30 {
            let mut m = HashMap::new();
            m.insert("x".to_string(), 1.0); // Always pass
            m.insert("y".to_string(), if i % 3 == 0 { 1.0 } else { 0.0 }); // Sometimes
            m.insert("z".to_string(), if i < 10 { 1.0 } else { 0.0 }); // New metric (not tracked)
            judge.score(&m);
        }

        let mut evolver = RubricEvolver::new(1, 0.05, 10);
        evolver.episodes_since_evolution = 1;
        let evolved = evolver.evolve(&judge);

        println!(
            "  Rubric v{}: {} criteria",
            evolved.version,
            evolved.num_criteria()
        );
        for c in &evolved.criteria {
            println!("    {} ({}) weight={:.2}", c.id, c.description, c.weight);
        }

        assert_eq!(evolved.version, 2);
        // "easy" should be pruned (saturated), "useful" should remain
        let ids: Vec<&str> = evolved.criteria.iter().map(|c| c.id.as_str()).collect();
        assert!(!ids.contains(&"easy"), "Saturated 'easy' should be pruned");
        assert!(ids.contains(&"useful"), "'useful' should survive");
    }

    #[test]
    fn test_judge_reward_conversion() {
        let rubric = Rubric::agent_default();
        let mut judge = RubricJudge::new(rubric);

        // Perfect rollout
        let good = HashMap::from([
            ("compile_success".to_string(), 1.0),
            ("test_pass_rate".to_string(), 1.0),
            ("primary_metric".to_string(), 0.5),
            ("action_efficiency".to_string(), 0.8),
            ("no_regressions".to_string(), 1.0),
        ]);
        let reward = judge.score_to_reward(&good, 5.0);
        assert!(
            reward > 0.0,
            "Good rollout should give positive reward: {}",
            reward
        );

        // Bad rollout
        let bad = HashMap::from([
            ("compile_success".to_string(), 0.0),
            ("test_pass_rate".to_string(), 0.3),
            ("primary_metric".to_string(), -0.2),
            ("action_efficiency".to_string(), 0.1),
            ("no_regressions".to_string(), 0.0),
        ]);
        let reward_bad = judge.score_to_reward(&bad, 5.0);
        assert!(reward_bad < reward, "Bad rollout should score lower");
    }
}
