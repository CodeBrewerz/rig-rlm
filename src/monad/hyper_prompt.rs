//! HyperPromptEvolver — Metacognitive system prompt self-modification.
//!
//! Applies the HyperAgent (DGM-H) Meta-Prompt Evolution pattern to the agent's
//! system prompt. When per-task-type success rates drop below a configurable
//! threshold, the system calls the LLM to rewrite the underperforming section(s)
//! of its own system prompt.
//!
//! ## Architecture
//!
//! ```text
//! Task Execution → Score (pass/fail per task type) → TaskTypeMetrics
//!   │                                                     │
//!   │  if pass_rate < threshold for any task type          │
//!   │      ↓                                              │
//!   │  LLM rewrites the system prompt section             │
//!   │  for that task type                                 │
//!   │      ↓                                              │
//!   │  New prompt version installed                       │
//!   └─────────────────────────────────────────────────────┘
//! ```

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════
// Task-Type Performance Tracking
// ═══════════════════════════════════════════════════════════════

/// Categories of tasks the agent performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskType {
    Coding,
    Analysis,
    Debugging,
    Refactoring,
    Documentation,
    Testing,
    Research,
    General,
}

impl TaskType {
    pub fn all() -> Vec<Self> {
        vec![
            Self::Coding,
            Self::Analysis,
            Self::Debugging,
            Self::Refactoring,
            Self::Documentation,
            Self::Testing,
            Self::Research,
            Self::General,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Coding => "coding",
            Self::Analysis => "analysis",
            Self::Debugging => "debugging",
            Self::Refactoring => "refactoring",
            Self::Documentation => "documentation",
            Self::Testing => "testing",
            Self::Research => "research",
            Self::General => "general",
        }
    }

    /// Detect task type from a user query string.
    pub fn classify(query: &str) -> Self {
        let q = query.to_lowercase();
        if q.contains("debug") || q.contains("fix") || q.contains("error") || q.contains("bug") {
            Self::Debugging
        } else if q.contains("test") || q.contains("assert") || q.contains("verify") {
            Self::Testing
        } else if q.contains("refactor") || q.contains("clean") || q.contains("improve code") {
            Self::Refactoring
        } else if q.contains("document") || q.contains("readme") || q.contains("explain") {
            Self::Documentation
        } else if q.contains("analyze") || q.contains("review") || q.contains("audit") {
            Self::Analysis
        } else if q.contains("research") || q.contains("investigate") || q.contains("explore") {
            Self::Research
        } else if q.contains("code") || q.contains("implement") || q.contains("write") || q.contains("create") || q.contains("build") {
            Self::Coding
        } else {
            Self::General
        }
    }
}

/// Per-task-type performance metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskTypeMetrics {
    pub successes: u32,
    pub failures: u32,
    /// Recent failure examples (last 5) for reflection.
    pub recent_failures: Vec<String>,
}

impl TaskTypeMetrics {
    pub fn new() -> Self {
        Self {
            successes: 0,
            failures: 0,
            recent_failures: Vec::new(),
        }
    }

    pub fn total(&self) -> u32 {
        self.successes + self.failures
    }

    pub fn pass_rate(&self) -> f64 {
        if self.total() == 0 {
            1.0 // No data = assume good
        } else {
            self.successes as f64 / self.total() as f64
        }
    }

    pub fn record_success(&mut self) {
        self.successes += 1;
    }

    pub fn record_failure(&mut self, description: &str) {
        self.failures += 1;
        self.recent_failures.push(description.to_string());
        if self.recent_failures.len() > 5 {
            self.recent_failures.remove(0);
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// HyperPromptEvolver — The metacognitive loop
// ═══════════════════════════════════════════════════════════════

/// HyperPromptEvolver: self-modifying agent system prompt.
///
/// Tracks per-task-type success rates and, when performance degrades
/// below `evolution_threshold`, calls the LLM to rewrite the
/// underperforming section of the system prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperPromptEvolver {
    /// Per-task-type performance metrics.
    pub metrics: HashMap<String, TaskTypeMetrics>,

    /// The current evolved instruction addendum (the "Additional Instructions"
    /// section that gets injected via `render_system_with_instruction`).
    /// `None` means the base template is used unmodified.
    pub evolved_instruction: Option<String>,

    /// Prompt version counter.
    pub version: u32,

    /// Version history: version → (instruction_text, trigger_reason).
    pub history: Vec<PromptVersion>,

    // ── Configuration ──

    /// Minimum pass rate before evolution triggers. Default: 0.60.
    pub evolution_threshold: f64,

    /// Minimum number of tasks of a given type before evolution triggers.
    pub min_tasks_before_evolve: u32,

    /// Number of evolution events so far.
    pub evolution_count: u32,
}

/// A recorded prompt version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptVersion {
    pub version: u32,
    pub instruction: String,
    pub trigger: String,
    pub task_type: String,
    pub pass_rate_before: f64,
}

/// The LLM prompt used to evolve the system prompt.
const HYPER_PROMPT_EVOLVE: &str = r#"You are a metacognitive AI architect. Your job is to improve an AI agent's system prompt.

## Current System Prompt Addendum
{current_instruction}

## Performance Problem
The agent is underperforming on {task_type} tasks:
- Pass rate: {pass_rate:.1}% (threshold: {threshold:.1}%)
- Total attempts: {total}
- Recent failures:
{failure_examples}

## Your Task
Rewrite the "Additional Instructions" section to help the agent perform better on {task_type} tasks.

Rules:
1. Keep instructions concise (max 200 words)
2. Be specific about what the agent should do differently for {task_type} tasks
3. Include concrete strategies, not vague advice
4. Preserve any existing instructions that work well for other task types
5. Output ONLY the new instruction text, no explanation

New instruction:"#;

impl HyperPromptEvolver {
    /// Create a new evolver with default config.
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            evolved_instruction: None,
            version: 0,
            history: Vec::new(),
            evolution_threshold: 0.60,
            min_tasks_before_evolve: 3,
            evolution_count: 0,
        }
    }

    /// Record a task outcome.
    pub fn record(&mut self, task_type: TaskType, success: bool, failure_desc: Option<&str>) {
        let entry = self
            .metrics
            .entry(task_type.name().to_string())
            .or_insert_with(TaskTypeMetrics::new);
        if success {
            entry.record_success();
        } else {
            entry.record_failure(failure_desc.unwrap_or("unspecified failure"));
        }
    }

    /// Check if any task type needs evolution.
    pub fn needs_evolution(&self) -> Option<(String, f64)> {
        for (task_type, metrics) in &self.metrics {
            if metrics.total() >= self.min_tasks_before_evolve
                && metrics.pass_rate() < self.evolution_threshold
            {
                return Some((task_type.clone(), metrics.pass_rate()));
            }
        }
        None
    }

    /// Build the evolution prompt for the LLM.
    pub fn build_evolution_prompt(&self, task_type: &str, pass_rate: f64) -> String {
        let current = self
            .evolved_instruction
            .as_deref()
            .unwrap_or("(no additional instructions — using base template)");

        let metrics = self.metrics.get(task_type);
        let total = metrics.map(|m| m.total()).unwrap_or(0);
        let failure_examples = metrics
            .map(|m| {
                m.recent_failures
                    .iter()
                    .enumerate()
                    .map(|(i, f)| format!("  {}. {}", i + 1, f))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_else(|| "  (none recorded)".to_string());

        HYPER_PROMPT_EVOLVE
            .replace("{current_instruction}", current)
            .replace("{task_type}", task_type)
            .replace("{pass_rate:.1}", &format!("{:.1}", pass_rate * 100.0))
            .replace("{threshold:.1}", &format!("{:.1}", self.evolution_threshold * 100.0))
            .replace("{total}", &total.to_string())
            .replace("{failure_examples}", &failure_examples)
    }

    /// Install a new evolved instruction (called after LLM generates the new prompt).
    pub fn install_evolution(&mut self, new_instruction: String, task_type: &str, pass_rate: f64) {
        self.version += 1;
        self.evolution_count += 1;

        self.history.push(PromptVersion {
            version: self.version,
            instruction: new_instruction.clone(),
            trigger: format!("{} pass_rate={:.2}", task_type, pass_rate),
            task_type: task_type.to_string(),
            pass_rate_before: pass_rate,
        });

        self.evolved_instruction = Some(new_instruction);

        // Reset metrics for the evolved task type so we measure the new prompt fairly
        if let Some(m) = self.metrics.get_mut(task_type) {
            m.successes = 0;
            m.failures = 0;
            m.recent_failures.clear();
        }
    }

    /// Get the current instruction for injection into the system prompt.
    pub fn current_instruction(&self) -> &str {
        self.evolved_instruction.as_deref().unwrap_or("")
    }

    /// Summary for diagnostics.
    pub fn summary(&self) -> String {
        let mut s = format!("HyperPromptEvolver v{} (evolved {} times)\n", self.version, self.evolution_count);
        for (task_type, metrics) in &self.metrics {
            s.push_str(&format!(
                "  {}: {}/{} ({:.1}% pass)\n",
                task_type,
                metrics.successes,
                metrics.total(),
                metrics.pass_rate() * 100.0
            ));
        }
        if let Some(ref inst) = self.evolved_instruction {
            s.push_str(&format!(
                "  Current instruction ({} chars): {}...\n",
                inst.len(),
                &inst[..inst.len().min(80)]
            ));
        }
        s
    }

    /// Save to disk.
    pub fn save(&self, path: &str) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("serialize error: {}", e))?;
        std::fs::write(path, json).map_err(|e| format!("write error: {}", e))
    }

    /// Load from disk.
    pub fn load(path: &str) -> Result<Self, String> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| format!("read error: {}", e))?;
        serde_json::from_str(&json).map_err(|e| format!("parse error: {}", e))
    }

    /// Load or create default.
    pub fn load_or_default(path: &str) -> Self {
        Self::load(path).unwrap_or_else(|_| Self::new())
    }
}

impl Default for HyperPromptEvolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_type_classification() {
        assert_eq!(TaskType::classify("debug this error"), TaskType::Debugging);
        assert_eq!(TaskType::classify("write a function"), TaskType::Coding);
        assert_eq!(TaskType::classify("analyze the code"), TaskType::Analysis);
        assert_eq!(TaskType::classify("add unit tests"), TaskType::Testing);
        assert_eq!(TaskType::classify("update the readme"), TaskType::Documentation);
        assert_eq!(TaskType::classify("refactor the module"), TaskType::Refactoring);
        assert_eq!(TaskType::classify("hello"), TaskType::General);
    }

    #[test]
    fn test_metrics_tracking() {
        let mut evolver = HyperPromptEvolver::new();

        // Record some successes
        evolver.record(TaskType::Coding, true, None);
        evolver.record(TaskType::Coding, true, None);

        assert!(evolver.needs_evolution().is_none(), "2 tasks < 3 min threshold");

        // Record failures to trigger evolution
        evolver.record(TaskType::Coding, false, Some("syntax error in output"));
        evolver.record(TaskType::Coding, false, Some("wrong function signature"));
        evolver.record(TaskType::Coding, false, Some("missing imports"));

        // Now: 2 success, 3 failure = 5 total, 40% pass rate
        let metrics = evolver.metrics.get("coding").unwrap();
        assert_eq!(metrics.total(), 5);
        assert!((metrics.pass_rate() - 0.40).abs() < 0.01);

        // Should trigger evolution
        let trigger = evolver.needs_evolution();
        assert!(trigger.is_some());
        let (task_type, rate) = trigger.unwrap();
        assert_eq!(task_type, "coding");
        assert!((rate - 0.40).abs() < 0.01);
    }

    #[test]
    fn test_evolution_prompt_generation() {
        let mut evolver = HyperPromptEvolver::new();
        evolver.record(TaskType::Analysis, false, Some("missed key insight"));
        evolver.record(TaskType::Analysis, false, Some("incomplete analysis"));
        evolver.record(TaskType::Analysis, false, Some("wrong conclusion"));

        let prompt = evolver.build_evolution_prompt("analysis", 0.0);
        assert!(prompt.contains("analysis"));
        assert!(prompt.contains("missed key insight"));
        assert!(prompt.contains("incomplete analysis"));
        assert!(prompt.contains("wrong conclusion"));
    }

    #[test]
    fn test_evolution_install() {
        let mut evolver = HyperPromptEvolver::new();
        evolver.record(TaskType::Debugging, false, Some("failed to find root cause"));
        evolver.record(TaskType::Debugging, false, Some("wrong fix"));
        evolver.record(TaskType::Debugging, false, Some("missed edge case"));

        assert_eq!(evolver.version, 0);

        evolver.install_evolution(
            "For debugging tasks: always reproduce the error first. Use binary search.".to_string(),
            "debugging",
            0.0,
        );

        assert_eq!(evolver.version, 1);
        assert_eq!(evolver.evolution_count, 1);
        assert!(evolver.current_instruction().contains("binary search"));

        // Metrics should be reset for the evolved task type
        let metrics = evolver.metrics.get("debugging").unwrap();
        assert_eq!(metrics.total(), 0);

        // History should record the evolution
        assert_eq!(evolver.history.len(), 1);
        assert_eq!(evolver.history[0].task_type, "debugging");
    }

    #[test]
    fn test_evolution_does_not_trigger_for_high_pass_rate() {
        let mut evolver = HyperPromptEvolver::new();
        // All successes
        for _ in 0..10 {
            evolver.record(TaskType::Coding, true, None);
        }
        assert!(evolver.needs_evolution().is_none());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut evolver = HyperPromptEvolver::new();
        evolver.record(TaskType::Coding, true, None);
        evolver.record(TaskType::Coding, false, Some("test failure"));
        evolver.install_evolution("test instruction".to_string(), "coding", 0.5);

        let path = "/tmp/test_hyper_prompt_evolver.json";
        evolver.save(path).unwrap();
        let loaded = HyperPromptEvolver::load(path).unwrap();

        assert_eq!(loaded.version, 1);
        assert_eq!(loaded.current_instruction(), "test instruction");
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_summary() {
        let mut evolver = HyperPromptEvolver::new();
        evolver.record(TaskType::Coding, true, None);
        evolver.record(TaskType::Analysis, false, Some("bad"));

        let summary = evolver.summary();
        assert!(summary.contains("HyperPromptEvolver v0"));
        assert!(summary.contains("coding"));
    }
}
