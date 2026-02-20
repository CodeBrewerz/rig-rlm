//! Phase 8: Recipe DSL — declarative multi-step agent pipelines.
//!
//! A `Recipe` is a DAG of steps where each step runs an agent task.
//! Steps can reference outputs from prior steps via `{{step_id.output}}`
//! templates. Recipes support cost estimation, validation, progress
//! tracking, and YAML serialization.
//!
//! # Example
//!
//! ```rust,no_run
//! use rig_rlm::monad::recipe::{Recipe, StepKind};
//!
//! let recipe = Recipe::new("Data Pipeline")
//!     .description("Analyze data and build model")
//!     .step("analyze", "Load and analyze the CSV file", StepKind::CodeGen)
//!     .step_with_deps(
//!         "model",
//!         "Build model from: {{analyze.output}}",
//!         StepKind::CodeGen,
//!         vec!["analyze"],
//!     )
//!     .step_with_deps(
//!         "report",
//!         "Write report on: {{model.output}}",
//!         StepKind::TextGen,
//!         vec!["model"],
//!     );
//!
//! recipe.validate().unwrap();
//! let estimate = recipe.estimate_cost();
//! println!("Estimated: {} steps, ~{} LLM calls", estimate.total_steps, estimate.estimated_llm_calls);
//! ```

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Duration;

// ─── Core Types ───────────────────────────────────────────────────

/// A declarative multi-step agent pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recipe {
    /// Human-readable name for the pipeline.
    pub name: String,
    /// Optional description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Ordered list of steps.
    pub steps: Vec<RecipeStep>,
}

/// A single step in a recipe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecipeStep {
    /// Unique identifier for this step.
    pub id: String,
    /// The task prompt. May contain `{{prev_step.output}}` templates.
    pub task: String,
    /// What kind of agent work this step performs.
    #[serde(default)]
    pub kind: StepKind,
    /// IDs of steps that must complete before this one.
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// Per-step max_turns override (None = use agent default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<usize>,
}

/// The type of agent work a step performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum StepKind {
    /// Full interaction loop — generate code, execute, feedback.
    #[default]
    CodeGen,
    /// Single LLM call — text generation only, no code execution.
    TextGen,
    /// Spawn a sub-agent for the step.
    SubAgent,
}

// ─── Builder API ──────────────────────────────────────────────────

impl Recipe {
    /// Create a new empty recipe.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            steps: Vec::new(),
        }
    }

    /// Set the description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a step with no dependencies.
    pub fn step(mut self, id: impl Into<String>, task: impl Into<String>, kind: StepKind) -> Self {
        self.steps.push(RecipeStep {
            id: id.into(),
            task: task.into(),
            kind,
            depends_on: Vec::new(),
            max_turns: None,
        });
        self
    }

    /// Add a step with explicit dependencies.
    pub fn step_with_deps(
        mut self,
        id: impl Into<String>,
        task: impl Into<String>,
        kind: StepKind,
        deps: Vec<&str>,
    ) -> Self {
        self.steps.push(RecipeStep {
            id: id.into(),
            task: task.into(),
            kind,
            depends_on: deps.into_iter().map(String::from).collect(),
            max_turns: None,
        });
        self
    }

    /// Add a step with max_turns override.
    pub fn step_with_turns(
        mut self,
        id: impl Into<String>,
        task: impl Into<String>,
        kind: StepKind,
        max_turns: usize,
    ) -> Self {
        self.steps.push(RecipeStep {
            id: id.into(),
            task: task.into(),
            kind,
            depends_on: Vec::new(),
            max_turns: Some(max_turns),
        });
        self
    }

    /// Look up a step by ID.
    pub fn get_step(&self, id: &str) -> Option<&RecipeStep> {
        self.steps.iter().find(|s| s.id == id)
    }

    // ─── Validation ───────────────────────────────────────────────

    /// Validate the recipe for correctness.
    ///
    /// Checks:
    /// - No duplicate step IDs
    /// - All dependencies reference existing steps
    /// - No dependency cycles (DAG property)
    /// - At least one step
    pub fn validate(&self) -> Result<(), RecipeError> {
        if self.steps.is_empty() {
            return Err(RecipeError::Empty);
        }

        // Check duplicate IDs
        let mut seen = HashSet::new();
        for step in &self.steps {
            if !seen.insert(&step.id) {
                return Err(RecipeError::DuplicateId(step.id.clone()));
            }
        }

        // Check missing dependencies
        let ids: HashSet<&str> = self.steps.iter().map(|s| s.id.as_str()).collect();
        for step in &self.steps {
            for dep in &step.depends_on {
                if !ids.contains(dep.as_str()) {
                    return Err(RecipeError::MissingDependency {
                        step: step.id.clone(),
                        dependency: dep.clone(),
                    });
                }
            }
        }

        // Check for cycles using DFS
        self.check_cycles()?;

        Ok(())
    }

    /// Detect cycles using iterative DFS with coloring.
    fn check_cycles(&self) -> Result<(), RecipeError> {
        // 0 = unvisited, 1 = in-progress, 2 = done
        let mut state: HashMap<&str, u8> = HashMap::new();
        for step in &self.steps {
            state.insert(&step.id, 0);
        }

        let adj: HashMap<&str, Vec<&str>> = self
            .steps
            .iter()
            .map(|s| {
                (
                    s.id.as_str(),
                    s.depends_on.iter().map(|d| d.as_str()).collect(),
                )
            })
            .collect();

        for step in &self.steps {
            if state[step.id.as_str()] == 0 {
                let mut stack = vec![(step.id.as_str(), false)];
                while let Some((node, returning)) = stack.pop() {
                    if returning {
                        state.insert(node, 2);
                        continue;
                    }
                    if state[node] == 1 {
                        return Err(RecipeError::CyclicDependency(node.to_string()));
                    }
                    if state[node] == 2 {
                        continue;
                    }
                    state.insert(node, 1);
                    stack.push((node, true)); // mark for completion
                    if let Some(deps) = adj.get(node) {
                        for dep in deps {
                            if state[*dep] == 1 {
                                return Err(RecipeError::CyclicDependency(dep.to_string()));
                            }
                            if state[*dep] == 0 {
                                stack.push((dep, false));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    // ─── Execution Order ──────────────────────────────────────────

    /// Compute topological execution order (Kahn's algorithm).
    ///
    /// Returns step IDs in an order where all dependencies are satisfied.
    pub fn execution_order(&self) -> Result<Vec<String>, RecipeError> {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut dependents: HashMap<&str, Vec<&str>> = HashMap::new();

        for step in &self.steps {
            in_degree.entry(&step.id).or_insert(0);
            for dep in &step.depends_on {
                *in_degree.entry(&step.id).or_insert(0) += 1;
                dependents.entry(dep.as_str()).or_default().push(&step.id);
            }
        }

        let mut queue: Vec<&str> = in_degree
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(id, _)| *id)
            .collect();
        // Stable sort for deterministic ordering
        queue.sort();

        let mut order = Vec::new();
        while let Some(id) = queue.pop() {
            order.push(id.to_string());
            if let Some(deps) = dependents.get(id) {
                for dep_id in deps {
                    if let Some(deg) = in_degree.get_mut(dep_id) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push(dep_id);
                            queue.sort();
                        }
                    }
                }
            }
        }

        if order.len() != self.steps.len() {
            return Err(RecipeError::CyclicDependency("unresolvable".to_string()));
        }

        Ok(order)
    }

    // ─── Template Resolution ──────────────────────────────────────

    /// Resolve `{{step_id.output}}` templates in a step's task.
    pub fn resolve_task(step: &RecipeStep, outputs: &IndexMap<String, StepResult>) -> String {
        let mut task = step.task.clone();
        for (id, result) in outputs {
            let placeholder = format!("{{{{{}.output}}}}", id);
            if task.contains(&placeholder) {
                let replacement = match &result.status {
                    StepStatus::Completed => &result.output,
                    StepStatus::Failed(e) => e,
                    StepStatus::Skipped => "[skipped]",
                };
                task = task.replace(&placeholder, replacement);
            }
        }
        task
    }

    // ─── Cost Estimation ──────────────────────────────────────────

    /// Estimate the cost of running this recipe.
    pub fn estimate_cost(&self) -> RecipeCostEstimate {
        let mut total_llm_calls = 0;
        let mut estimated_minutes = 0.0;

        for step in &self.steps {
            let max_turns = step.max_turns.unwrap_or(25);
            let (calls, mins) = match step.kind {
                StepKind::CodeGen => {
                    // CodeGen: typically ~3-8 turns (LLM + exec each)
                    let avg_turns = max_turns.min(8);
                    (avg_turns, avg_turns as f64 * 0.5)
                }
                StepKind::TextGen => {
                    // TextGen: single LLM call
                    (1, 0.3)
                }
                StepKind::SubAgent => {
                    // SubAgent: similar to CodeGen
                    let avg_turns = max_turns.min(6);
                    (avg_turns, avg_turns as f64 * 0.5)
                }
            };
            total_llm_calls += calls;
            estimated_minutes += mins;
        }

        RecipeCostEstimate {
            total_steps: self.steps.len(),
            estimated_llm_calls: total_llm_calls,
            estimated_minutes,
        }
    }

    // ─── YAML ─────────────────────────────────────────────────────

    /// Load a recipe from YAML.
    pub fn from_yaml(yaml: &str) -> Result<Self, RecipeError> {
        serde_yaml::from_str(yaml).map_err(|e| RecipeError::Yaml(e.to_string()))
    }

    /// Serialize to YAML.
    pub fn to_yaml(&self) -> Result<String, RecipeError> {
        serde_yaml::to_string(self).map_err(|e| RecipeError::Yaml(e.to_string()))
    }
}

// ─── Results ──────────────────────────────────────────────────────

/// The result of running an entire recipe.
#[derive(Debug, Clone)]
pub struct RecipeResult {
    /// Recipe name.
    pub recipe_name: String,
    /// Per-step results, in execution order.
    pub steps: IndexMap<String, StepResult>,
    /// Total turns across all steps.
    pub total_turns: usize,
    /// Wall-clock time for the entire recipe.
    pub elapsed: Duration,
    /// Total cost in USD for the entire recipe.
    pub total_cost_usd: f64,
}

/// The result of a single recipe step.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// The output from the agent.
    pub output: String,
    /// How many turns this step took.
    pub turns: usize,
    /// Wall-clock time for this step.
    pub elapsed: Duration,
    /// Completion status.
    pub status: StepStatus,
    /// Cost in USD for this step.
    pub cost_usd: f64,
}

/// Status of a recipe step.
#[derive(Debug, Clone)]
pub enum StepStatus {
    /// Step completed successfully.
    Completed,
    /// Step failed with an error.
    Failed(String),
    /// Step was skipped (e.g., dependency failed).
    Skipped,
}

impl StepResult {
    /// Create a completed step result.
    pub fn completed(output: String, turns: usize, elapsed: Duration, cost_usd: f64) -> Self {
        Self {
            output,
            turns,
            elapsed,
            status: StepStatus::Completed,
            cost_usd,
        }
    }

    /// Create a failed step result.
    pub fn failed(error: String, turns: usize, elapsed: Duration, cost_usd: f64) -> Self {
        Self {
            output: String::new(),
            turns,
            elapsed,
            status: StepStatus::Failed(error),
            cost_usd,
        }
    }

    /// Create a skipped step result.
    pub fn skipped() -> Self {
        Self {
            output: String::new(),
            turns: 0,
            elapsed: Duration::ZERO,
            status: StepStatus::Skipped,
            cost_usd: 0.0,
        }
    }
}

impl RecipeResult {
    /// Print a human-readable summary.
    pub fn print_summary(&self) {
        println!("═══════════════════════════════════════════");
        println!("  Recipe: {}", self.recipe_name);
        println!("  Steps: {}", self.steps.len());
        println!("  Total turns: {}", self.total_turns);
        println!("  Elapsed: {:.1}s", self.elapsed.as_secs_f64());
        if self.total_cost_usd > 0.0 {
            println!("  Total cost: ${:.6}", self.total_cost_usd);
        }
        println!("═══════════════════════════════════════════");
        for (id, result) in &self.steps {
            let icon = match &result.status {
                StepStatus::Completed => "✅",
                StepStatus::Failed(_) => "❌",
                StepStatus::Skipped => "⏭️",
            };
            println!(
                "  {icon} {id}: {} turns, {:.1}s",
                result.turns,
                result.elapsed.as_secs_f64()
            );
            match &result.status {
                StepStatus::Completed => {
                    let preview = if result.output.len() > 100 {
                        format!("{}...", &result.output[..100])
                    } else {
                        result.output.clone()
                    };
                    println!("     → {preview}");
                }
                StepStatus::Failed(e) => println!("     → Error: {e}"),
                StepStatus::Skipped => println!("     → Skipped"),
            }
        }
    }

    /// Get the final output (last completed step).
    pub fn final_output(&self) -> Option<&str> {
        self.steps
            .values()
            .rev()
            .find(|r| matches!(r.status, StepStatus::Completed))
            .map(|r| r.output.as_str())
    }
}

/// Cost estimate for running a recipe.
#[derive(Debug, Clone)]
pub struct RecipeCostEstimate {
    /// Number of steps in the recipe.
    pub total_steps: usize,
    /// Estimated total LLM calls.
    pub estimated_llm_calls: usize,
    /// Estimated wall-clock minutes.
    pub estimated_minutes: f64,
}

// ─── Errors ───────────────────────────────────────────────────────

/// Recipe validation/execution errors.
#[derive(Debug, Clone)]
pub enum RecipeError {
    /// Recipe has no steps.
    Empty,
    /// Duplicate step ID.
    DuplicateId(String),
    /// Step references a non-existent dependency.
    MissingDependency { step: String, dependency: String },
    /// Dependency cycle detected.
    CyclicDependency(String),
    /// YAML parsing error.
    Yaml(String),
    /// Step execution failed.
    StepFailed { step: String, error: String },
}

impl std::fmt::Display for RecipeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "recipe has no steps"),
            Self::DuplicateId(id) => write!(f, "duplicate step ID: {id}"),
            Self::MissingDependency { step, dependency } => {
                write!(f, "step '{step}' depends on unknown step '{dependency}'")
            }
            Self::CyclicDependency(id) => write!(f, "cyclic dependency involving '{id}'"),
            Self::Yaml(e) => write!(f, "YAML error: {e}"),
            Self::StepFailed { step, error } => write!(f, "step '{step}' failed: {error}"),
        }
    }
}

impl std::error::Error for RecipeError {}

impl From<RecipeError> for super::error::AgentError {
    fn from(e: RecipeError) -> Self {
        super::error::AgentError::Internal(e.to_string())
    }
}

// ─── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_recipe() -> Recipe {
        Recipe::new("Test Pipeline")
            .description("A test pipeline")
            .step("load", "Load the data", StepKind::CodeGen)
            .step_with_deps(
                "clean",
                "Clean: {{load.output}}",
                StepKind::CodeGen,
                vec!["load"],
            )
            .step_with_deps(
                "model",
                "Model: {{clean.output}}",
                StepKind::CodeGen,
                vec!["clean"],
            )
            .step_with_deps(
                "report",
                "Report: {{model.output}}",
                StepKind::TextGen,
                vec!["model"],
            )
    }

    #[test]
    fn recipe_validates_ok() {
        let recipe = sample_recipe();
        assert!(recipe.validate().is_ok());
    }

    #[test]
    fn recipe_detects_duplicate_id() {
        let recipe = Recipe::new("test")
            .step("a", "first", StepKind::CodeGen)
            .step("a", "duplicate", StepKind::CodeGen);
        match recipe.validate() {
            Err(RecipeError::DuplicateId(id)) => assert_eq!(id, "a"),
            other => panic!("expected DuplicateId, got: {other:?}"),
        }
    }

    #[test]
    fn recipe_detects_missing_dep() {
        let recipe = Recipe::new("test")
            .step("a", "task a", StepKind::CodeGen)
            .step_with_deps("b", "task b", StepKind::CodeGen, vec!["nonexistent"]);
        match recipe.validate() {
            Err(RecipeError::MissingDependency { step, dependency }) => {
                assert_eq!(step, "b");
                assert_eq!(dependency, "nonexistent");
            }
            other => panic!("expected MissingDependency, got: {other:?}"),
        }
    }

    #[test]
    fn recipe_detects_cycle() {
        let mut recipe = Recipe::new("test");
        recipe.steps.push(RecipeStep {
            id: "a".into(),
            task: "task a".into(),
            kind: StepKind::CodeGen,
            depends_on: vec!["b".into()],
            max_turns: None,
        });
        recipe.steps.push(RecipeStep {
            id: "b".into(),
            task: "task b".into(),
            kind: StepKind::CodeGen,
            depends_on: vec!["a".into()],
            max_turns: None,
        });
        assert!(matches!(
            recipe.validate(),
            Err(RecipeError::CyclicDependency(_))
        ));
    }

    #[test]
    fn recipe_empty_fails() {
        let recipe = Recipe::new("empty");
        assert!(matches!(recipe.validate(), Err(RecipeError::Empty)));
    }

    #[test]
    fn recipe_execution_order() {
        let recipe = sample_recipe();
        let order = recipe.execution_order().unwrap();
        assert_eq!(order, vec!["load", "clean", "model", "report"]);
    }

    #[test]
    fn recipe_execution_order_parallel_roots() {
        let recipe = Recipe::new("parallel")
            .step("a", "task a", StepKind::CodeGen)
            .step("b", "task b", StepKind::CodeGen)
            .step_with_deps("c", "task c", StepKind::CodeGen, vec!["a", "b"]);
        let order = recipe.execution_order().unwrap();
        // a and b should come before c, both are roots
        assert!(
            order.iter().position(|s| s == "a").unwrap()
                < order.iter().position(|s| s == "c").unwrap()
        );
        assert!(
            order.iter().position(|s| s == "b").unwrap()
                < order.iter().position(|s| s == "c").unwrap()
        );
    }

    #[test]
    fn recipe_resolve_task_templates() {
        let recipe = sample_recipe();
        let step = recipe.get_step("clean").unwrap();
        let mut outputs = IndexMap::new();
        outputs.insert(
            "load".to_string(),
            StepResult::completed("loaded data".to_string(), 3, Duration::from_secs(1), 0.0),
        );
        let resolved = Recipe::resolve_task(step, &outputs);
        assert_eq!(resolved, "Clean: loaded data");
    }

    #[test]
    fn recipe_cost_estimate() {
        let recipe = sample_recipe();
        let estimate = recipe.estimate_cost();
        assert_eq!(estimate.total_steps, 4);
        assert!(estimate.estimated_llm_calls > 0);
        assert!(estimate.estimated_minutes > 0.0);
    }

    #[test]
    fn recipe_yaml_roundtrip() {
        let recipe = sample_recipe();
        let yaml = recipe.to_yaml().unwrap();
        let parsed = Recipe::from_yaml(&yaml).unwrap();
        assert_eq!(parsed.name, recipe.name);
        assert_eq!(parsed.steps.len(), recipe.steps.len());
        assert_eq!(parsed.steps[0].id, "load");
        assert_eq!(parsed.steps[1].depends_on, vec!["load"]);
    }

    #[test]
    fn recipe_from_yaml_file_format() {
        let yaml = r#"
name: Data Pipeline
description: End-to-end analysis
steps:
  - id: load
    task: "Load the CSV file"
    kind: code_gen
  - id: model
    task: "Build model from: {{load.output}}"
    kind: code_gen
    depends_on: [load]
  - id: report
    task: "Write report"
    kind: text_gen
    depends_on: [model]
"#;
        let recipe = Recipe::from_yaml(yaml).unwrap();
        assert_eq!(recipe.name, "Data Pipeline");
        assert_eq!(recipe.steps.len(), 3);
        assert_eq!(recipe.steps[1].kind, StepKind::CodeGen);
        assert_eq!(recipe.steps[2].kind, StepKind::TextGen);
        assert!(recipe.validate().is_ok());
    }
}
