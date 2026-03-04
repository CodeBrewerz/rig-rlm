//! Phase 21: ARC-AGI benchmark runner.
//!
//! Provides baseline (direct evaluation) and optimized (GEPA train→eval)
//! benchmark modes. Produces a `BenchmarkReport` with per-task scores.

use anyhow::Result;
use dspy_rs::*;
use std::time::Instant;

use super::agent::ArcAgentModule;
use super::data::{ArcTask, Grid, load_arc_dataset, tasks_to_examples};
use crate::monad::provider::ProviderConfig;
use crate::sandbox::ExecutorKind;

/// Configuration for benchmark runs.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Path to the ARC dataset directory.
    pub dataset_dir: String,
    /// LLM provider configuration.
    pub provider: ProviderConfig,
    /// Executor kind.
    pub executor_kind: ExecutorKind,
    /// Whether to run GEPA optimization before evaluation.
    pub optimize: bool,
    /// Number of tasks to use as training set (rest = eval).
    /// Only used when `optimize` is true.
    pub train_split: usize,
    /// Number of GEPA iterations (only when optimizing).
    pub gepa_iterations: usize,
    /// Maximum tasks to evaluate (0 = all).
    pub max_tasks: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            dataset_dir: "./arc-agi-2/evaluation".to_string(),
            provider: ProviderConfig::local("gpt-4o"),
            executor_kind: ExecutorKind::Pyo3,
            optimize: false,
            train_split: 50,
            gepa_iterations: 10,
            max_tasks: 0,
        }
    }
}

/// Result of a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// Number of tasks where all test grids were correct.
    pub correct: usize,
    /// Total number of tasks evaluated.
    pub total: usize,
    /// Per-task results.
    pub task_results: Vec<TaskResult>,
    /// Total wall-clock time.
    pub elapsed_seconds: f64,
    /// Whether GEPA optimization was used.
    pub optimized: bool,
}

impl BenchmarkReport {
    /// Overall accuracy as a percentage.
    pub fn accuracy_pct(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        (self.correct as f64 / self.total as f64) * 100.0
    }

    /// Print a summary to stdout.
    pub fn print_summary(&self) {
        println!("\n{}", "=".repeat(60));
        println!("ARC-AGI Benchmark Report");
        println!("{}", "=".repeat(60));
        println!(
            "Mode:     {}",
            if self.optimized {
                "Optimized (GEPA)"
            } else {
                "Baseline"
            }
        );
        println!("Tasks:    {}/{} correct", self.correct, self.total);
        println!("Accuracy: {:.1}%", self.accuracy_pct());
        println!("Time:     {:.1}s", self.elapsed_seconds);
        println!("{}", "=".repeat(60));

        // Show per-task results
        for result in &self.task_results {
            let icon = if result.correct { "✓" } else { "✗" };
            println!(
                "  {icon} {}: {:.0}% ({:.1}s)",
                result.task_id,
                result.score * 100.0,
                result.elapsed_seconds,
            );
        }
    }
}

/// Result for a single ARC task.
#[derive(Debug, Clone)]
pub struct TaskResult {
    /// The task ID (filename without .json).
    pub task_id: String,
    /// Whether ALL test grids were correct.
    pub correct: bool,
    /// Score (0.0-1.0).
    pub score: f32,
    /// Time taken for this task.
    pub elapsed_seconds: f64,
    /// Detailed feedback.
    pub feedback: String,
}

/// Run baseline benchmark — direct evaluation, no optimization.
pub async fn run_baseline(config: &BenchmarkConfig) -> Result<BenchmarkReport> {
    let tasks = load_arc_dataset(&config.dataset_dir)?;
    let limited = if config.max_tasks > 0 && config.max_tasks < tasks.len() {
        &tasks[..config.max_tasks]
    } else {
        &tasks
    };

    let module = ArcAgentModule::new(
        &crate::arc::prompt::ARC_INITIAL_PROMPT,
        config.provider.clone(),
    )
    .with_executor(config.executor_kind.clone());

    let start = Instant::now();
    let mut report = BenchmarkReport {
        correct: 0,
        total: limited.len(),
        task_results: Vec::new(),
        elapsed_seconds: 0.0,
        optimized: false,
    };

    for (i, (task_id, task)) in limited.iter().enumerate() {
        let task_start = Instant::now();
        print!("[{}/{}] {} ... ", i + 1, report.total, task_id);

        let result = evaluate_single_task(&module, task_id, task).await;

        let task_result = match result {
            Ok(tr) => tr,
            Err(e) => TaskResult {
                task_id: task_id.clone(),
                correct: false,
                score: 0.0,
                elapsed_seconds: task_start.elapsed().as_secs_f64(),
                feedback: format!("Error: {e}"),
            },
        };

        if task_result.correct {
            report.correct += 1;
        }
        println!(
            "{} ({:.1}s)",
            if task_result.correct { "✓" } else { "✗" },
            task_result.elapsed_seconds,
        );

        report.task_results.push(task_result);
    }

    report.elapsed_seconds = start.elapsed().as_secs_f64();
    Ok(report)
}

/// Run optimized benchmark — GEPA train→eval split.
pub async fn run_optimized(config: &BenchmarkConfig) -> Result<BenchmarkReport> {
    let tasks = load_arc_dataset(&config.dataset_dir)?;

    anyhow::ensure!(
        config.train_split < tasks.len(),
        "Train split {} >= total tasks {}",
        config.train_split,
        tasks.len()
    );

    let (train_tasks, eval_tasks) = tasks.split_at(config.train_split);

    println!(
        "=== Phase 1: GEPA Optimization on {} training tasks ===",
        train_tasks.len()
    );

    let mut module = ArcAgentModule::new(
        &crate::arc::prompt::ARC_INITIAL_PROMPT,
        config.provider.clone(),
    )
    .with_executor(config.executor_kind.clone());

    let trainset = tasks_to_examples(train_tasks);

    let optimizer = GEPA::builder()
        .num_iterations(config.gepa_iterations)
        .minibatch_size(10.min(trainset.len()))
        .max_lm_calls(500)
        .build();

    let opt_report = optimizer
        .compile_with_feedback(&mut module, trainset)
        .await?;

    println!(
        "Optimization complete: best score = {:.1}%",
        opt_report.best_candidate.average_score() * 100.0
    );
    println!(
        "Optimized instruction: {}",
        module.predictor.signature.instruction()
    );

    // Phase 2: Evaluate on held-out set
    println!(
        "\n=== Phase 2: Evaluation on {} held-out tasks ===",
        eval_tasks.len()
    );

    let limited = if config.max_tasks > 0 && config.max_tasks < eval_tasks.len() {
        &eval_tasks[..config.max_tasks]
    } else {
        eval_tasks
    };

    let start = Instant::now();
    let mut report = BenchmarkReport {
        correct: 0,
        total: limited.len(),
        task_results: Vec::new(),
        elapsed_seconds: 0.0,
        optimized: true,
    };

    for (i, (task_id, task)) in limited.iter().enumerate() {
        let task_start = Instant::now();
        print!("[{}/{}] {} ... ", i + 1, report.total, task_id);

        let result = evaluate_single_task(&module, task_id, task).await;

        let task_result = match result {
            Ok(tr) => tr,
            Err(e) => TaskResult {
                task_id: task_id.clone(),
                correct: false,
                score: 0.0,
                elapsed_seconds: task_start.elapsed().as_secs_f64(),
                feedback: format!("Error: {e}"),
            },
        };

        if task_result.correct {
            report.correct += 1;
        }
        println!(
            "{} ({:.1}s)",
            if task_result.correct { "✓" } else { "✗" },
            task_result.elapsed_seconds,
        );

        report.task_results.push(task_result);
    }

    report.elapsed_seconds = start.elapsed().as_secs_f64();
    Ok(report)
}

/// Evaluate a single ARC task using the module.
async fn evaluate_single_task(
    module: &ArcAgentModule,
    task_id: &str,
    task: &ArcTask,
) -> Result<TaskResult> {
    let start = Instant::now();

    // Build Example for this task
    let mut data = std::collections::HashMap::new();
    data.insert(
        "task_id".to_string(),
        serde_json::Value::String(task_id.to_string()),
    );
    data.insert("examples".to_string(), serde_json::to_value(&task.train)?);
    data.insert(
        "challenges".to_string(),
        serde_json::to_value(&task.test.iter().map(|p| &p.input).collect::<Vec<_>>())?,
    );

    let example = Example::new(
        data,
        vec![
            "task_id".to_string(),
            "examples".to_string(),
            "challenges".to_string(),
        ],
        vec![],
    );

    // Run the module
    let prediction = module.forward(example.clone()).await?;

    // Check accuracy
    let predicted_outputs = prediction
        .get("outputs", None)
        .as_str()
        .unwrap_or("")
        .to_string();
    let predicted_grids: std::result::Result<Vec<Grid>, _> =
        serde_json::from_str(&predicted_outputs);

    let expected_grids: Vec<Grid> = task.test.iter().map(|p| p.output.clone()).collect();

    let (correct, score, feedback) = match predicted_grids {
        Ok(pred) => {
            let mut correct_count = 0;
            let mut lines = Vec::new();

            for (i, (p, e)) in pred.iter().zip(&expected_grids).enumerate() {
                if p == e {
                    correct_count += 1;
                    lines.push(format!("Test {}: ✓", i + 1));
                } else {
                    let mismatches = super::agent::count_cell_mismatches(p, e);
                    lines.push(format!("Test {}: ✗ ({} cells wrong)", i + 1, mismatches));
                }
            }

            let all_correct = correct_count == expected_grids.len() && !expected_grids.is_empty();
            let score = correct_count as f32 / expected_grids.len().max(1) as f32;
            (all_correct, score, lines.join("; "))
        }
        Err(e) => (false, 0.0, format!("Parse error: {e}")),
    };

    Ok(TaskResult {
        task_id: task_id.to_string(),
        correct,
        score,
        elapsed_seconds: start.elapsed().as_secs_f64(),
        feedback,
    })
}
