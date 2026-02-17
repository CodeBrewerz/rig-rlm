//! Phase 18: ARC-AGI data loader.
//!
//! Parses ARC-AGI JSON format (directory of .json files, each with
//! train/test pairs of input/output grids) into dspy-rs `Example` values
//! for use with the optimizer and benchmark runner.

use anyhow::{Context, Result};
use dspy_rs::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// A 2D grid of integers (0-9 color indices).
pub type Grid = Vec<Vec<i32>>;

/// A single ARC task loaded from JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArcTask {
    /// Training examples (input→output pairs the agent can study).
    pub train: Vec<ArcPair>,
    /// Test challenges (input grids to predict output for).
    pub test: Vec<ArcPair>,
}

/// An input/output grid pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArcPair {
    pub input: Grid,
    pub output: Grid,
}

/// Load all ARC-AGI tasks from a directory of JSON files.
///
/// Each file is one task. The filename (minus .json) becomes the task ID.
/// Returns `(task_id, task)` pairs sorted by ID.
///
/// Expected directory structure:
/// ```
/// evaluation/
///   00576224.json
///   007bbfb7.json
///   ...
/// ```
pub fn load_arc_dataset(dir: &str) -> Result<Vec<(String, ArcTask)>> {
    let path = Path::new(dir);
    anyhow::ensure!(path.is_dir(), "Not a directory: {dir}");

    let mut tasks = Vec::new();

    for entry in fs::read_dir(path).with_context(|| format!("Failed to read directory: {dir}"))? {
        let entry = entry?;
        let file_path = entry.path();

        if file_path.extension().is_some_and(|ext| ext == "json") {
            let task_id = file_path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            let content = fs::read_to_string(&file_path)
                .with_context(|| format!("Failed to read: {}", file_path.display()))?;

            let task: ArcTask = serde_json::from_str(&content)
                .with_context(|| format!("Failed to parse: {}", file_path.display()))?;

            tasks.push((task_id, task));
        }
    }

    tasks.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(tasks)
}

/// Convert ARC tasks into dspy-rs `Example` values for the optimizer.
///
/// Each Example has:
/// - `examples`: JSON string of training pairs
/// - `challenges`: JSON string of test input grids
/// - `expected_outputs`: JSON string of expected test output grids
pub fn tasks_to_examples(tasks: &[(String, ArcTask)]) -> Vec<Example> {
    tasks
        .iter()
        .map(|(id, task)| {
            let mut data = std::collections::HashMap::new();

            // Input fields
            data.insert("task_id".to_string(), serde_json::Value::String(id.clone()));
            data.insert(
                "examples".to_string(),
                serde_json::to_value(&task.train).unwrap(),
            );
            data.insert(
                "challenges".to_string(),
                serde_json::to_value(&task.test.iter().map(|p| &p.input).collect::<Vec<_>>())
                    .unwrap(),
            );

            // Expected output (ground truth)
            data.insert(
                "expected_outputs".to_string(),
                serde_json::to_value(&task.test.iter().map(|p| &p.output).collect::<Vec<_>>())
                    .unwrap(),
            );

            Example::new(
                data,
                vec![
                    "task_id".to_string(),
                    "examples".to_string(),
                    "challenges".to_string(),
                ],
                vec!["expected_outputs".to_string()],
            )
        })
        .collect()
}

/// Format grid as a human-readable string for display.
pub fn format_grid(grid: &Grid) -> String {
    grid.iter()
        .map(|row| {
            row.iter()
                .map(|cell| cell.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Parse a JSON string as a list of grids.
pub fn parse_grids(json: &str) -> Result<Vec<Grid>> {
    serde_json::from_str(json).context("Failed to parse grids JSON")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_task() -> ArcTask {
        ArcTask {
            train: vec![ArcPair {
                input: vec![vec![0, 0, 0], vec![0, 1, 0], vec![0, 0, 0]],
                output: vec![vec![1, 1, 1], vec![1, 0, 1], vec![1, 1, 1]],
            }],
            test: vec![ArcPair {
                input: vec![vec![0, 0], vec![2, 0]],
                output: vec![vec![2, 2], vec![0, 2]],
            }],
        }
    }

    #[test]
    fn test_json_round_trip() {
        let task = sample_task();
        let json = serde_json::to_string(&task).unwrap();
        let parsed: ArcTask = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.train.len(), 1);
        assert_eq!(parsed.test.len(), 1);
        assert_eq!(parsed.train[0].input, task.train[0].input);
    }

    #[test]
    fn test_tasks_to_examples() {
        let tasks = vec![("test001".to_string(), sample_task())];
        let examples = tasks_to_examples(&tasks);
        assert_eq!(examples.len(), 1);

        let ex = &examples[0];
        assert!(ex.data.contains_key("examples"));
        assert!(ex.data.contains_key("challenges"));
        assert!(ex.data.contains_key("expected_outputs"));
        assert_eq!(ex.input_keys.len(), 3);
        assert_eq!(ex.output_keys.len(), 1);
    }

    #[test]
    fn test_format_grid() {
        let grid = vec![vec![1, 2], vec![3, 4]];
        assert_eq!(format_grid(&grid), "1 2\n3 4");
    }

    #[test]
    fn test_parse_grids() {
        let json = "[[1,2],[3,4]]";
        let grids: Vec<Grid> = serde_json::from_str(&format!("[{json}]")).unwrap();
        assert_eq!(grids.len(), 1);
        assert_eq!(grids[0], vec![vec![1, 2], vec![3, 4]]);
    }
}
