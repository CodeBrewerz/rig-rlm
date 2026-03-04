//! Phase 19-20: ARC Agent Module — wraps the monadic loop for ARC tasks.
//!
//! Follows the same pattern as `agent_module.rs`:
//! - `Module::forward(Example) -> Result<Prediction>` — runs generate→execute→feedback
//! - `Optimizable` — exposes predictor for GEPA optimization
//! - `FeedbackEvaluator` — exact grid match + cell-level feedback (Phase 20)

use anyhow::Result;
use dspy_rs::*;
use indexmap::IndexMap;

use super::data::Grid;
use super::prompt::format_arc_task;
use super::signature::ArcSolver;
use crate::monad::interaction::agent_task;
use crate::monad::provider::ProviderConfig;
use crate::monad::{AgentConfig, AgentContext, Role};
use crate::sandbox::ExecutorKind;

/// Wraps the monadic agent loop for ARC-AGI tasks.
///
/// Same "hollow Predict" pattern as `AgentModule`:
/// the Predict stores the optimizable instruction + signature,
/// but `forward()` delegates to the full monadic loop with REPL,
/// code execution, and feedback.
pub struct ArcAgentModule {
    /// The optimizable instruction leaf.
    pub predictor: Predict,
    /// LLM provider config.
    pub provider_config: ProviderConfig,
    /// Which executor to use.
    pub executor_kind: ExecutorKind,
}

impl ArcAgentModule {
    /// Create with a specific instruction and provider.
    pub fn new(instruction: &str, provider_config: ProviderConfig) -> Self {
        let mut sig = ArcSolver::new();
        let _ = sig.update_instruction(instruction.to_string());
        Self {
            predictor: Predict::new(sig),
            provider_config,
            executor_kind: ExecutorKind::Pyo3,
        }
    }

    /// Create with default ARC prompt and a local provider.
    pub fn default_local() -> Self {
        Self {
            predictor: Predict::new(ArcSolver::new()),
            provider_config: ProviderConfig::local("qwen/qwen3-8b"),
            executor_kind: ExecutorKind::Pyo3,
        }
    }

    /// Set the executor kind.
    pub fn with_executor(mut self, kind: ExecutorKind) -> Self {
        self.executor_kind = kind;
        self
    }

    /// Set the provider config.
    pub fn with_provider(mut self, provider_config: ProviderConfig) -> Self {
        self.provider_config = provider_config;
        self
    }
}

impl Module for ArcAgentModule {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        // 1. Get optimizable instruction
        let instruction = self.predictor.signature.instruction();

        // 2. Extract training examples and test challenges
        let examples_json = inputs.get("examples", None);
        let challenges_json = inputs.get("challenges", None);

        // 3. Build the task prompt
        let user_prompt = format_arc_task(&examples_json, &challenges_json);
        let task_prompt = format!("{instruction}\n\n{user_prompt}");

        // 4. Build agent config
        let agent_config = AgentConfig {
            provider: self.provider_config.clone(),
            executor_kind: self.executor_kind.clone(),
            ..AgentConfig::default()
        };

        // 5. Run the monadic agent
        let mut ctx = AgentContext::new(agent_config);
        let program = agent_task(&task_prompt);

        let result = ctx
            .run(program)
            .await
            .map_err(|e| anyhow::anyhow!("ARC agent execution failed: {e}"))?;

        // 6. Extract code and outputs from the result
        let (code, outputs) = extract_arc_outputs(&result, &ctx);

        // 7. Return as a Prediction
        let prediction = dspy_rs::prediction! {
            "code" => code,
            "outputs" => outputs
        };

        Ok(prediction)
    }
}

impl Optimizable for ArcAgentModule {
    fn get_signature(&self) -> &dyn MetaSignature {
        self.predictor.get_signature()
    }

    fn parameters(&mut self) -> IndexMap<String, &mut dyn Optimizable> {
        let mut map = IndexMap::new();
        map.insert(
            "predictor".to_string(),
            &mut self.predictor as &mut dyn Optimizable,
        );
        map
    }

    fn update_signature_instruction(&mut self, instruction: String) -> Result<()> {
        self.predictor.update_signature_instruction(instruction)
    }
}

/// Phase 20: FeedbackEvaluator for ARC — exact grid match with cell-level feedback.
///
/// This drives GEPA's evolutionary search by providing:
/// - Binary per-test accuracy (correct/wrong)
/// - Cell-level mismatch count for wrong answers
/// - Dimension mismatch detection
impl FeedbackEvaluator for ArcAgentModule {
    async fn feedback_metric(&self, example: &Example, prediction: &Prediction) -> FeedbackMetric {
        let predicted_outputs = prediction
            .get("outputs", None)
            .as_str()
            .unwrap_or("")
            .to_string();
        let expected_outputs = example.get("expected_outputs", None);

        // Parse grids
        let predicted: std::result::Result<Vec<Grid>, _> = serde_json::from_str(&predicted_outputs);
        let expected: std::result::Result<Vec<Grid>, _> =
            serde_json::from_value(expected_outputs.clone());

        let (predicted, expected) = match (predicted, expected) {
            (Ok(p), Ok(e)) => (p, e),
            (Err(e), _) => {
                return FeedbackMetric::new(
                    0.0,
                    format!("✗ Could not parse predicted outputs: {e}"),
                );
            }
            (_, Err(e)) => {
                return FeedbackMetric::new(
                    0.0,
                    format!("✗ Could not parse expected outputs: {e}"),
                );
            }
        };

        if predicted.is_empty() {
            return FeedbackMetric::new(0.0, "✗ No predicted outputs".to_string());
        }

        let mut correct = 0;
        let mut feedback_lines = Vec::new();

        for (i, (pred, exp)) in predicted.iter().zip(&expected).enumerate() {
            if pred == exp {
                correct += 1;
                feedback_lines.push(format!("Test {}: ✓ Correct grid", i + 1));
            } else {
                let mismatches = count_cell_mismatches(pred, exp);
                let dim_match = pred.len() == exp.len()
                    && pred
                        .first()
                        .map_or(true, |r| exp.first().map_or(true, |er| r.len() == er.len()));

                feedback_lines.push(format!(
                    "Test {}: ✗ {} cell{} wrong{}",
                    i + 1,
                    mismatches,
                    if mismatches == 1 { "" } else { "s" },
                    if !dim_match {
                        format!(
                            ", dimensions differ (predicted {}x{} vs expected {}x{})",
                            pred.len(),
                            pred.first().map_or(0, |r| r.len()),
                            exp.len(),
                            exp.first().map_or(0, |r| r.len()),
                        )
                    } else {
                        String::new()
                    },
                ));
            }
        }

        // Missing test predictions
        if predicted.len() < expected.len() {
            let missing = expected.len() - predicted.len();
            feedback_lines.push(format!(
                "✗ Missing {missing} test prediction{}",
                if missing == 1 { "" } else { "s" }
            ));
        }

        let total = expected.len().max(1);
        let score = correct as f32 / total as f32;

        FeedbackMetric::new(score, feedback_lines.join("\n"))
    }
}

/// Count cell-level mismatches between two grids.
///
/// If dimensions differ, counts all cells in the overlap region
/// plus all cells in the non-overlapping area.
pub fn count_cell_mismatches(predicted: &Grid, expected: &Grid) -> usize {
    let max_rows = predicted.len().max(expected.len());
    let max_cols = predicted
        .first()
        .map_or(0, |r| r.len())
        .max(expected.first().map_or(0, |r| r.len()));

    let mut mismatches = 0;
    for row in 0..max_rows {
        for col in 0..max_cols {
            let pred_val = predicted.get(row).and_then(|r| r.get(col));
            let exp_val = expected.get(row).and_then(|r| r.get(col));

            match (pred_val, exp_val) {
                (Some(a), Some(b)) if a != b => mismatches += 1,
                (None, Some(_)) | (Some(_), None) => mismatches += 1,
                _ => {}
            }
        }
    }
    mismatches
}

/// Extract code and output grids from the monadic result.
///
/// Looks for:
/// 1. The SUBMIT result (if the agent called SUBMIT with code and outputs)
/// 2. Falls back to parsing from conversation history
fn extract_arc_outputs(result: &str, ctx: &AgentContext) -> (String, String) {
    // Try to parse as JSON (SUBMIT result)
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(result) {
        let code = val
            .get("code")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let outputs = val
            .get("outputs")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if !outputs.is_empty() {
            return (code, outputs);
        }
    }

    // Fallback: extract last code block from assistant messages
    let code = ctx
        .history
        .messages()
        .iter()
        .rev()
        .find(|m| m.role == Role::Assistant)
        .map(|m| {
            if let Some(start) = m.content.find("```") {
                let after_fence = &m.content[start + 3..];
                let lang_end = after_fence.find('\n').unwrap_or(0);
                let code_start = start + 3 + lang_end + 1;
                if let Some(end) = m.content[code_start..].find("```") {
                    return m.content[code_start..code_start + end].trim().to_string();
                }
            }
            String::new()
        })
        .unwrap_or_default();

    (code, result.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_cell_mismatches_identical() {
        let grid = vec![vec![1, 2], vec![3, 4]];
        assert_eq!(count_cell_mismatches(&grid, &grid), 0);
    }

    #[test]
    fn test_count_cell_mismatches_one_wrong() {
        let pred = vec![vec![1, 2], vec![3, 4]];
        let exp = vec![vec![1, 2], vec![3, 5]];
        assert_eq!(count_cell_mismatches(&pred, &exp), 1);
    }

    #[test]
    fn test_count_cell_mismatches_different_dims() {
        let pred = vec![vec![1, 2]]; // 1x2
        let exp = vec![vec![1, 2], vec![3, 4]]; // 2x2
        assert_eq!(count_cell_mismatches(&pred, &exp), 2);
    }

    #[test]
    fn test_count_cell_mismatches_all_wrong() {
        let pred = vec![vec![0, 0], vec![0, 0]];
        let exp = vec![vec![1, 1], vec![1, 1]];
        assert_eq!(count_cell_mismatches(&pred, &exp), 4);
    }
}
