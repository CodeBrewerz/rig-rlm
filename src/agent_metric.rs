//! Phase 15 + 16: Agent Metric — scoring agent output with execution feedback.
//!
//! Implements DSRs `FeedbackEvaluator` for `AgentModule`.
//! Scores are based on:
//! 1. Re-executing the predicted code in a sandbox (real verification)
//! 2. Answer comparison (exact/partial match)
//! 3. Code quality heuristics
//!
//! The textual feedback drives GEPA's evolutionary search — the richer
//! and more actionable the feedback, the better GEPA performs.

use crate::agent_module::AgentModule;
use crate::sandbox::CodeExecutor;
use crate::sandbox::Pyo3CodeExecutor;
use dspy_rs::*;

/// Implements FeedbackEvaluator for AgentModule so GEPA can optimize it.
///
/// This provides rich feedback (not just a score) to guide the
/// evolutionary search toward better instructions.
impl FeedbackEvaluator for AgentModule {
    async fn feedback_metric(&self, example: &Example, prediction: &Prediction) -> FeedbackMetric {
        let mut feedback_lines = Vec::new();
        let mut total_score: f32 = 0.0;
        let mut weight_sum: f32 = 0.0;

        // Extract predicted values
        let predicted_answer = prediction
            .get("answer", None)
            .as_str()
            .unwrap_or("")
            .to_string();
        let predicted_code = prediction
            .get("code", None)
            .as_str()
            .unwrap_or("")
            .to_string();

        // 1. Check if we got an answer at all
        if predicted_answer.trim().is_empty() {
            feedback_lines.push("✗ No answer produced".to_string());
            return FeedbackMetric::new(0.0, feedback_lines.join("\n"));
        }

        // 2. Re-execute code in a fresh sandbox (Phase 16)
        if !predicted_code.is_empty() {
            let mut executor = Pyo3CodeExecutor::new();
            match executor.execute(&predicted_code).await {
                Ok(result) => {
                    if result.is_error() {
                        let err_msg = result
                            .exception
                            .as_ref()
                            .map(|e| e.message.as_str())
                            .or(result.stderr.lines().last())
                            .unwrap_or("unknown error");
                        feedback_lines.push(format!("✗ Code execution failed: {}", err_msg));
                        total_score += 0.0;
                    } else if result.is_submitted() {
                        feedback_lines.push("✓ Code executed and called SUBMIT()".to_string());
                        total_score += 0.3;
                    } else {
                        feedback_lines.push(format!(
                            "✓ Code executed successfully (output: {} chars)",
                            result.stdout.len()
                        ));
                        total_score += 0.3;
                    }
                }
                Err(e) => {
                    feedback_lines.push(format!("✗ Sandbox error: {e}"));
                    total_score += 0.0;
                }
            }
            weight_sum += 0.3;
        } else {
            feedback_lines.push("✗ No code produced".to_string());
            weight_sum += 0.3;
        }

        // 3. Answer comparison (exact match or fuzzy)
        let expected = example
            .get("answer", None)
            .as_str()
            .unwrap_or("")
            .to_string();

        let answer_score = if predicted_answer.trim() == expected.trim() {
            feedback_lines.push("✓ Exact answer match".to_string());
            1.0
        } else if predicted_answer
            .to_lowercase()
            .contains(&expected.to_lowercase())
        {
            feedback_lines.push(format!(
                "◐ Partial match: expected '{}', got '{}'",
                expected, predicted_answer
            ));
            0.5
        } else {
            feedback_lines.push(format!(
                "✗ Wrong answer: expected '{}', got '{}'",
                expected, predicted_answer
            ));
            0.0
        };
        total_score += answer_score * 0.5;
        weight_sum += 0.5;

        // 4. Code structure quality
        if !predicted_code.is_empty() {
            let has_functions = predicted_code.contains("def ");
            let has_imports = predicted_code.contains("import ");
            let line_count = predicted_code.lines().count();

            if has_functions {
                feedback_lines.push("✓ Well-structured: has function definitions".to_string());
                total_score += 0.2;
            } else if has_imports && line_count > 3 {
                feedback_lines.push("◐ Code has imports but no functions".to_string());
                total_score += 0.1;
            } else {
                feedback_lines.push("◐ Code is minimal (no functions)".to_string());
                total_score += 0.05;
            }
            weight_sum += 0.2;
        }

        let final_score = if weight_sum > 0.0 {
            total_score / weight_sum
        } else {
            0.0
        };

        FeedbackMetric::new(final_score, feedback_lines.join("\n"))
    }
}
