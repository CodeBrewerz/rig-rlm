//! Phase 5: Structured execution result types.
//!
//! Rich feedback from code execution — stdout, stderr, return value,
//! exceptions with traceback. This is what makes the LLM able to
//! self-correct: it sees *exactly* what went wrong.

use serde::{Deserialize, Serialize};

/// Structured result from code execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Standard output captured during execution.
    pub stdout: String,
    /// Standard error output.
    pub stderr: String,
    /// Return value (if the code assigned to `my_answer`).
    pub return_value: Option<String>,
    /// Exception info (if execution raised an error).
    pub exception: Option<ExecutionError>,
    /// Whether the code produced a final result.
    pub has_result: bool,
    /// Whether this result came from SUBMIT() (not a regular return).
    pub submitted: bool,
}

/// Structured Python exception.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionError {
    pub name: String,
    pub message: String,
    pub traceback: Option<String>,
}

impl ExecutionResult {
    /// Create a successful result with stdout output.
    pub fn success(stdout: impl Into<String>) -> Self {
        Self {
            stdout: stdout.into(),
            stderr: String::new(),
            return_value: None,
            exception: None,
            has_result: false,
            submitted: false,
        }
    }

    /// Create a result with a final return value.
    pub fn with_return(stdout: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            stdout: stdout.into(),
            stderr: String::new(),
            return_value: Some(value.into()),
            exception: None,
            has_result: true,
            submitted: false,
        }
    }

    /// Create an error result from an exception.
    pub fn error(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            stdout: String::new(),
            stderr: String::new(),
            return_value: None,
            exception: Some(ExecutionError {
                name: name.into(),
                message: message.into(),
                traceback: None,
            }),
            has_result: false,
            submitted: false,
        }
    }

    /// Create a result from SUBMIT() — structured final output.
    ///
    /// The return_value contains the JSON-serialized SUBMIT fields.
    /// has_result is true, indicating the REPL loop should terminate.
    pub fn submitted(stdout: impl Into<String>, result_json: impl Into<String>) -> Self {
        Self {
            stdout: stdout.into(),
            stderr: String::new(),
            return_value: Some(result_json.into()),
            exception: None,
            has_result: true,
            submitted: true,
        }
    }

    /// Format as feedback for the LLM — this is what gets inserted
    /// into conversation history as an Execution role message.
    pub fn to_feedback(&self) -> String {
        let mut parts = Vec::new();

        // If this is a SUBMIT result, format it clearly
        if self.is_submitted() {
            if let Some(ref val) = self.return_value {
                parts.push(format!("[submitted] {val}"));
            }
            return parts.join("\n");
        }

        if !self.stdout.is_empty() {
            // Strip SUBMIT markers from stdout if present (cleanup)
            let clean_stdout = self.stdout.replace("__SUBMIT__", "").trim().to_string();
            if !clean_stdout.is_empty() {
                parts.push(format!("[stdout]\n{clean_stdout}"));
            }
        }
        if !self.stderr.is_empty() {
            parts.push(format!("[stderr]\n{}", self.stderr));
        }
        if let Some(ref val) = self.return_value {
            parts.push(format!("[return] {val}"));
        }
        if let Some(ref err) = self.exception {
            let mut error_msg = format!("[error] {}: {}", err.name, err.message);
            if let Some(ref tb) = err.traceback {
                error_msg.push_str(&format!("\n[traceback]\n{tb}"));
            }
            parts.push(error_msg);
        }

        if parts.is_empty() {
            "[no output]".to_string()
        } else {
            parts.join("\n\n")
        }
    }

    /// Did execution fail?
    pub fn is_error(&self) -> bool {
        self.exception.is_some()
    }

    /// Was this a SUBMIT result (structured final output)?
    pub fn is_submitted(&self) -> bool {
        self.submitted && self.return_value.is_some() && self.exception.is_none()
    }
}
