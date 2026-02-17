//! Phase 6: Generation type — parsed LLM output.
//!
//! The LLM's response is parsed into structured form: text explanation
//! + optional code block + optional final answer. This drives the
//! interaction monad's decision logic.

/// Parsed LLM generation — what the model actually produced.
#[derive(Debug, Clone)]
pub struct Generation {
    /// The full raw text from the model.
    pub raw: String,
    /// Extracted code block (if any ```repl block found).
    pub code: Option<String>,
    /// Final answer (if FINAL command found).
    pub final_answer: Option<String>,
    /// Shell command (if RUN command found).
    pub shell_command: Option<String>,
    /// Plain text explanation (everything outside special blocks).
    pub text: String,
}

impl Generation {
    /// Parse raw LLM output into a structured Generation.
    ///
    /// Recognizes:
    /// - ` ```repl ... ``` ` or ` ```python ... ``` ` code blocks
    /// - `FINAL <message>` commands
    /// - `RUN <command>` shell commands
    pub fn parse(raw: &str) -> Self {
        let trimmed = raw.trim();

        // Check for FINAL command
        if trimmed.starts_with("FINAL") || trimmed.contains("\nFINAL ") {
            let final_text = Self::extract_final(trimmed);
            return Self {
                raw: raw.to_string(),
                code: None,
                final_answer: Some(final_text),
                shell_command: None,
                text: raw.to_string(),
            };
        }

        // Check for RUN command
        if trimmed.starts_with("RUN ") {
            let cmd = trimmed
                .strip_prefix("RUN ")
                .unwrap_or("")
                .trim()
                .to_string();
            return Self {
                raw: raw.to_string(),
                code: None,
                final_answer: None,
                shell_command: Some(cmd),
                text: raw.to_string(),
            };
        }

        // Extract code blocks
        let code = Self::extract_code_block(trimmed);

        Self {
            raw: raw.to_string(),
            code,
            final_answer: None,
            shell_command: None,
            text: raw.to_string(),
        }
    }

    /// Extract code from ```repl or ```python fenced blocks.
    fn extract_code_block(text: &str) -> Option<String> {
        // Try ```repl first, then ```python, then generic ```
        for lang in &["```repl", "```python", "```py", "```"] {
            if let Some(start) = text.find(lang) {
                let code_start = start + lang.len();
                // Skip to next line after the opening fence
                let code_start = text[code_start..]
                    .find('\n')
                    .map(|i| code_start + i + 1)
                    .unwrap_or(code_start);

                if let Some(end) = text[code_start..].find("\n```") {
                    let code = text[code_start..code_start + end].trim().to_string();
                    if !code.is_empty() {
                        return Some(code);
                    }
                }
            }
        }
        None
    }

    /// Extract FINAL message text.
    fn extract_final(text: &str) -> String {
        // Find FINAL and take everything after it
        if let Some(pos) = text.find("FINAL") {
            let after = &text[pos + 5..];
            after.trim().to_string()
        } else {
            text.to_string()
        }
    }

    /// Does this generation contain executable code?
    pub fn has_code(&self) -> bool {
        self.code.is_some()
    }

    /// Is this a terminal response (FINAL answer)?
    pub fn is_final(&self) -> bool {
        self.final_answer.is_some()
    }
}

// ── Error guidance templates ─────────────────────────────────────

/// Guidance messages for common LLM errors.
/// Fed back to the model to help it self-correct.
pub struct ErrorGuidance;

impl ErrorGuidance {
    pub const EMPTY_RESPONSE: &'static str = "Your response was empty. Please provide either:\n\
         1. Python code in ```repl blocks to work on the task\n\
         2. FINAL <answer> if you have completed the task";

    pub const MISSING_CODE: &'static str =
        "No code block found in your response. To execute code, \
         wrap it in ```repl ... ``` blocks. To give a final answer, \
         use FINAL <your answer>.";

    pub const EXECUTION_ERROR: &'static str =
        "Your code raised an error. Please review the traceback above, \
         fix the issue, and try again with corrected code in ```repl blocks.";

    pub const MULTIPLE_CODE_BLOCKS: &'static str =
        "Multiple code blocks detected. Only the first code block will \
         be executed. Please combine your code into a single ```repl block.";
}
