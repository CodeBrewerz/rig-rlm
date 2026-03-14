//! Phase 6: Generation type — parsed LLM output.
//!
//! The LLM's response is parsed into structured form: text explanation
//! + optional code block + optional final answer. This drives the
//! interaction monad's decision logic.
//!
//! Phase 30: Added `orchestrate` field for LLM-triggered multi-agent
//! spawning via ```orchestrate blocks.

/// Specification for a sub-agent parsed from an ```orchestrate block.
#[derive(Debug, Clone)]
pub struct AgentSpec {
    /// Human-readable name for this agent.
    pub name: String,
    /// The task prompt for this agent.
    pub task: String,
}

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
    /// Unified diff patch (if ```diff block found).
    pub patch: Option<String>,
    /// Orchestration specs (if ```orchestrate block found).
    pub orchestrate: Option<Vec<AgentSpec>>,
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
                patch: None,
                orchestrate: None,
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
                patch: None,
                orchestrate: None,
                text: raw.to_string(),
            };
        }

        // Extract diff/patch blocks (```diff ... ```)
        let patch = Self::extract_diff_block(trimmed);

        // Extract orchestrate blocks (```orchestrate ... ```)
        let orchestrate = Self::extract_orchestrate_block(trimmed);

        // Extract ALL code blocks and concatenate them
        let code = Self::extract_all_code_blocks(trimmed);

        Self {
            raw: raw.to_string(),
            code,
            final_answer: None,
            shell_command: None,
            patch,
            orchestrate,
            text: raw.to_string(),
        }
    }

    /// Extract ALL code blocks from ```repl, ```python, or ``` fences
    /// and concatenate them into a single script.
    ///
    /// This allows the LLM to write multiple code blocks in one response
    /// (e.g., call API → parse response → call next API) and have them
    /// all execute together in a single Python context.
    fn extract_all_code_blocks(text: &str) -> Option<String> {
        let mut blocks = Vec::new();
        let mut search_from = 0;

        while search_from < text.len() {
            let remaining = &text[search_from..];

            // Find the next code fence opening
            let mut best_match: Option<(usize, usize)> = None; // (abs_start, fence_len)
            for lang in &["```repl", "```python", "```py"] {
                if let Some(rel_pos) = remaining.find(lang) {
                    let abs_pos = search_from + rel_pos;
                    if best_match.is_none() || abs_pos < best_match.unwrap().0 {
                        best_match = Some((abs_pos, lang.len()));
                    }
                }
            }

            let (fence_start, fence_len) = match best_match {
                Some(m) => m,
                None => break,
            };

            // After the fence tag, check if there's a newline or inline content.
            // Handles both:
            //   ```repl\n  code...  (standard: newline after fence)
            //   ```repl code...     (inline: code on same line as fence)
            let after_tag = &text[fence_start + fence_len..];
            let code_start = if after_tag.starts_with('\n') {
                // Standard: code starts on next line
                fence_start + fence_len + 1
            } else if after_tag.starts_with(" ") || after_tag.starts_with("\t") {
                // Inline: code on same line after whitespace, find start of actual content
                // But we still need to include the rest of the line as code
                let inline_start = fence_start + fence_len;
                // Check if there's a newline somewhere — treat everything from
                // after the fence to the closing ``` as code
                match after_tag.find('\n') {
                    Some(_) => inline_start, // include whitespace, trim later
                    None => break,           // no newline at all, malformed
                }
            } else {
                // No whitespace or newline after tag — try next line
                match after_tag.find('\n') {
                    Some(i) => fence_start + fence_len + i + 1,
                    None => break,
                }
            };

            // Find closing fence
            let code_end = match text[code_start..].find("\n```") {
                Some(i) => code_start + i,
                None => break,
            };

            let code = text[code_start..code_end].trim();
            if !code.is_empty() {
                blocks.push(code.to_string());
            }

            // Move past the closing fence
            search_from = code_end + 4; // skip past "\n```"
        }

        if blocks.is_empty() {
            // Fallback: try generic ``` blocks (but only first, to avoid
            // matching output blocks)
            Self::extract_single_generic_block(text)
        } else {
            Some(blocks.join("\n\n"))
        }
    }

    /// Fallback: extract a single generic ``` block (no language tag).
    fn extract_single_generic_block(text: &str) -> Option<String> {
        if let Some(start) = text.find("```\n") {
            let code_start = start + 4;
            if let Some(end) = text[code_start..].find("\n```") {
                let code = text[code_start..code_start + end].trim().to_string();
                if !code.is_empty() {
                    return Some(code);
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

    /// Does this generation contain a unified diff patch?
    pub fn has_patch(&self) -> bool {
        self.patch.is_some()
    }

    /// Does this generation contain an orchestrate directive?
    pub fn has_orchestrate(&self) -> bool {
        self.orchestrate.is_some()
    }

    /// Extract a ```diff block from the response.
    fn extract_diff_block(text: &str) -> Option<String> {
        for tag in &["```diff", "```patch"] {
            if let Some(start) = text.find(tag) {
                let code_start = start + tag.len();
                // Skip to next line
                let code_start = match text[code_start..].find('\n') {
                    Some(i) => code_start + i + 1,
                    None => continue,
                };
                // Find closing fence
                if let Some(end) = text[code_start..].find("\n```") {
                    let diff = text[code_start..code_start + end].trim().to_string();
                    if !diff.is_empty() && (diff.contains("---") || diff.contains("+++")) {
                        return Some(diff);
                    }
                }
            }
        }
        None
    }

    /// Extract a ```orchestrate block from the response.
    ///
    /// Parses a simple YAML-like list of agent specs:
    /// ```orchestrate
    /// - name: "auditor-1"
    ///   task: "Deep-audit manage_obligation workflow"
    /// - name: "reviewer"
    ///   task: "Review for hidden edge cases"
    /// ```
    fn extract_orchestrate_block(text: &str) -> Option<Vec<AgentSpec>> {
        let tag = "```orchestrate";
        let start = text.find(tag)?;
        let code_start = start + tag.len();
        let code_start = code_start + text[code_start..].find('\n')? + 1;
        let end = text[code_start..].find("\n```")?;
        let block = text[code_start..code_start + end].trim();

        if block.is_empty() {
            return None;
        }

        // Parse YAML-like format: lines starting with "- name:" followed by "  task:"
        let mut specs = Vec::new();
        let mut current_name: Option<String> = None;

        for line in block.lines() {
            let trimmed = line.trim();

            if trimmed.starts_with("- name:") {
                // If we had a previous spec without task, skip it
                let name = trimmed
                    .strip_prefix("- name:")
                    .unwrap_or("")
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'')
                    .to_string();
                current_name = Some(name);
            } else if trimmed.starts_with("task:") {
                let task = trimmed
                    .strip_prefix("task:")
                    .unwrap_or("")
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'')
                    .to_string();

                if let Some(name) = current_name.take() {
                    if !name.is_empty() && !task.is_empty() {
                        specs.push(AgentSpec { name, task });
                    }
                }
            }
        }

        if specs.is_empty() {
            None
        } else {
            Some(specs)
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_detects_diff_block() {
        let input = r#"Here's the change:

```diff
--- a/hello.py
+++ b/hello.py
@@ -1,3 +1,4 @@
 def hello():
-    print("hi")
+    print("hello world")
+    return True
     pass
```

This updates the greeting."#;

        let parsed = Generation::parse(input);
        assert!(parsed.has_patch(), "should detect diff block");
        let patch = parsed.patch.unwrap();
        assert!(patch.contains("--- a/hello.py"));
        assert!(patch.contains("+++ b/hello.py"));
        assert!(patch.contains("-    print(\"hi\")"));
        assert!(patch.contains("+    print(\"hello world\")"));
    }

    #[test]
    fn parse_detects_patch_block() {
        let input = "```patch\n--- a/foo.rs\n+++ b/foo.rs\n@@ -1 +1 @@\n-old\n+new\n```";
        let parsed = Generation::parse(input);
        assert!(parsed.has_patch());
    }

    #[test]
    fn parse_no_diff_when_no_markers() {
        let input = "```diff\nsome random text without diff markers\n```";
        let parsed = Generation::parse(input);
        assert!(
            !parsed.has_patch(),
            "should not detect diff without --- or +++"
        );
    }

    #[test]
    fn parse_diff_and_code_coexist() {
        let input = r#"```diff
--- a/x.py
+++ b/x.py
@@ -1 +1 @@
-old
+new
```

```repl
print("hello")
```"#;

        let parsed = Generation::parse(input);
        assert!(parsed.has_patch(), "should have patch");
        assert!(parsed.has_code(), "should have code too");
    }

    #[test]
    fn parse_final_takes_priority() {
        let parsed = Generation::parse("FINAL the answer is 42");
        assert!(parsed.is_final());
        assert!(!parsed.has_patch());
        assert!(!parsed.has_code());
    }

    #[test]
    fn parse_orchestrate_block() {
        let input = r#"I'll spawn two auditors to review independently.

```orchestrate
- name: "auditor-1"
  task: "Deep-audit the manage_obligation workflow"
- name: "edge-reviewer"
  task: "Review for hidden edge cases"
```
"#;
        let parsed = Generation::parse(input);
        assert!(parsed.has_orchestrate());
        let specs = parsed.orchestrate.unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].name, "auditor-1");
        assert_eq!(specs[1].name, "edge-reviewer");
        assert!(specs[0].task.contains("manage_obligation"));
    }

    #[test]
    fn parse_no_orchestrate_when_empty() {
        let input = "Just some text, no orchestrate block";
        let parsed = Generation::parse(input);
        assert!(!parsed.has_orchestrate());
    }
}
