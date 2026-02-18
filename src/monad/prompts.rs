//! Phase 7: Prompt templating with Tera.
//!
//! Separates prompt engineering from code. Templates are loaded from
//! embedded strings (will move to files when the template library grows).

use std::collections::HashMap;
use tera::{Context, Tera};

use super::error::{AgentError, Result};

/// Prompt rendering system using Tera templates.
pub struct PromptSystem {
    tera: Tera,
}

impl PromptSystem {
    /// Create a new prompt system with built-in templates.
    pub fn new() -> Result<Self> {
        let mut tera = Tera::default();

        // Register built-in templates
        tera.add_raw_templates(vec![
            ("system.txt", SYSTEM_TEMPLATE),
            ("user.txt", USER_TEMPLATE),
            ("error_empty.txt", ERROR_EMPTY),
            ("error_missing_code.txt", ERROR_MISSING_CODE),
            ("error_execution.txt", ERROR_EXECUTION),
        ])
        .map_err(|e| AgentError::Template(e.to_string()))?;

        Ok(Self { tera })
    }

    /// Render the system prompt.
    pub fn render_system(&self, session_info: &HashMap<String, String>) -> Result<String> {
        let mut ctx = Context::new();
        for (k, v) in session_info {
            ctx.insert(k, v);
        }
        ctx.insert("instruction", "");
        self.tera
            .render("system.txt", &ctx)
            .map_err(|e| AgentError::Template(e.to_string()))
    }

    /// Render the system prompt with a custom instruction override.
    ///
    /// When GEPA optimizes the instruction, the result is injected here
    /// so it appears in the system prompt alongside the base template.
    pub fn render_system_with_instruction(
        &self,
        instruction: &str,
        session_info: &HashMap<String, String>,
    ) -> Result<String> {
        let mut ctx = Context::new();
        for (k, v) in session_info {
            ctx.insert(k, v);
        }
        ctx.insert("instruction", instruction);
        self.tera
            .render("system.txt", &ctx)
            .map_err(|e| AgentError::Template(e.to_string()))
    }

    /// Render the initial user prompt for a task.
    pub fn render_user(&self, task: &str) -> Result<String> {
        let mut ctx = Context::new();
        ctx.insert("task", task);
        self.tera
            .render("user.txt", &ctx)
            .map_err(|e| AgentError::Template(e.to_string()))
    }

    /// Render an error guidance template.
    pub fn render_error(
        &self,
        template_name: &str,
        error_info: &HashMap<String, String>,
    ) -> Result<String> {
        let mut ctx = Context::new();
        for (k, v) in error_info {
            ctx.insert(k, v);
        }
        self.tera
            .render(template_name, &ctx)
            .map_err(|e| AgentError::Template(e.to_string()))
    }

    /// Register a custom template.
    pub fn add_template(&mut self, name: &str, content: &str) -> Result<()> {
        self.tera
            .add_raw_template(name, content)
            .map_err(|e| AgentError::Template(e.to_string()))
    }
}

impl Default for PromptSystem {
    fn default() -> Self {
        Self::new().expect("built-in templates should always parse")
    }
}

// ── Built-in templates ───────────────────────────────────────────

const SYSTEM_TEMPLATE: &str = r#"You are an expert AI agent that solves tasks by writing and executing Python code.

## How to work

1. **Think step-by-step** about what you need to do
2. **Write Python code** in a single ```repl block — it will be executed and you'll see the output
3. **Review the output** and iterate until you have the answer
4. **Return your answer** with FINAL <your answer>

## Available commands

- Write Python code in ```repl blocks — it will be executed and you'll see the output
- Use `FINAL <message>` when you have completed the task
- Use `RUN <command>` to run shell commands

## HTTP/JSON Toolkit (pre-loaded)

You have these helper functions available for making HTTP calls:

- `http_call(method, url, json_data=None, headers=None, timeout=30)` → HttpResponse
- `http_get(url)` / `http_post(url, json_data)` / `http_put(url, json_data)` / `http_delete(url)` — shorthand wrappers
- `json_extract(data, "key1", "key2", 0, ...)` — safely extract nested values (works on dicts or HttpResponses)
- `json_pretty(data)` — pretty-print JSON (works on dicts, HttpResponse, or strings)
- `fetch_all([(method, url, data), ...])` — call multiple APIs sequentially
- `assert_status(resp, expected=200)` — assert response status code

HttpResponse has: `.status_code`, `.body`, `.json()`, `.ok`, `.error`, `resp["key"]` shorthand.

Example:
```
resp = http_post("http://127.0.0.1:8080/MyWorkflow/key/run", json_data={"workflow_id": "test"})
print(resp.status_code, resp.ok)
json_pretty(resp)
value = json_extract(resp, "results", "unit", "name")
```

**IMPORTANT**: Do NOT use `requests` or `urllib` — they are not available. Use the toolkit functions above.

## Rules

- Write only ONE ```repl block per response, then wait for the output
- Always check execution output before giving a final answer
- If your code errors, fix the bug and retry
- You can use `print()` to inspect intermediate values
- Use `SUBMIT(answer="your result")` for structured final output
- **Once you see the expected output, immediately use FINAL to report the result. Do NOT re-run code that already succeeded.**
- Do NOT repeat the same code block — if execution was successful, move on to the next step or give FINAL
{% if instruction %}

## Additional Instructions

{{ instruction }}
{% endif %}
"#;

const USER_TEMPLATE: &str = r#"Please solve the following task:

{{ task }}

Think step-by-step, write code to solve it, and provide your final answer."#;

const ERROR_EMPTY: &str = r#"Your response was empty. Please provide either:
1. Python code in ```repl blocks to work on the task
2. FINAL <answer> if you have completed the task"#;

const ERROR_MISSING_CODE: &str = r#"No executable code found in your response. To execute code, wrap it in:
```repl
your_code_here
```
To give a final answer, use: FINAL <your answer>"#;

const ERROR_EXECUTION: &str = r#"Your code raised an error:
{{ error_type }}: {{ error_message }}
{% if traceback %}
Traceback:
{{ traceback }}
{% endif %}
Please fix the issue and try again."#;
