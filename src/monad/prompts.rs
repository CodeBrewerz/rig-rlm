//! Phase 7: Prompt templating with Tera.
//!
//! All prompt templates live in `src/templates/` as `.md` files, loaded at
//! compile time via `include_str!()`. This makes them easy to edit and tweak
//! without modifying Rust code — just rebuild.

use std::collections::HashMap;
use tera::{Context, Tera};

use super::error::{AgentError, Result};

/// Prompt rendering system using Tera templates.
pub struct PromptSystem {
    tera: Tera,
}

// ── Template files (loaded at compile time) ──────────────────────

const SYSTEM_TEMPLATE: &str = include_str!("../templates/system.md");
const USER_TEMPLATE: &str = include_str!("../templates/user.md");
const ERROR_EMPTY: &str = include_str!("../templates/errors/empty.md");
const ERROR_MISSING_CODE: &str = include_str!("../templates/errors/missing_code.md");
const ERROR_EXECUTION: &str = include_str!("../templates/errors/execution.md");

/// Compaction prompt — instructs LLM to create a handoff summary.
pub const COMPACT_PROMPT: &str = include_str!("../templates/compact/prompt.md");
/// Summary prefix — gives the resuming LLM context about prior work.
pub const SUMMARY_PREFIX: &str = include_str!("../templates/compact/summary_prefix.md");

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
        ctx.insert("project_instructions", "");
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
        ctx.insert("project_instructions", "");
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

    /// Render the compaction summarization prompt.
    pub fn compact_prompt(&self) -> &'static str {
        COMPACT_PROMPT
    }

    /// Render the summary prefix for handoff context.
    pub fn summary_prefix(&self) -> &'static str {
        SUMMARY_PREFIX
    }
}

impl Default for PromptSystem {
    fn default() -> Self {
        Self::new().expect("built-in templates should always parse")
    }
}
