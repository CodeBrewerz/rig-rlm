//! Phase 9: The interaction monad — generate→execute→feedback loop.
//!
//! This is the core agent loop, implemented as a monadic computation.
//! It composes the system prompt, user task, and interaction loop
//! into a single `AgentMonad` that can be run by `AgentContext`.
//!
//! Phase 23B addition: SUBMIT detection — when code calls SUBMIT(),
//! the loop terminates with the structured result.
//!
//! From the user's perspective: you call `interaction::agent_task("your task")`
//! and get back a final answer string.

use super::action::Role;
use super::generation::{ErrorGuidance, Generation};
use super::monad::AgentMonad;
use super::prompts::PromptSystem;

/// Build the complete agent computation for a task.
///
/// This is the top-level entry point. It composes:
/// 1. System prompt insertion
/// 2. User task insertion
/// 3. The generate→execute→feedback loop
///
/// Returns an `AgentMonad` ready to be run by `AgentContext::run()`.
pub fn agent_task(task: &str) -> AgentMonad {
    agent_task_with_instruction(task, None)
}

/// Build the agent computation with an optional instruction override.
///
/// When `instruction` is `Some(text)`, the text is injected into the
/// system prompt under "## Additional Instructions". This is how
/// GEPA-optimized instructions flow into the agent at runtime.
pub fn agent_task_with_instruction(task: &str, instruction: Option<&str>) -> AgentMonad {
    agent_task_full(task, instruction, None, None, vec![])
}

/// Build the agent computation with instruction override and memory config.
///
/// When `memory` is `Some(config)`, AGENTS.md content and skill listings
/// are appended to the system prompt (Phase 4).
///
/// When `recalled_memories` is non-empty, semantically relevant memories
/// from past sessions are injected into the system prompt (Phase 28).
pub fn agent_task_full(
    task: &str,
    instruction: Option<&str>,
    memory: Option<&super::memory::MemoryConfig>,
    recalled_memories: Option<&str>,
    attachments: Vec<super::attachment::Attachment>,
) -> AgentMonad {
    let prompts = PromptSystem::default();
    let mut system_prompt = match instruction {
        Some(instr) => prompts
            .render_system_with_instruction(instr, &Default::default())
            .unwrap_or_else(|_| "You are a helpful AI agent.".to_string()),
        None => prompts
            .render_system(&Default::default())
            .unwrap_or_else(|_| "You are a helpful AI agent.".to_string()),
    };

    // Phase 4: inject memory content (AGENTS.md + skills) into system prompt
    if let Some(mem) = memory {
        let memory_block = mem.format_for_prompt();
        if !memory_block.is_empty() {
            system_prompt.push_str("\n\n");
            system_prompt.push_str(&memory_block);
        }
    }

    // Phase 28: inject recalled semantic memories from past sessions
    if let Some(memories_block) = recalled_memories {
        if !memories_block.is_empty() {
            system_prompt.push_str("\n\n");
            system_prompt.push_str(memories_block);
        }
    }

    // If attachments present, mention in system prompt
    if !attachments.is_empty() {
        let att_desc: Vec<String> = attachments.iter().map(|a| a.label()).collect();
        system_prompt.push_str(&format!(
            "\n\n## Attached Files\nThe user has attached {} file(s): {}\n\
            You can see the content of these files in the user's first message. \
            Analyze them carefully and use the information to complete the task.",
            attachments.len(),
            att_desc.join(", "),
        ));
    }

    let user_prompt = prompts
        .render_user(task)
        .unwrap_or_else(|_| task.to_string());

    // Extract text from PDF/text attachments → LoadContext actions
    // Image attachments stay in the vec for insert_with_attachments
    let mut context_loads: Vec<AgentMonad> = Vec::new();
    let mut media_attachments = Vec::new();

    for att in attachments {
        if att.is_pdf() || att.is_text() {
            // Extract text and load as searchable context
            let ctx_name = att
                .filename
                .as_deref()
                .unwrap_or("uploaded_document")
                .to_string();
            match att.extract_text() {
                Ok(text) if !text.is_empty() => {
                    tracing::info!(
                        name = ctx_name,
                        chars = text.len(),
                        "Extracted text from attachment"
                    );
                    context_loads.push(AgentMonad::load_context(&ctx_name, text));
                }
                Ok(_) => {
                    tracing::warn!(name = ctx_name, "Attachment text extraction returned empty");
                }
                Err(e) => {
                    tracing::warn!(name = ctx_name, error = %e, "Failed to extract text from attachment");
                }
            }
        } else if att.is_image() || att.is_audio() {
            media_attachments.push(att);
        }
        // Other attachment types are silently ignored
    }

    // Compose: system → [context loads] → user (with media attachments) → interaction loop
    let user_insert = if media_attachments.is_empty() {
        AgentMonad::insert(Role::User, user_prompt)
    } else {
        AgentMonad::insert_with_attachments(Role::User, user_prompt, media_attachments)
    };

    let mut chain = AgentMonad::insert(Role::System, system_prompt);
    for load in context_loads {
        chain = chain.then(load);
    }
    chain.then(user_insert).then(interaction_loop())
}

/// The core interaction loop as a monadic computation.
///
/// Loop: call LLM → parse response → execute code or return final answer.
/// This is the Rust equivalent of Agentica's `interaction_monad`.
///
/// SUBMIT detection: When execute_code returns an ActionOutput::Submitted,
/// the `into_string()` extracts the SUBMIT JSON and the output contains
/// `[submitted]` marker, which triggers termination.
fn interaction_loop() -> AgentMonad {
    // Step 1: Call the LLM
    AgentMonad::model_inference().bind(|response| {
        let generation = Generation::parse(&response);

        // ── Thinking trace: emit the LLM's reasoning text ──
        // This is the Codex-style "• I've dispatched two independent audits..."
        // pattern. The LLM's text outside code blocks is its thinking trace.
        emit_thinking_trace(&generation);

        // Case 1: Final answer — we're done
        if let Some(answer) = generation.final_answer {
            return AgentMonad::pure(answer);
        }

        // Case 2: Code to execute
        if let Some(code) = generation.code {
            return AgentMonad::execute_code(code).bind(|output| {
                // Case 2a: SUBMIT() was called — structured result
                if output.starts_with("[submitted]") {
                    let result = output
                        .strip_prefix("[submitted] ")
                        .unwrap_or(&output)
                        .to_string();
                    return AgentMonad::pure(result);
                }

                // Case 2b: Legacy return value (my_answer)
                if output.contains("[return]") {
                    let result = output
                        .lines()
                        .find(|l| l.starts_with("[return] "))
                        .map(|l| l.strip_prefix("[return] ").unwrap_or(l))
                        .unwrap_or(&output)
                        .to_string();
                    return AgentMonad::pure(result);
                }

                // Case 2c: Continue the loop — feed output back
                AgentMonad::insert(
                    Role::Execution,
                    format!("Execution result:\n{output}"),
                )
                .then(AgentMonad::compact_context())
                .then(interaction_loop())
            });
        }

        // Case 3: Shell command
        if let Some(cmd) = generation.shell_command {
            return AgentMonad::execute_code(format!(
                "import subprocess; result = subprocess.run({cmd:?}, shell=True, capture_output=True, text=True); print(result.stdout); print(result.stderr)"
            ))
            .bind(|output| {
                AgentMonad::insert(
                    Role::Execution,
                    format!("Shell output:\n{output}"),
                )
                .then(AgentMonad::compact_context())
                .then(interaction_loop())
            });
        }

        // Case 3b: Orchestrate — spawn parallel sub-agents
        if let Some(agents) = generation.orchestrate {
            return AgentMonad::orchestrate_agents(agents).bind(|output| {
                AgentMonad::insert(
                    Role::Execution,
                    format!("Orchestration result:\n{output}"),
                )
                .then(AgentMonad::compact_context())
                .then(interaction_loop())
            });
        }

        // Case 4: Unified diff patch — apply to files
        if let Some(patch) = generation.patch {
            return AgentMonad::apply_patch(patch).bind(|output| {
                AgentMonad::insert(
                    Role::Execution,
                    format!("Patch result:\n{output}"),
                )
                .then(AgentMonad::compact_context())
                .then(interaction_loop())
            });
        }

        // Case 5: No code, no final — prompt the LLM to provide code
        if response.trim().is_empty() {
            AgentMonad::insert(Role::Execution, ErrorGuidance::EMPTY_RESPONSE.to_string())
                .then(AgentMonad::compact_context())
                .then(interaction_loop())
        } else {
            AgentMonad::insert(Role::Execution, ErrorGuidance::MISSING_CODE.to_string())
                .then(AgentMonad::compact_context())
                .then(interaction_loop())
        }
    })
}

/// Emit the LLM's reasoning text as a thinking trace.
///
/// Extracts the text outside code/diff/orchestrate blocks and displays
/// it as a Codex-style bullet point. This gives visibility into the
/// model's reasoning between actions.
fn emit_thinking_trace(generation: &Generation) {
    // Extract reasoning text (text before the first code/orchestrate block)
    let raw = &generation.raw;
    let thinking = extract_thinking_text(raw);

    if !thinking.is_empty() {
        // Truncate to 200 chars for display
        let preview = if thinking.len() > 200 {
            format!("{}...", &thinking[..200])
        } else {
            thinking
        };
        eprintln!("\n• {preview}");
    }
}

/// Extract the thinking/reasoning text from an LLM response.
///
/// Returns the text before the first structured block (```repl, ```diff,
/// ```orchestrate, FINAL, RUN). This is the model's "thinking out loud".
fn extract_thinking_text(raw: &str) -> String {
    let trimmed = raw.trim();

    // Find the earliest structured block marker
    let markers = [
        "```repl", "```python", "```diff", "```patch", "```orchestrate",
        "```\n",  // generic code block
    ];
    let earliest = markers
        .iter()
        .filter_map(|m| trimmed.find(m))
        .min();

    let text = match earliest {
        Some(pos) if pos > 0 => &trimmed[..pos],
        _ => {
            // No code blocks — check for FINAL/RUN prefixes
            if trimmed.starts_with("FINAL") || trimmed.starts_with("RUN ") {
                return String::new();
            }
            // Pure text response — show first 3 lines
            return trimmed.lines().take(3).collect::<Vec<_>>().join(" ");
        }
    };

    // Clean up: remove trailing whitespace and empty lines
    text.trim().to_string()
}

/// Build a simple agent computation that just does one LLM call
/// (no code execution loop). Useful for sub-agents.
pub fn simple_query(prompt: &str) -> AgentMonad {
    AgentMonad::insert(Role::User, prompt.to_string()).then(AgentMonad::model_inference())
}
