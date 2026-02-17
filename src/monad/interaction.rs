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
    let prompts = PromptSystem::default();
    let system_prompt = prompts
        .render_system(&Default::default())
        .unwrap_or_else(|_| "You are a helpful AI agent.".to_string());
    let user_prompt = prompts
        .render_user(task)
        .unwrap_or_else(|_| task.to_string());

    // Compose: system → user → interaction loop
    AgentMonad::insert(Role::System, system_prompt)
        .then(AgentMonad::insert(Role::User, user_prompt))
        .then(interaction_loop())
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
                .then(interaction_loop())
            });
        }

        // Case 4: No code, no final — prompt the LLM to provide code
        if response.trim().is_empty() {
            AgentMonad::insert(Role::Execution, ErrorGuidance::EMPTY_RESPONSE.to_string())
                .then(interaction_loop())
        } else {
            AgentMonad::insert(Role::Execution, ErrorGuidance::MISSING_CODE.to_string())
                .then(interaction_loop())
        }
    })
}

/// Build a simple agent computation that just does one LLM call
/// (no code execution loop). Useful for sub-agents.
pub fn simple_query(prompt: &str) -> AgentMonad {
    AgentMonad::insert(Role::User, prompt.to_string()).then(AgentMonad::model_inference())
}
