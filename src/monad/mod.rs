//! The monadic agent system.
//!
//! Core architecture: `AgentMonad` (free monad) → `Action` (side effects)
//! → `AgentContext` (interpreter) → `LlmProvider` (LLM calls).
//!
//! **User entry point**: `interaction::agent_task("your task")`

pub mod action;
pub mod attachment;
pub mod capabilities;
pub mod context;
pub mod context_manager;
pub mod context_memory;
pub mod cost;
pub mod diff_tracker;
pub mod error;
pub mod evidence;
pub mod execution;
pub mod generation;
pub mod history;
pub mod hooks;
pub mod interaction;
pub mod memories;
pub mod memory;
pub mod monad;
pub mod normalize;
pub mod orchestrator;
pub mod otel;
pub mod parallel;
pub mod project_doc;
pub mod prompts;
pub mod provider;
pub mod recipe;
pub mod restate;
pub mod restate_helpers;
pub mod truncation;

#[cfg(test)]
mod integration_tests;
#[cfg(test)]
mod live_llm_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_modules;

// Re-export the primary types for ergonomic imports.
pub use action::{Action, ActionOutput, LogLevel, Role};
pub use capabilities::Capabilities;
pub use context::{AgentConfig, AgentContext};
pub use context_manager::{ContextManager, IsolatedContext};
pub use error::{AgentError, Result};
pub use evidence::{Evidence, EvidenceSource};
pub use execution::{ExecutionError, ExecutionResult};
pub use generation::Generation;
pub use history::{ConversationHistory, HistoryMessage};
pub use hooks::{HookEvent, HookRegistry, HookResult};
pub use interaction::{agent_task, agent_task_full, agent_task_with_instruction};
pub use memory::MemoryConfig;
pub use monad::AgentMonad;
pub use orchestrator::{Orchestrator, OrchestratorStrategy, SubAgentResult, SubAgentSpec};
pub use parallel::{ActionSafety, ParallelRuntime};
pub use prompts::PromptSystem;
pub use provider::{LlmProvider, ProviderConfig};
pub use recipe::{Recipe, RecipeResult, RecipeStep, StepKind};
