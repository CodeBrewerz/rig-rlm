//! The monadic agent system.
//!
//! Core architecture: `AgentMonad` (free monad) → `Action` (side effects)
//! → `AgentContext` (interpreter) → `LlmProvider` (LLM calls).
//!
//! **User entry point**: `interaction::agent_task("your task")`

pub mod action;
pub mod capabilities;
pub mod context;
pub mod error;
pub mod execution;
pub mod generation;
pub mod history;
pub mod interaction;
pub mod monad;
pub mod prompts;
pub mod provider;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_modules;

// Re-export the primary types for ergonomic imports.
pub use action::{Action, ActionOutput, LogLevel, Role};
pub use capabilities::Capabilities;
pub use context::{AgentConfig, AgentContext};
pub use error::{AgentError, Result};
pub use execution::{ExecutionError, ExecutionResult};
pub use generation::Generation;
pub use history::{ConversationHistory, HistoryMessage};
pub use interaction::{agent_task, agent_task_with_instruction};
pub use monad::AgentMonad;
pub use prompts::PromptSystem;
pub use provider::{LlmProvider, ProviderConfig};
