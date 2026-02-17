//! Error types for the monadic agent system.

use std::fmt;

/// All errors the agent system can produce.
#[derive(Debug)]
pub enum AgentError {
    /// LLM inference failed.
    Inference(String),
    /// Code execution failed.
    Execution(String),
    /// Template rendering failed.
    Template(String),
    /// Context variable not found.
    VariableNotFound(String),
    /// Agent exceeded max turns without producing a result.
    MaxTurnsExceeded(usize),
    /// Agent action was cancelled or interrupted.
    Cancelled,
    /// Action denied by capability restrictions (Phase 11).
    PermissionDenied(String),
    /// Code failed safety validation (Phase 12).
    SafetyViolation(String),
    /// Catch-all for unexpected errors.
    Internal(String),
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inference(msg) => write!(f, "inference error: {msg}"),
            Self::Execution(msg) => write!(f, "execution error: {msg}"),
            Self::Template(msg) => write!(f, "template error: {msg}"),
            Self::VariableNotFound(name) => write!(f, "variable not found: {name}"),
            Self::MaxTurnsExceeded(n) => write!(f, "max turns exceeded: {n}"),
            Self::Cancelled => write!(f, "agent cancelled"),
            Self::PermissionDenied(msg) => write!(f, "permission denied: {msg}"),
            Self::SafetyViolation(msg) => write!(f, "safety violation: {msg}"),
            Self::Internal(msg) => write!(f, "internal error: {msg}"),
        }
    }
}

impl std::error::Error for AgentError {}

impl From<Box<dyn std::error::Error + Send + Sync>> for AgentError {
    fn from(e: Box<dyn std::error::Error + Send + Sync>) -> Self {
        Self::Internal(e.to_string())
    }
}

impl From<String> for AgentError {
    fn from(s: String) -> Self {
        Self::Internal(s)
    }
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, AgentError>;
