//! The `AgentContext` — interpreter for the agent monad.
//!
//! Holds all mutable state (conversation history, variables, provider, executor)
//! and walks the monad tree, executing each `Action` and feeding results
//! to continuations until a `Pure` value is reached.
//!
//! Phase 11 additions: capability-based action guards + SpawnSubAgent handler.
//! Phase 12 additions: safety validation + output sanitization.
//! Phase 23B additions: CodeExecutor, SUBMIT(), prelude, llm_query().

use std::collections::HashMap;

use tracing::{debug, info, warn, error as tracing_error};

use super::action::{Action, ActionOutput, LogLevel, Role};
use super::capabilities::Capabilities;
use super::error::{AgentError, Result};
use super::interaction::agent_task;
use super::monad::AgentMonad;
use super::history::{ConversationHistory, HistoryMessage};
use super::provider::{LlmProvider, ProviderConfig};
use crate::safety::{ExecutionLimits, validate_code, sanitize_output};
use crate::sandbox::{CodeExecutor, Pyo3CodeExecutor, ExecutorKind, create_executor};
use crate::session::SessionConfig;

/// Configuration for the agent context.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Maximum number of generate→execute→feedback turns before giving up.
    pub max_turns: usize,
    /// Provider configuration.
    pub provider: ProviderConfig,
    /// Executor kind (Microsandbox or PyO3).
    pub executor_kind: ExecutorKind,
    /// Session configuration for the sandbox.
    pub session: SessionConfig,
    /// Capability restrictions (Phase 11).
    pub capabilities: Capabilities,
    /// Execution safety limits (Phase 12).
    pub safety: ExecutionLimits,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_turns: 25,
            provider: ProviderConfig::local("qwen/qwen3-8b"),
            executor_kind: ExecutorKind::Pyo3,
            session: SessionConfig::default(),
            capabilities: Capabilities::root(),
            safety: ExecutionLimits::permissive(),
        }
    }
}

impl AgentConfig {
    /// Create config for OpenAI models.
    pub fn openai(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            max_turns: 25,
            provider: ProviderConfig::openai(model, api_key),
            executor_kind: ExecutorKind::Pyo3,
            session: SessionConfig::default(),
            capabilities: Capabilities::root(),
            safety: ExecutionLimits::permissive(),
        }
    }

    /// Create config for any OpenAI-compatible endpoint.
    pub fn openai_compatible(
        name: impl Into<String>,
        model: impl Into<String>,
        base_url: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            max_turns: 25,
            provider: ProviderConfig::openai_compatible(name, model, base_url, api_key),
            executor_kind: ExecutorKind::Pyo3,
            session: SessionConfig::default(),
            capabilities: Capabilities::root(),
            safety: ExecutionLimits::permissive(),
        }
    }

    /// Enable sub-LLM bridging (llm_query() callable from sandbox).
    /// Automatically uses the agent's own provider config for the bridge.
    pub fn with_sub_llm(mut self) -> Self {
        self.session = self.session
            .with_sub_llm()
            .with_provider(self.provider.clone());
        self
    }

    /// Set the executor kind.
    pub fn with_executor(mut self, kind: ExecutorKind) -> Self {
        self.executor_kind = kind;
        self
    }

    /// Set a custom session config.
    pub fn with_session(mut self, session: SessionConfig) -> Self {
        self.session = session;
        self
    }

    /// Set capability restrictions (Phase 11).
    pub fn with_capabilities(mut self, caps: Capabilities) -> Self {
        self.capabilities = caps;
        self
    }

    /// Set execution safety limits (Phase 12).
    pub fn with_safety(mut self, limits: ExecutionLimits) -> Self {
        self.safety = limits;
        self
    }
}

/// The execution context for the agent monad interpreter.
///
/// Owns: conversation history, variables, LLM provider, and code executor.
/// The executor maintains persistent state across execute() calls within
/// a single agent run (Phase 23B).
///
/// Phase 11: capability-guarded action dispatch.
/// Phase 12: safety-validated code execution with output sanitization.
pub struct AgentContext {
    /// Full conversation history.
    pub history: ConversationHistory,
    /// Named variables stored by the agent.
    pub variables: HashMap<String, String>,
    /// Configuration.
    pub config: AgentConfig,
    /// LLM provider.
    provider: LlmProvider,
    /// Code executor (persistent REPL session).
    executor: Box<dyn CodeExecutor>,
    /// Turn counter (for max_turns enforcement).
    turn: usize,
    /// Whether the session has been set up.
    session_ready: bool,
    /// Unique agent ID (for lineage tracking).
    pub agent_id: String,
    /// Parent agent ID (None for root agents).
    pub parent_id: Option<String>,
    /// Execution counter (for safety limit enforcement).
    execution_count: usize,
}

impl AgentContext {
    /// Create a new context with the given config.
    ///
    /// Uses PyO3 executor by default. For microsandbox,
    /// use `AgentContext::new_async()` which awaits sandbox creation.
    pub fn new(config: AgentConfig) -> Self {
        let provider = LlmProvider::new(config.provider.clone());
        Self {
            history: ConversationHistory::new(),
            variables: HashMap::new(),
            provider,
            executor: Box::new(Pyo3CodeExecutor::new()),
            config,
            turn: 0,
            session_ready: false,
            agent_id: uuid::Uuid::new_v4().to_string(),
            parent_id: None,
            execution_count: 0,
        }
    }

    /// Create a context with async executor initialization.
    /// Required for microsandbox (which needs async startup).
    pub async fn new_async(config: AgentConfig) -> anyhow::Result<Self> {
        let provider = LlmProvider::new(config.provider.clone());
        let executor = create_executor(&config.executor_kind).await?;
        Ok(Self {
            history: ConversationHistory::new(),
            variables: HashMap::new(),
            provider,
            executor,
            config,
            turn: 0,
            session_ready: false,
            agent_id: uuid::Uuid::new_v4().to_string(),
            parent_id: None,
            execution_count: 0,
        })
    }

    /// Create a context with default config (local LLM).
    pub fn new_local() -> Self {
        Self::new(AgentConfig::default())
    }

    /// Create a context for OpenAI.
    pub fn new_openai(model: &str, api_key: &str) -> Self {
        Self::new(AgentConfig::openai(model, api_key))
    }

    /// Get the current turn count.
    pub fn turn_count(&self) -> usize {
        self.turn
    }

    /// Ensure the session is set up (SUBMIT injected, prelude run, etc.).
    async fn ensure_session(&mut self) -> Result<()> {
        if !self.session_ready {
            self.executor.setup(&self.config.session).await
                .map_err(|e| AgentError::Execution(format!("session setup: {e}")))?;
            self.session_ready = true;
        }
        Ok(())
    }

    /// Run a monadic computation to completion.
    ///
    /// Iterative interpreter loop — walks the monad tree:
    /// - `Pure(value)` → return the value
    /// - `Perform { action, next }` → execute action, pass output to next
    pub async fn run(&mut self, mut computation: AgentMonad) -> Result<String> {
        // Ensure session is ready before first execution
        self.ensure_session().await?;

        loop {
            match computation {
                AgentMonad::Pure(value) => return Ok(value),

                AgentMonad::Perform { action, next } => {
                    // Check turn limit
                    self.turn += 1;
                    if self.turn > self.config.max_turns {
                        return Err(AgentError::MaxTurnsExceeded(self.config.max_turns));
                    }

                    // Execute the action
                    let output = self.interpret_action(action).await?;

                    // Feed output to the continuation → get next step
                    computation = next(output);
                }
            }
        }
    }

    /// Interpret a single action, producing an output.
    ///
    /// Phase 11: capability check before dispatch.
    /// Phase 12: safety validation + output sanitization for code execution.
    async fn interpret_action(&mut self, action: Action) -> Result<ActionOutput> {
        // Phase 11: capability guard
        let action_name = match &action {
            Action::Insert { .. } => "Insert",
            Action::ModelInference => "ModelInference",
            Action::ExecuteCode { .. } => "ExecuteCode",
            Action::Capture { .. } => "Capture",
            Action::Retrieve { .. } => "Retrieve",
            Action::Log { .. } => "Log",
            Action::SpawnSubAgent { .. } => "SpawnSubAgent",
        };
        self.config.capabilities.check_action(action_name)
            .map_err(|reason| AgentError::PermissionDenied(reason))?;

        match action {
            Action::Insert { role, content } => {
                debug!(role = %role, len = content.len(), "inserting message");
                self.history.push(HistoryMessage {
                    role,
                    content,
                });
                Ok(ActionOutput::Unit)
            }

            Action::ModelInference => {
                debug!(model = %self.config.provider.model, "calling LLM");
                let response = self.provider.chat(&self.history).await?;
                debug!(len = response.len(), "LLM response received");

                // Auto-insert the assistant response
                self.history.push(HistoryMessage {
                    role: Role::Assistant,
                    content: response.clone(),
                });

                Ok(ActionOutput::Value(response))
            }

            Action::ExecuteCode { source } => {
                // Phase 12: safety validation
                validate_code(&source, &self.config.safety)
                    .map_err(|v| AgentError::SafetyViolation(v.to_string()))?;

                // Phase 12: execution count limit
                self.execution_count += 1;
                if self.execution_count > self.config.safety.max_total_executions {
                    return Err(AgentError::SafetyViolation(format!(
                        "execution limit reached: {}/{}",
                        self.execution_count,
                        self.config.safety.max_total_executions
                    )));
                }

                debug!(len = source.len(), exec_count = self.execution_count, "executing code in persistent REPL");
                let result = self.executor.execute(&source).await
                    .map_err(|e| AgentError::Execution(e.to_string()))?;

                // Phase 12: sanitize output
                let feedback = sanitize_output(&result.to_feedback(), &self.config.safety);
                self.history.push(HistoryMessage {
                    role: Role::Execution,
                    content: feedback.clone(),
                });

                // If SUBMIT was called, return the structured result
                if result.is_submitted() {
                    if let Some(ref return_val) = result.return_value {
                        return Ok(ActionOutput::Submitted(return_val.clone()));
                    }
                }

                Ok(ActionOutput::Value(feedback))
            }

            Action::Capture { name, value } => {
                debug!(name = %name, "capturing variable");
                self.variables.insert(name, value);
                Ok(ActionOutput::Unit)
            }

            Action::Retrieve { name } => {
                match self.variables.get(&name) {
                    Some(value) => Ok(ActionOutput::Value(value.clone())),
                    None => Err(AgentError::VariableNotFound(name)),
                }
            }

            Action::Log { level, message } => {
                match level {
                    LogLevel::Debug => debug!("{message}"),
                    LogLevel::Info => info!("{message}"),
                    LogLevel::Warn => warn!("{message}"),
                    LogLevel::Error => tracing_error!("{message}"),
                }
                Ok(ActionOutput::Unit)
            }

            // Phase 11: spawn a sub-agent with restricted capabilities
            Action::SpawnSubAgent { task, capabilities } => {
                info!(
                    parent = %self.agent_id,
                    caps = %capabilities,
                    "spawning sub-agent"
                );

                let child_config = AgentConfig {
                    max_turns: self.config.max_turns,
                    provider: self.config.provider.clone(),
                    executor_kind: self.config.executor_kind.clone(),
                    session: self.config.session.clone(),
                    capabilities,
                    safety: ExecutionLimits::standard(),
                };

                let mut child_ctx = AgentContext::new(child_config);
                child_ctx.parent_id = Some(self.agent_id.clone());

                let computation = agent_task(&task);
                // Box::pin breaks the recursive async type cycle
                let result = Box::pin(child_ctx.run(computation)).await
                    .map_err(|e| AgentError::Internal(
                        format!("sub-agent failed: {e}")
                    ))?;

                Ok(ActionOutput::Value(result))
            }
        }
    }
}
