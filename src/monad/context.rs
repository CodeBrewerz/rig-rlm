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

use tracing::{debug, error as tracing_error, info, warn};

use super::action::{Action, ActionOutput, LogLevel, Role};
use super::capabilities::Capabilities;
use super::context_manager::ContextManager;
use super::error::{AgentError, Result};
use super::evidence::Evidence;
use super::history::{ConversationHistory, HistoryMessage};
use super::interaction::agent_task;
use super::memory::MemoryConfig;
use super::monad::AgentMonad;
use super::provider::{LlmProvider, ProviderConfig};
use crate::safety::{ExecutionLimits, sanitize_output, validate_code};
use crate::sandbox::{CodeExecutor, ExecutorKind, Pyo3CodeExecutor, create_executor};
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
    /// Memory/skills configuration (Phase 4).
    pub memory: MemoryConfig,
    /// Maximum cost budget in USD (Tier 1.2). None = unlimited.
    pub max_cost_usd: Option<f64>,
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
            memory: MemoryConfig::default(),
            max_cost_usd: None,
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
            memory: MemoryConfig::default(),
            max_cost_usd: None,
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
            memory: MemoryConfig::default(),
            max_cost_usd: None,
        }
    }

    /// Enable sub-LLM bridging (llm_query() callable from sandbox).
    /// Automatically uses the agent's own provider config for the bridge.
    pub fn with_sub_llm(mut self) -> Self {
        self.session = self
            .session
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

    /// Set memory/skills configuration (Phase 4).
    pub fn with_memory(mut self, memory: MemoryConfig) -> Self {
        self.memory = memory;
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
    pub turn: usize,
    /// Whether the session has been set up.
    session_ready: bool,
    /// Unique agent ID (for lineage tracking).
    pub agent_id: String,
    /// Parent agent ID (None for root agents).
    pub parent_id: Option<String>,
    /// Execution counter (for safety limit enforcement).
    execution_count: usize,
    /// Optional shared sandbox pool for multi-agent execution.
    pub sandbox_pool: Option<std::sync::Arc<crate::sandbox::SandboxPool>>,
    /// Evidence trail — auto-recorded during action interpretation (Phase 1).
    pub evidence: Vec<Evidence>,
    /// Isolated context manager for data analysis (Phase 2/3).
    pub context_manager: ContextManager,
    /// OTEL/LangFuse tracing context (session, user, tags, metadata).
    pub trace_ctx: super::otel::TraceContext,
    /// Cost tracker for budget enforcement (Tier 1.2).
    pub cost_tracker: super::cost::CostTracker,
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
            sandbox_pool: None,
            evidence: Vec::new(),
            context_manager: ContextManager::new(),
            trace_ctx: super::otel::TraceContext::new(),
            cost_tracker: super::cost::CostTracker::new(),
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
            sandbox_pool: None,
            evidence: Vec::new(),
            context_manager: ContextManager::new(),
            trace_ctx: super::otel::TraceContext::new(),
            cost_tracker: super::cost::CostTracker::new(),
        })
    }

    /// Create a context with a pre-built executor.
    ///
    /// Used for pool-backed multi-agent execution: the caller checks out
    /// an executor from the `SandboxPool`, passes it here, runs the agent,
    /// then returns the executor to the pool.
    pub fn new_with_executor(
        config: AgentConfig,
        executor: Box<dyn CodeExecutor>,
        pool: Option<std::sync::Arc<crate::sandbox::SandboxPool>>,
    ) -> Self {
        let provider = LlmProvider::new(config.provider.clone());
        Self {
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
            sandbox_pool: pool,
            evidence: Vec::new(),
            context_manager: ContextManager::new(),
            trace_ctx: super::otel::TraceContext::new(),
            cost_tracker: super::cost::CostTracker::new(),
        }
    }

    /// Create a context with default config (local LLM).
    pub fn new_local() -> Self {
        Self::new(AgentConfig::default())
    }

    /// Create a context for OpenAI.
    pub fn new_openai(model: &str, api_key: &str) -> Self {
        Self::new(AgentConfig::openai(model, api_key))
    }

    /// Shutdown the executor and release all resources.
    ///
    /// MUST be called after agent execution completes to clean up
    /// microsandbox VMs. No-op for PyO3 executor.
    pub async fn shutdown(&mut self) -> anyhow::Result<()> {
        self.executor.shutdown().await
    }

    /// Get the current turn count.
    pub fn turn_count(&self) -> usize {
        self.turn
    }

    /// Get the evidence trail.
    pub fn evidence(&self) -> &[Evidence] {
        &self.evidence
    }

    /// Get a human-readable summary of all collected evidence.
    pub fn evidence_summary(&self) -> String {
        super::evidence::summarize_evidence(&self.evidence)
    }

    /// Ensure the session is set up (SUBMIT injected, prelude run, etc.).
    async fn ensure_session(&mut self) -> Result<()> {
        if !self.session_ready {
            self.executor
                .setup(&self.config.session)
                .await
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
    ///
    /// Returns a boxed future because PlanRecipe can call run_recipe→run→interpret_action
    /// (recursive async requires boxing).
    fn interpret_action(
        &mut self,
        action: Action,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ActionOutput>> + '_>> {
        Box::pin(async move {
            // Phase 11: capability guard
            let action_name = match &action {
                Action::Insert { .. } => "Insert",
                Action::ModelInference => "ModelInference",
                Action::ExecuteCode { .. } => "ExecuteCode",
                Action::Capture { .. } => "Capture",
                Action::Retrieve { .. } => "Retrieve",
                Action::Log { .. } => "Log",
                Action::SpawnSubAgent { .. } => "SpawnSubAgent",
                Action::LoadContext { .. } => "LoadContext",
                Action::SearchContext { .. } => "SearchContext",
                Action::PeekContext { .. } => "PeekContext",
                Action::ListContexts => "ListContexts",
                Action::Think { .. } => "Think",
                Action::EvaluateProgress { .. } => "EvaluateProgress",
                Action::PlanRecipe { .. } => "PlanRecipe",
            };
            self.config
                .capabilities
                .check_action(action_name)
                .map_err(|reason| AgentError::PermissionDenied(reason))?;

            match action {
                Action::Insert { role, content } => {
                    debug!(role = %role, len = content.len(), "inserting message");
                    self.history.push(HistoryMessage { role, content });
                    Ok(ActionOutput::Unit)
                }

                Action::ModelInference => {
                    debug!(model = %self.config.provider.model, "calling LLM");

                    // Budget check before LLM call
                    if let Err((spent, limit)) =
                        self.cost_tracker.check_budget(self.config.max_cost_usd)
                    {
                        return Err(AgentError::BudgetExceeded { spent, limit });
                    }

                    let (response, usage) =
                        self.provider.chat(&self.history, &self.trace_ctx).await?;
                    debug!(len = response.len(), "LLM response received");

                    // Record cost (Tier 1.2)
                    let input_tokens = usage.input_tokens.unwrap_or(0) as u64;
                    let output_tokens = usage.output_tokens.unwrap_or(0) as u64;
                    let call_cost = self.cost_tracker.record(
                        &self.config.provider.model,
                        input_tokens,
                        output_tokens,
                    );
                    if call_cost > 0.0 {
                        eprintln!(
                            "💰 [cost] ${:.6} this call, ${:.6} cumulative",
                            call_cost, self.cost_tracker.total_cost_usd,
                        );
                    }

                    // Record evidence (Phase 1)
                    self.evidence.push(Evidence::from_inference(&response));

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
                            self.execution_count, self.config.safety.max_total_executions
                        )));
                    }

                    debug!(
                        len = source.len(),
                        exec_count = self.execution_count,
                        "executing code in persistent REPL"
                    );
                    let result = self
                        .executor
                        .execute(&source)
                        .await
                        .map_err(|e| AgentError::Execution(e.to_string()))?;

                    // Phase 12: sanitize output
                    let mut feedback = sanitize_output(&result.to_feedback(), &self.config.safety);

                    // Phase 2: auto-load large results into isolated context
                    let threshold = self.config.safety.auto_load_threshold;
                    if threshold > 0 && feedback.len() > threshold {
                        let ctx_id = format!("auto_exec_{}", self.turn);
                        let meta = self.context_manager.load(&ctx_id, &feedback);
                        let preview_len = 600.min(feedback.len());
                        let preview = &feedback[..preview_len];
                        info!(
                            ctx_id = %ctx_id,
                            size = meta.size_bytes,
                            lines = meta.line_count,
                            "auto-loaded large output into context"
                        );
                        feedback = format!(
                            "{preview}\n...[Full output ({} chars, {} lines) in context '{ctx_id}'. \
                         Use search_context/peek_context to explore.]",
                            meta.size_bytes, meta.line_count
                        );
                    }

                    // Record evidence (Phase 1)
                    self.evidence.push(Evidence::from_code_exec(
                        &feedback,
                        if result.is_error() { Some(1) } else { Some(0) },
                    ));

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

                Action::Retrieve { name } => match self.variables.get(&name) {
                    Some(value) => Ok(ActionOutput::Value(value.clone())),
                    None => Err(AgentError::VariableNotFound(name)),
                },

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
                        memory: self.config.memory.clone(),
                        max_cost_usd: self.config.max_cost_usd,
                    };

                    // Use pool if available, otherwise create executor matching parent's kind
                    let mut child_ctx = match &self.sandbox_pool {
                        Some(pool) => {
                            // Checkout a sandbox from the shared pool
                            match pool.checkout().await {
                                Ok(executor) => {
                                    let mut ctx = AgentContext::new_with_executor(
                                        child_config,
                                        Box::new(executor),
                                        Some(pool.clone()),
                                    );
                                    ctx.parent_id = Some(self.agent_id.clone());
                                    ctx
                                }
                                Err(e) => {
                                    warn!(
                                        "Pool checkout failed, falling back to create_executor: {e}"
                                    );
                                    let executor = create_executor(&child_config.executor_kind)
                                        .await
                                        .map_err(|e| {
                                            AgentError::Internal(format!(
                                                "executor creation failed: {e}"
                                            ))
                                        })?;
                                    let mut ctx = AgentContext::new_with_executor(
                                        child_config,
                                        executor,
                                        None,
                                    );
                                    ctx.parent_id = Some(self.agent_id.clone());
                                    ctx
                                }
                            }
                        }
                        None => {
                            // No pool — create executor based on config
                            let executor = create_executor(&child_config.executor_kind)
                                .await
                                .map_err(|e| {
                                    AgentError::Internal(format!("executor creation failed: {e}"))
                                })?;
                            let mut ctx =
                                AgentContext::new_with_executor(child_config, executor, None);
                            ctx.parent_id = Some(self.agent_id.clone());
                            ctx
                        }
                    };

                    let computation = agent_task(&task);
                    let sub_start = std::time::Instant::now();
                    // Box::pin breaks the recursive async type cycle
                    let sub_result = Box::pin(child_ctx.run(computation)).await;
                    let sub_duration = sub_start.elapsed();

                    // Record OTEL sub-agent span
                    let sub_success = sub_result.is_ok();
                    super::otel::record_subagent_span(
                        &self.agent_id,
                        &task,
                        sub_duration,
                        sub_success,
                        &self.trace_ctx,
                    );

                    let result = sub_result
                        .map_err(|e| AgentError::Internal(format!("sub-agent failed: {e}")))?;

                    // Record evidence (Phase 1)
                    self.evidence.push(Evidence::from_sub_agent(&result));

                    // Shutdown the child executor (stops VM if microsandbox)
                    if let Err(e) = child_ctx.shutdown().await {
                        warn!("Failed to shutdown sub-agent executor: {e}");
                    }

                    Ok(ActionOutput::Value(result))
                }

                // ─── Phase 3: Context operations ─────────────────────────────
                Action::LoadContext { id, content } => {
                    debug!(id = %id, len = content.len(), "loading content into context");
                    let meta = self.context_manager.load(&id, &content);
                    let summary = format!(
                        "Loaded context '{id}': {} lines, {} bytes, format={}",
                        meta.line_count, meta.size_bytes, meta.format
                    );
                    info!(%summary);
                    Ok(ActionOutput::Value(summary))
                }

                Action::SearchContext { id, pattern } => {
                    debug!(id = %id, pattern = %pattern, "searching context");
                    let results = self.context_manager.search(&id, &pattern);
                    let formatted = super::context_manager::format_search_results(&results);

                    // Record evidence for search
                    self.evidence
                        .push(Evidence::from_search(&id, &pattern, &formatted));

                    Ok(ActionOutput::Value(formatted))
                }

                Action::PeekContext { id, start, end } => {
                    debug!(id = %id, start, end, "peeking context");
                    match self.context_manager.peek(&id, start, end) {
                        Some(content) => Ok(ActionOutput::Value(content)),
                        None => Err(AgentError::Internal(format!("context '{id}' not found"))),
                    }
                }

                Action::ListContexts => {
                    let contexts = self.context_manager.list();
                    if contexts.is_empty() {
                        Ok(ActionOutput::Value("No contexts loaded.".to_string()))
                    } else {
                        let mut lines = vec![format!("{} context(s) loaded:", contexts.len())];
                        for (id, meta) in &contexts {
                            lines.push(format!(
                                "  {id}: {} lines, {} bytes, format={}",
                                meta.line_count, meta.size_bytes, meta.format
                            ));
                        }
                        Ok(ActionOutput::Value(lines.join("\n")))
                    }
                }

                // ─── Phase 7: Reasoning tools ──────────────────────────────
                Action::Think { reasoning } => {
                    debug!(len = reasoning.len(), "agent thinking");
                    // Record in evidence trail but do NOT insert into history
                    self.evidence.push(Evidence::from_think(&reasoning));
                    Ok(ActionOutput::Unit)
                }

                Action::EvaluateProgress {
                    confidence,
                    remaining,
                } => {
                    debug!(%confidence, %remaining, "evaluating progress");
                    self.evidence
                        .push(Evidence::from_evaluate_progress(confidence, &remaining));
                    let summary = format!(
                        "Progress: confidence={:.0}%, remaining: {}",
                        confidence * 100.0,
                        remaining
                    );
                    Ok(ActionOutput::Value(summary))
                }

                // ─── Phase 8: Recipe execution ─────────────────────────────
                Action::PlanRecipe { recipe_yaml } => {
                    info!(len = recipe_yaml.len(), "agent planning recipe pipeline");

                    // Parse and validate
                    let recipe = super::recipe::Recipe::from_yaml(&recipe_yaml)
                        .map_err(|e| AgentError::Internal(format!("invalid recipe YAML: {e}")))?;
                    recipe
                        .validate()
                        .map_err(|e| AgentError::Internal(format!("recipe validation: {e}")))?;

                    let estimate = recipe.estimate_cost();
                    info!(
                        name = %recipe.name,
                        steps = estimate.total_steps,
                        est_calls = estimate.estimated_llm_calls,
                        "executing agent-planned recipe"
                    );

                    // Record intent in evidence
                    self.evidence.push(Evidence::from_think(&format!(
                        "Planning recipe '{}': {} steps, ~{} LLM calls",
                        recipe.name, estimate.total_steps, estimate.estimated_llm_calls
                    )));

                    // Save current history — run_recipe resets it per step
                    let saved_history =
                        std::mem::replace(&mut self.history, ConversationHistory::new());
                    let saved_turn = self.turn;

                    let result = self.run_recipe(recipe).await;

                    // Restore original history + append recipe summary
                    self.history = saved_history;
                    self.turn = saved_turn;

                    match result {
                        Ok(recipe_result) => {
                            // Build summary string
                            let mut lines = vec![format!(
                                "Recipe '{}' completed: {} steps, {} total turns, {:.1}s",
                                recipe_result.recipe_name,
                                recipe_result.steps.len(),
                                recipe_result.total_turns,
                                recipe_result.elapsed.as_secs_f64()
                            )];
                            for (id, step) in &recipe_result.steps {
                                let status = match &step.status {
                                    super::recipe::StepStatus::Completed => "✅",
                                    super::recipe::StepStatus::Failed(_) => "❌",
                                    super::recipe::StepStatus::Skipped => "⏭️",
                                };
                                lines.push(format!("  {status} {id}: {} turns", step.turns));
                                if !step.output.is_empty() {
                                    let preview_len = 200.min(step.output.len());
                                    lines.push(format!("     → {}", &step.output[..preview_len]));
                                }
                            }
                            Ok(ActionOutput::Value(lines.join("\n")))
                        }
                        Err(e) => Err(e),
                    }
                }
            }
        }) // close async move + Box::pin
    }

    /// Phase 5: Maybe compact history if token estimate exceeds threshold.
    ///
    /// Called from the run loop. When history grows large, older messages
    /// are truncated and a summary is prepended.
    pub fn maybe_compact(&mut self, max_tokens: usize) {
        let estimated = self.history.estimate_tokens();
        if estimated <= max_tokens {
            return;
        }
        // Keep the last 6 messages + system prompt
        let keep_recent = 6;
        // First, truncate old long messages
        self.history.truncate_old_content(keep_recent, 500);

        // If still over budget, split off older messages
        let still_over = self.history.estimate_tokens() > max_tokens;
        if still_over {
            let old = self.history.split_at(keep_recent);
            if !old.is_empty() {
                // Build a brief summary of removed messages
                let summary = format!(
                    "[{} older messages removed for context compaction. \
                     Key topics: {}]",
                    old.len(),
                    old.iter()
                        .filter(|m| m.role == Role::Assistant)
                        .take(3)
                        .map(|m| {
                            let preview_len = 80.min(m.content.len());
                            m.content[..preview_len].to_string()
                        })
                        .collect::<Vec<_>>()
                        .join("; ")
                );
                self.history.prepend_summary(summary);
                info!(
                    removed = old.len(),
                    new_size = self.history.len(),
                    "compacted conversation history"
                );
            }
        }
    }

    /// Phase 8: Run a recipe (multi-step pipeline).
    ///
    /// Validates the recipe, computes execution order, then runs each step
    /// sequentially. Each step gets a fresh conversation history. Step
    /// outputs chain into subsequent steps via `{{step_id.output}}` templates.
    pub async fn run_recipe(
        &mut self,
        recipe: super::recipe::Recipe,
    ) -> std::result::Result<super::recipe::RecipeResult, AgentError> {
        use super::recipe::{Recipe, RecipeResult, StepResult, StepStatus};
        use indexmap::IndexMap;

        recipe
            .validate()
            .map_err(|e| AgentError::Internal(e.to_string()))?;

        let order = recipe
            .execution_order()
            .map_err(|e| AgentError::Internal(e.to_string()))?;

        let estimate = recipe.estimate_cost();
        info!(
            name = %recipe.name,
            steps = estimate.total_steps,
            est_calls = estimate.estimated_llm_calls,
            "starting recipe execution"
        );
        eprintln!(
            "🍳 [recipe] starting '{}' — {} steps",
            recipe.name, estimate.total_steps
        );

        let mut outputs: IndexMap<String, StepResult> = IndexMap::new();
        let mut total_turns = 0;
        let recipe_start = std::time::Instant::now();

        for step_id in &order {
            let step = recipe
                .get_step(step_id)
                .ok_or_else(|| AgentError::Internal(format!("step '{step_id}' not found")))?
                .clone();

            // Check if any dependency failed → skip
            let dep_failed = step.depends_on.iter().any(|dep| {
                outputs
                    .get(dep)
                    .map(|r| !matches!(r.status, StepStatus::Completed))
                    .unwrap_or(false)
            });
            if dep_failed {
                warn!(step = %step_id, "skipping step — dependency failed");
                eprintln!("⏭️  [recipe] skipping '{step_id}' — dependency failed");
                outputs.insert(step_id.clone(), StepResult::skipped());
                continue;
            }

            // Resolve templates
            let resolved_task = Recipe::resolve_task(&step, &outputs);
            info!(step = %step_id, kind = ?step.kind, "executing recipe step");
            eprintln!(
                "▶️  [recipe] step '{step_id}' ({:?}) starting...",
                step.kind
            );

            // Reset context for this step
            self.history = ConversationHistory::new();
            self.turn = 0;

            // Override max_turns if step specifies one
            let original_max_turns = self.config.max_turns;
            if let Some(mt) = step.max_turns {
                self.config.max_turns = mt;
            }

            let step_start = std::time::Instant::now();
            let cost_before = self.cost_tracker.total_cost_usd;
            let program = super::interaction::agent_task_full(
                &resolved_task,
                None,
                Some(&self.config.memory),
            );

            let step_result = match self.run(program).await {
                Ok(output) => {
                    let turns = self.turn;
                    total_turns += turns;
                    info!(step = %step_id, turns, "step completed");
                    eprintln!(
                        "✅  [recipe] step '{step_id}' completed — {turns} turns, {:.1}s",
                        step_start.elapsed().as_secs_f64()
                    );
                    let step_cost = self.cost_tracker.total_cost_usd - cost_before;
                    StepResult::completed(output, turns, step_start.elapsed(), step_cost)
                }
                Err(e) => {
                    let turns = self.turn;
                    total_turns += turns;
                    warn!(step = %step_id, error = %e, "step failed");
                    eprintln!(
                        "❌  [recipe] step '{step_id}' failed — {turns} turns, {:.1}s: {e}",
                        step_start.elapsed().as_secs_f64()
                    );
                    let step_cost = self.cost_tracker.total_cost_usd - cost_before;
                    StepResult::failed(e.to_string(), turns, step_start.elapsed(), step_cost)
                }
            };

            // Restore max_turns
            self.config.max_turns = original_max_turns;

            outputs.insert(step_id.clone(), step_result);
        }

        let result = RecipeResult {
            recipe_name: recipe.name,
            steps: outputs,
            total_turns,
            elapsed: recipe_start.elapsed(),
            total_cost_usd: self.cost_tracker.total_cost_usd,
        };

        // Record OTEL recipe span
        let all_success = result
            .steps
            .values()
            .all(|s| matches!(s.status, StepStatus::Completed));
        super::otel::record_recipe_span(
            &result.recipe_name,
            result.steps.len(),
            result.total_turns,
            result.elapsed,
            all_success,
            &self.trace_ctx,
        );

        Ok(result)
    }
}
