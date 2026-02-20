//! Restate durable agent workflow.
//!
//! Exposes the RLM agent as a Restate workflow so that agent execution survives
//! process crashes. Each agent task is wrapped in `ctx.run()` making it a
//! durable side-effect that Restate will replay on restart.

use restate_sdk::errors::HandlerError;
use restate_sdk::prelude::*;
use serde::{Deserialize, Serialize};

/// Request payload for starting a durable agent task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTaskRequest {
    /// The task description / user prompt.
    pub task: String,
    /// LLM model to use (e.g. "arcee-ai/trinity-large-preview:free").
    pub model: Option<String>,
    /// Provider base URL (e.g. "https://openrouter.ai/api/v1").
    pub base_url: Option<String>,
    /// API key for the provider.
    pub api_key: Option<String>,
    /// Provider name (e.g. "openrouter").
    pub provider_name: Option<String>,
    /// Maximum turns before giving up.
    pub max_turns: Option<usize>,
    /// Maximum cost budget in USD.
    pub max_cost_usd: Option<f64>,
    /// System preamble / instructions.
    pub preamble: Option<String>,
}

/// Status snapshot returned by the `status` handler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTaskStatus {
    /// Current turn number.
    pub turn: usize,
    /// Phase: "pending", "running", "completed", "failed".
    pub phase: String,
    /// Cumulative cost in USD.
    pub cost_usd: f64,
    /// Total tokens used.
    pub total_tokens: u64,
    /// Final output (if completed).
    pub output: Option<String>,
    /// Error message (if failed).
    pub error: Option<String>,
}

/// Result payload returned when the workflow completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTaskResult {
    /// The final answer from the agent.
    pub output: String,
    /// Total turns executed.
    pub turns: usize,
    /// Total cost in USD.
    pub cost_usd: f64,
    /// Total tokens consumed.
    pub total_tokens: u64,
}

const STATUS_KEY: &str = "agent_status";

// ── Restate workflow definition ─────────────────────────────────────────────

#[restate_sdk::workflow]
pub trait AgentWorkflow {
    /// Main entry point — executes the agent task durably.
    /// Runs exactly once per workflow ID.
    async fn run(req: Json<AgentTaskRequest>) -> Result<Json<AgentTaskResult>, HandlerError>;

    /// Query current agent status without blocking execution.
    #[shared]
    async fn status() -> Result<Json<AgentTaskStatus>, HandlerError>;
}

/// Implementation of the durable agent workflow.
pub struct AgentWorkflowImpl;

impl AgentWorkflow for AgentWorkflowImpl {
    /// Execute the agent task durably.
    ///
    /// The entire agent execution is wrapped in a single `ctx.run()`. On crash,
    /// Restate replays the journal and returns the cached result if the run
    /// already completed. This gives us exactly-once semantics for the task.
    async fn run(
        &self,
        ctx: WorkflowContext<'_>,
        req: Json<AgentTaskRequest>,
    ) -> Result<Json<AgentTaskResult>, HandlerError> {
        let req = req.into_inner();

        // Set initial status
        ctx.set(
            STATUS_KEY,
            serde_json::to_string(&AgentTaskStatus {
                turn: 0,
                phase: "running".to_string(),
                cost_usd: 0.0,
                total_tokens: 0,
                output: None,
                error: None,
            })
            .unwrap(),
        );

        // Serialize the request so the ctx.run() closure captures only a String
        let req_json = serde_json::to_string(&req).map_err(|e| {
            HandlerError::from(restate_sdk::errors::TerminalError::new(format!(
                "serialize request: {e}"
            )))
        })?;

        // Durable side-effect: run the agent task.
        // ctx.run() returns String (JSON-serialized result).
        let result_json: String = ctx
            .run(move || async move {
                let req: AgentTaskRequest = serde_json::from_str(&req_json).map_err(|e| {
                    HandlerError::from(restate_sdk::errors::TerminalError::new(format!(
                        "deserialize request: {e}"
                    )))
                })?;
                run_agent_task(req).await
            })
            .name("agent_execution")
            .await
            .map_err(HandlerError::from)?;

        // Parse result
        let result: AgentTaskResult = serde_json::from_str(&result_json).map_err(|e| {
            HandlerError::from(restate_sdk::errors::TerminalError::new(format!(
                "deserialize result: {e}"
            )))
        })?;

        // Update final status
        let final_status = AgentTaskStatus {
            turn: result.turns,
            phase: "completed".to_string(),
            cost_usd: result.cost_usd,
            total_tokens: result.total_tokens,
            output: Some(result.output.clone()),
            error: None,
        };
        ctx.set(STATUS_KEY, serde_json::to_string(&final_status).unwrap());

        Ok(Json::from(result))
    }

    /// Query current agent status.
    async fn status(
        &self,
        ctx: SharedWorkflowContext<'_>,
    ) -> Result<Json<AgentTaskStatus>, HandlerError> {
        let status_json = ctx.get::<String>(STATUS_KEY).await?;
        match status_json {
            Some(json) => {
                let status: AgentTaskStatus = serde_json::from_str(&json).map_err(|e| {
                    HandlerError::from(restate_sdk::errors::TerminalError::new(format!(
                        "failed to parse status: {e}"
                    )))
                })?;
                Ok(Json::from(status))
            }
            None => Ok(Json::from(AgentTaskStatus {
                turn: 0,
                phase: "pending".to_string(),
                cost_usd: 0.0,
                total_tokens: 0,
                output: None,
                error: None,
            })),
        }
    }
}

/// Build provider config from request fields or fall back to environment.
fn build_provider_config(req: &AgentTaskRequest) -> super::ProviderConfig {
    let model = req
        .model
        .clone()
        .or_else(|| std::env::var("RIG_RLM_MODEL").ok())
        .unwrap_or_else(|| "arcee-ai/trinity-large-preview:free".to_string());

    let base_url = req
        .base_url
        .clone()
        .or_else(|| std::env::var("OPENAI_BASE_URL").ok())
        .unwrap_or_else(|| "https://openrouter.ai/api/v1".to_string());

    let api_key = req
        .api_key
        .clone()
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .unwrap_or_default();

    let name = req
        .provider_name
        .clone()
        .unwrap_or_else(|| "openrouter".to_string());

    super::ProviderConfig::openai_compatible(name, model, base_url, api_key)
}

/// Run the agent task to completion and return JSON-serialized result.
///
/// AgentContext contains non-Send types (PyO3 executor), so we must
/// run the agent on a blocking thread via `spawn_blocking`. Inside,
/// we use a `LocalSet` to allow the async agent loop to execute.
///
/// After execution, persists the session and turns to Turso (RIG_RLM_DB).
async fn run_agent_task(req: AgentTaskRequest) -> Result<String, HandlerError> {
    use super::interaction::agent_task_full;
    use super::{AgentConfig, AgentContext, Role};
    use crate::persistence::{AgentStore, Session, Turn};

    let provider_config = build_provider_config(&req);

    let mut config = AgentConfig {
        max_turns: req.max_turns.unwrap_or(25),
        provider: provider_config,
        max_cost_usd: req.max_cost_usd,
        ..AgentConfig::default()
    };

    if let Some(ref preamble) = req.preamble {
        config.provider.preamble = Some(preamble.clone());
    }

    let task_clone = req.task.clone();

    // Capture the tokio runtime handle for use inside spawn_blocking
    let handle = tokio::runtime::Handle::current();

    let join_result = tokio::task::spawn_blocking(move || {
        handle.block_on(async {
            let local = tokio::task::LocalSet::new();
            local
                .run_until(async {
                    let model_name = config.provider.model.clone();
                    let program = agent_task_full(&req.task, None, Some(&config.memory));
                    let mut agent_ctx = AgentContext::new(config);

                    let run_result = agent_ctx.run(program).await;

                    // ── Turso persistence ────────────────────────────────
                    let db_path =
                        std::env::var("RIG_RLM_DB").unwrap_or_else(|_| "agent.db".to_string());
                    let session_id = uuid::Uuid::new_v4().to_string();

                    if let Ok(store) = AgentStore::open(&db_path).await {
                        let session = Session {
                            session_id: session_id.clone(),
                            model: model_name,
                            task: task_clone,
                            executor: "pyo3".to_string(),
                            optimizer: None,
                            optimized_instruction: None,
                            started_at: chrono::Utc::now().to_rfc3339(),
                            finished_at: None,
                            final_answer: None,
                            score: None,
                        };
                        let _ = store.create_session(&session).await;

                        // Record all turns
                        for (i, msg) in agent_ctx.history.messages().iter().enumerate() {
                            let _ = store
                                .record_turn(&Turn {
                                    session_id: session_id.clone(),
                                    turn_num: i as i32,
                                    role: format!("{:?}", msg.role),
                                    content: msg.content.clone(),
                                    code: None,
                                    exec_stdout: None,
                                    exec_stderr: None,
                                    exec_return: None,
                                    timestamp_ms: chrono::Utc::now().timestamp_millis(),
                                })
                                .await;
                        }

                        // Finish session
                        match &run_result {
                            Ok(answer) => {
                                let _ = store.finish_session(&session_id, Some(answer), None).await;
                            }
                            Err(_) => {
                                let _ = store.finish_session(&session_id, None, None).await;
                            }
                        }
                        eprintln!("📦 Restate session {session_id} saved to {db_path}");
                    }

                    // Return result
                    match run_result {
                        Ok(output) => {
                            let result = AgentTaskResult {
                                output,
                                turns: agent_ctx.turn,
                                cost_usd: agent_ctx.cost_tracker.total_cost_usd,
                                total_tokens: agent_ctx.cost_tracker.total_tokens(),
                            };
                            serde_json::to_string(&result).map_err(|e| format!("serialize: {e}"))
                        }
                        Err(e) => Err(format!("agent error: {e}")),
                    }
                })
                .await
        })
    })
    .await
    .map_err(|e| {
        HandlerError::from(restate_sdk::errors::TerminalError::new(format!(
            "task join: {e}"
        )))
    })?;

    join_result.map_err(|e| HandlerError::from(restate_sdk::errors::TerminalError::new(e)))
}
