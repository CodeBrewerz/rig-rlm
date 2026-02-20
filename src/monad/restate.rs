//! Restate durable agent workflow — granular journal entries with durable LLM replay.
//!
//! Architecture: the handler drives ctx.run() calls while a background thread
//! steps through the monad. Each I/O action is a separate journal entry.
//!
//! **Key durable guarantee**: LLM calls happen INSIDE ctx.run() closures, so on
//! Restate replay they return cached results — no duplicate API calls or costs.
//! Code execution runs on the bg thread (for REPL state) with output recorded
//! via ctx.run().
//!
//! ```text
//! Restate Handler                 Background Thread (spawn_blocking)
//!   │                               │
//!   │  ← NeedLlm { history, step }  │  monad hit ModelInference
//!   │                               │
//!   │ ctx.run("llm_3") {            │
//!   │   provider.chat(history)      │  (NOT called on replay — cached!)
//!   │ } → response                  │
//!   │                               │
//!   │  → LlmResult { response }    │  bg thread records cost, evidence
//!   │                               │
//!   │  ← ExecDone { output, step }  │  bg thread executed code (REPL state)
//!   │                               │
//!   │ ctx.run("exec_4") {           │
//!   │   return output               │  (just records output to journal)
//!   │ }                             │
//!   │  → ExecAck                    │
//!   │                               │
//!   │  ← Finished(result)           │  monad returned Pure(value)
//! ```

use restate_sdk::errors::HandlerError;
use restate_sdk::prelude::*;
use serde::{Deserialize, Serialize};

/// Request payload for starting a durable agent task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTaskRequest {
    pub task: String,
    pub model: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub provider_name: Option<String>,
    pub max_turns: Option<usize>,
    pub max_cost_usd: Option<f64>,
    pub preamble: Option<String>,
}

/// Status snapshot returned by the `status` handler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTaskStatus {
    pub turn: usize,
    pub phase: String,
    pub cost_usd: f64,
    pub total_tokens: u64,
    pub output: Option<String>,
    pub error: Option<String>,
}

/// Result payload returned when the workflow completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTaskResult {
    pub output: String,
    pub turns: usize,
    pub cost_usd: f64,
    pub total_tokens: u64,
}

const STATUS_KEY: &str = "agent_status";

// ── Messages between handler and background thread ─────────────────────────

/// Messages from bg thread → handler.
#[derive(Debug)]
enum BgToHandler {
    /// LLM call needed. Handler should make the call inside ctx.run().
    NeedLlm {
        /// Serialized history + provider config for the LLM call.
        call_request: String,
        step: usize,
    },
    /// Code execution completed on bg thread. Handler should record in journal.
    ExecDone { output: String, step: usize },
    /// Pure action completed (no ctx.run() needed).
    PureDone,
    /// Monad finished with a result.
    Finished(Result<String, String>),
}

/// Messages from handler → bg thread.
#[derive(Debug)]
enum HandlerToBg {
    /// LLM call result (from ctx.run() — either fresh or cached on replay).
    LlmResult(String), // JSON: LlmCallResult
    /// Ack that exec output was recorded.
    ExecAck,
}

/// Serializable LLM call request (sent from bg thread to handler).
#[derive(Serialize, Deserialize)]
struct LlmCallRequest {
    provider_config: super::ProviderConfig,
    history: Vec<HistoryMsg>,
}

/// Serializable history message for cross-thread transfer.
#[derive(Serialize, Deserialize, Clone)]
struct HistoryMsg {
    role: String,
    content: String,
}

/// Serializable LLM call result (returned from ctx.run() and sent to bg thread).
#[derive(Serialize, Deserialize)]
struct LlmCallResult {
    response: String,
    input_tokens: u64,
    output_tokens: u64,
}

// ── Restate workflow definition ─────────────────────────────────────────────

#[restate_sdk::workflow]
pub trait AgentWorkflow {
    async fn run(req: Json<AgentTaskRequest>) -> Result<Json<AgentTaskResult>, HandlerError>;

    #[shared]
    async fn status() -> Result<Json<AgentTaskStatus>, HandlerError>;
}

pub struct AgentWorkflowImpl;

impl AgentWorkflow for AgentWorkflowImpl {
    async fn run(
        &self,
        ctx: WorkflowContext<'_>,
        req: Json<AgentTaskRequest>,
    ) -> Result<Json<AgentTaskResult>, HandlerError> {
        let req = req.into_inner();

        set_status(&ctx, 0, "running", 0.0, 0, None, None);

        // Build config
        let provider_config = build_provider_config(&req);
        let mut config = super::AgentConfig {
            max_turns: req.max_turns.unwrap_or(25),
            provider: provider_config,
            max_cost_usd: req.max_cost_usd,
            ..super::AgentConfig::default()
        };
        if let Some(ref preamble) = req.preamble {
            config.provider.preamble = Some(preamble.clone());
        }

        let task_text = req.task.clone();

        // Channels
        let (bg_tx, mut bg_rx) = tokio::sync::mpsc::unbounded_channel::<BgToHandler>();
        let (handler_tx, handler_rx) = tokio::sync::mpsc::unbounded_channel::<HandlerToBg>();

        // Spawn background thread
        let handle = tokio::runtime::Handle::current();
        let bg_handle = tokio::task::spawn_blocking(move || {
            handle.block_on(async {
                let local = tokio::task::LocalSet::new();
                local
                    .run_until(run_agent_loop(config, &task_text, bg_tx, handler_rx))
                    .await
            })
        });

        // ── Handler loop ─────────────────────────────────────────────────

        let mut turn: usize = 0;
        let mut cost_usd: f64 = 0.0;
        let mut total_tokens: u64 = 0;

        let result: Result<AgentTaskResult, String> = loop {
            let msg = bg_rx
                .recv()
                .await
                .ok_or_else(|| "agent thread closed unexpectedly".to_string());

            match msg {
                Err(e) => break Err(e),

                Ok(BgToHandler::PureDone) => continue,

                Ok(BgToHandler::NeedLlm { call_request, step }) => {
                    turn = step;

                    // ── LLM call happens INSIDE ctx.run() ──────────────
                    // On replay, Restate returns cached result → no API call!
                    let journal_name = format!("llm_{step}");
                    let result_json: String = match ctx
                        .run(move || async move {
                            // This closure ONLY runs on fresh execution, NOT on replay.
                            let req: LlmCallRequest =
                                serde_json::from_str(&call_request).map_err(|e| {
                                    HandlerError::from(restate_sdk::errors::TerminalError::new(
                                        format!("deserialize llm request: {e}"),
                                    ))
                                })?;

                            // Create provider and make the call
                            let provider = super::provider::LlmProvider::new(req.provider_config);

                            // Rebuild history
                            let mut history = super::history::ConversationHistory::new();
                            for msg in &req.history {
                                let role = match msg.role.as_str() {
                                    "System" | "system" => super::Role::System,
                                    "User" | "user" => super::Role::User,
                                    "Assistant" | "assistant" => super::Role::Assistant,
                                    _ => super::Role::Execution,
                                };
                                history.push(super::history::HistoryMessage {
                                    role,
                                    content: msg.content.clone(),
                                });
                            }

                            let trace_ctx = super::otel::TraceContext::default();
                            let (response, usage) =
                                provider.chat(&history, &trace_ctx).await.map_err(|e| {
                                    HandlerError::from(restate_sdk::errors::TerminalError::new(
                                        format!("LLM call failed: {e}"),
                                    ))
                                })?;

                            let result = LlmCallResult {
                                response,
                                input_tokens: usage.input_tokens.unwrap_or(0) as u64,
                                output_tokens: usage.output_tokens.unwrap_or(0) as u64,
                            };
                            serde_json::to_string(&result).map_err(|e| {
                                HandlerError::from(restate_sdk::errors::TerminalError::new(
                                    format!("serialize llm result: {e}"),
                                ))
                            })
                        })
                        .name(&journal_name)
                        .await
                    {
                        Ok(json) => json,
                        Err(e) => break Err(format!("llm ctx.run failed: {e}")),
                    };

                    // Parse cost info
                    if let Ok(result) = serde_json::from_str::<LlmCallResult>(&result_json) {
                        cost_usd += 0.0; // Cost tracked on bg thread
                        total_tokens += result.input_tokens + result.output_tokens;
                    }

                    set_status(&ctx, turn, "running", cost_usd, total_tokens, None, None);

                    // Send result to bg thread for bookkeeping
                    let _ = handler_tx.send(HandlerToBg::LlmResult(result_json));
                }

                Ok(BgToHandler::ExecDone { output, step }) => {
                    turn = step;

                    // Record code execution output in journal
                    let journal_name = format!("exec_{step}");
                    let captured = output.clone();
                    let _ = ctx
                        .run(move || async move { Ok(captured) })
                        .name(&journal_name)
                        .await;

                    set_status(&ctx, turn, "running", cost_usd, total_tokens, None, None);

                    // Ack to bg thread
                    let _ = handler_tx.send(HandlerToBg::ExecAck);
                }

                Ok(BgToHandler::Finished(result)) => match result {
                    Ok(json) => {
                        let task_result: AgentTaskResult =
                            serde_json::from_str(&json).unwrap_or(AgentTaskResult {
                                output: json,
                                turns: turn,
                                cost_usd,
                                total_tokens,
                            });
                        break Ok(task_result);
                    }
                    Err(e) => break Err(e),
                },
            }
        };

        let _ = bg_handle.await;

        match result {
            Ok(task_result) => {
                set_status(
                    &ctx,
                    task_result.turns,
                    "completed",
                    task_result.cost_usd,
                    task_result.total_tokens,
                    Some(task_result.output.clone()),
                    None,
                );
                Ok(Json::from(task_result))
            }
            Err(e) => {
                set_status(
                    &ctx,
                    turn,
                    "failed",
                    cost_usd,
                    total_tokens,
                    None,
                    Some(e.clone()),
                );
                Err(HandlerError::from(restate_sdk::errors::TerminalError::new(
                    e,
                )))
            }
        }
    }

    async fn status(
        &self,
        ctx: SharedWorkflowContext<'_>,
    ) -> Result<Json<AgentTaskStatus>, HandlerError> {
        let status_json = ctx.get::<String>(STATUS_KEY).await?;
        match status_json {
            Some(json) => {
                let status: AgentTaskStatus = serde_json::from_str(&json).map_err(|e| {
                    HandlerError::from(restate_sdk::errors::TerminalError::new(format!(
                        "parse status: {e}"
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

fn set_status(
    ctx: &WorkflowContext<'_>,
    turn: usize,
    phase: &str,
    cost_usd: f64,
    total_tokens: u64,
    output: Option<String>,
    error: Option<String>,
) {
    let status = AgentTaskStatus {
        turn,
        phase: phase.to_string(),
        cost_usd,
        total_tokens,
        output,
        error,
    };
    ctx.set(
        STATUS_KEY,
        serde_json::to_string(&status).unwrap_or_default(),
    );
}

// ── Background agent loop ───────────────────────────────────────────────────

/// Drive the monad loop on a background thread (for !Send AgentContext).
///
/// For ModelInference: does NOT call the LLM. Instead, sends history to the
/// handler, which calls the LLM inside ctx.run() (durable on replay).
///
/// For ExecuteCode: executes on this thread (maintains REPL state), sends
/// output to handler for journal recording.
async fn run_agent_loop(
    config: super::AgentConfig,
    task: &str,
    tx: tokio::sync::mpsc::UnboundedSender<BgToHandler>,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<HandlerToBg>,
) {
    use super::action::{Action, ActionOutput, Role};
    use super::history::HistoryMessage;
    use super::interaction::agent_task_full;
    use super::{AgentContext, AgentMonad};
    use crate::persistence::{AgentStore, Session, Turn};

    let model_name = config.provider.model.clone();
    let provider_config = config.provider.clone();
    let task_owned = task.to_string();
    let program = agent_task_full(task, None, Some(&config.memory));
    let mut agent_ctx = AgentContext::new(config);

    let _ = agent_ctx.ensure_session_public().await;

    let mut computation = program;
    let mut step: usize = 0;

    let final_result: Result<String, String> = loop {
        match computation {
            AgentMonad::Pure(value) => break Ok(value),

            AgentMonad::Perform { action, next } => {
                step += 1;

                // Turn limit
                agent_ctx.turn += 1;
                if agent_ctx.turn > agent_ctx.config.max_turns {
                    break Err(format!(
                        "max turns exceeded: {}",
                        agent_ctx.config.max_turns
                    ));
                }

                match action {
                    // ── ModelInference: delegate to handler's ctx.run() ──
                    Action::ModelInference => {
                        // Budget check
                        if let Err((spent, limit)) = agent_ctx
                            .cost_tracker
                            .check_budget(agent_ctx.config.max_cost_usd)
                        {
                            break Err(format!("budget exceeded: ${spent:.4} >= ${limit:.4}"));
                        }

                        // Serialize history for the handler
                        let history_msgs: Vec<HistoryMsg> = agent_ctx
                            .history
                            .messages()
                            .iter()
                            .map(|m| HistoryMsg {
                                role: format!("{:?}", m.role),
                                content: m.content.clone(),
                            })
                            .collect();

                        let call_req = LlmCallRequest {
                            provider_config: provider_config.clone(),
                            history: history_msgs,
                        };
                        let call_json = serde_json::to_string(&call_req).unwrap_or_default();

                        // Send to handler — handler makes the LLM call in ctx.run()
                        let _ = tx.send(BgToHandler::NeedLlm {
                            call_request: call_json,
                            step,
                        });

                        // Wait for result (fresh or cached from replay)
                        let result = match rx.recv().await {
                            Some(HandlerToBg::LlmResult(json)) => json,
                            _ => {
                                break Err("handler closed during LLM call".to_string());
                            }
                        };

                        let llm_result: LlmCallResult = serde_json::from_str(&result)
                            .map_err(|e| format!("parse llm result: {e}"))
                            .unwrap_or(LlmCallResult {
                                response: result.clone(),
                                input_tokens: 0,
                                output_tokens: 0,
                            });

                        // Bookkeeping (on bg thread)
                        let _ = agent_ctx.cost_tracker.record(
                            &model_name,
                            llm_result.input_tokens,
                            llm_result.output_tokens,
                        );

                        agent_ctx
                            .evidence
                            .push(super::evidence::Evidence::from_inference(
                                &llm_result.response,
                            ));

                        agent_ctx.history.push(HistoryMessage {
                            role: Role::Assistant,
                            content: llm_result.response.clone(),
                        });

                        let output = ActionOutput::Value(llm_result.response);
                        computation = next(output);
                    }

                    // ── Code execution: run on bg thread, record in journal ──
                    Action::ExecuteCode { ref source } => {
                        let output = match agent_ctx.interpret_action(action).await {
                            Ok(out) => out,
                            Err(e) => break Err(format!("exec error: {e}")),
                        };

                        let _ = tx.send(BgToHandler::ExecDone {
                            output: output.clone().into_string(),
                            step,
                        });

                        // Wait for handler to record in journal
                        match rx.recv().await {
                            Some(HandlerToBg::ExecAck) => {}
                            _ => break Err("handler closed during exec".to_string()),
                        }

                        computation = next(output);
                    }

                    // ── All other actions: execute inline on bg thread ──
                    other => {
                        let is_io = other.is_io();
                        let output = match agent_ctx.interpret_action(other).await {
                            Ok(out) => out,
                            Err(e) => break Err(format!("action error: {e}")),
                        };

                        if is_io {
                            // SpawnSubAgent etc — record in journal
                            let _ = tx.send(BgToHandler::ExecDone {
                                output: output.clone().into_string(),
                                step,
                            });
                            match rx.recv().await {
                                Some(HandlerToBg::ExecAck) => {}
                                _ => break Err("handler closed".to_string()),
                            }
                        } else {
                            let _ = tx.send(BgToHandler::PureDone);
                        }

                        computation = next(output);
                    }
                }
            }
        }
    };

    // ── Turso persistence ────────────────────────────────────────────────
    let db_path = std::env::var("RIG_RLM_DB").unwrap_or_else(|_| "agent.db".to_string());
    let session_id = uuid::Uuid::new_v4().to_string();

    if let Ok(store) = AgentStore::open(&db_path).await {
        let session = Session {
            session_id: session_id.clone(),
            model: model_name.clone(),
            task: task_owned,
            executor: "pyo3".to_string(),
            optimizer: None,
            optimized_instruction: None,
            started_at: chrono::Utc::now().to_rfc3339(),
            finished_at: None,
            final_answer: None,
            score: None,
        };
        let _ = store.create_session(&session).await;

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

        match &final_result {
            Ok(answer) => {
                let _ = store.finish_session(&session_id, Some(answer), None).await;
            }
            Err(_) => {
                let _ = store.finish_session(&session_id, None, None).await;
            }
        }
        eprintln!("📦 Restate session {session_id} saved to {db_path}");
    }

    // Send final result
    let result_json = match &final_result {
        Ok(output) => serde_json::to_string(&AgentTaskResult {
            output: output.clone(),
            turns: agent_ctx.turn,
            cost_usd: agent_ctx.cost_tracker.total_cost_usd,
            total_tokens: agent_ctx.cost_tracker.total_tokens(),
        })
        .unwrap_or_default(),
        Err(_) => String::new(),
    };

    let _ = tx.send(BgToHandler::Finished(final_result.map(|_| result_json)));
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
