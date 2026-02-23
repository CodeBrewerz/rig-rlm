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
    /// Multimodal file attachments (images, PDFs, etc.).
    #[serde(default)]
    pub attachments: Vec<super::attachment::Attachment>,
    /// If true, this workflow delegates the task to a child sub-agent workflow.
    /// The parent acts as a supervisor and the child does the actual work.
    #[serde(default)]
    pub delegate: bool,
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
enum BgToHandler {
    /// LLM call needed. Handler should make the call inside ctx.run().
    NeedLlm {
        /// Serialized history + provider config for the LLM call.
        call_request: String,
        step: usize,
    },
    /// Code execution completed on bg thread. Handler should record in journal.
    ExecDone { output: String, step: usize },
    /// Sub-agent needed. Handler should invoke child workflow via ctx.workflow_client().
    NeedSubAgent {
        task: String,
        step: usize,
        /// If Some, send the result here (bridge thread for llm_query during code exec).
        /// If None, send via HandlerToBg::SubAgentResult (bg thread for explicit SpawnSubAgent).
        reply_tx: Option<std::sync::mpsc::SyncSender<String>>,
    },
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
    /// Sub-agent result (from child workflow — durable on replay).
    SubAgentResult(String),
}

/// Serializable LLM call request (sent from bg thread to handler).
#[derive(Serialize, Deserialize)]
struct LlmCallRequest {
    provider_config: super::ProviderConfig,
    history: Vec<HistoryMsg>,
    trace_ctx: super::otel::TraceContext,
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
        let attachments_for_sub = req.attachments.clone();

        // Channels
        let (bg_tx, mut bg_rx) = tokio::sync::mpsc::unbounded_channel::<BgToHandler>();
        let (handler_tx, handler_rx) = tokio::sync::mpsc::unbounded_channel::<HandlerToBg>();

        // Spawn background thread
        let handle = tokio::runtime::Handle::current();
        let bg_handle = tokio::task::spawn_blocking(move || {
            handle.block_on(async {
                let local = tokio::task::LocalSet::new();
                local
                    .run_until(run_agent_loop(
                        config,
                        &task_text,
                        req.delegate,
                        req.attachments.clone(),
                        bg_tx,
                        handler_rx,
                    ))
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
                                    content: std::borrow::Cow::Owned(msg.content.clone()),
                                    attachments: vec![],
                                });
                            }

                            let trace_ctx = req.trace_ctx;
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

                Ok(BgToHandler::NeedSubAgent {
                    task,
                    step,
                    reply_tx,
                }) => {
                    turn = step;

                    // ── Sub-agent as linked child workflow ────────────
                    // ctx.workflow_client() creates a LINKED child invocation
                    // in the parent's journal — visible in Restate UI as parent→child.
                    // On replay, Restate returns cached child result automatically.
                    let sub_id = format!("sub-{step}-{}", uuid::Uuid::new_v4());

                    // Inherit parent's provider config for the child
                    let sub_req = AgentTaskRequest {
                        task: task.clone(),
                        model: req.model.clone(), // None → child uses env vars
                        base_url: req.base_url.clone(), // None → child uses env vars
                        api_key: req.api_key.clone(), // None → child uses env vars
                        provider_name: req.provider_name.clone(),
                        max_turns: Some(15), // sub-agents get fewer turns
                        max_cost_usd: req.max_cost_usd,
                        preamble: req.preamble.clone(),
                        attachments: attachments_for_sub.clone(), // sub-agents inherit attachments
                        delegate: false, // child does actual work, no further delegation
                    };

                    // Invoke child workflow via Restate service communication
                    // This is durable — journaled as a Call entry in the parent
                    let child_result: Json<AgentTaskResult> = match ctx
                        .workflow_client::<AgentWorkflowClient>(&sub_id)
                        .run(Json::from(sub_req))
                        .call()
                        .await
                    {
                        Ok(result) => result,
                        Err(e) => break Err(format!("sub-agent workflow failed: {e}")),
                    };

                    let result = child_result.into_inner().output;

                    set_status(&ctx, turn, "running", cost_usd, total_tokens, None, None);

                    // Route result to appropriate recipient:
                    // - reply_tx (Some): bridge thread for llm_query during code execution
                    // - reply_tx (None): bg thread for explicit SpawnSubAgent action
                    if let Some(reply) = reply_tx {
                        let _ = reply.send(result);
                    } else {
                        let _ = handler_tx.send(HandlerToBg::SubAgentResult(result));
                    }
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
    mut config: super::AgentConfig,
    task: &str,
    delegate: bool,
    attachments: Vec<super::attachment::Attachment>,
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

    // Enable llm_query() in the Python REPL so generated code can use it.
    // The RESTATE_LLM_BRIDGE intercepts at runtime to route through child workflows,
    // but llm_query must be registered as a Python builtin during sandbox setup.
    config.session.enable_sub_llm = true;
    config.session.provider_config = Some(provider_config.clone());

    let task_owned = task.to_string();
    let program = if delegate {
        // Delegate mode: wrap task in a SpawnSubAgent action
        // This triggers the child workflow path
        use super::capabilities::Capabilities;
        AgentMonad::perform(
            Action::SpawnSubAgent {
                task: task_owned.clone(),
                capabilities: Capabilities::default(),
            },
            |output| AgentMonad::Pure(output.into_string()),
        )
    } else {
        // Phase 28: Recall relevant memories from past sessions
        // Open DB once — reused later for session persistence
        let db_path = std::env::var("RIG_RLM_DB").unwrap_or_else(|_| "agent.db".to_string());
        let recall_provider = super::provider::LlmProvider::new(provider_config.clone());
        let recalled = match recall_provider.embed(task).await {
            Ok(query_emb) => {
                match AgentStore::open(&db_path).await {
                    Ok(store) => match store.recall_memories(&query_emb, 5).await {
                        Ok(memories) if !memories.is_empty() => {
                            eprintln!(
                                "\n• Recalled {} memories for task",
                                memories.len()
                            );
                            for m in &memories {
                                eprintln!(
                                    "  └ [{}] {}",
                                    m.category,
                                    &m.content[..m.content.len().min(80)]
                                );
                            }
                            Some(AgentStore::format_memories_for_prompt(&memories))
                        }
                        Ok(_) => None,
                        Err(e) => { eprintln!("⚠️ recall query: {e}"); None }
                    },
                    Err(e) => { eprintln!("⚠️ recall db: {e}"); None }
                }
            }
            Err(e) => { eprintln!("⚠️ recall embed: {e}"); None }
        };
        agent_task_full(task, None, Some(&config.memory), recalled.as_deref(), attachments)
    };
    let mut agent_ctx = AgentContext::new(config);

    // ── Register lifecycle hooks (Codex pattern) ─────
    // Default logging hook — fires at every lifecycle point
    {
        use super::hooks::{HookEvent, HookResult};

        agent_ctx.hooks.on("restate_lifecycle_logger", |event| {
            match event {
                HookEvent::SessionStart { task } => {
                    eprintln!("🪝 [hook] session_start | task={}", &task[..task.len().min(80)]);
                }
                HookEvent::SessionEnd { turns, final_answer_len } => {
                    eprintln!("🪝 [hook] session_end | turns={turns} answer_len={final_answer_len}");
                }
                HookEvent::BeforeLlmCall { turn, message_count } => {
                    eprintln!("🪝 [hook] before_llm | turn={turn} messages={message_count}");
                }
                HookEvent::AfterLlmCall { turn, response_len, duration } => {
                    eprintln!("🪝 [hook] after_llm | turn={turn} response_len={response_len} duration={:.1}s", duration.as_secs_f64());
                }
                HookEvent::BeforeCodeExec { turn, code_preview } => {
                    eprintln!("🪝 [hook] before_exec | turn={turn} code={}", &code_preview[..code_preview.len().min(60)]);
                }
                HookEvent::AfterCodeExec { turn, success, duration, output_preview } => {
                    eprintln!("🪝 [hook] after_exec | turn={turn} success={success} duration={:.1}s output={}", duration.as_secs_f64(), &output_preview[..output_preview.len().min(60)]);
                }
                HookEvent::BeforeSubAgent { task } => {
                    eprintln!("🪝 [hook] before_subagent | task={}", &task[..task.len().min(80)]);
                }
                HookEvent::AfterSubAgent { task, result_len, success } => {
                    eprintln!("🪝 [hook] after_subagent | task={} result_len={result_len} success={success}", &task[..task.len().min(60)]);
                }
                HookEvent::AfterCompaction { messages_before, messages_after } => {
                    eprintln!("🪝 [hook] after_compaction | {messages_before}→{messages_after} messages");
                }
            }
            HookResult::Continue
        });
    }

    let _ = agent_ctx.ensure_session_public().await;

    // ── Initialize context memory store for this session ─────
    crate::sandbox::init_context_memory_store();

    // ── Set up Restate LLM bridge (lock-free, parallel-capable) ─────
    // No bridge thread — each llm_query() creates its own reply channel
    // and dispatches directly to the handler via a captured closure.
    {
        use crate::sandbox::{RestateLlmBridge, set_restate_llm_bridge};

        // Shared step counter: AtomicUsize avoids collision with bg thread's step.
        // Starts at 10_000 to leave headroom for the main loop's steps.
        let step_counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(10_000));

        let dispatch_tx = tx.clone(); // clone the BgToHandler sender
        let dispatch_step = step_counter.clone();

        let bridge = Box::new(RestateLlmBridge {
            dispatch: Box::new(move |prompt, reply_tx| {
                let step = dispatch_step.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                // Send directly to handler — no intermediary thread
                let _ = dispatch_tx.send(BgToHandler::NeedSubAgent {
                    task: prompt,
                    step,
                    reply_tx: Some(reply_tx),
                });
            }),
        });
        set_restate_llm_bridge(bridge);
    }

    let mut computation = program;
    let mut step: usize = 0;

    // 🪝 Session start hook
    {
        use super::hooks::HookEvent;
        let task_preview = task_owned.chars().take(100).collect::<String>();
        let _ = agent_ctx
            .hooks
            .fire(&HookEvent::SessionStart { task: task_preview });
    }

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
                        // 🪝 Hook: BeforeLlmCall
                        {
                            use super::hooks::HookEvent;
                            let msg_count = agent_ctx.history.messages().len();
                            if let Err(reason) = agent_ctx.hooks.fire(&HookEvent::BeforeLlmCall {
                                turn: agent_ctx.turn,
                                message_count: msg_count,
                            }) {
                                break Err(format!("Hook abort: {reason}"));
                            }
                        }
                        let llm_start = std::time::Instant::now();

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
                                content: m.content.to_string(),
                            })
                            .collect();

                        let call_req = LlmCallRequest {
                            provider_config: provider_config.clone(),
                            history: history_msgs,
                            trace_ctx: super::otel::TraceContext::new()
                                .with_session(task_owned.as_str())
                                .with_name(format!("restate:llm_{step}"))
                                .with_tags(vec!["restate".to_string()])
                                .with_metadata("model", model_name.as_str())
                                .with_metadata("step", step.to_string()),
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
                            content: std::borrow::Cow::Owned(llm_result.response.clone()),
                            attachments: vec![],
                        });

                        let output = ActionOutput::Value(llm_result.response);

                        // 🪝 Hook: AfterLlmCall
                        {
                            use super::hooks::HookEvent;
                            let llm_dur = llm_start.elapsed();
                            let _ = agent_ctx.hooks.fire(&HookEvent::AfterLlmCall {
                                turn: agent_ctx.turn,
                                response_len: output.clone().into_string().len(),
                                duration: llm_dur,
                            });
                        }

                        computation = next(output);
                    }

                    // ── Code execution: run on bg thread, record in journal ──
                    Action::ExecuteCode { source: ref _source } => {
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

                    Action::SpawnSubAgent { task, .. } => {
                        // 🪝 Hook: BeforeSubAgent
                        {
                            use super::hooks::HookEvent;
                            let _ = agent_ctx
                                .hooks
                                .fire(&HookEvent::BeforeSubAgent { task: task.clone() });
                        }

                        let _ = tx.send(BgToHandler::NeedSubAgent {
                            task: task.clone(),
                            step,
                            reply_tx: None, // bg thread reads result from rx
                        });

                        // Wait for handler to invoke child workflow
                        let result = match rx.recv().await {
                            Some(HandlerToBg::SubAgentResult(output)) => output,
                            _ => {
                                break Err("handler closed during sub-agent".to_string());
                            }
                        };

                        // Record sub-agent result in history
                        agent_ctx.history.push(super::history::HistoryMessage {
                            role: super::action::Role::Execution,
                            content: std::borrow::Cow::Owned(format!("Sub-agent result: {result}")),
                            attachments: vec![],
                        });

                        let output = super::action::ActionOutput::Value(result.clone());

                        // 🪝 Hook: AfterSubAgent
                        {
                            use super::hooks::HookEvent;
                            let _ = agent_ctx.hooks.fire(&HookEvent::AfterSubAgent {
                                task: task.clone(),
                                result_len: result.len(),
                                success: true,
                            });
                        }

                        computation = next(output);
                    }

                    // ── Orchestrate: durable multi-agent via sequential child workflows ──
                    //
                    // Under Restate, we dispatch each sub-agent as a sequential
                    // NeedSubAgent message. The handler invokes each as a durable
                    // child workflow (journaled). On replay, Restate returns cached
                    // results automatically.
                    //
                    // Tradeoff: sequential (not parallel) under Restate for journal
                    // determinism. Locally, interpret_action() does true parallel.
                    Action::Orchestrate { orchestrator } => {
                        use super::orchestrator::SubAgentResult;

                        let agent_count = orchestrator.agent_count();
                        let agent_names: Vec<&str> = orchestrator
                            .specs()
                            .iter()
                            .map(|s| s.name.as_str())
                            .collect();

                        // ── Codex-style: Agent spawned traces ──
                        for spec in orchestrator.specs() {
                            let prompt_preview: String =
                                spec.task.chars().take(120).collect();
                            eprintln!(
                                "\n• Agent spawned (Restate durable)\n  └ name: \"{}\"\n    status: pending init\n    prompt: {prompt_preview}...",
                                spec.name
                            );
                        }

                        eprintln!(
                            "\n• Waiting for agents (sequential under Restate)\n  └ receivers: {}",
                            agent_names.join(", ")
                        );

                        let orch_start = std::time::Instant::now();
                        let mut results: Vec<SubAgentResult> = Vec::new();

                        // Dispatch each sub-agent sequentially as a durable child workflow
                        for (i, spec) in orchestrator.specs().iter().enumerate() {
                            let agent_start = std::time::Instant::now();

                            eprintln!(
                                "\n• Agent running (Restate child workflow)\n  └ name: \"{}\"\n    status: running ({}/{})",
                                spec.name, i + 1, agent_count
                            );

                            // Send to handler — it calls ctx.workflow_client().run().call().await
                            let _ = tx.send(BgToHandler::NeedSubAgent {
                                task: spec.task.clone(),
                                step: step + i,
                                reply_tx: None,
                            });

                            // Wait for handler's durable result
                            let result_str = match rx.recv().await {
                                Some(HandlerToBg::SubAgentResult(output)) => output,
                                _ => {
                                    let err = format!(
                                        "handler closed during orchestrated agent '{}'",
                                        spec.name
                                    );
                                    eprintln!(
                                        "\n• Agent failed\n  └ name: \"{}\"\n    error: {err}",
                                        spec.name
                                    );
                                    results.push(SubAgentResult::failure(
                                        spec.name.clone(),
                                        err,
                                        agent_start.elapsed(),
                                    ));
                                    continue;
                                }
                            };

                            let duration = agent_start.elapsed();
                            let snapshot: String =
                                result_str.chars().take(150).collect();
                            eprintln!(
                                "\n• Agent complete\n  └ name: \"{}\"\n    status: success\n    duration: {:.1}s\n    snapshot: {snapshot}...",
                                spec.name,
                                duration.as_secs_f64(),
                            );

                            results.push(SubAgentResult::success(
                                spec.name.clone(),
                                result_str,
                                0, // turns not tracked via channel
                                duration,
                            ));
                        }

                        let total_duration = orch_start.elapsed();
                        let success_count = results.iter().filter(|r| r.success).count();
                        eprintln!(
                            "\n• Orchestration complete (Restate durable)\n  └ {success_count}/{agent_count} agents succeeded in {:.1}s",
                            total_duration.as_secs_f64()
                        );

                        // Format combined results
                        let formatted = super::orchestrator::Orchestrator::format_results(&results);
                        let output = super::action::ActionOutput::Value(formatted);

                        computation = next(output);
                    }

                    // ── All other actions: execute inline on bg thread ──
                    other => {
                        let output = match agent_ctx.interpret_action(other).await {
                            Ok(out) => out,
                            Err(e) => break Err(format!("action error: {e}")),
                        };

                        let _ = tx.send(BgToHandler::PureDone);

                        computation = next(output);
                    }
                }
            }
        }
    };

    // 🪝 Session end hook
    {
        use super::hooks::HookEvent;
        let result_len = match &final_result {
            Ok(s) => s.len(),
            Err(s) => s.len(),
        };
        let _ = agent_ctx.hooks.fire(&HookEvent::SessionEnd {
            turns: agent_ctx.turn,
            final_answer_len: result_len,
        });
    }

    // ── Turso persistence ────────────────────────────────────────────────
    let db_path = std::env::var("RIG_RLM_DB").unwrap_or_else(|_| "agent.db".to_string());
    let session_id = uuid::Uuid::new_v4().to_string();

    // Reuse the same DB handle (avoids reopening + schema re-init)
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
                    content: msg.content.to_string(),
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

        // Phase 28: Extract and store memories for future recall
        if final_result.is_ok() {
            match agent_ctx.extract_and_store_memories(&session_id).await {
                Ok(n) if n > 0 => eprintln!("🧠 {n} memories extracted and stored"),
                Ok(_) => {} // nothing worth remembering
                Err(e) => eprintln!("⚠️ memory extraction error: {e}"),
            }
        }
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
