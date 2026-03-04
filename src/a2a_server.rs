//! A2A (Agent-to-Agent) HTTP Server — expose rig-rlm as a Google A2A agent.
//!
//! Implements the A2A protocol so that agentgateway (or any A2A-compatible
//! client) can discover and interact with this agent.
//!
//! Entry points:
//!   - `GET  /.well-known/agent.json`  → Agent Card discovery
//!   - `POST /`                        → JSON-RPC (message/send, message/stream)

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use tower_http::cors::{Any, CorsLayer};
use uuid::Uuid;

use crate::monad::interaction::agent_task;
use crate::monad::{AgentConfig, AgentContext};

// ── A2A Protocol Types ────────────────────────────────────────────

/// The Agent Card — describes agent capabilities for discovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCard {
    pub name: String,
    pub description: String,
    pub url: String,
    pub version: String,
    pub protocol_version: String,
    pub capabilities: AgentCapabilities,
    pub default_input_modes: Vec<String>,
    pub default_output_modes: Vec<String>,
    pub skills: Vec<AgentSkill>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCapabilities {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub streaming: Option<bool>,
    #[serde(
        rename = "pushNotifications",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub push_notifications: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSkill {
    pub id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

/// A2A JSON-RPC request envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: serde_json::Value,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

/// A2A JSON-RPC response envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// A2A Message (from client or agent).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2aMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parts: Vec<A2aPart>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
}

/// A2A Part — text, file, or data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum A2aPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "file")]
    File { file: serde_json::Value },
    #[serde(rename = "data")]
    Data { data: serde_json::Value },
}

/// Parameters for `message/send` and `message/stream`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageSendParams {
    pub message: A2aMessage,
    #[serde(default)]
    pub metadata: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub configuration: Option<serde_json::Value>,
}

/// A2A Task status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStatus {
    pub state: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<A2aMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

/// A2A Task — returned by message/send.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct A2aTask {
    pub id: String,
    pub context_id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub kind: String,
    pub status: TaskStatus,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub history: Vec<A2aMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<serde_json::Value>,
}

// ── Server State ──────────────────────────────────────────────────

/// Shared state for the A2A server.
#[derive(Clone)]
pub struct A2aServerState {
    pub agent_card: AgentCard,
    pub agent_config: AgentConfig,
}

// ── Default Agent Card ────────────────────────────────────────────

impl A2aServerState {
    /// Create a new A2A server state with defaults.
    pub fn new(port: u16, agent_config: AgentConfig) -> Self {
        Self {
            agent_card: AgentCard {
                name: "rig-rlm".to_string(),
                description: "Monadic AI agent with durable execution, code sandbox, and prompt optimization. Powered by Rig.".to_string(),
                url: format!("http://localhost:{port}"),
                version: "0.1.0".to_string(),
                protocol_version: "0.2.6".to_string(),
                capabilities: AgentCapabilities {
                    streaming: Some(true),
                    push_notifications: Some(false),
                },
                default_input_modes: vec!["text".to_string()],
                default_output_modes: vec!["text".to_string()],
                skills: vec![
                    AgentSkill {
                        id: "run_task".to_string(),
                        name: "Run Task".to_string(),
                        description: "Execute a task using the monadic agent loop with LLM reasoning, Python code execution, and file editing.".to_string(),
                        tags: vec!["coding".to_string(), "reasoning".to_string(), "execution".to_string()],
                    },
                    AgentSkill {
                        id: "execute_python".to_string(),
                        name: "Execute Python".to_string(),
                        description: "Run Python code in a sandboxed environment (PyO3 or Microsandbox).".to_string(),
                        tags: vec!["python".to_string(), "sandbox".to_string()],
                    },
                    AgentSkill {
                        id: "apply_patch".to_string(),
                        name: "Apply Patch".to_string(),
                        description: "Apply unified diff patches to edit files on disk.".to_string(),
                        tags: vec!["editing".to_string(), "diff".to_string()],
                    },
                ],
            },
            agent_config,
        }
    }
}

// ── HTTP Handlers ─────────────────────────────────────────────────

/// GET /.well-known/agent.json — Agent Card discovery.
async fn handle_agent_card(State(state): State<Arc<A2aServerState>>) -> Json<AgentCard> {
    Json(state.agent_card.clone())
}

/// POST / — JSON-RPC dispatcher for A2A requests.
async fn handle_a2a_request(
    State(state): State<Arc<A2aServerState>>,
    Json(req): Json<JsonRpcRequest>,
) -> Response {
    match req.method.as_str() {
        "message/send" => handle_message_send(state, req).await,
        "message/stream" => handle_message_stream(state, req).await,
        method => {
            let error_resp = JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: req.id,
                result: None,
                error: Some(JsonRpcError {
                    code: -32601,
                    message: format!("Method not found: {method}"),
                    data: None,
                }),
            };
            Json(error_resp).into_response()
        }
    }
}

/// Handle `message/send` — synchronous task execution.
async fn handle_message_send(state: Arc<A2aServerState>, req: JsonRpcRequest) -> Response {
    // Parse params
    let params: MessageSendParams = match serde_json::from_value(req.params) {
        Ok(p) => p,
        Err(e) => {
            let error_resp = JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: req.id,
                result: None,
                error: Some(JsonRpcError {
                    code: -32602,
                    message: format!("Invalid params: {e}"),
                    data: None,
                }),
            };
            return Json(error_resp).into_response();
        }
    };

    // Extract text from the message parts
    let task_text = extract_text_from_message(&params.message);
    if task_text.is_empty() {
        let error_resp = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: req.id,
            result: None,
            error: Some(JsonRpcError {
                code: -32602,
                message: "Message must contain at least one text part".to_string(),
                data: None,
            }),
        };
        return Json(error_resp).into_response();
    }

    let task_id = Uuid::new_v4().to_string();
    let context_id = Uuid::new_v4().to_string();

    // Run the agent (on a dedicated OS thread to avoid PyO3 GIL deadlock)
    tracing::info!("[a2a] handle_message_send: dispatching to agent thread...");
    let rx = run_agent_task(&state.agent_config, &task_text);
    tracing::info!("[a2a] handle_message_send: waiting for oneshot result...");
    let result = rx
        .await
        .map_err(|e| anyhow::anyhow!("Task channel error: {e}"))
        .and_then(|r| r);
    tracing::info!(
        "[a2a] handle_message_send: got result (ok={})",
        result.is_ok()
    );

    let task = match result {
        Ok(answer) => A2aTask {
            id: task_id,
            context_id,
            kind: "task".to_string(),
            status: TaskStatus {
                state: "completed".to_string(),
                message: Some(A2aMessage {
                    role: "agent".to_string(),
                    parts: vec![A2aPart::Text { text: answer }],
                    metadata: None,
                }),
                timestamp: Some(chrono::Utc::now().to_rfc3339()),
            },
            history: vec![params.message],
            artifacts: vec![],
        },
        Err(e) => A2aTask {
            id: task_id,
            context_id,
            kind: "task".to_string(),
            status: TaskStatus {
                state: "failed".to_string(),
                message: Some(A2aMessage {
                    role: "agent".to_string(),
                    parts: vec![A2aPart::Text {
                        text: format!("Agent error: {e}"),
                    }],
                    metadata: None,
                }),
                timestamp: Some(chrono::Utc::now().to_rfc3339()),
            },
            history: vec![params.message],
            artifacts: vec![],
        },
    };

    tracing::info!("[a2a] handle_message_send: serializing response...");
    let resp = JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id: req.id,
        result: Some(serde_json::to_value(&task).unwrap_or_default()),
        error: None,
    };
    tracing::info!("[a2a] handle_message_send: returning response.");
    Json(resp).into_response()
}

/// Handle `message/stream` — streaming task execution via SSE.
///
/// Uses manual SSE formatting instead of axum's Sse wrapper to avoid
/// buffering issues that prevent events from reaching the client.
async fn handle_message_stream(state: Arc<A2aServerState>, req: JsonRpcRequest) -> Response {
    // Parse params
    let params: MessageSendParams = match serde_json::from_value(req.params) {
        Ok(p) => p,
        Err(e) => {
            let error_resp = JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: req.id,
                result: None,
                error: Some(JsonRpcError {
                    code: -32602,
                    message: format!("Invalid params: {e}"),
                    data: None,
                }),
            };
            return Json(error_resp).into_response();
        }
    };

    let task_text = extract_text_from_message(&params.message);
    if task_text.is_empty() {
        let error_resp = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: req.id,
            result: None,
            error: Some(JsonRpcError {
                code: -32602,
                message: "Message must contain at least one text part".to_string(),
                data: None,
            }),
        };
        return Json(error_resp).into_response();
    }

    let task_id = Uuid::new_v4().to_string();
    let context_id = Uuid::new_v4().to_string();
    let rpc_id = req.id.clone();

    // Run the agent (same as message/send)
    tracing::info!("[a2a] handle_message_stream: dispatching...");
    let result = run_agent_task(&state.agent_config, &task_text)
        .await
        .map_err(|e| anyhow::anyhow!("Task channel error: {e}"))
        .and_then(|r| r);
    tracing::info!(
        "[a2a] handle_message_stream: got result (ok={})",
        result.is_ok()
    );

    // Build SSE body manually: "working" event + final event
    let working = serde_json::json!({
        "jsonrpc": "2.0", "id": rpc_id,
        "result": {
            "id": task_id, "contextId": context_id, "kind": "task",
            "status": { "state": "working", "timestamp": chrono::Utc::now().to_rfc3339() }
        }
    });

    let final_event = match result {
        Ok(answer) => serde_json::json!({
            "jsonrpc": "2.0", "id": rpc_id,
            "result": {
                "id": task_id, "contextId": context_id, "kind": "task",
                "status": {
                    "state": "completed",
                    "message": { "role": "agent", "parts": [{"kind": "text", "text": answer}] },
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                }
            }
        }),
        Err(e) => serde_json::json!({
            "jsonrpc": "2.0", "id": rpc_id,
            "result": {
                "id": task_id, "contextId": context_id, "kind": "task",
                "status": {
                    "state": "failed",
                    "message": { "role": "agent", "parts": [{"kind": "text", "text": format!("Agent error: {e}")}] },
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                }
            }
        }),
    };

    // Format as text/event-stream body
    let body = format!(
        "data: {}\n\ndata: {}\n\n",
        serde_json::to_string(&working).unwrap(),
        serde_json::to_string(&final_event).unwrap(),
    );

    tracing::info!(
        "[a2a] handle_message_stream: returning {} bytes SSE body.",
        body.len()
    );

    axum::http::Response::builder()
        .status(200)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .body(axum::body::Body::from(body))
        .unwrap()
        .into_response()
}

// ── Helpers ───────────────────────────────────────────────────────

/// Extract all text content from an A2A message.
fn extract_text_from_message(msg: &A2aMessage) -> String {
    msg.parts
        .iter()
        .filter_map(|part| match part {
            A2aPart::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Run the agent on a task using the monadic architecture.
///
/// The agent's internal future is not `Send` (uses `Pin<Box<dyn Future>>`),
/// so we run it on a dedicated OS thread with its own tokio runtime.
/// We use `std::thread::spawn` instead of `tokio::task::spawn_blocking`
/// to avoid PyO3 GIL deadlocks with tokio's blocking thread pool.
fn run_agent_task(
    config: &AgentConfig,
    task: &str,
) -> tokio::sync::oneshot::Receiver<anyhow::Result<String>> {
    let config = config.clone();
    let task = task.to_string();
    let (tx, rx) = tokio::sync::oneshot::channel();

    tracing::info!(
        "[a2a] Spawning agent thread for task: {}",
        &task[..task.len().min(80)]
    );

    std::thread::spawn(move || {
        tracing::info!("[a2a] Agent thread started, building runtime...");

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to build tokio runtime");

        tracing::info!("[a2a] Runtime built, running agent...");

        let result = rt.block_on(async move {
            tracing::info!("[a2a] Creating AgentContext...");
            let mut ctx = AgentContext::new_async(config).await?;

            tracing::info!("[a2a] AgentContext created, starting monadic program...");
            let program = agent_task(&task);
            let result = ctx.run(program).await;

            tracing::info!(
                "[a2a] Monadic program finished (ok={}), shutting down...",
                result.is_ok()
            );

            if let Err(e) = ctx.shutdown().await {
                tracing::warn!("[a2a] Failed to shutdown executor: {e}");
            }

            tracing::info!("[a2a] Shutdown complete.");
            result.map_err(|e| anyhow::anyhow!("{e}"))
        });

        tracing::info!(
            "[a2a] Agent thread done (ok={}), sending result.",
            result.is_ok()
        );
        let _ = tx.send(result);
    });

    rx
}

// ── Server Entry Point ────────────────────────────────────────────

/// Start the A2A HTTP server.
///
/// # Arguments
/// * `port` — Port to listen on (default: 9999)
/// * `agent_config` — Configuration for the agent (provider, executor, etc.)
pub async fn start_a2a_server(port: u16, agent_config: AgentConfig) -> anyhow::Result<()> {
    let state = Arc::new(A2aServerState::new(port, agent_config));

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_headers(Any)
        .allow_methods(Any);

    let app = Router::new()
        .route("/.well-known/agent.json", get(handle_agent_card))
        // Also serve at the newer path
        .route("/.well-known/agent-card.json", get(handle_agent_card))
        .route("/", post(handle_a2a_request))
        .layer(cors)
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("🚀 A2A server listening on http://{addr}");
    eprintln!("🚀 A2A server listening on http://{addr}");
    eprintln!("   Agent card: http://{addr}/.well-known/agent.json");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install CTRL+C signal handler");
    eprintln!("\n⚡ Shutdown signal received");
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_card_serializes_correctly() {
        let state = A2aServerState::new(9999, AgentConfig::default());
        let json = serde_json::to_string_pretty(&state.agent_card).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed["name"], "rig-rlm");
        assert_eq!(parsed["protocolVersion"], "0.2.6");
        assert_eq!(parsed["url"], "http://localhost:9999");
        assert!(parsed["skills"].is_array());
        assert_eq!(parsed["skills"].as_array().unwrap().len(), 3);
        assert_eq!(parsed["defaultInputModes"][0], "text");
        assert_eq!(parsed["capabilities"]["streaming"], true);
    }

    #[test]
    fn agent_card_roundtrips_json() {
        let state = A2aServerState::new(9999, AgentConfig::default());
        let json = serde_json::to_value(&state.agent_card).unwrap();
        let deserialized: AgentCard = serde_json::from_value(json).unwrap();
        assert_eq!(deserialized.name, "rig-rlm");
        assert_eq!(deserialized.skills.len(), 3);
    }

    #[test]
    fn extract_text_from_message_works() {
        let msg = A2aMessage {
            role: "user".to_string(),
            parts: vec![
                A2aPart::Text {
                    text: "Hello ".to_string(),
                },
                A2aPart::Text {
                    text: "world".to_string(),
                },
            ],
            metadata: None,
        };
        assert_eq!(extract_text_from_message(&msg), "Hello \nworld");
    }

    #[test]
    fn extract_text_skips_non_text_parts() {
        let msg = A2aMessage {
            role: "user".to_string(),
            parts: vec![
                A2aPart::Text {
                    text: "hello".to_string(),
                },
                A2aPart::Data {
                    data: serde_json::json!({"key": "value"}),
                },
            ],
            metadata: None,
        };
        assert_eq!(extract_text_from_message(&msg), "hello");
    }

    #[test]
    fn jsonrpc_request_parses() {
        let json = r#"{
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Hello"}]
                },
                "metadata": {}
            }
        }"#;
        let req: JsonRpcRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.method, "message/send");

        let params: MessageSendParams = serde_json::from_value(req.params).unwrap();
        assert_eq!(params.message.role, "user");
        assert_eq!(extract_text_from_message(&params.message), "Hello");
    }

    #[test]
    fn task_status_serializes() {
        let task = A2aTask {
            id: "test-1".to_string(),
            context_id: "ctx-1".to_string(),
            kind: "task".to_string(),
            status: TaskStatus {
                state: "completed".to_string(),
                message: Some(A2aMessage {
                    role: "agent".to_string(),
                    parts: vec![A2aPart::Text {
                        text: "Done!".to_string(),
                    }],
                    metadata: None,
                }),
                timestamp: None,
            },
            history: vec![],
            artifacts: vec![],
        };
        let json = serde_json::to_value(&task).unwrap();
        assert_eq!(json["status"]["state"], "completed");
        assert_eq!(json["status"]["message"]["parts"][0]["text"], "Done!");
    }
}
