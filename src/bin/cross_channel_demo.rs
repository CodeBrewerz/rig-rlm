//! Cross-Channel LLM Coding Agent — Telegram + WebSocket
//!
//! A **real** coding agent that handles ANY programming question:
//! - Uses OpenRouter/OpenAI LLM for understanding and code generation
//! - Uses Pyo3Executor for running Python code
//! - Streams progress messages back to Telegram/WebSocket as it works
//! - Maintains per-user conversation history
//!
//! ## Setup
//!
//! 1. Set `OPENAI_API_KEY` in `.env` (OpenRouter or OpenAI key)
//! 2. Set `TELOXIDE_TOKEN` env var for Telegram
//! 3. `cargo run --bin cross-channel-demo`

use rig::{
    OneOrMany,
    agent::Text,
    client::{CompletionClient, ProviderClient},
    completion::Chat,
    message::{AssistantContent, Message, UserContent},
    providers::openai::CompletionModel,
};
use rig_rlm::channels::hub::ChannelHub;
use rig_rlm::channels::telegram::TelegramSpoke;
use rig_rlm::channels::ws::WsSpoke;
use rig_rlm::channels::{ChannelMeta, Spoke};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

// ── Coding agent preamble ────────────────────────────────────────

const CODING_PREAMBLE: &str = r#"You are a coding assistant that helps users solve programming problems. You communicate via a chat channel (Telegram or WebSocket).

Your workflow for each coding request:
1. UNDERSTAND the problem. If anything is ambiguous, ask ONE clear clarifying question.
2. PLAN your approach in 1-2 sentences.
3. WRITE the code. Put the complete solution in a Python code block.
4. EXPLAIN the solution briefly.

Format rules:
- Use ```python code blocks for executable code
- For clarifying questions, prefix with "❓ "
- For plans, prefix with "📝 "
- For results/output, prefix with "✅ "
- Keep responses concise — this is a chat, not an essay
- Always provide COMPLETE, RUNNABLE code (not pseudo-code)

Example interaction:
User: "solve n-queens"
You: "❓ What board size N should I solve for? (default: 8)"
User: "6"
You: "📝 I'll implement N-Queens using backtracking for N=6.

```python
def solve_n_queens(n):
    solutions = []
    def backtrack(row, cols, diag1, diag2, board):
        if row == n:
            solutions.append([''.join(r) for r in board])
            return
        for col in range(n):
            if col in cols or (row-col) in diag1 or (row+col) in diag2:
                continue
            board[row][col] = 'Q'
            backtrack(row+1, cols|{col}, diag1|{row-col}, diag2|{row+col}, board)
            board[row][col] = '.'
    backtrack(0, set(), set(), set(), [['.']*n for _ in range(n)])
    return solutions

results = solve_n_queens(6)
print(f'Found {len(results)} solutions for 6-Queens')
for i, sol in enumerate(results[:3]):
    print(f'\nSolution {i+1}:')
    for row in sol:
        print(row)
```

✅ This backtracking approach places queens row by row, pruning invalid positions."

IMPORTANT: Always include executable code that produces output via print()."#;

// ── Session with conversation history ────────────────────────────

#[derive(Clone, Debug)]
struct Session {
    source: String,
    reply_meta: ChannelMeta,
    history: Vec<Message>,
}

// ── Reply helper ─────────────────────────────────────────────────

async fn send_reply(
    ws: &WsSpoke,
    tg: &Option<Arc<TelegramSpoke>>,
    session: &Session,
    text: &str,
) {
    match session.source.as_str() {
        "ws" => {
            ws.reply(&session.reply_meta, text).await.ok();
        }
        "telegram" => {
            if let Some(tg) = tg {
                if let Err(e) = tg.reply(&session.reply_meta, text).await {
                    error!(%e, "Telegram reply failed");
                }
            }
        }
        _ => {}
    }
}

// ── Code extraction + execution ──────────────────────────────────

fn extract_python_code(text: &str) -> Option<String> {
    // Look for ```python ... ``` blocks
    let mut in_block = false;
    let mut code_lines = Vec::new();

    for line in text.lines() {
        if line.trim().starts_with("```python") {
            in_block = true;
            continue;
        }
        if line.trim() == "```" && in_block {
            break;
        }
        if in_block {
            code_lines.push(line);
        }
    }

    if code_lines.is_empty() {
        None
    } else {
        Some(code_lines.join("\n"))
    }
}

fn execute_python(code: &str) -> Result<String, String> {
    use pyo3::Python;
    use pyo3::types::{PyAnyMethods, PyModuleMethods, PyDict};
    use std::ffi::CString;

    let result: Result<String, pyo3::PyErr> = Python::attach(|py| {
        let io = py.import("io")?;
        let sys = py.import("sys")?;

        let string_io = io.call_method0("StringIO")?;
        sys.setattr("stdout", &string_io)?;

        let locals = PyDict::new(py);
        let code_c = CString::new(code).map_err(|e| {
            pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
        })?;

        if let Err(e) = py.run(&code_c, None, Some(&locals)) {
            return Ok(format!("⚠️ Error:\n{e}"));
        }

        let output = string_io.call_method0("getvalue")?;
        let out_str = output.to_string();

        if out_str.trim().is_empty() {
            Ok("(no output)".to_string())
        } else {
            Ok(out_str)
        }
    });
    result.map_err(|e| e.to_string())
}

// ── LLM call ─────────────────────────────────────────────────────

fn build_agent() -> rig::agent::Agent<CompletionModel> {
    dotenvy::dotenv().ok();

    // Check if OPENAI_API_KEY looks like OpenRouter
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    let is_openrouter = api_key.starts_with("sk-or-");

    if is_openrouter {
        let client = rig::providers::openai::Client::<reqwest::Client>::builder()
            .base_url("https://openrouter.ai/api/v1")
            .api_key(&api_key)
            .http_client(reqwest::Client::new())
            .build()
            .unwrap();

        client
            .completion_model("arcee-ai/trinity-large-preview:free")
            .completions_api()
            .into_agent_builder()
            .preamble(CODING_PREAMBLE)
            .build()
    } else {
        let client = rig::providers::openai::Client::<reqwest::Client>::from_env();
        client
            .completion_model("gpt-4o")
            .completions_api()
            .into_agent_builder()
            .preamble(CODING_PREAMBLE)
            .build()
    }
}

async fn llm_chat(
    agent: &rig::agent::Agent<CompletionModel>,
    prompt: &str,
    history: &[Message],
) -> Result<String, String> {
    agent
        .chat(prompt.to_string(), history.to_vec())
        .await
        .map_err(|e| format!("LLM error: {e}"))
}

// ── Process a message with the real agent ────────────────────────

const MAX_FIX_ATTEMPTS: usize = 3;

async fn process_message(
    ws: &WsSpoke,
    tg: &Option<Arc<TelegramSpoke>>,
    agent: &rig::agent::Agent<CompletionModel>,
    session: &mut Session,
    content: &str,
) {
    // Send "thinking" indicator
    send_reply(ws, tg, session, "🤔 Thinking...").await;

    // Call the LLM with conversation history
    let response = match llm_chat(agent, content, &session.history).await {
        Ok(r) => r,
        Err(e) => {
            send_reply(ws, tg, session, &format!("❌ LLM error: {e}")).await;
            return;
        }
    };

    // Update history with the user message + first LLM response
    session.history.push(Message::User {
        content: OneOrMany::one(UserContent::Text(Text {
            text: content.to_string(),
        })),
    });
    session.history.push(Message::Assistant {
        content: OneOrMany::one(AssistantContent::Text(Text {
            text: response.clone(),
        })),
        id: None,
    });

    // Check if response contains executable Python code
    let mut current_response = response;

    if extract_python_code(&current_response).is_none() {
        // No code — just a clarifying question or text response
        send_reply(ws, tg, session, &current_response).await;
        trim_history(session);
        return;
    }

    // ── Agentic code-execute-fix loop ────────────────────────────
    for attempt in 0..=MAX_FIX_ATTEMPTS {
        let code = match extract_python_code(&current_response) {
            Some(c) => c,
            None => {
                // LLM replied without code (maybe gave up or answered differently)
                send_reply(ws, tg, session, &current_response).await;
                break;
            }
        };

        if attempt == 0 {
            // First attempt: show the full plan + code
            send_reply(ws, tg, session, &current_response).await;
            send_reply(ws, tg, session, "🔧 Running code...").await;
        } else {
            // Fix attempt: show what the LLM changed
            send_reply(
                ws, tg, session,
                &format!("🔄 Fix attempt {attempt}/{MAX_FIX_ATTEMPTS}:\n{current_response}"),
            ).await;
            send_reply(ws, tg, session, "🔧 Re-running...").await;
        }

        // Execute Python code
        let code_clone = code.clone();
        let exec_result = tokio::task::spawn_blocking(move || {
            execute_python(&code_clone)
        })
        .await
        .unwrap_or_else(|e| Err(format!("Execution panicked: {e}")));

        match exec_result {
            Ok(ref output) if !output.contains("⚠️ Error:") => {
                // ✅ Success! Send the result and stop
                let truncated = if output.len() > 3000 {
                    format!("{}...\n(truncated)", &output[..3000])
                } else {
                    output.clone()
                };
                send_reply(ws, tg, session, &format!("✅ Output:\n{truncated}")).await;
                break;
            }
            Ok(error_output) | Err(error_output) => {
                // ❌ Error — feed it back to the LLM for correction
                if attempt >= MAX_FIX_ATTEMPTS {
                    send_reply(
                        ws, tg, session,
                        &format!("❌ Failed after {MAX_FIX_ATTEMPTS} fix attempts.\nLast error:\n{error_output}"),
                    ).await;
                    break;
                }

                let fix_prompt = format!(
                    "Your Python code produced this error:\n\n{error_output}\n\n\
                     Fix the code. Return the COMPLETE corrected code in a ```python block. \
                     Make sure all functions are defined before they are called. \
                     The code must be self-contained and runnable."
                );

                send_reply(
                    ws, tg, session,
                    &format!("⚠️ Error detected — asking LLM to fix...\n```\n{error_output}\n```"),
                ).await;

                // Add the error as user feedback in conversation history
                session.history.push(Message::User {
                    content: OneOrMany::one(UserContent::Text(Text {
                        text: fix_prompt.clone(),
                    })),
                });

                // Ask LLM to fix
                match llm_chat(agent, &fix_prompt, &session.history).await {
                    Ok(fix_response) => {
                        session.history.push(Message::Assistant {
                            content: OneOrMany::one(AssistantContent::Text(Text {
                                text: fix_response.clone(),
                            })),
                            id: None,
                        });
                        current_response = fix_response;
                        // Loop continues to extract + execute the fixed code
                    }
                    Err(e) => {
                        send_reply(ws, tg, session, &format!("❌ LLM fix failed: {e}")).await;
                        break;
                    }
                }
            }
        }
    }

    trim_history(session);
}

fn trim_history(session: &mut Session) {
    if session.history.len() > 20 {
        session.history.drain(0..session.history.len() - 20);
    }
}

// ── Main ─────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,rig_rlm::channels=debug".parse().unwrap()),
        )
        .init();

    dotenvy::dotenv().ok();

    let has_telegram = std::env::var("TELOXIDE_TOKEN").is_ok();
    let has_openai = std::env::var("OPENAI_API_KEY").is_ok();

    eprintln!("╔═══════════════════════════════════════════════════════════╗");
    eprintln!("║   rig-rlm LLM Coding Agent (Telegram + WebSocket)       ║");
    eprintln!("║                                                         ║");
    eprintln!("║   WebSocket: websocat ws://localhost:9090/ws            ║");
    if has_telegram {
        eprintln!("║   Telegram:  ✅ Bot active                              ║");
    } else {
        eprintln!("║   Telegram:  ⚠️  Set TELOXIDE_TOKEN to enable          ║");
    }
    if has_openai {
        eprintln!("║   LLM:       ✅ OPENAI_API_KEY found                   ║");
    } else {
        eprintln!("║   LLM:       ❌ Set OPENAI_API_KEY in .env             ║");
    }
    eprintln!("║                                                         ║");
    eprintln!("║   Ask ANY coding question! Examples:                    ║");
    eprintln!("║     \"solve n-queens for board size 6\"                    ║");
    eprintln!("║     \"write a binary search in python\"                   ║");
    eprintln!("║     \"implement dijkstra's shortest path\"                ║");
    eprintln!("║     \"reverse a linked list\"                             ║");
    eprintln!("║                                                         ║");
    eprintln!("║   The agent will ask clarifying questions if needed,    ║");
    eprintln!("║   write code, execute it, and send results back!        ║");
    eprintln!("║                                                         ║");
    eprintln!("║   Press Ctrl+C to stop                                  ║");
    eprintln!("╚═══════════════════════════════════════════════════════════╝");
    eprintln!();

    if !has_openai {
        eprintln!("❌ OPENAI_API_KEY not found. Add it to .env file.");
        std::process::exit(1);
    }

    // Build the LLM agent
    let agent = build_agent();
    eprintln!("✅ LLM agent initialized");

    // ── Build hub ────────────────────────────────────────────────
    let mut hub = ChannelHub::new();
    let (_all_id, _) = hub.subscribe("**");

    let ws_spoke = Arc::new(WsSpoke::new(9090));
    hub.add_spoke(ws_spoke.clone());

    let tg_spoke: Option<Arc<TelegramSpoke>> = if has_telegram {
        let spoke = Arc::new(TelegramSpoke::from_env());
        hub.add_spoke(spoke.clone());
        info!("Telegram spoke registered");
        Some(spoke)
    } else {
        warn!("TELOXIDE_TOKEN not set — running WebSocket only");
        None
    };

    let mut ingest_rx = hub.subscribe_ingest();
    let publisher = hub.publisher();

    ws_spoke.start(publisher.clone()).await?;
    eprintln!("✅ WebSocket spoke started on port 9090");

    if let Some(ref tg) = tg_spoke {
        tg.start(publisher.clone()).await?;
        eprintln!("✅ Telegram spoke started (long-polling)");
    }

    eprintln!();
    eprintln!("🚀 Ready! Send any coding question from Telegram or WebSocket.");
    eprintln!();

    // ── Session state per-user ───────────────────────────────────
    let sessions: Arc<RwLock<HashMap<String, Session>>> =
        Arc::new(RwLock::new(HashMap::new()));

    // ── Main agent loop ──────────────────────────────────────────
    loop {
        match ingest_rx.recv().await {
            Ok(event) => {
                let source = event.source.clone();
                let content = event.content.trim().to_string();
                let meta = event.meta.clone();

                // Skip empty or /start
                if content.is_empty() || content == "/start" {
                    if source == "telegram" {
                        if let Some(chat_id) = meta.get("chat_id") {
                            let session = Session {
                                source: "telegram".to_string(),
                                reply_meta: ChannelMeta::new()
                                    .insert("chat_id", chat_id.to_string()),
                                history: Vec::new(),
                            };
                            send_reply(&ws_spoke, &tg_spoke, &session,
                                "👋 Welcome to rig-rlm Coding Agent!\n\n\
                                 Ask me ANY coding question:\n\
                                 • \"solve n-queens for 8\"\n\
                                 • \"write dijkstra's algorithm\"\n\
                                 • \"implement a binary search tree\"\n\
                                 • \"calculate fibonacci with memoization\"\n\n\
                                 I'll write code, run it, and show results!"
                            ).await;
                        }
                    }
                    continue;
                }

                // Handle /clear command
                if content == "/clear" {
                    let session_key = match source.as_str() {
                        "ws" => format!("ws:{}", meta.get("conn_id").unwrap_or(&"0")),
                        "telegram" => format!("tg:{}", meta.get("chat_id").unwrap_or(&"0")),
                        _ => continue,
                    };
                    sessions.write().await.remove(&session_key);
                    let reply_meta = match source.as_str() {
                        "ws" => ChannelMeta::new().insert("conn_id", meta.get("conn_id").unwrap_or(&"0").to_string()),
                        _ => ChannelMeta::new().insert("chat_id", meta.get("chat_id").unwrap_or(&"0").to_string()),
                    };
                    let tmp_session = Session { source: source.clone(), reply_meta, history: vec![] };
                    send_reply(&ws_spoke, &tg_spoke, &tmp_session, "🧹 Conversation cleared!").await;
                    continue;
                }

                // Build session key
                let session_key = match source.as_str() {
                    "ws" => {
                        let conn_id = meta.get("conn_id").map(|s| s.to_string()).unwrap_or_default();
                        format!("ws:{conn_id}")
                    }
                    "telegram" => {
                        let chat_id = meta.get("chat_id").map(|s| s.to_string()).unwrap_or_default();
                        format!("tg:{chat_id}")
                    }
                    _ => continue,
                };

                eprintln!("📨 [{source}] {session_key}: {content}");

                // Get or create session
                let conn_id = meta.get("conn_id").map(|s| s.to_string()).unwrap_or_else(|| "0".to_string());
                let chat_id = meta.get("chat_id").map(|s| s.to_string()).unwrap_or_else(|| "0".to_string());

                let mut sessions_w = sessions.write().await;
                let session = sessions_w.entry(session_key.clone()).or_insert_with(|| {
                    let reply_meta = match source.as_str() {
                        "ws" => ChannelMeta::new().insert("conn_id", conn_id.clone()),
                        "telegram" => ChannelMeta::new().insert("chat_id", chat_id.clone()),
                        _ => ChannelMeta::new(),
                    };
                    Session {
                        source: source.clone(),
                        reply_meta,
                        history: Vec::new(),
                    }
                });

                // Process with the real LLM agent
                process_message(&ws_spoke, &tg_spoke, &agent, session, &content).await;
            }
            Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                warn!(n, "Lagged");
            }
            Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                info!("Hub closed");
                break;
            }
        }
    }

    Ok(())
}
