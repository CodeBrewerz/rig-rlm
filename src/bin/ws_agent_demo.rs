//! Interactive Coding Agent over WebSocket Channel
//!
//! Demonstrates a coding agent that:
//! 1. Receives a task from the user via WebSocket
//! 2. Writes code (generates it)
//! 3. Asks clarifying questions back to the user
//! 4. User replies via WebSocket
//! 5. Agent finalizes the code, runs it, and sends the result
//!
//! ```bash
//! cargo run --bin ws-agent-demo
//! # In another terminal:
//! websocat ws://localhost:9090/ws
//! > write a fibonacci function
//! < 🤖 Working on: "write a fibonacci function"
//! < 📝 Here's my initial plan...
//! < ❓ How many terms should I compute? (default: 10)
//! > 15
//! < 💻 Writing code...
//! < 🔧 Running fibonacci(15)...
//! < ✅ Result: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
//! ```

use rig_rlm::channels::hub::ChannelHub;
use rig_rlm::channels::ws::WsSpoke;
use rig_rlm::channels::{ChannelMeta, Spoke};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Session state for each connected user.
#[derive(Debug, Clone)]
enum SessionState {
    /// Waiting for a task from the user.
    Idle,
    /// Agent asked a question, waiting for the user's reply.
    WaitingForReply {
        task: String,
        question: String,
    },
}

/// Simple agent that simulates coding tasks.
struct CodingAgent {
    sessions: HashMap<String, SessionState>,
}

impl CodingAgent {
    fn new() -> Self {
        Self {
            sessions: HashMap::new(),
        }
    }

    /// Process a message from a user and return a sequence of replies.
    fn process(&mut self, conn_id: &str, message: &str) -> Vec<String> {
        let state = self
            .sessions
            .get(conn_id)
            .cloned()
            .unwrap_or(SessionState::Idle);

        match state {
            SessionState::Idle => {
                // New task from user
                self.handle_new_task(conn_id, message)
            }
            SessionState::WaitingForReply { task, question } => {
                // User replied to our question
                self.handle_reply(conn_id, message, &task, &question)
            }
        }
    }

    fn handle_new_task(&mut self, conn_id: &str, task: &str) -> Vec<String> {
        let task_lower = task.to_lowercase();
        let mut replies = Vec::new();

        replies.push(format!("🤖 Working on: \"{task}\""));

        if task_lower.contains("fibonacci") || task_lower.contains("fib") {
            replies.push("📝 Plan: Write a Fibonacci sequence generator in Rust".to_string());
            replies.push(
                "❓ How many terms should I compute? (just type a number, default: 10)"
                    .to_string(),
            );
            self.sessions.insert(
                conn_id.to_string(),
                SessionState::WaitingForReply {
                    task: task.to_string(),
                    question: "fib_count".to_string(),
                },
            );
        } else if task_lower.contains("sort") {
            replies.push("📝 Plan: Implement a sorting algorithm".to_string());
            replies.push(
                "❓ Which algorithm? (bubble, merge, quick) — just type the name"
                    .to_string(),
            );
            self.sessions.insert(
                conn_id.to_string(),
                SessionState::WaitingForReply {
                    task: task.to_string(),
                    question: "sort_algo".to_string(),
                },
            );
        } else if task_lower.contains("hello") || task_lower.contains("hi") {
            replies.push("👋 Hello! I'm the rig-rlm coding agent.".to_string());
            replies.push("💡 Try asking me to:".to_string());
            replies.push("   • \"write a fibonacci function\"".to_string());
            replies.push("   • \"implement a sorting algorithm\"".to_string());
            replies.push("   • \"calculate factorial of 10\"".to_string());
        } else if task_lower.contains("factorial") {
            // Extract number if present
            let n = extract_number(task).unwrap_or(10);
            replies.push(format!("📝 Plan: Calculate factorial of {n}"));
            replies.push(format!("💻 Writing code...\n```rust\nfn factorial(n: u64) -> u64 {{\n    (1..=n).product()\n}}\n```"));

            let result = factorial(n as u64);

            replies.push(format!("🔧 Running factorial({n})..."));
            replies.push(format!("✅ Result: {n}! = {result}"));
            self.sessions.insert(conn_id.to_string(), SessionState::Idle);
        } else {
            replies.push("🤔 I can help with coding tasks! Try:".to_string());
            replies.push("   • \"write a fibonacci function\"".to_string());
            replies.push("   • \"implement a sorting algorithm\"".to_string());
            replies.push("   • \"calculate factorial of 10\"".to_string());
        }

        replies
    }

    fn handle_reply(
        &mut self,
        conn_id: &str,
        reply: &str,
        task: &str,
        question: &str,
    ) -> Vec<String> {
        let mut replies = Vec::new();

        match question {
            "fib_count" => {
                let n = extract_number(reply).unwrap_or(10);
                replies.push(format!("📬 Got it! Computing {n} Fibonacci terms."));

                // "Write" the code
                replies.push(format!(
                    "💻 Writing code...\n```rust\nfn fibonacci(n: usize) -> Vec<u64> {{\n    let mut seq = vec![0u64; n];\n    if n > 1 {{ seq[1] = 1; }}\n    for i in 2..n {{\n        seq[i] = seq[i-1] + seq[i-2];\n    }}\n    seq\n}}\n```"
                ));

                // "Run" the code
                let fib = fibonacci(n);
                replies.push(format!("🔧 Running fibonacci({n})..."));
                replies.push(format!("✅ Result: {:?}", fib));

                // Reset to idle
                self.sessions.insert(conn_id.to_string(), SessionState::Idle);
                replies.push("💡 Send another task, or type 'hi' for help.".to_string());
            }
            "sort_algo" => {
                let algo = reply.trim().to_lowercase();
                let algo_name = match algo.as_str() {
                    "quick" => "quicksort",
                    "merge" => "merge sort",
                    "bubble" => "bubble sort",
                    _ => "quicksort (defaulting)",
                };

                replies.push(format!("📬 Got it! Implementing {algo_name}."));

                let data = vec![38, 27, 43, 3, 9, 82, 10];

                // "Write" the code
                replies.push(format!(
                    "💻 Writing code...\n```rust\nfn {algo}(arr: &mut [i32]) {{\n    // ... {algo_name} implementation ...\n    arr.sort(); // using stdlib for demo\n}}\n```",
                    algo = algo.replace(' ', "_")
                ));

                // "Run" the code
                let mut sorted_data = data.clone();
                sorted_data.sort();
                replies.push(format!("🔧 Running {algo_name} on {:?}...", data));
                replies.push(format!("✅ Sorted: {:?}", sorted_data));

                self.sessions.insert(conn_id.to_string(), SessionState::Idle);
                replies.push("💡 Send another task, or type 'hi' for help.".to_string());
            }
            _ => {
                replies.push(format!("📬 Got your reply: \"{reply}\""));
                replies.push(format!("🤖 Continuing with task: \"{task}\""));
                self.sessions.insert(conn_id.to_string(), SessionState::Idle);
            }
        }

        replies
    }
}

fn extract_number(text: &str) -> Option<usize> {
    text.split_whitespace()
        .filter_map(|w| w.trim_matches(|c: char| !c.is_ascii_digit()).parse::<usize>().ok())
        .next()
}

fn fibonacci(n: usize) -> Vec<u64> {
    if n == 0 {
        return vec![];
    }
    let mut seq = vec![0u64; n];
    if n > 1 {
        seq[1] = 1;
    }
    for i in 2..n {
        seq[i] = seq[i - 1] + seq[i - 2];
    }
    seq
}

fn factorial(n: u64) -> u64 {
    (1..=n).product()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,rig_rlm::channels=debug".parse().unwrap()),
        )
        .init();

    eprintln!("╔═══════════════════════════════════════════════════════════╗");
    eprintln!("║   rig-rlm Interactive Coding Agent — WebSocket Demo      ║");
    eprintln!("║                                                         ║");
    eprintln!("║   Connect: websocat ws://localhost:9090/ws              ║");
    eprintln!("║                                                         ║");
    eprintln!("║   Try: \"write a fibonacci function\"                     ║");
    eprintln!("║        \"implement a sorting algorithm\"                  ║");
    eprintln!("║        \"calculate factorial of 12\"                      ║");
    eprintln!("║                                                         ║");
    eprintln!("║   The agent will ask questions, you reply, it codes!    ║");
    eprintln!("║   Press Ctrl+C to stop                                  ║");
    eprintln!("╚═══════════════════════════════════════════════════════════╝");
    eprintln!();

    let hub = ChannelHub::new();
    let ws_spoke = Arc::new(WsSpoke::new(9090));
    let mut rx = hub.subscribe_ingest();
    let publisher = hub.publisher();
    ws_spoke.start(publisher).await?;

    eprintln!("✅ WebSocket server ready at ws://localhost:9090/ws");
    eprintln!();

    // Shared agent state
    let agent = Arc::new(RwLock::new(CodingAgent::new()));

    loop {
        match rx.recv().await {
            Ok(event) => {
                let content = &event.content;
                let conn_id = event.meta.get("conn_id").map(|s| s.to_string());

                if let Some(conn_id) = conn_id {
                    eprintln!("📨 [conn {conn_id}]: {content}");

                    // Process through the coding agent
                    let replies = {
                        let mut agent = agent.write().await;
                        agent.process(&conn_id, content)
                    };

                    // Send each reply with a small delay for readability
                    let meta = ChannelMeta::new().insert("conn_id", conn_id.clone());

                    for (i, reply) in replies.iter().enumerate() {
                        if i > 0 {
                            // Small delay between messages for readability
                            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                        }
                        match ws_spoke.reply(&meta, reply).await {
                            Ok(()) => eprintln!("📤 [conn {conn_id}]: {}", reply.lines().next().unwrap_or("")),
                            Err(e) => warn!(%e, "Failed to send reply"),
                        }
                    }
                }
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
