//! WebSocket Channel E2E Demo
//!
//! Starts a WebSocket server on port 9090 with a simple echo agent.
//! Connect with any WebSocket client (websocat, wscat, browser):
//!
//! ```bash
//! cargo run --bin ws-channel-demo
//! # In another terminal:
//! websocat ws://localhost:9090/ws
//! > Hello agent!
//! < 🤖 Agent received your message: "Hello agent!"
//! ```
//!
//! This demonstrates the full channel pipeline:
//! 1. WebSocket client connects → WsSpoke creates connection
//! 2. Client sends text → WsSpoke publishes ChannelEvent via HubPublisher
//! 3. Agent loop receives event from ingest channel
//! 4. Agent processes event → sends reply directly to WsSpoke
//! 5. WsSpoke sends reply back through WebSocket → client

use rig_rlm::channels::hub::ChannelHub;
use rig_rlm::channels::ws::WsSpoke;
use rig_rlm::channels::{ChannelMeta, Spoke};
use std::sync::Arc;
use tracing::{info, warn};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,rig_rlm::channels=debug".parse().unwrap()),
        )
        .init();

    eprintln!("╔═══════════════════════════════════════════════════════════╗");
    eprintln!("║   rig-rlm WebSocket Channel E2E Demo                    ║");
    eprintln!("║                                                         ║");
    eprintln!("║   Connect: websocat ws://localhost:9090/ws              ║");
    eprintln!("║   or: wscat -c ws://localhost:9090/ws                   ║");
    eprintln!("║                                                         ║");
    eprintln!("║   Send any message → agent echoes back with processing  ║");
    eprintln!("║   Press Ctrl+C to stop                                  ║");
    eprintln!("╚═══════════════════════════════════════════════════════════╝");
    eprintln!();

    // 1. Create the hub and WebSocket spoke
    let hub = ChannelHub::new();
    let ws_spoke = Arc::new(WsSpoke::new(9090));

    // 2. Get a receiver for spoke-published events.
    //    HubPublisher sends to ingest_tx; subscribe_ingest() taps that channel.
    let mut rx = hub.subscribe_ingest();

    // 3. Get the publisher handle and start the WsSpoke
    let publisher = hub.publisher();
    ws_spoke.start(publisher).await?;

    info!("WebSocket server started on port 9090");
    eprintln!("✅ WebSocket server ready at ws://localhost:9090/ws");
    eprintln!();

    // 4. Agent loop: listen for spoke events and reply
    loop {
        match rx.recv().await {
            Ok(event) => {
                let source = &event.source;
                let topic = &event.topic;
                let content = &event.content;

                eprintln!("📨 Received from {source} [{topic}]: {content}");

                // Extract conn_id for replying
                let conn_id = event.meta.get("conn_id").map(|s| s.to_string());

                // Simple agent processing: echo with formatting
                let reply = format!(
                    "🤖 Agent received your message: \"{content}\"\n\
                     📊 Length: {} chars | Topic: {topic} | Source: {source}",
                    content.len()
                );

                // Send reply back directly through the WsSpoke
                if let Some(conn_id) = conn_id {
                    let meta = ChannelMeta::new()
                        .insert("conn_id", conn_id.clone());

                    match ws_spoke.reply(&meta, &reply).await {
                        Ok(()) => eprintln!("📤 Reply sent to conn {conn_id}"),
                        Err(e) => warn!(%e, "Failed to send reply"),
                    }
                }
            }
            Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                warn!(n, "Agent lagged behind by {n} messages");
            }
            Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                info!("Channel hub closed, shutting down");
                break;
            }
        }
    }

    Ok(())
}
