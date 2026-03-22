//! WsSpoke — bidirectional WebSocket channel for real-time agent interaction.
//!
//! Uses axum's WebSocket upgrade to provide a persistent connection where:
//! - **Inbound**: Client sends text → ChannelEvent → Hub → Agent
//! - **Outbound**: Agent reply → Hub → WsSpoke → WebSocket → Client
//!
//! ## Usage
//!
//! ```bash
//! # Agent starts with WebSocket channel on port 9090
//! # Connect via any WebSocket client:
//! websocat ws://localhost:9090/ws
//! > Hello agent!
//! < [agent reply]
//! ```
//!
//! ## Topic Mapping
//!
//! All WebSocket messages use topic `ws/interactive/{connection_id}`.

use async_trait::async_trait;
use axum::extract::ws::{Message, WebSocket};
use axum::extract::WebSocketUpgrade;
use axum::response::IntoResponse;
use axum::routing::get;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex, RwLock};
use tracing::{debug, info, warn};

use super::{ChannelEvent, ChannelMeta, HubPublisher, Spoke};

/// Global connection counter for unique IDs.
static CONN_COUNTER: AtomicU64 = AtomicU64::new(1);

/// WsSpoke — bidirectional WebSocket channel.
///
/// Runs an axum HTTP server with a `/ws` upgrade endpoint.
/// Each connection gets a unique ID and communicates with the agent
/// through the channel hub.
pub struct WsSpoke {
    port: u16,
    /// Active connections: connection_id → sender for outbound messages.
    connections: Arc<RwLock<HashMap<u64, tokio::sync::mpsc::UnboundedSender<String>>>>,
}

impl WsSpoke {
    /// Create a new WebSocket spoke on the specified port.
    pub fn new(port: u16) -> Self {
        Self {
            port,
            connections: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get a handle to the active connections map (for testing).
    pub fn connections(
        &self,
    ) -> Arc<RwLock<HashMap<u64, tokio::sync::mpsc::UnboundedSender<String>>>> {
        Arc::clone(&self.connections)
    }
}

#[async_trait]
impl Spoke for WsSpoke {
    fn name(&self) -> &str {
        "ws"
    }

    fn instructions(&self) -> &str {
        "Real-time WebSocket messages arrive as <channel source=\"ws\" \
         topic=\"ws/interactive/{conn_id}\" conn_id=\"...\">message</channel>.\n\
         To reply, use channel_reply with spoke=\"ws\" and include conn_id in meta.\n\
         The reply text will be sent back to the connected WebSocket client."
    }

    fn supports_reply(&self) -> bool {
        true
    }

    async fn reply(&self, meta: &ChannelMeta, text: &str) -> anyhow::Result<()> {
        let conn_id_str = meta
            .get("conn_id")
            .ok_or_else(|| anyhow::anyhow!("Missing conn_id in meta for WS reply"))?;

        let conn_id: u64 = conn_id_str
            .parse()
            .map_err(|_| anyhow::anyhow!("Invalid conn_id: {conn_id_str}"))?;

        let conns = self.connections.read().await;
        if let Some(tx) = conns.get(&conn_id) {
            tx.send(text.to_string())
                .map_err(|_| anyhow::anyhow!("WebSocket connection {conn_id} closed"))?;
            debug!(conn_id, len = text.len(), "WS reply sent");
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "WebSocket connection {conn_id} not found"
            ))
        }
    }

    async fn start(&self, publisher: HubPublisher) -> anyhow::Result<()> {
        let port = self.port;
        let connections = Arc::clone(&self.connections);
        let publisher = Arc::new(publisher);

        info!(port, "WsSpoke starting WebSocket server on /ws");

        tokio::spawn(async move {
            let publisher = Arc::clone(&publisher);
            let connections = Arc::clone(&connections);

            let app = axum::Router::new().route(
                "/ws",
                get(move |ws: WebSocketUpgrade| {
                    let publisher = Arc::clone(&publisher);
                    let connections = Arc::clone(&connections);
                    async move { ws_upgrade(ws, publisher, connections).await }
                }),
            );

            let bind_addr = format!("0.0.0.0:{port}");
            let listener = match tokio::net::TcpListener::bind(&bind_addr).await {
                Ok(l) => l,
                Err(e) => {
                    tracing::error!(%e, %bind_addr, "WsSpoke failed to bind");
                    return;
                }
            };

            info!(%bind_addr, "WsSpoke WebSocket server listening");
            if let Err(e) = axum::serve(listener, app).await {
                tracing::error!(%e, "WsSpoke server error");
            }
        });

        Ok(())
    }
}

/// Handle WebSocket upgrade — spawn per-connection reader/writer tasks.
async fn ws_upgrade(
    ws: WebSocketUpgrade,
    publisher: Arc<HubPublisher>,
    connections: Arc<RwLock<HashMap<u64, tokio::sync::mpsc::UnboundedSender<String>>>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws(socket, publisher, connections))
}

/// Handle a single WebSocket connection.
async fn handle_ws(
    socket: WebSocket,
    publisher: Arc<HubPublisher>,
    connections: Arc<RwLock<HashMap<u64, tokio::sync::mpsc::UnboundedSender<String>>>>,
) {
    use futures_util::{SinkExt, StreamExt};

    let conn_id = CONN_COUNTER.fetch_add(1, Ordering::Relaxed);
    let topic = format!("ws/interactive/{conn_id}");

    info!(conn_id, "New WebSocket connection");

    // Split socket into reader and writer
    let (mut ws_sender, mut ws_receiver) = socket.split();

    // Create channel for outbound messages (agent → client)
    let (out_tx, mut out_rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    // Register this connection
    {
        let mut conns = connections.write().await;
        conns.insert(conn_id, out_tx);
    }

    // Send welcome message
    let welcome = format!(
        "🔌 Connected to rig-rlm agent (conn_id: {conn_id}). Send messages to interact."
    );
    let _ = ws_sender.send(Message::Text(welcome.into())).await;

    // Writer task: forward outbound messages from agent to WebSocket
    let write_handle = tokio::spawn(async move {
        while let Some(text) = out_rx.recv().await {
            if ws_sender
                .send(Message::Text(text.into()))
                .await
                .is_err()
            {
                break;
            }
        }
    });

    // Reader loop: forward inbound messages from WebSocket to hub
    while let Some(msg) = ws_receiver.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                let text = text.to_string();
                if text.is_empty() {
                    continue;
                }
                debug!(conn_id, len = text.len(), "WS message received");

                let meta = ChannelMeta::new()
                    .insert("conn_id", conn_id.to_string())
                    .insert("sender_id", format!("ws-{conn_id}"));

                let event = ChannelEvent::new("ws", &topic, text).with_meta(meta);
                publisher.publish(event);
            }
            Ok(Message::Close(_)) => {
                info!(conn_id, "WebSocket client disconnected");
                break;
            }
            Ok(Message::Ping(_)) => {
                // axum handles pong automatically
            }
            Err(e) => {
                warn!(conn_id, %e, "WebSocket error");
                break;
            }
            _ => {} // Binary, Pong — ignore
        }
    }

    // Cleanup
    {
        let mut conns = connections.write().await;
        conns.remove(&conn_id);
    }
    write_handle.abort();
    info!(conn_id, "WebSocket connection closed");
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ws_spoke_name() {
        let spoke = WsSpoke::new(9090);
        assert_eq!(spoke.name(), "ws");
        assert!(spoke.supports_reply());
    }

    #[test]
    fn ws_spoke_instructions() {
        let spoke = WsSpoke::new(9090);
        assert!(spoke.instructions().contains("ws/interactive"));
        assert!(spoke.instructions().contains("conn_id"));
    }
}
