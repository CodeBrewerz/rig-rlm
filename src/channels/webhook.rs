//! WebhookSpoke — HTTP listener channel (one-way, push only).
//!
//! Runs an axum HTTP server that receives POST requests and publishes
//! them as `ChannelEvent`s into the hub.
//!
//! ## Topic mapping
//!
//! The URL path maps to the topic:
//! - `POST /`                → topic = `webhook/raw` (default)
//! - `POST /ci/build/main`   → topic = `ci/build/main`
//! - Header `X-Topic: foo`   → topic = `foo` (override)
//!
//! ## Example
//!
//! ```bash
//! curl -X POST localhost:8788 -d "build failed on main"
//! curl -X POST localhost:8788/ci/build/main -d '{"status":"failed"}'
//! curl -H "X-Topic: alerts/critical" -X POST localhost:8788 -d "disk full"
//! ```

use async_trait::async_trait;
use axum::{
    body::Bytes,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    routing::post,
    Router,
};
use std::sync::Arc;
use tracing::{info, warn};

use super::{ChannelEvent, ChannelMeta, HubPublisher, Spoke};

/// WebhookSpoke — receives HTTP POSTs and publishes them as channel events.
pub struct WebhookSpoke {
    /// Display name for this spoke.
    name: String,
    /// Port to listen on.
    port: u16,
    /// Host to bind to.
    host: String,
    /// Default topic for events without a path or X-Topic header.
    default_topic: String,
}

impl WebhookSpoke {
    /// Create a new webhook spoke on the given port.
    pub fn new(port: u16) -> Self {
        Self {
            name: "webhook".to_string(),
            port,
            host: "0.0.0.0".to_string(),
            default_topic: "webhook/raw".to_string(),
        }
    }

    /// Set a custom name for this spoke.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set a custom default topic.
    pub fn with_default_topic(mut self, topic: impl Into<String>) -> Self {
        self.default_topic = topic.into();
        self
    }

    /// Set a custom bind host.
    pub fn with_host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }
}

/// Shared state for the axum handler.
#[derive(Clone)]
struct WebhookState {
    publisher: HubPublisher,
    spoke_name: String,
    default_topic: String,
}

#[async_trait]
impl Spoke for WebhookSpoke {
    fn name(&self) -> &str {
        &self.name
    }

    fn instructions(&self) -> &str {
        "Webhook events arrive via HTTP POST. They are one-way — you cannot reply \
         to webhook events. Events include the request body as content and may \
         include metadata from headers."
    }

    fn supports_reply(&self) -> bool {
        false
    }

    async fn reply(&self, _meta: &ChannelMeta, _text: &str) -> anyhow::Result<()> {
        Err(anyhow::anyhow!("WebhookSpoke does not support replies"))
    }

    async fn start(&self, publisher: HubPublisher) -> anyhow::Result<()> {
        let state = WebhookState {
            publisher,
            spoke_name: self.name.clone(),
            default_topic: self.default_topic.clone(),
        };

        let app = Router::new()
            .route("/", post(handle_webhook_root))
            .route("/{*path}", post(handle_webhook_path))
            .with_state(state);

        let bind_addr = format!("{}:{}", self.host, self.port);
        let listener = tokio::net::TcpListener::bind(&bind_addr).await?;

        info!(
            spoke = %self.name,
            addr = %bind_addr,
            default_topic = %self.default_topic,
            "WebhookSpoke listening"
        );

        // Spawn the server as a background task
        tokio::spawn(async move {
            if let Err(e) = axum::serve(listener, app).await {
                warn!("WebhookSpoke server error: {e}");
            }
        });

        Ok(())
    }
}

/// Handle POST to root path (/) — uses default topic.
async fn handle_webhook_root(
    State(state): State<WebhookState>,
    headers: HeaderMap,
    body: Bytes,
) -> StatusCode {
    let topic = headers
        .get("x-topic")
        .and_then(|v| v.to_str().ok())
        .unwrap_or(&state.default_topic)
        .to_string();

    publish_event(&state, &topic, &headers, body)
}

/// Handle POST to a path (e.g. /ci/build/main) — path becomes topic.
async fn handle_webhook_path(
    State(state): State<WebhookState>,
    Path(path): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> StatusCode {
    // X-Topic header overrides path
    let topic = headers
        .get("x-topic")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or(path);

    publish_event(&state, &topic, &headers, body)
}

/// Create and publish a ChannelEvent from the HTTP request.
fn publish_event(
    state: &WebhookState,
    topic: &str,
    headers: &HeaderMap,
    body: Bytes,
) -> StatusCode {
    let content = String::from_utf8_lossy(&body).to_string();
    if content.is_empty() {
        return StatusCode::BAD_REQUEST;
    }

    // Build metadata from selected headers
    let mut meta = ChannelMeta::new();
    if let Some(ct) = headers.get("content-type").and_then(|v| v.to_str().ok()) {
        meta = meta.insert("content_type", ct.to_string());
    }
    if let Some(sender) = headers.get("x-sender-id").and_then(|v| v.to_str().ok()) {
        meta = meta.insert("sender_id", sender.to_string());
    }

    let event = ChannelEvent::new(&state.spoke_name, topic, content).with_meta(meta);

    state.publisher.publish(event);
    StatusCode::ACCEPTED
}
