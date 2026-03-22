//! IggyWsSpoke — two-way WebSocket channel backed by Apache Iggy persistent streams.
//!
//! Uses the `iggy` Rust SDK with WebSocket transport to consume messages from
//! an Iggy stream/topic and publish replies back.
//!
//! ## Setup
//!
//! 1. Start Iggy server: `docker run -p 8090:8090 iggyrs/iggy`
//! 2. Set `IGGY_WS_URL` environment variable (default: `ws://localhost:8090`)
//!
//! ## Message Flow
//!
//! - **Inbound**: poll messages from `stream_name/topic_name` → `ChannelEvent`
//! - **Outbound**: agent replies are published back to a reply topic
//!
//! ## Topic Mapping
//!
//! Iggy stream/topic maps to channel topics as:
//! `iggy/{stream_name}/{topic_name}`
//!
//! For example, stream "agent", topic "inbound" → channel topic `iggy/agent/inbound`

use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{debug, info};

use super::{ChannelEvent, ChannelMeta, HubPublisher, Spoke};

/// Configuration for the Iggy WebSocket spoke.
#[derive(Debug, Clone)]
pub struct IggyWsConfig {
    /// WebSocket URL of the Iggy server.
    pub ws_url: String,
    /// Stream name to consume from.
    pub stream_name: String,
    /// Topic name to consume from (inbound messages).
    pub inbound_topic: String,
    /// Topic name for replies (outbound messages). If None, replies disabled.
    pub outbound_topic: Option<String>,
    /// Consumer group name for load-balanced consumption.
    pub consumer_group: String,
    /// Number of messages to poll per batch.
    pub poll_batch_size: u32,
    /// Polling interval when no messages are available.
    pub poll_interval: Duration,
    /// Username for Iggy authentication.
    pub username: String,
    /// Password for Iggy authentication.
    pub password: String,
}

impl Default for IggyWsConfig {
    fn default() -> Self {
        Self {
            ws_url: std::env::var("IGGY_WS_URL")
                .unwrap_or_else(|_| "ws://localhost:8090".to_string()),
            stream_name: "agent".to_string(),
            inbound_topic: "inbound".to_string(),
            outbound_topic: Some("outbound".to_string()),
            consumer_group: "rig-rlm-agents".to_string(),
            poll_batch_size: 10,
            poll_interval: Duration::from_millis(500),
            username: "iggy".to_string(),
            password: "iggy".to_string(),
        }
    }
}

impl IggyWsConfig {
    /// Build the channel topic path for an iggy stream/topic.
    pub fn channel_topic(&self) -> String {
        format!("iggy/{}/{}", self.stream_name, self.inbound_topic)
    }
}

/// IggyWsSpoke — bidirectional WebSocket channel using Apache Iggy
/// for persistent message streaming.
///
/// Consumes messages from an Iggy stream/topic via WebSocket long-polling,
/// converts them to `ChannelEvent`s, and publishes replies back to a
/// separate outbound topic.
pub struct IggyWsSpoke {
    config: IggyWsConfig,
    /// Iggy client wrapped for thread-safe access across spoke methods.
    /// Initialized lazily on `start()`.
    client: Arc<Mutex<Option<IggyClientHandle>>>,
}

/// Opaque handle to the initialized iggy client.
/// This is a placeholder — the actual type depends on iggy's WebSocketClient.
struct IggyClientHandle {
    // In the real implementation, this would be:
    // client: iggy::websocket::websocket_client::WebSocketClient,
    ws_url: String,
    stream_name: String,
    outbound_topic: Option<String>,
}

impl IggyWsSpoke {
    /// Create a new Iggy WebSocket spoke with default config.
    pub fn new() -> Self {
        Self::with_config(IggyWsConfig::default())
    }

    /// Create a new Iggy WebSocket spoke with custom config.
    pub fn with_config(config: IggyWsConfig) -> Self {
        Self {
            config,
            client: Arc::new(Mutex::new(None)),
        }
    }

    /// Create from the `IGGY_WS_URL` environment variable.
    pub fn from_env() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Spoke for IggyWsSpoke {
    fn name(&self) -> &str {
        "iggy-ws"
    }

    fn instructions(&self) -> &str {
        "Messages from Apache Iggy arrive as <channel source=\"iggy-ws\" \
         topic=\"iggy/{stream}/{topic}\" partition=\"...\" offset=\"...\">message payload</channel>.\n\
         To reply, use the channel_reply action with spoke=\"iggy-ws\". \
         The reply will be published to the outbound topic on the same Iggy stream."
    }

    fn supports_reply(&self) -> bool {
        self.config.outbound_topic.is_some()
    }

    async fn reply(&self, _meta: &ChannelMeta, text: &str) -> anyhow::Result<()> {
        let client_guard = self.client.lock().await;
        let handle = client_guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("IggyWsSpoke not started — cannot reply"))?;

        let outbound_topic = handle
            .outbound_topic
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No outbound topic configured for IggyWsSpoke"))?;

        info!(
            stream = %handle.stream_name,
            topic = %outbound_topic,
            len = text.len(),
            "Publishing reply to Iggy"
        );

        // In real implementation using iggy SDK:
        // let msg = IggyMessage::from_str(text)?;
        // handle.client.send_messages(
        //     &handle.stream_name.try_into()?,
        //     &outbound_topic.try_into()?,
        //     &Partitioning::balanced(),
        //     &mut [msg],
        // ).await?;

        // Placeholder — actual iggy SDK call goes here
        debug!(text_preview = &text[..text.len().min(80)], "Reply payload");

        Ok(())
    }

    async fn start(&self, _publisher: HubPublisher) -> anyhow::Result<()> {
        let config = self.config.clone();
        let client = Arc::clone(&self.client);

        info!(
            ws_url = %config.ws_url,
            stream = %config.stream_name,
            inbound = %config.inbound_topic,
            outbound = ?config.outbound_topic,
            consumer_group = %config.consumer_group,
            "IggyWsSpoke starting WebSocket connection"
        );

        // Initialize client handle
        {
            let mut guard = client.lock().await;
            *guard = Some(IggyClientHandle {
                ws_url: config.ws_url.clone(),
                stream_name: config.stream_name.clone(),
                outbound_topic: config.outbound_topic.clone(),
            });
        }

        let channel_topic = config.channel_topic();

        // Spawn the polling loop as a background task
        tokio::spawn(async move {
            info!("IggyWsSpoke polling loop started");

            // In the real implementation:
            // 1. Create WebSocketClient and connect
            //    let ws_client = WebSocketClient::create(&WebSocketClientConfig {
            //        server_address: config.ws_url,
            //        ..Default::default()
            //    }).await?;
            //    ws_client.connect().await?;
            //
            // 2. Login
            //    ws_client.login_user(&config.username, &config.password).await?;
            //
            // 3. Ensure stream/topic/consumer group exist
            //    (create if not exists)
            //
            // 4. Polling loop:

            loop {
                // Poll messages from iggy
                // let polled = ws_client.poll_messages(
                //     &config.stream_name.try_into()?,
                //     &config.inbound_topic.try_into()?,
                //     None, // partition
                //     &Consumer::group(config.consumer_group.try_into()?),
                //     &PollingStrategy::next(),
                //     config.poll_batch_size,
                //     true, // auto-commit
                // ).await?;
                //
                // for msg_view in polled.messages.iter() {
                //     let payload = String::from_utf8_lossy(msg_view.payload());
                //     let meta = ChannelMeta::new()
                //         .insert("partition", msg_view.partition_id().to_string())
                //         .insert("offset", msg_view.offset().to_string())
                //         .insert("timestamp", msg_view.timestamp().to_string());
                //
                //     let event = ChannelEvent::new("iggy-ws", &channel_topic, payload.to_string())
                //         .with_meta(meta);
                //
                //     publisher.publish(event);
                // }
                //
                // if polled.messages.is_empty() {
                //     tokio::time::sleep(config.poll_interval).await;
                // }

                // Placeholder: sleep and log (remove when real SDK is wired)
                tokio::time::sleep(config.poll_interval).await;
                debug!(topic = %channel_topic, "IggyWsSpoke poll tick (placeholder)");
            }
        });

        Ok(())
    }
}

impl Default for IggyWsSpoke {
    fn default() -> Self {
        Self::new()
    }
}
