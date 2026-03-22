//! Channel System — hub-and-spoke architecture for external event injection.
//!
//! The **Hub** is the central message bus that handles routing, broadcasting,
//! multiplexing, topic-based streaming, and sender gating. **Spokes** are
//! transport adapters (Webhook HTTP, Telegram bot, etc.) that connect external
//! sources to the hub.
//!
//! Inspired by Claude Code's Channels Reference, but built as a native Rust
//! trait system with proper pub/sub primitives.
//!
//! ## Key Design Principles
//!
//! - **Copy-on-Write (CoW)**: Events use `Arc` for zero-copy broadcasting.
//!   Meta keys/values use `Cow<'static, str>` — static strings stay borrowed.
//! - **Hub & Spoke**: All routing logic lives in the hub. Spokes only handle
//!   their transport. Cross-spoke routing is a hub concern.
//! - **Hierarchical Topics**: `/`-separated paths with upward propagation.
//!   Publishing to `chat/telegram/123` auto-propagates to `chat/telegram` and `chat`.
//! - **Broadcasting**: One event → N subscribers via `tokio::broadcast`.
//! - **Multiplexing**: N spokes → one subscriber stream, distinguished by `source`.

pub mod hub;
pub mod iggy_ws;
pub mod topic;
pub mod topology;
pub mod webhook;
pub mod telegram;
pub mod ws;

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ── Core Types ────────────────────────────────────────────────────

/// Metadata attached to a channel event.
///
/// Uses `Cow<'static, str>` for keys and values: static strings
/// (e.g. "chat_id", "severity") stay zero-copy borrowed, while
/// dynamic values (user IDs, timestamps) are owned.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChannelMeta {
    /// Key-value attributes.
    ///
    /// Stored as `BTreeMap<String, String>` for serialization compatibility,
    /// but constructed via `Cow` builders for zero-copy when possible.
    attributes: BTreeMap<String, String>,
}

impl ChannelMeta {
    /// Create empty metadata.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a key-value pair using CoW semantics.
    ///
    /// Static keys like `"chat_id"` avoid allocation when passed as `&'static str`.
    pub fn insert(mut self, key: impl Into<Cow<'static, str>>, value: impl Into<Cow<'static, str>>) -> Self {
        self.attributes.insert(key.into().into_owned(), value.into().into_owned());
        self
    }

    /// Get a value by key.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.attributes.get(key).map(|s| s.as_str())
    }

    /// Iterate over all attributes.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.attributes.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// Number of attributes.
    pub fn len(&self) -> usize {
        self.attributes.len()
    }

    /// Is empty?
    pub fn is_empty(&self) -> bool {
        self.attributes.is_empty()
    }

    /// Format attributes as XML-style key="value" pairs.
    ///
    /// Example: `severity="high" run_id="1234"`
    pub fn to_xml_attrs(&self) -> String {
        self.attributes
            .iter()
            .map(|(k, v)| format!("{k}=\"{v}\""))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// An event published by a spoke into the hub.
///
/// **CoW principle**: Events are wrapped in `Arc` for broadcasting.
/// When one event goes to N subscribers, ALL subscribers share the
/// same immutable `Arc<ChannelEvent>` — zero clones of the content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelEvent {
    /// Which spoke produced this event ("webhook", "telegram").
    pub source: String,
    /// Hierarchical topic path (e.g. "chat/telegram/12345").
    pub topic: String,
    /// Event body text.
    pub content: String,
    /// Structured metadata (becomes XML attributes in prompt).
    pub meta: ChannelMeta,
    /// When the event was created.
    pub timestamp: DateTime<Utc>,
}

impl ChannelEvent {
    /// Create a new channel event.
    pub fn new(
        source: impl Into<String>,
        topic: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            source: source.into(),
            topic: topic.into(),
            content: content.into(),
            meta: ChannelMeta::new(),
            timestamp: Utc::now(),
        }
    }

    /// Builder: attach metadata.
    pub fn with_meta(mut self, meta: ChannelMeta) -> Self {
        self.meta = meta;
        self
    }

    /// Format the event as an XML tag for injection into the agent's
    /// conversation history.
    ///
    /// Matches the Claude Code channel format:
    /// ```xml
    /// <channel source="webhook" topic="ci/build" severity="high">
    ///   build failed on main
    /// </channel>
    /// ```
    pub fn to_prompt(&self) -> String {
        let mut attrs = format!("source=\"{}\" topic=\"{}\"", self.source, self.topic);
        let meta_attrs = self.meta.to_xml_attrs();
        if !meta_attrs.is_empty() {
            attrs.push(' ');
            attrs.push_str(&meta_attrs);
        }
        format!("<channel {attrs}>{}</channel>", self.content)
    }

    /// Extract the parent topics for hierarchical propagation.
    ///
    /// `"chat/telegram/12345"` → `["chat/telegram", "chat"]`
    pub fn parent_topics(&self) -> Vec<String> {
        let mut parents = Vec::new();
        let mut path = self.topic.as_str();
        while let Some(pos) = path.rfind('/') {
            path = &path[..pos];
            parents.push(path.to_string());
        }
        parents
    }
}

impl fmt::Display for ChannelEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {}", self.source, self.topic, &self.content[..self.content.len().min(80)])
    }
}

// ── Spoke Trait ───────────────────────────────────────────────────

/// Handle given to spokes for publishing events into the hub.
///
/// Wraps a `tokio::sync::broadcast::Sender` and adds the Arc wrapping
/// for CoW broadcasting.
#[derive(Clone)]
pub struct HubPublisher {
    tx: tokio::sync::broadcast::Sender<Arc<ChannelEvent>>,
}

impl HubPublisher {
    /// Create a new publisher from a broadcast sender.
    pub(crate) fn new(tx: tokio::sync::broadcast::Sender<Arc<ChannelEvent>>) -> Self {
        Self { tx }
    }

    /// Publish an event into the hub.
    ///
    /// The event is wrapped in `Arc` for zero-copy broadcasting to all
    /// subscribers. Returns the number of subscribers that received it.
    pub fn publish(&self, event: ChannelEvent) -> usize {
        let arc_event = Arc::new(event);
        self.tx.send(arc_event).unwrap_or(0)
    }
}

/// A spoke connects an external transport to the hub.
///
/// Spokes handle their specific protocol (HTTP, Telegram long-poll,
/// WebSocket, etc.) and convert transport-specific messages into
/// `ChannelEvent`s that flow through the hub.
///
/// ## Hub & Spoke Contract
///
/// - The hub calls `start()` once, passing a `HubPublisher`.
/// - The spoke spawns its background polling/listening task.
/// - Events are published via `publisher.publish(event)`.
/// - If the spoke supports replies, the hub calls `reply()`.
#[async_trait]
pub trait Spoke: Send + Sync + 'static {
    /// Unique name for this spoke (e.g. "webhook", "telegram").
    fn name(&self) -> &str;

    /// Instructions injected into the agent's system prompt.
    ///
    /// Tells the LLM how to interpret events from this spoke and
    /// whether/how to reply. This follows the Claude Code pattern.
    fn instructions(&self) -> &str;

    /// Whether this spoke supports bidirectional communication.
    fn supports_reply(&self) -> bool;

    /// Send a reply back through this spoke's transport.
    ///
    /// The `meta` contains routing info (e.g. `chat_id` for Telegram).
    async fn reply(&self, meta: &ChannelMeta, text: &str) -> anyhow::Result<()>;

    /// Start the spoke's background transport.
    ///
    /// The spoke should spawn its own tokio tasks for polling/listening
    /// and publish events via the `publisher` handle. This method
    /// returns after setup — it should NOT block.
    async fn start(&self, publisher: HubPublisher) -> anyhow::Result<()>;
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_event_to_prompt_basic() {
        let event = ChannelEvent::new("webhook", "ci/build", "build failed on main");
        let prompt = event.to_prompt();
        assert_eq!(
            prompt,
            "<channel source=\"webhook\" topic=\"ci/build\">build failed on main</channel>"
        );
    }

    #[test]
    fn channel_event_to_prompt_with_meta() {
        let meta = ChannelMeta::new()
            .insert("severity", "high")
            .insert("run_id", "1234");
        let event = ChannelEvent::new("webhook", "ci/build", "build failed")
            .with_meta(meta);
        let prompt = event.to_prompt();
        assert!(prompt.contains("source=\"webhook\""));
        assert!(prompt.contains("topic=\"ci/build\""));
        assert!(prompt.contains("severity=\"high\""));
        assert!(prompt.contains("run_id=\"1234\""));
        assert!(prompt.contains(">build failed</channel>"));
    }

    #[test]
    fn channel_event_parent_topics() {
        let event = ChannelEvent::new("telegram", "chat/telegram/12345", "hello");
        let parents = event.parent_topics();
        assert_eq!(parents, vec!["chat/telegram", "chat"]);
    }

    #[test]
    fn channel_event_parent_topics_single_segment() {
        let event = ChannelEvent::new("webhook", "alerts", "fire!");
        let parents = event.parent_topics();
        assert!(parents.is_empty());
    }

    #[test]
    fn channel_meta_cow_insert() {
        // Static string — zero allocation for key
        let meta = ChannelMeta::new()
            .insert("chat_id", "12345")
            .insert("username", String::from("dynamic_user"));
        assert_eq!(meta.get("chat_id"), Some("12345"));
        assert_eq!(meta.get("username"), Some("dynamic_user"));
        assert_eq!(meta.len(), 2);
    }

    #[test]
    fn channel_meta_xml_attrs() {
        let meta = ChannelMeta::new()
            .insert("a", "1")
            .insert("b", "2");
        let attrs = meta.to_xml_attrs();
        // BTreeMap is sorted, so a comes before b
        assert_eq!(attrs, "a=\"1\" b=\"2\"");
    }

    #[test]
    fn hub_publisher_broadcasts_with_arc() {
        let (tx, mut rx) = tokio::sync::broadcast::channel::<Arc<ChannelEvent>>(16);
        let publisher = HubPublisher::new(tx);

        let event = ChannelEvent::new("test", "test/topic", "hello");
        let sent = publisher.publish(event);
        assert_eq!(sent, 1);

        let received = rx.try_recv().unwrap();
        assert_eq!(received.source, "test");
        assert_eq!(received.content, "hello");
        // Arc strong count = 1 (only the received copy)
        assert_eq!(Arc::strong_count(&received), 1);
    }

    #[test]
    fn hub_publisher_zero_copy_broadcast() {
        let (tx, mut rx1) = tokio::sync::broadcast::channel::<Arc<ChannelEvent>>(16);
        let mut rx2 = tx.subscribe();
        let publisher = HubPublisher::new(tx);

        let event = ChannelEvent::new("test", "t", "data");
        publisher.publish(event);

        let e1 = rx1.try_recv().unwrap();
        let e2 = rx2.try_recv().unwrap();

        // Both receivers got the SAME Arc — zero-copy broadcast
        assert!(Arc::ptr_eq(&e1, &e2));
    }
}
