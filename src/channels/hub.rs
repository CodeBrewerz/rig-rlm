//! The Channel Hub — central message bus for the hub-and-spoke architecture.
//!
//! The hub manages:
//! - **Spoke registration**: adding transport adapters (webhook, telegram, etc.)
//! - **Topic-based routing**: matching published events to subscriber filters
//! - **Broadcasting**: one event → N subscribers via `Arc<ChannelEvent>` (CoW)
//! - **Multiplexing**: N spokes → one subscriber stream
//! - **Sender gating**: per-spoke allowlists to block untrusted senders
//! - **Firehose**: all events (pre-filter) stream to an audit/logging subscriber
//! - **Unsubscribe**: dynamic subscription management via `SubscriptionId`
//! - **Hierarchical propagation**: events auto-match parent topic subscribers

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::broadcast;
use tracing::{debug, info, warn};

use super::topic::TopicFilter;
use super::{ChannelEvent, ChannelMeta, HubPublisher, Spoke};

/// Unique identifier for a subscription, used for unsubscribing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SubscriptionId(u64);

impl SubscriptionId {
    /// Create a SubscriptionId from a raw u64 (for testing/topology).
    pub fn from_raw(id: u64) -> Self {
        Self(id)
    }

    /// Get the raw u64 value.
    pub fn raw(&self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for SubscriptionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sub-{}", self.0)
    }
}

/// Global subscription ID counter.
static NEXT_SUB_ID: AtomicU64 = AtomicU64::new(1);

/// A topic subscription with its broadcast channel.
struct Subscription {
    id: SubscriptionId,
    filter: TopicFilter,
    tx: broadcast::Sender<Arc<ChannelEvent>>,
}

/// The central message bus.
///
/// ## CoW Broadcasting
///
/// Events are wrapped in `Arc<ChannelEvent>` at publish time. When one event
/// matches N subscriptions, the SAME `Arc` is sent to all — zero clones of
/// the underlying event content. Subscribers receive `Arc<ChannelEvent>` and
/// can read the event without any allocation.
///
/// ## Unsubscribe
///
/// `subscribe()` returns a `SubscriptionId` + `Receiver`. Call
/// `unsubscribe(id)` to remove the subscription. The receiver will
/// stop receiving events after unsubscribe.
pub struct ChannelHub {
    /// Registered spokes (transport adapters).
    spokes: Vec<Arc<dyn Spoke>>,
    /// Active topic subscriptions.
    subscriptions: Vec<Subscription>,
    /// Per-spoke sender allowlists (empty = allow all).
    allowlists: HashMap<String, HashSet<String>>,
    /// Firehose: ALL events (pre-filter) go here for audit/logging.
    firehose_tx: broadcast::Sender<Arc<ChannelEvent>>,
    /// Ingest channel: spokes publish here, hub consumes for routing.
    ingest_tx: broadcast::Sender<Arc<ChannelEvent>>,
    /// Broadcast capacity per subscription.
    broadcast_capacity: usize,
}

impl ChannelHub {
    /// Create a new hub with default capacity.
    pub fn new() -> Self {
        Self::with_capacity(256)
    }

    /// Create a new hub with specified broadcast capacity per subscription.
    pub fn with_capacity(capacity: usize) -> Self {
        let (firehose_tx, _) = broadcast::channel(capacity);
        let (ingest_tx, _) = broadcast::channel(capacity);
        Self {
            spokes: Vec::new(),
            subscriptions: Vec::new(),
            allowlists: HashMap::new(),
            firehose_tx,
            ingest_tx,
            broadcast_capacity: capacity,
        }
    }

    // ── Spoke Management ─────────────────────────────────────────

    /// Register a spoke with the hub.
    pub fn add_spoke(&mut self, spoke: Arc<dyn Spoke>) {
        info!(spoke = spoke.name(), "Registering spoke with hub");
        self.spokes.push(spoke);
    }

    /// Get a publisher handle for spokes to publish events.
    pub fn publisher(&self) -> HubPublisher {
        HubPublisher::new(self.ingest_tx.clone())
    }

    /// Start all spokes' background transports.
    ///
    /// Each spoke receives a `HubPublisher` for event injection.
    pub async fn start_all(&self) -> anyhow::Result<()> {
        let publisher = self.publisher();
        for spoke in &self.spokes {
            info!(spoke = spoke.name(), "Starting spoke transport");
            spoke.start(publisher.clone()).await?;
        }
        Ok(())
    }

    // ── Subscription Management ──────────────────────────────────

    /// Subscribe to events matching a topic filter pattern.
    ///
    /// Returns a `SubscriptionId` (for unsubscribing) and a `Receiver`
    /// that delivers matching events as `Arc<ChannelEvent>` (CoW).
    ///
    /// # Examples
    /// - `"ci/*"` — CI build events
    /// - `"chat/**"` — all chat messages from any platform
    /// - `"**"` — catch-all (firehose)
    pub fn subscribe(
        &mut self,
        filter_pattern: &str,
    ) -> (SubscriptionId, broadcast::Receiver<Arc<ChannelEvent>>) {
        let id = SubscriptionId(NEXT_SUB_ID.fetch_add(1, Ordering::Relaxed));
        let filter = TopicFilter::new(filter_pattern);
        let (tx, rx) = broadcast::channel(self.broadcast_capacity);

        debug!(id = %id, filter = filter_pattern, "New subscription");
        self.subscriptions.push(Subscription { id, filter, tx });

        (id, rx)
    }

    /// Remove a subscription by ID.
    ///
    /// After unsubscribing, the receiver will no longer receive events.
    /// Returns `true` if a subscription was found and removed.
    pub fn unsubscribe(&mut self, id: SubscriptionId) -> bool {
        let before = self.subscriptions.len();
        self.subscriptions.retain(|s| s.id != id);
        let removed = self.subscriptions.len() < before;
        if removed {
            debug!(id = %id, "Subscription removed");
        } else {
            warn!(id = %id, "Unsubscribe: subscription not found");
        }
        removed
    }

    /// Subscribe to the firehose (ALL events, pre-filter, for audit/logging).
    pub fn subscribe_firehose(&self) -> broadcast::Receiver<Arc<ChannelEvent>> {
        self.firehose_tx.subscribe()
    }

    /// Subscribe to the raw ingest channel (events from HubPublisher).
    ///
    /// This is used by the routing bridge task to consume spoke-published
    /// events and route them through the hub's topic filtering system.
    pub fn subscribe_ingest(&self) -> broadcast::Receiver<Arc<ChannelEvent>> {
        self.ingest_tx.subscribe()
    }

    /// Publish a pre-wrapped Arc<ChannelEvent> through the routing system.
    ///
    /// Unlike `publish()` which wraps in Arc, this takes an already-wrapped
    /// event — used by the routing bridge to avoid double-wrapping.
    pub fn publish_arc(&self, arc_event: Arc<ChannelEvent>) -> usize {
        // Step 1: Sender gating
        if !self.check_gate(&arc_event) {
            debug!(
                source = %arc_event.source,
                topic = %arc_event.topic,
                "Event gated (sender not in allowlist)"
            );
            return 0;
        }

        // Step 2: Firehose (all events, pre-filter)
        let _ = self.firehose_tx.send(Arc::clone(&arc_event));

        // Step 3: Route to matching subscriptions
        let mut delivered = 0;
        for sub in &self.subscriptions {
            if sub.filter.matches(&arc_event.topic) {
                match sub.tx.send(Arc::clone(&arc_event)) {
                    Ok(n) => delivered += n,
                    Err(_) => {
                        debug!(id = %sub.id, "Subscription has no active receivers");
                    }
                }
            }
        }

        if delivered > 0 {
            debug!(
                topic = %arc_event.topic,
                source = %arc_event.source,
                delivered,
                "Event routed to subscribers"
            );
        }

        delivered
    }

    /// Number of active subscriptions.
    pub fn subscription_count(&self) -> usize {
        self.subscriptions.len()
    }

    // ── Publishing ───────────────────────────────────────────────

    /// Publish an event, routing it through gating → firehose → topic matching.
    ///
    /// Returns the number of subscriptions the event was delivered to.
    /// Returns 0 if the event was gated (blocked by allowlist).
    ///
    /// ## Hierarchical propagation
    ///
    /// The event is matched against ALL subscriptions — both its exact topic
    /// AND its parent topics. The `TopicFilter` glob matching handles this:
    /// an event at `chat/telegram/123` will match `chat/telegram/*` and `chat/**`.
    pub fn publish(&self, event: ChannelEvent) -> usize {
        // Step 1: Sender gating
        if !self.check_gate(&event) {
            debug!(
                source = %event.source,
                topic = %event.topic,
                "Event gated (sender not in allowlist)"
            );
            return 0;
        }

        // Step 2: Wrap in Arc for CoW broadcasting
        let arc_event = Arc::new(event);

        // Step 3: Firehose (all events, pre-filter)
        let _ = self.firehose_tx.send(Arc::clone(&arc_event));

        // Step 4: Route to matching subscriptions
        let mut delivered = 0;
        for sub in &self.subscriptions {
            if sub.filter.matches(&arc_event.topic) {
                match sub.tx.send(Arc::clone(&arc_event)) {
                    Ok(n) => delivered += n,
                    Err(_) => {
                        // No active receivers — subscription will be cleaned up
                        debug!(id = %sub.id, "Subscription has no active receivers");
                    }
                }
            }
        }

        if delivered > 0 {
            debug!(
                topic = %arc_event.topic,
                source = %arc_event.source,
                delivered,
                "Event routed to subscribers"
            );
        }

        delivered
    }

    // ── Reply ────────────────────────────────────────────────────

    /// Send a reply through a named spoke.
    ///
    /// The hub looks up the spoke by name and delegates the reply.
    /// Returns an error if the spoke doesn't exist or doesn't support replies.
    pub async fn reply(
        &self,
        spoke_name: &str,
        meta: &ChannelMeta,
        text: &str,
    ) -> anyhow::Result<()> {
        let spoke = self
            .spokes
            .iter()
            .find(|s| s.name() == spoke_name)
            .ok_or_else(|| anyhow::anyhow!("Spoke not found: {spoke_name}"))?;

        if !spoke.supports_reply() {
            return Err(anyhow::anyhow!(
                "Spoke '{spoke_name}' does not support replies"
            ));
        }

        spoke.reply(meta, text).await
    }

    // ── Gating ───────────────────────────────────────────────────

    /// Configure sender allowlist for a spoke.
    ///
    /// When set, only events with a `sender_id` meta attribute in
    /// the allowlist will be routed. Empty allowlist = allow all.
    ///
    /// Gating rules are **inherited hierarchically**: if you set
    /// an allowlist for "telegram", it applies to all topics under
    /// `chat/telegram/*` unless a more specific allowlist is set.
    pub fn set_allowlist(&mut self, spoke_name: &str, senders: HashSet<String>) {
        info!(
            spoke = spoke_name,
            count = senders.len(),
            "Allowlist configured"
        );
        self.allowlists.insert(spoke_name.to_string(), senders);
    }

    /// Check if an event passes the sender gate.
    fn check_gate(&self, event: &ChannelEvent) -> bool {
        match self.allowlists.get(&event.source) {
            None => true, // No allowlist = allow all
            Some(allowed) if allowed.is_empty() => true,
            Some(allowed) => {
                // Check if the event's sender_id is in the allowlist
                match event.meta.get("sender_id") {
                    Some(sender) => allowed.contains(sender),
                    None => {
                        // No sender_id in meta — deny if allowlist is set
                        warn!(
                            source = %event.source,
                            "Event has no sender_id but allowlist is configured — denying"
                        );
                        false
                    }
                }
            }
        }
    }

    // ── Instructions ─────────────────────────────────────────────

    /// Collect system prompt instructions from all registered spokes.
    ///
    /// Each spoke contributes instructions telling the LLM how to
    /// interpret events from that spoke and how to reply.
    pub fn instructions(&self) -> String {
        if self.spokes.is_empty() {
            return String::new();
        }

        let mut instructions = String::from("## Active Channels\n\n");
        instructions.push_str(
            "You have active channel connections. Events from external sources \
             will appear as `<channel>` XML tags in the conversation. Each tag \
             has `source` and `topic` attributes.\n\n",
        );

        for spoke in &self.spokes {
            let instr = spoke.instructions();
            if !instr.is_empty() {
                instructions.push_str(&format!("### {} Channel\n{}\n\n", spoke.name(), instr));
            }
        }

        instructions
    }

    /// Get the names of all registered spokes.
    pub fn spoke_names(&self) -> Vec<&str> {
        self.spokes.iter().map(|s| s.name()).collect()
    }
}

impl Default for ChannelHub {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::ChannelMeta;

    fn test_event(topic: &str) -> ChannelEvent {
        ChannelEvent::new("test-spoke", topic, "test content")
    }

    fn test_event_with_sender(topic: &str, sender: &str) -> ChannelEvent {
        ChannelEvent::new("test-spoke", topic, "test content")
            .with_meta(ChannelMeta::new().insert("sender_id", sender.to_string()))
    }

    // ── Subscribe & Publish ─────────────────────────────────────

    #[test]
    fn subscribe_and_publish_exact() {
        let mut hub = ChannelHub::new();
        let (_id, mut rx) = hub.subscribe("ci/build");

        let delivered = hub.publish(test_event("ci/build"));
        assert!(delivered > 0);

        let event = rx.try_recv().unwrap();
        assert_eq!(event.topic, "ci/build");
        assert_eq!(event.content, "test content");
    }

    #[test]
    fn subscribe_wildcard_routing() {
        let mut hub = ChannelHub::new();
        let (_id, mut rx) = hub.subscribe("ci/*");

        hub.publish(test_event("ci/build"));
        hub.publish(test_event("ci/deploy"));
        hub.publish(test_event("chat/telegram")); // should NOT match

        assert!(rx.try_recv().is_ok()); // ci/build
        assert!(rx.try_recv().is_ok()); // ci/deploy
        assert!(rx.try_recv().is_err()); // no more
    }

    // ── Broadcasting (one event → N subscribers) ────────────────

    #[test]
    fn broadcast_to_multiple_subscribers() {
        let mut hub = ChannelHub::new();
        let (_id1, mut rx1) = hub.subscribe("ci/**");
        let (_id2, mut rx2) = hub.subscribe("ci/**");

        hub.publish(test_event("ci/build/main"));

        let e1 = rx1.try_recv().unwrap();
        let e2 = rx2.try_recv().unwrap();

        // Both got the event
        assert_eq!(e1.topic, "ci/build/main");
        assert_eq!(e2.topic, "ci/build/main");

        // CoW: they share the SAME Arc (zero-copy)
        assert!(Arc::ptr_eq(&e1, &e2));
    }

    // ── Multiplexing (N spokes → one subscriber) ────────────────

    #[test]
    fn multiplex_multiple_sources() {
        let mut hub = ChannelHub::new();
        let (_id, mut rx) = hub.subscribe("**"); // catch-all

        hub.publish(ChannelEvent::new("webhook", "ci/build", "build failed"));
        hub.publish(ChannelEvent::new("telegram", "chat/123", "hello"));

        let e1 = rx.try_recv().unwrap();
        let e2 = rx.try_recv().unwrap();

        // One subscriber got events from two different sources
        assert_eq!(e1.source, "webhook");
        assert_eq!(e2.source, "telegram");
    }

    // ── Unsubscribe ─────────────────────────────────────────────

    #[test]
    fn unsubscribe_removes_subscription() {
        let mut hub = ChannelHub::new();
        let (id, _rx) = hub.subscribe("ci/**");
        assert_eq!(hub.subscription_count(), 1);

        let removed = hub.unsubscribe(id);
        assert!(removed);
        assert_eq!(hub.subscription_count(), 0);
    }

    #[test]
    fn unsubscribe_nonexistent_returns_false() {
        let mut hub = ChannelHub::new();
        let removed = hub.unsubscribe(SubscriptionId(9999));
        assert!(!removed);
    }

    #[test]
    fn unsubscribe_stops_delivery() {
        let mut hub = ChannelHub::new();
        let (id, mut rx) = hub.subscribe("ci/**");

        // First event: should arrive
        hub.publish(test_event("ci/build"));
        assert!(rx.try_recv().is_ok());

        // Unsubscribe
        hub.unsubscribe(id);

        // Second event: should NOT arrive (subscription removed)
        hub.publish(test_event("ci/deploy"));
        assert!(rx.try_recv().is_err());
    }

    // ── Sender Gating ───────────────────────────────────────────

    #[test]
    fn gating_allows_permitted_sender() {
        let mut hub = ChannelHub::new();
        hub.set_allowlist("test-spoke", HashSet::from(["user42".to_string()]));
        let (_id, mut rx) = hub.subscribe("**");

        let delivered = hub.publish(test_event_with_sender("ci/build", "user42"));
        assert!(delivered > 0);
        assert!(rx.try_recv().is_ok());
    }

    #[test]
    fn gating_blocks_unpermitted_sender() {
        let mut hub = ChannelHub::new();
        hub.set_allowlist("test-spoke", HashSet::from(["user42".to_string()]));
        let (_id, mut rx) = hub.subscribe("**");

        let delivered = hub.publish(test_event_with_sender("ci/build", "hacker99"));
        assert_eq!(delivered, 0);
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn gating_blocks_missing_sender_id() {
        let mut hub = ChannelHub::new();
        hub.set_allowlist("test-spoke", HashSet::from(["user42".to_string()]));

        // Event without sender_id meta
        let delivered = hub.publish(test_event("ci/build"));
        assert_eq!(delivered, 0);
    }

    #[test]
    fn no_allowlist_allows_all() {
        let mut hub = ChannelHub::new();
        let (_id, mut rx) = hub.subscribe("**");

        // No allowlist set — everything passes
        let delivered = hub.publish(test_event("ci/build"));
        assert!(delivered > 0);
        assert!(rx.try_recv().is_ok());
    }

    // ── Firehose ────────────────────────────────────────────────

    #[test]
    fn firehose_receives_all_events() {
        let mut hub = ChannelHub::new();
        let mut firehose = hub.subscribe_firehose();
        let (_id, _) = hub.subscribe("ci/*"); // topic-specific sub

        hub.publish(test_event("ci/build"));
        hub.publish(test_event("chat/telegram")); // won't match ci/*

        // Firehose gets BOTH events regardless of topic subscriptions
        assert!(firehose.try_recv().is_ok());
        assert!(firehose.try_recv().is_ok());
    }

    // ── Hierarchical topic matching ─────────────────────────────

    #[test]
    fn hierarchical_topic_matching() {
        let mut hub = ChannelHub::new();
        let (_id1, mut rx_exact) = hub.subscribe("chat/telegram/12345");
        let (_id2, mut rx_wild) = hub.subscribe("chat/telegram/*");
        let (_id3, mut rx_recursive) = hub.subscribe("chat/**");

        hub.publish(test_event("chat/telegram/12345"));

        // All three should receive the event
        assert!(rx_exact.try_recv().is_ok());
        assert!(rx_wild.try_recv().is_ok());
        assert!(rx_recursive.try_recv().is_ok());
    }
}
