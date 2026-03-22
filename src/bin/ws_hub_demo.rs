//! Hub & Topics Demo — Topics, Broadcast, Multiplex, Routing
//!
//! Demonstrates the full hub-and-spoke architecture:
//! - **Topics**: hierarchical topic routing with pattern matching
//! - **Broadcast**: one event fans out to ALL matching subscribers
//! - **Multiplex**: one subscriber receives events from MULTIPLE sources
//! - **Hub routing**: gating, firehose, subscription management
//!
//! ## Commands (type in WebSocket)
//!
//! - `/help` — show available commands
//! - `/status` — show all active subscribers and their topic filters
//! - `/pub <topic> <message>` — publish an event to a specific topic
//! - `/sub <pattern>` — add a new subscriber with a topic filter
//! - `/unsub <id>` — remove a subscriber by ID
//! - `/broadcast <message>` — publish to multiple topics at once (fan-out)
//! - `/multiplex` — demonstrate N sources → 1 subscriber
//! - `/topo` — show the topology/clique analysis
//! - Just type normal text → goes to `ws/interactive/{conn_id}`

use rig_rlm::channels::hub::ChannelHub;
use rig_rlm::channels::topology::{SubscriptionEntry, SubscriptionGraph};
use rig_rlm::channels::ws::WsSpoke;
use rig_rlm::channels::{ChannelEvent, ChannelMeta, Spoke};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use tracing::{info, warn};

/// Named subscriber for tracking
struct NamedSubscriber {
    name: String,
    filter_pattern: String,
    rx: broadcast::Receiver<Arc<ChannelEvent>>,
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
    eprintln!("║   rig-rlm Hub & Topics Demo                             ║");
    eprintln!("║                                                         ║");
    eprintln!("║   Connect: websocat ws://localhost:9090/ws              ║");
    eprintln!("║                                                         ║");
    eprintln!("║   Commands:                                             ║");
    eprintln!("║     /help              — show all commands              ║");
    eprintln!("║     /status            — show subscribers               ║");
    eprintln!("║     /pub topic msg     — publish to a topic             ║");
    eprintln!("║     /sub pattern       — subscribe to a pattern         ║");
    eprintln!("║     /unsub N           — unsubscribe subscriber N       ║");
    eprintln!("║     /broadcast msg     — fan-out to multiple topics     ║");
    eprintln!("║     /multiplex         — demo N sources → 1 subscriber  ║");
    eprintln!("║     /topo              — show topology analysis         ║");
    eprintln!("║                                                         ║");
    eprintln!("║   Press Ctrl+C to stop                                  ║");
    eprintln!("╚═══════════════════════════════════════════════════════════╝");
    eprintln!();

    // 1. Create the hub
    let mut hub = ChannelHub::new();

    // 2. Pre-create named subscribers with different topic patterns
    let mut subscribers: Vec<NamedSubscriber> = Vec::new();

    let patterns = [
        ("CodeAgent", "code/**"),
        ("MathAgent", "math/**"),
        ("AllWatcher", "**"),
        ("ChatBot", "ws/interactive/*"),
    ];

    for (name, pattern) in &patterns {
        let (_id, rx) = hub.subscribe(pattern);
        subscribers.push(NamedSubscriber {
            name: name.to_string(),
            filter_pattern: pattern.to_string(),
            rx,
        });
        eprintln!("📋 Subscriber \"{name}\" watching '{pattern}'");
    }

    // 3. Start the WsSpoke
    let ws_spoke = Arc::new(WsSpoke::new(9090));
    let mut ingest_rx = hub.subscribe_ingest();
    let publisher = hub.publisher();
    ws_spoke.start(publisher.clone()).await?;

    // 4. Wrap hub for shared access (needed for /sub, /unsub, /pub commands)
    let hub = Arc::new(RwLock::new(hub));
    let subscribers = Arc::new(RwLock::new(subscribers));

    eprintln!("✅ WebSocket server ready at ws://localhost:9090/ws");
    eprintln!();

    // 5. Spawn subscriber drain tasks — each subscriber prints when it gets an event
    //    and forwards the event info back to the user via WsSpoke
    {
        let subs = Arc::clone(&subscribers);
        let ws_spoke_clone = Arc::clone(&ws_spoke);
        tokio::spawn(async move {
            // Continuously drain all subscriber receivers
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                let mut subs_w = subs.write().await;
                for sub in subs_w.iter_mut() {
                    while let Ok(event) = sub.rx.try_recv() {
                        eprintln!(
                            "📬 Subscriber \"{}\" [{}] received: [{}] {}",
                            sub.name,
                            sub.filter_pattern,
                            event.topic,
                            &event.content[..event.content.len().min(60)]
                        );
                    }
                }
            }
        });
    }

    // 6. Main loop — process incoming WebSocket messages
    loop {
        match ingest_rx.recv().await {
            Ok(event) => {
                let content = event.content.trim().to_string();
                let conn_id = event.meta.get("conn_id").map(|s| s.to_string());
                let conn_id = match conn_id {
                    Some(id) => id,
                    None => continue,
                };

                let meta = ChannelMeta::new().insert("conn_id", conn_id.clone());

                if content.starts_with('/') {
                    // Command mode
                    let parts: Vec<&str> = content.splitn(3, ' ').collect();
                    let cmd = parts[0];

                    match cmd {
                        "/help" => {
                            let help = "📖 Commands:\n\
                                /status — list all active subscribers & their filters\n\
                                /pub <topic> <msg> — publish to a topic (e.g. /pub math/calc 2+2)\n\
                                /sub <pattern> — add subscriber (e.g. /sub code/*)\n\
                                /unsub <N> — remove subscriber N\n\
                                /broadcast <msg> — fan-out to code/*, math/*, chat/*\n\
                                /multiplex — demo N sources → 1 subscriber (AllWatcher)\n\
                                /topo — show co-subscription topology + cliques\n\
                                Or just type text → goes to ws/interactive/{conn_id}";
                            ws_spoke.reply(&meta, help).await.ok();
                        }

                        "/status" => {
                            let subs = subscribers.read().await;
                            let hub_r = hub.read().await;
                            let mut status = format!(
                                "📊 Hub Status\n\
                                 ─────────────────────────\n\
                                 Subscribers: {}\n\
                                 Spokes: {:?}\n",
                                hub_r.subscription_count(),
                                hub_r.spoke_names(),
                            );
                            status.push_str("─────────────────────────\n");
                            for (i, sub) in subs.iter().enumerate() {
                                status.push_str(&format!(
                                    "  [{}] \"{}\" → filter: '{}'\n",
                                    i, sub.name, sub.filter_pattern
                                ));
                            }
                            ws_spoke.reply(&meta, &status).await.ok();
                        }

                        "/pub" => {
                            if parts.len() < 3 {
                                ws_spoke
                                    .reply(&meta, "Usage: /pub <topic> <message>\nExample: /pub math/calc 2+2=?")
                                    .await.ok();
                            } else {
                                let topic = parts[1];
                                let msg = parts[2];
                                let event = ChannelEvent::new("user", topic, msg);

                                // Route through hub
                                let hub_r = hub.read().await;
                                let delivered = hub_r.publish(event);

                                // Check which subscribers matched
                                let subs = subscribers.read().await;
                                let matching: Vec<&str> = subs
                                    .iter()
                                    .filter(|s| {
                                        let filter =
                                            rig_rlm::channels::topic::TopicFilter::new(&s.filter_pattern);
                                        filter.matches(topic)
                                    })
                                    .map(|s| s.name.as_str())
                                    .collect();

                                let reply = format!(
                                    "📤 Published to topic '{topic}'\n\
                                     ✅ Delivered to {delivered} subscriber(s)\n\
                                     📬 Matching: [{}]\n\
                                     🔀 This is {} — {} event → {} subscriber(s)",
                                    matching.join(", "),
                                    if matching.len() > 1 {
                                        "BROADCAST (1→N)"
                                    } else {
                                        "UNICAST (1→1)"
                                    },
                                    1,
                                    matching.len()
                                );
                                ws_spoke.reply(&meta, &reply).await.ok();
                            }
                        }

                        "/sub" => {
                            if parts.len() < 2 {
                                ws_spoke
                                    .reply(&meta, "Usage: /sub <pattern>\nExample: /sub alerts/**")
                                    .await.ok();
                            } else {
                                let pattern = parts[1];
                                let name = format!("User-{}", pattern.replace('/', "-"));
                                let mut hub_w = hub.write().await;
                                let (_id, rx) = hub_w.subscribe(pattern);
                                let mut subs = subscribers.write().await;
                                subs.push(NamedSubscriber {
                                    name: name.clone(),
                                    filter_pattern: pattern.to_string(),
                                    rx,
                                });
                                let reply = format!(
                                    "✅ Added subscriber \"{name}\" watching '{pattern}'\n\
                                     📊 Total subscribers: {}",
                                    subs.len()
                                );
                                ws_spoke.reply(&meta, &reply).await.ok();
                            }
                        }

                        "/unsub" => {
                            if parts.len() < 2 {
                                ws_spoke
                                    .reply(&meta, "Usage: /unsub <N>\nExample: /unsub 2")
                                    .await.ok();
                            } else if let Ok(idx) = parts[1].parse::<usize>() {
                                let mut subs = subscribers.write().await;
                                if idx < subs.len() {
                                    let removed = subs.remove(idx);
                                    let reply = format!(
                                        "🗑️ Removed subscriber \"{}\" (was watching '{}')\n\
                                         📊 Remaining: {}",
                                        removed.name,
                                        removed.filter_pattern,
                                        subs.len()
                                    );
                                    ws_spoke.reply(&meta, &reply).await.ok();
                                } else {
                                    ws_spoke
                                        .reply(&meta, &format!("❌ No subscriber at index {idx}"))
                                        .await.ok();
                                }
                            }
                        }

                        "/broadcast" => {
                            let msg = if parts.len() > 1 { parts[1] } else { "broadcast ping!" };
                            let topics = ["code/build", "math/calc", "ws/interactive/broadcast"];

                            let hub_r = hub.read().await;
                            let mut total_delivered = 0;
                            let mut results = Vec::new();

                            for topic in topics {
                                let event = ChannelEvent::new("broadcast", topic, msg);
                                let n = hub_r.publish(event);
                                total_delivered += n;
                                results.push(format!("  '{topic}' → {n} subscriber(s)"));
                            }

                            let reply = format!(
                                "📡 BROADCAST — 1 message → {} topics → {} total deliveries\n\
                                 ─────────────────────────\n\
                                 {}\n\
                                 ─────────────────────────\n\
                                 🔀 Same message fanned out to ALL matching subscribers\n\
                                 📊 Topics: {} | Deliveries: {}",
                                topics.len(),
                                total_delivered,
                                results.join("\n"),
                                topics.len(),
                                total_delivered,
                            );
                            ws_spoke.reply(&meta, &reply).await.ok();
                        }

                        "/multiplex" => {
                            let hub_r = hub.read().await;

                            // Publish from multiple "sources" to different topics
                            let sources = [
                                ("webhook", "code/ci", "build #42 passed ✅"),
                                ("telegram", "ws/interactive/1", "user says hello 💬"),
                                ("cron", "math/schedule", "daily report ready 📊"),
                            ];

                            for (source, topic, msg) in &sources {
                                let event = ChannelEvent::new(*source, *topic, *msg);
                                hub_r.publish(event);
                            }

                            // The AllWatcher subscriber (pattern "**") gets ALL of them
                            let reply = format!(
                                "🔀 MULTIPLEX — {} sources → 1 subscriber\n\
                                 ─────────────────────────\n\
                                 Sources → Topics:\n\
                                 {}\n\
                                 ─────────────────────────\n\
                                 ✅ AllWatcher (pattern '**') received ALL {} events\n\
                                 📊 This is MULTIPLEX: N sources → 1 subscriber stream",
                                sources.len(),
                                sources
                                    .iter()
                                    .map(|(s, t, m)| format!("  [{s}] '{t}' → \"{m}\""))
                                    .collect::<Vec<_>>()
                                    .join("\n"),
                                sources.len()
                            );
                            ws_spoke.reply(&meta, &reply).await.ok();
                        }

                        "/topo" => {
                            let subs = subscribers.read().await;
                            let mut graph = SubscriptionGraph::new();

                            // Build the co-subscription graph
                            for (i, sub) in subs.iter().enumerate() {
                                let topics: Vec<&str> = match sub.filter_pattern.as_str() {
                                    "code/**" => vec!["code/build", "code/review", "code/ci"],
                                    "math/**" => vec!["math/calc", "math/schedule"],
                                    "**" => vec!["code/build", "code/review", "math/calc", "ws/interactive/1"],
                                    "ws/interactive/*" => vec!["ws/interactive/1", "ws/interactive/2"],
                                    other => vec![other],
                                };
                                for t in topics {
                                    graph.add(SubscriptionEntry {
                                        id: rig_rlm::channels::hub::SubscriptionId::from_raw(i as u64),
                                        filter_pattern: t.to_string(),
                                    });
                                }
                            }

                            let adj = graph.co_subscription_adjacency();
                            let cliques = graph.find_cliques();

                            let mut reply = format!(
                                "🕸️ TOPOLOGY ANALYSIS\n\
                                 ─────────────────────────\n\
                                 Subscribers: {}\n\
                                 Co-subscription edges: {}\n",
                                subs.len(),
                                adj.values().map(|v| v.len()).sum::<usize>() / 2,
                            );

                            reply.push_str("─────────────────────────\n");
                            reply.push_str("Cliques (groups sharing topics):\n");
                            for (i, clique) in cliques.iter().enumerate() {
                                let members: Vec<String> = clique
                                    .members
                                    .iter()
                                    .map(|id| {
                                        subs.get(id.raw() as usize)
                                            .map(|s| s.name.as_str())
                                            .unwrap_or("?")
                                            .to_string()
                                    })
                                    .collect();
                                reply.push_str(&format!(
                                    "  Clique {}: [{}]\n",
                                    i + 1,
                                    members.join(", "),
                                ));
                            }

                            reply.push_str("─────────────────────────\n");
                            reply.push_str("📊 Cliques = groups of subscribers that share information spaces");
                            ws_spoke.reply(&meta, &reply).await.ok();
                        }

                        _ => {
                            ws_spoke
                                .reply(&meta, &format!("❌ Unknown command: {cmd}\nType /help for commands"))
                                .await.ok();
                        }
                    }
                } else {
                    // Normal message — route through hub and show routing info
                    let topic = format!("ws/interactive/{conn_id}");
                    let event = ChannelEvent::new("ws", &topic, &content);
                    let hub_r = hub.read().await;
                    let delivered = hub_r.publish(event);

                    let subs = subscribers.read().await;
                    let matching: Vec<&str> = subs
                        .iter()
                        .filter(|s| {
                            let filter =
                                rig_rlm::channels::topic::TopicFilter::new(&s.filter_pattern);
                            filter.matches(&topic)
                        })
                        .map(|s| s.name.as_str())
                        .collect();

                    let reply = format!(
                        "📨 Message received: \"{}\"\n\
                         📍 Topic: {topic}\n\
                         ✅ Routed to {} subscriber(s): [{}]\n\
                         💡 Try /pub math/calc 2+2 or /broadcast hello or /topo",
                        content,
                        delivered,
                        matching.join(", "),
                    );
                    ws_spoke.reply(&meta, &reply).await.ok();
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
