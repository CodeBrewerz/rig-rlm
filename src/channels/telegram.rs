//! TelegramSpoke — two-way Telegram bot channel.
//!
//! Uses `teloxide` for long-polling message reception and reply.
//! Each incoming text message becomes a `ChannelEvent` with topic
//! `chat/telegram/{chat_id}`.
//!
//! ## Setup
//!
//! Set `TELOXIDE_TOKEN` environment variable with your bot token.
//!
//! ## Topic mapping
//!
//! - Private message from user 123 → topic `chat/telegram/123`
//! - Group message in chat -456    → topic `chat/telegram/group/-456`
//!
//! ## Sender gating
//!
//! Configure `allowed_users` with Telegram user IDs to restrict which
//! users can interact with the agent.

use async_trait::async_trait;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn};

use super::{ChannelEvent, ChannelMeta, HubPublisher, Spoke};

/// TelegramSpoke — bidirectional Telegram Bot channel.
///
/// Receives text messages via long-polling and publishes them as
/// `ChannelEvent`s. Supports replies via `bot.send_message()`.
pub struct TelegramSpoke {
    /// Telegram Bot instance (from TELOXIDE_TOKEN env var).
    bot: teloxide::Bot,
    /// Allowed user IDs for sender gating.
    /// Empty = allow all (gating done at hub level instead).
    allowed_users: HashSet<i64>,
}

impl TelegramSpoke {
    /// Create a new Telegram spoke from an existing bot instance.
    pub fn new(bot: teloxide::Bot) -> Self {
        Self {
            bot,
            allowed_users: HashSet::new(),
        }
    }

    /// Create a new Telegram spoke from the `TELOXIDE_TOKEN` env var.
    pub fn from_env() -> Self {
        Self::new(teloxide::Bot::from_env())
    }

    /// Set allowed user IDs for spoke-level sender gating.
    pub fn with_allowed_users(mut self, users: HashSet<i64>) -> Self {
        self.allowed_users = users;
        self
    }
}

#[async_trait]
impl Spoke for TelegramSpoke {
    fn name(&self) -> &str {
        "telegram"
    }

    fn instructions(&self) -> &str {
        "Messages from Telegram arrive as <channel source=\"telegram\" topic=\"chat/telegram/{chat_id}\" \
         chat_id=\"...\" user_id=\"...\" username=\"...\">message text</channel>.\n\
         To reply, use the channel_reply action with spoke=\"telegram\" and include the chat_id in meta.\n\
         Always reply to the same chat_id that the message came from."
    }

    fn supports_reply(&self) -> bool {
        true
    }

    async fn reply(&self, meta: &ChannelMeta, text: &str) -> anyhow::Result<()> {
        use teloxide::prelude::*;

        let chat_id_str = meta
            .get("chat_id")
            .ok_or_else(|| anyhow::anyhow!("Missing chat_id in meta for Telegram reply"))?;

        let chat_id: i64 = chat_id_str
            .parse()
            .map_err(|_| anyhow::anyhow!("Invalid chat_id: {chat_id_str}"))?;

        self.bot
            .send_message(ChatId(chat_id), text)
            .await
            .map_err(|e| anyhow::anyhow!("Telegram send failed: {e}"))?;

        Ok(())
    }

    async fn start(&self, publisher: HubPublisher) -> anyhow::Result<()> {
        use teloxide::prelude::*;

        info!("TelegramSpoke starting long-polling...");

        let bot = self.bot.clone();
        let allowed = self.allowed_users.clone();

        // Wrap publisher in Arc<Mutex> for sharing across handler invocations
        let publisher = Arc::new(Mutex::new(publisher));

        tokio::spawn(async move {
            // Use teloxide's simple repl pattern for message handling
            teloxide::repl(bot, move |_bot: Bot, msg: Message| {
                let publisher = Arc::clone(&publisher);
                let allowed = allowed.clone();
                async move {
                    // Extract text content
                    let text = match msg.text() {
                        Some(t) => t.to_string(),
                        None => return Ok(()), // Skip non-text messages
                    };

                    // Spoke-level sender gating
                    let user_id = msg.from.as_ref().map(|u| u.id.0 as i64);
                    if !allowed.is_empty() {
                        if let Some(uid) = user_id {
                            if !allowed.contains(&uid) {
                                warn!(
                                    user_id = uid,
                                    "Telegram message from non-allowed user — skipping"
                                );
                                return Ok(());
                            }
                        } else {
                            return Ok(()); // No user info, skip
                        }
                    }

                    // Build topic: chat/telegram/{chat_id}
                    let chat_id = msg.chat.id.0;
                    let topic = format!("chat/telegram/{chat_id}");

                    // Build metadata
                    let mut meta = ChannelMeta::new()
                        .insert("chat_id", chat_id.to_string());

                    if let Some(uid) = user_id {
                        meta = meta.insert("sender_id", uid.to_string())
                            .insert("user_id", uid.to_string());
                    }

                    if let Some(user) = msg.from.as_ref() {
                        if let Some(ref username) = user.username {
                            meta = meta.insert("username", username.clone());
                        }
                        meta = meta.insert("first_name", user.first_name.clone());
                    }

                    let event = ChannelEvent::new("telegram", &topic, text)
                        .with_meta(meta);

                    let pub_lock = publisher.lock().await;
                    pub_lock.publish(event);

                    Ok(())
                }
            })
            .await;
        });

        Ok(())
    }
}
