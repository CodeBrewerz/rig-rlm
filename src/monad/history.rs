//! Conversation history management.
//!
//! Structured message history with role-based insertion and conversion
//! to rig's message format for LLM calls.

use std::borrow::Cow;

use rig::{
    OneOrMany,
    agent::Text,
    message::{AssistantContent, Message, UserContent},
};

use super::action::Role;
use super::attachment::Attachment;

/// A single message in the conversation history.
///
/// Uses `Cow<'static, str>` for content to enable cheap clones
/// when content is shared (e.g. in `to_rig_prompt`).
#[derive(Debug, Clone)]
pub struct HistoryMessage {
    pub role: Role,
    pub content: Cow<'static, str>,
    /// Multimodal attachments (images, PDFs, etc.).
    /// Empty for pure-text messages.
    #[allow(dead_code)]
    pub attachments: Vec<Attachment>,
}

/// Manages the full conversation history for an agent session.
#[derive(Debug, Clone)]
pub struct ConversationHistory {
    messages: Vec<HistoryMessage>,
    /// Token budget tracker for the model's context window.
    token_budget: TokenBudget,
}

impl ConversationHistory {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            token_budget: TokenBudget::default(),
        }
    }

    /// Add a message to the history.
    pub fn push(&mut self, message: HistoryMessage) {
        self.messages.push(message);
    }

    /// Insert a system prompt at the beginning.
    pub fn insert_system(&mut self, content: impl Into<String>) {
        self.messages.insert(
            0,
            HistoryMessage {
                role: Role::System,
                content: Cow::Owned(content.into()),
                attachments: vec![],
            },
        );
    }

    /// Get all messages.
    pub fn messages(&self) -> &[HistoryMessage] {
        &self.messages
    }

    /// Number of messages.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Whether history is empty.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Clear all messages.
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Insert a message at a specific position.
    pub fn insert_at(&mut self, index: usize, message: HistoryMessage) {
        let pos = index.min(self.messages.len());
        self.messages.insert(pos, message);
    }

    /// Convert to rig's prompt format.
    ///
    /// Returns `(latest_user_prompt, chat_history)` because rig's
    /// `.chat()` takes the current prompt separately from history.
    ///
    /// The system messages are folded into the first user message
    /// (rig handles preamble separately via agent builder).
    pub fn to_rig_prompt(&self) -> (String, Vec<Message>) {
        let mut history: Vec<Message> = Vec::new();
        let mut latest_user_prompt = String::new();

        for msg in &self.messages {
            match &msg.role {
                Role::System => {
                    // System messages are handled by rig's preamble.
                    // If we have one mid-conversation, fold it as user context.
                    history.push(Message::User {
                        content: OneOrMany::one(UserContent::Text(Text {
                            text: format!("[System] {}", msg.content),
                        })),
                    });
                }
                Role::User => {
                    // Track the latest user message as the prompt.
                    latest_user_prompt = msg.content.to_string();

                    // Build content parts: text + any image attachments
                    let mut parts: Vec<UserContent> = vec![UserContent::Text(Text {
                        text: msg.content.to_string(),
                    })];
                    for att in &msg.attachments {
                        if att.is_image() {
                            parts.push(UserContent::image_base64(
                                &att.data,
                                att.image_media_type(),
                                None,
                            ));
                        } else if att.is_audio() {
                            parts.push(UserContent::audio(&att.data, att.audio_media_type()));
                        }
                        // PDFs and text docs are handled as LoadContext
                        // (text extraction), not as direct content here.
                    }

                    let content = if parts.len() == 1 {
                        OneOrMany::one(parts.remove(0))
                    } else {
                        OneOrMany::many(parts).unwrap_or_else(|_| {
                            OneOrMany::one(UserContent::Text(Text {
                                text: msg.content.to_string(),
                            }))
                        })
                    };
                    history.push(Message::User { content });
                }
                Role::Assistant => {
                    history.push(Message::Assistant {
                        content: OneOrMany::one(AssistantContent::Text(Text {
                            text: msg.content.to_string(),
                        })),
                        id: None,
                    });
                }
                Role::Execution => {
                    // Execution results are inserted as user messages
                    // (the model sees them as feedback from the environment).
                    history.push(Message::User {
                        content: OneOrMany::one(UserContent::Text(Text {
                            text: msg.content.to_string(),
                        })),
                    });
                }
            }
        }

        // Remove the latest user message from history (rig takes it as `prompt`).
        if !history.is_empty() {
            history.pop();
        }

        (latest_user_prompt, history)
    }
}

impl ConversationHistory {
    // ─── Phase 5: Compaction support ──────────────────────────────

    /// Split off old messages, keeping only the most recent `keep_recent` messages.
    ///
    /// Returns the removed older messages. The system prompt (first message) is
    /// always preserved even if it would otherwise be removed.
    pub fn split_at(&mut self, keep_recent: usize) -> Vec<HistoryMessage> {
        if self.messages.len() <= keep_recent {
            return Vec::new();
        }
        let split_point = self.messages.len() - keep_recent;
        // Always keep the system prompt (index 0) if present
        let actual_split = if !self.messages.is_empty()
            && self.messages[0].role == Role::System
            && split_point > 0
        {
            split_point.max(1)
        } else {
            split_point
        };
        let old: Vec<_> = self.messages.drain(..actual_split).collect();
        old
    }

    /// Prepend a summary message at the beginning of history.
    ///
    /// Inserts after the system prompt if one exists, otherwise at position 0.
    pub fn prepend_summary(&mut self, summary: String) {
        let insert_pos = if !self.messages.is_empty() && self.messages[0].role == Role::System {
            1
        } else {
            0
        };
        self.messages.insert(
            insert_pos,
            HistoryMessage {
                role: Role::System,
                content: Cow::Owned(format!("[Context Summary]\n{summary}")),
                attachments: vec![],
            },
        );
    }

    /// Rough token estimate (total chars / 4).
    pub fn estimate_tokens(&self) -> usize {
        self.messages.iter().map(|m| m.content.len()).sum::<usize>() / 4
    }

    /// Truncate large content in old messages (not the most recent `keep_recent`).
    ///
    /// Replaces content over `max_length` chars with a truncated preview.
    pub fn truncate_old_content(&mut self, keep_recent: usize, max_length: usize) {
        let cutoff = self.messages.len().saturating_sub(keep_recent);
        for msg in self.messages[..cutoff].iter_mut() {
            if msg.content.len() > max_length {
                let preview = &msg.content[..max_length];
                msg.content = Cow::Owned(format!(
                    "{preview}\n...[truncated, was {} chars]",
                    msg.content.len()
                ));
            }
        }
    }
}

impl Default for ConversationHistory {
    fn default() -> Self {
        Self::new()
    }
}

// ── Token Usage Tracking ─────────────────────────────────────────

/// Token usage reported by the LLM after a completion.
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    /// Tokens consumed by the input (prompt + history).
    pub input_tokens: usize,
    /// Tokens generated in the output.
    pub output_tokens: usize,
    /// Total tokens (input + output).
    pub total_tokens: usize,
}

/// Budget tracker for the model's context window.
#[derive(Debug, Clone)]
pub struct TokenBudget {
    /// Model's maximum context window (e.g. 128_000 for GPT-4).
    pub context_window: usize,
    /// Accumulated token usage from the most recent turn.
    pub last_usage: TokenUsage,
    /// Number of LLM turns completed.
    pub turns: usize,
}

impl Default for TokenBudget {
    fn default() -> Self {
        Self {
            context_window: 128_000, // safe default
            last_usage: TokenUsage::default(),
            turns: 0,
        }
    }
}

impl TokenBudget {
    /// Create a budget with a specific context window size.
    pub fn with_window(context_window: usize) -> Self {
        Self {
            context_window,
            ..Default::default()
        }
    }

    /// Record token usage from an LLM response.
    pub fn record(&mut self, usage: TokenUsage) {
        self.last_usage = usage;
        self.turns += 1;
    }

    /// Estimated remaining tokens in the context window.
    pub fn remaining(&self) -> usize {
        self.context_window
            .saturating_sub(self.last_usage.total_tokens)
    }

    /// Fraction of the context window used (0.0 – 1.0).
    pub fn usage_ratio(&self) -> f64 {
        if self.context_window == 0 {
            return 1.0;
        }
        self.last_usage.total_tokens as f64 / self.context_window as f64
    }

    /// True when usage exceeds 85% of the context window.
    pub fn is_over_budget(&self) -> bool {
        self.usage_ratio() > 0.85
    }
}

// ── Turn-Level Operations ────────────────────────────────────────

impl ConversationHistory {
    /// Token budget tracker.
    pub fn token_budget(&self) -> &TokenBudget {
        &self.token_budget
    }

    /// Mutable token budget tracker.
    pub fn token_budget_mut(&mut self) -> &mut TokenBudget {
        &mut self.token_budget
    }

    /// Index of the most recent Assistant message, if any.
    pub fn last_model_turn_index(&self) -> Option<usize> {
        self.messages
            .iter()
            .rposition(|m| m.role == Role::Assistant)
    }

    /// Messages added after the last model (Assistant) response.
    ///
    /// These are execution results not yet reflected in token counts.
    pub fn items_after_last_model_turn(&self) -> &[HistoryMessage] {
        match self.last_model_turn_index() {
            Some(idx) if idx + 1 < self.messages.len() => &self.messages[idx + 1..],
            _ => &[],
        }
    }

    /// Drop the last `n` user turns and their paired responses.
    ///
    /// A "user turn" is a User or Execution message followed by an
    /// Assistant response. Dropping a turn removes both the user
    /// message and the assistant reply. The system prompt is always
    /// preserved.
    pub fn drop_last_n_user_turns(&mut self, n: usize) {
        if n == 0 {
            return;
        }

        let mut dropped = 0;
        while dropped < n && self.messages.len() > 1 {
            // Find the last User or Execution message (skip system at index 0)
            let pos = self
                .messages
                .iter()
                .rposition(|m| m.role == Role::User || m.role == Role::Execution);
            match pos {
                Some(p) if p > 0 => {
                    // Remove from this position to end (user msg + any following assistant/exec)
                    self.messages.truncate(p);
                    dropped += 1;
                }
                _ => break,
            }
        }
    }

    /// Replace the entire message list (e.g. after normalization).
    pub fn replace_messages(&mut self, messages: Vec<HistoryMessage>) {
        self.messages = messages;
    }

    /// Get a mutable reference to all messages (for in-place normalization).
    pub fn messages_mut(&mut self) -> &mut Vec<HistoryMessage> {
        &mut self.messages
    }
}
