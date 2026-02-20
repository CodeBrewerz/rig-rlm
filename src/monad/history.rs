//! Conversation history management.
//!
//! Structured message history with role-based insertion and conversion
//! to rig's message format for LLM calls.

use rig::{
    OneOrMany,
    agent::Text,
    message::{AssistantContent, Message, UserContent},
};

use super::action::Role;

/// A single message in the conversation history.
#[derive(Debug, Clone)]
pub struct HistoryMessage {
    pub role: Role,
    pub content: String,
}

/// Manages the full conversation history for an agent session.
#[derive(Debug, Clone)]
pub struct ConversationHistory {
    messages: Vec<HistoryMessage>,
}

impl ConversationHistory {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
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
                content: content.into(),
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
                    latest_user_prompt = msg.content.clone();
                    history.push(Message::User {
                        content: OneOrMany::one(UserContent::Text(Text {
                            text: msg.content.clone(),
                        })),
                    });
                }
                Role::Assistant => {
                    history.push(Message::Assistant {
                        content: OneOrMany::one(AssistantContent::Text(Text {
                            text: msg.content.clone(),
                        })),
                        id: None,
                    });
                }
                Role::Execution => {
                    // Execution results are inserted as user messages
                    // (the model sees them as feedback from the environment).
                    history.push(Message::User {
                        content: OneOrMany::one(UserContent::Text(Text {
                            text: msg.content.clone(),
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
                content: format!("[Context Summary]\n{summary}"),
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
                msg.content = format!("{preview}\n...[truncated, was {} chars]", msg.content.len());
            }
        }
    }
}

impl Default for ConversationHistory {
    fn default() -> Self {
        Self::new()
    }
}
