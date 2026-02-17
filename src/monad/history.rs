//! Conversation history management.
//!
//! Structured message history with role-based insertion and conversion
//! to rig's message format for LLM calls.

use rig::{
    agent::Text,
    message::{AssistantContent, Message, UserContent},
    OneOrMany,
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

impl Default for ConversationHistory {
    fn default() -> Self {
        Self::new()
    }
}
