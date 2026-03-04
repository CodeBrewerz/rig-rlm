//! History normalization passes.
//!
//! Prepares the conversation history for reliable LLM consumption by:
//! 1. Removing empty messages
//! 2. Ensuring strict role alternation (merging consecutive same-role msgs)
//! 3. Truncating oversized execution outputs

use std::borrow::Cow;

use super::action::Role;
use super::history::HistoryMessage;

/// Default maximum length for execution output (8 KB).
const DEFAULT_MAX_EXEC_OUTPUT: usize = 8_000;

/// Run all normalization passes on a message list.
///
/// Call this before each LLM invocation to ensure the history
/// conforms to model expectations.
pub fn normalize_history(messages: &mut Vec<HistoryMessage>) {
    remove_empty_messages(messages);
    ensure_role_alternation(messages);
    truncate_execution_output(messages, DEFAULT_MAX_EXEC_OUTPUT);
}

/// Remove messages with empty or whitespace-only content.
pub fn remove_empty_messages(messages: &mut Vec<HistoryMessage>) {
    messages.retain(|m| !m.content.trim().is_empty());
}

/// Merge consecutive messages from the same role.
///
/// Many LLM APIs (OpenAI, Anthropic) require strict user/assistant
/// alternation. When multiple execution results arrive in sequence,
/// or the agent produces multiple assistant chunks, they must be
/// concatenated into a single message.
pub fn ensure_role_alternation(messages: &mut Vec<HistoryMessage>) {
    if messages.len() < 2 {
        return;
    }

    let mut merged: Vec<HistoryMessage> = Vec::with_capacity(messages.len());
    for msg in messages.drain(..) {
        if let Some(last) = merged.last_mut() {
            if last.role == msg.role {
                last.content.to_mut().push('\n');
                last.content.to_mut().push_str(&msg.content);
                continue;
            }
        }
        merged.push(msg);
    }
    *messages = merged;
}

/// Truncate execution outputs that exceed `max_length` characters.
///
/// Only affects `Role::Execution` messages. Prevents a single large
/// stdout/stderr from dominating the context window.
pub fn truncate_execution_output(messages: &mut Vec<HistoryMessage>, max_length: usize) {
    for msg in messages.iter_mut() {
        if msg.role == Role::Execution && msg.content.len() > max_length {
            let original_len = msg.content.len();
            // Keep first half and last quarter of the budget
            let head = max_length * 2 / 3;
            let tail = max_length / 3;
            let tail_start = original_len.saturating_sub(tail);
            msg.content = Cow::Owned(format!(
                "{}\n\n...[truncated {} chars, showing first {} + last {}]...\n\n{}",
                &msg.content[..head],
                original_len - head - (original_len - tail_start),
                head,
                original_len - tail_start,
                &msg.content[tail_start..],
            ));
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: Role, content: &str) -> HistoryMessage {
        HistoryMessage {
            role,
            content: Cow::Owned(content.to_string()),
            attachments: vec![],
        }
    }

    #[test]
    fn remove_empty_drops_blank_and_whitespace() {
        let mut msgs = vec![
            msg(Role::User, "hello"),
            msg(Role::Execution, ""),
            msg(Role::Execution, "   "),
            msg(Role::Assistant, "world"),
        ];
        remove_empty_messages(&mut msgs);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].content, "hello");
        assert_eq!(msgs[1].content, "world");
    }

    #[test]
    fn alternation_merges_consecutive_same_role() {
        let mut msgs = vec![
            msg(Role::User, "task"),
            msg(Role::Execution, "line 1"),
            msg(Role::Execution, "line 2"),
            msg(Role::Execution, "line 3"),
            msg(Role::Assistant, "done"),
        ];
        ensure_role_alternation(&mut msgs);
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[1].role, Role::Execution);
        assert!(msgs[1].content.contains("line 1"));
        assert!(msgs[1].content.contains("line 2"));
        assert!(msgs[1].content.contains("line 3"));
        assert_eq!(msgs[2].role, Role::Assistant);
    }

    #[test]
    fn alternation_preserves_already_alternating() {
        let mut msgs = vec![
            msg(Role::User, "a"),
            msg(Role::Assistant, "b"),
            msg(Role::User, "c"),
            msg(Role::Assistant, "d"),
        ];
        ensure_role_alternation(&mut msgs);
        assert_eq!(msgs.len(), 4);
    }

    #[test]
    fn alternation_handles_empty_input() {
        let mut msgs: Vec<HistoryMessage> = vec![];
        ensure_role_alternation(&mut msgs);
        assert!(msgs.is_empty());
    }

    #[test]
    fn alternation_handles_single_message() {
        let mut msgs = vec![msg(Role::User, "solo")];
        ensure_role_alternation(&mut msgs);
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn truncate_caps_long_execution_output() {
        let long_output = "x".repeat(10_000);
        let mut msgs = vec![
            msg(Role::User, "run code"),
            msg(Role::Execution, &long_output),
            msg(Role::Assistant, "result"),
        ];
        truncate_execution_output(&mut msgs, 1000);
        assert!(msgs[1].content.len() < 10_000, "should be truncated");
        assert!(msgs[1].content.contains("[truncated"));
    }

    #[test]
    fn truncate_leaves_short_output_alone() {
        let mut msgs = vec![
            msg(Role::User, "run code"),
            msg(Role::Execution, "short output"),
            msg(Role::Assistant, "result"),
        ];
        truncate_execution_output(&mut msgs, 1000);
        assert_eq!(msgs[1].content, "short output");
    }

    #[test]
    fn truncate_only_affects_execution_role() {
        let long_msg = "x".repeat(10_000);
        let mut msgs = vec![msg(Role::User, &long_msg), msg(Role::Assistant, &long_msg)];
        truncate_execution_output(&mut msgs, 1000);
        assert_eq!(
            msgs[0].content.len(),
            10_000,
            "User msg should be untouched"
        );
        assert_eq!(
            msgs[1].content.len(),
            10_000,
            "Assistant msg should be untouched"
        );
    }

    #[test]
    fn normalize_full_pipeline() {
        let long_output = "x".repeat(10_000);
        let mut msgs = vec![
            msg(Role::System, "You are helpful"),
            msg(Role::User, "do something"),
            msg(Role::Execution, ""),
            msg(Role::Execution, &long_output),
            msg(Role::Execution, "more output"),
            msg(Role::Assistant, "done"),
        ];
        normalize_history(&mut msgs);
        // Empty execution removed, consecutive executions merged, long output truncated
        assert_eq!(msgs.len(), 4); // System, User, Execution(merged+truncated), Assistant
        assert!(msgs[2].role == Role::Execution);
        assert!(msgs[2].content.contains("[truncated"));
    }
}
