//! Lifecycle hooks system — inspired by Codex CLI.
//!
//! Pluggable hook system for injecting custom logic at key lifecycle points.
//! Hooks can be registered for events like LLM calls, code execution,
//! sub-agent spawning, and session lifecycle.
//!
//! Hook results control flow:
//! - `Continue` — proceed normally
//! - `FailedContinue(reason)` — log warning, keep going
//! - `Abort(reason)` — stop the current operation

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

/// Events that hooks can subscribe to.
#[derive(Debug, Clone)]
pub enum HookEvent {
    /// Fired before an LLM call.
    BeforeLlmCall { turn: usize, message_count: usize },

    /// Fired after an LLM call completes.
    AfterLlmCall {
        turn: usize,
        response_len: usize,
        duration: Duration,
    },

    /// Fired before code execution.
    BeforeCodeExec { turn: usize, code_preview: String },

    /// Fired after code execution completes.
    AfterCodeExec {
        turn: usize,
        success: bool,
        duration: Duration,
        output_preview: String,
    },

    /// Fired before a sub-agent is spawned.
    BeforeSubAgent { task: String },

    /// Fired after a sub-agent completes.
    AfterSubAgent {
        task: String,
        result_len: usize,
        success: bool,
    },

    /// Fired when a session starts.
    SessionStart { task: String },

    /// Fired when a session ends.
    SessionEnd {
        turns: usize,
        final_answer_len: usize,
    },

    /// Fired after context compaction.
    AfterCompaction {
        messages_before: usize,
        messages_after: usize,
    },
}

impl fmt::Display for HookEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BeforeLlmCall { turn, .. } => write!(f, "before_llm_call(turn={turn})"),
            Self::AfterLlmCall { turn, .. } => write!(f, "after_llm_call(turn={turn})"),
            Self::BeforeCodeExec { turn, .. } => write!(f, "before_code_exec(turn={turn})"),
            Self::AfterCodeExec { turn, success, .. } => {
                write!(f, "after_code_exec(turn={turn}, success={success})")
            }
            Self::BeforeSubAgent { task } => {
                write!(f, "before_sub_agent(task={})", &task[..task.len().min(40)])
            }
            Self::AfterSubAgent { task, success, .. } => {
                write!(
                    f,
                    "after_sub_agent(task={}, success={success})",
                    &task[..task.len().min(40)]
                )
            }
            Self::SessionStart { task } => {
                write!(f, "session_start(task={})", &task[..task.len().min(40)])
            }
            Self::SessionEnd { turns, .. } => write!(f, "session_end(turns={turns})"),
            Self::AfterCompaction {
                messages_before,
                messages_after,
            } => {
                write!(f, "after_compaction({messages_before}→{messages_after})")
            }
        }
    }
}

/// Result of a hook execution.
#[derive(Debug)]
pub enum HookResult {
    /// Hook completed successfully, proceed normally.
    Continue,
    /// Hook failed but other hooks should still execute and the
    /// operation should continue.
    FailedContinue(String),
    /// Hook failed, abort the current operation.
    Abort(String),
}

impl HookResult {
    /// Should the operation be aborted?
    pub fn should_abort(&self) -> bool {
        matches!(self, Self::Abort(_))
    }

    /// Get the error message if any.
    pub fn error_message(&self) -> Option<&str> {
        match self {
            Self::FailedContinue(msg) | Self::Abort(msg) => Some(msg),
            Self::Continue => None,
        }
    }
}

/// A registered hook function.
pub type HookFn = Arc<dyn Fn(&HookEvent) -> HookResult + Send + Sync>;

/// A named hook registration.
struct RegisteredHook {
    name: String,
    func: HookFn,
}

/// Registry of lifecycle hooks.
///
/// Hooks are executed in registration order. If any hook returns
/// `Abort`, subsequent hooks are skipped and the operation is aborted.
pub struct HookRegistry {
    hooks: Vec<RegisteredHook>,
}

impl Default for HookRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl HookRegistry {
    /// Create an empty hook registry.
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    /// Register a named hook function.
    pub fn register(&mut self, name: impl Into<String>, func: HookFn) {
        self.hooks.push(RegisteredHook {
            name: name.into(),
            func,
        });
    }

    /// Register a hook from a closure.
    pub fn on<F>(&mut self, name: impl Into<String>, func: F)
    where
        F: Fn(&HookEvent) -> HookResult + Send + Sync + 'static,
    {
        self.register(name, Arc::new(func));
    }

    /// Fire all hooks for an event.
    ///
    /// Returns `Ok(())` if all hooks passed, or the first abort error.
    pub fn fire(&self, event: &HookEvent) -> Result<(), String> {
        for hook in &self.hooks {
            let result = (hook.func)(event);
            match result {
                HookResult::Continue => {}
                HookResult::FailedContinue(reason) => {
                    tracing::warn!(
                        hook = %hook.name,
                        event = %event,
                        reason = %reason,
                        "hook failed (continuing)"
                    );
                }
                HookResult::Abort(reason) => {
                    tracing::error!(
                        hook = %hook.name,
                        event = %event,
                        reason = %reason,
                        "hook abort"
                    );
                    return Err(format!("Hook '{}' aborted: {reason}", hook.name));
                }
            }
        }
        Ok(())
    }

    /// Number of registered hooks.
    pub fn len(&self) -> usize {
        self.hooks.len()
    }

    /// Are there any hooks registered?
    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_registry_fires_ok() {
        let registry = HookRegistry::new();
        let event = HookEvent::SessionStart {
            task: "test".to_string(),
        };
        assert!(registry.fire(&event).is_ok());
    }

    #[test]
    fn continue_hooks_pass() {
        let mut registry = HookRegistry::new();
        registry.on("test", |_| HookResult::Continue);

        let event = HookEvent::BeforeLlmCall {
            turn: 1,
            message_count: 5,
        };
        assert!(registry.fire(&event).is_ok());
    }

    #[test]
    fn abort_hook_stops_execution() {
        let mut registry = HookRegistry::new();
        registry.on("aborter", |_| {
            HookResult::Abort("safety violation".to_string())
        });

        let event = HookEvent::BeforeCodeExec {
            turn: 1,
            code_preview: "rm -rf /".to_string(),
        };
        let result = registry.fire(&event);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("safety violation"));
    }

    #[test]
    fn failed_continue_does_not_abort() {
        let mut registry = HookRegistry::new();
        let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count_clone = call_count.clone();

        registry.on("logger", move |_| {
            count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            HookResult::FailedContinue("log failed".to_string())
        });

        let event = HookEvent::AfterLlmCall {
            turn: 1,
            response_len: 100,
            duration: Duration::from_millis(500),
        };
        assert!(registry.fire(&event).is_ok());
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn multiple_hooks_execute_in_order() {
        let mut registry = HookRegistry::new();
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));

        let order1 = order.clone();
        registry.on("first", move |_| {
            order1.lock().unwrap().push(1);
            HookResult::Continue
        });

        let order2 = order.clone();
        registry.on("second", move |_| {
            order2.lock().unwrap().push(2);
            HookResult::Continue
        });

        let event = HookEvent::SessionEnd {
            turns: 5,
            final_answer_len: 100,
        };
        registry.fire(&event).unwrap();
        assert_eq!(*order.lock().unwrap(), vec![1, 2]);
    }

    #[test]
    fn abort_skips_subsequent_hooks() {
        let mut registry = HookRegistry::new();
        let reached = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let reached_clone = reached.clone();

        registry.on("aborter", |_| HookResult::Abort("stop".to_string()));
        registry.on("never_reached", move |_| {
            reached_clone.store(true, std::sync::atomic::Ordering::SeqCst);
            HookResult::Continue
        });

        let event = HookEvent::BeforeLlmCall {
            turn: 1,
            message_count: 3,
        };
        let _ = registry.fire(&event);
        assert!(!reached.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn hook_event_display() {
        let event = HookEvent::AfterCodeExec {
            turn: 3,
            success: true,
            duration: Duration::from_secs(1),
            output_preview: "hello".to_string(),
        };
        let display = format!("{event}");
        assert!(display.contains("after_code_exec"));
        assert!(display.contains("turn=3"));
    }
}
