//! Frozen context store — holds suspended agent contexts for HITL resume.
//!
//! When an agent pauses (ElicitUser), the caller freezes the AgentContext
//! and continuation here. When the user responds, the caller thaws the
//! frozen task and resumes execution.

use std::collections::HashMap;
use std::sync::Mutex;

use super::context::AgentContext;
use super::monad::AgentMonad;

/// A suspended agent execution that can be resumed with user input.
pub struct FrozenTask {
    /// The agent context (conversation history, variables, executor, etc.)
    pub context: AgentContext,
    /// The monadic continuation to resume with.
    pub continuation: AgentMonad,
    /// When this task was frozen.
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// In-memory store for suspended tasks, keyed by A2A task_id.
///
/// Tasks auto-expire after `ttl` duration. Call `gc()` periodically
/// or rely on `thaw()` to skip expired entries.
pub struct FrozenStore {
    tasks: Mutex<HashMap<String, FrozenTask>>,
    /// Time-to-live for frozen tasks (default: 10 minutes).
    ttl: chrono::Duration,
}

impl FrozenStore {
    /// Create a new frozen store with a 10-minute TTL.
    pub fn new() -> Self {
        Self {
            tasks: Mutex::new(HashMap::new()),
            ttl: chrono::Duration::minutes(10),
        }
    }

    /// Create a store with a custom TTL.
    #[allow(dead_code)]
    pub fn with_ttl(ttl: chrono::Duration) -> Self {
        Self {
            tasks: Mutex::new(HashMap::new()),
            ttl,
        }
    }

    /// Freeze a task — store it for later resumption.
    pub fn freeze(&self, task_id: &str, task: FrozenTask) {
        let mut store = self.tasks.lock().expect("frozen store lock poisoned");
        store.insert(task_id.to_string(), task);
    }

    /// Thaw a task — remove and return it for resumption.
    /// Returns `None` if the task doesn't exist or has expired.
    pub fn thaw(&self, task_id: &str) -> Option<FrozenTask> {
        let mut store = self.tasks.lock().expect("frozen store lock poisoned");
        let task = store.remove(task_id)?;
        // Check TTL
        if chrono::Utc::now() - task.created_at > self.ttl {
            tracing::warn!(task_id, "frozen task expired (TTL exceeded)");
            return None;
        }
        Some(task)
    }

    /// Garbage-collect expired tasks.
    pub fn gc(&self) {
        let mut store = self.tasks.lock().expect("frozen store lock poisoned");
        let now = chrono::Utc::now();
        store.retain(|id, task| {
            let alive = now - task.created_at <= self.ttl;
            if !alive {
                tracing::info!(task_id = id, "GC: removing expired frozen task");
            }
            alive
        });
    }

    /// Number of currently frozen tasks.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.tasks.lock().expect("frozen store lock poisoned").len()
    }

    /// Whether the store is empty.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for FrozenStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monad::context::{AgentConfig, AgentContext};
    use crate::monad::monad::AgentMonad;

    #[test]
    fn test_freeze_and_thaw() {
        let store = FrozenStore::new();
        let ctx = AgentContext::new(AgentConfig::default());
        let continuation = AgentMonad::pure("test");

        store.freeze(
            "task-1",
            FrozenTask {
                context: ctx,
                continuation,
                created_at: chrono::Utc::now(),
            },
        );

        assert_eq!(store.len(), 1);

        let thawed = store.thaw("task-1");
        assert!(thawed.is_some());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_thaw_nonexistent() {
        let store = FrozenStore::new();
        assert!(store.thaw("nope").is_none());
    }

    #[test]
    fn test_expired_task() {
        let store = FrozenStore::with_ttl(chrono::Duration::seconds(0));
        let ctx = AgentContext::new(AgentConfig::default());

        store.freeze(
            "task-1",
            FrozenTask {
                context: ctx,
                continuation: AgentMonad::pure("expired"),
                created_at: chrono::Utc::now() - chrono::Duration::seconds(1),
            },
        );

        // Should be expired
        assert!(store.thaw("task-1").is_none());
    }

    #[test]
    fn test_gc() {
        let store = FrozenStore::with_ttl(chrono::Duration::seconds(0));
        let ctx = AgentContext::new(AgentConfig::default());

        store.freeze(
            "task-1",
            FrozenTask {
                context: ctx,
                continuation: AgentMonad::pure("gc-me"),
                created_at: chrono::Utc::now() - chrono::Duration::seconds(1),
            },
        );

        assert_eq!(store.len(), 1);
        store.gc();
        assert_eq!(store.len(), 0);
    }
}
