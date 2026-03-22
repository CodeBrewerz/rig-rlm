//! Parallel tool execution — inspired by Codex CLI.
//!
//! Enables multiple independent actions to run concurrently within a single
//! turn. Uses a read/write lock pattern:
//! - Read-lock for safe operations (context queries, retrieve)
//! - Write-lock for mutating operations (code execution, insert)
//!
//! This plus cancellation support allows much faster multi-tool turns.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use super::action::{Action, ActionOutput};

/// Classification of action safety for parallelism.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionSafety {
    /// Safe to run in parallel with other ReadOnly actions.
    /// Uses a read-lock (multiple concurrent allowed).
    ReadOnly,
    /// Mutates state — requires exclusive access.
    /// Uses a write-lock (blocks all other actions).
    Mutating,
}

impl Action {
    /// Classify whether this action is safe for parallel execution.
    pub fn safety(&self) -> ActionSafety {
        match self {
            // Read-only: these don't modify conversation or executor state
            Self::Retrieve { .. }
            | Self::SearchContext { .. }
            | Self::PeekContext { .. }
            | Self::ListContexts
            | Self::Think { .. }
            | Self::EvaluateProgress { .. }
            | Self::ListenChannels
            | Self::Log { .. } => ActionSafety::ReadOnly,

            // Mutating: these modify conversation history, execute code, etc.
            Self::Insert { .. }
            | Self::ModelInference
            | Self::ExecuteCode { .. }
            | Self::Capture { .. }
            | Self::SpawnSubAgent { .. }
            | Self::LoadContext { .. }
            | Self::PlanRecipe { .. }
            | Self::CompactContext
            | Self::Orchestrate { .. }
            | Self::ApplyPatch { .. }
            | Self::ElicitUser { .. }
            | Self::ChannelInject(_)
            | Self::ChannelReply { .. } => ActionSafety::Mutating,

            // ParallelBatch: inherently parallel (contains only read-only actions)
            Self::ParallelBatch { .. } => ActionSafety::ReadOnly,
        }
    }

    /// Can this action run in parallel with other actions?
    pub fn supports_parallel(&self) -> bool {
        self.safety() == ActionSafety::ReadOnly
    }
}

/// Runtime for executing actions with parallelism control.
///
/// Uses a read/write lock to allow multiple read-only actions to
/// execute concurrently while serializing mutating actions.
pub struct ParallelRuntime {
    /// RwLock controlling concurrent access to agent state.
    lock: Arc<RwLock<()>>,
}

impl Default for ParallelRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl ParallelRuntime {
    pub fn new() -> Self {
        Self {
            lock: Arc::new(RwLock::new(())),
        }
    }

    /// Execute multiple actions with appropriate locking.
    ///
    /// Read-only actions grab a read-lock (concurrent).
    /// Mutating actions grab a write-lock (exclusive).
    /// Returns results in the same order as input actions.
    pub async fn execute_batch<F, Fut>(
        &self,
        actions: Vec<Action>,
        executor: F,
    ) -> Vec<ActionResult>
    where
        F: Fn(Action) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<ActionOutput, String>> + Send + 'static,
    {
        let mut handles = Vec::new();

        for action in actions {
            let lock = self.lock.clone();
            let exec = executor.clone();
            let safety = action.safety();

            let handle = tokio::spawn(async move {
                let start = Instant::now();

                let result = match safety {
                    ActionSafety::ReadOnly => {
                        let _guard = lock.read().await;
                        exec(action.clone()).await
                    }
                    ActionSafety::Mutating => {
                        let _guard = lock.write().await;
                        exec(action.clone()).await
                    }
                };

                ActionResult {
                    action_label: action.label().to_string(),
                    output: result,
                    duration: start.elapsed(),
                    safety,
                }
            });

            handles.push(handle);
        }

        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(result) => results.push(result),
                Err(e) => results.push(ActionResult {
                    action_label: "unknown".to_string(),
                    output: Err(format!("Task panicked: {e}")),
                    duration: Duration::ZERO,
                    safety: ActionSafety::Mutating,
                }),
            }
        }

        results
    }
}

/// Result from a single parallel action execution.
#[derive(Debug)]
pub struct ActionResult {
    /// Label of the action (for logging).
    pub action_label: String,
    /// The action output, or an error message.
    pub output: Result<ActionOutput, String>,
    /// How long the action took.
    pub duration: Duration,
    /// Safety classification used for locking.
    pub safety: ActionSafety,
}

impl ActionResult {
    /// Did the action succeed?
    pub fn is_ok(&self) -> bool {
        self.output.is_ok()
    }

    /// Get the output string if successful.
    pub fn output_string(&self) -> Option<String> {
        self.output.as_ref().ok().map(|o| o.clone().into_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_only_actions_classified_correctly() {
        assert_eq!(
            Action::Retrieve {
                name: "x".to_string()
            }
            .safety(),
            ActionSafety::ReadOnly
        );
        assert_eq!(Action::ListContexts.safety(), ActionSafety::ReadOnly);
        assert_eq!(
            Action::Think {
                reasoning: "hmm".to_string()
            }
            .safety(),
            ActionSafety::ReadOnly
        );
    }

    #[test]
    fn mutating_actions_classified_correctly() {
        assert_eq!(Action::ModelInference.safety(), ActionSafety::Mutating);
        assert_eq!(
            Action::ExecuteCode {
                source: "x".to_string()
            }
            .safety(),
            ActionSafety::Mutating
        );
        assert_eq!(Action::CompactContext.safety(), ActionSafety::Mutating);
    }

    #[test]
    fn supports_parallel_mirrors_safety() {
        assert!(Action::ListContexts.supports_parallel());
        assert!(!Action::ModelInference.supports_parallel());
    }

    #[tokio::test]
    async fn parallel_runtime_executes_batch() {
        let runtime = ParallelRuntime::new();

        let actions = vec![
            Action::Retrieve {
                name: "x".to_string(),
            },
            Action::ListContexts,
        ];

        let results = runtime
            .execute_batch(actions, |_action| async {
                Ok(ActionOutput::Value("result".to_string()))
            })
            .await;

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.is_ok()));
    }
}
