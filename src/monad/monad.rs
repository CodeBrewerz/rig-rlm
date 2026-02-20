//! The `AgentMonad` — a free monad for composing agent computations.
//!
//! This is the Rust translation of Agentica's `HistoryMonad[A]` pattern.
//! Since Rust lacks higher-kinded types, we encode the monad as an enum
//! with boxed continuations. The key insight: we describe *what* the agent
//! should do as a data structure, then interpret it later in `AgentContext::run`.

use super::action::{Action, ActionOutput, Role};

/// A computation that produces a value of type `String` and may perform
/// side effects via `Action`s along the way.
///
/// This is deliberately monomorphic (`String` output only) because:
/// - Agent actions universally produce/consume strings (LLM text, code, etc.)
/// - Avoids the complexity of full generic HKTs in Rust
/// - The DSRs Signature layer (Phase 13) handles typed I/O on top of this
pub enum AgentMonad {
    /// A completed computation with a final value.
    Pure(String),

    /// An action to perform, followed by a continuation that receives
    /// the action's output and produces the next computation step.
    Perform {
        action: Action,
        next: Box<dyn FnOnce(ActionOutput) -> AgentMonad + Send>,
    },
}

impl AgentMonad {
    // ─── Constructors ───────────────────────────────────────────────

    /// Wrap a value — no side effects.
    pub fn pure(value: impl Into<String>) -> Self {
        Self::Pure(value.into())
    }

    /// Create a Perform step: do `action`, then pass output to `next`.
    pub fn perform<F>(action: Action, next: F) -> Self
    where
        F: FnOnce(ActionOutput) -> AgentMonad + Send + 'static,
    {
        Self::Perform {
            action,
            next: Box::new(next),
        }
    }

    /// Monadic bind — sequence this computation with `f`.
    ///
    /// `self.bind(f)` means "run self, then pass the result to f".
    /// This is the core sequencing primitive.
    pub fn bind<F>(self, f: F) -> Self
    where
        F: FnOnce(String) -> AgentMonad + Send + 'static,
    {
        match self {
            Self::Pure(value) => f(value),
            Self::Perform { action, next } => Self::Perform {
                action,
                next: Box::new(move |output| {
                    let rest = next(output);
                    rest.bind(f)
                }),
            },
        }
    }

    /// Sequence: run `self`, ignore its value, then run `next`.
    pub fn then(self, next: AgentMonad) -> Self {
        self.bind(move |_| next)
    }

    // ─── Convenience action builders ────────────────────────────────

    /// Insert a message into conversation history.
    pub fn insert(role: Role, content: impl Into<String>) -> Self {
        Self::perform(
            Action::Insert {
                role,
                content: content.into(),
            },
            |_| Self::Pure(String::new()),
        )
    }

    /// Call the LLM and return its response text.
    pub fn model_inference() -> Self {
        Self::perform(Action::ModelInference, |output| {
            Self::Pure(output.into_string())
        })
    }

    /// Execute code in the sandbox and return the output.
    pub fn execute_code(source: impl Into<String>) -> Self {
        Self::perform(
            Action::ExecuteCode {
                source: source.into(),
            },
            |output| Self::Pure(output.into_string()),
        )
    }

    /// Store a named variable.
    pub fn capture(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self::perform(
            Action::Capture {
                name: name.into(),
                value: value.into(),
            },
            |_| Self::Pure(String::new()),
        )
    }

    /// Retrieve a named variable.
    pub fn retrieve(name: impl Into<String>) -> Self {
        Self::perform(Action::Retrieve { name: name.into() }, |output| {
            Self::Pure(output.into_string())
        })
    }

    /// Emit a log message.
    pub fn log(level: super::action::LogLevel, message: impl Into<String>) -> Self {
        Self::perform(
            Action::Log {
                level,
                message: message.into(),
            },
            |_| Self::Pure(String::new()),
        )
    }

    // ─── Phase 3: Context operation builders ────────────────────────

    /// Load content into an isolated context.
    pub fn load_context(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::perform(
            Action::LoadContext {
                id: id.into(),
                content: content.into(),
            },
            |output| Self::Pure(output.into_string()),
        )
    }

    /// Search within a named context for a pattern.
    pub fn search_context(id: impl Into<String>, pattern: impl Into<String>) -> Self {
        Self::perform(
            Action::SearchContext {
                id: id.into(),
                pattern: pattern.into(),
            },
            |output| Self::Pure(output.into_string()),
        )
    }

    /// Peek at a range of lines in a context.
    pub fn peek_context(id: impl Into<String>, start: usize, end: usize) -> Self {
        Self::perform(
            Action::PeekContext {
                id: id.into(),
                start,
                end,
            },
            |output| Self::Pure(output.into_string()),
        )
    }

    /// List all loaded contexts.
    pub fn list_contexts() -> Self {
        Self::perform(Action::ListContexts, |output| {
            Self::Pure(output.into_string())
        })
    }

    // ─── Phase 7: Reasoning builders ────────────────────────────────

    /// Record structured reasoning (not inserted into conversation history).
    pub fn think(reasoning: impl Into<String>) -> Self {
        Self::perform(
            Action::Think {
                reasoning: reasoning.into(),
            },
            |_| Self::Pure(String::new()),
        )
    }

    /// Record a self-assessment of progress and confidence.
    pub fn evaluate_progress(confidence: f64, remaining: impl Into<String>) -> Self {
        Self::perform(
            Action::EvaluateProgress {
                confidence,
                remaining: remaining.into(),
            },
            |output| Self::Pure(output.into_string()),
        )
    }

    // ─── Phase 8: Recipe builder ────────────────────────────────────

    /// Submit a recipe YAML for dynamic pipeline execution.
    /// The runtime parses, validates, and executes all steps.
    pub fn plan_recipe(recipe_yaml: impl Into<String>) -> Self {
        Self::perform(
            Action::PlanRecipe {
                recipe_yaml: recipe_yaml.into(),
            },
            |output| Self::Pure(output.into_string()),
        )
    }

    // ─── Context compaction builder ─────────────────────────────────

    /// Check and compact context if over token budget.
    /// Returns empty string if no compaction needed, or "compacted" if it was.
    pub fn compact_context() -> Self {
        Self::perform(Action::CompactContext, |_| Self::Pure(String::new()))
    }

    // ─── Parallel + Orchestration builders (Codex patterns) ─────────

    /// Execute multiple read-only actions concurrently.
    /// Returns a JSON array of their results.
    pub fn parallel_batch(actions: Vec<Action>) -> Self {
        Self::perform(Action::ParallelBatch { actions }, |output| {
            Self::Pure(output.into_string())
        })
    }

    /// Run multiple sub-agents via the orchestrator.
    /// Returns a formatted summary of all results.
    pub fn orchestrate(orchestrator: super::orchestrator::Orchestrator) -> Self {
        Self::perform(Action::Orchestrate { orchestrator }, |output| {
            Self::Pure(output.into_string())
        })
    }
}

// Debug impl that doesn't try to print the continuation closure.
impl std::fmt::Debug for AgentMonad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pure(v) => f.debug_tuple("Pure").field(v).finish(),
            Self::Perform { action, .. } => {
                f.debug_struct("Perform").field("action", action).finish()
            }
        }
    }
}
