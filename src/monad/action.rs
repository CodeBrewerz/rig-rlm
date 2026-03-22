//! The `Action` trait — declarative side effects for the agent monad.
//!
//! Each action describes *what* should happen (insert text, call LLM, run code)
//! without specifying *how*. The `AgentContext` interpreter handles execution.

use std::fmt;

use super::capabilities::Capabilities;
use super::orchestrator::Orchestrator;

/// A side effect the agent can perform.
///
/// Actions are the leaves of the monadic tree. Each variant describes
/// a single operation. The context interpreter pattern-matches on the
/// action type and executes it.
///
/// This is intentionally a closed enum (not a trait) because:
/// - The set of primitive actions is fixed and small
/// - Pattern matching gives exhaustiveness checking
/// - No dynamic dispatch overhead
/// - Easy to serialize/log for observability
#[derive(Debug, Clone)]
pub enum Action {
    /// Insert content into conversation history.
    /// Optionally carries multimodal attachments (images, PDFs).
    Insert {
        role: Role,
        content: String,
        #[allow(dead_code)]
        attachments: Vec<super::attachment::Attachment>,
    },

    /// Call the LLM with current conversation state.
    /// Returns the model's text response.
    ModelInference,

    /// Execute code in the sandbox.
    /// Returns structured execution results.
    ExecuteCode { source: String },

    /// Store a named variable in the context.
    Capture { name: String, value: String },

    /// Retrieve a named variable from the context.
    Retrieve { name: String },

    /// Log a structured event (for observability / Turso persistence).
    Log { level: LogLevel, message: String },

    /// Spawn a sub-agent with restricted capabilities (Phase 11).
    ///
    /// The sub-agent runs the given task string through its own
    /// interaction loop with the specified capability restrictions.
    /// Returns the sub-agent's final answer.
    SpawnSubAgent {
        task: String,
        capabilities: Capabilities,
    },

    // ─── Phase 3: Context operations ─────────────────────────────
    /// Load content into an isolated context.
    /// Returns metadata (format, size, lines, token_estimate).
    LoadContext { id: String, content: String },

    /// Search within a named context for a pattern.
    /// Returns formatted search results.
    SearchContext { id: String, pattern: String },

    /// Peek at a range of lines (1-indexed, inclusive) in a context.
    /// Returns the requested lines.
    PeekContext {
        id: String,
        start: usize,
        end: usize,
    },

    /// List all loaded contexts with metadata.
    ListContexts,

    // ─── Phase 7: Reasoning tools ──────────────────────────────
    /// Structured reasoning scratchpad.
    /// Recorded in evidence trail but NOT inserted into conversation history.
    /// Gives the agent a private "thinking" space.
    Think { reasoning: String },

    /// Self-assess progress and confidence.
    /// Recorded in evidence trail for GEPA scoring. Not sent to LLM.
    EvaluateProgress { confidence: f64, remaining: String },

    // ─── Phase 8: Recipe execution ────────────────────────────────
    /// Dynamically define and execute a multi-step pipeline.
    /// The agent generates recipe YAML, the runtime validates, estimates
    /// cost, and executes the full pipeline. Returns a summary of all
    /// step results.
    PlanRecipe { recipe_yaml: String },

    // ─── Context compaction ───────────────────────────────────────
    /// Check if context window is over budget and compact if needed.
    /// Truncates large execution outputs first (fast path), then
    /// summarizes old turns via a one-shot LLM call (smart path).
    CompactContext,

    // ─── Parallel execution (Codex pattern) ───────────────────────
    /// Execute multiple read-only actions concurrently.
    /// All actions in the batch must be `ActionSafety::ReadOnly`.
    /// Returns a JSON array of results.
    ParallelBatch { actions: Vec<Action> },

    // ─── Multi-agent orchestration (Codex pattern) ────────────────
    /// Configure and run multiple sub-agents.
    /// Uses the Orchestrator's strategy (Sequential/Parallel/FanOutFanIn).
    /// Returns a formatted summary of all sub-agent results.
    Orchestrate { orchestrator: Orchestrator },

    // ─── Apply patch (Codex pattern) ──────────────────────────────
    /// Parse and apply a unified diff patch to files.
    /// The patch string follows standard unified diff format.
    /// Returns a summary of files changed.
    ApplyPatch { patch: String },

    // ─── HITL: Human-in-the-Loop elicitation ─────────────────────
    /// Pause execution and ask the user a question.
    /// Returns the user's response text when execution resumes.
    ElicitUser {
        question: String,
        partial_result: Option<String>,
    },

    // ─── Channels: event injection + reply ──────────────────────
    /// Inject a channel event into the conversation history.
    /// The event is formatted as an XML `<channel>` tag.
    ChannelInject(crate::channels::ChannelEvent),

    /// Send a reply through a named spoke.
    ChannelReply {
        spoke: String,
        meta: crate::channels::ChannelMeta,
        text: String,
    },

    /// Non-blocking drain of pending channel events.
    /// Returns a JSON array of any events waiting in the subscription.
    ListenChannels,
}

impl Action {
    /// Short human-readable label for Restate journal entry names.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Insert { .. } => "insert",
            Self::ModelInference => "llm",
            Self::ExecuteCode { .. } => "exec",
            Self::Capture { .. } => "capture",
            Self::Retrieve { .. } => "retrieve",
            Self::Log { .. } => "log",
            Self::SpawnSubAgent { .. } => "spawn",
            Self::LoadContext { .. } => "load_ctx",
            Self::SearchContext { .. } => "search_ctx",
            Self::PeekContext { .. } => "peek_ctx",
            Self::ListContexts => "list_ctx",
            Self::Think { .. } => "think",
            Self::EvaluateProgress { .. } => "eval",
            Self::PlanRecipe { .. } => "recipe",
            Self::CompactContext => "compact",
            Self::ParallelBatch { .. } => "parallel",
            Self::Orchestrate { .. } => "orchestrate",
            Self::ApplyPatch { .. } => "apply_patch",
            Self::ElicitUser { .. } => "elicit_user",
            Self::ChannelInject(_) => "channel_inject",
            Self::ChannelReply { .. } => "channel_reply",
            Self::ListenChannels => "listen_channels",
        }
    }

    /// Returns true if this action performs external I/O (LLM call, code exec,
    /// sub-agent spawn) and should be wrapped in a Restate `ctx.run()`.
    pub fn is_io(&self) -> bool {
        matches!(
            self,
            Self::ModelInference
                | Self::ExecuteCode { .. }
                | Self::SpawnSubAgent { .. }
                | Self::CompactContext
                | Self::ParallelBatch { .. }
                | Self::Orchestrate { .. }
                | Self::ApplyPatch { .. }
                | Self::ElicitUser { .. }
                | Self::ChannelReply { .. }
                | Self::ListenChannels
        )
    }
}

/// Conversation roles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    /// Execution result fed back to the model.
    Execution,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::System => write!(f, "system"),
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
            Self::Execution => write!(f, "execution"),
        }
    }
}

/// Log severity levels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

/// The result of interpreting a single action.
///
/// Actions that produce a string value (ModelInference, ExecuteCode, Retrieve)
/// return `ActionOutput::Value(String)`. Side-effect-only actions (Insert,
/// Capture, Log) return `ActionOutput::Unit`. SUBMIT() returns `Submitted`.
#[derive(Debug, Clone)]
pub enum ActionOutput {
    /// No return value (the action was a pure side effect).
    Unit,
    /// A string value produced by the action.
    Value(String),
    /// A structured SUBMIT result (JSON string from SUBMIT() call).
    /// Signals the interaction loop to terminate with this answer.
    Submitted(String),
    /// Signals the agent loop to suspend and wait for user input.
    /// The caller must freeze the context and return `input-required`.
    Suspended {
        question: String,
        partial_result: Option<String>,
    },
}

impl ActionOutput {
    /// Extract the string value, or return an empty string for Unit.
    ///
    /// For `Submitted`, prepends `[submitted] ` so that `interaction_loop`
    /// can detect the marker via `starts_with("[submitted]")`.
    pub fn into_string(self) -> String {
        match self {
            Self::Unit => String::new(),
            Self::Value(s) => s,
            Self::Submitted(s) => format!("[submitted] {s}"),
            Self::Suspended { question, .. } => format!("[elicit] {question}"),
        }
    }

    /// Returns true if this output contains a value.
    pub fn has_value(&self) -> bool {
        matches!(
            self,
            Self::Value(_) | Self::Submitted(_) | Self::Suspended { .. }
        )
    }

    /// Returns true if this is a SUBMIT result.
    pub fn is_submitted(&self) -> bool {
        matches!(self, Self::Submitted(_))
    }

    /// Returns true if this signals a suspension for user input.
    pub fn is_suspended(&self) -> bool {
        matches!(self, Self::Suspended { .. })
    }
}
