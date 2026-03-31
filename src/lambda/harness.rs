//! # Meta-Harness Techniques for λ-RLM
//!
//! Port of key ideas from Meta-Harness (76.4% on Terminal-Bench 2.0).
//! These are agent-loop optimizations that reduce friction and improve
//! reliability when the λ-RLM pipeline drives an autonomous agent.
//!
//! ## Techniques Implemented
//!
//! 1. **Environment Bootstrapping** — Gather a workspace snapshot before
//!    the first LLM call, eliminating 2-5 exploration turns.
//!
//! 2. **Marker-Based Early Completion** — Detect command completion via
//!    unique echo markers instead of waiting for fixed timeouts.
//!
//! 3. **Structured Analysis-Plan-Execute Loop** — Force explicit reasoning
//!    before every action, providing training signal for rubrics.
//!
//! 4. **Double Completion Confirmation** — Require verification against
//!    a checklist before finalizing a task.
//!
//! Reference: <https://github.com/meta-harness/meta-harness-tbench2-artifact>

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ═══════════════════════════════════════════════════════════════════════
// 1. ENVIRONMENT BOOTSTRAPPING
// ═══════════════════════════════════════════════════════════════════════

/// A snapshot of the execution environment, gathered once before the
/// agent loop starts. Eliminates 2-5 early exploration turns.
#[derive(Debug, Clone, Default)]
pub struct EnvironmentSnapshot {
    /// Current working directory.
    pub cwd: Option<String>,
    /// File listing from the workspace root.
    pub file_listing: Vec<String>,
    /// Available language runtimes (e.g., "python3 3.12.0", "rustc 1.83.0").
    pub available_languages: Vec<String>,
    /// Available package managers (e.g., "cargo", "pip3").
    pub package_managers: Vec<String>,
    /// System memory info (free -h output).
    pub memory_info: Option<String>,
    /// Custom key-value environment facts.
    pub extra: HashMap<String, String>,
}

impl EnvironmentSnapshot {
    /// Create a new empty snapshot.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a snapshot from raw shell output sections.
    ///
    /// Expected input: output from a compound bootstrap command with
    /// `@@SECTION@@` delimiters (same pattern as Meta-Harness).
    pub fn from_shell_output(raw: &str) -> Self {
        let mut snapshot = Self::new();
        let mut sections: HashMap<String, Vec<String>> = HashMap::new();
        let mut current_key: Option<String> = None;

        for line in raw.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("@@") && trimmed.ends_with("@@") {
                let key = trimmed.trim_matches('@').to_string();
                if let Some(ref k) = current_key {
                    sections.entry(k.clone()).or_default();
                }
                current_key = Some(key);
            } else if let Some(ref key) = current_key {
                sections.entry(key.clone()).or_default().push(line.to_string());
            }
        }

        if let Some(lines) = sections.get("PWD") {
            snapshot.cwd = lines.first().map(|s| s.trim().to_string());
        }
        if let Some(lines) = sections.get("LS") {
            snapshot.file_listing = lines.iter()
                .filter(|l| !l.trim().is_empty())
                .map(|l| l.trim().to_string())
                .collect();
        }
        if let Some(lines) = sections.get("LANG") {
            snapshot.available_languages = lines.iter()
                .filter(|l| !l.trim().is_empty() && !l.contains("not found"))
                .map(|l| l.trim().to_string())
                .collect();
        }
        if let Some(lines) = sections.get("PKG") {
            snapshot.package_managers = lines.iter()
                .filter(|l| !l.trim().is_empty() && !l.contains("not found"))
                .map(|l| l.trim().to_string())
                .collect();
        }
        if let Some(lines) = sections.get("MEM") {
            let mem = lines.join("\n").trim().to_string();
            if !mem.is_empty() {
                snapshot.memory_info = Some(mem);
            }
        }

        snapshot
    }

    /// Generate the bootstrap shell command (single compound command).
    ///
    /// This is the same pattern Meta-Harness uses: a single `&&`-chained
    /// command with `@@SECTION@@` markers for easy parsing.
    pub fn bootstrap_command() -> &'static str {
        concat!(
            "echo '@@PWD@@' && pwd && ",
            "echo '@@LS@@' && ls -la 2>/dev/null | head -30 && ",
            "echo '@@LANG@@' && ",
            "(python3 --version 2>&1 || echo 'python3: not found') && ",
            "(rustc --version 2>&1 || echo 'rustc: not found') && ",
            "(node --version 2>&1 || echo 'node: not found') && ",
            "(go version 2>&1 || echo 'go: not found') && ",
            "(gcc --version 2>&1 | head -1 || echo 'gcc: not found') && ",
            "echo '@@PKG@@' && ",
            "(cargo --version 2>&1 || echo 'cargo: not found') && ",
            "(pip3 --version 2>&1 || echo 'pip3: not found') && ",
            "(npm --version 2>&1 || echo 'npm: not found') && ",
            "echo '@@MEM@@' && free -h 2>/dev/null | head -2 || true",
        )
    }

    /// Format the snapshot as a compact string for prompt injection.
    pub fn to_prompt_block(&self) -> String {
        let mut parts = Vec::new();

        if let Some(ref cwd) = self.cwd {
            parts.push(format!("Working directory: {}", cwd));
        }

        if !self.file_listing.is_empty() {
            let count = self.file_listing.len();
            if count <= 20 {
                parts.push(format!("Files ({}):\n{}", count, self.file_listing.join("\n")));
            } else {
                let preview: Vec<_> = self.file_listing[..15].to_vec();
                parts.push(format!(
                    "Files ({} entries):\n{}\n... ({} more)",
                    count,
                    preview.join("\n"),
                    count - 15,
                ));
            }
        }

        if !self.available_languages.is_empty() {
            parts.push(format!("Languages: {}", self.available_languages.join("; ")));
        }

        if !self.package_managers.is_empty() {
            parts.push(format!("Packages: {}", self.package_managers.join("; ")));
        }

        if let Some(ref mem) = self.memory_info {
            parts.push(format!("Memory: {}", mem));
        }

        for (k, v) in &self.extra {
            parts.push(format!("{}: {}", k, v));
        }

        if parts.is_empty() {
            return String::new();
        }

        format!("[Environment Snapshot]\n{}", parts.join("\n"))
    }

    /// Check if the snapshot has any useful data.
    pub fn is_empty(&self) -> bool {
        self.cwd.is_none()
            && self.file_listing.is_empty()
            && self.available_languages.is_empty()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 2. MARKER-BASED EARLY COMPLETION
// ═══════════════════════════════════════════════════════════════════════

/// Unique marker prefix for command completion detection.
const MARKER_PREFIX: &str = "__CMDEND__";

/// A command with an associated timeout and completion marker.
#[derive(Debug, Clone)]
pub struct MarkedCommand {
    /// The command to execute.
    pub command: String,
    /// Maximum time to wait for completion.
    pub max_duration: Duration,
    /// Unique completion marker.
    pub marker: String,
    /// Sequence number for ordering.
    pub seq: usize,
}

/// Generator for marked commands.
///
/// Instead of waiting a fixed `duration_sec` for every command, append a
/// unique `echo '__CMDEND__N__'` after each command. Poll the output for
/// the marker — if found early, move on immediately.
#[derive(Debug)]
pub struct MarkerGenerator {
    seq: usize,
}

impl MarkerGenerator {
    pub fn new() -> Self {
        Self { seq: 0 }
    }

    /// Wrap a command with a completion marker.
    pub fn mark(&mut self, command: &str, max_duration: Duration) -> MarkedCommand {
        self.seq += 1;
        let marker = format!("{}{}__{}", MARKER_PREFIX, self.seq, "_END");

        MarkedCommand {
            command: command.to_string(),
            max_duration,
            marker,
            seq: self.seq,
        }
    }

    /// Generate the actual command string to send (command + marker echo).
    pub fn command_with_marker(cmd: &MarkedCommand) -> String {
        format!("{}\necho '{}'", cmd.command, cmd.marker)
    }

    /// Check if output contains the expected marker.
    pub fn is_complete(output: &str, cmd: &MarkedCommand) -> bool {
        output.contains(&cmd.marker)
    }

    /// Strip all markers from output so the LLM sees clean text.
    pub fn clean_output(output: &str) -> String {
        output
            .lines()
            .filter(|line| !line.contains(MARKER_PREFIX))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl Default for MarkerGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of polling for command completion.
#[derive(Debug, Clone)]
pub struct CompletionResult {
    /// The cleaned output (markers stripped).
    pub output: String,
    /// Whether the command completed before the timeout.
    pub completed_early: bool,
    /// Time saved by early detection (compared to max_duration).
    pub time_saved: Duration,
    /// Actual time waited.
    pub actual_duration: Duration,
}

// ═══════════════════════════════════════════════════════════════════════
// 3. ANALYSIS-PLAN-EXECUTE STRUCTURED LOOP
// ═══════════════════════════════════════════════════════════════════════

/// A structured step in the agent loop that forces explicit reasoning.
///
/// Every agent action must contain:
/// - **Analysis**: What do we observe? What has been accomplished?
/// - **Plan**: What will we do next and why?
/// - **Actions**: The actual commands/queries to execute.
///
/// This structure provides training signal for the rubric system
/// and prevents the agent from acting without reasoning.
#[derive(Debug, Clone)]
pub struct StructuredStep {
    /// Analysis of current state.
    pub analysis: String,
    /// Plan for next actions.
    pub plan: String,
    /// Actions to execute.
    pub actions: Vec<StepAction>,
    /// Whether this step signals task completion.
    pub is_complete: bool,
    /// Timestamp when step was created.
    pub timestamp: Instant,
}

/// A single action within a structured step.
#[derive(Debug, Clone)]
pub struct StepAction {
    /// The command or query to execute.
    pub command: String,
    /// Duration hint (for marker polling).
    pub duration_hint: Duration,
    /// Optional label for logging.
    pub label: Option<String>,
}

impl StructuredStep {
    /// Create a new structured step.
    pub fn new(analysis: &str, plan: &str) -> Self {
        Self {
            analysis: analysis.to_string(),
            plan: plan.to_string(),
            actions: Vec::new(),
            is_complete: false,
            timestamp: Instant::now(),
        }
    }

    /// Add an action to this step.
    pub fn add_action(&mut self, command: &str, duration_hint: Duration) -> &mut Self {
        self.actions.push(StepAction {
            command: command.to_string(),
            duration_hint,
            label: None,
        });
        self
    }

    /// Add a labeled action.
    pub fn add_labeled_action(&mut self, label: &str, command: &str, duration: Duration) -> &mut Self {
        self.actions.push(StepAction {
            command: command.to_string(),
            duration_hint: duration,
            label: Some(label.to_string()),
        });
        self
    }

    /// Mark this step as a completion step.
    pub fn mark_complete(&mut self) -> &mut Self {
        self.is_complete = true;
        self
    }

    /// Generate the structured prompt section for this step.
    ///
    /// Used to inject the analysis/plan structure into the LLM prompt,
    /// forcing explicit reasoning before action.
    pub fn to_prompt_suffix(&self) -> String {
        format!(
            concat!(
                "Before providing commands, you MUST include:\n",
                "1. **Analysis**: What do you observe? What has been accomplished? What needs to be done?\n",
                "2. **Plan**: What commands will you run and why? Be specific about expected outcomes.\n",
                "3. **Commands**: The actual commands to execute.\n\n",
                "Previous step context:\n",
                "- Analysis: {}\n",
                "- Plan: {}\n",
            ),
            if self.analysis.is_empty() { "(initial step)" } else { &self.analysis },
            if self.plan.is_empty() { "(initial step)" } else { &self.plan },
        )
    }
}

/// Parse an LLM response into a structured step.
///
/// Extracts `Analysis:`, `Plan:`, and command blocks from the response.
pub fn parse_structured_response(response: &str) -> StructuredStep {
    let mut analysis = String::new();
    let mut plan = String::new();
    let mut commands = Vec::new();

    let mut current_section = "";

    for line in response.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("**Analysis**:") || trimmed.starts_with("Analysis:") {
            current_section = "analysis";
            let content = trimmed
                .trim_start_matches("**Analysis**:")
                .trim_start_matches("Analysis:")
                .trim();
            if !content.is_empty() {
                analysis.push_str(content);
            }
        } else if trimmed.starts_with("**Plan**:") || trimmed.starts_with("Plan:") {
            current_section = "plan";
            let content = trimmed
                .trim_start_matches("**Plan**:")
                .trim_start_matches("Plan:")
                .trim();
            if !content.is_empty() {
                plan.push_str(content);
            }
        } else if trimmed.starts_with("```") {
            current_section = "code";
        } else {
            match current_section {
                "analysis" => {
                    analysis.push(' ');
                    analysis.push_str(trimmed);
                }
                "plan" => {
                    plan.push(' ');
                    plan.push_str(trimmed);
                }
                "code" => {
                    if !trimmed.is_empty() {
                        commands.push(StepAction {
                            command: trimmed.to_string(),
                            duration_hint: Duration::from_secs(1),
                            label: None,
                        });
                    }
                }
                _ => {}
            }
        }
    }

    StructuredStep {
        analysis: analysis.trim().to_string(),
        plan: plan.trim().to_string(),
        actions: commands,
        is_complete: false,
        timestamp: Instant::now(),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 4. DOUBLE COMPLETION CONFIRMATION
// ═══════════════════════════════════════════════════════════════════════

/// State machine for double-confirmation before task completion.
///
/// When the agent signals completion:
/// 1. First call → returns a verification checklist
/// 2. Second call → actually marks complete
///
/// This prevents premature termination, giving the agent one more
/// chance to review against the original task requirements.
#[derive(Debug, Clone)]
pub struct CompletionGate {
    /// Whether a completion request is pending verification.
    pending: bool,
    /// The original task instruction (for the verification prompt).
    original_task: String,
    /// How many completions were rejected (agent decided not ready).
    pub rejections: usize,
    /// How many completions were confirmed.
    pub confirmations: usize,
}

impl CompletionGate {
    /// Create a new completion gate for a task.
    pub fn new(task: &str) -> Self {
        Self {
            pending: false,
            original_task: task.to_string(),
            rejections: 0,
            confirmations: 0,
        }
    }

    /// Process a completion request. Returns:
    /// - `Ok(None)` — completion NOT yet confirmed, returns verification prompt
    /// - `Ok(Some(true))` — completion confirmed (second call)
    /// - `Ok(Some(false))` — completion was pending but agent sent non-complete action
    pub fn request_completion(&mut self, terminal_output: &str) -> (bool, Option<String>) {
        if self.pending {
            // Second call — actually complete
            self.pending = false;
            self.confirmations += 1;
            (true, None)
        } else {
            // First call — show verification checklist
            self.pending = true;
            let checklist = self.verification_prompt(terminal_output);
            (false, Some(checklist))
        }
    }

    /// Reset the pending state (agent sent a non-complete action).
    pub fn cancel_pending(&mut self) {
        if self.pending {
            self.pending = false;
            self.rejections += 1;
        }
    }

    /// Check if a completion is pending.
    pub fn is_pending(&self) -> bool {
        self.pending
    }

    /// Generate the verification prompt (Meta-Harness style).
    fn verification_prompt(&self, terminal_output: &str) -> String {
        format!(
            concat!(
                "Original task:\n{}\n\n",
                "Current state:\n{}\n\n",
                "═══ COMPLETION VERIFICATION CHECKLIST ═══\n\n",
                "Before confirming, verify each item:\n\n",
                "1. □ Does the solution meet ALL requirements in the original task?\n",
                "2. □ Does the solution handle edge cases (empty input, large data, etc.)?\n",
                "3. □ Have you verified from multiple perspectives?\n",
                "   - Test engineer: Does it pass all tests? □\n",
                "   - QA engineer: Are there regressions? □\n",
                "   - End user: Does it solve the actual problem? □\n",
                "4. □ Is the workspace clean (no leftover temp files, debug output)?\n\n",
                "If ALL checks pass, call task_complete again to confirm.\n",
                "If ANY check fails, continue working on the task.",
            ),
            self.original_task,
            &terminal_output[..terminal_output.len().min(2000)],
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════
// COMPOSITE: HarnessConfig — combines all techniques
// ═══════════════════════════════════════════════════════════════════════

/// Configuration for the full Meta-Harness optimization suite.
#[derive(Debug, Clone)]
pub struct HarnessConfig {
    /// Enable environment bootstrapping.
    pub bootstrap_env: bool,
    /// Enable marker-based early completion.
    pub use_markers: bool,
    /// Enable structured analysis-plan-execute.
    pub structured_steps: bool,
    /// Enable double completion confirmation.
    pub double_confirm: bool,
    /// Maximum output bytes before truncation (Meta-Harness uses 30KB).
    pub max_output_bytes: usize,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            bootstrap_env: true,
            use_markers: true,
            structured_steps: true,
            double_confirm: true,
            max_output_bytes: 30_000,
        }
    }
}

/// Truncate output to `max_bytes`, keeping head + tail if needed.
///
/// Matches Meta-Harness's `_limit_output_length` behavior.
pub fn limit_output_length(output: &str, max_bytes: usize) -> String {
    if output.len() <= max_bytes {
        return output.to_string();
    }

    let head_size = max_bytes * 2 / 3;
    let tail_size = max_bytes / 3;

    let head = &output[..head_size];
    let tail = &output[output.len() - tail_size..];
    let omitted = output.len() - head_size - tail_size;

    format!(
        "{}\n\n[... {} bytes omitted ...]\n\n{}",
        head, omitted, tail
    )
}

// ═══════════════════════════════════════════════════════════════════════
// 5. META-GEPA: HARNESS CONFIGURATION EVOLUTION
// ═══════════════════════════════════════════════════════════════════════

/// Numeric parameters within HarnessConfig that GEPA can mutate.
///
/// Boolean knobs (bootstrap, markers, structured_steps, double_confirm)
/// are encoded as floats: threshold > 0.5 → enabled. This lets GEPA
/// smoothly explore the enable/disable boundary.
#[derive(Debug, Clone)]
pub struct HarnessGenes {
    /// P(bootstrap_env) — 0.0–1.0, threshold=0.5 → enabled.
    pub bootstrap_prob: f64,
    /// P(use_markers) — 0.0–1.0, threshold=0.5 → enabled.
    pub markers_prob: f64,
    /// P(structured_steps) — 0.0–1.0, threshold=0.5 → enabled.
    pub structured_prob: f64,
    /// P(double_confirm) — 0.0–1.0, threshold=0.5 → enabled.
    pub confirm_prob: f64,
    /// max_output_bytes — 5000–100000.
    pub max_output_bytes: f64,
    /// marker_poll_interval_ms — 100–5000.
    pub marker_poll_ms: f64,
}

impl Default for HarnessGenes {
    fn default() -> Self {
        Self {
            bootstrap_prob: 0.8,   // strongly enabled by default
            markers_prob: 0.8,
            structured_prob: 0.8,
            confirm_prob: 0.8,
            max_output_bytes: 30_000.0,
            marker_poll_ms: 500.0,
        }
    }
}

impl HarnessGenes {
    /// Decode back to a concrete HarnessConfig.
    pub fn to_config(&self) -> HarnessConfig {
        HarnessConfig {
            bootstrap_env: self.bootstrap_prob > 0.5,
            use_markers: self.markers_prob > 0.5,
            structured_steps: self.structured_prob > 0.5,
            double_confirm: self.confirm_prob > 0.5,
            max_output_bytes: (self.max_output_bytes as usize).clamp(5_000, 100_000),
        }
    }

    /// Encode a concrete HarnessConfig into genes.
    pub fn from_config(config: &HarnessConfig) -> Self {
        Self {
            bootstrap_prob: if config.bootstrap_env { 0.8 } else { 0.2 },
            markers_prob: if config.use_markers { 0.8 } else { 0.2 },
            structured_prob: if config.structured_steps { 0.8 } else { 0.2 },
            confirm_prob: if config.double_confirm { 0.8 } else { 0.2 },
            max_output_bytes: config.max_output_bytes as f64,
            marker_poll_ms: 500.0,
        }
    }

    /// Get all gene values as a flat vector (for GEPA mutation).
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.bootstrap_prob,
            self.markers_prob,
            self.structured_prob,
            self.confirm_prob,
            self.max_output_bytes,
            self.marker_poll_ms,
        ]
    }

    /// Create from a flat vector.
    pub fn from_vec(v: &[f64]) -> Self {
        assert!(v.len() >= 6, "HarnessGenes requires 6 values");
        Self {
            bootstrap_prob: v[0].clamp(0.0, 1.0),
            markers_prob: v[1].clamp(0.0, 1.0),
            structured_prob: v[2].clamp(0.0, 1.0),
            confirm_prob: v[3].clamp(0.0, 1.0),
            max_output_bytes: v[4].clamp(5_000.0, 100_000.0),
            marker_poll_ms: v[5].clamp(100.0, 5_000.0),
        }
    }

    /// Mutate genes with Gaussian noise (GEPA-style numeric mutation).
    ///
    /// Uses different sigma for boolean-threshold genes vs continuous genes.
    pub fn mutate(&self, rng_seed: u64) -> Self {
        // Simple deterministic pseudo-random based on seed
        let mut hash = rng_seed;
        let mut next_f64 = || -> f64 {
            hash = hash.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((hash >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0 // ∈ [-1, 1]
        };

        let sigma_bool = 0.15;   // mutation sigma for boolean-threshold genes
        let sigma_output = 5000.0; // mutation sigma for max_output_bytes
        let sigma_poll = 200.0;   // mutation sigma for poll interval

        Self {
            bootstrap_prob: (self.bootstrap_prob + next_f64() * sigma_bool).clamp(0.0, 1.0),
            markers_prob: (self.markers_prob + next_f64() * sigma_bool).clamp(0.0, 1.0),
            structured_prob: (self.structured_prob + next_f64() * sigma_bool).clamp(0.0, 1.0),
            confirm_prob: (self.confirm_prob + next_f64() * sigma_bool).clamp(0.0, 1.0),
            max_output_bytes: (self.max_output_bytes + next_f64() * sigma_output).clamp(5_000.0, 100_000.0),
            marker_poll_ms: (self.marker_poll_ms + next_f64() * sigma_poll).clamp(100.0, 5_000.0),
        }
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "bootstrap={:.0}% markers={:.0}% structured={:.0}% confirm={:.0}% output={}KB poll={}ms",
            self.bootstrap_prob * 100.0,
            self.markers_prob * 100.0,
            self.structured_prob * 100.0,
            self.confirm_prob * 100.0,
            self.max_output_bytes as usize / 1000,
            self.marker_poll_ms as usize,
        )
    }
}

/// A trial record — one evaluation of a HarnessGenes configuration.
#[derive(Debug, Clone)]
pub struct HarnessTrial {
    /// The gene configuration tested.
    pub genes: HarnessGenes,
    /// Task score achieved (0.0–1.0).
    pub score: f64,
    /// Tokens consumed.
    pub total_tokens: i64,
    /// Latency in seconds.
    pub latency_secs: f64,
    /// Cost-efficiency: score / (tokens/1000).
    pub cost_efficiency: f64,
    /// Combined fitness: α·score + β·efficiency - γ·latency.
    pub fitness: f64,
}

/// Meta-GEPA: Evolves HarnessConfig parameters through GEPA selection.
///
/// This is the "automated harness evolution" from Meta-Harness applied
/// to λ-RLM's own GEPA loop. The system discovers which harness
/// techniques help for which task classes.
///
/// # Evolution Protocol
///
/// 1. Maintain a population of HarnessGenes candidates
/// 2. For each agent task, select the current best config
/// 3. After each task, record the trial (score, tokens, latency)
/// 4. Every `evolve_interval` tasks, run tournament selection + mutation
/// 5. Apply the 1/5th success rule to adapt mutation rate
///
/// # Fitness Function
///
/// `fitness = α·score + β·cost_efficiency - γ·normalized_latency`
///
/// Where:
/// - α = 0.6 (task success is primary)
/// - β = 0.3 (token efficiency matters)
/// - γ = 0.1 (latency is a tiebreaker)
#[derive(Debug, Clone)]
pub struct HarnessEvolver {
    /// Population of candidate configurations.
    pub population: Vec<HarnessGenes>,
    /// Trial history for each population member.
    pub trials: Vec<Vec<HarnessTrial>>,
    /// Index of the current best configuration.
    pub best_idx: usize,
    /// Number of evolution cycles completed.
    pub evolution_count: usize,
    /// How many tasks between evolution cycles.
    pub evolve_interval: usize,
    /// Tasks since last evolution.
    pub tasks_since_evolve: usize,
    /// Mutation rate (adapted by 1/5th rule).
    pub mutation_rate: f64,
    /// Count of successful mutations (score improved).
    pub successful_mutations: usize,
    /// Total mutations attempted.
    pub total_mutations: usize,

    // Fitness weights
    /// Weight for task score.
    pub alpha: f64,
    /// Weight for cost-efficiency.
    pub beta: f64,
    /// Weight for latency penalty.
    pub gamma: f64,
}

impl HarnessEvolver {
    /// Create a new evolver with a default population.
    pub fn new() -> Self {
        // Seed population with diverse configurations
        let population = vec![
            // 1. Default (all enabled, standard params)
            HarnessGenes::default(),
            // 2. Minimal (no bootstrap, no structured)
            HarnessGenes {
                bootstrap_prob: 0.2,
                markers_prob: 0.8,
                structured_prob: 0.2,
                confirm_prob: 0.8,
                max_output_bytes: 20_000.0,
                marker_poll_ms: 300.0,
            },
            // 3. Max structured (heavy analysis)
            HarnessGenes {
                bootstrap_prob: 0.8,
                markers_prob: 0.8,
                structured_prob: 0.95,
                confirm_prob: 0.95,
                max_output_bytes: 50_000.0,
                marker_poll_ms: 500.0,
            },
            // 4. Fast (aggressive polling, small output)
            HarnessGenes {
                bootstrap_prob: 0.8,
                markers_prob: 0.95,
                structured_prob: 0.5,
                confirm_prob: 0.3,
                max_output_bytes: 15_000.0,
                marker_poll_ms: 200.0,
            },
        ];

        let pop_size = population.len();
        Self {
            population,
            trials: vec![Vec::new(); pop_size],
            best_idx: 0,
            evolution_count: 0,
            evolve_interval: 5,
            tasks_since_evolve: 0,
            mutation_rate: 0.3,
            successful_mutations: 0,
            total_mutations: 0,
            alpha: 0.6,
            beta: 0.3,
            gamma: 0.1,
        }
    }

    /// Get the current best HarnessConfig.
    pub fn current_config(&self) -> HarnessConfig {
        self.population[self.best_idx].to_config()
    }

    /// Get the current best genes.
    pub fn current_genes(&self) -> &HarnessGenes {
        &self.population[self.best_idx]
    }

    /// Record a trial result for the current best configuration.
    pub fn record_trial(&mut self, score: f64, total_tokens: i64, latency_secs: f64) {
        let cost_efficiency = if total_tokens > 0 {
            score / (total_tokens as f64 / 1000.0)
        } else {
            0.0
        };

        let fitness = self.compute_fitness(score, cost_efficiency, latency_secs);

        let trial = HarnessTrial {
            genes: self.population[self.best_idx].clone(),
            score,
            total_tokens,
            latency_secs,
            cost_efficiency,
            fitness,
        };

        self.trials[self.best_idx].push(trial);
        self.tasks_since_evolve += 1;

        // Check if it's time to evolve
        if self.tasks_since_evolve >= self.evolve_interval {
            self.evolve();
        }
    }

    /// Compute fitness from components.
    fn compute_fitness(&self, score: f64, cost_efficiency: f64, latency_secs: f64) -> f64 {
        // Normalize latency (assume 60s is "bad", 1s is "good")
        let norm_latency = (latency_secs / 60.0).clamp(0.0, 1.0);
        // Normalize cost_efficiency (cap at 2.0 score/Ktok)
        let norm_eff = (cost_efficiency / 2.0).clamp(0.0, 1.0);

        self.alpha * score + self.beta * norm_eff - self.gamma * norm_latency
    }

    /// Run one evolution cycle: evaluate, select, mutate.
    pub fn evolve(&mut self) {
        self.tasks_since_evolve = 0;

        // Calculate mean fitness for each population member
        let mut fitnesses: Vec<(usize, f64)> = self.trials.iter().enumerate()
            .map(|(i, trials)| {
                if trials.is_empty() {
                    return (i, 0.0);
                }
                let mean_fitness = trials.iter()
                    .map(|t| t.fitness)
                    .sum::<f64>() / trials.len() as f64;
                (i, mean_fitness)
            })
            .collect();

        // Sort by fitness (descending)
        fitnesses.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let prev_best_fitness = fitnesses.iter()
            .find(|(i, _)| *i == self.best_idx)
            .map(|(_, f)| *f)
            .unwrap_or(0.0);

        // New best is the top-performing candidate
        self.best_idx = fitnesses[0].0;
        let new_best_fitness = fitnesses[0].1;

        eprintln!(
            "🧬 [MetaGEPA] Evolution #{}: best_idx={} fitness={:.4} (prev={:.4})",
            self.evolution_count, self.best_idx, new_best_fitness, prev_best_fitness
        );

        // Mutate the bottom half of the population, seeding from the top half
        let pop_size = self.population.len();
        let cutoff = pop_size / 2;

        for rank in cutoff..pop_size {
            let loser_idx = fitnesses[rank].0;
            let parent_idx = fitnesses[rank % cutoff].0;

            let parent_fitness = fitnesses.iter()
                .find(|(i, _)| *i == parent_idx)
                .map(|(_, f)| *f)
                .unwrap_or(0.0);

            // Seed for mutation based on evolution count + index for determinism
            let seed = (self.evolution_count as u64 * 1000 + loser_idx as u64) ^ 0xDEADBEEF;
            let child = self.population[parent_idx].mutate(seed);

            eprintln!(
                "🧬 [MetaGEPA]   slot {} ← mutate({}): {}",
                loser_idx, parent_idx, child.summary()
            );

            self.population[loser_idx] = child;
            self.trials[loser_idx].clear(); // reset trials for new config
            self.total_mutations += 1;
        }

        // 1/5th success rule for mutation rate adaptation
        if new_best_fitness > prev_best_fitness {
            self.successful_mutations += 1;
        }

        if self.total_mutations >= 5 {
            let success_ratio = self.successful_mutations as f64 / self.total_mutations as f64;
            if success_ratio > 0.2 {
                // Too many successes → increase mutation rate (explore more)
                self.mutation_rate = (self.mutation_rate * 1.2).min(0.8);
            } else {
                // Too few successes → decrease mutation rate (exploit more)
                self.mutation_rate = (self.mutation_rate * 0.8).max(0.05);
            }
            eprintln!(
                "🧬 [MetaGEPA]   1/5th rule: success_ratio={:.2} → mutation_rate={:.3}",
                success_ratio, self.mutation_rate
            );
        }

        self.evolution_count += 1;
    }

    /// Get summary statistics.
    pub fn summary(&self) -> String {
        let total_trials: usize = self.trials.iter().map(|t| t.len()).sum();
        let best_genes = &self.population[self.best_idx];
        format!(
            "🧬 MetaGEPA: {} evolutions, {} total trials, mutation_rate={:.3}\n  Best (slot {}): {}",
            self.evolution_count,
            total_trials,
            self.mutation_rate,
            self.best_idx,
            best_genes.summary(),
        )
    }

    /// Get the evolution history as structured data (for diagnostics).
    pub fn fitness_history(&self) -> Vec<(usize, f64)> {
        self.trials.iter().enumerate()
            .flat_map(|(slot, trials)| {
                trials.iter().map(move |t| (slot, t.fitness))
            })
            .collect()
    }
}

impl Default for HarnessEvolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_environment_snapshot_from_shell() {
        let raw = "\
@@PWD@@
/home/user/project
@@LS@@
total 48
drwxr-xr-x 5 user user 4096 Mar 30 src
-rw-r--r-- 1 user user 1234 Mar 30 Cargo.toml
-rw-r--r-- 1 user user 5678 Mar 30 README.md
@@LANG@@
Python 3.12.0
rustc 1.83.0
node: not found
@@PKG@@
cargo 1.83.0
pip3: not found
@@MEM@@
              total        used        free
Mem:           62Gi       8.2Gi        54Gi";

        let snap = EnvironmentSnapshot::from_shell_output(raw);

        assert_eq!(snap.cwd, Some("/home/user/project".to_string()));
        assert_eq!(snap.file_listing.len(), 4); // total + 3 entries
        assert_eq!(snap.available_languages.len(), 2); // python3, rustc
        assert_eq!(snap.package_managers.len(), 1); // cargo only (pip3 not found)
        assert!(snap.memory_info.is_some());
    }

    #[test]
    fn test_snapshot_to_prompt() {
        let mut snap = EnvironmentSnapshot::new();
        snap.cwd = Some("/app".to_string());
        snap.available_languages = vec!["Python 3.12".to_string(), "rustc 1.83".to_string()];
        snap.package_managers = vec!["cargo 1.83".to_string()];

        let prompt = snap.to_prompt_block();
        assert!(prompt.starts_with("[Environment Snapshot]"));
        assert!(prompt.contains("Working directory: /app"));
        assert!(prompt.contains("Python 3.12"));
        assert!(prompt.contains("cargo 1.83"));
    }

    #[test]
    fn test_empty_snapshot() {
        let snap = EnvironmentSnapshot::new();
        assert!(snap.is_empty());
        assert!(snap.to_prompt_block().is_empty());
    }

    #[test]
    fn test_marker_generator() {
        let mut marker_gen = MarkerGenerator::new();
        let cmd1 = marker_gen.mark("ls -la", Duration::from_secs(1));
        let cmd2 = marker_gen.mark("cargo build", Duration::from_secs(30));

        assert_eq!(cmd1.seq, 1);
        assert_eq!(cmd2.seq, 2);
        assert_ne!(cmd1.marker, cmd2.marker);

        let full = MarkerGenerator::command_with_marker(&cmd1);
        assert!(full.contains("ls -la"));
        assert!(full.contains(&cmd1.marker));

        // Test completion detection
        let output_with_marker = format!("some output\n{}\nmore output", cmd1.marker);
        assert!(MarkerGenerator::is_complete(&output_with_marker, &cmd1));
        assert!(!MarkerGenerator::is_complete(&output_with_marker, &cmd2));
    }

    #[test]
    fn test_marker_clean_output() {
        let output = format!(
            "file1.rs\nfile2.rs\n{}1___END\nfile3.rs",
            MARKER_PREFIX
        );
        let cleaned = MarkerGenerator::clean_output(&output);
        assert!(cleaned.contains("file1.rs"));
        assert!(cleaned.contains("file3.rs"));
        assert!(!cleaned.contains(MARKER_PREFIX));
    }

    #[test]
    fn test_completion_gate_double_confirm() {
        let mut gate = CompletionGate::new("Fix the bug in main.rs");

        // First completion request → returns verification prompt
        let (confirmed, prompt) = gate.request_completion("test output");
        assert!(!confirmed);
        assert!(prompt.is_some());
        assert!(prompt.unwrap().contains("COMPLETION VERIFICATION"));
        assert!(gate.is_pending());

        // Second completion request → actually confirms
        let (confirmed, prompt) = gate.request_completion("test output");
        assert!(confirmed);
        assert!(prompt.is_none());
        assert!(!gate.is_pending());
        assert_eq!(gate.confirmations, 1);
    }

    #[test]
    fn test_completion_gate_cancel() {
        let mut gate = CompletionGate::new("Fix the bug");

        // Request completion
        let (confirmed, _) = gate.request_completion("output");
        assert!(!confirmed);
        assert!(gate.is_pending());

        // Agent decides not ready, sends more commands
        gate.cancel_pending();
        assert!(!gate.is_pending());
        assert_eq!(gate.rejections, 1);

        // Next completion request starts fresh (needs 2 calls again)
        let (confirmed, prompt) = gate.request_completion("output");
        assert!(!confirmed);
        assert!(prompt.is_some());
    }

    #[test]
    fn test_structured_step() {
        let mut step = StructuredStep::new(
            "The tests are failing due to a missing import",
            "Add the import and rerun tests",
        );
        step.add_action("cargo test", Duration::from_secs(10));
        step.add_labeled_action("verify", "cat src/main.rs | head -5", Duration::from_millis(100));

        assert_eq!(step.actions.len(), 2);
        assert_eq!(step.actions[1].label, Some("verify".to_string()));
        assert!(!step.is_complete);

        let prompt = step.to_prompt_suffix();
        assert!(prompt.contains("Analysis"));
        assert!(prompt.contains("Plan"));
    }

    #[test]
    fn test_parse_structured_response() {
        let response = "\
**Analysis**: The build is failing because of a missing dependency.\
We need to add the `serde` crate to Cargo.toml.

**Plan**: Add serde to dependencies, then rebuild.

```bash
echo '[dependencies]' >> Cargo.toml
cargo build
```";

        let step = parse_structured_response(response);
        assert!(step.analysis.contains("missing dependency"));
        assert!(step.plan.contains("Add serde"));
        assert!(!step.actions.is_empty());
    }

    #[test]
    fn test_limit_output_length() {
        let short = "hello world";
        assert_eq!(limit_output_length(short, 100), short);

        let long = "x".repeat(100);
        let truncated = limit_output_length(&long, 50);
        assert!(truncated.len() < 100);
        assert!(truncated.contains("bytes omitted"));
    }

    #[test]
    fn test_harness_config_default() {
        let config = HarnessConfig::default();
        assert!(config.bootstrap_env);
        assert!(config.use_markers);
        assert!(config.structured_steps);
        assert!(config.double_confirm);
        assert_eq!(config.max_output_bytes, 30_000);
    }

    #[test]
    fn test_bootstrap_command_is_valid() {
        let cmd = EnvironmentSnapshot::bootstrap_command();
        assert!(cmd.contains("pwd"));
        assert!(cmd.contains("@@PWD@@"));
        assert!(cmd.contains("@@LANG@@"));
        assert!(cmd.contains("@@PKG@@"));
        assert!(cmd.contains("@@MEM@@"));
    }

    // ── Meta-GEPA Tests ──

    #[test]
    fn test_harness_genes_default_roundtrip() {
        let genes = HarnessGenes::default();
        let config = genes.to_config();
        assert!(config.bootstrap_env);
        assert!(config.use_markers);
        assert!(config.structured_steps);
        assert!(config.double_confirm);
        assert_eq!(config.max_output_bytes, 30_000);

        // Roundtrip: config → genes → config
        let genes2 = HarnessGenes::from_config(&config);
        let config2 = genes2.to_config();
        assert_eq!(config.bootstrap_env, config2.bootstrap_env);
        assert_eq!(config.max_output_bytes, config2.max_output_bytes);
    }

    #[test]
    fn test_harness_genes_vec_roundtrip() {
        let genes = HarnessGenes::default();
        let vec = genes.to_vec();
        assert_eq!(vec.len(), 6);

        let genes2 = HarnessGenes::from_vec(&vec);
        assert!((genes.bootstrap_prob - genes2.bootstrap_prob).abs() < f64::EPSILON);
        assert!((genes.max_output_bytes - genes2.max_output_bytes).abs() < f64::EPSILON);
    }

    #[test]
    fn test_harness_genes_mutation() {
        let parent = HarnessGenes::default();
        let child1 = parent.mutate(42);
        let child2 = parent.mutate(123);

        // Different seeds → different children
        assert!((child1.bootstrap_prob - child2.bootstrap_prob).abs() > f64::EPSILON
            || (child1.max_output_bytes - child2.max_output_bytes).abs() > f64::EPSILON);

        // Children should be within reasonable bounds
        assert!(child1.bootstrap_prob >= 0.0 && child1.bootstrap_prob <= 1.0);
        assert!(child1.max_output_bytes >= 5_000.0 && child1.max_output_bytes <= 100_000.0);
        assert!(child1.marker_poll_ms >= 100.0 && child1.marker_poll_ms <= 5_000.0);
    }

    #[test]
    fn test_harness_evolver_lifecycle() {
        let mut evolver = HarnessEvolver::new();
        assert_eq!(evolver.population.len(), 4);
        assert_eq!(evolver.evolution_count, 0);
        assert_eq!(evolver.best_idx, 0);

        // Record trials (not enough to trigger evolution)
        evolver.record_trial(0.8, 500, 2.0);
        evolver.record_trial(0.9, 400, 1.5);
        assert_eq!(evolver.tasks_since_evolve, 2);
        assert_eq!(evolver.evolution_count, 0);

        // Record more to trigger evolution (interval = 5)
        evolver.record_trial(0.7, 600, 3.0);
        evolver.record_trial(0.85, 450, 2.5);
        evolver.record_trial(0.95, 350, 1.0);
        // Should have evolved now
        assert_eq!(evolver.evolution_count, 1);
        assert_eq!(evolver.tasks_since_evolve, 0);
    }

    #[test]
    fn test_harness_evolver_fitness() {
        let evolver = HarnessEvolver::new();

        // High score, good efficiency, low latency → high fitness
        let f1 = evolver.compute_fitness(0.9, 1.5, 2.0);
        // Low score, poor efficiency, high latency → low fitness
        let f2 = evolver.compute_fitness(0.3, 0.1, 50.0);

        assert!(f1 > f2, "f1={:.4} should be > f2={:.4}", f1, f2);

        // Pure score comparison (same efficiency, same latency)
        let f3 = evolver.compute_fitness(1.0, 1.0, 5.0);
        let f4 = evolver.compute_fitness(0.5, 1.0, 5.0);
        assert!(f3 > f4);
    }

    #[test]
    fn test_harness_evolver_summary() {
        let mut evolver = HarnessEvolver::new();
        evolver.record_trial(0.8, 500, 2.0);
        let summary = evolver.summary();
        assert!(summary.contains("MetaGEPA"));
        assert!(summary.contains("1 total trials"));
        assert!(summary.contains("bootstrap="));
    }

    #[test]
    fn test_harness_genes_disabled_config() {
        let genes = HarnessGenes {
            bootstrap_prob: 0.1,  // below threshold
            markers_prob: 0.1,
            structured_prob: 0.1,
            confirm_prob: 0.1,
            max_output_bytes: 10_000.0,
            marker_poll_ms: 200.0,
        };
        let config = genes.to_config();
        assert!(!config.bootstrap_env);
        assert!(!config.use_markers);
        assert!(!config.structured_steps);
        assert!(!config.double_confirm);
        assert_eq!(config.max_output_bytes, 10_000);
    }
}
