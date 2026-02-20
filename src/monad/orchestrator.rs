//! Multi-agent orchestrator — inspired by Codex CLI.
//!
//! Structured orchestration for spawning and managing multiple sub-agents.
//! Supports sequential execution, parallel fan-out/fan-in, and
//! configurable timeouts per agent.

use std::time::Duration;

use super::capabilities::Capabilities;

/// Specification for a sub-agent to be spawned.
#[derive(Debug, Clone)]
pub struct SubAgentSpec {
    /// Human-readable name for this agent.
    pub name: String,
    /// The task prompt for this agent.
    pub task: String,
    /// Capability restrictions for this agent.
    pub capabilities: Capabilities,
    /// Maximum time this agent is allowed to run.
    pub timeout: Duration,
    /// Maximum number of turns before force-stopping.
    pub max_turns: usize,
}

impl SubAgentSpec {
    /// Create a new sub-agent spec with defaults.
    pub fn new(name: impl Into<String>, task: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            task: task.into(),
            capabilities: Capabilities::default(),
            timeout: Duration::from_secs(120),
            max_turns: 10,
        }
    }

    /// Set capability restrictions.
    pub fn with_capabilities(mut self, caps: Capabilities) -> Self {
        self.capabilities = caps;
        self
    }

    /// Set timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set max turns.
    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }
}

/// Result from a completed sub-agent.
#[derive(Debug, Clone)]
pub struct SubAgentResult {
    /// Name of the agent.
    pub name: String,
    /// The agent's final answer.
    pub answer: String,
    /// Number of turns the agent used.
    pub turns_used: usize,
    /// How long the agent took.
    pub duration: Duration,
    /// Whether the agent completed successfully.
    pub success: bool,
    /// Error message if the agent failed.
    pub error: Option<String>,
}

impl SubAgentResult {
    /// Create a successful result.
    pub fn success(name: String, answer: String, turns: usize, duration: Duration) -> Self {
        Self {
            name,
            answer,
            turns_used: turns,
            duration,
            success: true,
            error: None,
        }
    }

    /// Create a failed result.
    pub fn failure(name: String, error: String, duration: Duration) -> Self {
        Self {
            name,
            answer: String::new(),
            turns_used: 0,
            duration,
            success: false,
            error: Some(error),
        }
    }
}

/// Orchestration strategy for running multiple agents.
#[derive(Debug, Clone)]
pub enum OrchestratorStrategy {
    /// Run agents one after another in order.
    /// Each agent sees the results of previous agents.
    Sequential,
    /// Run all agents in parallel.
    /// Agents don't see each other's results.
    Parallel,
    /// Fan-out/fan-in: run agents in parallel, then combine results.
    /// The combiner function gets all results and produces a single answer.
    FanOutFanIn {
        /// Prompt for combining sub-agent results into a final answer.
        combiner_prompt: String,
    },
}

/// Multi-agent orchestrator.
///
/// Manages a set of sub-agent specifications and runs them according
/// to a chosen strategy.
#[derive(Debug, Clone)]
pub struct Orchestrator {
    /// Sub-agents to orchestrate.
    specs: Vec<SubAgentSpec>,
    /// Execution strategy.
    strategy: OrchestratorStrategy,
}

impl Orchestrator {
    /// Create a new orchestrator with sequential strategy.
    pub fn new() -> Self {
        Self {
            specs: Vec::new(),
            strategy: OrchestratorStrategy::Sequential,
        }
    }

    /// Set the orchestration strategy.
    pub fn with_strategy(mut self, strategy: OrchestratorStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Add a sub-agent.
    pub fn add_agent(mut self, spec: SubAgentSpec) -> Self {
        self.specs.push(spec);
        self
    }

    /// Get the sub-agent specs.
    pub fn specs(&self) -> &[SubAgentSpec] {
        &self.specs
    }

    /// Get the strategy.
    pub fn strategy(&self) -> &OrchestratorStrategy {
        &self.strategy
    }

    /// Number of sub-agents.
    pub fn agent_count(&self) -> usize {
        self.specs.len()
    }

    /// Format all results as a combined summary.
    pub fn format_results(results: &[SubAgentResult]) -> String {
        let mut output = String::new();
        output.push_str("## Orchestration Results\n\n");

        for result in results {
            let status = if result.success { "✅" } else { "❌" };
            output.push_str(&format!(
                "### {} {} ({:.1}s, {} turns)\n",
                status,
                result.name,
                result.duration.as_secs_f64(),
                result.turns_used
            ));

            if result.success {
                output.push_str(&result.answer);
            } else if let Some(ref err) = result.error {
                output.push_str(&format!("**Error**: {err}"));
            }
            output.push_str("\n\n");
        }

        output
    }

    /// Build a combiner prompt that includes all sub-agent results.
    pub fn build_combiner_prompt(base_prompt: &str, results: &[SubAgentResult]) -> String {
        let mut prompt = String::from(base_prompt);
        prompt.push_str("\n\n## Sub-Agent Results\n\n");

        for result in results {
            if result.success {
                prompt.push_str(&format!(
                    "### {} (success)\n{}\n\n",
                    result.name, result.answer
                ));
            } else {
                prompt.push_str(&format!(
                    "### {} (failed: {})\n\n",
                    result.name,
                    result.error.as_deref().unwrap_or("unknown error")
                ));
            }
        }

        prompt
    }
}

impl Default for Orchestrator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sub_agent_spec_builder() {
        let spec = SubAgentSpec::new("researcher", "find relevant papers")
            .with_timeout(Duration::from_secs(60))
            .with_max_turns(5);

        assert_eq!(spec.name, "researcher");
        assert_eq!(spec.max_turns, 5);
        assert_eq!(spec.timeout, Duration::from_secs(60));
    }

    #[test]
    fn orchestrator_builder() {
        let orch = Orchestrator::new()
            .with_strategy(OrchestratorStrategy::Parallel)
            .add_agent(SubAgentSpec::new("a1", "task1"))
            .add_agent(SubAgentSpec::new("a2", "task2"));

        assert_eq!(orch.agent_count(), 2);
        assert!(matches!(orch.strategy(), OrchestratorStrategy::Parallel));
    }

    #[test]
    fn format_results_shows_success_and_failure() {
        let results = vec![
            SubAgentResult::success(
                "agent1".to_string(),
                "answer 1".to_string(),
                3,
                Duration::from_secs(5),
            ),
            SubAgentResult::failure(
                "agent2".to_string(),
                "timeout".to_string(),
                Duration::from_secs(60),
            ),
        ];

        let output = Orchestrator::format_results(&results);
        assert!(output.contains("✅"));
        assert!(output.contains("❌"));
        assert!(output.contains("answer 1"));
        assert!(output.contains("timeout"));
    }

    #[test]
    fn combiner_prompt_includes_results() {
        let results = vec![SubAgentResult::success(
            "researcher".to_string(),
            "found 5 papers".to_string(),
            2,
            Duration::from_secs(10),
        )];

        let prompt = Orchestrator::build_combiner_prompt("Combine the results below:", &results);
        assert!(prompt.contains("found 5 papers"));
        assert!(prompt.contains("researcher"));
    }

    #[test]
    fn sub_agent_result_success_and_failure() {
        let ok = SubAgentResult::success(
            "a".to_string(),
            "done".to_string(),
            1,
            Duration::from_secs(1),
        );
        assert!(ok.success);
        assert!(ok.error.is_none());

        let fail = SubAgentResult::failure(
            "b".to_string(),
            "crashed".to_string(),
            Duration::from_secs(1),
        );
        assert!(!fail.success);
        assert_eq!(fail.error.as_deref(), Some("crashed"));
    }
}
