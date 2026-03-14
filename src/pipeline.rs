//! Phase 17: ProgramGraph — Multi-agent pipeline composition.
//!
//! Provides composable pipeline patterns for chaining multiple agents or
//! modules. When DSRs V6 ProgramGraph lands, these will be replaced with
//! the full graph-based composition. For now, we have:
//!
//! - `Pipeline<A, B>`: Two-stage A→B pipeline with a connector function
//! - `DelegateToSubAgents`: Fan-out to multiple sub-agents with code chunks
//! - `SubAgentConfig`: Configuration for sub-agent spawning

use crate::chunking::ast::CodeChunk;
use crate::monad::interaction::agent_task;
use crate::monad::provider::ProviderConfig;
use crate::monad::{AgentConfig, AgentContext};
use crate::sandbox::ExecutorKind;
use dspy_rs::*;

// ─── Sub-Agent Spawning ──────────────────────────────────────

/// Configuration for a sub-agent.
///
/// Sub-agents are lighter-weight agents that receive a specific chunk
/// of work. The root LLM stays pristine — only sub-agents get tools
/// and code execution capabilities.
#[derive(Debug, Clone)]
pub struct SubAgentConfig {
    /// LLM provider for the sub-agent (can be a smaller/cheaper model).
    pub provider: ProviderConfig,
    /// System instruction for the sub-agent.
    pub instruction: String,
    /// Maximum turns before the sub-agent gives up.
    pub max_turns: usize,
    /// Whether the sub-agent gets code execution.
    pub enable_code_execution: bool,
    /// Whether the sub-agent gets sub-LLM bridging.
    pub enable_sub_llm: bool,
}

impl Default for SubAgentConfig {
    fn default() -> Self {
        Self {
            provider: ProviderConfig::local("qwen/qwen3-8b"),
            instruction: "You are a helpful code analysis assistant.".to_string(),
            max_turns: 10,
            enable_code_execution: true,
            enable_sub_llm: false,
        }
    }
}

/// Result from a sub-agent execution.
#[derive(Debug, Clone)]
pub struct SubAgentResult {
    /// The sub-agent's final output.
    pub output: String,
    /// The code chunk this sub-agent was working on.
    pub chunk_name: Option<String>,
    /// Whether the sub-agent completed successfully.
    pub success: bool,
    /// Number of turns used.
    pub turns_used: usize,
}

/// Delegate code chunks to sub-agents for parallel analysis.
///
/// Each sub-agent gets:
/// - One code chunk to analyze
/// - Its own `AgentContext` (isolated history, executor)
/// - The sub-agent config (model, instruction, max_turns)
///
/// The root LLM never sees the raw code — it only sees the
/// aggregated results from sub-agents.
pub async fn delegate_to_sub_agents(
    chunks: Vec<CodeChunk>,
    config: &SubAgentConfig,
) -> Vec<SubAgentResult> {
    let mut results = Vec::new();

    // Run sub-agents sequentially for now (concurrent spawning
    // requires careful task isolation — future improvement)
    for chunk in &chunks {
        let result = spawn_sub_agent(chunk, config).await;
        results.push(result);
    }

    results
}

/// Spawn a single sub-agent to analyze a code chunk.
async fn spawn_sub_agent(chunk: &CodeChunk, config: &SubAgentConfig) -> SubAgentResult {
    let mut agent_config = AgentConfig {
        max_turns: config.max_turns,
        provider: config.provider.clone(),
        executor_kind: ExecutorKind::Pyo3,
        ..AgentConfig::default()
    };

    if config.enable_sub_llm {
        agent_config = agent_config.with_sub_llm();
    }

    let task = format!(
        "{}\n\nAnalyze this code:\n```python\n{}\n```\n\n{}",
        config.instruction,
        chunk.content,
        if let Some(ref name) = chunk.name {
            format!("This is the {} '{}'.", chunk.kind_str(), name)
        } else {
            format!("This is a {} block.", chunk.kind_str())
        }
    );

    let mut ctx = AgentContext::new(agent_config);
    let program = agent_task(&task);

    match ctx.run(program).await {
        Ok(run_result) => SubAgentResult {
            output: run_result.into_completed(),
            chunk_name: chunk.name.clone(),
            success: true,
            turns_used: ctx.turn_count(),
        },
        Err(e) => SubAgentResult {
            output: format!("Sub-agent failed: {e}"),
            chunk_name: chunk.name.clone(),
            success: false,
            turns_used: ctx.turn_count(),
        },
    }
}

/// Aggregate sub-agent results into a summary for the root LLM.
pub fn aggregate_results(results: &[SubAgentResult]) -> String {
    let mut summary = String::new();

    let successful = results.iter().filter(|r| r.success).count();
    summary.push_str(&format!(
        "Sub-agent analysis complete: {}/{} chunks analyzed successfully.\n\n",
        successful,
        results.len()
    ));

    for (i, result) in results.iter().enumerate() {
        let label = result.chunk_name.as_deref().unwrap_or("unnamed");
        if result.success {
            summary.push_str(&format!(
                "--- Chunk {} ({}) ---\n{}\n\n",
                i + 1,
                label,
                result.output
            ));
        } else {
            summary.push_str(&format!(
                "--- Chunk {} ({}) --- FAILED\n{}\n\n",
                i + 1,
                label,
                result.output
            ));
        }
    }

    summary
}

// ─── Pipeline Composition ────────────────────────────────────

/// A two-stage pipeline: module A produces output, connector transforms
/// it into input for module B, module B produces final output.
///
/// In dspy-rs 0.7.3, Module::forward takes Example and returns Result<Prediction>.
/// The connector transforms A's Prediction into B's Example input.
///
/// Both stages' `Predict` leaves are discoverable by optimizers via
/// the Optimizable trait — the optimizer can mutate both instructions jointly.
pub struct ModulePipeline<A, B>
where
    A: Module + Send + Sync,
    B: Module + Send + Sync,
{
    pub stage_a: A,
    pub stage_b: B,
    pub connector: Box<dyn Fn(Prediction) -> Example + Send + Sync>,
}

impl<A, B> ModulePipeline<A, B>
where
    A: Module + Send + Sync,
    B: Module + Send + Sync,
{
    /// Create a new pipeline from two modules and a connector function.
    pub fn new(
        stage_a: A,
        stage_b: B,
        connector: impl Fn(Prediction) -> Example + Send + Sync + 'static,
    ) -> Self {
        Self {
            stage_a,
            stage_b,
            connector: Box::new(connector),
        }
    }

    /// Run the pipeline: A → connector → B.
    pub async fn run(&self, input: Example) -> anyhow::Result<Prediction> {
        let a_output = self.stage_a.forward(input).await?;
        let b_input = (self.connector)(a_output);
        self.stage_b.forward(b_input).await
    }
}

// ─── Helper trait extension for CodeChunk ────────────────────

impl CodeChunk {
    /// Human-readable kind string.
    pub fn kind_str(&self) -> &'static str {
        match self.kind {
            crate::chunking::ast::ChunkKind::Function => "function",
            crate::chunking::ast::ChunkKind::Class => "class",
            crate::chunking::ast::ChunkKind::Import => "import",
            crate::chunking::ast::ChunkKind::TopLevel => "top-level",
            crate::chunking::ast::ChunkKind::Module => "module",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sub_agent_config_default() {
        let config = SubAgentConfig::default();
        assert_eq!(config.max_turns, 10);
        assert!(config.enable_code_execution);
        assert!(!config.enable_sub_llm);
    }

    #[test]
    fn test_aggregate_results_empty() {
        let results = vec![];
        let summary = aggregate_results(&results);
        assert!(summary.contains("0/0"));
    }

    #[test]
    fn test_aggregate_results_mixed() {
        let results = vec![
            SubAgentResult {
                output: "Analysis of transform function".to_string(),
                chunk_name: Some("transform".to_string()),
                success: true,
                turns_used: 3,
            },
            SubAgentResult {
                output: "Sub-agent failed: timeout".to_string(),
                chunk_name: Some("GridSolver".to_string()),
                success: false,
                turns_used: 10,
            },
        ];
        let summary = aggregate_results(&results);
        assert!(summary.contains("1/2 chunks"));
        assert!(summary.contains("transform"));
        assert!(summary.contains("FAILED"));
    }
}
