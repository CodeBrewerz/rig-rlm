//! Phase 14 + 16: Agent-as-Module — wrapping the monadic loop as a DSRs Module.
//!
//! The `AgentModule` wraps our monadic agent loop as a DSRs `Module`,
//! making it callable by optimizers (GEPA, COPRO, MIPROv2).
//!
//! dspy-rs 0.7.3 API:
//!   - Module::forward(&self, Example) -> Result<Prediction>
//!   - Optimizable: get_signature, parameters, update_signature_instruction
//!   - FeedbackEvaluator: feedback_metric(&self, &Example, &Prediction) -> FeedbackMetric

use anyhow::Result;
use dspy_rs::*;
use indexmap::IndexMap;

use crate::monad::interaction::agent_task_with_instruction;
use crate::monad::provider::ProviderConfig;
use crate::monad::{AgentConfig, AgentContext, Role};
use crate::safety::ExecutionLimits;
use crate::sandbox::ExecutorKind;
use crate::signature::CodeGenAgent;

/// Wraps the monadic agent loop as a DSRs Module.
///
/// The inner `Predict` stores the signature with the optimizable instruction.
/// GEPA/COPRO discover it via `Optimizable::parameters()`.
///
/// This is a "hollow Predict" pattern: the Predict stores the optimizable
/// parameters, but the module's forward() delegates actual execution to
/// the monadic loop (with its REPL, code execution, and feedback cycle).
pub struct AgentModule {
    /// The optimizable instruction leaf.
    pub predictor: Predict,
    /// LLM provider config.
    pub provider_config: ProviderConfig,
    /// Which executor to use.
    pub executor_kind: ExecutorKind,
}

impl AgentModule {
    /// Create with a specific instruction and provider.
    pub fn new(instruction: &str, provider_config: ProviderConfig) -> Self {
        let mut sig = CodeGenAgent::new();
        let _ = sig.update_instruction(instruction.to_string());
        Self {
            predictor: Predict::new(sig),
            provider_config,
            executor_kind: ExecutorKind::Pyo3,
        }
    }

    /// Create with default signature instruction and a local provider.
    pub fn default_local() -> Self {
        Self {
            predictor: Predict::new(CodeGenAgent::new()),
            provider_config: ProviderConfig::local("qwen/qwen3-8b"),
            executor_kind: ExecutorKind::Pyo3,
        }
    }

    /// Set the executor kind.
    pub fn with_executor(mut self, kind: ExecutorKind) -> Self {
        self.executor_kind = kind;
        self
    }
}

impl Module for AgentModule {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        // 1. Get instruction from the signature (GEPA-optimized)
        let instruction = self.predictor.signature.instruction();

        // 2. Extract task and context from the Example
        let task = inputs.get("task", None).as_str().unwrap_or("").to_string();
        let context = inputs.get("context", None).as_str().map(|s| s.to_string());

        // 3. Build the task description (context + task, NOT instruction)
        //    The instruction flows through agent_task_with_instruction()
        //    into the system prompt's {{ instruction }} variable.
        let task_prompt = if let Some(ctx) = &context {
            format!("Context:\n{ctx}\n\nTask: {task}")
        } else {
            task
        };

        // 4. Build agent config with the right provider
        let mut safety = ExecutionLimits::permissive();
        safety.max_code_length = 500_000; // 500KB — allow verbose LLM output during optimization
        let agent_config = AgentConfig {
            provider: self.provider_config.clone(),
            executor_kind: self.executor_kind.clone(),
            max_turns: 12, // Keep evaluations fast during optimization
            safety,
            ..AgentConfig::default()
        };

        // 5. Run the monadic agent with the optimized instruction
        let mut ctx = AgentContext::new(agent_config);
        let program = agent_task_with_instruction(&task_prompt, Some(&instruction));

        // Gracefully handle agent failures — during GEPA optimization,
        // individual evaluations may fail (max turns, execution errors).
        // Return an empty prediction so GEPA scores this as 0 and continues.
        let result = match ctx.run(program).await {
            Ok(answer) => answer,
            Err(e) => {
                tracing::warn!("Agent evaluation failed (will score 0): {e}");
                let prediction = dspy_rs::prediction! {
                    "code" => String::new(),
                    "answer" => String::new()
                };
                return Ok(prediction);
            }
        };

        // 6. Extract code and answer from the result
        let (code, answer) = extract_code_and_answer(&result, &ctx);

        // 7. Return as a Prediction (HashMap<String, serde_json::Value>)
        let prediction = dspy_rs::prediction! {
            "code" => code,
            "answer" => answer
        };

        Ok(prediction)
    }
}

impl Optimizable for AgentModule {
    fn get_signature(&self) -> &dyn MetaSignature {
        self.predictor.get_signature()
    }

    fn parameters(&mut self) -> IndexMap<String, &mut dyn Optimizable> {
        let mut map = IndexMap::new();
        map.insert(
            "predictor".to_string(),
            &mut self.predictor as &mut dyn Optimizable,
        );
        map
    }

    fn update_signature_instruction(&mut self, instruction: String) -> Result<()> {
        self.predictor.update_signature_instruction(instruction)
    }
}

/// Extract code and answer from the monadic result and context.
///
/// The answer is the final result string from the monad.
/// The code is the last code block from the assistant's messages.
fn extract_code_and_answer(result: &str, ctx: &AgentContext) -> (String, String) {
    let answer = result.to_string();

    // The code is the last executed code block from history
    let code = ctx
        .history
        .messages()
        .iter()
        .rev()
        .find(|m| m.role == Role::Assistant)
        .map(|m| {
            if let Some(start) = m.content.find("```") {
                let after_fence = &m.content[start + 3..];
                let lang_end = after_fence.find('\n').unwrap_or(0);
                let code_start = start + 3 + lang_end + 1;
                if let Some(end) = m.content[code_start..].find("```") {
                    return m.content[code_start..code_start + end].trim().to_string();
                }
            }
            m.content.to_string()
        })
        .unwrap_or_default();

    (code, answer)
}
