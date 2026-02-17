//! Phase 13: Agent Signature — typed I/O contract for the agent.
//!
//! Manually implements `MetaSignature` for the code-generation agent.
//! This is equivalent to what `#[Signature]` generates but gives us
//! control over visibility and avoids the `schemars` dependency.

use dspy_rs::*;
use serde_json::json;

/// Code-generation agent signature.
///
/// Optimizers (GEPA/COPRO) mutate the instruction during optimization.
/// The `instruction()` method returns the current (possibly optimized) prompt.
#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CodeGenAgent {
    pub instruction: String,
    pub input_fields: serde_json::Value,
    pub output_fields: serde_json::Value,
    pub demos: Vec<Example>,
}

impl CodeGenAgent {
    pub fn new() -> Self {
        Self {
            instruction: "You are an expert Python programmer. Given a task description, \
                write correct Python code that solves the task and return the final answer."
                .to_string(),
            input_fields: json!({
                "task": {
                    "type": "String",
                    "desc": "The task to solve",
                    "schema": "",
                    "__dsrs_field_type": "input"
                },
                "context": {
                    "type": "Option<String>",
                    "desc": "Optional context for the task",
                    "schema": "",
                    "__dsrs_field_type": "input"
                }
            }),
            output_fields: json!({
                "code": {
                    "type": "String",
                    "desc": "Python code that solves the task",
                    "schema": "",
                    "__dsrs_field_type": "output"
                },
                "answer": {
                    "type": "String",
                    "desc": "The final answer",
                    "schema": "",
                    "__dsrs_field_type": "output"
                }
            }),
            demos: vec![],
        }
    }
}

impl MetaSignature for CodeGenAgent {
    fn demos(&self) -> Vec<Example> {
        self.demos.clone()
    }

    fn set_demos(&mut self, demos: Vec<Example>) -> anyhow::Result<()> {
        self.demos = demos;
        Ok(())
    }

    fn instruction(&self) -> String {
        self.instruction.clone()
    }

    fn input_fields(&self) -> serde_json::Value {
        self.input_fields.clone()
    }

    fn output_fields(&self) -> serde_json::Value {
        self.output_fields.clone()
    }

    fn update_instruction(&mut self, instruction: String) -> anyhow::Result<()> {
        self.instruction = instruction;
        Ok(())
    }

    fn append(&mut self, name: &str, field_value: serde_json::Value) -> anyhow::Result<()> {
        match field_value["__dsrs_field_type"].as_str() {
            Some("input") => {
                self.input_fields[name] = field_value;
            }
            Some("output") => {
                self.output_fields[name] = field_value;
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Invalid field type: {:?}",
                    field_value["__dsrs_field_type"].as_str()
                ));
            }
        }
        Ok(())
    }
}
