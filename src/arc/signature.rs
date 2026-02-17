//! Phase 19: ARC solver signature — MetaSignature for ARC tasks.
//!
//! Same pattern as `CodeGenAgent` in signature.rs but for ARC grid tasks.
//! Manual MetaSignature impl (avoids `#[Signature]` macro issues).

use dspy_rs::*;
use serde_json::json;

/// ARC solver signature.
///
/// Inputs: training examples + test challenges (as JSON strings).
/// Outputs: Python transform code + predicted output grids (as JSON).
///
/// The instruction is mutated by GEPA/COPRO during optimization.
#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ArcSolver {
    pub instruction: String,
    pub input_fields: serde_json::Value,
    pub output_fields: serde_json::Value,
    pub demos: Vec<Example>,
}

impl ArcSolver {
    pub fn new() -> Self {
        Self {
            instruction: crate::arc::prompt::ARC_INITIAL_PROMPT.to_string(),
            input_fields: json!({
                "examples": {
                    "type": "String",
                    "desc": "JSON array of training input/output grid pairs",
                    "schema": "",
                    "__dsrs_field_type": "input"
                },
                "challenges": {
                    "type": "String",
                    "desc": "JSON array of test input grids to solve",
                    "schema": "",
                    "__dsrs_field_type": "input"
                }
            }),
            output_fields: json!({
                "code": {
                    "type": "String",
                    "desc": "Python transform function that maps input grids to output grids",
                    "schema": "",
                    "__dsrs_field_type": "output"
                },
                "outputs": {
                    "type": "String",
                    "desc": "JSON array of predicted output grids",
                    "schema": "",
                    "__dsrs_field_type": "output"
                }
            }),
            demos: vec![],
        }
    }
}

impl MetaSignature for ArcSolver {
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
