//! Phase 23B + 26: Session configuration for persistent REPL, typed SUBMIT,
//! and sub-LLM bridging.
//!
//! Adopted from Daytona PR #3565 analysis. Instead of Daytona's Flask broker
//! pattern (accidental complexity from cloud architecture), we use direct
//! Rust callbacks for PyO3 and stdin/stdout JSON-RPC for microsandbox.

use crate::monad::provider::ProviderConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Describes a single output field from a DSRs Signature.
/// Used to generate the typed SUBMIT() function signature in the sandbox.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputField {
    /// Field name, e.g. "answer", "code"
    pub name: String,
    /// Python type hint, e.g. "str", "int", "list"
    pub type_hint: String,
    /// Optional description (becomes docstring hint)
    pub description: Option<String>,
}

/// Configuration for a code execution session.
///
/// This is injected at executor setup time and controls:
/// - What output fields the SUBMIT() function accepts
/// - Whether sub-LLM bridging is enabled
/// - What custom tools are available in the sandbox
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Output fields — determines SUBMIT() function signature.
    /// If empty, SUBMIT takes a single positional `answer: str` arg.
    pub output_fields: Vec<OutputField>,

    /// Whether to inject llm_query() and llm_query_batched() bridges.
    pub enable_sub_llm: bool,

    /// LLM provider config for sub-LLM bridging.
    /// Required when `enable_sub_llm` is true.
    pub provider_config: Option<ProviderConfig>,

    /// Custom tool definitions to inject into the sandbox.
    /// Key = function name available in Python, Value = tool spec.
    pub tools: HashMap<String, ToolSpec>,

    /// Prelude code to run before any user code (imports, helpers).
    pub prelude: Option<String>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            output_fields: vec![OutputField {
                name: "answer".into(),
                type_hint: "str".into(),
                description: Some("The final answer to the task".into()),
            }],
            enable_sub_llm: false,
            provider_config: None,
            tools: HashMap::new(),
            prelude: None,
        }
    }
}

impl SessionConfig {
    /// Enable sub-LLM bridging with a specific provider.
    pub fn with_sub_llm(mut self) -> Self {
        self.enable_sub_llm = true;
        self
    }

    /// Set the LLM provider config for sub-LLM bridging.
    pub fn with_provider(mut self, config: ProviderConfig) -> Self {
        self.provider_config = Some(config);
        self.enable_sub_llm = true;
        self
    }

    /// Add an output field to the SUBMIT signature.
    pub fn with_output_field(
        mut self,
        name: impl Into<String>,
        type_hint: impl Into<String>,
    ) -> Self {
        self.output_fields.push(OutputField {
            name: name.into(),
            type_hint: type_hint.into(),
            description: None,
        });
        self
    }

    /// Set prelude code.
    pub fn with_prelude(mut self, prelude: impl Into<String>) -> Self {
        self.prelude = Some(prelude.into());
        self
    }

    /// Generate the Python SUBMIT function code to inject.
    ///
    /// Produces something like:
    /// ```python
    /// class FinalOutput(BaseException):
    ///     pass
    ///
    /// def SUBMIT(answer: str):
    ///     """Submit your final answer. Call this when you're done."""
    ///     import json, sys
    ///     result = {"answer": answer}
    ///     print("__SUBMIT__" + json.dumps(result) + "__SUBMIT__")
    ///     raise FinalOutput()
    /// ```
    pub fn generate_submit_code(&self) -> String {
        let fields = if self.output_fields.is_empty() {
            vec![OutputField {
                name: "answer".into(),
                type_hint: "str".into(),
                description: None,
            }]
        } else {
            self.output_fields.clone()
        };

        // Build function signature
        let params: Vec<String> = fields
            .iter()
            .map(|f| format!("{}: {}", f.name, f.type_hint))
            .collect();

        // Build result dict
        let dict_entries: Vec<String> = fields
            .iter()
            .map(|f| format!("        \"{name}\": {name}", name = f.name))
            .collect();

        // Build docstring
        let field_docs: Vec<String> = fields
            .iter()
            .map(|f| {
                let desc = f.description.as_deref().unwrap_or("");
                format!("        {}: {} — {}", f.name, f.type_hint, desc)
            })
            .collect();

        format!(
            r#"
class FinalOutput(BaseException):
    """Raised by SUBMIT() to halt execution with a result."""
    pass

def SUBMIT({params}):
    """Submit your final answer. Call this when the task is complete.

    Args:
{field_docs}
    """
    import json, sys
    _result = {{
{dict_entries}
    }}
    print("__SUBMIT__" + json.dumps(_result, default=str) + "__SUBMIT__")
    raise FinalOutput()
"#,
            params = params.join(", "),
            field_docs = field_docs.join("\n"),
            dict_entries = dict_entries.join(",\n"),
        )
    }

    /// Generate the Python llm_query bridge code (for microsandbox — stdin/stdout JSON-RPC).
    ///
    /// For PyO3, llm_query is injected as a #[pyfunction] directly — no code generation needed.
    pub fn generate_llm_bridge_code(&self) -> String {
        if !self.enable_sub_llm {
            return String::new();
        }

        r#"
import json, sys

def llm_query(prompt: str) -> str:
    """Call the host LLM with a prompt and get a text response.

    Use this for semantic tasks: summarization, classification,
    extraction, or any reasoning that Python can't do.
    """
    request = json.dumps({"method": "llm_query", "params": {"prompt": prompt}, "id": id(_rpc_counter := getattr(sys, '_rpc_counter', 0) + 1)})
    sys.stderr.write("__RPC__" + request + "__RPC__\n")
    sys.stderr.flush()
    # Block until host responds via stdin
    response_line = sys.stdin.readline().strip()
    response = json.loads(response_line)
    if "error" in response:
        raise RuntimeError(f"LLM query failed: {response['error']}")
    return response["result"]

def llm_query_batched(prompts: list[str]) -> list[str]:
    """Call the host LLM with multiple prompts concurrently.

    More efficient than calling llm_query() in a loop — the host
    runs all prompts in parallel.
    """
    request = json.dumps({"method": "llm_query_batched", "params": {"prompts": prompts}, "id": id(prompts)})
    sys.stderr.write("__RPC__" + request + "__RPC__\n")
    sys.stderr.flush()
    response_line = sys.stdin.readline().strip()
    response = json.loads(response_line)
    if "error" in response:
        raise RuntimeError(f"Batched LLM query failed: {response['error']}")
    return response["result"]
"#.to_string()
    }
}

/// Specification for a host-side tool to inject into the sandbox.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Human-readable description (becomes Python docstring).
    pub description: String,
    /// Parameter specifications.
    pub parameters: Vec<ToolParam>,
    /// Return type hint.
    pub return_type: String,
}

/// Parameter specification for a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParam {
    pub name: String,
    pub type_hint: String,
    pub description: Option<String>,
}

// ── SUBMIT result extraction ─────────────────────────────────

/// Marker used to delimit SUBMIT output in stdout.
pub const SUBMIT_MARKER: &str = "__SUBMIT__";

/// Try to extract a SUBMIT result from execution stdout.
///
/// Returns `Some(json_value)` if the output contains `__SUBMIT__{json}__SUBMIT__`,
/// otherwise `None`.
pub fn extract_submit_result(stdout: &str) -> Option<serde_json::Value> {
    let start = stdout.find(SUBMIT_MARKER)?;
    let json_start = start + SUBMIT_MARKER.len();
    let rest = &stdout[json_start..];
    let end = rest.find(SUBMIT_MARKER)?;
    let json_str = &rest[..end];
    serde_json::from_str(json_str).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_submit_code_generation() {
        let config = SessionConfig::default();
        let code = config.generate_submit_code();
        assert!(code.contains("def SUBMIT(answer: str)"));
        assert!(code.contains("class FinalOutput(BaseException)"));
        assert!(code.contains("__SUBMIT__"));
    }

    #[test]
    fn test_multi_field_submit() {
        let config = SessionConfig {
            output_fields: vec![
                OutputField {
                    name: "code".into(),
                    type_hint: "str".into(),
                    description: Some("Python code".into()),
                },
                OutputField {
                    name: "answer".into(),
                    type_hint: "str".into(),
                    description: Some("Final answer".into()),
                },
            ],
            ..Default::default()
        };
        let code = config.generate_submit_code();
        assert!(code.contains("def SUBMIT(code: str, answer: str)"));
    }

    #[test]
    fn test_extract_submit_result() {
        let stdout = r#"some output
__SUBMIT__{"answer": "42"}__SUBMIT__"#;
        let result = extract_submit_result(stdout).unwrap();
        assert_eq!(result["answer"], "42");
    }

    #[test]
    fn test_extract_submit_no_marker() {
        assert!(extract_submit_result("just regular output").is_none());
    }

    #[test]
    fn test_llm_bridge_disabled() {
        let config = SessionConfig::default();
        assert!(config.generate_llm_bridge_code().is_empty());
    }

    #[test]
    fn test_llm_bridge_enabled() {
        let config = SessionConfig::default().with_sub_llm();
        let code = config.generate_llm_bridge_code();
        assert!(code.contains("def llm_query(prompt: str)"));
        assert!(code.contains("def llm_query_batched(prompts: list[str])"));
    }
}
