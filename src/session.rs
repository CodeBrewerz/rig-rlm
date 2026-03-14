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

    /// Generate Python helper functions for context operations (Phase 3).
    ///
    /// These functions are available in the sandbox and print instructions
    /// that guide the agent's next action. They don't directly call the
    /// host — instead they produce structured output that the interaction
    /// loop can interpret.
    pub fn generate_context_tools_code() -> String {
        r#"
def search_context(context_id: str, pattern: str) -> str:
    """Search within a named context for a pattern.

    Use this when a large result was auto-loaded into a context.
    Returns matching lines with line numbers.

    Args:
        context_id: The context name (e.g. 'auto_exec_3')
        pattern: Text pattern to search for
    """
    print(f"[CONTEXT_SEARCH] id={context_id} pattern={pattern}")
    return f"Searching context '{context_id}' for '{pattern}'..."

def peek_context(context_id: str, start: int = 1, end: int = 50) -> str:
    """View a range of lines from a named context.

    Use this to examine specific sections of auto-loaded data.

    Args:
        context_id: The context name (e.g. 'auto_exec_3')
        start: First line to view (1-indexed)
        end: Last line to view (inclusive)
    """
    print(f"[CONTEXT_PEEK] id={context_id} start={start} end={end}")
    return f"Peeking context '{context_id}' lines {start}-{end}..."

def list_contexts() -> str:
    """List all loaded data contexts with metadata.

    Shows context names, sizes, line counts, and detected formats.
    """
    print("[CONTEXT_LIST]")
    return "Listing loaded contexts..."
"#
        .to_string()
    }

    /// Generate Python helper for recipe pipeline execution (Phase 8).
    ///
    /// Allows the agent to define and submit a multi-step pipeline
    /// from within the sandbox.
    pub fn generate_recipe_tools_code() -> String {
        r#"
def run_pipeline(recipe_yaml: str) -> str:
    """Define and execute a multi-step pipeline.

    Write a YAML recipe with named steps, dependencies, and task prompts.
    Each step runs as an independent agent task. Step outputs chain into
    downstream steps via {{step_id.output}} templates.

    Example:
        run_pipeline('''
        name: Analysis Pipeline
        steps:
          - id: load
            task: "Load and explore the data"
            kind: code_gen
          - id: model
            task: "Build model from: {{load.output}}"
            depends_on: [load]
          - id: report
            task: "Write report: {{model.output}}"
            kind: text_gen
            depends_on: [model]
        ''')

    Args:
        recipe_yaml: YAML string defining the pipeline steps
    """
    import json
    print("__PLAN_RECIPE__" + recipe_yaml + "__PLAN_RECIPE__")
    return "Pipeline submitted for execution..."
"#
        .to_string()
    }

    /// Generate the Python ELICIT() function for HITL pause/resume.
    ///
    /// ELICIT() is like SUBMIT() but non-terminal: it signals the agent
    /// loop to pause, present a question to the user, and resume with
    /// the user's response.
    pub fn generate_elicit_code() -> String {
        r#"
def ELICIT(question: str, partial_result: str = "") -> str:
    """Ask the user a question and wait for their response.

    Call this when you need clarification, confirmation, or additional
    input from the user before continuing. Execution pauses until the
    user responds.

    Args:
        question: The question to ask the user.
        partial_result: Optional partial work to show the user.

    Returns:
        The user's response text.
    """
    import json
    payload = {"question": question}
    if partial_result:
        payload["partial_result"] = partial_result
    print("__ELICIT__" + json.dumps(payload, default=str) + "__ELICIT__")
    return "[elicit] " + question
"#
        .to_string()
    }

    /// Generate the **blocking** ELICIT() function for microsandbox.
    ///
    /// This variant blocks on a sentinel file — the container can be
    /// stopped (zero resource use) while waiting. On resume, the host
    /// writes the user's response to the sentinel file.
    pub fn generate_elicit_code_blocking() -> String {
        r#"
def ELICIT(question: str, partial_result: str = "") -> str:
    """Ask the user a question and wait for their response.

    Execution blocks until the user responds. The sandbox may be
    suspended during the wait to conserve resources.

    Args:
        question: The question to ask the user.
        partial_result: Optional partial work to show the user.

    Returns:
        The user's response text.
    """
    import json, time, os, sys

    payload = {"question": question}
    if partial_result:
        payload["partial_result"] = partial_result

    # Clean up any stale sentinel from previous runs
    _sentinel = "/tmp/.elicit_response"
    if os.path.exists(_sentinel):
        os.remove(_sentinel)

    # Signal the host — markers are detected in stdout
    print("__ELICIT__" + json.dumps(payload, default=str) + "__ELICIT__", flush=True)
    sys.stdout.flush()

    # Block: poll for sentinel file written by host on resume
    while not os.path.exists(_sentinel):
        time.sleep(0.1)

    with open(_sentinel) as _f:
        _response = _f.read().strip()
    os.remove(_sentinel)

    print(f"[elicit-resumed] User responded: {_response}")
    return _response
"#
        .to_string()
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

// ── ELICIT result extraction ─────────────────────────────────────

/// Marker used to delimit ELICIT output in stdout.
pub const ELICIT_MARKER: &str = "__ELICIT__";

/// Try to extract an ELICIT request from execution stdout.
///
/// Returns `Some((question, partial_result))` if the output contains
/// `__ELICIT__{json}__ELICIT__`, otherwise `None`.
pub fn extract_elicit_request(stdout: &str) -> Option<(String, Option<String>)> {
    let start = stdout.find(ELICIT_MARKER)?;
    let json_start = start + ELICIT_MARKER.len();
    let rest = &stdout[json_start..];
    let end = rest.find(ELICIT_MARKER)?;
    let json_str = &rest[..end];
    let val: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let question = val["question"].as_str()?.to_string();
    let partial = val["partial_result"].as_str().map(|s| s.to_string());
    Some((question, partial))
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
