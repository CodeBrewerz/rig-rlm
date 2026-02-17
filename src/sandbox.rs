//! Phase 23 + 23B + 26: Code execution backend abstraction.
//!
//! Provides a `CodeExecutor` trait and implementations:
//! - `MicrosandboxExecutor` — hardware-isolated microVM via microsandbox
//! - `Pyo3CodeExecutor` — in-process Python via PyO3 (dev/test fallback)
//!
//! Phase 23B additions (from Daytona PR analysis):
//! - Persistent REPL state — variables survive across execute() calls
//! - Session setup — inject SUBMIT(), prelude, llm_query()
//! - Typed SUBMIT — structured output matching DSRs Signature fields
//!
//! Phase 26: Sub-LLM bridging — llm_query() as direct #[pyfunction] callback
//! Generated code can call back to the host LLM for semantic reasoning.

use async_trait::async_trait;
use crate::monad::ExecutionResult;
use crate::monad::provider::{ProviderConfig, LlmProvider};
use crate::session::{SessionConfig, extract_submit_result};

/// Abstraction over code execution backends.
///
/// Implementations must support:
/// - `execute()` — run Python code and return structured results
/// - `setup()` — inject session prelude (SUBMIT, llm_query, custom tools)
/// - Persistent state across execute() calls within a session
#[async_trait]
pub trait CodeExecutor: Send + Sync {
    /// Execute Python code and return structured results.
    ///
    /// State (variables, imports, function defs) MUST persist
    /// across calls within a session. This enables iterative
    /// problem-solving where the LLM builds on previous work.
    async fn execute(&mut self, code: &str) -> anyhow::Result<ExecutionResult>;

    /// Execute a shell command (convenience wrapper).
    async fn run_command(&mut self, cmd: &str) -> anyhow::Result<ExecutionResult> {
        let wrapper = format!(
            "import subprocess; _r = subprocess.run({cmd:?}, shell=True, capture_output=True, text=True)\nprint(_r.stdout)\nif _r.stderr: print(_r.stderr, end='')"
        );
        self.execute(&wrapper).await
    }

    /// Set up the session — inject SUBMIT(), prelude, llm_query(), etc.
    ///
    /// Called once before the first execute(). Implementations should:
    /// 1. Run the SUBMIT function definition
    /// 2. Run any prelude code
    /// 3. Inject llm_query bridge if enabled
    async fn setup(&mut self, config: &SessionConfig) -> anyhow::Result<()> {
        // Default: inject SUBMIT + prelude via execute()
        let submit_code = config.generate_submit_code();
        self.execute(&submit_code).await?;

        if let Some(ref prelude) = config.prelude {
            self.execute(prelude).await?;
        }

        // For microsandbox: inject llm_query as Python code
        // For PyO3: this is a no-op (injected as #[pyfunction] in setup())
        let bridge_code = config.generate_llm_bridge_code();
        if !bridge_code.is_empty() {
            self.execute(&bridge_code).await?;
        }

        Ok(())
    }

    /// Whether this executor supports persistent state across execute() calls.
    fn supports_persistent_state(&self) -> bool { false }

    /// Reset the session — clear all state.
    async fn reset(&mut self) -> anyhow::Result<()> { Ok(()) }
}

// ─── Microsandbox Executor ───────────────────────────────────

use microsandbox::{PythonSandbox, BaseSandbox};

/// Hardware-isolated code execution via microsandbox microVM.
///
/// Persistent state: microsandbox's `run_or_start()` keeps the
/// Python process alive — variables, imports, and definitions
/// persist across execute() calls automatically.
pub struct MicrosandboxExecutor {
    sandbox: PythonSandbox,
    session_ready: bool,
}

/// Configuration for the microsandbox executor.
pub struct SandboxConfig {
    pub name: String,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            name: "rig-rlm-sandbox".to_string(),
        }
    }
}

impl MicrosandboxExecutor {
    /// Create and start a new microsandbox executor.
    /// Requires microsandbox server running (`msandbox server start`).
    pub async fn new(config: &SandboxConfig) -> anyhow::Result<Self> {
        let mut sandbox = PythonSandbox::create(&config.name).await
            .map_err(|e| anyhow::anyhow!("microsandbox create: {e}"))?;
        sandbox.start(None).await
            .map_err(|e| anyhow::anyhow!("microsandbox start: {e}"))?;
        Ok(Self { sandbox, session_ready: false })
    }

    /// Stop the sandbox.
    pub async fn stop(&mut self) -> anyhow::Result<()> {
        self.sandbox.stop().await
            .map_err(|e| anyhow::anyhow!("microsandbox stop: {e}"))?;
        Ok(())
    }
}

#[async_trait]
impl CodeExecutor for MicrosandboxExecutor {
    async fn execute(&mut self, code: &str) -> anyhow::Result<ExecutionResult> {
        let execution = self.sandbox.run_or_start(code).await
            .map_err(|e| anyhow::anyhow!("microsandbox exec: {e}"))?;
        let stdout = execution.output().await.unwrap_or_default();
        let stderr = execution.error().await.unwrap_or_default();

        // Check for SUBMIT marker in stdout
        if let Some(submit_value) = extract_submit_result(&stdout) {
            let answer = submit_value
                .get("answer")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            return Ok(ExecutionResult::submitted(
                stdout.clone(),
                serde_json::to_string(&submit_value).unwrap_or_default(),
            ));
        }

        if execution.has_error() {
            // Check if error is FinalOutput (from SUBMIT) — not a real error
            if stderr.contains("FinalOutput") {
                // SUBMIT was called — extract from stdout
                if let Some(submit_value) = extract_submit_result(&stdout) {
                    return Ok(ExecutionResult::submitted(
                        stdout.clone(),
                        serde_json::to_string(&submit_value).unwrap_or_default(),
                    ));
                }
            }
            let msg = if stderr.is_empty() { &stdout } else { &stderr };
            Ok(ExecutionResult::error("ExecutionError", msg.trim()))
        } else if stdout.is_empty() {
            Ok(ExecutionResult::success("(no output)"))
        } else {
            Ok(ExecutionResult::success(&stdout))
        }
    }

    fn supports_persistent_state(&self) -> bool { true }

    async fn reset(&mut self) -> anyhow::Result<()> {
        self.sandbox.stop().await
            .map_err(|e| anyhow::anyhow!("microsandbox stop: {e}"))?;
        self.sandbox.start(None).await
            .map_err(|e| anyhow::anyhow!("microsandbox restart: {e}"))?;
        self.session_ready = false;
        Ok(())
    }
}

// ─── PyO3 Persistent REPL Executor ──────────────────────────

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::ffi::CString;
use std::sync::{Arc, Mutex};

// ─── Sub-LLM Bridge Infrastructure ──────────────────────────
//
// Phase 26: Generated Python code calls llm_query("prompt") which
// routes to the host LLM via a #[pyfunction] Rust callback.
//
// Architecture:
// - LlmBridge holds ProviderConfig + a tokio runtime handle
// - Stored in a global Arc once initialized
// - #[pyfunction] llm_query_bridge uses py.allow_threads() to
//   release GIL, then block_on the async LLM call
//
// No broker, no HTTP, no serialization overhead — just a Rust fn call.

/// Holds the LLM provider config for the sub-LLM bridge.
///
/// Initialized once when the executor is set up with a SessionConfig
/// that has enable_sub_llm=true and a provider_config.
struct LlmBridge {
    config: ProviderConfig,
    /// Handle to the async runtime for block_on calls from pyfunction.
    runtime: tokio::runtime::Handle,
}

/// Global bridge — set once, used by all #[pyfunction] llm_query calls.
static LLM_BRIDGE: std::sync::OnceLock<Arc<LlmBridge>> = std::sync::OnceLock::new();

/// Initialize the global LLM bridge. Called from Pyo3CodeExecutor::setup().
fn init_llm_bridge(config: ProviderConfig) {
    let bridge = Arc::new(LlmBridge {
        config,
        runtime: tokio::runtime::Handle::current(),
    });
    // Ignore error if already set (idempotent)
    let _ = LLM_BRIDGE.set(bridge);
}

/// Python-callable: llm_query(prompt) -> str
///
/// Releases the GIL via py.allow_threads(), blocks on the async LLM
/// call, returns the response text. This is the "recursive" in RLM —
/// generated code can call the LLM for semantic reasoning.
#[pyfunction]
fn llm_query_bridge(py: Python, prompt: String) -> PyResult<String> {
    let bridge = LLM_BRIDGE.get()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
            "LLM bridge not initialized. Ensure session has enable_sub_llm=true."
        ))?;
    let bridge = bridge.clone();

    py.detach(move || {
        let provider = LlmProvider::new(bridge.config.clone());
        bridge.runtime.block_on(async {
            provider.complete(&prompt).await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("LLM query failed: {e}")
                ))
        })
    })
}

/// Python-callable: llm_query_batched(prompts) -> list[str]
///
/// Runs multiple LLM calls concurrently for efficiency.
#[pyfunction]
fn llm_query_batched_bridge(py: Python, prompts: Vec<String>) -> PyResult<Vec<String>> {
    let bridge = LLM_BRIDGE.get()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
            "LLM bridge not initialized."
        ))?;
    let bridge = bridge.clone();

    py.detach(move || {
        bridge.runtime.block_on(async {
            let mut results = Vec::with_capacity(prompts.len());
            // Run sequentially to avoid spawning issues; still async I/O
            for prompt in &prompts {
                let provider = LlmProvider::new(bridge.config.clone());
                let result = provider.complete(prompt).await
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                        format!("Batched LLM query failed: {e}")
                    ))?;
                results.push(result);
            }
            Ok(results)
        })
    })
}

/// In-process Python execution via PyO3 with persistent REPL state.
///
/// Key design decisions (from Daytona PR analysis):
/// - **Persistent namespace**: A `PyDict` (`globals`) survives across
///   execute() calls. Variables, imports, and function definitions
///   persist. This is what makes iterative problem-solving work.
/// - **Direct callbacks**: `llm_query()` is injected as a `#[pyfunction]`
///   that calls Rust directly. No broker, no HTTP, no serialization.
///   This replaces Daytona's Flask broker pattern.
/// - **SUBMIT detection**: FinalOutput exception is caught specially.
///   The result is extracted from the shared `submit_result` Arc.
///
/// No isolation — use only for dev/test. Production uses MicrosandboxExecutor.
pub struct Pyo3CodeExecutor {
    /// Persistent Python globals dict — variables survive across calls.
    globals: Option<Py<PyDict>>,
    /// Shared state for SUBMIT results — set by the injected SUBMIT function.
    submit_result: Arc<Mutex<Option<String>>>,
    /// Whether session setup has been run.
    session_ready: bool,
}

impl Pyo3CodeExecutor {
    pub fn new() -> Self {
        Self {
            globals: None,
            submit_result: Arc::new(Mutex::new(None)),
            session_ready: false,
        }
    }

    /// Initialize the persistent Python namespace if needed.
    fn ensure_globals(&mut self) {
        if self.globals.is_none() {
            Python::attach(|py| {
                let globals = PyDict::new(py);
                // Import __builtins__ so built-in functions work
                let builtins = py.import("builtins").unwrap();
                globals.set_item("__builtins__", builtins).unwrap();
                self.globals = Some(globals.into());
            });
        }
    }
}

#[async_trait]
impl CodeExecutor for Pyo3CodeExecutor {
    async fn execute(&mut self, code: &str) -> anyhow::Result<ExecutionResult> {
        self.ensure_globals();

        // Clear any previous SUBMIT result
        *self.submit_result.lock().unwrap() = None;

        let code = code.to_string();
        // Py<T>::clone() requires GIL in modern PyO3 — use clone_ref(py)
        let globals_py = Python::attach(|py| {
            self.globals.as_ref().unwrap().clone_ref(py)
        });
        let submit_result = self.submit_result.clone();

        let result = tokio::task::spawn_blocking(move || {
            Python::attach(|py| {
                let globals = globals_py.bind(py);

                // Capture stdout & stderr
                let sys = py.import("sys").unwrap();
                let io = py.import("io").unwrap();
                let string_io = io.call_method0("StringIO").unwrap();
                sys.setattr("stdout", &string_io).unwrap();
                let err_io = io.call_method0("StringIO").unwrap();
                sys.setattr("stderr", &err_io).unwrap();

                let c_code = CString::new(code.as_str()).unwrap_or_else(|_|
                    CString::new("print('code contained null byte')").unwrap()
                );

                match py.run(&c_code, Some(&globals), None) {
                    Ok(()) => {
                        let stdout: String = string_io.call_method0("getvalue")
                            .and_then(|v| v.extract())
                            .unwrap_or_default();
                        let stderr: String = err_io.call_method0("getvalue")
                            .and_then(|v| v.extract())
                            .unwrap_or_default();

                        // Check for SUBMIT in stdout
                        if let Some(submit_value) = extract_submit_result(&stdout) {
                            return ExecutionResult::submitted(
                                stdout,
                                serde_json::to_string(&submit_value).unwrap_or_default(),
                            );
                        }

                        // Check shared SUBMIT result (set by #[pyfunction] SUBMIT)
                        if let Some(ref result_json) = *submit_result.lock().unwrap() {
                            return ExecutionResult::submitted(
                                stdout,
                                result_json.clone(),
                            );
                        }

                        if stderr.is_empty() {
                            ExecutionResult::success(
                                if stdout.is_empty() { "(no output)" } else { &stdout }
                            )
                        } else {
                            ExecutionResult {
                                stdout,
                                stderr,
                                return_value: None,
                                exception: None,
                                has_result: false,
                            }
                        }
                    }
                    Err(e) => {
                        // Check if this is a FinalOutput exception (from SUBMIT)
                        let err_str = e.to_string();
                        if err_str.contains("FinalOutput") {
                            let stdout: String = string_io.call_method0("getvalue")
                                .and_then(|v| v.extract())
                                .unwrap_or_default();

                            // SUBMIT prints to stdout before raising FinalOutput
                            if let Some(submit_value) = extract_submit_result(&stdout) {
                                return ExecutionResult::submitted(
                                    stdout,
                                    serde_json::to_string(&submit_value).unwrap_or_default(),
                                );
                            }

                            // Fallback: check shared state
                            if let Some(ref result_json) = *submit_result.lock().unwrap() {
                                return ExecutionResult::submitted(
                                    stdout,
                                    result_json.clone(),
                                );
                            }
                        }

                        ExecutionResult::error("PythonError", &err_str)
                    }
                }
            })
        }).await?;
        Ok(result)
    }

    async fn setup(&mut self, config: &SessionConfig) -> anyhow::Result<()> {
        self.ensure_globals();

        // 1. Inject SUBMIT function
        let submit_code = config.generate_submit_code();
        self.execute(&submit_code).await?;

        // 2. Inject prelude
        if let Some(ref prelude) = config.prelude {
            self.execute(prelude).await?;
        }

        // 3. Inject llm_query as #[pyfunction] if enabled (Phase 26)
        //    Direct Rust callback — no broker, no HTTP, no serialization.
        if config.enable_sub_llm {
            if let Some(ref provider_config) = config.provider_config {
                init_llm_bridge(provider_config.clone());

                // Inject the Python-callable functions into globals
                Python::attach(|py| -> PyResult<()> {
                    let globals = self.globals.as_ref().unwrap().bind(py);
                    let llm_fn = wrap_pyfunction!(llm_query_bridge, py)?;
                    globals.set_item("llm_query", llm_fn)?;
                    let batch_fn = wrap_pyfunction!(llm_query_batched_bridge, py)?;
                    globals.set_item("llm_query_batched", batch_fn)?;
                    Ok(())
                }).map_err(|e| anyhow::anyhow!("Failed to inject llm_query bridge: {e}"))?;

                tracing::info!("Sub-LLM bridge injected: llm_query() + llm_query_batched() available in sandbox");
            } else {
                tracing::warn!("enable_sub_llm is true but no provider_config set — llm_query will not be available");
            }
        }

        self.session_ready = true;
        Ok(())
    }

    fn supports_persistent_state(&self) -> bool { true }

    async fn reset(&mut self) -> anyhow::Result<()> {
        self.globals = None;
        *self.submit_result.lock().unwrap() = None;
        self.session_ready = false;
        Ok(())
    }
}

// ─── Executor Factory ────────────────────────────────────────

/// Which execution backend to use.
#[derive(Debug, Clone, Default)]
pub enum ExecutorKind {
    /// Microsandbox (hardware-isolated microVM) — primary.
    Microsandbox,
    /// PyO3 (in-process, persistent REPL) — dev/test fallback.
    #[default]
    Pyo3,
}

/// Create an executor based on kind.
pub async fn create_executor(kind: &ExecutorKind) -> anyhow::Result<Box<dyn CodeExecutor>> {
    match kind {
        ExecutorKind::Microsandbox => {
            let config = SandboxConfig::default();
            let executor = MicrosandboxExecutor::new(&config).await?;
            Ok(Box::new(executor))
        }
        ExecutorKind::Pyo3 => {
            Ok(Box::new(Pyo3CodeExecutor::new()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pyo3_persistent_state() {
        let mut executor = Pyo3CodeExecutor::new();

        // First call: define a variable
        let r1 = executor.execute("x = 42\nprint(x)").await.unwrap();
        assert!(r1.stdout.contains("42"));

        // Second call: variable persists
        let r2 = executor.execute("print(x + 8)").await.unwrap();
        assert!(r2.stdout.contains("50"));
    }

    #[tokio::test]
    async fn test_pyo3_persistent_imports() {
        let mut executor = Pyo3CodeExecutor::new();

        executor.execute("import json").await.unwrap();
        let r = executor.execute("print(json.dumps({'a': 1}))").await.unwrap();
        assert!(r.stdout.contains(r#"{"a": 1}"#));
    }

    #[tokio::test]
    async fn test_pyo3_persistent_functions() {
        let mut executor = Pyo3CodeExecutor::new();

        executor.execute("def greet(name):\n    return f'Hello, {name}!'").await.unwrap();
        let r = executor.execute("print(greet('World'))").await.unwrap();
        assert!(r.stdout.contains("Hello, World!"));
    }

    #[tokio::test]
    async fn test_pyo3_submit() {
        let mut executor = Pyo3CodeExecutor::new();
        let config = SessionConfig::default();
        executor.setup(&config).await.unwrap();

        let r = executor.execute("SUBMIT(answer='42')").await.unwrap();
        assert!(r.has_result);
        assert!(r.return_value.is_some());
        let val: serde_json::Value = serde_json::from_str(r.return_value.as_ref().unwrap()).unwrap();
        assert_eq!(val["answer"], "42");
    }

    #[tokio::test]
    async fn test_pyo3_reset() {
        let mut executor = Pyo3CodeExecutor::new();

        executor.execute("x = 42").await.unwrap();
        executor.reset().await.unwrap();

        let r = executor.execute("print(x)").await.unwrap();
        assert!(r.is_error());
    }
}
