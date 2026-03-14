//! Phase 23 + 23B + 26: Code execution backend abstraction.
//!
//! Provides a `CodeExecutor` trait and implementations:
//! - `MicrosandboxExecutor` — hardware-isolated microVM via microsandbox
//! - `Pyo3CodeExecutor` — in-process Python via PyO3 (dev/test fallback)
//!
//! ## Module organization
//!
//! Single file (not split) because PyO3's `#[pyfunction]` macros and
//! `wrap_pyfunction!` require same-module scope.
//!
//! **Sections** (search by `── Section Name ──`):
//! - `CodeExecutor Trait` — trait + ExecutionResult (~90 lines)  
//! - `Microsandbox Executor` — MicrosandboxExecutor (~130 lines)
//! - `LLM + Restate Bridges` — global bridges for llm_query() (~290 lines)
//! - `PyO3 Executor` — Pyo3CodeExecutor (~220 lines)
//! - `Factory + Pool` — ExecutorKind, create_executor, SandboxPool (~125 lines)
//! - `Helper Toolkit` — Python HTTP helpers injected at setup (~230 lines)
//! - `Tests` — unit tests (~65 lines)
//!
//! Phase 23B additions (from Daytona PR analysis):
//! - Persistent REPL state — variables survive across execute() calls
//! - Session setup — inject SUBMIT(), prelude, llm_query()
//! - Typed SUBMIT — structured output matching DSRs Signature fields
//!
//! Phase 26: Sub-LLM bridging — llm_query() as direct #[pyfunction] callback
//! Generated code can call back to the host LLM for semantic reasoning.

use crate::monad::ExecutionResult;
use crate::monad::provider::{LlmProvider, ProviderConfig};
use crate::session::{SessionConfig, extract_submit_result};
use async_trait::async_trait;

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
    /// 3. Inject HTTP/JSON helper toolkit
    /// 4. Inject llm_query bridge if enabled
    async fn setup(&mut self, config: &SessionConfig) -> anyhow::Result<()> {
        // Default: inject SUBMIT + prelude via execute()
        let submit_code = config.generate_submit_code();
        self.execute(&submit_code).await?;

        // Inject ELICIT() function (HITL pause/resume)
        let elicit_code = SessionConfig::generate_elicit_code();
        self.execute(&elicit_code).await?;

        if let Some(ref prelude) = config.prelude {
            self.execute(prelude).await?;
        }

        // Inject HTTP/JSON helper toolkit
        self.execute(HELPER_TOOLKIT_CODE).await?;

        // For microsandbox: inject llm_query as Python code
        // For PyO3: this is a no-op (injected as #[pyfunction] in setup())
        let bridge_code = config.generate_llm_bridge_code();
        if !bridge_code.is_empty() {
            self.execute(&bridge_code).await?;
        }

        Ok(())
    }

    /// Whether this executor supports persistent state across execute() calls.
    fn supports_persistent_state(&self) -> bool {
        false
    }

    /// Reset the session — clear all state.
    async fn reset(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    /// Shutdown the executor and release all resources.
    ///
    /// For microsandbox: stops and destroys the microVM.
    /// For PyO3: no-op (in-process, nothing to clean up).
    async fn shutdown(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}

// ─── Microsandbox Executor ───────────────────────────────────

use microsandbox::{BaseSandbox, PythonSandbox};

/// Hardware-isolated code execution via microsandbox microVM.
///
/// Persistent state: microsandbox's `run_or_start()` keeps the
/// Python process alive — variables, imports, and definitions
/// persist across execute() calls automatically.
pub struct MicrosandboxExecutor {
    sandbox: PythonSandbox,
    session_ready: bool,
    /// Sandbox name (for logging/debugging).
    pub name: String,
}

/// Configuration for the microsandbox executor.
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    pub name: String,
    /// Number of vCPUs for the sandbox (default: 1).
    pub cpus: Option<u32>,
    /// Memory in MiB for the sandbox (default: 256).
    pub memory_mb: Option<u32>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            name: format!("rlm-{}", uuid::Uuid::new_v4().simple()),
            cpus: Some(1),
            memory_mb: Some(256),
        }
    }
}

impl MicrosandboxExecutor {
    /// Create and start a new microsandbox executor.
    /// Requires microsandbox server running (`msb server start --dev`).
    pub async fn new(config: &SandboxConfig) -> anyhow::Result<Self> {
        tracing::info!(name = %config.name, "Creating microsandbox");
        let mut sandbox = PythonSandbox::create(&config.name)
            .await
            .map_err(|e| anyhow::anyhow!("microsandbox create: {e}"))?;
        sandbox
            .start(None)
            .await
            .map_err(|e| anyhow::anyhow!("microsandbox start: {e}"))?;
        tracing::info!(name = %config.name, "Microsandbox started");
        Ok(Self {
            sandbox,
            session_ready: false,
            name: config.name.clone(),
        })
    }

    /// Stop the sandbox.
    pub async fn stop(&mut self) -> anyhow::Result<()> {
        tracing::info!(name = %self.name, "Stopping microsandbox");
        self.sandbox
            .stop()
            .await
            .map_err(|e| anyhow::anyhow!("microsandbox stop: {e}"))?;
        Ok(())
    }
}

#[async_trait]
impl CodeExecutor for MicrosandboxExecutor {
    async fn execute(&mut self, code: &str) -> anyhow::Result<ExecutionResult> {
        let execution = self
            .sandbox
            .run_or_start(code)
            .await
            .map_err(|e| anyhow::anyhow!("microsandbox exec: {e}"))?;
        let stdout = execution.output().await.unwrap_or_default();
        let stderr = execution.error().await.unwrap_or_default();

        // Check for SUBMIT marker in stdout
        if let Some(submit_value) = extract_submit_result(&stdout) {
            let _answer = submit_value
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

    fn supports_persistent_state(&self) -> bool {
        true
    }

    async fn reset(&mut self) -> anyhow::Result<()> {
        self.sandbox
            .stop()
            .await
            .map_err(|e| anyhow::anyhow!("microsandbox stop: {e}"))?;
        self.sandbox
            .start(None)
            .await
            .map_err(|e| anyhow::anyhow!("microsandbox restart: {e}"))?;
        self.session_ready = false;
        Ok(())
    }

    async fn shutdown(&mut self) -> anyhow::Result<()> {
        self.stop().await
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
    /// Dedicated runtime for sub-LLM calls.
    ///
    /// MUST be a separate runtime from the main tokio runtime.
    /// `llm_query_bridge` is called from `spawn_blocking` on the main runtime,
    /// and calling `Handle::block_on()` from within that context would deadlock.
    runtime: tokio::runtime::Runtime,
}

/// Global LLM bridge — updated per-session via atomic swap.
/// Uses AtomicPtr (not OnceLock) so provider config can change between sessions.
static LLM_BRIDGE: std::sync::atomic::AtomicPtr<LlmBridge> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// Initialize the global LLM bridge. Called from Pyo3CodeExecutor::setup().
///
/// When the Restate bridge is active, we skip creating a dedicated runtime
/// because all llm_query() calls are intercepted by RESTATE_LLM_BRIDGE.
/// Creating a runtime in an async context panics on drop.
fn init_llm_bridge(config: ProviderConfig) {
    // Skip if Restate bridge is active — llm_query() will route through child workflows
    if !RESTATE_LLM_BRIDGE
        .load(std::sync::atomic::Ordering::Acquire)
        .is_null()
    {
        return;
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("Failed to create sub-LLM runtime");
    let bridge = Box::new(LlmBridge { config, runtime });
    let new_ptr = Box::into_raw(bridge);
    let old_ptr = LLM_BRIDGE.swap(new_ptr, std::sync::atomic::Ordering::AcqRel);
    if !old_ptr.is_null() {
        // SAFETY: old_ptr was created by Box::into_raw. Sequential session setup
        // guarantees no concurrent readers of the old bridge.
        unsafe {
            drop(Box::from_raw(old_ptr));
        }
    }
}

/// Get a reference to the current LLM bridge (lock-free).
fn get_llm_bridge() -> Option<&'static LlmBridge> {
    let ptr = LLM_BRIDGE.load(std::sync::atomic::Ordering::Acquire);
    if ptr.is_null() {
        None
    } else {
        // SAFETY: ptr was created by Box::into_raw and is valid until next swap.
        // Session setup is sequential — no concurrent swap during use.
        Some(unsafe { &*ptr })
    }
}

// ─── Restate Sub-Agent Bridge ────────────────────────────────
//
// When running under Restate, llm_query() spawns a linked child
// workflow. Each call gets its own reply channel for parallel support.
// No bridge thread — pyfunction dispatches directly to the handler.
//
// Architecture (per-call):
//   Python REPL (sync)                    Handler (async)
//       │                                     │
//       │ llm_query("prompt")                 │
//       │ create reply channel (tx, rx)       │
//       │ dispatch(prompt, reply_tx) ────────→│
//       │                                     │ ctx.workflow_client()
//       │ reply_rx.recv_timeout(120s) ←───────│ reply_tx.send(result)
//       │ return text                         │

/// Bridge for routing llm_query() through Restate child workflows.
///
/// Holds a type-erased dispatch function that sends (prompt, reply_tx)
/// to the handler. Each call creates its own reply channel — supports
/// parallel llm_query() calls naturally.
pub struct RestateLlmBridge {
    /// Dispatch function: captures handler_tx + step counter internally.
    /// Called with (prompt, reply_tx) — sends to handler, which invokes
    /// child workflow and sends result on reply_tx.
    pub dispatch: Box<dyn Fn(String, std::sync::mpsc::SyncSender<String>) + Send + Sync>,
}

/// Lock-free global Restate bridge — updated per-workflow-invocation via atomic swap.
/// Uses AtomicPtr for zero-cost reads (no locking, no contention).
static RESTATE_LLM_BRIDGE: std::sync::atomic::AtomicPtr<RestateLlmBridge> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// Set the Restate LLM bridge for the current invocation (atomic swap).
/// Called from restate::run_agent_loop().
///
/// Drops the previous bridge safely — invocations are sequential,
/// so no concurrent llm_query() reads the old bridge during swap.
pub fn set_restate_llm_bridge(bridge: Box<RestateLlmBridge>) {
    let new_ptr = Box::into_raw(bridge);
    let old_ptr = RESTATE_LLM_BRIDGE.swap(new_ptr, std::sync::atomic::Ordering::AcqRel);
    if !old_ptr.is_null() {
        // SAFETY: old_ptr was created by Box::into_raw. Sequential invocations
        // guarantee no concurrent readers of the old bridge.
        unsafe {
            drop(Box::from_raw(old_ptr));
        }
    }
}

/// Get a reference to the current Restate bridge (lock-free).
fn get_restate_llm_bridge() -> Option<&'static RestateLlmBridge> {
    let ptr = RESTATE_LLM_BRIDGE.load(std::sync::atomic::Ordering::Acquire);
    if ptr.is_null() {
        None
    } else {
        // SAFETY: ptr was created by Box::into_raw and is never freed.
        // It's valid for 'static because we intentionally leak it.
        Some(unsafe { &*ptr })
    }
}

// ─── Context Memory Bridge ──────────────────────────────────────
//
// Agent-driven context memory: offload/recall/manifest/search.
// Uses a simple Arc<Mutex<ContextMemoryStore>> global.
// The Python functions lock the mutex directly — safe because the bg
// thread is blocked during code execution (no contention).

use crate::monad::context_memory::{ContextMemoryStore, SharedMemoryStore};

/// Global shared memory store — set per session.
static CONTEXT_MEMORY_STORE: std::sync::OnceLock<SharedMemoryStore> = std::sync::OnceLock::new();

/// Initialize or get the global context memory store.
pub fn init_context_memory_store() -> SharedMemoryStore {
    CONTEXT_MEMORY_STORE
        .get_or_init(ContextMemoryStore::new_shared)
        .clone()
}

/// Reset the store (for new sessions).
pub fn reset_context_memory_store() {
    if let Some(store) = CONTEXT_MEMORY_STORE.get() {
        let Ok(mut s) = store.lock() else {
            tracing::error!("context memory store lock poisoned during reset");
            return;
        };
        *s = ContextMemoryStore::new();
    }
}

#[pyfunction]
fn memory_offload_bridge(_py: Python, label: String, content: String) -> PyResult<String> {
    let store = CONTEXT_MEMORY_STORE.get().ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err("Context memory not initialized")
    })?;
    let mut s = store.lock().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("memory lock failed: {e}"))
    })?;
    Ok(s.offload(&label, content))
}

#[pyfunction]
fn memory_recall_bridge(_py: Python, segment_id: String) -> PyResult<String> {
    let store = CONTEXT_MEMORY_STORE.get().ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err("Context memory not initialized")
    })?;
    let s = store.lock().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("memory lock failed: {e}"))
    })?;
    Ok(s.recall(&segment_id))
}

#[pyfunction]
fn memory_manifest_bridge(_py: Python) -> PyResult<String> {
    let store = CONTEXT_MEMORY_STORE.get().ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err("Context memory not initialized")
    })?;
    let s = store.lock().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("memory lock failed: {e}"))
    })?;
    Ok(s.manifest())
}

#[pyfunction]
fn memory_search_bridge(_py: Python, query: String) -> PyResult<String> {
    let store = CONTEXT_MEMORY_STORE.get().ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err("Context memory not initialized")
    })?;
    let s = store.lock().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("memory lock failed: {e}"))
    })?;
    Ok(s.search(&query))
}

/// Timeout for child workflow completion (2 minutes).
const SUB_AGENT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// Python-callable: llm_query(prompt) -> str
///
/// Releases the GIL via py.allow_threads(), blocks on the async LLM
/// call, returns the response text. This is the "recursive" in RLM —
/// generated code can call the LLM for semantic reasoning.
///
/// **Dual-mode**: When running under Restate, routes through per-call
/// reply channels to spawn linked child workflows. Supports parallel
/// calls. Otherwise, calls the LLM directly via LLM_BRIDGE.
#[pyfunction]
fn llm_query_bridge(py: Python, prompt: String) -> PyResult<String> {
    // Check Restate bridge first — lock-free atomic load
    if let Some(bridge) = get_restate_llm_bridge() {
        return py.detach(move || {
            // Per-call reply channel — supports parallel llm_query() calls
            let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel::<String>(1);

            // Dispatch to handler (captured handler_tx + step counter)
            (bridge.dispatch)(prompt, reply_tx);

            // Block with timeout until child workflow returns result
            reply_rx.recv_timeout(SUB_AGENT_TIMEOUT).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Restate sub-agent timed out or failed after {}s: {e}",
                    SUB_AGENT_TIMEOUT.as_secs()
                ))
            })
        });
    }

    // Fallback: direct LLM call (non-Restate mode)
    let bridge = get_llm_bridge().ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err(
            "LLM bridge not initialized. Ensure session has enable_sub_llm=true.",
        )
    })?;

    py.detach(move || {
        let provider = LlmProvider::new(bridge.config.clone());
        bridge.runtime.block_on(async {
            provider.complete(&prompt).await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("LLM query failed: {e}"))
            })
        })
    })
}

/// Python-callable: llm_query_batched(prompts) -> list[str]
///
/// Runs multiple LLM calls concurrently for efficiency.
#[pyfunction]
fn llm_query_batched_bridge(py: Python, prompts: Vec<String>) -> PyResult<Vec<String>> {
    let bridge = get_llm_bridge()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("LLM bridge not initialized."))?;

    py.detach(move || {
        bridge.runtime.block_on(async {
            let mut results = Vec::with_capacity(prompts.len());
            // Run sequentially to avoid spawning issues; still async I/O
            for prompt in &prompts {
                let provider = LlmProvider::new(bridge.config.clone());
                let result = provider.complete(prompt).await.map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Batched LLM query failed: {e}"
                    ))
                })?;
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
        let globals_py = Python::attach(|py| self.globals.as_ref().unwrap().clone_ref(py));
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

                let c_code = CString::new(code.as_str())
                    .unwrap_or_else(|_| CString::new("print('code contained null byte')").unwrap());

                match py.run(&c_code, Some(&globals), None) {
                    Ok(()) => {
                        let stdout: String = string_io
                            .call_method0("getvalue")
                            .and_then(|v| v.extract())
                            .unwrap_or_default();
                        let stderr: String = err_io
                            .call_method0("getvalue")
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
                            return ExecutionResult::submitted(stdout, result_json.clone());
                        }

                        if stderr.is_empty() {
                            ExecutionResult::success(if stdout.is_empty() {
                                "(no output)"
                            } else {
                                &stdout
                            })
                        } else {
                            ExecutionResult {
                                stdout,
                                stderr,
                                return_value: None,
                                exception: None,
                                has_result: false,
                                submitted: false,
                            }
                        }
                    }
                    Err(e) => {
                        // Check if this is a FinalOutput exception (from SUBMIT)
                        let err_str = e.to_string();
                        if err_str.contains("FinalOutput") {
                            let stdout: String = string_io
                                .call_method0("getvalue")
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
                                return ExecutionResult::submitted(stdout, result_json.clone());
                            }
                        }

                        ExecutionResult::error("PythonError", &err_str)
                    }
                }
            })
        })
        .await?;
        Ok(result)
    }

    async fn setup(&mut self, config: &SessionConfig) -> anyhow::Result<()> {
        self.ensure_globals();

        // 1. Inject SUBMIT function
        let submit_code = config.generate_submit_code();
        self.execute(&submit_code).await?;

        // 1b. Inject ELICIT() function (HITL pause/resume)
        let elicit_code = SessionConfig::generate_elicit_code();
        self.execute(&elicit_code).await?;

        // 2. Inject prelude
        if let Some(ref prelude) = config.prelude {
            self.execute(prelude).await?;
        }

        // 3. Inject HTTP/JSON helper toolkit
        self.execute(HELPER_TOOLKIT_CODE).await?;

        // 4. Inject llm_query as #[pyfunction] if enabled (Phase 26)
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
                })
                .map_err(|e| anyhow::anyhow!("Failed to inject llm_query bridge: {e}"))?;

                tracing::info!(
                    "Sub-LLM bridge injected: llm_query() + llm_query_batched() available in sandbox"
                );
            } else {
                tracing::warn!(
                    "enable_sub_llm is true but no provider_config set — llm_query will not be available"
                );
            }
        }

        // 5. Inject context memory bridge functions (always available)
        Python::attach(|py| -> PyResult<()> {
            let globals = self.globals.as_ref().unwrap().bind(py);
            let offload_fn = wrap_pyfunction!(memory_offload_bridge, py)?;
            globals.set_item("memory_offload", offload_fn)?;
            let recall_fn = wrap_pyfunction!(memory_recall_bridge, py)?;
            globals.set_item("memory_recall", recall_fn)?;
            let manifest_fn = wrap_pyfunction!(memory_manifest_bridge, py)?;
            globals.set_item("memory_manifest", manifest_fn)?;
            let search_fn = wrap_pyfunction!(memory_search_bridge, py)?;
            globals.set_item("memory_search", search_fn)?;
            Ok(())
        })
        .map_err(|e| anyhow::anyhow!("Failed to inject memory bridge: {e}"))?;

        tracing::info!(
            "Context memory bridge injected: memory_offload/recall/manifest/search available"
        );

        self.session_ready = true;
        Ok(())
    }

    fn supports_persistent_state(&self) -> bool {
        true
    }

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
        ExecutorKind::Pyo3 => Ok(Box::new(Pyo3CodeExecutor::new())),
    }
}

// ─── Sandbox Pool ────────────────────────────────────────────

use std::sync::atomic::{AtomicUsize, Ordering};

/// A pool of pre-warmed microsandbox executors for reuse.
///
/// Avoids the overhead of creating a new microVM for every agent run.
/// VMs are reset (not destroyed) when returned to the pool.
pub struct SandboxPool {
    available: tokio::sync::Mutex<Vec<MicrosandboxExecutor>>,
    /// Maximum number of sandboxes the pool can manage.
    pub max_size: usize,
    /// Number of sandboxes currently checked out.
    active_count: AtomicUsize,
    /// Configuration template for new sandboxes.
    config_template: SandboxConfig,
}

impl SandboxPool {
    /// Create a new pool and optionally pre-warm some sandboxes.
    pub async fn new(
        max_size: usize,
        pre_warm: usize,
        config: Option<SandboxConfig>,
    ) -> anyhow::Result<Self> {
        let config_template = config.unwrap_or_default();
        let mut pool = Vec::with_capacity(max_size);
        for _ in 0..pre_warm.min(max_size) {
            let sandbox_config = SandboxConfig {
                name: format!("rlm-{}", uuid::Uuid::new_v4().simple()),
                ..config_template.clone()
            };
            let executor = MicrosandboxExecutor::new(&sandbox_config).await?;
            pool.push(executor);
        }
        tracing::info!(max_size, pre_warmed = pool.len(), "Sandbox pool created");
        Ok(Self {
            available: tokio::sync::Mutex::new(pool),
            max_size,
            active_count: AtomicUsize::new(0),
            config_template,
        })
    }

    /// Check out a sandbox from the pool.
    ///
    /// Returns a pre-warmed sandbox if available, otherwise creates a new one
    /// (up to max_size). Returns an error if the pool is exhausted.
    pub async fn checkout(&self) -> anyhow::Result<MicrosandboxExecutor> {
        let current = self.active_count.load(Ordering::Relaxed);
        {
            let mut pool = self.available.lock().await;
            if let Some(executor) = pool.pop() {
                self.active_count.fetch_add(1, Ordering::Relaxed);
                tracing::debug!(name = %executor.name, active = current + 1, "Checked out sandbox from pool");
                return Ok(executor);
            }
        }
        // Pool empty — create a new one if under max
        if current < self.max_size {
            let sandbox_config = SandboxConfig {
                name: format!("rlm-{}", uuid::Uuid::new_v4().simple()),
                ..self.config_template.clone()
            };
            let executor = MicrosandboxExecutor::new(&sandbox_config).await?;
            self.active_count.fetch_add(1, Ordering::Relaxed);
            tracing::info!(name = %executor.name, active = current + 1, "Created new sandbox (pool was empty)");
            Ok(executor)
        } else {
            Err(anyhow::anyhow!(
                "Sandbox pool exhausted (max_size={}, all active)",
                self.max_size
            ))
        }
    }

    /// Return a sandbox to the pool for reuse.
    ///
    /// Resets the sandbox state before making it available.
    /// If the pool is full, the sandbox is stopped and discarded.
    pub async fn return_to_pool(&self, mut executor: MicrosandboxExecutor) {
        self.active_count.fetch_sub(1, Ordering::Relaxed);
        // Reset the session so the next user gets a clean slate
        if let Err(e) = executor.reset().await {
            tracing::warn!(name = %executor.name, error = %e, "Failed to reset sandbox, stopping instead");
            let _ = executor.stop().await;
            return;
        }
        let mut pool = self.available.lock().await;
        if pool.len() < self.max_size {
            tracing::debug!(name = %executor.name, pool_size = pool.len() + 1, "Returned sandbox to pool");
            pool.push(executor);
        } else {
            tracing::debug!(name = %executor.name, "Pool full, stopping excess sandbox");
            let _ = executor.stop().await;
        }
    }

    /// Shutdown all sandboxes in the pool (both available and inform about active).
    pub async fn shutdown_all(&self) {
        let mut pool = self.available.lock().await;
        let count = pool.len();
        for mut executor in pool.drain(..) {
            if let Err(e) = executor.stop().await {
                tracing::warn!(name = %executor.name, error = %e, "Failed to stop sandbox during pool shutdown");
            }
        }
        let active = self.active_count.load(Ordering::Relaxed);
        tracing::info!(
            stopped = count,
            still_active = active,
            "Sandbox pool shutdown complete"
        );
        if active > 0 {
            tracing::warn!(
                active,
                "Some sandboxes are still checked out — they will be orphaned unless returned"
            );
        }
    }
}

// ── Helper toolkit injected into every executor session ───────────────────

/// Python helper functions pre-loaded into the executor.
///
/// Uses `subprocess.run(["curl", ...])` for HTTP calls to avoid
/// urllib/requests deadlocks in PyO3's embedded interpreter.
const HELPER_TOOLKIT_CODE: &str = r#"
import subprocess as _subprocess
import json as _json

class HttpResponse:
    """Structured HTTP response from http_call()."""
    def __init__(self, status_code, body, headers=None, ok=True, error=None):
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}
        self.ok = ok
        self.error = error
        self._json = None

    def json(self):
        """Parse body as JSON, cached after first call."""
        if self._json is None:
            try:
                self._json = _json.loads(self.body) if self.body else {}
            except _json.JSONDecodeError:
                self._json = {"_raw": self.body}
        return self._json

    def __repr__(self):
        preview = self.body[:200] if self.body else ""
        return f"HttpResponse(status={self.status_code}, ok={self.ok}, body={preview!r})"

    def __getitem__(self, key):
        """Allow resp['key'] shorthand for resp.json()['key']."""
        return self.json()[key]

    def get(self, key, default=None):
        """Allow resp.get('key') shorthand for resp.json().get('key')."""
        return self.json().get(key, default)


def http_call(method, url, json_data=None, headers=None, timeout=30):
    """Make an HTTP request using curl. Returns an HttpResponse.

    Args:
        method: HTTP method (GET, POST, PUT, DELETE, PATCH)
        url: Full URL to call
        json_data: Optional dict to send as JSON body
        headers: Optional dict of extra headers
        timeout: Request timeout in seconds (default 30)

    Returns:
        HttpResponse with .status_code, .body, .json(), .ok, .error

    Example:
        resp = http_call("POST", "http://127.0.0.1:8080/MyWorkflow/key/run",
                         json_data={"workflow_id": "test"})
        print(resp.status_code)  # 200
        print(resp.json())       # {"is_success": True, ...}
        print(resp["results"])   # shorthand for resp.json()["results"]
    """
    cmd = ["curl", "-s", "-w", "\n__HTTP_STATUS__%{http_code}", "--max-time", str(timeout),
           "-X", method.upper()]

    # Add default headers
    cmd.extend(["-H", "Content-Type: application/json"])
    cmd.extend(["-H", "Accept: application/json"])

    # Add custom headers
    if headers:
        for k, v in headers.items():
            cmd.extend(["-H", f"{k}: {v}"])

    # Add JSON body
    if json_data is not None:
        cmd.extend(["-d", _json.dumps(json_data) if isinstance(json_data, dict) else str(json_data)])

    cmd.append(url)

    try:
        result = _subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
        output = result.stdout

        # Extract HTTP status code from curl's -w flag
        status_code = 0
        body = output
        if "__HTTP_STATUS__" in output:
            parts = output.rsplit("__HTTP_STATUS__", 1)
            body = parts[0].rstrip("\n")
            try:
                status_code = int(parts[1].strip())
            except ValueError:
                status_code = 0

        ok = 200 <= status_code < 400

        if result.returncode != 0 and status_code == 0:
            return HttpResponse(0, "", ok=False,
                                error=f"curl failed (exit {result.returncode}): {result.stderr.strip()}")

        return HttpResponse(status_code, body, ok=ok)

    except _subprocess.TimeoutExpired:
        return HttpResponse(0, "", ok=False, error=f"Request timed out after {timeout}s")
    except Exception as e:
        return HttpResponse(0, "", ok=False, error=str(e))


def http_get(url, headers=None, timeout=30):
    """Shorthand for http_call('GET', url, ...)."""
    return http_call("GET", url, headers=headers, timeout=timeout)


def http_post(url, json_data=None, headers=None, timeout=30):
    """Shorthand for http_call('POST', url, json_data, ...)."""
    return http_call("POST", url, json_data=json_data, headers=headers, timeout=timeout)


def http_put(url, json_data=None, headers=None, timeout=30):
    """Shorthand for http_call('PUT', url, json_data, ...)."""
    return http_call("PUT", url, json_data=json_data, headers=headers, timeout=timeout)


def http_delete(url, headers=None, timeout=30):
    """Shorthand for http_call('DELETE', url, ...)."""
    return http_call("DELETE", url, headers=headers, timeout=timeout)


def json_extract(data, *keys, default=None):
    """Safely extract nested values from a dict/JSON.

    Args:
        data: dict or HttpResponse to extract from
        *keys: sequence of keys/indices to traverse
        default: value to return if path doesn't exist

    Example:
        json_extract(resp, "results", "unit-guid", "created_sub_ledgers", 0, "name")
        json_extract({"a": {"b": [1,2,3]}}, "a", "b", 2)  # => 3
    """
    current = data.json() if hasattr(data, 'json') and callable(data.json) else data
    for key in keys:
        try:
            if isinstance(current, dict):
                current = current[key]
            elif isinstance(current, (list, tuple)):
                current = current[int(key)]
            else:
                return default
        except (KeyError, IndexError, TypeError, ValueError):
            return default
    return current


def json_pretty(data, indent=2):
    """Pretty-print JSON data. Accepts dicts, HttpResponse, or JSON strings.

    Example:
        json_pretty(resp)
        json_pretty({"key": "value"})
    """
    if hasattr(data, 'json') and callable(data.json):
        data = data.json()
    elif isinstance(data, str):
        try:
            data = _json.loads(data)
        except _json.JSONDecodeError:
            print(data)
            return
    formatted = _json.dumps(data, indent=indent, default=str)
    print(formatted)
    return formatted


def fetch_all(calls):
    """Execute multiple HTTP calls sequentially and return all responses.

    Args:
        calls: list of (method, url, json_data) tuples or (method, url) tuples

    Returns:
        list of HttpResponse objects

    Example:
        responses = fetch_all([
            ("GET", "http://127.0.0.1:8080/health"),
            ("POST", "http://127.0.0.1:8080/WorkflowA/key/run", {"id": "a"}),
            ("POST", "http://127.0.0.1:8080/WorkflowB/key/run", {"id": "b"}),
        ])
        for r in responses:
            print(r.status_code, r.json())
    """
    results = []
    for call in calls:
        method = call[0]
        url = call[1]
        json_data = call[2] if len(call) > 2 else None
        headers = call[3] if len(call) > 3 else None
        results.append(http_call(method, url, json_data=json_data, headers=headers))
    return results


def assert_status(resp, expected=200, msg=None):
    """Assert an HttpResponse has the expected status code.

    Args:
        resp: HttpResponse to check
        expected: expected status code (default 200), can be int or list of ints
        msg: optional error message

    Raises:
        AssertionError if status doesn't match

    Example:
        resp = http_post("http://...", json_data={...})
        assert_status(resp, [200, 201])
    """
    if isinstance(expected, int):
        expected = [expected]
    if resp.status_code not in expected:
        detail = msg or f"Expected status {expected}, got {resp.status_code}"
        body_preview = resp.body[:500] if resp.body else "(empty)"
        raise AssertionError(f"{detail}\nResponse body: {body_preview}")
    return resp


# Signal that the toolkit loaded successfully
print("__TOOLKIT_READY__")
"#;

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
        let r = executor
            .execute("print(json.dumps({'a': 1}))")
            .await
            .unwrap();
        assert!(r.stdout.contains(r#"{"a": 1}"#));
    }

    #[tokio::test]
    async fn test_pyo3_persistent_functions() {
        let mut executor = Pyo3CodeExecutor::new();

        executor
            .execute("def greet(name):\n    return f'Hello, {name}!'")
            .await
            .unwrap();
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
        let val: serde_json::Value =
            serde_json::from_str(r.return_value.as_ref().unwrap()).unwrap();
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
