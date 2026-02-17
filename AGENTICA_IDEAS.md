# Agentica → rig-rlm: Architecture Analysis & Integration Ideas

## Part 1: What Agentica Server Really Does

### High-Level Summary

Agentica is a **production-grade, type-safe AI agent framework** by Symbolica. It is *not* just "another LLM wrapper" – it's an entire runtime for agents that can interact with **live user code, objects, and SDKs** inside a sandboxed REPL. The server handles:

1. **Session Management** — multiplexed WebSocket connections, multiple agents per session, invocation lifecycle  
2. **Agent Core** — a monadic composition engine for building inference-REPL loops  
3. **Sandboxed Code Execution** — WASM-based sandboxes where agents can actually run code  
4. **Multi-Provider Inference** — abstracts OpenAI (Responses + Chat Completions) and Anthropic (Messages API)  
5. **Observability** — OpenTelemetry tracing, Prometheus metrics, structured logs  

### Core Architecture Breakdown

#### 1. The Monad System (`com/` module)
This is the *heart* of Agentica's design and the most novel part:

- **`HistoryMonad[A]`** — An abstract monad representing a computation that produces side effects on chat history and yields a value of type `A`.
  - `Pure(value)` — wraps a value with no side effects
  - `Do(action, continuation)` — an action followed by "what to do next"
  - `bind(f)` — standard monadic bind (sequencing)
  - `>>` operator overloaded for ergonomic chaining

- **`Action[A]`** — Abstract base class for side effects (insert text, run code, call inference, etc.)
  - `Insert(content, role)` — insert text into conversation
  - `ModelInference()` — call the LLM
  - `ReplRunCode(source)` — execute code in the sandbox  
  - `Capture(name, value)` / `Retrieve(name)` — context variable management
  - `SendLog(message)` — structured event logging

- **`Context`** — The execution context that the monad runner uses. Holds the sandbox, inference system, captured variables, and config. `Context.run(monad)` is the interpreter that walks the `Do` chain.

- **`do(MonadT)` decorator** — Haskell-style do-notation using Python generators:
  ```python
  @do(HistoryMonad[None])
  def interaction_monad():
      session: ReplSessionInfo = yield repl_session_info()
      while True:
          response: Generation = yield model_inference()
          code_block = response.code
          if code_block is None:
              msg = yield _explain("missing-code.txt")
              yield insert_execution_result(msg)
              continue
          summary = yield repl_run_code(code_block)
          if summary.has_result:
              return
          yield insert_execution_result(summary.output)
  ```

**Why this matters**: The monad system cleanly separates the *description* of what an agent should do from *how* it gets executed. This makes the agent loop composable, testable, and provider-agnostic.

#### 2. Agent Monads — The Three Phases
Each agent invocation goes through three composed monads:

| Phase | Monad | Purpose |
|-------|-------|---------|
| System | `system_monad` | Set up system prompt + few-shot examples |
| User | `user_monad` | Insert user task/prompt |
| Interaction | `interaction_monad` | The generate→execute→feedback loop |

These are composed:  
```python
user_monad(task) >> interaction_monad
```

The `interaction_monad` is the core agent loop:
1. Call `model_inference()` → get `Generation` (text + optional code)
2. If code found → `repl_run_code(code)` → get execution result
3. Feed result back → `insert_execution_result(output)` 
4. Repeat until the code does a `return` or `raise`

#### 3. Prompt Templating with Jinja2
- Templates are organized by provider (`openai/`, `anthropic/`) and role (`agent/`, `function/`)
- Jinja2 templates with recursive inclusion, variable resolution from session info
- Few-shot examples loaded from YAML files
- Provider-specific formatting (Anthropic uses XML tags, OpenAI uses usernames)

#### 4. Multi-Provider Inference System
- **`InferenceSystem`** abstract base class with three implementations:
  - `ResponsesSystem` — OpenAI Responses API (supports tool calls)
  - `ChatCompletionsSystem` — OpenAI Chat Completions API
  - `MessagesSystem` — Anthropic Messages API
- Provider routing via `InferenceProvider` with fnmatch patterns
- Streaming support with delta-based chunk delivery
- Retry with exponential backoff for rate limits

#### 5. Sandboxed Execution
- **Guest** (runs in WASM): `AgentRepl` + `AgentWorld` — provides a Python REPL with type-safe builtins, preloaded modules, and a message loop communicating via WIT bindings
- **Host** (`Sandbox`): manages the WASM runtime, handles send/recv of binary messages (the "warp" protocol)
- Code execution is truly sandboxed — the agent can't escape the WASM container

#### 6. Session & Connection Management
- **`ServerSessionManager`**: manages sessions (CID), agents (UID), invocations (IID)
- **`Multiplexer`**: handles multiplexed invocations over a single WebSocket
  - Multiple agents per socket
  - Per-invocation recv queues
  - Concurrency limits
  - Full lifecycle: invoke → run → cancel → cleanup

---

## Part 2: What rig-rlm Currently Does

rig-rlm is a **Rust implementation of the Recursive Language Model (RLM) pattern**:

1. **`RigRlm<T: ExecutionEnvironment>`** — Core struct holding a rig `Agent` and a `REPL<T>`
2. **`REPL<T>`** — Stores context (HashMap), dispatches commands to the executor
3. **`Command`** — Parsed from LLM output: `RUN`, `FINAL`, or ` ```repl` code blocks
4. **`ExecutionEnvironment` trait** — Abstract code execution; currently only `Pyo3Executor`
5. **`Pyo3Executor`** — Runs Python code via PyO3, captures stdout, supports `query_llm()` recursion

The agent loop in `query()`:
1. Build initial prompt with instructions
2. Send to LLM, get response
3. Parse response for commands
4. If `FINAL` → return the answer
5. If code/RUN → execute, feed result back as next prompt
6. Loop

---

## Part 3: Integration Ideas (Agentica → rig-rlm)

### 🔴 Idea 1: Monadic Agent Loop (HIGH IMPACT)

**What Agentica does**: Uses `HistoryMonad` to compose agent phases (system → user → interaction → code execution → feedback) in a declarative, reusable way.

**What rig-rlm does**: Has a hardcoded `loop` in `query()`.

**Integration plan**:
```rust
// A monadic action trait
trait AgentAction {
    type Output;
    async fn perform(&self, ctx: &mut AgentContext) -> Result<Self::Output, AgentError>;
}

// Concrete actions
struct ModelInference;      // Call the LLM
struct ExecuteCode(String); // Run code in REPL
struct InsertMessage { role: Role, content: String }
struct ExtractCommand;      // Parse LLM output for commands

// The monad (simplified — Rust doesn't have HKTs, so use an enum)
enum AgentStep<T> {
    Pure(T),
    Action(Box<dyn AgentAction<Output = Box<dyn Any>>>, Box<dyn FnOnce(Box<dyn Any>) -> AgentStep<T>>),
}

impl<T> AgentStep<T> {
    fn bind<U>(self, f: impl FnOnce(T) -> AgentStep<U>) -> AgentStep<U> { ... }
}
```

**Why**: Makes the agent loop testable (mock actions), composable (swap phases), and extensible (add new action types without touching the loop).

---

### 🔴 Idea 2: Proper Conversation History Management (HIGH IMPACT)

**What Agentica does**: The `InferenceSystem` maintains full conversation history with role-based insertion (`system`, `user`, `assistant`, `execution`). History is structured and provider-aware.

**What rig-rlm does**: Manually builds `Vec<Message>` in the loop, and uses rig's `.chat()` which takes a prompt + history.

**Integration plan**:
```rust
struct ConversationHistory {
    messages: Vec<Message>,
}

impl ConversationHistory {
    fn insert_system(&mut self, content: &str) { ... }
    fn insert_user(&mut self, content: &str) { ... }
    fn insert_assistant(&mut self, content: &str) { ... }
    fn insert_execution_result(&mut self, output: &str) { ... }
    fn to_rig_messages(&self) -> Vec<Message> { ... }
}
```

**Why**: Currently the history management is ad-hoc. Structured history enables better context management, truncation strategies, and debugging.

---

### 🟡 Idea 3: Provider Abstraction & Multiple Backend Support (MEDIUM)

**What Agentica does**: `InferenceProvider` with pattern-matching routing (`openai/*` → OpenAI, `anthropic/*` → Anthropic). Config-driven via YAML.

**What rig-rlm does**: Hardcoded to OpenAI-compatible endpoints only.

**Integration plan**:
```rust
trait InferenceProvider: Send + Sync {
    fn matches(&self, model_id: &str) -> bool;
    async fn complete(&self, messages: &[Message], config: &InferenceConfig) -> Result<Generation, Error>;
}

struct ProviderRouter {
    providers: Vec<Box<dyn InferenceProvider>>,
}

impl ProviderRouter {
    fn route(&self, model_id: &str) -> Option<&dyn InferenceProvider> {
        self.providers.iter().find(|p| p.matches(model_id))
    }
}
```

**Why**: Lets you switch between OpenAI, Anthropic, local models without code changes. Load config from YAML/TOML.

---

### 🟡 Idea 4: Better Error Handling in Code Execution (MEDIUM)

**What Agentica does**: 
- Returns `ReplEvaluationInfo` with structured error info (exception name, traceback, has_result flag)
- Provides explanatory prompts for common errors ("empty-response", "missing-code", "uncaught-exit", "multiple-code-blocks")
- Feeds error context back to the LLM so it can self-correct

**What rig-rlm does**: Returns errors as strings; panics on missing args.

**Integration plan**:
```rust
struct ExecutionResult {
    stdout: String,
    stderr: String,
    return_value: Option<String>,
    exception: Option<ExecutionError>,
    has_result: bool,
}

struct ExecutionError {
    name: String,
    message: String,
    traceback: Option<String>,
}

// Error guidance templates
const EMPTY_RESPONSE: &str = "Your response was empty. Please provide code or a final answer.";
const MISSING_CODE: &str = "No code block found. Wrap your code in ```repl blocks.";
```

**Why**: The LLM needs structured feedback to fix its own mistakes. Agentica's approach of providing specific guidance for each error type dramatically improves agent reliability.

---

### 🟡 Idea 5: Prompt Templating System (MEDIUM)

**What Agentica does**: Jinja2 templates organized by provider & role, with variable injection from session state.

**What rig-rlm does**: One giant `const PREAMBLE` string.

**Integration plan**:
```rust
// Use Tera (Rust's Jinja2-like template engine)
use tera::Tera;

struct PromptSystem {
    tera: Tera,
}

impl PromptSystem {
    fn new() -> Self {
        let mut tera = Tera::new("templates/**/*.txt").unwrap();
        Self { tera }
    }
    
    fn render_system_prompt(&self, ctx: &PromptContext) -> String { ... }
    fn render_user_prompt(&self, task: &str, ctx: &PromptContext) -> String { ... }
    fn render_error_guidance(&self, error_type: &str) -> String { ... }
}
```

**Why**: Separates prompt engineering from code. Makes it easy to iterate on prompts, A/B test, and support different providers.

---

### 🟢 Idea 6: Execution Environment Safety (LOWER but VALUABLE)

**What Agentica does**: Full WASM sandbox with WIT bindings. The agent runs in a constrained environment.

**What rig-rlm does**: Direct PyO3 calls (no sandboxing) and raw `std::process::Command` for shell.

**Integration plan** (incremental):
```rust
// Step 1: Add resource limits to Pyo3Executor
impl Pyo3Executor {
    fn execute_code_with_limits(&self, code: String, limits: ExecutionLimits) -> Result<String, Error> {
        // Set sys.settrace for instruction counting
        // Capture stderr separately
        // Add timeout via tokio::time::timeout
    }
}

// Step 2: Sandbox shell commands
impl Pyo3Executor {
    fn execute_shell_with_sandbox(cmd: &str, allowed_dirs: &[PathBuf]) -> Result<String, Error> {
        // Restrict to allowed directories
        // No network access
        // Resource limits
    }
}

struct ExecutionLimits {
    max_time_ms: u64,
    max_memory_bytes: u64,
    max_output_chars: usize,
    allowed_modules: Vec<String>,
}
```

**Why**: Without sandboxing, the LLM can delete files, exfiltrate data, or DoS the host. Even basic limits (timeouts, output truncation) dramatically improve safety.

---

### 🟢 Idea 7: Few-Shot Example System (LOWER but EASY WIN)

**What Agentica does**: Loads few-shot examples from YAML, injects them as conversation history (supporting both tool-call and markdown formats).

**What rig-rlm does**: No few-shot examples at all.

**Integration plan**:
```rust
// Load from YAML
#[derive(Deserialize)]
struct FewShotExample {
    role: String,         // "instructions", "assistant", "execution"
    value: Option<String>,
    code: Option<String>,
    text: Option<String>,
}

fn load_few_shot_examples() -> Vec<Message> {
    let yaml = include_str!("../templates/few-shot.yaml");
    let examples: Vec<FewShotExample> = serde_yaml::from_str(yaml).unwrap();
    examples.into_iter().map(|e| e.to_message()).collect()
}
```

**Why**: Few-shot examples are the single most effective way to improve agent behavior without changing the model.

---

### 🟢 Idea 8: Session & State Persistence (LOWER PRIORITY)

**What Agentica does**: Full session management with UUID-based tracking, WebSocket multiplexing, invocation lifecycle.

**What rig-rlm does**: Single-shot — one query, one response.

**Integration plan** (when needed):
```rust
struct Session {
    id: Uuid,
    agents: Vec<AgentHandle>,
    history: ConversationHistory,
    context: HashMap<String, String>,
}

// Support multi-turn conversations
impl RigRlm<T> {
    fn new_session(&self) -> Session { ... }
    async fn query_in_session(&self, session: &mut Session, input: &str) -> Result<String, Error> { ... }
}
```

---

## Part 4: Recommended Implementation Order

| Priority | Idea | Effort | Impact |
|----------|------|--------|--------|
| 1 | **Conversation History Management** (#2) | Low | High |
| 2 | **Error Handling in Execution** (#4) | Low-Med | High |
| 3 | **Few-Shot Examples** (#7) | Low | Medium |
| 4 | **Prompt Templating** (#5) | Medium | Medium |
| 5 | **Execution Safety** (#6) | Medium | Medium |
| 6 | **Provider Abstraction** (#3) | Medium | Medium |
| 7 | **Monadic Agent Loop** (#1) | High | High (long-term) |
| 8 | **Session Persistence** (#8) | High | Low (for now) |

Start with #2 and #4 — they're low-effort, high-impact improvements that will immediately make the agent more reliable. Then add #7 (few-shot) for easy quality gains. The monadic loop (#1) is the biggest architectural improvement but also the most work — tackle it when you're ready to refactor the core.

---

## Part 5: Key Takeaways

1. **Agentica's monad system is its secret weapon** — it makes agent logic composable, testable, and provider-agnostic. Rust's type system (enums + traits) is actually a *better* fit for this pattern than Python.

2. **The generate→execute→feedback loop is universal** — both projects implement exactly the same core loop. The difference is in how cleanly it's abstracted.

3. **Error recovery is critical** — Agentica invests heavily in helping the LLM fix its own mistakes (guidance templates, structured error info). This is the lowest-effort, highest-impact improvement for rig-rlm.

4. **Sandboxing matters** — rig-rlm's direct `std::process::Command` and unsandboxed PyO3 are fine for demos but dangerous for anything user-facing.

5. **The `ExecutionEnvironment` trait in rig-rlm is already a good foundation** — it maps directly to Agentica's sandbox abstraction. Just extend it with richer output types and safety limits.
