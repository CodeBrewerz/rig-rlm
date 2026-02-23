# rig-rlm

Monadic AI agent with durable execution, MCP server, sandboxed Python, and DSPy-rs prompt optimization.

**Stack:** [Rig](https://rig.rs) (LLM) · [Restate](https://restate.dev) (durable execution) · [rmcp](https://crates.io/crates/rmcp) (MCP server) · [PyO3](https://pyo3.rs) (Python sandbox) · [DSPy-rs](https://crates.io/crates/dspy) (optimization) · [Turso](https://turso.tech) (persistence)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Entry Points                      │
├──────────┬──────────────┬───────────────────────────┤
│  CLI     │  MCP Server  │  Restate Workflow          │
│ (main)   │  (stdio)     │  (HTTP, durable)           │
└────┬─────┴──────┬───────┴───────────┬───────────────┘
     │            │                   │
     ▼            ▼                   ▼
┌─────────────────────────────────────────────────────┐
│              AgentMonad (Free Monad)                 │
│  Pure(value) | Perform { action, continuation }      │
└──────────────────────┬──────────────────────────────┘
                       │ interpreted by
                       ▼
┌─────────────────────────────────────────────────────┐
│              AgentContext (Interpreter)               │
│                                                      │
│  Action::ModelInference  → Rig LLM Provider          │
│  Action::ExecuteCode     → PyO3 Sandbox REPL         │
│  Action::ApplyPatch      → Unified Diff Engine       │
│  Action::SpawnSubAgent   → Child Context             │
│  Action::CompactContext  → Normalize + Compress      │
│  Action::Log             → Evidence + OTEL           │
└──────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Rust 1.75+ (`rustup update`)
- Python 3.10+ (for PyO3 sandbox)
- Optional: [Restate CLI](https://docs.restate.dev/develop/local_dev#running-restate-server--cli-locally) for durable mode

### Environment Variables

| Variable | When Needed | Example |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | Using OpenRouter | `export OPENROUTER_API_KEY=sk-...` |
| `OPENAI_API_KEY` | Using OpenAI | `export OPENAI_API_KEY=sk-...` |
| `RUST_LOG` | Debug tracing | `export RUST_LOG=rig_rlm=debug` |

### Build

```bash
cargo build --release
```

## Usage

### 1. CLI — Direct Mode

```bash
# Single task
cargo run --release -- run "Write fibonacci in Python" \
  --model gpt-4o --executor pyo3 --max-turns 25 --db agent.db

# DSPy-rs GEPA optimization
cargo run --release -- optimize --trainset examples.json --iterations 10

# ARC-AGI benchmark
cargo run --release -- arc-bench --dataset ./arc-agi-2/evaluation --model gpt-4o

# E2E smoke test
cargo run --release -- e2e-test --executor pyo3 --db test.db
```

### 2. Restate — Durable Mode

The agent runs as a Restate workflow with journaled execution, crash recovery, and lifecycle hooks.

```bash
# Terminal 1: Start Restate
restate-server --dev

# Terminal 2: Start the agent workflow server
cargo run --bin restate-server

# Terminal 3: Register + invoke
curl -X POST http://localhost:9070/deployments \
  -H 'Content-Type: application/json' \
  -d '{"uri": "http://localhost:9091"}'

curl -X POST http://localhost:8080/AgentWorkflow/task-1/run \
  -H 'Content-Type: application/json' \
  -d '{"task": "Write a sorting algorithm in Python"}'

# Check status
curl http://localhost:8080/AgentWorkflow/task-1/status
```

**Response:**
```json
{"turn": 5, "phase": "completed", "total_tokens": 2341, "output": "...", "error": null}
```

### 3. MCP Server — Claude Desktop / VS Code

The agent exposes 4 tools via the Model Context Protocol:

| Tool | Description |
|------|-------------|
| `run_task` | Full agent workflow via Restate |
| `execute_python` | Policy-checked Python execution |
| `apply_patch` | Unified diff file editing |
| `check_policy` | Command safety evaluation |

```bash
# Start MCP server (stdio)
cargo run --bin mcp-server
```

**Claude Desktop config** (`~/.config/claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "rig-rlm": {
      "command": "/path/to/target/release/mcp-server"
    }
  }
}
```

## Key Features

| Feature | Module | Description |
|---------|--------|-------------|
| Monadic composition | `monad/` | Free monad for composable, testable agent logic |
| Durable execution | `monad/restate.rs` | Crash-recoverable workflows via Restate |
| Execution policy | `exec_policy.rs` | Configurable allow/deny for commands |
| Apply-patch | `apply_patch.rs` | Unified diff editing (like Codex) |
| Context normalization | `monad/normalize.rs` | Role alternation, output truncation |
| Token tracking | `monad/history.rs` | Budget tracking with 85% threshold |
| Cost tracking | `monad/cost.rs` | Per-model USD cost with budget limits |
| Lifecycle hooks | `monad/hooks.rs` | Pluggable before/after LLM, exec, session |
| Multi-agent | `monad/orchestrator.rs` | Sub-agent spawning with capability scoping |
| Evidence trail | `monad/evidence.rs` | Auto-recorded proof of work |
| OTEL + LangFuse | `monad/otel.rs` | Distributed tracing with enrichment |

## Development

```bash
# Check (zero warnings)
cargo check

# Test (221 tests)
cargo test --lib

# Lint
cargo clippy --all-targets -- -D warnings

# Format
cargo fmt --all
```

## License

See [LICENSE](LICENSE).
