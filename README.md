# rig-rlm

Monadic AI agent with durable execution, MCP server, A2A protocol support, sandboxed Python, DSPy-rs prompt optimization, and self-learning via Category Theory + HyperAgents.

**Stack:** [Rig](https://rig.rs) (LLM) · [Restate](https://restate.dev) (durable execution) · [rmcp](https://crates.io/crates/rmcp) (MCP server) · [A2A](https://a2aproject.github.io/A2A/) (agent-to-agent) · [agentgateway](https://github.com/agentgateway/agentgateway) (proxy) · [PyO3](https://pyo3.rs) (Python sandbox) · [DSPy-rs](https://crates.io/crates/dspy) (optimization) · [Turso](https://turso.tech) (persistence) · [Burn](https://burn.dev) (GNN)

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Entry Points                            │
├──────────┬──────────────┬───────────────────┬───────────────────┤
│  CLI     │  MCP Server  │  A2A Server       │  Restate Workflow │
│ (main)   │  (stdio)     │  (HTTP, JSON-RPC) │  (HTTP, durable)  │
└────┬─────┴──────┬───────┴────────┬──────────┴───────────────────┘
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

## Module Map

The codebase is organized into several major subsystems. Each subsystem has its own detailed documentation.

```
rig-rlm/
├── src/
│   ├── monad/              # Free monad agent core
│   │   ├── context.rs      #   AgentContext interpreter
│   │   ├── monad.rs        #   Free monad definition
│   │   ├── provider.rs     #   LLM provider (OpenRouter, OpenAI)
│   │   ├── interaction.rs  #   agent_task_full() entry point
│   │   └── ...             #   hooks, evidence, cost, normalize, otel
│   │
│   ├── lambda/             # λ-RLM + Yoneda + HyperAgent (→ LAMBDA_RLM.md)
│   │   ├── executor.rs     #   Recursive Φ(P) = SPLIT → MAP → REDUCE
│   │   ├── planner.rs      #   Analytical k* via Theorem 4
│   │   ├── yoneda.rs       #   YonedaContext — lazy representable functor
│   │   ├── profunctor.rs   #   TypedPipeline — type-safe C → D via dimap
│   │   ├── adaptive_yoneda.rs # Self-learning + HyperCostModel + HyperMutator
│   │   ├── rubric.rs       #   DR-Tulu rubrics + HyperRubricGenerator
│   │   └── live_tests.rs   #   End-to-end tests with Trinity LLM
│   │
│   ├── gnn/hehrgnn/        # GNN platform + MSA (→ src/gnn/hehrgnn/README.md)
│   │   ├── model/          #   GraphSAGE, RGCN/mHC, GAT, GPS Transformer
│   │   ├── model/msa/      #   Memory Sparse Attention (MSA)
│   │   ├── eval/           #   Fiduciary engine, probabilistic circuits
│   │   └── optimizer/      #   GEPA self-improvement
│   │
│   ├── nuggets/            # Holographic memory (HRR vectors + disk-backed facts)
│   │   ├── core.rs         #   AVX2-optimized HRR vector operations
│   │   ├── shelf.rs        #   Nugget persistence and shelf management
│   │   └── memory.rs       #   Nugget memory engine
│   │
│   ├── channels/           # Event channels (broadcast, WebSocket, Iggy)
│   ├── sandbox/            # PyO3 Python sandbox
│   └── ...                 # exec_policy, apply_patch, safety, etc.
│
├── LAMBDA_RLM.md           # Full developer guide for λ-RLM + HyperAgent
└── README.md               # This file
```

## Self-Learning Stack — The Three Levels

The system implements a **3-level self-improvement hierarchy** inspired by the [HyperAgents](https://arxiv.org/abs/2502.xxxxx) framework:

```
┌─────────────────── Level 2: Metacognitive (HyperAgent) ──────────┐
│  HyperRubricGenerator: evolves its own rubric generation prompt   │
│  HyperCostModel:       evolves planner params (ρ, A₀, A⊕) via GEPA│
│  HyperMutator:         adapts mutation rate via 1/5th success rule│
├──────────────────────────────────────────────────────────────────┤
│  Level 1: Self-Improvement (DR-Tulu + GEPA)                      │
│  RubricBuffer: persistent + adaptive + retired rubric lifecycle   │
│  MorphismPopulation: ε-greedy selection of query morphisms        │
│  TrajectoryStore: past (query, morphism, score) for co-evolution  │
├──────────────────────────────────────────────────────────────────┤
│  Level 0: Task Execution (λ-RLM)                                 │
│  LambdaExecutor: recursive SPLIT → MAP → REDUCE                  │
│  YonedaContext: lazy representable functor for massive documents   │
│  TypedPipeline: Profunctor optics for type-safe C → D             │
└──────────────────────────────────────────────────────────────────┘
```

**How it works:**

| Level | What Improves | Signal |
|-------|--------------|--------|
| **0** | Task execution quality | LLM response to query |
| **1** | How we evaluate + guide the task agent | Rubric scores, morphism fitness |
| **2** | How we generate rubrics + tune mutation/planner | Discriminative ratio, success rate |

```bash
# Standard agent (Level 0 + 1):
AdaptiveYoneda::with_rubrics(doc, provider, config)

# Full HyperAgent (Level 0 + 1 + 2):
AdaptiveYoneda::hyper(doc, provider, config)
```

See [`LAMBDA_RLM.md`](LAMBDA_RLM.md) for the complete developer guide.

## Quick Start

### Prerequisites

- Rust 1.75+ (`rustup update`)
- Python 3.10+ (for PyO3 sandbox)
- Optional: [Restate CLI](https://docs.restate.dev/develop/local_dev#running-restate-server--cli-locally) for durable mode
- Optional: [agentgateway](https://github.com/agentgateway/agentgateway) for A2A/MCP proxying

> **Tip:** Use the included **Nix flake** to get all dependencies automatically — see [Development Environment](#development-environment) below.

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

### 4. A2A Server — Agent-to-Agent Protocol

The agent exposes itself as an [A2A](https://a2aproject.github.io/A2A/)-compatible agent over HTTP, discoverable via `/.well-known/agent.json`.

| Method | Description |
|--------|-------------|
| `GET /.well-known/agent.json` | Agent Card (name, skills, capabilities) |
| `POST /` — `message/send` | Synchronous task execution (JSON-RPC response) |
| `POST /` — `message/stream` | Streaming execution via SSE (`text/event-stream`) |

#### Start the server

```bash
# With Nix flake (recommended — sets LD_LIBRARY_PATH automatically)
cargo run --bin a2a-server -- --port 9999

# Without Nix flake (set LD_LIBRARY_PATH for PyO3 manually)
LD_LIBRARY_PATH="/path/to/python/lib:$LD_LIBRARY_PATH" \
  cargo run --bin a2a-server -- --port 9999
```

The server auto-detects OpenRouter API keys (`sk-or-*`) from `OPENAI_API_KEY` in `.env` and sets the base URL accordingly. Default model: `arcee-ai/trinity-large-preview:free`.

Logs are written to `a2a-server.log` in the working directory.

#### Test: Agent Card Discovery

```bash
curl -s http://localhost:9999/.well-known/agent.json | python3 -m json.tool
```

**Expected:** JSON with agent name, description, skills, and supported protocols.

#### Test: `message/send` — Simple Math

```bash
curl -s -m 30 -X POST http://localhost:9999/ \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "What is 2+2?"}]
      },
      "metadata": {}
    }
  }'
```

#### Test: `message/send` — Coding Task (Python execution via PyO3)

```bash
curl -s -m 120 -X POST http://localhost:9999/ \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Write a Python function that implements merge sort. Test it with [38, 27, 43, 3, 9, 82, 10] and print the sorted result."}]
      },
      "metadata": {}
    }
  }'
```

> **Note:** Complex coding tasks may take 30–70s depending on the LLM. Use `-m 120` for generous timeouts.

### 5. Agent Gateway — Proxied A2A & MCP

[agentgateway](https://github.com/agentgateway/agentgateway) is a Linux Foundation data plane that proxies A2A and MCP traffic with security, observability, and governance.

Config files: [`agentgateway-a2a.yaml`](agentgateway-a2a.yaml) · [`agentgateway-mcp.yaml`](agentgateway-mcp.yaml) · [`agentgateway-mcp-multiplex.yaml`](agentgateway-mcp-multiplex.yaml)

```bash
# A2A proxying (direct → port 9999, gateway → port 3001)
cargo run --bin a2a-server -- --port 9999
cd ~/dev-stuff/agentgateway && cargo run -- -f ~/dev-stuff/rig-rlm/agentgateway-a2a.yaml

# MCP proxying (gateway spawns mcp-server automatically)
cd ~/dev-stuff/agentgateway && cargo run -- -f ~/dev-stuff/rig-rlm/agentgateway-mcp.yaml
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
| **λ-RLM engine** | `lambda/executor.rs` | Recursive long-context Map-Reduce ([arXiv:2603.20105](https://arxiv.org/abs/2603.20105)) |
| **Yoneda context** | `lambda/yoneda.rs` | Lazy representable functor for massive documents |
| **Profunctor optics** | `lambda/profunctor.rs` | Type-safe `C → D` pipeline via `dimap` |
| **Adaptive GEPA** | `lambda/adaptive_yoneda.rs` | Self-learning morphism evolution from trajectories |
| **Evolving rubrics** | `lambda/rubric.rs` | DR-Tulu-style LLM-as-judge with adaptive criteria ([arXiv:2511.19399](https://arxiv.org/abs/2511.19399)) |
| **HyperAgent** | `lambda/rubric.rs`, `adaptive_yoneda.rs` | Metacognitive self-modification (HyperRubricGenerator, HyperCostModel, HyperMutator) |
| **HEHRGNN** | `gnn/hehrgnn/` | Heterogeneous GNN platform with 4 models + GEPA auto-tune |
| **MSA** | `gnn/hehrgnn/src/model/msa/` | Memory Sparse Attention — trainable long-range attention |
| **Nuggets** | `nuggets/` | Holographic memory with HRR vectors + disk-backed persistence |

### λ-RLM Engine — Recursive Long-Context Processing

The `lambda/` module implements the λ-RLM framework from [arXiv:2603.20105](https://arxiv.org/abs/2603.20105), enabling any LLM to process documents that exceed its context window via recursive Map-Reduce decomposition.

**Key capabilities:**
- **Analytical planning** — optimal `k*` via closed-form formula (Theorem 4)
- **Parallel execution** — `MAP` branches run concurrently via `join_all`
- **Category theory** — Yoneda representable functors, Profunctor optics, query morphisms with verified laws
- **Self-learning** — GEPA co-evolves execution parameters + query morphisms from past trajectories
- **Evolving rubrics** — DR-Tulu-inspired LLM-as-judge scoring with adaptive rubric generation and zero-std retirement
- **HyperAgent** — Metacognitive self-modification: the rubric generator can rewrite its own prompt, mutation rates self-adapt, planner params co-evolve

```bash
# Quick test (no LLM required)
cargo test --lib lambda:: -- --nocapture

# Live test with Trinity model (requires OPENAI_API_KEY)
cargo test live_yoneda_representable_functor -- --ignored --nocapture

# Full DR-Tulu evolving rubric lifecycle test
cargo test live_evolving_rubric_reward -- --ignored --nocapture

# Full HyperAgent metacognitive pipeline test
cargo test live_hyperagent -- --ignored --nocapture
```

**Full developer guide**: [`LAMBDA_RLM.md`](LAMBDA_RLM.md)

### HEHRGNN — Graph Neural Network Platform

The `gnn/hehrgnn/` sub-crate provides a heterogeneous GNN platform for relational graph intelligence, with 4 GNN models (GraphSAGE, RGCN/mHC, GAT, GPS Transformer), a fiduciary engine (18 action types), probabilistic circuits for calibrated risk, and GEPA auto-tuning.

New: **Memory Sparse Attention (MSA)** in `model/msa/` provides trainable long-range attention with configurable routing, RoPE embeddings, and memory banks.

```bash
# Run all GNN tests
cargo test -p hehrgnn

# Run the full ensemble pipeline
cargo test -p hehrgnn --test ensemble_pipeline_test -- --nocapture
```

**Full developer guide**: [`src/gnn/hehrgnn/README.md`](src/gnn/hehrgnn/README.md)

## Development Environment

### Nix Flake (recommended)

The project includes a Nix flake that provides all dependencies (Rust, Python, OpenSSL, protobuf, Node.js, etc.) and also supports building [agentgateway](https://github.com/agentgateway/agentgateway).

```bash
# With direnv (auto-activates on cd)
direnv allow

# Without direnv
nix develop
```

The dev shell includes sccache, wild linker, and `LD_LIBRARY_PATH` for PyO3.

### Commands

```bash
# Check
cargo check

# Test (unit — no LLM)
cargo test --lib

# Test (live — requires OPENAI_API_KEY)
cargo test lambda_live -- --ignored --nocapture

# Lint
cargo clippy --all-targets -- -D warnings

# Format
cargo fmt --all

# Build agentgateway (from within the nix shell)
cd ~/dev-stuff/agentgateway && cargo build --release
```

## Cross-References

| Document | What It Covers |
|----------|---------------|
| [`README.md`](README.md) | This file — project overview, entry points, usage |
| [`LAMBDA_RLM.md`](LAMBDA_RLM.md) | λ-RLM engine, Yoneda, Profunctor, GEPA, DR-Tulu rubrics, HyperAgent |
| [`src/gnn/hehrgnn/README.md`](src/gnn/hehrgnn/README.md) | HEHRGNN platform, GNN models, MSA, fiduciary engine, tests |
| [`AGENTICA_IDEAS.md`](AGENTICA_IDEAS.md) | Research notes and future directions |

## License

See [LICENSE](LICENSE).
