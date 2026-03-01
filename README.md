# rig-rlm

Monadic AI agent with durable execution, MCP server, A2A protocol support, sandboxed Python, and DSPy-rs prompt optimization.

**Stack:** [Rig](https://rig.rs) (LLM) · [Restate](https://restate.dev) (durable execution) · [rmcp](https://crates.io/crates/rmcp) (MCP server) · [A2A](https://a2aproject.github.io/A2A/) (agent-to-agent) · [agentgateway](https://github.com/agentgateway/agentgateway) (proxy) · [PyO3](https://pyo3.rs) (Python sandbox) · [DSPy-rs](https://crates.io/crates/dspy) (optimization) · [Turso](https://turso.tech) (persistence)

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

**Expected:**
```json
{"jsonrpc":"2.0","id":1,"result":{"status":{"state":"completed","message":{"role":"agent","parts":[{"kind":"text","text":"4"}]}}}}
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

**Expected:** `"text":"The sorted result is [3, 9, 10, 27, 38, 43, 82]"`

> **Note:** Complex coding tasks may take 30–70s depending on the LLM. Use `-m 120` for generous timeouts.

#### Test: `message/stream` — SSE Streaming

```bash
curl -s -m 120 -N -X POST http://localhost:9999/ \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "message/stream",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Write a Python class for a binary search tree with insert and in-order traversal. Insert [50,30,70,20,40,60,80] and print the traversal."}]
      },
      "metadata": {}
    }
  }'
```

**Expected:** Two SSE events — `working` then `completed`:
```
data: {"jsonrpc":"2.0","id":3,"result":{"status":{"state":"working",...}}}

data: {"jsonrpc":"2.0","id":3,"result":{"status":{"state":"completed","message":{"role":"agent","parts":[{"kind":"text","text":"In-order traversal: [20, 30, 40, 50, 60, 70, 80]"}]}}}}
```

#### Test: Error Handling — Unknown Method

```bash
curl -s -X POST http://localhost:9999/ \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":99,"method":"unknown/method","params":{}}'
```

**Expected:** JSON-RPC error `-32601 Method not found`:
```json
{"jsonrpc":"2.0","id":99,"error":{"code":-32601,"message":"Method not found: unknown/method"}}
```

### 5. Agent Gateway — Proxied A2A & MCP

[agentgateway](https://github.com/agentgateway/agentgateway) is a Linux Foundation data plane that proxies A2A and MCP traffic with security, observability, and governance.

#### Prerequisites

1. **Clone agentgateway** (if not already done):
   ```bash
   cd ~/dev-stuff
   git clone https://github.com/agentgateway/agentgateway.git
   ```

2. **Build agentgateway** (requires protobuf — included in the Nix flake):
   ```bash
   cd ~/dev-stuff/agentgateway
   cargo build --release
   ```

3. **Set up `.env`** in the rig-rlm directory:
   ```bash
   # .env (rig-rlm root)
   OPENAI_API_KEY=sk-or-v1-...   # OpenRouter key (auto-detected by sk-or-* prefix)
   ```

#### A2A Proxying — Gateway fronts the A2A server

```bash
# Terminal 1: Start rig-rlm A2A server on port 9999
cargo run --bin a2a-server -- --port 9999

# Terminal 2: Start agentgateway (proxies :9999 → :3001)
cd ~/dev-stuff/agentgateway
cargo run -- -f ~/dev-stuff/rig-rlm/agentgateway-a2a.yaml
```

**Test through the gateway (port 3001):**

```bash
# Agent card via gateway
curl -s http://localhost:3001/.well-known/agent.json | python3 -m json.tool

# message/send via gateway
curl -s -m 120 -X POST http://localhost:3001/ \
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

# message/stream SSE via gateway
curl -s -m 120 -N -X POST http://localhost:3001/ \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "message/stream",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "What is 5+3?"}]
      },
      "metadata": {}
    }
  }'
```

#### MCP Proxying — Gateway spawns the MCP server via stdio

```bash
# Terminal 1: Start agentgateway (spawns mcp-server automatically)
cd ~/dev-stuff/agentgateway
cargo run -- -f ~/dev-stuff/rig-rlm/agentgateway-mcp.yaml

# Terminal 2: Connect MCP Inspector
npx @modelcontextprotocol/inspector
# → URL: http://localhost:3001/mcp
```

> **Note:** If not using the Nix flake, set `LD_LIBRARY_PATH` for PyO3 before starting either server:
> ```bash
> export LD_LIBRARY_PATH="/path/to/python3.14/lib:$LD_LIBRARY_PATH"
> ```

Config files: [`agentgateway-a2a.yaml`](agentgateway-a2a.yaml) · [`agentgateway-mcp.yaml`](agentgateway-mcp.yaml)

#### MCP Multiplexing — Multiple servers, one endpoint

Federate multiple MCP servers behind a single agentgateway endpoint. Tools from all servers are automatically namespaced and accessible at one URL.

A multiplexing config is included that combines:
- **rig-rlm** — `run_task`, `execute_python`, `apply_patch`, `check_policy`
- **@modelcontextprotocol/server-everything** — `echo`, `add`, `longRunningOperation`, etc.

```bash
# Terminal 1: Start agentgateway with multiplexed config
cd ~/dev-stuff/agentgateway
cargo run --release -- -f ~/dev-stuff/rig-rlm/agentgateway-mcp-multiplex.yaml

# Terminal 2: Connect MCP Inspector
npx @modelcontextprotocol/inspector
# → Transport: Streamable HTTP
# → URL: http://localhost:3001/mcp
# → Click "Connect", then "List Tools"
```

You should see tools from **both** servers in a single tool list, automatically prefixed with the server name to avoid naming conflicts.

Config file: [`agentgateway-mcp-multiplex.yaml`](agentgateway-mcp-multiplex.yaml)

#### Gateway Policies

All agentgateway configs include enterprise-grade policies:

| Policy | A2A | MCP | What it does |
|--------|-----|-----|-------------|
| **Timeouts** | 120s | 60s | Request + backend timeout for agent/tool calls |
| **Rate Limiting** | 10 req/min | 20 req/min | In-memory token-bucket rate limiter |
| **Header Modification** | ✅ | ✅ | Adds `x-gateway`, `x-powered-by` headers |
| **CORS** | ✅ | ✅ | Cross-origin access for Inspector/IDE |
| **A2A Telemetry** | ✅ | — | Agent card URL rewriting + A2A metrics |
| **MCP OAuth Auth** | — | 🔧 | OAuth token validation (commented out, ready to enable) |

**To enable MCP OAuth authentication**, uncomment the `mcp_authentication` block in `agentgateway-mcp-multiplex.yaml` and configure your OAuth provider (Auth0, Keycloak, etc.):

```yaml
mcp_authentication:
  issuer: https://your-idp.example.com
  audiences: [rig-rlm-mcp]
  jwks:
    remote: https://your-idp.example.com/.well-known/jwks.json
  mode: required
```

**OTEL Tracing** can be added as a frontend policy — the rig-rlm codebase already integrates LangFuse for LLM observability; gateway-level OTEL tracing adds spans for proxy/routing latency.

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
# Check (zero warnings)
cargo check

# Test
cargo test --lib

# Lint
cargo clippy --all-targets -- -D warnings

# Format
cargo fmt --all

# Build agentgateway (from within the nix shell)
cd ~/dev-stuff/agentgateway && cargo build --release
```

## License

See [LICENSE](LICENSE).
