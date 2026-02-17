# rig-rlm

Monadic AI agent with DSPy-rs prompt optimization, Turso persistence, and sandboxed Python execution.

**Stack:** Rig (LLM) · DSPy-rs (optimization) · Turso (persistence) · PyO3/Microsandbox (execution)

### Environment Variables

Variable	When Needed	Example
OPENAI_API_KEY	Using OpenAI models	export OPENAI_API_KEY=sk-...
RUST_LOG	Debug tracing	export RUST_LOG=rig_rlm=debug

## Quick Start

```bash
cargo build --release
```

## Commands

### `run` — Single task

```bash
cargo run --release -- run "Write a function to compute fibonacci" \
  --model gpt-4o \
  --executor pyo3 \
  --max-turns 25 \
  --db agent.db
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gpt-4o` | LLM model name |
| `--executor` | `pyo3` | `pyo3` (in-process) or `microsandbox` (isolated VM) |
| `--max-turns` | `25` | Max generate→execute→feedback cycles |
| `--db` | `agent.db` | Turso database path |

---

### `optimize` — DSPy-rs GEPA prompt optimization

```bash
cargo run --release -- optimize \
  --trainset examples.json \
  --iterations 10 \
  --db optimize.db
```

| Flag | Default | Description |
|------|---------|-------------|
| `--trainset` | *(required)* | Path to training examples JSON |
| `--iterations` | `10` | Number of GEPA iterations |
| `--db` | `optimize.db` | Turso database path |

---

### `arc-bench` — ARC-AGI benchmark

```bash
# Baseline — direct evaluation
cargo run --release -- arc-bench \
  --dataset ./arc-agi-2/evaluation \
  --model gpt-4o

# Optimized — GEPA trains on first N tasks, evaluates on the rest
cargo run --release -- arc-bench \
  --dataset ./arc-agi-2 \
  --model gpt-4o \
  --optimize \
  --train-split 50 \
  --iterations 10

# Quick test — just 1 task
cargo run --release -- arc-bench \
  --dataset ./arc-test \
  --max-tasks 1 \
  --model gpt-4o
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `./arc-agi-2/evaluation` | Directory of ARC JSON task files |
| `--model` | `gpt-4o` | LLM model |
| `--optimize` | `false` | Run GEPA optimization before evaluation |
| `--train-split` | `50` | Tasks for training (rest = eval). Only with `--optimize` |
| `--iterations` | `10` | GEPA iterations. Only with `--optimize` |
| `--max-tasks` | `0` (all) | Limit number of tasks to evaluate |

---

### `e2e-test` — End-to-end smoke test

```bash
cargo run --release -- e2e-test --executor pyo3 --db test.db
```

### `summary` — Session summary from Turso

```bash
cargo run --release -- summary turso.db --session latest
```

---

## Toy ARC Problem

Create a test task and run it:

```bash
mkdir -p arc-test

cat > arc-test/invert_fill.json << 'EOF'
{
  "train": [
    {
      "input":  [[0,0,0],[0,1,0],[0,0,0]],
      "output": [[1,1,1],[1,0,1],[1,1,1]]
    },
    {
      "input":  [[0,0,0,0],[0,0,2,0],[0,0,0,0],[0,0,0,0]],
      "output": [[2,2,2,2],[2,2,0,2],[2,2,2,2],[2,2,2,2]]
    }
  ],
  "test": [
    {
      "input":  [[0,0,0],[0,3,0],[0,0,0]],
      "output": [[3,3,3],[3,0,3],[3,3,3]]
    }
  ]
}
EOF

cargo run --release -- arc-bench --dataset ./arc-test --max-tasks 1 --model gpt-4o
```

The pattern: find the non-zero cell, flood-fill the grid with that color, set the original cell to 0.

## Architecture

```
User Task
  → AgentMonad (Pure | Perform { action, next })
    → AgentContext::run() interpreter loop
      → Action::ModelInference  → Rig LLM provider
      → Action::ExecuteCode     → PyO3/Microsandbox REPL
      → Action::SpawnSubAgent   → child context (capability-restricted)
    → SUBMIT() terminates loop with structured result
  → DSPy-rs GEPA optimizes the instruction via FeedbackEvaluator
  → Turso persists sessions, turns, optimization history
```
