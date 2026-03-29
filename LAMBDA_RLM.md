# λ-RLM: Recursive Long-Context Map-Reduce Engine

**Paper**: "The Y-Combinator for LLMs: Solving Long-Context Rot with λ-Calculus" — Roy et al., arXiv:2603.20105v1 (Mar 2026)

**Key results**: +21.9pp accuracy over standard RLM, 3.3–4.1× latency reduction, wins 29/36 model-task comparisons.

---

## Architecture Overview

```
┌───────────── Level 2: HyperAgent (Metacognitive) ──────────────┐
│  HyperRubricGenerator: evolves its own rubric generation prompt │
│  HyperCostModel:       evolves planner params (ρ, A₀, A⊕)     │
│  HyperMutator:         adapts mutation rate via 1/5th rule      │
├────────────────────────────────────────────────────────────────┤
│  Level 1: Self-Improvement (GEPA + DR-Tulu)                    │
│                                                                 │
│  ┌────────────┐    ┌──────────────────────┐    ┌────────────┐  │
│  │ Trajectory │    │ MorphismPopulation   │    │  Rubric    │  │
│  │ Store      │◄──►│ (ε-greedy selection) │    │  Buffer    │  │
│  └─────┬──────┘    └──────────┬───────────┘    │  (DR-Tulu) │  │
│        └──────────┬───────────┘                └─────┬──────┘  │
│                   ▼                                  │         │
├───────────── Level 0: Task Execution ────────────────┤─────────┤
│                                                      │         │
│  ┌─────────────────────────────────────────────┐     │         │
│  │         YonedaContext — y(P)                 │     │         │
│  │  probe(q): y(P)(q)                          │     │         │
│  │  fmap(f, q): y(P)(f(q))                     │     │         │
│  │  probe_typed(q, lmap, rmap): Q → R          │     │         │
│  └──────────────────┬──────────────────────┘     │         │
│                     ▼                            │         │
│  ┌──────────────────────────────┐                │         │
│  │   lambda_rlm() — 5 Phases   │    ◄── scored by┘         │
│  │  1. Preview (peek 500)      │                            │
│  │  2. Task detection (1 LLM)  │                            │
│  │  3. Planning (0 LLM, pure)  │                            │
│  │  4. Cost estimation         │                            │
│  │  5. Execute Φ(P)            │                            │
│  └───────────┬─────────────────┘                            │
│              ▼                                               │
│  ┌───────────────────────────────┐                           │
│  │   LambdaExecutor — Φ(P)      │                           │
│  │  if |P| ≤ τ* → leaf M(P)    │                           │
│  │  else: SPLIT → MAP → REDUCE │                           │
│  └───────────────────────────────┘                           │
└──────────────────────────────────────────────────────────────┘
```

## Quick Start

### Environment Setup

```bash
# .env (project root)
OPENAI_API_KEY=sk-or-v1-...           # OpenRouter API key
OPENAI_BASE_URL=https://openrouter.ai/api/v1  # (auto-detected for sk-or-* keys)
RIG_RLM_MODEL=arcee-ai/trinity-large-preview:free  # Default model
```

For high-volume testing, switch to a paid model:
```bash
RIG_RLM_MODEL=google/gemini-2.5-flash-8b
```

### Build

```bash
# Full build
cargo build

# Check only (faster)
cargo check

# With Python LD_LIBRARY_PATH (if not using Nix flake)
LD_LIBRARY_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH \
  cargo build
```

## Test Commands

### Unit Tests (no LLM, instant)

```bash
# All lambda module tests
cargo test --lib lambda:: -- --nocapture

# Specific test suites
cargo test --lib lambda::combinators       # SPLIT, REDUCE, PEEK, FILTER
cargo test --lib lambda::planner           # Analytical k*, accuracy loop
cargo test --lib lambda::profunctor        # Profunctor dimap
cargo test --lib lambda::yoneda            # QueryMorphism category laws
cargo test --lib lambda::adaptive_yoneda   # Trajectory store, morphism population
cargo test --lib lambda::rubric            # Rubric buffer lifecycle, LLM judge parsing
cargo test --lib lambda::executor          # Relevance checker
```

### Live Tests (require LLM credits)

> **Note**: Free-tier OpenRouter allows 50 requests/day for `trinity-large-preview:free`.
> Each test uses ~3–15 LLM calls. Budget accordingly.

```bash
# Run all live tests
cargo test lambda_live -- --ignored --nocapture

# Individual live tests:

# 1. λ-RLM on a real source file (reads executor.rs, decomposes, analyzes)
cargo test live_lambda_rlm_on_real_source_file -- --ignored --nocapture

# 2. Yoneda representable functor — probe, fmap, morphism composition
cargo test live_yoneda_representable_functor -- --ignored --nocapture

# 3. Profunctor typed pipeline — AnalysisRequest → AnalysisReport
cargo test live_profunctor_typed_pipeline -- --ignored --nocapture

# 4. Yoneda equivalence — full faithfulness check (P₁ ≅? P₂)
cargo test live_yoneda_equivalence_check -- --ignored --nocapture

# 5. Adaptive Yoneda — self-learning loop with GEPA trajectory evolution
cargo test live_adaptive_yoneda_self_learning -- --ignored --nocapture

# 6. Open coding — Trinity reads files, browses, writes code
cargo test live_open_coding_read_files_and_write_code -- --ignored --nocapture

# 7. HITL iteration — model asks questions, user responds, model iterates
cargo test live_open_coding_with_hitl_iteration -- --ignored --nocapture

# 8. Full loop — read → analyze → code → "make it dynamic" → iterate
cargo test live_read_analyze_code_then_iterate -- --ignored --nocapture

# 9. DR-Tulu evolving rubric — LLM-as-judge with adaptive criteria lifecycle
cargo test live_evolving_rubric_reward -- --ignored --nocapture

# 10. HyperAgent metacognitive — simulated system prompt evolution
cargo test live_hyperagent_metacognitive_pipeline -- --ignored --nocapture

# 11-15. Isolated Metacognitive Modules (Testing individual Hyper* behaviors)
cargo test live_hyper_prompt_evolver -- --ignored --nocapture
cargo test live_hyper_llm_mutator -- --ignored --nocapture
cargo test live_hyper_router -- --ignored --nocapture
cargo test live_hyper_fiduciary -- --ignored --nocapture
cargo test live_hyper_exec_policy -- --ignored --nocapture

# 16. TRUE E2E Metacognitive Loop (ALL 5 modules, real Trinity signals, no hardcoding)
# Runs the full system: real probes → real LLM → dynamic rubric generation → 
# auto-adjusted weights → router load balancing → prompt evolution.
cargo test live_e2e_all_hyper -- --ignored --nocapture
```

### GEPA Optimization Daemon

```bash
# Run the GEPA evolutionary optimizer
cargo run --bin optimize_rlm
```

## Module Guide

### `src/lambda/` — The λ-RLM Engine

| File | Purpose |
|------|---------|
| `mod.rs` | Top-level entry: `lambda_rlm()`, `lambda_rlm_typed()`, config, re-exports |
| `combinators.rs` | Deterministic combinators: `split`, `reduce`, `peek`, `filter`, `concat`, `cross` |
| `planner.rs` | Analytical planner: `k* = ⌈√(n·c_in/c_⊕)⌉`, accuracy constraint loop, `CostParams` |
| `executor.rs` | Recursive executor `Φ(P)` with parallel MAP via `join_all` + `AtomicMetrics` |
| `templates.rs` | Leaf/synthesis prompt templates per task type |
| `yoneda.rs` | Yoneda Lemma: `YonedaContext`, `QueryMorphism`, `yoneda_equivalence()`, `check_naturality()` |
| `profunctor.rs` | Profunctor optics: `TypedPipeline<C, D>`, `AsyncProfunctor`, `dimap_async` |
| `adaptive_yoneda.rs` | Self-learning: `AdaptiveYoneda`, `TrajectoryStore`, `MorphismPopulation`, GEPA co-evolution, **`HyperCostModel`**, **`HyperMutator`** |
| `rubric.rs` | Evolving rubric reward: `RubricItem`, `RubricBuffer`, LLM-as-judge, adaptive generation (DR-Tulu), **`HyperRubricGenerator`** |
| `gepa_rlm.rs` | GEPA evaluator: `LambdaExecutorEvaluator` — uses `planner::plan()` for dynamic depth |
| `effects.rs` | Algebraic effects prototype for LLM interaction |
| `live_tests.rs` | All integration tests (pure + live LLM) |

### Key Types

```rust
// Top-level entry (untyped)
lambda_rlm(prompt, query, provider, config) -> Result<String>

// Top-level entry (typed via Profunctor)
lambda_rlm_typed(input, query, provider, config, lmap, rmap) -> Result<D>

// Yoneda lazy context
let y = YonedaContext::lift(document, provider, config);
let result = y.probe("summarize").await?;
let result = y.fmap(&morphism, "summarize").await?;

// Typed pipeline via Profunctor
let pipeline = TypedPipeline::new(provider, config, query, lmap, rmap);
let report: Report = pipeline.execute(&request).await?;

// Self-learning adaptive probe (Level 1)
let mut adaptive = AdaptiveYoneda::with_rubrics(document, provider, config);
let (result, score, per_rubric) = adaptive.adaptive_probe_with_rubrics("query").await?;

// Full HyperAgent (Level 0 + 1 + 2)
let mut agent = AdaptiveYoneda::hyper(document, provider, config);
let (result, score, per_rubric) = agent.adaptive_probe_with_rubrics("query").await?;
println!("{}", agent.hyper_summary());
```

## HyperAgent — Level 2 Metacognitive Self-Modification

The HyperAgent layer transforms λ-RLM from a DGM (Dynamic Generative Model) into a **DGM-H** — a system that can **improve its own improvement process**.

### The Three Hyper Components

| Component | File | What It Evolves | Trigger Signal |
|-----------|------|----------------|----------------|
| **HyperRubricGenerator** | `rubric.rs` | The rubric generation prompt | `discriminative_ratio < 0.50` |
| **HyperCostModel** | `adaptive_yoneda.rs` | Planner params: ρ, A₀, A⊕ | GEPA fitness scores |
| **HyperMutator** | `adaptive_yoneda.rs` | Mutation rate (0.02–0.80) | 1/5th success rule |

### HyperRubricGenerator — Self-Modifying Rubric Prompt

When the rubric buffer's `discriminative_ratio` drops below 0.50, the system:

1. Collects performance metrics: `avg_std`, `zero_rubrics_ratio`, `discriminative_ratio`
2. Gathers examples of retired (non-discriminative) rubrics
3. Calls the LLM with `HYPER_EVOLVE_PROMPT` + current prompt + metrics
4. The LLM rewrites the generation prompt to produce more discriminative rubrics
5. The new prompt is installed as `v{n+1}` and all future rubric generation uses it

```
v0 (hardcoded prompt, 935 chars)
  → generates rubrics → disc_ratio=0.25 → BAD
  → metacognitive trigger fires
  → LLM rewrites prompt
v1 (LLM-evolved prompt, 1234 chars)
  → generates rubrics → disc_ratio=0.67 → GOOD
```

Verified in the live test (`live_hyperagent`):
```
Scores: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.12, 0.20, 0.10]
                                                 ↑ v0→v1 evolution here
```

### HyperCostModel — Evolving Planner Parameters

Makes the planner's cost parameters part of the GEPA candidate space:

| Parameter | Range | Default | Purpose |
|-----------|-------|---------|---------|
| `ρ` | [0.5, 0.99] | 0.85 | Token retention per decomposition |
| `A₀` | [0.7, 1.0] | 0.95 | Base accuracy of a single M call |
| `A⊕` | [0.8, 1.0] | 0.98 | Accuracy of REDUCE composition |

### HyperMutator — Adaptive Mutation Rate

Uses the **1/5th success rule** from Evolution Strategy theory:

```
if success_rate > 0.20: rate *= 1.2   (more exploration)
if success_rate < 0.20: rate *= 0.83  (more exploitation)
rate ∈ [0.02, 0.80]
```

### Usage

```rust
// Create full HyperAgent (all three layers)
let mut agent = AdaptiveYoneda::hyper(document, provider, config);

// Customize thresholds
if let Some(ref mut hyper_gen) = agent.hyper_rubric_gen {
    hyper_gen.discriminative_threshold = 0.40;
    hyper_gen.min_generations_before_evolve = 3;
}

// Run probes — metacognitive layers activate automatically
for query in queries {
    let (result, score, per_rubric) = agent.adaptive_probe_with_rubrics(query).await?;
}

// Inspect metacognitive state
println!("{}", agent.hyper_summary());
```

## Category Theory Guide

### Yoneda Lemma — Representable Functors

```
Nat(Hom(A, −), F) ≅ F(A)
```

- **Object P** = a massive document (held lazily, never collapsed to an embedding)
- **Functor y(P)** = `YonedaContext` — maps queries to results via λ-RLM
- `probe(q)` = evaluate `y(P)` on a query object → `y(P)(q)`
- `fmap(f, q)` = evaluate `y(P)` on a query morphism → `y(P)(f(q))`
- **Full Faithfulness**: `yoneda_equivalence(P₁, P₂, queries, sim, threshold)`
- **Naturality**: `check_naturality(P, q, f, transform, sim)`

### QueryMorphism — Category of Queries

```rust
let id = QueryMorphism::identity();       // id(q) = q
let refine = QueryMorphism::new("focus_science",
    |q| format!("{} Focus on science.", q));
let composed = g.compose(f);              // (g ∘ f)(q) = g(f(q))
let result = yoneda.fmap(&composed, base_query).await?;
```

### Profunctor Optics — Typed I/O

```
lmap          Φ (Hylomorphism)        rmap
C ────────→ String ──────────→ String ────────→ D
  contra-                                covariant
  variant
```

### Left Kan Extension — Extrapolation

```
    Observed ──J──→ AllQueries
       │               │
       F               Lan_J F
       │               │
       ↓               ↓
      Set             Set
```

### Evolving Rubric Reward — LLM-as-Judge (DR-Tulu)

Inspired by DR-Tulu ([arXiv:2511.19399](https://arxiv.org/abs/2511.19399)), the rubric system replaces hardcoded scoring with an **evolving, LLM-judged evaluation**.

```
┌──────────────────── Rubric Buffer ────────────────────┐
│                                                        │
│  Persistent Rubrics ─── always scored ──→  Weighted    │
│  (Factual Recall, Answer Relevance,        Reward      │
│   Completeness)                               │        │
│                                               │        │
│  Active Adaptive  ──── scored + filtered ──→  ↑        │
│  (LLM-generated)                                       │
│                     filter_and_retire()                 │
│  Inactive Adaptive ── zero-std retired ──→  ∅          │
│                                                        │
│           ↓ (metrics fed to Level 2)                   │
│  HyperRubricGenerator ── rewrites generation prompt    │
│  when disc_ratio < 0.50                                │
└────────────────────────────────────────────────────────┘
```

| Layer | Description | Action |
|-------|-------------|--------|
| **Persistent** | Ground truth (always scored) | Never retired |
| **Active** | LLM-discovered criteria | Scored, tracked for std |
| **Inactive** | Non-discriminative (std ≈ 0) | Retired, no longer scored |

**Automatic evolution loop** (runs inside `adaptive_probe_with_rubrics`):

1. **Score**: LLM judge evaluates response against ALL active rubrics → `{"score": 0-2}`
2. **Record**: Per-rubric scores tracked for std computation
3. **Z-Score**: Normalized advantage `(score - mean) / std * weight`
4. **Metrics**: Emit `RubricMetrics` (avg_mean, avg_std, zero_ratio, discriminative_ratio)
5. **Generate** (every N probes): LLM compares recent responses → new rubrics
6. **Retire** (every 3 probes): Zero-std rubrics → inactive
7. **Meta-Evolve** (disc_ratio < 0.50): HyperRubricGenerator rewrites its own prompt
8. **Save**: Auto-save to disk (if `rubric_save_path` set)
9. **Cap**: Active adaptive rubrics capped at `max_active` (default 5)

## Execution Parameters

### `LambdaConfig`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `context_window` | 32,000 | Model context window K (tokens) |
| `accuracy_target` | 0.80 | Target accuracy α ∈ (0, 1] |
| `cost_params.c_invoke` | 1.0 | Cost per leaf M invocation |
| `cost_params.c_compose` | 0.0 | Cost per REDUCE composition |

### Analytical Planning (Theorem 4)

```
k* = ⌈√(n · c_invoke / c_compose)⌉     (when c_compose > 0)
k* = ⌈n / K⌉                            (when c_compose ≈ 0, symbolic)
```

### Task Types

| Type | Compose (⊕) | Pre-filter | Neural Compose |
|------|-------------|------------|----------------|
| `Search` | Concatenate matches | ✅ Yes | No |
| `MultiHop` | Chain reasoning | ✅ Yes | Yes |
| `Summarise` | Merge summaries | No | Yes |
| `Aggregate` | Combine counts | No | No |
| `Pairwise` | Bracket tournament | No | No |

## Troubleshooting

### Rate Limit (429)

**Fix**: Add $10 credits to OpenRouter, or switch to a paid model:
```bash
RIG_RLM_MODEL=google/gemini-2.5-flash-8b
```

### Python LD_LIBRARY_PATH

```bash
export LD_LIBRARY_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH
```

Or use the Nix flake which sets this automatically.

## Cross-References

| Document | What It Covers |
|----------|---------------|
| [`README.md`](README.md) | Project overview, entry points, usage, deployment |
| [`LAMBDA_RLM.md`](LAMBDA_RLM.md) | This file — λ-RLM engine, Yoneda, HyperAgent |
| [`src/gnn/hehrgnn/README.md`](src/gnn/hehrgnn/README.md) | HEHRGNN platform, GNN models, MSA, fiduciary engine |
