# λ-RLM: Recursive Long-Context Map-Reduce Engine

**Paper**: "The Y-Combinator for LLMs: Solving Long-Context Rot with λ-Calculus" — Roy et al., arXiv:2603.20105v1 (Mar 2026)

**Key results**: +21.9pp accuracy over standard RLM, 3.3–4.1× latency reduction, wins 29/36 model-task comparisons.

---

## Architecture Overview

```
                    ┌─────────────────────────────────────────────┐
                    │           AdaptiveYoneda (Self-Learning)     │
                    │  ┌────────────┐    ┌──────────────────────┐ │
                    │  │ Trajectory │    │ MorphismPopulation   │ │
                    │  │ Store      │◄──►│ (ε-greedy selection) │ │
                    │  └─────┬──────┘    └──────────┬───────────┘ │
                    │        │                      │             │
                    │        └──────────┬───────────┘             │
                    │                   ▼                         │
┌──────────┐    ┌───┴───────────────────────────────────────┐     │
│  Caller  │    │         YonedaContext — y(P)                │     │
│          │───►│  probe(q): y(P)(q)                         │     │
│ Typed:   │    │  fmap(f, q): y(P)(f(q))                    │     │
│  C → D   │    │  probe_typed(q, lmap, rmap): Q → R         │     │
│ via      │    └──────────────────┬──────────────────────┘     │
│ Profunctor│                      │                          │
└──────────┘                      ▼                          │
                    ┌──────────────────────────────┐          │
                    │   lambda_rlm() — 5 Phases    │          │
                    │  1. Preview (peek 500 tokens) │          │
                    │  2. Task detection (1 LLM)    │          │
                    │  3. Planning (0 LLM, pure)    │          │
                    │  4. Cost estimation            │          │
                    │  5. Execute Φ(P)               │          │
                    └───────────┬──────────────────┘          │
                                │                             │
                    ┌───────────▼──────────────────┐          │
                    │   LambdaExecutor — Φ(P)      │          │
                    │                              │          │
                    │  if |P| ≤ τ* → leaf M(P)     │          │
                    │  else:                       │          │
                    │    SPLIT(P, k*)              │          │
                    │    FILTER (optional)         │          │
                    │    MAP(Φ) — parallel join_all │          │
                    │    REDUCE(⊕)                 │          │
                    └──────────────────────────────┘          │
                    │                                         │
                    │  GEPA co-evolves k*, τ*, morphisms ────►│
                    └─────────────────────────────────────────┘
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

# 9. Evolving rubric reward — LLM-as-judge scoring with adaptive criteria
cargo test live_evolving_rubric_reward -- --ignored --nocapture
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
| `adaptive_yoneda.rs` | Self-learning loop: `AdaptiveYoneda`, `TrajectoryStore`, `MorphismPopulation`, GEPA co-evolution |
| `rubric.rs` | Evolving rubric reward: `RubricItem`, `RubricBuffer`, LLM-as-judge scoring, adaptive rubric generation (DR-Tulu inspired) |
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

// Self-learning adaptive probe
let mut adaptive = AdaptiveYoneda::new(document, provider, config);
let (result, score) = adaptive.adaptive_probe("query", scorer).await?;

// Self-learning with evolving rubric scoring (DR-Tulu style)
let mut adaptive = AdaptiveYoneda::with_rubrics(document, provider, config);
let (result, score, per_rubric) = adaptive.adaptive_probe_with_rubrics("query").await?;
// per_rubric: {"Factual Recall" => 0.85, "Answer Relevance" => 0.90, ...}
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
  - Two documents are semantically isomorphic iff they produce the same results for all queries
- **Naturality**: `check_naturality(P, q, f, transform, sim)`
  - Verifies `probe(f(q)) ≈ transform(probe(q))`

### QueryMorphism — Category of Queries

```rust
// Identity
let id = QueryMorphism::identity();       // id(q) = q

// Named morphism
let refine = QueryMorphism::new("focus_science",
    |q| format!("{} Focus on science.", q));

// Composition (associative)
let composed = g.compose(f);              // (g ∘ f)(q) = g(f(q))

// Functorial evaluation
let result = yoneda.fmap(&composed, base_query).await?;
```

### Profunctor Optics — Typed I/O

```
lmap          Φ (Hylomorphism)        rmap
C ────────→ String ──────────→ String ────────→ D
  contra-                                covariant
  variant                                
```

- `lmap: &C → String` — serialize domain input to prompt (contravariant)
- `rmap: String → D` — parse LLM output to domain type (covariant)
- Composition `rmap ∘ Φ ∘ lmap` is type-safe end-to-end

### Left Kan Extension — Extrapolation

The `AdaptiveYoneda` implements a Left Kan Extension:

```
    Observed ──J──→ AllQueries
       │               │
       F               Lan_J F
       │               │
       ↓               ↓
      Set             Set
```

Where `F(q) = best_trajectory_result(q)` on observed queries, and
`Lan_J F(q') ≈ F(nearest(q'))` for unseen queries.

### Evolving Rubric Reward — LLM-as-Judge (DR-Tulu)

Inspired by DR-Tulu (arXiv:2511.19399), the rubric system replaces hardcoded
scoring with an **evolving, LLM-judged evaluation** that discovers task-specific
quality dimensions automatically.

```
┌──────────────────── Rubric Buffer ────────────────────┐
│                                                        │
│  Persistent Rubrics ─── always scored ──→  Weighted    │
│  (Factual Recall,                          Reward      │
│   Answer Relevance,                           │        │
│   Completeness)                               │        │
│                                               │        │
│  Active Adaptive  ──── scored + filtered ──→  ↑        │
│  (LLM-generated)                                       │
│                     filter_and_retire()                 │
│  Inactive Adaptive ── zero-std retired ──→  ∅          │
│                                                        │
└────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
   LLM-as-Judge                  Rubric Generator
   score 0–2 per criterion       compare N responses
   → normalize to [0,1]          → find discriminative criteria
```

**3-layer lifecycle** (mirrors DR-Tulu's `rubric_buffer`):

| Layer | Description | Action |
|-------|-------------|--------|
| **Persistent** | Ground truth (always scored) | Never retired |
| **Active** | LLM-discovered criteria | Scored, tracked for std |
| **Inactive** | Non-discriminative (std ≈ 0) | Retired, no longer scored |

**Usage:**

```rust
// Create with evolving rubrics (3 default persistent rubrics)
let mut adaptive = AdaptiveYoneda::with_rubrics(document, provider, config);
adaptive.rubric_gen_interval = 3;  // generate new rubrics every 3 probes

// Probe — scores via LLM judge, returns per-rubric breakdown
let (result, score, per_rubric) = adaptive.adaptive_probe_with_rubrics("query").await?;
// per_rubric: {"Factual Recall" => 0.5, "Answer Relevance" => 0.75, ...}

// Custom rubric buffer
let mut buf = RubricBuffer::default();
buf.persistent.push(RubricItem::persistent("Code Quality", "Output compiles and follows idioms"));
buf.persistent.push(RubricItem::persistent("Correctness", "Logic is sound and handles edge cases"));
let mut adaptive = AdaptiveYoneda::with_custom_rubrics(doc, provider, config, buf);
```

**Automatic evolution loop** (runs inside `adaptive_probe_with_rubrics`):

1. **Score**: LLM judge evaluates response against ALL active rubrics → `{"score": 0-2}` per criterion
2. **Record**: Per-rubric scores tracked for std computation
3. **Generate** (every `rubric_gen_interval` probes): LLM compares recent responses → generates positive/negative adaptive rubrics
4. **Retire** (every 3 probes): Zero-std rubrics → inactive (non-discriminative = useless)
5. **Cap**: Active adaptive rubrics capped at `max_active` (default 5)

**Category theory framing:**

- **Criterion Category** `Crit` — objects are rubric items
- **Judge Functor** `J: Crit → [0,1]` — LLM scores each criterion
- **Natural Transformation** `η_t → η_{t+1}` — each generation refines scoring criteria
- **Quotient** `r₁ ~ r₂ iff std(J(r)) = 0` — non-discriminative rubrics identified to zero object

## Execution Parameters

### `LambdaConfig`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `context_window` | 32,000 | Model context window K (tokens) |
| `accuracy_target` | 0.80 | Target accuracy α ∈ (0, 1] |
| `cost_params.c_invoke` | 1.0 | Cost per leaf M invocation |
| `cost_params.c_compose` | 0.0 | Cost per REDUCE composition (0 = symbolic, free) |

### Analytical Planning (Theorem 4)

The planner computes optimal k* analytically:

```
k* = ⌈√(n · c_invoke / c_compose)⌉     (when c_compose > 0)
k* = ⌈n / K⌉                            (when c_compose ≈ 0, symbolic)
```

Then validates the accuracy constraint:
```
A(K)^d · A_⊕^d ≥ α                      (loop increments k* until satisfied)
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

```
Rate limit exceeded: free-models-per-day
```

**Fix**: Add $10 credits to OpenRouter, or switch to a paid model:
```bash
RIG_RLM_MODEL=google/gemini-2.5-flash-8b
```

### Python LD_LIBRARY_PATH

```
error while loading shared libraries: libpython3.xx.so
```

**Fix**: Set the library path:
```bash
export LD_LIBRARY_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH
```

Or use the Nix flake which sets this automatically.

### Compiler Warnings

The `nuggets/core.rs` module has persistent `unsafe` warnings from AVX2 intrinsics. These are harmless (the unsafe blocks exist at the function level) and will be resolved when the crate migrates to Rust 2024 edition's stricter unsafe scoping.
