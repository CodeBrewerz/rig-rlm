# HEHRGNN — Heterogeneous Entity-Hyper-Relational Graph Neural Network

> Finverse GNN Platform: relational graph intelligence for finance, built on [Burn](https://burn.dev/).

## Quick Start

```bash
# All commands run from the repo root
cd rig-rlm

# Run all unit + integration tests (skips LLM tests)
cargo test -p hehrgnn

# Run a specific test
cargo test -p hehrgnn --test ensemble_pipeline_test -- --nocapture

# Run the full ensemble pipeline (4 GNN models + GEPA auto-tune)
cargo test -p hehrgnn --test ensemble_100k_test -- --nocapture

# Start the HTTP server (MCP/A2A)
cargo run -p hehrgnn --bin hehrgnn-server
```

## Architecture Overview

```
GraphFacts (TQL triples)
    │
    ▼
┌─────────────────────────────────────────┐
│  Graph Builder (data/)                  │
│  TQL → HeteroGraph<B> with node feats   │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────┼─────────────┬──────────────┐
    ▼             ▼             ▼              ▼
┌────────┐  ┌──────────┐  ┌────────┐    ┌──────────┐
│GraphSAGE│  │RGCN (mHC)│  │  GAT   │    │GPS Trans.│
│+ DoRA   │  │8L×4 str. │  │4-head  │    │4-head FFN│
└────┬───┘  └────┬─────┘  └───┬────┘    └────┬─────┘
     └────────┬──┴────────────┴───────────────┘
              ▼
     Ensemble Embeddings (per node type)
              │
     ┌────────┼──────────────┐
     ▼        ▼              ▼
┌─────────┐ ┌──────────┐ ┌──────────────┐
│Anomaly  │ │Fiduciary │ │Probabilistic │
│Detection│ │Engine    │ │Circuit (PC)  │
│(SAGE)   │ │18 actions│ │risk calibrate│
└─────────┘ └────┬─────┘ └──────┬───────┘
                 └───────┬──────┘
                         ▼
              Fiduciary Recommendations
              (ranked, PC-calibrated)
                         │
                         ▼
              ┌─────────────────────┐
              │ GEPA Auto-Tune      │
              │ Self-improves every  │
              │ pipeline run         │
              └─────────────────────┘
```

## Project Structure

```
src/gnn/hehrgnn/
├── Cargo.toml
├── src/
│   ├── main.rs                    # CLI binary
│   ├── server_main.rs             # HTTP server binary (MCP/A2A)
│   ├── lib.rs                     # Library root
│   │
│   ├── data/                      # Graph data layer
│   │   ├── graph_builder.rs       # TQL facts → HeteroGraph<B>
│   │   ├── hetero_graph.rs        # Core heterogeneous graph struct
│   │   ├── fact.rs                # HehrFact triple representation
│   │   ├── batcher.rs             # Burn DataLoader batching
│   │   ├── negative_sampling.rs   # Negative edge sampling for training
│   │   ├── neighbor_sampler.rs    # Mini-batch neighbor sampling
│   │   ├── synthetic.rs           # Procedural test graph generation
│   │   └── vocab.rs               # Entity/relation vocabularies
│   │
│   ├── model/                     # GNN architectures + training
│   │   ├── graphsage.rs           # GraphSAGE (mean aggregation)
│   │   ├── rgcn.rs                # Relational GCN (basis decomposition)
│   │   ├── mhc.rs                 # Multi-Hop Convolution RGCN (8 layers, 4 streams)
│   │   ├── gat.rs                 # Graph Attention Network (multi-head)
│   │   ├── graph_transformer.rs   # GPS Transformer (global+local attention)
│   │   ├── hehrgnn.rs             # HEHRGNN entity embedding model
│   │   ├── gnn_layer.rs           # Shared GNN layer primitives
│   │   ├── backbone.rs            # NodeEmbeddings output type
│   │   ├── embedding.rs           # Embedding initialization
│   │   ├── lora.rs                # LoRA / DoRA adapter (HeteroDoRA)
│   │   ├── jepa.rs                # JEPA training (InfoNCE + uniformity)
│   │   ├── trainer.rs             # Training loop: link-pred AUC, early stopping
│   │   ├── probe.rs               # Linear probing for interpretability
│   │   ├── weights.rs             # Model checkpoint save/load
│   │   ├── ensemble_pipeline.rs   # Full pipeline: 5 models + GEPA auto-tune
│   │   └── pc/                    # Probabilistic Circuit
│   │       ├── circuit.rs         # CompiledCircuit (sum-product network)
│   │       ├── node.rs            # Sum/Product/Leaf nodes
│   │       ├── structure.rs       # Structure learning (CLTree + independence)
│   │       ├── distribution.rs    # Categorical distributions
│   │       ├── em.rs              # EM parameter learning
│   │       ├── query.rs           # Exact inference: marginal, conditional, lift
│   │       ├── bridge.rs          # GNN embeddings → PC training data
│   │       └── fiduciary_pc.rs    # PC analysis: risk, lift, counterfactuals
│   │
│   ├── eval/                      # Evaluation & scoring
│   │   ├── fiduciary.rs           # 18 fiduciary action types, recommend()
│   │   ├── learnable_scorer.rs    # Thompson sampling scorer
│   │   ├── bench.rs               # Alignment benchmark framework
│   │   ├── metrics.rs             # AUC, Kendall τ, NDCG
│   │   ├── probing.rs             # Feature probing (anomaly, type)
│   │   ├── sae.rs                 # Sparse Autoencoder for interpretability
│   │   ├── explanation.rs         # Natural language explanations
│   │   └── evaluate.rs            # Graph-level evaluation
│   │
│   ├── optimizer/                 # Self-improvement
│   │   └── gepa.rs                # GEPA optimizer: Pareto-evolutionary search
│   │                              #   NumericMutator, LlmMutator (Trinity),
│   │                              #   auto_tune_weights(), OptimizedWeights
│   │
│   ├── feedback/                  # Online learning
│   │   ├── collector.rs           # Feedback signal collection
│   │   └── retrainer.rs           # Incremental retraining
│   │
│   ├── server/                    # HTTP API
│   │   ├── state.rs               # Server state (graph, models, pipeline)
│   │   └── handlers.rs            # MCP/A2A request handlers
│   │
│   ├── ingest/                    # Data ingestion
│   ├── training/                  # Training utilities
│   ├── tasks/                     # Task definitions
│   └── past_runs/                 # Run history
│
└── tests/                         # 47 integration tests (see below)
```

## Key Concepts

### Graph Facts (TQL Triples)

Everything starts with `GraphFact` — a `(src_type, src_name, relation, dst_type, dst_name)` triple:

```rust
GraphFact {
    src: ("user", "alice"),
    relation: "owns",
    dst: ("account", "checking"),
}
```

These get built into a `HeteroGraph<B>` with typed node features and edge indices.

### 4 GNN Models (Ensemble)

| Model | Key Feature | Best Config |
|-------|-------------|-------------|
| **GraphSAGE** | Mean aggregation + DoRA adapter | +7.9% AUC with JEPA |
| **RGCN mHC** | 8-layer multi-hop, 4 streams | +4.2% AUC with JEPA |
| **GAT** | 4-head attention | +9.9% AUC with JEPA |
| **GPS Transformer** | Global + local attention | +3.8% AUC with JEPA |

All models train with **JEPA** (InfoNCE + uniformity regularization), not standard link prediction.

### Fiduciary Engine (18 Action Types)

The system generates financial recommendations across 18 action types:

| Domain | Actions |
|--------|---------|
| **Core** | `should_investigate`, `should_avoid`, `should_pay` |
| **Debt** | `should_refinance`, `should_pay_down_lien`, `should_dispute` |
| **Subscriptions** | `should_cancel`, `should_review_recurring` |
| **Goals** | `should_fund_goal`, `should_adjust_budget` |
| **Accounts** | `should_transfer`, `should_consolidate` |
| **Assets** | `should_revalue_asset` |
| **Tax** | `should_prepare_tax`, `should_claim_exemption`, `should_run_tax_scenario`, `should_fund_tax_sinking` |
| **Reconciliation** | `should_reconcile` |

### Probabilistic Circuit (PC)

A sum-product network that provides:
- **Calibrated risk probability** P(risky | features)
- **Lift factors** (which variable drives risk)
- **Counterfactuals** ("if anomaly drops to low, risk drops by X%")
- **Exact inference** (no approximate sampling)

### GEPA Optimizer (Self-Improvement)

Genetic-Pareto optimizer that tunes parameters automatically:

| Target | What It Tunes | Persistence |
|--------|---------------|-------------|
| **Fiduciary weights** | GNN/PC blend α/β, axes weights | `gepa_weights.json` |
| **Training hyperparams** | lr, weight_decay, neg_ratio, perturb_frac | `/tmp/gepa_train_config.json` |
| **Prediction thresholds** | recommend, anomaly, urgency cutoffs | `/tmp/gepa_prediction_config.json` |
| **Auto-tune (pipeline)** | Runs 5 evals every `run_pipeline()` call | `gepa_weights.json` |

---

## Commands Reference

### Build & Test

```bash
# Build the library
cargo build -p hehrgnn

# Run all tests (fast — skips LLM tests)
cargo test -p hehrgnn

# Run all tests with output
cargo test -p hehrgnn -- --nocapture

# Run a specific test file
cargo test -p hehrgnn --test <test_name> -- --nocapture

# Run a specific test function
cargo test -p hehrgnn --test <test_file> <test_fn> -- --nocapture

# Run ignored tests (LLM tests that call Trinity)
cargo test -p hehrgnn --test <test_file> <test_fn> -- --ignored --nocapture
```

### Binaries

```bash
# Run main CLI
cargo run -p hehrgnn --bin hehrgnn

# Run HTTP server (MCP/A2A endpoints)
cargo run -p hehrgnn --bin hehrgnn-server
```

### Environment Variables

```bash
# Required for LLM-guided GEPA tests (Trinity via OpenRouter)
OPENAI_API_KEY=sk-or-v1-...    # in .env file at repo root
```

---

## Test Catalog

### Core Pipeline Tests

| Test | Command | What It Verifies | Time |
|------|---------|------------------|------|
| **Ensemble Pipeline** | `--test ensemble_pipeline_test` | 3 sequential `run_pipeline()` runs prove models checkpoint, reload, and continue learning. GEPA auto-tune fires each run. | ~64s |
| **Ensemble 100K** | `--test ensemble_100k_test` | 500 users × 47 merchants, full anomaly detection, novelty signals, HEHRGNN KG scoring. | ~24s |

### GNN Model Tests

| Test | Command | What It Verifies |
|------|---------|------------------|
| **GNN Training** | `--test gnn_training_test` | GraphSAGE trains, AUC improves, checkpoint save/load |
| **JEPA Training** | `--test jepa_test` | InfoNCE + uniformity training on all 4 models |
| **HEHRGNN JEPA** | `--test hehrgnn_jepa_test` | HEHRGNN entity embedding JEPA training |
| **LoRA/DoRA** | `--test lora_test` | HeteroDoRA adapter training and AUC improvement |
| **mHC RGCN** | `--test mhc_test` | Multi-hop convolution with 8 layers |
| **Combo Features** | `--test combo_features_test` | Best model+feature combos (DoRA, JEPA, mHC) |
| **Per-Model Sweep** | `--test per_model_sweep_test` | Hyperparameter sweep across all 4 models |
| **Tuning** | `--test tuning_test` | Learning rate, layer count, dropout tuning |
| **Progressive Learning** | `--test progressive_learning_test` | Multi-stage curriculum learning |

### Fiduciary Tests

| Test | Command | What It Verifies |
|------|---------|------------------|
| **Alignment Bench** | `--test fiduciary_alignment_bench_test` | 10 scenarios, ground truth, precision@K, NDCG, misalignment rate |
| **Fiduciary Actions** | `--test fiduciary_actions_test` | All 18 action types trigger correctly |
| **Fiduciary Scenarios** | `--test fiduciary_scenarios_test` | Complex financial scenarios produce correct recommendations |
| **Negative Tests** | `--test fiduciary_negative_test` | System does NOT recommend harmful actions |
| **Schema Validation** | `--test fiduciary_schema_validation_test` | Output schema correctness |
| **Generalization** | `--test fiduciary_generalization_test` | Works across varied graph topologies |
| **Model Comparison** | `--test fiduciary_model_comparison_test` | Compare fiduciary quality across GNN models |
| **Recommendations** | `--test scenario_recommendations_test` | End-to-end recommendation pipeline |

### Probabilistic Circuit Tests

| Test | Command | What It Verifies |
|------|---------|------------------|
| **PC Fiduciary** | `--test pc_fiduciary_test` | PC analysis on all 18 action types, lift, counterfactuals, 10 checks |
| **Large Graph PC** | `--test large_graph_pc_test` | 130+ entities, 5 risk profiles, PC risk differentiation |
| **Rich PC Comparison** | `--test rich_pc_comparison_test` | PC vs naive anomaly scoring comparison |
| **Circuit Self-Learning** | `--test circuit_self_learning_test` | PC EM training improves with more data |

### GEPA Optimizer Tests

| Test | Command | What It Verifies |
|------|---------|------------------|
| **Fiduciary Weights** | `--test gepa_optimizer_test` | Optimize GNN/PC blend and axes weights (NumericMutator, 30 evals) |
| **Training Hyperparams** | `--test gepa_training_test` | Optimize lr/weight_decay/neg_ratio/perturb_frac (NumericMutator) |
| **Prediction Quality** | `--test gepa_prediction_test` | Optimize threshold params for end-to-end prediction quality |

#### Live LLM Tests (require `OPENAI_API_KEY`)

These call Trinity via OpenRouter. Each run builds on the previous one's saved weights:

```bash
# Fiduciary weights (saves to /tmp/gepa_weights.json)
cargo test -p hehrgnn --test gepa_optimizer_test test_gepa_llm -- --ignored --nocapture

# Training hyperparams (saves to /tmp/gepa_train_config.json)
cargo test -p hehrgnn --test gepa_training_test test_gepa_llm_training -- --ignored --nocapture

# Prediction thresholds (saves to /tmp/gepa_prediction_config.json)
cargo test -p hehrgnn --test gepa_prediction_test test_gepa_llm_prediction -- --ignored --nocapture

# Run again to keep improving from checkpoint!
```

### Anomaly Detection Tests

| Test | Command | What It Verifies |
|------|---------|------------------|
| **Anomaly Realworld** | `--test anomaly_realworld_test` | Anomaly scoring on realistic financial patterns |
| **Ensemble Anomaly** | `--test ensemble_anomaly_test` | Cross-model anomaly consensus |
| **HEHRGNN Anomaly** | `--test hehrgnn_anomaly_test` | KG-based anomaly scoring |

### Interpretability Tests

| Test | Command | What It Verifies |
|------|---------|------------------|
| **SAE Financial Health** | `--test sae_financial_health_test` | Sparse autoencoder feature discovery |
| **All Models Probe** | `--test all_models_probe_test` | Linear probing across all 4 models |
| **Probe Reward** | `--test probe_reward_test` | Probing with reward signal |
| **Learnable Scorer** | `--test learnable_scorer_test` | Thompson sampling scorer training |

### Scenario Tests (Financial Use Cases)

| Test | Command | Financial Scenario |
|------|---------|-------------------|
| **Entity Resolution** | `--test scenario_entity_resolution_test` | Linked account detection |
| **GL Tax** | `--test scenario_gl_tax_test` | General ledger + tax computation |
| **Peer Splits** | `--test scenario_peer_splits_test` | Peer-to-peer split tracking |
| **Receipt Linking** | `--test scenario_receipt_linking_test` | Receipt → transaction matching |
| **Recon Matching** | `--test scenario_recon_matching_test` | Bank reconciliation |
| **Recurring Bills** | `--test scenario_recurring_bills_test` | Recurring payment detection |
| **Tax Estimation** | `--test scenario_tax_estimation_test` | Tax obligation estimation |

### Scale & Evolution Tests

| Test | Command | What It Verifies |
|------|---------|------------------|
| **Large Scale** | `--test large_scale_test` | Performance with large graphs |
| **Evolving Graph** | `--test evolving_graph_simulation_test` | Graph changes over time (add/remove entities) |
| **Real Ensemble Evolution** | `--test real_ensemble_evolution_test` | Full ensemble over multiple graph evolution steps |
| **Multi-hop** | `--test multihop_test` | Multi-hop traversal for deep financial patterns |
| **E2E** | `--test e2e_test` | End-to-end: ingest → train → score → recommend |
| **All Features** | `--test all_features_test` | Every feature in one pipeline run |

---

## Self-Improvement Feedback Loop

The system automatically improves on every pipeline run:

```
Run 1 (fresh):
  → Train 4 GNNs from scratch
  → GEPA auto-tune: seed with default weights → 5 evals → save best
  → Save model checkpoints

Run 2 (loaded):
  → Load 4 GNN checkpoints (start where Run 1 left off)
  → Train further → better embeddings
  → GEPA auto-tune: load Run 1's best weights → 5 more evals → save if improved
  → Cumulative improvement compounds

Run N:
  → Models keep improving from checkpoints
  → GEPA weights keep improving from persistence
  → Fiduciary recommendations get better each run
```

**Persistence files:**

| File | What | Reset Command |
|------|------|---------------|
| `/tmp/gnn_weights/` | Model checkpoints (all 5 models) | `rm -rf /tmp/gnn_weights` |
| `gepa_weights.json` | Fiduciary blend + axes weights | `rm gepa_weights.json` |
| `/tmp/gepa_train_config.json` | Training hyperparameters | `rm /tmp/gepa_train_config.json` |
| `/tmp/gepa_prediction_config.json` | Prediction thresholds | `rm /tmp/gepa_prediction_config.json` |

---

## Dependencies

| Crate | Purpose |
|-------|---------|
| `burn` | Deep learning framework (NdArray + WGPU backends) |
| `serde` / `serde_json` | Serialization for weights, configs, PC circuits |
| `rand` | Random sampling (negative edges, mutations) |
| `axum` / `tokio` / `tower-http` | HTTP server (MCP/A2A) |
| `chrono` | Timestamps for weight persistence |
| `rayon` | Parallel iteration for large graphs |
| `reqwest` | HTTP client for Trinity LLM API |
| `dotenvy` | Load `.env` file for API keys |

---

## Development Tips

### Running Tests Fast

```bash
# Run just the fast unit tests in gepa.rs
cargo test -p hehrgnn -- gepa::tests --nocapture

# Run just the alignment benchmark (3s)
cargo test -p hehrgnn --test fiduciary_alignment_bench_test -- --nocapture

# Run just the PC fiduciary check (1.6s)
cargo test -p hehrgnn --test pc_fiduciary_test -- --nocapture
```

### Adding a New GNN Model

1. Create `src/model/your_model.rs` implementing the `Module<B>` trait
2. Implement `forward(&self, graph: &HeteroGraph<B>) -> NodeEmbeddings<B>`
3. Add to `ensemble_pipeline.rs`: init → load checkpoint → train → save → extract embeddings
4. Add a test in `tests/`

### Adding a New Fiduciary Action Type

1. Add variant to `FiduciaryActionType` enum in `eval/fiduciary.rs`
2. Add matching logic in `generate_candidates()` to trigger from graph patterns
3. Add scoring logic in `compute_fiduciary_axes()` to set axis values
4. Add domain mapping in `FiduciaryActionType::domain()`
5. Add scenario to `fiduciary_alignment_bench_test.rs`

### Debugging PC Issues

```bash
# Check if probability distributions sum to 1.0
cargo test -p hehrgnn --test pc_fiduciary_test -- --nocapture 2>&1 | grep "Check 2"

# Check anomaly correlation
cargo test -p hehrgnn --test pc_fiduciary_test -- --nocapture 2>&1 | grep "Check 4"
```
