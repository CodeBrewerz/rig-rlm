# HEHRGNN — Heterogeneous Entity-Hyper-Relational Graph Neural Network

> Finverse GNN Platform: relational graph intelligence for finance, built on [Burn](https://burn.dev/).
>
> Part of the [rig-rlm](../../README.md) monorepo. See also: [λ-RLM + HyperAgent](../../LAMBDA_RLM.md).

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
│   │   │
│   │   ├── msa/                   # Memory Sparse Attention
│   │   │   ├── mod.rs             # MsaBlock, MsaLayer, forward pass
│   │   │   ├── sparse_attn.rs     # Top-K sparse attention with masking
│   │   │   ├── memory_bank.rs     # Persistent memory bank with routing
│   │   │   ├── router.rs          # Expert router (top-K gating)
│   │   │   ├── rope.rs            # Rotary Position Embeddings (RoPE)
│   │   │   ├── interleave.rs      # Local/global attention interleaving
│   │   │   ├── scoring.rs         # Attention scoring functions
│   │   │   ├── pooling.rs         # Attention pooling strategies
│   │   │   └── loss.rs            # MSA-specific loss functions
│   │   │
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
└── tests/                         # 47+ integration tests
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

### 4 GNN Models (Ensemble)

| Model | Key Feature | Best Config |
|-------|-------------|-------------|
| **GraphSAGE** | Mean aggregation + DoRA adapter | +7.9% AUC with JEPA |
| **RGCN mHC** | 8-layer multi-hop, 4 streams | +4.2% AUC with JEPA |
| **GAT** | 4-head attention | +9.9% AUC with JEPA |
| **GPS Transformer** | Global + local attention | +3.8% AUC with JEPA |

All models train with **JEPA** (InfoNCE + uniformity regularization).

### Memory Sparse Attention (MSA)

The `model/msa/` module implements trainable long-range attention for scaling context beyond what standard attention handles efficiently.

```
Input Sequence
      │
      ▼
┌─────────────────────────────┐
│  MsaLayer                    │
│  ┌───────────┬────────────┐ │
│  │ Local Attn│Global Attn │ │   ← Interleaved (configurable ratio)
│  │ (window)  │(sparse top-K)│
│  └─────┬─────┴─────┬──────┘ │
│        │            │        │
│        ▼            ▼        │
│  ┌────────┐  ┌────────────┐ │
│  │  RoPE  │  │Expert Router│ │   ← Top-K gating across experts
│  └────┬───┘  └─────┬──────┘ │
│       └─────┬──────┘        │
│             ▼               │
│  ┌──────────────────┐       │
│  │  Memory Bank     │       │   ← Persistent document memory
│  │  (route + store) │       │
│  └──────────────────┘       │
└─────────────────────────────┘
```

| File | What It Does |
|------|-------------|
| `mod.rs` | `MsaBlock` + `MsaLayer` — stackable attention layers |
| `sparse_attn.rs` | Top-K attention — only attends to K most relevant positions |
| `memory_bank.rs` | Persistent memory bank — stores and routes to document embeddings |
| `router.rs` | Expert routing — top-K gating for MoE-style processing |
| `rope.rs` | Rotary Position Embeddings for position-aware attention |
| `interleave.rs` | Interleaves local (window) and global (sparse) attention |
| `scoring.rs` | Attention scoring: dot-product, additive, cosine |
| `pooling.rs` | CLS, mean, max pooling over attention outputs |
| `loss.rs` | Contrastive + diversity losses for MSA training |

### Fiduciary Engine (18 Action Types)

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

Sum-product network providing calibrated risk probability, lift factors, counterfactuals, and exact inference.

### GEPA Optimizer (Self-Improvement)

| Target | What It Tunes | Persistence |
|--------|---------------|-------------|
| **Fiduciary weights** | GNN/PC blend α/β, axes weights | `gepa_weights.json` |
| **Training hyperparams** | lr, weight_decay, neg_ratio | `/tmp/gepa_train_config.json` |
| **Prediction thresholds** | recommend, anomaly, urgency cutoffs | `/tmp/gepa_prediction_config.json` |
| **Auto-tune (pipeline)** | Runs 5 evals every `run_pipeline()` | `gepa_weights.json` |

> **Note**: The λ-RLM module ([`LAMBDA_RLM.md`](../../LAMBDA_RLM.md)) uses a separate GEPA instance for query morphism evolution and HyperAgent parameter co-evolution.

---

## Test Catalog

### Core Pipeline

| Test | Command | Time |
|------|---------|------|
| **Ensemble Pipeline** | `--test ensemble_pipeline_test` | ~64s |
| **Ensemble 100K** | `--test ensemble_100k_test` | ~24s |

### GNN Models

`gnn_training_test` · `jepa_test` · `hehrgnn_jepa_test` · `lora_test` · `mhc_test` · `combo_features_test` · `per_model_sweep_test` · `tuning_test` · `progressive_learning_test`

### Fiduciary

`fiduciary_alignment_bench_test` · `fiduciary_actions_test` · `fiduciary_scenarios_test` · `fiduciary_negative_test` · `fiduciary_schema_validation_test` · `fiduciary_generalization_test` · `fiduciary_model_comparison_test` · `scenario_recommendations_test`

### Probabilistic Circuits

`pc_fiduciary_test` · `large_graph_pc_test` · `rich_pc_comparison_test` · `circuit_self_learning_test`

### GEPA Optimizer

`gepa_optimizer_test` · `gepa_training_test` · `gepa_prediction_test`

### Live LLM (require `OPENAI_API_KEY`)

```bash
cargo test -p hehrgnn --test gepa_optimizer_test test_gepa_llm -- --ignored --nocapture
cargo test -p hehrgnn --test gepa_training_test test_gepa_llm_training -- --ignored --nocapture
cargo test -p hehrgnn --test gepa_prediction_test test_gepa_llm_prediction -- --ignored --nocapture
```

### Anomaly · Interpretability · Scenarios · Scale

`anomaly_realworld_test` · `ensemble_anomaly_test` · `hehrgnn_anomaly_test` · `sae_financial_health_test` · `all_models_probe_test` · `probe_reward_test` · `learnable_scorer_test` · `scenario_entity_resolution_test` · `scenario_gl_tax_test` · `scenario_peer_splits_test` · `scenario_receipt_linking_test` · `scenario_recon_matching_test` · `scenario_recurring_bills_test` · `scenario_tax_estimation_test` · `large_scale_test` · `evolving_graph_simulation_test` · `real_ensemble_evolution_test` · `multihop_test` · `e2e_test` · `all_features_test`

---

## Self-Improvement Feedback Loop

```
Run 1: Train 4 GNNs → GEPA auto-tune → save checkpoints
Run 2: Load checkpoints → train further → GEPA improves → save if better
Run N: Cumulative improvement compounds
```

| Persistence File | Reset |
|-----------------|-------|
| `/tmp/gnn_weights/` | `rm -rf /tmp/gnn_weights` |
| `gepa_weights.json` | `rm gepa_weights.json` |

---

## Development Tips

### Adding a New GNN Model

1. Create `src/model/your_model.rs` implementing `Module<B>`
2. Implement `forward(&self, graph: &HeteroGraph<B>) -> NodeEmbeddings<B>`
3. Add to `ensemble_pipeline.rs`
4. Add test in `tests/`

### Adding a New Fiduciary Action

1. Add variant to `FiduciaryActionType` in `eval/fiduciary.rs`
2. Add matching in `generate_candidates()`
3. Add scoring in `compute_fiduciary_axes()`
4. Add domain in `FiduciaryActionType::domain()`
5. Add scenario to alignment bench

---

## Cross-References

| Document | What It Covers |
|----------|---------------|
| [README.md](../../README.md) | Project overview, entry points, usage, deployment |
| [LAMBDA_RLM.md](../../LAMBDA_RLM.md) | λ-RLM engine, Yoneda, DR-Tulu rubrics, HyperAgent |
| [hehrgnn/README.md](README.md) | This file — GNN platform, MSA, fiduciary engine |
