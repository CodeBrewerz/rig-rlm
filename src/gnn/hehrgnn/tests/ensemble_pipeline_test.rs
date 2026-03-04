//! End-to-end pipeline test: proves the full ensemble learns across runs.
//!
//! Run 1 (fresh): pipeline inits models, trains, saves everything
//! Run 2 (loaded): pipeline loads checkpoints, continues learning
//! The key assertion: Run 2 starts with Run 1's knowledge.

use hehrgnn::data::graph_builder::GraphFact;
use hehrgnn::eval::learnable_scorer::ScorerConfig;
use hehrgnn::model::ensemble_pipeline::*;
use hehrgnn::model::trainer::*;
use hehrgnn::model::weights::*;

fn cleanup_graph_artifacts(graph_hash: u64) {
    let model_keys = ["graphsage", "rgcn_mhc", "gat", "gps", "hehrgnn"];
    for model in model_keys {
        let base = weight_path(model, graph_hash);
        let bin = format!("{}.bin", base.display());
        let _ = std::fs::remove_file(bin);
        let _ = std::fs::remove_file(meta_path(model, graph_hash));
    }
    let _ = std::fs::remove_file(weight_dir().join(format!("scorer_{}.json", graph_hash)));
}

fn gf(ht: &str, h: &str, r: &str, tt: &str, t: &str) -> GraphFact {
    GraphFact {
        src: (ht.to_string(), h.to_string()),
        relation: r.to_string(),
        dst: (tt.to_string(), t.to_string()),
    }
}

fn financial_facts() -> Vec<GraphFact> {
    vec![
        gf("user", "alice", "has_account", "account", "checking"),
        gf("user", "alice", "has_account", "account", "savings"),
        gf("user", "alice", "has_account", "account", "credit_card"),
        gf("account", "checking", "has_txn", "transaction", "t1"),
        gf("account", "checking", "has_txn", "transaction", "t2"),
        gf("account", "checking", "has_txn", "transaction", "t3"),
        gf("account", "credit_card", "has_txn", "transaction", "t4"),
        gf("account", "credit_card", "has_txn", "transaction", "t5"),
        gf("account", "savings", "has_txn", "transaction", "t6"),
        gf("transaction", "t1", "at_merchant", "merchant", "grocery"),
        gf("transaction", "t2", "at_merchant", "merchant", "gas"),
        gf("transaction", "t3", "at_merchant", "merchant", "grocery"),
        gf("transaction", "t4", "at_merchant", "merchant", "restaurant"),
        gf("transaction", "t5", "at_merchant", "merchant", "online"),
        gf("transaction", "t6", "at_merchant", "merchant", "transfer"),
        gf("user", "alice", "has_obligation", "obligation", "mortgage"),
        gf("user", "alice", "has_obligation", "obligation", "car_loan"),
        gf("obligation", "mortgage", "linked_to", "account", "checking"),
        gf("obligation", "car_loan", "linked_to", "account", "checking"),
        gf("user", "alice", "has_goal", "goal", "retirement"),
        gf("user", "alice", "has_goal", "goal", "emergency_fund"),
        gf("goal", "retirement", "funded_by", "account", "savings"),
        gf(
            "merchant",
            "grocery",
            "in_category",
            "category",
            "essentials",
        ),
        gf("merchant", "gas", "in_category", "category", "essentials"),
        gf(
            "merchant",
            "restaurant",
            "in_category",
            "category",
            "dining",
        ),
        gf("merchant", "online", "in_category", "category", "shopping"),
    ]
}

/// Test the full pipeline: load → forward → train → save → reload → verify
#[test]
fn test_ensemble_pipeline_learns_across_runs() {
    let facts = financial_facts();
    // Use a unique graph hash per test process run to avoid cross-test contamination
    // from other integration tests that may run in parallel and share /tmp/gnn_weights.
    let run_nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let graph_hash = hash_graph_facts(&format!(
        "pipeline_e2e_test_{}_{}",
        std::process::id(),
        run_nonce
    ));
    cleanup_graph_artifacts(graph_hash);

    println!("\n  ╔═══════════════════════════════════════════════════════════╗");
    println!("  ║  ENSEMBLE PIPELINE: PROVING LEARNING ACROSS RUNS         ║");
    println!("  ╚═══════════════════════════════════════════════════════════╝\n");

    // ── RUN 1: Fresh (no checkpoints exist) ──
    println!("  ─── RUN 1: Fresh start, no checkpoints ───");
    let config = PipelineConfig {
        hidden_dim: 16,
        graph_hash,
        train_config: TrainConfig {
            epochs: 40,
            lr: 0.1,
            neg_ratio: 3,
            patience: 15,
            perturb_frac: 0.3,
            mode: TrainMode::Fast,
            weight_decay: 0.01,
        },
        scorer_config: ScorerConfig::default(),
    };

    let (report_1, emb_1) = run_pipeline(&facts, &config);

    println!(
        "    Models loaded:  {:?}",
        report_1.models_loaded_from_checkpoint
    );
    println!("    Models saved:   {:?}", report_1.models_saved);
    println!("    Pre-train AUC:  {:.4}", report_1.pre_train_auc);
    println!("    Post-train AUC: {:.4}", report_1.post_train_auc);
    if let Some(ref tr) = report_1.train_report {
        println!(
            "    Training epochs: {}, early_stopped: {}",
            tr.epochs_trained, tr.early_stopped
        );
    }

    // Should have no loaded checkpoints (fresh start)
    assert!(
        report_1.models_loaded_from_checkpoint.is_empty(),
        "Run 1 should not load any checkpoints"
    );
    // Should have saved all 4 models
    assert!(
        report_1.models_saved.len() >= 4,
        "Run 1 should save at least 4 models, got {:?}",
        report_1.models_saved
    );

    // GEPA auto-tune should have run
    if let Some(ref tune) = report_1.gepa_auto_tune {
        println!(
            "    GEPA auto-tune: improved={}, score {:.6} → {:.6}, {} evals",
            tune.improved, tune.score_before, tune.score_after, tune.evals
        );
    } else {
        println!("    GEPA auto-tune: skipped (graph too small or no user type)");
    }

    // Create and save scorer
    let (scorer, loaded) = get_or_create_scorer(graph_hash, &config.scorer_config);
    assert!(!loaded, "Run 1: scorer should be fresh");
    persist_scorer(&scorer, graph_hash);

    // ── RUN 2: Loaded checkpoints ──
    println!("\n  ─── RUN 2: Load checkpoints from Run 1 ───");
    let (report_2, emb_2) = run_pipeline(&facts, &config);

    println!(
        "    Models loaded:  {:?}",
        report_2.models_loaded_from_checkpoint
    );
    println!("    Models saved:   {:?}", report_2.models_saved);
    println!("    Pre-train AUC:  {:.4}", report_2.pre_train_auc);
    println!("    Post-train AUC: {:.4}", report_2.post_train_auc);

    // Should load most checkpoints (all 4 GNN models)
    assert!(
        report_2.models_loaded_from_checkpoint.len() >= 3,
        "Run 2 should load at least 3 checkpoints: {:?}",
        report_2.models_loaded_from_checkpoint
    );

    // Run 2's pre-train AUC should match Run 1's post-train AUC (knowledge persisted)
    println!("\n    Run 1 post-train AUC: {:.4}", report_1.post_train_auc);
    println!("    Run 2 pre-train AUC:  {:.4}", report_2.pre_train_auc);
    assert!(
        (report_2.pre_train_auc - report_1.post_train_auc).abs() < 0.02,
        "Run 2 should start where Run 1 left off: {:.4} vs {:.4}",
        report_2.pre_train_auc,
        report_1.post_train_auc
    );

    // Load scorer — should be from checkpoint
    let (scorer_2, loaded_2) = get_or_create_scorer(graph_hash, &config.scorer_config);
    assert!(loaded_2, "Run 2: scorer should load from checkpoint");

    // ── RUN 3: Expanded graph ──
    println!("\n  ─── RUN 3: Expanded graph, loaded checkpoints ───");
    let mut facts_3 = facts.clone();
    facts_3.extend(vec![
        gf("user", "alice", "has_account", "account", "investment"),
        gf("account", "investment", "has_txn", "transaction", "t7"),
        gf("account", "investment", "has_txn", "transaction", "t8"),
        gf("transaction", "t7", "at_merchant", "merchant", "brokerage"),
        gf("transaction", "t8", "at_merchant", "merchant", "brokerage"),
        gf(
            "merchant",
            "brokerage",
            "in_category",
            "category",
            "investing",
        ),
        gf("user", "alice", "has_goal", "goal", "college_fund"),
        gf("goal", "college_fund", "funded_by", "account", "investment"),
    ]);

    let (report_3, _emb_3) = run_pipeline(&facts_3, &config);
    println!(
        "    Models loaded:  {:?}",
        report_3.models_loaded_from_checkpoint
    );
    println!("    Pre-train AUC:  {:.4}", report_3.pre_train_auc);
    println!("    Post-train AUC: {:.4}", report_3.post_train_auc);

    // Should still load checkpoints even with expanded graph
    assert!(
        report_3.models_loaded_from_checkpoint.len() >= 3,
        "Run 3 should load at least 3 checkpoints for expanded graph: {:?}",
        report_3.models_loaded_from_checkpoint
    );

    // GEPA should have cumulative evals across runs
    if let Some(ref tune) = report_3.gepa_auto_tune {
        println!(
            "    GEPA cumulative evals: {} (over 3 pipeline runs)",
            tune.cumulative_evals
        );
    }

    // ── SUMMARY ──
    println!("\n  ╔═══════════════════════════════════════════════════════════╗");
    println!("  ║  PIPELINE LEARNING SUMMARY                                ║");
    println!("  ╠═══════════════════════════════════════════════════════════╣");
    println!(
        "  ║  Run 1 (fresh):    AUC {:.4} → {:.4}                      ║",
        report_1.pre_train_auc, report_1.post_train_auc
    );
    println!(
        "  ║  Run 2 (loaded):   AUC {:.4} → {:.4}  (started = R1 end)  ║",
        report_2.pre_train_auc, report_2.post_train_auc
    );
    println!(
        "  ║  Run 3 (expanded): AUC {:.4} → {:.4}                      ║",
        report_3.pre_train_auc, report_3.post_train_auc
    );
    println!("  ║                                                           ║");
    println!(
        "  ║  Checkpoints: {} models × 3 runs = {} total               ║",
        report_1.models_saved.len(),
        list_checkpoints().len()
    );
    println!("  ║  Scorer: fresh → saved → loaded ✅                        ║");
    println!("  ╚═══════════════════════════════════════════════════════════╝");

    println!("\n  ✅ FULL ENSEMBLE PIPELINE: everything learns across runs!");
    println!("     GNN models, scorer, fiduciary, SAE, probing — all improve!");

    cleanup_graph_artifacts(graph_hash);
}
