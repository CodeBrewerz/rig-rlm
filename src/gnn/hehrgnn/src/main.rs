//! HEHRGNN end-to-end pipeline binary.
//!
//! Reads a TQL schema file, generates synthetic hyper-relational data,
//! trains the HEHRGNN model, evaluates link prediction, and produces
//! an interactive HTML dashboard.

use std::fs;

use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;

use hehrgnn::data::synthetic::{SyntheticDataConfig, generate_synthetic_dataset};
use hehrgnn::training::scoring::{DistMultScorer, Scorer, TransEScorer};
use hehrgnn::training::train::{TrainConfig, train};

type TrainBackend = Autodiff<NdArray>;
type InferBackend = NdArray;

fn main() {
    // Parse command-line arguments (simple manual parsing)
    let args: Vec<String> = std::env::args().collect();

    let schema_path =
        get_arg(&args, "--schema").unwrap_or_else(|| "src/gnn/SchemaFinverse.tql".to_string());
    let epochs: usize = get_arg(&args, "--epochs")
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let batch_size: usize = get_arg(&args, "--batch-size")
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let hidden_dim: usize = get_arg(&args, "--hidden-dim")
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let lr: f64 = get_arg(&args, "--lr")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.001);
    let num_facts: usize = get_arg(&args, "--num-facts")
        .and_then(|v| v.parse().ok())
        .unwrap_or(500);
    let instances_per_type: usize = get_arg(&args, "--instances-per-type")
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let scorer_type = get_arg(&args, "--scorer").unwrap_or_else(|| "distmult".to_string());
    let output_dir = get_arg(&args, "--output").unwrap_or_else(|| "hehrgnn_output".to_string());

    if args.contains(&"--help".to_string()) || args.contains(&"-h".to_string()) {
        print_help();
        return;
    }

    println!();
    println!("  ╔═══════════════════════════════════════════╗");
    println!("  ║      HEHRGNN Pipeline Runner              ║");
    println!("  ╚═══════════════════════════════════════════╝");
    println!();

    // --- Step 1: Load and parse TQL schema ---
    println!("  [1/5] Loading schema from: {}", schema_path);
    let tql_content = match fs::read_to_string(&schema_path) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("  Error reading schema file: {}", e);
            std::process::exit(1);
        }
    };

    // --- Step 2: Generate synthetic data ---
    println!("  [2/5] Generating synthetic dataset...");
    let syn_config = SyntheticDataConfig {
        instances_per_type,
        num_facts,
        max_qualifiers: 3,
        seed: 42,
    };

    let (raw_facts, vocab, indexed_facts) = generate_synthetic_dataset(&tql_content, &syn_config);

    println!(
        "         Generated {} raw facts → {} indexed facts",
        raw_facts.len(),
        indexed_facts.len()
    );
    println!(
        "         Entities: {} | Relations: {}",
        vocab.num_entities(),
        vocab.num_relations()
    );

    if indexed_facts.is_empty() {
        eprintln!("  Error: No indexed facts generated. Check schema file.");
        std::process::exit(1);
    }

    // --- Step 3: Train/test split (80/20) ---
    println!("  [3/5] Splitting data (80/20)...");
    let split_idx = (indexed_facts.len() as f64 * 0.8) as usize;
    let train_facts = &indexed_facts[..split_idx];
    let test_facts = &indexed_facts[split_idx..];
    println!(
        "         Train: {} | Test: {}",
        train_facts.len(),
        test_facts.len()
    );

    if train_facts.is_empty() || test_facts.is_empty() {
        eprintln!("  Error: Not enough facts for train/test split.");
        std::process::exit(1);
    }

    // --- Step 4: Train the model ---
    println!("  [4/5] Training HEHRGNN model...\n");

    let device = <TrainBackend as Backend>::Device::default();

    let train_config = TrainConfig {
        epochs,
        lr,
        margin: 1.0,
        batch_size,
        negatives_per_positive: 3,
        hidden_dim,
        num_layers: 2,
        dropout: 0.1,
        eval_every: std::cmp::max(1, epochs / 5),
        scorer_type: scorer_type.clone(),
        output_dir: output_dir.clone(),
    };

    let result = train::<TrainBackend>(
        &train_config,
        train_facts,
        test_facts,
        vocab.num_entities(),
        vocab.num_relations(),
        &device,
    );

    // --- Step 5: Final predictions demo ---
    println!("\n  [5/5] Running sample predictions...\n");

    let infer_device = result
        .model
        .embeddings
        .entity_embedding
        .weight
        .val()
        .device();
    let infer_scorer: Box<dyn Scorer<InferBackend>> = match scorer_type.as_str() {
        "transe" => Box::new(TransEScorer::l2()),
        _ => Box::new(DistMultScorer::new()),
    };

    // Score a few test facts
    println!("  Sample Predictions (test facts):");
    println!(
        "  {:<6}  {:<6}  {:<6}  {:<10}  {:>8}",
        "Head", "Rel", "Tail", "Quals", "Score"
    );
    println!("  {}", "─".repeat(44));

    let batcher = hehrgnn::data::batcher::HehrBatcher::new();
    use burn::data::dataloader::batcher::Batcher;

    let sample_count = std::cmp::min(10, test_facts.len());
    for fact in &test_facts[..sample_count] {
        let item = hehrgnn::data::batcher::HehrFactItem {
            fact: fact.clone(),
            label: 1.0,
        };
        let batch: hehrgnn::data::batcher::HehrBatch<InferBackend> =
            batcher.batch(vec![item], &infer_device);
        let score = result.model.score_batch(&batch, infer_scorer.as_ref());
        let score_val: f32 = score
            .into_data()
            .as_slice::<f32>()
            .expect("Failed to read score")[0];

        println!(
            "  {:<6}  {:<6}  {:<6}  {:<10}  {:>8.4}",
            fact.head,
            fact.relation,
            fact.tail,
            format!("{}q", fact.qualifiers.len()),
            score_val
        );
    }

    // Final summary
    println!();
    println!("  ╔═══════════════════════════════════════════╗");
    println!("  ║  Pipeline Complete!                       ║");
    println!("  ╠═══════════════════════════════════════════╣");
    if let Some(last) = result.logger.history.last() {
        println!("  ║  Final Loss: {:<29.6}  ║", last.train_loss);
        if let Some(mrr) = last.mrr {
            println!("  ║  Final MRR:  {:<29.4}  ║", mrr);
        }
        if let Some(h10) = last.hits_at_10 {
            println!("  ║  Final H@10: {:<29.4}  ║", h10);
        }
    }
    println!(
        "  ║  Dashboard: {:<30}  ║",
        format!("{}/dashboard.html", output_dir)
    );
    println!(
        "  ║  Metrics:   {:<30}  ║",
        format!("{}/metrics.jsonl", output_dir)
    );
    println!("  ╚═══════════════════════════════════════════╝");

    // --- Step 6: Archive run to past_runs ---
    let past_runs_dir = {
        // Resolve relative to the crate source directory
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        format!("{}/src/past_runs", manifest_dir)
    };

    let now = chrono::Local::now();
    let run_name = format!("{}_{}", scorer_type, now.format("%Y%m%d_%H%M%S"));
    let archive_dir = format!("{}/{}", past_runs_dir, run_name);

    if let Err(e) = fs::create_dir_all(&archive_dir) {
        eprintln!("  Warning: could not create archive dir: {}", e);
    } else {
        // Copy dashboard
        let src_dashboard = format!("{}/dashboard.html", output_dir);
        let dst_dashboard = format!("{}/dashboard.html", archive_dir);
        let _ = fs::copy(&src_dashboard, &dst_dashboard);

        // Copy metrics
        let src_metrics = format!("{}/metrics.jsonl", output_dir);
        let dst_metrics = format!("{}/metrics.jsonl", archive_dir);
        let _ = fs::copy(&src_metrics, &dst_metrics);

        // Write run summary
        let summary = serde_json::json!({
            "run_name": run_name,
            "timestamp": now.to_rfc3339(),
            "gnn_type": "hehrgnn",
            "scorer": scorer_type,
            "config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "hidden_dim": hidden_dim,
                "lr": lr,
                "num_facts": num_facts,
                "instances_per_type": instances_per_type,
                "num_layers": 2,
                "margin": 1.0,
                "negatives_per_positive": 3,
                "dropout": 0.1,
            },
            "data": {
                "schema": schema_path,
                "raw_facts": raw_facts.len(),
                "indexed_facts": indexed_facts.len(),
                "entities": vocab.num_entities(),
                "relations": vocab.num_relations(),
                "train_facts": train_facts.len(),
                "test_facts": test_facts.len(),
            },
            "results": {
                "final_loss": result.logger.history.last().map(|h| h.train_loss),
                "final_mrr": result.logger.history.last().and_then(|h| h.mrr),
                "final_hits_at_1": result.logger.history.last().and_then(|h| h.hits_at_1),
                "final_hits_at_3": result.logger.history.last().and_then(|h| h.hits_at_3),
                "final_hits_at_10": result.logger.history.last().and_then(|h| h.hits_at_10),
                "final_mean_rank": result.logger.history.last().and_then(|h| h.mean_rank),
            }
        });

        let summary_path = format!("{}/run_summary.json", archive_dir);
        if let Ok(json) = serde_json::to_string_pretty(&summary) {
            let _ = fs::write(&summary_path, json);
        }

        println!();
        println!("  📁 Run archived to: {}", archive_dir);
        println!("     ├── dashboard.html");
        println!("     ├── metrics.jsonl");
        println!("     └── run_summary.json");
    }

    println!();
}

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn print_help() {
    println!("HEHRGNN Training Pipeline");
    println!();
    println!("USAGE:");
    println!("  cargo run -p hehrgnn -- [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!(
        "  --schema <PATH>          Path to TQL schema file (default: src/gnn/SchemaFinverse.tql)"
    );
    println!("  --epochs <N>             Number of training epochs (default: 50)");
    println!("  --batch-size <N>         Mini-batch size (default: 64)");
    println!("  --hidden-dim <N>         Embedding dimension (default: 32)");
    println!("  --lr <FLOAT>             Learning rate (default: 0.001)");
    println!("  --num-facts <N>          Number of synthetic facts to generate (default: 500)");
    println!("  --instances-per-type <N> Entity instances per type (default: 5)");
    println!("  --scorer <TYPE>          Scoring function: distmult or transe (default: distmult)");
    println!("  --output <DIR>           Output directory (default: hehrgnn_output)");
    println!("  --help, -h               Print this help message");
}
