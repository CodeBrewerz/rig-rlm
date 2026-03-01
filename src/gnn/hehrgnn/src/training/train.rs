//! Training loop for HEHRGNN.
//!
//! Implements mini-batch contrastive training with negative sampling,
//! using Burn's autodiff backend for gradient computation and Adam optimizer.

use burn::data::dataloader::batcher::Batcher;
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use crate::data::batcher::{HehrBatch, HehrBatcher, HehrFactItem};
use crate::data::fact::HehrFact;
use crate::data::negative_sampling::NegativeSampler;
use crate::model::hehrgnn::{HehrgnnModel, HehrgnnModelConfig};
use crate::training::loss::margin_ranking_loss;
use crate::training::observe::MetricsLogger;
use crate::training::scoring::{DistMultScorer, Scorer, TransEScorer};

/// Configuration for the training pipeline.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Number of training epochs.
    pub epochs: usize,
    /// Learning rate.
    pub lr: f64,
    /// Margin for margin ranking loss.
    pub margin: f64,
    /// Mini-batch size (number of positive facts per batch).
    pub batch_size: usize,
    /// Number of negative samples per positive fact.
    pub negatives_per_positive: usize,
    /// Embedding dimension.
    pub hidden_dim: usize,
    /// Number of GNN layers.
    pub num_layers: usize,
    /// Dropout rate.
    pub dropout: f64,
    /// Evaluate every N epochs (0 = only at end).
    pub eval_every: usize,
    /// Scoring function type: "transe" or "distmult".
    pub scorer_type: String,
    /// Output directory for metrics and dashboard.
    pub output_dir: String,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 50,
            lr: 0.001,
            margin: 1.0,
            batch_size: 64,
            negatives_per_positive: 5,
            hidden_dim: 32,
            num_layers: 2,
            dropout: 0.1,
            eval_every: 10,
            scorer_type: "distmult".to_string(),
            output_dir: "hehrgnn_output".to_string(),
        }
    }
}

/// Result of a full training run.
pub struct TrainResult<B: Backend> {
    /// The trained model (inner module without autodiff).
    pub model: HehrgnnModel<B>,
    /// Metrics logger with full history.
    pub logger: MetricsLogger,
}

/// Run the full training pipeline.
///
/// # Arguments
/// - `config`: training hyperparameters
/// - `train_facts`: indexed training facts
/// - `test_facts`: indexed test facts (for evaluation)
/// - `num_entities`: total entities in vocabulary
/// - `num_relations`: total relations in vocabulary
///
/// # Returns
/// `TrainResult` containing the trained model and metrics history.
pub fn train<B: AutodiffBackend>(
    config: &TrainConfig,
    train_facts: &[HehrFact],
    test_facts: &[HehrFact],
    num_entities: usize,
    num_relations: usize,
    device: &B::Device,
) -> TrainResult<B::InnerBackend> {
    // Ensure output directory exists
    let _ = std::fs::create_dir_all(&config.output_dir);

    let jsonl_path = format!("{}/metrics.jsonl", config.output_dir);
    let dashboard_path = format!("{}/dashboard.html", config.output_dir);

    let mut logger = MetricsLogger::new(Some(jsonl_path));

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║               HEHRGNN Training Pipeline                     ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║  Entities: {:>6}  │  Relations: {:>6}  │  Hidden: {:>4}   ║",
        num_entities, num_relations, config.hidden_dim
    );
    println!(
        "║  Epochs: {:>8}  │  Batch: {:>10}  │  LR: {:>8.6}  ║",
        config.epochs, config.batch_size, config.lr
    );
    println!(
        "║  Neg/Pos: {:>7}  │  Margin: {:>9.2}  │  Layers: {:>4}   ║",
        config.negatives_per_positive, config.margin, config.num_layers
    );
    println!(
        "║  Scorer: {:>8}  │  Dropout: {:>8.2}  │  Train: {:>5}   ║",
        config.scorer_type,
        config.dropout,
        train_facts.len()
    );
    println!(
        "║  Test: {:>10}  │  Eval every: {:>5}                    ║",
        test_facts.len(),
        config.eval_every
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Initialize model
    let mut model = HehrgnnModelConfig {
        num_entities,
        num_relations,
        hidden_dim: config.hidden_dim,
        num_layers: config.num_layers,
        dropout: config.dropout,
    }
    .init::<B>(device);

    // Initialize optimizer
    let mut optim = AdamConfig::new().init::<B, HehrgnnModel<B>>();

    // Create scorer
    let scorer: Box<dyn Scorer<B>> = match config.scorer_type.as_str() {
        "transe" => Box::new(TransEScorer::l2()),
        _ => Box::new(DistMultScorer::new()),
    };

    // Create scorer for inner backend (used during evaluation)
    let inner_scorer: Box<dyn Scorer<B::InnerBackend>> = match config.scorer_type.as_str() {
        "transe" => Box::new(TransEScorer::l2()),
        _ => Box::new(DistMultScorer::new()),
    };

    let neg_sampler = NegativeSampler::new(num_entities, config.negatives_per_positive);
    let batcher = HehrBatcher::new();

    println!("  Training started...\n");

    for epoch in 0..config.epochs {
        let epoch_start = std::time::Instant::now();
        logger.start_epoch();

        // Shuffle training facts
        let mut epoch_facts = train_facts.to_vec();
        use rand::seq::SliceRandom;
        epoch_facts.shuffle(&mut rand::rng());

        let total_batches = (epoch_facts.len() + config.batch_size - 1) / config.batch_size;
        let mut batch_idx = 0;

        // Process mini-batches
        for chunk in epoch_facts.chunks(config.batch_size) {
            batch_idx += 1;
            if batch_idx % 200 == 0 || batch_idx == total_batches {
                print!(
                    "\r    Epoch {}/{}: batch {}/{} ({:.0}%)   ",
                    epoch + 1,
                    config.epochs,
                    batch_idx,
                    total_batches,
                    batch_idx as f64 / total_batches as f64 * 100.0
                );
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
            // Build positive items
            let mut pos_items: Vec<HehrFactItem> = Vec::new();
            let mut neg_items: Vec<HehrFactItem> = Vec::new();

            for fact in chunk {
                pos_items.push(HehrFactItem {
                    fact: fact.clone(),
                    label: 1.0,
                });

                // Generate negatives for this positive
                let negatives = neg_sampler.sample(fact);
                for neg_fact in negatives {
                    neg_items.push(HehrFactItem {
                        fact: neg_fact,
                        label: 0.0,
                    });
                }
            }

            // Batch positives and negatives
            let pos_batch: HehrBatch<B> = batcher.batch(pos_items, device);
            let neg_batch: HehrBatch<B> = batcher.batch(neg_items, device);

            // Score positives and negatives
            let pos_scores = model.score_batch(&pos_batch, scorer.as_ref());
            let neg_scores_raw = model.score_batch(&neg_batch, scorer.as_ref());

            // Aggregate negative scores: mean over negatives_per_positive for each positive
            let batch_len = pos_scores.dims()[0];
            let neg_scores = if config.negatives_per_positive > 1 {
                // Reshape neg scores to [batch_len, negatives_per_positive]
                // then take max (hardest negative)
                let reshaped = neg_scores_raw.reshape([batch_len, config.negatives_per_positive]);
                // Use mean of negatives for more stable training
                reshaped.mean_dim(1).reshape([batch_len])
            } else {
                neg_scores_raw
            };

            // Compute loss
            let loss = margin_ranking_loss(pos_scores, neg_scores, config.margin);
            let loss_scalar: f32 = loss
                .clone()
                .into_data()
                .as_slice::<f32>()
                .expect("Failed to read loss")[0];
            logger.record_batch_loss(loss_scalar as f64);

            // Backward pass + optimizer step
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optim.step(config.lr, model, grads_params);
        }

        // Evaluate if needed
        let eval_metrics = if config.eval_every > 0
            && (epoch % config.eval_every == 0 || epoch == config.epochs - 1)
        {
            // Convert model to inner backend for evaluation (no autodiff)
            let inner_model = model.valid();
            let inner_device = inner_model
                .embeddings
                .entity_embedding
                .weight
                .val()
                .device();

            let metrics = crate::eval::evaluate::evaluate_link_prediction(
                &inner_model,
                inner_scorer.as_ref(),
                test_facts,
                train_facts,
                num_entities,
                &inner_device,
            );
            Some(metrics)
        } else {
            None
        };

        let epoch_time = epoch_start.elapsed();
        println!(
            "\r    Epoch {}/{}: {:.2}s, loss={:.4}                      ",
            epoch + 1,
            config.epochs,
            epoch_time.as_secs_f64(),
            logger.history.last().map(|h| h.train_loss).unwrap_or(0.0)
        );
        logger.end_epoch(epoch, config.lr, eval_metrics.as_ref());
    }

    println!("\n  Training complete!");

    // Generate dashboard
    if let Err(e) = logger.generate_dashboard(&dashboard_path) {
        eprintln!("  Warning: Failed to generate dashboard: {}", e);
    }

    // Return the trained inner model
    let inner_model = model.valid();

    TrainResult {
        model: inner_model,
        logger,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::fact::Qualifier;
    use burn::backend::{Autodiff, NdArray};

    type TestAutodiff = Autodiff<NdArray>;

    #[test]
    fn test_train_runs_small() {
        let device = <TestAutodiff as Backend>::Device::default();

        let train_facts = vec![
            HehrFact {
                head: 0,
                relation: 0,
                tail: 1,
                qualifiers: vec![],
            },
            HehrFact {
                head: 2,
                relation: 1,
                tail: 3,
                qualifiers: vec![],
            },
            HehrFact {
                head: 4,
                relation: 0,
                tail: 5,
                qualifiers: vec![Qualifier {
                    relation_id: 2,
                    entity_id: 6,
                }],
            },
            HehrFact {
                head: 1,
                relation: 1,
                tail: 4,
                qualifiers: vec![],
            },
        ];
        let test_facts = vec![HehrFact {
            head: 0,
            relation: 1,
            tail: 3,
            qualifiers: vec![],
        }];

        let config = TrainConfig {
            epochs: 3,
            lr: 0.01,
            margin: 1.0,
            batch_size: 4,
            negatives_per_positive: 2,
            hidden_dim: 8,
            num_layers: 1,
            dropout: 0.0,
            eval_every: 2,
            scorer_type: "distmult".to_string(),
            output_dir: "/tmp/hehrgnn_test_train".to_string(),
        };

        let result = train::<TestAutodiff>(
            &config,
            &train_facts,
            &test_facts,
            8, // num_entities
            3, // num_relations
            &device,
        );

        // Check that training produced results
        assert!(!result.logger.history.is_empty());
        // Check that the last epoch has a loss
        let last = result.logger.history.last().unwrap();
        assert!(last.train_loss.is_finite());

        // Cleanup
        let _ = std::fs::remove_dir_all("/tmp/hehrgnn_test_train");
    }
}
