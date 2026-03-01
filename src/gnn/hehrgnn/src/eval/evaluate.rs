//! Link prediction evaluation.
//!
//! For each test fact, corrupts head/tail with every entity in the vocabulary,
//! scores all candidates, and computes filtered ranks to produce MRR/Hits@K.

use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;

use crate::data::batcher::{HehrBatch, HehrBatcher, HehrFactItem};
use crate::data::fact::HehrFact;
use crate::model::hehrgnn::HehrgnnModel;
use crate::training::scoring::Scorer;

use super::metrics::{filtered_rank, LinkPredictionMetrics};

/// Evaluate link prediction on a set of test facts.
///
/// For each test fact, replaces the tail with every entity and scores all
/// candidates to compute the filtered rank of the true tail. Then aggregates
/// into standard metrics (MRR, Hits@K).
///
/// # Arguments
/// - `model`: trained HEHRGNN model
/// - `scorer`: scoring function (TransE, DistMult, etc.)
/// - `test_facts`: facts to evaluate
/// - `all_facts`: all known facts (train + test) for filtering
/// - `num_entities`: total entities in vocabulary
/// - `device`: compute device
///
/// # Returns
/// `LinkPredictionMetrics` with MRR, Hits@1/3/10, Mean Rank.
pub fn evaluate_link_prediction<B: Backend>(
    model: &HehrgnnModel<B>,
    scorer: &dyn Scorer<B>,
    test_facts: &[HehrFact],
    all_facts: &[HehrFact],
    num_entities: usize,
    device: &B::Device,
) -> LinkPredictionMetrics {
    let batcher = HehrBatcher::new();
    let mut ranks = Vec::with_capacity(test_facts.len());

    for fact in test_facts {
        // --- Tail prediction ---
        // Build candidates: replace tail with each entity
        let mut candidate_items: Vec<HehrFactItem> = Vec::with_capacity(num_entities);
        for ent_id in 0..num_entities {
            let mut corrupted = fact.clone();
            corrupted.tail = ent_id;
            candidate_items.push(HehrFactItem {
                fact: corrupted,
                label: 0.0,
            });
        }

        // Score all candidates in a single batch
        let batch: HehrBatch<B> = batcher.batch(candidate_items, device);
        let scores_tensor = model.score_batch(&batch, scorer);

        // Convert scores to Vec<f64>
        let scores_data = scores_tensor.into_data();
        let scores_vec: Vec<f64> = scores_data
            .as_slice::<f32>()
            .expect("Failed to read scores")
            .iter()
            .map(|&s| s as f64)
            .collect();

        // Find other true tails for this (head, rel, ?, qualifiers) — filtering
        let true_tails: Vec<usize> = all_facts
            .iter()
            .filter(|f| {
                f.head == fact.head
                    && f.relation == fact.relation
                    && f.tail != fact.tail
                    && f.qualifiers == fact.qualifiers
            })
            .map(|f| f.tail)
            .collect();

        let rank = filtered_rank(&scores_vec, fact.tail, &true_tails);
        ranks.push(rank);
    }

    LinkPredictionMetrics::from_ranks(&ranks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::hehrgnn::HehrgnnModelConfig;
    use crate::training::scoring::DistMultScorer;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_evaluate_link_prediction_runs() {
        let device = <TestBackend as Backend>::Device::default();

        let num_entities = 10;
        let num_relations = 3;
        let hidden_dim = 8;

        let model = HehrgnnModelConfig {
            num_entities,
            num_relations,
            hidden_dim,
            num_layers: 1,
            dropout: 0.0,
        }
        .init::<TestBackend>(&device);

        let scorer = DistMultScorer::new();

        let test_facts = vec![
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
        ];

        let all_facts = test_facts.clone();

        let metrics = evaluate_link_prediction(
            &model,
            &scorer,
            &test_facts,
            &all_facts,
            num_entities,
            &device,
        );

        // Just check it runs and produces valid metrics
        assert!(metrics.mrr >= 0.0 && metrics.mrr <= 1.0);
        assert!(metrics.hits_at_1 >= 0.0 && metrics.hits_at_1 <= 1.0);
        assert!(metrics.mean_rank >= 1.0);
    }
}
