//! Link prediction evaluation.
//!
//! Vectorized evaluation: instead of creating N candidate facts per test fact,
//! uses direct embedding lookup + matmul to score all entities at once.
//! DistMult: scores = (h ⊙ r) · E^T — single matmul per test fact.

use burn::prelude::*;

use crate::data::fact::HehrFact;
use crate::model::hehrgnn::HehrgnnModel;

use super::metrics::{LinkPredictionMetrics, filtered_rank};

/// Evaluate link prediction using vectorized scoring.
///
/// For each test fact (h, r, t), computes scores against ALL entities
/// using a single matmul: scores = (h_emb ⊙ r_emb) · E_all^T
///
/// This is O(d × N) per test fact instead of O(N × batch_overhead).
pub fn evaluate_link_prediction<B: Backend>(
    model: &HehrgnnModel<B>,
    _scorer: &dyn crate::training::scoring::Scorer<B>,
    test_facts: &[HehrFact],
    all_facts: &[HehrFact],
    num_entities: usize,
    _device: &B::Device,
) -> LinkPredictionMetrics {
    let total = test_facts.len();

    eprintln!(
        "      [Eval] Vectorized evaluation: {} test facts, {} entities (matmul scoring)",
        total, num_entities
    );
    let eval_start = std::time::Instant::now();

    // Get the raw embedding tables directly (no forward pass needed —
    // embeddings were already updated during training)
    let entity_emb = model.embeddings.entity_embedding.weight.val(); // [N, d]
    let relation_emb = model.embeddings.relation_embedding.weight.val(); // [R, d]

    let entity_emb_t = entity_emb.clone().transpose(); // [d, N]

    let mut ranks = Vec::with_capacity(total);

    for (fact_idx, fact) in test_facts.iter().enumerate() {
        if fact_idx % 50 == 0 {
            let elapsed = eval_start.elapsed().as_secs_f64();
            let rate = if fact_idx > 0 {
                fact_idx as f64 / elapsed
            } else {
                0.0
            };
            let eta = if rate > 0.0 {
                (total - fact_idx) as f64 / rate
            } else {
                0.0
            };
            eprintln!(
                "      [Eval] fact {}/{} ({:.0}%) [{:.1}s elapsed, {:.0} facts/s, ETA {:.0}s]",
                fact_idx,
                total,
                fact_idx as f64 / total as f64 * 100.0,
                elapsed,
                rate,
                eta
            );
        }

        // Get head and relation embeddings: [1, d]
        let h_emb = entity_emb.clone().slice([fact.head..fact.head + 1]); // [1, d]
        let r_emb = relation_emb
            .clone()
            .slice([fact.relation..fact.relation + 1]); // [1, d]

        // DistMult vectorized: scores = (h ⊙ r) · E^T
        // h_emb * r_emb = [1, d], entity_emb_t = [d, N]
        // result = [1, N]
        let hr = h_emb * r_emb; // [1, d] element-wise product
        let scores_2d = hr.matmul(entity_emb_t.clone()); // [1, N]
        let scores_1d = scores_2d.reshape([num_entities]); // [N]

        // Convert to Vec<f64>
        let scores_data = scores_1d.into_data();
        let scores_vec: Vec<f64> = scores_data
            .as_slice::<f32>()
            .expect("Failed to read scores")
            .iter()
            .map(|&s| s as f64)
            .collect();

        // Filtered rank: exclude other true tails for this (h, r, ?)
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

    let eval_time = eval_start.elapsed();
    eprintln!(
        "      [Eval] ✅ Done in {:.1}s ({} facts, {:.0} facts/s)",
        eval_time.as_secs_f64(),
        ranks.len(),
        ranks.len() as f64 / eval_time.as_secs_f64()
    );

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
