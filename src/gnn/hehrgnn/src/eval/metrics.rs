//! Evaluation metrics for knowledge graph link prediction.
//!
//! Implements the standard evaluation protocol:
//! - **Filtered ranking** — true facts in the training set are excluded from
//!   being counted as false negatives.
//! - **MRR** (Mean Reciprocal Rank)
//! - **Hits@K** for K ∈ {1, 3, 10}

/// Compute Mean Reciprocal Rank (MRR) from a list of ranks.
///
/// MRR = (1/|Q|) Σ (1 / rank_i)
///
/// Ranks are 1-indexed (rank 1 = best).
pub fn mean_reciprocal_rank(ranks: &[usize]) -> f64 {
    if ranks.is_empty() {
        return 0.0;
    }
    let sum: f64 = ranks.iter().map(|&r| 1.0 / r as f64).sum();
    sum / ranks.len() as f64
}

/// Compute Hits@K: fraction of predictions where the true entity
/// was ranked within the top K positions.
///
/// Ranks are 1-indexed (rank 1 = best).
pub fn hits_at_k(ranks: &[usize], k: usize) -> f64 {
    if ranks.is_empty() {
        return 0.0;
    }
    let hits = ranks.iter().filter(|&&r| r <= k).count();
    hits as f64 / ranks.len() as f64
}

/// A collection of standard link prediction metrics.
#[derive(Debug, Clone)]
pub struct LinkPredictionMetrics {
    pub mrr: f64,
    pub hits_at_1: f64,
    pub hits_at_3: f64,
    pub hits_at_10: f64,
    pub mean_rank: f64,
}

impl LinkPredictionMetrics {
    /// Compute all metrics from a list of ranks.
    pub fn from_ranks(ranks: &[usize]) -> Self {
        let mrr = mean_reciprocal_rank(ranks);
        let hits_at_1 = hits_at_k(ranks, 1);
        let hits_at_3 = hits_at_k(ranks, 3);
        let hits_at_10 = hits_at_k(ranks, 10);
        let mean_rank = if ranks.is_empty() {
            0.0
        } else {
            ranks.iter().sum::<usize>() as f64 / ranks.len() as f64
        };

        Self {
            mrr,
            hits_at_1,
            hits_at_3,
            hits_at_10,
            mean_rank,
        }
    }
}

impl std::fmt::Display for LinkPredictionMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MRR: {:.4} | Hits@1: {:.4} | Hits@3: {:.4} | Hits@10: {:.4} | MeanRank: {:.1}",
            self.mrr, self.hits_at_1, self.hits_at_3, self.hits_at_10, self.mean_rank
        )
    }
}

/// Compute the filtered rank for a target entity.
///
/// Given a vector of scores (one per entity), the target entity index,
/// and a set of other true entity indices (to be filtered out),
/// returns the 1-indexed rank of the target entity.
///
/// In the **filtered setting**, entities that are known true answers
/// (other than the target) do not count against the rank.
pub fn filtered_rank(scores: &[f64], target_idx: usize, true_entity_indices: &[usize]) -> usize {
    let target_score = scores[target_idx];
    let mut rank = 1usize;

    for (idx, &score) in scores.iter().enumerate() {
        if idx == target_idx {
            continue;
        }
        // Skip other true entities (filtered setting)
        if true_entity_indices.contains(&idx) {
            continue;
        }
        // Count entities that score higher than the target
        if score > target_score {
            rank += 1;
        }
    }

    rank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mrr() {
        // Ranks: [1, 2, 5] → MRR = (1/1 + 1/2 + 1/5) / 3
        let ranks = vec![1, 2, 5];
        let mrr = mean_reciprocal_rank(&ranks);
        let expected = (1.0 + 0.5 + 0.2) / 3.0;
        assert!((mrr - expected).abs() < 1e-6);
    }

    #[test]
    fn test_hits_at_k() {
        let ranks = vec![1, 2, 5, 8, 15];
        assert!((hits_at_k(&ranks, 1) - 0.2).abs() < 1e-6); // 1 hit
        assert!((hits_at_k(&ranks, 3) - 0.4).abs() < 1e-6); // 2 hits
        assert!((hits_at_k(&ranks, 10) - 0.8).abs() < 1e-6); // 4 hits
    }

    #[test]
    fn test_filtered_rank() {
        // 5 entities, target is idx=2
        let scores = vec![0.8, 0.3, 0.6, 0.9, 0.5];
        // Without filtering: rank of idx=2 (score 0.6) → entities 0 (0.8) and 3 (0.9) score higher → rank 3
        let rank = filtered_rank(&scores, 2, &[]);
        assert_eq!(rank, 3);

        // With filtering: entity 0 is a known true answer → skip it → rank 2
        let rank_filtered = filtered_rank(&scores, 2, &[0]);
        assert_eq!(rank_filtered, 2);
    }

    #[test]
    fn test_link_prediction_metrics() {
        let ranks = vec![1, 3, 10, 50, 100];
        let metrics = LinkPredictionMetrics::from_ranks(&ranks);

        assert!((metrics.hits_at_1 - 0.2).abs() < 1e-6);
        assert!((metrics.hits_at_3 - 0.4).abs() < 1e-6);
        assert!((metrics.hits_at_10 - 0.6).abs() < 1e-6);
        assert!((metrics.mean_rank - 32.8).abs() < 1e-6);
    }
}
