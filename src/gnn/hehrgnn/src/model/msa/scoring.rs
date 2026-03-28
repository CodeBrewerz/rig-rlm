//! Relevance Scoring and Top-k Document Selection for MSA.
//!
//! MSA Paper §3.2.1, Eq. 2:
//!   S_ij = max_t ( mean_h ( cos(Q^R_{q,h})_t, K̄^R_{ij,h}) ) )
//!
//! Document-level score is the max over its chunks: s_i = max_j S_ij
//! Top-k indices: I = Top-k({s_i}_{i=1}^N)

use burn::prelude::*;

/// Compute cosine similarity between two tensors along the last dimension.
///
/// # Arguments
/// * `a` - Tensor [n, dim]
/// * `b` - Tensor [m, dim]
///
/// # Returns
/// * Similarity matrix [n, m]
pub fn cosine_similarity_matrix<B: Backend>(a: Tensor<B, 2>, b: Tensor<B, 2>) -> Tensor<B, 2> {
    // Normalize along last dimension
    let a_norm = l2_normalize(a);
    let b_norm = l2_normalize(b);
    // [n, dim] × [dim, m] → [n, m]
    a_norm.matmul(b_norm.transpose())
}

/// L2-normalize a tensor along the last dimension.
fn l2_normalize<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let [n, d] = x.dims();
    // Compute L2 norm per row
    let sq = x.clone().powf_scalar(2.0);
    let norm = sq.sum_dim(1).sqrt().clamp_min(1e-8); // [n, 1]
    x / norm.reshape([n, 1]).expand([n, d])
}

/// Compute relevance scores between a query and all document chunks.
///
/// Implements Eq. 2 from the MSA paper:
///   S_ij = max_t ( mean_h ( cos(Q^R_t,h, K̄^R_{ij,h}) ) )
///
/// # Arguments
/// * `query_routing` - Query routing representations [query_len, num_heads * head_dim]
/// * `doc_routing_chunks` - List of (doc_id, chunk routing keys [num_chunks, num_heads * head_dim])
/// * `num_heads` - Number of attention heads for mean aggregation
///
/// # Returns
/// * Vec of (doc_id, relevance_score)
pub fn compute_document_scores<B: Backend>(
    query_routing: Tensor<B, 2>,
    doc_routing_chunks: &[(usize, Tensor<B, 2>)],
    num_heads: usize,
) -> Vec<(usize, f32)> {
    let [query_len, total_dim] = query_routing.dims();
    let head_dim = total_dim / num_heads;

    let mut scores: Vec<(usize, f32)> = Vec::with_capacity(doc_routing_chunks.len());

    for (doc_id, chunk_keys) in doc_routing_chunks {
        let [num_chunks, _] = chunk_keys.dims();

        // Compute per-head cosine similarity, then mean across heads
        // Reshape: [query_len, num_heads, head_dim] and [num_chunks, num_heads, head_dim]
        let mut head_scores_sum =
            Tensor::<B, 2>::zeros([query_len, num_chunks], &query_routing.device());

        for h in 0..num_heads {
            let h_start = h * head_dim;
            let h_end = h_start + head_dim;

            let q_head = query_routing
                .clone()
                .slice([0..query_len, h_start..h_end]); // [query_len, head_dim]
            let k_head = chunk_keys
                .clone()
                .slice([0..num_chunks, h_start..h_end]); // [num_chunks, head_dim]

            let sim = cosine_similarity_matrix(q_head, k_head); // [query_len, num_chunks]
            head_scores_sum = head_scores_sum + sim;
        }

        // Mean across heads
        let mean_sim = head_scores_sum / (num_heads as f32); // [query_len, num_chunks]

        // Max over query tokens (max_t), then max over chunks (max_j for doc-level score)
        let max_over_tokens = mean_sim.max_dim(0); // [1, num_chunks]
        let doc_score = max_over_tokens.max(); // scalar

        let score_val: f32 = doc_score
            .into_data()
            .as_slice::<f32>()
            .unwrap()[0];
        scores.push((*doc_id, score_val));
    }

    scores
}

/// Select Top-k documents by relevance score.
///
/// # Arguments
/// * `scores` - Vec of (doc_id, score)
/// * `k` - Number of documents to select
///
/// # Returns
/// * Vec of selected (doc_id, score), sorted by score descending
pub fn topk_select(mut scores: Vec<(usize, f32)>, k: usize) -> Vec<(usize, f32)> {
    // Sort by score descending
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(k);
    scores
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn test_cosine_similarity_identity() {
        let device = <B as Backend>::Device::default();
        let a = Tensor::<B, 2>::from_data([[1.0f32, 0.0], [0.0, 1.0]], &device);
        let sim = cosine_similarity_matrix(a.clone(), a);
        let data: Vec<f32> = sim.into_data().as_slice::<f32>().unwrap().to_vec();
        // Diagonal should be ~1.0, off-diagonal ~0.0
        assert!((data[0] - 1.0).abs() < 1e-5, "self-sim should be 1.0");
        assert!(data[1].abs() < 1e-5, "orthogonal should be 0.0");
        println!("✅ Cosine similarity: identity verified");
    }

    #[test]
    fn test_topk_select() {
        let scores = vec![(0, 0.5f32), (1, 0.9), (2, 0.3), (3, 0.7), (4, 0.1)];
        let top3 = topk_select(scores, 3);
        assert_eq!(top3.len(), 3);
        assert_eq!(top3[0].0, 1); // highest score
        assert_eq!(top3[1].0, 3);
        assert_eq!(top3[2].0, 0);
        println!("✅ Top-k selection: correct ordering");
    }

    #[test]
    fn test_document_scoring() {
        let device = <B as Backend>::Device::default();
        let num_heads = 2;
        let head_dim = 4;
        let total_dim = num_heads * head_dim;

        // Query: 3 tokens
        let query = Tensor::<B, 2>::random(
            [3, total_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        // 3 docs with different chunk counts
        let doc0 = Tensor::<B, 2>::random(
            [2, total_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let doc1 = query.clone().slice([0..1, 0..total_dim]).repeat_dim(0, 2); // similar to query
        let doc2 = Tensor::<B, 2>::random(
            [3, total_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let docs = vec![(0, doc0), (1, doc1), (2, doc2)];
        let scores = compute_document_scores(query, &docs, num_heads);

        assert_eq!(scores.len(), 3);
        // Doc 1 should have highest score (it's a copy of query)
        let top = topk_select(scores.clone(), 1);
        assert_eq!(top[0].0, 1, "Doc copied from query should rank highest");
        println!("✅ Document scoring: similar doc ranks first (score={})", top[0].1);
    }
}
