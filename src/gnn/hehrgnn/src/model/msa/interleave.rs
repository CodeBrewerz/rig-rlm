//! Memory Interleave mechanism for multi-hop reasoning.
//!
//! MSA Paper §3.5: "MSA incorporates an adaptive Memory Interleave Mechanism
//! that essentially performs the routing and context assembly (Stage 2) and
//! Sparse Generation (Stage 3) in an iterative manner."
//!
//! Unlike single-shot retrieval, the inference process alternates between:
//! 1. Generative Retrieval — model generates document IDs
//! 2. Context Expansion — retrieved documents appended to query
//!
//! The cycle repeats adaptively until the model generates a final answer.

use burn::prelude::*;

use super::memory_bank::MemoryBank;
use super::sparse_attn::MsaLayer;

/// Result of a single interleave step.
#[derive(Debug)]
pub struct InterleaveStep {
    /// Document IDs retrieved in this step.
    pub retrieved_doc_ids: Vec<usize>,
    /// Relevance scores for retrieved documents.
    pub scores: Vec<f32>,
    /// Step number (0-indexed).
    pub step: usize,
}

/// Configuration for the Memory Interleave mechanism.
#[derive(Debug, Clone)]
pub struct InterleaveConfig {
    /// Maximum number of interleave iterations.
    pub max_rounds: usize,
    /// Top-k documents per round.
    pub topk_per_round: usize,
    /// Minimum relevance score to continue interleaving.
    pub min_score_threshold: f32,
}

impl Default for InterleaveConfig {
    fn default() -> Self {
        Self {
            max_rounds: 5,
            topk_per_round: 4,
            min_score_threshold: 0.1,
        }
    }
}

/// Memory Interleave: iterative retrieval for multi-hop reasoning.
///
/// Paper §3.5: "After loading the KV-cache for the document corpus, the model
/// first autoregressively generates a sequence of document IDs ending with a
/// special delimiter based on the given query."
pub struct MemoryInterleave {
    /// Interleave configuration.
    pub config: InterleaveConfig,
}

impl MemoryInterleave {
    /// Create a new Memory Interleave mechanism.
    pub fn new(config: InterleaveConfig) -> Self {
        Self { config }
    }

    /// Run the interleaved retrieval loop.
    ///
    /// Each round:
    /// 1. Compute routing query from current accumulated context
    /// 2. Route against memory bank to get top-k
    /// 3. Retrieve compressed KV for selected documents
    /// 4. Expand context with retrieved document representations
    ///
    /// Returns when:
    /// - Max rounds reached
    /// - No new documents found (scores below threshold)
    /// - All documents already retrieved
    ///
    /// # Arguments
    /// * `initial_query` - Initial query hidden states [query_len, hidden_dim]
    /// * `bank` - Memory bank with encoded documents
    /// * `layer` - MSA layer for routing and generation
    ///
    /// # Returns
    /// * `(final_mem_k, final_mem_v, steps)` — accumulated KV context and retrieval log
    pub fn run<B: Backend>(
        &self,
        initial_query: Tensor<B, 2>,
        bank: &MemoryBank<B>,
        layer: &MsaLayer<B>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Vec<InterleaveStep>) {
        let device = initial_query.device();
        let [_q_len, hidden_dim] = initial_query.dims();

        let mut current_context = initial_query;
        let mut all_retrieved: Vec<usize> = Vec::new();
        let mut steps: Vec<InterleaveStep> = Vec::new();
        let mut accumulated_k: Vec<Tensor<B, 2>> = Vec::new();
        let mut accumulated_v: Vec<Tensor<B, 2>> = Vec::new();

        for round in 0..self.config.max_rounds {
            // Compute routing query from current context
            let routing_query = layer.compute_routing_query(current_context.clone());

            // Route against memory bank
            let candidates = bank.route(routing_query);

            // Filter out already-retrieved documents and below-threshold scores
            let new_candidates: Vec<(usize, f32)> = candidates
                .into_iter()
                .filter(|(id, score)| {
                    !all_retrieved.contains(id) && *score >= self.config.min_score_threshold
                })
                .take(self.config.topk_per_round)
                .collect();

            if new_candidates.is_empty() {
                break; // No new relevant documents
            }

            let new_ids: Vec<usize> = new_candidates.iter().map(|(id, _)| *id).collect();
            let new_scores: Vec<f32> = new_candidates.iter().map(|(_, s)| *s).collect();

            // Retrieve KV for new documents
            if let Some((k, v)) = bank.retrieve_kv(&new_ids) {
                accumulated_k.push(k.clone());
                accumulated_v.push(v.clone());

                // Expand context: append retrieved document representations
                // In the paper, the original document text is appended. Here we
                // use the compressed KV as a proxy for context expansion.
                current_context = Tensor::cat(vec![current_context, k], 0);
            }

            all_retrieved.extend(&new_ids);
            steps.push(InterleaveStep {
                retrieved_doc_ids: new_ids,
                scores: new_scores,
                step: round,
            });
        }

        // Assemble final accumulated KV
        let final_k = if accumulated_k.is_empty() {
            Tensor::<B, 2>::zeros([0, hidden_dim], &device)
        } else {
            Tensor::cat(accumulated_k, 0)
        };

        let final_v = if accumulated_v.is_empty() {
            Tensor::<B, 2>::zeros([0, hidden_dim], &device)
        } else {
            Tensor::cat(accumulated_v, 0)
        };

        (final_k, final_v, steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::sparse_attn::MsaLayerConfig;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn test_interleave_multi_hop() {
        let device = <B as Backend>::Device::default();

        let layer_config = MsaLayerConfig {
            hidden_dim: 64,
            num_heads: 4,
            router_dim: 8,
            chunk_size: 16,
            topk: 2,
            ffn_ratio: 2,
        };
        let layer = MsaLayer::<B>::new(&layer_config, &device);

        let mut bank = MemoryBank::<B>::new(4, 4);

        // Encode 8 documents
        for i in 0..8 {
            let doc = Tensor::<B, 2>::random(
                [64, 64],
                burn::tensor::Distribution::Normal(0.0, 0.1),
                &device,
            );
            bank.encode_document(i, doc, &layer);
        }

        // Run interleaved retrieval
        let query = Tensor::<B, 2>::random(
            [10, 64],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );

        let interleave = MemoryInterleave::new(InterleaveConfig {
            max_rounds: 3,
            topk_per_round: 2,
            min_score_threshold: -1.0, // accept all
        });

        let (final_k, final_v, steps) = interleave.run(query, &bank, &layer);

        assert!(!steps.is_empty(), "Should have at least one retrieval step");

        // Each round retrieves 2 new docs, so after 3 rounds we should have 6
        let total_retrieved: usize = steps.iter().map(|s| s.retrieved_doc_ids.len()).sum();

        println!("✅ Interleave multi-hop:");
        for step in &steps {
            println!(
                "   Round {}: retrieved {:?}, scores {:?}",
                step.step, step.retrieved_doc_ids, step.scores
            );
        }
        println!(
            "   Total retrieved: {}, final_k={:?}, final_v={:?}",
            total_retrieved,
            final_k.dims(),
            final_v.dims()
        );

        // Verify no duplicate retrievals
        let mut all_ids: Vec<usize> = steps
            .iter()
            .flat_map(|s| s.retrieved_doc_ids.iter().copied())
            .collect();
        let orig_len = all_ids.len();
        all_ids.sort();
        all_ids.dedup();
        assert_eq!(
            all_ids.len(),
            orig_len,
            "No duplicate documents should be retrieved across rounds"
        );
    }

    #[test]
    fn test_interleave_early_stop() {
        let device = <B as Backend>::Device::default();

        let layer_config = MsaLayerConfig {
            hidden_dim: 64,
            num_heads: 4,
            router_dim: 8,
            chunk_size: 16,
            topk: 2,
            ffn_ratio: 2,
        };
        let layer = MsaLayer::<B>::new(&layer_config, &device);

        let mut bank = MemoryBank::<B>::new(4, 2);

        // Only 2 documents — should stop after 1 round
        for i in 0..2 {
            let doc = Tensor::<B, 2>::random(
                [64, 64],
                burn::tensor::Distribution::Normal(0.0, 0.1),
                &device,
            );
            bank.encode_document(i, doc, &layer);
        }

        let query = Tensor::<B, 2>::random(
            [5, 64],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );

        let interleave = MemoryInterleave::new(InterleaveConfig {
            max_rounds: 10,
            topk_per_round: 3,
            min_score_threshold: -1.0,
        });

        let (_, _, steps) = interleave.run(query, &bank, &layer);

        // Should stop after 1 round since there are only 2 docs
        assert!(
            steps.len() <= 2,
            "Should stop early with only 2 docs, got {} rounds",
            steps.len()
        );
        println!("✅ Interleave early stop: {} rounds for 2 docs", steps.len());
    }
}
