//! Memory Bank: KV Cache Store for MSA Inference.
//!
//! MSA Paper §3.4: Three-Stage Inference Process.
//!
//! Stage 1 (Offline): Encode all documents → cache compressed K̄, V̄, K̄^R
//! Stage 2 (Online): Route query Q^R against cached K̄^R → select Top-k
//! Stage 3 (Online): Attend to selected K̄, V̄ + query local KV → generate

use burn::prelude::*;

use super::scoring::{compute_document_scores, topk_select};
use super::sparse_attn::MsaLayer;

/// Cached representation of a single document in the memory bank.
#[derive(Debug)]
pub struct DocumentCache<B: Backend> {
    /// Document ID.
    pub doc_id: usize,
    /// Compressed Key matrix K̄ [num_chunks, hidden_dim]
    pub k_bar: Tensor<B, 2>,
    /// Compressed Value matrix V̄ [num_chunks, hidden_dim]
    pub v_bar: Tensor<B, 2>,
    /// Compressed Routing Key matrix K̄^R [num_chunks, router_total_dim]
    pub kr_bar: Tensor<B, 2>,
    /// Number of chunks in this document.
    pub num_chunks: usize,
    /// Original document length in tokens.
    pub original_len: usize,
}

/// Memory Bank: stores compressed KV caches for all documents.
///
/// Paper §3.4.1: "This stage converts the raw text corpus into a structured,
/// retrievable latent store."
pub struct MemoryBank<B: Backend> {
    /// All cached documents.
    documents: Vec<DocumentCache<B>>,
    /// Number of attention heads (for scoring).
    num_heads: usize,
    /// Top-k for retrieval.
    topk: usize,
}

impl<B: Backend> MemoryBank<B> {
    /// Create an empty memory bank.
    pub fn new(num_heads: usize, topk: usize) -> Self {
        Self {
            documents: Vec::new(),
            num_heads,
            topk,
        }
    }

    /// Stage 1: Encode a document and add to the memory bank.
    ///
    /// Paper: "For every document, the model performs a forward pass to generate
    /// the standard K and V matrices. Simultaneously, the specialized Router K
    /// Projector generates the routing key matrix K^R."
    ///
    /// # Arguments
    /// * `doc_id` - Unique document identifier
    /// * `doc_hidden` - Document hidden states [doc_len, hidden_dim]
    /// * `layer` - MSA layer to use for encoding
    pub fn encode_document(
        &mut self,
        doc_id: usize,
        doc_hidden: Tensor<B, 2>,
        layer: &MsaLayer<B>,
    ) {
        let original_len = doc_hidden.dims()[0];
        let (k_bar, v_bar, kr_bar) = layer.encode_document(doc_hidden);
        let num_chunks = k_bar.dims()[0];

        self.documents.push(DocumentCache {
            doc_id,
            k_bar,
            v_bar,
            kr_bar,
            num_chunks,
            original_len,
        });
    }

    /// Stage 2: Route a query against the memory bank and select Top-k documents.
    ///
    /// Paper: "First, the model computes the question's hidden states and projects
    /// them via the Router Q Projector to obtain the routing query Q^R_q. This query
    /// is then matched against the cached global routing keys K̄^R."
    ///
    /// # Arguments
    /// * `query_routing` - Query routing representation Q^R [query_len, router_total_dim]
    ///
    /// # Returns
    /// * Vec of (doc_id, score) for the top-k documents
    pub fn route(&self, query_routing: Tensor<B, 2>) -> Vec<(usize, f32)> {
        let doc_chunks: Vec<(usize, Tensor<B, 2>)> = self
            .documents
            .iter()
            .map(|d| (d.doc_id, d.kr_bar.clone()))
            .collect();

        let scores = compute_document_scores(query_routing, &doc_chunks, self.num_heads);
        topk_select(scores, self.topk)
    }

    /// Retrieve the compressed K and V caches for selected documents.
    ///
    /// Paper: "Only the compact Key and Value matrices (K̄, V̄) of these selected
    /// documents are loaded."
    ///
    /// # Arguments
    /// * `selected_ids` - Document IDs to retrieve
    ///
    /// # Returns
    /// * `(concatenated_K̄, concatenated_V̄)` — [total_chunks, hidden_dim] each
    pub fn retrieve_kv(&self, selected_ids: &[usize]) -> Option<(Tensor<B, 2>, Tensor<B, 2>)> {
        let mut k_parts: Vec<Tensor<B, 2>> = Vec::new();
        let mut v_parts: Vec<Tensor<B, 2>> = Vec::new();

        for &doc_id in selected_ids {
            if let Some(doc) = self.documents.iter().find(|d| d.doc_id == doc_id) {
                k_parts.push(doc.k_bar.clone());
                v_parts.push(doc.v_bar.clone());
            }
        }

        if k_parts.is_empty() {
            return None;
        }

        Some((
            Tensor::cat(k_parts, 0),
            Tensor::cat(v_parts, 0),
        ))
    }

    /// Full retrieval pipeline: route + retrieve.
    ///
    /// Combines Stage 2 routing with KV retrieval.
    ///
    /// # Returns
    /// * `(memory_k, memory_v, selected_doc_ids)`
    pub fn route_and_retrieve(
        &self,
        query_routing: Tensor<B, 2>,
    ) -> Option<(Tensor<B, 2>, Tensor<B, 2>, Vec<usize>)> {
        let selected = self.route(query_routing);
        let ids: Vec<usize> = selected.iter().map(|(id, _)| *id).collect();

        if let Some((k, v)) = self.retrieve_kv(&ids) {
            Some((k, v, ids))
        } else {
            None
        }
    }

    /// Number of documents in the memory bank.
    pub fn num_documents(&self) -> usize {
        self.documents.len()
    }

    /// Total number of chunks across all documents.
    pub fn total_chunks(&self) -> usize {
        self.documents.iter().map(|d| d.num_chunks).sum()
    }

    /// Total original tokens represented.
    pub fn total_tokens(&self) -> usize {
        self.documents.iter().map(|d| d.original_len).sum()
    }

    /// Clear all cached documents.
    pub fn clear(&mut self) {
        self.documents.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::sparse_attn::MsaLayerConfig;
    use burn::backend::NdArray;

    type B = NdArray;

    fn make_layer(device: &<B as Backend>::Device) -> MsaLayer<B> {
        let config = MsaLayerConfig {
            hidden_dim: 64,
            num_heads: 4,
            router_dim: 8,
            chunk_size: 16,
            topk: 2,
            ffn_ratio: 2,
        };
        MsaLayer::<B>::new(&config, device)
    }

    #[test]
    fn test_memory_bank_encode() {
        let device = <B as Backend>::Device::default();
        let layer = make_layer(&device);
        let mut bank = MemoryBank::<B>::new(4, 2);

        // Encode 3 documents of different lengths
        for i in 0..3 {
            let doc_len = 64 + i * 32;
            let doc = Tensor::<B, 2>::random(
                [doc_len, 64],
                burn::tensor::Distribution::Normal(0.0, 0.1),
                &device,
            );
            bank.encode_document(i, doc, &layer);
        }

        assert_eq!(bank.num_documents(), 3);
        println!(
            "✅ MemoryBank encode: {} docs, {} chunks, {} tokens",
            bank.num_documents(),
            bank.total_chunks(),
            bank.total_tokens()
        );
    }

    #[test]
    fn test_memory_bank_route_and_retrieve() {
        let device = <B as Backend>::Device::default();
        let layer = make_layer(&device);
        let mut bank = MemoryBank::<B>::new(4, 2);

        // Encode 5 documents
        let mut doc_hiddens = Vec::new();
        for i in 0..5 {
            let doc = Tensor::<B, 2>::random(
                [64, 64],
                burn::tensor::Distribution::Normal(0.0, 0.1),
                &device,
            );
            doc_hiddens.push(doc.clone());
            bank.encode_document(i, doc, &layer);
        }

        // Use doc 2's hidden as query (should retrieve doc 2)
        let query_routing = layer.compute_routing_query(
            doc_hiddens[2].clone().slice([0..5, 0..64]),
        );

        let result = bank.route_and_retrieve(query_routing);
        assert!(result.is_some(), "Should retrieve documents");

        let (mem_k, mem_v, ids) = result.unwrap();
        assert_eq!(ids.len(), 2, "Should retrieve top-2 documents");
        assert!(
            ids.contains(&2),
            "Doc 2 (similar to query) should be in top-2, got {:?}",
            ids
        );

        println!(
            "✅ MemoryBank route+retrieve: top-2 = {:?}, mem_k={:?}, mem_v={:?}",
            ids,
            mem_k.dims(),
            mem_v.dims()
        );
    }

    #[test]
    fn test_memory_bank_empty() {
        let bank = MemoryBank::<B>::new(4, 2);
        assert_eq!(bank.num_documents(), 0);
        assert_eq!(bank.total_chunks(), 0);

        let device = <B as Backend>::Device::default();
        let qr = Tensor::<B, 2>::random(
            [3, 32],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let result = bank.route_and_retrieve(qr);
        // Empty bank routing should return empty scores → None
        assert!(result.is_none() || {
            let (_, _, ids) = result.unwrap();
            ids.is_empty()
        });
        println!("✅ MemoryBank: empty bank handled gracefully");
    }
}
