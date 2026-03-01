//! GNN backbone trait: unified interface for graph neural network architectures.
//!
//! All GNN architectures (GraphSAGE, RGCN, RGT) implement this trait so that
//! task heads can be plugged onto any backbone interchangeably.

use burn::prelude::*;
use std::collections::HashMap;

use crate::data::hetero_graph::NodeType;

/// Stores node embeddings produced by a GNN backbone, keyed by node type.
#[derive(Debug, Clone)]
pub struct NodeEmbeddings<B: Backend> {
    /// Embeddings per node type: `[num_nodes_of_type, hidden_dim]`.
    pub embeddings: HashMap<NodeType, Tensor<B, 2>>,
}

impl<B: Backend> NodeEmbeddings<B> {
    pub fn new() -> Self {
        Self {
            embeddings: HashMap::new(),
        }
    }

    /// Get embeddings for a specific node type.
    pub fn get(&self, node_type: &str) -> Option<&Tensor<B, 2>> {
        self.embeddings.get(node_type)
    }

    /// Get embeddings for specific node indices within a type.
    pub fn select(&self, node_type: &str, indices: Tensor<B, 1, Int>) -> Option<Tensor<B, 2>> {
        self.embeddings
            .get(node_type)
            .map(|emb| emb.clone().select(0, indices))
    }

    /// Insert embeddings for a node type.
    pub fn insert(&mut self, node_type: &str, embeddings: Tensor<B, 2>) {
        self.embeddings.insert(node_type.to_string(), embeddings);
    }
}
