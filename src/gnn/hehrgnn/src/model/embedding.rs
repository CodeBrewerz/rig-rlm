//! Embedding layers for entities and relations.
//!
//! Wraps Burn's `nn::Embedding` module with config structs
//! following Burn's `Module` / `Config` derive patterns.

use burn::nn;
use burn::prelude::*;

/// Configuration for the KG embedding layers.
#[derive(Config, Debug)]
pub struct KgEmbeddingConfig {
    /// Number of unique entities.
    pub num_entities: usize,
    /// Number of unique relations.
    pub num_relations: usize,
    /// Dimensionality of embedding vectors.
    pub hidden_dim: usize,
}

/// Entity and relation embedding tables.
#[derive(Module, Debug)]
pub struct KgEmbedding<B: Backend> {
    /// Entity embedding lookup table.
    pub entity_embedding: nn::Embedding<B>,
    /// Relation embedding lookup table.
    pub relation_embedding: nn::Embedding<B>,
}

impl KgEmbeddingConfig {
    /// Initialize the embedding tables on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> KgEmbedding<B> {
        let entity_embedding =
            nn::EmbeddingConfig::new(self.num_entities, self.hidden_dim).init(device);
        let relation_embedding =
            nn::EmbeddingConfig::new(self.num_relations, self.hidden_dim).init(device);

        KgEmbedding {
            entity_embedding,
            relation_embedding,
        }
    }
}

impl<B: Backend> KgEmbedding<B> {
    /// Look up entity embeddings: `indices: [*, N]` → `[*, N, hidden_dim]`.
    pub fn embed_entities(&self, indices: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.entity_embedding.forward(indices)
    }

    /// Look up relation embeddings: `indices: [*, N]` → `[*, N, hidden_dim]`.
    pub fn embed_relations(&self, indices: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.relation_embedding.forward(indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_embedding_shapes() {
        let device = <TestBackend as Backend>::Device::default();
        let config = KgEmbeddingConfig {
            num_entities: 100,
            num_relations: 20,
            hidden_dim: 32,
        };
        let embedding = config.init::<TestBackend>(&device);

        // Batch of 4 entities, each referencing 3 entity IDs
        let indices = Tensor::<TestBackend, 2, Int>::from_data(
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
            &device,
        );
        let result = embedding.embed_entities(indices);
        assert_eq!(result.dims(), [4, 3, 32]);

        // Batch of 4, each referencing 1 relation ID
        let rel_indices = Tensor::<TestBackend, 2, Int>::from_data([[0], [1], [2], [3]], &device);
        let rel_result = embedding.embed_relations(rel_indices);
        assert_eq!(rel_result.dims(), [4, 1, 32]);
    }
}
