//! Full HEHRGNN model that stacks multiple GNN layers.
//!
//! Combines embedding layers with N message-passing layers to produce
//! updated entity and relation embeddings for downstream scoring.

use burn::prelude::*;

use super::embedding::{KgEmbedding, KgEmbeddingConfig};
use super::gnn_layer::{GnnLayer, GnnLayerConfig};
use crate::data::batcher::HehrBatch;

/// Configuration for the full HEHRGNN model.
#[derive(Config, Debug)]
pub struct HehrgnnModelConfig {
    /// Number of unique entities.
    pub num_entities: usize,
    /// Number of unique relations.
    pub num_relations: usize,
    /// Embedding / hidden dimension.
    pub hidden_dim: usize,
    /// Number of stacked GNN layers.
    #[config(default = "2")]
    pub num_layers: usize,
    /// Dropout rate within GNN layers.
    #[config(default = "0.1")]
    pub dropout: f64,
}

/// Full HEHRGNN model.
#[derive(Module, Debug)]
pub struct HehrgnnModel<B: Backend> {
    /// Entity and relation embedding tables.
    pub embeddings: KgEmbedding<B>,
    /// Stacked GNN message-passing layers.
    pub gnn_layers: Vec<GnnLayer<B>>,
}

impl HehrgnnModelConfig {
    /// Initialize the model on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> HehrgnnModel<B> {
        let embeddings = KgEmbeddingConfig {
            num_entities: self.num_entities,
            num_relations: self.num_relations,
            hidden_dim: self.hidden_dim,
        }
        .init(device);

        let gnn_layers = (0..self.num_layers)
            .map(|_| {
                GnnLayerConfig {
                    hidden_dim: self.hidden_dim,
                    dropout: self.dropout,
                }
                .init(device)
            })
            .collect();

        HehrgnnModel {
            embeddings,
            gnn_layers,
        }
    }
}

/// Output of the HEHRGNN forward pass.
pub struct HehrgnnOutput<B: Backend> {
    /// Updated entity embeddings: `[num_entities, hidden_dim]`.
    pub entity_emb: Tensor<B, 2>,
    /// Updated relation embeddings: `[num_relations, hidden_dim]`.
    pub relation_emb: Tensor<B, 2>,
}

impl<B: Backend> HehrgnnModel<B> {
    /// Forward pass: run the batch through all GNN layers.
    ///
    /// Returns updated entity and relation embeddings.
    pub fn forward(&self, batch: &HehrBatch<B>) -> HehrgnnOutput<B> {
        // Start with the raw embedding tables
        let mut entity_emb = self.embeddings.entity_embedding.weight.val();
        let mut relation_emb = self.embeddings.relation_embedding.weight.val();

        // Pass through each GNN layer
        for layer in &self.gnn_layers {
            let (ent, rel) = layer.forward(
                entity_emb,
                relation_emb,
                batch.primary_triples.clone(),
                batch.qualifier_entities.clone(),
                batch.qualifier_relations.clone(),
                batch.qualifier_mask.clone(),
            );
            entity_emb = ent;
            relation_emb = rel;
        }

        HehrgnnOutput {
            entity_emb,
            relation_emb,
        }
    }

    /// Score a batch of facts using the given scoring function.
    ///
    /// First runs the forward pass to get updated embeddings, then
    /// computes scores for the primary triples in the batch.
    pub fn score_batch(
        &self,
        batch: &HehrBatch<B>,
        scorer: &dyn crate::training::scoring::Scorer<B>,
    ) -> Tensor<B, 1> {
        let output = self.forward(batch);

        let batch_size = batch.primary_triples.dims()[0];

        // Extract head, relation, tail IDs
        let head_ids = batch
            .primary_triples
            .clone()
            .slice([0..batch_size, 0..1])
            .reshape([batch_size]);
        let rel_ids = batch
            .primary_triples
            .clone()
            .slice([0..batch_size, 1..2])
            .reshape([batch_size]);
        let tail_ids = batch
            .primary_triples
            .clone()
            .slice([0..batch_size, 2..3])
            .reshape([batch_size]);

        // Look up updated embeddings for each triple
        let head_emb = output.entity_emb.clone().select(0, head_ids);
        let rel_emb = output.relation_emb.select(0, rel_ids);
        let tail_emb = output.entity_emb.select(0, tail_ids);

        scorer.score(head_emb, rel_emb, tail_emb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::batcher::{HehrBatcher, HehrFactItem};
    use crate::data::fact::{HehrFact, Qualifier};
    use burn::backend::NdArray;
    use burn::data::dataloader::batcher::Batcher;

    type TestBackend = NdArray;

    #[test]
    fn test_model_forward_shapes() {
        let device = <TestBackend as Backend>::Device::default();

        let num_entities = 20;
        let num_relations = 5;
        let hidden_dim = 8;

        let model = HehrgnnModelConfig {
            num_entities,
            num_relations,
            hidden_dim,
            num_layers: 2,
            dropout: 0.0,
        }
        .init::<TestBackend>(&device);

        // Create a small batch
        let items = vec![
            HehrFactItem {
                fact: HehrFact {
                    head: 0,
                    relation: 0,
                    tail: 1,
                    qualifiers: vec![Qualifier {
                        relation_id: 1,
                        entity_id: 2,
                    }],
                },
                label: 1.0,
            },
            HehrFactItem {
                fact: HehrFact {
                    head: 3,
                    relation: 2,
                    tail: 4,
                    qualifiers: vec![],
                },
                label: 0.0,
            },
        ];

        let batcher = HehrBatcher::new();
        let batch: HehrBatch<TestBackend> = batcher.batch(items, &device);

        let output = model.forward(&batch);

        assert_eq!(output.entity_emb.dims(), [num_entities, hidden_dim]);
        assert_eq!(output.relation_emb.dims(), [num_relations, hidden_dim]);
    }
}
