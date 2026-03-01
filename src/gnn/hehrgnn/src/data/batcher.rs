//! Batcher for creating Burn-compatible training batches from HEHR facts.
//!
//! Implements Burn's `Batcher` trait to convert a vector of fact items
//! into padded tensors suitable for the GNN forward pass.

use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;

use super::fact::HehrFact;

/// A single item fed to the batcher.
///
/// Contains the indexed fact and a label (1.0 for positive, 0.0 for negative).
#[derive(Debug, Clone)]
pub struct HehrFactItem {
    pub fact: HehrFact,
    pub label: f32,
}

/// A collated batch of HEHR facts, ready for the GNN.
#[derive(Debug, Clone)]
pub struct HehrBatch<B: Backend> {
    /// Primary triples: `[batch_size, 3]` — columns are (head, relation, tail).
    pub primary_triples: Tensor<B, 2, Int>,
    /// Qualifier relation IDs: `[batch_size, max_qualifiers]` (0-padded).
    pub qualifier_relations: Tensor<B, 2, Int>,
    /// Qualifier entity IDs: `[batch_size, max_qualifiers]` (0-padded).
    pub qualifier_entities: Tensor<B, 2, Int>,
    /// Qualifier mask: `[batch_size, max_qualifiers]` — 1.0 where valid, 0.0 for padding.
    pub qualifier_mask: Tensor<B, 2>,
    /// Labels: `[batch_size]` — 1.0 positive, 0.0 negative.
    pub labels: Tensor<B, 1>,
}

/// Batcher that collates `HehrFactItem`s into a padded `HehrBatch`.
#[derive(Debug, Clone)]
pub struct HehrBatcher;

impl HehrBatcher {
    pub fn new() -> Self {
        Self
    }
}

impl Default for HehrBatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Batcher<B, HehrFactItem, HehrBatch<B>> for HehrBatcher {
    fn batch(&self, items: Vec<HehrFactItem>, device: &B::Device) -> HehrBatch<B> {
        let batch_size = items.len();
        let max_qualifiers = items
            .iter()
            .map(|item| item.fact.qualifiers.len())
            .max()
            .unwrap_or(0)
            // Ensure at least 1 to avoid zero-dim tensors
            .max(1);

        // Build flat vectors for primary triples
        let mut triples_data: Vec<i64> = Vec::with_capacity(batch_size * 3);
        let mut qual_rel_data: Vec<i64> = Vec::with_capacity(batch_size * max_qualifiers);
        let mut qual_ent_data: Vec<i64> = Vec::with_capacity(batch_size * max_qualifiers);
        let mut mask_data: Vec<f32> = Vec::with_capacity(batch_size * max_qualifiers);
        let mut label_data: Vec<f32> = Vec::with_capacity(batch_size);

        for item in &items {
            let f = &item.fact;

            // Primary triple
            triples_data.push(f.head as i64);
            triples_data.push(f.relation as i64);
            triples_data.push(f.tail as i64);

            // Qualifiers (pad to max_qualifiers)
            for i in 0..max_qualifiers {
                if i < f.qualifiers.len() {
                    qual_rel_data.push(f.qualifiers[i].relation_id as i64);
                    qual_ent_data.push(f.qualifiers[i].entity_id as i64);
                    mask_data.push(1.0);
                } else {
                    qual_rel_data.push(0);
                    qual_ent_data.push(0);
                    mask_data.push(0.0);
                }
            }

            label_data.push(item.label);
        }

        let primary_triples = Tensor::<B, 1, Int>::from_data(triples_data.as_slice(), device)
            .reshape([batch_size, 3]);

        let qualifier_relations = Tensor::<B, 1, Int>::from_data(qual_rel_data.as_slice(), device)
            .reshape([batch_size, max_qualifiers]);

        let qualifier_entities = Tensor::<B, 1, Int>::from_data(qual_ent_data.as_slice(), device)
            .reshape([batch_size, max_qualifiers]);

        let qualifier_mask = Tensor::<B, 1>::from_data(mask_data.as_slice(), device)
            .reshape([batch_size, max_qualifiers]);

        let labels = Tensor::<B, 1>::from_data(label_data.as_slice(), device);

        HehrBatch {
            primary_triples,
            qualifier_relations,
            qualifier_entities,
            qualifier_mask,
            labels,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::fact::{HehrFact, Qualifier};
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_batcher_shapes() {
        let items = vec![
            HehrFactItem {
                fact: HehrFact {
                    head: 0,
                    relation: 0,
                    tail: 1,
                    qualifiers: vec![
                        Qualifier {
                            relation_id: 1,
                            entity_id: 2,
                        },
                        Qualifier {
                            relation_id: 2,
                            entity_id: 3,
                        },
                    ],
                },
                label: 1.0,
            },
            HehrFactItem {
                fact: HehrFact {
                    head: 4,
                    relation: 1,
                    tail: 5,
                    qualifiers: vec![Qualifier {
                        relation_id: 3,
                        entity_id: 6,
                    }],
                },
                label: 0.0,
            },
        ];

        let device = <TestBackend as Backend>::Device::default();
        let batcher = HehrBatcher::new();
        let batch: HehrBatch<TestBackend> = batcher.batch(items, &device);

        // batch_size = 2, max_qualifiers = 2
        assert_eq!(batch.primary_triples.dims(), [2, 3]);
        assert_eq!(batch.qualifier_relations.dims(), [2, 2]);
        assert_eq!(batch.qualifier_entities.dims(), [2, 2]);
        assert_eq!(batch.qualifier_mask.dims(), [2, 2]);
        assert_eq!(batch.labels.dims(), [2]);
    }
}
