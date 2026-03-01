//! HEHRGNN message-passing layer.
//!
//! Implements the 3-step message propagation from the HEHRGNN paper:
//! 1. **Gather** — pool entity embeddings (head, tail, qualifier entities) into a
//!    hyperedge instance representation.
//! 2. **Apply** — combine the hyperedge instance with the relation embedding through
//!    a linear transform + ReLU to get an updated relation representation.
//! 3. **Scatter** — project updated representations back to entity space, accumulating
//!    updates for each entity via scatter-add.

use burn::nn;
use burn::prelude::*;
use burn::tensor::IndexingUpdateOp;

/// Configuration for a single GNN message-passing layer.
#[derive(Config, Debug)]
pub struct GnnLayerConfig {
    /// Dimensionality of entity / relation embeddings.
    pub hidden_dim: usize,
    /// Dropout rate applied after Apply step.
    #[config(default = "0.1")]
    pub dropout: f64,
}

/// A single HEHRGNN message-passing layer.
#[derive(Module, Debug)]
pub struct GnnLayer<B: Backend> {
    /// Linear transform for Apply step: projects concatenated
    /// [hyperedge_instance || relation] (2 * hidden_dim) → hidden_dim.
    pub apply_linear: nn::Linear<B>,
    /// Dropout applied after the Apply step.
    pub dropout: nn::Dropout,
    /// Hidden dimension (stored for scatter step).
    hidden_dim: usize,
}

impl GnnLayerConfig {
    /// Initialize the GNN layer on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> GnnLayer<B> {
        // Apply linear takes concatenation of hyperedge_instance + relation → hidden_dim
        let apply_linear = nn::LinearConfig::new(self.hidden_dim * 2, self.hidden_dim).init(device);
        let dropout = nn::DropoutConfig::new(self.dropout).init();

        GnnLayer {
            apply_linear,
            dropout,
            hidden_dim: self.hidden_dim,
        }
    }
}

impl<B: Backend> GnnLayer<B> {
    /// Run the 3-step message passing.
    ///
    /// # Arguments
    /// - `entity_emb`: all entity embeddings `[num_entities, hidden_dim]`
    /// - `relation_emb`: all relation embeddings `[num_relations, hidden_dim]`
    /// - `primary_triples`: `[batch_size, 3]` int tensor (head, rel, tail)
    /// - `qualifier_entities`: `[batch_size, max_qual]` int tensor (padded entity IDs)
    /// - `qualifier_relations`: `[batch_size, max_qual]` int tensor (padded relation IDs)
    /// - `qualifier_mask`: `[batch_size, max_qual]` float mask (1 = valid, 0 = pad)
    ///
    /// # Returns
    /// `(updated_entity_emb, updated_relation_emb)` with same shapes as inputs.
    pub fn forward(
        &self,
        entity_emb: Tensor<B, 2>,
        relation_emb: Tensor<B, 2>,
        primary_triples: Tensor<B, 2, Int>,
        qualifier_entities: Tensor<B, 2, Int>,
        _qualifier_relations: Tensor<B, 2, Int>,
        qualifier_mask: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let device = entity_emb.device();
        let batch_size = primary_triples.dims()[0];
        let num_entities = entity_emb.dims()[0];

        // --- Step 1: Gather ---
        // Extract head, relation, tail indices from primary triples
        let head_ids = primary_triples
            .clone()
            .slice([0..batch_size, 0..1])
            .reshape([batch_size]);
        let rel_ids = primary_triples
            .clone()
            .slice([0..batch_size, 1..2])
            .reshape([batch_size]);
        let tail_ids = primary_triples
            .slice([0..batch_size, 2..3])
            .reshape([batch_size]);

        // Look up embeddings for head and tail: [batch_size, hidden_dim]
        let head_emb = entity_emb.clone().select(0, head_ids.clone());
        let tail_emb = entity_emb.clone().select(0, tail_ids.clone());

        // Look up qualifier entity embeddings: [batch_size, max_qual, hidden_dim]
        let max_qual = qualifier_entities.dims()[1];
        let qual_ent_emb = entity_emb
            .clone()
            .select(
                0,
                qualifier_entities.clone().reshape([batch_size * max_qual]),
            )
            .reshape([batch_size, max_qual, self.hidden_dim]);

        // Mask out padded qualifier positions and mean-pool
        let qual_mask_expanded = qualifier_mask
            .clone()
            .unsqueeze_dim::<3>(2)
            .repeat_dim(2, self.hidden_dim); // [batch_size, max_qual, hidden_dim]

        let masked_qual = qual_ent_emb.mul(qual_mask_expanded);

        // Sum qualifier embeddings and divide by count of valid qualifiers
        let qual_sum: Tensor<B, 2> = masked_qual
            .sum_dim(1)
            .reshape([batch_size, self.hidden_dim]); // [batch_size, hidden_dim]
        let qual_count = qualifier_mask
            .sum_dim(1) // [batch_size, 1]
            .clamp_min(1.0) // avoid div-by-zero
            .reshape([batch_size, 1]); // ensure shape
        let qual_mean = qual_sum.div(qual_count); // [batch_size, hidden_dim]

        // Hyperedge instance = mean(head, tail, qual_mean)
        let hyperedge_instance = (head_emb + tail_emb + qual_mean) / 3.0;

        // --- Step 2: Apply ---
        // Look up primary relation embedding: [batch_size, hidden_dim]
        let primary_rel_emb = relation_emb.clone().select(0, rel_ids.clone());

        // Concatenate [hyperedge_instance, primary_rel_emb] → [batch_size, 2*hidden_dim]
        let concat = Tensor::cat(vec![hyperedge_instance, primary_rel_emb], 1);

        // Linear + ReLU + Dropout → updated relation representation
        let updated_rel_repr = self.apply_linear.forward(concat);
        let updated_rel_repr = burn::tensor::activation::relu(updated_rel_repr);
        let updated_rel_repr = self.dropout.forward(updated_rel_repr);

        // --- Step 3: Scatter ---
        // Scatter updated representations back to the global entity embedding.
        // We add the updated relation repr as a message to both head and tail entities.
        let mut entity_update = Tensor::<B, 2>::zeros([num_entities, self.hidden_dim], &device);

        // Scatter-add to head positions
        let head_scatter_idx = head_ids
            .reshape([batch_size, 1])
            .repeat_dim(1, self.hidden_dim);
        entity_update = entity_update.scatter(
            0,
            head_scatter_idx,
            updated_rel_repr.clone(),
            IndexingUpdateOp::Add,
        );

        // Scatter-add to tail positions
        let tail_scatter_idx = tail_ids
            .reshape([batch_size, 1])
            .repeat_dim(1, self.hidden_dim);
        entity_update = entity_update.scatter(
            0,
            tail_scatter_idx,
            updated_rel_repr.clone(),
            IndexingUpdateOp::Add,
        );

        // Updated entity embeddings = original + scattered messages
        let updated_entity_emb = entity_emb + entity_update;

        // Updated relation embeddings: scatter the updated_rel_repr back to relation positions
        let mut relation_update = Tensor::<B, 2>::zeros_like(&relation_emb);
        let rel_scatter_idx = rel_ids
            .reshape([batch_size, 1])
            .repeat_dim(1, self.hidden_dim);
        relation_update =
            relation_update.scatter(0, rel_scatter_idx, updated_rel_repr, IndexingUpdateOp::Add);

        let updated_relation_emb = relation_emb + relation_update;

        (updated_entity_emb, updated_relation_emb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_gnn_layer_output_shapes() {
        let device = <TestBackend as Backend>::Device::default();
        let hidden_dim = 16;
        let num_entities = 10;
        let num_relations = 5;

        let layer = GnnLayerConfig {
            hidden_dim,
            dropout: 0.0, // no dropout for deterministic test
        }
        .init::<TestBackend>(&device);

        let entity_emb = Tensor::random(
            [num_entities, hidden_dim],
            burn::tensor::Distribution::Default,
            &device,
        );
        let relation_emb = Tensor::random(
            [num_relations, hidden_dim],
            burn::tensor::Distribution::Default,
            &device,
        );

        // Primary triples: [batch_size, 3]
        let primary_triples =
            Tensor::<TestBackend, 2, Int>::from_data([[0, 0, 1], [2, 1, 3], [4, 2, 5]], &device);

        // Qualifier entities: [batch_size, max_qual]
        let qualifier_entities =
            Tensor::<TestBackend, 2, Int>::from_data([[6, 7], [8, 0], [9, 0]], &device);

        // Qualifier relations: [batch_size, max_qual]
        let qualifier_relations =
            Tensor::<TestBackend, 2, Int>::from_data([[3, 4], [3, 0], [4, 0]], &device);

        // Mask: [batch_size, max_qual]
        let qualifier_mask =
            Tensor::<TestBackend, 2>::from_data([[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]], &device);

        let (updated_ent, updated_rel) = layer.forward(
            entity_emb,
            relation_emb,
            primary_triples,
            qualifier_entities,
            qualifier_relations,
            qualifier_mask,
        );

        assert_eq!(updated_ent.dims(), [num_entities, hidden_dim]);
        assert_eq!(updated_rel.dims(), [num_relations, hidden_dim]);
    }
}
