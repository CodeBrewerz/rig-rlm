//! Feature engineering: converts raw entity attributes into GNN node features.
//!
//! Takes the numeric attributes extracted by json_loader and constructs
//! properly-shaped tensors for the HeteroGraph, optionally normalizing features.

use burn::prelude::*;

use super::json_loader::DataExport;
use crate::data::hetero_graph::HeteroGraph;

/// Feature engineering configuration.
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Target feature dimension (pad or truncate to this).
    pub target_dim: usize,
    /// Whether to normalize features to zero mean/unit variance.
    pub normalize: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            target_dim: 16,
            normalize: true,
        }
    }
}

/// Compute features from entity attributes and inject into a HeteroGraph.
///
/// For each entity type:
/// 1. Extract numeric attributes
/// 2. Pad/truncate to `target_dim`
/// 3. Optionally normalize
/// 4. Replace the random features in the graph
pub fn engineer_features<B: Backend>(
    graph: &mut HeteroGraph<B>,
    export: &DataExport,
    config: &FeatureConfig,
    device: &B::Device,
) {
    let raw_features = super::json_loader::extract_numeric_features(export);

    for (entity_type, id_features) in &raw_features {
        let count = graph.node_counts.get(entity_type).copied().unwrap_or(0);
        if count == 0 {
            continue;
        }

        // Collect feature vectors in order
        let mut all_features: Vec<Vec<f32>> = Vec::with_capacity(count);

        // Build ordered list (node IDs are 0..count)
        for i in 0..count {
            let entity_id = format!("{}_{}", entity_type, i);
            let raw = id_features.get(&entity_id).cloned().unwrap_or_default();

            // Pad or truncate to target_dim
            let mut features = vec![0.0f32; config.target_dim];
            for (j, &val) in raw.iter().enumerate().take(config.target_dim) {
                features[j] = val;
            }
            all_features.push(features);
        }

        // Normalize if requested
        if config.normalize && !all_features.is_empty() {
            normalize_features(&mut all_features, config.target_dim);
        }

        // Convert to tensor and replace in graph
        let flat: Vec<f32> = all_features.into_iter().flatten().collect();
        let tensor =
            Tensor::<B, 1>::from_data(flat.as_slice(), device).reshape([count, config.target_dim]);

        graph.node_features.insert(entity_type.clone(), tensor);
    }
}

/// Z-score normalize feature columns (zero mean, unit variance).
fn normalize_features(features: &mut [Vec<f32>], dim: usize) {
    let n = features.len() as f32;
    if n < 2.0 {
        return;
    }

    for col in 0..dim {
        let mean: f32 = features.iter().map(|f| f[col]).sum::<f32>() / n;
        let var: f32 = features
            .iter()
            .map(|f| (f[col] - mean).powi(2))
            .sum::<f32>()
            / n;
        let std = var.sqrt();

        if std > 1e-8 {
            for row in features.iter_mut() {
                row[col] = (row[col] - mean) / std;
            }
        } else {
            // Constant column → zero out
            for row in features.iter_mut() {
                row[col] = 0.0;
            }
        }
    }
}

/// Compute basic feature statistics per node type.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FeatureStats {
    pub node_type: String,
    pub num_nodes: usize,
    pub feature_dim: usize,
    pub mean_magnitude: f32,
    pub max_magnitude: f32,
}

/// Compute feature statistics from a HeteroGraph.
pub fn feature_stats<B: Backend>(graph: &HeteroGraph<B>) -> Vec<FeatureStats> {
    let mut stats = Vec::new();

    for (node_type, tensor) in &graph.node_features {
        let dims = tensor.dims();
        let num_nodes = dims[0];
        let feature_dim = dims[1];

        let flat = tensor
            .clone()
            .reshape([num_nodes * feature_dim])
            .into_data();
        let values: Vec<f32> = flat.as_slice::<f32>().unwrap().to_vec();

        let magnitudes: Vec<f32> = (0..num_nodes)
            .map(|i| {
                let start = i * feature_dim;
                let end = start + feature_dim;
                values[start..end].iter().map(|x| x.abs()).sum::<f32>() / feature_dim as f32
            })
            .collect();

        let mean_mag = magnitudes.iter().sum::<f32>() / num_nodes.max(1) as f32;
        let max_mag = magnitudes.iter().copied().fold(0.0f32, f32::max);

        stats.push(FeatureStats {
            node_type: node_type.clone(),
            num_nodes,
            feature_dim,
            mean_magnitude: mean_mag,
            max_magnitude: max_mag,
        });
    }

    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_features() {
        let mut features = vec![vec![1.0, 10.0], vec![3.0, 20.0], vec![5.0, 30.0]];

        normalize_features(&mut features, 2);

        // After normalization, mean ≈ 0 and std ≈ 1 for each column
        let col0: Vec<f32> = features.iter().map(|f| f[0]).collect();
        let mean0: f32 = col0.iter().sum::<f32>() / 3.0;
        assert!(mean0.abs() < 0.01, "Mean should be ~0, got {}", mean0);
    }
}
