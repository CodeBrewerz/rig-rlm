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
    /// Whether to append queue-regime bucketing features (4 one-hot bins).
    /// From the queue-reactive LOB paper (Huang, Lehalle, Rosenbaum 2014):
    /// classifies each node's primary feature magnitude into
    /// {Empty, Small, Usual, Large} using 33rd/67th percentile thresholds.
    pub enable_queue_regime: bool,
    /// Whether to compute and inject flow ratio ρ(n) edge features.
    /// ρ = in_degree / (out_degree + 1) for each edge endpoint.
    /// From the queue-reactive LOB paper: the invariant distribution
    /// is entirely determined by ρ(n).
    pub enable_flow_ratio: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            target_dim: 16,
            normalize: true,
            enable_queue_regime: true,
            enable_flow_ratio: true,
        }
    }
}

/// Queue-size regime from the queue-reactive LOB paper.
///
/// S_{m,l}(x) classifies a value into 4 regimes using quantile thresholds:
/// - Empty: x == 0
/// - Small: 0 < x <= q33
/// - Usual: q33 < x <= q67
/// - Large: x > q67
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueRegime {
    Empty = 0,
    Small = 1,
    Usual = 2,
    Large = 3,
}

impl QueueRegime {
    /// Classify a value into a queue regime using quantile thresholds.
    pub fn classify(value: f32, q33: f32, q67: f32) -> Self {
        if value.abs() < 1e-8 {
            QueueRegime::Empty
        } else if value <= q33 {
            QueueRegime::Small
        } else if value <= q67 {
            QueueRegime::Usual
        } else {
            QueueRegime::Large
        }
    }

    /// Return a 4-element one-hot vector for this regime.
    pub fn one_hot(&self) -> [f32; 4] {
        let mut v = [0.0f32; 4];
        v[*self as usize] = 1.0;
        v
    }
}

/// Number of queue-regime bins (one-hot dimensions appended to features).
pub const QUEUE_REGIME_BINS: usize = 4;

/// Compute the 33rd and 67th percentile of a sorted slice of positive values.
fn quantile_thresholds(values: &[f32]) -> (f32, f32) {
    let positives: Vec<f32> = values.iter().copied().filter(|v| *v > 1e-8).collect();
    if positives.is_empty() {
        return (0.0, 0.0);
    }
    let mut sorted = positives;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let q33 = sorted[(n as f32 * 0.33) as usize];
    let q67 = sorted[((n as f32 * 0.67) as usize).min(n - 1)];
    (q33, q67)
}

/// Compute features from entity attributes and inject into a HeteroGraph.
///
/// For each entity type:
/// 1. Extract numeric attributes
/// 2. Pad/truncate to `target_dim`
/// 3. Optionally append queue-regime bucketing (4 one-hot bins)
/// 4. Optionally normalize (before regime bins, to preserve one-hot)
/// 5. Replace the random features in the graph
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

        // Normalize if requested (before adding regime bins)
        if config.normalize && !all_features.is_empty() {
            normalize_features(&mut all_features, config.target_dim);
        }

        // Append queue-regime bucketing (4 one-hot bins)
        if config.enable_queue_regime && !all_features.is_empty() {
            // Use the first feature (primary magnitude) as the regime signal
            let magnitudes: Vec<f32> = all_features
                .iter()
                .map(|f| f.iter().map(|v| v.abs()).sum::<f32>())
                .collect();
            let (q33, q67) = quantile_thresholds(&magnitudes);

            for (feat, &mag) in all_features.iter_mut().zip(magnitudes.iter()) {
                let regime = QueueRegime::classify(mag, q33, q67);
                feat.extend_from_slice(&regime.one_hot());
            }
        }

        // Compute actual feature dimension
        let feat_dim = if config.enable_queue_regime {
            config.target_dim + QUEUE_REGIME_BINS
        } else {
            config.target_dim
        };

        // Convert to tensor and replace in graph
        let flat: Vec<f32> = all_features.into_iter().flatten().collect();
        let tensor = Tensor::<B, 1>::from_data(flat.as_slice(), device).reshape([count, feat_dim]);

        graph.node_features.insert(entity_type.clone(), tensor);
    }

    // Compute and inject flow ratio edge features if enabled
    if config.enable_flow_ratio {
        compute_flow_ratio_edges(graph, device);
    }
}

/// Compute the arrival/departure ratio ρ(n) for each edge.
///
/// From the queue-reactive LOB paper (Huang et al. 2014), the invariant
/// distribution is entirely determined by ρ(n) = arrival_rate / departure_rate.
///
/// In the financial graph context:
/// - ρ = in_degree / (out_degree + 1) for each node
/// - High ρ → accumulation pressure (more inflows than outflows)
/// - Low ρ → depletion pressure (more outflows than inflows)
///
/// For each edge (src→dst), we store a 2-dim feature: [ρ_src, ρ_dst].
/// This allows message-passing layers to weight messages by the flow
/// pressure at each endpoint.
pub fn compute_flow_ratio_edges<B: Backend>(graph: &mut HeteroGraph<B>, device: &B::Device) {
    // Compute in-degree and out-degree per (node_type, node_idx)
    let mut in_degree: std::collections::HashMap<String, Vec<f32>> =
        std::collections::HashMap::new();
    let mut out_degree: std::collections::HashMap<String, Vec<f32>> =
        std::collections::HashMap::new();

    for (nt, &count) in &graph.node_counts {
        in_degree.insert(nt.clone(), vec![0.0; count]);
        out_degree.insert(nt.clone(), vec![0.0; count]);
    }

    // Count degrees from edge indices
    for (et, _) in &graph.edge_index {
        let (src_type, _, dst_type) = et;
        if let Some((src_vec, dst_vec)) = graph.edges_as_vecs(et) {
            if let Some(out_deg) = out_degree.get_mut(src_type) {
                for &s in &src_vec {
                    let s = s as usize;
                    if s < out_deg.len() {
                        out_deg[s] += 1.0;
                    }
                }
            }
            if let Some(in_deg) = in_degree.get_mut(dst_type) {
                for &d in &dst_vec {
                    let d = d as usize;
                    if d < in_deg.len() {
                        in_deg[d] += 1.0;
                    }
                }
            }
        }
    }

    // Compute ρ(n) = in_degree / (out_degree + 1) per node
    let mut rho: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
    for (nt, in_deg) in &in_degree {
        let out_deg = out_degree.get(nt).unwrap();
        let node_rho: Vec<f32> = in_deg
            .iter()
            .zip(out_deg.iter())
            .map(|(&i, &o)| i / (o + 1.0))
            .collect();
        rho.insert(nt.clone(), node_rho);
    }

    // For each edge type, build edge features [ρ_src, ρ_dst]
    let edge_types: Vec<_> = graph.edge_index.keys().cloned().collect();
    for et in &edge_types {
        let (src_type, _, dst_type) = et;
        if let Some((src_vec, dst_vec)) = graph.edges_as_vecs(et) {
            let src_rho = match rho.get(src_type) {
                Some(r) => r,
                None => continue,
            };
            let dst_rho = match rho.get(dst_type) {
                Some(r) => r,
                None => continue,
            };

            let num_edges = src_vec.len();
            let mut edge_feat_data = Vec::with_capacity(num_edges * 2);
            for (&s, &d) in src_vec.iter().zip(dst_vec.iter()) {
                let s = s as usize;
                let d = d as usize;
                let r_src = if s < src_rho.len() { src_rho[s] } else { 1.0 };
                let r_dst = if d < dst_rho.len() { dst_rho[d] } else { 1.0 };
                edge_feat_data.push(r_src);
                edge_feat_data.push(r_dst);
            }

            let edge_features = Tensor::<B, 1>::from_data(edge_feat_data.as_slice(), device)
                .reshape([num_edges, 2]);
            graph.add_edge_features(et, edge_features);
        }
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
