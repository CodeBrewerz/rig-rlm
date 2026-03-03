//! Neural activation probing for GNN interpretability.
//!
//! Inspired by GuideLabs' concept discovery in Steerling-8B, this module
//! probes GNN neurons to discover which graph concepts each neuron detects.
//!
//! Approach:
//! 1. Define known graph concepts (node degree, neighbor homogeneity, etc.)
//! 2. Compute ground-truth concept labels for every node from the graph
//! 3. Train linear probes: for each neuron, regress activation → concept value
//! 4. Neurons with high R² are "concept detectors"

use std::collections::HashMap;

/// A known graph concept that we probe for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub enum GraphConcept {
    /// Number of edges connected to this node (in + out).
    NodeDegree,
    /// Number of distinct edge types touching this node.
    EdgeTypeDiversity,
    /// Number of distinct neighbor node types.
    NeighborTypeDiversity,
    /// Whether this node is a "hub" (degree > mean + 2σ).
    IsHub,
    /// Fraction of 1-hop neighbors that are anomalous (score >= 0.5).
    AnomalousNeighborRatio,
    /// Local clustering coefficient (how connected are this node's neighbors).
    LocalDensity,
}

impl GraphConcept {
    pub fn all() -> Vec<GraphConcept> {
        vec![
            Self::NodeDegree,
            Self::EdgeTypeDiversity,
            Self::NeighborTypeDiversity,
            Self::IsHub,
            Self::AnomalousNeighborRatio,
            Self::LocalDensity,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::NodeDegree => "node_degree",
            Self::EdgeTypeDiversity => "edge_type_diversity",
            Self::NeighborTypeDiversity => "neighbor_type_diversity",
            Self::IsHub => "is_hub",
            Self::AnomalousNeighborRatio => "anomalous_neighbor_ratio",
            Self::LocalDensity => "local_density",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::NodeDegree => "Total number of edges connected to this node",
            Self::EdgeTypeDiversity => "Number of distinct relation types touching this node",
            Self::NeighborTypeDiversity => "Number of distinct neighbor entity types",
            Self::IsHub => "Whether this node is a high-degree hub (degree > mean + 2σ)",
            Self::AnomalousNeighborRatio => "Fraction of direct neighbors flagged as anomalous",
            Self::LocalDensity => "Clustering coefficient: connectivity among neighbors",
        }
    }
}

/// Ground-truth concept labels computed from graph structure.
/// node_type → concept → Vec<f32> indexed by node_id.
pub struct ConceptLabels {
    pub labels: HashMap<String, HashMap<GraphConcept, Vec<f32>>>,
}

impl ConceptLabels {
    /// Compute concept labels from graph edge structure.
    pub fn compute(
        edges: &HashMap<(String, String, String), Vec<(usize, usize)>>,
        node_counts: &HashMap<String, usize>,
        anomaly_scores: &HashMap<String, Vec<f32>>, // node_type → norm scores
    ) -> Self {
        let mut labels: HashMap<String, HashMap<GraphConcept, Vec<f32>>> = HashMap::new();

        for (nt, &count) in node_counts {
            if count == 0 {
                continue;
            }

            // Initialize per-node accumulators
            let mut degree = vec![0.0f32; count];
            let mut edge_type_sets: Vec<std::collections::HashSet<String>> = (0..count)
                .map(|_| std::collections::HashSet::new())
                .collect();
            let mut neighbor_type_sets: Vec<std::collections::HashSet<String>> = (0..count)
                .map(|_| std::collections::HashSet::new())
                .collect();
            let mut neighbor_ids: Vec<Vec<(String, usize)>> =
                (0..count).map(|_| Vec::new()).collect();

            // Scan all edges to compute per-node features
            for ((src_type, relation, dst_type), edge_list) in edges {
                if dst_type == nt {
                    // Incoming edges
                    for &(src_id, dst_id) in edge_list {
                        if dst_id < count {
                            degree[dst_id] += 1.0;
                            edge_type_sets[dst_id].insert(relation.clone());
                            neighbor_type_sets[dst_id].insert(src_type.clone());
                            neighbor_ids[dst_id].push((src_type.clone(), src_id));
                        }
                    }
                }
                if src_type == nt {
                    // Outgoing edges
                    for &(src_id, dst_id) in edge_list {
                        if src_id < count {
                            degree[src_id] += 1.0;
                            edge_type_sets[src_id].insert(relation.clone());
                            neighbor_type_sets[src_id].insert(dst_type.clone());
                            neighbor_ids[src_id].push((dst_type.clone(), dst_id));
                        }
                    }
                }
            }

            // Compute derived concepts
            let mean_degree = degree.iter().sum::<f32>() / count as f32;
            let deg_std = (degree
                .iter()
                .map(|d| (d - mean_degree).powi(2))
                .sum::<f32>()
                / count as f32)
                .sqrt();

            let is_hub: Vec<f32> = degree
                .iter()
                .map(|d| {
                    if *d > mean_degree + 2.0 * deg_std {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();

            let edge_type_div: Vec<f32> = edge_type_sets.iter().map(|s| s.len() as f32).collect();
            let nbr_type_div: Vec<f32> =
                neighbor_type_sets.iter().map(|s| s.len() as f32).collect();

            // Anomalous neighbor ratio
            let anom_nbr_ratio: Vec<f32> = neighbor_ids
                .iter()
                .map(|nbrs| {
                    if nbrs.is_empty() {
                        return 0.0;
                    }
                    let anom_count = nbrs
                        .iter()
                        .filter(|(ntype, nid)| {
                            anomaly_scores
                                .get(ntype)
                                .and_then(|scores| scores.get(*nid))
                                .map(|s| *s >= 0.5)
                                .unwrap_or(false)
                        })
                        .count();
                    anom_count as f32 / nbrs.len() as f32
                })
                .collect();

            // Local density (simplified: unique neighbors / degree²)
            let local_density: Vec<f32> = neighbor_ids
                .iter()
                .zip(degree.iter())
                .map(|(nbrs, deg)| {
                    if *deg <= 1.0 {
                        return 0.0;
                    }
                    let unique: std::collections::HashSet<_> = nbrs.iter().collect();
                    unique.len() as f32 / deg.max(1.0)
                })
                .collect();

            let mut concept_map = HashMap::new();
            concept_map.insert(GraphConcept::NodeDegree, degree);
            concept_map.insert(GraphConcept::EdgeTypeDiversity, edge_type_div);
            concept_map.insert(GraphConcept::NeighborTypeDiversity, nbr_type_div);
            concept_map.insert(GraphConcept::IsHub, is_hub);
            concept_map.insert(GraphConcept::AnomalousNeighborRatio, anom_nbr_ratio);
            concept_map.insert(GraphConcept::LocalDensity, local_density);

            labels.insert(nt.clone(), concept_map);
        }

        ConceptLabels { labels }
    }
}

/// Result of training a linear probe: neuron → concept alignment.
#[derive(Debug, Clone, serde::Serialize)]
pub struct NeuronConceptAlignment {
    pub layer: usize,
    pub dimension: usize,
    pub concept: String,
    pub correlation: f32, // Pearson r
    pub r_squared: f32,   // R²
}

/// All concept probe results for one model.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelProbeResults {
    pub model_name: String,
    /// Top neuron-concept alignments sorted by R² (strongest first).
    pub top_alignments: Vec<NeuronConceptAlignment>,
    /// Per-concept: which neurons detect it best.
    pub concept_detectors: HashMap<String, Vec<NeuronConceptAlignment>>,
    /// How many concepts are "well-detected" (R² > 0.3).
    pub concepts_detected: usize,
}

/// All probe results across all models.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ProbeResults {
    pub models: HashMap<String, ModelProbeResults>,
    /// Cross-model comparison: which concepts each model specializes in.
    pub model_specializations: HashMap<String, Vec<String>>,
}

impl ProbeResults {
    /// Train linear probes for all models.
    ///
    /// `activations`: model_name → layer_idx → { node_type → Vec<Vec<f32>> }
    /// Each Vec<Vec<f32>> is [N nodes × D dims].
    pub fn train(
        activations: &HashMap<String, Vec<HashMap<String, Vec<Vec<f32>>>>>,
        concept_labels: &ConceptLabels,
        hidden_dim: usize,
    ) -> Self {
        let mut models = HashMap::new();
        let mut model_specializations: HashMap<String, Vec<String>> = HashMap::new();

        for (model_name, layer_acts) in activations {
            let mut all_alignments: Vec<NeuronConceptAlignment> = Vec::new();

            for (layer_idx, layer_data) in layer_acts.iter().enumerate() {
                for (node_type, node_vecs) in layer_data {
                    let concept_map = match concept_labels.labels.get(node_type) {
                        Some(c) => c,
                        None => continue,
                    };

                    let n = node_vecs.len();
                    if n < 10 {
                        continue; // Need enough samples
                    }

                    // For each dimension × concept, compute Pearson correlation
                    for dim in 0..hidden_dim.min(node_vecs.first().map(|v| v.len()).unwrap_or(0)) {
                        let activations_col: Vec<f32> = node_vecs.iter().map(|v| v[dim]).collect();

                        for concept in GraphConcept::all() {
                            let concept_values = match concept_map.get(&concept) {
                                Some(v) => v,
                                None => continue,
                            };

                            let r = pearson_r(&activations_col, concept_values);
                            let r_sq = r * r;

                            if r_sq > 0.1 {
                                // Only keep meaningful correlations
                                all_alignments.push(NeuronConceptAlignment {
                                    layer: layer_idx,
                                    dimension: dim,
                                    concept: concept.name().to_string(),
                                    correlation: r,
                                    r_squared: r_sq,
                                });
                            }
                        }
                    }
                }
            }

            // Sort by R² descending
            all_alignments.sort_by(|a, b| b.r_squared.partial_cmp(&a.r_squared).unwrap());

            // Build concept_detectors map
            let mut concept_detectors: HashMap<String, Vec<NeuronConceptAlignment>> =
                HashMap::new();
            for alignment in &all_alignments {
                concept_detectors
                    .entry(alignment.concept.clone())
                    .or_default()
                    .push(alignment.clone());
            }

            // Limit each concept to top-3 detectors
            for detectors in concept_detectors.values_mut() {
                detectors.sort_by(|a, b| b.r_squared.partial_cmp(&a.r_squared).unwrap());
                detectors.truncate(3);
            }

            let concepts_detected = concept_detectors
                .values()
                .filter(|v| v.first().map(|a| a.r_squared > 0.3).unwrap_or(false))
                .count();

            // Model specializations: concepts with R² > 0.3
            let specialties: Vec<String> = concept_detectors
                .iter()
                .filter(|(_, v)| v.first().map(|a| a.r_squared > 0.3).unwrap_or(false))
                .map(|(c, _)| c.clone())
                .collect();
            model_specializations.insert(model_name.clone(), specialties);

            models.insert(
                model_name.clone(),
                ModelProbeResults {
                    model_name: model_name.clone(),
                    top_alignments: all_alignments.into_iter().take(20).collect(),
                    concept_detectors,
                    concepts_detected,
                },
            );
        }

        ProbeResults {
            models,
            model_specializations,
        }
    }
}

/// Activation profile for a single node — which neurons fired and what concepts.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ActivationProfile {
    /// Per-layer: which neurons fired strongest.
    pub layer_activations: Vec<LayerProfile>,
    /// Detected concepts (from probed neurons).
    pub detected_concepts: Vec<DetectedConcept>,
    /// Cross-model agreement on concepts.
    pub concept_agreement: HashMap<String, Vec<String>>,
    /// Summary narrative.
    pub summary: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LayerProfile {
    pub layer: usize,
    pub layer_name: String,
    /// Top-5 most active neurons in this layer.
    pub top_neurons: Vec<ActiveNeuron>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ActiveNeuron {
    pub dimension: usize,
    pub activation: f32,
    /// If this neuron is a concept detector, which concept.
    pub detected_concept: Option<String>,
    /// R² of the probe for this neuron-concept pair.
    pub probe_r_squared: Option<f32>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct DetectedConcept {
    pub concept: String,
    pub description: String,
    /// Which model's neurons detected this concept.
    pub detecting_model: String,
    /// Best R² among detectors for this concept.
    pub best_r_squared: f32,
    /// Actual concept value for this node.
    pub value: f32,
}

impl ActivationProfile {
    /// Build activation profile for a node from its per-layer activations.
    pub fn build(
        node_activations: &HashMap<String, Vec<Vec<f32>>>, // model → [layer][dim]
        probe_results: &ProbeResults,
        concept_labels: &ConceptLabels,
        node_type: &str,
        node_id: usize,
    ) -> Self {
        let mut layer_profiles = Vec::new();
        let mut detected_concepts = Vec::new();
        let mut concept_agreement: HashMap<String, Vec<String>> = HashMap::new();

        // Build per-model layer profiles
        for (model_name, layer_acts) in node_activations {
            let probe = probe_results.models.get(model_name);

            for (layer_idx, activations) in layer_acts.iter().enumerate() {
                // Find top-5 most active neurons
                let mut indexed: Vec<(usize, f32)> = activations
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (i, v.abs()))
                    .collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let top_neurons: Vec<ActiveNeuron> = indexed
                    .iter()
                    .take(5)
                    .map(|&(dim, act)| {
                        // Check if this neuron is a known concept detector
                        let (concept, r_sq) = probe
                            .and_then(|p| {
                                p.top_alignments
                                    .iter()
                                    .find(|a| a.layer == layer_idx && a.dimension == dim)
                            })
                            .map(|a| (Some(a.concept.clone()), Some(a.r_squared)))
                            .unwrap_or((None, None));

                        ActiveNeuron {
                            dimension: dim,
                            activation: act,
                            detected_concept: concept,
                            probe_r_squared: r_sq,
                        }
                    })
                    .collect();

                layer_profiles.push(LayerProfile {
                    layer: layer_idx,
                    layer_name: format!("{}:layer_{}", model_name, layer_idx),
                    top_neurons,
                });
            }

            // Collect detected concepts from this model
            if let Some(probe) = probe {
                for (concept_name, detectors) in &probe.concept_detectors {
                    if let Some(best) = detectors.first() {
                        if best.r_squared > 0.2 {
                            // Get actual concept value for this node
                            let value = concept_labels
                                .labels
                                .get(node_type)
                                .and_then(|cm| {
                                    GraphConcept::all()
                                        .iter()
                                        .find(|c| c.name() == concept_name)
                                        .and_then(|c| cm.get(c))
                                })
                                .and_then(|v| v.get(node_id))
                                .copied()
                                .unwrap_or(0.0);

                            concept_agreement
                                .entry(concept_name.clone())
                                .or_default()
                                .push(model_name.clone());

                            detected_concepts.push(DetectedConcept {
                                concept: concept_name.clone(),
                                description: GraphConcept::all()
                                    .iter()
                                    .find(|c| c.name() == concept_name)
                                    .map(|c| c.description().to_string())
                                    .unwrap_or_default(),
                                detecting_model: model_name.clone(),
                                best_r_squared: best.r_squared,
                                value,
                            });
                        }
                    }
                }
            }
        }

        // Deduplicate detected concepts
        detected_concepts.sort_by(|a, b| b.best_r_squared.partial_cmp(&a.best_r_squared).unwrap());
        let unique_concepts: Vec<String> = detected_concepts
            .iter()
            .map(|c| c.concept.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let summary = format!(
            "Activation probing detected {} graph concepts across {} model(s). \
             Top detected: {}. \
             Cross-model agreement: {} concepts detected by multiple models.",
            unique_concepts.len(),
            node_activations.len(),
            detected_concepts
                .first()
                .map(|c| format!("{} (R²={:.2})", c.concept, c.best_r_squared))
                .unwrap_or_else(|| "none".into()),
            concept_agreement.values().filter(|v| v.len() > 1).count()
        );

        ActivationProfile {
            layer_activations: layer_profiles,
            detected_concepts,
            concept_agreement,
            summary,
        }
    }
}

/// Pearson correlation coefficient between two slices.
fn pearson_r(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    let mean_x = x.iter().take(n).sum::<f32>() / n as f32;
    let mean_y = y.iter().take(n).sum::<f32>() / n as f32;

    let mut cov = 0.0f32;
    let mut var_x = 0.0f32;
    let mut var_y = 0.0f32;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        cov / denom
    }
}
