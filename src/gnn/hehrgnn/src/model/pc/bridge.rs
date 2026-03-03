//! Bridge: discretize GNN features → PC variables + train HCLT.
//!
//! Converts continuous GNN embeddings, anomaly scores, and graph structural
//! features into discrete categories suitable for PC inference. Then builds
//! an HCLT structure and trains via EM.

use std::collections::HashMap;

use super::circuit::CompiledCircuit;
use super::em::{train_em, EmConfig, EmReport};
use super::structure;

/// PC variable schema: maps variable indices to interpretable names.
///
/// Variables:
/// 0: anomaly_level    — {low=0, medium=1, high=2}
/// 1: embedding_affinity — {negative=0, neutral=1, positive=2}
/// 2: node_degree      — {isolated=0, normal=1, hub=2}
/// 3: action_priority  — {low=0, medium=1, high=2}
/// 4: outcome_risk     — {safe=0, moderate=1, risky=2}
pub const NUM_PC_VARS: usize = 5;
pub const NUM_CATEGORIES: usize = 5;

/// Variable names for interpretability.
pub const VAR_NAMES: [&str; NUM_PC_VARS] = [
    "anomaly_level",
    "embedding_affinity",
    "node_degree",
    "action_priority",
    "outcome_risk",
];

/// Discretize an anomaly score (0.0–1.0) into 5 bins.
pub fn discretize_anomaly(score: f32) -> usize {
    if score < 0.15 {
        0 // very_low
    } else if score < 0.35 {
        1 // low
    } else if score < 0.55 {
        2 // medium
    } else if score < 0.75 {
        3 // high
    } else {
        4 // very_high
    }
}

/// Discretize embedding affinity (cosine sim, -1 to 1) into 5 bins.
pub fn discretize_affinity(affinity: f32) -> usize {
    if affinity < -0.3 {
        0 // very_negative
    } else if affinity < 0.0 {
        1 // negative
    } else if affinity < 0.3 {
        2 // neutral
    } else if affinity < 0.7 {
        3 // positive
    } else {
        4 // strong_positive
    }
}

/// Discretize node degree into 5 bins.
pub fn discretize_degree(degree: usize) -> usize {
    match degree {
        0 => 0,     // isolated
        1 => 1,     // leaf
        2..=3 => 2, // normal
        4..=6 => 3, // connected
        _ => 4,     // hub
    }
}

/// Discretize action priority weight (0.0–1.0) into 5 bins.
pub fn discretize_priority(priority: f32) -> usize {
    if priority < 0.2 {
        0 // very_low
    } else if priority < 0.4 {
        1 // low
    } else if priority < 0.6 {
        2 // medium
    } else if priority < 0.8 {
        3 // high
    } else {
        4 // very_high
    }
}

/// Discretize an outcome risk (fiduciary score, 0.0–1.0) into 5 bins.
/// Inverted: higher fiduciary = safer.
pub fn discretize_risk(fiduciary_score: f32) -> usize {
    if fiduciary_score >= 0.8 {
        0 // very_safe
    } else if fiduciary_score >= 0.6 {
        1 // safe
    } else if fiduciary_score >= 0.4 {
        2 // moderate
    } else if fiduciary_score >= 0.2 {
        3 // risky
    } else {
        4 // very_risky
    }
}

/// A discretized observation for PC training/inference.
pub type PcObservation = Vec<Option<usize>>;

/// Create a PC observation from fiduciary features.
pub fn create_observation(
    anomaly_score: f32,
    affinity: f32,
    degree: usize,
    action_priority: f32,
    outcome_risk: f32,
) -> PcObservation {
    vec![
        Some(discretize_anomaly(anomaly_score)),
        Some(discretize_affinity(affinity)),
        Some(discretize_degree(degree)),
        Some(discretize_priority(action_priority)),
        Some(discretize_risk(outcome_risk)),
    ]
}

/// Build and train a PC from fiduciary observations.
///
/// Takes historical observations (can be synthetic from graph features)
/// and builds an HCLT structure, then trains via EM.
pub fn build_fiduciary_pc(
    observations: &[Vec<usize>],
    em_epochs: usize,
) -> (CompiledCircuit, EmReport) {
    // Build HCLT structure from data
    let num_latents = 8; // 8 mixture components per tree node for richer circuit
    let builder = structure::hclt(observations, NUM_CATEGORIES, num_latents);

    // Compile
    let mut circuit = CompiledCircuit::compile(&builder);

    // Convert to evidence format
    let data: Vec<PcObservation> = observations
        .iter()
        .map(|obs| obs.iter().map(|&v| Some(v)).collect())
        .collect();

    // Train via EM
    let config = EmConfig {
        step_size: 0.1,
        pseudocount: 0.01,
        epochs: em_epochs,
    };
    let report = train_em(&mut circuit, &data, &config);

    (circuit, report)
}

/// Generate synthetic training data from graph features.
///
/// Scans all nodes in the graph, collects their features, and creates
/// PC training observations.
pub fn generate_training_data(
    anomaly_scores: &HashMap<String, HashMap<String, Vec<f32>>>,
    embeddings: &HashMap<String, Vec<Vec<f32>>>,
    edges: &HashMap<(String, String, String), Vec<(usize, usize)>>,
    node_counts: &HashMap<String, usize>,
    user_emb: &[f32],
) -> Vec<Vec<usize>> {
    let mut observations = Vec::new();

    for (node_type, count) in node_counts {
        for node_id in 0..*count {
            // Get anomaly score
            let anomaly = anomaly_scores
                .values()
                .filter_map(|model_scores| {
                    model_scores
                        .get(node_type)
                        .and_then(|scores| scores.get(node_id).copied())
                })
                .next()
                .unwrap_or(0.0);

            // Get embedding affinity with user
            let affinity = embeddings
                .get(node_type)
                .and_then(|embs| embs.get(node_id))
                .map(|emb| cosine_sim_f32(user_emb, emb))
                .unwrap_or(0.0);

            // Count degree
            let degree = count_node_edges(edges, node_type, node_id);

            // Create observations with different action priorities
            // to explore the full combinatorial space
            for priority_level in 0..NUM_CATEGORIES {
                let priority = match priority_level {
                    0 => 0.1,
                    1 => 0.3,
                    2 => 0.5,
                    3 => 0.7,
                    _ => 0.9,
                };

                // Heuristic outcome: high anomaly + low affinity = risky
                // Anomaly is the dominant signal (0.6 weight)
                let risk_signal = anomaly * 0.6
                    + (1.0 - affinity.clamp(-1.0, 1.0)) * 0.2
                    + if degree == 0 { 0.15 } else { 0.0 }
                    + priority * 0.05;
                let outcome = if risk_signal > 0.50 {
                    4 // very_risky
                } else if risk_signal > 0.40 {
                    3 // risky
                } else if risk_signal > 0.25 {
                    2 // moderate
                } else if risk_signal > 0.12 {
                    1 // low_risk
                } else {
                    0 // safe
                };

                observations.push(vec![
                    discretize_anomaly(anomaly),
                    discretize_affinity(affinity),
                    discretize_degree(degree),
                    priority_level,
                    outcome,
                ]);
            }
        }
    }

    // Ensure minimum data for EM
    if observations.is_empty() {
        for a in 0..NUM_CATEGORIES {
            for b in 0..NUM_CATEGORIES {
                observations.push(vec![a, b, 1, 1, a.max(b)]);
            }
        }
    }

    observations
}

fn cosine_sim_f32(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na * nb > 1e-10 {
        dot / (na * nb)
    } else {
        0.0
    }
}

pub fn count_node_edges(
    edges: &HashMap<(String, String, String), Vec<(usize, usize)>>,
    node_type: &str,
    node_id: usize,
) -> usize {
    let mut count = 0;
    for ((src_type, _, dst_type), edge_list) in edges {
        for &(src, dst) in edge_list {
            if (src_type == node_type && src == node_id)
                || (dst_type == node_type && dst == node_id)
            {
                count += 1;
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discretize() {
        assert_eq!(discretize_anomaly(0.1), 0); // very_low
        assert_eq!(discretize_anomaly(0.25), 1); // low
        assert_eq!(discretize_anomaly(0.5), 2); // medium
        assert_eq!(discretize_anomaly(0.7), 3); // high
        assert_eq!(discretize_anomaly(0.9), 4); // very_high

        assert_eq!(discretize_affinity(-0.5), 0); // very_negative
        assert_eq!(discretize_affinity(-0.1), 1); // negative
        assert_eq!(discretize_affinity(0.1), 2); // neutral
        assert_eq!(discretize_affinity(0.5), 3); // positive
        assert_eq!(discretize_affinity(0.8), 4); // strong_positive
    }

    #[test]
    fn test_build_pc() {
        let data: Vec<Vec<usize>> = (0..50)
            .map(|i| vec![i % 3, (i + 1) % 3, i % 3, i % 3, (i * 2) % 3])
            .collect();

        let (circuit, report) = build_fiduciary_pc(&data, 10);
        assert!(circuit.num_nodes() > 5);
        assert!(report.final_ll.is_finite());
        println!(
            "PC: {} nodes, final LL = {:.4}",
            circuit.num_nodes(),
            report.final_ll
        );
    }
}
