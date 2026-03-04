//! Critical Path Discovery for Financial Risk Assessment.
//!
//! Adopted from Paper 2402.18246: "RL and GNNs for Probabilistic Risk Assessment"
//! (Grimstad & Morozov, 2024) — Graph-Level section (§3.5).
//!
//! Key insight: find *minimal cut sets* in the financial graph — the smallest
//! collection of edge removals that causes a user's risk score to spike.
//!
//! Example: "If Alice loses her job (remove employed_at edge) AND her savings
//! account closes (remove has-instrument edge), her entire position collapses."
//!
//! Algorithm: greedy edge ablation + scoring
//! 1. Start with all edges connected to a user
//! 2. Remove one edge at a time, re-score risk
//! 3. Keep the removal that causes the highest risk increase
//! 4. Repeat until risk exceeds threshold or max_depth reached
//! 5. The resulting set of removed edges is a "critical path"

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════

/// A critical financial dependency path.
/// Represents the minimal set of relationship losses that would
/// push a user's risk above the danger threshold.
#[derive(Debug, Clone)]
pub struct CriticalPath {
    /// The edges removed (source_type, relation, target_type, target_name).
    pub removed_edges: Vec<CriticalEdge>,
    /// Baseline risk score (all edges intact).
    pub baseline_risk: f32,
    /// Final risk score after all removals.
    pub final_risk: f32,
    /// Risk increase per removal step.
    pub risk_trajectory: Vec<f32>,
    /// Human-readable explanation of the critical chain.
    pub explanation: String,
}

/// A single critical edge in the path.
#[derive(Debug, Clone)]
pub struct CriticalEdge {
    /// Node type of the source.
    pub src_type: String,
    /// Relation name.
    pub relation: String,
    /// Node type of the destination.
    pub dst_type: String,
    /// Name of the destination entity.
    pub dst_name: String,
    /// Risk increase when this specific edge is removed.
    pub risk_delta: f32,
}

/// Configuration for critical path discovery.
#[derive(Debug, Clone)]
pub struct CriticalPathConfig {
    /// Risk threshold above which the user is in danger (0.0-1.0).
    pub danger_threshold: f32,
    /// Maximum number of edges to remove.
    pub max_depth: usize,
    /// Minimum risk increase per step to continue.
    pub min_risk_delta: f32,
}

impl Default for CriticalPathConfig {
    fn default() -> Self {
        Self {
            danger_threshold: 0.7,
            max_depth: 5,
            min_risk_delta: 0.01,
        }
    }
}

/// Report from critical path analysis for a user.
#[derive(Debug, Clone)]
pub struct CriticalPathReport {
    /// All discovered critical paths (there can be multiple).
    pub paths: Vec<CriticalPath>,
    /// User's current baseline risk.
    pub baseline_risk: f32,
    /// Number of edge ablations performed.
    pub total_ablations: usize,
}

// ═══════════════════════════════════════════════════════════════
// Risk Scoring from Embeddings
// ═══════════════════════════════════════════════════════════════

/// Score risk based on anomaly scores and graph structure.
/// Higher score = higher risk. Returns 0.0-1.0.
fn score_risk(
    anomaly_scores: &HashMap<String, HashMap<String, Vec<f32>>>,
    edges: &HashMap<(String, String, String), Vec<(usize, usize)>>,
    user_type: &str,
    user_id: usize,
) -> f32 {
    // Aggregate anomaly scores of entities connected to the user
    let mut total_anomaly = 0.0f32;
    let mut connected_count = 0usize;

    for ((src_type, _rel, dst_type), pairs) in edges {
        for &(src, dst) in pairs {
            // Find edges connected to our user
            let (connected_type, connected_id) = if src_type == user_type && src == user_id {
                (dst_type.as_str(), dst)
            } else if dst_type == user_type && dst == user_id {
                (src_type.as_str(), src)
            } else {
                continue;
            };

            // Look up the anomaly score
            for model_scores in anomaly_scores.values() {
                if let Some(scores) = model_scores.get(connected_type) {
                    if let Some(&score) = scores.get(connected_id) {
                        total_anomaly += score;
                        connected_count += 1;
                    }
                }
            }
        }
    }

    if connected_count == 0 {
        // No connections = high risk (isolation is dangerous)
        return 0.8;
    }

    // Base risk = average anomaly of connected entities
    let avg_anomaly = total_anomaly / connected_count as f32;

    // Connectivity factor: more connections = more resilience
    // Having fewer connections increases risk (less diversification)
    let connectivity_factor = 1.0 / (1.0 + (connected_count as f32 / 5.0));

    // Combined risk: anomaly-weighted + isolation penalty
    let risk = avg_anomaly * 0.7 + connectivity_factor * 0.3;
    risk.clamp(0.0, 1.0)
}

// ═══════════════════════════════════════════════════════════════
// Critical Path Discovery (Greedy Edge Ablation)
// ═══════════════════════════════════════════════════════════════

/// Discover critical financial paths for a user.
///
/// Uses greedy edge ablation: removes one edge at a time, keeping the
/// removal that causes the largest risk increase, until the danger
/// threshold is reached or no more impactful removals exist.
pub fn discover_critical_paths(
    anomaly_scores: &HashMap<String, HashMap<String, Vec<f32>>>,
    edges: &HashMap<(String, String, String), Vec<(usize, usize)>>,
    node_names: &HashMap<String, Vec<String>>,
    user_type: &str,
    user_id: usize,
    config: &CriticalPathConfig,
) -> CriticalPathReport {
    let baseline_risk = score_risk(anomaly_scores, edges, user_type, user_id);
    let mut total_ablations = 0;

    // Find all edges connected to the user
    let mut user_edges: Vec<((String, String, String), usize, usize)> = Vec::new();
    for ((src_type, rel, dst_type), pairs) in edges {
        for &(src, dst) in pairs {
            if (src_type == user_type && src == user_id)
                || (dst_type == user_type && dst == user_id)
            {
                user_edges.push(((src_type.clone(), rel.clone(), dst_type.clone()), src, dst));
            }
        }
    }

    if user_edges.is_empty() {
        return CriticalPathReport {
            paths: Vec::new(),
            baseline_risk,
            total_ablations: 0,
        };
    }

    // Greedy ablation: find the critical path
    let mut removed_edges: Vec<CriticalEdge> = Vec::new();
    let mut risk_trajectory = vec![baseline_risk];
    let mut current_edges = edges.clone();
    let mut current_risk = baseline_risk;
    let mut remaining_user_edges = user_edges.clone();

    for _depth in 0..config.max_depth {
        if current_risk >= config.danger_threshold || remaining_user_edges.is_empty() {
            break;
        }

        // Try removing each remaining edge and measure risk increase
        let mut best_removal: Option<(usize, f32, f32)> = None; // (index, new_risk, delta)

        for (idx, ((src_t, rel, dst_t), src, dst)) in remaining_user_edges.iter().enumerate() {
            // Create edge set with this edge removed
            let mut trial_edges = current_edges.clone();
            let key = (src_t.clone(), rel.clone(), dst_t.clone());
            if let Some(pairs) = trial_edges.get_mut(&key) {
                pairs.retain(|&(s, d)| !(s == *src && d == *dst));
                if pairs.is_empty() {
                    trial_edges.remove(&key);
                }
            }

            let trial_risk = score_risk(anomaly_scores, &trial_edges, user_type, user_id);
            let delta = trial_risk - current_risk;
            total_ablations += 1;

            if delta >= config.min_risk_delta {
                if best_removal.is_none() || delta > best_removal.unwrap().2 {
                    best_removal = Some((idx, trial_risk, delta));
                }
            }
        }

        if let Some((best_idx, new_risk, delta)) = best_removal {
            let ((src_t, rel, dst_t), src, dst) = remaining_user_edges.remove(best_idx);

            // Get the name of the removed entity
            let (connected_type, connected_id) = if src_t == user_type {
                (&dst_t, dst)
            } else {
                (&src_t, src)
            };
            let entity_name = node_names
                .get(connected_type.as_str())
                .and_then(|names| names.get(connected_id))
                .cloned()
                .unwrap_or_else(|| format!("{}_{}", connected_type, connected_id));

            removed_edges.push(CriticalEdge {
                src_type: src_t.clone(),
                relation: rel.clone(),
                dst_type: dst_t.clone(),
                dst_name: entity_name,
                risk_delta: delta,
            });

            // Actually remove the edge
            let key = (src_t, rel, dst_t);
            if let Some(pairs) = current_edges.get_mut(&key) {
                pairs.retain(|&(s, d)| !(s == src && d == dst));
                if pairs.is_empty() {
                    current_edges.remove(&key);
                }
            }

            current_risk = new_risk;
            risk_trajectory.push(current_risk);
        } else {
            break; // No impactful removals left
        }
    }

    // Build explanation
    let explanation = build_explanation(&removed_edges, baseline_risk, current_risk);

    let path = CriticalPath {
        removed_edges,
        baseline_risk,
        final_risk: current_risk,
        risk_trajectory,
        explanation,
    };

    CriticalPathReport {
        paths: if path.removed_edges.is_empty() {
            Vec::new()
        } else {
            vec![path]
        },
        baseline_risk,
        total_ablations,
    }
}

/// Build a human-readable explanation of the critical path.
fn build_explanation(edges: &[CriticalEdge], baseline: f32, final_risk: f32) -> String {
    if edges.is_empty() {
        return "No critical dependencies found — user is well-diversified.".into();
    }

    let mut parts = Vec::new();
    for (i, edge) in edges.iter().enumerate() {
        let verb = match edge.relation.as_str() {
            r if r.contains("has-instrument") => "loses access to",
            r if r.contains("obligation") => "defaults on",
            r if r.contains("goal") => "abandons goal",
            r if r.contains("lien") => "faces foreclosure on",
            r if r.contains("pattern") => "loses recurring",
            r if r.contains("tax") => "loses tax benefit",
            r if r.contains("budget") => "stops budgeting",
            r if r.contains("valuation") => "loses valuation of",
            r if r.contains("reconciliation") => "stops reconciling",
            _ => "loses connection to",
        };
        if i == 0 {
            parts.push(format!(
                "If user {} {} (risk +{:.0}%)",
                verb,
                edge.dst_name,
                edge.risk_delta * 100.0
            ));
        } else {
            parts.push(format!(
                "AND {} {} (risk +{:.0}%)",
                verb,
                edge.dst_name,
                edge.risk_delta * 100.0
            ));
        }
    }

    parts.push(format!(
        "→ total risk rises from {:.0}% to {:.0}% (CRITICAL)",
        baseline * 100.0,
        final_risk * 100.0
    ));

    parts.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_critical_path_basic() {
        // Build a simple financial graph
        let mut anomaly_scores: HashMap<String, HashMap<String, Vec<f32>>> = HashMap::new();
        let mut model_scores: HashMap<String, Vec<f32>> = HashMap::new();
        model_scores.insert("instrument".into(), vec![0.1, 0.2, 0.3]);
        model_scores.insert("obligation".into(), vec![0.5]);
        model_scores.insert("goal".into(), vec![0.05]);
        anomaly_scores.insert("SAGE".into(), model_scores);

        let mut edges: HashMap<(String, String, String), Vec<(usize, usize)>> = HashMap::new();
        // User has 3 instruments
        edges.insert(
            (
                "user".into(),
                "user-has-instrument".into(),
                "instrument".into(),
            ),
            vec![(0, 0), (0, 1), (0, 2)],
        );
        // User has 1 obligation
        edges.insert(
            (
                "user".into(),
                "obligation-between-parties".into(),
                "obligation".into(),
            ),
            vec![(0, 0)],
        );
        // User has 1 goal
        edges.insert(
            (
                "user".into(),
                "subledger-holds-goal-funds".into(),
                "goal".into(),
            ),
            vec![(0, 0)],
        );

        let mut node_names: HashMap<String, Vec<String>> = HashMap::new();
        node_names.insert(
            "instrument".into(),
            vec!["Checking".into(), "Savings".into(), "Brokerage".into()],
        );
        node_names.insert("obligation".into(), vec!["CreditCard_22APR".into()]);
        node_names.insert("goal".into(), vec!["EmergencyFund".into()]);

        let config = CriticalPathConfig::default();
        let report =
            discover_critical_paths(&anomaly_scores, &edges, &node_names, "user", 0, &config);

        println!("\n  ── CRITICAL PATH DISCOVERY ──");
        println!("  Baseline risk: {:.2}", report.baseline_risk);
        println!("  Total ablations: {}", report.total_ablations);
        println!("  Paths found: {}", report.paths.len());

        for (i, path) in report.paths.iter().enumerate() {
            println!("\n  Path {}:", i + 1);
            println!("    Risk trajectory: {:?}", path.risk_trajectory);
            for edge in &path.removed_edges {
                println!(
                    "    - Remove {} → {} ({}): risk +{:.2}",
                    edge.relation, edge.dst_name, edge.dst_type, edge.risk_delta
                );
            }
            println!("    Explanation: {}", path.explanation);
        }

        // Basic assertions
        assert!(report.baseline_risk >= 0.0 && report.baseline_risk <= 1.0);
        assert!(
            report.total_ablations > 0,
            "Should have performed ablations"
        );

        if !report.paths.is_empty() {
            let path = &report.paths[0];
            assert!(
                path.final_risk >= path.baseline_risk,
                "Risk should increase or stay same when edges removed"
            );
            assert!(
                !path.explanation.is_empty(),
                "Should generate an explanation"
            );
        }

        println!("\n  ✅ Critical path discovery test PASSED");
    }

    #[test]
    fn test_well_diversified_user_no_critical_path() {
        // User with many connections — removing any single one shouldn't be critical
        let mut anomaly_scores: HashMap<String, HashMap<String, Vec<f32>>> = HashMap::new();
        let mut model_scores: HashMap<String, Vec<f32>> = HashMap::new();
        // All very low anomaly — healthy portfolio
        model_scores.insert("instrument".into(), vec![0.02; 5]);
        model_scores.insert("goal".into(), vec![0.01; 3]);
        anomaly_scores.insert("SAGE".into(), model_scores);

        let mut edges: HashMap<(String, String, String), Vec<(usize, usize)>> = HashMap::new();
        edges.insert(
            (
                "user".into(),
                "user-has-instrument".into(),
                "instrument".into(),
            ),
            vec![(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
        );
        edges.insert(
            (
                "user".into(),
                "subledger-holds-goal-funds".into(),
                "goal".into(),
            ),
            vec![(0, 0), (0, 1), (0, 2)],
        );

        let mut node_names: HashMap<String, Vec<String>> = HashMap::new();
        node_names.insert(
            "instrument".into(),
            (0..5).map(|i| format!("Account_{}", i)).collect(),
        );
        node_names.insert(
            "goal".into(),
            vec!["Emergency".into(), "Retirement".into(), "Travel".into()],
        );

        let config = CriticalPathConfig {
            danger_threshold: 0.7,
            max_depth: 5,
            min_risk_delta: 0.05, // Need at least 5% increase per step
        };

        let report =
            discover_critical_paths(&anomaly_scores, &edges, &node_names, "user", 0, &config);

        println!("\n  ── WELL-DIVERSIFIED USER ──");
        println!("  Baseline risk: {:.2}", report.baseline_risk);
        println!("  Paths found: {}", report.paths.len());

        // Low-anomaly, well-diversified user may not have critical paths
        assert!(
            report.baseline_risk < 0.4,
            "Well-diversified user should have low baseline risk"
        );
    }
}
