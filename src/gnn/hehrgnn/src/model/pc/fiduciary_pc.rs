//! PC-enhanced fiduciary analysis.
//!
//! Adds calibrated probabilities, lift factors, and counterfactual scenarios
//! to fiduciary recommendations using the trained Probabilistic Circuit.

use std::collections::HashMap;

use super::bridge::{
    self, discretize_affinity, discretize_anomaly, discretize_degree, discretize_priority,
    NUM_CATEGORIES, NUM_PC_VARS, VAR_NAMES,
};
use super::circuit::CompiledCircuit;
use super::query;

/// PC analysis result for a single fiduciary recommendation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PcAnalysis {
    /// Calibrated probability of the outcome given the evidence.
    /// P(outcome=risky | anomaly, affinity, degree, priority)
    pub risk_probability: f64,
    /// Full probability distribution over outcomes.
    pub outcome_distribution: OutcomeDistribution,
    /// Top risk factors with lift values.
    pub risk_factors: Vec<RiskFactor>,
    /// Counterfactual: what happens if we change one variable.
    pub counterfactuals: Vec<Counterfactual>,
    /// Confidence indicator.
    pub inference_type: String,
}

/// Distribution over outcome categories.
#[derive(Debug, Clone, serde::Serialize)]
pub struct OutcomeDistribution {
    pub safe: f64,
    pub low_risk: f64,
    pub moderate: f64,
    pub risky: f64,
    pub very_risky: f64,
}

/// A risk factor with its causal lift.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RiskFactor {
    /// Variable name.
    pub variable: String,
    /// Current value (discretized).
    pub current_value: String,
    /// Lift: how much this value increases risk vs baseline.
    pub lift: f64,
    /// Human-readable contribution.
    pub contribution: String,
}

/// A counterfactual scenario.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Counterfactual {
    /// What we change.
    pub scenario: String,
    /// New risk probability.
    pub new_risk_probability: f64,
    /// Risk reduction percentage.
    pub risk_reduction_pct: f64,
}

/// Category labels for interpretability.
const ANOMALY_LABELS: [&str; 5] = ["very_low", "low", "medium", "high", "very_high"];
const AFFINITY_LABELS: [&str; 5] = [
    "very_negative",
    "negative",
    "neutral",
    "positive",
    "strong_positive",
];
const DEGREE_LABELS: [&str; 5] = ["isolated", "leaf", "normal", "connected", "hub"];
const PRIORITY_LABELS: [&str; 5] = ["very_low", "low", "medium", "high", "very_high"];
const OUTCOME_LABELS: [&str; 5] = ["safe", "low_risk", "moderate", "risky", "very_risky"];

/// Run PC analysis for a fiduciary action candidate.
///
/// Given the features of the action candidate, compute:
/// 1. Calibrated risk probability P(risky | features)
/// 2. Full outcome distribution
/// 3. Lift of each variable on risk
/// 4. Counterfactual scenarios
pub fn analyze(
    circuit: &mut CompiledCircuit,
    anomaly_score: f32,
    affinity: f32,
    degree: usize,
    action_priority: f32,
) -> PcAnalysis {
    let anomaly_disc = discretize_anomaly(anomaly_score);
    let affinity_disc = discretize_affinity(affinity);
    let degree_disc = discretize_degree(degree);
    let priority_disc = discretize_priority(action_priority);

    // Evidence: observe feature variables, target variable (outcome) is unknown
    let evidence = vec![
        Some(anomaly_disc),
        Some(affinity_disc),
        Some(degree_disc),
        Some(priority_disc),
        None, // outcome_risk — we're querying this
    ];

    // 1. Compute conditional P(outcome | features)
    let outcome_probs = query::conditional(circuit, &evidence, &[4]);
    let probs = outcome_probs.get(&4).cloned().unwrap_or(vec![0.2; 5]);

    let outcome_dist = OutcomeDistribution {
        safe: *probs.get(0).unwrap_or(&0.2),
        low_risk: *probs.get(1).unwrap_or(&0.2),
        moderate: *probs.get(2).unwrap_or(&0.2),
        risky: *probs.get(3).unwrap_or(&0.2),
        very_risky: *probs.get(4).unwrap_or(&0.2),
    };

    // Risk probability = P(risky) + P(very_risky)
    let risk_probability = outcome_dist.risky + outcome_dist.very_risky;

    // 2. Compute lift factors for each feature variable
    let mut risk_factors = Vec::new();
    let feature_vars = [
        (0, anomaly_disc, ANOMALY_LABELS, "anomaly_level"),
        (1, affinity_disc, AFFINITY_LABELS, "embedding_affinity"),
        (2, degree_disc, DEGREE_LABELS, "node_degree"),
        (3, priority_disc, PRIORITY_LABELS, "action_priority"),
    ];

    for (var, val, labels, name) in &feature_vars {
        let l = query::lift(circuit, &evidence, *var, *val, 4, 4); // target: outcome=very_risky (bin 4)
        if l > 0.01 && l.is_finite() {
            let contribution = if l > 1.5 {
                format!(
                    "{} = {} increases risk by {:.0}%",
                    name,
                    labels[*val],
                    (l - 1.0) * 100.0
                )
            } else if l < 0.7 {
                format!(
                    "{} = {} decreases risk by {:.0}%",
                    name,
                    labels[*val],
                    (1.0 - l) * 100.0
                )
            } else {
                format!("{} = {} has neutral effect on risk", name, labels[*val])
            };

            risk_factors.push(RiskFactor {
                variable: name.to_string(),
                current_value: labels[*val].to_string(),
                lift: l,
                contribution,
            });
        }
    }

    // Sort by lift magnitude (most influential first)
    risk_factors.sort_by(|a, b| {
        (b.lift - 1.0)
            .abs()
            .partial_cmp(&(a.lift - 1.0).abs())
            .unwrap()
    });

    // 3. Counterfactual scenarios
    let mut counterfactuals = Vec::new();

    // What if anomaly drops to low?
    if anomaly_disc > 0 {
        let mut cf_evidence = evidence.clone();
        cf_evidence[0] = Some(0); // anomaly=low
        let cf_probs = query::conditional(circuit, &cf_evidence, &[4]);
        let cf_risk = cf_probs
            .get(&4)
            .map(|p| p.get(3).unwrap_or(&0.0) + p.get(4).unwrap_or(&0.0))
            .unwrap_or(risk_probability);
        let reduction = if risk_probability > 0.0 {
            (risk_probability - cf_risk) / risk_probability * 100.0
        } else {
            0.0
        };
        counterfactuals.push(Counterfactual {
            scenario: "If anomaly level drops to low".into(),
            new_risk_probability: cf_risk,
            risk_reduction_pct: reduction,
        });
    }

    // What if affinity improves to positive?
    if affinity_disc < 3 {
        let mut cf_evidence = evidence.clone();
        cf_evidence[1] = Some(3); // affinity=positive (bin 3)
        let cf_probs = query::conditional(circuit, &cf_evidence, &[4]);
        let cf_risk = cf_probs
            .get(&4)
            .map(|p| p.get(3).unwrap_or(&0.0) + p.get(4).unwrap_or(&0.0))
            .unwrap_or(risk_probability);
        let reduction = if risk_probability > 0.0 {
            (risk_probability - cf_risk) / risk_probability * 100.0
        } else {
            0.0
        };
        counterfactuals.push(Counterfactual {
            scenario: "If embedding affinity improves to positive".into(),
            new_risk_probability: cf_risk,
            risk_reduction_pct: reduction,
        });
    }

    // What if degree normalizes?
    if degree_disc != 2 {
        let mut cf_evidence = evidence.clone();
        cf_evidence[2] = Some(2); // degree=normal (bin 2)
        let cf_probs = query::conditional(circuit, &cf_evidence, &[4]);
        let cf_risk = cf_probs
            .get(&4)
            .map(|p| p.get(3).unwrap_or(&0.0) + p.get(4).unwrap_or(&0.0))
            .unwrap_or(risk_probability);
        let reduction = if risk_probability > 0.0 {
            (risk_probability - cf_risk) / risk_probability * 100.0
        } else {
            0.0
        };
        counterfactuals.push(Counterfactual {
            scenario: "If node connectivity normalizes".into(),
            new_risk_probability: cf_risk,
            risk_reduction_pct: reduction,
        });
    }

    // What if action priority increases?
    if priority_disc < 3 {
        let mut cf_evidence = evidence.clone();
        cf_evidence[3] = Some(3); // priority=high (bin 3)
        let cf_probs = query::conditional(circuit, &cf_evidence, &[4]);
        let cf_risk = cf_probs
            .get(&4)
            .map(|p| p.get(3).unwrap_or(&0.0) + p.get(4).unwrap_or(&0.0))
            .unwrap_or(risk_probability);
        let reduction = if risk_probability > 0.0 {
            (risk_probability - cf_risk) / risk_probability * 100.0
        } else {
            0.0
        };
        counterfactuals.push(Counterfactual {
            scenario: "If action priority is raised to high".into(),
            new_risk_probability: cf_risk,
            risk_reduction_pct: reduction,
        });
    }

    PcAnalysis {
        risk_probability,
        outcome_distribution: outcome_dist,
        risk_factors,
        counterfactuals,
        inference_type: "exact (tractable probabilistic circuit)".into(),
    }
}

/// Batch analyze: run PC analysis for multiple fiduciary candidates.
pub fn batch_analyze(
    circuit: &mut CompiledCircuit,
    candidates: &[(f32, f32, usize, f32)], // (anomaly, affinity, degree, priority)
) -> Vec<PcAnalysis> {
    candidates
        .iter()
        .map(|&(anomaly, affinity, degree, priority)| {
            analyze(circuit, anomaly, affinity, degree, priority)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::pc::bridge::build_fiduciary_pc;

    #[test]
    fn test_pc_analysis() {
        // Build a simple PC from synthetic data
        let data: Vec<Vec<usize>> = vec![
            // High anomaly → very_risky (bin 4)
            vec![4, 0, 0, 4, 4],
            vec![4, 1, 1, 3, 4],
            vec![3, 0, 2, 4, 3],
            vec![4, 0, 0, 4, 4],
            // Low anomaly → safe (bin 0)
            vec![0, 4, 2, 0, 0],
            vec![0, 3, 2, 1, 0],
            vec![1, 3, 2, 0, 0],
            vec![0, 4, 3, 0, 0],
            // Medium anomaly → moderate (bin 2)
            vec![2, 2, 2, 2, 2],
            vec![2, 1, 2, 2, 2],
            vec![2, 2, 3, 2, 1],
            vec![2, 2, 2, 3, 2],
        ];

        let (mut circuit, report) = build_fiduciary_pc(&data, 30);
        println!(
            "Trained PC: {} nodes, LL={:.4}",
            circuit.num_nodes(),
            report.final_ll
        );

        // Analyze a high-risk scenario
        let analysis = analyze(&mut circuit, 0.8, -0.3, 0, 0.9);
        println!("\n  High-risk scenario:");
        println!("    Risk probability: {:.4}", analysis.risk_probability);
        println!(
            "    Distribution: safe={:.3}, moderate={:.3}, risky={:.3}",
            analysis.outcome_distribution.safe,
            analysis.outcome_distribution.moderate,
            analysis.outcome_distribution.risky
        );
        for rf in &analysis.risk_factors {
            println!(
                "    Risk factor: {} (lift={:.2}): {}",
                rf.variable, rf.lift, rf.contribution
            );
        }
        for cf in &analysis.counterfactuals {
            println!(
                "    Counterfactual: {} → P(risky)={:.3} ({:.1}% reduction)",
                cf.scenario, cf.new_risk_probability, cf.risk_reduction_pct
            );
        }

        // Analyze a low-risk scenario
        let analysis_safe = analyze(&mut circuit, 0.1, 0.5, 2, 0.3);
        println!("\n  Low-risk scenario:");
        println!(
            "    Risk probability: {:.4}",
            analysis_safe.risk_probability
        );
        println!(
            "    Distribution: safe={:.3}, moderate={:.3}, risky={:.3}",
            analysis_safe.outcome_distribution.safe,
            analysis_safe.outcome_distribution.moderate,
            analysis_safe.outcome_distribution.risky
        );

        // Verify: high anomaly should have higher risk than low anomaly
        println!(
            "\n  Risk comparison: high_anomaly={:.4} vs low_anomaly={:.4}",
            analysis.risk_probability, analysis_safe.risk_probability
        );
    }
}
