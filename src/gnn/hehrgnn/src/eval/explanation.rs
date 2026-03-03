//! Prediction explainability for anomaly detection.
//!
//! Provides structured, human-readable explanations for why a transaction
//! was flagged as anomalous by the 7-signal ensemble.

use serde::Serialize;
use std::fmt;

/// A single signal's contribution to the anomaly score.
#[derive(Debug, Clone, Serialize)]
pub struct SignalTrace {
    /// Signal name: "Z-Score", "Novelty", "DistMult", "SAGE", etc.
    pub name: String,
    /// Raw (unnormalized) score from this signal.
    pub raw_score: f64,
    /// Min-max normalized score [0, 1].
    pub normalized_score: f64,
    /// Attention weight assigned by the ensemble (softmax).
    pub attention_weight: f64,
    /// Effective contribution = normalized_score × attention_weight.
    pub contribution: f64,
    /// Human-readable explanation of what triggered this signal.
    pub reason: String,
}

/// Graph neighborhood context for a transaction.
#[derive(Debug, Clone, Serialize)]
pub struct GraphContext {
    pub user: String,
    pub merchant: String,
    pub amount: f64,
    pub user_avg_amount: Option<f64>,
    pub merchant_avg_amount: Option<f64>,
    pub user_merchant_prior_visits: usize,
    pub merchant_category: Option<String>,
    pub anomaly_type: Option<String>,
}

/// Full explanation trace for one transaction's anomaly prediction.
#[derive(Debug, Clone, Serialize)]
pub struct AnomalyExplanation {
    /// Transaction index in the dataset.
    pub tx_idx: usize,
    /// Final ensemble score (normalized).
    pub ensemble_score: f64,
    /// Whether the ensemble flagged this as anomalous (score >= threshold).
    pub is_anomalous: bool,
    /// Signal traces, sorted by contribution (highest first).
    pub signals: Vec<SignalTrace>,
    /// Top 3 human-readable factors.
    pub top_factors: Vec<String>,
    /// Graph context.
    pub context: GraphContext,
}

impl AnomalyExplanation {
    /// Build an explanation from raw signal data.
    ///
    /// # Arguments
    /// - `tx_idx`: transaction index
    /// - `ensemble_score`: final normalized ensemble score
    /// - `threshold`: anomaly threshold (e.g., 0.5)
    /// - `raw_scores`: raw scores for each of the 7 signals
    /// - `norm_scores`: normalized scores for each signal
    /// - `attn_weights`: attention weights from ensemble fusion
    /// - `signal_names`: names of each signal
    /// - `reasons`: human-readable reasons for each signal
    /// - `context`: graph neighborhood context
    pub fn build(
        tx_idx: usize,
        ensemble_score: f64,
        threshold: f64,
        raw_scores: &[f64; 7],
        norm_scores: &[f64; 7],
        attn_weights: &[f64; 7],
        signal_names: &[&str; 7],
        reasons: &[String; 7],
        context: GraphContext,
    ) -> Self {
        let mut signals: Vec<SignalTrace> = (0..7)
            .map(|i| SignalTrace {
                name: signal_names[i].to_string(),
                raw_score: raw_scores[i],
                normalized_score: norm_scores[i],
                attention_weight: attn_weights[i],
                contribution: norm_scores[i] * attn_weights[i],
                reason: reasons[i].clone(),
            })
            .collect();

        // Sort by contribution (highest first)
        signals.sort_by(|a, b| b.contribution.partial_cmp(&a.contribution).unwrap());

        // Top 3 factors = top 3 signals with non-trivial contributions
        let top_factors: Vec<String> = signals
            .iter()
            .filter(|s| s.contribution > 0.01)
            .take(3)
            .map(|s| {
                if s.reason.is_empty() {
                    format!("{} (score={:.2})", s.name, s.normalized_score)
                } else {
                    format!("{} ({})", s.reason, s.name)
                }
            })
            .collect();

        AnomalyExplanation {
            tx_idx,
            ensemble_score,
            is_anomalous: ensemble_score >= threshold,
            signals,
            top_factors,
            context,
        }
    }
}

impl fmt::Display for AnomalyExplanation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.is_anomalous {
            "🚨 ANOMALOUS"
        } else {
            "✅ NORMAL"
        };

        writeln!(f, "  ═══ EXPLANATION: Tx #{} ═══", self.tx_idx)?;
        writeln!(
            f,
            "    Ensemble Score: {:.3} ({})",
            self.ensemble_score, status
        )?;

        // Context
        writeln!(
            f,
            "    {} → {} │ ${:.2}{}",
            self.context.user,
            self.context.merchant,
            self.context.amount,
            self.context
                .anomaly_type
                .as_ref()
                .map(|t| format!(" │ type: {}", t))
                .unwrap_or_default()
        )?;
        if let Some(avg) = self.context.merchant_avg_amount {
            write!(f, "    Merchant avg: ${:.2}", avg)?;
            if avg > 0.0 {
                write!(f, " (this tx is {:.1}×)", self.context.amount / avg)?;
            }
            writeln!(f)?;
        }
        writeln!(
            f,
            "    Prior visits to this merchant: {}",
            self.context.user_merchant_prior_visits
        )?;

        // Signal breakdown
        writeln!(f, "\n    Signal Breakdown (by contribution):")?;
        for (rank, s) in self.signals.iter().enumerate() {
            writeln!(
                f,
                "      #{} {:<10} │ score={:.2} │ wt={:.2} │ contrib={:.3}",
                rank + 1,
                s.name,
                s.normalized_score,
                s.attention_weight,
                s.contribution
            )?;
            if !s.reason.is_empty() {
                writeln!(f, "         → {}", s.reason)?;
            }
        }

        // Top factors
        if !self.top_factors.is_empty() {
            writeln!(f, "\n    Top Factors:")?;
            for (i, factor) in self.top_factors.iter().enumerate() {
                writeln!(f, "      {}. {}", i + 1, factor)?;
            }
        }

        Ok(())
    }
}

// ─── Per-signal reason generators ──────────────────────────────────────

/// Generate Z-Score reason.
pub fn zscore_reason(amount: f64, merchant: &str, merchant_avg: Option<f64>) -> String {
    match merchant_avg {
        Some(avg) if avg > 0.0 => {
            let ratio = amount / avg;
            if ratio > 2.0 {
                format!(
                    "${:.0} is {:.1}× the avg ${:.0} at {}",
                    amount, ratio, avg, merchant
                )
            } else if ratio < 0.3 {
                format!(
                    "${:.0} is unusually low ({:.1}× avg ${:.0}) at {}",
                    amount, ratio, avg, merchant
                )
            } else {
                format!("${:.0} is within normal range for {}", amount, merchant)
            }
        }
        _ => format!("${:.0} at {} (no baseline available)", amount, merchant),
    }
}

/// Generate Novelty reason.
pub fn novelty_reason(user: &str, merchant: &str, prior_visits: usize) -> String {
    if prior_visits == 0 {
        format!(
            "First time {} transacted at {} (0 prior visits)",
            user, merchant
        )
    } else {
        format!(
            "{} has visited {} {} time(s) before",
            user, merchant, prior_visits
        )
    }
}

/// Generate DistMult reason.
pub fn distmult_reason(
    user: &str,
    merchant: &str,
    raw_score: f64,
    normalized_score: f64,
) -> String {
    if normalized_score > 0.5 {
        format!(
            "KG triple ({}, transacts_at, {}) scored {:.2} (implausible)",
            user, merchant, raw_score
        )
    } else {
        format!(
            "KG triple ({}, transacts_at, {}) scored {:.2} (plausible)",
            user, merchant, raw_score
        )
    }
}

/// Generate GNN embedding reason (SAGE, RGCN, GAT, GT).
pub fn gnn_reason(model_name: &str, normalized_score: f64) -> String {
    if normalized_score > 0.7 {
        format!(
            "{} embedding highly anomalous (score={:.2}): node features deviate from neighborhood",
            model_name, normalized_score
        )
    } else if normalized_score > 0.4 {
        format!(
            "{} embedding moderately unusual (score={:.2})",
            model_name, normalized_score
        )
    } else {
        format!(
            "{} embedding within normal range (score={:.2})",
            model_name, normalized_score
        )
    }
}
