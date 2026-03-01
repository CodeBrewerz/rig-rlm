//! Retrainer: fine-tunes the GNN model using collected feedback.
//!
//! Converts accepted/rejected/corrected feedback into training signals:
//! - Accepted matches → positive pairs (increase score)
//! - Rejected matches → negative pairs (decrease score)
//! - Corrected matches → positive for correct, negative for predicted
//! - Classification corrections → supervised labels

use serde::Serialize;
use std::collections::HashMap;

use super::collector::{FeedbackStore, Verdict};

/// A training signal derived from feedback.
#[derive(Debug, Clone, Serialize)]
pub struct TrainingSignal {
    pub signal_type: SignalType,
    pub src_type: String,
    pub src_id: usize,
    pub dst_type: Option<String>,
    pub dst_id: Option<usize>,
    pub label: f32,  // 1.0 = positive, 0.0 = negative
    pub weight: f32, // confidence weight
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SignalType {
    PositivePair,
    NegativePair,
    ClassLabel,
}

/// Configuration for the retraining process.
#[derive(Debug, Clone)]
pub struct RetrainConfig {
    /// Minimum feedback entries before triggering retrain.
    pub min_feedback: usize,
    /// Weight multiplier for corrected feedback (more important).
    pub correction_weight: f32,
    /// Weight for accepted predictions (reinforcement).
    pub accept_weight: f32,
    /// Weight for rejected predictions.
    pub reject_weight: f32,
}

impl Default for RetrainConfig {
    fn default() -> Self {
        Self {
            min_feedback: 10,
            correction_weight: 2.0,
            accept_weight: 0.5,
            reject_weight: 1.0,
        }
    }
}

/// Convert collected feedback into training signals.
pub fn feedback_to_signals(store: &FeedbackStore, config: &RetrainConfig) -> Vec<TrainingSignal> {
    let mut signals = Vec::new();

    for entry in store.entries() {
        match entry.prediction_type.as_str() {
            "match" => {
                match entry.verdict {
                    Verdict::Accepted => {
                        // Reinforce the predicted pair
                        signals.push(TrainingSignal {
                            signal_type: SignalType::PositivePair,
                            src_type: entry.prediction.src_type.clone(),
                            src_id: entry.prediction.src_id,
                            dst_type: entry.prediction.dst_type.clone(),
                            dst_id: entry.prediction.dst_id,
                            label: 1.0,
                            weight: config.accept_weight,
                        });
                    }
                    Verdict::Rejected => {
                        // Penalize the predicted pair
                        signals.push(TrainingSignal {
                            signal_type: SignalType::NegativePair,
                            src_type: entry.prediction.src_type.clone(),
                            src_id: entry.prediction.src_id,
                            dst_type: entry.prediction.dst_type.clone(),
                            dst_id: entry.prediction.dst_id,
                            label: 0.0,
                            weight: config.reject_weight,
                        });
                    }
                    Verdict::Corrected => {
                        // Penalize wrong prediction
                        signals.push(TrainingSignal {
                            signal_type: SignalType::NegativePair,
                            src_type: entry.prediction.src_type.clone(),
                            src_id: entry.prediction.src_id,
                            dst_type: entry.prediction.dst_type.clone(),
                            dst_id: entry.prediction.dst_id,
                            label: 0.0,
                            weight: config.correction_weight,
                        });
                        // Reinforce correct pair from correction
                        if let Some(ref corr) = entry.correction {
                            if corr.correct_dst_id.is_some() {
                                signals.push(TrainingSignal {
                                    signal_type: SignalType::PositivePair,
                                    src_type: entry.prediction.src_type.clone(),
                                    src_id: entry.prediction.src_id,
                                    dst_type: entry.prediction.dst_type.clone(),
                                    dst_id: corr.correct_dst_id,
                                    label: 1.0,
                                    weight: config.correction_weight,
                                });
                            }
                        }
                    }
                }
            }
            "classify" => {
                match entry.verdict {
                    Verdict::Accepted => {
                        if let Some(class) = entry.prediction.predicted_class {
                            signals.push(TrainingSignal {
                                signal_type: SignalType::ClassLabel,
                                src_type: entry.prediction.src_type.clone(),
                                src_id: entry.prediction.src_id,
                                dst_type: None,
                                dst_id: Some(class),
                                label: 1.0,
                                weight: config.accept_weight,
                            });
                        }
                    }
                    Verdict::Corrected => {
                        if let Some(ref corr) = entry.correction {
                            if let Some(correct_class) = corr.correct_class {
                                signals.push(TrainingSignal {
                                    signal_type: SignalType::ClassLabel,
                                    src_type: entry.prediction.src_type.clone(),
                                    src_id: entry.prediction.src_id,
                                    dst_type: None,
                                    dst_id: Some(correct_class),
                                    label: 1.0,
                                    weight: config.correction_weight,
                                });
                            }
                        }
                    }
                    Verdict::Rejected => {
                        // Rejected with no correction — can't do much
                    }
                }
            }
            _ => {} // Other types (anomaly) — no retraining signal
        }
    }

    signals
}

/// Check if enough feedback has been collected to justify retraining.
pub fn should_retrain(store: &FeedbackStore, config: &RetrainConfig) -> RetrainDecision {
    let stats = store.stats();
    let total = stats.total_entries;

    if total < config.min_feedback {
        return RetrainDecision {
            should_retrain: false,
            reason: format!(
                "Not enough feedback ({} < {} min)",
                total, config.min_feedback
            ),
            num_signals: 0,
            acceptance_rates: HashMap::new(),
        };
    }

    let signals = feedback_to_signals(store, config);

    let mut acceptance_rates = HashMap::new();
    for (pred_type, ts) in &stats.by_type {
        acceptance_rates.insert(pred_type.clone(), ts.acceptance_rate());
    }

    RetrainDecision {
        should_retrain: true,
        reason: format!(
            "{} feedback entries collected, {} training signals",
            total,
            signals.len()
        ),
        num_signals: signals.len(),
        acceptance_rates,
    }
}

/// Decision about whether to retrain.
#[derive(Debug, Clone, Serialize)]
pub struct RetrainDecision {
    pub should_retrain: bool,
    pub reason: String,
    pub num_signals: usize,
    pub acceptance_rates: HashMap<String, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feedback::collector::*;

    fn make_match_feedback(verdict: Verdict, correction: Option<usize>) -> FeedbackEntry {
        FeedbackEntry {
            id: String::new(),
            timestamp: "2026-01-01".into(),
            prediction_type: "match".into(),
            prediction: PredictionRecord {
                src_type: "tx".into(),
                src_id: 0,
                dst_type: Some("account".into()),
                dst_id: Some(1),
                predicted_class: None,
                predicted_score: Some(0.9),
            },
            verdict,
            correction: correction.map(|id| CorrectionRecord {
                correct_dst_id: Some(id),
                correct_class: None,
                notes: None,
            }),
        }
    }

    #[test]
    fn test_feedback_to_signals() {
        let mut store = FeedbackStore::new();

        store.record(make_match_feedback(Verdict::Accepted, None));
        store.record(make_match_feedback(Verdict::Rejected, None));
        store.record(make_match_feedback(Verdict::Corrected, Some(5)));

        let config = RetrainConfig::default();
        let signals = feedback_to_signals(&store, &config);

        // Accepted → 1 positive
        // Rejected → 1 negative
        // Corrected → 1 negative + 1 positive (correction)
        assert_eq!(signals.len(), 4);
        assert_eq!(
            signals
                .iter()
                .filter(|s| s.signal_type == SignalType::PositivePair)
                .count(),
            2
        );
        assert_eq!(
            signals
                .iter()
                .filter(|s| s.signal_type == SignalType::NegativePair)
                .count(),
            2
        );
    }

    #[test]
    fn test_should_retrain() {
        let store = FeedbackStore::new();
        let config = RetrainConfig {
            min_feedback: 2,
            ..Default::default()
        };

        let decision = should_retrain(&store, &config);
        assert!(!decision.should_retrain);

        let mut store2 = FeedbackStore::new();
        store2.record(make_match_feedback(Verdict::Accepted, None));
        store2.record(make_match_feedback(Verdict::Rejected, None));
        store2.record(make_match_feedback(Verdict::Corrected, Some(5)));

        let decision2 = should_retrain(&store2, &config);
        assert!(decision2.should_retrain);
        assert_eq!(decision2.num_signals, 4);
    }
}
