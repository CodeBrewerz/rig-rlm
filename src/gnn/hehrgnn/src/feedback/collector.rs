//! Feedback collector: stores user feedback on predictions for retraining.
//!
//! After the GNN makes predictions (match rankings, classifications, anomaly
//! scores), users accept/reject/correct them. This module stores that feedback
//! as structured data that the retrainer can use.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// A single feedback entry for a prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackEntry {
    /// Unique feedback ID.
    pub id: String,
    /// Timestamp (ISO 8601).
    pub timestamp: String,
    /// Type of prediction: "match", "classify", "anomaly".
    pub prediction_type: String,
    /// The prediction that was made.
    pub prediction: PredictionRecord,
    /// User's verdict.
    pub verdict: Verdict,
    /// Optional correction (e.g., correct class, correct match).
    #[serde(default)]
    pub correction: Option<CorrectionRecord>,
}

/// Record of what was predicted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRecord {
    pub src_type: String,
    pub src_id: usize,
    #[serde(default)]
    pub dst_type: Option<String>,
    #[serde(default)]
    pub dst_id: Option<usize>,
    #[serde(default)]
    pub predicted_class: Option<usize>,
    #[serde(default)]
    pub predicted_score: Option<f32>,
}

/// User verdict on a prediction.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Accepted,
    Rejected,
    Corrected,
}

/// A correction provided by the user.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionRecord {
    /// Correct destination (for match corrections).
    #[serde(default)]
    pub correct_dst_id: Option<usize>,
    /// Correct class (for classification corrections).
    #[serde(default)]
    pub correct_class: Option<usize>,
    /// Free-text notes.
    #[serde(default)]
    pub notes: Option<String>,
}

/// Feedback store: collects and persists feedback entries.
#[derive(Debug)]
pub struct FeedbackStore {
    entries: Vec<FeedbackEntry>,
    file_path: Option<String>,
    next_id: usize,
}

impl FeedbackStore {
    /// Create a new in-memory feedback store.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            file_path: None,
            next_id: 0,
        }
    }

    /// Create a feedback store backed by a JSONL file.
    pub fn with_file(path: &str) -> Self {
        let mut store = Self::new();
        store.file_path = Some(path.to_string());

        // Load existing entries if file exists
        if Path::new(path).exists() {
            if let Ok(content) = std::fs::read_to_string(path) {
                for line in content.lines() {
                    if let Ok(entry) = serde_json::from_str::<FeedbackEntry>(line) {
                        store.next_id = store.next_id.max(
                            entry
                                .id
                                .strip_prefix("fb_")
                                .and_then(|s| s.parse::<usize>().ok())
                                .unwrap_or(0)
                                + 1,
                        );
                        store.entries.push(entry);
                    }
                }
            }
        }

        store
    }

    /// Record a feedback entry.
    pub fn record(&mut self, mut entry: FeedbackEntry) {
        entry.id = format!("fb_{}", self.next_id);
        self.next_id += 1;

        // Persist to file if configured
        if let Some(ref path) = self.file_path {
            use std::io::Write;
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                if let Ok(json) = serde_json::to_string(&entry) {
                    let _ = writeln!(file, "{}", json);
                }
            }
        }

        self.entries.push(entry);
    }

    /// Get all entries.
    pub fn entries(&self) -> &[FeedbackEntry] {
        &self.entries
    }

    /// Get entries filtered by verdict.
    pub fn by_verdict(&self, verdict: Verdict) -> Vec<&FeedbackEntry> {
        self.entries
            .iter()
            .filter(|e| e.verdict == verdict)
            .collect()
    }

    /// Get entries filtered by prediction type.
    pub fn by_type(&self, prediction_type: &str) -> Vec<&FeedbackEntry> {
        self.entries
            .iter()
            .filter(|e| e.prediction_type == prediction_type)
            .collect()
    }

    /// Compute feedback statistics.
    pub fn stats(&self) -> FeedbackStats {
        let mut by_type: HashMap<String, TypeStats> = HashMap::new();

        for entry in &self.entries {
            let ts = by_type.entry(entry.prediction_type.clone()).or_default();
            ts.total += 1;
            match entry.verdict {
                Verdict::Accepted => ts.accepted += 1,
                Verdict::Rejected => ts.rejected += 1,
                Verdict::Corrected => ts.corrected += 1,
            }
        }

        FeedbackStats {
            total_entries: self.entries.len(),
            by_type,
        }
    }
}

/// Statistics about collected feedback.
#[derive(Debug, Clone, Serialize)]
pub struct FeedbackStats {
    pub total_entries: usize,
    pub by_type: HashMap<String, TypeStats>,
}

/// Per-type feedback breakdown.
#[derive(Debug, Clone, Default, Serialize)]
pub struct TypeStats {
    pub total: usize,
    pub accepted: usize,
    pub rejected: usize,
    pub corrected: usize,
}

impl TypeStats {
    pub fn acceptance_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.accepted as f64 / self.total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedback_store() {
        let mut store = FeedbackStore::new();

        store.record(FeedbackEntry {
            id: String::new(),
            timestamp: "2026-01-01T00:00:00Z".into(),
            prediction_type: "match".into(),
            prediction: PredictionRecord {
                src_type: "transaction".into(),
                src_id: 0,
                dst_type: Some("account".into()),
                dst_id: Some(1),
                predicted_class: None,
                predicted_score: Some(0.95),
            },
            verdict: Verdict::Accepted,
            correction: None,
        });

        store.record(FeedbackEntry {
            id: String::new(),
            timestamp: "2026-01-01T00:01:00Z".into(),
            prediction_type: "match".into(),
            prediction: PredictionRecord {
                src_type: "transaction".into(),
                src_id: 1,
                dst_type: Some("account".into()),
                dst_id: Some(2),
                predicted_class: None,
                predicted_score: Some(0.3),
            },
            verdict: Verdict::Corrected,
            correction: Some(CorrectionRecord {
                correct_dst_id: Some(5),
                correct_class: None,
                notes: Some("Wrong account".into()),
            }),
        });

        assert_eq!(store.entries().len(), 2);
        assert_eq!(store.entries()[0].id, "fb_0");
        assert_eq!(store.entries()[1].id, "fb_1");

        let stats = store.stats();
        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.by_type["match"].accepted, 1);
        assert_eq!(stats.by_type["match"].corrected, 1);
    }

    #[test]
    fn test_feedback_file_persistence() {
        let path = "/tmp/test_feedback.jsonl";
        let _ = std::fs::remove_file(path); // clean slate

        {
            let mut store = FeedbackStore::with_file(path);
            store.record(FeedbackEntry {
                id: String::new(),
                timestamp: "2026-01-01".into(),
                prediction_type: "classify".into(),
                prediction: PredictionRecord {
                    src_type: "tx".into(),
                    src_id: 0,
                    dst_type: None,
                    dst_id: None,
                    predicted_class: Some(3),
                    predicted_score: Some(0.8),
                },
                verdict: Verdict::Rejected,
                correction: None,
            });
        }

        // Reload from file
        let store2 = FeedbackStore::with_file(path);
        assert_eq!(store2.entries().len(), 1);
        assert_eq!(store2.entries()[0].verdict, Verdict::Rejected);

        let _ = std::fs::remove_file(path);
    }
}
