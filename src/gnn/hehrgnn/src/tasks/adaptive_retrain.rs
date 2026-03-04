//! Adaptive retrain monitor (InstantGNN §4.5).
//!
//! Tracks cumulative representation drift ‖ΔZ‖_F and recommends
//! retraining when the drift exceeds threshold θ.

use serde::Serialize;

/// Monitors representation drift and decides when to retrain.
#[derive(Debug, Clone)]
pub struct RetrainMonitor {
    /// Cumulative Frobenius norm of representation change since last retrain.
    cumulative_drift: f64,
    /// Threshold θ: retrain when cumulative_drift > theta.
    theta: f64,
    /// History of (timestamp, drift_at_retrain) for adaptive scheduling.
    retrain_history: Vec<(String, f64)>,
    /// Number of mutations since last retrain.
    mutations_since_retrain: usize,
}

/// Status report from the retrain monitor.
#[derive(Debug, Clone, Serialize)]
pub struct RetrainStatus {
    pub cumulative_drift: f64,
    pub threshold: f64,
    pub retrain_recommended: bool,
    pub mutations_since_retrain: usize,
    pub total_retrains: usize,
}

impl RetrainMonitor {
    /// Create a new monitor with threshold θ.
    pub fn new(theta: f64) -> Self {
        Self {
            cumulative_drift: 0.0,
            theta,
            retrain_history: Vec::new(),
            mutations_since_retrain: 0,
        }
    }

    /// Record a representation change from a mutation.
    /// `delta_z` is the Frobenius norm of the embedding change.
    pub fn record_drift(&mut self, delta_z: f64) {
        self.cumulative_drift += delta_z;
        self.mutations_since_retrain += 1;
    }

    /// Check if retraining is recommended.
    pub fn should_retrain(&self) -> bool {
        self.cumulative_drift > self.theta
    }

    /// Record that a retrain occurred — resets drift counter.
    pub fn record_retrain(&mut self, timestamp: &str) {
        self.retrain_history
            .push((timestamp.to_string(), self.cumulative_drift));
        self.cumulative_drift = 0.0;
        self.mutations_since_retrain = 0;
    }

    /// Get current status.
    pub fn status(&self) -> RetrainStatus {
        RetrainStatus {
            cumulative_drift: self.cumulative_drift,
            threshold: self.theta,
            retrain_recommended: self.should_retrain(),
            mutations_since_retrain: self.mutations_since_retrain,
            total_retrains: self.retrain_history.len(),
        }
    }

    /// Adaptive threshold: after enough retrains, adjust θ based on
    /// average drift-at-retrain (paper §4.5).
    pub fn adapt_threshold(&mut self) {
        if self.retrain_history.len() >= 3 {
            let avg_drift: f64 = self.retrain_history.iter().map(|(_, d)| d).sum::<f64>()
                / self.retrain_history.len() as f64;
            self.theta = avg_drift;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retrain_monitor() {
        let mut mon = RetrainMonitor::new(1.0);
        assert!(!mon.should_retrain());

        mon.record_drift(0.3);
        mon.record_drift(0.4);
        assert!(!mon.should_retrain());
        assert_eq!(mon.mutations_since_retrain, 2);

        mon.record_drift(0.5);
        assert!(mon.should_retrain()); // 1.2 > 1.0

        mon.record_retrain("2026-03-03T21:00:00Z");
        assert!(!mon.should_retrain());
        assert_eq!(mon.mutations_since_retrain, 0);

        let status = mon.status();
        assert_eq!(status.total_retrains, 1);
    }

    #[test]
    fn test_adaptive_threshold() {
        let mut mon = RetrainMonitor::new(1.0);

        // Simulate 3 retrains at different drift levels
        mon.cumulative_drift = 0.8;
        mon.record_retrain("t1");
        mon.cumulative_drift = 1.2;
        mon.record_retrain("t2");
        mon.cumulative_drift = 1.0;
        mon.record_retrain("t3");

        mon.adapt_threshold();
        assert!((mon.theta - 1.0).abs() < 0.01); // avg of 0.8, 1.2, 1.0 = 1.0
    }
}
