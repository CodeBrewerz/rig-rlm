//! Queue-Reactive Market Simulator
//!
//! From the queue-reactive LOB paper (Huang et al. 2014):
//! implements a Markov jump process simulator calibrated from historical
//! graph data. Used to generate synthetic scenarios for stress testing
//! and as a FeedbackEvaluator for RLVR training.
//!
//! In the hehrgnn financial graph context:
//! - λ^L(n) → arrival rate (new edges being created)
//! - λ^C(n) → cancellation rate (edges being removed/reversed)
//! - λ^M(n) → execution rate (edges leading to terminal state)
//!
//! The simulator generates synthetic GraphFact sequences that follow
//! the calibrated Markov transition rates.

use std::collections::HashMap;

/// Intensity function parameters for a single entity type.
///
/// Calibrated from historical edge data using MLE.
#[derive(Debug, Clone)]
pub struct IntensityParams {
    /// Entity type these parameters apply to.
    pub entity_type: String,
    /// Arrival rate per queue size bucket: λ^L(n) for n=0,1,2,...
    /// Index = queue size bucket, value = rate.
    pub arrival_rates: Vec<f32>,
    /// Departure rate per queue size bucket: λ^C(n) + λ^M(n)
    pub departure_rates: Vec<f32>,
}

impl IntensityParams {
    /// Compute ρ(n) = λ^L(n) / λ^D(n) for bucket n.
    pub fn rho(&self, bucket: usize) -> f32 {
        let arrival = self.arrival_rates.get(bucket).copied().unwrap_or(0.0);
        let departure = self.departure_rates.get(bucket).copied().unwrap_or(1.0);
        arrival / departure.max(1e-8)
    }

    /// Check if the queue is stable at bucket n (ρ(n) < 1).
    pub fn is_stable_at(&self, bucket: usize) -> bool {
        self.rho(bucket) < 1.0
    }
}

/// Calibrated simulator state.
#[derive(Debug, Clone)]
pub struct QueueReactiveSimulator {
    /// Intensity parameters per entity type.
    pub params: HashMap<String, IntensityParams>,
    /// Maximum queue size for simulation.
    pub max_queue_size: usize,
    /// Random seed.
    seed: u64,
}

impl QueueReactiveSimulator {
    /// Create a new simulator with given parameters.
    pub fn new(params: HashMap<String, IntensityParams>, max_queue_size: usize) -> Self {
        Self {
            params,
            max_queue_size,
            seed: 42,
        }
    }

    /// Calibrate intensity parameters from historical graph edge data.
    ///
    /// Uses simple MLE: count arrivals and departures per queue-size bucket.
    ///
    /// Args:
    /// - `edge_events`: sequence of (entity_type, is_arrival, queue_size_at_event)
    /// - `num_buckets`: number of queue-size buckets
    pub fn calibrate(
        edge_events: &[(String, bool, usize)],
        num_buckets: usize,
    ) -> HashMap<String, IntensityParams> {
        let mut arrival_counts: HashMap<String, Vec<f32>> = HashMap::new();
        let mut departure_counts: HashMap<String, Vec<f32>> = HashMap::new();
        let mut bucket_times: HashMap<String, Vec<f32>> = HashMap::new();

        for (entity_type, is_arrival, queue_size) in edge_events {
            let bucket = (*queue_size).min(num_buckets - 1);

            let arrivals = arrival_counts
                .entry(entity_type.clone())
                .or_insert_with(|| vec![0.0; num_buckets]);
            let departures = departure_counts
                .entry(entity_type.clone())
                .or_insert_with(|| vec![0.0; num_buckets]);
            let times = bucket_times
                .entry(entity_type.clone())
                .or_insert_with(|| vec![0.0; num_buckets]);

            if *is_arrival {
                arrivals[bucket] += 1.0;
            } else {
                departures[bucket] += 1.0;
            }
            times[bucket] += 1.0; // Unit time per event (simplified)
        }

        let mut params = HashMap::new();
        for (entity_type, arrivals) in &arrival_counts {
            let departures = departure_counts
                .get(entity_type)
                .cloned()
                .unwrap_or_else(|| vec![0.0; num_buckets]);
            let times = bucket_times
                .get(entity_type)
                .cloned()
                .unwrap_or_else(|| vec![1.0; num_buckets]);

            let arrival_rates: Vec<f32> = arrivals
                .iter()
                .zip(times.iter())
                .map(|(&a, &t)| if t > 0.0 { a / t } else { 0.0 })
                .collect();

            let departure_rates: Vec<f32> = departures
                .iter()
                .zip(times.iter())
                .map(|(&d, &t)| if t > 0.0 { d / t } else { 0.0 })
                .collect();

            params.insert(
                entity_type.clone(),
                IntensityParams {
                    entity_type: entity_type.clone(),
                    arrival_rates,
                    departure_rates,
                },
            );
        }

        params
    }

    /// Run a Monte Carlo simulation for a single entity type.
    ///
    /// Returns a trajectory: Vec<(time, queue_size)>.
    pub fn simulate_trajectory(
        &mut self,
        entity_type: &str,
        initial_queue_size: usize,
        num_steps: usize,
    ) -> Vec<(f64, usize)> {
        let params = match self.params.get(entity_type) {
            Some(p) => p.clone(),
            None => return vec![(0.0, initial_queue_size)],
        };

        let mut trajectory = Vec::with_capacity(num_steps + 1);
        let mut t = 0.0f64;
        let mut q = initial_queue_size;
        trajectory.push((t, q));

        for _ in 0..num_steps {
            let bucket = q.min(params.arrival_rates.len().saturating_sub(1));
            let lambda_a = params.arrival_rates.get(bucket).copied().unwrap_or(0.1);
            let lambda_d = params.departure_rates.get(bucket).copied().unwrap_or(0.1);
            let total_rate = (lambda_a + lambda_d).max(1e-6);

            // Exponential inter-event time
            let u = self.next_uniform();
            let dt = -(1.0 - u).max(1e-10).ln() as f64 / total_rate as f64;
            t += dt;

            // Decide event type
            let p_arrival = lambda_a / total_rate;
            if self.next_uniform() < p_arrival {
                // Arrival: queue grows
                if q < self.max_queue_size {
                    q += 1;
                }
            } else {
                // Departure: queue shrinks
                if q > 0 {
                    q -= 1;
                }
            }

            trajectory.push((t, q));
        }

        trajectory
    }

    /// Compute slippage distribution from Monte Carlo simulation.
    ///
    /// Slippage = number of steps until queue position is reached.
    /// Returns (mean_slippage, std_slippage, fill_probability).
    pub fn estimate_slippage(
        &mut self,
        entity_type: &str,
        target_position: usize,
        num_simulations: usize,
        max_steps: usize,
    ) -> (f32, f32, f32) {
        let mut fill_times = Vec::new();
        let mut fills = 0;

        for _ in 0..num_simulations {
            let trajectory = self.simulate_trajectory(entity_type, target_position, max_steps);
            // Check if queue ever depletes to 0 (fill)
            if let Some(pos) = trajectory.iter().position(|&(_, q)| q == 0) {
                fill_times.push(pos as f32);
                fills += 1;
            }
        }

        let fill_prob = fills as f32 / num_simulations.max(1) as f32;

        if fill_times.is_empty() {
            return (f32::INFINITY, 0.0, fill_prob);
        }

        let mean = fill_times.iter().sum::<f32>() / fill_times.len() as f32;
        let variance =
            fill_times.iter().map(|&t| (t - mean).powi(2)).sum::<f32>() / fill_times.len() as f32;

        (mean, variance.sqrt(), fill_prob)
    }

    /// Generate synthetic edge events following the calibrated Markov process.
    ///
    /// These can be used as augmented training data or for stress testing.
    pub fn generate_synthetic_events(
        &mut self,
        entity_type: &str,
        entity_name: &str,
        num_events: usize,
    ) -> Vec<(String, String, String, String, String)> {
        // (src_type, src_name, relation, dst_type, dst_name)
        let mut events = Vec::new();
        let trajectory = self.simulate_trajectory(entity_type, 0, num_events);

        let mut prev_q = 0usize;
        for (i, &(_t, q)) in trajectory.iter().enumerate().skip(1) {
            if q > prev_q {
                // Arrival → new edge
                events.push((
                    entity_type.to_string(),
                    entity_name.to_string(),
                    "sim_arrival".to_string(),
                    "sim_event".to_string(),
                    format!("event_{}", i),
                ));
            } else if q < prev_q {
                // Departure → edge removal (logged as event)
                events.push((
                    entity_type.to_string(),
                    entity_name.to_string(),
                    "sim_departure".to_string(),
                    "sim_event".to_string(),
                    format!("event_{}", i),
                ));
            }
            prev_q = q;
        }

        events
    }

    fn next_uniform(&mut self) -> f32 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.seed >> 33) as f32 / (1u64 << 31) as f32
    }
}

/// Summary statistics from a simulation run.
#[derive(Debug, Clone)]
pub struct SimulationSummary {
    pub entity_type: String,
    pub mean_queue_size: f32,
    pub max_queue_size: usize,
    pub fill_probability: f32,
    pub mean_fill_time: f32,
    pub num_simulations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intensity_params() {
        let params = IntensityParams {
            entity_type: "account".to_string(),
            arrival_rates: vec![0.5, 0.4, 0.3, 0.2],
            departure_rates: vec![0.1, 0.2, 0.3, 0.4],
        };

        // ρ(0) = 0.5/0.1 = 5.0 (unstable: queue grows)
        assert!((params.rho(0) - 5.0).abs() < 0.01);
        assert!(!params.is_stable_at(0));

        // ρ(3) = 0.2/0.4 = 0.5 (stable: queue shrinks)
        assert!((params.rho(3) - 0.5).abs() < 0.01);
        assert!(params.is_stable_at(3));
    }

    #[test]
    fn test_calibrate() {
        let events: Vec<(String, bool, usize)> = vec![
            ("account".into(), true, 0),
            ("account".into(), true, 1),
            ("account".into(), false, 2),
            ("account".into(), true, 1),
            ("account".into(), false, 2),
            ("account".into(), false, 1),
        ];

        let params = QueueReactiveSimulator::calibrate(&events, 4);
        assert!(params.contains_key("account"));
        let p = &params["account"];
        assert_eq!(p.arrival_rates.len(), 4);
        assert_eq!(p.departure_rates.len(), 4);
    }

    #[test]
    fn test_simulate_trajectory() {
        let mut params = HashMap::new();
        params.insert(
            "account".to_string(),
            IntensityParams {
                entity_type: "account".to_string(),
                arrival_rates: vec![0.5, 0.4, 0.3],
                departure_rates: vec![0.1, 0.3, 0.5],
            },
        );

        let mut sim = QueueReactiveSimulator::new(params, 10);
        let trajectory = sim.simulate_trajectory("account", 3, 100);
        assert!(!trajectory.is_empty());
        assert_eq!(trajectory[0], (0.0, 3));

        // Queue sizes should be bounded
        for &(_t, q) in &trajectory {
            assert!(q <= 10);
        }
    }

    #[test]
    fn test_slippage_estimation() {
        let mut params = HashMap::new();
        params.insert(
            "account".to_string(),
            IntensityParams {
                entity_type: "account".to_string(),
                // High departure rates → should fill quickly
                arrival_rates: vec![0.1, 0.1, 0.1],
                departure_rates: vec![0.5, 0.5, 0.5],
            },
        );

        let mut sim = QueueReactiveSimulator::new(params, 10);
        let (mean, _std, fill_prob) = sim.estimate_slippage("account", 2, 50, 100);

        // With high departure rates starting from position 2,
        // fills should happen relatively quickly
        assert!(mean.is_finite(), "Mean should be finite");
        assert!(fill_prob > 0.0, "Should have some fills");
    }

    #[test]
    fn test_synthetic_event_generation() {
        let mut params = HashMap::new();
        params.insert(
            "user".to_string(),
            IntensityParams {
                entity_type: "user".to_string(),
                arrival_rates: vec![0.5, 0.4],
                departure_rates: vec![0.3, 0.4],
            },
        );

        let mut sim = QueueReactiveSimulator::new(params, 10);
        let events = sim.generate_synthetic_events("user", "test_user", 20);
        assert!(!events.is_empty(), "Should generate some events");
        for (src_type, _, rel, _, _) in &events {
            assert_eq!(src_type, "user");
            assert!(
                rel == "sim_arrival" || rel == "sim_departure",
                "Unknown relation: {}",
                rel
            );
        }
    }
}
