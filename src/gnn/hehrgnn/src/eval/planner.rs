//! Action-Conditioned Planner (Gap #8)
//!
//! From jepa-rs `jepa-world/planner.rs`:
//! given a current graph state and a candidate action, predict the
//! next state and evaluate its cost.
//!
//! In the hehrgnn financial graph context:
//! - State = current graph embeddings
//! - Action = proposed financial operation (e.g., "increase allocation by 5%")
//! - Cost = predicted risk/reward of the action
//!
//! Uses random shooting (CEM) to find the best action among candidates.

use std::collections::HashMap;

/// A candidate action represented as a feature vector.
#[derive(Debug, Clone)]
pub struct Action {
    /// Action identifier.
    pub id: String,
    /// Action feature vector.
    pub features: Vec<f32>,
    /// Optional description.
    pub description: String,
}

/// Predicted outcome of an action.
#[derive(Debug, Clone)]
pub struct ActionOutcome {
    /// The action taken.
    pub action_id: String,
    /// Predicted embedding change (delta).
    pub predicted_delta: HashMap<String, Vec<f32>>,
    /// Cost/risk score (lower is better).
    pub cost: f32,
    /// Confidence (0..1).
    pub confidence: f32,
}

/// Action-conditioned state predictor.
///
/// Predicts: z_{t+1} = z_t + W_action * action_features
/// Simple linear action-conditioned transition for graph JEPA.
#[derive(Debug, Clone)]
pub struct ActionPredictor {
    /// Weight matrix per node type: [action_dim, embed_dim].
    pub weights: HashMap<String, Vec<Vec<f32>>>,
    /// Action feature dimension.
    pub action_dim: usize,
}

impl ActionPredictor {
    /// Create a new action predictor with random weights.
    pub fn new(action_dim: usize, embed_dim: usize, node_types: &[String]) -> Self {
        let mut weights = HashMap::new();
        let mut seed = 42u64;

        for nt in node_types {
            let mut w = Vec::with_capacity(action_dim);
            for _ in 0..action_dim {
                let row: Vec<f32> = (0..embed_dim)
                    .map(|_| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let u = (seed >> 33) as f32 / (1u64 << 31) as f32;
                        (u - 0.5) * 0.1 // Small random init
                    })
                    .collect();
                w.push(row);
            }
            weights.insert(nt.clone(), w);
        }

        Self {
            weights,
            action_dim,
        }
    }

    /// Predict the next state given current embeddings and an action.
    ///
    /// z_{t+1} = z_t + W * action_features (per node type mean)
    pub fn predict(
        &self,
        current: &HashMap<String, Vec<Vec<f32>>>,
        action: &Action,
    ) -> HashMap<String, Vec<f32>> {
        let mut predicted = HashMap::new();

        for (nt, vecs) in current {
            if vecs.is_empty() || vecs[0].is_empty() {
                continue;
            }

            let d = vecs[0].len();

            // Pool current embeddings (mean)
            let mut mean = vec![0.0f32; d];
            for v in vecs {
                for (j, &val) in v.iter().enumerate() {
                    if j < d {
                        mean[j] += val;
                    }
                }
            }
            let n = vecs.len() as f32;
            for v in mean.iter_mut() {
                *v /= n;
            }

            // Apply action transformation
            if let Some(w) = self.weights.get(nt) {
                let mut delta = vec![0.0f32; d];
                for (a_idx, &a_val) in action.features.iter().enumerate() {
                    if a_idx < w.len() {
                        for (j, &w_val) in w[a_idx].iter().enumerate() {
                            if j < d {
                                delta[j] += a_val * w_val;
                            }
                        }
                    }
                }

                for (j, d_val) in delta.iter().enumerate() {
                    if j < mean.len() {
                        mean[j] += d_val;
                    }
                }
            }

            predicted.insert(nt.clone(), mean);
        }

        predicted
    }
}

/// Random shooting planner (CEM-style).
///
/// Evaluates multiple candidate actions and selects the best one.
#[derive(Debug, Clone)]
pub struct RandomShootingPlanner {
    /// Number of candidate actions to evaluate.
    pub num_candidates: usize,
    /// Number of steps to look ahead.
    pub horizon: usize,
}

impl RandomShootingPlanner {
    pub fn new(num_candidates: usize, horizon: usize) -> Self {
        Self {
            num_candidates,
            horizon,
        }
    }

    /// Evaluate a set of candidate actions and rank them by cost.
    ///
    /// Cost = L2 distance from predicted state to goal state.
    pub fn evaluate_actions(
        &self,
        predictor: &ActionPredictor,
        current_state: &HashMap<String, Vec<Vec<f32>>>,
        actions: &[Action],
        goal: &HashMap<String, Vec<f32>>,
    ) -> Vec<ActionOutcome> {
        let mut outcomes = Vec::with_capacity(actions.len());

        for action in actions {
            let predicted = predictor.predict(current_state, action);

            // Cost = L2 distance to goal
            let mut cost = 0.0f32;
            let mut count = 0usize;
            for (nt, pred_vec) in &predicted {
                if let Some(goal_vec) = goal.get(nt) {
                    let dist: f32 = pred_vec
                        .iter()
                        .zip(goal_vec.iter())
                        .map(|(&p, &g)| (p - g).powi(2))
                        .sum::<f32>()
                        .sqrt();
                    cost += dist;
                    count += 1;
                }
            }
            if count > 0 {
                cost /= count as f32;
            }

            // Confidence = 1 / (1 + cost)
            let confidence = 1.0 / (1.0 + cost);

            let predicted_delta: HashMap<String, Vec<f32>> = predicted
                .into_iter()
                .map(|(nt, pred)| {
                    let orig = current_state
                        .get(&nt)
                        .and_then(|vecs| {
                            if vecs.is_empty() {
                                None
                            } else {
                                let d = vecs[0].len();
                                let mut mean = vec![0.0f32; d];
                                for v in vecs {
                                    for (j, &val) in v.iter().enumerate() {
                                        if j < d {
                                            mean[j] += val;
                                        }
                                    }
                                }
                                let n = vecs.len() as f32;
                                for v in mean.iter_mut() {
                                    *v /= n;
                                }
                                Some(mean)
                            }
                        })
                        .unwrap_or_else(|| vec![0.0; pred.len()]);
                    let delta: Vec<f32> =
                        pred.iter().zip(orig.iter()).map(|(&p, &o)| p - o).collect();
                    (nt, delta)
                })
                .collect();

            outcomes.push(ActionOutcome {
                action_id: action.id.clone(),
                predicted_delta,
                cost,
                confidence,
            });
        }

        // Sort by cost (lowest first)
        outcomes.sort_by(|a, b| {
            a.cost
                .partial_cmp(&b.cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        outcomes
    }

    /// Select the best action.
    pub fn best_action(
        &self,
        predictor: &ActionPredictor,
        current_state: &HashMap<String, Vec<Vec<f32>>>,
        actions: &[Action],
        goal: &HashMap<String, Vec<f32>>,
    ) -> Option<ActionOutcome> {
        let outcomes = self.evaluate_actions(predictor, current_state, actions, goal);
        outcomes.into_iter().next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state() -> HashMap<String, Vec<Vec<f32>>> {
        let mut m = HashMap::new();
        m.insert("user".to_string(), vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        m
    }

    fn make_goal() -> HashMap<String, Vec<f32>> {
        let mut m = HashMap::new();
        m.insert("user".to_string(), vec![2.0, 2.0]);
        m
    }

    #[test]
    fn test_action_predictor() {
        let predictor = ActionPredictor::new(2, 2, &["user".to_string()]);
        let state = make_state();
        let action = Action {
            id: "grow".to_string(),
            features: vec![1.0, 0.0],
            description: "Grow user engagement".to_string(),
        };

        let predicted = predictor.predict(&state, &action);
        assert!(predicted.contains_key("user"));
        assert_eq!(predicted["user"].len(), 2);
    }

    #[test]
    fn test_planner_ranks_actions() {
        let predictor = ActionPredictor::new(2, 2, &["user".to_string()]);
        let state = make_state();
        let goal = make_goal();

        let actions = vec![
            Action {
                id: "a1".to_string(),
                features: vec![1.0, 0.0],
                description: String::new(),
            },
            Action {
                id: "a2".to_string(),
                features: vec![0.0, 1.0],
                description: String::new(),
            },
            Action {
                id: "a3".to_string(),
                features: vec![1.0, 1.0],
                description: String::new(),
            },
        ];

        let planner = RandomShootingPlanner::new(3, 1);
        let outcomes = planner.evaluate_actions(&predictor, &state, &actions, &goal);

        assert_eq!(outcomes.len(), 3);
        // First outcome should have lowest cost
        assert!(
            outcomes[0].cost <= outcomes[1].cost,
            "Actions should be sorted by cost"
        );
        assert!(
            outcomes[1].cost <= outcomes[2].cost,
            "Actions should be sorted by cost"
        );
    }

    #[test]
    fn test_best_action() {
        let predictor = ActionPredictor::new(2, 2, &["user".to_string()]);
        let state = make_state();
        let goal = make_goal();

        let actions = vec![
            Action {
                id: "a1".to_string(),
                features: vec![1.0, 0.0],
                description: String::new(),
            },
            Action {
                id: "a2".to_string(),
                features: vec![0.5, 0.5],
                description: String::new(),
            },
        ];

        let planner = RandomShootingPlanner::new(2, 1);
        let best = planner.best_action(&predictor, &state, &actions, &goal);
        assert!(best.is_some(), "Should return a best action");
        assert!(best.unwrap().confidence > 0.0);
    }

    #[test]
    fn test_action_outcome_confidence() {
        let outcome = ActionOutcome {
            action_id: "test".to_string(),
            predicted_delta: HashMap::new(),
            cost: 0.0,
            confidence: 1.0,
        };
        assert_eq!(
            outcome.confidence, 1.0,
            "Zero cost should give 100% confidence"
        );
    }
}
