//! InstantGNN: incremental PPR-based propagation for dynamic heterogeneous graphs.
//!
//! Based on: Zheng et al., "Instant Graph Neural Networks for Dynamic Graphs", KDD 2022.
//!
//! Core idea: Maintain estimated propagation vector π̂ and residual vector r per node.
//! When edges are inserted/deleted, only update residuals for affected nodes and their
//! neighbors, then run BasicPropagation to reduce residuals below threshold ε.
//! This gives O(1) expected update time per edge mutation.

use crate::data::graph_mutation::GraphEvent;
use std::collections::HashMap;

/// Key for a typed node: (node_type, node_id).
type NodeKey = (String, usize);

/// Per-relation degree info for a node.
#[derive(Debug, Clone, Default)]
struct DegreeInfo {
    /// Total in-degree across all relation types.
    in_degree: usize,
    /// Total out-degree across all relation types.
    out_degree: usize,
    /// Per-relation in-degree: relation → count.
    relation_in: HashMap<String, usize>,
    /// Per-relation out-degree: relation → count.
    relation_out: HashMap<String, usize>,
}

/// Adjacency entry: (neighbor_type, neighbor_id, relation).
type AdjEntry = (String, usize, String);

/// PPR-based propagation state for incremental updates.
#[derive(Debug, Clone)]
pub struct PropagationState {
    /// Estimated propagation vectors π̂: node_type → [node_id][feat_dim].
    pi_hat: HashMap<String, Vec<Vec<f32>>>,
    /// Residual vectors r: node_type → [node_id][feat_dim].
    residuals: HashMap<String, Vec<Vec<f32>>>,
    /// Node degrees: node_type → [node_id] → DegreeInfo.
    degrees: HashMap<String, Vec<DegreeInfo>>,
    /// Adjacency lists: node_type → [node_id] → Vec<(neighbor_type, neighbor_id, relation)>.
    adjacency: HashMap<String, Vec<Vec<AdjEntry>>>,
    /// Feature dimension.
    feat_dim: usize,
    /// Error threshold ε — controls accuracy/speed tradeoff.
    epsilon: f32,
    /// Teleport probability α (PPR damping, typically 0.15-0.25).
    alpha: f32,
    /// Convolution coefficient β (degree normalization, typically 0.5 for symmetric).
    beta: f32,
    /// Cumulative representation drift ‖ΔZ‖_F since last retrain.
    cumulative_drift: f64,
    /// Number of mutations applied.
    mutation_count: usize,
}

/// Result of applying a batch of mutations.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MutationResult {
    /// Number of events processed.
    pub events_processed: usize,
    /// Number of nodes whose embeddings were updated.
    pub nodes_updated: usize,
    /// Current cumulative drift.
    pub cumulative_drift: f64,
    /// Whether retrain is recommended based on drift.
    pub drift_delta: f64,
    /// Node types that had embedding changes.
    pub affected_types: Vec<String>,
}

impl PropagationState {
    /// Create a new propagation state from pre-computed embeddings.
    ///
    /// Initializes π̂ from the existing embeddings and sets residuals to zero
    /// (since embeddings are already the result of full GNN training).
    pub fn init_from_embeddings(
        embeddings: &HashMap<String, Vec<Vec<f32>>>,
        edges: &HashMap<(String, String, String), Vec<(usize, usize)>>,
        feat_dim: usize,
    ) -> Self {
        let alpha = 0.15;
        let epsilon = 0.01;
        let beta = 0.5;

        // Initialize π̂ from existing embeddings
        let mut pi_hat: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
        let mut residuals: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
        let mut degrees: HashMap<String, Vec<DegreeInfo>> = HashMap::new();
        let mut adjacency: HashMap<String, Vec<Vec<AdjEntry>>> = HashMap::new();

        for (node_type, node_embs) in embeddings {
            let n = node_embs.len();
            pi_hat.insert(node_type.clone(), node_embs.clone());
            residuals.insert(node_type.clone(), vec![vec![0.0; feat_dim]; n]);
            degrees.insert(node_type.clone(), vec![DegreeInfo::default(); n]);
            adjacency.insert(node_type.clone(), vec![Vec::new(); n]);
        }

        // Build degree info and adjacency from edges
        for ((src_type, relation, dst_type), edge_list) in edges {
            for &(src_id, dst_id) in edge_list {
                // Out-degree for source
                if let Some(deg_vec) = degrees.get_mut(src_type) {
                    if src_id < deg_vec.len() {
                        deg_vec[src_id].out_degree += 1;
                        *deg_vec[src_id]
                            .relation_out
                            .entry(relation.clone())
                            .or_insert(0) += 1;
                    }
                }
                // In-degree for destination
                if let Some(deg_vec) = degrees.get_mut(dst_type) {
                    if dst_id < deg_vec.len() {
                        deg_vec[dst_id].in_degree += 1;
                        *deg_vec[dst_id]
                            .relation_in
                            .entry(relation.clone())
                            .or_insert(0) += 1;
                    }
                }
                // Adjacency: src → dst
                if let Some(adj_vec) = adjacency.get_mut(src_type) {
                    if src_id < adj_vec.len() {
                        adj_vec[src_id].push((dst_type.clone(), dst_id, relation.clone()));
                    }
                }
                // Adjacency: dst → src (undirected)
                if let Some(adj_vec) = adjacency.get_mut(dst_type) {
                    if dst_id < adj_vec.len() {
                        adj_vec[dst_id].push((src_type.clone(), src_id, relation.clone()));
                    }
                }
            }
        }

        Self {
            pi_hat,
            residuals,
            degrees,
            adjacency,
            feat_dim,
            epsilon,
            alpha,
            beta,
            cumulative_drift: 0.0,
            mutation_count: 0,
        }
    }

    /// Apply a batch of graph events and incrementally update embeddings.
    pub fn apply_mutations(&mut self, events: &[GraphEvent]) -> MutationResult {
        let mut affected_types: Vec<String> = Vec::new();
        let mut total_drift = 0.0f64;

        for event in events {
            match event {
                GraphEvent::InsertEdge {
                    src_type,
                    src_id,
                    dst_type,
                    dst_id,
                    relation,
                } => {
                    let drift =
                        self.update_edge(src_type, *src_id, dst_type, *dst_id, relation, true);
                    total_drift += drift;
                    for t in event.affected_types() {
                        if !affected_types.contains(&t.to_string()) {
                            affected_types.push(t.to_string());
                        }
                    }
                }
                GraphEvent::DeleteEdge {
                    src_type,
                    src_id,
                    dst_type,
                    dst_id,
                    relation,
                } => {
                    let drift =
                        self.update_edge(src_type, *src_id, dst_type, *dst_id, relation, false);
                    total_drift += drift;
                    for t in event.affected_types() {
                        if !affected_types.contains(&t.to_string()) {
                            affected_types.push(t.to_string());
                        }
                    }
                }
                GraphEvent::UpdateFeatures {
                    node_type,
                    node_id,
                    new_features,
                } => {
                    let drift = self.update_features(node_type, *node_id, new_features);
                    total_drift += drift;
                    if !affected_types.contains(node_type) {
                        affected_types.push(node_type.clone());
                    }
                }
                GraphEvent::InsertNode {
                    node_type,
                    features,
                } => {
                    self.insert_node(node_type, features);
                    if !affected_types.contains(node_type) {
                        affected_types.push(node_type.clone());
                    }
                }
            }
            self.mutation_count += 1;
        }

        // Run basic propagation to reduce residuals
        let nodes_updated = self.basic_propagation();

        self.cumulative_drift += total_drift;

        MutationResult {
            events_processed: events.len(),
            nodes_updated,
            cumulative_drift: self.cumulative_drift,
            drift_delta: total_drift,
            affected_types,
        }
    }

    /// InstantGNN Algorithm 2 UPDATE for edge insert/delete.
    ///
    /// Updates residuals for nodes u, v, and their neighbors based on
    /// degree change and new/removed neighbor contribution.
    fn update_edge(
        &mut self,
        src_type: &str,
        src_id: usize,
        dst_type: &str,
        dst_id: usize,
        relation: &str,
        is_insert: bool,
    ) -> f64 {
        let alpha = self.alpha;
        let beta = self.beta;
        let mut total_drift = 0.0f64;

        // Update degrees
        if is_insert {
            if let Some(deg_vec) = self.degrees.get_mut(src_type) {
                if src_id < deg_vec.len() {
                    deg_vec[src_id].out_degree += 1;
                    *deg_vec[src_id]
                        .relation_out
                        .entry(relation.to_string())
                        .or_insert(0) += 1;
                }
            }
            if let Some(deg_vec) = self.degrees.get_mut(dst_type) {
                if dst_id < deg_vec.len() {
                    deg_vec[dst_id].in_degree += 1;
                    *deg_vec[dst_id]
                        .relation_in
                        .entry(relation.to_string())
                        .or_insert(0) += 1;
                }
            }
            // Update adjacency
            if let Some(adj) = self.adjacency.get_mut(src_type) {
                if src_id < adj.len() {
                    adj[src_id].push((dst_type.to_string(), dst_id, relation.to_string()));
                }
            }
            if let Some(adj) = self.adjacency.get_mut(dst_type) {
                if dst_id < adj.len() {
                    adj[dst_id].push((src_type.to_string(), src_id, relation.to_string()));
                }
            }
        } else {
            // Delete: decrement degrees
            if let Some(deg_vec) = self.degrees.get_mut(src_type) {
                if src_id < deg_vec.len() && deg_vec[src_id].out_degree > 0 {
                    deg_vec[src_id].out_degree -= 1;
                    if let Some(count) = deg_vec[src_id].relation_out.get_mut(relation) {
                        *count = count.saturating_sub(1);
                    }
                }
            }
            if let Some(deg_vec) = self.degrees.get_mut(dst_type) {
                if dst_id < deg_vec.len() && deg_vec[dst_id].in_degree > 0 {
                    deg_vec[dst_id].in_degree -= 1;
                    if let Some(count) = deg_vec[dst_id].relation_in.get_mut(relation) {
                        *count = count.saturating_sub(1);
                    }
                }
            }
            // Remove from adjacency
            if let Some(adj) = self.adjacency.get_mut(src_type) {
                if src_id < adj.len() {
                    adj[src_id]
                        .retain(|(t, id, r)| !(t == dst_type && *id == dst_id && r == relation));
                }
            }
            if let Some(adj) = self.adjacency.get_mut(dst_type) {
                if dst_id < adj.len() {
                    adj[dst_id]
                        .retain(|(t, id, r)| !(t == src_type && *id == src_id && r == relation));
                }
            }
        }

        // Compute residual increments for u (src) and v (dst)
        // From paper Eq. 5: Δr(u) based on degree change + new/removed neighbor
        total_drift +=
            self.update_node_residual(src_type, src_id, dst_type, dst_id, is_insert, alpha, beta);
        total_drift +=
            self.update_node_residual(dst_type, dst_id, src_type, src_id, is_insert, alpha, beta);

        // Update residuals for neighbors of u and v (degree normalization changed)
        let src_neighbors: Vec<AdjEntry> = self
            .adjacency
            .get(src_type)
            .and_then(|a| a.get(src_id))
            .cloned()
            .unwrap_or_default();
        for (nt, nid, _) in &src_neighbors {
            if nt == dst_type && *nid == dst_id {
                continue; // already handled
            }
            total_drift += self.update_neighbor_residual(nt, *nid, src_type, src_id, alpha, beta);
        }

        let dst_neighbors: Vec<AdjEntry> = self
            .adjacency
            .get(dst_type)
            .and_then(|a| a.get(dst_id))
            .cloned()
            .unwrap_or_default();
        for (nt, nid, _) in &dst_neighbors {
            if nt == src_type && *nid == src_id {
                continue;
            }
            total_drift += self.update_neighbor_residual(nt, *nid, dst_type, dst_id, alpha, beta);
        }

        total_drift
    }

    /// Update residual for node u when edge (u, v) is inserted/deleted.
    fn update_node_residual(
        &mut self,
        u_type: &str,
        u_id: usize,
        v_type: &str,
        v_id: usize,
        is_insert: bool,
        alpha: f32,
        beta: f32,
    ) -> f64 {
        let d_u = self.total_degree(u_type, u_id) as f32;
        let d_v = self.total_degree(v_type, v_id) as f32;
        if d_u < 1.0 || d_v < 1.0 {
            return 0.0;
        }

        let pi_hat_u = self
            .pi_hat
            .get(u_type)
            .and_then(|v| v.get(u_id))
            .cloned()
            .unwrap_or_else(|| vec![0.0; self.feat_dim]);

        let pi_hat_v = self
            .pi_hat
            .get(v_type)
            .and_then(|v| v.get(v_id))
            .cloned()
            .unwrap_or_else(|| vec![0.0; self.feat_dim]);

        let r_u = self
            .residuals
            .get(u_type)
            .and_then(|v| v.get(u_id))
            .cloned()
            .unwrap_or_else(|| vec![0.0; self.feat_dim]);

        // Degree change factor: (d_u_old^β - d_u_new^β) / d_u_new^β
        let d_old = if is_insert { d_u - 1.0 } else { d_u + 1.0 };
        let degree_factor = (d_old.powf(beta) - d_u.powf(beta)) / (d_u.powf(beta) * alpha);

        // Neighbor contribution: γ(u', v') / α
        let neighbor_factor = (1.0 - alpha) / (alpha * d_u.powf(beta) * d_v.powf(1.0 - beta));

        let mut drift = 0.0f64;
        if let Some(res) = self.residuals.get_mut(u_type) {
            if let Some(r) = res.get_mut(u_id) {
                for d in 0..self.feat_dim {
                    let delta = (pi_hat_u[d] + alpha * r_u[d]) * degree_factor
                        + if is_insert { 1.0 } else { -1.0 } * pi_hat_v[d] * neighbor_factor;
                    r[d] += delta;
                    drift += (delta as f64).powi(2);
                }
            }
        }

        drift.sqrt()
    }

    /// Update residual for neighbor w of node u (degree normalization changed).
    fn update_neighbor_residual(
        &mut self,
        w_type: &str,
        w_id: usize,
        u_type: &str,
        u_id: usize,
        alpha: f32,
        beta: f32,
    ) -> f64 {
        let d_u = self.total_degree(u_type, u_id) as f32;
        let d_w = self.total_degree(w_type, w_id) as f32;
        if d_u < 1.0 || d_w < 1.0 {
            return 0.0;
        }

        let pi_hat_u = self
            .pi_hat
            .get(u_type)
            .and_then(|v| v.get(u_id))
            .cloned()
            .unwrap_or_else(|| vec![0.0; self.feat_dim]);

        // Normalization change: 1/d_u_new^(1-β) - 1/d_u_old^(1-β)
        let d_u_old = d_u; // degree already updated
        let norm_diff = 1.0 / d_u.powf(1.0 - beta) - 1.0 / (d_u_old).powf(1.0 - beta);

        let factor = (1.0 - alpha) / (alpha * d_w.powf(beta));

        let mut drift = 0.0f64;
        if let Some(res) = self.residuals.get_mut(w_type) {
            if let Some(r) = res.get_mut(w_id) {
                for d in 0..self.feat_dim {
                    let delta = pi_hat_u[d] * factor * norm_diff;
                    r[d] += delta;
                    drift += (delta as f64).powi(2);
                }
            }
        }

        drift.sqrt()
    }

    /// Handle feature updates: add Δx to residuals (paper §4.3).
    fn update_features(&mut self, node_type: &str, node_id: usize, new_features: &[f32]) -> f64 {
        let mut drift = 0.0f64;
        if let Some(res) = self.residuals.get_mut(node_type) {
            if let Some(r) = res.get_mut(node_id) {
                // Get old features from π̂ (approximation)
                let old_features = self
                    .pi_hat
                    .get(node_type)
                    .and_then(|v| v.get(node_id))
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; self.feat_dim]);

                for d in 0..self
                    .feat_dim
                    .min(new_features.len())
                    .min(old_features.len())
                {
                    let delta = new_features[d] - old_features[d];
                    r[d] += delta;
                    drift += (delta as f64).powi(2);
                }
            }
        }
        drift.sqrt()
    }

    /// Insert a new node (extend vectors).
    fn insert_node(&mut self, node_type: &str, features: &[f32]) {
        let feat = if features.len() >= self.feat_dim {
            features[..self.feat_dim].to_vec()
        } else {
            let mut v = features.to_vec();
            v.resize(self.feat_dim, 0.0);
            v
        };

        self.pi_hat
            .entry(node_type.to_string())
            .or_insert_with(Vec::new)
            .push(feat);

        self.residuals
            .entry(node_type.to_string())
            .or_insert_with(Vec::new)
            .push(vec![0.0; self.feat_dim]);

        self.degrees
            .entry(node_type.to_string())
            .or_insert_with(Vec::new)
            .push(DegreeInfo::default());

        self.adjacency
            .entry(node_type.to_string())
            .or_insert_with(Vec::new)
            .push(Vec::new());
    }

    /// BasicPropagation (Algorithm 1): reduce residuals exceeding threshold.
    ///
    /// Iteratively processes nodes where |r(s)| > ε × d(s)^(1-β).
    /// Returns number of nodes that were updated.
    fn basic_propagation(&mut self) -> usize {
        let alpha = self.alpha;
        let beta = self.beta;
        let epsilon = self.epsilon;
        let mut nodes_updated = 0;
        let max_iterations = 50; // Cap iterations to avoid infinite loops

        for _iter in 0..max_iterations {
            // Find nodes exceeding threshold
            let mut active_nodes: Vec<(String, usize)> = Vec::new();

            for (node_type, res_vec) in &self.residuals {
                for (node_id, r) in res_vec.iter().enumerate() {
                    let d = self.total_degree_from(node_type, node_id) as f32;
                    let threshold = epsilon * d.max(1.0).powf(1.0 - beta);
                    let r_norm: f32 = r.iter().map(|x| x.abs()).sum();
                    if r_norm > threshold {
                        active_nodes.push((node_type.clone(), node_id));
                    }
                }
            }

            if active_nodes.is_empty() {
                break;
            }

            for (node_type, node_id) in &active_nodes {
                let r = match self.residuals.get(node_type).and_then(|v| v.get(*node_id)) {
                    Some(r) => r.clone(),
                    None => continue,
                };

                // Convert α-fraction of residual to estimated mass
                if let Some(pi) = self.pi_hat.get_mut(node_type) {
                    if let Some(pi_s) = pi.get_mut(*node_id) {
                        for d in 0..self.feat_dim {
                            pi_s[d] += alpha * r[d];
                        }
                    }
                }

                // Distribute (1-α) fraction to neighbors
                let neighbors: Vec<AdjEntry> = self
                    .adjacency
                    .get(node_type.as_str())
                    .and_then(|a| a.get(*node_id))
                    .cloned()
                    .unwrap_or_default();

                let d_s = self.total_degree_from(node_type, *node_id) as f32;
                if d_s > 0.0 && !neighbors.is_empty() {
                    let push_factor = (1.0 - alpha) / d_s;
                    for (nt, nid, _) in &neighbors {
                        if let Some(res) = self.residuals.get_mut(nt.as_str()) {
                            if let Some(r_n) = res.get_mut(*nid) {
                                for dim in 0..self.feat_dim {
                                    r_n[dim] += push_factor * r[dim];
                                }
                            }
                        }
                    }
                }

                // Zero out the residual for this node
                if let Some(res) = self.residuals.get_mut(node_type.as_str()) {
                    if let Some(r_s) = res.get_mut(*node_id) {
                        for dim in 0..self.feat_dim {
                            r_s[dim] = 0.0;
                        }
                    }
                }

                nodes_updated += 1;
            }
        }

        nodes_updated
    }

    /// Extract current π̂ as PlainEmbeddings-compatible format.
    pub fn extract_embeddings(&self) -> HashMap<String, Vec<Vec<f32>>> {
        self.pi_hat.clone()
    }

    /// Get cumulative drift since last reset.
    pub fn drift(&self) -> f64 {
        self.cumulative_drift
    }

    /// Reset drift (called after retrain).
    pub fn reset_drift(&mut self) {
        self.cumulative_drift = 0.0;
    }

    /// Total degree of a node (in + out + 1 for self-loop).
    fn total_degree(&self, node_type: &str, node_id: usize) -> usize {
        self.degrees
            .get(node_type)
            .and_then(|v| v.get(node_id))
            .map(|d| d.in_degree + d.out_degree + 1) // +1 self-loop
            .unwrap_or(1)
    }

    /// Same as total_degree but from the degrees map directly (avoids borrow issues).
    fn total_degree_from(&self, node_type: &str, node_id: usize) -> usize {
        self.total_degree(node_type, node_id)
    }

    /// Get mutation count.
    pub fn mutation_count(&self) -> usize {
        self.mutation_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_state() -> PropagationState {
        let mut embeddings: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
        embeddings.insert(
            "user".into(),
            vec![vec![1.0, 0.0, 0.5], vec![0.0, 1.0, 0.5]],
        );
        embeddings.insert(
            "item".into(),
            vec![
                vec![0.5, 0.5, 0.0],
                vec![0.3, 0.7, 0.1],
                vec![0.1, 0.9, 0.2],
            ],
        );

        let mut edges: HashMap<(String, String, String), Vec<(usize, usize)>> = HashMap::new();
        edges.insert(
            ("user".into(), "purchased".into(), "item".into()),
            vec![(0, 0), (0, 1), (1, 2)],
        );

        PropagationState::init_from_embeddings(&embeddings, &edges, 3)
    }

    #[test]
    fn test_init_from_embeddings() {
        let state = make_test_state();
        assert_eq!(state.pi_hat.get("user").unwrap().len(), 2);
        assert_eq!(state.pi_hat.get("item").unwrap().len(), 3);
        assert_eq!(state.feat_dim, 3);

        // Check degrees: user_0 has out_degree=2 (purchased item_0, item_1)
        let user_0_deg = &state.degrees["user"][0];
        assert_eq!(user_0_deg.out_degree, 2);

        // item_0 has in_degree=1
        let item_0_deg = &state.degrees["item"][0];
        assert_eq!(item_0_deg.in_degree, 1);
    }

    #[test]
    fn test_edge_insert_updates_embeddings() {
        let mut state = make_test_state();

        let old_pi = state.pi_hat["user"][1].clone();

        let result = state.apply_mutations(&[GraphEvent::InsertEdge {
            src_type: "user".into(),
            src_id: 1,
            dst_type: "item".into(),
            dst_id: 0,
            relation: "purchased".into(),
        }]);

        assert_eq!(result.events_processed, 1);
        assert!(result.drift_delta > 0.0);
        assert!(result.affected_types.contains(&"user".to_string()));
        assert!(result.affected_types.contains(&"item".to_string()));

        // User 1's embedding should have changed
        let new_pi = &state.pi_hat["user"][1];
        let changed = old_pi
            .iter()
            .zip(new_pi.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed, "user_1 embedding should change after edge insert");
    }

    #[test]
    fn test_edge_delete() {
        let mut state = make_test_state();
        let result = state.apply_mutations(&[GraphEvent::DeleteEdge {
            src_type: "user".into(),
            src_id: 0,
            dst_type: "item".into(),
            dst_id: 0,
            relation: "purchased".into(),
        }]);

        assert_eq!(result.events_processed, 1);
        assert!(result.drift_delta > 0.0);
    }

    #[test]
    fn test_node_insert() {
        let mut state = make_test_state();
        assert_eq!(state.pi_hat["item"].len(), 3);

        state.apply_mutations(&[GraphEvent::InsertNode {
            node_type: "item".into(),
            features: vec![0.5, 0.5, 0.5],
        }]);

        assert_eq!(state.pi_hat["item"].len(), 4);
    }

    #[test]
    fn test_feature_update() {
        let mut state = make_test_state();
        let result = state.apply_mutations(&[GraphEvent::UpdateFeatures {
            node_type: "user".into(),
            node_id: 0,
            new_features: vec![2.0, 0.0, 1.0],
        }]);

        assert!(result.drift_delta > 0.0);
    }

    #[test]
    fn test_cumulative_drift() {
        let mut state = make_test_state();

        let r1 = state.apply_mutations(&[GraphEvent::InsertEdge {
            src_type: "user".into(),
            src_id: 1,
            dst_type: "item".into(),
            dst_id: 0,
            relation: "purchased".into(),
        }]);
        let d1 = r1.cumulative_drift;

        let r2 = state.apply_mutations(&[GraphEvent::InsertEdge {
            src_type: "user".into(),
            src_id: 0,
            dst_type: "item".into(),
            dst_id: 2,
            relation: "purchased".into(),
        }]);
        let d2 = r2.cumulative_drift;

        assert!(
            d2 > d1,
            "Cumulative drift should increase with more mutations"
        );
        assert_eq!(state.mutation_count(), 2);
    }
}
