//! Compiled Probabilistic Circuit — efficient forward/backward inference.
//!
//! Compiles a `CircuitBuilder` DAG into a flat topologically-ordered
//! representation for O(n) inference passes. All computation in log-space.
//!
//! # Forward Pass (bottom-up)
//! - Input nodes: log P(X=x | params)
//! - Product nodes: Σ children log-values (independence)
//! - Sum nodes: log-sum-exp(log_w_i + child_log_value_i) (mixture)
//!
//! # Backward Pass (top-down)
//! - Root flow = 1.0
//! - Sum child flow: parent_flow × exp(log_w_i + child_value - parent_value)
//! - Product child flow: parent_flow (independence)

use serde::{Deserialize, Serialize};

use super::node::{CircuitBuilder, CircuitNode, NodeId, NodeKind};

/// Evidence for a single data point.
/// `values[var]` = observed value (or `None` if marginalized).
pub type Evidence = Vec<Option<usize>>;

/// Compiled circuit ready for inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledCircuit {
    /// Arena of circuit nodes (owned).
    pub nodes: Vec<CircuitNode>,
    /// Topological order (bottom-up: leaves first, root last).
    pub topo_order: Vec<NodeId>,
    /// Root node index.
    pub root: NodeId,
    /// Number of variables.
    pub num_vars: usize,
    /// Forward pass cache: log-values per node.
    #[serde(skip)]
    log_values: Vec<f64>,
    /// Backward pass cache: flows per node.
    #[serde(skip)]
    flows: Vec<f64>,
    /// Parent map: for each node, list of (parent_id, child_index_in_parent).
    parents: Vec<Vec<(NodeId, usize)>>,
}

impl CompiledCircuit {
    /// Compile from a builder. The root is assumed to be the last node.
    pub fn compile(builder: &CircuitBuilder) -> Self {
        let nodes = builder.nodes.clone();
        let root = builder.root();
        let n = nodes.len();

        // Build parent map
        let mut parents: Vec<Vec<(NodeId, usize)>> = vec![vec![]; n];
        for node in &nodes {
            for (ci, &child) in node.children.iter().enumerate() {
                parents[child].push((node.id, ci));
            }
        }

        // Topological sort (Kahn's algorithm)
        let mut in_degree = vec![0usize; n];
        for node in &nodes {
            for &child in &node.children {
                in_degree[child] += 1;
            }
        }
        // Wait — in_degree should count *parents*, and topo order goes leaves→root.
        // Actually for bottom-up: leaf = in_degree from parents = 0 is wrong.
        // Let's use: out-degree = 0 (no children) → leaf. Process bottom-up.
        // Better: reverse the edge direction for Kahn's.

        // Simple DFS-based topological sort, leaves first.
        let mut topo_order = Vec::with_capacity(n);
        let mut visited = vec![false; n];
        fn dfs(
            node: NodeId,
            nodes: &[CircuitNode],
            visited: &mut Vec<bool>,
            order: &mut Vec<NodeId>,
        ) {
            if visited[node] {
                return;
            }
            visited[node] = true;
            for &child in &nodes[node].children {
                dfs(child, nodes, visited, order);
            }
            order.push(node);
        }
        dfs(root, &nodes, &mut visited, &mut topo_order);

        let num_vars = if n > 0 { nodes[root].scope.len() } else { 0 };

        CompiledCircuit {
            nodes,
            topo_order,
            root,
            num_vars,
            log_values: vec![0.0; n],
            flows: vec![0.0; n],
            parents,
        }
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Forward pass: compute log P(evidence) bottom-up.
    ///
    /// For marginalized variables (evidence[var] = None), input nodes
    /// return log(1) = 0 (uniform marginal over all values).
    pub fn forward(&mut self, evidence: &Evidence) -> f64 {
        for &node_id in &self.topo_order {
            let node = &self.nodes[node_id];
            let val = match &node.kind {
                NodeKind::Input { var, dist } => {
                    if let Some(x) = evidence.get(*var).and_then(|v| *v) {
                        dist.log_prob(x)
                    } else {
                        // Marginalize: log Σ_x P(x) = log(1) = 0 for normalized dists
                        0.0
                    }
                }
                NodeKind::Product => {
                    // log P = Σ log P_child
                    node.children
                        .iter()
                        .map(|&c| self.log_values[c])
                        .sum::<f64>()
                }
                NodeKind::Sum { log_weights } => {
                    // log P = log Σ exp(log_w_i + log P_child_i)
                    log_sum_exp(
                        &node
                            .children
                            .iter()
                            .zip(log_weights.iter())
                            .map(|(&c, &w)| w + self.log_values[c])
                            .collect::<Vec<_>>(),
                    )
                }
            };
            self.log_values[node_id] = val;
        }
        self.log_values[self.root]
    }

    /// Forward pass with continuous evidence.
    pub fn forward_f64(&mut self, evidence: &[Option<f64>]) -> f64 {
        for &node_id in &self.topo_order {
            let node = &self.nodes[node_id];
            let val = match &node.kind {
                NodeKind::Input { var, dist } => {
                    if let Some(x) = evidence.get(*var).and_then(|v| *v) {
                        dist.log_prob_f64(x)
                    } else {
                        0.0
                    }
                }
                NodeKind::Product => node
                    .children
                    .iter()
                    .map(|&c| self.log_values[c])
                    .sum::<f64>(),
                NodeKind::Sum { log_weights } => log_sum_exp(
                    &node
                        .children
                        .iter()
                        .zip(log_weights.iter())
                        .map(|(&c, &w)| w + self.log_values[c])
                        .collect::<Vec<_>>(),
                ),
            };
            self.log_values[node_id] = val;
        }
        self.log_values[self.root]
    }

    /// Backward pass: compute flows (parameter gradients) top-down.
    ///
    /// Must be called after `forward()`. Returns flows for EM.
    pub fn backward(&mut self) {
        // Reset flows
        for f in self.flows.iter_mut() {
            *f = 0.0;
        }
        self.flows[self.root] = 1.0;

        // Top-down (reverse topo order)
        for &node_id in self.topo_order.iter().rev() {
            let flow = self.flows[node_id];
            if flow == 0.0 {
                continue;
            }

            let node = self.nodes[node_id].clone();
            match &node.kind {
                NodeKind::Sum { log_weights } => {
                    // Child i gets: flow × P(child_i chosen) =
                    //   flow × exp(log_w_i + child_value - parent_value)
                    let parent_val = self.log_values[node_id];
                    for (i, (&child, &log_w)) in
                        node.children.iter().zip(log_weights.iter()).enumerate()
                    {
                        let child_contribution =
                            (log_w + self.log_values[child] - parent_val).exp();
                        self.flows[child] += flow * child_contribution;
                    }
                }
                NodeKind::Product => {
                    // Each child gets the full parent flow (independence)
                    for &child in &node.children {
                        self.flows[child] += flow;
                    }
                }
                NodeKind::Input { .. } => {
                    // Leaf — no children to propagate to
                }
            }
        }
    }

    /// Get log-value of a node after forward pass.
    pub fn log_value(&self, node: NodeId) -> f64 {
        self.log_values[node]
    }

    /// Get flow of a node after backward pass.
    pub fn flow(&self, node_id: NodeId) -> f64 {
        self.flows[node_id]
    }

    /// Get the sum node edge flows (for EM):
    /// For sum node `node_id`, returns flows for each child edge.
    pub fn sum_edge_flows(&self, node_id: NodeId) -> Vec<f64> {
        let node = &self.nodes[node_id];
        match &node.kind {
            NodeKind::Sum { log_weights } => {
                let parent_val = self.log_values[node_id];
                let parent_flow = self.flows[node_id];
                node.children
                    .iter()
                    .zip(log_weights.iter())
                    .map(|(&child, &log_w)| {
                        parent_flow * (log_w + self.log_values[child] - parent_val).exp()
                    })
                    .collect()
            }
            _ => vec![],
        }
    }

    /// Get all sum node IDs.
    pub fn sum_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter_map(|n| match &n.kind {
                NodeKind::Sum { .. } => Some(n.id),
                _ => None,
            })
            .collect()
    }

    /// Get all input node IDs.
    pub fn input_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter_map(|n| match &n.kind {
                NodeKind::Input { .. } => Some(n.id),
                _ => None,
            })
            .collect()
    }

    /// Update sum node weights (in log-space).
    pub fn set_sum_weights(&mut self, node_id: NodeId, new_log_weights: Vec<f64>) {
        if let NodeKind::Sum {
            ref mut log_weights,
        } = self.nodes[node_id].kind
        {
            *log_weights = new_log_weights;
        }
    }

    /// Update input node distribution.
    pub fn set_input_dist(&mut self, node_id: NodeId, new_dist: super::distribution::Distribution) {
        if let NodeKind::Input { ref mut dist, .. } = self.nodes[node_id].kind {
            *dist = new_dist;
        }
    }
}

/// Numerically stable log-sum-exp.
pub fn log_sum_exp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let sum_exp: f64 = values.iter().map(|v| (v - max).exp()).sum();
    max + sum_exp.ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::pc::distribution::Distribution;
    use crate::model::pc::node::CircuitBuilder;

    #[test]
    fn test_simple_forward() {
        // P(X0, X1) = P(X0) × P(X1) with Bernoulli(0.5) each
        let mut b = CircuitBuilder::new();
        let x0 = b.add_input(0, Distribution::bernoulli(0.5));
        let x1 = b.add_input(1, Distribution::bernoulli(0.5));
        let p = b.add_product(vec![x0, x1]);

        let mut c = CompiledCircuit::compile(&b);

        // P(X0=0, X1=0) = 0.5 × 0.5 = 0.25
        let ll = c.forward(&vec![Some(0), Some(0)]);
        assert!((ll.exp() - 0.25).abs() < 1e-10, "P(0,0) = {}", ll.exp());

        // P(X0=1, X1=1) = 0.5 × 0.5 = 0.25
        let ll = c.forward(&vec![Some(1), Some(1)]);
        assert!((ll.exp() - 0.25).abs() < 1e-10, "P(1,1) = {}", ll.exp());
    }

    #[test]
    fn test_marginal() {
        // P(X0) with X1 marginalized should = P(X0)
        let mut b = CircuitBuilder::new();
        let x0 = b.add_input(0, Distribution::bernoulli(0.3));
        let x1 = b.add_input(1, Distribution::bernoulli(0.7));
        let p = b.add_product(vec![x0, x1]);

        let mut c = CompiledCircuit::compile(&b);

        // P(X0=1, X1=?) = P(X0=1) × 1 = 0.3
        let ll = c.forward(&vec![Some(1), None]);
        assert!(
            (ll.exp() - 0.3).abs() < 1e-6,
            "P(X0=1, X1=?) = {}",
            ll.exp()
        );
    }

    #[test]
    fn test_mixture() {
        // P(X) = 0.6 × Bernoulli(0.2) + 0.4 × Bernoulli(0.8)
        let mut b = CircuitBuilder::new();
        let x0a = b.add_input(0, Distribution::bernoulli(0.2));
        let x0b = b.add_input(0, Distribution::bernoulli(0.8));
        let log_weights = vec![0.6_f64.ln(), 0.4_f64.ln()];
        let _s = b.add_sum_weighted(vec![x0a, x0b], log_weights);

        let mut c = CompiledCircuit::compile(&b);

        // P(X=1) = 0.6 × 0.2 + 0.4 × 0.8 = 0.12 + 0.32 = 0.44
        let ll = c.forward(&vec![Some(1)]);
        assert!((ll.exp() - 0.44).abs() < 1e-6, "P(X=1) = {}", ll.exp());

        // P(X=0) = 0.6 × 0.8 + 0.4 × 0.2 = 0.48 + 0.08 = 0.56
        let ll = c.forward(&vec![Some(0)]);
        assert!((ll.exp() - 0.56).abs() < 1e-6, "P(X=0) = {}", ll.exp());
    }

    #[test]
    fn test_backward_flows() {
        let mut b = CircuitBuilder::new();
        let x0a = b.add_input(0, Distribution::bernoulli(0.3));
        let x0b = b.add_input(0, Distribution::bernoulli(0.7));
        let _s = b.add_sum(vec![x0a, x0b]);

        let mut c = CompiledCircuit::compile(&b);
        c.forward(&vec![Some(1)]);
        c.backward();

        // Root flow should be 1.0
        assert!((c.flows[c.root] - 1.0).abs() < 1e-10);
        // Child flows should sum to 1.0 (for a sum node)
        let child_flow_sum: f64 = c.flows[x0a] + c.flows[x0b];
        assert!(
            (child_flow_sum - 1.0).abs() < 1e-6,
            "Child flow sum = {}",
            child_flow_sum
        );
    }
}
