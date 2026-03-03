//! Circuit DAG nodes for Probabilistic Circuits.
//!
//! A PC is a DAG of three node types:
//! - **Sum** nodes: weighted mixtures (log-space weights)
//! - **Product** nodes: independent factorizations
//! - **Input** nodes: leaf distributions
//!
//! Nodes are arena-allocated (Vec<CircuitNode>) and reference children by index.

use serde::{Deserialize, Serialize};

use super::distribution::Distribution;

/// Unique index into the circuit's node arena.
pub type NodeId = usize;

/// The type of a circuit node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeKind {
    /// Weighted mixture: log P = log Σ_i exp(w_i + log P_child_i).
    Sum {
        /// Log-space mixture weights (one per child).
        log_weights: Vec<f64>,
    },
    /// Independent factorization: log P = Σ_i log P_child_i.
    Product,
    /// Leaf distribution over a single variable.
    Input {
        /// Which variable this input node is defined on.
        var: usize,
        /// The distribution.
        dist: Distribution,
    },
}

/// A node in the circuit DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitNode {
    /// Unique ID (index in arena).
    pub id: NodeId,
    /// Node type and parameters.
    pub kind: NodeKind,
    /// Indices of children in the arena.
    pub children: Vec<NodeId>,
    /// Set of variables in this node's scope (sorted).
    pub scope: Vec<usize>,
}

/// Arena-based circuit builder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBuilder {
    pub nodes: Vec<CircuitNode>,
}

impl CircuitBuilder {
    pub fn new() -> Self {
        CircuitBuilder { nodes: Vec::new() }
    }

    /// Add an input (leaf) node for variable `var` with distribution `dist`.
    pub fn add_input(&mut self, var: usize, dist: Distribution) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(CircuitNode {
            id,
            kind: NodeKind::Input { var, dist },
            children: vec![],
            scope: vec![var],
        });
        id
    }

    /// Add a product node combining children with disjoint scopes.
    pub fn add_product(&mut self, children: Vec<NodeId>) -> NodeId {
        let id = self.nodes.len();
        let mut scope: Vec<usize> = children
            .iter()
            .flat_map(|&c| self.nodes[c].scope.clone())
            .collect();
        scope.sort();
        scope.dedup();
        self.nodes.push(CircuitNode {
            id,
            kind: NodeKind::Product,
            children,
            scope,
        });
        id
    }

    /// Add a sum node with uniform weights over children.
    /// All children must have the same scope (smoothness).
    pub fn add_sum(&mut self, children: Vec<NodeId>) -> NodeId {
        let n = children.len();
        let log_w = -(n as f64).ln();
        self.add_sum_weighted(children, vec![log_w; n])
    }

    /// Add a sum node with specified log-space weights.
    pub fn add_sum_weighted(&mut self, children: Vec<NodeId>, log_weights: Vec<f64>) -> NodeId {
        assert_eq!(children.len(), log_weights.len());
        let id = self.nodes.len();
        // Scope = union of children scopes (should be same for smooth PCs)
        let mut scope: Vec<usize> = children
            .iter()
            .flat_map(|&c| self.nodes[c].scope.clone())
            .collect();
        scope.sort();
        scope.dedup();
        self.nodes.push(CircuitNode {
            id,
            kind: NodeKind::Sum { log_weights },
            children,
            scope,
        });
        id
    }

    /// Number of nodes in the circuit.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Get the root node (last added).
    pub fn root(&self) -> NodeId {
        self.nodes.len() - 1
    }

    /// Total number of variables in the circuit's scope.
    pub fn num_vars(&self) -> usize {
        let root = self.root();
        self.nodes[root].scope.len()
    }

    /// Count total learnable parameters.
    pub fn num_params(&self) -> usize {
        self.nodes
            .iter()
            .map(|n| match &n.kind {
                NodeKind::Sum { log_weights } => log_weights.len(),
                NodeKind::Input { dist, .. } => dist.num_params(),
                NodeKind::Product => 0,
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_simple_circuit() {
        let mut b = CircuitBuilder::new();
        // Two bernoulli inputs
        let x0 = b.add_input(0, Distribution::bernoulli(0.5));
        let x1 = b.add_input(1, Distribution::bernoulli(0.5));
        // Product
        let p = b.add_product(vec![x0, x1]);
        // Sum (trivial — one component)
        let s = b.add_sum(vec![p]);

        assert_eq!(b.len(), 4);
        assert_eq!(b.nodes[p].scope, vec![0, 1]);
        assert_eq!(b.nodes[s].scope, vec![0, 1]);
        assert_eq!(b.num_vars(), 2);
    }
}
