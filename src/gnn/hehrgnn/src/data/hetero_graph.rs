//! Heterogeneous graph data structure.
//!
//! This is the Burn equivalent of PyG's `HeteroData`. It stores typed nodes
//! and typed edges, where each node type has its own feature tensor and each
//! edge type (src_type, relation, dst_type) has its own edge index.
//!
//! In the finance domain, a "transaction-as-node" pattern expands each
//! hyperedge (transaction touching multiple entities) into a node connected
//! to all its participants via typed edges.

use burn::prelude::*;
use std::collections::HashMap;

/// Identifies a node type (e.g., "transaction", "account", "merchant").
pub type NodeType = String;

/// Identifies a relation type (e.g., "posted_to", "matched_to").
pub type RelationType = String;

/// A typed edge triplet: (source_node_type, relation_type, dest_node_type).
pub type EdgeType = (NodeType, RelationType, NodeType);

/// A heterogeneous graph with typed nodes and typed edges.
///
/// Each node type has:
/// - A feature tensor `[num_nodes_of_type, feat_dim]`
/// - A count of how many nodes of that type exist
///
/// Each edge type has:
/// - An edge index tensor `[2, num_edges]` (row 0 = source indices, row 1 = dest indices)
/// - Optional edge feature tensor `[num_edges, edge_feat_dim]`
///
/// Node indices are **local** to each type (0..num_nodes_of_type).
#[derive(Debug, Clone)]
pub struct HeteroGraph<B: Backend> {
    /// Node features per type: `[num_nodes, feat_dim]`.
    pub node_features: HashMap<NodeType, Tensor<B, 2>>,

    /// Number of nodes per type.
    pub node_counts: HashMap<NodeType, usize>,

    /// Feature dimension per node type (can differ across types).
    pub node_feat_dims: HashMap<NodeType, usize>,

    /// Edge indices per edge type: `[2, num_edges]` (source_local, dest_local).
    pub edge_index: HashMap<EdgeType, Tensor<B, 2, Int>>,

    /// Optional edge features per edge type: `[num_edges, edge_feat_dim]`.
    pub edge_features: HashMap<EdgeType, Tensor<B, 2>>,

    /// Number of edges per type.
    pub edge_counts: HashMap<EdgeType, usize>,
}

impl<B: Backend> HeteroGraph<B> {
    /// Create an empty heterogeneous graph.
    pub fn new() -> Self {
        Self {
            node_features: HashMap::new(),
            node_counts: HashMap::new(),
            node_feat_dims: HashMap::new(),
            edge_index: HashMap::new(),
            edge_features: HashMap::new(),
            edge_counts: HashMap::new(),
        }
    }

    /// Add a node type with its feature tensor.
    ///
    /// # Arguments
    /// - `node_type`: name of the node type
    /// - `features`: tensor of shape `[num_nodes, feat_dim]`
    pub fn add_node_type(&mut self, node_type: &str, features: Tensor<B, 2>) {
        let dims = features.dims();
        let num_nodes = dims[0];
        let feat_dim = dims[1];

        self.node_counts.insert(node_type.to_string(), num_nodes);
        self.node_feat_dims.insert(node_type.to_string(), feat_dim);
        self.node_features.insert(node_type.to_string(), features);
    }

    /// Add an edge type with its edge index.
    ///
    /// # Arguments
    /// - `edge_type`: `(src_type, relation, dst_type)`
    /// - `edge_idx`: tensor of shape `[2, num_edges]`
    pub fn add_edge_type(&mut self, edge_type: EdgeType, edge_idx: Tensor<B, 2, Int>) {
        let num_edges = edge_idx.dims()[1];
        self.edge_counts.insert(edge_type.clone(), num_edges);
        self.edge_index.insert(edge_type, edge_idx);
    }

    /// Add edge features for an existing edge type.
    pub fn add_edge_features(&mut self, edge_type: &EdgeType, features: Tensor<B, 2>) {
        self.edge_features.insert(edge_type.clone(), features);
    }

    /// Get the total number of nodes across all types.
    pub fn total_nodes(&self) -> usize {
        self.node_counts.values().sum()
    }

    /// Get the total number of edges across all types.
    pub fn total_edges(&self) -> usize {
        self.edge_counts.values().sum()
    }

    /// Get all node types.
    pub fn node_types(&self) -> Vec<&NodeType> {
        let mut types: Vec<_> = self.node_counts.keys().collect();
        types.sort();
        types
    }

    /// Get all edge types.
    pub fn edge_types(&self) -> Vec<&EdgeType> {
        let mut types: Vec<_> = self.edge_counts.keys().collect();
        types.sort();
        types
    }

    /// Get the global offset for a node type (for converting local→global indices).
    ///
    /// Node types are sorted alphabetically; each type's global offset is the
    /// cumulative sum of previous types' counts.
    pub fn global_offset(&self, node_type: &str) -> usize {
        let mut offset = 0;
        for nt in self.node_types() {
            if nt == node_type {
                return offset;
            }
            offset += self.node_counts[nt];
        }
        offset
    }

    /// Move all tensors to the given device.
    pub fn to_device(&self, device: &B::Device) -> Self {
        let node_features = self
            .node_features
            .iter()
            .map(|(k, v)| (k.clone(), v.clone().to_device(device)))
            .collect();

        let edge_index = self
            .edge_index
            .iter()
            .map(|(k, v)| (k.clone(), v.clone().to_device(device)))
            .collect();

        let edge_features = self
            .edge_features
            .iter()
            .map(|(k, v)| (k.clone(), v.clone().to_device(device)))
            .collect();

        Self {
            node_features,
            node_counts: self.node_counts.clone(),
            node_feat_dims: self.node_feat_dims.clone(),
            edge_index,
            edge_features,
            edge_counts: self.edge_counts.clone(),
        }
    }

    /// Extract edges of a given type as (source_indices, dest_indices) Vec pairs.
    pub fn edges_as_vecs(&self, edge_type: &EdgeType) -> Option<(Vec<i64>, Vec<i64>)> {
        self.edge_index.get(edge_type).map(|idx| {
            let data = idx.clone().into_data();
            let flat: Vec<i64> = data
                .as_slice::<i64>()
                .expect("Failed to read edge index")
                .to_vec();
            let num_edges = idx.dims()[1];
            let src = flat[..num_edges].to_vec();
            let dst = flat[num_edges..].to_vec();
            (src, dst)
        })
    }

    /// Get neighbor node indices for a given node in a given edge type.
    ///
    /// Returns destination node indices where `source == node_idx`.
    pub fn neighbors(
        &self,
        edge_type: &EdgeType,
        node_idx: usize,
        direction: EdgeDirection,
    ) -> Vec<usize> {
        let Some((src, dst)) = self.edges_as_vecs(edge_type) else {
            return Vec::new();
        };

        match direction {
            EdgeDirection::Outgoing => src
                .iter()
                .zip(dst.iter())
                .filter(|&(s, _)| *s as usize == node_idx)
                .map(|(_, d)| *d as usize)
                .collect(),
            EdgeDirection::Incoming => dst
                .iter()
                .zip(src.iter())
                .filter(|&(d, _)| *d as usize == node_idx)
                .map(|(_, s)| *s as usize)
                .collect(),
        }
    }
}

/// Direction for edge traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeDirection {
    /// Follow edges from source → destination.
    Outgoing,
    /// Follow edges from destination → source (reverse).
    Incoming,
}

/// Metadata about a mini-batch subgraph extracted via neighbor sampling.
#[derive(Debug, Clone)]
pub struct MiniBatchGraph<B: Backend> {
    /// The subgraph with re-indexed nodes.
    pub graph: HeteroGraph<B>,

    /// Mapping from local index → original global index, per node type.
    pub original_ids: HashMap<NodeType, Vec<usize>>,

    /// Which nodes in the subgraph are the seed (target) nodes, per type.
    pub seed_mask: HashMap<NodeType, Vec<bool>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_hetero_graph_construction() {
        let device = <TestBackend as Backend>::Device::default();
        let mut graph = HeteroGraph::<TestBackend>::new();

        // Add node types
        let tx_features = Tensor::random([5, 8], burn::tensor::Distribution::Default, &device);
        let acct_features = Tensor::random([3, 8], burn::tensor::Distribution::Default, &device);
        graph.add_node_type("transaction", tx_features);
        graph.add_node_type("account", acct_features);

        assert_eq!(graph.node_counts["transaction"], 5);
        assert_eq!(graph.node_counts["account"], 3);
        assert_eq!(graph.total_nodes(), 8);

        // Add edge type: transaction --posted_to--> account
        let edge_idx =
            Tensor::<TestBackend, 2, Int>::from_data([[0, 1, 2, 3, 4], [0, 0, 1, 1, 2]], &device);
        graph.add_edge_type(
            ("transaction".into(), "posted_to".into(), "account".into()),
            edge_idx,
        );

        assert_eq!(graph.total_edges(), 5);
        assert_eq!(graph.node_types().len(), 2);
        assert_eq!(graph.edge_types().len(), 1);
    }

    #[test]
    fn test_hetero_graph_neighbors() {
        let device = <TestBackend as Backend>::Device::default();
        let mut graph = HeteroGraph::<TestBackend>::new();

        let tx_features = Tensor::random([4, 4], burn::tensor::Distribution::Default, &device);
        let acct_features = Tensor::random([3, 4], burn::tensor::Distribution::Default, &device);
        graph.add_node_type("tx", tx_features);
        graph.add_node_type("acct", acct_features);

        // tx0 → acct0, tx0 → acct1, tx1 → acct0, tx2 → acct2
        let edge_idx =
            Tensor::<TestBackend, 2, Int>::from_data([[0, 0, 1, 2], [0, 1, 0, 2]], &device);
        let et = ("tx".into(), "posted_to".into(), "acct".into());
        graph.add_edge_type(et.clone(), edge_idx);

        // tx0 has outgoing neighbors acct0, acct1
        let neighbors = graph.neighbors(&et, 0, EdgeDirection::Outgoing);
        assert_eq!(neighbors, vec![0, 1]);

        // acct0 has incoming neighbors tx0, tx1
        let neighbors = graph.neighbors(&et, 0, EdgeDirection::Incoming);
        assert_eq!(neighbors, vec![0, 1]);
    }

    #[test]
    fn test_hetero_graph_global_offsets() {
        let device = <TestBackend as Backend>::Device::default();
        let mut graph = HeteroGraph::<TestBackend>::new();

        graph.add_node_type(
            "account",
            Tensor::random([3, 4], burn::tensor::Distribution::Default, &device),
        );
        graph.add_node_type(
            "merchant",
            Tensor::random([5, 4], burn::tensor::Distribution::Default, &device),
        );
        graph.add_node_type(
            "transaction",
            Tensor::random([10, 4], burn::tensor::Distribution::Default, &device),
        );

        // Sorted: account(3), merchant(5), transaction(10)
        assert_eq!(graph.global_offset("account"), 0);
        assert_eq!(graph.global_offset("merchant"), 3);
        assert_eq!(graph.global_offset("transaction"), 8);
    }

    #[test]
    fn test_hetero_graph_to_device() {
        let device = <TestBackend as Backend>::Device::default();
        let mut graph = HeteroGraph::<TestBackend>::new();

        graph.add_node_type(
            "tx",
            Tensor::random([4, 4], burn::tensor::Distribution::Default, &device),
        );
        let edge_idx = Tensor::<TestBackend, 2, Int>::from_data([[0, 1], [1, 0]], &device);
        graph.add_edge_type(("tx".into(), "self_loop".into(), "tx".into()), edge_idx);

        let graph2 = graph.to_device(&device);
        assert_eq!(graph2.total_nodes(), 4);
        assert_eq!(graph2.total_edges(), 2);
    }
}
