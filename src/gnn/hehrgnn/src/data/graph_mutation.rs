//! Graph mutation events for dynamic graph updates (InstantGNN, KDD 2022).
//!
//! Defines the event types that can mutate the graph at runtime:
//! edge insert/delete, node insert, and feature updates.

use serde::{Deserialize, Serialize};

/// A single graph mutation event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GraphEvent {
    /// Insert an edge between two existing nodes.
    InsertEdge {
        src_type: String,
        src_id: usize,
        dst_type: String,
        dst_id: usize,
        relation: String,
    },
    /// Delete an edge between two existing nodes.
    DeleteEdge {
        src_type: String,
        src_id: usize,
        dst_type: String,
        dst_id: usize,
        relation: String,
    },
    /// Insert a new node with initial features.
    InsertNode {
        node_type: String,
        features: Vec<f32>,
    },
    /// Update features of an existing node.
    UpdateFeatures {
        node_type: String,
        node_id: usize,
        new_features: Vec<f32>,
    },
}

impl GraphEvent {
    /// Returns the node types affected by this event.
    pub fn affected_types(&self) -> Vec<&str> {
        match self {
            GraphEvent::InsertEdge {
                src_type, dst_type, ..
            }
            | GraphEvent::DeleteEdge {
                src_type, dst_type, ..
            } => {
                vec![src_type.as_str(), dst_type.as_str()]
            }
            GraphEvent::InsertNode { node_type, .. }
            | GraphEvent::UpdateFeatures { node_type, .. } => {
                vec![node_type.as_str()]
            }
        }
    }
}

/// A batch of mutation events to apply atomically.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationBatch {
    pub events: Vec<GraphEvent>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_event_serde() {
        let event = GraphEvent::InsertEdge {
            src_type: "transaction".into(),
            src_id: 0,
            dst_type: "transaction-category".into(),
            dst_id: 2,
            relation: "categorized-as".into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("insert_edge"));
        let parsed: GraphEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, GraphEvent::InsertEdge { .. }));
    }

    #[test]
    fn test_affected_types() {
        let event = GraphEvent::InsertEdge {
            src_type: "a".into(),
            src_id: 0,
            dst_type: "b".into(),
            dst_id: 1,
            relation: "r".into(),
        };
        let types = event.affected_types();
        assert_eq!(types, vec!["a", "b"]);
    }
}
