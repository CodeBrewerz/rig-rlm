//! JSON/CSV loader: ingests real entity and relationship data into HeteroGraph.
//!
//! Supports two input formats:
//! 1. **Entities JSON** — `[{type, id, attributes: {key: value}}]`
//! 2. **Relations JSON** — `[{src_type, src_id, relation, dst_type, dst_id, attributes}]`
//!
//! This bridges real Finverse data (exported from TypeDB, APIs, or databases)
//! into the HeteroGraph format used by GNN models.

use burn::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::data::graph_builder::{GraphBuildConfig, GraphFact, build_hetero_graph};
use crate::data::hetero_graph::HeteroGraph;

/// A raw entity record from JSON input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRecord {
    /// Entity type (e.g., "transaction", "account", "user").
    #[serde(rename = "type")]
    pub entity_type: String,
    /// Unique identifier within the type.
    pub id: String,
    /// Arbitrary key-value attributes.
    #[serde(default)]
    pub attributes: HashMap<String, serde_json::Value>,
}

/// A raw relation record from JSON input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationRecord {
    pub src_type: String,
    pub src_id: String,
    pub relation: String,
    pub dst_type: String,
    pub dst_id: String,
    #[serde(default)]
    pub attributes: HashMap<String, serde_json::Value>,
}

/// A complete data export: entities + relations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataExport {
    #[serde(default)]
    pub entities: Vec<EntityRecord>,
    pub relations: Vec<RelationRecord>,
}

/// Load entities and relations from a JSON string.
pub fn load_from_json(json_str: &str) -> Result<DataExport, serde_json::Error> {
    serde_json::from_str(json_str)
}

/// Load entities and relations from a JSON file.
pub fn load_from_file(path: &str) -> Result<DataExport, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let export = load_from_json(&content)?;
    Ok(export)
}

/// Convert a DataExport into a HeteroGraph.
///
/// Numeric attributes are extracted as node features.
/// Non-numeric attributes are ignored (could be used for text embeddings later).
pub fn build_graph_from_export<B: Backend>(
    export: &DataExport,
    config: &GraphBuildConfig,
    device: &B::Device,
) -> HeteroGraph<B> {
    // Convert relations to GraphFacts
    let graph_facts: Vec<GraphFact> = export
        .relations
        .iter()
        .map(|r| GraphFact {
            src: (r.src_type.clone(), r.src_id.clone()),
            relation: r.relation.clone(),
            dst: (r.dst_type.clone(), r.dst_id.clone()),
        })
        .collect();

    // Ensure all entities appear even if they have no relations
    let mut all_facts = graph_facts;
    for entity in &export.entities {
        // Add a self-referencing fact to ensure the entity appears
        // (it will be deduplicated in the graph builder)
        all_facts.push(GraphFact {
            src: (entity.entity_type.clone(), entity.id.clone()),
            relation: "_exists".into(),
            dst: (entity.entity_type.clone(), entity.id.clone()),
        });
    }

    build_hetero_graph(&all_facts, config, device)
}

/// Extract numeric features from entity attributes.
///
/// Returns a map: entity_type → entity_id → Vec<f32> of feature values.
pub fn extract_numeric_features(export: &DataExport) -> HashMap<String, HashMap<String, Vec<f32>>> {
    // First pass: collect all numeric attribute keys per type
    let mut keys_per_type: HashMap<String, Vec<String>> = HashMap::new();

    for entity in &export.entities {
        let keys = keys_per_type.entry(entity.entity_type.clone()).or_default();

        for (key, value) in &entity.attributes {
            if value.is_number() && !keys.contains(key) {
                keys.push(key.clone());
            }
        }
    }

    // Sort keys for consistent ordering
    for keys in keys_per_type.values_mut() {
        keys.sort();
    }

    // Second pass: extract features
    let mut features: HashMap<String, HashMap<String, Vec<f32>>> = HashMap::new();

    for entity in &export.entities {
        let keys = keys_per_type
            .get(&entity.entity_type)
            .cloned()
            .unwrap_or_default();

        let values: Vec<f32> = keys
            .iter()
            .map(|key| {
                entity
                    .attributes
                    .get(key)
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32
            })
            .collect();

        features
            .entry(entity.entity_type.clone())
            .or_default()
            .insert(entity.id.clone(), values);
    }

    features
}

/// Summary statistics for loaded data.
#[derive(Debug, Serialize)]
pub struct LoadSummary {
    pub num_entities: usize,
    pub num_relations: usize,
    pub entity_types: HashMap<String, usize>,
    pub relation_types: HashMap<String, usize>,
}

/// Generate a summary of the loaded data.
pub fn summarize(export: &DataExport) -> LoadSummary {
    let mut entity_types: HashMap<String, usize> = HashMap::new();
    for e in &export.entities {
        *entity_types.entry(e.entity_type.clone()).or_default() += 1;
    }

    let mut relation_types: HashMap<String, usize> = HashMap::new();
    for r in &export.relations {
        *relation_types.entry(r.relation.clone()).or_default() += 1;
    }

    LoadSummary {
        num_entities: export.entities.len(),
        num_relations: export.relations.len(),
        entity_types,
        relation_types,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_load_from_json() {
        let json = r#"{
            "entities": [
                {"type": "user", "id": "u1", "attributes": {"age": 30, "name": "Alice"}},
                {"type": "user", "id": "u2", "attributes": {"age": 25}},
                {"type": "account", "id": "a1", "attributes": {"balance": 1000.50}},
                {"type": "account", "id": "a2", "attributes": {"balance": 500.00}}
            ],
            "relations": [
                {"src_type": "user", "src_id": "u1", "relation": "owns", "dst_type": "account", "dst_id": "a1"},
                {"src_type": "user", "src_id": "u2", "relation": "owns", "dst_type": "account", "dst_id": "a2"}
            ]
        }"#;

        let export = load_from_json(json).unwrap();
        assert_eq!(export.entities.len(), 4);
        assert_eq!(export.relations.len(), 2);

        let summary = summarize(&export);
        assert_eq!(summary.entity_types["user"], 2);
        assert_eq!(summary.entity_types["account"], 2);
        assert_eq!(summary.relation_types["owns"], 2);
    }

    #[test]
    fn test_build_graph_from_export() {
        let json = r#"{
            "entities": [
                {"type": "user", "id": "u1"},
                {"type": "account", "id": "a1"},
                {"type": "account", "id": "a2"}
            ],
            "relations": [
                {"src_type": "user", "src_id": "u1", "relation": "owns", "dst_type": "account", "dst_id": "a1"},
                {"src_type": "user", "src_id": "u1", "relation": "owns", "dst_type": "account", "dst_id": "a2"}
            ]
        }"#;

        let export = load_from_json(json).unwrap();
        let device = <TestBackend as Backend>::Device::default();

        let config = GraphBuildConfig {
            node_feat_dim: 8,
            add_reverse_edges: true,
            add_self_loops: false,
            add_positional_encoding: true,
        };

        let graph = build_graph_from_export::<TestBackend>(&export, &config, &device);
        assert!(graph.total_nodes() >= 3); // at least user + 2 accounts
        assert!(graph.total_edges() >= 2); // at least 2 owns edges
    }

    #[test]
    fn test_extract_numeric_features() {
        let json = r#"{
            "entities": [
                {"type": "user", "id": "u1", "attributes": {"age": 30, "score": 0.9, "name": "Alice"}},
                {"type": "user", "id": "u2", "attributes": {"age": 25, "score": 0.7}}
            ],
            "relations": []
        }"#;

        let export = load_from_json(json).unwrap();
        let features = extract_numeric_features(&export);

        let user_features = &features["user"];
        assert_eq!(user_features["u1"].len(), 2); // age, score (name is non-numeric)
        assert_eq!(user_features["u2"].len(), 2);
    }
}
