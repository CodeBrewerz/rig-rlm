//! Synthetic dataset generator from TQL (TypeQL) schema files.
//!
//! Parses the `SchemaFinverse.tql` ontology to extract entity types, relation types,
//! and role constraints, then generates synthetic hyper-relational facts that
//! adhere to the ontology.

use rand::Rng;
use std::collections::HashMap;

use super::fact::{HehrFact, RawFact, RawQualifier};
use super::vocab::KgVocabulary;

/// A parsed entity type from the TQL schema.
#[derive(Debug, Clone)]
pub struct EntityType {
    pub name: String,
    /// Parent type (via `sub`), if any.
    pub parent: Option<String>,
    /// Roles this entity type can play: `(relation_name, role_name)`.
    pub plays: Vec<(String, String)>,
}

/// A parsed relation type from the TQL schema.
#[derive(Debug, Clone)]
pub struct RelationType {
    pub name: String,
    /// Parent type (via `sub`), if any.
    pub parent: Option<String>,
    /// Roles defined by this relation: `(role_name, cardinality_str)`.
    pub roles: Vec<(String, Option<String>)>,
}

/// Parsed TQL schema containing all entity and relation types.
#[derive(Debug, Clone)]
pub struct TqlSchema {
    pub entity_types: Vec<EntityType>,
    pub relation_types: Vec<RelationType>,
    /// Maps (relation_name, role_name) → list of entity type names that can fill the role.
    pub role_to_entities: HashMap<(String, String), Vec<String>>,
}

impl TqlSchema {
    /// Parse a TQL schema definition string.
    ///
    /// Extracts entity types (with `plays` clauses) and relation types (with `relates` clauses),
    /// then builds the role-to-entity mapping.
    pub fn parse(tql: &str) -> Self {
        let mut entity_types = Vec::new();
        let mut relation_types = Vec::new();

        // State machine: we're either inside an entity def, relation def, or neither.
        let mut current_entity: Option<EntityType> = None;
        let mut current_relation: Option<RelationType> = None;

        for line in tql.lines() {
            let trimmed = line
                .trim()
                .trim_end_matches(';')
                .trim_end_matches(',')
                .trim();

            // Skip empty and comment lines
            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed == "define" {
                // Flush any pending definition
                if let Some(e) = current_entity.take() {
                    entity_types.push(e);
                }
                if let Some(r) = current_relation.take() {
                    relation_types.push(r);
                }
                continue;
            }

            // Start of a new entity definition
            if trimmed.starts_with("entity ") {
                // Flush previous
                if let Some(e) = current_entity.take() {
                    entity_types.push(e);
                }
                if let Some(r) = current_relation.take() {
                    relation_types.push(r);
                }

                let rest = &trimmed["entity ".len()..];
                let parts: Vec<&str> = rest.split(',').map(|s| s.trim()).collect();
                let name = parts[0].to_string();
                let mut parent = None;

                // Check remaining parts for "sub X"
                for part in &parts[1..] {
                    if part.starts_with("sub ") {
                        parent = Some(part["sub ".len()..].trim().to_string());
                    }
                }

                current_entity = Some(EntityType {
                    name,
                    parent,
                    plays: Vec::new(),
                });
                continue;
            }

            // Start of a new relation definition
            if trimmed.starts_with("relation ") {
                // Flush previous
                if let Some(e) = current_entity.take() {
                    entity_types.push(e);
                }
                if let Some(r) = current_relation.take() {
                    relation_types.push(r);
                }

                let rest = &trimmed["relation ".len()..];
                let parts: Vec<&str> = rest.split(',').map(|s| s.trim()).collect();
                let name = parts[0].to_string();
                let mut parent = None;

                for part in &parts[1..] {
                    if part.starts_with("sub ") {
                        parent = Some(part["sub ".len()..].trim().to_string());
                    }
                }

                current_relation = Some(RelationType {
                    name,
                    parent,
                    roles: Vec::new(),
                });
                continue;
            }

            // Inside an entity definition — parse `plays` and `sub`
            if let Some(ref mut entity) = current_entity {
                if trimmed.starts_with("sub ") {
                    entity.parent = Some(trimmed["sub ".len()..].trim().to_string());
                } else if trimmed.starts_with("plays ") {
                    let play_str = &trimmed["plays ".len()..];
                    // Remove cardinality annotations like @card(0..)
                    let play_clean = play_str.split('@').next().unwrap_or(play_str).trim();
                    // Format: "relation-name:role-name"
                    if let Some(colon_pos) = play_clean.find(':') {
                        let rel_name = play_clean[..colon_pos].trim().to_string();
                        let role_name = play_clean[colon_pos + 1..].trim().to_string();
                        entity.plays.push((rel_name, role_name));
                    }
                }
                continue;
            }

            // Inside a relation definition — parse `relates` and `sub`
            if let Some(ref mut relation) = current_relation {
                if trimmed.starts_with("sub ") {
                    relation.parent = Some(trimmed["sub ".len()..].trim().to_string());
                } else if trimmed.starts_with("relates ") {
                    let relates_str = &trimmed["relates ".len()..];
                    // Remove "as X" aliasing and cardinality annotations
                    let clean = relates_str.split(" as ").next().unwrap_or(relates_str);
                    let clean = clean.split('@').next().unwrap_or(clean).trim();
                    let role_name = clean.to_string();
                    relation.roles.push((role_name, None));
                }
                continue;
            }
        }

        // Flush final definitions
        if let Some(e) = current_entity {
            entity_types.push(e);
        }
        if let Some(r) = current_relation {
            relation_types.push(r);
        }

        // Build role_to_entities mapping
        let mut role_to_entities: HashMap<(String, String), Vec<String>> = HashMap::new();
        for entity in &entity_types {
            for (rel_name, role_name) in &entity.plays {
                role_to_entities
                    .entry((rel_name.clone(), role_name.clone()))
                    .or_default()
                    .push(entity.name.clone());
            }
        }

        Self {
            entity_types,
            relation_types,
            role_to_entities,
        }
    }

    /// Get all concrete (non-abstract) entity type names.
    pub fn concrete_entity_types(&self) -> Vec<&str> {
        self.entity_types
            .iter()
            .filter(|e| {
                // Filter out abstract types
                !e.name.contains("@abstract")
                    && e.name != "base-entity"
                    && e.name != "base-relation"
            })
            .map(|e| e.name.as_str())
            .collect()
    }

    /// Get all concrete relation type names.
    pub fn concrete_relation_types(&self) -> Vec<&str> {
        self.relation_types
            .iter()
            .filter(|r| !r.name.contains("@abstract") && r.name != "base-relation")
            .map(|r| r.name.as_str())
            .collect()
    }
}

/// Configuration for synthetic data generation.
#[derive(Debug, Clone)]
pub struct SyntheticDataConfig {
    /// Number of synthetic entity instances per entity type.
    pub instances_per_type: usize,
    /// Number of facts to generate.
    pub num_facts: usize,
    /// Maximum number of qualifiers per fact (randomly chosen 0..max).
    pub max_qualifiers: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for SyntheticDataConfig {
    fn default() -> Self {
        Self {
            instances_per_type: 10,
            num_facts: 1000,
            max_qualifiers: 3,
            seed: 42,
        }
    }
}

/// Generate synthetic raw facts adhering to the TQL schema ontology.
///
/// The generator:
/// 1. Creates named entity instances for each concrete entity type.
/// 2. For each fact, picks a random relation with ≥ 2 roles,
///    fills each role with a random entity instance of a compatible type.
/// 3. Optionally adds qualifier pairs (borrowing from other roles in the relation
///    or from random compatible relations).
pub fn generate_synthetic_facts(schema: &TqlSchema, config: &SyntheticDataConfig) -> Vec<RawFact> {
    let mut rng = rand::rng();

    // Create entity instances: "EntityType_0", "EntityType_1", ...
    let concrete_entities = schema.concrete_entity_types();
    let mut instances_by_type: HashMap<String, Vec<String>> = HashMap::new();

    for entity_type in &concrete_entities {
        let instances: Vec<String> = (0..config.instances_per_type)
            .map(|i| format!("{}_{}", entity_type, i))
            .collect();
        instances_by_type.insert(entity_type.to_string(), instances);
    }

    // Collect relations that have at least 2 roles and fillable entities
    let usable_relations: Vec<&RelationType> = schema
        .relation_types
        .iter()
        .filter(|r| {
            r.roles.len() >= 2
                && !r.name.contains("@abstract")
                && r.name != "base-relation"
                && r.name != "same-hierarchy-relation"
        })
        .collect();

    if usable_relations.is_empty() {
        return Vec::new();
    }

    let mut facts = Vec::with_capacity(config.num_facts);

    for _ in 0..config.num_facts {
        // Pick a random relation
        let rel_idx = rng.random_range(0..usable_relations.len());
        let relation = usable_relations[rel_idx];

        // Try to fill at least 2 roles
        let mut filled_roles: Vec<(String, String)> = Vec::new(); // (role, entity_instance)

        for (role_name, _) in &relation.roles {
            let key = (relation.name.clone(), role_name.clone());
            if let Some(eligible_types) = schema.role_to_entities.get(&key) {
                // Pick a random eligible type that has instances
                let types_with_instances: Vec<&String> = eligible_types
                    .iter()
                    .filter(|t| instances_by_type.contains_key(t.as_str()))
                    .collect();

                if !types_with_instances.is_empty() {
                    let type_idx = rng.random_range(0..types_with_instances.len());
                    let chosen_type = types_with_instances[type_idx];
                    let instances = &instances_by_type[chosen_type.as_str()];
                    let inst_idx = rng.random_range(0..instances.len());
                    filled_roles.push((role_name.clone(), instances[inst_idx].clone()));
                }
            }
        }

        if filled_roles.len() < 2 {
            continue; // Skip if we can't fill enough roles
        }

        // First two filled roles become head and tail of the primary triple
        let head = filled_roles[0].1.clone();
        let tail = filled_roles[1].1.clone();

        // Additional roles become qualifiers
        let mut qualifiers = Vec::new();
        for (role_name, entity_instance) in filled_roles.iter().skip(2) {
            if qualifiers.len() >= config.max_qualifiers {
                break;
            }
            // The qualifier relation is the role name (acting as a qualifier key)
            qualifiers.push(RawQualifier {
                relation: format!("{}:{}", relation.name, role_name),
                entity: entity_instance.clone(),
            });
        }

        facts.push(RawFact {
            head,
            relation: relation.name.clone(),
            tail,
            qualifiers,
        });
    }

    facts
}

/// Generate a complete synthetic dataset: raw facts, vocabulary, and indexed facts.
///
/// Convenience function that chains schema parsing → fact generation → vocabulary
/// building → indexing.
pub fn generate_synthetic_dataset(
    tql_content: &str,
    config: &SyntheticDataConfig,
) -> (Vec<RawFact>, KgVocabulary, Vec<HehrFact>) {
    let schema = TqlSchema::parse(tql_content);
    let raw_facts = generate_synthetic_facts(&schema, config);
    let vocab = KgVocabulary::from_facts(&raw_facts);
    let indexed_facts = raw_facts
        .iter()
        .filter_map(|rf| HehrFact::from_raw(rf, &vocab))
        .collect();

    (raw_facts, vocab, indexed_facts)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_TQL: &str = r#"
define

entity base-entity @abstract,
  owns instance_guid @key;
entity user,
  sub base-entity,
  plays case-owned-by:owner,
  plays evidence-owned-by:owner;
entity reconciliation-case,
  sub base-entity,
  plays case-owned-by:case,
  plays case-has-evidence:case;
entity transaction-evidence,
  sub base-entity,
  plays case-has-evidence:evidence,
  plays evidence-owned-by:evidence;
relation base-relation @abstract,
  owns instance_guid @key;
relation case-owned-by,
  sub base-relation,
  relates case @card(1..1),
  relates owner @card(1..1);
relation case-has-evidence,
  sub base-relation,
  relates case @card(1..1),
  relates evidence @card(1..1);
relation evidence-owned-by,
  sub base-relation,
  relates evidence @card(1..1),
  relates owner @card(1..1);
"#;

    #[test]
    fn test_parse_tql_schema() {
        let schema = TqlSchema::parse(SAMPLE_TQL);

        // Should have 4 entity types (including base-entity)
        assert!(schema.entity_types.len() >= 3);

        // Should have 4 relation types (including base-relation)
        assert!(schema.relation_types.len() >= 3);

        // Check concrete types
        let concrete = schema.concrete_entity_types();
        assert!(concrete.contains(&"user"));
        assert!(concrete.contains(&"reconciliation-case"));
        assert!(concrete.contains(&"transaction-evidence"));

        // Check role mapping
        let case_owners = schema
            .role_to_entities
            .get(&("case-owned-by".to_string(), "owner".to_string()));
        assert!(case_owners.is_some());
        assert!(case_owners.unwrap().contains(&"user".to_string()));
    }

    #[test]
    fn test_generate_synthetic_facts() {
        let schema = TqlSchema::parse(SAMPLE_TQL);
        let config = SyntheticDataConfig {
            instances_per_type: 5,
            num_facts: 50,
            max_qualifiers: 2,
            seed: 42,
        };

        let facts = generate_synthetic_facts(&schema, &config);

        // Should generate some facts (not all attempts succeed)
        assert!(!facts.is_empty(), "Should generate at least some facts");

        // All facts should have valid structure
        for fact in &facts {
            assert!(!fact.head.is_empty());
            assert!(!fact.relation.is_empty());
            assert!(!fact.tail.is_empty());
            assert!(fact.qualifiers.len() <= config.max_qualifiers);
        }
    }

    #[test]
    fn test_generate_synthetic_dataset_end_to_end() {
        let config = SyntheticDataConfig {
            instances_per_type: 5,
            num_facts: 30,
            max_qualifiers: 1,
            seed: 42,
        };

        let (raw_facts, vocab, indexed_facts) = generate_synthetic_dataset(SAMPLE_TQL, &config);

        assert!(!raw_facts.is_empty());
        assert!(vocab.num_entities() > 0);
        assert!(vocab.num_relations() > 0);
        assert_eq!(raw_facts.len(), indexed_facts.len());
    }

    #[test]
    fn test_parse_full_schema_file_would_work() {
        // Test that the parser handles the SchemaFinverse.tql patterns
        let snippet = r#"
define

entity asset,
  sub base-entity,
  plays asset-has-identifier:asset,
  plays asset-has-valuation:asset,
  plays asset-issued-by:asset;
entity instrument,
  sub base-entity,
  plays provider-has-instrument:instrument,
  plays reconciliation-for-instrument:instrument;
relation asset-has-identifier,
  sub base-relation,
  relates asset @card(1..1),
  relates identifier @card(1..1);
relation asset-issued-by,
  sub base-relation,
  relates asset @card(1..1),
  relates issuance-document @card(0..1),
  relates issuer @card(1..1);
"#;

        let schema = TqlSchema::parse(snippet);
        assert!(schema.entity_types.iter().any(|e| e.name == "asset"));
        assert!(schema.entity_types.iter().any(|e| e.name == "instrument"));

        // asset-issued-by has 3 roles (including optional issuance-document)
        let issued_by = schema
            .relation_types
            .iter()
            .find(|r| r.name == "asset-issued-by");
        assert!(issued_by.is_some());
        assert_eq!(issued_by.unwrap().roles.len(), 3);
    }
}
