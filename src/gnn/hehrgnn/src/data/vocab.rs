//! Vocabulary mapping for entities and relations.
//!
//! Provides bidirectional string ↔ integer ID mappings used to convert
//! raw string-based knowledge graph facts into integer-indexed tensors.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::fact::RawFact;

/// Bidirectional vocabulary: string ↔ integer ID.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocabulary {
    /// String → ID mapping.
    pub str_to_id: HashMap<String, usize>,
    /// ID → String mapping (reverse lookup).
    pub id_to_str: Vec<String>,
}

impl Vocabulary {
    /// Create an empty vocabulary.
    pub fn new() -> Self {
        Self {
            str_to_id: HashMap::new(),
            id_to_str: Vec::new(),
        }
    }

    /// Insert a token and return its ID. If already present, returns existing ID.
    pub fn get_or_insert(&mut self, token: &str) -> usize {
        if let Some(&id) = self.str_to_id.get(token) {
            return id;
        }
        let id = self.id_to_str.len();
        self.id_to_str.push(token.to_string());
        self.str_to_id.insert(token.to_string(), id);
        id
    }

    /// Look up the ID for a token, returning `None` if not present.
    pub fn get_id(&self, token: &str) -> Option<usize> {
        self.str_to_id.get(token).copied()
    }

    /// Look up the token for an ID, returning `None` if out of range.
    pub fn get_token(&self, id: usize) -> Option<&str> {
        self.id_to_str.get(id).map(|s| s.as_str())
    }

    /// Number of unique tokens in the vocabulary.
    pub fn len(&self) -> usize {
        self.id_to_str.len()
    }

    /// Whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.id_to_str.is_empty()
    }
}

impl Default for Vocabulary {
    fn default() -> Self {
        Self::new()
    }
}

/// Entity and relation vocabularies built from a set of raw facts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KgVocabulary {
    pub entities: Vocabulary,
    pub relations: Vocabulary,
}

impl KgVocabulary {
    /// Build entity and relation vocabularies from raw facts.
    ///
    /// Scans all heads, tails, qualifier entities (into entity vocab) and
    /// all primary relations + qualifier relations (into relation vocab).
    pub fn from_facts(facts: &[RawFact]) -> Self {
        let mut entities = Vocabulary::new();
        let mut relations = Vocabulary::new();

        for fact in facts {
            entities.get_or_insert(&fact.head);
            entities.get_or_insert(&fact.tail);
            relations.get_or_insert(&fact.relation);

            for qual in &fact.qualifiers {
                relations.get_or_insert(&qual.relation);
                entities.get_or_insert(&qual.entity);
            }
        }

        Self {
            entities,
            relations,
        }
    }

    /// Number of unique entities.
    pub fn num_entities(&self) -> usize {
        self.entities.len()
    }

    /// Number of unique relations.
    pub fn num_relations(&self) -> usize {
        self.relations.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::fact::RawQualifier;

    #[test]
    fn test_vocabulary_basic() {
        let mut vocab = Vocabulary::new();
        let id_a = vocab.get_or_insert("Alice");
        let id_b = vocab.get_or_insert("Bob");
        let id_a2 = vocab.get_or_insert("Alice");

        assert_eq!(id_a, 0);
        assert_eq!(id_b, 1);
        assert_eq!(id_a, id_a2); // same token → same ID
        assert_eq!(vocab.len(), 2);
        assert_eq!(vocab.get_token(0), Some("Alice"));
        assert_eq!(vocab.get_id("Bob"), Some(1));
        assert_eq!(vocab.get_id("Unknown"), None);
    }

    #[test]
    fn test_kg_vocabulary_from_facts() {
        let facts = vec![
            RawFact {
                head: "Q42".into(),
                relation: "P31".into(),
                tail: "Q5".into(),
                qualifiers: vec![RawQualifier {
                    relation: "P580".into(),
                    entity: "1952".into(),
                }],
            },
            RawFact {
                head: "Q42".into(),
                relation: "P27".into(),
                tail: "Q145".into(),
                qualifiers: vec![],
            },
        ];

        let vocab = KgVocabulary::from_facts(&facts);

        // Entities: Q42, Q5, 1952, Q145 = 4 unique
        assert_eq!(vocab.num_entities(), 4);
        // Relations: P31, P580, P27 = 3 unique
        assert_eq!(vocab.num_relations(), 3);
        // Verify specific lookups
        assert!(vocab.entities.get_id("Q42").is_some());
        assert!(vocab.relations.get_id("P580").is_some());
    }
}
