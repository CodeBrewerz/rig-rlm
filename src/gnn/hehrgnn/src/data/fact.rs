//! HEHR (HyperEdge Hyper-Relational) fact representation.
//!
//! Defines both raw (string-based) and indexed (ID-based) fact structures
//! for representing n-ary knowledge graph facts with qualifiers.

use serde::{Deserialize, Serialize};

use super::vocab::KgVocabulary;

// ---------------------------------------------------------------------------
// Raw (string-based) fact types — used for parsing dataset files
// ---------------------------------------------------------------------------

/// A qualifier key-value pair in string form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawQualifier {
    pub relation: String,
    pub entity: String,
}

/// A raw fact from a dataset file, using string identifiers.
///
/// Represents a primary triple (head, relation, tail) with zero or more
/// qualifier pairs that modify / contextualize the primary fact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawFact {
    pub head: String,
    pub relation: String,
    pub tail: String,
    #[serde(default)]
    pub qualifiers: Vec<RawQualifier>,
}

// ---------------------------------------------------------------------------
// Indexed (ID-based) fact types — used for tensor creation
// ---------------------------------------------------------------------------

/// A qualifier pair using integer IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qualifier {
    pub relation_id: usize,
    pub entity_id: usize,
}

/// An indexed knowledge graph fact ready for embedding.
///
/// The primary triple `(head, relation, tail)` plus qualifier pairs are all
/// stored as integer IDs referencing into the [`KgVocabulary`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HehrFact {
    pub head: usize,
    pub relation: usize,
    pub tail: usize,
    pub qualifiers: Vec<Qualifier>,
}

impl HehrFact {
    /// Convert a [`RawFact`] into an indexed [`HehrFact`] using the given vocabulary.
    ///
    /// Returns `None` if any token in the raw fact is missing from the vocabulary.
    pub fn from_raw(raw: &RawFact, vocab: &KgVocabulary) -> Option<Self> {
        let head = vocab.entities.get_id(&raw.head)?;
        let relation = vocab.relations.get_id(&raw.relation)?;
        let tail = vocab.entities.get_id(&raw.tail)?;

        let mut qualifiers = Vec::with_capacity(raw.qualifiers.len());
        for q in &raw.qualifiers {
            qualifiers.push(Qualifier {
                relation_id: vocab.relations.get_id(&q.relation)?,
                entity_id: vocab.entities.get_id(&q.entity)?,
            });
        }

        Some(Self {
            head,
            relation,
            tail,
            qualifiers,
        })
    }

    /// All entity IDs participating in this fact (head + tail + qualifier entities).
    pub fn all_entity_ids(&self) -> Vec<usize> {
        let mut ids = vec![self.head, self.tail];
        for q in &self.qualifiers {
            ids.push(q.entity_id);
        }
        ids
    }

    /// All relation IDs in this fact (primary + qualifier relations).
    pub fn all_relation_ids(&self) -> Vec<usize> {
        let mut ids = vec![self.relation];
        for q in &self.qualifiers {
            ids.push(q.relation_id);
        }
        ids
    }
}

/// Parse a JSON array of [`RawFact`]s from a string.
pub fn parse_raw_facts_json(json_str: &str) -> Result<Vec<RawFact>, serde_json::Error> {
    serde_json::from_str(json_str)
}

/// Convert a slice of [`RawFact`]s into indexed [`HehrFact`]s.
///
/// Facts whose tokens are missing from the vocabulary are silently skipped.
pub fn index_facts(raw_facts: &[RawFact], vocab: &KgVocabulary) -> Vec<HehrFact> {
    raw_facts
        .iter()
        .filter_map(|rf| HehrFact::from_raw(rf, vocab))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_raw_facts() -> Vec<RawFact> {
        vec![
            RawFact {
                head: "Q42".into(),
                relation: "P31".into(),
                tail: "Q5".into(),
                qualifiers: vec![
                    RawQualifier {
                        relation: "P580".into(),
                        entity: "1952".into(),
                    },
                    RawQualifier {
                        relation: "P582".into(),
                        entity: "2001".into(),
                    },
                ],
            },
            RawFact {
                head: "Q42".into(),
                relation: "P27".into(),
                tail: "Q145".into(),
                qualifiers: vec![],
            },
        ]
    }

    #[test]
    fn test_parse_json() {
        let json = r#"[
            {"head":"A","relation":"R","tail":"B","qualifiers":[{"relation":"QR","entity":"QE"}]},
            {"head":"C","relation":"R2","tail":"D"}
        ]"#;
        let facts = parse_raw_facts_json(json).unwrap();
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].qualifiers.len(), 1);
        assert_eq!(facts[1].qualifiers.len(), 0);
    }

    #[test]
    fn test_hehr_fact_from_raw() {
        let raw_facts = sample_raw_facts();
        let vocab = KgVocabulary::from_facts(&raw_facts);

        let indexed = index_facts(&raw_facts, &vocab);
        assert_eq!(indexed.len(), 2);

        // First fact should have 2 qualifiers
        assert_eq!(indexed[0].qualifiers.len(), 2);
        // Head should be the same across both facts (both "Q42")
        assert_eq!(indexed[0].head, indexed[1].head);
    }

    #[test]
    fn test_all_entity_ids() {
        let fact = HehrFact {
            head: 0,
            relation: 0,
            tail: 1,
            qualifiers: vec![
                Qualifier {
                    relation_id: 1,
                    entity_id: 2,
                },
                Qualifier {
                    relation_id: 2,
                    entity_id: 3,
                },
            ],
        };
        assert_eq!(fact.all_entity_ids(), vec![0, 1, 2, 3]);
        assert_eq!(fact.all_relation_ids(), vec![0, 1, 2]);
    }
}
