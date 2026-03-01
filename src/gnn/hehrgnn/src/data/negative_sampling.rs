//! Negative sampling for knowledge graph link prediction training.
//!
//! Generates corrupted (negative) facts by randomly replacing the head, tail,
//! or a qualifier entity with a random entity from the vocabulary.

use rand::Rng;

use super::fact::{HehrFact, Qualifier};

/// Strategy for which part of the fact to corrupt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorruptionTarget {
    Head,
    Tail,
    /// Corrupt a specific qualifier entity by index.
    QualifierEntity(usize),
}

/// Negative sampler for HEHR facts.
#[derive(Debug, Clone)]
pub struct NegativeSampler {
    /// Total number of entities in the vocabulary.
    pub num_entities: usize,
    /// Number of negative samples to generate per positive fact.
    pub num_negatives: usize,
}

impl NegativeSampler {
    pub fn new(num_entities: usize, num_negatives: usize) -> Self {
        Self {
            num_entities,
            num_negatives,
        }
    }

    /// Generate negative samples for a single positive fact.
    ///
    /// For each negative, randomly picks a corruption target (head, tail, or
    /// qualifier entity) and replaces it with a random entity different from
    /// the original.
    pub fn sample(&self, fact: &HehrFact) -> Vec<HehrFact> {
        let mut rng = rand::rng();
        let mut negatives = Vec::with_capacity(self.num_negatives);

        // Build list of valid corruption targets
        let mut targets = vec![CorruptionTarget::Head, CorruptionTarget::Tail];
        for i in 0..fact.qualifiers.len() {
            targets.push(CorruptionTarget::QualifierEntity(i));
        }

        for _ in 0..self.num_negatives {
            let target_idx = rng.random_range(0..targets.len());
            let target = targets[target_idx];
            let corrupted = self.corrupt(fact, target, &mut rng);
            negatives.push(corrupted);
        }

        negatives
    }

    /// Corrupt a fact at the given target position.
    fn corrupt(&self, fact: &HehrFact, target: CorruptionTarget, rng: &mut impl Rng) -> HehrFact {
        let mut corrupted = fact.clone();

        match target {
            CorruptionTarget::Head => {
                corrupted.head = self.random_entity_excluding(fact.head, rng);
            }
            CorruptionTarget::Tail => {
                corrupted.tail = self.random_entity_excluding(fact.tail, rng);
            }
            CorruptionTarget::QualifierEntity(idx) => {
                if idx < corrupted.qualifiers.len() {
                    let original = corrupted.qualifiers[idx].entity_id;
                    corrupted.qualifiers[idx] = Qualifier {
                        relation_id: corrupted.qualifiers[idx].relation_id,
                        entity_id: self.random_entity_excluding(original, rng),
                    };
                }
            }
        }

        corrupted
    }

    /// Pick a random entity ID that is different from `exclude`.
    fn random_entity_excluding(&self, exclude: usize, rng: &mut impl Rng) -> usize {
        loop {
            let candidate = rng.random_range(0..self.num_entities);
            if candidate != exclude {
                return candidate;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::fact::Qualifier;

    #[test]
    fn test_negative_sampler_produces_different_facts() {
        let fact = HehrFact {
            head: 0,
            relation: 0,
            tail: 1,
            qualifiers: vec![Qualifier {
                relation_id: 1,
                entity_id: 2,
            }],
        };

        let sampler = NegativeSampler::new(100, 5);
        let negatives = sampler.sample(&fact);

        assert_eq!(negatives.len(), 5);
        // Each negative should differ from the original in at least one entity position
        for neg in &negatives {
            let differs = neg.head != fact.head
                || neg.tail != fact.tail
                || neg
                    .qualifiers
                    .iter()
                    .zip(fact.qualifiers.iter())
                    .any(|(a, b)| a.entity_id != b.entity_id);
            assert!(differs, "Negative sample must differ from original");
        }
    }

    #[test]
    fn test_negative_sampler_no_qualifiers() {
        let fact = HehrFact {
            head: 0,
            relation: 0,
            tail: 1,
            qualifiers: vec![],
        };

        let sampler = NegativeSampler::new(50, 3);
        let negatives = sampler.sample(&fact);

        assert_eq!(negatives.len(), 3);
        for neg in &negatives {
            // With no qualifiers, only head or tail can be corrupted
            assert!(neg.head != fact.head || neg.tail != fact.tail);
        }
    }
}
