//! Inverted keyword index + bi-directional link graph for O(1) fact lookup.
//!
//! Replaces the O(N) fuzzy string scan in `resolve_tag()` with:
//!   1. **Exact key HashMap** — O(1)
//!   2. **Inverted token index** — tokenize query, intersect posting lists, rank by overlap
//!   3. **Link graph** — bi-directional edges between facts sharing entities
//!   4. **Fuzzy fallback** — only if steps 1-3 produce no matches

use std::collections::{HashMap, HashSet, BTreeSet};

/// Tokenize a key into searchable tokens.
/// "Alice:worksAt" → ["alice", "worksat", "alice:worksat"]
/// "Google:ceo" → ["google", "ceo", "google:ceo"]
/// "Mountain View:state" → ["mountain", "view", "state", "mountain view:state"]
fn tokenize(key: &str) -> Vec<String> {
    let lower = key.to_lowercase();
    let lower = lower.trim();
    let mut tokens = Vec::new();

    // Full key (normalized)
    tokens.push(lower.to_string());

    // Split on delimiters: ':', ' ', '_', '-', '.'
    for part in lower.split(|c: char| c == ':' || c == ' ' || c == '_' || c == '-' || c == '.') {
        let t = part.trim();
        if !t.is_empty() && t.len() >= 2 {
            tokens.push(t.to_string());
        }
    }

    tokens.sort();
    tokens.dedup();
    tokens
}

/// A bi-directional link between two facts.
#[derive(Debug, Clone)]
struct FactLink {
    /// The other fact's index.
    target: usize,
    /// Shared tokens that caused the link.
    shared_tokens: Vec<String>,
}

/// Inverted keyword index + link graph.
#[derive(Debug, Clone)]
pub struct KeyIndex {
    /// Exact key → fact index (case-insensitive).
    exact: HashMap<String, usize>,

    /// Token → set of fact indices that contain this token.
    inverted: HashMap<String, BTreeSet<usize>>,

    /// Per-fact adjacency list: fact_index → list of linked facts.
    links: Vec<Vec<FactLink>>,

    /// Per-fact token sets (for link computation).
    fact_tokens: Vec<Vec<String>>,

    /// Per-fact value tokens (for entity-based linking).
    value_tokens: Vec<Vec<String>>,
}

impl KeyIndex {
    pub fn new() -> Self {
        Self {
            exact: HashMap::new(),
            inverted: HashMap::new(),
            links: Vec::new(),
            fact_tokens: Vec::new(),
            value_tokens: Vec::new(),
        }
    }

    /// Build the entire index from a facts list.
    pub fn rebuild(&mut self, facts: &[(String, String)]) {
        self.exact.clear();
        self.inverted.clear();
        self.links.clear();
        self.fact_tokens.clear();
        self.value_tokens.clear();

        let n = facts.len();
        self.links.resize(n, Vec::new());
        self.fact_tokens.reserve(n);
        self.value_tokens.reserve(n);

        for (i, (key, value)) in facts.iter().enumerate() {
            let lower_key = key.to_lowercase();
            self.exact.insert(lower_key, i);

            let ktokens = tokenize(key);
            for t in &ktokens {
                self.inverted.entry(t.clone()).or_default().insert(i);
            }
            self.fact_tokens.push(ktokens);

            // Tokenize value for entity linking
            let vtokens = tokenize(value);
            for t in &vtokens {
                self.inverted.entry(t.clone()).or_default().insert(i);
            }
            self.value_tokens.push(vtokens);
        }

        // Build bi-directional links: facts sharing tokens are linked
        self.build_links(n);
    }

    /// Incrementally add a single fact (avoids full rebuild).
    pub fn add_fact(&mut self, idx: usize, key: &str, value: &str) {
        let lower_key = key.to_lowercase();
        self.exact.insert(lower_key, idx);

        let ktokens = tokenize(key);
        for t in &ktokens {
            self.inverted.entry(t.clone()).or_default().insert(idx);
        }

        let vtokens = tokenize(value);
        for t in &vtokens {
            self.inverted.entry(t.clone()).or_default().insert(idx);
        }

        // Extend links/fact_tokens arrays
        while self.links.len() <= idx { self.links.push(Vec::new()); }
        while self.fact_tokens.len() <= idx { self.fact_tokens.push(Vec::new()); }
        while self.value_tokens.len() <= idx { self.value_tokens.push(Vec::new()); }

        // Find neighbors: other facts sharing any token with this new fact
        let all_tokens: HashSet<&str> = ktokens.iter().chain(vtokens.iter())
            .map(|s| s.as_str()).collect();

        let mut neighbors: HashMap<usize, Vec<String>> = HashMap::new();
        for token in &all_tokens {
            if let Some(posting) = self.inverted.get(*token) {
                for &other_idx in posting {
                    if other_idx != idx {
                        neighbors.entry(other_idx)
                            .or_default()
                            .push(token.to_string());
                    }
                }
            }
        }

        // Create bi-directional links
        for (other_idx, shared) in neighbors {
            let deduped: Vec<String> = {
                let mut s = shared;
                s.sort();
                s.dedup();
                s
            };

            // Link: this → other
            self.links[idx].push(FactLink {
                target: other_idx,
                shared_tokens: deduped.clone(),
            });

            // Link: other → this
            if other_idx < self.links.len() {
                self.links[other_idx].push(FactLink {
                    target: idx,
                    shared_tokens: deduped,
                });
            }
        }

        self.fact_tokens[idx] = ktokens;
        self.value_tokens[idx] = vtokens;
    }

    /// Update a fact's value (key stays the same).
    pub fn update_value(&mut self, idx: usize, _key: &str, new_value: &str) {
        // Remove old value tokens from inverted index
        if idx < self.value_tokens.len() {
            let old_vtokens = std::mem::take(&mut self.value_tokens[idx]);
            for t in &old_vtokens {
                if let Some(set) = self.inverted.get_mut(t) {
                    set.remove(&idx);
                    if set.is_empty() {
                        self.inverted.remove(t);
                    }
                }
            }

            // Remove old links involving this fact
            let old_targets: Vec<usize> = self.links[idx].iter().map(|l| l.target).collect();
            self.links[idx].clear();
            for target in old_targets {
                if target < self.links.len() {
                    self.links[target].retain(|l| l.target != idx);
                }
            }
        }

        // Add new value tokens
        let vtokens = tokenize(new_value);
        for t in &vtokens {
            self.inverted.entry(t.clone()).or_default().insert(idx);
        }

        if idx < self.value_tokens.len() {
            self.value_tokens[idx] = vtokens.clone();
        }

        // Rebuild links for this fact
        let all_tokens: HashSet<&str> = self.fact_tokens.get(idx)
            .into_iter().flat_map(|v| v.iter())
            .chain(vtokens.iter())
            .map(|s| s.as_str())
            .collect();

        let mut neighbors: HashMap<usize, Vec<String>> = HashMap::new();
        for token in &all_tokens {
            if let Some(posting) = self.inverted.get(*token) {
                for &other_idx in posting {
                    if other_idx != idx {
                        neighbors.entry(other_idx)
                            .or_default()
                            .push(token.to_string());
                    }
                }
            }
        }

        for (other_idx, shared) in neighbors {
            let mut deduped = shared;
            deduped.sort();
            deduped.dedup();
            self.links[idx].push(FactLink {
                target: other_idx,
                shared_tokens: deduped.clone(),
            });
            if other_idx < self.links.len() {
                // Avoid duplicates in other's link list
                self.links[other_idx].retain(|l| l.target != idx);
                self.links[other_idx].push(FactLink {
                    target: idx,
                    shared_tokens: deduped,
                });
            }
        }
    }

    /// Remove a fact from the index.
    pub fn remove_fact(&mut self, idx: usize) {
        // Remove from exact index
        self.exact.retain(|_, &mut v| v != idx);

        // Remove from inverted index
        if idx < self.fact_tokens.len() {
            for t in &self.fact_tokens[idx] {
                if let Some(set) = self.inverted.get_mut(t) {
                    set.remove(&idx);
                }
            }
        }
        if idx < self.value_tokens.len() {
            for t in &self.value_tokens[idx] {
                if let Some(set) = self.inverted.get_mut(t) {
                    set.remove(&idx);
                }
            }
        }

        // Remove links
        if idx < self.links.len() {
            let targets: Vec<usize> = self.links[idx].iter().map(|l| l.target).collect();
            self.links[idx].clear();
            for target in targets {
                if target < self.links.len() {
                    self.links[target].retain(|l| l.target != idx);
                }
            }
        }
    }

    /// Resolve a query to a fact key. Returns `Some(key)` or `None`.
    ///
    /// Resolution order:
    ///   1. Exact match (O(1) HashMap lookup)
    ///   2. Token overlap ranking (inverted index)
    ///   3. None (caller falls back to fuzzy if desired)
    pub fn resolve(&self, query: &str, fact_keys: &[String]) -> Option<String> {
        if fact_keys.is_empty() { return None; }

        let lower = query.to_lowercase();
        let lower = lower.trim();

        // 1. Exact match — O(1)
        if let Some(&idx) = self.exact.get(lower) {
            if idx < fact_keys.len() {
                return Some(fact_keys[idx].clone());
            }
        }

        // 2. Token-overlap ranking via inverted index — O(K) where K = query tokens
        let query_tokens = tokenize(query);
        if query_tokens.is_empty() { return None; }

        // Collect candidate facts and their overlap scores
        let mut scores: HashMap<usize, f64> = HashMap::new();
        for qt in &query_tokens {
            if let Some(posting) = self.inverted.get(qt) {
                for &fact_idx in posting {
                    if fact_idx < fact_keys.len() {
                        // Weight: longer token matches are more valuable
                        let weight = qt.len() as f64;
                        *scores.entry(fact_idx).or_default() += weight;
                    }
                }
            }
        }

        if scores.is_empty() { return None; }

        // Normalize by total query token length
        let query_total: f64 = query_tokens.iter().map(|t| t.len() as f64).sum();

        // Find best candidate
        let (best_idx, best_score) = scores.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(&idx, &score)| (idx, score / query_total))
            .unwrap();

        // Only return if there's reasonable overlap (at least 50% of query tokens matched)
        if best_score >= 0.4 {
            return Some(fact_keys[best_idx].clone());
        }

        None
    }

    /// Get linked facts for a given fact index.
    /// Returns vec of (linked_fact_index, shared_token_count).
    pub fn get_links(&self, fact_idx: usize) -> Vec<(usize, usize)> {
        if fact_idx >= self.links.len() { return Vec::new(); }
        self.links[fact_idx].iter()
            .map(|l| (l.target, l.shared_tokens.len()))
            .collect()
    }

    /// Get related facts for a query (via link graph traversal).
    /// First resolves the query, then returns linked facts sorted by relevance.
    pub fn related(&self, query: &str, fact_keys: &[String], max_results: usize) -> Vec<usize> {
        let lower = query.to_lowercase();

        // Find the anchor fact
        let anchor = match self.exact.get(lower.trim()) {
            Some(&idx) => idx,
            None => return Vec::new(),
        };

        // Collect 1-hop neighbors sorted by shared token count
        let mut neighbors: Vec<(usize, usize)> = self.get_links(anchor);
        neighbors.sort_by(|a, b| b.1.cmp(&a.1));
        neighbors.truncate(max_results);
        neighbors.iter().map(|&(idx, _)| idx).collect()
    }

    /// Estimated RAM usage in bytes.
    pub fn ram_bytes(&self) -> usize {
        let exact_bytes = self.exact.len() * (32 + 8); // key String + usize
        let inverted_bytes: usize = self.inverted.iter()
            .map(|(k, v)| k.len() + 24 + v.len() * 8) // key + BTreeSet overhead + entries
            .sum();
        let links_bytes: usize = self.links.iter()
            .map(|v| v.len() * 48)  // FactLink: usize + Vec<String>
            .sum();
        let token_bytes: usize = self.fact_tokens.iter().chain(self.value_tokens.iter())
            .map(|v| v.iter().map(|s| s.len() + 24).sum::<usize>())
            .sum();
        exact_bytes + inverted_bytes + links_bytes + token_bytes
    }

    // -- private --------------------------------------------------------------

    fn build_links(&mut self, n: usize) {
        // For each pair of facts, check if they share tokens
        // Optimization: iterate inverted index instead of O(N²) pairs
        let mut edge_map: HashMap<(usize, usize), Vec<String>> = HashMap::new();

        for (token, posting) in &self.inverted {
            let indices: Vec<usize> = posting.iter().copied().collect();
            for i in 0..indices.len() {
                for j in (i+1)..indices.len() {
                    let (a, b) = (indices[i], indices[j]);
                    let key = (a.min(b), a.max(b));
                    edge_map.entry(key).or_default().push(token.clone());
                }
            }
        }

        // Create bi-directional links (limit to top-10 per fact to avoid explosion)
        for ((a, b), tokens) in &edge_map {
            let mut deduped = tokens.clone();
            deduped.sort();
            deduped.dedup();
            let shared_count = deduped.len();

            if shared_count >= 1 && *a < n && *b < n {
                self.links[*a].push(FactLink {
                    target: *b,
                    shared_tokens: deduped.clone(),
                });
                self.links[*b].push(FactLink {
                    target: *a,
                    shared_tokens: deduped,
                });
            }
        }

        // Sort each adjacency list by shared token count (descending) and limit
        for adj in &mut self.links {
            adj.sort_by(|a, b| b.shared_tokens.len().cmp(&a.shared_tokens.len()));
            adj.truncate(20); // Keep top-20 links per fact
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Alice:worksAt");
        assert!(tokens.contains(&"alice".to_string()));
        assert!(tokens.contains(&"worksat".to_string()));
        assert!(tokens.contains(&"alice:worksat".to_string()));
    }

    #[test]
    fn test_exact_lookup() {
        let mut idx = KeyIndex::new();
        let facts = vec![
            ("Alice:worksAt".into(), "Google".into()),
            ("Bob:worksAt".into(), "Meta".into()),
        ];
        idx.rebuild(&facts);
        let keys: Vec<String> = facts.iter().map(|(k, _)| k.clone()).collect();

        assert_eq!(idx.resolve("Alice:worksAt", &keys), Some("Alice:worksAt".into()));
        assert_eq!(idx.resolve("alice:worksat", &keys), Some("Alice:worksAt".into()));
        assert_eq!(idx.resolve("Bob:worksAt", &keys), Some("Bob:worksAt".into()));
    }

    #[test]
    fn test_token_overlap_lookup() {
        let mut idx = KeyIndex::new();
        let facts = vec![
            ("Alice:worksAt".into(), "Google".into()),
            ("Alice:age".into(), "32".into()),
            ("Bob:worksAt".into(), "Meta".into()),
        ];
        idx.rebuild(&facts);
        let keys: Vec<String> = facts.iter().map(|(k, _)| k.clone()).collect();

        // "Alice" should match Alice facts
        let result = idx.resolve("Alice", &keys);
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.starts_with("Alice:"));
    }

    #[test]
    fn test_link_graph() {
        let mut idx = KeyIndex::new();
        let facts = vec![
            ("Alice:worksAt".into(), "Google".into()),     // 0
            ("Google:ceo".into(), "Sundar Pichai".into()),  // 1
            ("Bob:worksAt".into(), "Meta".into()),          // 2
        ];
        idx.rebuild(&facts);

        // Fact 0 (Alice:worksAt → Google) and Fact 1 (Google:ceo) share "google"
        let links_0 = idx.get_links(0);
        assert!(links_0.iter().any(|&(target, _)| target == 1),
            "Alice:worksAt should link to Google:ceo via shared 'google' token");

        // Fact 1 should link back to Fact 0
        let links_1 = idx.get_links(1);
        assert!(links_1.iter().any(|&(target, _)| target == 0),
            "Google:ceo should link back to Alice:worksAt");

        // Bob and Google shouldn't be strongly linked
        let links_2 = idx.get_links(2);
        assert!(!links_2.iter().any(|&(target, _)| target == 1),
            "Bob:worksAt should not link to Google:ceo");
    }

    #[test]
    fn test_incremental_add() {
        let mut idx = KeyIndex::new();
        idx.add_fact(0, "Alice:worksAt", "Google");
        idx.add_fact(1, "Google:ceo", "Sundar Pichai");

        let keys = vec!["Alice:worksAt".into(), "Google:ceo".into()];
        assert_eq!(idx.resolve("Alice:worksAt", &keys), Some("Alice:worksAt".into()));
        assert_eq!(idx.resolve("Google:ceo", &keys), Some("Google:ceo".into()));

        // Should be linked via "google"
        let links = idx.get_links(0);
        assert!(links.iter().any(|&(t, _)| t == 1));
    }

    #[test]
    fn test_ram_is_small() {
        let mut idx = KeyIndex::new();
        let facts: Vec<(String, String)> = (0..1000)
            .map(|i| (format!("key_{i:04}:pred"), format!("value_{i:04}")))
            .collect();
        idx.rebuild(&facts);
        let ram = idx.ram_bytes();
        println!("KeyIndex RAM for 1000 facts: {} bytes ({:.1} KB)", ram, ram as f64 / 1024.0);
        assert!(ram < 2_000_000, "KeyIndex should use < 2MB for 1000 facts");
    }
}
