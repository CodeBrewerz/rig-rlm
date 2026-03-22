//! Topology analysis for the channel subscription graph.
//!
//! The subscription system forms a **bipartite graph** (Topics ↔ Subscribers).
//! Projecting onto subscribers gives a **co-subscription graph** where edges
//! connect subscribers who share at least one matching topic. **Cliques** in
//! this graph reveal groups of agents sharing the same information space.
//!
//! ## Reactive Clique Detection
//!
//! Cliques are recomputed automatically when subscriptions change (subscribe
//! or unsubscribe). Call `find_cliques()` for on-demand introspection.
//!
//! ## Bron-Kerbosch Algorithm
//!
//! Used for maximal clique enumeration on the co-subscription adjacency graph.
//! O(3^(n/3)) worst case but subscription graphs are typically sparse.

use std::collections::{BTreeSet, HashMap, HashSet};

use chrono::{DateTime, Utc};

use super::hub::SubscriptionId;

// ── Core Types ────────────────────────────────────────────────────

/// A subscription entry: what topic filter a subscriber is watching.
#[derive(Debug, Clone)]
pub struct SubscriptionEntry {
    pub id: SubscriptionId,
    pub filter_pattern: String,
}

/// A maximal clique in the co-subscription graph.
///
/// A clique is a set of subscribers where every pair shares at least
/// one topic filter. This means they're all in the same "information space."
#[derive(Debug, Clone)]
pub struct Clique {
    /// Subscriber IDs forming the clique.
    pub members: Vec<SubscriptionId>,
    /// Topic filters shared by ALL members of the clique.
    pub shared_filters: HashSet<String>,
    /// When the clique was detected.
    pub detected_at: DateTime<Utc>,
}

impl Clique {
    /// Number of members in the clique.
    pub fn size(&self) -> usize {
        self.members.len()
    }

    /// Is this a trivial clique (single member)?
    pub fn is_trivial(&self) -> bool {
        self.members.len() <= 1
    }
}

/// A broadcast optimization group: subscribers with identical filters
/// that can share a single broadcast sender (CoW at the routing level).
#[derive(Debug, Clone)]
pub struct BroadcastGroup {
    /// The common topic filter pattern.
    pub filter_pattern: String,
    /// Subscriber IDs that all use this exact filter.
    pub subscribers: Vec<SubscriptionId>,
}

// ── Subscription Graph ────────────────────────────────────────────

/// Bipartite subscription graph: subscribers ↔ topic filters.
///
/// Built from the hub's active subscriptions. Used for clique detection
/// and broadcast optimization.
#[derive(Debug, Default)]
pub struct SubscriptionGraph {
    /// subscriber → set of topic filter patterns they're watching.
    sub_to_filters: HashMap<SubscriptionId, HashSet<String>>,
    /// filter pattern → set of subscriber IDs watching it.
    filter_to_subs: HashMap<String, HashSet<SubscriptionId>>,
}

impl SubscriptionGraph {
    /// Create an empty subscription graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a subscription entry.
    pub fn add(&mut self, entry: SubscriptionEntry) {
        self.sub_to_filters
            .entry(entry.id)
            .or_default()
            .insert(entry.filter_pattern.clone());
        self.filter_to_subs
            .entry(entry.filter_pattern)
            .or_default()
            .insert(entry.id);
    }

    /// Remove a subscription by ID.
    pub fn remove(&mut self, id: SubscriptionId) {
        if let Some(filters) = self.sub_to_filters.remove(&id) {
            for filter in filters {
                if let Some(subs) = self.filter_to_subs.get_mut(&filter) {
                    subs.remove(&id);
                    if subs.is_empty() {
                        self.filter_to_subs.remove(&filter);
                    }
                }
            }
        }
    }

    /// Number of subscribers in the graph.
    pub fn subscriber_count(&self) -> usize {
        self.sub_to_filters.len()
    }

    /// Number of unique filter patterns.
    pub fn filter_count(&self) -> usize {
        self.filter_to_subs.len()
    }

    /// Get all filter patterns for a subscriber.
    pub fn filters_for(&self, id: SubscriptionId) -> Option<&HashSet<String>> {
        self.sub_to_filters.get(&id)
    }

    /// Get all subscribers watching a filter pattern.
    pub fn subscribers_for(&self, filter: &str) -> Option<&HashSet<SubscriptionId>> {
        self.filter_to_subs.get(filter)
    }

    // ── Co-subscription Adjacency ────────────────────────────────

    /// Project onto subscribers: two subscribers share an edge if they
    /// have at least one topic filter in common.
    ///
    /// Returns adjacency list: subscriber → set of neighbors.
    pub fn co_subscription_adjacency(&self) -> HashMap<SubscriptionId, HashSet<SubscriptionId>> {
        let mut adj: HashMap<SubscriptionId, HashSet<SubscriptionId>> = HashMap::new();

        // For each filter, all its subscribers are pairwise connected
        for subs in self.filter_to_subs.values() {
            let sub_list: Vec<_> = subs.iter().copied().collect();
            for i in 0..sub_list.len() {
                for j in (i + 1)..sub_list.len() {
                    adj.entry(sub_list[i]).or_default().insert(sub_list[j]);
                    adj.entry(sub_list[j]).or_default().insert(sub_list[i]);
                }
            }
        }

        // Ensure all subscribers appear in the adjacency (even isolated ones)
        for &id in self.sub_to_filters.keys() {
            adj.entry(id).or_default();
        }

        adj
    }

    // ── Clique Detection (Bron-Kerbosch) ─────────────────────────

    /// Find all maximal cliques in the co-subscription graph.
    ///
    /// Uses the Bron-Kerbosch algorithm with pivot selection for efficiency.
    /// Returns only non-trivial cliques (size ≥ 2).
    pub fn find_cliques(&self) -> Vec<Clique> {
        let adj = self.co_subscription_adjacency();
        let all_vertices: BTreeSet<SubscriptionId> = adj.keys().copied().collect();

        let mut cliques = Vec::new();
        Self::bron_kerbosch(
            BTreeSet::new(),      // R: current clique
            all_vertices,         // P: candidates
            BTreeSet::new(),      // X: excluded
            &adj,
            &mut cliques,
        );

        // Convert to Clique structs with shared filters
        let now = Utc::now();
        cliques
            .into_iter()
            .filter(|members| members.len() >= 2) // skip trivial
            .map(|members| {
                let shared = self.shared_filters_for(&members);
                Clique {
                    members: members.into_iter().collect(),
                    shared_filters: shared,
                    detected_at: now,
                }
            })
            .collect()
    }

    /// Bron-Kerbosch with pivot: find all maximal cliques.
    fn bron_kerbosch(
        r: BTreeSet<SubscriptionId>,
        mut p: BTreeSet<SubscriptionId>,
        mut x: BTreeSet<SubscriptionId>,
        adj: &HashMap<SubscriptionId, HashSet<SubscriptionId>>,
        results: &mut Vec<BTreeSet<SubscriptionId>>,
    ) {
        if p.is_empty() && x.is_empty() {
            // R is a maximal clique
            results.push(r);
            return;
        }

        // Choose pivot: vertex in P ∪ X with most neighbors in P
        let pivot = p
            .union(&x)
            .max_by_key(|v| {
                adj.get(v)
                    .map(|n| n.iter().filter(|u| p.contains(u)).count())
                    .unwrap_or(0)
            })
            .copied();

        let pivot_neighbors: HashSet<SubscriptionId> = pivot
            .and_then(|pv| adj.get(&pv))
            .cloned()
            .unwrap_or_default();

        // Iterate over P \ N(pivot)
        let candidates: Vec<_> = p
            .iter()
            .filter(|v| !pivot_neighbors.contains(v))
            .copied()
            .collect();

        for v in candidates {
            let neighbors: BTreeSet<_> = adj.get(&v).map(|n| n.iter().copied().collect()).unwrap_or_default();

            let mut new_r = r.clone();
            new_r.insert(v);

            let new_p: BTreeSet<_> = p.intersection(&neighbors).copied().collect();
            let new_x: BTreeSet<_> = x.intersection(&neighbors).copied().collect();

            Self::bron_kerbosch(new_r, new_p, new_x, adj, results);

            p.remove(&v);
            x.insert(v);
        }
    }

    /// Find topic filters shared by ALL members of a set.
    fn shared_filters_for(&self, members: &BTreeSet<SubscriptionId>) -> HashSet<String> {
        let mut iter = members.iter();
        let first = match iter.next() {
            Some(id) => id,
            None => return HashSet::new(),
        };

        let mut shared: HashSet<String> = self
            .sub_to_filters
            .get(first)
            .cloned()
            .unwrap_or_default();

        for id in iter {
            if let Some(filters) = self.sub_to_filters.get(id) {
                shared.retain(|f| filters.contains(f));
            } else {
                return HashSet::new();
            }
        }

        shared
    }

    // ── Broadcast Optimization ───────────────────────────────────

    /// Find groups of subscribers with identical filter patterns.
    ///
    /// These groups can share a single broadcast::Sender (CoW optimization).
    pub fn broadcast_groups(&self) -> Vec<BroadcastGroup> {
        self.filter_to_subs
            .iter()
            .filter(|(_, subs)| subs.len() >= 2)
            .map(|(filter, subs)| BroadcastGroup {
                filter_pattern: filter.clone(),
                subscribers: subs.iter().copied().collect(),
            })
            .collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sub_entry(id: u64, filter: &str) -> SubscriptionEntry {
        SubscriptionEntry {
            id: SubscriptionId::from_raw(id),
            filter_pattern: filter.to_string(),
        }
    }

    #[test]
    fn empty_graph() {
        let graph = SubscriptionGraph::new();
        assert_eq!(graph.subscriber_count(), 0);
        assert_eq!(graph.filter_count(), 0);
        assert!(graph.find_cliques().is_empty());
    }

    #[test]
    fn single_subscriber_no_cliques() {
        let mut graph = SubscriptionGraph::new();
        graph.add(sub_entry(1, "ci/**"));
        assert_eq!(graph.subscriber_count(), 1);
        assert!(graph.find_cliques().is_empty()); // trivial cliques filtered
    }

    #[test]
    fn two_subscribers_same_filter_form_clique() {
        let mut graph = SubscriptionGraph::new();
        graph.add(sub_entry(1, "ci/**"));
        graph.add(sub_entry(2, "ci/**"));

        let cliques = graph.find_cliques();
        assert_eq!(cliques.len(), 1);
        assert_eq!(cliques[0].size(), 2);
        assert!(cliques[0].shared_filters.contains("ci/**"));
    }

    #[test]
    fn three_subscribers_triangle_clique() {
        let mut graph = SubscriptionGraph::new();
        // All three share "ci/**"
        graph.add(sub_entry(1, "ci/**"));
        graph.add(sub_entry(2, "ci/**"));
        graph.add(sub_entry(3, "ci/**"));

        let cliques = graph.find_cliques();
        assert_eq!(cliques.len(), 1);
        assert_eq!(cliques[0].size(), 3);
    }

    #[test]
    fn disjoint_groups_form_separate_cliques() {
        let mut graph = SubscriptionGraph::new();
        // Group A: subscribe to ci/**
        graph.add(sub_entry(1, "ci/**"));
        graph.add(sub_entry(2, "ci/**"));
        // Group B: subscribe to chat/**
        graph.add(sub_entry(3, "chat/**"));
        graph.add(sub_entry(4, "chat/**"));

        let cliques = graph.find_cliques();
        assert_eq!(cliques.len(), 2);
    }

    #[test]
    fn overlapping_subscriptions_create_larger_clique() {
        let mut graph = SubscriptionGraph::new();
        graph.add(sub_entry(1, "ci/**"));
        graph.add(sub_entry(1, "chat/**")); // sub 1 watches both
        graph.add(sub_entry(2, "ci/**"));
        graph.add(sub_entry(3, "chat/**"));

        // Co-subscription graph:
        // 1-2 connected (share ci/**)
        // 1-3 connected (share chat/**)
        // 2-3 NOT connected (no shared filter)
        // So two cliques: {1,2} and {1,3}
        let cliques = graph.find_cliques();
        assert_eq!(cliques.len(), 2);
        for clique in &cliques {
            assert_eq!(clique.size(), 2);
        }
    }

    #[test]
    fn remove_subscriber_breaks_clique() {
        let mut graph = SubscriptionGraph::new();
        graph.add(sub_entry(1, "ci/**"));
        graph.add(sub_entry(2, "ci/**"));

        assert_eq!(graph.find_cliques().len(), 1);

        // Remove subscriber 2
        graph.remove(SubscriptionId::from_raw(2));
        assert!(graph.find_cliques().is_empty()); // no more cliques
    }

    #[test]
    fn co_subscription_adjacency() {
        let mut graph = SubscriptionGraph::new();
        graph.add(sub_entry(1, "ci/**"));
        graph.add(sub_entry(2, "ci/**"));
        graph.add(sub_entry(3, "chat/**"));

        let adj = graph.co_subscription_adjacency();
        // 1 and 2 are neighbors (share ci/**)
        assert!(adj[&SubscriptionId::from_raw(1)].contains(&SubscriptionId::from_raw(2)));
        assert!(adj[&SubscriptionId::from_raw(2)].contains(&SubscriptionId::from_raw(1)));
        // 3 is isolated
        assert!(adj[&SubscriptionId::from_raw(3)].is_empty());
    }

    #[test]
    fn broadcast_groups_finds_duplicates() {
        let mut graph = SubscriptionGraph::new();
        graph.add(sub_entry(1, "ci/**"));
        graph.add(sub_entry(2, "ci/**"));
        graph.add(sub_entry(3, "ci/**"));
        graph.add(sub_entry(4, "chat/**")); // alone

        let groups = graph.broadcast_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].filter_pattern, "ci/**");
        assert_eq!(groups[0].subscribers.len(), 3);
    }
}
