//! Short-Term Memory for graphs (Gap #6)
//!
//! From jepa-rs `jepa-world/memory.rs`:
//! bounded ring buffer of recent graph state representations.
//! Enables temporal reasoning by maintaining a sliding window
//! of past graph snapshots that the model can attend to.
//!
//! In the hehrgnn context:
//! - Each "state" is a snapshot of graph embeddings
//! - The memory allows comparing current vs. historical embeddings
//! - Useful for detecting drift, temporal anomalies, and trend prediction

use std::collections::HashMap;

/// A snapshot of graph embeddings at a point in time.
#[derive(Debug, Clone)]
pub struct GraphState {
    /// Embeddings per node type at this timestep.
    pub embeddings: HashMap<String, Vec<Vec<f32>>>,
    /// Timestamp or step when this state was captured.
    pub step: usize,
    /// Optional metadata (e.g., loss at this step).
    pub loss: Option<f32>,
}

/// Bounded ring buffer of recent graph states.
#[derive(Debug, Clone)]
pub struct ShortTermMemory {
    /// Ring buffer storage.
    buffer: Vec<Option<GraphState>>,
    /// Maximum capacity.
    capacity: usize,
    /// Write position (wraps around).
    write_pos: usize,
    /// Number of entries currently stored.
    count: usize,
}

impl ShortTermMemory {
    /// Create a new short-term memory with given capacity.
    ///
    /// # Panics
    /// Panics if capacity is 0.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Capacity must be positive");
        Self {
            buffer: (0..capacity).map(|_| None).collect(),
            capacity,
            write_pos: 0,
            count: 0,
        }
    }

    /// Push a new graph state into memory.
    /// Evicts oldest entry if at capacity.
    pub fn push(&mut self, state: GraphState) {
        self.buffer[self.write_pos] = Some(state);
        self.write_pos = (self.write_pos + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    /// Get the most recent state.
    pub fn latest(&self) -> Option<&GraphState> {
        if self.count == 0 {
            return None;
        }
        let idx = if self.write_pos == 0 {
            self.capacity - 1
        } else {
            self.write_pos - 1
        };
        self.buffer[idx].as_ref()
    }

    /// Get entries in chronological order (oldest first).
    pub fn entries_chronological(&self) -> Vec<&GraphState> {
        if self.count == 0 {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(self.count);

        if self.count < self.capacity {
            // Buffer hasn't wrapped yet
            for i in 0..self.count {
                if let Some(ref state) = self.buffer[i] {
                    result.push(state);
                }
            }
        } else {
            // Buffer has wrapped — start from write_pos (oldest)
            for i in 0..self.capacity {
                let idx = (self.write_pos + i) % self.capacity;
                if let Some(ref state) = self.buffer[idx] {
                    result.push(state);
                }
            }
        }

        result
    }

    /// Number of stored entries.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Whether the buffer is at capacity.
    pub fn is_full(&self) -> bool {
        self.count == self.capacity
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        for slot in self.buffer.iter_mut() {
            *slot = None;
        }
        self.write_pos = 0;
        self.count = 0;
    }

    /// Compute embedding drift: average L2 distance between
    /// latest and earliest states per node type.
    pub fn compute_drift(&self) -> f32 {
        let entries = self.entries_chronological();
        if entries.len() < 2 {
            return 0.0;
        }

        let oldest = &entries[0].embeddings;
        let newest = &entries[entries.len() - 1].embeddings;

        let mut total_drift = 0.0f32;
        let mut count = 0usize;

        for (nt, old_vecs) in oldest {
            if let Some(new_vecs) = newest.get(nt) {
                for (old, new) in old_vecs.iter().zip(new_vecs.iter()) {
                    let dist: f32 = old
                        .iter()
                        .zip(new.iter())
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum::<f32>()
                        .sqrt();
                    total_drift += dist;
                    count += 1;
                }
            }
        }

        if count > 0 {
            total_drift / count as f32
        } else {
            0.0
        }
    }

    /// Compute temporal consistency: average cosine similarity
    /// between consecutive states. Higher = more stable.
    pub fn compute_temporal_consistency(&self) -> f32 {
        let entries = self.entries_chronological();
        if entries.len() < 2 {
            return 1.0; // Trivially consistent
        }

        let mut total_sim = 0.0f32;
        let mut count = 0usize;

        for window in entries.windows(2) {
            let prev = &window[0].embeddings;
            let curr = &window[1].embeddings;

            for (nt, prev_vecs) in prev {
                if let Some(curr_vecs) = curr.get(nt) {
                    for (p, c) in prev_vecs.iter().zip(curr_vecs.iter()) {
                        let dot: f32 = p.iter().zip(c.iter()).map(|(&a, &b)| a * b).sum();
                        let norm_p = p.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                        let norm_c = c.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                        total_sim += dot / (norm_p * norm_c);
                        count += 1;
                    }
                }
            }
        }

        if count > 0 {
            total_sim / count as f32
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(step: usize, val: f32) -> GraphState {
        let mut emb = HashMap::new();
        emb.insert("user".to_string(), vec![vec![val, val * 2.0]]);
        GraphState {
            embeddings: emb,
            step,
            loss: Some(1.0 / (step as f32 + 1.0)),
        }
    }

    #[test]
    fn test_new_memory_is_empty() {
        let mem = ShortTermMemory::new(5);
        assert!(mem.is_empty());
        assert_eq!(mem.len(), 0);
        assert!(mem.latest().is_none());
    }

    #[test]
    fn test_push_and_latest() {
        let mut mem = ShortTermMemory::new(3);
        mem.push(make_state(0, 1.0));
        assert_eq!(mem.len(), 1);
        assert_eq!(mem.latest().unwrap().step, 0);

        mem.push(make_state(1, 2.0));
        assert_eq!(mem.latest().unwrap().step, 1);
    }

    #[test]
    fn test_eviction_at_capacity() {
        let mut mem = ShortTermMemory::new(2);
        mem.push(make_state(0, 1.0));
        mem.push(make_state(1, 2.0));
        assert!(mem.is_full());

        // Push a third — should evict oldest
        mem.push(make_state(2, 3.0));
        assert_eq!(mem.len(), 2);

        let entries = mem.entries_chronological();
        assert_eq!(entries[0].step, 1); // Oldest surviving
        assert_eq!(entries[1].step, 2); // Newest
    }

    #[test]
    fn test_chronological_order() {
        let mut mem = ShortTermMemory::new(5);
        for i in 0..5 {
            mem.push(make_state(i, i as f32));
        }

        let entries = mem.entries_chronological();
        assert_eq!(entries.len(), 5);
        for (i, e) in entries.iter().enumerate() {
            assert_eq!(e.step, i, "Entry {} should have step {}", i, i);
        }
    }

    #[test]
    fn test_drift_computation() {
        let mut mem = ShortTermMemory::new(5);
        mem.push(make_state(0, 0.0)); // [0.0, 0.0]
        mem.push(make_state(1, 1.0)); // [1.0, 2.0]

        let drift = mem.compute_drift();
        // L2 from [0,0] to [1,2] = sqrt(1+4) ≈ 2.236
        assert!(drift > 2.0 && drift < 2.5, "Drift = {}", drift);
    }

    #[test]
    fn test_temporal_consistency_identical() {
        let mut mem = ShortTermMemory::new(3);
        mem.push(make_state(0, 1.0));
        mem.push(make_state(1, 1.0)); // Same embeddings

        let consistency = mem.compute_temporal_consistency();
        assert!(
            (consistency - 1.0).abs() < 1e-4,
            "Identical states should have consistency ≈ 1.0, got {}",
            consistency
        );
    }

    #[test]
    fn test_clear() {
        let mut mem = ShortTermMemory::new(5);
        mem.push(make_state(0, 1.0));
        mem.push(make_state(1, 2.0));
        mem.clear();
        assert!(mem.is_empty());
        assert_eq!(mem.len(), 0);
    }
}
