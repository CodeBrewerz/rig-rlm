//! Circular replay buffer for RL transitions (adapted from burn-rl).
//!
//! Uses `Vec<f32>` instead of burn Tensors for SPSA compatibility.

// ──────────────────────────────────────────────────────
// Transition
// ──────────────────────────────────────────────────────

/// A single RL transition: (s, a, r, s', done).
#[derive(Debug, Clone)]
pub struct Transition {
    pub state: Vec<f32>,
    pub action_id: usize,
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub done: bool,
    pub constraint_cost: f32,
}

// ──────────────────────────────────────────────────────
// Transition batch
// ──────────────────────────────────────────────────────

/// A batch of transitions for training.
#[derive(Debug, Clone)]
pub struct TransitionBatch {
    pub states: Vec<Vec<f32>>,
    pub action_ids: Vec<usize>,
    pub rewards: Vec<f32>,
    pub next_states: Vec<Vec<f32>>,
    pub dones: Vec<bool>,
    pub constraint_costs: Vec<f32>,
}

impl TransitionBatch {
    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Mean reward in this batch.
    pub fn mean_reward(&self) -> f32 {
        if self.rewards.is_empty() {
            0.0
        } else {
            self.rewards.iter().sum::<f32>() / self.rewards.len() as f32
        }
    }
}

// ──────────────────────────────────────────────────────
// Transition buffer (circular, from burn-rl)
// ──────────────────────────────────────────────────────

/// Circular replay buffer for transitions.
pub struct TransitionBuffer {
    transitions: Vec<Transition>,
    capacity: usize,
    write_head: usize,
    len: usize,
    rng_seed: u64,
}

impl TransitionBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            transitions: Vec::with_capacity(capacity),
            capacity,
            write_head: 0,
            len: 0,
            rng_seed: 42,
        }
    }

    /// Add a transition, overwriting oldest if full.
    pub fn push(&mut self, transition: Transition) {
        if self.transitions.len() < self.capacity {
            self.transitions.push(transition);
            self.len = self.transitions.len();
        } else {
            let idx = self.write_head % self.capacity;
            self.transitions[idx] = transition;
        }
        self.write_head += 1;
    }

    /// Sample a random batch.
    pub fn sample(&mut self, batch_size: usize) -> TransitionBatch {
        assert!(batch_size <= self.len, "batch_size exceeds buffer length");

        let mut states = Vec::with_capacity(batch_size);
        let mut action_ids = Vec::with_capacity(batch_size);
        let mut rewards = Vec::with_capacity(batch_size);
        let mut next_states = Vec::with_capacity(batch_size);
        let mut dones = Vec::with_capacity(batch_size);
        let mut constraint_costs = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            self.rng_seed = self
                .rng_seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            let idx = (self.rng_seed >> 33) as usize % self.len;
            let t = &self.transitions[idx];
            states.push(t.state.clone());
            action_ids.push(t.action_id);
            rewards.push(t.reward);
            next_states.push(t.next_state.clone());
            dones.push(t.done);
            constraint_costs.push(t.constraint_cost);
        }

        TransitionBatch {
            states,
            action_ids,
            rewards,
            next_states,
            dones,
            constraint_costs,
        }
    }

    /// Retrieve the last `n` transitions added to the buffer (chronological order).
    pub fn pop_last(&self, n: usize) -> TransitionBatch {
        let count = n.min(self.len);
        let mut states = Vec::with_capacity(count);
        let mut action_ids = Vec::with_capacity(count);
        let mut rewards = Vec::with_capacity(count);
        let mut next_states = Vec::with_capacity(count);
        let mut dones = Vec::with_capacity(count);
        let mut constraint_costs = Vec::with_capacity(count);

        let start_idx = if self.write_head >= count {
            self.write_head - count
        } else {
            self.capacity + self.write_head - count
        };

        for i in 0..count {
            let idx = (start_idx + i) % self.capacity;
            let t = &self.transitions[idx];
            states.push(t.state.clone());
            action_ids.push(t.action_id);
            rewards.push(t.reward);
            next_states.push(t.next_state.clone());
            dones.push(t.done);
            constraint_costs.push(t.constraint_cost);
        }

        TransitionBatch {
            states,
            action_ids,
            rewards,
            next_states,
            dones,
            constraint_costs,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn clear(&mut self) {
        self.transitions.clear();
        self.write_head = 0;
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_transition(val: f32) -> Transition {
        Transition {
            state: vec![val, val],
            action_id: val as usize,
            reward: val,
            next_state: vec![val + 1.0, val + 1.0],
            done: false,
            constraint_cost: 0.0,
        }
    }

    #[test]
    fn test_push_increments_len() {
        let mut buf = TransitionBuffer::new(5);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());

        buf.push(make_transition(1.0));
        assert_eq!(buf.len(), 1);

        buf.push(make_transition(2.0));
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_push_overwrites_when_full() {
        let mut buf = TransitionBuffer::new(3);
        for i in 0..5 {
            buf.push(make_transition(i as f32));
        }
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.capacity(), 3);
    }

    #[test]
    fn test_sample_returns_correct_size() {
        let mut buf = TransitionBuffer::new(10);
        for i in 0..5 {
            buf.push(make_transition(i as f32));
        }

        let batch = buf.sample(3);
        assert_eq!(batch.len(), 3);
        assert_eq!(batch.states.len(), 3);
        assert_eq!(batch.rewards.len(), 3);
    }

    #[test]
    #[should_panic(expected = "batch_size exceeds buffer length")]
    fn test_sample_panics_when_too_large() {
        let mut buf = TransitionBuffer::new(5);
        buf.push(make_transition(1.0));
        buf.sample(5);
    }

    #[test]
    fn test_clear() {
        let mut buf = TransitionBuffer::new(5);
        buf.push(make_transition(1.0));
        buf.push(make_transition(2.0));
        buf.clear();
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_batch_mean_reward() {
        let mut buf = TransitionBuffer::new(10);
        buf.push(make_transition(1.0));
        buf.push(make_transition(3.0));
        buf.push(make_transition(5.0));

        let batch = buf.sample(3);
        let mean = batch.mean_reward();
        assert!(mean.is_finite());
    }
}
