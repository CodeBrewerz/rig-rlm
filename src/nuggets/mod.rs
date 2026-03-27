//! Nuggets — disk-backed fuzzy key-value memory for LLM agents.
//!
//! A fast, persistent key-value memory with fuzzy key lookup.
//! Think of it as an associative cache for an agent — sub-millisecond
//! recall, fuzzy matching, and fully disk-backed with an LRU cache.
//!
//! # Architecture
//!
//! - **Nugget**: A single disk-backed memory unit storing key→value facts
//!   as JSON.  Recall uses fuzzy string matching — no vectors or complex
//!   math at runtime.
//!
//! - **NuggetShelf**: Multi-nugget manager with LRU cache.  At startup
//!   only a lightweight catalog is built by scanning the save directory.
//!   Nugget data is loaded on-demand and evicted when the cache is full.
//!
//! - **Promote**: Promotes frequently-recalled facts (3+ hits) to MEMORY.md.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use rig_rlm::nuggets::{Nugget, NuggetOpts};
//!
//! let mut n = Nugget::new(NuggetOpts {
//!     name: "project".into(),
//!     ..Default::default()
//! });
//! n.remember("test_cmd", "cargo test -p accounting");
//! let result = n.recall("test", "");
//! assert!(result.found);
//! assert_eq!(result.answer.unwrap(), "cargo test -p accounting");
//! ```

pub mod advanced;
#[cfg(test)]
mod bench;
#[cfg(test)]
mod bench_turboquant;
#[cfg(test)]
mod stress_5k;
#[cfg(test)]
mod test_knowledge_graph;
#[cfg(test)]
mod test_temporal_lattice;
#[cfg(test)]
mod test_temporal_e2e;
pub mod core;
pub mod turboquant;
pub mod keyindex;
pub mod memory;
pub mod promote;
pub mod shelf;

// Re-export main types
pub use memory::{Nugget, NuggetOpts, RecallResult};
pub use shelf::NuggetShelf;

// HRR engine primitives — kept for direct use, benchmarking, and future experimentation
pub use advanced::{CleanupNetwork, Resonator, RffDecorrelator};
