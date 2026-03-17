//! Nuggets — Holographic Reduced Representation (HRR) memory for LLM agents.
//!
//! A fast, fixed-capacity associative key-value memory backed by complex-valued
//! tensors. Think of it as L1 cache for an agent — tiny, sub-microsecond recall,
//! lossy, and associative (fuzzy key lookup).
//!
//! # Architecture
//!
//! - **Nugget**: A single holographic memory unit storing key→value facts
//!   as superposed complex vectors. Deterministic rebuild from facts using
//!   seeded PRNG (vectors are never serialized, only facts are).
//!
//! - **NuggetShelf**: Multi-nugget manager supporting broadcast recall
//!   across all loaded nuggets (e.g., "prefs", "locations", "debug").
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
pub mod core;
pub mod memory;
pub mod promote;
pub mod shelf;

pub use advanced::{CleanupNetwork, Resonator, RffDecorrelator};
pub use memory::{Nugget, NuggetOpts, RecallResult};
pub use shelf::NuggetShelf;
