//! # Optimizer module
//!
//! Optimization algorithms for tunable system parameters.
//!
//! - [`gepa`] — GEPA (Genetic-Pareto) optimizer: Pareto-efficient search.
//! - [`newton_schulz`] — Gram Newton-Schulz polar decomposition for the
//!   TEON/Muon optimizer. Implements the stabilized algorithm from
//!   Dao AI Lab (2026): <https://dao-lab.ai/blog/2026/gram-newton-schulz/>

pub mod gepa;
pub mod newton_schulz;
