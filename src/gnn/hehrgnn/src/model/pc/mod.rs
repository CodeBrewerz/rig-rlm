//! Probabilistic Circuits Engine.
//!
//! A Rust implementation of Probabilistic Circuits (PCs) from first principles.
//! PCs provide **exact tractable inference** — guaranteed correct probabilities,
//! not approximations.
//!
//! # Architecture
//! - `node` — Circuit DAG nodes (Sum, Product, Input)
//! - `distribution` — Leaf distributions (Categorical, Gaussian, Bernoulli)
//! - `circuit` — Compiled circuit for efficient forward/backward inference
//! - `em` — Expectation-Maximization parameter learning
//! - `structure` — Structure learning (HCLT from mutual information)
//! - `query` — High-level query API (marginal, conditional, sample)
//!
//! # Key Property: Tractability
//! A PC is a DAG of sum (mixture), product (factorization), and input (distribution)
//! nodes. When the circuit is **decomposable** (products partition variables) and
//! **smooth** (sums have the same scope), inference is exact and polynomial-time.

pub mod bridge;
pub mod circuit;
pub mod distribution;
pub mod em;
pub mod fiduciary_pc;
pub mod node;
pub mod query;
pub mod structure;
