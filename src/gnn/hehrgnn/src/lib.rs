//! # Finverse GNN Platform
//!
//! A relational GNN platform for finance graph intelligence, built on Burn 0.20.
//!
//! ## Architecture
//!
//! - [`data`] — HeteroGraph, vocabulary, batching, neighbor sampling, synthetic data.
//! - [`model`] — GraphSAGE, RGCN, HEHRGNN backbone architectures.
//! - [`tasks`] — Task heads: link prediction, node classification, anomaly, forecasting.
//! - [`training`] — Scoring functions, loss, training loop, observability.
//! - [`eval`] — Link prediction evaluation with MRR and Hits@K metrics.

pub mod data;
pub mod eval;
pub mod feedback;
pub mod ingest;
pub mod model;
pub mod optimizer;
pub mod server;
pub mod tasks;
pub mod training;
