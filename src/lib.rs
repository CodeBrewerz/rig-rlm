//! rig-rlm: Monadic AI agent library with durable execution.
//!
//! This library crate exposes the core modules so both the CLI binary
//! (`src/main.rs`) and the Restate server binary (`src/restate_server.rs`)
//! can share the same implementation.

// Core monadic architecture
pub mod monad;

// Infrastructure
pub mod a2a_server;
pub mod apply_patch;
pub mod cancellation;
pub mod exec_policy;
pub mod mcp_server;
pub mod persistence;
pub mod safety;
pub mod sandbox;
pub mod session;

// Supporting modules
pub mod exec;
pub mod llm;
pub mod repl;

// DSRs integration
pub mod agent_metric;
pub mod agent_module;
pub mod signature;

// Infrastructure
pub mod chunking;
pub mod pipeline;

// ARC-AGI Benchmark
pub mod arc;
