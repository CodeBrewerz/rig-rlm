//! rig-rlm: Monadic AI agent library with durable execution.
//!
//! This library crate exposes the core modules so both the CLI binary
//! (`src/main.rs`) and the Restate server binary (`src/restate_server.rs`)
//! can share the same implementation.
//!
//! ## Feature gates (2026-05-11 Finverse split)
//!
//! See `FINVERSE-HERMES.md` for the feature shape and the
//! `sandbox-worker` default-build contract. Heavy/research modules are
//! gated behind named features; the default build is the slim sandbox
//! worker.

// Core monadic architecture (always on)
pub mod monad;

// Infrastructure (always on)
pub mod apply_patch;
pub mod cancellation;
pub mod exec_policy;
pub mod persistence;
pub mod safety;
pub mod session;

// Sandbox (core trait + MicrosandboxExecutor under sandbox-worker;
// Pyo3CodeExecutor gated by dev-pyo3 inside the module).
pub mod sandbox;

// Legacy pyo3+lambda+llm execution environment (research path only).
#[cfg(feature = "research-rlm")]
pub mod exec;

// REPL is a thin wrapper over `exec::Pyo3Executor`; research-rlm only.
#[cfg(feature = "research-rlm")]
pub mod repl;

// λ-RLM (lambda) — research-rlm only
#[cfg(feature = "research-rlm")]
pub mod lambda;

// llm.rs is the λ-RLM API; requires `lambda`.
#[cfg(feature = "research-rlm")]
pub mod llm;

// DSRs integration (research-rlm)
#[cfg(feature = "research-rlm")]
pub mod agent_metric;
#[cfg(feature = "research-rlm")]
pub mod agent_module;
#[cfg(feature = "research-rlm")]
pub mod signature;

// Pipeline / chunking (pipeline is DSRs-coupled; chunking is light but
// only used by research paths today — gate together for simplicity).
#[cfg(feature = "research-rlm")]
pub mod chunking;
#[cfg(feature = "research-rlm")]
pub mod pipeline;

// ARC-AGI Benchmark (DSRs-heavy)
#[cfg(feature = "research-rlm")]
pub mod arc;

// Holographic memory (research-rlm)
#[cfg(feature = "research-rlm")]
pub mod nuggets;

// Channel system (telegram + ws demos)
#[cfg(feature = "channels")]
pub mod channels;

// A2A HTTP server (axum/tower-http heavy)
#[cfg(feature = "a2a-worker")]
pub mod a2a_server;

// Standalone MCP server (uses nuggets + rmcp)
#[cfg(feature = "mcp-worker")]
pub mod mcp_server;
