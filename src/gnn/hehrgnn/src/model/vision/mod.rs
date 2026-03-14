//! Vision modules for document understanding.
//!
//! Adapted from jepa-rs `jepa-vision` crate for processing
//! fiduciary documents (receipts, bank statements) as graph node features.
//!
//! Uses DroPE (paper 2512.12167) instead of RoPE for position encoding.

pub mod document_drope;
pub mod document_patch;
pub mod document_vit;
