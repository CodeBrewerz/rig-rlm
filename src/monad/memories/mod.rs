//! Background memory extraction pipeline — inspired by Codex CLI.
//!
//! Two-phase pipeline:
//! - Phase 1 (extraction): After a session ends, extract structured memories
//!   (rollout summary, raw memory) via an LLM call.
//! - Phase 2 (consolidation): Merge raw memories into persistent knowledge
//!   files (MEMORY.md, memory_summary.md).
//!
//! Memory files are stored at `~/.rig-rlm/memories/`.

pub mod consolidation;
pub mod extraction;
pub mod storage;

use std::path::{Path, PathBuf};

/// Template for memory extraction (Phase 1).
pub const EXTRACTION_PROMPT: &str = include_str!("../../templates/memories/extraction.md");
/// Template for memory consolidation (Phase 2).
pub const CONSOLIDATION_PROMPT: &str = include_str!("../../templates/memories/consolidation.md");
/// Template injected into system prompt when memories exist.
pub const READ_PATH_TEMPLATE: &str = include_str!("../../templates/memories/read_path.md");

/// Get the memory root directory.
pub fn memory_root() -> PathBuf {
    let home = std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    home.join(".rig-rlm").join("memories")
}

/// Ensure the memory directory layout exists.
pub fn ensure_layout(root: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(root.join("rollout_summaries"))?;
    Ok(())
}

/// Build the memory read-path instructions for injection into system prompt.
///
/// Returns `None` if no memory_summary.md exists.
pub fn build_memory_instructions(root: &Path) -> Option<String> {
    let summary_path = root.join("memory_summary.md");
    let summary = std::fs::read_to_string(&summary_path).ok()?;
    let summary = summary.trim().to_string();

    if summary.is_empty() {
        return None;
    }

    let base_path = root.display().to_string();
    Some(
        READ_PATH_TEMPLATE
            .replace("{{ base_path }}", &base_path)
            .replace("{{ memory_summary }}", &summary),
    )
}
