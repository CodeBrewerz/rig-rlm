//! Phase 2: Memory Consolidation
//!
//! Merges raw memories from multiple sessions into consolidated
//! knowledge files: MEMORY.md and memory_summary.md.

use super::storage;

/// Build the consolidation prompt with raw memories injected.
pub fn build_consolidation_prompt(memory_root: &std::path::Path, raw_memories: &str) -> String {
    let root_str = memory_root.display().to_string();
    super::CONSOLIDATION_PROMPT
        .replace("{{ memory_root }}", &root_str)
        .replace("{{ raw_memories }}", raw_memories)
}

/// Check if consolidation is needed (enough raw memories accumulated).
pub fn should_consolidate(memory_root: &std::path::Path) -> bool {
    let raw_path = memory_root.join("raw_memories.md");
    match std::fs::metadata(&raw_path) {
        Ok(meta) => meta.len() > 1024, // Consolidate when > 1KB of raw memories
        Err(_) => false,
    }
}

/// Save consolidation outputs (MEMORY.md and memory_summary.md).
pub fn save_consolidation(
    memory_root: &std::path::Path,
    memory_md: &str,
    summary_md: &str,
) -> std::io::Result<()> {
    storage::save_memory_md(memory_root, memory_md)?;
    storage::save_memory_summary(memory_root, summary_md)?;

    // Clear raw memories after successful consolidation
    let raw_path = memory_root.join("raw_memories.md");
    if raw_path.exists() {
        std::fs::write(&raw_path, "")?;
    }

    Ok(())
}
