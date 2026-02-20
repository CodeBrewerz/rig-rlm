//! File I/O for the memory system.
//!
//! Handles reading/writing memory files to disk at `~/.rig-rlm/memories/`.

use std::io::{self, Write};
use std::path::Path;

/// Save a rollout summary to `rollout_summaries/<slug>.md`.
pub fn save_rollout_summary(root: &Path, slug: &str, content: &str) -> io::Result<()> {
    let dir = root.join("rollout_summaries");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{slug}.md"));
    std::fs::write(&path, content)
}

/// Append raw memory to `raw_memories.md`.
pub fn append_raw_memory(root: &Path, content: &str) -> io::Result<()> {
    let path = root.join("raw_memories.md");
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    writeln!(file, "\n---\n")?;
    writeln!(file, "{content}")
}

/// Read raw memories from `raw_memories.md`.
pub fn read_raw_memories(root: &Path) -> Option<String> {
    let path = root.join("raw_memories.md");
    std::fs::read_to_string(&path).ok()
}

/// Save MEMORY.md.
pub fn save_memory_md(root: &Path, content: &str) -> io::Result<()> {
    std::fs::write(root.join("MEMORY.md"), content)
}

/// Save memory_summary.md.
pub fn save_memory_summary(root: &Path, content: &str) -> io::Result<()> {
    std::fs::write(root.join("memory_summary.md"), content)
}

/// Read MEMORY.md.
pub fn read_memory_md(root: &Path) -> Option<String> {
    std::fs::read_to_string(root.join("MEMORY.md")).ok()
}

/// Read memory_summary.md.
pub fn read_memory_summary(root: &Path) -> Option<String> {
    std::fs::read_to_string(root.join("memory_summary.md")).ok()
}

/// List all rollout summary files.
pub fn list_rollout_summaries(root: &Path) -> Vec<String> {
    let dir = root.join("rollout_summaries");
    match std::fs::read_dir(&dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "md"))
            .filter_map(|e| e.file_name().into_string().ok())
            .collect(),
        Err(_) => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_and_read_rollout_summary() {
        let dir = tempfile::tempdir().unwrap();
        save_rollout_summary(dir.path(), "test-session", "summary content").unwrap();

        let summaries = list_rollout_summaries(dir.path());
        assert_eq!(summaries.len(), 1);
        assert!(summaries[0].contains("test-session"));
    }

    #[test]
    fn append_and_read_raw_memories() {
        let dir = tempfile::tempdir().unwrap();
        append_raw_memory(dir.path(), "memory 1").unwrap();
        append_raw_memory(dir.path(), "memory 2").unwrap();

        let raw = read_raw_memories(dir.path()).unwrap();
        assert!(raw.contains("memory 1"));
        assert!(raw.contains("memory 2"));
    }

    #[test]
    fn save_and_read_memory_summary() {
        let dir = tempfile::tempdir().unwrap();
        save_memory_summary(dir.path(), "# Summary").unwrap();

        let summary = read_memory_summary(dir.path()).unwrap();
        assert_eq!(summary, "# Summary");
    }
}
