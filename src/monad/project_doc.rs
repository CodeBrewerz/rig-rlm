//! AGENTS.md / Project Doc Loading — inspired by Codex CLI.
//!
//! Automatically loads project-specific instructions from `AGENTS.md` files
//! found in the working directory or parent directories. These instructions
//! are injected into the system prompt so the agent understands project
//! conventions, coding standards, and preferences.

use std::path::{Path, PathBuf};

/// Maximum file size to load (64 KB).
const MAX_FILE_SIZE: u64 = 64 * 1024;

/// Walk up from `cwd` looking for `AGENTS.md` files.
/// Returns the concatenated contents of all found files (closest first).
///
/// Stops after finding the first one to keep instructions focused.
/// If you need hierarchical merging, change this to collect all.
pub fn load_project_instructions(cwd: &Path) -> Option<String> {
    let mut current = cwd.to_path_buf();

    loop {
        let agents_md = current.join("AGENTS.md");
        if let Some(content) = try_read_agents_file(&agents_md) {
            return Some(content);
        }

        // Also check .agents/AGENTS.md
        let dot_agents = current.join(".agents").join("AGENTS.md");
        if let Some(content) = try_read_agents_file(&dot_agents) {
            return Some(content);
        }

        // Walk up to parent
        if !current.pop() {
            break;
        }
    }

    None
}

/// Load all AGENTS.md files from cwd up to root, merging them
/// with the most specific (closest to cwd) first.
pub fn load_project_instructions_hierarchical(cwd: &Path) -> Option<String> {
    let mut instructions = Vec::new();
    let mut current = cwd.to_path_buf();

    loop {
        let agents_md = current.join("AGENTS.md");
        if let Some(content) = try_read_agents_file(&agents_md) {
            instructions.push((current.clone(), content));
        }

        let dot_agents = current.join(".agents").join("AGENTS.md");
        if let Some(content) = try_read_agents_file(&dot_agents) {
            instructions.push((current.clone(), content));
        }

        if !current.pop() {
            break;
        }
    }

    if instructions.is_empty() {
        return None;
    }

    // Reverse so root instructions appear first, most specific last
    instructions.reverse();

    let merged: Vec<String> = instructions
        .into_iter()
        .map(|(path, content)| format!("# Instructions from {}\n\n{}", path.display(), content))
        .collect();

    Some(merged.join("\n\n---\n\n"))
}

/// Try to read an AGENTS.md file, checking size limits.
fn try_read_agents_file(path: &PathBuf) -> Option<String> {
    let metadata = std::fs::metadata(path).ok()?;
    if !metadata.is_file() || metadata.len() > MAX_FILE_SIZE {
        return None;
    }
    std::fs::read_to_string(path).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn load_finds_agents_md_in_current_dir() {
        let dir = tempfile::tempdir().unwrap();
        let agents_path = dir.path().join("AGENTS.md");
        fs::write(&agents_path, "# Test Project\n\nUse pytest for tests.").unwrap();

        let result = load_project_instructions(dir.path());
        assert!(result.is_some());
        assert!(result.unwrap().contains("pytest"));
    }

    #[test]
    fn load_returns_none_when_no_file() {
        let dir = tempfile::tempdir().unwrap();
        let result = load_project_instructions(dir.path());
        assert!(result.is_none());
    }

    #[test]
    fn load_finds_in_parent_dir() {
        let parent = tempfile::tempdir().unwrap();
        let child = parent.path().join("src");
        fs::create_dir(&child).unwrap();
        fs::write(parent.path().join("AGENTS.md"), "# Root Instructions").unwrap();

        let result = load_project_instructions(&child);
        assert!(result.is_some());
        assert!(result.unwrap().contains("Root Instructions"));
    }
}
