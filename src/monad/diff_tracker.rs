//! Turn Diff Tracker — inspired by Codex CLI.
//!
//! Tracks file changes per agent turn by maintaining in-memory baseline
//! snapshots. After each turn, can compute unified diffs showing exactly
//! what the agent changed. Uses the `similar` crate for in-memory diffing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Tracks sets of changes to files and exposes the overall unified diff.
///
/// Workflow:
/// 1. Before code execution, call `snapshot_before()` for files that might change.
/// 2. After execution, call `get_unified_diff()` to see what changed.
/// 3. Call `reset()` to start a new turn.
#[derive(Default)]
pub struct TurnDiffTracker {
    /// In-memory baseline snapshots (file path → content before change).
    baselines: HashMap<PathBuf, Option<String>>,
    /// Files touched during this turn (ordered).
    touched_files: Vec<PathBuf>,
}

impl TurnDiffTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot a file's current content before any modifications.
    /// If the file doesn't exist yet, records `None` (new file).
    ///
    /// Only snapshots on first call per file per turn (idempotent).
    pub fn snapshot_before(&mut self, path: &Path) {
        let canonical = normalize_path(path);
        if self.baselines.contains_key(&canonical) {
            return; // Already have a baseline for this turn
        }

        let content = std::fs::read_to_string(&canonical).ok();
        self.baselines.insert(canonical.clone(), content);
        self.touched_files.push(canonical);
    }

    /// Snapshot multiple files at once.
    pub fn snapshot_files(&mut self, paths: &[PathBuf]) {
        for path in paths {
            self.snapshot_before(path);
        }
    }

    /// Compute the unified diff for a single file.
    ///
    /// Compares the baseline snapshot with the current file on disk.
    /// Returns `None` if no changes were detected.
    pub fn get_file_diff(&self, path: &Path) -> Option<String> {
        let canonical = normalize_path(path);
        let baseline = self.baselines.get(&canonical)?;

        let old_content = baseline.as_deref().unwrap_or("");
        let new_content = std::fs::read_to_string(&canonical).unwrap_or_default();

        if old_content == new_content {
            return None;
        }

        let diff =
            compute_unified_diff(&canonical.display().to_string(), old_content, &new_content);

        if diff.is_empty() { None } else { Some(diff) }
    }

    /// Compute the aggregated unified diff for all tracked files this turn.
    ///
    /// Returns `None` if no changes were made.
    pub fn get_unified_diff(&self) -> Option<String> {
        let mut diffs = Vec::new();

        for path in &self.touched_files {
            if let Some(diff) = self.get_file_diff(path) {
                diffs.push(diff);
            }
        }

        if diffs.is_empty() {
            None
        } else {
            Some(diffs.join("\n"))
        }
    }

    /// Get a summary of changed files (without full diffs).
    pub fn changed_files_summary(&self) -> Vec<(PathBuf, ChangeKind)> {
        let mut summary = Vec::new();

        for path in &self.touched_files {
            let baseline = self.baselines.get(path);
            let exists_now = path.exists();

            let kind = match (baseline, exists_now) {
                (Some(None), true) => ChangeKind::Created,
                (Some(Some(_)), false) => ChangeKind::Deleted,
                (Some(Some(old)), true) => {
                    let new = std::fs::read_to_string(path).unwrap_or_default();
                    if old == &new {
                        continue; // No change
                    }
                    ChangeKind::Modified
                }
                _ => continue,
            };

            summary.push((path.clone(), kind));
        }

        summary
    }

    /// Reset the tracker for a new turn.
    pub fn reset(&mut self) {
        self.baselines.clear();
        self.touched_files.clear();
    }

    /// Check if any files have been tracked this turn.
    pub fn has_tracked_files(&self) -> bool {
        !self.baselines.is_empty()
    }
}

/// Kind of file change detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeKind {
    Created,
    Modified,
    Deleted,
}

impl std::fmt::Display for ChangeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Created => write!(f, "created"),
            Self::Modified => write!(f, "modified"),
            Self::Deleted => write!(f, "deleted"),
        }
    }
}

/// Normalize a path by resolving it canonically, falling back to the original.
fn normalize_path(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Compute a unified diff between old and new content using the `similar` crate.
fn compute_unified_diff(filename: &str, old: &str, new: &str) -> String {
    use similar::TextDiff;

    let diff = TextDiff::from_lines(old, new);
    let mut output = String::new();

    // Header
    output.push_str(&format!("--- a/{filename}\n"));
    output.push_str(&format!("+++ b/{filename}\n"));

    for hunk in diff.unified_diff().context_radius(3).iter_hunks() {
        output.push_str(&format!("{hunk}"));
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn tracks_file_modifications() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("test.txt");
        fs::write(&file, "line1\nline2\nline3\n").unwrap();

        let mut tracker = TurnDiffTracker::new();
        tracker.snapshot_before(&file);

        // Modify the file
        fs::write(&file, "line1\nmodified\nline3\n").unwrap();

        let diff = tracker.get_unified_diff();
        assert!(diff.is_some());
        let diff = diff.unwrap();
        assert!(diff.contains("-line2"));
        assert!(diff.contains("+modified"));
    }

    #[test]
    fn tracks_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("new_file.txt");

        let mut tracker = TurnDiffTracker::new();
        tracker.snapshot_before(&file); // File doesn't exist yet

        // Create the file
        fs::write(&file, "new content\n").unwrap();

        let diff = tracker.get_unified_diff();
        assert!(diff.is_some());
        assert!(diff.unwrap().contains("+new content"));
    }

    #[test]
    fn no_diff_when_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("test.txt");
        fs::write(&file, "unchanged\n").unwrap();

        let mut tracker = TurnDiffTracker::new();
        tracker.snapshot_before(&file);

        // Don't modify
        let diff = tracker.get_unified_diff();
        assert!(diff.is_none());
    }

    #[test]
    fn reset_clears_state() {
        let mut tracker = TurnDiffTracker::new();
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("test.txt");
        fs::write(&file, "content").unwrap();
        tracker.snapshot_before(&file);
        assert!(tracker.has_tracked_files());

        tracker.reset();
        assert!(!tracker.has_tracked_files());
    }

    #[test]
    fn changed_files_summary_works() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("test.txt");
        fs::write(&file, "old\n").unwrap();

        let mut tracker = TurnDiffTracker::new();
        tracker.snapshot_before(&file);
        fs::write(&file, "new\n").unwrap();

        let summary = tracker.changed_files_summary();
        assert_eq!(summary.len(), 1);
        assert_eq!(summary[0].1, ChangeKind::Modified);
    }
}
