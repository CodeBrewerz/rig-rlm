//! Apply-Patch Tool (inspired by Codex `apply-patch` crate).
//!
//! Parses unified diff format and applies file changes (Add, Delete, Update).
//! This enables the LLM to write code patches instead of full file contents,
//! which is more efficient for large files.
//!
//! # Supported Format
//!
//! ```text
//! *** Begin Patch ***
//! --- a/path/to/file.py
//! +++ b/path/to/file.py
//! @@ -10,3 +10,4 @@
//!  context line
//! -old line
//! +new line
//! +added line
//!  context line
//! *** End Patch ***
//! ```

use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

// ── ParseError ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum PatchError {
    /// Could not parse the patch format.
    ParseError(String),
    /// Target file not found.
    FileNotFound(PathBuf),
    /// Hunk does not match file contents at expected location.
    HunkMismatch {
        file: PathBuf,
        expected_line: usize,
        context: String,
    },
    /// I/O error.
    IoError(String),
}

impl fmt::Display for PatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParseError(msg) => write!(f, "patch parse error: {msg}"),
            Self::FileNotFound(path) => write!(f, "file not found: {}", path.display()),
            Self::HunkMismatch {
                file,
                expected_line,
                context,
            } => write!(
                f,
                "hunk mismatch in {} at line {expected_line}: {context}",
                file.display()
            ),
            Self::IoError(msg) => write!(f, "I/O error: {msg}"),
        }
    }
}

impl std::error::Error for PatchError {}

// ── Hunk ──────────────────────────────────────────────────────────────────

/// A single hunk from a unified diff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hunk {
    /// Line number in the old file (1-indexed).
    pub old_start: usize,
    /// Number of lines in the old file.
    pub old_count: usize,
    /// Line number in the new file (1-indexed).
    pub new_start: usize,
    /// Number of lines in the new file.
    pub new_count: usize,
    /// Lines in the hunk: (type, content).
    /// Types: ' ' = context, '-' = removed, '+' = added
    pub lines: Vec<HunkLine>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HunkLine {
    pub kind: HunkLineKind,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HunkLineKind {
    Context,
    Remove,
    Add,
}

// ── FileChange ────────────────────────────────────────────────────────────

/// Describes a change to be applied to a file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileChange {
    /// Create a new file with the given content.
    Add { content: String },
    /// Delete the file.
    Delete,
    /// Update the file by applying hunks.
    Update { hunks: Vec<Hunk> },
    /// Rename the file (with optional content update).
    Rename { new_path: PathBuf, hunks: Vec<Hunk> },
}

// ── PatchAction ───────────────────────────────────────────────────────────

/// A parsed patch with all file changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchAction {
    /// Map of file paths to their changes.
    pub changes: Vec<(PathBuf, FileChange)>,
}

impl PatchAction {
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    pub fn file_count(&self) -> usize {
        self.changes.len()
    }

    /// Human-readable summary of the patch.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        for (path, change) in &self.changes {
            let desc = match change {
                FileChange::Add { content } => {
                    format!("add {} ({} bytes)", path.display(), content.len())
                }
                FileChange::Delete => format!("delete {}", path.display()),
                FileChange::Update { hunks } => {
                    let added: usize = hunks
                        .iter()
                        .flat_map(|h| &h.lines)
                        .filter(|l| l.kind == HunkLineKind::Add)
                        .count();
                    let removed: usize = hunks
                        .iter()
                        .flat_map(|h| &h.lines)
                        .filter(|l| l.kind == HunkLineKind::Remove)
                        .count();
                    format!(
                        "update {} ({} hunks, +{added}/-{removed})",
                        path.display(),
                        hunks.len()
                    )
                }
                FileChange::Rename { new_path, .. } => {
                    format!("rename {} → {}", path.display(), new_path.display())
                }
            };
            parts.push(desc);
        }
        parts.join("; ")
    }
}

// ── Parser ────────────────────────────────────────────────────────────────

/// Parse a unified diff string into a `PatchAction`.
pub fn parse_patch(patch: &str) -> Result<PatchAction, PatchError> {
    let mut changes = Vec::new();
    let lines: Vec<&str> = patch.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];

        // Skip patch envelope markers
        if line.starts_with("***") || line.trim().is_empty() {
            i += 1;
            continue;
        }

        // Look for --- a/path header
        if line.starts_with("--- ") {
            let old_path = parse_file_path(line, "--- ");

            // Must be followed by +++ b/path
            i += 1;
            if i >= lines.len() || !lines[i].starts_with("+++ ") {
                return Err(PatchError::ParseError(
                    "expected +++ line after ---".to_string(),
                ));
            }
            let new_path = parse_file_path(lines[i], "+++ ");
            i += 1;

            // Determine change type
            if old_path == "/dev/null" {
                // New file — collect all added lines until next --- or end
                let mut content = String::new();
                while i < lines.len() {
                    if lines[i].starts_with("--- ") || lines[i].starts_with("***") {
                        break;
                    }
                    if lines[i].starts_with("@@ ") {
                        i += 1;
                        continue;
                    }
                    if let Some(stripped) = lines[i].strip_prefix('+') {
                        content.push_str(stripped);
                        content.push('\n');
                    }
                    i += 1;
                }
                changes.push((PathBuf::from(&new_path), FileChange::Add { content }));
            } else if new_path == "/dev/null" {
                // Deleted file
                changes.push((PathBuf::from(&old_path), FileChange::Delete));
                // Skip hunk lines
                while i < lines.len()
                    && !lines[i].starts_with("--- ")
                    && !lines[i].starts_with("***")
                {
                    i += 1;
                }
            } else {
                // Update — parse hunks
                let mut hunks = Vec::new();
                while i < lines.len()
                    && !lines[i].starts_with("--- ")
                    && !lines[i].starts_with("***")
                {
                    if lines[i].starts_with("@@ ") {
                        let (hunk, next_i) = parse_hunk(&lines, i)?;
                        hunks.push(hunk);
                        i = next_i;
                    } else {
                        i += 1;
                    }
                }

                let path = if old_path != new_path {
                    let file_change = FileChange::Rename {
                        new_path: PathBuf::from(&new_path),
                        hunks,
                    };
                    changes.push((PathBuf::from(&old_path), file_change));
                    continue;
                } else {
                    PathBuf::from(&old_path)
                };

                if !hunks.is_empty() {
                    changes.push((path, FileChange::Update { hunks }));
                }
            }
        } else {
            i += 1;
        }
    }

    Ok(PatchAction { changes })
}

/// Parse a file path from a --- or +++ line.
fn parse_file_path(line: &str, prefix: &str) -> String {
    let raw = line.strip_prefix(prefix).unwrap_or("").trim();
    // Strip git a/ b/ prefixes
    if raw.starts_with("a/") || raw.starts_with("b/") {
        raw[2..].to_string()
    } else {
        raw.to_string()
    }
}

/// Parse a single @@ hunk and return it along with the next line index.
fn parse_hunk(lines: &[&str], start: usize) -> Result<(Hunk, usize), PatchError> {
    let header = lines[start];
    let (old_start, old_count, new_start, new_count) = parse_hunk_header(header)?;

    let mut hunk_lines = Vec::new();
    let mut i = start + 1;

    while i < lines.len() {
        let line = lines[i];

        if line.starts_with("@@ ")
            || line.starts_with("--- ")
            || line.starts_with("+++ ")
            || line.starts_with("***")
        {
            break;
        }

        if let Some(content) = line.strip_prefix('+') {
            hunk_lines.push(HunkLine {
                kind: HunkLineKind::Add,
                content: content.to_string(),
            });
        } else if let Some(content) = line.strip_prefix('-') {
            hunk_lines.push(HunkLine {
                kind: HunkLineKind::Remove,
                content: content.to_string(),
            });
        } else if let Some(content) = line.strip_prefix(' ') {
            hunk_lines.push(HunkLine {
                kind: HunkLineKind::Context,
                content: content.to_string(),
            });
        } else if line == "\\ No newline at end of file" {
            // Skip this marker
        } else {
            // Treat as context line (some diffs don't have leading space)
            hunk_lines.push(HunkLine {
                kind: HunkLineKind::Context,
                content: line.to_string(),
            });
        }

        i += 1;
    }

    Ok((
        Hunk {
            old_start,
            old_count,
            new_start,
            new_count,
            lines: hunk_lines,
        },
        i,
    ))
}

/// Parse a @@ -a,b +c,d @@ header.
fn parse_hunk_header(header: &str) -> Result<(usize, usize, usize, usize), PatchError> {
    // Format: @@ -old_start,old_count +new_start,new_count @@
    let stripped = header
        .strip_prefix("@@ ")
        .and_then(|s| {
            let at_pos = s.find(" @@")?;
            Some(&s[..at_pos])
        })
        .ok_or_else(|| PatchError::ParseError(format!("invalid hunk header: {header}")))?;

    let parts: Vec<&str> = stripped.split_whitespace().collect();
    if parts.len() < 2 {
        return Err(PatchError::ParseError(format!(
            "invalid hunk header: {header}"
        )));
    }

    let (old_start, old_count) = parse_range(parts[0].strip_prefix('-').unwrap_or(parts[0]))?;
    let (new_start, new_count) = parse_range(parts[1].strip_prefix('+').unwrap_or(parts[1]))?;

    Ok((old_start, old_count, new_start, new_count))
}

/// Parse a range spec like "10,3" or "10" into (start, count).
fn parse_range(spec: &str) -> Result<(usize, usize), PatchError> {
    if let Some((start, count)) = spec.split_once(',') {
        let s = start
            .parse::<usize>()
            .map_err(|e| PatchError::ParseError(format!("invalid line number: {e}")))?;
        let c = count
            .parse::<usize>()
            .map_err(|e| PatchError::ParseError(format!("invalid count: {e}")))?;
        Ok((s, c))
    } else {
        let s = spec
            .parse::<usize>()
            .map_err(|e| PatchError::ParseError(format!("invalid line number: {e}")))?;
        Ok((s, 1))
    }
}

// ── Applier ───────────────────────────────────────────────────────────────

/// Apply a parsed patch to the filesystem.
///
/// Returns a list of (path, description) for each file changed.
pub fn apply_patch(
    patch: &PatchAction,
    base_dir: &Path,
) -> Result<Vec<(PathBuf, String)>, PatchError> {
    let mut results = Vec::new();

    for (path, change) in &patch.changes {
        let full_path = if path.is_absolute() {
            path.clone()
        } else {
            base_dir.join(path)
        };

        match change {
            FileChange::Add { content } => {
                if let Some(parent) = full_path.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| PatchError::IoError(format!("mkdir: {e}")))?;
                }
                std::fs::write(&full_path, content)
                    .map_err(|e| PatchError::IoError(format!("write: {e}")))?;
                results.push((full_path, "created".to_string()));
            }

            FileChange::Delete => {
                if full_path.exists() {
                    std::fs::remove_file(&full_path)
                        .map_err(|e| PatchError::IoError(format!("delete: {e}")))?;
                    results.push((full_path, "deleted".to_string()));
                }
            }

            FileChange::Update { hunks } => {
                let contents = std::fs::read_to_string(&full_path)
                    .map_err(|_| PatchError::FileNotFound(full_path.clone()))?;

                let new_contents = apply_hunks_to_content(&contents, hunks, &full_path)?;

                std::fs::write(&full_path, &new_contents)
                    .map_err(|e| PatchError::IoError(format!("write: {e}")))?;
                results.push((full_path, format!("{} hunks applied", hunks.len())));
            }

            FileChange::Rename { new_path, hunks } => {
                let new_full = if new_path.is_absolute() {
                    new_path.clone()
                } else {
                    base_dir.join(new_path)
                };

                if !hunks.is_empty() {
                    let contents = std::fs::read_to_string(&full_path)
                        .map_err(|_| PatchError::FileNotFound(full_path.clone()))?;

                    let new_contents = apply_hunks_to_content(&contents, hunks, &full_path)?;

                    if let Some(parent) = new_full.parent() {
                        std::fs::create_dir_all(parent)
                            .map_err(|e| PatchError::IoError(format!("mkdir: {e}")))?;
                    }
                    std::fs::write(&new_full, &new_contents)
                        .map_err(|e| PatchError::IoError(format!("write: {e}")))?;
                } else {
                    std::fs::rename(&full_path, &new_full)
                        .map_err(|e| PatchError::IoError(format!("rename: {e}")))?;
                }

                if full_path.exists() && full_path != new_full {
                    let _ = std::fs::remove_file(&full_path);
                }

                results.push((new_full, format!("renamed from {}", path.display())));
            }
        }
    }

    Ok(results)
}

/// Apply hunks to file content, returning the new content.
pub fn apply_hunks_to_content(
    content: &str,
    hunks: &[Hunk],
    file_path: &Path,
) -> Result<String, PatchError> {
    let mut lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();

    // Apply hunks in reverse order to preserve line numbers
    let mut sorted_hunks: Vec<&Hunk> = hunks.iter().collect();
    sorted_hunks.sort_by(|a, b| b.old_start.cmp(&a.old_start));

    for hunk in sorted_hunks {
        let start_idx = if hunk.old_start > 0 {
            hunk.old_start - 1
        } else {
            0
        };

        // Build the replacement lines
        let mut new_lines = Vec::new();
        let mut old_line_idx = start_idx;

        for hunk_line in &hunk.lines {
            match hunk_line.kind {
                HunkLineKind::Context => {
                    // Verify context matches
                    if old_line_idx < lines.len() {
                        new_lines.push(lines[old_line_idx].clone());
                        old_line_idx += 1;
                    } else {
                        new_lines.push(hunk_line.content.clone());
                    }
                }
                HunkLineKind::Remove => {
                    // Skip the old line (verify it matches if possible)
                    if old_line_idx < lines.len() {
                        let expected = lines[old_line_idx].trim();
                        let got = hunk_line.content.trim();
                        if expected != got && !expected.is_empty() && !got.is_empty() {
                            return Err(PatchError::HunkMismatch {
                                file: file_path.to_path_buf(),
                                expected_line: old_line_idx + 1,
                                context: format!(
                                    "expected '{}', got '{}'",
                                    &expected[..expected.len().min(50)],
                                    &got[..got.len().min(50)]
                                ),
                            });
                        }
                    }
                    old_line_idx += 1;
                }
                HunkLineKind::Add => {
                    new_lines.push(hunk_line.content.clone());
                }
            }
        }

        // Calculate how many old lines this hunk replaces
        let old_lines_consumed = old_line_idx - start_idx;

        // Replace the range
        let end_idx = (start_idx + old_lines_consumed).min(lines.len());
        lines.splice(start_idx..end_idx, new_lines);
    }

    let mut result = lines.join("\n");
    // Preserve trailing newline if original had one
    if content.ends_with('\n') && !result.ends_with('\n') {
        result.push('\n');
    }

    Ok(result)
}

/// Dry-run: parse and validate a patch without applying it.
pub fn validate_patch(patch: &str) -> Result<PatchAction, PatchError> {
    parse_patch(patch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_update_patch() {
        let patch = r#"
--- a/hello.py
+++ b/hello.py
@@ -1,3 +1,4 @@
 def hello():
-    print("hello")
+    print("hello world")
+    return True
     pass
"#;
        let action = parse_patch(patch).unwrap();
        assert_eq!(action.file_count(), 1);
        assert!(!action.summary().is_empty());
    }

    #[test]
    fn parse_add_file_patch() {
        let patch = r#"
--- /dev/null
+++ b/new_file.py
@@ -0,0 +1,3 @@
+def new_func():
+    return 42
+
"#;
        let action = parse_patch(patch).unwrap();
        assert_eq!(action.file_count(), 1);
        match &action.changes[0].1 {
            FileChange::Add { content } => {
                assert!(content.contains("return 42"));
            }
            _ => panic!("expected FileChange::Add"),
        }
    }

    #[test]
    fn parse_delete_file_patch() {
        let patch = r#"
--- a/old_file.py
+++ /dev/null
@@ -1,3 +0,0 @@
-def old_func():
-    return 0
-
"#;
        let action = parse_patch(patch).unwrap();
        assert_eq!(action.file_count(), 1);
        assert!(matches!(&action.changes[0].1, FileChange::Delete));
    }

    #[test]
    fn parse_multi_file_patch() {
        let patch = r#"
--- a/file1.py
+++ b/file1.py
@@ -1,2 +1,2 @@
 def foo():
-    return 1
+    return 2
--- a/file2.py
+++ b/file2.py
@@ -1,2 +1,3 @@
 def bar():
     return 3
+    # added comment
"#;
        let action = parse_patch(patch).unwrap();
        assert_eq!(action.file_count(), 2);
    }

    #[test]
    fn apply_hunks_simple_update() {
        let original = "line 1\nold line 2\nline 3\n";
        let hunk = Hunk {
            old_start: 2,
            old_count: 1,
            new_start: 2,
            new_count: 1,
            lines: vec![
                HunkLine {
                    kind: HunkLineKind::Remove,
                    content: "old line 2".to_string(),
                },
                HunkLine {
                    kind: HunkLineKind::Add,
                    content: "new line 2".to_string(),
                },
            ],
        };

        let result = apply_hunks_to_content(original, &[hunk], Path::new("test.txt")).unwrap();
        assert!(result.contains("new line 2"));
        assert!(!result.contains("old line 2"));
    }

    #[test]
    fn apply_hunks_add_lines() {
        let original = "line 1\nline 2\nline 3\n";
        let hunk = Hunk {
            old_start: 2,
            old_count: 1,
            new_start: 2,
            new_count: 3,
            lines: vec![
                HunkLine {
                    kind: HunkLineKind::Context,
                    content: "line 2".to_string(),
                },
                HunkLine {
                    kind: HunkLineKind::Add,
                    content: "new line a".to_string(),
                },
                HunkLine {
                    kind: HunkLineKind::Add,
                    content: "new line b".to_string(),
                },
            ],
        };

        let result = apply_hunks_to_content(original, &[hunk], Path::new("test.txt")).unwrap();
        assert!(result.contains("new line a"));
        assert!(result.contains("new line b"));
    }

    #[test]
    fn patch_summary_format() {
        let action = PatchAction {
            changes: vec![(
                PathBuf::from("file.py"),
                FileChange::Update {
                    hunks: vec![Hunk {
                        old_start: 1,
                        old_count: 1,
                        new_start: 1,
                        new_count: 2,
                        lines: vec![
                            HunkLine {
                                kind: HunkLineKind::Remove,
                                content: "old".to_string(),
                            },
                            HunkLine {
                                kind: HunkLineKind::Add,
                                content: "new".to_string(),
                            },
                            HunkLine {
                                kind: HunkLineKind::Add,
                                content: "extra".to_string(),
                            },
                        ],
                    }],
                },
            )],
        };
        let summary = action.summary();
        assert!(summary.contains("update file.py"));
        assert!(summary.contains("+2/-1"));
    }

    #[test]
    fn hunk_header_parsing() {
        let (s, c, ns, nc) = parse_hunk_header("@@ -10,3 +12,5 @@ some context").unwrap();
        assert_eq!((s, c, ns, nc), (10, 3, 12, 5));
    }

    #[test]
    fn hunk_header_single_line() {
        let (s, c, ns, nc) = parse_hunk_header("@@ -5 +5,2 @@").unwrap();
        assert_eq!((s, c, ns, nc), (5, 1, 5, 2));
    }

    #[test]
    fn validate_patch_returns_action() {
        let patch = "--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,1 @@\n-old\n+new\n";
        let action = validate_patch(patch).unwrap();
        assert_eq!(action.file_count(), 1);
    }
}
