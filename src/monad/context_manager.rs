//! Multi-context isolation manager.
//!
//! Provides named, isolated workspaces for agent data analysis.
//! Each context has its own content buffer, metadata, and can be
//! searched/peeked independently — keeping large data out of the
//! conversation history and the LLM's limited attention.
//!
//! Phase 2: Auto-load large results into contexts.
//! Phase 3: Full context API (search, peek, diff, chunk, cross-search).

use std::collections::HashMap;

/// Default threshold (in chars) above which execution output is
/// auto-loaded into a context instead of being placed inline.
pub const DEFAULT_AUTO_LOAD_THRESHOLD: usize = 10_000;

/// Metadata about an isolated context's content.
#[derive(Debug, Clone)]
pub struct ContextMetadata {
    /// Detected format ("text", "json", "csv", etc.).
    pub format: String,
    /// Size in bytes.
    pub size_bytes: usize,
    /// Number of lines.
    pub line_count: usize,
    /// Rough token estimate (size / 4).
    pub token_estimate: usize,
}

/// A single search result within a context.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Line number (1-indexed).
    pub line_number: usize,
    /// The matching line content.
    pub line_content: String,
}

/// An isolated workspace for data the agent is analyzing.
///
/// Contexts keep large data out of the conversation history.
/// The agent can search, peek, and diff contexts without polluting
/// the LLM's context window.
#[derive(Debug, Clone)]
pub struct IsolatedContext {
    /// Unique identifier for this context.
    pub id: String,
    /// The full content (may be large).
    pub content: String,
    /// Metadata about the content.
    pub metadata: ContextMetadata,
}

/// Manages multiple isolated contexts.
///
/// The `ContextManager` is owned by `AgentContext` and provides
/// the backing store for auto-loaded large results (Phase 2)
/// and explicit context operations (Phase 3).
#[derive(Debug, Default)]
pub struct ContextManager {
    contexts: HashMap<String, IsolatedContext>,
}

impl ContextManager {
    /// Create a new empty context manager.
    pub fn new() -> Self {
        Self {
            contexts: HashMap::new(),
        }
    }

    /// Load content into a named context, returning metadata.
    ///
    /// If a context with this ID already exists, it is replaced.
    pub fn load(&mut self, id: &str, content: &str) -> ContextMetadata {
        let metadata = ContextMetadata {
            format: detect_format(content),
            size_bytes: content.len(),
            line_count: content.lines().count(),
            token_estimate: content.len() / 4,
        };
        let ctx = IsolatedContext {
            id: id.to_string(),
            content: content.to_string(),
            metadata: metadata.clone(),
        };
        self.contexts.insert(id.to_string(), ctx);
        metadata
    }

    /// Peek at a range of lines (1-indexed, inclusive) in a context.
    ///
    /// Returns `None` if the context doesn't exist.
    pub fn peek(&self, id: &str, start: usize, end: usize) -> Option<String> {
        let ctx = self.contexts.get(id)?;
        let lines: Vec<&str> = ctx.content.lines().collect();
        let start_idx = start.saturating_sub(1);
        let end_idx = end.min(lines.len());
        if start_idx >= lines.len() || start_idx >= end_idx {
            return Some(String::new());
        }
        Some(lines[start_idx..end_idx].join("\n"))
    }

    /// Search for a pattern (case-insensitive substring) within a context.
    ///
    /// Returns matching lines with their line numbers.
    pub fn search(&self, id: &str, pattern: &str) -> Vec<SearchResult> {
        let ctx = match self.contexts.get(id) {
            Some(c) => c,
            None => return Vec::new(),
        };
        let pattern_lower = pattern.to_lowercase();
        ctx.content
            .lines()
            .enumerate()
            .filter(|(_, line)| line.to_lowercase().contains(&pattern_lower))
            .map(|(i, line)| SearchResult {
                line_number: i + 1,
                line_content: line.to_string(),
            })
            .collect()
    }

    /// Search across all contexts for a pattern.
    ///
    /// Returns (context_id, search_results) pairs.
    pub fn cross_search(&self, pattern: &str) -> Vec<(String, Vec<SearchResult>)> {
        self.contexts
            .keys()
            .filter_map(|id| {
                let results = self.search(id, pattern);
                if results.is_empty() {
                    None
                } else {
                    Some((id.clone(), results))
                }
            })
            .collect()
    }

    /// Diff two contexts, producing a simple line-level diff.
    pub fn diff(&self, a_id: &str, b_id: &str) -> Option<String> {
        let a = self.contexts.get(a_id)?;
        let b = self.contexts.get(b_id)?;

        let a_lines: Vec<&str> = a.content.lines().collect();
        let b_lines: Vec<&str> = b.content.lines().collect();

        let mut result = Vec::new();
        let max_lines = a_lines.len().max(b_lines.len());

        for i in 0..max_lines {
            match (a_lines.get(i), b_lines.get(i)) {
                (Some(al), Some(bl)) if al == bl => {
                    result.push(format!("  {al}"));
                }
                (Some(al), Some(bl)) => {
                    result.push(format!("- {al}"));
                    result.push(format!("+ {bl}"));
                }
                (Some(al), None) => {
                    result.push(format!("- {al}"));
                }
                (None, Some(bl)) => {
                    result.push(format!("+ {bl}"));
                }
                (None, None) => {}
            }
        }

        Some(result.join("\n"))
    }

    /// Split a context's content into chunks of `size` chars with `overlap`.
    pub fn chunk(&self, id: &str, size: usize, overlap: usize) -> Option<Vec<String>> {
        let ctx = self.contexts.get(id)?;
        let content = &ctx.content;
        if content.is_empty() || size == 0 {
            return Some(Vec::new());
        }

        let mut chunks = Vec::new();
        let mut start = 0;
        let step = size.saturating_sub(overlap).max(1);

        while start < content.len() {
            let end = (start + size).min(content.len());
            chunks.push(content[start..end].to_string());
            start += step;
            if end == content.len() {
                break;
            }
        }

        Some(chunks)
    }

    /// List all loaded contexts with their metadata.
    pub fn list(&self) -> Vec<(&str, &ContextMetadata)> {
        self.contexts
            .iter()
            .map(|(id, ctx)| (id.as_str(), &ctx.metadata))
            .collect()
    }

    /// Get the content of a context.
    pub fn get(&self, id: &str) -> Option<&str> {
        self.contexts.get(id).map(|c| c.content.as_str())
    }

    /// Remove a context, returning true if it existed.
    pub fn remove(&mut self, id: &str) -> bool {
        self.contexts.remove(id).is_some()
    }

    /// Check if a context exists.
    pub fn contains(&self, id: &str) -> bool {
        self.contexts.contains_key(id)
    }

    /// Number of loaded contexts.
    pub fn len(&self) -> usize {
        self.contexts.len()
    }

    /// Whether no contexts are loaded.
    pub fn is_empty(&self) -> bool {
        self.contexts.is_empty()
    }
}

/// Detect the format of content based on simple heuristics.
fn detect_format(content: &str) -> String {
    let trimmed = content.trim();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        "json".to_string()
    } else if trimmed.contains(',') && trimmed.lines().count() > 1 {
        // Very rough CSV heuristic
        let first_line_commas = trimmed.lines().next().map_or(0, |l| l.matches(',').count());
        if first_line_commas > 0 {
            "csv".to_string()
        } else {
            "text".to_string()
        }
    } else {
        "text".to_string()
    }
}

/// Format search results for display to the LLM.
pub fn format_search_results(results: &[SearchResult]) -> String {
    if results.is_empty() {
        return "No matches found.".to_string();
    }
    results
        .iter()
        .map(|r| format!("L{}: {}", r.line_number, r.line_content))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_and_peek() {
        let mut mgr = ContextManager::new();
        let meta = mgr.load("test", "line1\nline2\nline3\nline4\nline5");
        assert_eq!(meta.line_count, 5);
        assert_eq!(meta.format, "text");

        let peek = mgr.peek("test", 2, 4).unwrap();
        assert_eq!(peek, "line2\nline3\nline4");
    }

    #[test]
    fn peek_nonexistent() {
        let mgr = ContextManager::new();
        assert!(mgr.peek("nope", 1, 5).is_none());
    }

    #[test]
    fn peek_out_of_range() {
        let mut mgr = ContextManager::new();
        mgr.load("test", "a\nb\nc");
        let peek = mgr.peek("test", 10, 20).unwrap();
        assert!(peek.is_empty());
    }

    #[test]
    fn search_finds_matches() {
        let mut mgr = ContextManager::new();
        mgr.load("test", "hello world\nfoo bar\nhello again\nbaz");

        let results = mgr.search("test", "hello");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].line_number, 1);
        assert_eq!(results[1].line_number, 3);
    }

    #[test]
    fn search_case_insensitive() {
        let mut mgr = ContextManager::new();
        mgr.load("test", "Hello World\nHELLO AGAIN");

        let results = mgr.search("test", "hello");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn search_no_matches() {
        let mut mgr = ContextManager::new();
        mgr.load("test", "foo bar baz");
        assert!(mgr.search("test", "xyz").is_empty());
    }

    #[test]
    fn cross_search() {
        let mut mgr = ContextManager::new();
        mgr.load("ctx1", "TODO: fix this\ndone");
        mgr.load("ctx2", "no match here");
        mgr.load("ctx3", "another TODO item");

        let results = mgr.cross_search("TODO");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn diff_contexts() {
        let mut mgr = ContextManager::new();
        mgr.load("a", "same\ndifferent_a\nsame2");
        mgr.load("b", "same\ndifferent_b\nsame2");

        let diff = mgr.diff("a", "b").unwrap();
        assert!(diff.contains("- different_a"));
        assert!(diff.contains("+ different_b"));
        assert!(diff.contains("  same"));
    }

    #[test]
    fn chunk_content() {
        let mut mgr = ContextManager::new();
        mgr.load("test", "0123456789abcdef");

        let chunks = mgr.chunk("test", 8, 2).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "01234567");
        assert_eq!(chunks[1], "6789abcd");
        assert_eq!(chunks[2], "cdef");
    }

    #[test]
    fn list_and_remove() {
        let mut mgr = ContextManager::new();
        mgr.load("a", "content_a");
        mgr.load("b", "content_b");

        assert_eq!(mgr.len(), 2);
        assert!(mgr.remove("a"));
        assert_eq!(mgr.len(), 1);
        assert!(!mgr.contains("a"));
        assert!(mgr.contains("b"));
    }

    #[test]
    fn detect_json_format() {
        let mut mgr = ContextManager::new();
        let meta = mgr.load("j", "{\"key\": \"value\"}");
        assert_eq!(meta.format, "json");
    }

    #[test]
    fn detect_csv_format() {
        let mut mgr = ContextManager::new();
        let meta = mgr.load("c", "name,age\nAlice,30\nBob,25");
        assert_eq!(meta.format, "csv");
    }

    #[test]
    fn format_search_results_empty() {
        assert_eq!(format_search_results(&[]), "No matches found.");
    }

    #[test]
    fn format_search_results_with_matches() {
        let results = vec![
            SearchResult {
                line_number: 5,
                line_content: "TODO: fix".to_string(),
            },
            SearchResult {
                line_number: 12,
                line_content: "TODO: refactor".to_string(),
            },
        ];
        let formatted = format_search_results(&results);
        assert!(formatted.contains("L5:"));
        assert!(formatted.contains("L12:"));
    }
}
