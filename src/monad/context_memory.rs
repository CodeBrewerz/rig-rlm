//! Agent-driven context memory — virtual memory for the context window.
//!
//! The agent manages its own context by offloading conversation segments
//! to an in-memory store and recalling them when needed.
//!
//! Threading model:
//! - The store lives behind `Arc<Mutex<...>>`, shared with the bridge dispatch.
//! - During code execution (when Python calls memory_*), the bg thread is blocked,
//!   so the Mutex is uncontended.
//! - Segments store raw string content (JSON-serialized by the Python side).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A stored segment of conversation content.
#[derive(Debug, Clone)]
pub struct MemorySegment {
    pub id: String,
    pub label: String,
    /// Raw content — could be JSON array of messages, or plain text summary.
    pub content: String,
    /// One-line summary for the manifest.
    pub summary: String,
}

/// In-memory store for context segments.
#[derive(Debug)]
pub struct ContextMemoryStore {
    segments: HashMap<String, MemorySegment>,
    next_id: usize,
}

pub type SharedMemoryStore = Arc<Mutex<ContextMemoryStore>>;

impl ContextMemoryStore {
    pub fn new() -> Self {
        Self {
            segments: HashMap::new(),
            next_id: 1,
        }
    }

    pub fn new_shared() -> SharedMemoryStore {
        Arc::new(Mutex::new(Self::new()))
    }

    /// Store content as a new segment. Returns JSON result.
    pub fn offload(&mut self, label: &str, content: String) -> String {
        let seg_id = format!("seg_{}", self.next_id);
        self.next_id += 1;

        // Auto-generate summary from first 120 chars
        let summary = if content.len() > 120 {
            format!("{}...", &content[..120])
        } else {
            content.clone()
        };

        self.segments.insert(
            seg_id.clone(),
            MemorySegment {
                id: seg_id.clone(),
                label: label.to_string(),
                content,
                summary,
            },
        );

        serde_json::json!({
            "segment_id": seg_id,
            "label": label,
            "status": "stored"
        })
        .to_string()
    }

    /// Retrieve a segment's content. Returns the content string.
    /// Does NOT remove the segment (use `forget` for that).
    pub fn recall(&self, segment_id: &str) -> String {
        match self.segments.get(segment_id) {
            Some(seg) => serde_json::json!({
                "segment_id": segment_id,
                "label": seg.label,
                "content": seg.content,
            })
            .to_string(),
            None => serde_json::json!({
                "error": format!("segment '{}' not found", segment_id)
            })
            .to_string(),
        }
    }

    /// Remove a segment after recall (free the memory).
    pub fn forget(&mut self, segment_id: &str) -> bool {
        self.segments.remove(segment_id).is_some()
    }

    /// Return a manifest of all stored segments.
    pub fn manifest(&self) -> String {
        let entries: Vec<serde_json::Value> = self
            .segments
            .values()
            .map(|seg| {
                serde_json::json!({
                    "id": seg.id,
                    "label": seg.label,
                    "summary": seg.summary,
                })
            })
            .collect();

        serde_json::to_string_pretty(&entries).unwrap_or_else(|_| "[]".to_string())
    }

    /// Search all stored segments for a query string (case-insensitive).
    pub fn search(&self, query: &str) -> String {
        let query_lower = query.to_lowercase();
        let mut results: Vec<serde_json::Value> = Vec::new();

        for seg in self.segments.values() {
            if seg.content.to_lowercase().contains(&query_lower)
                || seg.label.to_lowercase().contains(&query_lower)
            {
                // Extract matching lines
                let matches: Vec<&str> = seg
                    .content
                    .lines()
                    .filter(|line| line.to_lowercase().contains(&query_lower))
                    .take(5)
                    .collect();

                results.push(serde_json::json!({
                    "segment_id": seg.id,
                    "label": seg.label,
                    "match_count": matches.len(),
                    "matches": matches,
                }));
            }
        }

        serde_json::to_string_pretty(&results).unwrap_or_else(|_| "[]".to_string())
    }
}
