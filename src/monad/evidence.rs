//! Evidence tracking with provenance.
//!
//! Every significant action (code execution, LLM inference, search, etc.)
//! is automatically recorded as `Evidence` with full provenance metadata.
//! This enables cited conclusions, GEPA evaluation scoring based on
//! evidence quality, and post-hoc analysis of agent reasoning.

use chrono::{DateTime, Utc};
use std::fmt;

/// The source/type of an evidence record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceSource {
    /// Code was executed in the sandbox.
    CodeExec,
    /// LLM inference produced a response.
    ModelInference,
    /// A sub-agent produced a result.
    SubAgent,
    /// A search was performed (context or filesystem).
    Search,
    /// A context peek returned data.
    ContextPeek,
    /// Structured reasoning (Think action).
    Think,
    /// Self-assessment (EvaluateProgress action).
    EvaluateProgress,
}

impl fmt::Display for EvidenceSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CodeExec => write!(f, "code_exec"),
            Self::ModelInference => write!(f, "model_inference"),
            Self::SubAgent => write!(f, "sub_agent"),
            Self::Search => write!(f, "search"),
            Self::ContextPeek => write!(f, "context_peek"),
            Self::Think => write!(f, "think"),
            Self::EvaluateProgress => write!(f, "evaluate_progress"),
        }
    }
}

/// A single piece of evidence gathered during agent execution.
///
/// Evidence is automatically recorded by the interpreter in
/// `AgentContext::interpret_action()` for key actions. It captures
/// what was found, where it came from, and when.
#[derive(Debug, Clone)]
pub struct Evidence {
    /// What kind of action produced this evidence.
    pub source: EvidenceSource,
    /// Which isolated context this came from (if applicable).
    pub context_id: Option<String>,
    /// The actual evidence content (truncated for large outputs).
    pub snippet: String,
    /// Line range within a context or file (if applicable).
    pub line_range: Option<(usize, usize)>,
    /// Search pattern that found this (if applicable).
    pub pattern: Option<String>,
    /// Exit code from code execution (if applicable).
    pub exit_code: Option<i32>,
    /// When this evidence was recorded.
    pub timestamp: DateTime<Utc>,
}

/// Maximum snippet length stored in evidence (to avoid bloat).
const MAX_SNIPPET_LENGTH: usize = 500;

impl Evidence {
    /// Create a new evidence record for code execution.
    pub fn from_code_exec(output: &str, exit_code: Option<i32>) -> Self {
        Self {
            source: EvidenceSource::CodeExec,
            context_id: None,
            snippet: truncate_snippet(output),
            line_range: None,
            pattern: None,
            exit_code,
            timestamp: Utc::now(),
        }
    }

    /// Create a new evidence record for model inference.
    pub fn from_inference(response: &str) -> Self {
        Self {
            source: EvidenceSource::ModelInference,
            context_id: None,
            snippet: truncate_snippet(response),
            line_range: None,
            pattern: None,
            exit_code: None,
            timestamp: Utc::now(),
        }
    }

    /// Create a new evidence record for a sub-agent result.
    pub fn from_sub_agent(result: &str) -> Self {
        Self {
            source: EvidenceSource::SubAgent,
            context_id: None,
            snippet: truncate_snippet(result),
            line_range: None,
            pattern: None,
            exit_code: None,
            timestamp: Utc::now(),
        }
    }

    /// Create a new evidence record for a search result.
    pub fn from_search(context_id: &str, pattern: &str, results: &str) -> Self {
        Self {
            source: EvidenceSource::Search,
            context_id: Some(context_id.to_string()),
            snippet: truncate_snippet(results),
            line_range: None,
            pattern: Some(pattern.to_string()),
            exit_code: None,
            timestamp: Utc::now(),
        }
    }

    /// Create a new evidence record for structured reasoning.
    pub fn from_think(reasoning: &str) -> Self {
        Self {
            source: EvidenceSource::Think,
            context_id: None,
            snippet: truncate_snippet(reasoning),
            line_range: None,
            pattern: None,
            exit_code: None,
            timestamp: Utc::now(),
        }
    }

    /// Create a new evidence record for self-assessment.
    pub fn from_evaluate_progress(confidence: f64, remaining: &str) -> Self {
        Self {
            source: EvidenceSource::EvaluateProgress,
            context_id: None,
            snippet: format!("confidence={confidence:.2}, remaining: {remaining}"),
            line_range: None,
            pattern: None,
            exit_code: None,
            timestamp: Utc::now(),
        }
    }
}

/// Generate a human-readable summary of an evidence trail.
///
/// Groups evidence by source type and shows counts + latest snippet.
pub fn summarize_evidence(evidence: &[Evidence]) -> String {
    if evidence.is_empty() {
        return "No evidence collected.".to_string();
    }

    let mut by_source: std::collections::HashMap<String, Vec<&Evidence>> =
        std::collections::HashMap::new();
    for e in evidence {
        by_source.entry(e.source.to_string()).or_default().push(e);
    }

    let mut lines = vec![format!("Evidence summary ({} items):", evidence.len())];
    for (source, items) in &by_source {
        lines.push(format!("  {source}: {} item(s)", items.len()));
        if let Some(latest) = items.last() {
            let preview = if latest.snippet.len() > 80 {
                format!("{}...", &latest.snippet[..80])
            } else {
                latest.snippet.clone()
            };
            lines.push(format!("    latest: {preview}"));
        }
    }
    lines.join("\n")
}

/// Truncate a snippet to MAX_SNIPPET_LENGTH, appending "..." if truncated.
fn truncate_snippet(s: &str) -> String {
    if s.len() <= MAX_SNIPPET_LENGTH {
        s.to_string()
    } else {
        format!("{}...", &s[..MAX_SNIPPET_LENGTH])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evidence_from_code_exec() {
        let e = Evidence::from_code_exec("hello world\n", Some(0));
        assert_eq!(e.source, EvidenceSource::CodeExec);
        assert_eq!(e.snippet, "hello world\n");
        assert_eq!(e.exit_code, Some(0));
    }

    #[test]
    fn evidence_from_inference() {
        let e = Evidence::from_inference("I'll solve this by...");
        assert_eq!(e.source, EvidenceSource::ModelInference);
        assert!(e.snippet.contains("solve"));
    }

    #[test]
    fn evidence_snippet_truncation() {
        let long = "x".repeat(1000);
        let e = Evidence::from_code_exec(&long, None);
        assert_eq!(e.snippet.len(), MAX_SNIPPET_LENGTH + 3); // +3 for "..."
        assert!(e.snippet.ends_with("..."));
    }

    #[test]
    fn summarize_empty() {
        assert_eq!(summarize_evidence(&[]), "No evidence collected.");
    }

    #[test]
    fn summarize_groups_by_source() {
        let evidence = vec![
            Evidence::from_code_exec("result1", Some(0)),
            Evidence::from_code_exec("result2", Some(0)),
            Evidence::from_inference("response"),
        ];
        let summary = summarize_evidence(&evidence);
        assert!(summary.contains("3 items"));
        assert!(summary.contains("code_exec: 2 item(s)"));
        assert!(summary.contains("model_inference: 1 item(s)"));
    }

    #[test]
    fn evidence_from_search() {
        let e = Evidence::from_search("ctx1", "TODO", "line 42: TODO fix this");
        assert_eq!(e.source, EvidenceSource::Search);
        assert_eq!(e.context_id, Some("ctx1".to_string()));
        assert_eq!(e.pattern, Some("TODO".to_string()));
    }

    #[test]
    fn evidence_from_think() {
        let e = Evidence::from_think("I should try approach B because...");
        assert_eq!(e.source, EvidenceSource::Think);
        assert!(e.snippet.contains("approach B"));
    }

    #[test]
    fn evidence_from_evaluate_progress() {
        let e = Evidence::from_evaluate_progress(0.75, "need to verify edge case");
        assert_eq!(e.source, EvidenceSource::EvaluateProgress);
        assert!(e.snippet.contains("0.75"));
        assert!(e.snippet.contains("edge case"));
    }
}
