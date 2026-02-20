//! Phase 1: Memory Extraction
//!
//! After a session ends, extracts structured memories from the conversation
//! transcript via an LLM call. Outputs are saved as rollout summaries and
//! raw memories for later consolidation.

use serde::{Deserialize, Serialize};

use super::storage;

/// Phase 1 extraction output — structured JSON from the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractionOutput {
    /// Comprehensive summary of the session.
    pub rollout_summary: String,
    /// Structured memory entries as bullet points.
    pub raw_memory: String,
    /// Comma-separated searchable keywords.
    pub keywords: String,
}

impl ExtractionOutput {
    /// Check if this extraction has meaningful content.
    pub fn is_empty(&self) -> bool {
        self.rollout_summary.is_empty() && self.raw_memory.is_empty() && self.keywords.is_empty()
    }
}

/// Build the extraction prompt with the session transcript injected.
pub fn build_extraction_prompt(session_transcript: &str) -> String {
    super::EXTRACTION_PROMPT.replace("{{ session_transcript }}", session_transcript)
}

/// Parse the LLM's extraction output.
pub fn parse_extraction_output(llm_response: &str) -> Option<ExtractionOutput> {
    // Try to parse as JSON directly
    if let Ok(output) = serde_json::from_str::<ExtractionOutput>(llm_response) {
        if !output.is_empty() {
            return Some(output);
        }
    }

    // Try to extract JSON from markdown code block
    let trimmed = llm_response.trim();
    if let Some(json_start) = trimmed.find('{') {
        if let Some(json_end) = trimmed.rfind('}') {
            let json_str = &trimmed[json_start..=json_end];
            if let Ok(output) = serde_json::from_str::<ExtractionOutput>(json_str) {
                if !output.is_empty() {
                    return Some(output);
                }
            }
        }
    }

    None
}

/// Save extraction output to the memory directory.
pub fn save_extraction(
    memory_root: &std::path::Path,
    session_id: &str,
    output: &ExtractionOutput,
) -> std::io::Result<()> {
    super::ensure_layout(memory_root)?;

    // Save rollout summary
    if !output.rollout_summary.is_empty() {
        let slug = session_id
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect::<String>();
        storage::save_rollout_summary(memory_root, &slug, &output.rollout_summary)?;
    }

    // Append raw memory
    if !output.raw_memory.is_empty() {
        storage::append_raw_memory(memory_root, &output.raw_memory)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_json() {
        let json =
            r#"{"rollout_summary": "Did X", "raw_memory": "- learned Y", "keywords": "x, y"}"#;
        let result = parse_extraction_output(json);
        assert!(result.is_some());
        let output = result.unwrap();
        assert_eq!(output.rollout_summary, "Did X");
    }

    #[test]
    fn parse_empty_returns_none() {
        let json = r#"{"rollout_summary": "", "raw_memory": "", "keywords": ""}"#;
        let result = parse_extraction_output(json);
        assert!(result.is_none());
    }

    #[test]
    fn parse_json_in_markdown() {
        let response = "Here's the output:\n```json\n{\"rollout_summary\": \"summary\", \"raw_memory\": \"mem\", \"keywords\": \"k\"}\n```";
        let result = parse_extraction_output(response);
        assert!(result.is_some());
    }
}
