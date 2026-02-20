//! Truncation utilities — adapted from OpenAI Codex CLI.
//!
//! Provides `TruncationPolicy` for token/byte-based truncation with
//! UTF-8-safe middle truncation (keeps prefix + suffix, cuts the middle).

/// Approximate bytes per token (industry standard heuristic).
const APPROX_BYTES_PER_TOKEN: usize = 4;

/// Policy controlling how text is truncated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TruncationPolicy {
    /// Hard byte limit.
    Bytes(usize),
    /// Token-based limit (uses 4 bytes/token heuristic).
    Tokens(usize),
}

impl TruncationPolicy {
    /// Returns a token budget derived from this policy.
    pub fn token_budget(&self) -> usize {
        match self {
            Self::Tokens(t) => *t,
            Self::Bytes(b) => *b / APPROX_BYTES_PER_TOKEN,
        }
    }

    /// Returns a byte budget derived from this policy.
    pub fn byte_budget(&self) -> usize {
        match self {
            Self::Bytes(b) => *b,
            Self::Tokens(t) => *t * APPROX_BYTES_PER_TOKEN,
        }
    }
}

/// Approximate token count for a string (byte length / 4).
pub fn approx_token_count(text: &str) -> usize {
    text.len() / APPROX_BYTES_PER_TOKEN
}

/// Approximate bytes for a given token count.
pub fn approx_bytes_for_tokens(tokens: usize) -> usize {
    tokens * APPROX_BYTES_PER_TOKEN
}

/// Truncate text using middle truncation — preserves prefix and suffix,
/// cuts the middle. Returns the truncated string.
///
/// This is much more information-preserving than head-only truncation
/// because the agent sees both the beginning (setup) and end (conclusion).
pub fn truncate_text(content: &str, policy: TruncationPolicy) -> String {
    let budget = policy.byte_budget();
    if content.len() <= budget {
        return content.to_string();
    }

    let (prefix_budget, suffix_budget) = split_budget(budget);
    let (removed, prefix, suffix) = split_string(content, prefix_budget, suffix_budget);

    if removed == 0 {
        return content.to_string();
    }

    let marker = format_truncation_marker(policy, removed);
    format!("{prefix}{marker}{suffix}")
}

/// Truncate text and prepend a formatted header showing original vs truncated size.
pub fn formatted_truncate_text(content: &str, policy: TruncationPolicy) -> String {
    let budget = policy.byte_budget();
    if content.len() <= budget {
        return content.to_string();
    }

    let original_tokens = approx_token_count(content);
    let truncated = truncate_text(content, policy);
    let truncated_tokens = approx_token_count(&truncated);
    format!("[Truncated: {original_tokens} → {truncated_tokens} tokens]\n{truncated}")
}

/// Split a budget into prefix (60%) and suffix (40%) portions.
fn split_budget(budget: usize) -> (usize, usize) {
    let prefix = (budget * 3) / 5; // 60%
    let suffix = budget - prefix; // 40%
    (prefix, suffix)
}

/// Split a string into prefix and suffix on UTF-8 boundaries.
/// Returns (removed_bytes, prefix_str, suffix_str).
fn split_string(s: &str, prefix_bytes: usize, suffix_bytes: usize) -> (usize, &str, &str) {
    if s.len() <= prefix_bytes + suffix_bytes {
        return (0, s, "");
    }

    // Find UTF-8 safe boundary for prefix
    let prefix_end = find_utf8_boundary(s, prefix_bytes, true);
    // Find UTF-8 safe boundary for suffix
    let suffix_start = find_utf8_boundary(s, s.len().saturating_sub(suffix_bytes), false);

    if suffix_start <= prefix_end {
        return (0, s, "");
    }

    let removed = suffix_start - prefix_end;
    (removed, &s[..prefix_end], &s[suffix_start..])
}

/// Find a UTF-8 safe boundary near the target byte position.
/// If `round_down` is true, rounds down; otherwise rounds up.
fn find_utf8_boundary(s: &str, target: usize, round_down: bool) -> usize {
    let target = target.min(s.len());
    if s.is_char_boundary(target) {
        return target;
    }
    if round_down {
        (0..=target)
            .rev()
            .find(|&i| s.is_char_boundary(i))
            .unwrap_or(0)
    } else {
        (target..=s.len())
            .find(|&i| s.is_char_boundary(i))
            .unwrap_or(s.len())
    }
}

/// Format the truncation marker showing removed content.
fn format_truncation_marker(policy: TruncationPolicy, removed_bytes: usize) -> String {
    match policy {
        TruncationPolicy::Tokens(_) => {
            let removed_tokens = removed_bytes / APPROX_BYTES_PER_TOKEN;
            format!("\n\n[...{removed_tokens} tokens truncated...]\n\n")
        }
        TruncationPolicy::Bytes(_) => {
            format!("\n\n[...{removed_bytes} bytes truncated...]\n\n")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_truncation_when_within_budget() {
        let text = "hello world";
        let result = truncate_text(text, TruncationPolicy::Tokens(100));
        assert_eq!(result, text);
    }

    #[test]
    fn middle_truncation_preserves_prefix_and_suffix() {
        let text = "AAAA_BBBB_CCCC_DDDD_EEEE";
        let result = truncate_text(text, TruncationPolicy::Bytes(16));
        assert!(result.starts_with("AAAA_"));
        assert!(result.ends_with("EEEE"));
        assert!(result.contains("truncated"));
    }

    #[test]
    fn utf8_safe_truncation() {
        let text = "Hello 🌍 World 🎉 End";
        let result = truncate_text(text, TruncationPolicy::Bytes(16));
        // Should not panic on UTF-8 boundaries
        assert!(!result.is_empty());
    }

    #[test]
    fn approx_token_count_reasonable() {
        let text = "word ".repeat(100);
        let tokens = approx_token_count(&text);
        assert!(tokens > 50 && tokens < 200);
    }

    #[test]
    fn formatted_truncate_adds_header() {
        let text = "x".repeat(1000);
        let result = formatted_truncate_text(&text, TruncationPolicy::Bytes(100));
        assert!(result.contains("Truncated:"));
    }

    #[test]
    fn split_budget_60_40() {
        let (prefix, suffix) = split_budget(100);
        assert_eq!(prefix, 60);
        assert_eq!(suffix, 40);
    }
}
