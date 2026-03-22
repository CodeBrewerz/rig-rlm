//! Topic-based pub/sub routing with glob matching and hierarchical propagation.
//!
//! Topics are `/`-separated hierarchical paths:
//!   - `chat/telegram/12345` — message from Telegram chat
//!   - `ci/build/main`       — CI build event on main
//!   - `alerts/critical`     — critical monitoring alert
//!
//! ## Subscription patterns
//!
//! - `ci/build/main` — exact match
//! - `ci/*`          — single-level wildcard (matches `ci/build` but NOT `ci/build/main`)
//! - `ci/**`         — recursive wildcard (matches `ci/build/main`, `ci/deploy/staging/v2`)
//! - `*`             — catch-all (matches every topic)
//!
//! ## Hierarchical propagation
//!
//! Events auto-propagate up the topic tree. Publishing to `chat/telegram/123`
//! also matches subscribers at `chat/telegram/*` and `chat/**`.

use std::fmt;

/// A topic subscription pattern with glob matching.
///
/// Supports exact matches, single-level wildcards (`*`), and
/// recursive wildcards (`**`). Patterns are compared segment-by-segment
/// using the `/` delimiter.
#[derive(Debug, Clone)]
pub struct TopicFilter {
    /// The raw pattern string.
    pattern: String,
    /// Pre-split segments for fast matching.
    segments: Vec<Segment>,
}

/// A parsed segment of a topic filter.
#[derive(Debug, Clone, PartialEq)]
enum Segment {
    /// Exact text match.
    Literal(String),
    /// `*` — matches any single segment.
    SingleWild,
    /// `**` — matches zero or more segments (recursive).
    RecursiveWild,
}

impl TopicFilter {
    /// Create a new topic filter from a pattern string.
    ///
    /// # Examples
    /// ```
    /// let f = TopicFilter::new("ci/*");
    /// assert!(f.matches("ci/build"));
    /// assert!(!f.matches("ci/build/main"));
    /// ```
    pub fn new(pattern: impl Into<String>) -> Self {
        let pattern: String = pattern.into();
        let segments = pattern
            .split('/')
            .map(|s| match s {
                "*" => Segment::SingleWild,
                "**" => Segment::RecursiveWild,
                other => Segment::Literal(other.to_string()),
            })
            .collect();
        Self { pattern, segments }
    }

    /// Test whether a topic path matches this filter.
    ///
    /// Matching is done segment-by-segment:
    /// - `Literal("x")` matches only `"x"`
    /// - `SingleWild` matches any single segment
    /// - `RecursiveWild` matches zero or more remaining segments
    pub fn matches(&self, topic: &str) -> bool {
        let topic_segments: Vec<&str> = topic.split('/').collect();
        Self::match_segments(&self.segments, &topic_segments)
    }

    /// Recursive segment matching with backtracking for `**`.
    fn match_segments(filter: &[Segment], topic: &[&str]) -> bool {
        match (filter.first(), topic.first()) {
            // Both empty → match
            (None, None) => true,

            // Filter exhausted but topic has remaining segments → no match
            (None, Some(_)) => false,

            // `**` at the end of filter → matches everything remaining
            (Some(Segment::RecursiveWild), _) if filter.len() == 1 => true,

            // `**` with more filter segments → try skipping 0..N topic segments
            (Some(Segment::RecursiveWild), _) => {
                let rest_filter = &filter[1..];
                // Try matching rest_filter against topic[i..] for each i
                for i in 0..=topic.len() {
                    if Self::match_segments(rest_filter, &topic[i..]) {
                        return true;
                    }
                }
                false
            }

            // Topic exhausted but filter has remaining non-** segments → no match
            (Some(Segment::Literal(_) | Segment::SingleWild), None) => false,

            // `*` matches any single segment
            (Some(Segment::SingleWild), Some(_)) => {
                Self::match_segments(&filter[1..], &topic[1..])
            }

            // Literal must match exactly
            (Some(Segment::Literal(expected)), Some(actual)) => {
                expected == actual && Self::match_segments(&filter[1..], &topic[1..])
            }
        }
    }

    /// Get the raw pattern string.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }
}

impl fmt::Display for TopicFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.pattern)
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Exact match ─────────────────────────────────────────────

    #[test]
    fn exact_match() {
        let f = TopicFilter::new("ci/build/main");
        assert!(f.matches("ci/build/main"));
        assert!(!f.matches("ci/build"));
        assert!(!f.matches("ci/build/main/extra"));
        assert!(!f.matches("ci/deploy/main"));
    }

    // ── Single-level wildcard ───────────────────────────────────

    #[test]
    fn single_wildcard() {
        let f = TopicFilter::new("ci/*");
        assert!(f.matches("ci/build"));
        assert!(f.matches("ci/deploy"));
        assert!(!f.matches("ci/build/main")); // * doesn't cross levels
        assert!(!f.matches("ci"));            // * requires one segment
    }

    #[test]
    fn single_wildcard_in_middle() {
        let f = TopicFilter::new("chat/*/messages");
        assert!(f.matches("chat/telegram/messages"));
        assert!(f.matches("chat/discord/messages"));
        assert!(!f.matches("chat/telegram/group/messages"));
    }

    // ── Recursive wildcard ──────────────────────────────────────

    #[test]
    fn recursive_wildcard_at_end() {
        let f = TopicFilter::new("ci/**");
        assert!(f.matches("ci/build"));
        assert!(f.matches("ci/build/main"));
        assert!(f.matches("ci/deploy/staging/v2"));
        assert!(f.matches("ci")); // ** matches zero segments
        assert!(!f.matches("chat/telegram"));
    }

    #[test]
    fn recursive_wildcard_in_middle() {
        let f = TopicFilter::new("chat/**/messages");
        assert!(f.matches("chat/messages"));              // ** = 0 segments
        assert!(f.matches("chat/telegram/messages"));     // ** = 1 segment
        assert!(f.matches("chat/telegram/group/messages")); // ** = 2 segments
        assert!(!f.matches("chat/telegram/files"));
    }

    // ── Catch-all ───────────────────────────────────────────────

    #[test]
    fn catch_all_single() {
        let f = TopicFilter::new("*");
        assert!(f.matches("alerts"));
        assert!(!f.matches("alerts/critical")); // * is single-level
    }

    #[test]
    fn catch_all_recursive() {
        let f = TopicFilter::new("**");
        assert!(f.matches("anything"));
        assert!(f.matches("a/b/c/d"));
        assert!(f.matches(""));
    }

    // ── Hierarchical propagation ────────────────────────────────
    // Events at "chat/telegram/12345" should match:
    //   - "chat/telegram/12345" (exact)
    //   - "chat/telegram/*"     (single wild)
    //   - "chat/**"             (recursive wild)

    #[test]
    fn hierarchical_propagation_scenario() {
        let exact = TopicFilter::new("chat/telegram/12345");
        let single = TopicFilter::new("chat/telegram/*");
        let recursive = TopicFilter::new("chat/**");
        let catch_all = TopicFilter::new("**");

        let topic = "chat/telegram/12345";

        assert!(exact.matches(topic));
        assert!(single.matches(topic));
        assert!(recursive.matches(topic));
        assert!(catch_all.matches(topic));

        // But a filter for a different path shouldn't match
        let other = TopicFilter::new("ci/**");
        assert!(!other.matches(topic));
    }
}
