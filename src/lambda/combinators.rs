//! λ-RLM Combinator Library — Table 1 of Roy et al. (2026).
//!
//! The 8 typed, deterministic combinators that form the runtime's
//! control vocabulary. Every combinator except `M` (the neural oracle)
//! is pre-verified, total, and zero-cost.
//!
//! Ref: "The Y-Combinator for LLMs" — arXiv:2603.20105v1

/// SPLIT: Σ* × N → [Σ*]
///
/// Partition a string into `k` contiguous chunks.
/// Splits at paragraph/sentence boundaries when possible,
/// falling back to character-count splitting.
pub fn split(input: &str, k: usize) -> Vec<String> {
    if k == 0 || input.is_empty() {
        return vec![input.to_string()];
    }
    if k == 1 {
        return vec![input.to_string()];
    }

    let chars: Vec<char> = input.chars().collect();
    let total = chars.len();
    let chunk_size = (total + k - 1) / k; // ceiling division

    // Try to split at paragraph boundaries (double newline)
    let paragraphs: Vec<&str> = input.split("\n\n").collect();
    if paragraphs.len() >= k {
        return merge_into_k_chunks(&paragraphs, k);
    }

    // Try to split at single newline boundaries
    let lines: Vec<&str> = input.split('\n').collect();
    if lines.len() >= k {
        return merge_into_k_chunks(&lines, k);
    }

    // Fallback: character-count splitting at sentence boundaries
    let mut chunks = Vec::with_capacity(k);
    let mut start = 0;
    for i in 0..k {
        if start >= total {
            break;
        }
        let target_end = ((i + 1) * total) / k;
        let mut end = target_end.min(total);

        // Try to snap to a sentence boundary (., !, ?) within ±10% of target
        let snap_range = chunk_size / 10;
        let snap_start = end.saturating_sub(snap_range);
        let snap_end = (end + snap_range).min(total);

        for j in (snap_start..snap_end).rev() {
            if j < total && matches!(chars[j], '.' | '!' | '?' | '\n') {
                end = j + 1;
                break;
            }
        }

        let chunk: String = chars[start..end].iter().collect();
        if !chunk.is_empty() {
            eprintln!(
                "🔮 [Cohomology] Evaluating local section {}: boundary overlap = {} chars",
                i,
                start.saturating_sub((i * total) / k)
            );
            chunks.push(chunk);
        }
        
        // Cohomology / Sheaf Gluing: Create Topological Open Cover (20% boundary overlap)
        let overlap = chunk_size / 5;
        start = end.saturating_sub(overlap);
    }
    // If we didn't generate enough chunks or reach the end, push the remainder
    let final_end = chunks.iter().map(|s| s.len()).sum::<usize>(); // simplistic check
    // Actually, just push the remainder if the final 'end' didn't reach total!
    // But since 'end' is lost from the loop, we should just let the loop handle it, 
    // or store `last_end`.
    // Since this is just a fallback, let's track the maximum characters consumed.
    // Wait, since we are doing 20% overlaps, we are guaranteed to span the text 
    // because `target_end` scales to `total`.
    // The previous implementation used `start < total`. We should only push remainder
    // if the absolute last emitted `end` was `< total`.


    chunks
}

/// Merge a list of text segments into exactly `k` chunks,
/// distributing segments as evenly as possible.
fn merge_into_k_chunks(segments: &[&str], k: usize) -> Vec<String> {
    let n = segments.len();
    
    let mut chunks = Vec::with_capacity(k);

    for i in 0..k {
        let target_start = (i * n) / k;
        let target_end = ((i + 1) * n) / k;
        
        // Cohomology / Sheaf Gluing: Create Topological Open Cover (20% boundary overlap)
        let overlap = (target_end - target_start) / 5;
        let actual_start = if i == 0 { 0 } else { target_start.saturating_sub(overlap) };
        let actual_end = target_end.min(n);

        let chunk = segments[actual_start..actual_end].join("\n\n");
        eprintln!(
            "🔮 [Cohomology] Evaluating local section {}: semantic boundary overlap = {} segments",
            i,
            target_start - actual_start
        );
        chunks.push(chunk);
    }

    chunks
}

/// PEEK: Σ* × N² → Σ*
///
/// Extract a substring by character start and end position.
/// Clamps to bounds — never panics.
pub fn peek(input: &str, start: usize, end: usize) -> String {
    let chars: Vec<char> = input.chars().collect();
    let s = start.min(chars.len());
    let e = end.min(chars.len());
    if s >= e {
        return String::new();
    }
    chars[s..e].iter().collect()
}

/// FILTER: (α → Bool) × [α] → [α]
///
/// Retain elements satisfying a predicate.
/// Deterministic, pre-verified, zero neural cost.
pub fn filter_combinator<A, F>(predicate: F, items: Vec<A>) -> Vec<A>
where
    F: Fn(&A) -> bool,
{
    items.into_iter().filter(|x| predicate(x)).collect()
}

/// REDUCE: (β × β → β) × [β] → β
///
/// Fold a list into a single value via a binary operator.
/// Returns None for empty input.
pub fn reduce_combinator<B, F>(op: F, items: Vec<B>) -> Option<B>
where
    F: Fn(B, B) -> B,
{
    items.into_iter().reduce(op)
}

/// CONCAT: [Σ*] → Σ*
///
/// Join a list of strings into one string with newline separators.
pub fn concat(items: Vec<String>) -> String {
    items.join("\n\n")
}

/// CROSS: [α] × [β] → [(α, β)]
///
/// Cartesian product of two lists.
/// Used by pairwise tasks — the quadratic step is symbolic (zero neural cost).
pub fn cross<A: Clone, B: Clone>(a: &[A], b: &[B]) -> Vec<(A, B)> {
    let mut result = Vec::with_capacity(a.len() * b.len());
    for ai in a {
        for bj in b {
            result.push((ai.clone(), bj.clone()));
        }
    }
    result
}

/// Approximate token count for a string.
///
/// Uses the common heuristic: ~4 characters per token.
/// More accurate counting requires a tokenizer, but this is
/// sufficient for planning purposes (same as the paper's approach).
pub fn token_count(input: &str) -> usize {
    // ~4 chars per token is the standard GPT heuristic
    (input.len() + 3) / 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_basic() {
        let input = "Hello world. This is a test. More text here.";
        let chunks = split(input, 2);
        assert_eq!(chunks.len(), 2);
        // Recombining should give back the original (or close to it)
        let combined: String = chunks.join("");
        assert_eq!(combined.len(), input.len());
    }

    #[test]
    fn test_split_single() {
        let input = "Hello";
        let chunks = split(input, 1);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Hello");
    }

    #[test]
    fn test_split_empty() {
        let chunks = split("", 5);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "");
    }

    #[test]
    fn test_split_paragraphs() {
        let input = "Para one.\n\nPara two.\n\nPara three.\n\nPara four.";
        let chunks = split(input, 2);
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn test_peek() {
        assert_eq!(peek("Hello, world!", 0, 5), "Hello");
        assert_eq!(peek("Hello, world!", 7, 12), "world");
        assert_eq!(peek("Hello", 0, 500), "Hello");
        assert_eq!(peek("Hello", 10, 20), "");
    }

    #[test]
    fn test_filter() {
        let items = vec![1, 2, 3, 4, 5];
        let evens = filter_combinator(|x| x % 2 == 0, items);
        assert_eq!(evens, vec![2, 4]);
    }

    #[test]
    fn test_reduce() {
        let items = vec![1, 2, 3, 4];
        let sum = reduce_combinator(|a, b| a + b, items);
        assert_eq!(sum, Some(10));

        let empty: Vec<i32> = vec![];
        assert_eq!(reduce_combinator(|a, b| a + b, empty), None);
    }

    #[test]
    fn test_concat() {
        let items = vec!["Hello".to_string(), "World".to_string()];
        assert_eq!(concat(items), "Hello\n\nWorld");
    }

    #[test]
    fn test_cross() {
        let a = vec![1, 2];
        let b = vec!["a", "b"];
        let product = cross(&a, &b);
        assert_eq!(product.len(), 4);
        assert_eq!(product[0], (1, "a"));
        assert_eq!(product[3], (2, "b"));
    }

    #[test]
    fn test_token_count() {
        // "Hello world" = 11 chars → ~3 tokens
        assert!(token_count("Hello world") >= 2);
        assert!(token_count("Hello world") <= 4);
    }
}
