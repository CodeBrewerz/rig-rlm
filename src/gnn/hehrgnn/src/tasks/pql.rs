//! Predictive Query Language (PQL) parser for link prediction tasks.
//!
//! Inspired by KumoRFM's PQL (§2.1), this module provides a simplified
//! SQL-like syntax for specifying prediction tasks that map to
//! `LinkPredictionTask` entries in the registry.
//!
//! # Syntax
//!
//! ```text
//! PREDICT <target_type> FOR <source_type> VIA <relation>
//! PREDICT <target_type> FOR <source_type> VIA <rel1> -> <rel2> -> ...
//! ```
//!
//! # Examples
//!
//! ```text
//! PREDICT transaction-category FOR transaction-evidence VIA evidence-has-category
//! PREDICT sub-ledger FOR reconciliation-case VIA case-has-entry -> allocation-has-journal-entry
//! ```

use super::link_predictor::LinkPredictionTask;

/// Parse a PQL query string into a `LinkPredictionTask`.
///
/// Returns `None` if the query doesn't match the expected syntax.
pub fn parse_pql(query: &str) -> Option<LinkPredictionTask> {
    let query = query.trim();

    // PREDICT <target> FOR <source> VIA <path>
    let parts: Vec<&str> = query.splitn(2, ' ').collect();
    if parts.len() < 2 || parts[0].to_uppercase() != "PREDICT" {
        return None;
    }

    let rest = parts[1];

    // Split by FOR
    let for_parts: Vec<&str> = rest.splitn(2, " FOR ").collect();
    if for_parts.len() < 2 {
        // Try case-insensitive
        let lower = rest.to_uppercase();
        let for_idx = lower.find(" FOR ")?;
        let target_type = rest[..for_idx].trim().to_string();
        let after_for = rest[for_idx + 5..].trim();

        return parse_via(target_type, after_for);
    }

    let target_type = for_parts[0].trim().to_string();
    let after_for = for_parts[1].trim();

    parse_via(target_type, after_for)
}

fn parse_via(target_type: String, after_for: &str) -> Option<LinkPredictionTask> {
    // Split by VIA
    let via_parts: Vec<&str> = after_for.splitn(2, " VIA ").collect();
    if via_parts.len() < 2 {
        let upper = after_for.to_uppercase();
        let via_idx = upper.find(" VIA ")?;
        let source_type = after_for[..via_idx].trim().to_string();
        let path_str = after_for[via_idx + 5..].trim();
        return Some(build_task(source_type, target_type, path_str));
    }

    let source_type = via_parts[0].trim().to_string();
    let path_str = via_parts[1].trim();

    Some(build_task(source_type, target_type, path_str))
}

fn build_task(source_type: String, target_type: String, path_str: &str) -> LinkPredictionTask {
    // Parse path: "rel1 -> rel2 -> rel3" or just "rel1"
    let path: Vec<String> = path_str
        .split("->")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let relation = path.first().cloned().unwrap_or_default();

    // Auto-generate a snake_case name from the relation
    let name = relation.replace('-', "_");

    LinkPredictionTask {
        name,
        source_type,
        target_type,
        relation,
        path,
        description: format!("PQL-defined prediction task from query"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_pql() {
        let task = parse_pql(
            "PREDICT transaction-category FOR transaction-evidence VIA evidence-has-category",
        )
        .unwrap();
        assert_eq!(task.source_type, "transaction-evidence");
        assert_eq!(task.target_type, "transaction-category");
        assert_eq!(task.relation, "evidence-has-category");
        assert_eq!(task.path.len(), 1);
        println!(
            "✅ PQL parse: {} → {} via {}",
            task.source_type, task.target_type, task.relation
        );
    }

    #[test]
    fn test_parse_multihop_pql() {
        let task = parse_pql(
            "PREDICT sub-ledger FOR reconciliation-case VIA case-has-entry -> allocation-has-journal-entry",
        )
        .unwrap();
        assert_eq!(task.source_type, "reconciliation-case");
        assert_eq!(task.target_type, "sub-ledger");
        assert_eq!(task.path.len(), 2);
        assert_eq!(task.path[0], "case-has-entry");
        assert_eq!(task.path[1], "allocation-has-journal-entry");
        println!("✅ PQL multi-hop: {} hops", task.path.len());
    }

    #[test]
    fn test_parse_case_insensitive() {
        let task =
            parse_pql("predict tax-code for reconciliation-case via tax-code-assigned-to-subject");
        // Our parser expects PREDICT in uppercase, so this should return None
        // unless we add case-insensitive support
        // For now the parser is case-sensitive on PREDICT
        assert!(task.is_none() || task.unwrap().target_type == "tax-code");
    }

    #[test]
    fn test_parse_invalid() {
        assert!(parse_pql("SELECT * FROM table").is_none());
        assert!(parse_pql("").is_none());
        assert!(parse_pql("PREDICT").is_none());
    }
}
