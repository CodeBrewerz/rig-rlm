//! MEMORY.md promotion — bridge nuggets to permanent context.
//!
//! Facts recalled 3+ times across sessions are promoted to MEMORY.md
//! for permanent context inclusion. The file is idempotently merged.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use super::shelf::NuggetShelf;

const PROMOTE_THRESHOLD: u32 = 3;

const MEMORY_MD_HEADER: &str =
    "# Memory\n\nAuto-promoted from nuggets (3+ recalls across sessions).\n";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Sections = BTreeMap<String, BTreeMap<String, String>>;

// ---------------------------------------------------------------------------
// Parsing & rendering
// ---------------------------------------------------------------------------

fn parse_memory_md(content: &str) -> Sections {
    let mut sections = Sections::new();
    let mut current_section = String::new();

    for line in content.lines() {
        let stripped = line.trim();

        // Section header: ## Name
        if let Some(rest) = stripped.strip_prefix("## ") {
            current_section = rest.trim().to_string();
            sections
                .entry(current_section.clone())
                .or_insert_with(BTreeMap::new);
            continue;
        }

        // Fact entry: - **key**: value
        if !current_section.is_empty() {
            if let Some(rest) = stripped.strip_prefix("- **") {
                if let Some(sep_pos) = rest.find("**:") {
                    let key = rest[..sep_pos].trim().to_string();
                    let value = rest[sep_pos + 3..].trim().to_string();
                    if let Some(section) = sections.get_mut(&current_section) {
                        section.insert(key, value);
                    }
                }
            }
        }
    }

    sections
}

fn render_memory_md(sections: &Sections) -> String {
    if sections.is_empty() {
        return MEMORY_MD_HEADER.to_string();
    }

    // Ordering: "learnings" first, "preferences" second, then alphabetical
    let priority = ["learnings", "preferences"];
    let mut ordered: Vec<&str> = Vec::new();
    for p in &priority {
        if sections.contains_key(*p) {
            ordered.push(p);
        }
    }
    for key in sections.keys() {
        if !priority.contains(&key.as_str()) {
            ordered.push(key);
        }
    }

    let mut lines = vec![MEMORY_MD_HEADER.to_string()];
    for section_name in ordered {
        if let Some(facts) = sections.get(section_name) {
            if facts.is_empty() {
                continue;
            }
            lines.push(format!("## {section_name}\n"));
            for (key, value) in facts {
                lines.push(format!("- **{key}**: {value}"));
            }
            lines.push(String::new());
        }
    }

    lines.join("\n")
}

// ---------------------------------------------------------------------------
// Promote
// ---------------------------------------------------------------------------

/// Promote facts with hits >= threshold to a MEMORY.md file.
///
/// Merges into existing MEMORY.md (idempotent). Returns count of
/// newly promoted facts.
pub fn promote_facts(shelf: &mut NuggetShelf, memory_path: &Path) -> usize {
    // Collect promotable facts
    let mut candidates: Vec<(String, String, String)> = Vec::new(); // (nugget, key, value)
    for info in shelf.list() {
        let name = &info.name;
        let nugget = shelf.get(name);
        for fact in nugget.facts() {
            if fact.hits >= PROMOTE_THRESHOLD {
                candidates.push((name.clone(), fact.key.clone(), fact.value.clone()));
            }
        }
    }

    if candidates.is_empty() {
        return 0;
    }

    // Load existing MEMORY.md
    let existing_content = if memory_path.exists() {
        fs::read_to_string(memory_path).unwrap_or_default()
    } else {
        String::new()
    };

    let mut sections = if existing_content.is_empty() {
        Sections::new()
    } else {
        parse_memory_md(&existing_content)
    };

    // Merge candidates
    let mut new_count = 0;
    for (nugget_name, key, value) in &candidates {
        let section = sections
            .entry(nugget_name.clone())
            .or_insert_with(BTreeMap::new);
        let existing = section.get(key);
        if existing.map(|v| v.as_str()) != Some(value.as_str()) {
            if existing.is_none() {
                new_count += 1;
            }
            section.insert(key.clone(), value.clone());
        }
    }

    if new_count == 0 && !existing_content.is_empty() {
        let new_content = render_memory_md(&sections);
        if new_content == existing_content {
            return 0;
        }
    }

    // Atomic write
    if let Some(parent) = memory_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let tmp_path = memory_path.with_extension("md.tmp");
    let new_content = render_memory_md(&sections);
    if fs::write(&tmp_path, &new_content).is_ok() {
        let _ = fs::rename(&tmp_path, memory_path);
    }

    new_count
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_render_roundtrip() {
        let input = r#"# Memory

Auto-promoted from nuggets (3+ recalls across sessions).

## prefs

- **color**: blue
- **font**: monospace

## debug

- **cors_fix**: add origin to allowlist
"#;

        let sections = parse_memory_md(input);
        assert_eq!(sections.len(), 2);
        assert_eq!(sections["prefs"]["color"], "blue");
        assert_eq!(sections["debug"]["cors_fix"], "add origin to allowlist");

        let rendered = render_memory_md(&sections);
        // Re-parse should yield same sections
        let re_parsed = parse_memory_md(&rendered);
        assert_eq!(re_parsed, sections);
    }

    #[test]
    fn promote_creates_memory_md() {
        let tmp = tempfile::tempdir().unwrap();
        let shelf_dir = tmp.path().join("shelf");
        let memory_path = tmp.path().join("MEMORY.md");

        let mut shelf = NuggetShelf::new(Some(shelf_dir), false);
        shelf.create("prefs", Some(512), Some(2), None);
        shelf.remember("prefs", "color", "blue");

        // Manually bump hits to threshold
        {
            let nugget = shelf.get_mut("prefs");
            for _ in 0..3 {
                nugget.recall("color", &format!("session-{}", rand_id()));
            }
        }

        let promoted = promote_facts(&mut shelf, &memory_path);
        assert!(promoted > 0);
        assert!(memory_path.exists());

        let content = fs::read_to_string(&memory_path).unwrap();
        assert!(content.contains("**color**: blue"));
    }

    fn rand_id() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }

    #[test]
    fn promote_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let memory_path = tmp.path().join("MEMORY.md");

        // Write initial
        let mut sections = Sections::new();
        let mut prefs = BTreeMap::new();
        prefs.insert("color".into(), "blue".into());
        sections.insert("prefs".into(), prefs);
        let content = render_memory_md(&sections);
        fs::write(&memory_path, &content).unwrap();

        // Parse back
        let parsed = parse_memory_md(&content);
        assert_eq!(parsed["prefs"]["color"], "blue");
    }
}
