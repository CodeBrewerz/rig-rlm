//! Skills and memory loading for agent enrichment.
//!
//! Phase 4: Loads external knowledge sources into the agent's system prompt.
//! - **AGENTS.md files**: Project-level instructions/context from `.agents/`
//!   or custom paths
//! - **Skill directories**: Folders containing `SKILL.md` files with
//!   specialized instructions the agent can reference
//!
//! This mirrors rlmagents' skill loading pattern but integrates cleanly
//! with rig-rlm's monadic architecture.

use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Configuration for memory/skill sources injected at agent startup.
#[derive(Debug, Clone, Default)]
pub struct MemoryConfig {
    /// Paths to AGENTS.md files (project-level instructions).
    pub agents_md_paths: Vec<PathBuf>,
    /// Directories containing SKILL.md files.
    pub skill_dirs: Vec<PathBuf>,
}

/// Summary of a discovered skill.
#[derive(Debug, Clone)]
pub struct SkillSummary {
    /// Skill name (derived from directory name or SKILL.md frontmatter).
    pub name: String,
    /// Short description (from SKILL.md frontmatter, if present).
    pub description: String,
    /// Path to the SKILL.md file.
    pub path: PathBuf,
}

impl MemoryConfig {
    /// Create a new memory config with the given paths.
    pub fn new(agents_md_paths: Vec<PathBuf>, skill_dirs: Vec<PathBuf>) -> Self {
        Self {
            agents_md_paths,
            skill_dirs,
        }
    }

    /// Auto-discover AGENTS.md files in standard locations relative to `cwd`.
    ///
    /// Checks: `.agents/`, `.agent/`, `_agents/`, `_agent/`
    pub fn auto_discover(cwd: &Path) -> Self {
        let mut agents_md_paths = Vec::new();
        let mut skill_dirs = Vec::new();

        let agent_dirs = [".agents", ".agent", "_agents", "_agent"];
        for dir in &agent_dirs {
            let agent_dir = cwd.join(dir);
            let agents_md = agent_dir.join("AGENTS.md");
            if agents_md.is_file() {
                info!(path = %agents_md.display(), "Found AGENTS.md");
                agents_md_paths.push(agents_md);
            }
            let workflows_dir = agent_dir.join("workflows");
            if workflows_dir.is_dir() {
                info!(path = %workflows_dir.display(), "Found workflows directory");
                skill_dirs.push(workflows_dir);
            }
        }

        Self {
            agents_md_paths,
            skill_dirs,
        }
    }

    /// Load and concatenate all AGENTS.md files.
    ///
    /// Returns the combined content or an empty string if none found.
    pub fn load_agents_md(&self) -> String {
        let mut parts = Vec::new();
        for path in &self.agents_md_paths {
            match std::fs::read_to_string(path) {
                Ok(content) => {
                    debug!(path = %path.display(), len = content.len(), "Loaded AGENTS.md");
                    parts.push(content);
                }
                Err(e) => {
                    warn!(path = %path.display(), error = %e, "Failed to load AGENTS.md");
                }
            }
        }
        parts.join("\n\n---\n\n")
    }

    /// Discover all skills from configured skill directories.
    ///
    /// A skill is a directory containing a `SKILL.md` file.
    /// The name/description are extracted from YAML frontmatter if present.
    pub fn discover_skills(&self) -> Vec<SkillSummary> {
        let mut skills = Vec::new();
        for dir in &self.skill_dirs {
            if !dir.is_dir() {
                continue;
            }
            // Check for SKILL.md files directly in the dir
            match std::fs::read_dir(dir) {
                Ok(entries) => {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.is_file() && path.extension().map_or(false, |e| e == "md") {
                            // Treat each .md file as a skill
                            let name = path
                                .file_stem()
                                .map(|s| s.to_string_lossy().to_string())
                                .unwrap_or_default();
                            let description = extract_frontmatter_description(&path);
                            skills.push(SkillSummary {
                                name,
                                description,
                                path,
                            });
                        } else if path.is_dir() {
                            // Check for SKILL.md inside subdirectory
                            let skill_md = path.join("SKILL.md");
                            if skill_md.is_file() {
                                let name = path
                                    .file_name()
                                    .map(|s| s.to_string_lossy().to_string())
                                    .unwrap_or_default();
                                let description = extract_frontmatter_description(&skill_md);
                                skills.push(SkillSummary {
                                    name,
                                    description,
                                    path: skill_md,
                                });
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!(dir = %dir.display(), error = %e, "Failed to read skill directory");
                }
            }
        }
        debug!(count = skills.len(), "Discovered skills");
        skills
    }

    /// Read a specific skill by name.
    ///
    /// Searches all skill directories for a matching skill and returns its content.
    pub fn read_skill(&self, name: &str) -> Option<String> {
        let skills = self.discover_skills();
        let skill = skills.iter().find(|s| s.name == name)?;
        std::fs::read_to_string(&skill.path).ok()
    }

    /// Format the memory block for injection into the system prompt.
    ///
    /// Returns a formatted string suitable for appending to the system prompt,
    /// including AGENTS.md content and a skill listing.
    pub fn format_for_prompt(&self) -> String {
        let mut sections = Vec::new();

        // AGENTS.md content
        let agents_md = self.load_agents_md();
        if !agents_md.is_empty() {
            sections.push(format!(
                "## Project Instructions (from AGENTS.md)\n\n{}",
                agents_md
            ));
        }

        // Skill listing
        let skills = self.discover_skills();
        if !skills.is_empty() {
            let mut skill_list = String::from("## Available Skills\n\n");
            skill_list.push_str(
                "The following skills are available. You can reference them for specialized tasks:\n\n",
            );
            for skill in &skills {
                if skill.description.is_empty() {
                    skill_list.push_str(&format!(
                        "- **{}** ({})\n",
                        skill.name,
                        skill.path.display()
                    ));
                } else {
                    skill_list.push_str(&format!(
                        "- **{}**: {} ({})\n",
                        skill.name,
                        skill.description,
                        skill.path.display()
                    ));
                }
            }
            sections.push(skill_list);
        }

        sections.join("\n\n")
    }
}

/// Extract the `description` field from YAML frontmatter in a markdown file.
///
/// Expects format:
/// ```text
/// ---
/// description: some description here
/// ---
/// ```
fn extract_frontmatter_description(path: &Path) -> String {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return String::new(),
    };

    if !content.starts_with("---") {
        return String::new();
    }

    // Find the closing ---
    if let Some(end) = content[3..].find("---") {
        let frontmatter = &content[3..3 + end];
        for line in frontmatter.lines() {
            let line = line.trim();
            if let Some(desc) = line.strip_prefix("description:") {
                return desc.trim().to_string();
            }
        }
    }

    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn setup_test_dir() -> TempDir {
        let dir = TempDir::new().unwrap();

        // Create an AGENTS.md file
        let agents_dir = dir.path().join(".agents");
        fs::create_dir_all(&agents_dir).unwrap();
        fs::write(
            agents_dir.join("AGENTS.md"),
            "# Project Rules\n\nAlways write tests.",
        )
        .unwrap();

        // Create a workflows dir with skill files
        let workflows_dir = agents_dir.join("workflows");
        fs::create_dir_all(&workflows_dir).unwrap();
        fs::write(
            workflows_dir.join("deploy.md"),
            "---\ndescription: How to deploy the app\n---\n\n1. Run cargo build\n2. Deploy",
        )
        .unwrap();
        fs::write(
            workflows_dir.join("test.md"),
            "---\ndescription: How to run tests\n---\n\n1. Run cargo test",
        )
        .unwrap();

        dir
    }

    #[test]
    fn auto_discover_finds_agents_md() {
        let dir = setup_test_dir();
        let config = MemoryConfig::auto_discover(dir.path());
        assert_eq!(config.agents_md_paths.len(), 1);
        assert!(config.agents_md_paths[0].ends_with("AGENTS.md"));
    }

    #[test]
    fn auto_discover_finds_workflow_dir() {
        let dir = setup_test_dir();
        let config = MemoryConfig::auto_discover(dir.path());
        assert_eq!(config.skill_dirs.len(), 1);
    }

    #[test]
    fn load_agents_md_content() {
        let dir = setup_test_dir();
        let config = MemoryConfig::auto_discover(dir.path());
        let content = config.load_agents_md();
        assert!(content.contains("Always write tests"));
    }

    #[test]
    fn discover_skills_finds_md_files() {
        let dir = setup_test_dir();
        let config = MemoryConfig::auto_discover(dir.path());
        let skills = config.discover_skills();
        assert_eq!(skills.len(), 2);
        let names: Vec<&str> = skills.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"deploy"));
        assert!(names.contains(&"test"));
    }

    #[test]
    fn skill_description_from_frontmatter() {
        let dir = setup_test_dir();
        let config = MemoryConfig::auto_discover(dir.path());
        let skills = config.discover_skills();
        let deploy = skills.iter().find(|s| s.name == "deploy").unwrap();
        assert_eq!(deploy.description, "How to deploy the app");
    }

    #[test]
    fn read_skill_by_name() {
        let dir = setup_test_dir();
        let config = MemoryConfig::auto_discover(dir.path());
        let content = config.read_skill("deploy").unwrap();
        assert!(content.contains("cargo build"));
    }

    #[test]
    fn read_skill_nonexistent() {
        let dir = setup_test_dir();
        let config = MemoryConfig::auto_discover(dir.path());
        assert!(config.read_skill("nope").is_none());
    }

    #[test]
    fn format_for_prompt_includes_both() {
        let dir = setup_test_dir();
        let config = MemoryConfig::auto_discover(dir.path());
        let prompt = config.format_for_prompt();
        assert!(prompt.contains("Project Instructions"));
        assert!(prompt.contains("Always write tests"));
        assert!(prompt.contains("Available Skills"));
        assert!(prompt.contains("deploy"));
    }

    #[test]
    fn empty_config_produces_empty_prompt() {
        let config = MemoryConfig::default();
        assert!(config.format_for_prompt().is_empty());
    }

    #[test]
    fn extract_frontmatter_no_frontmatter() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.md");
        fs::write(&path, "# No frontmatter here").unwrap();
        assert!(extract_frontmatter_description(&path).is_empty());
    }
}
