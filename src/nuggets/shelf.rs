//! NuggetShelf — multi-nugget manager.
//!
//! Organises multiple Nugget instances under a shared directory and
//! supports broadcast recall across all nuggets. Each nugget is topic-scoped
//! (e.g., "prefs", "locations", "debug") and broadcast recall returns the
//! best match across all of them.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use serde::Serialize;

use super::memory::{default_save_dir, Nugget, NuggetOpts, NuggetStatus, RecallResult};

// ---------------------------------------------------------------------------
// Shelf recall result (extends RecallResult with nugget_name)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct ShelfRecallResult {
    #[serde(flatten)]
    pub result: RecallResult,
    pub nugget_name: Option<String>,
}

// ---------------------------------------------------------------------------
// NuggetShelf
// ---------------------------------------------------------------------------

/// Multi-nugget manager with broadcast recall.
pub struct NuggetShelf {
    save_dir: PathBuf,
    auto_save: bool,
    nuggets: HashMap<String, Nugget>,
}

impl NuggetShelf {
    pub fn new(save_dir: Option<PathBuf>, auto_save: bool) -> Self {
        Self {
            save_dir: save_dir.unwrap_or_else(default_save_dir),
            auto_save,
            nuggets: HashMap::new(),
        }
    }

    // -- nugget lifecycle ----------------------------------------------------

    /// Create a new nugget with the given name and options.
    /// Panics if a nugget with that name already exists.
    pub fn create(
        &mut self,
        name: &str,
        d: Option<usize>,
        banks: Option<usize>,
        ensembles: Option<usize>,
    ) -> &mut Nugget {
        if self.nuggets.contains_key(name) {
            panic!("Nugget {:?} already exists", name);
        }
        let n = Nugget::new(NuggetOpts {
            name: name.into(),
            d: d.unwrap_or(2048),
            banks: banks.unwrap_or(4),
            ensembles: ensembles.unwrap_or(1),
            auto_save: self.auto_save,
            save_dir: self.save_dir.clone(),
            ..Default::default()
        });
        self.nuggets.insert(name.to_string(), n);
        self.nuggets.get_mut(name).unwrap()
    }

    /// Get a nugget by name. Panics if not found.
    pub fn get(&self, name: &str) -> &Nugget {
        self.nuggets
            .get(name)
            .unwrap_or_else(|| panic!("Nugget {:?} not found", name))
    }

    /// Get a mutable reference to a nugget. Panics if not found.
    pub fn get_mut(&mut self, name: &str) -> &mut Nugget {
        self.nuggets
            .get_mut(name)
            .unwrap_or_else(|| panic!("Nugget {:?} not found", name))
    }

    /// Get or create a nugget with default settings.
    pub fn get_or_create(&mut self, name: &str) -> &mut Nugget {
        if !self.nuggets.contains_key(name) {
            self.create(name, None, None, None);
        }
        self.nuggets.get_mut(name).unwrap()
    }

    /// Remove a nugget and its persisted file.
    pub fn remove(&mut self, name: &str) {
        if !self.nuggets.contains_key(name) {
            panic!("Nugget {:?} not found", name);
        }
        let path = self.save_dir.join(format!("{name}.nugget.json"));
        if path.exists() {
            let _ = fs::remove_file(&path);
        }
        self.nuggets.remove(name);
    }

    /// List status of all nuggets.
    pub fn list(&self) -> Vec<NuggetStatus> {
        self.nuggets.values().map(|n| n.status()).collect()
    }

    // -- convenience pass-throughs -------------------------------------------

    /// Remember a fact in a specific nugget.
    pub fn remember(&mut self, nugget_name: &str, key: &str, value: &str) {
        self.get_mut(nugget_name).remember(key, value);
    }

    /// Recall from a specific nugget or broadcast across all.
    pub fn recall(
        &mut self,
        query: &str,
        nugget_name: Option<&str>,
        session_id: &str,
    ) -> ShelfRecallResult {
        if let Some(name) = nugget_name {
            let result = self.get_mut(name).recall(query, session_id);
            return ShelfRecallResult {
                result,
                nugget_name: Some(name.to_string()),
            };
        }

        // Broadcast: find best confidence across all nuggets
        let mut best = ShelfRecallResult {
            result: RecallResult {
                answer: None,
                confidence: 0.0,
                margin: 0.0,
                found: false,
                key: String::new(),
            },
            nugget_name: None,
        };

        let names: Vec<String> = self.nuggets.keys().cloned().collect();
        for name in &names {
            let result = self
                .nuggets
                .get_mut(name)
                .unwrap()
                .recall(query, session_id);
            if result.found && result.confidence > best.result.confidence {
                best = ShelfRecallResult {
                    result,
                    nugget_name: Some(name.clone()),
                };
            }
        }
        best
    }

    /// Forget a fact from a specific nugget.
    pub fn forget(&mut self, nugget_name: &str, key: &str) -> bool {
        self.get_mut(nugget_name).forget(key)
    }

    // -- persistence ---------------------------------------------------------

    /// Load all `.nugget.json` files from the save directory.
    pub fn load_all(&mut self) {
        if !self.save_dir.exists() {
            return;
        }
        let entries = match fs::read_dir(&self.save_dir) {
            Ok(e) => e,
            Err(_) => return,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let fname = match path.file_name().and_then(|f| f.to_str()) {
                Some(f) => f.to_string(),
                None => continue,
            };
            if !fname.ends_with(".nugget.json") {
                continue;
            }
            match Nugget::load(&path, self.auto_save) {
                Ok(n) => {
                    self.nuggets.insert(n.name.clone(), n);
                }
                Err(_) => continue, // skip corrupt files
            }
        }
    }

    /// Save all nuggets to disk.
    pub fn save_all(&self) {
        for n in self.nuggets.values() {
            let _ = n.save(None);
        }
    }

    /// Check if a nugget exists.
    pub fn has(&self, name: &str) -> bool {
        self.nuggets.contains_key(name)
    }

    /// Number of loaded nuggets.
    pub fn size(&self) -> usize {
        self.nuggets.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_shelf() -> (NuggetShelf, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let shelf = NuggetShelf::new(Some(tmp.path().to_path_buf()), false);
        (shelf, tmp)
    }

    #[test]
    fn create_and_get() {
        let (mut shelf, _tmp) = test_shelf();
        shelf.create("test", Some(512), Some(2), None);
        assert_eq!(shelf.get("test").name, "test");
        assert_eq!(shelf.size(), 1);
    }

    #[test]
    #[should_panic(expected = "already exists")]
    fn duplicate_create_panics() {
        let (mut shelf, _tmp) = test_shelf();
        shelf.create("test", None, None, None);
        shelf.create("test", None, None, None);
    }

    #[test]
    #[should_panic(expected = "not found")]
    fn get_missing_panics() {
        let (shelf, _tmp) = test_shelf();
        shelf.get("nope");
    }

    #[test]
    fn get_or_create() {
        let (mut shelf, _tmp) = test_shelf();
        shelf.get_or_create("test");
        assert!(shelf.has("test"));
        // Second call returns existing
        shelf.get_or_create("test");
        assert_eq!(shelf.size(), 1);
    }

    #[test]
    fn remove_nugget() {
        let (mut shelf, _tmp) = test_shelf();
        shelf.create("doomed", Some(512), Some(2), None);
        shelf.remember("doomed", "key", "val");
        shelf.remove("doomed");
        assert_eq!(shelf.size(), 0);
    }

    #[test]
    fn remember_and_recall_across_nuggets() {
        let (mut shelf, _tmp) = test_shelf();
        shelf.create("prefs", Some(512), Some(2), None);
        shelf.create("facts", Some(512), Some(2), None);

        shelf.remember("prefs", "color", "blue");
        shelf.remember("facts", "city", "London");

        // Targeted recall
        let r1 = shelf.recall("color", Some("prefs"), "");
        assert!(r1.result.found);
        assert_eq!(r1.result.answer.unwrap(), "blue");
        assert_eq!(r1.nugget_name.unwrap(), "prefs");

        // Broadcast recall
        let r2 = shelf.recall("city", None, "");
        assert!(r2.result.found);
        assert_eq!(r2.result.answer.unwrap(), "London");
        assert_eq!(r2.nugget_name.unwrap(), "facts");
    }

    #[test]
    fn save_and_load_all() {
        let tmp = tempfile::tempdir().unwrap();
        let tmp_path = tmp.path().to_path_buf();

        // Create and populate
        {
            let mut shelf1 = NuggetShelf::new(Some(tmp_path.clone()), true);
            shelf1.create("a", Some(512), Some(2), None);
            shelf1.create("b", Some(512), Some(2), None);
            shelf1.remember("a", "k1", "v1");
            shelf1.remember("b", "k2", "v2");
            shelf1.save_all();
        }

        // Reload
        let mut shelf2 = NuggetShelf::new(Some(tmp_path), false);
        shelf2.load_all();
        assert_eq!(shelf2.size(), 2);
        assert_eq!(
            shelf2.recall("k1", Some("a"), "").result.answer.unwrap(),
            "v1"
        );
        assert_eq!(
            shelf2.recall("k2", Some("b"), "").result.answer.unwrap(),
            "v2"
        );
    }

    #[test]
    fn list_statuses() {
        let (mut shelf, _tmp) = test_shelf();
        shelf.create("one", Some(512), Some(2), None);
        shelf.create("two", Some(512), Some(2), None);
        let list = shelf.list();
        assert_eq!(list.len(), 2);
        let mut names: Vec<String> = list.iter().map(|s| s.name.clone()).collect();
        names.sort();
        assert_eq!(names, vec!["one", "two"]);
    }

    #[test]
    fn has_checks_existence() {
        let (mut shelf, _tmp) = test_shelf();
        shelf.create("exists", None, None, None);
        assert!(shelf.has("exists"));
        assert!(!shelf.has("nope"));
    }

    #[test]
    fn broadcast_recall_returns_best() {
        let (mut shelf, _tmp) = test_shelf();
        shelf.create("a", Some(512), Some(2), None);
        shelf.create("b", Some(512), Some(2), None);

        shelf.remember("a", "color", "red");
        shelf.remember("b", "color", "blue");

        let r = shelf.recall("color", None, "");
        assert!(r.result.found);
        // Either "red" or "blue" — depends on which has higher confidence
        let ans = r.result.answer.as_deref();
        assert!(ans == Some("red") || ans == Some("blue"));
    }
}
