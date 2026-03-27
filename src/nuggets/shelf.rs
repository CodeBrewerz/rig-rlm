//! NuggetShelf — multi-nugget manager with LRU cache.
//!
//! **Zero RAM at startup.**  On `new()` / `load_all()` only a lightweight
//! catalog of (name → disk path) is built by scanning the save directory.
//! Nugget data is loaded on-demand the first time it is accessed and kept
//! in an LRU cache.  When the cache exceeds `max_cached` entries, the
//! least-recently-used nugget is flushed to disk and dropped from RAM.
//!
//! All mutations are write-through: `save()` is called immediately after
//! each `remember()` / `forget()` / `clear()` when `auto_save` is true.

use hashbrown::HashMap;
use std::fs;
use std::path::PathBuf;

use serde::Serialize;
use rayon::prelude::*;

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
// LRU entry — wraps a loaded Nugget with an access counter
// ---------------------------------------------------------------------------

struct CachedNugget {
    nugget: Nugget,
    /// Monotonically increasing counter — higher = more recently used.
    last_access: u64,
}

// ---------------------------------------------------------------------------
// NuggetShelf
// ---------------------------------------------------------------------------

/// Multi-nugget manager with on-demand disk loading and LRU eviction.
///
/// At startup only the directory listing is scanned — no JSON is parsed,
/// no facts are loaded.  Each `get_or_load()` brings a nugget into the
/// cache; if the cache is full the LRU entry is flushed and dropped.
pub struct NuggetShelf {
    save_dir: PathBuf,
    auto_save: bool,

    /// Known nugget names → disk paths.  Built from directory scan.
    catalog: HashMap<String, PathBuf>,

    /// In-memory cache of loaded nuggets, keyed by name.
    cache: HashMap<String, CachedNugget>,

    /// Maximum number of nuggets to keep in memory at once.
    max_cached: usize,

    /// Maximum total HRR bytes across all cached nuggets (default 1 GB).
    max_hrr_bytes: usize,

    /// Monotonic access counter for LRU ordering.
    access_counter: u64,
}

impl NuggetShelf {
    /// Create a new shelf.  Nothing is loaded into memory yet.
    pub fn new(save_dir: Option<PathBuf>, auto_save: bool) -> Self {
        Self {
            save_dir: save_dir.unwrap_or_else(default_save_dir),
            auto_save,
            catalog: HashMap::new(),
            cache: HashMap::new(),
            max_cached: 8,
            max_hrr_bytes: 1_073_741_824, // 1 GB
            access_counter: 0,
        }
    }

    /// Set the maximum number of nuggets kept in the LRU cache.
    pub fn set_max_cached(&mut self, n: usize) {
        self.max_cached = n.max(1);
    }

    /// Set the maximum total HRR bytes across all cached nuggets.
    pub fn set_max_hrr_bytes(&mut self, bytes: usize) {
        self.max_hrr_bytes = bytes;
    }

    /// Total HRR bytes currently in RAM across all cached nuggets.
    pub fn total_hrr_bytes(&self) -> usize {
        self.cache.values().map(|e| e.nugget.hrr_bytes()).sum()
    }

    // -- catalog management --------------------------------------------------

    /// Scan the save directory and build the catalog of available nuggets.
    /// **Does NOT load any nugget data into memory.**
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
            // Use peek_metadata to get just the name without loading all facts
            match Nugget::peek_metadata(&path) {
                Ok((name, _count)) => {
                    self.catalog.insert(name, path);
                }
                Err(_) => continue, // skip corrupt files
            }
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
        if self.catalog.contains_key(name) || self.cache.contains_key(name) {
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

        // Register in catalog
        let path = self.save_dir.join(format!("{name}.nugget.json"));
        self.catalog.insert(name.to_string(), path);

        // Insert into cache (may evict LRU)
        self.insert_into_cache(name.to_string(), n);
        &mut self.cache.get_mut(name).unwrap().nugget
    }

    /// Get a nugget by name.  Loads from disk if not cached.
    /// Panics if the nugget doesn't exist in catalog or cache.
    pub fn get(&mut self, name: &str) -> &Nugget {
        self.ensure_loaded(name);
        &self.cache.get(name).unwrap().nugget
    }

    /// Get a mutable reference to a nugget.  Loads from disk if not cached.
    /// Panics if the nugget doesn't exist.
    pub fn get_mut(&mut self, name: &str) -> &mut Nugget {
        self.ensure_loaded(name);
        &mut self.cache.get_mut(name).unwrap().nugget
    }

    /// Get or create a nugget with default settings.
    pub fn get_or_create(&mut self, name: &str) -> &mut Nugget {
        if !self.catalog.contains_key(name) && !self.cache.contains_key(name) {
            self.create(name, None, None, None);
        } else {
            self.ensure_loaded(name);
        }
        &mut self.cache.get_mut(name).unwrap().nugget
    }

    /// Remove a nugget and its persisted file.
    pub fn remove(&mut self, name: &str) {
        if !self.catalog.contains_key(name) && !self.cache.contains_key(name) {
            panic!("Nugget {:?} not found", name);
        }
        let path = self.save_dir.join(format!("{name}.nugget.json"));
        if path.exists() {
            let _ = fs::remove_file(&path);
        }
        self.catalog.remove(name);
        self.cache.remove(name);
    }

    /// List status of all nuggets (reads from disk on-demand for uncached ones).
    pub fn list(&mut self) -> Vec<NuggetStatus> {
        let names: Vec<String> = self.catalog.keys().cloned().collect();
        let mut statuses = Vec::with_capacity(names.len());

        // For cached nuggets, use in-memory status
        // For uncached, peek metadata to avoid full load
        for name in &names {
            if let Some(entry) = self.cache.get(name) {
                statuses.push(entry.nugget.status());
            } else if let Some(path) = self.catalog.get(name) {
                // Peek without loading — light disk read
                if let Ok((n, count)) = Nugget::peek_metadata(path) {
                    statuses.push(NuggetStatus {
                        name: n,
                        fact_count: count,
                        dimension: 0,
                        banks: 0,
                        ensembles: 0,
                        capacity_used_pct: 0.0,
                        capacity_warning: String::new(),
                        max_facts: 0,
                    });
                }
            }
        }
        statuses
    }

    // -- convenience pass-throughs -------------------------------------------

    /// Remember a fact in a specific nugget.
    /// Pre-warms HRR eagerly (build cost is ~2-3ms per nugget).
    pub fn remember(&mut self, nugget_name: &str, key: &str, value: &str) {
        let nugget = self.get_mut(nugget_name);
        nugget.remember(key, value);
        nugget.ensure_hrr_hot();
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
            self.enforce_hrr_budget();
            return ShelfRecallResult {
                result,
                nugget_name: Some(name.to_string()),
            };
        }

        // Broadcast: parallel recall across all nuggets
        // 1. Ensure all nuggets are loaded (sequential — disk I/O)
        let names: Vec<String> = self.catalog.keys().cloned().collect();
        for name in &names {
            self.ensure_loaded(name);
        }

        // 2. Extract nuggets for parallel processing
        let mut entries: Vec<(String, CachedNugget)> = names.iter()
            .filter_map(|n| self.cache.remove_entry(n))
            .collect();

        // 3. Parallel recall via rayon (Nugget is Send + Sync)
        let query_owned = query.to_string();
        let session_owned = session_id.to_string();
        let results: Vec<(String, CachedNugget, RecallResult)> = entries
            .into_par_iter()
            .map(|(name, mut entry)| {
                let result = entry.nugget.recall(&query_owned, &session_owned);
                (name, entry, result)
            })
            .collect();

        // 4. Put nuggets back and find best result
        // We track the best match, with a massive bias for EXACT key matches
        let mut best_is_exact = false;
        let mut best = ShelfRecallResult {
            result: RecallResult {
                answer: None, confidence: 0.0, margin: 0.0,
                found: false, key: String::new(),
            },
            nugget_name: None,
        };
        for (name, entry, result) in results {
            if result.found {
                let is_exact = result.key.eq_ignore_ascii_case(&query_owned);
                
                let should_update = if is_exact && !best_is_exact {
                    true // Always upgrade from fuzzy to exact
                } else if !is_exact && best_is_exact {
                    false // Never let a fuzzy match overwrite an exact match
                } else {
                    // Both are exact, or both are fuzzy -> break tie with confidence
                    result.confidence > best.result.confidence
                };

                if should_update {
                    best_is_exact = is_exact;
                    best = ShelfRecallResult {
                        result: result.clone(),
                        nugget_name: Some(name.clone()),
                    };
                    best.result = result;
                }
            }
            self.cache.insert(name, entry);
        }

        self.enforce_hrr_budget();
        best
    }

    /// Forget a fact from a specific nugget.
    pub fn forget(&mut self, nugget_name: &str, key: &str) -> bool {
        self.get_mut(nugget_name).forget(key)
    }

    // -- persistence ---------------------------------------------------------

    /// Save all *cached* nuggets to disk.
    pub fn save_all(&self) {
        for entry in self.cache.values() {
            let _ = entry.nugget.save(None);
        }
    }

    /// Check if a nugget exists (in catalog or cache).
    pub fn has(&self, name: &str) -> bool {
        self.catalog.contains_key(name) || self.cache.contains_key(name)
    }

    /// Number of known nuggets (catalog size).
    pub fn size(&self) -> usize {
        self.catalog.len()
    }

    /// Number of nuggets currently resident in the LRU cache.
    pub fn cached_count(&self) -> usize {
        self.cache.len()
    }

    // -- LRU internals -------------------------------------------------------

    /// Evict HRR vectors from LRU nuggets until total is under budget.
    /// Nugget facts stay in RAM — only the HRR vectors are spilled to disk.
    fn enforce_hrr_budget(&mut self) {
        let total: usize = self.cache.values().map(|e| e.nugget.hrr_bytes()).sum();
        if total <= self.max_hrr_bytes {
            return;
        }

        // Sort by LRU (oldest access first)
        let mut entries: Vec<(String, u64, usize)> = self.cache.iter()
            .filter(|(_, e)| e.nugget.hrr_bytes() > 0)
            .map(|(name, e)| (name.clone(), e.last_access, e.nugget.hrr_bytes()))
            .collect();
        entries.sort_by_key(|(_, access, _)| *access);

        let mut remaining = total;
        for (name, _, bytes) in entries {
            if remaining <= self.max_hrr_bytes { break; }
            if let Some(entry) = self.cache.get_mut(&name) {
                entry.nugget.evict_hrr();
                remaining -= bytes;
            }
        }
    }

    /// Ensure a nugget is loaded into the cache.  If it's already cached,
    /// just bump its access counter.  If not, load from disk and evict
    /// the LRU entry if the cache is full.
    fn ensure_loaded(&mut self, name: &str) {
        if self.cache.contains_key(name) {
            // Bump access counter
            self.access_counter += 1;
            self.cache.get_mut(name).unwrap().last_access = self.access_counter;
            return;
        }

        // Must load from disk
        let t_load = std::time::Instant::now();
        let path = self
            .catalog
            .get(name)
            .unwrap_or_else(|| panic!("Nugget {:?} not found", name))
            .clone();

        let mut nugget = Nugget::load(&path, self.auto_save)
            .unwrap_or_else(|e| panic!("Failed to load nugget {:?}: {}", name, e));
        let load_ms = t_load.elapsed().as_millis();

        // Pre-warm HRR on load so first recall is instant
        let t_warm = std::time::Instant::now();
        nugget.ensure_hrr_hot();
        let warm_ms = t_warm.elapsed().as_millis();
        
        eprintln!("[Shelf] ensure_loaded {} load={}ms prewarm={}ms", name, load_ms, warm_ms);

        self.insert_into_cache(name.to_string(), nugget);
    }

    /// Insert a nugget into the cache, evicting the LRU entry if needed.
    fn insert_into_cache(&mut self, name: String, nugget: Nugget) {
        // Evict LRU if cache is full
        while self.cache.len() >= self.max_cached {
            self.evict_lru();
        }

        self.access_counter += 1;
        self.cache.insert(
            name,
            CachedNugget {
                nugget,
                last_access: self.access_counter,
            },
        );
    }

    /// Evict the least-recently-used nugget from cache.
    /// Saves to disk before evicting.
    fn evict_lru(&mut self) {
        if self.cache.is_empty() {
            return;
        }

        let lru_name = self
            .cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_access)
            .map(|(name, _)| name.clone());

        if let Some(name) = lru_name {
            // Save before evicting
            if let Some(entry) = self.cache.get(&name) {
                let _ = entry.nugget.save(None);
            }
            self.cache.remove(&name);
        }
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
        let (mut shelf, _tmp) = test_shelf();
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

        // Reload — should NOT load anything into memory, just catalog
        let mut shelf2 = NuggetShelf::new(Some(tmp_path), false);
        shelf2.load_all();
        assert_eq!(shelf2.size(), 2);
        assert_eq!(shelf2.cached_count(), 0); // nothing loaded yet!

        // Now access one — should load on demand
        assert_eq!(
            shelf2.recall("k1", Some("a"), "").result.answer.unwrap(),
            "v1"
        );
        assert_eq!(shelf2.cached_count(), 1); // only "a" is loaded
        assert_eq!(
            shelf2.recall("k2", Some("b"), "").result.answer.unwrap(),
            "v2"
        );
        assert_eq!(shelf2.cached_count(), 2); // now both loaded
    }

    #[test]
    fn lru_eviction() {
        let tmp = tempfile::tempdir().unwrap();
        let tmp_path = tmp.path().to_path_buf();

        let mut shelf = NuggetShelf::new(Some(tmp_path), true);
        shelf.set_max_cached(2); // only keep 2 in cache

        shelf.create("a", None, None, None);
        shelf.create("b", None, None, None);
        shelf.remember("a", "k1", "v1");
        shelf.remember("b", "k2", "v2");
        shelf.save_all();

        assert_eq!(shelf.cached_count(), 2);

        // Creating a third should evict the LRU
        shelf.create("c", None, None, None);
        shelf.remember("c", "k3", "v3");

        assert_eq!(shelf.cached_count(), 2);
        assert_eq!(shelf.size(), 3);

        // All data should still be accessible (loaded on demand)
        let r1 = shelf.recall("k1", Some("a"), "");
        assert!(r1.result.found);
        assert_eq!(r1.result.answer.unwrap(), "v1");
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

    #[test]
    fn zero_ram_on_startup() {
        let tmp = tempfile::tempdir().unwrap();
        let tmp_path = tmp.path().to_path_buf();

        // Create some nuggets on disk
        {
            let mut shelf = NuggetShelf::new(Some(tmp_path.clone()), true);
            for i in 0..5 {
                let name = format!("nugget_{i}");
                shelf.create(&name, None, None, None);
                shelf.remember(&name, "key", &format!("value_{i}"));
            }
            shelf.save_all();
        }

        // Fresh shelf — should scan catalog but load ZERO nuggets
        let mut shelf = NuggetShelf::new(Some(tmp_path), false);
        shelf.load_all();
        assert_eq!(shelf.size(), 5);
        assert_eq!(shelf.cached_count(), 0);

        // Accessing one loads only that one
        let r = shelf.recall("key", Some("nugget_3"), "");
        assert!(r.result.found);
        assert_eq!(r.result.answer.unwrap(), "value_3");
        assert_eq!(shelf.cached_count(), 1);
    }
}
