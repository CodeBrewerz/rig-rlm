//! Nugget — a disk-backed memory unit with tiered HRR.
//!
//! Facts persist as JSON at `~/.nuggets/<name>.nugget.json`.
//! HRR vectors are serialized to `~/.nuggets/<name>.hrr.bin`.
//!
//! **Tiered HRR design:**
//! - **Hot**: HRR vectors in RAM — fast decode via unbind + cleanup.
//! - **Cold**: HRR vectors on disk (`.hrr.bin`) — loaded on demand.
//! - **NotBuilt**: No HRR yet — built lazily on first recall.
//!
//! The HRR engine (`core.rs`, `advanced.rs`) is used with **bounded
//! params** (D capped at 2048, single bank, single ensemble) so each
//! nugget uses ~5–15 MB regardless of fact count.  The `NuggetShelf`
//! enforces a global byte budget and evicts HRR from LRU nuggets when
//! exceeded.

use hashbrown::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::advanced::CleanupNetwork;
use super::core::{
    bind, dot_product, make_role_keys, make_vocab_keys, orthogonalize,
    seed_from_name, softmax_temp, stack_and_unit_norm, unbind, ComplexVector, Mulberry32,
};

// ---------------------------------------------------------------------------
// Default save directory
// ---------------------------------------------------------------------------

pub fn default_save_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".nuggets")
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub key: String,
    pub value: String,
    #[serde(default)]
    pub hits: u32,
    #[serde(default)]
    pub last_hit_session: String,
}

/// On-disk JSON envelope (backward-compat).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NuggetFile {
    version: u32,
    name: String,
    #[serde(rename = "D")]
    d: usize,
    banks: usize,
    ensembles: usize,
    max_facts: usize,
    facts: Vec<Fact>,
    #[serde(default)]
    config: NuggetConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NuggetConfig {
    #[serde(default = "default_sharpen_p")]
    sharpen_p: f64,
    #[serde(default)]
    corvacs_a: f64,
    #[serde(default = "default_temp_t")]
    temp_t: f64,
    #[serde(default = "default_orth_iters")]
    orth_iters: usize,
}

impl Default for NuggetConfig {
    fn default() -> Self {
        Self { sharpen_p: 1.0, corvacs_a: 0.0, temp_t: 0.9, orth_iters: 1 }
    }
}

fn default_sharpen_p() -> f64 { 1.0 }
fn default_temp_t() -> f64 { 0.9 }
fn default_orth_iters() -> usize { 1 }

// ---------------------------------------------------------------------------
// HRR tiered cache
// ---------------------------------------------------------------------------

/// Pre-built HRR data for fast recall.
struct HrrData {
    /// Per-bank memory vectors (facts distributed round-robin).
    memories: Vec<ComplexVector>,
    sent_key: ComplexVector,
    /// Shared role keys — one per fact position.
    role_keys: Vec<ComplexVector>,
    /// Shared cleanup network — one codebook for all banks.
    cleanup: CleanupNetwork,
    vocab_words: Vec<String>,
    /// O(1) lookup map for fast patching during swaps.
    vocab_idx: HashMap<String, usize>,
    /// Effective dimension used (may differ from configured d).
    effective_d: usize,
    /// Number of banks.
    num_banks: usize,
    /// Estimated RAM in bytes.
    pub byte_size: usize,
}

/// State of HRR vectors for a nugget.
enum HrrState {
    /// Not built yet — facts are new or changed.
    NotBuilt,
    /// Vectors in RAM — fast recall.
    Hot(Box<HrrData>),
    /// Evicted from RAM; `.hrr.bin` exists on disk.
    Cold,
}

fn estimate_hrr_bytes(d: usize, v: usize, n: usize, num_banks: usize) -> usize {
    let cv = 2 * d * 8; // one ComplexVector
    let memories = num_banks * cv;          // per-bank memory vectors
    let vocab_norm = v * 2 * d * 8;         // shared
    let sent_key = cv;                      // shared
    let role_keys = n * cv;                 // shared
    let cleanup = v * cv + v * 2 * d * 8;   // shared codebook + norms
    let overhead = v * 64;
    memories + vocab_norm + sent_key + role_keys + cleanup + overhead
}

// ---------------------------------------------------------------------------
// Options & Result types
// ---------------------------------------------------------------------------

pub struct NuggetOpts {
    pub name: String,
    pub d: usize,
    pub banks: usize,
    pub ensembles: usize,
    pub auto_save: bool,
    pub save_dir: PathBuf,
    pub max_facts: usize,
}

impl Default for NuggetOpts {
    fn default() -> Self {
        Self {
            name: "default".into(),
            d: 512,
            banks: 4,
            ensembles: 1,
            auto_save: true,
            save_dir: default_save_dir(),
            max_facts: 500,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RecallResult {
    pub answer: Option<String>,
    pub confidence: f64,
    pub margin: f64,
    pub found: bool,
    pub key: String,
}

impl RecallResult {
    fn empty() -> Self {
        Self { answer: None, confidence: 0.0, margin: 0.0, found: false, key: String::new() }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct NuggetStatus {
    pub name: String,
    pub fact_count: usize,
    pub dimension: usize,
    pub banks: usize,
    pub ensembles: usize,
    pub capacity_used_pct: f64,
    pub capacity_warning: String,
    pub max_facts: usize,
}

// ---------------------------------------------------------------------------
// Nugget
// ---------------------------------------------------------------------------

pub struct Nugget {
    pub name: String,
    d: usize,
    banks: usize,
    ensembles: usize,
    pub auto_save: bool,
    pub save_dir: PathBuf,
    pub max_facts: usize,

    fuzzy_threshold: f64,

    facts: Vec<Fact>,
    tag_to_pos: HashMap<String, usize>,
    dirty: bool,

    /// Fact-level LRU: indices into `facts` currently in HRR bank.
    /// Ordered MRU-first (index 0 = most recently used).
    hot_indices: Vec<usize>,

    /// Tiered HRR cache.
    hrr_state: HrrState,
}

impl Nugget {
    pub fn new(opts: NuggetOpts) -> Self {
        Self {
            name: opts.name, d: opts.d, banks: opts.banks, ensembles: opts.ensembles,
            auto_save: opts.auto_save, save_dir: opts.save_dir, max_facts: opts.max_facts,
            fuzzy_threshold: 0.55,
            facts: Vec::new(), tag_to_pos: HashMap::new(), dirty: false,
            hot_indices: Vec::new(),
            hrr_state: HrrState::NotBuilt,
        }
    }

    // -- public API ----------------------------------------------------------

    pub fn remember(&mut self, key: &str, value: &str) {
        let key = key.trim();
        let value = value.trim();
        if key.is_empty() || value.is_empty() { return; }

        let mut found = false;
        for f in &mut self.facts {
            if f.key.eq_ignore_ascii_case(key) {
                f.value = value.to_string();
                found = true;
                break;
            }
        }
        if !found {
            self.facts.push(Fact {
                key: key.to_string(), value: value.to_string(),
                hits: 0, last_hit_session: String::new(),
            });
        }

        // Update hot_indices: add new fact
        let new_pos = if found {
            self.facts.iter().position(|f| f.key.eq_ignore_ascii_case(key)).unwrap()
        } else {
            self.facts.len() - 1
        };
        // Remove if already in hot set (will re-add at end)
        self.hot_indices.retain(|&i| i != new_pos);
        // Evict LRU (first = oldest) if over capacity
        if self.max_facts > 0 && self.hot_indices.len() >= self.max_facts {
            self.hot_indices.remove(0);
        }
        // Add at end (MRU)
        self.hot_indices.push(new_pos);

        self.dirty = true;
        self.invalidate_hrr();
        if self.auto_save { let _ = self.save(None); }
    }

    /// Recall a fact.  Ensures the fact is in the HRR bank (swapping
    /// in if cold) so every recall gets full HRR accuracy.
    pub fn recall(&mut self, query: &str, session_id: &str) -> RecallResult {
        if self.facts.is_empty() { return RecallResult::empty(); }

        if self.dirty {
            self.refresh_index();
            self.dirty = false;
        }

        let tag = match self.resolve_tag(query) {
            Some(t) => t,
            None => return RecallResult::empty(),
        };
        let pos = match self.tag_to_pos.get(&tag) {
            Some(&p) => p,
            None => return RecallResult::empty(),
        };

        // Ensure this fact is in the hot set (swap if cold)
        let (needs_update, patch_info) = self.ensure_fact_hot(pos);
        if let Some((evicted, new_pos, slot_idx)) = patch_info {
            if !self.try_patch_hrr(evicted, new_pos, slot_idx) {
                self.invalidate_hrr();
            }
        } else if needs_update {
            self.invalidate_hrr();
        }
        self.ensure_hrr_hot();

        // Decode via HRR — fact is guaranteed to be in the bank
        let hrr_pos = self.hot_indices.iter().position(|&i| i == pos);
        let (word, confidence, margin) = match (&self.hrr_state, hrr_pos) {
            (HrrState::Hot(hrr), Some(hp)) if hp < hrr.role_keys.len() => {
                Self::decode_hrr(hrr, hp)
            }
            _ => {
                // Safety fallback (shouldn't normally hit)
                (self.facts[pos].value.clone(), 1.0, 1.0)
            }
        };

        // Hit tracking
        if !session_id.is_empty() {
            if let Some(fact) = self.facts.get_mut(pos) {
                if fact.last_hit_session != session_id {
                    fact.hits += 1;
                    fact.last_hit_session = session_id.to_string();
                    if self.auto_save { let _ = self.save(None); }
                }
            }
        }

        RecallResult { answer: Some(word), confidence, margin, found: true, key: tag }
    }

    pub fn forget(&mut self, key: &str) -> bool {
        let lower = key.to_lowercase();
        let lower = lower.trim();
        let before = self.facts.len();
        self.facts.retain(|f| f.key.to_lowercase() != lower);
        let removed = self.facts.len() < before;
        if removed {
            // Rebuild hot_indices (positions may have shifted)
            self.rebuild_hot_indices();
            self.dirty = true;
            self.invalidate_hrr();
            if self.auto_save { let _ = self.save(None); }
        }
        removed
    }

    pub fn facts(&self) -> Vec<Fact> {
        self.facts.iter().map(|f| Fact {
            key: f.key.clone(), value: f.value.clone(),
            hits: f.hits, last_hit_session: f.last_hit_session.clone(),
        }).collect()
    }

    pub fn clear(&mut self) {
        self.facts.clear();
        self.tag_to_pos.clear();
        self.hot_indices.clear();
        self.dirty = false;
        self.invalidate_hrr();
        if self.auto_save { let _ = self.save(None); }
    }

    pub fn status(&self) -> NuggetStatus {
        let n = self.facts.len();
        let warning = if self.max_facts > 0 && n > self.max_facts * 9 / 10 {
            "WARNING: approaching max_facts limit".into()
        } else { String::new() };
        let pct = if self.max_facts > 0 { (n as f64 / self.max_facts as f64) * 100.0 } else { 0.0 };
        NuggetStatus {
            name: self.name.clone(), fact_count: n, dimension: self.d,
            banks: self.banks, ensembles: self.ensembles,
            capacity_used_pct: (pct * 10.0).round() / 10.0,
            capacity_warning: warning, max_facts: self.max_facts,
        }
    }

    pub fn fact_count(&self) -> usize { self.facts.len() }

    /// Current HRR RAM usage in bytes (0 if Cold or NotBuilt).
    pub fn hrr_bytes(&self) -> usize {
        match &self.hrr_state {
            HrrState::Hot(hrr) => hrr.byte_size,
            _ => 0,
        }
    }

    /// Evict HRR vectors from RAM → Cold.  Saves `.hrr.bin` first.
    pub fn evict_hrr(&mut self) {
        if let HrrState::Hot(ref hrr) = self.hrr_state {
            let _ = self.save_hrr(hrr);
            self.hrr_state = HrrState::Cold;
        }
    }

    // -- persistence ---------------------------------------------------------

    pub fn save(&self, path: Option<&Path>) -> Result<PathBuf, std::io::Error> {
        let path = match path {
            Some(p) => p.to_path_buf(),
            None => {
                fs::create_dir_all(&self.save_dir)?;
                self.save_dir.join(format!("{}.nugget.json", self.name))
            }
        };
        let data = NuggetFile {
            version: 4, name: self.name.clone(), d: self.d,
            banks: self.banks, ensembles: self.ensembles,
            max_facts: self.max_facts, facts: self.facts.clone(),
            config: NuggetConfig::default(),
        };
        let json = serde_json::to_string(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, &json)?;
        fs::rename(&tmp, &path)?;
        Ok(path)
    }

    pub fn load(path: &Path, auto_save: bool) -> Result<Self, std::io::Error> {
        let raw = fs::read_to_string(path)?;
        let data: NuggetFile = serde_json::from_str(&raw)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let save_dir = path.parent().map(|p| p.to_path_buf()).unwrap_or_else(default_save_dir);

        let hrr_path = save_dir.join(format!("{}.hrr.bin", data.name));
        let hrr_state = if hrr_path.exists() { HrrState::Cold } else { HrrState::NotBuilt };

        let mut n = Self {
            name: data.name, d: data.d, banks: data.banks, ensembles: data.ensembles,
            auto_save, save_dir, max_facts: data.max_facts,
            fuzzy_threshold: 0.55,
            facts: data.facts, tag_to_pos: HashMap::new(), dirty: false,
            hot_indices: Vec::new(),
            hrr_state,
        };
        if !n.facts.is_empty() {
            n.refresh_index();
            // Initialize hot_indices: most recent max_facts (or all)
            let total = n.facts.len();
            let start = if n.max_facts > 0 && total > n.max_facts {
                total - n.max_facts
            } else { 0 };
            n.hot_indices = (start..total).collect();
        }
        Ok(n)
    }

    pub fn peek_metadata(path: &Path) -> Result<(String, usize), std::io::Error> {
        let raw = fs::read_to_string(path)?;
        let data: NuggetFile = serde_json::from_str(&raw)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok((data.name, data.facts.len()))
    }

    // -- HRR internals -------------------------------------------------------

    fn hrr_path(&self) -> PathBuf {
        self.save_dir.join(format!("{}.hrr.bin", self.name))
    }

    /// Invalidate HRR — called on any mutation.
    fn invalidate_hrr(&mut self) {
        self.hrr_state = HrrState::NotBuilt;
        let _ = fs::remove_file(self.hrr_path()); // delete stale .hrr.bin
    }

    /// Ensure HRR vectors are in RAM (Hot).
    fn ensure_hrr_hot(&mut self) {
        match &self.hrr_state {
            HrrState::Hot(_) => return,
            HrrState::Cold => {
                if let Ok(hrr) = self.load_hrr() {
                    self.hrr_state = HrrState::Hot(Box::new(hrr));
                } else {
                    self.build_and_cache_hrr();
                }
            }
            HrrState::NotBuilt => {
                if !self.facts.is_empty() {
                    self.build_and_cache_hrr();
                }
            }
        }
    }

    fn build_and_cache_hrr(&mut self) {
        let hrr = self.build_hrr();
        if self.auto_save { let _ = self.save_hrr(&hrr); }
        self.hrr_state = HrrState::Hot(Box::new(hrr));
    }

    /// Ensure a fact is in the hot set. Returns (needs_update, patch_info).
    /// patch_info = Some((evicted_pos, new_pos, slot_idx))
    fn ensure_fact_hot(&mut self, pos: usize) -> (bool, Option<(Option<usize>, usize, usize)>) {
        // Already in hot set? No change needed.
        if self.hot_indices.contains(&pos) {
            return (false, None);
        }
        // Not in hot set — swap in, evict LRU (last entry) if needed
        let mut evicted = None;
        if self.max_facts > 0 && self.hot_indices.len() >= self.max_facts {
            evicted = self.hot_indices.pop(); // evict LRU (last = oldest)
        }
        let slot_idx = self.hot_indices.len();
        self.hot_indices.push(pos);
        (true, Some((evicted, pos, slot_idx)))
    }

    /// Try to patch the existing HRR by subtracting the evicted fact and adding the new fact.
    /// Returns true if successful, false if a full rebuild is required.
    fn try_patch_hrr(&mut self, evicted_idx: Option<usize>, new_pos: usize, hot_idx: usize) -> bool {
        let hrr = match &mut self.hrr_state {
            HrrState::Hot(hrr) => hrr,
            _ => return false,
        };

        if hot_idx >= hrr.role_keys.len() {
            return false;
        }

        let new_word = &self.facts[new_pos].value;
        let new_w_idx = match hrr.vocab_idx.get(new_word) {
            Some(&i) => i,
            None => return false,
        };

        let old_w_idx = match evicted_idx {
            Some(epos) => {
                let old_word = &self.facts[epos].value;
                match hrr.vocab_idx.get(old_word) {
                    Some(&i) => Some(i),
                    None => return false,
                }
            }
            None => None,
        };

        let bank = hot_idx % hrr.num_banks;
        
        let num_facts = hrr.role_keys.len();
        let mut bank_count = 0;
        for i in 0..num_facts {
            if i % hrr.num_banks == bank { bank_count += 1; }
        }
        let inv_scale = if bank_count > 0 { (bank_count as f64).sqrt() } else { 1.0 };
        let scale = if bank_count > 0 { 1.0 / (bank_count as f64).sqrt() } else { 1.0 };

        let sent_key = &hrr.sent_key;
        let role_key = &hrr.role_keys[hot_idx];

        let mut old_binding = None;
        if let Some(owi) = old_w_idx {
            let old_w_key = &hrr.cleanup.codebook()[owi];
            old_binding = Some(bind(&bind(sent_key, role_key), old_w_key));
        }

        let new_w_key = &hrr.cleanup.codebook()[new_w_idx];
        let new_binding = bind(&bind(sent_key, role_key), new_w_key);

        let mem = &mut hrr.memories[bank];
        for d in 0..hrr.effective_d {
            let mut re = mem.re[d] * inv_scale;
            let mut im = mem.im[d] * inv_scale;
            if let Some(ob) = &old_binding {
                re -= ob.re[d];
                im -= ob.im[d];
            }
            re += new_binding.re[d];
            im += new_binding.im[d];
            mem.re[d] = re * scale;
            mem.im[d] = im * scale;
        }

        true
    }

    /// Rebuild hot_indices after facts were deleted (positions may have shifted).
    fn rebuild_hot_indices(&mut self) {
        let total = self.facts.len();
        self.hot_indices.retain(|&i| i < total);
        // If hot set is smaller than capacity, fill from recent facts
        if self.max_facts > 0 {
            for i in (0..total).rev() {
                if self.hot_indices.len() >= self.max_facts { break; }
                if !self.hot_indices.contains(&i) {
                    self.hot_indices.push(i);
                }
            }
        }
    }

    /// Build HRR from facts with bounded params and multi-bank sharding.
    /// Encodes only the facts in `hot_indices` (the dynamic hot set).
    fn build_hrr(&self) -> HrrData {
        // Gather hot facts
        let hot_facts: Vec<&Fact> = self.hot_indices.iter()
            .filter_map(|&i| self.facts.get(i))
            .collect();
        let num_facts = hot_facts.len();

        // Unique values → vocab (from ALL facts, so any swap can be patched)
        let mut seen = HashSet::new();
        let mut vocab_words = Vec::new();
        for f in &self.facts {
            if seen.insert(f.value.clone()) { vocab_words.push(f.value.clone()); }
        }
        let v = vocab_words.len();
        let mut idx_w: HashMap<String, usize> = HashMap::with_capacity(v);
        for (i, w) in vocab_words.iter().enumerate() { idx_w.insert(w.clone(), i); }

        // Bounded dimension: cap at 2048, ensure power-of-two, min 256
        let effective_d = self.d.max(v.saturating_mul(8).max(256)).min(2048).next_power_of_two();

        // Auto-scale banks: max 20 facts per bank for high accuracy
        let max_per_bank = 20usize;
        let num_banks = if num_facts == 0 { 1 } else {
            self.banks.max((num_facts + max_per_bank - 1) / max_per_bank)
        };

        // Deterministic keys from name seed (shared across banks)
        let seed = seed_from_name(&self.name);
        let mut rng = Mulberry32::new(seed);

        let mut vocab_keys = make_vocab_keys(v, effective_d, &mut rng);
        if v <= 100 {
            vocab_keys = orthogonalize(&vocab_keys, 1, 0.4);
        }

        let sent_keys = make_vocab_keys(1, effective_d, &mut rng);
        let sent_key = sent_keys[0].clone();
        let role_keys = make_role_keys(effective_d, num_facts);

        // Build per-bank memory vectors (round-robin assignment)
        let mut memories: Vec<ComplexVector> = (0..num_banks)
            .map(|_| ComplexVector::zeros(effective_d))
            .collect();
        let mut bank_counts = vec![0usize; num_banks];

        for (i, f) in hot_facts.iter().enumerate() {
            let bank = i % num_banks;
            let w_key = &vocab_keys[idx_w[&f.value]];
            let binding = bind(&bind(&sent_key, &role_keys[i]), w_key);
            for d in 0..effective_d {
                memories[bank].re[d] += binding.re[d];
                memories[bank].im[d] += binding.im[d];
            }
            bank_counts[bank] += 1;
        }
        // Scale each bank by 1/sqrt(items_in_bank)
        for (b, mem) in memories.iter_mut().enumerate() {
            if bank_counts[b] > 0 {
                let scale = 1.0 / (bank_counts[b] as f64).sqrt();
                for d in 0..effective_d {
                    mem.re[d] *= scale;
                    mem.im[d] *= scale;
                }
            }
        }

        let cleanup = CleanupNetwork::new(&vocab_keys);
        let byte_size = estimate_hrr_bytes(effective_d, v, num_facts, num_banks);

        HrrData {
            memories, sent_key, role_keys, cleanup,
            vocab_words, vocab_idx: idx_w, effective_d, num_banks, byte_size,
        }
    }

    /// Decode a fact position from HRR memory.
    /// Selects the correct bank (round-robin), unbinds, then uses
    /// cleanup + cosine sim for accurate recall.
    fn decode_hrr(hrr: &HrrData, pos: usize) -> (String, f64, f64) {
        let v = hrr.vocab_words.len();
        if v == 0 { return (String::new(), 0.0, 0.0); }

        // Select bank containing this fact
        let bank = pos % hrr.num_banks;
        let bank_mem = &hrr.memories[bank];

        // Unbind sentence key, then role key using pre-allocated zero-buffers
        let d = hrr.effective_d;
        let mut tmp = ComplexVector::zeros(d);
        super::core::unbind_into(bank_mem, &hrr.sent_key, &mut tmp);
        
        let d2 = d * 2;
        let mut rec2 = vec![0.0; d2];
        super::core::unbind_into_real(&tmp, &hrr.role_keys[pos], &mut rec2);

        // Cleanup Network: exact nearest-neighbor & sims (Zero copy, mutating rec2)
        let (cleanup_idx, cleanup_sim, _, sims) = hrr.cleanup.cleanup_with_sims_real(&mut rec2);

        let probs = softmax_temp(&sims, 0.9);
        let softmax_best = probs.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0);

        let best_idx = if cleanup_sim > 0.3 { cleanup_idx } else { softmax_best };

        // Confidence & margin
        let confidence = cleanup_sim.max(probs[softmax_best]);
        let mut sorted = probs.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let margin = if sorted.len() > 1 { sorted[0] - sorted[1] } else { 1.0 };

        (hrr.vocab_words[best_idx].clone(), confidence, margin)
    }

    // -- HRR binary serialization --------------------------------------------
    // Format: "HRR2" | D:u32 | V:u32 | N:u32 | B:u32 | memories[B×2D] | vocab_json

    fn save_hrr(&self, hrr: &HrrData) -> Result<PathBuf, std::io::Error> {
        let path = self.hrr_path();
        fs::create_dir_all(&self.save_dir)?;
        let d = hrr.effective_d;
        let v = hrr.vocab_words.len();
        let n = hrr.role_keys.len();
        let b = hrr.num_banks;

        let mut buf = Vec::with_capacity(20 + b * 2 * d * 8 + 512);
        buf.extend_from_slice(b"HRR2");
        buf.extend_from_slice(&(d as u32).to_le_bytes());
        buf.extend_from_slice(&(v as u32).to_le_bytes());
        buf.extend_from_slice(&(n as u32).to_le_bytes());
        buf.extend_from_slice(&(b as u32).to_le_bytes());
        for mem in &hrr.memories {
            for &x in &mem.re { buf.extend_from_slice(&x.to_le_bytes()); }
            for &x in &mem.im { buf.extend_from_slice(&x.to_le_bytes()); }
        }

        let vocab_json = serde_json::to_string(&hrr.vocab_words)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        buf.extend_from_slice(vocab_json.as_bytes());

        let tmp = path.with_extension("bin.tmp");
        fs::write(&tmp, &buf)?;
        fs::rename(&tmp, &path)?;
        Ok(path)
    }

    fn load_hrr(&self) -> Result<HrrData, std::io::Error> {
        let buf = fs::read(self.hrr_path())?;
        if buf.len() < 20 || &buf[0..4] != b"HRR2" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad HRR magic"));
        }

        let d = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
        let v = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
        let n = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;
        let num_banks = u32::from_le_bytes(buf[16..20].try_into().unwrap()) as usize;

        let mem_data_size = num_banks * 2 * d * 8;
        let expected = 20 + mem_data_size;
        if buf.len() < expected {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "truncated"));
        }

        let mut memories = Vec::with_capacity(num_banks);
        for b in 0..num_banks {
            let bank_off = 20 + b * 2 * d * 8;
            let mut re = vec![0.0f64; d];
            let mut im = vec![0.0f64; d];
            for i in 0..d {
                let off = bank_off + i * 8;
                re[i] = f64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
            }
            for i in 0..d {
                let off = bank_off + d * 8 + i * 8;
                im[i] = f64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
            }
            memories.push(ComplexVector { re, im });
        }

        let vocab_json = std::str::from_utf8(&buf[expected..])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let vocab_words: Vec<String> = serde_json::from_str(vocab_json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        if vocab_words.len() != v {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "vocab mismatch"));
        }

        let seed = seed_from_name(&self.name);
        let mut rng = Mulberry32::new(seed);
        let mut vocab_keys = make_vocab_keys(v, d, &mut rng);
        if v <= 100 { vocab_keys = orthogonalize(&vocab_keys, 1, 0.4); }
        let sent_keys = make_vocab_keys(1, d, &mut rng);
        let sent_key = sent_keys[0].clone();
        let role_keys = make_role_keys(d, n);
        let cleanup = CleanupNetwork::new(&vocab_keys);
        let byte_size = estimate_hrr_bytes(d, v, n, num_banks);

        let mut vocab_idx: HashMap<String, usize> = HashMap::with_capacity(v);
        for (i, w) in vocab_words.iter().enumerate() { vocab_idx.insert(w.clone(), i); }

        Ok(HrrData {
            memories, sent_key, role_keys, cleanup,
            vocab_words, vocab_idx, effective_d: d, num_banks, byte_size,
        })
    }

    // -- index & fuzzy match -------------------------------------------------

    fn refresh_index(&mut self) {
        self.tag_to_pos.clear();
        for (i, f) in self.facts.iter().enumerate() {
            self.tag_to_pos.insert(f.key.clone(), i);
        }
    }

    fn resolve_tag(&self, query: &str) -> Option<String> {
        if self.tag_to_pos.is_empty() { return None; }
        let text = query.to_lowercase();
        let text = text.trim();
        let tags: Vec<&String> = self.tag_to_pos.keys().collect();

        // Exact
        for t in &tags { if t.to_lowercase() == text { return Some((*t).clone()); } }
        // Substring
        for t in &tags {
            let lower = t.to_lowercase();
            if lower.contains(text) || text.contains(lower.as_str()) { return Some((*t).clone()); }
        }
        // Fuzzy
        let mut best: Option<&String> = None;
        let mut best_score = 0.0;
        for t in &tags {
            let s = sequence_match_ratio(text, &t.to_lowercase());
            if s > best_score { best = Some(t); best_score = s; }
        }
        if best_score >= self.fuzzy_threshold { best.map(|t| (*t).clone()) } else { None }
    }
}

// ---------------------------------------------------------------------------
// Fuzzy matching
// ---------------------------------------------------------------------------

fn sequence_match_ratio(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() { return 1.0; }
    if a.is_empty() || b.is_empty() { return 0.0; }
    let matches = count_matches(a.as_bytes(), b.as_bytes());
    (2.0 * matches as f64) / (a.len() + b.len()) as f64
}

fn count_matches(a: &[u8], b: &[u8]) -> usize {
    let (m, n) = (a.len(), b.len());
    let mut total = 0;
    let mut used_a = vec![false; m];
    let mut used_b = vec![false; n];
    loop {
        let (mut best_len, mut best_i, mut best_j) = (0, 0, 0);
        for i in 0..m {
            if used_a[i] { continue; }
            for j in 0..n {
                if used_b[j] { continue; }
                let mut len = 0;
                while i + len < m && j + len < n && !used_a[i + len] && !used_b[j + len] && a[i + len] == b[j + len] { len += 1; }
                if len > best_len { best_len = len; best_i = i; best_j = j; }
            }
        }
        if best_len == 0 { break; }
        for k in 0..best_len { used_a[best_i + k] = true; used_b[best_j + k] = true; }
        total += best_len;
    }
    total
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_opts(name: &str) -> NuggetOpts {
        NuggetOpts { name: name.into(), d: 512, banks: 2, auto_save: false, ..Default::default() }
    }

    #[test]
    fn remember_and_recall() {
        let mut n = Nugget::new(test_opts("test"));
        n.remember("color", "blue");
        let r = n.recall("color", "");
        assert!(r.found);
        assert_eq!(r.answer.unwrap(), "blue");
        assert!(r.confidence > 0.0);
    }

    #[test]
    fn upsert_on_duplicate_key() {
        let mut n = Nugget::new(test_opts("test"));
        n.remember("color", "blue");
        n.remember("color", "red");
        assert_eq!(n.facts().len(), 1);
        assert_eq!(n.recall("color", "").answer.unwrap(), "red");
    }

    #[test]
    fn forget_fact() {
        let mut n = Nugget::new(test_opts("test"));
        n.remember("color", "blue");
        assert!(n.forget("color"));
        assert_eq!(n.facts().len(), 0);
        assert!(!n.forget("nonexistent"));
    }

    #[test]
    fn clear_all() {
        let mut n = Nugget::new(test_opts("test"));
        n.remember("a", "1");
        n.remember("b", "2");
        n.clear();
        assert_eq!(n.facts().len(), 0);
    }

    #[test]
    fn status_info() {
        let mut n = Nugget::new(test_opts("test"));
        n.remember("a", "1");
        let s = n.status();
        assert_eq!(s.name, "test");
        assert_eq!(s.fact_count, 1);
    }

    #[test]
    fn save_and_load() {
        let tmp = tempfile::tempdir().unwrap();
        let mut n = Nugget::new(NuggetOpts {
            name: "persist".into(), d: 512, banks: 2, auto_save: false,
            save_dir: tmp.path().to_path_buf(), ..Default::default()
        });
        n.remember("lang", "typescript");
        n.remember("color", "green");
        let path = n.save(None).unwrap();

        let mut loaded = Nugget::load(&path, false).unwrap();
        assert_eq!(loaded.facts().len(), 2);
        let r = loaded.recall("lang", "");
        assert!(r.found);
        assert_eq!(r.answer.unwrap(), "typescript");
    }

    #[test]
    fn hit_tracking_per_session() {
        let mut n = Nugget::new(test_opts("hits"));
        n.remember("key", "value");
        n.recall("key", "session-1");
        n.recall("key", "session-1");
        assert_eq!(n.facts()[0].hits, 1);
        n.recall("key", "session-2");
        assert_eq!(n.facts()[0].hits, 2);
    }

    #[test]
    fn max_facts_hot_set_limit() {
        let mut n = Nugget::new(NuggetOpts {
            name: "limited".into(), d: 512, banks: 2, auto_save: false,
            max_facts: 3, ..Default::default()
        });
        n.remember("a", "1"); n.remember("b", "2");
        n.remember("c", "3"); n.remember("d", "4");
        // All 4 facts are stored (no eviction from vec)
        assert_eq!(n.facts().len(), 4);
        // But only 3 are in the hot set
        assert_eq!(n.hot_indices.len(), 3);
        // All 4 are still recallable (cold ones swap in)
        assert_eq!(n.recall("a", "").answer.unwrap(), "1");
        assert_eq!(n.recall("d", "").answer.unwrap(), "4");
    }

    #[test]
    fn multiple_facts_recall() {
        let mut n = Nugget::new(NuggetOpts {
            name: "multi".into(), d: 1024, banks: 4, auto_save: false, ..Default::default()
        });
        n.remember("name", "Alice");
        n.remember("pet", "cat");
        n.remember("city", "London");
        assert_eq!(n.recall("name", "").answer.unwrap(), "Alice");
        assert_eq!(n.recall("pet", "").answer.unwrap(), "cat");
        assert_eq!(n.recall("city", "").answer.unwrap(), "London");
    }

    #[test]
    fn fuzzy_key_match() {
        let mut n = Nugget::new(test_opts("fuzzy"));
        n.remember("favorite color", "blue");
        let r = n.recall("color", "");
        assert!(r.found);
        assert_eq!(r.answer.unwrap(), "blue");
    }

    #[test]
    fn unknown_query_not_found() {
        let mut n = Nugget::new(test_opts("empty"));
        assert!(!n.recall("anything", "").found);
    }

    #[test]
    fn ignores_empty_key_value() {
        let mut n = Nugget::new(test_opts("test"));
        n.remember("", "value"); n.remember("key", ""); n.remember("  ", "value");
        assert_eq!(n.facts().len(), 0);
    }

    #[test]
    fn sequence_match_ratio_works() {
        assert!((sequence_match_ratio("abc", "abc") - 1.0).abs() < f64::EPSILON);
        assert!(sequence_match_ratio("abc", "xyz") < 0.5);
    }

    #[test]
    fn hrr_survives_disk_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let mut n = Nugget::new(NuggetOpts {
            name: "hrr_disk".into(), d: 512, auto_save: true,
            save_dir: tmp.path().to_path_buf(), ..Default::default()
        });
        n.remember("color", "blue");
        n.remember("pet", "cat");
        n.remember("city", "London");

        // Build HRR (triggers on first recall)
        let r1 = n.recall("color", "");
        assert!(r1.found);
        assert_eq!(r1.answer.unwrap(), "blue");
        assert!(n.hrr_bytes() > 0, "HRR should be Hot");

        // Evict HRR to Cold (saves .hrr.bin)
        n.evict_hrr();
        assert_eq!(n.hrr_bytes(), 0, "HRR should be Cold");
        assert!(tmp.path().join("hrr_disk.hrr.bin").exists());

        // Recall again — loads from disk, accuracy preserved
        let r2 = n.recall("pet", "");
        assert!(r2.found);
        assert_eq!(r2.answer.unwrap(), "cat");
        assert!(n.hrr_bytes() > 0, "HRR should be Hot again");

        let r3 = n.recall("city", "");
        assert_eq!(r3.answer.unwrap(), "London");
    }

    #[test]
    fn hrr_invalidated_on_mutation() {
        let tmp = tempfile::tempdir().unwrap();
        let mut n = Nugget::new(NuggetOpts {
            name: "hrr_mut".into(), d: 512, auto_save: true,
            save_dir: tmp.path().to_path_buf(), ..Default::default()
        });
        n.remember("a", "1");
        n.recall("a", ""); // builds HRR
        assert!(n.hrr_bytes() > 0);

        n.remember("b", "2"); // invalidates HRR
        // .hrr.bin should be deleted
        assert!(!tmp.path().join("hrr_mut.hrr.bin").exists());
    }
}
