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
use rayon::prelude::*;

use super::turboquant::QjlCleanupNetwork;
use super::keyindex::KeyIndex;
use super::core::{
    bind, bind_role_inline, bind_role_lut, make_role_keys, make_vocab_keys, orthogonalize,
    regenerate_vocab_key, regenerate_sent_key, unbind_role_inline, unbind_role_lut,
    seed_from_name, unbind, ComplexVector, Mulberry32, RopeLut,
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
    // ── Value bank (existing) ──────────────────────────────────
    /// Per-bank memory vectors (facts distributed round-robin), quantized to 4-bits.
    memories: Vec<Vec<u8>>,
    /// Seed for deterministic codebook regeneration.
    seed: u32,
    /// Number of unique vocab words (needed for lazy regeneration).
    vocab_size: usize,
    /// Cached sentence key — only 32KB, avoids O(V×d) PRNG skip on every decode.
    cached_sent_key: ComplexVector,
    /// Pre-computed sin/cos LUT for RoPE (32KB) — eliminates per-element sin_cos.
    rope_lut: RopeLut,
    /// Shared cleanup network — QJL 2-stage (sign-sketch Hamming + top-K f64 re-rank).
    cleanup: QjlCleanupNetwork,
    vocab_words: Vec<String>,
    /// O(1) lookup map for fast patching during swaps.
    vocab_idx: HashMap<String, usize>,
    /// Effective dimension used (may differ from configured d).
    effective_d: usize,
    /// Number of banks.
    num_banks: usize,
    /// Estimated RAM in bytes.
    pub byte_size: usize,

    // ── Key bank (NEW: enables HRR-based key lookup) ──────────
    /// Per-bank memory vectors encoding fact KEYS (same sharding as values).
    key_memories: Vec<Vec<u8>>,
    /// QJL cleanup network for key lookup.
    key_cleanup: QjlCleanupNetwork,
    /// Unique key strings in codebook order.
    key_words: Vec<String>,
    /// key string → codebook index.
    key_idx: HashMap<String, usize>,
    /// Separate sentence key for key encoding (independent PRNG stream).
    key_sent_key: ComplexVector,
}

impl HrrData {
    /// Lazily regenerate a single codebook vector by word index.
    fn regen_vocab_key(&self, idx: usize) -> ComplexVector {
        regenerate_vocab_key(self.seed, idx, self.effective_d)
    }
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

fn estimate_hrr_bytes(d: usize, v: usize, _n: usize, num_banks: usize) -> usize {
    let memories = num_banks * d;           // per-bank memory vectors (4-bit quantized)
    // vocab_keys: NOT stored — regenerated lazily from seed
    // role_keys:  NOT stored — computed inline (RoPE-style)
    // sent_key:   NOT stored — regenerated lazily from seed
    let cleanup_sketches = v * (d / 64 + 1) * 8; // sign-bit sketches only
    let vocab_overhead = v * 64;                  // vocab_words + vocab_idx
    memories + cleanup_sketches + vocab_overhead
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

    /// Inverted keyword index + bi-directional link graph for O(1) lookup.
    key_index: KeyIndex,
}

// Compile-time assertion: Nugget must be Send + Sync for parallel broadcast.
const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    fn check() { assert_send_sync::<Nugget>(); }
};

impl Nugget {
    pub fn new(opts: NuggetOpts) -> Self {
        Self {
            name: opts.name, d: opts.d, banks: opts.banks, ensembles: opts.ensembles,
            auto_save: opts.auto_save, save_dir: opts.save_dir, max_facts: opts.max_facts,
            fuzzy_threshold: 0.55,
            facts: Vec::new(), tag_to_pos: HashMap::new(), dirty: false,
            hot_indices: Vec::new(),
            hrr_state: HrrState::NotBuilt,
            key_index: KeyIndex::new(),
        }
    }

    // -- public API ----------------------------------------------------------

    pub fn remember(&mut self, key: &str, value: &str) {
        let key = key.trim();
        let value = value.trim();
        if key.is_empty() || value.is_empty() { return; }

        let mut found = false;
        let mut found_idx = 0;
        for (i, f) in self.facts.iter_mut().enumerate() {
            if f.key.eq_ignore_ascii_case(key) {
                f.value = value.to_string();
                found = true;
                found_idx = i;
                break;
            }
        }
        if found {
            // Update the index for this fact's new value
            self.key_index.update_value(found_idx, key, value);
        } else {
            let idx = self.facts.len();
            self.facts.push(Fact {
                key: key.to_string(), value: value.to_string(),
                hits: 0, last_hit_session: String::new(),
            });
            // Incrementally add to the index
            self.key_index.add_fact(idx, key, value);
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

        let (tag, match_quality) = match self.resolve_tag(query) {
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
            (HrrState::Hot(hrr), Some(hp)) if hp < self.hot_indices.len() => {
                Self::decode_hrr(hrr, hp)
            }
            _ => {
                // Safety fallback (shouldn't normally hit)
                (self.facts[pos].value.clone(), 1.0, 1.0)
            }
        };

        // Weight confidence by key match quality  
        // Exact match (1.0) preserves full confidence; fuzzy (0.8) reduces it
        let weighted_confidence = confidence * match_quality;

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

        RecallResult { answer: Some(word), confidence: weighted_confidence, margin, found: true, key: tag }
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
        let json = simd_json::to_string(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, &json)?;
        fs::rename(&tmp, &path)?;
        Ok(path)
    }

    pub fn load(path: &Path, auto_save: bool) -> Result<Self, std::io::Error> {
        let mut raw = fs::read(path)?;
        let t_json = std::time::Instant::now();
        let data: NuggetFile = simd_json::from_slice(&mut raw)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let json_ms = t_json.elapsed().as_millis();
        
        let save_dir = path.parent().map(|p| p.to_path_buf()).unwrap_or_else(default_save_dir);

        let hrr_path = save_dir.join(format!("{}.hrr.bin", data.name));
        let hrr_state = if hrr_path.exists() { HrrState::Cold } else { HrrState::NotBuilt };

        let mut n = Self {
            name: data.name.clone(), d: data.d, banks: data.banks, ensembles: data.ensembles,
            auto_save, save_dir, max_facts: data.max_facts,
            fuzzy_threshold: 0.55,
            facts: data.facts, tag_to_pos: HashMap::new(), dirty: false,
            hot_indices: Vec::new(),
            hrr_state,
            key_index: KeyIndex::new(),
        };
        let t_idx = std::time::Instant::now();
        if !n.facts.is_empty() {
            n.refresh_index();
            // Initialize hot_indices: most recent max_facts (or all)
            let total = n.facts.len();
            let start = if n.max_facts > 0 && total > n.max_facts {
                total - n.max_facts
            } else { 0 };
            n.hot_indices = (start..total).collect();
        }
        let idx_ms = t_idx.elapsed().as_millis();
        if json_ms + idx_ms > 10 {
            eprintln!("[Nugget::load] {} json={}ms index={}ms", data.name, json_ms, idx_ms);
        }
        Ok(n)
    }

    pub fn peek_metadata(path: &Path) -> Result<(String, usize), std::io::Error> {
        let mut raw = fs::read(path)?;
        let data: NuggetFile = simd_json::from_slice(&mut raw)
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
    pub fn ensure_hrr_hot(&mut self) {
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
        let t0 = std::time::Instant::now();
        let hrr = self.build_hrr();
        let build_ms = t0.elapsed().as_millis();
        if self.auto_save {
            let t1 = std::time::Instant::now();
            let _ = self.save_hrr(&hrr);
            let save_ms = t1.elapsed().as_millis();
            eprintln!("[HRR] {} build={}ms save={}ms facts={} d={} banks={}",
                     self.name, build_ms, save_ms,
                     self.hot_indices.len(), hrr.effective_d, hrr.num_banks);
        } else {
            eprintln!("[HRR] {} build={}ms facts={} d={} banks={}",
                     self.name, build_ms,
                     self.hot_indices.len(), hrr.effective_d, hrr.num_banks);
        }
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

        if hot_idx >= self.hot_indices.len() {
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
        
        let num_facts = self.hot_indices.len();
        let mut bank_count = 0;
        for i in 0..num_facts {
            if i % hrr.num_banks == bank { bank_count += 1; }
        }
        let inv_scale = if bank_count > 0 { (bank_count as f64).sqrt() } else { 1.0 };
        let scale = if bank_count > 0 { 1.0 / (bank_count as f64).sqrt() } else { 1.0 };

        // Use cached sent_key; direct sin_cos for role (faster than LUT in debug)
        let sent_key = &hrr.cached_sent_key;

        let mut old_binding = None;
        if let Some(owi) = old_w_idx {
            let old_w_key = hrr.regen_vocab_key(owi);
            let bound_sent = bind(sent_key, &old_w_key);
            old_binding = Some(bind_role_inline(&bound_sent, hot_idx));
        }

        let new_w_key = hrr.regen_vocab_key(new_w_idx);
        let bound_sent = bind(sent_key, &new_w_key);
        let new_binding = bind_role_inline(&bound_sent, hot_idx);

        let mut mem = super::turboquant::dequantize_mse_4bit(&hrr.memories[bank]);
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
        hrr.memories[bank] = super::turboquant::quantize_mse_4bit(&mem);

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

        // Auto-scale banks: facts/bank ratio derived from dimension.
        // SNR ≈ sqrt(D / N_bank), need SNR > 4 for reliable QJL cleanup.
        // Use sqrt(D)/2 as a practical sweet spot (conservative for key discrimination).
        // D=256→8/bank, D=512→11/bank, D=1024→16/bank, D=2048→22/bank.
        let max_per_bank = ((effective_d as f64).sqrt() / 2.0).ceil() as usize;
        let max_per_bank = max_per_bank.max(4);
        let num_banks = if num_facts == 0 { 1 } else {
            self.banks.max((num_facts + max_per_bank - 1) / max_per_bank)
        };

        // Deterministic keys from name seed (shared across banks)
        let seed = seed_from_name(&self.name);
        let mut rng = Mulberry32::new(seed);

        // Generate vocab keys temporarily for encoding + cleanup network build
        let t_vocab = std::time::Instant::now();
        let mut vocab_keys = make_vocab_keys(v, effective_d, &mut rng);
        if v <= 100 {
            vocab_keys = orthogonalize(&vocab_keys, 1, 0.4);
        }
        let vocab_ms = t_vocab.elapsed().as_millis();

        // Sent key is the (V+1)th key in the PRNG stream
        let sent_keys = make_vocab_keys(1, effective_d, &mut rng);
        let sent_key = sent_keys[0].clone();
        // Build RoPE LUT for this dimension (32KB, eliminates per-element sin_cos)
        let rope_lut = RopeLut::new(effective_d);

        // Build per-bank memory vectors in parallel using Rayon (up to B threads)
        let memories_raw: Vec<ComplexVector> = (0..num_banks).into_par_iter().map(|bank| {
            let mut mem = ComplexVector::zeros(effective_d);
            let mut bank_count = 0usize;

            for (i, f) in hot_facts.iter().enumerate() {
                if i % num_banks == bank {
                    let w_key = &vocab_keys[idx_w[&f.value]];
                    // RoPE LUT: bind sent_key first, then apply role rotation via LUT
                    let bound_sent = bind(&sent_key, w_key);
                    let binding = bind_role_lut(&bound_sent, i, &rope_lut);
                    for d in 0..effective_d {
                        mem.re[d] += binding.re[d];
                        mem.im[d] += binding.im[d];
                    }
                    bank_count += 1;
                }
            }

            // Scale by 1/sqrt(items_in_bank)
            if bank_count > 0 {
                let scale = 1.0 / (bank_count as f64).sqrt();
                for d in 0..effective_d {
                    mem.re[d] *= scale;
                    mem.im[d] *= scale;
                }
            }
            mem
        }).collect();

        // Apply TurboQuant MSE
        let t_mem = std::time::Instant::now();
        let memories: Vec<Vec<u8>> = memories_raw
            .par_iter()
            .map(|m| super::turboquant::quantize_mse_4bit(m))
            .collect();
        let mem_ms = t_mem.elapsed().as_millis();

        let t_qjl = std::time::Instant::now();
        let cleanup = QjlCleanupNetwork::new(&vocab_keys, 100);
        let qjl_ms = t_qjl.elapsed().as_millis();

        let byte_size = estimate_hrr_bytes(effective_d, v, num_facts, num_banks);
        // vocab_keys dropped here — no longer stored in HrrData

        // ── Build KEY bank (parallel to value bank) ─────────────────
        let t_key = std::time::Instant::now();
        let mut key_seen = HashSet::new();
        let mut key_words_vec = Vec::new();
        for f in &self.facts {
            if key_seen.insert(f.key.clone()) { key_words_vec.push(f.key.clone()); }
        }
        let kv = key_words_vec.len();
        let mut key_idx_map: HashMap<String, usize> = HashMap::with_capacity(kv);
        for (i, k) in key_words_vec.iter().enumerate() { key_idx_map.insert(k.clone(), i); }

        let key_seed = seed.wrapping_add(0xCAFE_1337);
        let mut key_rng = Mulberry32::new(key_seed);
        let mut key_vocab_keys = make_vocab_keys(kv, effective_d, &mut key_rng);
        if kv <= 100 {
            key_vocab_keys = orthogonalize(&key_vocab_keys, 1, 0.4);
        }
        let key_sent_keys = make_vocab_keys(1, effective_d, &mut key_rng);
        let key_sent_key = key_sent_keys[0].clone();

        let key_memories_raw: Vec<ComplexVector> = (0..num_banks).into_par_iter().map(|bank| {
            let mut mem = ComplexVector::zeros(effective_d);
            let mut bank_count = 0usize;
            for (i, f) in hot_facts.iter().enumerate() {
                if i % num_banks == bank {
                    let k_idx = key_idx_map[&f.key];
                    let k_key = &key_vocab_keys[k_idx];
                    let bound_sent = bind(&key_sent_key, k_key);
                    let binding = bind_role_lut(&bound_sent, i, &rope_lut);
                    for d in 0..effective_d {
                        mem.re[d] += binding.re[d];
                        mem.im[d] += binding.im[d];
                    }
                    bank_count += 1;
                }
            }
            if bank_count > 0 {
                let scale = 1.0 / (bank_count as f64).sqrt();
                for d in 0..effective_d { mem.re[d] *= scale; mem.im[d] *= scale; }
            }
            mem
        }).collect();

        let key_memories: Vec<Vec<u8>> = key_memories_raw
            .par_iter()
            .map(|m| super::turboquant::quantize_mse_4bit(m))
            .collect();

        let key_cleanup = QjlCleanupNetwork::new(&key_vocab_keys, 100);
        let key_ms = t_key.elapsed().as_millis();
        
        eprintln!("  [HRR detailed] {} vocab={}ms mem={}ms qjl={}ms keybank={}ms", 
                   self.name, vocab_ms, mem_ms, qjl_ms, key_ms);

        HrrData {
            memories, seed, vocab_size: v, cached_sent_key: sent_key, rope_lut, cleanup,
            vocab_words, vocab_idx: idx_w, effective_d, num_banks, byte_size,
            key_memories, key_cleanup, key_words: key_words_vec, key_idx: key_idx_map,
            key_sent_key,
        }
    }

    /// Decode a fact position from HRR memory.
    /// Selects the correct bank (round-robin), unbinds, then uses
    /// QJL 2-stage cleanup (sign-bit Hamming + top-K re-rank) for accurate recall.
    fn decode_hrr(hrr: &HrrData, pos: usize) -> (String, f64, f64) {
        let v = hrr.vocab_words.len();
        if v == 0 { return (String::new(), 0.0, 0.0); }

        // Select bank containing this fact
        let bank = pos % hrr.num_banks;
        let bank_mem = super::turboquant::dequantize_mse_4bit(&hrr.memories[bank]);

        // Unbind sentence key, then role key (direct sin_cos — faster than LUT in debug)
        let d = hrr.effective_d;
        let sent_key = &hrr.cached_sent_key;
        let mut tmp = ComplexVector::zeros(d);
        super::core::unbind_into(&bank_mem, sent_key, &mut tmp);
        let recovered = unbind_role_inline(&tmp, pos);

        // QJL 2-stage cleanup: sign-bit Hamming coarse filter → top-K f64 re-rank
        let (best_idx, cleanup_sim, _) = hrr.cleanup.cleanup(&recovered);

        (hrr.vocab_words[best_idx].clone(), cleanup_sim, 1.0)
    }

    /// Resolve a query key via HRR key bank.
    /// Tries every position in the hot set; returns the best-matching key
    /// (the one with highest QJL cleanup similarity).
    fn resolve_key_hrr(hrr: &HrrData, hot_indices: &[usize]) -> Option<(String, f64)> {
        let kv = hrr.key_words.len();
        if kv == 0 { return None; }

        let d = hrr.effective_d;
        let mut best_key: Option<String> = None;
        let mut best_sim: f64 = 0.0;

        for (hot_idx, &_fact_pos) in hot_indices.iter().enumerate() {
            let bank = hot_idx % hrr.num_banks;
            let bank_mem = super::turboquant::dequantize_mse_4bit(&hrr.key_memories[bank]);

            let sent_key = &hrr.key_sent_key;
            let mut tmp = ComplexVector::zeros(d);
            super::core::unbind_into(&bank_mem, sent_key, &mut tmp);
            let recovered = unbind_role_inline(&tmp, hot_idx);

            let (best_idx, sim, _) = hrr.key_cleanup.cleanup(&recovered);
            if sim > best_sim {
                best_sim = sim;
                best_key = Some(hrr.key_words[best_idx].clone());
            }
        }

        best_key.map(|k| (k, best_sim))
    }

    // -- HRR binary serialization --------------------------------------------
    // Format: "HRR2" | D:u32 | V:u32 | N:u32 | B:u32 | memories[B×2D] | vocab_json

    fn save_hrr(&self, hrr: &HrrData) -> Result<PathBuf, std::io::Error> {
        let path = self.hrr_path();
        fs::create_dir_all(&self.save_dir)?;
        let d = hrr.effective_d;
        let v = hrr.vocab_words.len();
        let n = self.hot_indices.len();
        let b = hrr.num_banks;

        let mut buf = Vec::with_capacity(20 + b * d + 512);
        buf.extend_from_slice(b"HRR3"); // Upgraded magic for TurboQuant
        buf.extend_from_slice(&(d as u32).to_le_bytes());
        buf.extend_from_slice(&(v as u32).to_le_bytes());
        buf.extend_from_slice(&(n as u32).to_le_bytes());
        buf.extend_from_slice(&(b as u32).to_le_bytes());
        for mem in &hrr.memories {
            buf.extend_from_slice(mem);
        }

        let vocab_json = simd_json::to_string(&hrr.vocab_words)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        buf.extend_from_slice(vocab_json.as_bytes());

        let tmp = path.with_extension("bin.tmp");
        fs::write(&tmp, &buf)?;
        fs::rename(&tmp, &path)?;
        Ok(path)
    }

    fn load_hrr(&self) -> Result<HrrData, std::io::Error> {
        let buf = fs::read(self.hrr_path())?;
        if buf.len() < 20 || &buf[0..4] != b"HRR3" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad HRR magic"));
        }

        let d = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
        let v = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
        let n = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;
        let num_banks = u32::from_le_bytes(buf[16..20].try_into().unwrap()) as usize;

        let mem_data_size = num_banks * d; // 4-bit packed
        let expected = 20 + mem_data_size;
        if buf.len() < expected {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "truncated"));
        }

        let mut memories = Vec::with_capacity(num_banks);
        for b in 0..num_banks {
            let bank_off = 20 + b * d;
            memories.push(buf[bank_off..bank_off + d].to_vec());
        }

        let mut vocab_json_vec = buf[expected..].to_vec();
        let vocab_words: Vec<String> = simd_json::from_slice(&mut vocab_json_vec)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        if vocab_words.len() != v {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "vocab mismatch"));
        }

        let seed = seed_from_name(&self.name);
        let mut rng = Mulberry32::new(seed);
        let mut vocab_keys = make_vocab_keys(v, d, &mut rng);
        if v <= 100 { vocab_keys = orthogonalize(&vocab_keys, 1, 0.4); }
        // Cache sent_key on load (32KB); role_keys computed via RoPE LUT
        let sent_keys = make_vocab_keys(1, d, &mut rng);
        let cached_sent_key = sent_keys[0].clone();
        let rope_lut = RopeLut::new(d);
        let cleanup = QjlCleanupNetwork::new(&vocab_keys, 5);
        let byte_size = estimate_hrr_bytes(d, v, n, num_banks);

        let mut vocab_idx: HashMap<String, usize> = HashMap::with_capacity(v);
        for (i, w) in vocab_words.iter().enumerate() { vocab_idx.insert(w.clone(), i); }

        // Rebuild key bank (not serialized — always fresh from current facts)
        let mut key_seen = HashSet::new();
        let mut key_words_vec = Vec::new();
        for f in &self.facts {
            if key_seen.insert(f.key.clone()) { key_words_vec.push(f.key.clone()); }
        }
        let kv = key_words_vec.len();
        let mut key_idx_map: HashMap<String, usize> = HashMap::with_capacity(kv);
        for (i, k) in key_words_vec.iter().enumerate() { key_idx_map.insert(k.clone(), i); }

        let key_seed = seed.wrapping_add(0xCAFE_1337);
        let mut key_rng = Mulberry32::new(key_seed);
        let mut key_vocab_keys = make_vocab_keys(kv, d, &mut key_rng);
        if kv <= 100 { key_vocab_keys = orthogonalize(&key_vocab_keys, 1, 0.4); }
        let key_sent_keys = make_vocab_keys(1, d, &mut key_rng);
        let key_sent_key = key_sent_keys[0].clone();
        let key_cleanup = QjlCleanupNetwork::new(&key_vocab_keys, 100);

        // Build key memories from hot facts
        let hot_facts: Vec<&Fact> = self.hot_indices.iter()
            .filter_map(|&i| self.facts.get(i)).collect();
        let key_memories_raw: Vec<ComplexVector> = (0..num_banks).map(|bank| {
            let mut mem = ComplexVector::zeros(d);
            let mut bank_count = 0usize;
            for (i, f) in hot_facts.iter().enumerate() {
                if i % num_banks == bank {
                    if let Some(&k_idx) = key_idx_map.get(&f.key) {
                        let k_key = &key_vocab_keys[k_idx];
                        let bound_sent = bind(&key_sent_key, k_key);
                        let binding = bind_role_lut(&bound_sent, i, &rope_lut);
                        for dd in 0..d {
                            mem.re[dd] += binding.re[dd];
                            mem.im[dd] += binding.im[dd];
                        }
                        bank_count += 1;
                    }
                }
            }
            if bank_count > 0 {
                let scale = 1.0 / (bank_count as f64).sqrt();
                for dd in 0..d { mem.re[dd] *= scale; mem.im[dd] *= scale; }
            }
            mem
        }).collect();
        let key_memories: Vec<Vec<u8>> = key_memories_raw.iter()
            .map(|m| super::turboquant::quantize_mse_4bit(m)).collect();

        Ok(HrrData {
            memories, seed, vocab_size: v, cached_sent_key, rope_lut, cleanup,
            vocab_words, vocab_idx, effective_d: d, num_banks, byte_size,
            key_memories, key_cleanup, key_words: key_words_vec, key_idx: key_idx_map,
            key_sent_key,
        })
    }

    // -- index & fuzzy match -------------------------------------------------

    fn refresh_index(&mut self) {
        self.tag_to_pos.clear();
        for (i, f) in self.facts.iter().enumerate() {
            self.tag_to_pos.insert(f.key.clone(), i);
        }
        // Rebuild the inverted keyword index + link graph
        let pairs: Vec<(String, String)> = self.facts.iter()
            .map(|f| (f.key.clone(), f.value.clone()))
            .collect();
        self.key_index.rebuild(&pairs);
    }

    /// Resolve a query to a stored fact key. Returns (key, match_quality)
    /// where match_quality ∈ [0, 1] indicates how well the query matched.
    /// 1.0 = exact match, 0.99 = case-insensitive, <1.0 = fuzzy.
    fn resolve_tag(&self, query: &str) -> Option<(String, f64)> {
        if self.tag_to_pos.is_empty() { return None; }

        // 0. Exact match — O(1) HashMap lookup on original key
        if self.tag_to_pos.contains_key(query) {
            return Some((query.to_string(), 1.0));
        }

        // 0b. Case-insensitive exact match — O(N) but fast for small N
        let lower_query = query.to_lowercase();
        let lower_query = lower_query.trim();
        for tag in self.tag_to_pos.keys() {
            if tag.to_lowercase() == lower_query {
                return Some((tag.clone(), 0.99));
            }
        }

        // 1. Inverted keyword index (O(K) token overlap)
        let fact_keys: Vec<String> = self.facts.iter().map(|f| f.key.clone()).collect();
        if let Some(key) = self.key_index.resolve(query, &fact_keys) {
            // Verify the key actually exists in tag_to_pos (sanity check)
            if self.tag_to_pos.contains_key(&key) {
                // Compute match quality from sequence ratio
                let qual = sequence_match_ratio(lower_query, &key.to_lowercase());
                return Some((key, qual.max(0.5)));
            }
        }

        // 2. Fuzzy string matching — rank by overlap quality, not first-match
        let tags: Vec<&String> = self.tag_to_pos.keys().collect();

        // Rank all candidates by sequence match ratio
        let mut best: Option<&String> = None;
        let mut best_score = 0.0;
        for t in &tags {
            let s = sequence_match_ratio(lower_query, &t.to_lowercase());
            if s > best_score { best = Some(t); best_score = s; }
        }
        // Require high confidence: 0.80 minimum
        if best_score >= 0.80 { return best.map(|t| ((*t).clone(), best_score)); }

        None
    }

    /// Get related facts for a query via the link graph.
    pub fn related(&self, query: &str, max_results: usize) -> Vec<(String, String)> {
        let fact_keys: Vec<String> = self.facts.iter().map(|f| f.key.clone()).collect();
        let indices = self.key_index.related(query, &fact_keys, max_results);
        indices.iter()
            .filter_map(|&i| self.facts.get(i).map(|f| (f.key.clone(), f.value.clone())))
            .collect()
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
