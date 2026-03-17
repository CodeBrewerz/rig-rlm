//! Nugget — a single holographic memory unit.
//!
//! Stores key-value facts as superposed complex-valued vectors and retrieves
//! them via algebraic unbinding. Deterministic rebuild from facts using
//! seeded RNG so vectors are never serialised — only the facts are persisted
//! as JSON.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::advanced::CleanupNetwork;
use super::core::{
    bind, corvacs_lite, dot_product, make_role_keys, make_vocab_keys, orthogonalize,
    seed_from_name, sharpen, softmax_temp, stack_and_unit_norm, unbind, ComplexVector, Mulberry32,
};

// ---------------------------------------------------------------------------
// Default save directory: ~/.nuggets/
// ---------------------------------------------------------------------------

/// Default save directory for nugget files.
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

#[derive(Debug, Clone)]
struct BankData {
    memory: ComplexVector,
    vocab_norm: Vec<Vec<f64>>,
    sent_keys: Vec<ComplexVector>,
    role_keys: Vec<ComplexVector>,
    /// Cleanup network for exact nearest-neighbor recall.
    cleanup: CleanupNetwork,
}

#[derive(Debug, Clone)]
struct EnsembleData {
    banks: Vec<BankData>,
}

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
        Self {
            sharpen_p: 1.0,
            corvacs_a: 0.0,
            temp_t: 0.9,
            orth_iters: 1,
        }
    }
}

fn default_sharpen_p() -> f64 {
    1.0
}
fn default_temp_t() -> f64 {
    0.9
}
fn default_orth_iters() -> usize {
    1
}

// ---------------------------------------------------------------------------
// Options & Result types
// ---------------------------------------------------------------------------

/// Configuration for creating a new Nugget.
pub struct NuggetOpts {
    pub name: String,
    /// Vector dimension (default 2048). Higher = more capacity, less interference.
    pub d: usize,
    /// Number of shards (default 4). Facts are distributed round-robin.
    pub banks: usize,
    /// Number of redundant ensemble copies (default 1).
    pub ensembles: usize,
    /// Whether to auto-save on every mutation.
    pub auto_save: bool,
    /// Directory for persistence (default ~/.nuggets/).
    pub save_dir: PathBuf,
    /// Maximum facts before oldest are evicted (0 = unlimited).
    pub max_facts: usize,
}

impl Default for NuggetOpts {
    fn default() -> Self {
        Self {
            name: "default".into(),
            d: 2048,
            banks: 4,
            ensembles: 1,
            auto_save: true,
            save_dir: default_save_dir(),
            max_facts: 0,
        }
    }
}

/// Result from a recall operation.
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
        Self {
            answer: None,
            confidence: 0.0,
            margin: 0.0,
            found: false,
            key: String::new(),
        }
    }
}

/// Status info for a nugget.
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

/// A single holographic memory unit.
///
/// Stores key→value facts as superposed complex-valued vectors (HRR).
/// Recall is algebraic and sub-millisecond. Facts persist as JSON at
/// `~/.nuggets/<name>.nugget.json`.
pub struct Nugget {
    pub name: String,
    d: usize,
    banks: usize,
    ensembles: usize,
    pub auto_save: bool,
    pub save_dir: PathBuf,
    pub max_facts: usize,

    // Hyperparameters
    sharpen_p: f64,
    corvacs_a: f64,
    temp_t: f64,
    orth_iters: usize,
    orth_step: f64,
    fuzzy_threshold: f64,

    // State
    facts: Vec<Fact>,
    ensemble_data: Option<Vec<EnsembleData>>,
    vocab_words: Vec<String>,
    tag_to_pos: HashMap<String, usize>,
    dirty: bool,
}

impl Nugget {
    pub fn new(opts: NuggetOpts) -> Self {
        Self {
            name: opts.name,
            d: opts.d,
            banks: opts.banks,
            ensembles: opts.ensembles,
            auto_save: opts.auto_save,
            save_dir: opts.save_dir,
            max_facts: opts.max_facts,

            sharpen_p: 1.0,
            corvacs_a: 0.0,
            temp_t: 0.9,
            orth_iters: 1,
            orth_step: 0.4,
            fuzzy_threshold: 0.55,

            facts: Vec::new(),
            ensemble_data: None,
            vocab_words: Vec::new(),
            tag_to_pos: HashMap::new(),
            dirty: false,
        }
    }

    // -- public API ----------------------------------------------------------

    /// Store a key→value fact. If key already exists (case-insensitive), update it.
    pub fn remember(&mut self, key: &str, value: &str) {
        let key = key.trim();
        let value = value.trim();
        if key.is_empty() || value.is_empty() {
            return;
        }

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
                key: key.to_string(),
                value: value.to_string(),
                hits: 0,
                last_hit_session: String::new(),
            });
        }

        // Evict oldest if max_facts exceeded
        if self.max_facts > 0 && self.facts.len() > self.max_facts {
            let drain_count = self.facts.len() - self.max_facts;
            self.facts.drain(..drain_count);
        }

        self.dirty = true;
        if self.auto_save {
            let _ = self.save(None);
        }
    }

    /// Recall a fact by fuzzy-matching query against stored keys.
    pub fn recall(&mut self, query: &str, session_id: &str) -> RecallResult {
        if self.facts.is_empty() {
            return RecallResult::empty();
        }

        if self.dirty || self.ensemble_data.is_none() {
            self.rebuild();
            self.dirty = false;
        }

        let tag = match self.resolve_tag(query) {
            Some(t) => t,
            None => return RecallResult::empty(),
        };

        if !self.tag_to_pos.contains_key(&tag) {
            return RecallResult::empty();
        }

        let (word, _sims, probs) = self.decode(&tag);

        // Top-2 for confidence/margin
        let mut top1 = f64::NEG_INFINITY;
        let mut top2 = f64::NEG_INFINITY;
        for &p in &probs {
            if p > top1 {
                top2 = top1;
                top1 = p;
            } else if p > top2 {
                top2 = p;
            }
        }
        let confidence = top1;
        let margin = if top2 == f64::NEG_INFINITY {
            top1
        } else {
            top1 - top2
        };

        // Hit tracking (per-session dedup)
        if !session_id.is_empty() {
            if let Some(&pos) = self.tag_to_pos.get(&tag) {
                if let Some(fact) = self.facts.get_mut(pos) {
                    if fact.last_hit_session != session_id {
                        fact.hits += 1;
                        fact.last_hit_session = session_id.to_string();
                        if self.auto_save {
                            let _ = self.save(None);
                        }
                    }
                }
            }
        }

        RecallResult {
            answer: Some(word),
            confidence,
            margin,
            found: true,
            key: tag,
        }
    }

    /// Remove a fact by key (case-insensitive).
    pub fn forget(&mut self, key: &str) -> bool {
        let lower = key.to_lowercase();
        let lower = lower.trim();
        let before = self.facts.len();
        self.facts.retain(|f| f.key.to_lowercase() != lower);
        let removed = self.facts.len() < before;
        if removed {
            self.dirty = true;
            if self.auto_save {
                let _ = self.save(None);
            }
        }
        removed
    }

    /// Return all stored facts.
    pub fn facts(&self) -> Vec<Fact> {
        self.facts
            .iter()
            .map(|f| Fact {
                key: f.key.clone(),
                value: f.value.clone(),
                hits: f.hits,
                last_hit_session: f.last_hit_session.clone(),
            })
            .collect()
    }

    /// Clear all facts and reset memory.
    pub fn clear(&mut self) {
        self.facts.clear();
        self.ensemble_data = None;
        self.vocab_words.clear();
        self.tag_to_pos.clear();
        self.dirty = false;
        if self.auto_save {
            let _ = self.save(None);
        }
    }

    /// Get status summary.
    pub fn status(&self) -> NuggetStatus {
        // Compute effective banks and dimension (same logic as rebuild)
        let max_per_bank = 20usize;
        let n = self.facts.len();
        let v = {
            let mut seen = std::collections::HashSet::new();
            self.facts.iter().filter(|f| seen.insert(&f.value)).count()
        };
        let effective_banks = self.banks.max(if n == 0 {
            self.banks
        } else {
            (n + max_per_bank - 1) / max_per_bank
        });
        let effective_d = if v == 0 {
            self.d
        } else {
            self.d.max(v * 16).min(8192).next_power_of_two()
        };
        let capacity_est = effective_banks * (effective_d as f64).sqrt() as usize;
        let used_pct = if capacity_est > 0 {
            (self.facts.len() as f64 / capacity_est as f64) * 100.0
        } else {
            0.0
        };
        let capacity_warning = if used_pct > 90.0 {
            "CRITICAL: nearly full".to_string()
        } else if used_pct > 80.0 {
            "WARNING: approaching capacity".to_string()
        } else {
            String::new()
        };

        NuggetStatus {
            name: self.name.clone(),
            fact_count: self.facts.len(),
            dimension: self.d,
            banks: effective_banks,
            ensembles: self.ensembles,
            capacity_used_pct: (used_pct * 10.0).round() / 10.0,
            capacity_warning,
            max_facts: self.max_facts,
        }
    }

    // -- persistence ---------------------------------------------------------

    /// Save to disk. Returns the path written.
    pub fn save(&self, path: Option<&Path>) -> Result<PathBuf, std::io::Error> {
        let path = match path {
            Some(p) => p.to_path_buf(),
            None => {
                fs::create_dir_all(&self.save_dir)?;
                self.save_dir.join(format!("{}.nugget.json", self.name))
            }
        };

        let data = NuggetFile {
            version: 3,
            name: self.name.clone(),
            d: self.d,
            banks: self.banks,
            ensembles: self.ensembles,
            max_facts: self.max_facts,
            facts: self.facts.clone(),
            config: NuggetConfig {
                sharpen_p: self.sharpen_p,
                corvacs_a: self.corvacs_a,
                temp_t: self.temp_t,
                orth_iters: self.orth_iters,
            },
        };

        let json = serde_json::to_string(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let tmp_path = path.with_extension("json.tmp");
        fs::write(&tmp_path, &json)?;
        fs::rename(&tmp_path, &path)?;
        Ok(path)
    }

    /// Load from a JSON file.
    pub fn load(path: &Path, auto_save: bool) -> Result<Self, std::io::Error> {
        let raw = fs::read_to_string(path)?;
        let data: NuggetFile = serde_json::from_str(&raw)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let save_dir = path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(default_save_dir);

        let mut n = Self {
            name: data.name,
            d: data.d,
            banks: data.banks,
            ensembles: data.ensembles,
            auto_save,
            save_dir,
            max_facts: data.max_facts,

            sharpen_p: data.config.sharpen_p,
            corvacs_a: data.config.corvacs_a,
            temp_t: data.config.temp_t,
            orth_iters: data.config.orth_iters,
            orth_step: 0.4,
            fuzzy_threshold: 0.55,

            facts: data.facts,
            ensemble_data: None,
            vocab_words: Vec::new(),
            tag_to_pos: HashMap::new(),
            dirty: false,
        };

        if !n.facts.is_empty() {
            n.rebuild();
        }
        Ok(n)
    }

    // -- internals -----------------------------------------------------------

    /// Rebuild the entire HRR memory from facts. Called on mutation or first recall.
    fn rebuild(&mut self) {
        use rayon::prelude::*;

        if self.facts.is_empty() {
            self.ensemble_data = None;
            self.vocab_words.clear();
            self.tag_to_pos.clear();
            return;
        }

        // Build vocabulary from unique values
        let mut seen = std::collections::HashSet::new();
        let mut vocab: Vec<String> = Vec::new();
        for f in &self.facts {
            if seen.insert(f.value.clone()) {
                vocab.push(f.value.clone());
            }
        }
        self.vocab_words = vocab.clone();

        // Tag → position mapping
        self.tag_to_pos.clear();
        for (i, f) in self.facts.iter().enumerate() {
            self.tag_to_pos.insert(f.key.clone(), i);
        }
        let num_facts = self.facts.len();

        // Deterministic seed from name
        let seed = seed_from_name(&self.name);
        let mut rng = Mulberry32::new(seed);

        let v = vocab.len();
        let mut idx_w: HashMap<String, usize> = HashMap::with_capacity(v);
        for (i, word) in vocab.iter().enumerate() {
            idx_w.insert(word.clone(), i);
        }

        // Auto-scale dimension: D must be large enough for vocab separation.
        // Random unit vectors in D dimensions have expected cosine similarity ~0,
        // but noise scales as ~1/√D. For V codebook entries to be distinguishable,
        // we need D ≥ V * 4 (conservative, gives ~2x safety margin).
        let effective_d = self.d.max(v * 16).min(8192).next_power_of_two();

        // Auto-scale banks: keep each bank under ~30 facts (within √D capacity)
        // Rule: effective_banks = max(configured_banks, ceil(facts / 30))
        let max_per_bank = 20usize;
        let effective_banks = self
            .banks
            .max((num_facts + max_per_bank - 1) / max_per_bank);

        // Round-robin bank assignment
        let mut items_by_bank: Vec<Vec<(usize, usize, String)>> =
            (0..effective_banks).map(|_| Vec::new()).collect();
        for (i, f) in self.facts.iter().enumerate() {
            items_by_bank[i % effective_banks].push((0, i, f.value.clone()));
        }

        let mut ensembles = Vec::with_capacity(self.ensembles);
        for _ in 0..self.ensembles {
            let mut vocab_keys = make_vocab_keys(v, effective_d, &mut rng);
            // Skip Gram-Schmidt for large vocabs — it's O(V²×2D) and at V>100
            // the high dimensionality + cleanup network provide sufficient separation.
            if self.orth_iters > 0 && v <= 100 {
                vocab_keys = orthogonalize(&vocab_keys, self.orth_iters, self.orth_step);
            }
            let vocab_norm = stack_and_unit_norm(&vocab_keys);

            let sent_keys = make_vocab_keys(1, effective_d, &mut rng);
            let role_keys = make_role_keys(effective_d, num_facts);

            let banks: Vec<BankData> = (0..effective_banks)
                .into_par_iter()
                .map(|b| {
                    let items = &items_by_bank[b];
                    let mut bindings: Vec<ComplexVector> = Vec::with_capacity(items.len());

                    for (sid, pos, word) in items {
                        let s_key = &sent_keys[*sid];
                        let r_key = &role_keys[*pos];
                        let w_key = &vocab_keys[idx_w[word]];
                        bindings.push(bind(&bind(s_key, r_key), w_key));
                    }

                    let memory = if bindings.is_empty() {
                        ComplexVector::zeros(effective_d)
                    } else {
                        // Sum all bindings
                        let mut re = vec![0.0; effective_d];
                        let mut im = vec![0.0; effective_d];
                        for binding in &bindings {
                            for d in 0..effective_d {
                                re[d] += binding.re[d];
                                im[d] += binding.im[d];
                            }
                        }
                        // Scale by 1/sqrt(n)
                        let scale = 1.0 / (bindings.len() as f64).sqrt();
                        for d in 0..effective_d {
                            re[d] *= scale;
                            im[d] *= scale;
                        }
                        ComplexVector { re, im }
                    };

                    BankData {
                        memory,
                        vocab_norm: vocab_norm.clone(),
                        sent_keys: sent_keys.clone(),
                        role_keys: role_keys.clone(),
                        cleanup: CleanupNetwork::new(&vocab_keys),
                    }
                })
                .collect();
            ensembles.push(EnsembleData { banks });
        }
        self.ensemble_data = Some(ensembles);
    }

    /// Decode a tag position from holographic memory.
    ///
    /// Uses the Cleanup Network (codebook nearest-neighbor) for exact recall,
    /// with cosine similarity accumulation across all banks/ensembles as tiebreaker.
    fn decode(&self, tag: &str) -> (String, Vec<f64>, Vec<f64>) {
        use rayon::prelude::*;

        let pos = self.tag_to_pos[tag];
        let sid = 0;
        let v = self.vocab_words.len();

        // Collect all banks from all ensembles into a flat list for parallel processing
        let all_banks: Vec<&BankData> = self
            .ensemble_data
            .as_ref()
            .map(|ensembles| ensembles.iter().flat_map(|ens| ens.banks.iter()).collect())
            .unwrap_or_default();

        if all_banks.is_empty() {
            let sims = vec![0.0; v];
            let probs = softmax_temp(&sims, self.temp_t);
            return (self.vocab_words[0].clone(), sims, probs);
        }

        // Each bank produces (cleanup_idx, cleanup_sim, sims_for_vocab)
        let sharpen_p = self.sharpen_p;
        let corvacs_a = self.corvacs_a;

        let bank_results: Vec<(usize, f64, Vec<f64>)> = all_banks
            .par_iter()
            .map(|bank| {
                // Unbind sentence, then role
                let rec = unbind(
                    &unbind(&bank.memory, &bank.sent_keys[sid]),
                    &bank.role_keys[pos],
                );
                let rec = corvacs_lite(&sharpen(&rec, sharpen_p), corvacs_a);

                // Cleanup Network: exact nearest-neighbor in codebook
                let (cleanup_idx, cleanup_sim, _) = bank.cleanup.cleanup(&rec);

                // Cosine sims for softmax fallback
                let d = rec.dim();
                let d2 = d * 2;
                let mut rec2 = vec![0.0; d2];
                rec2[..d].copy_from_slice(&rec.re);
                rec2[d..d2].copy_from_slice(&rec.im);
                let norm = dot_product(&rec2, &rec2).sqrt() + 1e-12;
                let inv_norm = 1.0 / norm;
                for x in rec2.iter_mut() {
                    *x *= inv_norm;
                }
                let sims: Vec<f64> = bank
                    .vocab_norm
                    .iter()
                    .map(|row| dot_product(row, &rec2))
                    .collect();

                (cleanup_idx, cleanup_sim, sims)
            })
            .collect();

        // Reduce results
        let mut sims_sum = vec![0.0; v];
        let mut cleanup_votes = vec![0.0_f64; v];
        for (cleanup_idx, cleanup_sim, sims) in &bank_results {
            cleanup_votes[*cleanup_idx] += cleanup_sim;
            for (vi, s) in sims.iter().enumerate() {
                sims_sum[vi] += s;
            }
        }

        // Use cleanup votes as primary signal (exact recall)
        let cleanup_best = cleanup_votes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Softmax as secondary signal
        let probs = softmax_temp(&sims_sum, self.temp_t);
        let softmax_best = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        // If cleanup and softmax agree, or cleanup has strong signal, use cleanup
        let best_idx = if cleanup_best == softmax_best || cleanup_votes[cleanup_best] > 0.3 {
            cleanup_best
        } else {
            softmax_best
        };

        // Build confidence from cleanup votes (much higher than softmax)
        let total_votes: f64 = cleanup_votes.iter().sum();
        let cleanup_probs: Vec<f64> = if total_votes > 0.0 {
            cleanup_votes.iter().map(|v| v / total_votes).collect()
        } else {
            probs.clone()
        };

        (self.vocab_words[best_idx].clone(), sims_sum, cleanup_probs)
    }

    /// Fuzzy-match query to stored keys (threshold >= 0.55).
    fn resolve_tag(&self, query: &str) -> Option<String> {
        if self.tag_to_pos.is_empty() {
            return None;
        }
        let text = query.to_lowercase();
        let text = text.trim();
        let tags: Vec<&String> = self.tag_to_pos.keys().collect();

        // Exact match
        for t in &tags {
            if t.to_lowercase() == text {
                return Some((*t).clone());
            }
        }

        // Substring match
        for t in &tags {
            let lower = t.to_lowercase();
            if lower.contains(text) || text.contains(lower.as_str()) {
                return Some((*t).clone());
            }
        }

        // Fuzzy match (SequenceMatcher equivalent)
        let mut best: Option<&String> = None;
        let mut best_score = 0.0;
        for t in &tags {
            let s = sequence_match_ratio(text, &t.to_lowercase());
            if s > best_score {
                best = Some(t);
                best_score = s;
            }
        }

        if best_score >= self.fuzzy_threshold {
            best.map(|t| (*t).clone())
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Fuzzy matching — port of Python's SequenceMatcher.ratio()
// ---------------------------------------------------------------------------

fn sequence_match_ratio(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let matches = count_matches(a.as_bytes(), b.as_bytes());
    (2.0 * matches as f64) / (a.len() + b.len()) as f64
}

/// Count matching characters using greedy longest common substring blocks.
/// Approximates Python's SequenceMatcher.
fn count_matches(a: &[u8], b: &[u8]) -> usize {
    let m = a.len();
    let n = b.len();
    let mut total = 0;
    let mut used_a = vec![false; m];
    let mut used_b = vec![false; n];

    loop {
        let mut best_len = 0;
        let mut best_i = 0;
        let mut best_j = 0;

        for i in 0..m {
            if used_a[i] {
                continue;
            }
            for j in 0..n {
                if used_b[j] {
                    continue;
                }
                let mut len = 0;
                while i + len < m
                    && j + len < n
                    && !used_a[i + len]
                    && !used_b[j + len]
                    && a[i + len] == b[j + len]
                {
                    len += 1;
                }
                if len > best_len {
                    best_len = len;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if best_len == 0 {
            break;
        }

        for k in 0..best_len {
            used_a[best_i + k] = true;
            used_b[best_j + k] = true;
        }
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
        NuggetOpts {
            name: name.into(),
            d: 512,
            banks: 2,
            auto_save: false,
            ..Default::default()
        }
    }

    #[test]
    fn remember_and_recall() {
        let mut n = Nugget::new(test_opts("test"));
        n.remember("color", "blue");
        let result = n.recall("color", "");
        assert!(result.found);
        assert_eq!(result.answer.unwrap(), "blue");
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn upsert_on_duplicate_key() {
        let mut n = Nugget::new(test_opts("test"));
        n.remember("color", "blue");
        n.remember("color", "red");
        assert_eq!(n.facts().len(), 1);
        let result = n.recall("color", "");
        assert_eq!(result.answer.unwrap(), "red");
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
        assert_eq!(s.dimension, 512);
        assert_eq!(s.banks, 2);
    }

    #[test]
    fn save_and_load() {
        let tmp = tempfile::tempdir().unwrap();
        let tmp_path = tmp.path().to_path_buf();

        let mut n = Nugget::new(NuggetOpts {
            name: "persist".into(),
            d: 512,
            banks: 2,
            auto_save: false,
            save_dir: tmp_path.clone(),
            ..Default::default()
        });
        n.remember("lang", "typescript");
        n.remember("color", "green");
        let path = n.save(None).unwrap();

        let mut loaded = Nugget::load(&path, false).unwrap();
        assert_eq!(loaded.name, "persist");
        assert_eq!(loaded.facts().len(), 2);

        let result = loaded.recall("lang", "");
        assert!(result.found);
        assert_eq!(result.answer.unwrap(), "typescript");
    }

    #[test]
    fn hit_tracking_per_session() {
        let mut n = Nugget::new(test_opts("hits"));
        n.remember("key", "value");

        n.recall("key", "session-1");
        n.recall("key", "session-1"); // duplicate — no increment
        assert_eq!(n.facts()[0].hits, 1);

        n.recall("key", "session-2");
        assert_eq!(n.facts()[0].hits, 2);
    }

    #[test]
    fn max_facts_eviction() {
        let mut n = Nugget::new(NuggetOpts {
            name: "limited".into(),
            d: 512,
            banks: 2,
            auto_save: false,
            max_facts: 3,
            ..Default::default()
        });
        n.remember("a", "1");
        n.remember("b", "2");
        n.remember("c", "3");
        n.remember("d", "4"); // should evict "a"
        assert_eq!(n.facts().len(), 3);
        let keys: Vec<String> = n.facts().iter().map(|f| f.key.clone()).collect();
        assert_eq!(keys, vec!["b", "c", "d"]);
    }

    #[test]
    fn multiple_facts_distinct_values() {
        let mut n = Nugget::new(NuggetOpts {
            name: "multi".into(),
            d: 1024,
            banks: 4,
            auto_save: false,
            ..Default::default()
        });
        n.remember("name", "Alice");
        n.remember("pet", "cat");
        n.remember("city", "London");

        let r1 = n.recall("name", "");
        assert!(r1.found);
        assert_eq!(r1.answer.unwrap(), "Alice");

        let r2 = n.recall("pet", "");
        assert!(r2.found);
        assert_eq!(r2.answer.unwrap(), "cat");

        let r3 = n.recall("city", "");
        assert!(r3.found);
        assert_eq!(r3.answer.unwrap(), "London");
    }

    #[test]
    fn fuzzy_key_match() {
        let mut n = Nugget::new(test_opts("fuzzy"));
        n.remember("favorite color", "blue");

        // Substring match
        let r = n.recall("color", "");
        assert!(r.found);
        assert_eq!(r.answer.unwrap(), "blue");
    }

    #[test]
    fn unknown_query_not_found() {
        let n = Nugget::new(test_opts("empty"));
        // Brand new, no facts
        let mut n = n;
        let result = n.recall("anything", "");
        assert!(!result.found);
        assert!(result.answer.is_none());
    }

    #[test]
    fn ignores_empty_key_value() {
        let mut n = Nugget::new(test_opts("test"));
        n.remember("", "value");
        n.remember("key", "");
        n.remember("  ", "value");
        assert_eq!(n.facts().len(), 0);
    }

    #[test]
    fn sequence_match_ratio_works() {
        assert!((sequence_match_ratio("abc", "abc") - 1.0).abs() < f64::EPSILON);
        assert!(sequence_match_ratio("abc", "xyz") < 0.5);
        assert!(sequence_match_ratio("", "") == 1.0);
        assert!(sequence_match_ratio("abc", "") == 0.0);
    }
}
