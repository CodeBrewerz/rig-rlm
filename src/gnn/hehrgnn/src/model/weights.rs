//! GNN model weight persistence.
//!
//! Save and load model weights to disk for all 4 GNN architectures.
//! Uses Burn's native `save_file` / `load_file` for Module-derived models
//! (GraphSAGE, RGCN, GAT, GPS Transformer).

use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

/// Metadata about a saved model checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightMeta {
    pub model_type: String,
    pub graph_hash: u64,
    pub epochs_trained: usize,
    pub final_loss: f32,
    pub final_auc: f32,
    pub hidden_dim: usize,
    pub timestamp: String,
}

/// Weight checkpoint directory.
pub fn weight_dir() -> PathBuf {
    PathBuf::from("/tmp/gnn_weights")
}

/// Path for a specific model's weights file.
pub fn weight_path(model_type: &str, graph_hash: u64) -> PathBuf {
    weight_dir().join(format!("{}_{}", model_type, graph_hash))
}

/// Path for a specific model's metadata file.
pub fn meta_path(model_type: &str, graph_hash: u64) -> PathBuf {
    weight_dir().join(format!("{}_{}_meta.json", model_type, graph_hash))
}

/// Hash a set of graph facts for cache key.
pub fn hash_graph_facts(facts_desc: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    facts_desc.hash(&mut hasher);
    hasher.finish()
}

/// Save model weights and metadata.
pub fn save_model<B: Backend, M: burn::module::Module<B>>(
    model: &M,
    model_type: &str,
    graph_hash: u64,
    meta: &WeightMeta,
    device: &B::Device,
) -> Result<(), String> {
    std::fs::create_dir_all(weight_dir()).map_err(|e| format!("mkdir: {}", e))?;

    let path = weight_path(model_type, graph_hash);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    model
        .clone()
        .save_file(path.to_str().unwrap(), &recorder)
        .map_err(|e| format!("save: {}", e))?;

    // Save metadata
    let meta_json = serde_json::to_string_pretty(meta).map_err(|e| format!("json: {}", e))?;
    std::fs::write(meta_path(model_type, graph_hash), meta_json)
        .map_err(|e| format!("meta write: {}", e))?;

    Ok(())
}

/// Load model weights if checkpoint exists.
pub fn load_model<B: Backend, M: burn::module::Module<B>>(
    model: M,
    model_type: &str,
    graph_hash: u64,
    device: &B::Device,
) -> Option<(M, WeightMeta)> {
    let path = weight_path(model_type, graph_hash);
    let meta_file = meta_path(model_type, graph_hash);

    // Check both files exist
    let bin_path = format!("{}.bin", path.to_str()?);
    if !Path::new(&bin_path).exists() || !meta_file.exists() {
        return None;
    }

    // Load metadata
    let meta_json = std::fs::read_to_string(&meta_file).ok()?;
    let meta: WeightMeta = serde_json::from_str(&meta_json).ok()?;

    // Load model weights
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    let loaded = model.load_file(path.to_str()?, &recorder, device).ok()?;

    Some((loaded, meta))
}

/// Check if a weight checkpoint exists for a model/hash combo.
pub fn has_checkpoint(model_type: &str, graph_hash: u64) -> bool {
    let path = weight_path(model_type, graph_hash);
    let bin_path = format!("{}.bin", path.to_str().unwrap_or(""));
    Path::new(&bin_path).exists() && meta_path(model_type, graph_hash).exists()
}

/// List all saved checkpoints.
pub fn list_checkpoints() -> Vec<WeightMeta> {
    let dir = weight_dir();
    if !dir.exists() {
        return vec![];
    }

    let mut metas = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "json") {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if let Ok(meta) = serde_json::from_str::<WeightMeta>(&content) {
                        metas.push(meta);
                    }
                }
            }
        }
    }
    metas
}

// ═══════════════════════════════════════════════════════════════
// LearnableScorer persistence
// ═══════════════════════════════════════════════════════════════

/// Save a LearnableScorer to disk (JSON).
pub fn save_scorer(
    scorer: &crate::eval::learnable_scorer::LearnableScorer,
    graph_hash: u64,
) -> Result<(), String> {
    std::fs::create_dir_all(weight_dir()).map_err(|e| format!("mkdir: {}", e))?;
    let path = weight_dir().join(format!("scorer_{}.json", graph_hash));
    let json = serde_json::to_string(scorer).map_err(|e| format!("json: {}", e))?;
    std::fs::write(&path, json).map_err(|e| format!("write: {}", e))?;
    Ok(())
}

/// Load a LearnableScorer from disk if checkpoint exists.
pub fn load_scorer(graph_hash: u64) -> Option<crate::eval::learnable_scorer::LearnableScorer> {
    let path = weight_dir().join(format!("scorer_{}.json", graph_hash));
    if !path.exists() {
        return None;
    }
    let json = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&json).ok()
}
