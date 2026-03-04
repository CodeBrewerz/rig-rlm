//! Sparse Autoencoder (SAE) for GNN embedding interpretability.
//!
//! Decomposes dense GNN embeddings into sparse, monosemantic features.
//! Each SAE feature learns to represent a single financial concept,
//! enabling transparent explanations for fiduciary recommendations.
//!
//! Architecture:
//!   Encoder: x → ReLU(W_enc @ (x - b_dec) + b_enc)   [hidden → expansion * hidden]
//!   Decoder: z → W_dec @ z + b_dec                     [expansion * hidden → hidden]
//!   Loss:    MSE(x, decode(encode(x))) + λ · L1(encode(x))
//!
//! Based on Anthropic's approach to mechanistic interpretability.

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════
// SAE Weights (plain Rust — works on Vec<f32> embeddings)
// ═══════════════════════════════════════════════════════════════

/// Trained Sparse Autoencoder weights.
#[derive(Debug, Clone)]
pub struct SparseAutoencoder {
    /// Encoder weight: [expansion_dim, hidden_dim]
    pub w_enc: Vec<Vec<f32>>,
    /// Encoder bias: [expansion_dim]
    pub b_enc: Vec<f32>,
    /// Decoder weight: [hidden_dim, expansion_dim]
    pub w_dec: Vec<Vec<f32>>,
    /// Decoder bias (also used for centering): [hidden_dim]
    pub b_dec: Vec<f32>,
    /// Input dimension.
    pub hidden_dim: usize,
    /// Expanded dimension (typically 4-8× hidden_dim).
    pub expansion_dim: usize,
    /// L1 penalty coefficient.
    pub l1_coeff: f32,
    /// Training reconstruction loss (final).
    pub final_mse: f32,
    /// Average sparsity (fraction of zeros in latent).
    pub avg_sparsity: f32,
}

/// SAE training configuration.
pub struct SaeConfig {
    /// Expansion factor (expansion_dim = hidden_dim * expansion_factor).
    pub expansion_factor: usize,
    /// L1 penalty coefficient for sparsity.
    pub l1_coeff: f32,
    /// Learning rate.
    pub lr: f32,
    /// Number of training epochs.
    pub epochs: usize,
}

impl Default for SaeConfig {
    fn default() -> Self {
        Self {
            expansion_factor: 8,
            l1_coeff: 0.01,
            lr: 0.001,
            epochs: 50,
        }
    }
}

impl SparseAutoencoder {
    /// Train an SAE on a collection of node embeddings.
    ///
    /// Takes all node embeddings (concatenated across types) and learns
    /// sparse features that reconstruct them.
    pub fn train(embeddings: &[Vec<f32>], config: &SaeConfig) -> Self {
        let n = embeddings.len();
        if n == 0 || embeddings[0].is_empty() {
            return Self::empty(0, config);
        }

        let hidden_dim = embeddings[0].len();
        let expansion_dim = hidden_dim * config.expansion_factor;

        // Initialize weights (Xavier/He initialization)
        let scale_enc = (2.0 / hidden_dim as f32).sqrt();
        let scale_dec = (2.0 / expansion_dim as f32).sqrt();

        let mut w_enc = vec![vec![0.0f32; hidden_dim]; expansion_dim];
        let mut b_enc = vec![0.0f32; expansion_dim];
        let mut w_dec = vec![vec![0.0f32; expansion_dim]; hidden_dim];
        let mut b_dec = vec![0.0f32; hidden_dim];

        // Deterministic pseudo-random init
        for i in 0..expansion_dim {
            for j in 0..hidden_dim {
                let seed = (i * 31 + j * 17 + 7) as f32;
                w_enc[i][j] = (seed * 0.1).sin() * scale_enc;
            }
        }
        for i in 0..hidden_dim {
            for j in 0..expansion_dim {
                let seed = (i * 23 + j * 13 + 11) as f32;
                w_dec[i][j] = (seed * 0.1).sin() * scale_dec;
            }
        }

        // Compute mean for b_dec initialization
        for emb in embeddings {
            for (j, &v) in emb.iter().enumerate() {
                b_dec[j] += v / n as f32;
            }
        }

        let mut final_mse = 0.0f32;
        let mut avg_sparsity = 0.0f32;

        // Training loop (SGD with mini-batch)
        let batch_size = 64.min(n);
        for epoch in 0..config.epochs {
            let mut epoch_mse = 0.0f32;
            let mut _epoch_l1 = 0.0f32;
            let mut epoch_sparsity = 0.0f32;

            // Process batches
            for batch_start in (0..n).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n);
                let batch = &embeddings[batch_start..batch_end];
                let bs = batch.len();

                // Forward pass for batch
                let mut latents = vec![vec![0.0f32; expansion_dim]; bs];
                let mut reconstructed = vec![vec![0.0f32; hidden_dim]; bs];

                for (bi, emb) in batch.iter().enumerate() {
                    // Encode: z = ReLU(W_enc @ (x - b_dec) + b_enc)
                    let centered: Vec<f32> =
                        emb.iter().zip(b_dec.iter()).map(|(x, b)| x - b).collect();

                    for i in 0..expansion_dim {
                        let mut val = b_enc[i];
                        for j in 0..hidden_dim {
                            val += w_enc[i][j] * centered[j];
                        }
                        latents[bi][i] = val.max(0.0); // ReLU
                    }

                    // Decode: x_hat = W_dec @ z + b_dec
                    for i in 0..hidden_dim {
                        let mut val = b_dec[i];
                        for j in 0..expansion_dim {
                            val += w_dec[i][j] * latents[bi][j];
                        }
                        reconstructed[bi][i] = val;
                    }
                }

                // Compute loss and gradients
                // dL/d_reconstructed = 2(reconstructed - x) / hidden_dim  (MSE grad)
                let mut grad_w_enc = vec![vec![0.0f32; hidden_dim]; expansion_dim];
                let mut grad_b_enc = vec![0.0f32; expansion_dim];
                let mut grad_w_dec = vec![vec![0.0f32; expansion_dim]; hidden_dim];
                let mut grad_b_dec = vec![0.0f32; hidden_dim];

                for (bi, emb) in batch.iter().enumerate() {
                    let centered: Vec<f32> =
                        emb.iter().zip(b_dec.iter()).map(|(x, b)| x - b).collect();

                    // MSE loss
                    let mut sample_mse = 0.0f32;
                    let mut grad_recon = vec![0.0f32; hidden_dim];
                    for i in 0..hidden_dim {
                        let diff = reconstructed[bi][i] - emb[i];
                        sample_mse += diff * diff;
                        grad_recon[i] = 2.0 * diff / hidden_dim as f32;
                    }
                    epoch_mse += sample_mse / hidden_dim as f32;

                    // L1 sparsity loss
                    let active = latents[bi].iter().filter(|v| **v > 0.0).count();
                    epoch_sparsity += 1.0 - (active as f32 / expansion_dim as f32);
                    for v in &latents[bi] {
                        _epoch_l1 += v.abs();
                    }

                    // Backward: decoder gradients
                    // grad_w_dec[i][j] = grad_recon[i] * latents[bi][j]
                    // grad_b_dec[i] = grad_recon[i]
                    let mut grad_latent = vec![0.0f32; expansion_dim];
                    for i in 0..hidden_dim {
                        grad_b_dec[i] += grad_recon[i];
                        for j in 0..expansion_dim {
                            grad_w_dec[i][j] += grad_recon[i] * latents[bi][j];
                            grad_latent[j] += grad_recon[i] * w_dec[i][j];
                        }
                    }

                    // Backward through ReLU + L1
                    for j in 0..expansion_dim {
                        if latents[bi][j] > 0.0 {
                            // ReLU gradient = 1
                            let l1_grad = config.l1_coeff * latents[bi][j].signum();
                            let total_grad = grad_latent[j] + l1_grad;
                            grad_b_enc[j] += total_grad;
                            for k in 0..hidden_dim {
                                grad_w_enc[j][k] += total_grad * centered[k];
                            }
                        }
                    }
                }

                // SGD update
                let lr = config.lr / bs as f32;
                for i in 0..expansion_dim {
                    b_enc[i] -= lr * grad_b_enc[i];
                    for j in 0..hidden_dim {
                        w_enc[i][j] -= lr * grad_w_enc[i][j];
                    }
                }
                for i in 0..hidden_dim {
                    b_dec[i] -= lr * grad_b_dec[i];
                    for j in 0..expansion_dim {
                        w_dec[i][j] -= lr * grad_w_dec[i][j];
                    }
                }

                // Normalize decoder columns to unit norm (prevents feature collapse)
                for j in 0..expansion_dim {
                    let norm: f32 = (0..hidden_dim)
                        .map(|i| w_dec[i][j] * w_dec[i][j])
                        .sum::<f32>()
                        .sqrt();
                    if norm > 1.0 {
                        for i in 0..hidden_dim {
                            w_dec[i][j] /= norm;
                        }
                    }
                }
            }

            final_mse = epoch_mse / n as f32;
            avg_sparsity = epoch_sparsity / n as f32;

            // Early stopping if converged
            if epoch > 5 && final_mse < 0.001 {
                break;
            }
        }

        Self {
            w_enc,
            b_enc,
            w_dec,
            b_dec,
            hidden_dim,
            expansion_dim,
            l1_coeff: config.l1_coeff,
            final_mse,
            avg_sparsity,
        }
    }

    fn empty(hidden_dim: usize, config: &SaeConfig) -> Self {
        let expansion_dim = hidden_dim * config.expansion_factor;
        Self {
            w_enc: vec![vec![0.0; hidden_dim]; expansion_dim],
            b_enc: vec![0.0; expansion_dim],
            w_dec: vec![vec![0.0; expansion_dim]; hidden_dim],
            b_dec: vec![0.0; hidden_dim],
            hidden_dim,
            expansion_dim,
            l1_coeff: config.l1_coeff,
            final_mse: 0.0,
            avg_sparsity: 0.0,
        }
    }

    /// Encode an embedding into sparse features.
    pub fn encode(&self, x: &[f32]) -> Vec<f32> {
        let mut latent = vec![0.0f32; self.expansion_dim];
        for i in 0..self.expansion_dim {
            let mut val = self.b_enc[i];
            for j in 0..self.hidden_dim {
                val += self.w_enc[i][j] * (x[j] - self.b_dec[j]);
            }
            latent[i] = val.max(0.0); // ReLU
        }
        latent
    }

    /// Decode sparse features back to embedding.
    pub fn decode(&self, z: &[f32]) -> Vec<f32> {
        let mut x_hat = vec![0.0f32; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let mut val = self.b_dec[i];
            for j in 0..self.expansion_dim {
                val += self.w_dec[i][j] * z[j];
            }
            x_hat[i] = val;
        }
        x_hat
    }

    /// Encode and return only the active (non-zero) features.
    pub fn active_features(&self, x: &[f32]) -> Vec<(usize, f32)> {
        let latent = self.encode(x);
        let mut active: Vec<(usize, f32)> = latent
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > 0.0)
            .map(|(i, v)| (i, *v))
            .collect();
        active.sort_by(|a, b| b.1.total_cmp(&a.1));
        active
    }

    /// Compute reconstruction error for an embedding.
    pub fn reconstruction_error(&self, x: &[f32]) -> f32 {
        let latent = self.encode(x);
        let x_hat = self.decode(&latent);
        let mse: f32 = x
            .iter()
            .zip(x_hat.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / self.hidden_dim as f32;
        mse
    }

    /// Compute sparsity for an embedding (fraction of zero features).
    pub fn sparsity(&self, x: &[f32]) -> f32 {
        let latent = self.encode(x);
        let zeros = latent.iter().filter(|v| **v == 0.0).count();
        zeros as f32 / self.expansion_dim as f32
    }
}

// ═══════════════════════════════════════════════════════════════
// Feature Labeling — correlate SAE features with financial concepts
// ═══════════════════════════════════════════════════════════════

/// A labeled SAE feature with its financial meaning.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SaeFeatureLabel {
    pub feature_id: usize,
    pub label: String,
    pub description: String,
    /// Pearson R² with the concept.
    pub correlation: f32,
    /// Domain this concept belongs to.
    pub domain: String,
}

/// Financial concepts we can detect in SAE features.
#[derive(Debug, Clone, Copy)]
pub enum FinancialConcept {
    HighDegreeHub,
    AnomalousEntity,
    DebtHolder,
    GoalConnected,
    TaxRelated,
    RecurringPattern,
    AssetHolder,
    MerchantRelation,
    HighRisk,
    LowActivity,
}

impl FinancialConcept {
    fn all() -> Vec<Self> {
        vec![
            Self::HighDegreeHub,
            Self::AnomalousEntity,
            Self::DebtHolder,
            Self::GoalConnected,
            Self::TaxRelated,
            Self::RecurringPattern,
            Self::AssetHolder,
            Self::MerchantRelation,
            Self::HighRisk,
            Self::LowActivity,
        ]
    }

    fn name(&self) -> &'static str {
        match self {
            Self::HighDegreeHub => "high_degree_hub",
            Self::AnomalousEntity => "anomalous_entity",
            Self::DebtHolder => "debt_holder",
            Self::GoalConnected => "goal_connected",
            Self::TaxRelated => "tax_related",
            Self::RecurringPattern => "recurring_pattern",
            Self::AssetHolder => "asset_holder",
            Self::MerchantRelation => "merchant_relation",
            Self::HighRisk => "high_risk",
            Self::LowActivity => "low_activity",
        }
    }

    fn description(&self) -> &'static str {
        match self {
            Self::HighDegreeHub => "Node with many connections — central to financial network",
            Self::AnomalousEntity => "Entity flagged as anomalous by ensemble models",
            Self::DebtHolder => "Connected to obligations, loans, or debt instruments",
            Self::GoalConnected => "Related to user financial goals or savings targets",
            Self::TaxRelated => "Connected to tax entities, obligations, or planning",
            Self::RecurringPattern => "Part of a recurring financial pattern or subscription",
            Self::AssetHolder => "Connected to owned assets or valuations",
            Self::MerchantRelation => "Entity involved in merchant transactions",
            Self::HighRisk => "Elevated anomaly score indicating financial risk",
            Self::LowActivity => "Low engagement — potential wasted cost",
        }
    }

    fn domain(&self) -> &'static str {
        match self {
            Self::HighDegreeHub => "structural",
            Self::AnomalousEntity | Self::HighRisk => "risk",
            Self::DebtHolder => "debt_obligations",
            Self::GoalConnected => "goals_budgets",
            Self::TaxRelated => "tax_optimization",
            Self::RecurringPattern | Self::LowActivity => "recurring_patterns",
            Self::AssetHolder => "asset_management",
            Self::MerchantRelation => "transactions",
        }
    }
}

/// Compute concept labels for each node from graph structure.
pub fn compute_concept_labels(
    edges: &HashMap<(String, String, String), Vec<(usize, usize)>>,
    anomaly_scores: &HashMap<String, HashMap<String, Vec<f32>>>,
    node_type: &str,
    num_nodes: usize,
) -> Vec<Vec<f32>> {
    let concepts = FinancialConcept::all();
    let mut labels = vec![vec![0.0f32; concepts.len()]; num_nodes];

    for node_id in 0..num_nodes {
        // Degree (hub detection)
        let degree = count_node_edges(edges, node_type, node_id);
        labels[node_id][0] = (degree as f32 / 50.0).min(1.0);

        // Anomaly score
        let anomaly = anomaly_scores
            .values()
            .next()
            .and_then(|m| m.get(node_type))
            .and_then(|scores| scores.get(node_id))
            .copied()
            .unwrap_or(0.0);
        labels[node_id][1] = anomaly;

        // Debt connection
        let has_debt = edges.iter().any(|((src, rel, dst), elist)| {
            (rel.contains("obligation") || rel.contains("lien") || rel.contains("debt"))
                && ((src == node_type && elist.iter().any(|(s, _)| *s == node_id))
                    || (dst == node_type && elist.iter().any(|(_, d)| *d == node_id)))
        });
        labels[node_id][2] = if has_debt { 1.0 } else { 0.0 };

        // Goal connection
        let has_goal = edges.iter().any(|((src, rel, dst), elist)| {
            (rel.contains("goal") || dst.contains("goal") || src.contains("goal"))
                && ((src == node_type && elist.iter().any(|(s, _)| *s == node_id))
                    || (dst == node_type && elist.iter().any(|(_, d)| *d == node_id)))
        });
        labels[node_id][3] = if has_goal { 1.0 } else { 0.0 };

        // Tax related
        let has_tax = edges.iter().any(|((src, rel, dst), elist)| {
            (rel.contains("tax") || dst.contains("tax") || src.contains("tax"))
                && ((src == node_type && elist.iter().any(|(s, _)| *s == node_id))
                    || (dst == node_type && elist.iter().any(|(_, d)| *d == node_id)))
        });
        labels[node_id][4] = if has_tax { 1.0 } else { 0.0 };

        // Recurring pattern
        let has_recurring = edges.iter().any(|((src, rel, dst), elist)| {
            (rel.contains("recurring")
                || rel.contains("pattern")
                || dst.contains("recurring")
                || src.contains("recurring"))
                && ((src == node_type && elist.iter().any(|(s, _)| *s == node_id))
                    || (dst == node_type && elist.iter().any(|(_, d)| *d == node_id)))
        });
        labels[node_id][5] = if has_recurring { 1.0 } else { 0.0 };

        // Asset holder
        let has_asset = edges.iter().any(|((src, rel, dst), elist)| {
            (rel.contains("asset") || dst.contains("asset") || src.contains("asset"))
                && ((src == node_type && elist.iter().any(|(s, _)| *s == node_id))
                    || (dst == node_type && elist.iter().any(|(_, d)| *d == node_id)))
        });
        labels[node_id][6] = if has_asset { 1.0 } else { 0.0 };

        // Merchant relation
        let has_merchant = edges.iter().any(|((src, rel, dst), elist)| {
            (dst.contains("merchant") || src.contains("merchant") || rel.contains("merchant"))
                && ((src == node_type && elist.iter().any(|(s, _)| *s == node_id))
                    || (dst == node_type && elist.iter().any(|(_, d)| *d == node_id)))
        });
        labels[node_id][7] = if has_merchant { 1.0 } else { 0.0 };

        // High risk (anomaly > 0.5)
        labels[node_id][8] = if anomaly >= 0.5 { 1.0 } else { 0.0 };

        // Low activity (low degree)
        labels[node_id][9] = if degree <= 1 { 1.0 } else { 0.0 };
    }

    labels
}

/// Label SAE features by correlating them with financial concepts.
pub fn label_features(
    sae: &SparseAutoencoder,
    embeddings: &[Vec<f32>],
    concept_labels: &[Vec<f32>],
) -> Vec<SaeFeatureLabel> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }

    let concepts = FinancialConcept::all();
    let num_concepts = concepts.len();

    // Compute SAE activations for all embeddings
    let activations: Vec<Vec<f32>> = embeddings.iter().map(|emb| sae.encode(emb)).collect();

    let mut labels = Vec::new();

    // For each SAE feature, find the best-correlated financial concept
    for feat_id in 0..sae.expansion_dim {
        let feat_activations: Vec<f32> = activations.iter().map(|a| a[feat_id]).collect();

        // Skip features that never activate
        let max_act = feat_activations.iter().cloned().fold(0.0f32, f32::max);
        if max_act < 1e-6 {
            continue;
        }

        let mut best_concept = 0;
        let mut best_corr = 0.0f32;

        for ci in 0..num_concepts {
            let concept_vals: Vec<f32> = concept_labels.iter().map(|c| c[ci]).collect();

            let corr = pearson_r(&feat_activations, &concept_vals).abs();
            if corr > best_corr {
                best_corr = corr;
                best_concept = ci;
            }
        }

        if best_corr > 0.1 {
            labels.push(SaeFeatureLabel {
                feature_id: feat_id,
                label: concepts[best_concept].name().to_string(),
                description: concepts[best_concept].description().to_string(),
                correlation: best_corr,
                domain: concepts[best_concept].domain().to_string(),
            });
        }
    }

    // Sort by correlation strength
    labels.sort_by(|a, b| b.correlation.total_cmp(&a.correlation));
    labels
}

// ═══════════════════════════════════════════════════════════════
// SAE Explanation for Fiduciary
// ═══════════════════════════════════════════════════════════════

/// SAE explanation for a single node/recommendation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SaeExplanation {
    /// Active features with their labels and activations.
    pub active_features: Vec<ActiveSaeFeature>,
    /// Sparsity of this encoding (fraction of zero features).
    pub sparsity: f32,
    /// How well the SAE reconstructs this embedding.
    pub reconstruction_quality: f32,
    /// Summary narrative.
    pub summary: String,
}

/// A single active SAE feature in an explanation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ActiveSaeFeature {
    pub feature_id: usize,
    pub label: String,
    pub domain: String,
    pub activation: f32,
    pub description: String,
}

/// Build an SAE explanation for a node embedding.
pub fn explain(
    sae: &SparseAutoencoder,
    embedding: &[f32],
    feature_labels: &[SaeFeatureLabel],
) -> SaeExplanation {
    let active = sae.active_features(embedding);
    let sparsity = sae.sparsity(embedding);
    let recon_error = sae.reconstruction_error(embedding);
    let recon_quality = 1.0 - recon_error.min(1.0);

    // Build label map
    let label_map: HashMap<usize, &SaeFeatureLabel> =
        feature_labels.iter().map(|l| (l.feature_id, l)).collect();

    let active_features: Vec<ActiveSaeFeature> = active
        .iter()
        .take(10) // Top 10 active features
        .map(|&(feat_id, activation)| {
            if let Some(label) = label_map.get(&feat_id) {
                ActiveSaeFeature {
                    feature_id: feat_id,
                    label: label.label.clone(),
                    domain: label.domain.clone(),
                    activation,
                    description: label.description.clone(),
                }
            } else {
                ActiveSaeFeature {
                    feature_id: feat_id,
                    label: format!("unlabeled_{}", feat_id),
                    domain: "unknown".into(),
                    activation,
                    description: "Feature not yet correlated with a known concept".into(),
                }
            }
        })
        .collect();

    let labeled_active: Vec<&ActiveSaeFeature> = active_features
        .iter()
        .filter(|f| !f.label.starts_with("unlabeled"))
        .collect();

    let summary = if labeled_active.is_empty() {
        format!(
            "SAE encoded into {} active features (sparsity: {:.0}%, reconstruction: {:.0}%). \
             No labeled concepts detected.",
            active.len(),
            sparsity * 100.0,
            recon_quality * 100.0,
        )
    } else {
        let top_concepts: Vec<String> = labeled_active
            .iter()
            .take(3)
            .map(|f| format!("{} ({:.2})", f.label, f.activation))
            .collect();

        let domains: std::collections::HashSet<&str> =
            labeled_active.iter().map(|f| f.domain.as_str()).collect();

        format!(
            "SAE detected {} active features (sparsity: {:.0}%, reconstruction: {:.0}%). \
             Top concepts: {}. Domains: {:?}.",
            active.len(),
            sparsity * 100.0,
            recon_quality * 100.0,
            top_concepts.join(", "),
            domains,
        )
    };

    SaeExplanation {
        active_features,
        sparsity,
        reconstruction_quality: recon_quality,
        summary,
    }
}

// ═══════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════

fn pearson_r(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    if n < 2.0 {
        return 0.0;
    }

    let mean_a = a.iter().sum::<f32>() / n;
    let mean_b = b.iter().sum::<f32>() / n;

    let mut cov = 0.0f32;
    let mut var_a = 0.0f32;
    let mut var_b = 0.0f32;

    for i in 0..a.len() {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    if var_a < 1e-10 || var_b < 1e-10 {
        0.0
    } else {
        cov / (var_a.sqrt() * var_b.sqrt())
    }
}

fn count_node_edges(
    edges: &HashMap<(String, String, String), Vec<(usize, usize)>>,
    node_type: &str,
    node_id: usize,
) -> usize {
    let mut count = 0;
    for ((src_type, _, dst_type), elist) in edges {
        if src_type == node_type {
            count += elist.iter().filter(|(s, _)| *s == node_id).count();
        }
        if dst_type == node_type {
            count += elist.iter().filter(|(_, d)| *d == node_id).count();
        }
    }
    count
}
