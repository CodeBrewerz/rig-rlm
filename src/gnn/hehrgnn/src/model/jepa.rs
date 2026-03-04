//! Graph-JEPA: Joint Embedding Predictive Architecture for GNNs.
//!
//! Implements concepts from VL-JEPA (Meta FAIR, Yann LeCun) for graph learning:
//! - InfoNCE contrastive loss (replaces BPR) with alignment + uniformity
//! - Edge embedding predictor MLP (predicts in embedding space, not raw scores)
//!
//! Key JEPA insight: "predict in embedding space instead of raw data space"
//! - Alignment: positive edge embeddings should be close
//! - Uniformity: all embeddings should spread out (prevents collapse)
//!
//! Reference: VL-JEPA (arXiv 2512.10942v2)

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════
// InfoNCE Loss (replaces BPR)
// ═══════════════════════════════════════════════════════════════

/// Compute InfoNCE contrastive loss for link prediction.
///
/// Unlike BPR which only compares one positive vs one negative,
/// InfoNCE compares each positive against ALL negatives in the batch.
/// This gives richer gradients and prevents embedding collapse.
///
/// L_InfoNCE = -log(exp(sim(z_u, z_v)/τ) / Σ_k exp(sim(z_u, z_k)/τ))
///
/// where τ is the temperature parameter.
pub fn compute_infonce_loss(
    embeddings: &HashMap<String, Vec<Vec<f32>>>,
    positive: &[(String, usize, String, usize)],
    negative: &[(String, usize, String, usize)],
    temperature: f32,
) -> f32 {
    let temperature = temperature.max(1e-3);
    let n = positive.len().min(negative.len());
    if n == 0 {
        return 0.0;
    }

    let mut total_loss = 0.0f32;

    for i in 0..n {
        // Positive score
        let pos_sim = cosine_sim(
            embeddings,
            &positive[i].0,
            positive[i].1,
            &positive[i].2,
            positive[i].3,
        ) / temperature;

        // Collect all negative scores for this anchor
        let mut neg_sims = Vec::new();
        for j in 0..negative.len() {
            let neg_sim = cosine_sim(
                embeddings,
                &positive[i].0,
                positive[i].1,
                &negative[j].2,
                negative[j].3,
            ) / temperature;
            neg_sims.push(neg_sim);
        }

        // Numerator: positive
        let numerator = pos_sim;

        // Denominator: positive + all negatives (log-sum-exp for stability)
        let max_val = neg_sims.iter().cloned().fold(numerator, f32::max);
        let sum_exp: f32 =
            (numerator - max_val).exp() + neg_sims.iter().map(|s| (s - max_val).exp()).sum::<f32>();
        let sum_exp = sum_exp.max(1e-20);
        let log_denominator = max_val + sum_exp.ln();

        // InfoNCE loss = -log(exp(pos) / sum(exp(all)))
        total_loss += -(numerator - log_denominator).max(-10.0);
    }

    total_loss / n as f32
}

/// Compute uniformity regularization loss.
///
/// From JEPA: prevents representation collapse by pushing embeddings
/// apart. Uses the log of average pairwise Gaussian kernel.
///
/// L_uniform = log E[exp(-2||z_i - z_j||²)]
///
/// Lower uniformity loss = more uniformly distributed embeddings.
pub fn compute_uniformity_loss(embeddings: &HashMap<String, Vec<Vec<f32>>>) -> f32 {
    // Collect all embeddings
    let mut all_emb: Vec<&Vec<f32>> = Vec::new();
    for (_nt, vecs) in embeddings {
        for v in vecs {
            all_emb.push(v);
        }
    }

    let n = all_emb.len();
    if n < 2 {
        return 0.0;
    }

    // Sample pairs (limit to 200 pairs for efficiency)
    let max_pairs = 200.min(n * (n - 1) / 2);
    let mut log_terms = Vec::with_capacity(max_pairs);
    let mut count = 0;
    let mut seed: u64 = 42;

    for _ in 0..max_pairs {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let i = (seed >> 33) as usize % n;
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut j = (seed >> 33) as usize % n;
        if j == i {
            j = (j + 1) % n;
        }

        let dist_sq: f32 = all_emb[i]
            .iter()
            .zip(all_emb[j].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        log_terms.push((-2.0 * dist_sq).clamp(-80.0, 20.0));
        count += 1;
    }

    if count == 0 || log_terms.is_empty() {
        return 0.0;
    }

    // log(mean(exp(x))) with log-sum-exp stabilization.
    let max_log = log_terms.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max_log.is_finite() {
        return -20.0;
    }
    let sum_exp: f32 = log_terms.iter().map(|x| (*x - max_log).exp()).sum();
    if !sum_exp.is_finite() || sum_exp <= 0.0 {
        return -20.0;
    }
    let mean_log = max_log + (sum_exp / count as f32).ln();
    if mean_log.is_finite() {
        mean_log.clamp(-30.0, 30.0)
    } else {
        -20.0
    }
}

/// Combined JEPA loss: InfoNCE + λ * uniformity.
///
/// From VL-JEPA paper: "prediction error + regularization that avoids
/// representation collapse"
pub fn compute_jepa_loss(
    embeddings: &HashMap<String, Vec<Vec<f32>>>,
    positive: &[(String, usize, String, usize)],
    negative: &[(String, usize, String, usize)],
    temperature: f32,
    uniformity_weight: f32,
) -> f32 {
    let infonce = compute_infonce_loss(embeddings, positive, negative, temperature);
    let uniformity = compute_uniformity_loss(embeddings);
    infonce + uniformity_weight * uniformity
}

// ═══════════════════════════════════════════════════════════════
// Edge Embedding Predictor (JEPA Phase 2)
// ═══════════════════════════════════════════════════════════════

/// Edge embedding predictor MLP.
///
/// From JEPA: instead of computing raw dot-product scores,
/// predict a target edge embedding in a shared space.
///
/// Architecture: concat(z_u, z_v) → hidden → target_embedding
/// This is the "Predictor" component from VL-JEPA.
#[derive(Debug, Clone)]
pub struct EdgePredictor {
    /// W1: [2*dim → hidden_dim]
    pub w1: Vec<Vec<f32>>,
    pub b1: Vec<f32>,
    /// W2: [hidden_dim → embed_dim]
    pub w2: Vec<Vec<f32>>,
    pub b2: Vec<f32>,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub embed_dim: usize,
}

impl EdgePredictor {
    pub fn new(node_dim: usize, hidden_dim: usize, embed_dim: usize) -> Self {
        let input_dim = node_dim * 2; // concat two node embeddings
        let mut seed: u64 = 77777;

        let scale1 = (2.0 / (input_dim + hidden_dim) as f64).sqrt() as f32;
        let mut w1 = vec![vec![0.0f32; input_dim]; hidden_dim];
        for row in &mut w1 {
            for w in row.iter_mut() {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                *w = ((seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 2.0 * scale1;
            }
        }

        let scale2 = (2.0 / (hidden_dim + embed_dim) as f64).sqrt() as f32;
        let mut w2 = vec![vec![0.0f32; hidden_dim]; embed_dim];
        for row in &mut w2 {
            for w in row.iter_mut() {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                *w = ((seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 2.0 * scale2;
            }
        }

        EdgePredictor {
            w1,
            b1: vec![0.1; hidden_dim],
            w2,
            b2: vec![0.0; embed_dim],
            input_dim,
            hidden_dim,
            embed_dim,
        }
    }

    /// Forward pass: predict edge embedding from two node embeddings.
    pub fn predict(&self, z_u: &[f32], z_v: &[f32]) -> Vec<f32> {
        // Concat: [z_u || z_v]
        let mut input: Vec<f32> = z_u.to_vec();
        input.extend_from_slice(z_v);

        // Hidden layer: ReLU(W1 @ input + b1)
        let mut hidden = self.b1.clone();
        for (h, w_row) in hidden.iter_mut().zip(&self.w1) {
            for (j, &x) in input.iter().enumerate() {
                if j < w_row.len() {
                    *h += w_row[j] * x;
                }
            }
            *h = if *h > 0.0 { *h } else { 0.01 * *h }; // LeakyReLU
        }

        // Output: W2 @ hidden + b2
        let mut output = self.b2.clone();
        for (o, w_row) in output.iter_mut().zip(&self.w2) {
            for (j, &h) in hidden.iter().enumerate() {
                if j < w_row.len() {
                    *o += w_row[j] * h;
                }
            }
        }

        // L2 normalize
        let norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        output.iter_mut().for_each(|x| *x /= norm);

        output
    }

    /// SPSA training step: perturb predictor weights to minimize loss.
    pub fn spsa_step(
        &mut self,
        embeddings: &HashMap<String, Vec<Vec<f32>>>,
        positive: &[(String, usize, String, usize)],
        negative: &[(String, usize, String, usize)],
        lr: f32,
        temperature: f32,
        epoch: usize,
    ) {
        let eps = 0.01f32;

        // Flatten weights for perturbation
        let flat: Vec<f32> = self
            .w1
            .iter()
            .flatten()
            .chain(self.b1.iter())
            .chain(self.w2.iter().flatten())
            .chain(self.b2.iter())
            .cloned()
            .collect();

        // Rademacher perturbation
        let mut seed: u64 = (epoch as u64 * 997).wrapping_add(31);
        let delta: Vec<f32> = (0..flat.len())
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                if (seed >> 33) % 2 == 0 { 1.0 } else { -1.0 }
            })
            .collect();

        // Loss with +ε*δ
        let mut flat_plus: Vec<f32> = flat.iter().zip(&delta).map(|(w, d)| w + eps * d).collect();
        self.unflatten(&flat_plus);
        let loss_plus = self.predictor_infonce_loss(embeddings, positive, negative, temperature);

        // Loss with -ε*δ
        let flat_minus: Vec<f32> = flat.iter().zip(&delta).map(|(w, d)| w - eps * d).collect();
        self.unflatten(&flat_minus);
        let loss_minus = self.predictor_infonce_loss(embeddings, positive, negative, temperature);

        // SPSA gradient and update
        let grad = (loss_plus - loss_minus) / (2.0 * eps);
        let updated: Vec<f32> = flat
            .iter()
            .zip(&delta)
            .map(|(w, d)| w - lr * grad * d)
            .collect();
        self.unflatten(&updated);
    }

    /// InfoNCE loss using predictor embeddings.
    fn predictor_infonce_loss(
        &self,
        embeddings: &HashMap<String, Vec<Vec<f32>>>,
        positive: &[(String, usize, String, usize)],
        negative: &[(String, usize, String, usize)],
        temperature: f32,
    ) -> f32 {
        let temperature = temperature.max(1e-3);
        let n = positive.len().min(negative.len());
        if n == 0 {
            return 0.0;
        }

        let mut total = 0.0f32;
        for i in 0..n {
            let z_u = get_emb(embeddings, &positive[i].0, positive[i].1);
            let z_v = get_emb(embeddings, &positive[i].2, positive[i].3);
            let pred_pos = self.predict(&z_u, &z_v);

            // Positive similarity
            let pos_sim = emb_cosine(&pred_pos, &z_v) / temperature;

            // Negative similarities
            let mut neg_sims = Vec::new();
            for j in 0..negative.len().min(n * 3) {
                let z_neg = get_emb(embeddings, &negative[j].2, negative[j].3);
                let pred_neg = self.predict(&z_u, &z_neg);
                neg_sims.push(emb_cosine(&pred_neg, &z_neg) / temperature);
            }

            let max_val = neg_sims.iter().cloned().fold(pos_sim, f32::max);
            let sum_exp = (pos_sim - max_val).exp()
                + neg_sims.iter().map(|s| (s - max_val).exp()).sum::<f32>();
            let sum_exp = sum_exp.max(1e-20);
            total += -(pos_sim - max_val - sum_exp.ln()).max(-10.0);
        }
        total / n as f32
    }

    fn unflatten(&mut self, flat: &[f32]) {
        let mut idx = 0;
        for row in &mut self.w1 {
            for w in row.iter_mut() {
                *w = flat[idx];
                idx += 1;
            }
        }
        for b in &mut self.b1 {
            *b = flat[idx];
            idx += 1;
        }
        for row in &mut self.w2 {
            for w in row.iter_mut() {
                *w = flat[idx];
                idx += 1;
            }
        }
        for b in &mut self.b2 {
            *b = flat[idx];
            idx += 1;
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════

fn get_emb(embeddings: &HashMap<String, Vec<Vec<f32>>>, nt: &str, idx: usize) -> Vec<f32> {
    embeddings
        .get(nt)
        .and_then(|v| v.get(idx))
        .cloned()
        .unwrap_or_default()
}

fn emb_cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    dot / (na * nb)
}

/// Cosine similarity between two nodes in the embedding map.
fn cosine_sim(
    embeddings: &HashMap<String, Vec<Vec<f32>>>,
    src_type: &str,
    src_idx: usize,
    dst_type: &str,
    dst_idx: usize,
) -> f32 {
    let a = get_emb(embeddings, src_type, src_idx);
    let b = get_emb(embeddings, dst_type, dst_idx);
    emb_cosine(&a, &b)
}

fn epoch_temperature(base: f32, epoch: usize, total_epochs: usize) -> f32 {
    let base = base.clamp(0.03, 2.0);
    if total_epochs <= 1 {
        return base;
    }
    let warm = (total_epochs / 5).max(1);
    let cool_start = (total_epochs * 2) / 3;
    let start = (base * 1.5).min(2.0);
    let end = (base * 0.7).max(0.03);

    if epoch < warm {
        let t = epoch as f32 / warm as f32;
        start + (base - start) * t
    } else if epoch >= cool_start {
        let span = (total_epochs - cool_start).max(1) as f32;
        let t = (epoch - cool_start) as f32 / span;
        base + (end - base) * t
    } else {
        base
    }
}

// ═══════════════════════════════════════════════════════════════
// HEHRGNN JEPA Training
// ═══════════════════════════════════════════════════════════════

use burn::prelude::*;

/// JEPA training for HEHRGNN model: InfoNCE + uniformity on entity embeddings.
///
/// Since HEHRGNN uses entity/relation embedding tables (not HeteroGraph),
/// this function applies SPSA perturbation to the entity embedding weights
/// with InfoNCE contrastive loss + uniformity regularization.
///
/// From VL-JEPA: "predict in embedding space instead of raw data space"
pub fn train_hehrgnn_jepa<B: Backend>(
    model: &mut crate::model::hehrgnn::HehrgnnModel<B>,
    batch: &crate::data::batcher::HehrBatch<B>,
    epochs: usize,
    lr: f32,
    uniformity_weight: f32,
) -> HehrgnnJepaReport {
    let num_entities = model.embeddings.entity_embedding.weight.val().dims()[0];
    let hidden_dim = model.embeddings.entity_embedding.weight.val().dims()[1];
    let device = model.embeddings.entity_embedding.weight.val().device();

    // Extract pairs from batch triples for InfoNCE
    let triples_data: Vec<i64> = batch
        .primary_triples
        .clone()
        .into_data()
        .as_slice::<i64>()
        .unwrap()
        .to_vec();
    let batch_size = batch.primary_triples.dims()[0];

    // Build positive pairs: (head, tail) from each triple
    let mut positive_pairs: Vec<(usize, usize)> = Vec::new();
    for b in 0..batch_size {
        let head = triples_data[b * 3] as usize;
        let tail = triples_data[b * 3 + 2] as usize;
        if head < num_entities && tail < num_entities {
            positive_pairs.push((head, tail));
        }
    }

    if positive_pairs.is_empty() {
        return HehrgnnJepaReport {
            epochs_trained: 0,
            initial_loss: 0.0,
            final_loss: 0.0,
            initial_uniformity: 0.0,
            final_uniformity: 0.0,
        };
    }

    // Get initial embeddings and compute initial loss
    let initial_emb = extract_entity_emb_map::<B>(&model.embeddings.entity_embedding.weight.val());
    let base_temperature = 0.1f32;
    let initial_loss = entity_infonce_loss(
        &initial_emb,
        &positive_pairs,
        num_entities,
        base_temperature,
    );
    let initial_uniformity = compute_uniformity_loss(&initial_emb);

    let mut best_loss = initial_loss;
    let mut patience_counter: usize = 0;
    let mut final_loss = initial_loss;
    let mut final_uniformity = initial_uniformity;

    for epoch in 0..epochs {
        let epoch_temp = epoch_temperature(base_temperature, epoch, epochs);
        let eps = 0.01f32;

        // Get current weights
        let w = model.embeddings.entity_embedding.weight.val();
        let w_data: Vec<f32> = w.clone().into_data().as_slice::<f32>().unwrap().to_vec();

        // Rademacher perturbation
        let mut seed: u64 = (epoch as u64 * 7919).wrapping_add(42);
        let delta: Vec<f32> = (0..w_data.len())
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                if (seed >> 33) % 2 == 0 { 1.0 } else { -1.0 }
            })
            .collect();

        // +ε*δ perturbation
        let w_plus: Vec<f32> = w_data
            .iter()
            .zip(&delta)
            .map(|(w, d)| w + eps * d)
            .collect();
        let wp_tensor = Tensor::<B, 1>::from_data(w_plus.as_slice(), &device)
            .reshape([num_entities, hidden_dim]);
        model.embeddings.entity_embedding.weight = model
            .embeddings
            .entity_embedding
            .weight
            .clone()
            .map(|_| wp_tensor);
        let emb_plus = extract_entity_emb_map::<B>(&model.embeddings.entity_embedding.weight.val());
        let loss_plus = entity_infonce_loss(&emb_plus, &positive_pairs, num_entities, epoch_temp)
            + uniformity_weight * compute_uniformity_loss(&emb_plus);

        // -ε*δ perturbation
        let w_minus: Vec<f32> = w_data
            .iter()
            .zip(&delta)
            .map(|(w, d)| w - eps * d)
            .collect();
        let wm_tensor = Tensor::<B, 1>::from_data(w_minus.as_slice(), &device)
            .reshape([num_entities, hidden_dim]);
        model.embeddings.entity_embedding.weight = model
            .embeddings
            .entity_embedding
            .weight
            .clone()
            .map(|_| wm_tensor);
        let emb_minus =
            extract_entity_emb_map::<B>(&model.embeddings.entity_embedding.weight.val());
        let loss_minus = entity_infonce_loss(&emb_minus, &positive_pairs, num_entities, epoch_temp)
            + uniformity_weight * compute_uniformity_loss(&emb_minus);

        // SPSA gradient and update
        let grad = (loss_plus - loss_minus) / (2.0 * eps);
        let w_updated: Vec<f32> = w_data
            .iter()
            .zip(&delta)
            .map(|(w, d)| w - lr * grad * d)
            .collect();
        let wu_tensor = Tensor::<B, 1>::from_data(w_updated.as_slice(), &device)
            .reshape([num_entities, hidden_dim]);
        model.embeddings.entity_embedding.weight = model
            .embeddings
            .entity_embedding
            .weight
            .clone()
            .map(|_| wu_tensor);

        let current_emb =
            extract_entity_emb_map::<B>(&model.embeddings.entity_embedding.weight.val());
        let current_loss =
            entity_infonce_loss(&current_emb, &positive_pairs, num_entities, epoch_temp);
        let current_uniform = compute_uniformity_loss(&current_emb);
        let combined = current_loss + uniformity_weight * current_uniform;

        final_loss = current_loss;
        final_uniformity = current_uniform;

        if epoch % 5 == 0 || epoch == epochs - 1 {
            eprintln!(
                "  [hehrgnn:jepa] epoch {}: infonce={:.4}, uniform={:.4}, temp={:.3}",
                epoch, current_loss, current_uniform, epoch_temp
            );
        }

        if combined < best_loss - 0.001 {
            best_loss = combined;
            patience_counter = 0;
        } else {
            patience_counter += 1;
            if patience_counter >= 10 {
                break;
            }
        }
    }

    HehrgnnJepaReport {
        epochs_trained: epochs,
        initial_loss,
        final_loss,
        initial_uniformity,
        final_uniformity,
    }
}

/// Report from HEHRGNN JEPA training.
#[derive(Debug, Clone)]
pub struct HehrgnnJepaReport {
    pub epochs_trained: usize,
    pub initial_loss: f32,
    pub final_loss: f32,
    pub initial_uniformity: f32,
    pub final_uniformity: f32,
}

/// InfoNCE loss for entity pairs (used by HEHRGNN which has flat entity IDs).
fn entity_infonce_loss(
    emb_map: &HashMap<String, Vec<Vec<f32>>>,
    positive_pairs: &[(usize, usize)],
    num_entities: usize,
    temperature: f32,
) -> f32 {
    let temperature = temperature.max(1e-3);
    let entities = match emb_map.get("entity") {
        Some(e) => e,
        None => return 0.0,
    };

    let n = positive_pairs.len();
    if n == 0 {
        return 0.0;
    }

    let mut total = 0.0f32;
    for &(head, tail) in positive_pairs {
        if head >= entities.len() || tail >= entities.len() {
            continue;
        }

        let pos_sim = emb_cosine(&entities[head], &entities[tail]) / temperature;

        // Sample negatives: random entities
        let mut neg_sims = Vec::new();
        let mut seed: u64 = (head as u64 * 997 + tail as u64).wrapping_add(31);
        for _ in 0..10.min(num_entities) {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let neg = (seed >> 33) as usize % entities.len();
            if neg != head && neg != tail {
                neg_sims.push(emb_cosine(&entities[head], &entities[neg]) / temperature);
            }
        }

        if neg_sims.is_empty() {
            continue;
        }

        let max_val = neg_sims.iter().cloned().fold(pos_sim, f32::max);
        let sum_exp =
            (pos_sim - max_val).exp() + neg_sims.iter().map(|s| (s - max_val).exp()).sum::<f32>();
        let sum_exp = sum_exp.max(1e-20);
        total += -(pos_sim - max_val - sum_exp.ln()).max(-10.0);
    }

    total / n as f32
}

/// Extract entity embeddings from a Burn tensor into our HashMap format.
fn extract_entity_emb_map<B: Backend>(
    entity_weight: &Tensor<B, 2>,
) -> HashMap<String, Vec<Vec<f32>>> {
    let dims = entity_weight.dims();
    let (n, d) = (dims[0], dims[1]);
    let data: Vec<f32> = entity_weight
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec();

    let mut entities = Vec::with_capacity(n);
    for i in 0..n {
        entities.push(data[i * d..(i + 1) * d].to_vec());
    }

    let mut map = HashMap::new();
    map.insert("entity".to_string(), entities);
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embeddings() -> HashMap<String, Vec<Vec<f32>>> {
        let mut emb = HashMap::new();
        emb.insert(
            "user".to_string(),
            vec![
                vec![1.0, 0.0, 0.0, 0.0], // alice
                vec![0.9, 0.1, 0.0, 0.0], // bob
            ],
        );
        emb.insert(
            "account".to_string(),
            vec![
                vec![0.8, 0.2, 0.0, 0.0], // acct_alice (similar to alice)
                vec![0.0, 0.0, 1.0, 0.0], // acct_other (dissimilar)
            ],
        );
        emb
    }

    #[test]
    fn test_infonce_loss() {
        let emb = make_embeddings();
        let pos = vec![("user".to_string(), 0, "account".to_string(), 0)]; // alice→acct_alice
        let neg = vec![("user".to_string(), 0, "account".to_string(), 1)]; // alice→acct_other

        let loss = compute_infonce_loss(&emb, &pos, &neg, 0.1);
        println!("  InfoNCE loss (τ=0.1): {:.4}", loss);
        assert!(loss > 0.0, "Loss should be positive");
        assert!(loss < 10.0, "Loss should be bounded");

        // Higher temperature smooths the distribution
        let loss_warm = compute_infonce_loss(&emb, &pos, &neg, 1.0);
        println!("  InfoNCE loss (τ=1.0): {:.4}", loss_warm);
        // Both should be valid positive losses
        assert!(loss_warm > 0.0, "Warm loss should be positive");
    }

    #[test]
    fn test_uniformity_loss() {
        let emb = make_embeddings();
        let uniform = compute_uniformity_loss(&emb);
        println!("  Uniformity loss: {:.4}", uniform);

        // Collapsed embeddings should have worse uniformity
        let mut collapsed = HashMap::new();
        collapsed.insert(
            "a".to_string(),
            vec![
                vec![0.5, 0.5, 0.5, 0.5],
                vec![0.5, 0.5, 0.5, 0.5],
                vec![0.5, 0.5, 0.5, 0.5],
            ],
        );
        let collapsed_uniform = compute_uniformity_loss(&collapsed);
        println!("  Collapsed uniformity: {:.4}", collapsed_uniform);
        assert!(
            collapsed_uniform > uniform,
            "Collapsed should have higher (worse) uniformity loss"
        );
    }

    #[test]
    fn test_edge_predictor() {
        let pred = EdgePredictor::new(4, 8, 4);
        let z_u = vec![1.0, 0.0, 0.0, 0.0];
        let z_v = vec![0.8, 0.2, 0.0, 0.0];

        let output = pred.predict(&z_u, &z_v);
        println!("  Edge prediction: {:?}", output);

        // Output should be L2 normalized
        let norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Output should be L2 normalized");
    }

    #[test]
    fn test_jepa_combined_loss() {
        let emb = make_embeddings();
        let pos = vec![("user".to_string(), 0, "account".to_string(), 0)];
        let neg = vec![("user".to_string(), 0, "account".to_string(), 1)];

        let bpr = crate::model::trainer::compute_bpr_loss(&emb, &pos, &neg);
        let jepa = compute_jepa_loss(&emb, &pos, &neg, 0.1, 0.5);

        println!("  BPR loss:  {:.4}", bpr);
        println!("  JEPA loss: {:.4}", jepa);

        // BPR should be positive. JEPA can be negative (uniformity term is negative for spread embeddings)
        assert!(bpr > 0.0);
        assert!(jepa.is_finite(), "JEPA loss should be finite");
    }
}
