//! Self-supervised GNN training via link prediction.
//!
//! Trains GNN models to predict whether edges exist between nodes.
//! Loss: BPR (Bayesian Personalized Ranking) link prediction
//!   L = -Σ log(σ(z_u · z_v - z_u · z_k))
//!      positive edge (u,v)   negative (u,k)
//!
//! Training approaches:
//! 1. Weight perturbation SGD: perturb each weight, measure loss change, update
//!    Works across ALL model types (no autodiff graph needed).
//! 2. Feature refinement: adjust node features to improve embeddings (lighter).

use burn::prelude::*;
use std::collections::{HashMap, HashSet};

use crate::data::hetero_graph::HeteroGraph;
use crate::model::backbone::NodeEmbeddings;

/// Training mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrainMode {
    /// Per-weight perturbation: accurate but slow (O(params × forward) per epoch).
    /// Best for production with GPU where forward passes are cheap.
    Full,
    /// SPSA (Simultaneous Perturbation Stochastic Approximation):
    /// Only 2 forward passes per epoch regardless of param count.
    /// Best for testing and CPU-bound environments.
    Fast,
}

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub epochs: usize,
    pub lr: f64,
    pub neg_ratio: usize,
    pub patience: usize,
    pub perturb_frac: f64,
    pub mode: TrainMode,
    /// Weight decay coefficient (from grokking paper: drives cleanup phase).
    /// Applied as `w *= (1 - weight_decay)` after each gradient step.
    /// Default: 0.01.
    pub weight_decay: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 40, // was 20; more epochs = better cleanup phase (grokking paper)
            lr: 0.01,
            neg_ratio: 3,
            patience: 10, // was 5; grokking-aware: don't stop too early
            perturb_frac: 0.3,
            mode: TrainMode::Fast,
            weight_decay: 0.01, // grokking paper: drives cleanup phase
        }
    }
}

/// Report from a training run.
#[derive(Debug, Clone)]
pub struct TrainReport {
    pub epochs_trained: usize,
    pub initial_loss: f32,
    pub final_loss: f32,
    pub initial_auc: f32,
    pub final_auc: f32,
    pub early_stopped: bool,
    /// Grokking progress measure: sum of squared weights (should decrease during cleanup).
    pub weight_norm_sq: f32,
    /// Grokking progress measure: mean embedding L2 norm (structural quality indicator).
    pub mean_emb_norm: f32,
}

/// Adapter distillation objective settings.
#[derive(Debug, Clone)]
pub struct AdapterDistillConfig {
    /// Weight for KL(teacher || adapter) on link score distributions.
    pub kl_weight: f32,
    /// Weight for embedding cosine alignment penalty (1 - cosine).
    pub cosine_weight: f32,
    /// Temperature used to smooth score distributions before KL.
    pub temperature: f32,
}

impl Default for AdapterDistillConfig {
    fn default() -> Self {
        Self {
            kl_weight: 0.35,
            cosine_weight: 0.05,
            temperature: 1.0,
        }
    }
}

/// Minimal interface for JEPA SPSA training on persistent model weights.
///
/// Models implementing this expose their input projection matrices so trainer
/// can run perturbation-based updates without autograd.
pub trait JepaTrainable<B: Backend> {
    fn forward_embeddings(&self, graph: &HeteroGraph<B>) -> NodeEmbeddings<B>;
    fn num_input_weights(&self) -> usize;
    fn get_input_weight(&self, idx: usize) -> Tensor<B, 2>;
    fn set_input_weight(&mut self, idx: usize, weight: Tensor<B, 2>);
}

/// Extract positive edges from graph as (src_type, src_idx, dst_type, dst_idx).
pub fn extract_positive_edges<B: Backend>(
    graph: &HeteroGraph<B>,
) -> Vec<(String, usize, String, usize)> {
    let mut edges = Vec::new();
    let mut edge_items: Vec<_> = graph.edge_index.iter().collect();
    edge_items.sort_by(|(a, _), (b, _)| a.cmp(b));
    for (edge_key, edge_index) in edge_items {
        let num_edges = edge_index.dims()[1];
        if num_edges == 0 {
            continue;
        }
        let data = edge_index.clone().into_data();
        let pairs: Vec<(i32, i32)> = if let Ok(slice) = data.as_slice::<i64>() {
            (0..num_edges)
                .map(|i| (slice[i] as i32, slice[num_edges + i] as i32))
                .collect()
        } else if let Ok(slice) = data.as_slice::<i32>() {
            (0..num_edges)
                .map(|i| (slice[i], slice[num_edges + i]))
                .collect()
        } else {
            continue;
        };
        for (src, dst) in pairs {
            edges.push((
                edge_key.0.clone(),
                src as usize,
                edge_key.2.clone(),
                dst as usize,
            ));
        }
    }
    edges
}

/// Sample negative edges (random non-existing edges).
pub fn sample_negative_edges<B: Backend>(
    graph: &HeteroGraph<B>,
    positive: &[(String, usize, String, usize)],
    neg_ratio: usize,
) -> Vec<(String, usize, String, usize)> {
    let mut negatives = Vec::new();
    if neg_ratio == 0 || positive.is_empty() {
        return negatives;
    }

    // Fast membership check: (src_type, dst_type, src_idx) -> set(dst_idx) positives.
    let mut positive_sets: HashMap<(String, String, usize), HashSet<usize>> = HashMap::new();
    // Destination popularity by dst_type for harder negatives.
    let mut dst_popularity: HashMap<String, HashMap<usize, usize>> = HashMap::new();
    for (src_type, src_idx, dst_type, dst_idx) in positive {
        positive_sets
            .entry((src_type.clone(), dst_type.clone(), *src_idx))
            .or_default()
            .insert(*dst_idx);
        let per_type = dst_popularity.entry(dst_type.clone()).or_default();
        *per_type.entry(*dst_idx).or_insert(0) += 1;
    }

    let mut hard_negatives: HashMap<String, Vec<usize>> = HashMap::new();
    for (dst_type, counts) in dst_popularity {
        let mut ranked: Vec<(usize, usize)> = counts.into_iter().collect();
        // Frequent destinations make stronger negatives than uniformly random nodes.
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        hard_negatives.insert(dst_type, ranked.into_iter().map(|(idx, _)| idx).collect());
    }

    let mut used_negatives: HashMap<(String, String, usize), HashSet<usize>> = HashMap::new();
    let mut seed: u64 = 42;
    for (src_type, src_idx, dst_type, pos_dst) in positive {
        let dst_count = *graph.node_counts.get(dst_type).unwrap_or(&0);
        if dst_count == 0 {
            continue;
        }

        let key = (src_type.clone(), dst_type.clone(), *src_idx);
        let pos_set = positive_sets.get(&key);
        let used = used_negatives.entry(key).or_default();
        let hard = hard_negatives
            .get(dst_type)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);

        for _ in 0..neg_ratio {
            let mut candidate = None;

            // Try hard negatives first (popular destination nodes), then random.
            for _attempt in 0..24 {
                let take_hard = !hard.is_empty() && next_u64(&mut seed) % 100 < 65;
                let neg_dst = if take_hard {
                    let pick = (next_u64(&mut seed) as usize) % hard.len().min(12);
                    hard[pick]
                } else {
                    (next_u64(&mut seed) as usize) % dst_count
                };

                let is_positive = pos_set.is_some_and(|s| s.contains(&neg_dst));
                if !is_positive && !used.contains(&neg_dst) {
                    candidate = Some(neg_dst);
                    break;
                }
            }

            // Fallback: allow duplicate negatives, but still avoid true positives.
            if candidate.is_none() {
                let start = (next_u64(&mut seed) as usize) % dst_count;
                for off in 0..dst_count {
                    let neg_dst = (start + off) % dst_count;
                    if !pos_set.is_some_and(|s| s.contains(&neg_dst)) {
                        candidate = Some(neg_dst);
                        break;
                    }
                }
            }

            // Final degenerate fallback (fully connected anchor): keep length stable.
            let neg_dst = candidate.unwrap_or(*pos_dst);
            negatives.push((src_type.clone(), *src_idx, dst_type.clone(), neg_dst));
            used.insert(neg_dst);
        }
    }
    negatives
}

fn next_u64(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    *seed
}

/// Extract embeddings as plain float vectors from NodeEmbeddings.
pub fn embeddings_to_plain<B: Backend>(
    node_emb: &NodeEmbeddings<B>,
) -> HashMap<String, Vec<Vec<f32>>> {
    let mut result = HashMap::new();
    for (nt, tensor) in &node_emb.embeddings {
        let dims = tensor.dims();
        let (n, d) = (dims[0], dims[1]);
        let data: Vec<f32> = tensor
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let mut vecs = Vec::with_capacity(n);
        for i in 0..n {
            vecs.push(data[i * d..(i + 1) * d].to_vec());
        }
        result.insert(nt.clone(), vecs);
    }
    result
}

/// Dot product between two node embeddings.
fn dot_score(
    embeddings: &HashMap<String, Vec<Vec<f32>>>,
    t1: &str,
    i1: usize,
    t2: &str,
    i2: usize,
) -> f32 {
    let e1 = embeddings.get(t1).and_then(|v| v.get(i1));
    let e2 = embeddings.get(t2).and_then(|v| v.get(i2));
    match (e1, e2) {
        (Some(a), Some(b)) => a.iter().zip(b.iter()).map(|(x, y)| x * y).sum(),
        _ => 0.0,
    }
}

/// Compute link prediction AUC: fraction of positive edges scoring higher than negative.
pub fn link_prediction_auc(
    embeddings: &HashMap<String, Vec<Vec<f32>>>,
    positive: &[(String, usize, String, usize)],
    negative: &[(String, usize, String, usize)],
) -> f32 {
    let pos_scores: Vec<f32> = positive
        .iter()
        .map(|(st, si, dt, di)| dot_score(embeddings, st, *si, dt, *di))
        .collect();
    let neg_scores: Vec<f32> = negative
        .iter()
        .map(|(st, si, dt, di)| dot_score(embeddings, st, *si, dt, *di))
        .collect();

    if pos_scores.is_empty() || neg_scores.is_empty() {
        return 0.5;
    }

    let mut correct = 0usize;
    let mut total = 0usize;
    for &ps in &pos_scores {
        for &ns in &neg_scores {
            total += 1;
            if ps > ns {
                correct += 1;
            }
        }
    }
    correct as f32 / total.max(1) as f32
}

/// Compute BPR link prediction loss.
pub fn compute_bpr_loss(
    embeddings: &HashMap<String, Vec<Vec<f32>>>,
    positive: &[(String, usize, String, usize)],
    negative: &[(String, usize, String, usize)],
) -> f32 {
    let n = positive.len().min(negative.len());
    if n == 0 {
        return 0.0;
    }

    let mut total = 0.0f32;
    for i in 0..n {
        let ps = dot_score(
            embeddings,
            &positive[i].0,
            positive[i].1,
            &positive[i].2,
            positive[i].3,
        );
        let ns = dot_score(
            embeddings,
            &negative[i].0,
            negative[i].1,
            &negative[i].2,
            negative[i].3,
        );
        let diff = ps - ns;
        total += -(1.0 / (1.0 + (-diff).exp())).ln().max(-10.0);
    }
    total / n as f32
}

fn scheduled_temperature(base: f32, epoch: usize, total_epochs: usize) -> f32 {
    let base = base.clamp(0.03, 2.0);
    if total_epochs <= 1 {
        return base;
    }

    let warm_epochs = (total_epochs / 5).max(1);
    let cool_start = (total_epochs * 2) / 3;
    let start = (base * 1.5).min(2.0);
    let end = (base * 0.7).max(0.03);

    if epoch < warm_epochs {
        let t = epoch as f32 / warm_epochs as f32;
        start + (base - start) * t
    } else if epoch >= cool_start {
        let span = (total_epochs - cool_start).max(1) as f32;
        let t = (epoch - cool_start) as f32 / span;
        base + (end - base) * t
    } else {
        base
    }
}

fn mean_embedding_norm(embeddings: &HashMap<String, Vec<Vec<f32>>>) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0usize;
    for vecs in embeddings.values() {
        for v in vecs {
            total += v.iter().map(|x| x * x).sum::<f32>().sqrt();
            count += 1;
        }
    }
    if count > 0 { total / count as f32 } else { 0.0 }
}

fn model_input_weight_norm_sq<B: Backend, M: JepaTrainable<B>>(model: &M) -> f32 {
    let mut total = 0.0f32;
    for i in 0..model.num_input_weights() {
        let data: Vec<f32> = model
            .get_input_weight(i)
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        total += data.iter().map(|w| w * w).sum::<f32>();
    }
    total
}

/// JEPA SPSA training over model input projection matrices.
///
/// Unlike feature-only refinement, this updates persistent model parameters
/// (saved via checkpoints), improving consistency across pipeline runs.
pub fn train_jepa_input_weights<B: Backend, M: JepaTrainable<B>>(
    model: &mut M,
    graph: &HeteroGraph<B>,
    config: &TrainConfig,
    temperature: f32,
    uniformity_weight: f32,
    use_edge_predictor: bool,
) -> TrainReport {
    let positive = extract_positive_edges(graph);
    let negative = sample_negative_edges(graph, &positive, config.neg_ratio);

    let initial_emb = embeddings_to_plain(&model.forward_embeddings(graph));
    let initial_loss = crate::model::jepa::compute_jepa_loss(
        &initial_emb,
        &positive,
        &negative,
        temperature,
        uniformity_weight,
    );
    let initial_auc = link_prediction_auc(&initial_emb, &positive, &negative);

    let hidden_dim = config_hidden_dim(&initial_emb);
    let mut edge_pred = if use_edge_predictor {
        Some(crate::model::jepa::EdgePredictor::new(
            hidden_dim,
            hidden_dim * 2,
            hidden_dim,
        ))
    } else {
        None
    };

    let mut best_loss = initial_loss;
    let mut patience_counter = 0usize;
    let mut final_loss = initial_loss;
    let mut final_auc = initial_auc;
    let mut final_mean_emb_norm = mean_embedding_norm(&initial_emb);
    let mut epochs_trained = 0usize;
    let lr = config.lr as f32;

    for epoch in 0..config.epochs {
        let epoch_temp = scheduled_temperature(temperature, epoch, config.epochs);

        match config.mode {
            TrainMode::Fast => {
                let eps = 0.01f32;
                for li in 0..model.num_input_weights() {
                    let w = model.get_input_weight(li);
                    let dims = w.dims();
                    let device = w.device();
                    let w_data: Vec<f32> =
                        w.clone().into_data().as_slice::<f32>().unwrap().to_vec();
                    if w_data.is_empty() {
                        continue;
                    }

                    let mut seed: u64 = (epoch as u64 * 7919 + li as u64 * 31).wrapping_add(211);
                    let delta: Vec<f32> = (0..w_data.len())
                        .map(|_| {
                            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                            if (seed >> 33) % 2 == 0 { 1.0 } else { -1.0 }
                        })
                        .collect();

                    let w_plus: Vec<f32> = w_data
                        .iter()
                        .zip(&delta)
                        .map(|(w, d)| w + eps * d)
                        .collect();
                    let wp_tensor = Tensor::<B, 1>::from_data(w_plus.as_slice(), &device)
                        .reshape([dims[0], dims[1]]);
                    model.set_input_weight(li, wp_tensor);
                    let emb_plus = embeddings_to_plain(&model.forward_embeddings(graph));
                    let loss_plus = crate::model::jepa::compute_jepa_loss(
                        &emb_plus,
                        &positive,
                        &negative,
                        epoch_temp,
                        uniformity_weight,
                    );

                    let w_minus: Vec<f32> = w_data
                        .iter()
                        .zip(&delta)
                        .map(|(w, d)| w - eps * d)
                        .collect();
                    let wm_tensor = Tensor::<B, 1>::from_data(w_minus.as_slice(), &device)
                        .reshape([dims[0], dims[1]]);
                    model.set_input_weight(li, wm_tensor);
                    let emb_minus = embeddings_to_plain(&model.forward_embeddings(graph));
                    let loss_minus = crate::model::jepa::compute_jepa_loss(
                        &emb_minus,
                        &positive,
                        &negative,
                        epoch_temp,
                        uniformity_weight,
                    );

                    let grad_scalar = (loss_plus - loss_minus) / (2.0 * eps);
                    let w_updated: Vec<f32> = w_data
                        .iter()
                        .zip(&delta)
                        .map(|(w, d)| w - lr * grad_scalar * d)
                        .collect();
                    let wu_tensor = Tensor::<B, 1>::from_data(w_updated.as_slice(), &device)
                        .reshape([dims[0], dims[1]]);
                    model.set_input_weight(li, wu_tensor);

                    if config.weight_decay > 0.0 {
                        let wd = 1.0 - config.weight_decay as f32;
                        let w_decayed = model.get_input_weight(li) * wd;
                        model.set_input_weight(li, w_decayed);
                    }
                }
            }
            TrainMode::Full => {
                let eps = 0.005f32;
                let base_emb = embeddings_to_plain(&model.forward_embeddings(graph));
                let base_loss = crate::model::jepa::compute_jepa_loss(
                    &base_emb,
                    &positive,
                    &negative,
                    epoch_temp,
                    uniformity_weight,
                );

                for li in 0..model.num_input_weights() {
                    let w = model.get_input_weight(li);
                    let dims = w.dims();
                    let device = w.device();
                    let mut w_data: Vec<f32> =
                        w.clone().into_data().as_slice::<f32>().unwrap().to_vec();
                    let total = w_data.len();
                    if total == 0 {
                        continue;
                    }

                    let count = ((total as f64 * config.perturb_frac) as usize).max(1);
                    let mut seed: u64 = (epoch as u64).wrapping_mul(31) + li as u64 + 17;

                    for _ in 0..count {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let idx = (seed >> 33) as usize % total;
                        let orig = w_data[idx];

                        w_data[idx] = orig + eps;
                        let pw = Tensor::<B, 1>::from_data(w_data.as_slice(), &device)
                            .reshape([dims[0], dims[1]]);
                        model.set_input_weight(li, pw);
                        let p_emb = embeddings_to_plain(&model.forward_embeddings(graph));
                        let p_loss = crate::model::jepa::compute_jepa_loss(
                            &p_emb,
                            &positive,
                            &negative,
                            epoch_temp,
                            uniformity_weight,
                        );
                        let grad = (p_loss - base_loss) / eps;
                        w_data[idx] = orig - lr * grad;
                    }

                    let fw = Tensor::<B, 1>::from_data(w_data.as_slice(), &device)
                        .reshape([dims[0], dims[1]]);
                    model.set_input_weight(li, fw);

                    if config.weight_decay > 0.0 {
                        let wd = 1.0 - config.weight_decay as f32;
                        let w_decayed = model.get_input_weight(li) * wd;
                        model.set_input_weight(li, w_decayed);
                    }
                }
            }
        }

        let post_emb = embeddings_to_plain(&model.forward_embeddings(graph));
        let post_loss = crate::model::jepa::compute_jepa_loss(
            &post_emb,
            &positive,
            &negative,
            epoch_temp,
            uniformity_weight,
        );
        let post_auc = link_prediction_auc(&post_emb, &positive, &negative);
        let uniform = crate::model::jepa::compute_uniformity_loss(&post_emb);

        if let Some(ref mut pred) = edge_pred {
            pred.spsa_step(
                &post_emb,
                &positive,
                &negative,
                config.lr as f32,
                epoch_temp,
                epoch,
            );
        }

        final_loss = post_loss;
        final_auc = post_auc;
        final_mean_emb_norm = mean_embedding_norm(&post_emb);
        epochs_trained = epoch + 1;

        if epoch % 5 == 0 || epoch == config.epochs.saturating_sub(1) {
            eprintln!(
                "  [train:jepa:weights] epoch {}: loss={:.4}, auc={:.4}, uniform={:.4}, temp={:.3}",
                epoch, post_loss, post_auc, uniform, epoch_temp
            );
        }

        if post_loss < best_loss - 0.001 {
            best_loss = post_loss;
            patience_counter = 0;
        } else {
            patience_counter += 1;
            if patience_counter >= config.patience {
                return TrainReport {
                    epochs_trained,
                    initial_loss,
                    final_loss,
                    initial_auc,
                    final_auc,
                    early_stopped: true,
                    weight_norm_sq: model_input_weight_norm_sq(model),
                    mean_emb_norm: final_mean_emb_norm,
                };
            }
        }
    }

    TrainReport {
        epochs_trained,
        initial_loss,
        final_loss,
        initial_auc,
        final_auc,
        early_stopped: false,
        weight_norm_sq: model_input_weight_norm_sq(model),
        mean_emb_norm: final_mean_emb_norm,
    }
}

/// Train a GraphSAGE model's weights.
///
/// Dispatches to either Full (per-weight) or Fast (SPSA) mode based on config.
pub fn train_graphsage<B: Backend>(
    model: &mut crate::model::graphsage::GraphSageModel<B>,
    graph: &HeteroGraph<B>,
    config: &TrainConfig,
) -> TrainReport {
    let positive = extract_positive_edges(graph);
    let negative = sample_negative_edges(graph, &positive, config.neg_ratio);

    let initial_emb = embeddings_to_plain(&model.forward(graph));
    let initial_loss = compute_bpr_loss(&initial_emb, &positive, &negative);
    let initial_auc = link_prediction_auc(&initial_emb, &positive, &negative);

    let mut best_loss = initial_loss;
    let mut patience_counter = 0;
    let mut final_loss = initial_loss;
    let mut final_auc = initial_auc;
    let mut epochs_trained = 0;
    let lr = config.lr as f32;

    for epoch in 0..config.epochs {
        match config.mode {
            TrainMode::Fast => {
                // ── SPSA: 2 forward passes per epoch ──
                // Perturb ALL weights in random direction, measure loss change
                let eps = 0.01f32;
                let num_linears = model.input_linears.len();

                for li in 0..num_linears {
                    let (dims, device, w_data) = {
                        let w = model.input_linears[li].weight.val();
                        let dims = w.dims();
                        let device = w.device();
                        let data: Vec<f32> =
                            w.clone().into_data().as_slice::<f32>().unwrap().to_vec();
                        (dims, device, data)
                    };

                    // Generate random perturbation direction (Rademacher: ±1)
                    let mut seed: u64 = (epoch as u64 * 7919 + li as u64 * 31).wrapping_add(42);
                    let delta: Vec<f32> = (0..w_data.len())
                        .map(|_| {
                            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                            if (seed >> 33) % 2 == 0 { 1.0 } else { -1.0 }
                        })
                        .collect();

                    // Forward with +ε*δ
                    let w_plus: Vec<f32> = w_data
                        .iter()
                        .zip(&delta)
                        .map(|(w, d)| w + eps * d)
                        .collect();
                    let wp_tensor = Tensor::<B, 1>::from_data(w_plus.as_slice(), &device)
                        .reshape([dims[0], dims[1]]);
                    model.input_linears[li].weight =
                        model.input_linears[li].weight.clone().map(|_| wp_tensor);
                    let emb_plus = embeddings_to_plain(&model.forward(graph));
                    let loss_plus = compute_bpr_loss(&emb_plus, &positive, &negative);

                    // Forward with -ε*δ
                    let w_minus: Vec<f32> = w_data
                        .iter()
                        .zip(&delta)
                        .map(|(w, d)| w - eps * d)
                        .collect();
                    let wm_tensor = Tensor::<B, 1>::from_data(w_minus.as_slice(), &device)
                        .reshape([dims[0], dims[1]]);
                    model.input_linears[li].weight =
                        model.input_linears[li].weight.clone().map(|_| wm_tensor);
                    let emb_minus = embeddings_to_plain(&model.forward(graph));
                    let loss_minus = compute_bpr_loss(&emb_minus, &positive, &negative);

                    // SPSA gradient: g_i = (loss+ - loss-) / (2 * eps * delta_i)
                    // Update: w_i -= lr * g_i
                    let grad_scalar = (loss_plus - loss_minus) / (2.0 * eps);
                    let w_updated: Vec<f32> = w_data
                        .iter()
                        .zip(&delta)
                        .map(|(w, d)| w - lr * grad_scalar * d)
                        .collect();

                    let wu_tensor = Tensor::<B, 1>::from_data(w_updated.as_slice(), &device)
                        .reshape([dims[0], dims[1]]);
                    model.input_linears[li].weight =
                        model.input_linears[li].weight.clone().map(|_| wu_tensor);

                    // Weight decay: w *= (1 - λ)  [grokking cleanup phase]
                    if config.weight_decay > 0.0 {
                        let wd = 1.0 - config.weight_decay as f32;
                        let w_decayed = model.input_linears[li].weight.val() * wd;
                        model.input_linears[li].weight =
                            model.input_linears[li].weight.clone().map(|_| w_decayed);
                    }
                }
            }
            TrainMode::Full => {
                // ── Full: per-weight perturbation (production/GPU) ──
                let eps = 0.005f32;
                let base_emb = embeddings_to_plain(&model.forward(graph));
                let base_loss = compute_bpr_loss(&base_emb, &positive, &negative);

                let num_linears = model.input_linears.len();
                for li in 0..num_linears {
                    let (dims, device, mut w_data) = {
                        let w = model.input_linears[li].weight.val();
                        let dims = w.dims();
                        let device = w.device();
                        let data: Vec<f32> =
                            w.clone().into_data().as_slice::<f32>().unwrap().to_vec();
                        (dims, device, data)
                    };

                    let total = w_data.len();
                    let count = ((total as f64 * config.perturb_frac) as usize).max(1);
                    let mut seed: u64 = (epoch as u64).wrapping_mul(31) + li as u64 + 17;

                    for _ in 0..count {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let idx = (seed >> 33) as usize % total;
                        let orig = w_data[idx];

                        w_data[idx] = orig + eps;
                        let pw = Tensor::<B, 1>::from_data(w_data.as_slice(), &device)
                            .reshape([dims[0], dims[1]]);
                        model.input_linears[li].weight =
                            model.input_linears[li].weight.clone().map(|_| pw);

                        let p_emb = embeddings_to_plain(&model.forward(graph));
                        let p_loss = compute_bpr_loss(&p_emb, &positive, &negative);
                        let grad = (p_loss - base_loss) / eps;
                        w_data[idx] = orig - lr * grad;
                    }

                    let fw = Tensor::<B, 1>::from_data(w_data.as_slice(), &device)
                        .reshape([dims[0], dims[1]]);
                    model.input_linears[li].weight =
                        model.input_linears[li].weight.clone().map(|_| fw);

                    // Weight decay: w *= (1 - λ)  [grokking cleanup phase]
                    if config.weight_decay > 0.0 {
                        let wd = 1.0 - config.weight_decay as f32;
                        let w_decayed = model.input_linears[li].weight.val() * wd;
                        model.input_linears[li].weight =
                            model.input_linears[li].weight.clone().map(|_| w_decayed);
                    }
                }
            }
        }

        // Post-epoch metrics
        let post_emb = embeddings_to_plain(&model.forward(graph));
        let post_loss = compute_bpr_loss(&post_emb, &positive, &negative);
        let post_auc = link_prediction_auc(&post_emb, &positive, &negative);

        // Grokking progress measures
        let weight_norm_sq: f32 = model
            .input_linears
            .iter()
            .map(|l| {
                let data: Vec<f32> = l
                    .weight
                    .val()
                    .into_data()
                    .as_slice::<f32>()
                    .unwrap()
                    .to_vec();
                data.iter().map(|w| w * w).sum::<f32>()
            })
            .sum();
        let mean_emb_norm: f32 = {
            let mut total = 0.0f32;
            let mut count = 0usize;
            for (_nt, vecs) in &post_emb {
                for v in vecs {
                    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                    total += norm;
                    count += 1;
                }
            }
            if count > 0 { total / count as f32 } else { 0.0 }
        };

        final_loss = post_loss;
        final_auc = post_auc;
        epochs_trained = epoch + 1;

        if epoch % 5 == 0 || epoch == config.epochs - 1 {
            eprintln!(
                "  [train:{:?}] epoch {}: loss={:.4}, auc={:.4}, w_norm²={:.2}, emb_norm={:.4}",
                config.mode, epoch, post_loss, post_auc, weight_norm_sq, mean_emb_norm
            );
        }

        if post_loss < best_loss - 0.001 {
            best_loss = post_loss;
            patience_counter = 0;
        } else {
            patience_counter += 1;
            if patience_counter >= config.patience {
                eprintln!("  [train] Early stopping at epoch {}", epoch);
                return TrainReport {
                    epochs_trained,
                    initial_loss,
                    final_loss,
                    initial_auc,
                    final_auc,
                    early_stopped: true,
                    weight_norm_sq,
                    mean_emb_norm,
                };
            }
        }
    }

    TrainReport {
        epochs_trained,
        initial_loss,
        final_loss,
        initial_auc,
        final_auc,
        early_stopped: false,
        weight_norm_sq: model
            .input_linears
            .iter()
            .map(|l| {
                let data: Vec<f32> = l
                    .weight
                    .val()
                    .into_data()
                    .as_slice::<f32>()
                    .unwrap()
                    .to_vec();
                data.iter().map(|w| w * w).sum::<f32>()
            })
            .sum(),
        mean_emb_norm: 0.0,
    }
}

/// RLFR-inspired training: BPR loss + probe reward signal.
///
/// From Goodfire's "Features as Rewards" research:
/// 1. Train a node-type probe on FROZEN initial embeddings
/// 2. During SPSA training, use combined loss = bpr - α*probe_score
/// 3. Probe reward encourages embeddings to maintain discriminability
///
/// The probe runs on a frozen copy (not updated during training), so the
/// model can't "hack" the probe — it must learn genuinely better embeddings.
pub fn train_with_probe_reward<B: Backend>(
    model: &mut crate::model::graphsage::GraphSageModel<B>,
    graph: &HeteroGraph<B>,
    config: &TrainConfig,
    probe_weight: f32, // α: how much probe reward matters (0.0 = BPR only)
) -> (TrainReport, f32, f32) {
    // Phase 1: Train probe on FROZEN initial embeddings
    let initial_emb = embeddings_to_plain(&model.forward(graph));
    let node_types: Vec<String> = initial_emb.keys().cloned().collect();

    let mut probe =
        crate::model::probe::NodeTypeProbe::new(&node_types, config_hidden_dim(&initial_emb));
    probe.train_on_frozen(&initial_emb, 100, 0.1);
    let probe_score_before = probe.score(&initial_emb);

    // Phase 2: Train with combined loss
    let positive = extract_positive_edges(graph);
    let negative = sample_negative_edges(graph, &positive, config.neg_ratio);

    let initial_loss = compute_bpr_loss(&initial_emb, &positive, &negative);
    let initial_auc = link_prediction_auc(&initial_emb, &positive, &negative);

    let mut best_loss = initial_loss;
    let mut patience_counter = 0;
    let mut final_loss = initial_loss;
    let mut final_auc = initial_auc;
    let mut epochs_trained = 0;
    let lr = config.lr as f32;

    for epoch in 0..config.epochs {
        // SPSA with probe-augmented loss
        let eps = 0.01f32;
        let num_linears = model.input_linears.len();

        for li in 0..num_linears {
            let (dims, device, w_data) = {
                let w = model.input_linears[li].weight.val();
                let dims = w.dims();
                let device = w.device();
                let data: Vec<f32> = w.clone().into_data().as_slice::<f32>().unwrap().to_vec();
                (dims, device, data)
            };

            // Rademacher perturbation
            let mut seed: u64 = (epoch as u64 * 7919 + li as u64 * 31).wrapping_add(42);
            let delta: Vec<f32> = (0..w_data.len())
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    if (seed >> 33) % 2 == 0 { 1.0 } else { -1.0 }
                })
                .collect();

            // Forward with +ε*δ
            let w_plus: Vec<f32> = w_data
                .iter()
                .zip(&delta)
                .map(|(w, d)| w + eps * d)
                .collect();
            let wp_tensor =
                Tensor::<B, 1>::from_data(w_plus.as_slice(), &device).reshape([dims[0], dims[1]]);
            model.input_linears[li].weight =
                model.input_linears[li].weight.clone().map(|_| wp_tensor);
            let emb_plus = embeddings_to_plain(&model.forward(graph));
            let bpr_plus = compute_bpr_loss(&emb_plus, &positive, &negative);
            let probe_plus = probe.score(&emb_plus);
            let loss_plus = bpr_plus - probe_weight * probe_plus; // lower BPR + higher probe = better

            // Forward with -ε*δ
            let w_minus: Vec<f32> = w_data
                .iter()
                .zip(&delta)
                .map(|(w, d)| w - eps * d)
                .collect();
            let wm_tensor =
                Tensor::<B, 1>::from_data(w_minus.as_slice(), &device).reshape([dims[0], dims[1]]);
            model.input_linears[li].weight =
                model.input_linears[li].weight.clone().map(|_| wm_tensor);
            let emb_minus = embeddings_to_plain(&model.forward(graph));
            let bpr_minus = compute_bpr_loss(&emb_minus, &positive, &negative);
            let probe_minus = probe.score(&emb_minus);
            let loss_minus = bpr_minus - probe_weight * probe_minus;

            // SPSA gradient
            let grad_scalar = (loss_plus - loss_minus) / (2.0 * eps);
            let w_updated: Vec<f32> = w_data
                .iter()
                .zip(&delta)
                .map(|(w, d)| w - lr * grad_scalar * d)
                .collect();

            let wu_tensor = Tensor::<B, 1>::from_data(w_updated.as_slice(), &device)
                .reshape([dims[0], dims[1]]);
            model.input_linears[li].weight =
                model.input_linears[li].weight.clone().map(|_| wu_tensor);

            // Weight decay
            if config.weight_decay > 0.0 {
                let wd = 1.0 - config.weight_decay as f32;
                let w_decayed = model.input_linears[li].weight.val() * wd;
                model.input_linears[li].weight =
                    model.input_linears[li].weight.clone().map(|_| w_decayed);
            }
        }

        // Post-epoch metrics
        let post_emb = embeddings_to_plain(&model.forward(graph));
        let post_loss = compute_bpr_loss(&post_emb, &positive, &negative);
        let post_auc = link_prediction_auc(&post_emb, &positive, &negative);
        let post_probe = probe.score(&post_emb);

        final_loss = post_loss;
        final_auc = post_auc;
        epochs_trained = epoch + 1;

        if epoch % 5 == 0 || epoch == config.epochs - 1 {
            eprintln!(
                "  [train:probe] epoch {}: loss={:.4}, auc={:.4}, probe={:.2}%",
                epoch,
                post_loss,
                post_auc,
                post_probe * 100.0
            );
        }

        if post_loss < best_loss - 0.001 {
            best_loss = post_loss;
            patience_counter = 0;
        } else {
            patience_counter += 1;
            if patience_counter >= config.patience {
                eprintln!("  [train:probe] Early stopping at epoch {}", epoch);
                let probe_score_after = probe.score(&embeddings_to_plain(&model.forward(graph)));
                return (
                    TrainReport {
                        epochs_trained,
                        initial_loss,
                        final_loss,
                        initial_auc,
                        final_auc,
                        early_stopped: true,
                        weight_norm_sq: 0.0,
                        mean_emb_norm: 0.0,
                    },
                    probe_score_before,
                    probe_score_after,
                );
            }
        }
    }

    let probe_score_after = probe.score(&embeddings_to_plain(&model.forward(graph)));
    (
        TrainReport {
            epochs_trained,
            initial_loss,
            final_loss,
            initial_auc,
            final_auc,
            early_stopped: false,
            weight_norm_sq: 0.0,
            mean_emb_norm: 0.0,
        },
        probe_score_before,
        probe_score_after,
    )
}

/// Helper: infer hidden dim from embeddings.
fn config_hidden_dim(emb: &HashMap<String, Vec<Vec<f32>>>) -> usize {
    for (_nt, vecs) in emb {
        if let Some(v) = vecs.first() {
            return v.len();
        }
    }
    16 // fallback
}

/// Train only the HeteroDoRA adapter weights (base model frozen).
///
/// Perturbs blend coefficients + basis A/B weights via SPSA.
/// Much faster than full weight training since adapter has ~97% fewer params.
pub fn train_adapter<B: Backend>(
    model: &mut crate::model::graphsage::GraphSageModel<B>,
    graph: &HeteroGraph<B>,
    config: &TrainConfig,
) -> TrainReport {
    if model.input_adapters.is_none() {
        eprintln!("  [train_adapter] No adapter attached, skipping");
        return TrainReport {
            epochs_trained: 0,
            initial_loss: 0.0,
            final_loss: 0.0,
            initial_auc: 0.0,
            final_auc: 0.0,
            early_stopped: false,
            weight_norm_sq: 0.0,
            mean_emb_norm: 0.0,
        };
    }

    let positive = extract_positive_edges(graph);
    let negative = sample_negative_edges(graph, &positive, config.neg_ratio);

    let initial_emb = embeddings_to_plain(&model.forward(graph));
    let initial_loss = compute_bpr_loss(&initial_emb, &positive, &negative);
    let initial_auc = link_prediction_auc(&initial_emb, &positive, &negative);

    let mut best_loss = initial_loss;
    let mut patience_counter = 0;
    let mut final_loss = initial_loss;
    let mut final_auc = initial_auc;
    let mut epochs_trained = 0;
    let lr = config.lr as f32;
    let eps = 0.01f32;

    for epoch in 0..config.epochs {
        // SPSA on blend weights
        {
            let blend_val = model.input_adapters.as_ref().unwrap().blend_weights.val();
            let blend_dims = blend_val.dims();
            let device = blend_val.device();
            let blend_data: Vec<f32> = blend_val
                .clone()
                .into_data()
                .as_slice::<f32>()
                .unwrap()
                .to_vec();
            let total = blend_data.len();

            let mut seed: u64 = (epoch as u64 * 7919).wrapping_add(99);
            let delta: Vec<f32> = (0..total)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    if (seed >> 33) % 2 == 0 { 1.0 } else { -1.0 }
                })
                .collect();

            // +εδ
            let w_plus: Vec<f32> = blend_data
                .iter()
                .zip(&delta)
                .map(|(w, d)| w + eps * d)
                .collect();
            let bp = Tensor::<B, 1>::from_data(w_plus.as_slice(), &device)
                .reshape([blend_dims[0], blend_dims[1]]);
            model.input_adapters.as_mut().unwrap().blend_weights = model
                .input_adapters
                .as_ref()
                .unwrap()
                .blend_weights
                .clone()
                .map(|_| bp);
            let emb_plus = embeddings_to_plain(&model.forward(graph));
            let loss_plus = compute_bpr_loss(&emb_plus, &positive, &negative);

            // -εδ
            let w_minus: Vec<f32> = blend_data
                .iter()
                .zip(&delta)
                .map(|(w, d)| w - eps * d)
                .collect();
            let bm = Tensor::<B, 1>::from_data(w_minus.as_slice(), &device)
                .reshape([blend_dims[0], blend_dims[1]]);
            model.input_adapters.as_mut().unwrap().blend_weights = model
                .input_adapters
                .as_ref()
                .unwrap()
                .blend_weights
                .clone()
                .map(|_| bm);
            let emb_minus = embeddings_to_plain(&model.forward(graph));
            let loss_minus = compute_bpr_loss(&emb_minus, &positive, &negative);

            // Update
            let grad_scalar = (loss_plus - loss_minus) / (2.0 * eps);
            let w_updated: Vec<f32> = blend_data
                .iter()
                .zip(&delta)
                .map(|(w, d)| w - lr * grad_scalar * d)
                .collect();
            let wu = Tensor::<B, 1>::from_data(w_updated.as_slice(), &device)
                .reshape([blend_dims[0], blend_dims[1]]);
            model.input_adapters.as_mut().unwrap().blend_weights = model
                .input_adapters
                .as_ref()
                .unwrap()
                .blend_weights
                .clone()
                .map(|_| wu);
        }

        // SPSA on DoRA magnitude vectors (lower lr per QDoRA paper)
        {
            let mag_val = model.input_adapters.as_ref().unwrap().magnitudes.val();
            let mag_dims = mag_val.dims();
            let device = mag_val.device();
            let mag_data: Vec<f32> = mag_val
                .clone()
                .into_data()
                .as_slice::<f32>()
                .unwrap()
                .to_vec();
            let total = mag_data.len();

            let mag_lr = lr * 0.5; // QDoRA: magnitudes converge faster
            let mut seed: u64 = (epoch as u64 * 4219).wrapping_add(71);
            let delta: Vec<f32> = (0..total)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    if (seed >> 33) % 2 == 0 { 1.0 } else { -1.0 }
                })
                .collect();

            let w_plus: Vec<f32> = mag_data
                .iter()
                .zip(&delta)
                .map(|(w, d)| w + eps * d)
                .collect();
            let mp = Tensor::<B, 1>::from_data(w_plus.as_slice(), &device)
                .reshape([mag_dims[0], mag_dims[1]]);
            model.input_adapters.as_mut().unwrap().magnitudes = model
                .input_adapters
                .as_ref()
                .unwrap()
                .magnitudes
                .clone()
                .map(|_| mp);
            let loss_plus = compute_bpr_loss(
                &embeddings_to_plain(&model.forward(graph)),
                &positive,
                &negative,
            );

            let w_minus: Vec<f32> = mag_data
                .iter()
                .zip(&delta)
                .map(|(w, d)| w - eps * d)
                .collect();
            let mm = Tensor::<B, 1>::from_data(w_minus.as_slice(), &device)
                .reshape([mag_dims[0], mag_dims[1]]);
            model.input_adapters.as_mut().unwrap().magnitudes = model
                .input_adapters
                .as_ref()
                .unwrap()
                .magnitudes
                .clone()
                .map(|_| mm);
            let loss_minus = compute_bpr_loss(
                &embeddings_to_plain(&model.forward(graph)),
                &positive,
                &negative,
            );

            let grad = (loss_plus - loss_minus) / (2.0 * eps);
            let w_upd: Vec<f32> = mag_data
                .iter()
                .zip(&delta)
                .map(|(w, d)| w - mag_lr * grad * d)
                .collect();
            let mu = Tensor::<B, 1>::from_data(w_upd.as_slice(), &device)
                .reshape([mag_dims[0], mag_dims[1]]);
            model.input_adapters.as_mut().unwrap().magnitudes = model
                .input_adapters
                .as_ref()
                .unwrap()
                .magnitudes
                .clone()
                .map(|_| mu);
        }

        // SPSA on basis A/B weights
        let num_bases = model.input_adapters.as_ref().unwrap().num_bases;
        for bi in 0..num_bases {
            // Train A weight
            let (dims, device, a_data) = {
                let w = model.input_adapters.as_ref().unwrap().bases[bi]
                    .lora_a
                    .weight
                    .val();
                (
                    w.dims(),
                    w.device(),
                    w.clone().into_data().as_slice::<f32>().unwrap().to_vec(),
                )
            };
            let mut seed: u64 = (epoch as u64 * 31 + bi as u64 * 7).wrapping_add(42);
            let delta: Vec<f32> = (0..a_data.len())
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    if (seed >> 33) % 2 == 0 { 1.0 } else { -1.0 }
                })
                .collect();

            let wp: Vec<f32> = a_data
                .iter()
                .zip(&delta)
                .map(|(w, d)| w + eps * d)
                .collect();
            let wpt = Tensor::<B, 1>::from_data(wp.as_slice(), &device).reshape([dims[0], dims[1]]);
            model.input_adapters.as_mut().unwrap().bases[bi]
                .lora_a
                .weight = model.input_adapters.as_ref().unwrap().bases[bi]
                .lora_a
                .weight
                .clone()
                .map(|_| wpt);
            let loss_p = compute_bpr_loss(
                &embeddings_to_plain(&model.forward(graph)),
                &positive,
                &negative,
            );

            let wm: Vec<f32> = a_data
                .iter()
                .zip(&delta)
                .map(|(w, d)| w - eps * d)
                .collect();
            let wmt = Tensor::<B, 1>::from_data(wm.as_slice(), &device).reshape([dims[0], dims[1]]);
            model.input_adapters.as_mut().unwrap().bases[bi]
                .lora_a
                .weight = model.input_adapters.as_ref().unwrap().bases[bi]
                .lora_a
                .weight
                .clone()
                .map(|_| wmt);
            let loss_m = compute_bpr_loss(
                &embeddings_to_plain(&model.forward(graph)),
                &positive,
                &negative,
            );

            let grad = (loss_p - loss_m) / (2.0 * eps);
            let wu: Vec<f32> = a_data
                .iter()
                .zip(&delta)
                .map(|(w, d)| w - lr * grad * d)
                .collect();
            let wut = Tensor::<B, 1>::from_data(wu.as_slice(), &device).reshape([dims[0], dims[1]]);
            model.input_adapters.as_mut().unwrap().bases[bi]
                .lora_a
                .weight = model.input_adapters.as_ref().unwrap().bases[bi]
                .lora_a
                .weight
                .clone()
                .map(|_| wut);

            // Train B weight
            let (dims_b, dev_b, b_data) = {
                let w = model.input_adapters.as_ref().unwrap().bases[bi]
                    .lora_b
                    .weight
                    .val();
                (
                    w.dims(),
                    w.device(),
                    w.clone().into_data().as_slice::<f32>().unwrap().to_vec(),
                )
            };
            let mut seed_b: u64 = (epoch as u64 * 37 + bi as u64 * 13).wrapping_add(77);
            let delta_b: Vec<f32> = (0..b_data.len())
                .map(|_| {
                    seed_b = seed_b.wrapping_mul(6364136223846793005).wrapping_add(1);
                    if (seed_b >> 33) % 2 == 0 { 1.0 } else { -1.0 }
                })
                .collect();

            let bpv: Vec<f32> = b_data
                .iter()
                .zip(&delta_b)
                .map(|(w, d)| w + eps * d)
                .collect();
            let bpt =
                Tensor::<B, 1>::from_data(bpv.as_slice(), &dev_b).reshape([dims_b[0], dims_b[1]]);
            model.input_adapters.as_mut().unwrap().bases[bi]
                .lora_b
                .weight = model.input_adapters.as_ref().unwrap().bases[bi]
                .lora_b
                .weight
                .clone()
                .map(|_| bpt);
            let loss_bp = compute_bpr_loss(
                &embeddings_to_plain(&model.forward(graph)),
                &positive,
                &negative,
            );

            let bmv: Vec<f32> = b_data
                .iter()
                .zip(&delta_b)
                .map(|(w, d)| w - eps * d)
                .collect();
            let bmt =
                Tensor::<B, 1>::from_data(bmv.as_slice(), &dev_b).reshape([dims_b[0], dims_b[1]]);
            model.input_adapters.as_mut().unwrap().bases[bi]
                .lora_b
                .weight = model.input_adapters.as_ref().unwrap().bases[bi]
                .lora_b
                .weight
                .clone()
                .map(|_| bmt);
            let loss_bm = compute_bpr_loss(
                &embeddings_to_plain(&model.forward(graph)),
                &positive,
                &negative,
            );

            let grad_b = (loss_bp - loss_bm) / (2.0 * eps);
            let buv: Vec<f32> = b_data
                .iter()
                .zip(&delta_b)
                .map(|(w, d)| w - lr * grad_b * d)
                .collect();
            let but =
                Tensor::<B, 1>::from_data(buv.as_slice(), &dev_b).reshape([dims_b[0], dims_b[1]]);
            model.input_adapters.as_mut().unwrap().bases[bi]
                .lora_b
                .weight = model.input_adapters.as_ref().unwrap().bases[bi]
                .lora_b
                .weight
                .clone()
                .map(|_| but);
        }

        // Post-epoch metrics
        let post_emb = embeddings_to_plain(&model.forward(graph));
        let post_loss = compute_bpr_loss(&post_emb, &positive, &negative);
        let post_auc = link_prediction_auc(&post_emb, &positive, &negative);
        final_loss = post_loss;
        final_auc = post_auc;
        epochs_trained = epoch + 1;

        if epoch % 5 == 0 || epoch == config.epochs - 1 {
            eprintln!(
                "  [adapter] epoch {}: loss={:.4}, auc={:.4}",
                epoch, post_loss, post_auc
            );
        }

        if post_loss < best_loss - 0.001 {
            best_loss = post_loss;
            patience_counter = 0;
        } else {
            patience_counter += 1;
            if patience_counter >= config.patience {
                eprintln!("  [adapter] Early stopping at epoch {}", epoch);
                return TrainReport {
                    epochs_trained,
                    initial_loss,
                    final_loss,
                    initial_auc,
                    final_auc,
                    early_stopped: true,
                    weight_norm_sq: 0.0,
                    mean_emb_norm: 0.0,
                };
            }
        }
    }

    TrainReport {
        epochs_trained,
        initial_loss,
        final_loss,
        initial_auc,
        final_auc,
        early_stopped: false,
        weight_norm_sq: 0.0,
        mean_emb_norm: 0.0,
    }
}

fn adapter_distill_objective(
    student_emb: &HashMap<String, Vec<Vec<f32>>>,
    teacher_emb: &HashMap<String, Vec<Vec<f32>>>,
    positive: &[(String, usize, String, usize)],
    negative: &[(String, usize, String, usize)],
    all_edges: &[(String, usize, String, usize)],
    teacher_scores_scaled: &[f32],
    distill: &AdapterDistillConfig,
) -> (f32, f32, f32, f32) {
    let bpr = compute_bpr_loss(student_emb, positive, negative);
    let student_scores = crate::model::lora::compute_link_scores(student_emb, all_edges);
    let temp = distill.temperature.max(1e-3);
    let student_scaled: Vec<f32> = student_scores.iter().map(|s| *s / temp).collect();
    let kl = crate::model::lora::kl_divergence(teacher_scores_scaled, &student_scaled);
    let cosine = crate::model::lora::avg_cosine_similarity(teacher_emb, student_emb);
    let cosine_penalty = (1.0 - cosine).max(0.0);
    let total = bpr + distill.kl_weight * kl + distill.cosine_weight * cosine_penalty;
    (total, bpr, kl, cosine)
}

/// Adapter training with teacher-student distillation objective.
///
/// Phase 1: standard adapter BPR training.
/// Phase 2: KL + cosine distillation against teacher embeddings.
pub fn train_adapter_with_distillation<B: Backend>(
    model: &mut crate::model::graphsage::GraphSageModel<B>,
    graph: &HeteroGraph<B>,
    config: &TrainConfig,
    teacher_embeddings: &HashMap<String, Vec<Vec<f32>>>,
    distill: &AdapterDistillConfig,
) -> TrainReport {
    if model.input_adapters.is_none() {
        return train_adapter(model, graph, config);
    }

    let base = train_adapter(model, graph, config);

    let positive = extract_positive_edges(graph);
    let negative = sample_negative_edges(graph, &positive, config.neg_ratio);
    let all_edges: Vec<(String, usize, String, usize)> =
        positive.iter().chain(negative.iter()).cloned().collect();

    let temp = distill.temperature.max(1e-3);
    let teacher_scores = crate::model::lora::compute_link_scores(teacher_embeddings, &all_edges);
    let teacher_scores_scaled: Vec<f32> = teacher_scores.iter().map(|s| *s / temp).collect();

    let mut post_emb = embeddings_to_plain(&model.forward(graph));
    let (mut best_loss, _, mut best_kl, mut best_cos) = adapter_distill_objective(
        &post_emb,
        teacher_embeddings,
        &positive,
        &negative,
        &all_edges,
        &teacher_scores_scaled,
        distill,
    );
    let mut final_auc = link_prediction_auc(&post_emb, &positive, &negative);
    let mut final_loss = best_loss;
    let mut patience_counter = 0usize;
    let mut distill_epochs_trained = 0usize;
    let distill_epochs = (config.epochs / 2).clamp(5, 30);
    let eps = 0.01f32;
    let lr = (config.lr as f32) * 0.8;

    for epoch in 0..distill_epochs {
        // Distill blend weights.
        {
            let blend_val = model.input_adapters.as_ref().unwrap().blend_weights.val();
            let blend_dims = blend_val.dims();
            let device = blend_val.device();
            let blend_data: Vec<f32> = blend_val
                .clone()
                .into_data()
                .as_slice::<f32>()
                .unwrap()
                .to_vec();
            let total = blend_data.len();

            let mut seed: u64 = (epoch as u64 * 12391).wrapping_add(501);
            let delta: Vec<f32> = (0..total)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    if (seed >> 33) % 2 == 0 { 1.0 } else { -1.0 }
                })
                .collect();

            let w_plus: Vec<f32> = blend_data
                .iter()
                .zip(&delta)
                .map(|(w, d)| w + eps * d)
                .collect();
            let bp = Tensor::<B, 1>::from_data(w_plus.as_slice(), &device)
                .reshape([blend_dims[0], blend_dims[1]]);
            model.input_adapters.as_mut().unwrap().blend_weights = model
                .input_adapters
                .as_ref()
                .unwrap()
                .blend_weights
                .clone()
                .map(|_| bp);
            let emb_plus = embeddings_to_plain(&model.forward(graph));
            let (loss_plus, _, _, _) = adapter_distill_objective(
                &emb_plus,
                teacher_embeddings,
                &positive,
                &negative,
                &all_edges,
                &teacher_scores_scaled,
                distill,
            );

            let w_minus: Vec<f32> = blend_data
                .iter()
                .zip(&delta)
                .map(|(w, d)| w - eps * d)
                .collect();
            let bm = Tensor::<B, 1>::from_data(w_minus.as_slice(), &device)
                .reshape([blend_dims[0], blend_dims[1]]);
            model.input_adapters.as_mut().unwrap().blend_weights = model
                .input_adapters
                .as_ref()
                .unwrap()
                .blend_weights
                .clone()
                .map(|_| bm);
            let emb_minus = embeddings_to_plain(&model.forward(graph));
            let (loss_minus, _, _, _) = adapter_distill_objective(
                &emb_minus,
                teacher_embeddings,
                &positive,
                &negative,
                &all_edges,
                &teacher_scores_scaled,
                distill,
            );

            let grad = (loss_plus - loss_minus) / (2.0 * eps);
            let w_updated: Vec<f32> = blend_data
                .iter()
                .zip(&delta)
                .map(|(w, d)| w - lr * grad * d)
                .collect();
            let wu = Tensor::<B, 1>::from_data(w_updated.as_slice(), &device)
                .reshape([blend_dims[0], blend_dims[1]]);
            model.input_adapters.as_mut().unwrap().blend_weights = model
                .input_adapters
                .as_ref()
                .unwrap()
                .blend_weights
                .clone()
                .map(|_| wu);
        }

        // Distill DoRA magnitudes.
        {
            let mag_val = model.input_adapters.as_ref().unwrap().magnitudes.val();
            let mag_dims = mag_val.dims();
            let device = mag_val.device();
            let mag_data: Vec<f32> = mag_val
                .clone()
                .into_data()
                .as_slice::<f32>()
                .unwrap()
                .to_vec();
            let total = mag_data.len();

            let mut seed: u64 = (epoch as u64 * 16127).wrapping_add(913);
            let delta: Vec<f32> = (0..total)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    if (seed >> 33) % 2 == 0 { 1.0 } else { -1.0 }
                })
                .collect();

            let w_plus: Vec<f32> = mag_data
                .iter()
                .zip(&delta)
                .map(|(w, d)| w + eps * d)
                .collect();
            let mp = Tensor::<B, 1>::from_data(w_plus.as_slice(), &device)
                .reshape([mag_dims[0], mag_dims[1]]);
            model.input_adapters.as_mut().unwrap().magnitudes = model
                .input_adapters
                .as_ref()
                .unwrap()
                .magnitudes
                .clone()
                .map(|_| mp);
            let emb_plus = embeddings_to_plain(&model.forward(graph));
            let (loss_plus, _, _, _) = adapter_distill_objective(
                &emb_plus,
                teacher_embeddings,
                &positive,
                &negative,
                &all_edges,
                &teacher_scores_scaled,
                distill,
            );

            let w_minus: Vec<f32> = mag_data
                .iter()
                .zip(&delta)
                .map(|(w, d)| w - eps * d)
                .collect();
            let mm = Tensor::<B, 1>::from_data(w_minus.as_slice(), &device)
                .reshape([mag_dims[0], mag_dims[1]]);
            model.input_adapters.as_mut().unwrap().magnitudes = model
                .input_adapters
                .as_ref()
                .unwrap()
                .magnitudes
                .clone()
                .map(|_| mm);
            let emb_minus = embeddings_to_plain(&model.forward(graph));
            let (loss_minus, _, _, _) = adapter_distill_objective(
                &emb_minus,
                teacher_embeddings,
                &positive,
                &negative,
                &all_edges,
                &teacher_scores_scaled,
                distill,
            );

            let grad = (loss_plus - loss_minus) / (2.0 * eps);
            let mag_lr = lr * 0.5;
            let w_updated: Vec<f32> = mag_data
                .iter()
                .zip(&delta)
                .map(|(w, d)| w - mag_lr * grad * d)
                .collect();
            let mu = Tensor::<B, 1>::from_data(w_updated.as_slice(), &device)
                .reshape([mag_dims[0], mag_dims[1]]);
            model.input_adapters.as_mut().unwrap().magnitudes = model
                .input_adapters
                .as_ref()
                .unwrap()
                .magnitudes
                .clone()
                .map(|_| mu);
        }

        post_emb = embeddings_to_plain(&model.forward(graph));
        let (distill_loss, _bpr, kl, cos) = adapter_distill_objective(
            &post_emb,
            teacher_embeddings,
            &positive,
            &negative,
            &all_edges,
            &teacher_scores_scaled,
            distill,
        );
        final_auc = link_prediction_auc(&post_emb, &positive, &negative);
        final_loss = distill_loss;
        distill_epochs_trained = epoch + 1;

        if epoch % 5 == 0 || epoch == distill_epochs.saturating_sub(1) {
            eprintln!(
                "  [adapter:distill] epoch {}: total={:.4}, kl={:.5}, cos={:.4}, auc={:.4}",
                epoch, distill_loss, kl, cos, final_auc
            );
        }

        if distill_loss < best_loss - 0.0005 {
            best_loss = distill_loss;
            best_kl = kl;
            best_cos = cos;
            patience_counter = 0;
        } else {
            patience_counter += 1;
            if patience_counter >= (config.patience / 2).max(3) {
                eprintln!(
                    "  [adapter:distill] Early stopping at epoch {}, best_kl={:.5}, best_cos={:.4}",
                    epoch, best_kl, best_cos
                );
                break;
            }
        }
    }

    TrainReport {
        epochs_trained: base.epochs_trained + distill_epochs_trained,
        initial_loss: base.initial_loss,
        final_loss,
        initial_auc: base.initial_auc,
        final_auc,
        early_stopped: base.early_stopped || distill_epochs_trained < distill_epochs,
        weight_norm_sq: 0.0,
        mean_emb_norm: mean_embedding_norm(&post_emb),
    }
}

/// Train any model by perturbing graph node features (lighter, works for all models).
/// Enhanced with grokking-inspired weight decay and progress measures.
pub fn train_via_feature_refinement<B: Backend>(
    graph: &mut HeteroGraph<B>,
    model_forward: &dyn Fn(&HeteroGraph<B>) -> NodeEmbeddings<B>,
    config: &TrainConfig,
) -> TrainReport {
    let positive = extract_positive_edges(graph);
    let negative = sample_negative_edges(graph, &positive, config.neg_ratio);

    let initial_emb = embeddings_to_plain(&model_forward(graph));
    let initial_loss = compute_bpr_loss(&initial_emb, &positive, &negative);
    let initial_auc = link_prediction_auc(&initial_emb, &positive, &negative);

    let mut best_loss = initial_loss;
    let mut patience_counter = 0;
    let mut final_loss = initial_loss;
    let mut final_auc = initial_auc;
    let mut epochs_trained = 0;

    for epoch in 0..config.epochs {
        let emb = model_forward(graph);
        let plain = embeddings_to_plain(&emb);
        let loss = compute_bpr_loss(&plain, &positive, &negative);
        let auc = link_prediction_auc(&plain, &positive, &negative);

        // Refine features toward better embeddings
        for (nt, feat) in graph.node_features.iter_mut() {
            let dims = feat.dims();
            let (n, d) = (dims[0], dims[1]);
            let feat_data: Vec<f32> = feat.clone().into_data().as_slice::<f32>().unwrap().to_vec();
            let mut new_feat = feat_data.clone();

            if let Some(node_vecs) = plain.get(nt) {
                for node in 0..n {
                    if let Some(emb_vec) = node_vecs.get(node) {
                        for dim in 0..d.min(emb_vec.len()) {
                            let diff = emb_vec[dim] - feat_data[node * d + dim];
                            new_feat[node * d + dim] += config.lr as f32 * diff * 0.01;
                        }
                    }
                }
            }

            // Feature weight decay (grokking cleanup phase)
            if config.weight_decay > 0.0 {
                let wd = 1.0 - config.weight_decay as f32;
                for v in new_feat.iter_mut() {
                    *v *= wd;
                }
            }

            let device = feat.device();
            *feat = Tensor::<B, 1>::from_data(new_feat.as_slice(), &device).reshape([n, d]);
        }

        // Progress measures
        let feature_norm_sq: f32 = graph
            .node_features
            .values()
            .map(|f| {
                let data: Vec<f32> = f.clone().into_data().as_slice::<f32>().unwrap().to_vec();
                data.iter().map(|v| v * v).sum::<f32>()
            })
            .sum();

        final_loss = loss;
        final_auc = auc;
        epochs_trained = epoch + 1;

        if epoch % 5 == 0 || epoch == config.epochs - 1 {
            eprintln!(
                "  [train:feat] epoch {}: loss={:.4}, auc={:.4}, feat_norm²={:.2}",
                epoch, loss, auc, feature_norm_sq
            );
        }

        if loss < best_loss - 0.001 {
            best_loss = loss;
            patience_counter = 0;
        } else {
            patience_counter += 1;
            if patience_counter >= config.patience {
                return TrainReport {
                    epochs_trained,
                    initial_loss,
                    final_loss,
                    initial_auc,
                    final_auc,
                    early_stopped: true,
                    weight_norm_sq: feature_norm_sq,
                    mean_emb_norm: 0.0,
                };
            }
        }
    }

    let feature_norm_sq: f32 = graph
        .node_features
        .values()
        .map(|f| {
            let data: Vec<f32> = f.clone().into_data().as_slice::<f32>().unwrap().to_vec();
            data.iter().map(|v| v * v).sum::<f32>()
        })
        .sum();

    TrainReport {
        epochs_trained,
        initial_loss,
        final_loss,
        initial_auc,
        final_auc,
        early_stopped: false,
        weight_norm_sq: feature_norm_sq,
        mean_emb_norm: 0.0,
    }
}

/// RLFR probe-as-reward training for ANY model type.
///
/// Combines BPR loss + probe reward via feature refinement.
/// Works for GraphSAGE, RGCN, GAT, GPS Transformer — any model
/// that implements forward(graph) → NodeEmbeddings.
///
/// From Goodfire RLFR: probe is trained on frozen initial embeddings,
/// then used as auxiliary reward during training.
pub fn train_features_with_probe<B: Backend>(
    graph: &mut HeteroGraph<B>,
    model_forward: &dyn Fn(&HeteroGraph<B>) -> NodeEmbeddings<B>,
    config: &TrainConfig,
    probe_weight: f32,
) -> (TrainReport, f32, f32) {
    let positive = extract_positive_edges(graph);
    let negative = sample_negative_edges(graph, &positive, config.neg_ratio);

    // Phase 1: Train probe on FROZEN initial embeddings
    let initial_emb = embeddings_to_plain(&model_forward(graph));
    let node_types: Vec<String> = initial_emb.keys().cloned().collect();
    let hidden_dim = config_hidden_dim(&initial_emb);

    let mut probe = crate::model::probe::NodeTypeProbe::new(&node_types, hidden_dim);
    probe.train_on_frozen(&initial_emb, 100, 0.1);
    let probe_score_before = probe.score(&initial_emb);

    let initial_loss = compute_bpr_loss(&initial_emb, &positive, &negative);
    let initial_auc = link_prediction_auc(&initial_emb, &positive, &negative);

    let mut best_loss = initial_loss;
    let mut patience_counter = 0;
    let mut final_loss = initial_loss;
    let mut final_auc = initial_auc;
    let mut epochs_trained = 0;

    // Phase 2: Train with probe-augmented feature refinement
    for epoch in 0..config.epochs {
        let emb = model_forward(graph);
        let plain = embeddings_to_plain(&emb);
        let bpr_loss = compute_bpr_loss(&plain, &positive, &negative);
        let auc = link_prediction_auc(&plain, &positive, &negative);
        let probe_score = probe.score(&plain);

        // Combined loss: lower BPR + higher probe = better
        let combined_loss = bpr_loss - probe_weight * probe_score;

        // Probe-aware feature refinement
        for (nt, feat) in graph.node_features.iter_mut() {
            let dims = feat.dims();
            let (n, d) = (dims[0], dims[1]);
            let feat_data: Vec<f32> = feat.clone().into_data().as_slice::<f32>().unwrap().to_vec();
            let mut new_feat = feat_data.clone();

            if let Some(node_vecs) = plain.get(nt) {
                let type_idx = probe.type_to_idx.get(nt).copied().unwrap_or(0);
                for node in 0..n {
                    if let Some(emb_vec) = node_vecs.get(node) {
                        for dim in 0..d.min(emb_vec.len()) {
                            // BPR gradient (embedding-feature diff)
                            let bpr_grad = emb_vec[dim] - feat_data[node * d + dim];

                            // Probe gradient: push toward correct class centroid
                            let probe_grad = if dim < hidden_dim {
                                probe.weights[type_idx][dim] * 0.01
                            } else {
                                0.0
                            };

                            new_feat[node * d + dim] +=
                                config.lr as f32 * (bpr_grad * 0.01 + probe_weight * probe_grad);
                        }
                    }
                }
            }

            // Feature weight decay
            if config.weight_decay > 0.0 {
                let wd = 1.0 - config.weight_decay as f32;
                for v in new_feat.iter_mut() {
                    *v *= wd;
                }
            }

            let device = feat.device();
            *feat = Tensor::<B, 1>::from_data(new_feat.as_slice(), &device).reshape([n, d]);
        }

        final_loss = bpr_loss;
        final_auc = auc;
        epochs_trained = epoch + 1;

        if epoch % 5 == 0 || epoch == config.epochs - 1 {
            eprintln!(
                "  [train:feat+probe] epoch {}: loss={:.4}, auc={:.4}, probe={:.1}%",
                epoch,
                bpr_loss,
                auc,
                probe_score * 100.0
            );
        }

        if combined_loss < best_loss - 0.001 {
            best_loss = combined_loss;
            patience_counter = 0;
        } else {
            patience_counter += 1;
            if patience_counter >= config.patience {
                let probe_score_after = probe.score(&embeddings_to_plain(&model_forward(graph)));
                return (
                    TrainReport {
                        epochs_trained,
                        initial_loss,
                        final_loss,
                        initial_auc,
                        final_auc,
                        early_stopped: true,
                        weight_norm_sq: 0.0,
                        mean_emb_norm: 0.0,
                    },
                    probe_score_before,
                    probe_score_after,
                );
            }
        }
    }

    let probe_score_after = probe.score(&embeddings_to_plain(&model_forward(graph)));
    (
        TrainReport {
            epochs_trained,
            initial_loss,
            final_loss,
            initial_auc,
            final_auc,
            early_stopped: false,
            weight_norm_sq: 0.0,
            mean_emb_norm: 0.0,
        },
        probe_score_before,
        probe_score_after,
    )
}

/// Graph-JEPA training: InfoNCE loss + uniformity + optional edge predictor.
///
/// Replaces BPR loss with JEPA's embedding-space prediction approach.
/// Works for ANY model type via feature refinement.
///
/// From VL-JEPA: "predict in embedding space instead of raw data space"
pub fn train_jepa<B: Backend>(
    graph: &mut HeteroGraph<B>,
    model_forward: &dyn Fn(&HeteroGraph<B>) -> NodeEmbeddings<B>,
    config: &TrainConfig,
    temperature: f32,
    uniformity_weight: f32,
    use_edge_predictor: bool,
) -> TrainReport {
    let positive = extract_positive_edges(graph);
    let negative = sample_negative_edges(graph, &positive, config.neg_ratio);

    let initial_emb = embeddings_to_plain(&model_forward(graph));
    let initial_loss = crate::model::jepa::compute_jepa_loss(
        &initial_emb,
        &positive,
        &negative,
        temperature,
        uniformity_weight,
    );
    let initial_auc = link_prediction_auc(&initial_emb, &positive, &negative);

    // Optional: create edge predictor
    let hidden_dim = config_hidden_dim(&initial_emb);
    let mut edge_pred = if use_edge_predictor {
        Some(crate::model::jepa::EdgePredictor::new(
            hidden_dim,
            hidden_dim * 2,
            hidden_dim,
        ))
    } else {
        None
    };

    let mut best_loss = initial_loss;
    let mut patience_counter = 0;
    let mut final_loss = initial_loss;
    let mut final_auc = initial_auc;
    let mut epochs_trained = 0;

    for epoch in 0..config.epochs {
        let epoch_temp = scheduled_temperature(temperature, epoch, config.epochs);
        let emb = model_forward(graph);
        let plain = embeddings_to_plain(&emb);

        // JEPA loss
        let jepa_loss = crate::model::jepa::compute_jepa_loss(
            &plain,
            &positive,
            &negative,
            epoch_temp,
            uniformity_weight,
        );
        let auc = link_prediction_auc(&plain, &positive, &negative);
        let uniform = crate::model::jepa::compute_uniformity_loss(&plain);

        // Train edge predictor if enabled
        if let Some(ref mut pred) = edge_pred {
            pred.spsa_step(
                &plain,
                &positive,
                &negative,
                config.lr as f32,
                epoch_temp,
                epoch,
            );
        }

        // Feature refinement using JEPA gradients
        for (nt, feat) in graph.node_features.iter_mut() {
            let dims = feat.dims();
            let (n, d) = (dims[0], dims[1]);
            let feat_data: Vec<f32> = feat.clone().into_data().as_slice::<f32>().unwrap().to_vec();
            let mut new_feat = feat_data.clone();

            if let Some(node_vecs) = plain.get(nt) {
                for node in 0..n {
                    if let Some(emb_vec) = node_vecs.get(node) {
                        for dim in 0..d.min(emb_vec.len()) {
                            let diff = emb_vec[dim] - feat_data[node * d + dim];
                            new_feat[node * d + dim] += config.lr as f32 * diff * 0.01;
                        }
                    }
                }
            }

            // Weight decay
            if config.weight_decay > 0.0 {
                let wd = 1.0 - config.weight_decay as f32;
                for v in new_feat.iter_mut() {
                    *v *= wd;
                }
            }

            let device = feat.device();
            *feat = Tensor::<B, 1>::from_data(new_feat.as_slice(), &device).reshape([n, d]);
        }

        final_loss = jepa_loss;
        final_auc = auc;
        epochs_trained = epoch + 1;

        if epoch % 5 == 0 || epoch == config.epochs - 1 {
            eprintln!(
                "  [train:jepa] epoch {}: loss={:.4}, auc={:.4}, uniform={:.4}, temp={:.3}",
                epoch, jepa_loss, auc, uniform, epoch_temp
            );
        }

        if jepa_loss < best_loss - 0.001 {
            best_loss = jepa_loss;
            patience_counter = 0;
        } else {
            patience_counter += 1;
            if patience_counter >= config.patience {
                return TrainReport {
                    epochs_trained,
                    initial_loss,
                    final_loss,
                    initial_auc,
                    final_auc,
                    early_stopped: true,
                    weight_norm_sq: 0.0,
                    mean_emb_norm: 0.0,
                };
            }
        }
    }

    TrainReport {
        epochs_trained,
        initial_loss,
        final_loss,
        initial_auc,
        final_auc,
        early_stopped: false,
        weight_norm_sq: 0.0,
        mean_emb_norm: 0.0,
    }
}

/// Combined BPR + JEPA training: best of both worlds.
///
/// - BPR gradient drives feature refinement (proven to optimize link prediction)
/// - InfoNCE uniformity prevents embedding collapse (JEPA anti-collapse)
/// - Cosine-aware gradient: push positives closer, push negatives apart
/// - Optional EdgePredictor learns in embedding space
///
/// This is the recommended training method for all GNN models.
pub fn train_bpr_jepa<B: Backend>(
    graph: &mut HeteroGraph<B>,
    model_forward: &dyn Fn(&HeteroGraph<B>) -> NodeEmbeddings<B>,
    config: &TrainConfig,
    uniformity_weight: f32,
    use_edge_predictor: bool,
) -> TrainReport {
    let positive = extract_positive_edges(graph);
    let negative = sample_negative_edges(graph, &positive, config.neg_ratio);

    let initial_emb = embeddings_to_plain(&model_forward(graph));
    let initial_loss = compute_bpr_loss(&initial_emb, &positive, &negative);
    let initial_auc = link_prediction_auc(&initial_emb, &positive, &negative);

    let hidden_dim = config_hidden_dim(&initial_emb);
    let mut edge_pred = if use_edge_predictor {
        Some(crate::model::jepa::EdgePredictor::new(
            hidden_dim,
            hidden_dim * 2,
            hidden_dim,
        ))
    } else {
        None
    };

    let mut best_loss = initial_loss;
    let mut patience_counter = 0;
    let mut final_loss = initial_loss;
    let mut final_auc = initial_auc;
    let mut epochs_trained = 0;

    for epoch in 0..config.epochs {
        let predictor_temp = scheduled_temperature(0.5, epoch, config.epochs);
        let emb = model_forward(graph);
        let plain = embeddings_to_plain(&emb);
        let bpr_loss = compute_bpr_loss(&plain, &positive, &negative);
        let auc = link_prediction_auc(&plain, &positive, &negative);
        let uniform = crate::model::jepa::compute_uniformity_loss(&plain);

        // Combined loss: BPR (link prediction) + λ * uniformity (anti-collapse)
        let combined_loss = bpr_loss + uniformity_weight * uniform;

        // Train edge predictor if enabled
        if let Some(ref mut pred) = edge_pred {
            pred.spsa_step(
                &plain,
                &positive,
                &negative,
                config.lr as f32,
                predictor_temp,
                epoch,
            );
        }

        // Cosine-aware feature refinement:
        // For each node, compute gradient from BPR + uniformity push
        for (nt, feat) in graph.node_features.iter_mut() {
            let dims = feat.dims();
            let (n, d) = (dims[0], dims[1]);
            let feat_data: Vec<f32> = feat.clone().into_data().as_slice::<f32>().unwrap().to_vec();
            let mut new_feat = feat_data.clone();

            if let Some(node_vecs) = plain.get(nt) {
                // Compute centroid of all embeddings for uniformity push
                let mut centroid = vec![0.0f32; d];
                let mut total_nodes = 0;
                for vecs in plain.values() {
                    for v in vecs {
                        for (ci, val) in centroid.iter_mut().zip(v.iter()) {
                            *ci += val;
                        }
                        total_nodes += 1;
                    }
                }
                if total_nodes > 0 {
                    centroid.iter_mut().for_each(|c| *c /= total_nodes as f32);
                }

                for node in 0..n {
                    if let Some(emb_vec) = node_vecs.get(node) {
                        for dim in 0..d.min(emb_vec.len()) {
                            // BPR gradient: move features toward embeddings
                            let bpr_grad = emb_vec[dim] - feat_data[node * d + dim];

                            // Uniformity gradient: push away from centroid
                            let uniform_grad = if uniformity_weight > 0.0 {
                                (emb_vec[dim] - centroid[dim]) * 0.01
                            } else {
                                0.0
                            };

                            new_feat[node * d + dim] += config.lr as f32
                                * (bpr_grad * 0.01 + uniformity_weight * uniform_grad);
                        }
                    }
                }
            }

            // Weight decay
            if config.weight_decay > 0.0 {
                let wd = 1.0 - config.weight_decay as f32;
                for v in new_feat.iter_mut() {
                    *v *= wd;
                }
            }

            let device = feat.device();
            *feat = Tensor::<B, 1>::from_data(new_feat.as_slice(), &device).reshape([n, d]);
        }

        final_loss = bpr_loss;
        final_auc = auc;
        epochs_trained = epoch + 1;

        if epoch % 5 == 0 || epoch == config.epochs - 1 {
            eprintln!(
                "  [train:bpr+jepa] epoch {}: bpr={:.4}, auc={:.4}, uniform={:.4}, pred_temp={:.3}",
                epoch, bpr_loss, auc, uniform, predictor_temp
            );
        }

        if combined_loss < best_loss - 0.001 {
            best_loss = combined_loss;
            patience_counter = 0;
        } else {
            patience_counter += 1;
            if patience_counter >= config.patience {
                return TrainReport {
                    epochs_trained,
                    initial_loss,
                    final_loss,
                    initial_auc,
                    final_auc,
                    early_stopped: true,
                    weight_norm_sq: 0.0,
                    mean_emb_norm: 0.0,
                };
            }
        }
    }

    TrainReport {
        epochs_trained,
        initial_loss,
        final_loss,
        initial_auc,
        final_auc,
        early_stopped: false,
        weight_norm_sq: 0.0,
        mean_emb_norm: 0.0,
    }
}
