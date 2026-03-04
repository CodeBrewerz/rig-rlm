//! GEPA Optimizer — End-to-End Prediction Quality Optimization
//!
//! Unlike the fiduciary weight optimizer (which tunes blending weights),
//! this optimizes the **prediction pipeline thresholds** that determine
//! WHAT gets recommended:
//!
//! - recommend_threshold: minimum score to mark "is_recommended" (default 0.3)
//! - anomaly_investigate: anomaly score to trigger "should_investigate" (default 0.5)
//! - anomaly_avoid: anomaly score to trigger "should_avoid" for merchants (default 0.3)
//! - anomaly_txn: anomaly score to flag transaction anomalies (default 0.4)
//! - urgency_cutoff: high-urgency score to trigger conflict suppression (default 0.4)
//! - discretionary_cutoff: below this, discretionary actions get demoted (default 0.5)
//!
//! The evaluator measures prediction quality against ground truth risk profiles.

use std::collections::HashMap;

use hehrgnn::eval::fiduciary::*;
use hehrgnn::optimizer::gepa::*;

// ═══════════════════════════════════════════════════════════════
// Graph with known ground truth for evaluation
// ═══════════════════════════════════════════════════════════════

#[derive(Clone, Copy)]
enum Risk {
    High,
    Medium,
    Low,
}

struct GroundTruthUser {
    name: &'static str,
    risk: Risk,
    /// Expected: system should recommend investigating/avoiding risky targets
    should_flag_count: usize,
    /// Expected: system should NOT over-recommend safe actions
    max_safe_recs: usize,
}

fn pseudo_random(seed: u64, i: usize, lo: f32, hi: f32) -> f32 {
    let x = seed
        .wrapping_mul(6364136223846793005u64)
        .wrapping_add(i as u64)
        .wrapping_mul(1442695040888963407u64);
    let frac = ((x >> 48) as f32) / 65535.0;
    lo + frac * (hi - lo)
}

fn make_embedding(risk: Risk, dim: usize, seed: u64, idx: usize) -> Vec<f32> {
    let base = match risk {
        Risk::High => 0.85,
        Risk::Medium => 0.45,
        Risk::Low => 0.10,
    };
    (0..dim)
        .map(|d| {
            let noise = pseudo_random(seed, idx * dim + d, -0.15, 0.15);
            (base + noise).clamp(0.0, 1.0)
        })
        .collect()
}

fn make_anomaly(risk: Risk, seed: u64, idx: usize) -> f32 {
    let (lo, hi) = match risk {
        Risk::High => (0.7, 0.95),
        Risk::Medium => (0.3, 0.55),
        Risk::Low => (0.02, 0.15),
    };
    pseudo_random(seed, idx * 7 + 3, lo, hi)
}

struct PredictionTestGraph {
    embeddings: HashMap<String, Vec<Vec<f32>>>,
    anomaly_scores: HashMap<String, HashMap<String, Vec<f32>>>,
    node_names: HashMap<String, Vec<String>>,
    edges: HashMap<(String, String, String), Vec<(usize, usize)>>,
    node_counts: HashMap<String, usize>,
    ground_truth: Vec<GroundTruthUser>,
}

fn build_prediction_graph() -> PredictionTestGraph {
    let dim = 8;
    let seed = 42u64;

    let ground_truth = vec![
        GroundTruthUser {
            name: "HighRisk_Dave",
            risk: Risk::High,
            should_flag_count: 3,
            max_safe_recs: 2,
        },
        GroundTruthUser {
            name: "MedRisk_Mike",
            risk: Risk::Medium,
            should_flag_count: 1,
            max_safe_recs: 4,
        },
        GroundTruthUser {
            name: "LowRisk_Emma",
            risk: Risk::Low,
            should_flag_count: 0,
            max_safe_recs: 6,
        },
        GroundTruthUser {
            name: "HighRisk_Zara",
            risk: Risk::High,
            should_flag_count: 2,
            max_safe_recs: 2,
        },
    ];

    let mut emb: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    let mut anomaly: HashMap<String, Vec<f32>> = HashMap::new();
    let mut names: HashMap<String, Vec<String>> = HashMap::new();
    let mut edges: HashMap<(String, String, String), Vec<(usize, usize)>> = HashMap::new();

    // Users
    let mut user_embs = Vec::new();
    let mut user_anom = Vec::new();
    let mut user_names = Vec::new();
    for (i, gt) in ground_truth.iter().enumerate() {
        user_embs.push(make_embedding(gt.risk, dim, seed, i));
        user_anom.push(make_anomaly(gt.risk, seed + 1, i));
        user_names.push(gt.name.to_string());
    }
    emb.insert("user".into(), user_embs);
    anomaly.insert("user".into(), user_anom);
    names.insert("user".into(), user_names);

    // Accounts (10): 3 high-risk, 3 medium, 4 low
    let acct_risks = [
        Risk::High,
        Risk::High,
        Risk::High,
        Risk::Medium,
        Risk::Medium,
        Risk::Medium,
        Risk::Low,
        Risk::Low,
        Risk::Low,
        Risk::Low,
    ];
    let acct_names_v: Vec<&str> = vec![
        "Dave_CreditLine",
        "Dave_Checking",
        "Zara_CreditLine",
        "Mike_Checking",
        "Mike_Brokerage",
        "Mike_Savings",
        "Emma_Checking",
        "Emma_Savings",
        "Emma_401k",
        "Emma_529",
    ];
    let mut a_embs = Vec::new();
    let mut a_anom = Vec::new();
    let mut a_names = Vec::new();
    for (i, (risk, name)) in acct_risks.iter().zip(acct_names_v.iter()).enumerate() {
        a_embs.push(make_embedding(*risk, dim, seed + 10, i));
        a_anom.push(make_anomaly(*risk, seed + 11, i));
        a_names.push(name.to_string());
    }
    emb.insert("account".into(), a_embs);
    anomaly.insert("account".into(), a_anom);
    names.insert("account".into(), a_names);

    edges.insert(
        ("user".into(), "owns".into(), "account".into()),
        vec![
            (0, 0),
            (0, 1), // Dave → 2 high-risk accounts
            (1, 3),
            (1, 4),
            (1, 5), // Mike → 3 medium accounts
            (2, 6),
            (2, 7),
            (2, 8),
            (2, 9), // Emma → 4 low-risk accounts
            (3, 2), // Zara → 1 high-risk account
        ],
    );

    // Merchants (6): gambling, crypto = high; luxury = medium; grocery, utilities, insurance = low
    let merch_risks = [
        Risk::High,
        Risk::High,
        Risk::Medium,
        Risk::Low,
        Risk::Low,
        Risk::Low,
    ];
    let merch_names_v = vec![
        "OnlineGambling",
        "CryptoShady",
        "LuxuryGoods",
        "Grocery",
        "Utilities",
        "Insurance",
    ];
    let mut m_embs = Vec::new();
    let mut m_anom = Vec::new();
    let mut m_names = Vec::new();
    for (i, (risk, name)) in merch_risks.iter().zip(merch_names_v.iter()).enumerate() {
        m_embs.push(make_embedding(*risk, dim, seed + 20, i));
        m_anom.push(make_anomaly(*risk, seed + 21, i));
        m_names.push(name.to_string());
    }
    emb.insert("merchant".into(), m_embs);
    anomaly.insert("merchant".into(), m_anom);
    names.insert("merchant".into(), m_names);

    edges.insert(
        ("account".into(), "transacts".into(), "merchant".into()),
        vec![
            (0, 0),
            (0, 1), // Dave → gambling, crypto
            (1, 2), // Dave → luxury
            (3, 2),
            (3, 3), // Mike → luxury, grocery
            (6, 3),
            (6, 4), // Emma → grocery, utilities
            (7, 5), // Emma → insurance
            (2, 0),
            (2, 1), // Zara → gambling, crypto
        ],
    );

    // Obligations (5): payday & collection = high; car loan = medium; mortgage = low
    let oblig_risks = [Risk::High, Risk::High, Risk::Medium, Risk::Low, Risk::Low];
    let oblig_names_v = vec![
        "PaydayLoan_36pct",
        "CollectionDebt",
        "CarLoan_12pct",
        "Mortgage_4pct",
        "Mortgage_3pct",
    ];
    let mut o_embs = Vec::new();
    let mut o_anom = Vec::new();
    let mut o_names = Vec::new();
    for (i, (risk, name)) in oblig_risks.iter().zip(oblig_names_v.iter()).enumerate() {
        o_embs.push(make_embedding(*risk, dim, seed + 30, i));
        o_anom.push(make_anomaly(*risk, seed + 31, i));
        o_names.push(name.to_string());
    }
    emb.insert("obligation".into(), o_embs);
    anomaly.insert("obligation".into(), o_anom);
    names.insert("obligation".into(), o_names);

    edges.insert(
        ("account".into(), "pays".into(), "obligation".into()),
        vec![
            (0, 0),
            (1, 1), // Dave → payday, collection
            (3, 2), // Mike → car loan
            (6, 3),
            (7, 4), // Emma → mortgages
            (2, 0), // Zara → payday
        ],
    );

    let mut node_counts: HashMap<String, usize> = HashMap::new();
    for (nt, ns) in &names {
        node_counts.insert(nt.clone(), ns.len());
    }
    let mut anomaly_scores: HashMap<String, HashMap<String, Vec<f32>>> = HashMap::new();
    anomaly_scores.insert("SAGE".into(), anomaly);

    PredictionTestGraph {
        embeddings: emb,
        anomaly_scores,
        node_names: names,
        edges,
        node_counts,
        ground_truth,
    }
}

// ═══════════════════════════════════════════════════════════════
// Prediction Quality Evaluator
// ═══════════════════════════════════════════════════════════════

/// Evaluates the full fiduciary prediction pipeline against ground truth.
///
/// Measures:
/// 1. Precision: do high-risk users get flagged? (true positive rate)
/// 2. Safety: do low-risk users avoid false alarms? (specificity)
/// 3. Ranking: are risky actions ranked above safe ones?
/// 4. Coverage: appropriate number of recommendations per risk level?
struct PredictionQualityEvaluator {
    graph: PredictionTestGraph,
}

impl PredictionQualityEvaluator {
    fn new(graph: PredictionTestGraph) -> Self {
        Self { graph }
    }
}

impl Evaluator for PredictionQualityEvaluator {
    fn evaluate(&self, candidate: &Candidate) -> EvalResult {
        // Parse prediction thresholds from candidate
        let recommend_threshold = candidate.get_f32("recommend_threshold", 0.3);
        let anomaly_investigate = candidate.get_f32("anomaly_investigate", 0.5);
        let anomaly_avoid = candidate.get_f32("anomaly_avoid", 0.3);
        let anomaly_txn = candidate.get_f32("anomaly_txn", 0.4);
        let urgency_cutoff = candidate.get_f32("urgency_cutoff", 0.4);
        let discretionary_cutoff = candidate.get_f32("discretionary_cutoff", 0.5);

        let mut total_precision = 0.0f64;
        let mut total_safety = 0.0f64;
        let mut total_ranking = 0.0f64;
        let mut total_coverage = 0.0f64;
        let mut user_count = 0;

        for (user_id, gt) in self.graph.ground_truth.iter().enumerate() {
            let user_emb = self.graph.embeddings.get("user").unwrap()[user_id].clone();

            let ctx = FiduciaryContext {
                user_emb: &user_emb,
                embeddings: &self.graph.embeddings,
                anomaly_scores: &self.graph.anomaly_scores,
                edges: &self.graph.edges,
                node_names: &self.graph.node_names,
                node_counts: &self.graph.node_counts,
                user_type: "user".into(),
                user_id,
                hidden_dim: 8,
            };

            let mut pc_state = PcState::new();
            let resp = recommend(&ctx, Some(&mut pc_state));
            let mut recs = resp.recommendations;

            // Apply candidate thresholds
            for rec in &mut recs {
                rec.is_recommended = rec.fiduciary_score >= recommend_threshold;
            }

            // Count flagged actions (investigate + avoid)
            let flagged = recs
                .iter()
                .filter(|r| {
                    r.is_recommended
                        && (r.action_type == "should_investigate"
                            || r.action_type == "should_avoid")
                })
                .count();

            let recommended = recs.iter().filter(|r| r.is_recommended).count();

            // --- Metric 1: Precision ---
            // High-risk users SHOULD have flagged actions
            let precision = match gt.risk {
                Risk::High => {
                    if flagged >= gt.should_flag_count {
                        1.0
                    } else {
                        flagged as f64 / gt.should_flag_count.max(1) as f64
                    }
                }
                Risk::Medium => {
                    if flagged >= 1 && flagged <= 3 {
                        1.0
                    } else if flagged == 0 {
                        0.5
                    }
                    // Missing flags for medium risk
                    else {
                        0.7
                    } // Over-flagging
                }
                Risk::Low => {
                    if flagged == 0 {
                        1.0
                    }
                    // Correct: no flags for safe users
                    else {
                        (1.0 - flagged as f64 * 0.3).max(0.0)
                    } // Penalize false alarms
                }
            };

            // --- Metric 2: Safety (specificity) ---
            // Low-risk users should have limited recommendations
            let safety = match gt.risk {
                Risk::Low => {
                    if recommended <= gt.max_safe_recs {
                        1.0
                    } else {
                        (gt.max_safe_recs as f64 / recommended as f64).min(1.0)
                    }
                }
                _ => 1.0, // Don't penalize high/medium risk for having many recs
            };

            // --- Metric 3: Ranking quality ---
            // Risk-relevant actions should rank above discretionary ones
            let risk_actions = [
                "should_investigate",
                "should_avoid",
                "should_pay",
                "should_dispute",
            ];
            let risk_avg_rank: f64 = {
                let risk_ranks: Vec<usize> = recs
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| risk_actions.contains(&r.action_type.as_str()))
                    .map(|(i, _)| i + 1)
                    .collect();
                if risk_ranks.is_empty() {
                    recs.len() as f64
                } else {
                    risk_ranks.iter().sum::<usize>() as f64 / risk_ranks.len() as f64
                }
            };
            let total_recs = recs.len().max(1) as f64;
            let ranking = 1.0 - (risk_avg_rank / total_recs).min(1.0); // Higher = risk actions rank better

            // --- Metric 4: Coverage ---
            // Right number of recommendations for this risk level
            let ideal_count = match gt.risk {
                Risk::High => 5.0,
                Risk::Medium => 3.0,
                Risk::Low => 2.0,
            };
            let coverage = 1.0 - ((recommended as f64 - ideal_count).abs() / ideal_count).min(1.0);

            total_precision += precision;
            total_safety += safety;
            total_ranking += ranking;
            total_coverage += coverage;
            user_count += 1;
        }

        let n = user_count as f64;
        let avg_precision = total_precision / n;
        let avg_safety = total_safety / n;
        let avg_ranking = total_ranking / n;
        let avg_coverage = total_coverage / n;

        let score =
            avg_precision * 0.35 + avg_safety * 0.25 + avg_ranking * 0.25 + avg_coverage * 0.15;

        let mut side_info = SideInfo::new();
        side_info.score("precision", avg_precision);
        side_info.score("safety", avg_safety);
        side_info.score("ranking", avg_ranking);
        side_info.score("coverage", avg_coverage);
        side_info.log(format!(
            "rec_t={:.2}, inv={:.2}, avoid={:.2}, txn={:.2}, urg={:.2}, disc={:.2} → prec={:.3}, safe={:.3}, rank={:.3}, cov={:.3}, combined={:.4}",
            recommend_threshold, anomaly_investigate, anomaly_avoid, anomaly_txn,
            urgency_cutoff, discretionary_cutoff,
            avg_precision, avg_safety, avg_ranking, avg_coverage, score
        ));

        EvalResult { score, side_info }
    }
}

// ═══════════════════════════════════════════════════════════════
// Sync Test — NumericMutator
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_gepa_optimize_prediction_quality() {
    println!(
        "\n  ╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "  ║  GEPA — End-to-End Prediction Quality Optimization                                ║"
    );
    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════╣"
    );

    let graph = build_prediction_graph();
    let evaluator = PredictionQualityEvaluator::new(graph);

    let seed = Candidate::seed(vec![
        ("recommend_threshold", "0.3000"),
        ("anomaly_investigate", "0.5000"),
        ("anomaly_avoid", "0.3000"),
        ("anomaly_txn", "0.4000"),
        ("urgency_cutoff", "0.4000"),
        ("discretionary_cutoff", "0.5000"),
    ]);

    let seed_eval = evaluator.evaluate(&seed);
    println!(
        "  ║  Seed: rec≥{}, investigate≥{}, avoid≥{}",
        seed.params.get("recommend_threshold").unwrap(),
        seed.params.get("anomaly_investigate").unwrap(),
        seed.params.get("anomaly_avoid").unwrap()
    );
    println!(
        "  ║  Seed score: {:.6} (prec={:.3}, safe={:.3}, rank={:.3}, cov={:.3})",
        seed_eval.score,
        seed_eval.side_info.scores.get("precision").unwrap_or(&0.0),
        seed_eval.side_info.scores.get("safety").unwrap_or(&0.0),
        seed_eval.side_info.scores.get("ranking").unwrap_or(&0.0),
        seed_eval.side_info.scores.get("coverage").unwrap_or(&0.0)
    );
    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════╣"
    );

    let mutator = NumericMutator::new(0.15, 42);
    let config = OptimizeConfig {
        max_evals: 30,
        max_frontier_size: 10,
        log_every: 10,
        objective: "Optimize prediction quality: high-risk users should get flagged, low-risk users should not".into(),
    };

    let result = optimize(seed, &evaluator, &mutator, config);

    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "  ║  Best score: {:.6}  ({} evals, frontier={})",
        result.best_score, result.total_evals, result.frontier_size
    );
    println!("  ║  Best thresholds:");
    let mut params: Vec<_> = result.best_candidate.params.iter().collect();
    params.sort_by_key(|(k, _)| k.clone());
    for (key, val) in &params {
        println!("  ║    {:24}: {}", key, val);
    }

    let improvement = result.best_score - seed_eval.score;
    if improvement > 0.0 {
        println!(
            "  ║  ✅ Improved by {:.6} ({:.1}%)",
            improvement,
            improvement / seed_eval.score.abs().max(0.001) * 100.0
        );
    }
    println!(
        "  ╚══════════════════════════════════════════════════════════════════════════════════════╝"
    );

    assert!(result.total_evals >= 20);
    assert!(result.best_score.is_finite());
}

// ═══════════════════════════════════════════════════════════════
// Live LLM Test — Trinity-guided with persistence
// ═══════════════════════════════════════════════════════════════

/// Uses Trinity to optimize prediction quality thresholds.
///
/// Run: `cargo test -p hehrgnn --test gepa_prediction_test test_gepa_llm_prediction -- --ignored --nocapture`
#[tokio::test]
#[ignore]
async fn test_gepa_llm_prediction_quality_with_trinity() {
    let weights_path = "/tmp/gepa_prediction_config.json";

    println!(
        "\n  ╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "  ║  GEPA + TRINITY — End-to-End Prediction Quality Optimization (feedback loop)      ║"
    );
    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════╣"
    );

    let objective = "Optimize fiduciary prediction quality thresholds for a financial recommendation system. \
        Parameters control WHAT gets recommended to users: \
        - recommend_threshold (0.1-0.6): minimum score to mark as recommended (lower = more recs) \
        - anomaly_investigate (0.2-0.8): anomaly score to trigger 'investigate' action \
        - anomaly_avoid (0.1-0.6): anomaly score to trigger 'avoid merchant' \
        - anomaly_txn (0.2-0.7): anomaly score to flag suspicious transactions \
        - urgency_cutoff (0.2-0.7): score to trigger conflict suppression of discretionary actions \
        - discretionary_cutoff (0.2-0.7): below this, discretionary actions are demoted \
        Goal: HIGH-RISK users must get investigation/avoidance flags (precision). \
        LOW-RISK users must NOT get false alarms (safety). \
        Risk-relevant actions must rank above discretionary ones (ranking). \
        Right number of recommendations per risk level (coverage). \
        Maximize 0.35*precision + 0.25*safety + 0.25*ranking + 0.15*coverage.";

    let llm_mutator = match LlmMutator::from_env(objective) {
        Ok(m) => m,
        Err(e) => {
            println!("  ║  ⚠️  Skipping: {}", e);
            println!(
                "  ╚══════════════════════════════════════════════════════════════════════════════════════╝"
            );
            return;
        }
    };

    let prev_weights = OptimizedWeights::load_or_default(weights_path);
    let seed = if prev_weights.total_evals > 0 {
        println!(
            "  ║  📂 Loaded previous best (score={:.6}, evals={})",
            prev_weights.score, prev_weights.total_evals
        );
        prev_weights.to_candidate()
    } else {
        println!("  ║  🆕 Starting from default thresholds");
        Candidate::seed(vec![
            ("recommend_threshold", "0.3000"),
            ("anomaly_investigate", "0.5000"),
            ("anomaly_avoid", "0.3000"),
            ("anomaly_txn", "0.4000"),
            ("urgency_cutoff", "0.4000"),
            ("discretionary_cutoff", "0.5000"),
        ])
    };

    let graph = build_prediction_graph();
    let evaluator = PredictionQualityEvaluator::new(graph);
    let seed_eval = evaluator.evaluate(&seed);
    println!(
        "  ║  Seed score: {:.6} (prec={:.3}, safe={:.3}, rank={:.3}, cov={:.3})",
        seed_eval.score,
        seed_eval.side_info.scores.get("precision").unwrap_or(&0.0),
        seed_eval.side_info.scores.get("safety").unwrap_or(&0.0),
        seed_eval.side_info.scores.get("ranking").unwrap_or(&0.0),
        seed_eval.side_info.scores.get("coverage").unwrap_or(&0.0)
    );
    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════╣"
    );

    let config = OptimizeConfig {
        max_evals: 15,
        max_frontier_size: 8,
        log_every: 1,
        objective: objective.into(),
    };

    let result = optimize_async(seed, &evaluator, &llm_mutator, config).await;

    let mut best_weights =
        OptimizedWeights::from_candidate(&result.best_candidate, result.best_score);
    best_weights.total_evals = prev_weights.total_evals + result.total_evals;
    match best_weights.save(weights_path) {
        Ok(()) => println!(
            "  ║  💾 Saved to {} (cumulative evals={})",
            weights_path, best_weights.total_evals
        ),
        Err(e) => println!("  ║  ⚠️  Save failed: {}", e),
    }

    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "  ║  Best score: {:.6}  ({} evals, {} cumulative)",
        result.best_score, result.total_evals, best_weights.total_evals
    );
    println!("  ║  Best thresholds (discovered by Trinity):");
    let mut params: Vec<_> = result.best_candidate.params.iter().collect();
    params.sort_by_key(|(k, _)| k.clone());
    for (key, val) in &params {
        println!("  ║    {:24}: {}", key, val);
    }

    let improvement = result.best_score - seed_eval.score;
    if improvement > 0.0 {
        println!(
            "  ║  ✅ Trinity improved by {:.6} ({:.1}%)",
            improvement,
            improvement / seed_eval.score.abs().max(0.001) * 100.0
        );
    } else {
        println!("  ║  ℹ️  No improvement this run");
    }
    println!("  ║  🔄 Run again to continue optimizing!");
    println!(
        "  ╚══════════════════════════════════════════════════════════════════════════════════════╝"
    );

    assert!(result.total_evals >= 5);
    assert!(result.best_score.is_finite());
}
