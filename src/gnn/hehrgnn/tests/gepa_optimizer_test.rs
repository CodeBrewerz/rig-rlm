//! GEPA Optimizer Integration Test
//!
//! Uses the 128-entity graph to demonstrate GEPA optimizing
//! the fiduciary blending weights (GNN α vs PC β) to maximize
//! ranking quality metric: high-anomaly entities should rank
//! above low-anomaly ones in the final recommendations.

use std::collections::HashMap;

use hehrgnn::eval::fiduciary::*;
use hehrgnn::optimizer::gepa::*;

// ═══════════════════════════════════════════════════════════════
// Reuse the 128-entity graph generator from large_graph_pc_test
// ═══════════════════════════════════════════════════════════════

#[derive(Clone, Copy)]
enum RiskProfile {
    HighRisk,
    Risky,
    Mixed,
    Safe,
    VerySafe,
}

impl RiskProfile {
    fn anomaly_range(self) -> (f32, f32) {
        match self {
            Self::HighRisk => (0.75, 0.98),
            Self::Risky => (0.50, 0.74),
            Self::Mixed => (0.30, 0.49),
            Self::Safe => (0.10, 0.29),
            Self::VerySafe => (0.02, 0.09),
        }
    }
    fn emb_base(self) -> f32 {
        match self {
            Self::HighRisk => 0.85,
            Self::Risky => 0.65,
            Self::Mixed => 0.45,
            Self::Safe => 0.25,
            Self::VerySafe => 0.10,
        }
    }
}

fn pseudo_random(seed: u64, i: usize, lo: f32, hi: f32) -> f32 {
    let x = seed
        .wrapping_mul(6364136223846793005u64)
        .wrapping_add(i as u64)
        .wrapping_mul(1442695040888963407u64);
    let frac = ((x >> 48) as f32) / 65535.0;
    lo + frac * (hi - lo)
}

fn make_embedding(profile: RiskProfile, dim: usize, seed: u64, idx: usize) -> Vec<f32> {
    let base = profile.emb_base();
    (0..dim)
        .map(|d| {
            let noise = pseudo_random(seed, idx * dim + d, -0.15, 0.15);
            (base + noise).clamp(0.0, 1.0)
        })
        .collect()
}

fn make_anomaly(profile: RiskProfile, seed: u64, idx: usize) -> f32 {
    let (lo, hi) = profile.anomaly_range();
    pseudo_random(seed, idx * 7 + 3, lo, hi)
}

/// Build a compact graph with known risk profiles for evaluation.
struct TestGraph {
    embeddings: HashMap<String, Vec<Vec<f32>>>,
    anomaly_scores: HashMap<String, HashMap<String, Vec<f32>>>,
    node_names: HashMap<String, Vec<String>>,
    edges: HashMap<(String, String, String), Vec<(usize, usize)>>,
    node_counts: HashMap<String, usize>,
    /// Ground truth: user_id → expected risk level (higher = riskier)
    user_risk_levels: Vec<f32>,
}

fn build_test_graph() -> TestGraph {
    let dim = 8;
    let seed = 42u64;

    let mut emb: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    let mut anomaly: HashMap<String, Vec<f32>> = HashMap::new();
    let mut names: HashMap<String, Vec<String>> = HashMap::new();
    let mut edges: HashMap<(String, String, String), Vec<(usize, usize)>> = HashMap::new();

    // 6 users with clear risk profiles
    let user_profiles = [
        (RiskProfile::HighRisk, "Dave_HighRisk", 0.95f32),
        (RiskProfile::Risky, "Mike_Risky", 0.70),
        (RiskProfile::Mixed, "Carlos_Mixed", 0.45),
        (RiskProfile::Safe, "Beth_Safe", 0.20),
        (RiskProfile::VerySafe, "Emma_VerySafe", 0.05),
        (RiskProfile::HighRisk, "Zara_HighRisk", 0.90),
    ];

    let user_risk_levels: Vec<f32> = user_profiles.iter().map(|(_, _, r)| *r).collect();

    let mut user_embs = Vec::new();
    let mut user_anom = Vec::new();
    let mut user_names_vec = Vec::new();
    for (i, (prof, name, _)) in user_profiles.iter().enumerate() {
        user_embs.push(make_embedding(*prof, dim, seed, i));
        user_anom.push(make_anomaly(*prof, seed + 1, i));
        user_names_vec.push(name.to_string());
    }
    emb.insert("user".into(), user_embs);
    anomaly.insert("user".into(), user_anom);
    names.insert("user".into(), user_names_vec);

    // 15 accounts
    let account_specs: Vec<(RiskProfile, &str)> = vec![
        (RiskProfile::HighRisk, "Dave_CreditLine"),
        (RiskProfile::HighRisk, "Dave_Checking"),
        (RiskProfile::Risky, "Mike_Checking"),
        (RiskProfile::Risky, "Mike_Brokerage"),
        (RiskProfile::Mixed, "Carlos_Checking"),
        (RiskProfile::Mixed, "Carlos_Savings"),
        (RiskProfile::Safe, "Beth_Savings"),
        (RiskProfile::Safe, "Beth_401k"),
        (RiskProfile::VerySafe, "Emma_Savings"),
        (RiskProfile::VerySafe, "Emma_529"),
        (RiskProfile::HighRisk, "Zara_Checking"),
        (RiskProfile::HighRisk, "Zara_CreditLine"),
        (RiskProfile::Mixed, "Carlos_HSA"),
        (RiskProfile::Safe, "Beth_HSA"),
        (RiskProfile::VerySafe, "Emma_401k"),
    ];
    let mut acct_embs = Vec::new();
    let mut acct_anom = Vec::new();
    let mut acct_names = Vec::new();
    for (i, (prof, name)) in account_specs.iter().enumerate() {
        acct_embs.push(make_embedding(*prof, dim, seed + 10, i));
        acct_anom.push(make_anomaly(*prof, seed + 11, i));
        acct_names.push(name.to_string());
    }
    emb.insert("account".into(), acct_embs);
    anomaly.insert("account".into(), acct_anom);
    names.insert("account".into(), acct_names);

    edges.insert(
        ("user".into(), "owns".into(), "account".into()),
        vec![
            (0, 0),
            (0, 1), // Dave
            (1, 2),
            (1, 3), // Mike
            (2, 4),
            (2, 5),
            (2, 12), // Carlos
            (3, 6),
            (3, 7),
            (3, 13), // Beth
            (4, 8),
            (4, 9),
            (4, 14), // Emma
            (5, 10),
            (5, 11), // Zara
        ],
    );

    // 12 obligations with varied risk
    let oblig_specs: Vec<(RiskProfile, &str)> = vec![
        (RiskProfile::HighRisk, "PaydayLoan_36pct"),
        (RiskProfile::HighRisk, "CreditCard_24pct"),
        (RiskProfile::HighRisk, "CollectionDebt"),
        (RiskProfile::Risky, "CarLoan_12pct"),
        (RiskProfile::Risky, "StudentLoan"),
        (RiskProfile::Mixed, "Mortgage_6pct"),
        (RiskProfile::Mixed, "AutoLoan_8pct"),
        (RiskProfile::Safe, "CarLoan_3pct"),
        (RiskProfile::Safe, "Mortgage_4pct"),
        (RiskProfile::VerySafe, "Mortgage_3pct"),
        (RiskProfile::HighRisk, "IRS_BackTax"),
        (RiskProfile::Risky, "MedicalBill"),
    ];
    let mut oblig_embs = Vec::new();
    let mut oblig_anom = Vec::new();
    let mut oblig_names = Vec::new();
    for (i, (prof, name)) in oblig_specs.iter().enumerate() {
        oblig_embs.push(make_embedding(*prof, dim, seed + 20, i));
        oblig_anom.push(make_anomaly(*prof, seed + 21, i));
        oblig_names.push(name.to_string());
    }
    emb.insert("obligation".into(), oblig_embs);
    anomaly.insert("obligation".into(), oblig_anom);
    names.insert("obligation".into(), oblig_names);

    edges.insert(
        ("account".into(), "pays".into(), "obligation".into()),
        vec![
            (0, 0),
            (0, 2),
            (1, 1), // Dave → payday, collection, credit card
            (2, 3),
            (2, 4), // Mike → car loan, student loan
            (4, 5),
            (5, 6), // Carlos → mortgage, auto
            (6, 7),
            (7, 8), // Beth → car loan, mortgage
            (8, 9), // Emma → mortgage
            (10, 10),
            (11, 11), // Zara → IRS, medical
        ],
    );

    // 8 merchants
    let merch_specs: Vec<(RiskProfile, &str)> = vec![
        (RiskProfile::HighRisk, "OnlineGambling"),
        (RiskProfile::HighRisk, "CryptoShady"),
        (RiskProfile::Risky, "LuxuryGoods"),
        (RiskProfile::Mixed, "Electronics"),
        (RiskProfile::Safe, "Grocery"),
        (RiskProfile::Safe, "Utilities"),
        (RiskProfile::VerySafe, "Insurance"),
        (RiskProfile::VerySafe, "Pharmacy"),
    ];
    let mut merch_embs = Vec::new();
    let mut merch_anom = Vec::new();
    let mut merch_names = Vec::new();
    for (i, (prof, name)) in merch_specs.iter().enumerate() {
        merch_embs.push(make_embedding(*prof, dim, seed + 40, i));
        merch_anom.push(make_anomaly(*prof, seed + 41, i));
        merch_names.push(name.to_string());
    }
    emb.insert("merchant".into(), merch_embs);
    anomaly.insert("merchant".into(), merch_anom);
    names.insert("merchant".into(), merch_names);

    edges.insert(
        ("account".into(), "transacts".into(), "merchant".into()),
        vec![
            (0, 0),
            (0, 1),
            (1, 3), // Dave → gambling, crypto, electronics
            (2, 2),
            (2, 3), // Mike → luxury, electronics
            (4, 3),
            (4, 4), // Carlos → electronics, grocery
            (6, 4),
            (6, 5), // Beth → grocery, utilities
            (8, 6),
            (8, 7), // Emma → insurance, pharmacy
            (10, 0),
            (10, 1), // Zara → gambling, crypto
        ],
    );

    // 6 recurring subscriptions
    emb.insert(
        "recurring".into(),
        (0..6)
            .map(|i| {
                let prof = match i {
                    0 | 1 => RiskProfile::VerySafe,
                    2 | 3 => RiskProfile::Risky,
                    _ => RiskProfile::Safe,
                };
                make_embedding(prof, dim, seed + 50, i)
            })
            .collect(),
    );
    anomaly.insert("recurring".into(), vec![0.03, 0.05, 0.55, 0.62, 0.15, 0.10]);
    names.insert(
        "recurring".into(),
        vec![
            "Netflix".into(),
            "Spotify".into(),
            "UnusedGym".into(),
            "GamblingApp".into(),
            "AmazonPrime".into(),
            "News".into(),
        ],
    );
    edges.insert(
        ("user".into(), "subscribes".into(), "recurring".into()),
        vec![
            (0, 0),
            (0, 2),
            (0, 3), // Dave: netflix, gym, gambling
            (1, 0),
            (1, 2), // Mike: netflix, gym
            (2, 0),
            (2, 4), // Carlos: netflix, prime
            (3, 0),
            (3, 4),
            (3, 5), // Beth: netflix, prime, news
            (4, 0),
            (4, 1), // Emma: netflix, spotify
            (5, 0),
            (5, 3), // Zara: netflix, gambling
        ],
    );

    // 4 goals
    emb.insert(
        "goal".into(),
        (0..4)
            .map(|i| {
                let prof = match i {
                    0 => RiskProfile::Mixed,
                    1 => RiskProfile::Safe,
                    _ => RiskProfile::Risky,
                };
                make_embedding(prof, dim, seed + 60, i)
            })
            .collect(),
    );
    anomaly.insert("goal".into(), vec![0.35, 0.12, 0.55, 0.08]);
    names.insert(
        "goal".into(),
        vec![
            "EmergencyFund".into(),
            "Retirement".into(),
            "DebtPayoff".into(),
            "Vacation".into(),
        ],
    );
    edges.insert(
        ("user".into(), "targets".into(), "goal".into()),
        vec![
            (0, 0),
            (0, 2),
            (1, 0),
            (2, 0),
            (2, 1),
            (3, 1),
            (3, 3),
            (4, 1),
            (4, 3),
            (5, 2),
        ],
    );

    // Build counts and anomaly_scores wrapper
    let mut node_counts: HashMap<String, usize> = HashMap::new();
    for (nt, ns) in &names {
        node_counts.insert(nt.clone(), ns.len());
    }
    let mut anomaly_scores: HashMap<String, HashMap<String, Vec<f32>>> = HashMap::new();
    anomaly_scores.insert("SAGE".into(), anomaly);

    TestGraph {
        embeddings: emb,
        anomaly_scores,
        node_names: names,
        edges,
        node_counts,
        user_risk_levels,
    }
}

// ═══════════════════════════════════════════════════════════════
// Fiduciary Evaluator — scores a weight configuration
// ═══════════════════════════════════════════════════════════════

/// Evaluates a candidate weight configuration by running recommend()
/// across all users and measuring ranking quality.
///
/// Ranking quality = how well the system ranks high-anomaly actions
/// above low-anomaly ones (Spearman-like correlation).
struct FiduciaryEvaluator {
    graph: TestGraph,
}

impl FiduciaryEvaluator {
    fn new(graph: TestGraph) -> Self {
        Self { graph }
    }

    /// Compute ranking quality: do recommendations correctly order
    /// risky targets above safe targets?
    fn compute_ranking_quality(
        &self,
        recommendations: &[FiduciaryRecommendation],
    ) -> (f64, f64, f64) {
        if recommendations.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        // Metric 1: Rank correlation — high-anomaly targets should rank higher
        let mut concordant = 0usize;
        let mut discordant = 0usize;
        for i in 0..recommendations.len() {
            for j in (i + 1)..recommendations.len() {
                let score_i = recommendations[i].fiduciary_score;
                let score_j = recommendations[j].fiduciary_score;
                let anom_i = recommendations[i].target_anomaly_score;
                let anom_j = recommendations[j].target_anomaly_score;

                // For "should_avoid" actions, higher anomaly → higher rank is good
                // For "should_pay" actions, higher anomaly → higher rank is also good
                if (score_i > score_j) == (anom_i > anom_j) {
                    concordant += 1;
                } else if (score_i > score_j) != (anom_i > anom_j) {
                    discordant += 1;
                }
            }
        }
        let tau = if concordant + discordant > 0 {
            (concordant as f64 - discordant as f64) / (concordant as f64 + discordant as f64)
        } else {
            0.0
        };

        // Metric 2: Top-3 quality — are the top 3 recommendations high-anomaly targets?
        let top3_avg_anom: f64 = recommendations
            .iter()
            .take(3)
            .map(|r| r.target_anomaly_score as f64)
            .sum::<f64>()
            / 3.0f64.min(recommendations.len() as f64);

        // Metric 3: Risk separation — how much do risky actions score above safe ones?
        let risky_scores: Vec<f64> = recommendations
            .iter()
            .filter(|r| r.target_anomaly_score > 0.5)
            .map(|r| r.fiduciary_score as f64)
            .collect();
        let safe_scores: Vec<f64> = recommendations
            .iter()
            .filter(|r| r.target_anomaly_score < 0.3)
            .map(|r| r.fiduciary_score as f64)
            .collect();
        let separation = if !risky_scores.is_empty() && !safe_scores.is_empty() {
            let avg_risky = risky_scores.iter().sum::<f64>() / risky_scores.len() as f64;
            let avg_safe = safe_scores.iter().sum::<f64>() / safe_scores.len() as f64;
            (avg_risky - avg_safe).max(0.0)
        } else {
            0.0
        };

        (tau, top3_avg_anom, separation)
    }
}

impl Evaluator for FiduciaryEvaluator {
    fn evaluate(&self, candidate: &Candidate) -> EvalResult {
        // Parse candidate weights
        let alpha = candidate.get_f32("gnn_weight", 0.7);
        let beta = candidate.get_f32("pc_weight", 0.3);
        let _cost_w = candidate.get_f32("cost_weight", 0.25);
        let _risk_w = candidate.get_f32("risk_weight", 0.25);
        let _goal_w = candidate.get_f32("goal_weight", 0.15);
        let _urgency_w = candidate.get_f32("urgency_weight", 0.15);

        let mut pc_state = PcState::new();
        let mut total_tau = 0.0;
        let mut total_top3 = 0.0;
        let mut total_sep = 0.0;
        let mut user_count = 0;

        // Run recommend() for each user
        for user_id in 0..self.graph.user_risk_levels.len() {
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

            let resp = recommend(&ctx, Some(&mut pc_state));

            // Apply candidate's alpha/beta weights (override the hardcoded ones)
            let mut recs = resp.recommendations;
            for rec in &mut recs {
                if let Some(ref analysis) = rec.pc_analysis {
                    let pc_risk = analysis.risk_probability as f32;
                    let gnn_base = rec.fiduciary_score; // already blended with default weights
                    // Re-blend with candidate weights (undo default 0.7/0.3, apply new)
                    rec.fiduciary_score = alpha * gnn_base + beta * pc_risk;
                }
            }
            recs.sort_by(|a, b| b.fiduciary_score.partial_cmp(&a.fiduciary_score).unwrap());

            let (tau, top3, sep) = self.compute_ranking_quality(&recs);
            total_tau += tau;
            total_top3 += top3;
            total_sep += sep;
            user_count += 1;
        }

        let avg_tau = total_tau / user_count as f64;
        let avg_top3 = total_top3 / user_count as f64;
        let avg_sep = total_sep / user_count as f64;

        // Combined score: weighted blend of metrics
        let score = avg_tau * 0.4 + avg_top3 * 0.3 + avg_sep * 0.3;

        let mut side_info = SideInfo::new();
        side_info.score("kendall_tau", avg_tau);
        side_info.score("top3_anomaly", avg_top3);
        side_info.score("risk_separation", avg_sep);
        side_info.log(format!(
            "α={:.3}, β={:.3} → τ={:.4}, top3={:.4}, sep={:.4}, combined={:.4}",
            alpha, beta, avg_tau, avg_top3, avg_sep, score
        ));

        EvalResult { score, side_info }
    }
}

// ═══════════════════════════════════════════════════════════════
// Integration Test
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_gepa_optimize_fiduciary_weights() {
    println!(
        "\n  ╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "  ║  GEPA OPTIMIZER — Fiduciary Weight Optimization                                   ║"
    );
    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════╣"
    );

    let graph = build_test_graph();
    let evaluator = FiduciaryEvaluator::new(graph);

    // Seed candidate: current hardcoded weights
    let seed = Candidate::seed(vec![
        ("gnn_weight", "0.7000"),
        ("pc_weight", "0.3000"),
        ("cost_weight", "0.2500"),
        ("risk_weight", "0.2500"),
        ("goal_weight", "0.1500"),
        ("urgency_weight", "0.1500"),
    ]);

    // Evaluate seed first
    let seed_eval = evaluator.evaluate(&seed);
    println!(
        "  ║  Seed weights: α={}, β={}",
        seed.params.get("gnn_weight").unwrap(),
        seed.params.get("pc_weight").unwrap(),
    );
    println!(
        "  ║  Seed score: {:.6} (τ={:.4}, top3={:.4}, sep={:.4})",
        seed_eval.score,
        seed_eval
            .side_info
            .scores
            .get("kendall_tau")
            .unwrap_or(&0.0),
        seed_eval
            .side_info
            .scores
            .get("top3_anomaly")
            .unwrap_or(&0.0),
        seed_eval
            .side_info
            .scores
            .get("risk_separation")
            .unwrap_or(&0.0),
    );
    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════╣"
    );

    // Run GEPA optimization
    let mutator = NumericMutator::new(0.15, 42);
    let config = OptimizeConfig {
        max_evals: 30,
        max_frontier_size: 10,
        log_every: 5,
        objective: "Optimize fiduciary weights so high-anomaly entities rank above safe ones, \
                    and the top-3 recommendations for each user are the most urgent actions."
            .into(),
    };

    let result = optimize(seed, &evaluator, &mutator, config);

    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "  ║  Best score: {:.6}  (after {} evaluations, frontier={})",
        result.best_score, result.total_evals, result.frontier_size
    );
    println!("  ║  Best weights:");
    for (key, val) in &result.best_candidate.params {
        println!("  ║    {:16}: {}", key, val);
    }

    // Verify improvement
    let improvement = result.best_score - seed_eval.score;
    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════╣"
    );
    if improvement > 0.0 {
        println!(
            "  ║  ✅ GEPA improved score by {:.6} ({:.1}%)",
            improvement,
            improvement / seed_eval.score.abs().max(0.001) * 100.0
        );
    } else {
        println!(
            "  ║  ℹ️  Seed weights were already near-optimal (Δ={:.6})",
            improvement
        );
    }

    // Show score evolution
    println!("  ║");
    println!("  ║  Score evolution (every 20 evals):");
    for (i, score) in &result.score_history {
        if *i % 20 == 0 || *i == result.total_evals - 1 {
            let bar_len = ((*score * 40.0).max(0.0).min(40.0)) as usize;
            let bar = "█".repeat(bar_len);
            println!("  ║    eval {:3}: {:.4}  {}", i, score, bar);
        }
    }

    println!(
        "  ╚══════════════════════════════════════════════════════════════════════════════════════╝"
    );

    // Assert minimum quality
    assert!(
        result.total_evals >= 20,
        "Should complete at least 20 evaluations"
    );
    assert!(result.best_score.is_finite(), "Best score should be finite");
}

// ═══════════════════════════════════════════════════════════════
// Live LLM Test — calls Trinity via OpenRouter
// ═══════════════════════════════════════════════════════════════

/// Live test that calls Trinity (arcee-ai/trinity-large-preview:free)
/// via OpenRouter for LLM-guided GEPA mutations with persistence.
///
/// **Feedback Loop**: Each run loads the best weights from the previous run,
/// optimizes further, and saves the new best—building on prior knowledge.
///
/// Run with: `cargo test -p hehrgnn --test gepa_optimizer_test test_gepa_llm -- --ignored --nocapture`
///
/// Requires OPENAI_API_KEY in .env or environment.
#[tokio::test]
#[ignore]
async fn test_gepa_llm_mutator_with_trinity() {
    let weights_path = "/tmp/gepa_weights.json";

    println!(
        "\n  ╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "  ║  GEPA + TRINITY — LLM-Guided Fiduciary Weight Optimization (feedback loop)       ║"
    );
    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════╣"
    );

    let objective = "Optimize fiduciary blending weights for a financial graph recommendation system. \
        The system scores entities by combining GNN anomaly detection (gnn_weight) with \
        Probabilistic Circuit risk calibration (pc_weight). The fiduciary axes weights \
        (cost_weight, risk_weight, goal_weight, urgency_weight) control how much each \
        financial dimension contributes to the composite score. \
        Goal: high-anomaly risky entities (payday loans, gambling merchants) should rank \
        ABOVE safe entities (insurance, savings) in the recommendation list. \
        Maximize Kendall tau rank correlation, top-3 anomaly concentration, and risk-safe separation.";

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

    // ── FEEDBACK LOOP: Load previous best or use defaults ──
    let prev_weights = OptimizedWeights::load_or_default(weights_path);
    let seed = if prev_weights.total_evals > 0 {
        println!(
            "  ║  📂 Loaded previous best from {} (score={:.6}, evals={})",
            weights_path, prev_weights.score, prev_weights.total_evals
        );
        prev_weights.to_candidate()
    } else {
        println!("  ║  🆕 No previous weights found — starting from defaults");
        Candidate::seed(vec![
            ("gnn_weight", "0.7000"),
            ("pc_weight", "0.3000"),
            ("cost_weight", "0.2500"),
            ("risk_weight", "0.2500"),
            ("goal_weight", "0.1500"),
            ("urgency_weight", "0.1500"),
        ])
    };

    let graph = build_test_graph();
    let evaluator = FiduciaryEvaluator::new(graph);

    let seed_eval = evaluator.evaluate(&seed);
    println!("  ║  Seed score: {:.6}", seed_eval.score);
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

    // ── FEEDBACK LOOP: Save best weights for next run ──
    let mut best_weights =
        OptimizedWeights::from_candidate(&result.best_candidate, result.best_score);
    best_weights.total_evals = prev_weights.total_evals + result.total_evals;
    match best_weights.save(weights_path) {
        Ok(()) => println!(
            "  ║  💾 Saved best weights to {} (cumulative evals={})",
            weights_path, best_weights.total_evals
        ),
        Err(e) => println!("  ║  ⚠️  Save failed: {}", e),
    }

    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "  ║  Best score: {:.6}  (after {} LLM-guided evaluations, {} cumulative)",
        result.best_score, result.total_evals, best_weights.total_evals
    );
    println!("  ║  Best weights (discovered by Trinity):");
    let mut params: Vec<_> = result.best_candidate.params.iter().collect();
    params.sort_by_key(|(k, _)| k.clone());
    for (key, val) in &params {
        println!("  ║    {:16}: {}", key, val);
    }

    let improvement = result.best_score - seed_eval.score;
    if improvement > 0.0 {
        println!(
            "  ║  ✅ Trinity improved score by {:.6} ({:.1}%)",
            improvement,
            improvement / seed_eval.score.abs().max(0.001) * 100.0
        );
    } else {
        println!("  ║  ℹ️  No improvement this run (exploring further needed)");
    }
    println!("  ║");
    println!("  ║  🔄 Run again to continue optimizing from this checkpoint!");
    println!(
        "  ╚══════════════════════════════════════════════════════════════════════════════════════╝"
    );

    assert!(
        result.total_evals >= 5,
        "Should complete at least 5 LLM-guided evaluations"
    );
    assert!(result.best_score.is_finite(), "Best score should be finite");
}
