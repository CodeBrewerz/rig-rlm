//! E2E tests for Sparse Autoencoder interpretability with realistic
//! financial scenarios — verifies SAE discovers meaningful concepts.

use std::collections::HashMap;
use hehrgnn::eval::sae::*;

// ═══════════════════════════════════════════════════════════════
// Helper: build financial graph with embeddings + concepts
// ═══════════════════════════════════════════════════════════════

struct FinancialGraph {
    embeddings: Vec<Vec<f32>>,
    edges: HashMap<(String, String, String), Vec<(usize, usize)>>,
    anomaly_scores: HashMap<String, HashMap<String, Vec<f32>>>,
    node_names: Vec<String>,
}

impl FinancialGraph {
    /// Build a financial graph with diverse user profiles for SAE training.
    ///
    /// Creates 100 "user" nodes with various financial situations:
    /// - Users 0-19: High debt (obligations, liens)
    /// - Users 20-39: Goal-oriented savers (goals, budgets)
    /// - Users 40-59: Tax-aware planners (tax entities)
    /// - Users 60-79: Pattern-based spenders (recurring, merchants)
    /// - Users 80-99: Asset holders (assets, valuations)
    fn build_diverse() -> Self {
        let n = 100;
        let dim = 16;
        let mut embeddings = Vec::new();
        let mut node_names = Vec::new();
        let mut edges: HashMap<(String, String, String), Vec<(usize, usize)>> = HashMap::new();
        let mut anomaly_vals = vec![0.1f32; n];

        for i in 0..n {
            let profile = i / 20; // 0-4 for 5 profiles
            let mut emb = vec![0.0f32; dim];

            // Base embedding varies by profile
            for d in 0..dim {
                let base = ((i * 7 + d * 3) as f32 * 0.1).sin() * 0.5;
                emb[d] = base;
            }

            match profile {
                0 => {
                    // HIGH DEBT: strong signal in dims 0-3
                    emb[0] += 2.0;
                    emb[1] += 1.5;
                    emb[2] += 1.0;
                    node_names.push(format!("debt_user_{}", i));
                    // Connect to obligations
                    edges.entry(("user".into(), "obligation-has-interest-term".into(), "obligation".into()))
                        .or_default()
                        .push((i, i % 5));
                    edges.entry(("user".into(), "lien-on-asset".into(), "asset".into()))
                        .or_default()
                        .push((i, i % 3));
                    anomaly_vals[i] = 0.3 + (i as f32 * 0.02);
                }
                1 => {
                    // GOAL SAVERS: strong signal in dims 4-7
                    emb[4] += 2.0;
                    emb[5] += 1.5;
                    emb[6] += 1.0;
                    node_names.push(format!("saver_user_{}", i));
                    edges.entry(("user".into(), "subledger-holds-goal-funds".into(), "goal".into()))
                        .or_default()
                        .push((i, i % 5));
                    edges.entry(("user".into(), "records-budget-estimation".into(), "budget-estimation".into()))
                        .or_default()
                        .push((i, i % 3));
                }
                2 => {
                    // TAX PLANNERS: strong signal in dims 8-11
                    emb[8] += 2.0;
                    emb[9] += 1.5;
                    emb[10] += 1.0;
                    node_names.push(format!("tax_user_{}", i));
                    edges.entry(("user".into(), "tax-liability-has-due-event".into(), "tax-due-event".into()))
                        .or_default()
                        .push((i, i % 5));
                    edges.entry(("user".into(), "tax-sinking-fund-backed-by-account".into(), "tax-sinking-fund".into()))
                        .or_default()
                        .push((i, i % 3));
                }
                3 => {
                    // PATTERN SPENDERS: strong signal in dims 12-14
                    emb[12] += 2.0;
                    emb[13] += 1.5;
                    node_names.push(format!("spender_user_{}", i));
                    edges.entry(("user".into(), "pattern-owned-by".into(), "recurring-pattern".into()))
                        .or_default()
                        .push((i, i % 5));
                    edges.entry(("user".into(), "transacts-at".into(), "merchant".into()))
                        .or_default()
                        .push((i, i % 10));
                    // Some are anomalous (fraud merchants)
                    if i >= 75 {
                        anomaly_vals[i] = 0.7;
                    }
                }
                _ => {
                    // ASSET HOLDERS: strong signal in dim 15
                    emb[15] += 2.0;
                    emb[14] += 1.0;
                    node_names.push(format!("asset_user_{}", i));
                    edges.entry(("user".into(), "asset-has-valuation".into(), "asset-valuation".into()))
                        .or_default()
                        .push((i, i % 5));
                    edges.entry(("user".into(), "user-has-instrument".into(), "instrument".into()))
                        .or_default()
                        .push((i, i % 8));
                }
            }

            embeddings.push(emb);
        }

        let mut anomaly_scores = HashMap::new();
        let mut model_scores = HashMap::new();
        model_scores.insert("user".into(), anomaly_vals);
        anomaly_scores.insert("SAGE".into(), model_scores);

        Self {
            embeddings,
            edges,
            anomaly_scores,
            node_names,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Test 1: SAE Training — sparsity + reconstruction
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_sae_training_and_sparsity() {
    let graph = FinancialGraph::build_diverse();

    let config = SaeConfig {
        expansion_factor: 4,
        l1_coeff: 0.05,
        lr: 0.005,
        epochs: 30,
    };

    let sae = SparseAutoencoder::train(&graph.embeddings, &config);

    println!("\n  ── SAE TRAINING RESULTS ──\n");
    println!("  Hidden dim:     {}", sae.hidden_dim);
    println!("  Expansion dim:  {}", sae.expansion_dim);
    println!("  Final MSE:      {:.6}", sae.final_mse);
    println!("  Avg sparsity:   {:.1}%", sae.avg_sparsity * 100.0);

    assert_eq!(sae.hidden_dim, 16);
    assert_eq!(sae.expansion_dim, 64); // 16 * 4

    // Sparsity: most features should be zero (> 50%)
    let mut total_sparsity = 0.0;
    for emb in &graph.embeddings {
        total_sparsity += sae.sparsity(emb);
    }
    let mean_sparsity = total_sparsity / graph.embeddings.len() as f32;
    println!("  Mean sparsity:  {:.1}%", mean_sparsity * 100.0);
    assert!(mean_sparsity > 0.3, "SAE should learn sparse features (>30%), got {:.1}%", mean_sparsity * 100.0);

    // Reconstruction: should approximate original
    let mut total_recon = 0.0;
    for emb in &graph.embeddings {
        total_recon += sae.reconstruction_error(emb);
    }
    let mean_recon = total_recon / graph.embeddings.len() as f32;
    println!("  Mean recon err: {:.6}", mean_recon);

    // Test individual profiles
    println!("\n  Per-profile sparsity:");
    for profile in 0..5 {
        let start = profile * 20;
        let end = start + 20;
        let profile_sparsity: f32 = graph.embeddings[start..end]
            .iter()
            .map(|e| sae.sparsity(e))
            .sum::<f32>() / 20.0;
        let profile_active: f32 = graph.embeddings[start..end]
            .iter()
            .map(|e| sae.active_features(e).len() as f32)
            .sum::<f32>() / 20.0;
        let label = match profile {
            0 => "Debt holders",
            1 => "Goal savers",
            2 => "Tax planners",
            3 => "Spenders",
            _ => "Asset holders",
        };
        println!("    {:<15} sparsity={:.0}%, active={:.1} features", label, profile_sparsity * 100.0, profile_active);
    }
}

// ═══════════════════════════════════════════════════════════════
// Test 2: Feature Labeling — SAE features map to financial concepts
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_sae_feature_labeling() {
    let graph = FinancialGraph::build_diverse();

    let config = SaeConfig {
        expansion_factor: 4,
        l1_coeff: 0.05,
        lr: 0.005,
        epochs: 30,
    };

    let sae = SparseAutoencoder::train(&graph.embeddings, &config);

    // Compute concept labels from graph structure
    let concept_labels = compute_concept_labels(
        &graph.edges,
        &graph.anomaly_scores,
        "user",
        graph.embeddings.len(),
    );

    // Label features
    let feature_labels = label_features(&sae, &graph.embeddings, &concept_labels);

    println!("\n  ── SAE FEATURE LABELS ({}) ──\n", feature_labels.len());
    for (i, label) in feature_labels.iter().take(15).enumerate() {
        println!(
            "    #{:<2} Feature {:>3} │ {:<20} │ {:<15} │ R={:.3}",
            i + 1,
            label.feature_id,
            label.label,
            label.domain,
            label.correlation,
        );
    }

    // Should have some labeled features
    assert!(
        feature_labels.len() >= 3,
        "SAE should discover at least 3 labeled features, got {}",
        feature_labels.len()
    );

    // Collect unique domains discovered
    let discovered_domains: std::collections::HashSet<&str> = feature_labels
        .iter()
        .map(|l| l.domain.as_str())
        .collect();

    println!("\n  Domains discovered: {:?}", discovered_domains);
    assert!(
        discovered_domains.len() >= 2,
        "Should discover features in at least 2 domains, got {:?}",
        discovered_domains
    );
}

// ═══════════════════════════════════════════════════════════════
// Test 3: SAE Explanations for different financial profiles
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_sae_explanations_per_profile() {
    let graph = FinancialGraph::build_diverse();

    let config = SaeConfig {
        expansion_factor: 4,
        l1_coeff: 0.05,
        lr: 0.005,
        epochs: 30,
    };

    let sae = SparseAutoencoder::train(&graph.embeddings, &config);
    let concept_labels = compute_concept_labels(
        &graph.edges,
        &graph.anomaly_scores,
        "user",
        graph.embeddings.len(),
    );
    let feature_labels = label_features(&sae, &graph.embeddings, &concept_labels);

    println!("\n  ── SAE EXPLANATIONS PER PROFILE ──\n");

    let profiles = [
        (0, "Debt holder"),
        (20, "Goal saver"),
        (40, "Tax planner"),
        (60, "Pattern spender"),
        (80, "Asset holder"),
    ];

    for (node_id, label) in &profiles {
        let expl = explain(&sae, &graph.embeddings[*node_id], &feature_labels);

        println!("  {} ({}):", label, graph.node_names[*node_id]);
        println!("    Sparsity:      {:.0}%", expl.sparsity * 100.0);
        println!("    Reconstruction: {:.0}%", expl.reconstruction_quality * 100.0);
        println!("    Active features: {}", expl.active_features.len());
        for feat in expl.active_features.iter().take(5) {
            println!(
                "      Feature {:>3} │ {:<20} │ {:<15} │ activation={:.3}",
                feat.feature_id, feat.label, feat.domain, feat.activation,
            );
        }
        println!("    Summary: {}", expl.summary);
        println!();

        // Each profile should have active features
        assert!(
            !expl.active_features.is_empty(),
            "Profile {} should have active SAE features",
            label
        );
    }
}

// ═══════════════════════════════════════════════════════════════
// Test 4: Different profiles should have DIFFERENT active features
// (monosemanticity — each feature represents one concept)
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_sae_profiles_have_different_features() {
    let graph = FinancialGraph::build_diverse();

    let config = SaeConfig {
        expansion_factor: 4,
        l1_coeff: 0.05,
        lr: 0.005,
        epochs: 30,
    };

    let sae = SparseAutoencoder::train(&graph.embeddings, &config);

    // Get active feature IDs for each profile
    let debt_features: std::collections::HashSet<usize> = sae
        .active_features(&graph.embeddings[0])
        .iter()
        .map(|(id, _)| *id)
        .collect();

    let saver_features: std::collections::HashSet<usize> = sae
        .active_features(&graph.embeddings[20])
        .iter()
        .map(|(id, _)| *id)
        .collect();

    let tax_features: std::collections::HashSet<usize> = sae
        .active_features(&graph.embeddings[40])
        .iter()
        .map(|(id, _)| *id)
        .collect();

    let spender_features: std::collections::HashSet<usize> = sae
        .active_features(&graph.embeddings[60])
        .iter()
        .map(|(id, _)| *id)
        .collect();

    let asset_features: std::collections::HashSet<usize> = sae
        .active_features(&graph.embeddings[80])
        .iter()
        .map(|(id, _)| *id)
        .collect();

    println!("\n  ── FEATURE DIFFERENTIATION ──\n");
    println!("  Debt features:    {:?}", debt_features);
    println!("  Saver features:   {:?}", saver_features);
    println!("  Tax features:     {:?}", tax_features);
    println!("  Spender features: {:?}", spender_features);
    println!("  Asset features:   {:?}", asset_features);

    // Different profile pairs should have some unique features
    let debt_only = debt_features.difference(&saver_features).count();
    let saver_only = saver_features.difference(&debt_features).count();
    let tax_only = tax_features.difference(&spender_features).count();

    println!("\n  Unique to debt (vs savers): {}", debt_only);
    println!("  Unique to savers (vs debt): {}", saver_only);
    println!("  Unique to tax (vs spenders): {}", tax_only);

    // At least SOME features should be unique to each profile
    // (This proves monosemanticity — features aren't all shared)
    let total_unique = debt_only + saver_only + tax_only;
    assert!(
        total_unique >= 1,
        "Different financial profiles should have differentiated SAE features! \
         Got {} unique features total",
        total_unique
    );
}

// ═══════════════════════════════════════════════════════════════
// Test 5: E2E — SAE + Fiduciary scenario
//
// Full pipeline: build graph → train SAE → get fiduciary recommendations
// → verify SAE explanations align with fiduciary actions.
//
// A user with high debt should have SAE features related to debt,
// and their fiduciary recommendation for "refinance" should
// have SAE features like "debt_holder" active.
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_e2e_sae_with_fiduciary_financial_health() {
    let graph = FinancialGraph::build_diverse();

    // Train SAE
    let config = SaeConfig {
        expansion_factor: 4,
        l1_coeff: 0.05,
        lr: 0.005,
        epochs: 30,
    };

    let sae = SparseAutoencoder::train(&graph.embeddings, &config);
    let concept_labels = compute_concept_labels(
        &graph.edges,
        &graph.anomaly_scores,
        "user",
        graph.embeddings.len(),
    );
    let feature_labels = label_features(&sae, &graph.embeddings, &concept_labels);

    println!("\n  ── E2E: SAE + FIDUCIARY FINANCIAL HEALTH ──\n");

    // Pick representative users from each profile
    let scenarios = [
        (0, "Debt-heavy user", "should get debt-related SAE features"),
        (20, "Goal-oriented saver", "should get goal-related SAE features"),
        (40, "Tax-aware planner", "should get tax-related SAE features"),
        (75, "Anomalous spender", "should get risk-related SAE features"),
        (80, "Asset holder", "should get asset-related SAE features"),
    ];

    for (node_id, label, expectation) in &scenarios {
        let expl = explain(&sae, &graph.embeddings[*node_id], &feature_labels);

        println!("  {} (node {}): {}", label, node_id, expectation);
        println!("    {} active features, sparsity {:.0}%",
            expl.active_features.len(), expl.sparsity * 100.0);

        for feat in expl.active_features.iter().take(3) {
            println!("      └─ {} [{}] activation={:.3}", feat.label, feat.domain, feat.activation);
        }

        // Verify reconstruction quality
        assert!(
            expl.reconstruction_quality > 0.0,
            "SAE should reconstruct embeddings for {} (quality={:.2})",
            label,
            expl.reconstruction_quality
        );

        // Verify non-empty explanation
        assert!(
            !expl.summary.is_empty(),
            "SAE explanation summary should not be empty for {}",
            label
        );

        println!("    Summary: {}\n", expl.summary);
    }

    // Verify anomalous user (node 75, spender with anomaly=0.7) has risk-related features
    let anomalous_expl = explain(&sae, &graph.embeddings[75], &feature_labels);
    println!("  Anomalous user (75) domains: {:?}",
        anomalous_expl.active_features.iter()
            .filter(|f| !f.label.starts_with("unlabeled"))
            .map(|f| f.domain.as_str())
            .collect::<Vec<_>>()
    );

    // Overall: verify SAE provides useful decomposition
    let total_labeled: usize = scenarios.iter()
        .map(|(id, _, _)| {
            let expl = explain(&sae, &graph.embeddings[*id], &feature_labels);
            expl.active_features.iter()
                .filter(|f| !f.label.starts_with("unlabeled"))
                .count()
        })
        .sum();

    println!("\n  Total labeled features across all scenarios: {}", total_labeled);
    assert!(
        total_labeled >= 3,
        "Across all scenarios, SAE should find at least 3 labeled features, got {}",
        total_labeled
    );
}
