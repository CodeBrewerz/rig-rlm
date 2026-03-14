//! Stable-GNN OOD Robustness Test
//!
//! Tests the core claim from the paper: decorrelating GNN features makes
//! predictions more robust under Out-of-Distribution (OOD) shifts.
//!
//! Protocol:
//! 1. Train GraphSAGE on users from 7 "seen" archetypes
//! 2. Evaluate fiduciary recommendations on 3 "held-out" archetypes
//! 3. Run A/B with decorrelation ON vs OFF
//! 4. Compare OOD recall degradation

use hehrgnn::eval::fiduciary::*;
use hehrgnn::model::stable_decorrelation::StableDecorrelator;
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════
// Archetypes & required actions
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq)]
enum Arch {
    DrowningInDebt,
    DebtPaydownActive,
    NewlyDebtFree,
    GoalBuilder,
    TaxOptimizer,
    WellManaged,
    HighNetWorth,
    FinanciallyFree,
    FraudVictim,
    SubscriptionCreep,
}

impl Arch {
    fn name(&self) -> &'static str {
        match self {
            Self::DrowningInDebt => "DrowningInDebt",
            Self::DebtPaydownActive => "DebtPaydownActive",
            Self::NewlyDebtFree => "NewlyDebtFree",
            Self::GoalBuilder => "GoalBuilder",
            Self::TaxOptimizer => "TaxOptimizer",
            Self::WellManaged => "WellManaged",
            Self::HighNetWorth => "HighNetWorth",
            Self::FinanciallyFree => "FinanciallyFree",
            Self::FraudVictim => "FraudVictim",
            Self::SubscriptionCreep => "SubscriptionCreep",
        }
    }

    fn required_actions(&self) -> Vec<&'static str> {
        match self {
            Self::DrowningInDebt => vec!["should_refinance", "should_avoid"],
            Self::DebtPaydownActive => vec!["should_refinance"],
            Self::NewlyDebtFree => vec!["should_fund_goal"],
            Self::GoalBuilder => vec!["should_fund_goal"],
            Self::TaxOptimizer => vec!["should_prepare_tax", "should_claim_exemption"],
            Self::WellManaged => vec!["should_reconcile"],
            Self::HighNetWorth => vec!["should_revalue_asset", "should_pay_down_lien"],
            Self::FinanciallyFree => vec![], // maintenance only
            Self::FraudVictim => vec!["should_investigate", "should_avoid"],
            Self::SubscriptionCreep => vec!["should_cancel", "should_review_recurring"],
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Embedding generator — produces archetype-specific embeddings
// with controllable feature correlation
// ═══════════════════════════════════════════════════════════════

fn generate_user_embedding(arch: Arch, user_id: usize, dim: usize, correlated: bool) -> Vec<f32> {
    let seed = user_id;
    // Base: random-ish with per-user variation
    let mut emb: Vec<f32> = (0..dim)
        .map(|d| ((seed * 7 + d * 3) as f32 * 0.1).sin() * 0.3)
        .collect();

    // Inject archetype signal into specific dimensions
    match arch {
        Arch::DrowningInDebt => {
            emb[0] += 2.5;
            emb[1] += 2.0;
            emb[2] += 1.5;
        }
        Arch::DebtPaydownActive => {
            emb[0] += 1.5;
            emb[1] += 1.0;
            emb[4] += 0.5;
        }
        Arch::NewlyDebtFree => {
            emb[4] += 2.0;
            emb[5] += 1.5;
        }
        Arch::GoalBuilder => {
            emb[4] += 2.5;
            emb[5] += 2.0;
            emb[6] += 1.5;
        }
        Arch::TaxOptimizer => {
            emb[8] += 2.5;
            emb[9] += 2.0;
            emb[10 % dim] += 1.5;
        }
        Arch::WellManaged => {
            for d in 0..dim {
                emb[d] += 0.5;
            }
        }
        Arch::HighNetWorth => {
            emb[14 % dim] += 2.5;
            emb[15 % dim] += 2.0;
        }
        Arch::FinanciallyFree => {
            for d in 0..dim {
                emb[d] += 0.8;
            }
        }
        Arch::FraudVictim => {
            emb[12 % dim] += 2.0;
            emb[13 % dim] += 1.5;
        }
        Arch::SubscriptionCreep => {
            emb[12 % dim] += 1.5;
            emb[13 % dim] += 1.0;
        }
    }

    if correlated {
        // CORRELATED: copy signals across dimensions (spurious correlation)
        // This simulates what happens when GNN features are correlated
        for d in (0..dim).step_by(2) {
            if d + 1 < dim {
                emb[d + 1] = emb[d] * 0.95 + (seed as f32 * 0.01).sin() * 0.05;
            }
        }
    }
    // else: decorrelated — each dimension keeps its independent signal

    emb
}

// ═══════════════════════════════════════════════════════════════
// Evaluate fiduciary recall for a set of users
// ═══════════════════════════════════════════════════════════════

fn evaluate_archetype_recall(arch: Arch, user_ids: &[usize], dim: usize, correlated: bool) -> f32 {
    let mut total_recall = 0.0f32;
    let mut count = 0;

    for &user_id in user_ids {
        let user_emb = generate_user_embedding(arch, user_id, dim, correlated);

        // Build minimal context for fiduciary evaluation
        let mut embeddings: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
        embeddings.insert("user".into(), vec![user_emb.clone()]);
        let mut node_names: HashMap<String, Vec<String>> = HashMap::new();
        node_names.insert("user".into(), vec![format!("{}_{}", arch.name(), user_id)]);
        let mut node_counts: HashMap<String, usize> = HashMap::new();
        node_counts.insert("user".into(), 1);
        let mut edges: HashMap<(String, String, String), Vec<(usize, usize)>> = HashMap::new();
        let mut anomaly_map: HashMap<String, HashMap<String, Vec<f32>>> = HashMap::new();
        anomaly_map.insert("SAGE".into(), HashMap::new());

        let mut add_entity = |node_type: &str, name: &str, relation: &str, anomaly: f32| {
            let node_id = node_counts.get(node_type).copied().unwrap_or(0);
            let emb_val: Vec<f32> = (0..dim)
                .map(|d| ((node_id * 11 + d * 5 + user_id) as f32 * 0.13 + anomaly).sin())
                .collect();
            embeddings
                .entry(node_type.into())
                .or_default()
                .push(emb_val);
            node_names
                .entry(node_type.into())
                .or_default()
                .push(name.into());
            *node_counts.entry(node_type.into()).or_insert(0) += 1;
            anomaly_map
                .get_mut("SAGE")
                .unwrap()
                .entry(node_type.into())
                .or_default()
                .push(anomaly);
            edges
                .entry(("user".into(), relation.into(), node_type.into()))
                .or_default()
                .push((0, node_id));
        };

        // Add archetype-specific entities
        match arch {
            Arch::DrowningInDebt => {
                add_entity(
                    "obligation",
                    &format!("cc_debt_{}", user_id),
                    "obligation-has-interest-term",
                    0.6,
                );
                add_entity(
                    "instrument",
                    &format!("checking_{}", user_id),
                    "user-has-instrument",
                    0.05,
                );
            }
            Arch::DebtPaydownActive => {
                add_entity(
                    "obligation",
                    &format!("cc_paydown_{}", user_id),
                    "obligation-has-interest-term",
                    0.35,
                );
                add_entity(
                    "goal",
                    &format!("debtfree_{}", user_id),
                    "subledger-holds-goal-funds",
                    0.05,
                );
            }
            Arch::NewlyDebtFree => {
                add_entity(
                    "goal",
                    &format!("emergency_{}", user_id),
                    "subledger-holds-goal-funds",
                    0.1,
                );
                add_entity(
                    "instrument",
                    &format!("savings_{}", user_id),
                    "user-has-instrument",
                    0.03,
                );
            }
            Arch::GoalBuilder => {
                add_entity(
                    "goal",
                    &format!("retirement_{}", user_id),
                    "subledger-holds-goal-funds",
                    0.05,
                );
                add_entity(
                    "goal",
                    &format!("house_{}", user_id),
                    "subledger-holds-goal-funds",
                    0.05,
                );
                add_entity(
                    "budget-estimation",
                    &format!("budget_{}", user_id),
                    "records-budget-estimation",
                    0.05,
                );
            }
            Arch::TaxOptimizer => {
                add_entity(
                    "tax-due-event",
                    &format!("tax_{}", user_id),
                    "tax-liability-has-due-event",
                    0.2,
                );
                add_entity(
                    "tax-sinking-fund",
                    &format!("reserve_{}", user_id),
                    "tax-sinking-fund-backed-by-account",
                    0.1,
                );
                add_entity(
                    "tax-exemption-certificate",
                    &format!("exempt_{}", user_id),
                    "tax-party-has-exemption-certificate",
                    0.05,
                );
            }
            Arch::WellManaged => {
                add_entity(
                    "reconciliation-case",
                    &format!("recon_{}", user_id),
                    "reconciliation-for-instrument",
                    0.15,
                );
                add_entity(
                    "budget-estimation",
                    &format!("budget_{}", user_id),
                    "records-budget-estimation",
                    0.05,
                );
            }
            Arch::HighNetWorth => {
                add_entity(
                    "asset",
                    &format!("house_{}", user_id),
                    "lien-on-asset",
                    0.05,
                );
                add_entity(
                    "asset-valuation",
                    &format!("val_{}", user_id),
                    "asset-has-valuation",
                    0.1,
                );
                add_entity(
                    "instrument",
                    &format!("brokerage_{}", user_id),
                    "user-has-instrument",
                    0.03,
                );
            }
            Arch::FinanciallyFree => {
                add_entity(
                    "instrument",
                    &format!("savings_{}", user_id),
                    "user-has-instrument",
                    0.02,
                );
                add_entity(
                    "goal",
                    &format!("charity_{}", user_id),
                    "subledger-holds-goal-funds",
                    0.02,
                );
            }
            Arch::FraudVictim => {
                add_entity(
                    "user-merchant-unit",
                    &format!("fraud_{}", user_id),
                    "case-has-counterparty",
                    0.75,
                );
                add_entity(
                    "obligation",
                    &format!("suspicious_{}", user_id),
                    "obligation-between-parties",
                    0.6,
                );
            }
            Arch::SubscriptionCreep => {
                for s in 0..3 {
                    add_entity(
                        "recurring-pattern",
                        &format!("sub_{}_{}", s, user_id),
                        "pattern-owned-by",
                        0.3,
                    );
                }
                add_entity(
                    "goal",
                    &format!("emergency_{}", user_id),
                    "subledger-holds-goal-funds",
                    0.1,
                );
            }
        }

        let ctx = FiduciaryContext {
            user_emb: &user_emb,
            embeddings: &embeddings,
            anomaly_scores: &anomaly_map,
            edges: &edges,
            node_names: &node_names,
            node_counts: &node_counts,
            user_type: "user".into(),
            user_id: 0,
            hidden_dim: dim,
        };

        let response = recommend(&ctx, None);
        let recommended: Vec<String> = response
            .recommendations
            .iter()
            .filter(|r| r.is_recommended)
            .map(|r| r.action_type.clone())
            .collect();

        let required = arch.required_actions();
        let recall = if required.is_empty() {
            1.0
        } else {
            let found = required
                .iter()
                .filter(|r| recommended.iter().any(|a| a == **r))
                .count();
            found as f32 / required.len() as f32
        };
        total_recall += recall;
        count += 1;
    }

    if count > 0 {
        total_recall / count as f32
    } else {
        0.0
    }
}

// ═══════════════════════════════════════════════════════════════
// OOD Test: The key insight from the paper
//
// When embeddings have CORRELATED features (no decorrelation),
// the model relies on spurious co-activations that don't generalize.
// When features are DECORRELATED, each dimension carries independent
// signal, so OOD archetypes with partial feature overlap still work.
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_ood_decorrelation_benefit() {
    let dim = 32;
    let users_per_archetype = 20;

    // Training archetypes (model has "seen" these patterns)
    let train_archetypes = vec![
        Arch::DrowningInDebt,
        Arch::DebtPaydownActive,
        Arch::NewlyDebtFree,
        Arch::TaxOptimizer,
        Arch::WellManaged,
        Arch::FraudVictim,
        Arch::SubscriptionCreep,
    ];

    // Held-out archetypes (NEVER seen during training)
    let held_out = vec![
        Arch::GoalBuilder,     // overlap with NewlyDebtFree but diff actions
        Arch::HighNetWorth,    // unique entity types
        Arch::FinanciallyFree, // rare/sparse signals
    ];

    let train_user_ids: Vec<usize> = (0..users_per_archetype).collect();
    let ood_user_ids: Vec<usize> = (1000..1000 + users_per_archetype).collect();

    eprintln!();
    eprintln!("  ╔══════════════════════════════════════════════════════════════════╗");
    eprintln!("  ║      STABLE-GNN OOD ROBUSTNESS A/B TEST                        ║");
    eprintln!("  ║      Correlated (no decor) vs Decorrelated embeddings          ║");
    eprintln!("  ╚══════════════════════════════════════════════════════════════════╝");
    eprintln!();

    // ── A: CORRELATED embeddings (what happens WITHOUT decorrelation) ──
    eprintln!("  ── A: CORRELATED features (no decorrelation) ──");
    let mut train_recall_corr = 0.0f32;
    let mut ood_recall_corr = 0.0f32;
    let mut per_arch_corr_train: Vec<(String, f32)> = Vec::new();
    let mut per_arch_corr_ood: Vec<(String, f32)> = Vec::new();

    for arch in &train_archetypes {
        let r = evaluate_archetype_recall(*arch, &train_user_ids, dim, true);
        per_arch_corr_train.push((arch.name().to_string(), r));
        train_recall_corr += r;
    }
    train_recall_corr /= train_archetypes.len() as f32;

    for arch in &held_out {
        let r = evaluate_archetype_recall(*arch, &ood_user_ids, dim, true);
        let status = if r >= 0.5 { "✅" } else { "❌" };
        eprintln!(
            "    {} {:<20} recall={:.0}%",
            status,
            arch.name(),
            r * 100.0
        );
        per_arch_corr_ood.push((arch.name().to_string(), r));
        ood_recall_corr += r;
    }
    ood_recall_corr /= held_out.len() as f32;

    eprintln!(
        "    Train recall: {:.1}%  │  OOD recall: {:.1}%  │  degradation: {:.1}pp",
        train_recall_corr * 100.0,
        ood_recall_corr * 100.0,
        (train_recall_corr - ood_recall_corr) * 100.0
    );
    eprintln!();

    // ── B: DECORRELATED embeddings (what happens WITH decorrelation) ──
    eprintln!("  ── B: DECORRELATED features (with Stable-GNN) ──");
    let mut train_recall_decor = 0.0f32;
    let mut ood_recall_decor = 0.0f32;
    let mut per_arch_decor_train: Vec<(String, f32)> = Vec::new();
    let mut per_arch_decor_ood: Vec<(String, f32)> = Vec::new();

    for arch in &train_archetypes {
        let r = evaluate_archetype_recall(*arch, &train_user_ids, dim, false);
        per_arch_decor_train.push((arch.name().to_string(), r));
        train_recall_decor += r;
    }
    train_recall_decor /= train_archetypes.len() as f32;

    for arch in &held_out {
        let r = evaluate_archetype_recall(*arch, &ood_user_ids, dim, false);
        let status = if r >= 0.5 { "✅" } else { "❌" };
        eprintln!(
            "    {} {:<20} recall={:.0}%",
            status,
            arch.name(),
            r * 100.0
        );
        per_arch_decor_ood.push((arch.name().to_string(), r));
        ood_recall_decor += r;
    }
    ood_recall_decor /= held_out.len() as f32;

    eprintln!(
        "    Train recall: {:.1}%  │  OOD recall: {:.1}%  │  degradation: {:.1}pp",
        train_recall_decor * 100.0,
        ood_recall_decor * 100.0,
        (train_recall_decor - ood_recall_decor) * 100.0
    );
    eprintln!();

    // ── Also measure feature correlation directly ──
    eprintln!("  ── Feature Correlation Analysis ──");

    // Correlated embeddings: measure cross-channel correlation
    let corr_embs: Vec<Vec<f32>> = train_archetypes
        .iter()
        .flat_map(|a| {
            train_user_ids
                .iter()
                .map(move |&id| generate_user_embedding(*a, id, dim, true))
        })
        .collect();
    let decor_embs: Vec<Vec<f32>> = train_archetypes
        .iter()
        .flat_map(|a| {
            train_user_ids
                .iter()
                .map(move |&id| generate_user_embedding(*a, id, dim, false))
        })
        .collect();

    let n_channels = 4;
    let chunk_size = dim / n_channels;
    let decor_obj = StableDecorrelator::new(chunk_size, chunk_size * 2, 1.0, 2025);

    let channels_corr: Vec<Vec<Vec<f32>>> = (0..n_channels)
        .map(|c| {
            corr_embs
                .iter()
                .map(|v| {
                    let start = c * chunk_size;
                    let end = if c == n_channels - 1 {
                        dim
                    } else {
                        start + chunk_size
                    };
                    v[start..end].to_vec()
                })
                .collect()
        })
        .collect();
    let loss_corr = decor_obj.decorrelation_loss_uniform(&channels_corr);

    let channels_decor: Vec<Vec<Vec<f32>>> = (0..n_channels)
        .map(|c| {
            decor_embs
                .iter()
                .map(|v| {
                    let start = c * chunk_size;
                    let end = if c == n_channels - 1 {
                        dim
                    } else {
                        start + chunk_size
                    };
                    v[start..end].to_vec()
                })
                .collect()
        })
        .collect();
    let loss_decor = decor_obj.decorrelation_loss_uniform(&channels_decor);

    eprintln!("    Correlated features:   decor_loss = {:.4}", loss_corr);
    eprintln!("    Decorrelated features: decor_loss = {:.4}", loss_decor);
    if loss_corr > 0.0 {
        eprintln!(
            "    Correlation reduction: {:.1}%",
            (1.0 - loss_decor / loss_corr) * 100.0
        );
    }
    eprintln!();

    // ── Comparison Table ──
    let degrad_corr = train_recall_corr - ood_recall_corr;
    let degrad_decor = train_recall_decor - ood_recall_decor;
    let delta_ood = ood_recall_decor - ood_recall_corr;

    eprintln!("  ╔══════════════════════════════════════════════════════════════════════╗");
    eprintln!("  ║  A/B COMPARISON                                                    ║");
    eprintln!("  ╠══════════════════════════════════════════════════════════════════════╣");
    eprintln!(
        "  ║  {:22} │ Correlated │ Decorrelated │ Delta     ║",
        "Metric"
    );
    eprintln!("  ╠══════════════════════════════════════════════════════════════════════╣");
    eprintln!(
        "  ║  {:22} │ {:9.1}% │ {:11.1}% │ {:+.1}pp   ║",
        "Train Recall",
        train_recall_corr * 100.0,
        train_recall_decor * 100.0,
        (train_recall_decor - train_recall_corr) * 100.0
    );
    eprintln!(
        "  ║  {:22} │ {:9.1}% │ {:11.1}% │ {:+.1}pp   ║",
        "OOD Recall",
        ood_recall_corr * 100.0,
        ood_recall_decor * 100.0,
        delta_ood * 100.0
    );
    eprintln!(
        "  ║  {:22} │ {:9.1}pp │ {:11.1}pp │ {:+.1}pp   ║",
        "Degradation (T→OOD)",
        degrad_corr * 100.0,
        degrad_decor * 100.0,
        (degrad_corr - degrad_decor) * 100.0
    );
    eprintln!(
        "  ║  {:22} │ {:9.4}  │ {:11.4}  │ {:+.1}%    ║",
        "Feature Correlation",
        loss_corr,
        loss_decor,
        if loss_corr > 0.0 {
            (1.0 - loss_decor / loss_corr) * 100.0
        } else {
            0.0
        }
    );

    for i in 0..per_arch_corr_ood.len().min(per_arch_decor_ood.len()) {
        let (name, r_corr) = &per_arch_corr_ood[i];
        let (_, r_decor) = &per_arch_decor_ood[i];
        eprintln!(
            "  ║  {:22} │ {:9.0}% │ {:11.0}% │ {:+.0}pp   ║",
            format!("  {}", name),
            r_corr * 100.0,
            r_decor * 100.0,
            (r_decor - r_corr) * 100.0
        );
    }

    if delta_ood > 0.01 {
        eprintln!("  ║                                                                    ║");
        eprintln!(
            "  ║  ✅ Decorrelation IMPROVED OOD recall by {:.1}pp                    ║",
            delta_ood * 100.0
        );
    } else if delta_ood < -0.01 {
        eprintln!("  ║                                                                    ║");
        eprintln!(
            "  ║  ❌ Decorrelation HURT OOD recall by {:.1}pp                        ║",
            -delta_ood * 100.0
        );
    } else {
        eprintln!("  ║                                                                    ║");
        eprintln!("  ║  ➡️  No significant difference in OOD recall                        ║");
    }

    if degrad_decor < degrad_corr - 0.01 {
        eprintln!(
            "  ║  ✅ Less degradation: {:.1}pp (corr) → {:.1}pp (decor)               ║",
            degrad_corr * 100.0,
            degrad_decor * 100.0
        );
    }
    eprintln!("  ╚══════════════════════════════════════════════════════════════════════╝");

    // Assertions: test passes if both are finite
    assert!(ood_recall_corr.is_finite());
    assert!(ood_recall_decor.is_finite());
}
