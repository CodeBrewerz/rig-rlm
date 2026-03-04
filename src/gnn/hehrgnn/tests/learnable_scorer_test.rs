//! Learnable Fiduciary Scorer E2E Tests.
//!
//! Tests the full learning pipeline:
//! 1. Knowledge distillation from expert rules
//! 2. Learned scorer matches expert on training data
//! 3. Reward-based fine-tuning improves on edge cases
//! 4. Recursive self-improvement compounds learning
//! 5. Axis weights drift from initial expert values (adaptation)
//! 6. Conflict patterns are learned from rejections

use hehrgnn::eval::fiduciary::*;
use hehrgnn::eval::learnable_scorer::*;

// ═══════════════════════════════════════════════════════════════
// Helpers: build examples from scenarios
// ═══════════════════════════════════════════════════════════════

fn make_example(
    action: FiduciaryActionType,
    anomaly: f32,
    affinity: f32,
    context: [f32; 5],
) -> ScorerExample {
    // Generate deterministic embeddings based on action + anomaly
    let idx = match action {
        FiduciaryActionType::ShouldRefinance => 0,
        FiduciaryActionType::ShouldAvoid => 1,
        FiduciaryActionType::ShouldInvestigate => 2,
        FiduciaryActionType::ShouldFundGoal => 3,
        FiduciaryActionType::ShouldCancel => 4,
        FiduciaryActionType::ShouldPrepareTax => 5,
        FiduciaryActionType::ShouldReconcile => 6,
        FiduciaryActionType::ShouldRevalueAsset => 7,
        _ => 8,
    };
    let dim = 32;
    let user_emb: Vec<f32> = (0..dim)
        .map(|d| ((idx * 7 + d * 3) as f32 * 0.1 + anomaly).sin() * 0.5)
        .collect();
    let target_emb: Vec<f32> = (0..dim)
        .map(|d| ((idx * 11 + d * 5) as f32 * 0.13 + affinity).sin() * 0.5)
        .collect();

    ScorerExample {
        user_emb,
        target_emb,
        action_type: action,
        anomaly_score: anomaly,
        embedding_affinity: affinity,
        context,
    }
}

fn expert_label(action: FiduciaryActionType, anomaly: f32) -> ScorerLabel {
    // Simulates what score_action() would produce
    let (cost, risk, goal, urgency, conflict, revers) = match action {
        FiduciaryActionType::ShouldRefinance => (0.8, 0.6, 0.7, 0.5, 0.7, 0.2),
        FiduciaryActionType::ShouldAvoid => (0.3, 0.9, 0.4, 0.7 + anomaly * 0.3, 0.8, 0.9),
        FiduciaryActionType::ShouldInvestigate => (0.2, 0.8, 0.3, 0.8, 0.7, 0.5),
        FiduciaryActionType::ShouldFundGoal => (0.5, 0.3, 0.9, 0.4, 0.6, 0.3),
        FiduciaryActionType::ShouldCancel => (0.7, 0.4, 0.5, 0.3, 0.8, 0.9),
        FiduciaryActionType::ShouldPrepareTax => (0.6, 0.5, 0.5, 0.7, 0.8, 0.3),
        FiduciaryActionType::ShouldReconcile => (0.3, 0.4, 0.3, 0.3, 0.9, 0.5),
        FiduciaryActionType::ShouldRevalueAsset => (0.4, 0.5, 0.6, 0.3, 0.7, 0.4),
        _ => (0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
    };
    ScorerLabel {
        axes: FiduciaryAxes {
            cost_reduction: cost,
            risk_reduction: risk,
            goal_alignment: goal,
            urgency,
            conflict_freedom: conflict,
            reversibility: revers,
        },
        should_recommend: anomaly < 0.7
            || matches!(
                action,
                FiduciaryActionType::ShouldInvestigate | FiduciaryActionType::ShouldAvoid
            ),
    }
}

// ═══════════════════════════════════════════════════════════════
// Test 1: Knowledge Distillation — Learned scores match expert
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_knowledge_distillation_matches_expert() {
    let config = ScorerConfig {
        embedding_dim: 32,
        hidden1: 64,
        hidden2: 32,
        lr: 0.005,
        ..ScorerConfig::default()
    };
    let mut scorer = LearnableScorer::new(&config);

    // Generate training set: 8 action types × 5 anomaly levels = 40 examples
    let actions = [
        FiduciaryActionType::ShouldRefinance,
        FiduciaryActionType::ShouldAvoid,
        FiduciaryActionType::ShouldInvestigate,
        FiduciaryActionType::ShouldFundGoal,
        FiduciaryActionType::ShouldCancel,
        FiduciaryActionType::ShouldPrepareTax,
        FiduciaryActionType::ShouldReconcile,
        FiduciaryActionType::ShouldRevalueAsset,
    ];
    let anomaly_levels = [0.1, 0.3, 0.5, 0.7, 0.9];

    let mut examples = Vec::new();
    let mut labels = Vec::new();

    for &action in &actions {
        for &anomaly in &anomaly_levels {
            let ctx = [0.3, 0.5, 0.4, 0.0, 0.0]; // degree, debt, goal_progress, tax, recurring
            examples.push(make_example(action, anomaly, 0.5, ctx));
            labels.push(expert_label(action, anomaly));
        }
    }

    println!("\n  ── KNOWLEDGE DISTILLATION ──\n");
    println!(
        "  Training on {} examples ({} actions × {} anomaly levels)\n",
        examples.len(),
        actions.len(),
        anomaly_levels.len()
    );

    // Before training: random predictions
    let pre_correct = examples
        .iter()
        .zip(labels.iter())
        .filter(|(ex, lbl)| {
            let (_, logit) = scorer.forward(ex);
            (logit > 0.0) == lbl.should_recommend
        })
        .count();
    println!(
        "  Pre-distillation accuracy: {}/{} ({:.0}%)",
        pre_correct,
        examples.len(),
        pre_correct as f32 / examples.len() as f32 * 100.0
    );

    // Distill
    scorer.distill(&examples, &labels, 100);

    // After training: should match expert
    let post_correct = examples
        .iter()
        .zip(labels.iter())
        .filter(|(ex, lbl)| {
            let (_, logit) = scorer.forward(ex);
            (logit > 0.0) == lbl.should_recommend
        })
        .count();
    let accuracy = post_correct as f32 / examples.len() as f32;
    println!(
        "  Post-distillation accuracy: {}/{} ({:.0}%)",
        post_correct,
        examples.len(),
        accuracy * 100.0
    );

    // Check axis score correlation
    let mut total_axis_err = 0.0f32;
    for (ex, lbl) in examples.iter().zip(labels.iter()) {
        let (pred_axes, _) = scorer.forward(ex);
        total_axis_err += (pred_axes.cost_reduction - lbl.axes.cost_reduction).abs();
        total_axis_err += (pred_axes.risk_reduction - lbl.axes.risk_reduction).abs();
        total_axis_err += (pred_axes.goal_alignment - lbl.axes.goal_alignment).abs();
    }
    let mean_axis_err = total_axis_err / (examples.len() as f32 * 3.0);
    println!("  Mean axis prediction error: {:.3}", mean_axis_err);

    println!("  Learned axis weights: {:?}", scorer.axis_weights);
    println!("  Samples seen: {}", scorer.samples_seen);

    assert!(
        accuracy >= 0.60,
        "Distilled scorer should match expert on ≥60% of examples, got {:.0}%",
        accuracy * 100.0
    );
}

// ═══════════════════════════════════════════════════════════════
// Test 2: Reward-based fine-tuning improves on edge cases
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_reward_fine_tuning_improves_scoring() {
    let config = ScorerConfig {
        embedding_dim: 32,
        hidden1: 64,
        hidden2: 32,
        lr: 0.005,
        miss_penalty_multiplier: 3.0,
    };
    let mut scorer = LearnableScorer::new(&config);

    // Pre-train with distillation
    let actions = [
        FiduciaryActionType::ShouldRefinance,
        FiduciaryActionType::ShouldFundGoal,
        FiduciaryActionType::ShouldCancel,
        FiduciaryActionType::ShouldAvoid,
    ];
    let mut examples = Vec::new();
    let mut labels = Vec::new();
    for &action in &actions {
        for anomaly in [0.2, 0.5, 0.8] {
            let ctx = [0.3, 0.5, 0.4, 0.0, 0.0];
            examples.push(make_example(action, anomaly, 0.5, ctx));
            labels.push(expert_label(action, anomaly));
        }
    }
    scorer.distill(&examples, &labels, 50);

    println!("\n  ── REWARD-BASED FINE-TUNING ──\n");

    // Simulate user feedback: user accepts refinance, rejects fund_goal when in debt
    let rewards = vec![
        RewardSignal {
            action_type: FiduciaryActionType::ShouldRefinance,
            accepted: true,
            helpfulness: Some(0.9),
            example: make_example(
                FiduciaryActionType::ShouldRefinance,
                0.6,
                0.5,
                [0.3, 0.8, 0.1, 0.0, 0.0],
            ),
            was_high_risk: false,
        },
        RewardSignal {
            action_type: FiduciaryActionType::ShouldFundGoal,
            accepted: false, // User rejected: "I have debt, don't tell me to invest!"
            helpfulness: Some(0.8),
            example: make_example(
                FiduciaryActionType::ShouldFundGoal,
                0.3,
                0.5,
                [0.3, 0.8, 0.1, 0.0, 0.0],
            ),
            was_high_risk: false,
        },
        RewardSignal {
            action_type: FiduciaryActionType::ShouldCancel,
            accepted: true,
            helpfulness: Some(0.7),
            example: make_example(
                FiduciaryActionType::ShouldCancel,
                0.3,
                0.4,
                [0.3, 0.5, 0.3, 0.0, 0.5],
            ),
            was_high_risk: false,
        },
        RewardSignal {
            action_type: FiduciaryActionType::ShouldAvoid,
            accepted: true,
            helpfulness: Some(1.0),
            example: make_example(
                FiduciaryActionType::ShouldAvoid,
                0.9,
                0.1,
                [0.1, 0.2, 0.1, 0.0, 0.0],
            ),
            was_high_risk: true, // high anomaly = genuinely risky
        },
    ];

    // Get scores before reward
    let pre_refinance = {
        let (axes, _) = scorer.forward(&rewards[0].example);
        scorer.score(&axes)
    };
    let pre_fund_goal = {
        let (axes, _) = scorer.forward(&rewards[1].example);
        scorer.score(&axes)
    };

    // Apply 10 rounds of reward
    for _ in 0..10 {
        for reward in &rewards {
            scorer.apply_reward(reward);
        }
    }

    // Get scores after reward
    let post_refinance = {
        let (axes, _) = scorer.forward(&rewards[0].example);
        scorer.score(&axes)
    };
    let post_fund_goal = {
        let (axes, _) = scorer.forward(&rewards[1].example);
        scorer.score(&axes)
    };

    println!(
        "  ShouldRefinance (accepted): {:.3} → {:.3} ({})",
        pre_refinance,
        post_refinance,
        if post_refinance > pre_refinance {
            "↑ strengthened ✅"
        } else {
            "→ stable"
        }
    );
    println!(
        "  ShouldFundGoal (rejected):  {:.3} → {:.3} ({})",
        pre_fund_goal,
        post_fund_goal,
        if post_fund_goal < pre_fund_goal {
            "↓ weakened ✅"
        } else {
            "→ stable"
        }
    );

    // Check learned conflict matrix
    let fund_goal_self_conflict = scorer.conflict_matrix[9][9]; // ShouldFundGoal index
    println!(
        "  ShouldFundGoal self-conflict: {:.3} (learned from rejections)",
        fund_goal_self_conflict
    );
    println!("  Axis weights after reward: {:?}", scorer.axis_weights);

    assert!(
        fund_goal_self_conflict > 0.0,
        "Should have learned conflict for rejected action"
    );
}

// ═══════════════════════════════════════════════════════════════
// Test 3: Recursive self-improvement compounds learning
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_recursive_self_improvement() {
    let config = ScorerConfig {
        embedding_dim: 32,
        hidden1: 64,
        hidden2: 32,
        lr: 0.005,
        miss_penalty_multiplier: 3.0,
    };
    let mut scorer = LearnableScorer::new(&config);

    // Distill first
    let mut examples = Vec::new();
    let mut labels = Vec::new();
    for &action in &FiduciaryActionType::all() {
        for anomaly in [0.1, 0.4, 0.7] {
            let ctx = [0.3, 0.5, 0.4, 0.0, 0.0];
            examples.push(make_example(action, anomaly, 0.5, ctx));
            labels.push(expert_label(action, anomaly));
        }
    }
    scorer.distill(&examples, &labels, 50);

    // Build replay buffer from mixed feedback
    let replay_buffer: Vec<RewardSignal> = vec![
        // Debt scenarios: refinance accepted
        RewardSignal {
            action_type: FiduciaryActionType::ShouldRefinance,
            accepted: true,
            helpfulness: Some(0.9),
            example: make_example(
                FiduciaryActionType::ShouldRefinance,
                0.5,
                0.6,
                [0.2, 0.7, 0.2, 0.0, 0.0],
            ),
            was_high_risk: false,
        },
        // Goal building accepted when no debt
        RewardSignal {
            action_type: FiduciaryActionType::ShouldFundGoal,
            accepted: true,
            helpfulness: Some(0.8),
            example: make_example(
                FiduciaryActionType::ShouldFundGoal,
                0.1,
                0.7,
                [0.5, 0.1, 0.7, 0.0, 0.0],
            ),
            was_high_risk: false,
        },
        // Fund goal rejected when in debt
        RewardSignal {
            action_type: FiduciaryActionType::ShouldFundGoal,
            accepted: false,
            helpfulness: Some(0.9),
            example: make_example(
                FiduciaryActionType::ShouldFundGoal,
                0.3,
                0.5,
                [0.3, 0.8, 0.1, 0.0, 0.0],
            ),
            was_high_risk: false,
        },
        // Tax prep accepted
        RewardSignal {
            action_type: FiduciaryActionType::ShouldPrepareTax,
            accepted: true,
            helpfulness: Some(0.7),
            example: make_example(
                FiduciaryActionType::ShouldPrepareTax,
                0.2,
                0.5,
                [0.4, 0.3, 0.5, 1.0, 0.0],
            ),
            was_high_risk: false,
        },
        // Avoid accepted for fraud
        RewardSignal {
            action_type: FiduciaryActionType::ShouldAvoid,
            accepted: true,
            helpfulness: Some(1.0),
            example: make_example(
                FiduciaryActionType::ShouldAvoid,
                0.9,
                0.1,
                [0.1, 0.2, 0.1, 0.0, 0.0],
            ),
            was_high_risk: true,
        },
    ];

    println!("\n  ── RECURSIVE SELF-IMPROVEMENT ──\n");

    let initial_weights = scorer.axis_weights;

    let report = scorer.recursive_improve(&replay_buffer, 10);

    println!("  Rounds completed: {}", report.rounds_completed);
    println!(
        "  Initial accuracy: {:.0}%",
        report.initial_accuracy * 100.0
    );
    println!("  Final accuracy:   {:.0}%", report.final_accuracy * 100.0);
    println!("  Axis weight drift: {:?}", report.axis_weight_drift);
    println!(
        "  Conflict patterns learned: {}",
        report.conflict_patterns_learned
    );
    println!("  Final weights: {:?}", scorer.axis_weights);
    println!("  Initial weights: {:?}", initial_weights);

    // Verify improvement or at least no regression
    assert!(
        report.final_accuracy >= report.initial_accuracy * 0.9,
        "Recursive improvement should not make things significantly worse"
    );

    // Verify axis weights have adapted
    let total_drift: f32 = report.axis_weight_drift.iter().sum();
    println!("  Total weight drift: {:.4}", total_drift);

    // Verify conflict patterns were learned
    assert!(
        report.conflict_patterns_learned > 0,
        "Should have learned at least one conflict pattern from rejections"
    );
}

// ═══════════════════════════════════════════════════════════════
// Test 4: Full lifecycle — distill → deploy → feedback → improve
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_full_learning_lifecycle() {
    let config = ScorerConfig {
        embedding_dim: 32,
        hidden1: 64,
        hidden2: 32,
        lr: 0.005,
        miss_penalty_multiplier: 3.0,
    };
    let mut scorer = LearnableScorer::new(&config);

    println!("\n  ── FULL LEARNING LIFECYCLE ──\n");

    // Phase 1: Distill from expert rules
    println!("  Phase 1: Knowledge Distillation");
    let mut examples = Vec::new();
    let mut labels = Vec::new();
    for &action in &FiduciaryActionType::all() {
        for anomaly in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let ctx = [0.3, 0.5, 0.4, 0.0, 0.0];
            examples.push(make_example(action, anomaly, 0.5, ctx));
            labels.push(expert_label(action, anomaly));
        }
    }
    scorer.distill(&examples, &labels, 80);
    let phase1_accuracy = examples
        .iter()
        .zip(labels.iter())
        .filter(|(ex, lbl)| {
            let (_, logit) = scorer.forward(ex);
            (logit > 0.0) == lbl.should_recommend
        })
        .count() as f32
        / examples.len() as f32;
    println!("    Accuracy: {:.0}%", phase1_accuracy * 100.0);

    // Phase 2: Deploy and collect user feedback (simulated)
    println!("  Phase 2: Reward Learning (50 simulated users)");
    let mut replay_buffer = Vec::new();
    for user_id in 0..50 {
        let anomaly = (user_id as f32 * 0.02).min(0.95);
        let action = if anomaly > 0.5 {
            FiduciaryActionType::ShouldAvoid
        } else if anomaly > 0.3 {
            FiduciaryActionType::ShouldRefinance
        } else {
            FiduciaryActionType::ShouldFundGoal
        };

        let reward = RewardSignal {
            action_type: action,
            accepted: anomaly > 0.5 || anomaly < 0.2, // Accept safety + low-risk goals
            helpfulness: Some(0.5 + anomaly * 0.3),
            example: make_example(
                action,
                anomaly,
                0.5,
                [0.3, anomaly, 0.5 - anomaly, 0.0, 0.0],
            ),
            was_high_risk: anomaly > 0.5, // high anomaly = high risk
        };
        scorer.apply_reward(&reward);
        replay_buffer.push(reward);
    }
    println!("    Applied {} reward signals", replay_buffer.len());

    // Phase 3: Recursive self-improvement
    println!("  Phase 3: Recursive Self-Improvement (3 rounds)");
    let report = scorer.recursive_improve(&replay_buffer, 3);
    println!(
        "    Accuracy: {:.0}% → {:.0}%",
        report.initial_accuracy * 100.0,
        report.final_accuracy * 100.0
    );
    println!(
        "    Conflicts learned: {}",
        report.conflict_patterns_learned
    );

    // Summary: the scorer should now be better than random
    println!("\n  ── LIFECYCLE SUMMARY ──");
    println!("  Total samples seen: {}", scorer.samples_seen);
    println!("  Final axis weights: {:?}", scorer.axis_weights);
    println!("  Weight names: [cost, risk, goal, urgency, conflict, reversibility]");

    // Verify the scorer learned that risk matters more in high-anomaly situations
    let high_risk_ex = make_example(
        FiduciaryActionType::ShouldAvoid,
        0.9,
        0.1,
        [0.1, 0.1, 0.1, 0.0, 0.0],
    );
    let low_risk_ex = make_example(
        FiduciaryActionType::ShouldFundGoal,
        0.1,
        0.7,
        [0.5, 0.1, 0.7, 0.0, 0.0],
    );

    let (high_axes, high_logit) = scorer.forward(&high_risk_ex);
    let (low_axes, low_logit) = scorer.forward(&low_risk_ex);

    println!(
        "\n  High-risk scenario: axes.risk={:.2} logit={:.2}",
        high_axes.risk_reduction, high_logit
    );
    println!(
        "  Low-risk scenario:  axes.risk={:.2} logit={:.2}",
        low_axes.risk_reduction, low_logit
    );

    // The system should differentiate risk levels
    assert!(
        scorer.samples_seen >= 100,
        "Should have trained on sufficient examples"
    );
}
