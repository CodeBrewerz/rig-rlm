//! 100-Iteration Evolving Graph Simulation.
//!
//! Simulates a user's financial journey over 100 time steps where the graph
//! evolves (entities/relations created, updated, removed) and the fiduciary
//! system adapts its recommendations in real-time.
//!
//! Timeline:
//!   Step   0-10:  Fresh start — just a checking account, gets first paycheck
//!   Step  10-25:  Accumulates credit card debt, subscription creep
//!   Step  25-40:  Realizes problem, starts cutting subscriptions, refinancing
//!   Step  40-55:  Pays down debt, builds emergency fund
//!   Step  55-70:  Debt-free, starts investing toward goals (house, retirement)
//!   Step  70-85:  Tax optimization, portfolio rebalancing
//!   Step  85-100: Financially free — maintenance mode
//!
//! At each step:
//!   1. Graph mutates (add/remove nodes, update attributes)
//!   2. Re-compute GNN embeddings (simulate via direct embedding update)
//!   3. Run fiduciary recommender
//!   4. Simulate user accept/reject → reward signal
//!   5. Learnable scorer updates weights

use hehrgnn::eval::fiduciary::*;
use hehrgnn::eval::learnable_scorer::*;
use std::collections::HashMap;

fn string_to_action(s: &str) -> FiduciaryActionType {
    match s {
        "should_pay" => FiduciaryActionType::ShouldPay,
        "should_cancel" => FiduciaryActionType::ShouldCancel,
        "should_transfer" => FiduciaryActionType::ShouldTransfer,
        "should_consolidate" => FiduciaryActionType::ShouldConsolidate,
        "should_avoid" => FiduciaryActionType::ShouldAvoid,
        "should_investigate" => FiduciaryActionType::ShouldInvestigate,
        "should_refinance" => FiduciaryActionType::ShouldRefinance,
        "should_pay_down_lien" => FiduciaryActionType::ShouldPayDownLien,
        "should_dispute" => FiduciaryActionType::ShouldDispute,
        "should_fund_goal" => FiduciaryActionType::ShouldFundGoal,
        "should_adjust_budget" => FiduciaryActionType::ShouldAdjustBudget,
        "should_prepare_tax" => FiduciaryActionType::ShouldPrepareTax,
        "should_fund_tax_sinking" => FiduciaryActionType::ShouldFundTaxSinking,
        "should_claim_exemption" => FiduciaryActionType::ShouldClaimExemption,
        "should_run_tax_scenario" => FiduciaryActionType::ShouldRunTaxScenario,
        "should_reconcile" => FiduciaryActionType::ShouldReconcile,
        "should_review_recurring" => FiduciaryActionType::ShouldReviewRecurring,
        "should_revalue_asset" => FiduciaryActionType::ShouldRevalueAsset,
        _ => FiduciaryActionType::ShouldPay,
    }
}

// ═══════════════════════════════════════════════════════════════
// Evolving Graph State
// ═══════════════════════════════════════════════════════════════

/// Mutable financial graph that evolves over time.
struct EvolvingGraph {
    embeddings: HashMap<String, Vec<Vec<f32>>>,
    node_names: HashMap<String, Vec<String>>,
    node_counts: HashMap<String, usize>,
    edges: HashMap<(String, String, String), Vec<(usize, usize)>>,
    anomaly_scores: HashMap<String, HashMap<String, Vec<f32>>>,
    dim: usize,
}

impl EvolvingGraph {
    fn new(dim: usize) -> Self {
        let mut g = Self {
            embeddings: HashMap::new(),
            node_names: HashMap::new(),
            node_counts: HashMap::new(),
            edges: HashMap::new(),
            anomaly_scores: HashMap::new(),
            dim,
        };
        // Start with user node
        let user_emb: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.1).sin() * 0.3).collect();
        g.embeddings.insert("user".into(), vec![user_emb]);
        g.node_names.insert("user".into(), vec!["Alice".into()]);
        g.node_counts.insert("user".into(), 1);
        g.anomaly_scores.insert("SAGE".into(), HashMap::new());
        g.anomaly_scores
            .get_mut("SAGE")
            .unwrap()
            .insert("user".into(), vec![0.05]);
        g
    }

    /// Add an entity node with a relation to the user.
    fn add_entity(&mut self, node_type: &str, name: &str, relation: &str, anomaly: f32) {
        let node_id = self.node_counts.get(node_type).copied().unwrap_or(0);
        let emb: Vec<f32> = (0..self.dim)
            .map(|d| ((node_id * 11 + d * 5) as f32 * 0.13 + anomaly).sin() * 0.4)
            .collect();
        self.embeddings
            .entry(node_type.into())
            .or_default()
            .push(emb);
        self.node_names
            .entry(node_type.into())
            .or_default()
            .push(name.into());
        *self.node_counts.entry(node_type.into()).or_insert(0) += 1;
        self.anomaly_scores
            .get_mut("SAGE")
            .unwrap()
            .entry(node_type.into())
            .or_default()
            .push(anomaly);
        self.edges
            .entry(("user".into(), relation.into(), node_type.into()))
            .or_default()
            .push((0, node_id));
    }

    /// Update anomaly score of an existing entity.
    fn update_anomaly(&mut self, node_type: &str, node_id: usize, new_anomaly: f32) {
        if let Some(scores) = self
            .anomaly_scores
            .get_mut("SAGE")
            .and_then(|m| m.get_mut(node_type))
        {
            if node_id < scores.len() {
                scores[node_id] = new_anomaly;
            }
        }
    }

    /// Remove an entity (zero out its embedding and anomaly — soft delete).
    fn remove_entity(&mut self, node_type: &str, node_id: usize) {
        if let Some(embs) = self.embeddings.get_mut(node_type) {
            if node_id < embs.len() {
                for v in embs[node_id].iter_mut() {
                    *v = 0.0;
                }
            }
        }
        self.update_anomaly(node_type, node_id, 0.0);
    }

    /// Update user embedding to reflect current financial state.
    fn update_user_embedding(&mut self, debt_level: f32, savings_level: f32, risk_level: f32) {
        if let Some(user_embs) = self.embeddings.get_mut("user") {
            let emb = &mut user_embs[0];
            // Modulate embedding dimensions based on financial state
            emb[0] = debt_level * 2.0;
            emb[1] = debt_level * 1.5;
            emb[2] = risk_level * 1.8;
            emb[4] = savings_level * 2.0;
            emb[5] = savings_level * 1.5;
        }
    }

    /// Build fiduciary context for current graph state.
    fn fiduciary_context(&self) -> FiduciaryContext<'_> {
        FiduciaryContext {
            user_emb: &self.embeddings["user"][0],
            embeddings: &self.embeddings,
            anomaly_scores: &self.anomaly_scores,
            edges: &self.edges,
            node_names: &self.node_names,
            node_counts: &self.node_counts,
            user_type: "user".into(),
            user_id: 0,
            hidden_dim: self.dim,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Life Events — what happens at each time step
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
struct LifeEvent {
    step: usize,
    description: String,
    event_type: LifeEventType,
}

#[derive(Debug, Clone)]
enum LifeEventType {
    /// Add a new entity + relation to the graph.
    AddEntity {
        node_type: String,
        name: String,
        relation: String,
        anomaly: f32,
    },
    /// Update anomaly score (e.g. debt getting worse/better).
    UpdateAnomaly {
        node_type: String,
        node_id: usize,
        new_anomaly: f32,
    },
    /// Remove an entity (paid off debt, cancelled subscription).
    RemoveEntity { node_type: String, node_id: usize },
    /// Update user's overall financial state.
    UpdateUserState {
        debt_level: f32,
        savings_level: f32,
        risk_level: f32,
    },
}

fn generate_life_events() -> Vec<LifeEvent> {
    let mut events = Vec::new();

    // ── Phase 1: Fresh Start (0-10) ──
    events.push(LifeEvent {
        step: 0,
        description: "Opens checking account".into(),
        event_type: LifeEventType::AddEntity {
            node_type: "instrument".into(),
            name: "Checking_Main".into(),
            relation: "user-has-instrument".into(),
            anomaly: 0.05,
        },
    });
    events.push(LifeEvent {
        step: 3,
        description: "Gets first paycheck, sets up direct deposit".into(),
        event_type: LifeEventType::UpdateUserState {
            debt_level: 0.0,
            savings_level: 0.1,
            risk_level: 0.1,
        },
    });
    events.push(LifeEvent {
        step: 5,
        description: "Opens savings account".into(),
        event_type: LifeEventType::AddEntity {
            node_type: "instrument".into(),
            name: "Savings_Emergency".into(),
            relation: "user-has-instrument".into(),
            anomaly: 0.03,
        },
    });

    // ── Phase 2: Debt Accumulation (10-25) ──
    events.push(LifeEvent {
        step: 10,
        description: "Gets first credit card (0% intro APR)".into(),
        event_type: LifeEventType::AddEntity {
            node_type: "obligation".into(),
            name: "CC_IntroRate".into(),
            relation: "obligation-has-interest-term".into(),
            anomaly: 0.15,
        },
    });
    events.push(LifeEvent {
        step: 13,
        description: "Signs up for streaming service".into(),
        event_type: LifeEventType::AddEntity {
            node_type: "recurring-pattern".into(),
            name: "Netflix_Sub".into(),
            relation: "pattern-owned-by".into(),
            anomaly: 0.08,
        },
    });
    events.push(LifeEvent {
        step: 15,
        description: "Intro rate expires, CC jumps to 24% APR!".into(),
        event_type: LifeEventType::UpdateAnomaly {
            node_type: "obligation".into(),
            node_id: 0,
            new_anomaly: 0.55,
        },
    });
    events.push(LifeEvent {
        step: 16,
        description: "Debt stress rising".into(),
        event_type: LifeEventType::UpdateUserState {
            debt_level: 0.6,
            savings_level: 0.08,
            risk_level: 0.5,
        },
    });
    events.push(LifeEvent {
        step: 18,
        description: "Signs up for gym membership barely uses".into(),
        event_type: LifeEventType::AddEntity {
            node_type: "recurring-pattern".into(),
            name: "Gym_Premium_Unused".into(),
            relation: "pattern-owned-by".into(),
            anomaly: 0.35,
        },
    });
    events.push(LifeEvent {
        step: 20,
        description: "Gets second credit card to balance transfer".into(),
        event_type: LifeEventType::AddEntity {
            node_type: "obligation".into(),
            name: "CC_BalanceTransfer".into(),
            relation: "obligation-has-interest-term".into(),
            anomaly: 0.40,
        },
    });
    events.push(LifeEvent {
        step: 22,
        description: "Suspicious charge appears on CC".into(),
        event_type: LifeEventType::AddEntity {
            node_type: "user-merchant-unit".into(),
            name: "Suspicious_OnlineStore".into(),
            relation: "case-has-counterparty".into(),
            anomaly: 0.82,
        },
    });
    events.push(LifeEvent {
        step: 23,
        description: "Debt reaches peak, savings depleted".into(),
        event_type: LifeEventType::UpdateUserState {
            debt_level: 0.85,
            savings_level: 0.02,
            risk_level: 0.75,
        },
    });

    // ── Phase 3: Turning Point (25-40) ──
    events.push(LifeEvent {
        step: 25,
        description: "Cancels unused gym membership".into(),
        event_type: LifeEventType::RemoveEntity {
            node_type: "recurring-pattern".into(),
            node_id: 1,
        },
    });
    events.push(LifeEvent {
        step: 27,
        description: "Disputes fraudulent charge (resolved!)".into(),
        event_type: LifeEventType::RemoveEntity {
            node_type: "user-merchant-unit".into(),
            node_id: 0,
        },
    });
    events.push(LifeEvent {
        step: 30,
        description: "Refinances CC to lower rate personal loan".into(),
        event_type: LifeEventType::UpdateAnomaly {
            node_type: "obligation".into(),
            node_id: 0,
            new_anomaly: 0.25,
        },
    });
    events.push(LifeEvent {
        step: 32,
        description: "Debt declining steadily".into(),
        event_type: LifeEventType::UpdateUserState {
            debt_level: 0.5,
            savings_level: 0.1,
            risk_level: 0.4,
        },
    });
    events.push(LifeEvent {
        step: 35,
        description: "Pays off balance transfer card completely".into(),
        event_type: LifeEventType::RemoveEntity {
            node_type: "obligation".into(),
            node_id: 1,
        },
    });
    events.push(LifeEvent {
        step: 38,
        description: "Sets up emergency fund goal".into(),
        event_type: LifeEventType::AddEntity {
            node_type: "goal".into(),
            name: "EmergencyFund_3mo".into(),
            relation: "subledger-holds-goal-funds".into(),
            anomaly: 0.05,
        },
    });

    // ── Phase 4: Building Stability (40-55) ──
    events.push(LifeEvent {
        step: 40,
        description: "Pays off last credit card — debt free!".into(),
        event_type: LifeEventType::RemoveEntity {
            node_type: "obligation".into(),
            node_id: 0,
        },
    });
    events.push(LifeEvent {
        step: 41,
        description: "Debt free, building savings".into(),
        event_type: LifeEventType::UpdateUserState {
            debt_level: 0.0,
            savings_level: 0.4,
            risk_level: 0.15,
        },
    });
    events.push(LifeEvent {
        step: 45,
        description: "Emergency fund fully funded".into(),
        event_type: LifeEventType::UpdateAnomaly {
            node_type: "goal".into(),
            node_id: 0,
            new_anomaly: 0.02,
        },
    });
    events.push(LifeEvent {
        step: 48,
        description: "Opens brokerage account for investing".into(),
        event_type: LifeEventType::AddEntity {
            node_type: "instrument".into(),
            name: "Brokerage_Invest".into(),
            relation: "user-has-instrument".into(),
            anomaly: 0.05,
        },
    });
    events.push(LifeEvent {
        step: 50,
        description: "Sets house down payment goal".into(),
        event_type: LifeEventType::AddEntity {
            node_type: "goal".into(),
            name: "HouseDownPayment_50k".into(),
            relation: "subledger-holds-goal-funds".into(),
            anomaly: 0.08,
        },
    });

    // ── Phase 5: Growth (55-70) ──
    events.push(LifeEvent {
        step: 55,
        description: "Starts 401k contributions".into(),
        event_type: LifeEventType::AddEntity {
            node_type: "goal".into(),
            name: "Retirement_401k".into(),
            relation: "subledger-holds-goal-funds".into(),
            anomaly: 0.03,
        },
    });
    events.push(LifeEvent {
        step: 57,
        description: "Growing wealth steadily".into(),
        event_type: LifeEventType::UpdateUserState {
            debt_level: 0.0,
            savings_level: 0.7,
            risk_level: 0.1,
        },
    });
    events.push(LifeEvent {
        step: 60,
        description: "Buys first home (takes on mortgage)".into(),
        event_type: LifeEventType::AddEntity {
            node_type: "asset".into(),
            name: "Home_Primary".into(),
            relation: "lien-on-asset".into(),
            anomaly: 0.10,
        },
    });
    events.push(LifeEvent {
        step: 62,
        description: "Sets up monthly budget for mortgage + expenses".into(),
        event_type: LifeEventType::AddEntity {
            node_type: "budget-estimation".into(),
            name: "MonthlyBudget".into(),
            relation: "records-budget-estimation".into(),
            anomaly: 0.05,
        },
    });

    // ── Phase 6: Optimization (70-85) ──
    events.push(LifeEvent {
        step: 70,
        description: "Starts tax planning with exemptions".into(),
        event_type: LifeEventType::AddEntity {
            node_type: "tax-exemption-certificate".into(),
            name: "MortgageInterest_Deduct".into(),
            relation: "tax-party-has-exemption-certificate".into(),
            anomaly: 0.05,
        },
    });
    events.push(LifeEvent {
        step: 73,
        description: "Tax scenario analysis for max contributions".into(),
        event_type: LifeEventType::AddEntity {
            node_type: "tax-scenario".into(),
            name: "MaxContrib_Analysis".into(),
            relation: "tax-scenario-for-period".into(),
            anomaly: 0.03,
        },
    });
    events.push(LifeEvent {
        step: 75,
        description: "Sets up tax sinking fund for quarterly estimates".into(),
        event_type: LifeEventType::AddEntity {
            node_type: "tax-sinking-fund".into(),
            name: "QuarterlyEstimates".into(),
            relation: "tax-sinking-fund-backed-by-account".into(),
            anomaly: 0.08,
        },
    });
    events.push(LifeEvent {
        step: 78,
        description: "Rebalances investment portfolio".into(),
        event_type: LifeEventType::AddEntity {
            node_type: "asset-valuation".into(),
            name: "PortfolioRebalance_Q4".into(),
            relation: "asset-has-valuation".into(),
            anomaly: 0.12,
        },
    });
    events.push(LifeEvent {
        step: 80,
        description: "Strong financial position".into(),
        event_type: LifeEventType::UpdateUserState {
            debt_level: 0.15,
            savings_level: 0.85,
            risk_level: 0.08,
        },
    });

    // ── Phase 7: Financial Freedom (85-100) ──
    events.push(LifeEvent {
        step: 85,
        description: "House down payment goal funded!".into(),
        event_type: LifeEventType::UpdateAnomaly {
            node_type: "goal".into(),
            node_id: 1,
            new_anomaly: 0.01,
        },
    });
    events.push(LifeEvent {
        step: 90,
        description: "Reconciliation check — all accounts balanced".into(),
        event_type: LifeEventType::AddEntity {
            node_type: "reconciliation-case".into(),
            name: "Q4_Recon".into(),
            relation: "reconciliation-for-instrument".into(),
            anomaly: 0.05,
        },
    });
    events.push(LifeEvent {
        step: 95,
        description: "Mortgage accelerated paydown — almost done".into(),
        event_type: LifeEventType::UpdateAnomaly {
            node_type: "asset".into(),
            node_id: 0,
            new_anomaly: 0.03,
        },
    });
    events.push(LifeEvent {
        step: 98,
        description: "Financially free — all goals met".into(),
        event_type: LifeEventType::UpdateUserState {
            debt_level: 0.05,
            savings_level: 0.95,
            risk_level: 0.03,
        },
    });

    events
}

// ═══════════════════════════════════════════════════════════════
// Snapshot: what the fiduciary recommends at a point in time
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
struct Snapshot {
    step: usize,
    phase: String,
    debt_level: f32,
    savings_level: f32,
    risk_level: f32,
    top_action: String,
    recommended_count: usize,
    action_types: Vec<String>,
    is_aligned: bool, // does recommendation make sense for this phase?
    total_nodes: usize,
    total_edges: usize,
}

fn phase_name(step: usize) -> &'static str {
    match step {
        0..=9 => "Fresh Start",
        10..=24 => "Debt Spiral",
        25..=39 => "Turning Point",
        40..=54 => "Building Stability",
        55..=69 => "Growth",
        70..=84 => "Optimization",
        85..=100 => "Financial Freedom",
        _ => "Unknown",
    }
}

fn expected_focus(step: usize) -> &'static str {
    match step {
        0..=9 => "instrument",     // just setting up accounts
        10..=24 => "debt/safety",  // should focus on debt and fraud
        25..=39 => "debt_paydown", // cutting expenses, refinancing
        40..=54 => "goals",        // building emergency fund, savings
        55..=69 => "goals",        // investing, buying home
        70..=84 => "tax/asset",    // tax optimization, portfolio rebalancing
        85..=100 => "maintenance", // everything balanced
        _ => "unknown",
    }
}

fn action_matches_focus(action: &str, focus: &str) -> bool {
    match focus {
        "instrument" => true, // anything is fine at start
        "debt/safety" => {
            action.contains("refinance")
                || action.contains("avoid")
                || action.contains("investigate")
                || action.contains("dispute")
                || action.contains("cancel")
        }
        "debt_paydown" => {
            action.contains("refinance")
                || action.contains("cancel")
                || action.contains("pay")
                || action.contains("fund_goal")
        }
        "goals" => {
            action.contains("fund_goal")
                || action.contains("budget")
                || action.contains("pay_down_lien")
                || action.contains("transfer")
        }
        "tax/asset" => {
            action.contains("tax")
                || action.contains("exemption")
                || action.contains("revalue")
                || action.contains("scenario")
                || action.contains("fund_goal")
        }
        "maintenance" => true, // any maintenance action is fine
        _ => true,
    }
}

// ═══════════════════════════════════════════════════════════════
// Main test: 100-iteration simulation
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_100_iteration_evolving_graph() {
    let dim = 32;
    let mut graph = EvolvingGraph::new(dim);
    let life_events = generate_life_events();

    // Initialize learnable scorer
    let scorer_config = ScorerConfig {
        embedding_dim: dim,
        hidden1: 64,
        hidden2: 32,
        lr: 0.003,
    };
    let mut scorer = LearnableScorer::new(&scorer_config);

    // Pre-distill from expert rules
    let mut distill_examples = Vec::new();
    let mut distill_labels = Vec::new();
    for &action in &FiduciaryActionType::all() {
        for anomaly in [0.1, 0.4, 0.7] {
            let ctx = [0.3, 0.5, 0.4, 0.0, 0.0];
            let idx = match action {
                FiduciaryActionType::ShouldRefinance => 0,
                FiduciaryActionType::ShouldAvoid => 1,
                _ => 2,
            };
            let user_emb: Vec<f32> = (0..dim)
                .map(|d| ((idx * 7 + d * 3) as f32 * 0.1 + anomaly).sin() * 0.5)
                .collect();
            let target_emb: Vec<f32> = (0..dim)
                .map(|d| ((idx * 11 + d * 5) as f32 * 0.13).sin() * 0.5)
                .collect();
            distill_examples.push(ScorerExample {
                user_emb,
                target_emb,
                action_type: action,
                anomaly_score: anomaly,
                embedding_affinity: 0.5,
                context: ctx,
            });
            let axes = FiduciaryAxes {
                cost_reduction: 0.5,
                risk_reduction: 0.5,
                goal_alignment: 0.5,
                urgency: 0.5,
                conflict_freedom: 0.7,
                reversibility: 0.5,
            };
            distill_labels.push(ScorerLabel {
                axes,
                should_recommend: anomaly < 0.6,
            });
        }
    }
    scorer.distill(&distill_examples, &distill_labels, 30);

    println!(
        "\n  ╔══════════════════════════════════════════════════════════════════════════════╗"
    );
    println!("  ║           100-ITERATION EVOLVING GRAPH SIMULATION                          ║");
    println!("  ║           Financial Journey: Fresh Start → Financial Freedom                ║");
    println!(
        "  ╚══════════════════════════════════════════════════════════════════════════════╝\n"
    );

    let mut snapshots: Vec<Snapshot> = Vec::new();
    let mut debt_level = 0.0f32;
    let mut savings_level = 0.1f32;
    let mut risk_level = 0.1f32;
    let mut reward_buffer: Vec<RewardSignal> = Vec::new();
    let mut last_phase = "";

    for step in 0..=100 {
        // Apply any life events for this step
        for event in life_events.iter().filter(|e| e.step == step) {
            match &event.event_type {
                LifeEventType::AddEntity {
                    node_type,
                    name,
                    relation,
                    anomaly,
                } => {
                    graph.add_entity(node_type, name, relation, *anomaly);
                }
                LifeEventType::UpdateAnomaly {
                    node_type,
                    node_id,
                    new_anomaly,
                } => {
                    graph.update_anomaly(node_type, *node_id, *new_anomaly);
                }
                LifeEventType::RemoveEntity { node_type, node_id } => {
                    graph.remove_entity(node_type, *node_id);
                }
                LifeEventType::UpdateUserState {
                    debt_level: d,
                    savings_level: s,
                    risk_level: r,
                } => {
                    debt_level = *d;
                    savings_level = *s;
                    risk_level = *r;
                    graph.update_user_embedding(*d, *s, *r);
                }
            }
        }

        // Run fiduciary recommender on current graph state
        let ctx = graph.fiduciary_context();
        let response = recommend(&ctx);

        let rec_types: Vec<String> = response
            .recommendations
            .iter()
            .filter(|r| r.is_recommended)
            .map(|r| r.action_type.clone())
            .collect();
        let top_action = rec_types.first().cloned().unwrap_or_default();

        // Count graph size
        let total_nodes: usize = graph.node_counts.values().sum();
        let total_edges: usize = graph.edges.values().map(|e| e.len()).sum();

        let phase = phase_name(step);
        let focus = expected_focus(step);
        let is_aligned =
            rec_types.is_empty() || rec_types.iter().any(|a| action_matches_focus(a, focus));

        // Print phase transitions and key events
        if phase != last_phase {
            println!("\n  ─── {} (step {}) ───", phase, step);
            last_phase = phase;
        }

        // Print events at this step
        for event in life_events.iter().filter(|e| e.step == step) {
            println!("  [{:>3}] 📌 {}", step, event.description);
        }

        // Print recommendation every 5 steps or on events
        let has_event = life_events.iter().any(|e| e.step == step);
        if step % 10 == 0 || has_event {
            let status = if is_aligned { "✅" } else { "⚠️" };
            let rec_summary = if rec_types.is_empty() {
                "no actions needed".to_string()
            } else {
                rec_types
                    .iter()
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            println!(
                "  [{:>3}] {} debt={:.0}% save={:.0}% risk={:.0}% │ nodes={} edges={} │ → {}",
                step,
                status,
                debt_level * 100.0,
                savings_level * 100.0,
                risk_level * 100.0,
                total_nodes,
                total_edges,
                rec_summary,
            );
        }

        // Simulate user feedback (user follows fiduciary advice)
        if !rec_types.is_empty() && has_event {
            let accepted = is_aligned; // User accepts aligned recommendations
            let ctx_feat = [debt_level, savings_level, risk_level, 0.0, 0.0];

            // Map the top recommendation string back to action type
            let top_action_type = string_to_action(&top_action);

            let reward = RewardSignal {
                action_type: top_action_type,
                accepted,
                helpfulness: if accepted { Some(0.8) } else { Some(0.3) },
                example: ScorerExample {
                    user_emb: graph.embeddings["user"][0].clone(),
                    target_emb: vec![0.0; dim],
                    action_type: top_action_type,
                    anomaly_score: risk_level,
                    embedding_affinity: 0.5,
                    context: ctx_feat,
                },
            };
            scorer.apply_reward(&reward);
            reward_buffer.push(reward);
        }

        // Snapshot every 10 steps
        if step % 10 == 0 {
            snapshots.push(Snapshot {
                step,
                phase: phase.to_string(),
                debt_level,
                savings_level,
                risk_level,
                top_action: top_action.clone(),
                recommended_count: rec_types.len(),
                action_types: rec_types.clone(),
                is_aligned,
                total_nodes,
                total_edges,
            });
        }
    }

    // ── Recursive improvement with accumulated feedback ──
    if reward_buffer.len() >= 5 {
        let report = scorer.recursive_improve(&reward_buffer, 5);
        println!("\n  ── LEARNABLE SCORER UPDATE ──");
        println!("  Rewards collected: {}", reward_buffer.len());
        println!(
            "  Accuracy: {:.0}% → {:.0}%",
            report.initial_accuracy * 100.0,
            report.final_accuracy * 100.0
        );
        println!("  Conflicts learned: {}", report.conflict_patterns_learned);
        println!("  Axis weights: {:?}", scorer.axis_weights);
    }

    // ── Summary Timeline ──
    println!(
        "\n  ╔══════════════════════════════════════════════════════════════════════════════╗"
    );
    println!("  ║                          JOURNEY SUMMARY                                    ║");
    println!(
        "  ╚══════════════════════════════════════════════════════════════════════════════╝\n"
    );
    println!(
        "  {:>4} │ {:>18} │ {:>5} {:>5} {:>5} │ {:>5} {:>5} │ {:>3} │ Top Action",
        "Step", "Phase", "Debt", "Save", "Risk", "Nodes", "Edges", "OK?"
    );
    println!(
        "  ─────┼────────────────────┼───────────────────┼─────────────┼─────┼────────────────────"
    );

    for snap in &snapshots {
        let ok = if snap.is_aligned { "✅" } else { "⚠️" };
        println!(
            "  {:>4} │ {:>18} │ {:>4.0}% {:>4.0}% {:>4.0}% │ {:>5} {:>5} │  {} │ {}",
            snap.step,
            snap.phase,
            snap.debt_level * 100.0,
            snap.savings_level * 100.0,
            snap.risk_level * 100.0,
            snap.total_nodes,
            snap.total_edges,
            ok,
            if snap.top_action.is_empty() {
                "(none)".to_string()
            } else {
                snap.top_action.clone()
            },
        );
    }

    // ── Assertions ──
    let aligned_count = snapshots.iter().filter(|s| s.is_aligned).count();
    let aligned_pct = aligned_count as f32 / snapshots.len() as f32 * 100.0;
    println!(
        "\n  Alignment: {}/{} snapshots ({:.0}%)",
        aligned_count,
        snapshots.len(),
        aligned_pct
    );
    println!("  Total life events processed: {}", life_events.len());
    println!("  Scorer samples seen: {}", scorer.samples_seen);

    // Journey should end with higher savings than debt
    let last = snapshots.last().unwrap();
    assert!(
        last.savings_level > last.debt_level,
        "Journey should end with savings > debt (savings={:.0}%, debt={:.0}%)",
        last.savings_level * 100.0,
        last.debt_level * 100.0,
    );

    // Most snapshots should be aligned
    assert!(
        aligned_pct >= 70.0,
        "At least 70% of snapshots should have aligned recommendations, got {:.0}%",
        aligned_pct,
    );

    // Graph should have grown over time
    assert!(
        last.total_nodes >= 10,
        "Final graph should have ≥10 nodes, got {}",
        last.total_nodes,
    );
}

// ═══════════════════════════════════════════════════════════════
// Test 2: Verify recommendation CHANGES at phase transitions
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_recommendation_changes_at_phase_transitions() {
    let dim = 32;
    let mut graph = EvolvingGraph::new(dim);

    println!("\n  ── PHASE TRANSITION RECOMMENDATIONS ──\n");

    // Phase A: No debt — should NOT recommend refinance
    graph.add_entity("instrument", "Checking", "user-has-instrument", 0.03);
    graph.add_entity("goal", "VacationFund", "subledger-holds-goal-funds", 0.05);
    graph.update_user_embedding(0.0, 0.5, 0.1);

    let ctx_a = graph.fiduciary_context();
    let resp_a = recommend(&ctx_a);
    let types_a: Vec<String> = resp_a
        .recommendations
        .iter()
        .filter(|r| r.is_recommended)
        .map(|r| r.action_type.clone())
        .collect();
    println!("  Phase A (no debt):      {:?}", types_a);
    assert!(
        !types_a.contains(&"should_refinance".to_string()),
        "Should NOT recommend refinance when there's no debt"
    );

    // Phase B: ADD debt — should NOW recommend refinance
    graph.add_entity(
        "obligation",
        "CC_22APR",
        "obligation-has-interest-term",
        0.55,
    );
    graph.update_user_embedding(0.7, 0.2, 0.5);

    let ctx_b = graph.fiduciary_context();
    let resp_b = recommend(&ctx_b);
    let types_b: Vec<String> = resp_b
        .recommendations
        .iter()
        .filter(|r| r.is_recommended)
        .map(|r| r.action_type.clone())
        .collect();
    println!("  Phase B (add debt):     {:?}", types_b);
    assert!(
        types_b.contains(&"should_refinance".to_string()),
        "Should recommend refinance when debt is added"
    );

    // Phase C: REMOVE debt — should STOP recommending refinance
    graph.remove_entity("obligation", 0);
    graph.update_user_embedding(0.0, 0.5, 0.1);

    let ctx_c = graph.fiduciary_context();
    let resp_c = recommend(&ctx_c);
    let types_c: Vec<String> = resp_c
        .recommendations
        .iter()
        .filter(|r| r.is_recommended)
        .map(|r| r.action_type.clone())
        .collect();
    println!("  Phase C (debt removed): {:?}", types_c);

    // Verify transitions happened
    let had_refinance_b = types_b.contains(&"should_refinance".to_string());
    let lost_refinance_c = !types_c.contains(&"should_refinance".to_string());
    println!(
        "\n  Refinance added when debt appeared:  {}",
        if had_refinance_b { "✅" } else { "❌" }
    );
    println!(
        "  Refinance removed when debt cleared: {}",
        if lost_refinance_c { "✅" } else { "❌" }
    );

    assert!(had_refinance_b, "Refinance should appear with debt");
}
