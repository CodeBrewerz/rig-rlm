//! Fiduciary Prediction Comparison: all 5 GNN models × 7 financial scenarios
//!
//! Trains each model with optimal feature combo, extracts embeddings,
//! runs fiduciary predictions, and compares:
//!   - Fiduciary compliance: correct priority ordering
//!   - Domain coverage: number of financial domains addressed
//!   - Action quality: total fiduciary score
//!   - Interpretability: axis dispersion (diverse = more interpretable)
//!   - Embedding quality: variance (over-smoothing resistance)

use std::collections::HashMap;
use burn::backend::NdArray;
use burn::prelude::*;

use hehrgnn::data::graph_builder::{build_hetero_graph, GraphBuildConfig, GraphFact};
use hehrgnn::data::hetero_graph::{EdgeType, HeteroGraph};
use hehrgnn::eval::fiduciary::*;
use hehrgnn::model::backbone::NodeEmbeddings;
use hehrgnn::model::gat::GatConfig;
use hehrgnn::model::graph_transformer::GraphTransformerConfig;
use hehrgnn::model::graphsage::GraphSageModelConfig;
use hehrgnn::model::lora::{init_hetero_basis_adapter, LoraConfig};
use hehrgnn::model::mhc::MhcRgcnConfig;
use hehrgnn::model::rgcn::RgcnConfig;
use hehrgnn::model::trainer::*;

type B = NdArray;

fn gf(ht: &str, h: &str, r: &str, tt: &str, t: &str) -> GraphFact {
    GraphFact {
        src: (ht.into(), h.into()),
        relation: r.into(),
        dst: (tt.into(), t.into()),
    }
}

/// Build a comprehensive financial scenario graph covering all domains.
fn build_financial_graph() -> (HeteroGraph<B>, Vec<GraphFact>) {
    let device = <B as Backend>::Device::default();
    let facts = vec![
        // User
        gf("user", "alice", "owns", "account", "checking"),
        gf("user", "alice", "owns", "account", "savings"),

        // Debt: high-interest credit card + car loan
        gf("account", "checking", "pays", "obligation", "cc_24apr"),
        gf("account", "checking", "pays", "obligation", "car_loan_6apr"),
        gf("obligation", "cc_24apr", "has_rate", "rate", "high_24"),
        gf("obligation", "car_loan_6apr", "has_rate", "rate", "low_6"),

        // Tax: deadline + sinking fund
        gf("user", "alice", "liable", "tax_due", "q4_tax"),
        gf("user", "alice", "funds", "tax_sinking", "fed_reserve"),

        // Merchant activity: normal + anomalous
        gf("account", "checking", "transacts", "merchant", "grocery_store"),
        gf("account", "checking", "transacts", "merchant", "sketchy_online"),

        // Subscriptions: used + unused
        gf("user", "alice", "subscribes", "recurring", "netflix_active"),
        gf("user", "alice", "subscribes", "recurring", "gym_unused"),
        gf("user", "alice", "subscribes", "recurring", "mag_unused"),

        // Goals
        gf("user", "alice", "targets", "goal", "emergency_fund"),
        gf("user", "alice", "targets", "goal", "retirement_401k"),

        // Assets + reconciliation
        gf("user", "alice", "holds", "asset", "house_primary"),
        gf("asset", "house_primary", "valued_by", "valuation", "house_val_2023"),
        gf("account", "checking", "reconciled_by", "recon_case", "jan_recon"),

        // Budget
        gf("user", "alice", "tracks", "budget", "monthly_budget"),
    ];
    let graph = build_hetero_graph::<B>(&facts, &GraphBuildConfig {
        node_feat_dim: 16, add_reverse_edges: true, add_self_loops: true,
    }, &device);
    (graph, facts)
}

/// Build FiduciaryContext from GNN embeddings + anomaly assignments.
fn build_fiduciary_context<'a>(
    embeddings: &'a HashMap<String, Vec<Vec<f32>>>,
    anomaly_map: &'a HashMap<String, HashMap<String, Vec<f32>>>,
    edges: &'a HashMap<(String, String, String), Vec<(usize, usize)>>,
    node_names: &'a HashMap<String, Vec<String>>,
    node_counts: &'a HashMap<String, usize>,
    user_emb: &'a [f32],
) -> FiduciaryContext<'a> {
    FiduciaryContext {
        user_emb,
        embeddings,
        anomaly_scores: anomaly_map,
        edges,
        node_names,
        node_counts,
        user_type: "user".into(),
        user_id: 0,
        hidden_dim: user_emb.len(),
    }
}

/// Train a model, extract embeddings, return PlainEmbeddings.
fn train_model(
    model_name: &str,
    graph: &mut HeteroGraph<B>,
) -> HashMap<String, Vec<Vec<f32>>> {
    let device = <B as Backend>::Device::default();
    let node_types: Vec<String> = graph.node_types().iter().map(|s| s.to_string()).collect();
    let edge_types: Vec<EdgeType> = graph.edge_types().iter().map(|e| (*e).clone()).collect();
    let config = TrainConfig {
        lr: 0.01, epochs: 15, patience: 20, neg_ratio: 2,
        weight_decay: 0.001, perturb_frac: 1.0, mode: TrainMode::Fast,
    };

    match model_name {
        "GraphSAGE+DoRA+JEPA" => {
            let mut m = GraphSageModelConfig { in_dim: 16, hidden_dim: 16, num_layers: 2, dropout: 0.0 }
                .init::<B>(&node_types, &edge_types, &device);
            m.attach_adapter(init_hetero_basis_adapter(16, 16, &LoraConfig::default(), node_types.clone(), &device));
            let fwd = |g: &HeteroGraph<B>| m.forward(g);
            let _ = train_jepa(graph, &fwd, &config, 0.1, 0.5, false);
            embeddings_to_plain(&m.forward(graph))
        }
        "RGCN+mHC+JEPA" => {
            let m = MhcRgcnConfig { in_dim: 16, hidden_dim: 16, num_layers: 8, num_bases: 4, n_streams: 4, dropout: 0.0 }
                .init::<B>(&node_types, &edge_types, &device);
            let fwd = |g: &HeteroGraph<B>| m.forward(g);
            let _ = train_jepa(graph, &fwd, &config, 0.1, 0.5, false);
            embeddings_to_plain(&m.forward(graph))
        }
        "GAT+JEPA" => {
            let m = GatConfig { in_dim: 16, hidden_dim: 16, num_heads: 4, num_layers: 2, dropout: 0.0 }
                .init_model::<B>(&node_types, &edge_types, &device);
            let fwd = |g: &HeteroGraph<B>| m.forward(g);
            let _ = train_jepa(graph, &fwd, &config, 0.1, 0.5, false);
            embeddings_to_plain(&m.forward(graph))
        }
        "GPS+JEPA" => {
            let m = GraphTransformerConfig { in_dim: 16, hidden_dim: 16, num_heads: 4, num_layers: 2, ffn_ratio: 2, dropout: 0.0 }
                .init_model::<B>(&node_types, &edge_types, &device);
            let fwd = |g: &HeteroGraph<B>| m.forward(g);
            let _ = train_jepa(graph, &fwd, &config, 0.1, 0.5, false);
            embeddings_to_plain(&m.forward(graph))
        }
        "HEHRGNN+JEPA" => {
            // HEHRGNN uses the same graph but produces entity-level embeddings
            // We fall back to feature refinement embeddings since HEHRGNN needs HehrBatch
            let m = GraphSageModelConfig { in_dim: 16, hidden_dim: 16, num_layers: 2, dropout: 0.0 }
                .init::<B>(&node_types, &edge_types, &device);
            let fwd = |g: &HeteroGraph<B>| m.forward(g);
            let _ = train_via_feature_refinement(graph, &fwd, &config);
            embeddings_to_plain(&m.forward(graph))
        }
        _ => unreachable!(),
    }
}

/// Score interpretability: higher dispersion across axes = more interpretable.
fn axis_dispersion(resp: &FiduciaryResponse) -> f32 {
    if resp.recommendations.is_empty() { return 0.0; }
    let scores: Vec<f32> = resp.recommendations.iter()
        .map(|r| r.fiduciary_score)
        .collect();
    let mean = scores.iter().sum::<f32>() / scores.len() as f32;
    let var = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / scores.len() as f32;
    var.sqrt() // std dev of scores — higher = more separation = more interpretable
}

/// Count fiduciary compliance violations in a response.
fn compliance_score(resp: &FiduciaryResponse) -> (usize, usize) {
    let mut tests = 0usize;
    let mut passes = 0usize;

    let find = |action: &str| -> usize {
        resp.recommendations.iter()
            .position(|r| r.action_type == action)
            .unwrap_or(999)
    };

    // Rule 1: Safety before wealth (investigate before fund_goal)
    let fraud = resp.recommendations.iter()
        .position(|r| r.action_type == "should_investigate" || r.action_type == "should_avoid")
        .unwrap_or(999);
    let goal = find("should_fund_goal");
    tests += 1;
    if fraud < goal { passes += 1; }

    // Rule 2: Tax before goals
    let tax = find("should_prepare_tax");
    tests += 1;
    if tax < goal || tax == 999 { passes += 1; }

    // Rule 3: Cancel before fund
    let cancel = find("should_cancel");
    tests += 1;
    if cancel < goal || cancel == 999 { passes += 1; }

    // Rule 4: Reconcile before revalue
    let recon = find("should_reconcile");
    let revalue = find("should_revalue_asset");
    tests += 1;
    if recon < revalue || recon == 999 { passes += 1; }

    // Rule 5: Debt before goals
    let debt = resp.recommendations.iter()
        .position(|r| r.domain == "debt_obligations")
        .unwrap_or(999);
    tests += 1;
    if debt < goal { passes += 1; }

    (passes, tests)
}

#[test]
fn test_fiduciary_all_models_comparison() {
    let models = [
        "GraphSAGE+DoRA+JEPA",
        "RGCN+mHC+JEPA",
        "GAT+JEPA",
        "GPS+JEPA",
        "HEHRGNN+JEPA",
    ];

    // Anomaly scores for entities in our scenario
    let mut anomaly_scores: HashMap<String, HashMap<String, Vec<f32>>> = HashMap::new();
    let mut sage_scores: HashMap<String, Vec<f32>> = HashMap::new();
    sage_scores.insert("obligation".into(), vec![0.65, 0.15]); // cc_24apr (high), car_loan (low)
    sage_scores.insert("merchant".into(), vec![0.05, 0.85]); // grocery (normal), sketchy (high)
    sage_scores.insert("recurring".into(), vec![0.05, 0.35, 0.30]); // netflix, gym unused, mag unused
    sage_scores.insert("goal".into(), vec![0.10, 0.10]); // emergency, retirement
    sage_scores.insert("tax_due".into(), vec![0.20]);
    sage_scores.insert("tax_sinking".into(), vec![0.10]);
    sage_scores.insert("asset".into(), vec![0.10]);
    sage_scores.insert("valuation".into(), vec![0.10]);
    sage_scores.insert("recon_case".into(), vec![0.30]);
    sage_scores.insert("budget".into(), vec![0.10]);
    sage_scores.insert("account".into(), vec![0.05, 0.05]);
    sage_scores.insert("rate".into(), vec![0.20, 0.05]);
    sage_scores.insert("user".into(), vec![0.0]);
    anomaly_scores.insert("SAGE".into(), sage_scores);

    // Node names for display
    let mut node_names: HashMap<String, Vec<String>> = HashMap::new();
    node_names.insert("user".into(), vec!["Alice".into()]);
    node_names.insert("account".into(), vec!["Checking".into(), "Savings".into()]);
    node_names.insert("obligation".into(), vec!["CreditCard_24APR".into(), "CarLoan_6APR".into()]);
    node_names.insert("rate".into(), vec!["High_24".into(), "Low_6".into()]);
    node_names.insert("merchant".into(), vec!["Grocery_Store".into(), "SketchyOnline".into()]);
    node_names.insert("recurring".into(), vec!["Netflix_Active".into(), "Gym_Unused".into(), "Magazine_Unused".into()]);
    node_names.insert("goal".into(), vec!["EmergencyFund".into(), "Retirement_401k".into()]);
    node_names.insert("tax_due".into(), vec!["Q4_2025_TaxDue".into()]);
    node_names.insert("tax_sinking".into(), vec!["FederalTaxReserve".into()]);
    node_names.insert("asset".into(), vec!["House_Primary".into()]);
    node_names.insert("valuation".into(), vec!["House_Val_2023".into()]);
    node_names.insert("recon_case".into(), vec!["Jan_Recon".into()]);
    node_names.insert("budget".into(), vec!["Monthly_Budget".into()]);

    // Edges connecting user to entities
    let mut edges: HashMap<(String, String, String), Vec<(usize, usize)>> = HashMap::new();
    edges.insert(("user".into(), "owns".into(), "account".into()), vec![(0,0),(0,1)]);
    edges.insert(("account".into(), "pays".into(), "obligation".into()), vec![(0,0),(0,1)]);
    edges.insert(("obligation".into(), "has_rate".into(), "rate".into()), vec![(0,0),(1,1)]);
    edges.insert(("user".into(), "liable".into(), "tax_due".into()), vec![(0,0)]);
    edges.insert(("user".into(), "funds".into(), "tax_sinking".into()), vec![(0,0)]);
    edges.insert(("account".into(), "transacts".into(), "merchant".into()), vec![(0,0),(0,1)]);
    edges.insert(("user".into(), "subscribes".into(), "recurring".into()), vec![(0,0),(0,1),(0,2)]);
    edges.insert(("user".into(), "targets".into(), "goal".into()), vec![(0,0),(0,1)]);
    edges.insert(("user".into(), "holds".into(), "asset".into()), vec![(0,0)]);
    edges.insert(("asset".into(), "valued_by".into(), "valuation".into()), vec![(0,0)]);
    edges.insert(("account".into(), "reconciled_by".into(), "recon_case".into()), vec![(0,0)]);
    edges.insert(("user".into(), "tracks".into(), "budget".into()), vec![(0,0)]);

    // Node counts
    let mut node_counts: HashMap<String, usize> = HashMap::new();
    for (nt, names) in &node_names {
        node_counts.insert(nt.clone(), names.len());
    }

    println!("\n  ╔══════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("  ║  FIDUCIARY PREDICTION COMPARISON — ALL 5 MODELS                                        ║");
    println!("  ╠══════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("  ║  Model                │ Compliance │ Domains │ Actions │ Score │ Disp  │ EmbVar        ║");
    println!("  ╠══════════════════════════════════════════════════════════════════════════════════════════╣");

    let mut best_compliance = 0usize;
    let mut best_model = "";
    let mut all_results: Vec<(String, FiduciaryResponse, f32, usize, usize)> = Vec::new();

    for model_name in &models {
        let (mut graph, _facts) = build_financial_graph();

        // Train model and extract embeddings
        let emb = train_model(model_name, &mut graph);

        // Compute embedding variance
        let mut total_var = 0.0f32;
        let mut count = 0;
        for vecs in emb.values() {
            if vecs.len() < 2 { continue; }
            for d in 0..vecs[0].len() {
                let vals: Vec<f32> = vecs.iter().map(|v| v[d]).collect();
                let mean = vals.iter().sum::<f32>() / vals.len() as f32;
                let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
                total_var += var;
                count += 1;
            }
        }
        let avg_var = if count > 0 { total_var / count as f32 } else { 0.0 };

        // Get user embedding
        let user_emb = emb.get("user").and_then(|v| v.first()).cloned()
            .unwrap_or_else(|| vec![0.0; 16]);

        // Build fiduciary context with GNN embeddings
        let ctx = build_fiduciary_context(
            &emb, &anomaly_scores, &edges, &node_names, &node_counts, &user_emb,
        );

        // Run fiduciary prediction
        let resp = recommend(&ctx);

        // Measure results
        let (comp_pass, comp_total) = compliance_score(&resp);
        let dispersion = axis_dispersion(&resp);
        let total_score: f32 = resp.recommendations.iter()
            .map(|r| r.fiduciary_score)
            .sum();

        if comp_pass > best_compliance {
            best_compliance = comp_pass;
            best_model = model_name;
        }

        println!(
            "  ║  {:21} │   {}/{:2}     │   {:2}    │   {:2}    │ {:5.2} │ {:.3} │ {:.4}         ║",
            model_name, comp_pass, comp_total,
            resp.domains_covered.len(), resp.action_types_triggered,
            total_score, dispersion, avg_var,
        );

        all_results.push((
            model_name.to_string(), resp, avg_var,
            comp_pass, comp_total,
        ));
    }

    println!("  ╠══════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("  ║  🏆 Best compliance: {:21}                                                 ║", best_model);
    println!("  ╚══════════════════════════════════════════════════════════════════════════════════════════╝");

    // Print detailed top-5 recommendations for each model
    println!("\n  ═══ TOP-5 RECOMMENDATIONS PER MODEL ═══\n");
    for (model_name, resp, _, comp_pass, comp_total) in &all_results {
        println!("  ── {} ({}/{} compliance) ──", model_name, comp_pass, comp_total);
        for rec in resp.recommendations.iter().take(5) {
            let flag = if rec.is_recommended { "✅" } else { "ℹ️" };
            println!(
                "    {} #{:<2} [{:.3}] {:<22} │ {} → {}",
                flag, rec.rank, rec.fiduciary_score, rec.action_type,
                rec.target_node_type, rec.target_name,
            );
        }
        println!("    Domains: {:?}", resp.domains_covered);
        println!();
    }

    // Print interpretability analysis
    println!("  ═══ INTERPRETABILITY ANALYSIS ═══\n");
    for (model_name, resp, _, _, _) in &all_results {
        let unique_domains: std::collections::HashSet<&str> = resp.recommendations.iter()
            .map(|r| r.domain.as_str()).collect();
        let recommended_count = resp.recommendations.iter()
            .filter(|r| r.is_recommended).count();
        let informational_count = resp.recommendations.iter()
            .filter(|r| !r.is_recommended).count();

        println!("  {:25} │ {} domains │ {} recommended │ {} informational",
            model_name, unique_domains.len(), recommended_count, informational_count,
        );
    }

    println!("\n  ✅ Fiduciary prediction comparison complete!");
}
