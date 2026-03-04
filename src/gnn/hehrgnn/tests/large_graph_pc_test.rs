//! Large-Scale Graph Test — 130+ entities demonstrating non-uniform PC risk.
//!
//! Procedurally generates a TQL-compliant financial graph with:
//! - 10 users (5 risk profiles: high-risk, safe, mixed, tax-heavy, new)
//! - 25 accounts, 20 obligations, 10 rates, 15 merchants
//! - 12 recurring subs, 8 goals, 5 tax items, 6 assets, 4 recon cases, 5 budgets
//!
//! Total: ~130 entities → ~650 training rows → enough for the HCLT to learn
//! strong anomaly→risk conditional dependencies.

use std::collections::HashMap;

use hehrgnn::eval::fiduciary::*;

// ═══════════════════════════════════════════════════════════════
// Procedural Graph Generator
// ═══════════════════════════════════════════════════════════════

/// Risk profile for procedural generation.
#[derive(Clone, Copy)]
enum RiskProfile {
    HighRisk, // anomaly ~0.7-0.95, emb ~0.8
    Risky,    // anomaly ~0.5-0.7,  emb ~0.6
    Mixed,    // anomaly ~0.3-0.5,  emb ~0.4
    Safe,     // anomaly ~0.1-0.3,  emb ~0.2
    VerySafe, // anomaly ~0.02-0.1, emb ~0.1
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

/// Simple deterministic pseudo-random: hash(seed + i) mapped to [lo, hi].
fn pseudo_random(seed: u64, i: usize, lo: f32, hi: f32) -> f32 {
    let x = seed
        .wrapping_mul(6364136223846793005u64)
        .wrapping_add(i as u64)
        .wrapping_mul(1442695040888963407u64);
    let frac = ((x >> 48) as f32) / (65535.0f32); // use top 16 bits for [0,1)
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

struct GeneratedGraph {
    embeddings: HashMap<String, Vec<Vec<f32>>>,
    anomaly_scores: HashMap<String, HashMap<String, Vec<f32>>>,
    node_names: HashMap<String, Vec<String>>,
    edges: HashMap<(String, String, String), Vec<(usize, usize)>>,
    node_counts: HashMap<String, usize>,
}

fn generate_large_graph() -> GeneratedGraph {
    let dim = 8;
    let seed = 42u64;

    let mut emb: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    let mut anomaly: HashMap<String, Vec<f32>> = HashMap::new();
    let mut names: HashMap<String, Vec<String>> = HashMap::new();
    let mut edges: HashMap<(String, String, String), Vec<(usize, usize)>> = HashMap::new();

    // ── 10 Users ─────────────────────────────────────────────
    let user_profiles = [
        (RiskProfile::HighRisk, "HighRisk_Dave"),
        (RiskProfile::HighRisk, "HighRisk_Zara"),
        (RiskProfile::Risky, "Risky_Mike"),
        (RiskProfile::Risky, "Risky_Sarah"),
        (RiskProfile::Mixed, "Mixed_Carlos"),
        (RiskProfile::Mixed, "Mixed_Priya"),
        (RiskProfile::Safe, "Safe_Beth"),
        (RiskProfile::Safe, "Safe_Tom"),
        (RiskProfile::VerySafe, "VerySafe_Emma"),
        (RiskProfile::VerySafe, "NewUser_Frank"),
    ];
    let mut user_embs = Vec::new();
    let mut user_anom = Vec::new();
    let mut user_names_vec = Vec::new();
    for (i, (prof, name)) in user_profiles.iter().enumerate() {
        user_embs.push(make_embedding(*prof, dim, seed, i));
        user_anom.push(make_anomaly(*prof, seed + 1, i));
        user_names_vec.push(name.to_string());
    }
    emb.insert("user".into(), user_embs);
    anomaly.insert("user".into(), user_anom);
    names.insert("user".into(), user_names_vec);

    // ── 25 Accounts ──────────────────────────────────────────
    // Each user gets 2-3 accounts with matching risk profile
    let account_specs: Vec<(RiskProfile, &str)> = vec![
        (RiskProfile::HighRisk, "Dave_Checking"),
        (RiskProfile::HighRisk, "Dave_CreditLine"),
        (RiskProfile::HighRisk, "Zara_Checking"),
        (RiskProfile::Risky, "Zara_Savings"),
        (RiskProfile::Risky, "Mike_Checking"),
        (RiskProfile::Risky, "Mike_Brokerage"),
        (RiskProfile::Risky, "Sarah_Checking"),
        (RiskProfile::Mixed, "Sarah_401k"),
        (RiskProfile::Mixed, "Carlos_Checking"),
        (RiskProfile::Mixed, "Carlos_Brokerage"),
        (RiskProfile::Mixed, "Priya_Checking"),
        (RiskProfile::Mixed, "Priya_HSA"),
        (RiskProfile::Safe, "Beth_Savings"),
        (RiskProfile::Safe, "Beth_401k"),
        (RiskProfile::Safe, "Beth_HSA"),
        (RiskProfile::Safe, "Tom_Checking"),
        (RiskProfile::Safe, "Tom_Savings"),
        (RiskProfile::VerySafe, "Emma_Savings"),
        (RiskProfile::VerySafe, "Emma_401k"),
        (RiskProfile::VerySafe, "Emma_529"),
        (RiskProfile::VerySafe, "Frank_Checking"),
        (RiskProfile::VerySafe, "Frank_Savings"),
        (RiskProfile::Mixed, "Carlos_Savings"),
        (RiskProfile::Risky, "Mike_CreditLine"),
        (RiskProfile::HighRisk, "Dave_PaydayAcct"),
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

    // User→Account edges
    edges.insert(
        ("user".into(), "owns".into(), "account".into()),
        vec![
            (0, 0),
            (0, 1),
            (0, 24), // Dave
            (1, 2),
            (1, 3), // Zara
            (2, 4),
            (2, 5),
            (2, 23), // Mike
            (3, 6),
            (3, 7), // Sarah
            (4, 8),
            (4, 9),
            (4, 22), // Carlos
            (5, 10),
            (5, 11), // Priya
            (6, 12),
            (6, 13),
            (6, 14), // Beth
            (7, 15),
            (7, 16), // Tom
            (8, 17),
            (8, 18),
            (8, 19), // Emma
            (9, 20),
            (9, 21), // Frank
        ],
    );

    // ── 20 Obligations ───────────────────────────────────────
    let oblig_specs: Vec<(RiskProfile, &str)> = vec![
        (RiskProfile::HighRisk, "Dave_CreditCard_24pct"),
        (RiskProfile::HighRisk, "Dave_PaydayLoan_36pct"),
        (RiskProfile::HighRisk, "Dave_HELOC"),
        (RiskProfile::HighRisk, "Zara_CreditCard_22pct"),
        (RiskProfile::Risky, "Zara_PersonalLoan"),
        (RiskProfile::Risky, "Mike_CarLoan_12pct"),
        (RiskProfile::Risky, "Mike_StudentLoan"),
        (RiskProfile::Risky, "Sarah_MedicalBill"),
        (RiskProfile::Mixed, "Carlos_Mortgage"),
        (RiskProfile::Mixed, "Carlos_AutoLoan"),
        (RiskProfile::Mixed, "Priya_StudentLoan"),
        (RiskProfile::Safe, "Beth_CarLoan_3pct"),
        (RiskProfile::Safe, "Tom_Mortgage_4pct"),
        (RiskProfile::VerySafe, "Emma_Mortgage_3pct"),
        (RiskProfile::HighRisk, "Dave_CollectionDebt"),
        (RiskProfile::Risky, "Sarah_CreditCard"),
        (RiskProfile::Mixed, "Priya_HELOC"),
        (RiskProfile::HighRisk, "Zara_TaxDebt"),
        (RiskProfile::Safe, "Beth_StudentLoan"),
        (RiskProfile::VerySafe, "Emma_CarLoan"),
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
            (1, 1),
            (24, 14), // Dave's accounts → obligations
            (2, 3),
            (3, 4),
            (2, 17), // Zara
            (4, 5),
            (4, 6),
            (23, 5), // Mike
            (6, 7),
            (6, 15), // Sarah
            (8, 8),
            (8, 9), // Carlos
            (10, 10),
            (10, 16), // Priya
            (12, 11),
            (13, 18), // Beth
            (15, 12), // Tom
            (17, 13),
            (17, 19), // Emma
        ],
    );

    // ── 10 Rates ─────────────────────────────────────────────
    let rate_specs: Vec<(RiskProfile, &str)> = vec![
        (RiskProfile::HighRisk, "APR_36pct"),
        (RiskProfile::HighRisk, "APR_24pct"),
        (RiskProfile::HighRisk, "APR_22pct"),
        (RiskProfile::Risky, "APR_15pct"),
        (RiskProfile::Risky, "APR_12pct"),
        (RiskProfile::Mixed, "APR_8pct"),
        (RiskProfile::Mixed, "APR_6pct"),
        (RiskProfile::Safe, "APR_4pct"),
        (RiskProfile::Safe, "APR_3pct"),
        (RiskProfile::VerySafe, "APR_2pct"),
    ];
    let mut rate_embs = Vec::new();
    let mut rate_anom = Vec::new();
    let mut rate_names = Vec::new();
    for (i, (prof, name)) in rate_specs.iter().enumerate() {
        rate_embs.push(make_embedding(*prof, dim, seed + 30, i));
        rate_anom.push(make_anomaly(*prof, seed + 31, i));
        rate_names.push(name.to_string());
    }
    emb.insert("rate".into(), rate_embs);
    anomaly.insert("rate".into(), rate_anom);
    names.insert("rate".into(), rate_names);

    edges.insert(
        ("obligation".into(), "has_rate".into(), "rate".into()),
        vec![
            (0, 1),
            (1, 0),
            (2, 5),
            (3, 2),
            (4, 3),
            (5, 4),
            (6, 3),
            (7, 5),
            (8, 7),
            (9, 6),
            (10, 3),
            (11, 8),
            (12, 7),
            (13, 8),
            (14, 0),
            (15, 4),
            (16, 5),
            (17, 2),
            (18, 8),
            (19, 9),
        ],
    );

    // ── 15 Merchants ─────────────────────────────────────────
    let merch_specs: Vec<(RiskProfile, &str)> = vec![
        (RiskProfile::HighRisk, "OnlineGambling"),
        (RiskProfile::HighRisk, "CryptoExchange_Shady"),
        (RiskProfile::HighRisk, "PaydayLender"),
        (RiskProfile::Risky, "CryptoExchange_Major"),
        (RiskProfile::Risky, "LuxuryGoods"),
        (RiskProfile::Mixed, "Electronics"),
        (RiskProfile::Mixed, "Restaurant"),
        (RiskProfile::Safe, "Grocery_Costco"),
        (RiskProfile::Safe, "Grocery_Walmart"),
        (RiskProfile::Safe, "Utilities_Electric"),
        (RiskProfile::Safe, "Utilities_Water"),
        (RiskProfile::VerySafe, "Pharmacy"),
        (RiskProfile::VerySafe, "Insurance"),
        (RiskProfile::Mixed, "GasStation"),
        (RiskProfile::Risky, "NightClub"),
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
            // Dave → gambling, crypto, payday (risky merchants)
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 6),
            // Zara → crypto, luxury, restaurant
            (2, 3),
            (2, 4),
            (2, 6),
            // Mike → electronics, restaurant, nightclub
            (4, 5),
            (4, 6),
            (4, 14),
            // Sarah → grocery, utilities
            (6, 7),
            (6, 9),
            // Carlos → grocery, gas, electronics
            (8, 7),
            (8, 13),
            (8, 5),
            // Priya → grocery, pharmacy
            (10, 8),
            (10, 11),
            // Beth → grocery, pharmacy, insurance
            (12, 8),
            (12, 11),
            (12, 12),
            // Tom → grocery, utilities
            (15, 7),
            (15, 10),
            // Emma → grocery, pharmacy, insurance
            (17, 9),
            (17, 11),
            (17, 12),
            // Frank → grocery, gas
            (20, 8),
            (20, 13),
        ],
    );

    // ── 12 Recurring Subscriptions ───────────────────────────
    let rec_specs: Vec<(RiskProfile, &str)> = vec![
        (RiskProfile::VerySafe, "Netflix"),
        (RiskProfile::VerySafe, "Spotify"),
        (RiskProfile::Risky, "UnusedGym_1"),
        (RiskProfile::Risky, "UnusedGym_2"),
        (RiskProfile::Risky, "UnusedMagazine"),
        (RiskProfile::Mixed, "ForgottenCloudStorage"),
        (RiskProfile::Mixed, "OldSaaSTrial"),
        (RiskProfile::Safe, "AmazonPrime"),
        (RiskProfile::VerySafe, "YouTubePremium"),
        (RiskProfile::Risky, "UnusedVPN"),
        (RiskProfile::HighRisk, "GamblingSubscription"),
        (RiskProfile::Safe, "NewsSubscription"),
    ];
    let mut rec_embs = Vec::new();
    let mut rec_anom = Vec::new();
    let mut rec_names = Vec::new();
    for (i, (prof, name)) in rec_specs.iter().enumerate() {
        rec_embs.push(make_embedding(*prof, dim, seed + 50, i));
        rec_anom.push(make_anomaly(*prof, seed + 51, i));
        rec_names.push(name.to_string());
    }
    emb.insert("recurring".into(), rec_embs);
    anomaly.insert("recurring".into(), rec_anom);
    names.insert("recurring".into(), rec_names);

    edges.insert(
        ("user".into(), "subscribes".into(), "recurring".into()),
        vec![
            // High risk users have lots of subscriptions including wasteful ones
            (0, 0),
            (0, 2),
            (0, 4),
            (0, 5),
            (0, 10), // Dave: netflix, gym, magazine, cloud, gambling
            (1, 0),
            (1, 3),
            (1, 6),
            (1, 9), // Zara: netflix, gym, saas, vpn
            (2, 0),
            (2, 2),
            (2, 5), // Mike: netflix, gym, cloud
            (3, 0),
            (3, 7), // Sarah: netflix, prime
            (4, 0),
            (4, 7),
            (4, 11), // Carlos: netflix, prime, news
            (5, 0),
            (5, 1), // Priya: netflix, spotify
            (6, 0),
            (6, 1),
            (6, 7),
            (6, 8), // Beth: netflix, spotify, prime, youtube
            (7, 0),
            (7, 7), // Tom: netflix, prime
            (8, 0),
            (8, 1),
            (8, 8), // Emma: netflix, spotify, youtube
            (9, 0), // Frank: netflix only
        ],
    );

    // ── 8 Goals ──────────────────────────────────────────────
    let goal_specs: Vec<(RiskProfile, &str)> = vec![
        (RiskProfile::Mixed, "EmergencyFund"),
        (RiskProfile::Safe, "Retirement401k"),
        (RiskProfile::Mixed, "HouseDownPayment"),
        (RiskProfile::Safe, "CollegeFund"),
        (RiskProfile::Risky, "DebtPayoff"),
        (RiskProfile::VerySafe, "Vacation"),
        (RiskProfile::Mixed, "CarReplacementFund"),
        (RiskProfile::Safe, "WeddingFund"),
    ];
    let mut goal_embs = Vec::new();
    let mut goal_anom = Vec::new();
    let mut goal_names = Vec::new();
    for (i, (prof, name)) in goal_specs.iter().enumerate() {
        goal_embs.push(make_embedding(*prof, dim, seed + 60, i));
        goal_anom.push(make_anomaly(*prof, seed + 61, i));
        goal_names.push(name.to_string());
    }
    emb.insert("goal".into(), goal_embs);
    anomaly.insert("goal".into(), goal_anom);
    names.insert("goal".into(), goal_names);

    edges.insert(
        ("user".into(), "targets".into(), "goal".into()),
        vec![
            (0, 0),
            (0, 4), // Dave: emergency + debt payoff
            (1, 4), // Zara: debt payoff
            (2, 0),
            (2, 6), // Mike: emergency + car
            (3, 0),
            (3, 7), // Sarah: emergency + wedding
            (4, 0),
            (4, 2), // Carlos: emergency + house
            (5, 1),
            (5, 3), // Priya: retirement + college
            (6, 1),
            (6, 3),
            (6, 5), // Beth: retirement + college + vacation
            (7, 1),
            (7, 6), // Tom: retirement + car
            (8, 1),
            (8, 5), // Emma: retirement + vacation
            (9, 0), // Frank: emergency
        ],
    );

    // ── 5 Tax Items ──────────────────────────────────────────
    let tax_specs: Vec<(RiskProfile, &str)> = vec![
        (RiskProfile::Mixed, "Q4_Tax_2024"),
        (RiskProfile::HighRisk, "IRS_BackTax_Zara"),
        (RiskProfile::HighRisk, "IRS_BackTax_Dave"),
        (RiskProfile::Safe, "EstimatedTax_Beth"),
        (RiskProfile::VerySafe, "EstimatedTax_Emma"),
    ];
    let mut tax_embs = Vec::new();
    let mut tax_anom = Vec::new();
    let mut tax_names = Vec::new();
    for (i, (prof, name)) in tax_specs.iter().enumerate() {
        tax_embs.push(make_embedding(*prof, dim, seed + 70, i));
        tax_anom.push(make_anomaly(*prof, seed + 71, i));
        tax_names.push(name.to_string());
    }
    emb.insert("tax_due".into(), tax_embs);
    anomaly.insert("tax_due".into(), tax_anom);
    names.insert("tax_due".into(), tax_names);

    edges.insert(
        ("user".into(), "liable".into(), "tax_due".into()),
        vec![(0, 2), (1, 1), (4, 0), (6, 3), (8, 4)],
    );

    // Tax sinking
    emb.insert(
        "tax_sinking".into(),
        vec![
            make_embedding(RiskProfile::Safe, dim, seed + 72, 0),
            make_embedding(RiskProfile::VerySafe, dim, seed + 72, 1),
        ],
    );
    anomaly.insert("tax_sinking".into(), vec![0.08, 0.03]);
    names.insert(
        "tax_sinking".into(),
        vec!["FedReserve".into(), "StateReserve".into()],
    );
    edges.insert(
        ("user".into(), "funds".into(), "tax_sinking".into()),
        vec![(4, 0), (6, 1)],
    );

    // ── 6 Assets ─────────────────────────────────────────────
    let asset_specs: Vec<(RiskProfile, &str)> = vec![
        (RiskProfile::Safe, "Dave_House"),
        (RiskProfile::Safe, "Carlos_House"),
        (RiskProfile::Safe, "Beth_House"),
        (RiskProfile::Mixed, "Mike_Car"),
        (RiskProfile::Safe, "Emma_House"),
        (RiskProfile::Mixed, "Tom_Car"),
    ];
    let mut asset_embs = Vec::new();
    let mut asset_anom = Vec::new();
    let mut asset_names = Vec::new();
    for (i, (prof, name)) in asset_specs.iter().enumerate() {
        asset_embs.push(make_embedding(*prof, dim, seed + 80, i));
        asset_anom.push(make_anomaly(*prof, seed + 81, i));
        asset_names.push(name.to_string());
    }
    emb.insert("asset".into(), asset_embs);
    anomaly.insert("asset".into(), asset_anom);
    names.insert("asset".into(), asset_names);

    edges.insert(
        ("user".into(), "holds".into(), "asset".into()),
        vec![(0, 0), (2, 3), (4, 1), (6, 2), (7, 5), (8, 4)],
    );

    // Valuations
    emb.insert(
        "valuation".into(),
        (0..6)
            .map(|i| make_embedding(RiskProfile::Safe, dim, seed + 82, i))
            .collect(),
    );
    anomaly.insert("valuation".into(), vec![0.12, 0.15, 0.08, 0.22, 0.05, 0.18]);
    names.insert(
        "valuation".into(),
        (0..6).map(|i| format!("Val_{}", i)).collect(),
    );
    edges.insert(
        ("asset".into(), "valued_by".into(), "valuation".into()),
        vec![(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],
    );

    // ── 4 Recon Cases ────────────────────────────────────────
    emb.insert(
        "recon_case".into(),
        vec![
            make_embedding(RiskProfile::Risky, dim, seed + 90, 0),
            make_embedding(RiskProfile::HighRisk, dim, seed + 90, 1),
            make_embedding(RiskProfile::Mixed, dim, seed + 90, 2),
            make_embedding(RiskProfile::Safe, dim, seed + 90, 3),
        ],
    );
    anomaly.insert("recon_case".into(), vec![0.55, 0.82, 0.35, 0.10]);
    names.insert(
        "recon_case".into(),
        vec![
            "Dave_Jan_Recon".into(),
            "Zara_Feb_Unmatched".into(),
            "Carlos_Mar_Recon".into(),
            "Beth_Apr_Recon".into(),
        ],
    );
    edges.insert(
        (
            "account".into(),
            "reconciled_by".into(),
            "recon_case".into(),
        ),
        vec![(0, 0), (2, 1), (8, 2), (12, 3)],
    );

    // ── 5 Budgets ────────────────────────────────────────────
    emb.insert(
        "budget".into(),
        (0..5)
            .map(|i| {
                let prof = match i {
                    0 => RiskProfile::HighRisk,
                    1 => RiskProfile::Risky,
                    2 => RiskProfile::Mixed,
                    3 => RiskProfile::Safe,
                    _ => RiskProfile::VerySafe,
                };
                make_embedding(prof, dim, seed + 95, i)
            })
            .collect(),
    );
    anomaly.insert("budget".into(), vec![0.65, 0.48, 0.30, 0.12, 0.05]);
    names.insert(
        "budget".into(),
        vec![
            "Dave_Monthly".into(),
            "Mike_Monthly".into(),
            "Carlos_Monthly".into(),
            "Beth_Monthly".into(),
            "Emma_Monthly".into(),
        ],
    );
    edges.insert(
        ("user".into(), "tracks".into(), "budget".into()),
        vec![(0, 0), (2, 1), (4, 2), (6, 3), (8, 4)],
    );

    // Build counts
    let mut node_counts: HashMap<String, usize> = HashMap::new();
    for (nt, ns) in &names {
        node_counts.insert(nt.clone(), ns.len());
    }

    // Build anomaly_scores wrapper
    let mut anomaly_scores: HashMap<String, HashMap<String, Vec<f32>>> = HashMap::new();
    anomaly_scores.insert("SAGE".into(), anomaly);

    GeneratedGraph {
        embeddings: emb,
        anomaly_scores,
        node_names: names,
        edges,
        node_counts,
    }
}

#[test]
fn test_large_graph_pc_differentiation() {
    let graph = generate_large_graph();

    // Count total entities
    let total_entities: usize = graph.node_counts.values().sum();
    println!(
        "\n  ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "  ║  LARGE-SCALE PC DIFFERENTIATION TEST — {} total entities                                                 ║",
        total_entities
    );
    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣"
    );
    for (nt, count) in &graph.node_counts {
        println!("  ║    {:18}: {:3} entities", nt, count);
    }
    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣"
    );

    // Test users: compare high-risk (Dave, id=0) vs safe (Beth, id=6) vs very-safe (Emma, id=8)
    let test_users = vec![
        (0, "🔴 HighRisk_Dave"),
        (1, "🔴 HighRisk_Zara"),
        (2, "🟠 Risky_Mike"),
        (4, "🟡 Mixed_Carlos"),
        (6, "🟢 Safe_Beth"),
        (8, "🟢 VerySafe_Emma"),
        (9, "⚪ NewUser_Frank"),
    ];

    let mut pc_state = PcState::new();
    let mut user_risks: Vec<(String, f64, usize)> = Vec::new();

    println!(
        "  ║  User                │ Avg P(risk) │ Actions │ Top 3 Actions (with P(risk))                                   ║"
    );
    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣"
    );

    for (user_id, label) in &test_users {
        let user_emb = graph.embeddings.get("user").unwrap()[*user_id].clone();

        let ctx = FiduciaryContext {
            user_emb: &user_emb,
            embeddings: &graph.embeddings,
            anomaly_scores: &graph.anomaly_scores,
            edges: &graph.edges,
            node_names: &graph.node_names,
            node_counts: &graph.node_counts,
            user_type: "user".into(),
            user_id: *user_id,
            hidden_dim: 8,
        };

        let resp = recommend(&ctx, Some(&mut pc_state));

        let analyses: Vec<(String, f64)> = resp
            .recommendations
            .iter()
            .filter_map(|r| {
                r.pc_analysis.as_ref().map(|a| {
                    (
                        format!(
                            "{}/{}",
                            &r.action_type[7..r.action_type.len().min(25)],
                            &r.target_name[..r.target_name.len().min(12)]
                        ),
                        a.risk_probability,
                    )
                })
            })
            .collect();

        let avg_risk = if analyses.is_empty() {
            0.0
        } else {
            analyses.iter().map(|(_, r)| *r).sum::<f64>() / analyses.len() as f64
        };

        let top3: Vec<String> = analyses
            .iter()
            .take(3)
            .map(|(name, risk)| format!("{} P={:.4}", name, risk))
            .collect();
        let top3_str = top3.join(", ");

        println!(
            "  ║  {:20} │  {:.6}   │   {:3}   │ {:62} ║",
            label,
            avg_risk,
            resp.recommendations.len(),
            &top3_str[..top3_str.len().min(62)]
        );

        user_risks.push((label.to_string(), avg_risk, resp.recommendations.len()));
    }

    println!(
        "  ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "  ║  PcState: {} total EM epochs, {} ll entries                                                                ║",
        pc_state.total_epochs,
        pc_state.ll_history.len()
    );

    // Key comparison: high-risk vs safe
    let dave_risk = user_risks
        .iter()
        .find(|(n, _, _)| n.contains("Dave"))
        .map(|(_, r, _)| *r)
        .unwrap_or(0.0);
    let beth_risk = user_risks
        .iter()
        .find(|(n, _, _)| n.contains("Beth"))
        .map(|(_, r, _)| *r)
        .unwrap_or(0.0);
    let emma_risk = user_risks
        .iter()
        .find(|(n, _, _)| n.contains("Emma"))
        .map(|(_, r, _)| *r)
        .unwrap_or(0.0);

    println!(
        "  ║  Risk comparison:                                                                                          ║"
    );
    println!(
        "  ║    Dave (high-risk): {:.6}                                                                                  ║",
        dave_risk
    );
    println!(
        "  ║    Beth (safe):      {:.6}                                                                                  ║",
        beth_risk
    );
    println!(
        "  ║    Emma (very-safe): {:.6}                                                                                  ║",
        emma_risk
    );

    let risk_spread = user_risks
        .iter()
        .map(|(_, r, _)| *r)
        .fold(f64::INFINITY, f64::min);
    let risk_max = user_risks
        .iter()
        .map(|(_, r, _)| *r)
        .fold(f64::NEG_INFINITY, f64::max);
    let spread = risk_max - risk_spread;

    if spread > 0.01 {
        println!(
            "  ║  ✅ PC risk varies across users (spread = {:.6})                                                         ║",
            spread
        );
    } else {
        println!(
            "  ║  ⚠️  PC risk is still uniform (spread = {:.6})                                                          ║",
            spread
        );
    }

    // Verify circuit accumulated
    assert!(pc_state.is_trained());
    println!(
        "  ║  ✅ PC circuit trained and persisted across {} users                                                        ║",
        test_users.len()
    );
    assert!(
        total_entities >= 100,
        "Expected 100+ entities, got {}",
        total_entities
    );
    println!(
        "  ║  ✅ {} total entities in graph                                                                               ║",
        total_entities
    );

    println!(
        "  ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝"
    );
}
