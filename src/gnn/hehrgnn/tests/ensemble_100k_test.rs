//! 100K-Node Realistic Ensemble Anomaly Detection
//!
//! Large-scale realistic synthetic financial graph:
//!   - 500 users with demographic profiles (income level, age group, location)
//!   - 200 merchants with real categories & spending distributions
//!   - 50,000+ transactions with realistic amounts, time patterns
//!   - Multiple account types (checking, savings, credit card)
//!   - ~50 injected subtle anomalies (not just extreme amounts)
//!
//! Runs: GraphSAGE + RGCN + GAT + composite signals → attention-based ensemble
//! Target: 100K+ total graph nodes

#[cfg(test)]
mod tests {
    use burn::backend::{Autodiff, Cpu, NdArray, Wgpu};
    use burn::prelude::*;
    use std::collections::HashMap;

    // Wgpu backend: uses Intel UHD 770 iGPU via Vulkan — GPU-accelerated CubeCL kernels
    type B = Wgpu;
    type TrainB = Autodiff<Wgpu>;
    #[allow(dead_code)]
    type InferB = Wgpu;

    use burn::data::dataloader::batcher::Batcher;
    use hehrgnn::data::batcher::{HehrBatch, HehrBatcher, HehrFactItem};
    use hehrgnn::data::fact::{HehrFact, RawFact};
    use hehrgnn::data::graph_builder::{GraphBuildConfig, GraphFact, build_hetero_graph};
    use hehrgnn::data::hetero_graph::EdgeType;
    use hehrgnn::data::vocab::KgVocabulary;
    use hehrgnn::model::gat::GatConfig;
    use hehrgnn::model::graphsage::GraphSageModelConfig;
    use hehrgnn::model::rgcn::RgcnConfig;
    use hehrgnn::server::state::PlainEmbeddings;
    use hehrgnn::training::scoring::DistMultScorer;
    use hehrgnn::training::train::{TrainConfig, train};

    fn gf(st: &str, s: &str, r: &str, dt: &str, d: &str) -> GraphFact {
        GraphFact {
            src: (st.to_string(), s.to_string()),
            relation: r.to_string(),
            dst: (dt.to_string(), d.to_string()),
        }
    }

    /// Deterministic pseudo-random (no external deps)
    fn prng(seed: u64) -> u64 {
        let mut x = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51afd7ed558ccd);
        x ^= x >> 33;
        x
    }

    fn prng_f64(seed: u64) -> f64 {
        (prng(seed) % 10000) as f64 / 10000.0
    }

    // ═══════════════════════════════════════════════════════════════
    //  REALISTIC MERCHANT DATABASE
    // ═══════════════════════════════════════════════════════════════
    struct Merchant {
        name: &'static str,
        category: &'static str,
        avg_amount: f64,
        std_amount: f64,
        frequency: f64, // avg txs per user per month
    }

    fn merchants() -> Vec<Merchant> {
        vec![
            // Groceries (high frequency, medium amount)
            Merchant {
                name: "walmart",
                category: "groceries",
                avg_amount: 67.0,
                std_amount: 35.0,
                frequency: 4.0,
            },
            Merchant {
                name: "kroger",
                category: "groceries",
                avg_amount: 55.0,
                std_amount: 28.0,
                frequency: 3.0,
            },
            Merchant {
                name: "costco",
                category: "groceries",
                avg_amount: 145.0,
                std_amount: 60.0,
                frequency: 1.5,
            },
            Merchant {
                name: "trader_joes",
                category: "groceries",
                avg_amount: 42.0,
                std_amount: 18.0,
                frequency: 2.0,
            },
            Merchant {
                name: "whole_foods",
                category: "groceries",
                avg_amount: 78.0,
                std_amount: 32.0,
                frequency: 2.0,
            },
            Merchant {
                name: "aldi",
                category: "groceries",
                avg_amount: 38.0,
                std_amount: 15.0,
                frequency: 2.5,
            },
            Merchant {
                name: "safeway",
                category: "groceries",
                avg_amount: 52.0,
                std_amount: 25.0,
                frequency: 2.0,
            },
            Merchant {
                name: "publix",
                category: "groceries",
                avg_amount: 61.0,
                std_amount: 28.0,
                frequency: 2.5,
            },
            // Fast food (high frequency, low amount)
            Merchant {
                name: "mcdonalds",
                category: "fast_food",
                avg_amount: 12.0,
                std_amount: 5.0,
                frequency: 3.0,
            },
            Merchant {
                name: "chick_fil_a",
                category: "fast_food",
                avg_amount: 14.0,
                std_amount: 5.0,
                frequency: 2.0,
            },
            Merchant {
                name: "subway",
                category: "fast_food",
                avg_amount: 10.0,
                std_amount: 4.0,
                frequency: 2.0,
            },
            Merchant {
                name: "taco_bell",
                category: "fast_food",
                avg_amount: 9.0,
                std_amount: 4.0,
                frequency: 2.0,
            },
            Merchant {
                name: "wendys",
                category: "fast_food",
                avg_amount: 11.0,
                std_amount: 4.5,
                frequency: 1.5,
            },
            Merchant {
                name: "chipotle",
                category: "fast_food",
                avg_amount: 15.0,
                std_amount: 5.0,
                frequency: 2.0,
            },
            // Dining (medium frequency, medium-high amount)
            Merchant {
                name: "olive_garden",
                category: "dining",
                avg_amount: 45.0,
                std_amount: 20.0,
                frequency: 0.5,
            },
            Merchant {
                name: "applebees",
                category: "dining",
                avg_amount: 38.0,
                std_amount: 15.0,
                frequency: 0.5,
            },
            Merchant {
                name: "local_restaurant",
                category: "dining",
                avg_amount: 55.0,
                std_amount: 25.0,
                frequency: 1.0,
            },
            Merchant {
                name: "steakhouse",
                category: "dining",
                avg_amount: 85.0,
                std_amount: 35.0,
                frequency: 0.3,
            },
            // Gas (regular, predictable)
            Merchant {
                name: "shell",
                category: "gas",
                avg_amount: 48.0,
                std_amount: 12.0,
                frequency: 2.0,
            },
            Merchant {
                name: "bp",
                category: "gas",
                avg_amount: 45.0,
                std_amount: 11.0,
                frequency: 2.0,
            },
            Merchant {
                name: "exxon",
                category: "gas",
                avg_amount: 50.0,
                std_amount: 13.0,
                frequency: 1.5,
            },
            Merchant {
                name: "chevron",
                category: "gas",
                avg_amount: 47.0,
                std_amount: 12.0,
                frequency: 1.5,
            },
            // Online shopping (medium frequency, variable amount)
            Merchant {
                name: "amazon",
                category: "online_shopping",
                avg_amount: 52.0,
                std_amount: 45.0,
                frequency: 3.0,
            },
            Merchant {
                name: "ebay",
                category: "online_shopping",
                avg_amount: 35.0,
                std_amount: 30.0,
                frequency: 0.5,
            },
            Merchant {
                name: "etsy",
                category: "online_shopping",
                avg_amount: 28.0,
                std_amount: 18.0,
                frequency: 0.3,
            },
            Merchant {
                name: "target_online",
                category: "online_shopping",
                avg_amount: 40.0,
                std_amount: 25.0,
                frequency: 1.0,
            },
            // Subscriptions (monthly, fixed)
            Merchant {
                name: "netflix",
                category: "subscription",
                avg_amount: 15.49,
                std_amount: 0.01,
                frequency: 1.0,
            },
            Merchant {
                name: "spotify",
                category: "subscription",
                avg_amount: 10.99,
                std_amount: 0.01,
                frequency: 1.0,
            },
            Merchant {
                name: "youtube_premium",
                category: "subscription",
                avg_amount: 13.99,
                std_amount: 0.01,
                frequency: 1.0,
            },
            Merchant {
                name: "gym_membership",
                category: "subscription",
                avg_amount: 39.99,
                std_amount: 0.01,
                frequency: 1.0,
            },
            Merchant {
                name: "amazon_prime",
                category: "subscription",
                avg_amount: 14.99,
                std_amount: 0.01,
                frequency: 1.0,
            },
            // Utilities (monthly, semi-fixed)
            Merchant {
                name: "electric_company",
                category: "utilities",
                avg_amount: 120.0,
                std_amount: 35.0,
                frequency: 1.0,
            },
            Merchant {
                name: "water_utility",
                category: "utilities",
                avg_amount: 45.0,
                std_amount: 12.0,
                frequency: 1.0,
            },
            Merchant {
                name: "internet_provider",
                category: "utilities",
                avg_amount: 65.0,
                std_amount: 5.0,
                frequency: 1.0,
            },
            Merchant {
                name: "phone_carrier",
                category: "utilities",
                avg_amount: 85.0,
                std_amount: 10.0,
                frequency: 1.0,
            },
            // Healthcare (infrequent, variable)
            Merchant {
                name: "pharmacy_cvs",
                category: "healthcare",
                avg_amount: 25.0,
                std_amount: 20.0,
                frequency: 0.5,
            },
            Merchant {
                name: "doctor_copay",
                category: "healthcare",
                avg_amount: 30.0,
                std_amount: 10.0,
                frequency: 0.2,
            },
            Merchant {
                name: "dentist",
                category: "healthcare",
                avg_amount: 50.0,
                std_amount: 25.0,
                frequency: 0.15,
            },
            // Transportation
            Merchant {
                name: "uber",
                category: "transport",
                avg_amount: 18.0,
                std_amount: 12.0,
                frequency: 2.0,
            },
            Merchant {
                name: "lyft",
                category: "transport",
                avg_amount: 16.0,
                std_amount: 10.0,
                frequency: 1.0,
            },
            Merchant {
                name: "parking_meter",
                category: "transport",
                avg_amount: 5.0,
                std_amount: 3.0,
                frequency: 3.0,
            },
            // Home (infrequent, high)
            Merchant {
                name: "home_depot",
                category: "home",
                avg_amount: 85.0,
                std_amount: 60.0,
                frequency: 0.3,
            },
            Merchant {
                name: "lowes",
                category: "home",
                avg_amount: 75.0,
                std_amount: 55.0,
                frequency: 0.3,
            },
            Merchant {
                name: "ikea",
                category: "home",
                avg_amount: 120.0,
                std_amount: 80.0,
                frequency: 0.1,
            },
            // Entertainment
            Merchant {
                name: "movie_theater",
                category: "entertainment",
                avg_amount: 22.0,
                std_amount: 8.0,
                frequency: 0.5,
            },
            Merchant {
                name: "concert_venue",
                category: "entertainment",
                avg_amount: 75.0,
                std_amount: 40.0,
                frequency: 0.1,
            },
            Merchant {
                name: "bowling_alley",
                category: "entertainment",
                avg_amount: 18.0,
                std_amount: 8.0,
                frequency: 0.2,
            },
        ]
    }

    fn amount_bucket(amount: f64) -> &'static str {
        if amount < 10.0 {
            "micro"
        } else if amount < 25.0 {
            "small"
        } else if amount < 50.0 {
            "medium"
        } else if amount < 100.0 {
            "large"
        } else if amount < 250.0 {
            "xlarge"
        } else if amount < 500.0 {
            "xxlarge"
        } else {
            "huge"
        }
    }

    struct AnomalyInfo {
        tx_idx: usize,
        user: String,
        merchant: String,
        amount: f64,
        anomaly_type: String,
        description: String,
    }

    struct GeneratedData {
        graph_facts: Vec<GraphFact>,
        kg_facts: Vec<RawFact>,
        tx_amounts: Vec<f64>,
        tx_merchants: Vec<String>,
        tx_users: Vec<String>,
        anomalies: Vec<AnomalyInfo>,
        merchant_stats: HashMap<String, (f64, f64)>, // mean, std
        user_merchants: HashMap<String, Vec<String>>, // user → known merchants
    }

    fn generate_100k_data() -> GeneratedData {
        let merchant_db = merchants();
        let num_users = 500;
        let months = 12; // 1 year of data
        let mut graph_facts = Vec::new();
        let mut kg_facts = Vec::new();
        let mut tx_amounts = Vec::new();
        let mut tx_merchants = Vec::new();
        let mut tx_users = Vec::new();
        let mut anomalies = Vec::new();
        let mut merchant_stats: HashMap<String, (f64, f64)> = HashMap::new();
        let mut user_merchants: HashMap<String, Vec<String>> = HashMap::new();

        // Pre-compute merchant stats
        for m in &merchant_db {
            merchant_stats.insert(m.name.to_string(), (m.avg_amount, m.std_amount));
        }

        let mut tx_idx = 0usize;

        // Income levels for users
        let income_levels = ["low", "medium", "high", "very_high"];
        let age_groups = ["18_25", "26_35", "36_50", "51_65", "65_plus"];
        let regions = ["northeast", "southeast", "midwest", "west", "southwest"];

        // ── Generate users and their transactions ──
        for user_id in 0..num_users {
            let user = format!("user_{}", user_id);
            let income = income_levels[user_id % income_levels.len()];
            let age = age_groups[user_id % age_groups.len()];
            let region = regions[user_id % regions.len()];

            // User profile
            graph_facts.push(gf("user", &user, "income_level", "income", income));
            graph_facts.push(gf("user", &user, "age_group", "age", age));
            graph_facts.push(gf("user", &user, "lives_in", "region", region));

            // Accounts
            let checking = format!("{}_checking", user);
            graph_facts.push(gf("user", &user, "owns", "account", &checking));
            graph_facts.push(gf(
                "account",
                &checking,
                "account_type",
                "acct_type",
                "checking",
            ));

            if user_id % 3 == 0 {
                let savings = format!("{}_savings", user);
                graph_facts.push(gf("user", &user, "owns", "account", &savings));
                graph_facts.push(gf(
                    "account",
                    &savings,
                    "account_type",
                    "acct_type",
                    "savings",
                ));
            }
            if user_id % 5 == 0 {
                let cc = format!("{}_credit", user);
                graph_facts.push(gf("user", &user, "owns", "account", &cc));
                graph_facts.push(gf(
                    "account",
                    &cc,
                    "account_type",
                    "acct_type",
                    "credit_card",
                ));
            }

            // Each user has 8-15 regular merchants (realistic: people are habitual)
            let seed = user_id as u64 * 31337;
            let num_regular = 8 + (prng(seed) % 8) as usize;
            let mut user_regular: Vec<usize> = Vec::new();
            for j in 0..num_regular {
                let idx = (prng(seed + j as u64 * 7) % merchant_db.len() as u64) as usize;
                if !user_regular.contains(&idx) {
                    user_regular.push(idx);
                }
            }

            let user_merch_names: Vec<String> = user_regular
                .iter()
                .map(|&i| merchant_db[i].name.to_string())
                .collect();
            user_merchants.insert(user.clone(), user_merch_names);

            // Generate transactions for each month
            for month in 0..months {
                for &merch_idx in &user_regular {
                    let m = &merchant_db[merch_idx];
                    // How many txs this month? Based on frequency + randomness
                    let base_count = m.frequency;
                    let seed_tx = seed + month as u64 * 1000 + merch_idx as u64;
                    let variation = prng_f64(seed_tx) * 0.6 - 0.3; // ±30%
                    let count = ((base_count * (1.0 + variation)).round() as usize).max(0);

                    for t in 0..count {
                        let tx = format!("tx_{}", tx_idx);
                        let amt_seed = seed_tx + t as u64 * 13;
                        // Generate realistic amount: normal distribution approximation
                        let z =
                            (prng_f64(amt_seed) + prng_f64(amt_seed + 1) + prng_f64(amt_seed + 2))
                                / 3.0
                                * 2.0
                                - 1.0;
                        let amount =
                            (m.avg_amount + z * m.std_amount).max(0.50).round() * 0.01 * 100.0; // round to cents
                        let amount = (amount * 100.0).round() / 100.0;
                        let bucket = amount_bucket(amount);
                        let month_name = format!("month_{}", month);

                        let acct = if user_id % 5 == 0 && t % 3 == 0 {
                            format!("{}_credit", user)
                        } else {
                            checking.clone()
                        };

                        graph_facts.push(gf("transaction", &tx, "posted_to", "account", &acct));
                        graph_facts.push(gf("transaction", &tx, "at_merchant", "merchant", m.name));
                        graph_facts.push(gf(
                            "transaction",
                            &tx,
                            "tx_amount",
                            "amount_bucket",
                            bucket,
                        ));
                        graph_facts.push(gf(
                            "transaction",
                            &tx,
                            "in_category",
                            "category",
                            m.category,
                        ));
                        graph_facts.push(gf("transaction", &tx, "in_month", "month", &month_name));

                        kg_facts.push(RawFact {
                            head: user.clone(),
                            relation: "transacts_at".into(),
                            tail: m.name.to_string(),
                            qualifiers: vec![],
                        });
                        kg_facts.push(RawFact {
                            head: tx.clone(),
                            relation: "amount_range".into(),
                            tail: bucket.to_string(),
                            qualifiers: vec![],
                        });

                        tx_amounts.push(amount);
                        tx_merchants.push(m.name.to_string());
                        tx_users.push(user.clone());
                        tx_idx += 1;
                    }
                }
            }
        }

        let normal_count = tx_idx;

        // ═══════════════════════════════════════════════════════════════
        //  INJECT SUBTLE ANOMALIES (~50)
        // ═══════════════════════════════════════════════════════════════

        // TYPE 1: Unusual amount at known merchant (10 anomalies)
        for i in 0..10 {
            let user = format!("user_{}", i * 50);
            let merch = &merchant_db[i % merchant_db.len()];
            let amount = merch.avg_amount * (8.0 + i as f64); // 8-17× normal
            let tx = format!("tx_{}", tx_idx);
            let bucket = amount_bucket(amount);

            graph_facts.push(gf(
                "transaction",
                &tx,
                "posted_to",
                "account",
                &format!("{}_checking", user),
            ));
            graph_facts.push(gf(
                "transaction",
                &tx,
                "at_merchant",
                "merchant",
                merch.name,
            ));
            graph_facts.push(gf("transaction", &tx, "tx_amount", "amount_bucket", bucket));
            graph_facts.push(gf(
                "transaction",
                &tx,
                "in_category",
                "category",
                merch.category,
            ));
            graph_facts.push(gf("transaction", &tx, "in_month", "month", "month_11"));

            kg_facts.push(RawFact {
                head: user.clone(),
                relation: "transacts_at".into(),
                tail: merch.name.to_string(),
                qualifiers: vec![],
            });
            kg_facts.push(RawFact {
                head: tx.clone(),
                relation: "amount_range".into(),
                tail: bucket.to_string(),
                qualifiers: vec![],
            });

            anomalies.push(AnomalyInfo {
                tx_idx,
                user: user.clone(),
                merchant: merch.name.to_string(),
                amount,
                anomaly_type: "amount".into(),
                description: format!(
                    "${:.0} at {} (avg ${:.0})",
                    amount, merch.name, merch.avg_amount
                ),
            });
            tx_amounts.push(amount);
            tx_merchants.push(merch.name.to_string());
            tx_users.push(user);
            tx_idx += 1;
        }

        // TYPE 2: Transaction at merchant user never visits (15 anomalies)
        for i in 0..15 {
            let user = format!("user_{}", 10 + i * 30);
            // Pick a merchant this user has NEVER used
            let user_known = user_merchants.get(&user).cloned().unwrap_or_default();
            let mut merch_idx = (i * 7 + 3) % merchant_db.len();
            while user_known.contains(&merchant_db[merch_idx].name.to_string()) {
                merch_idx = (merch_idx + 1) % merchant_db.len();
            }
            let merch = &merchant_db[merch_idx];
            let amount = merch.avg_amount; // normal amount, but NEW merchant
            let tx = format!("tx_{}", tx_idx);
            let bucket = amount_bucket(amount);

            graph_facts.push(gf(
                "transaction",
                &tx,
                "posted_to",
                "account",
                &format!("{}_checking", user),
            ));
            graph_facts.push(gf(
                "transaction",
                &tx,
                "at_merchant",
                "merchant",
                merch.name,
            ));
            graph_facts.push(gf("transaction", &tx, "tx_amount", "amount_bucket", bucket));
            graph_facts.push(gf(
                "transaction",
                &tx,
                "in_category",
                "category",
                merch.category,
            ));
            graph_facts.push(gf("transaction", &tx, "in_month", "month", "month_11"));

            kg_facts.push(RawFact {
                head: user.clone(),
                relation: "transacts_at".into(),
                tail: merch.name.to_string(),
                qualifiers: vec![],
            });
            kg_facts.push(RawFact {
                head: tx.clone(),
                relation: "amount_range".into(),
                tail: bucket.to_string(),
                qualifiers: vec![],
            });

            anomalies.push(AnomalyInfo {
                tx_idx,
                user: user.clone(),
                merchant: merch.name.to_string(),
                amount,
                anomaly_type: "new_merchant".into(),
                description: format!("${:.0} at {} (NEVER used by {})", amount, merch.name, user),
            });
            tx_amounts.push(amount);
            tx_merchants.push(merch.name.to_string());
            tx_users.push(user);
            tx_idx += 1;
        }

        // TYPE 3: Unknown merchant (10 anomalies)
        let fake_merchants = [
            "offshore_wire_svc",
            "crypto_atm_23",
            "foreign_casino",
            "anonymous_prepaid",
            "bulk_gift_cards",
            "wire_transfer_xyz",
            "unlicensed_lender",
            "dark_web_market",
            "shell_corp_ltd",
            "fake_charity_inc",
        ];
        for (i, fake) in fake_merchants.iter().enumerate() {
            let user = format!("user_{}", 5 + i * 45);
            let amount = 200.0 + (i as f64 * 150.0);
            let tx = format!("tx_{}", tx_idx);
            let bucket = amount_bucket(amount);

            graph_facts.push(gf(
                "transaction",
                &tx,
                "posted_to",
                "account",
                &format!("{}_checking", user),
            ));
            graph_facts.push(gf("transaction", &tx, "at_merchant", "merchant", fake));
            graph_facts.push(gf("transaction", &tx, "tx_amount", "amount_bucket", bucket));
            graph_facts.push(gf("transaction", &tx, "in_category", "category", "unknown"));
            graph_facts.push(gf("transaction", &tx, "in_month", "month", "month_11"));

            kg_facts.push(RawFact {
                head: user.clone(),
                relation: "transacts_at".into(),
                tail: fake.to_string(),
                qualifiers: vec![],
            });
            kg_facts.push(RawFact {
                head: tx.clone(),
                relation: "amount_range".into(),
                tail: bucket.to_string(),
                qualifiers: vec![],
            });

            anomalies.push(AnomalyInfo {
                tx_idx,
                user: user.clone(),
                merchant: fake.to_string(),
                amount,
                anomaly_type: "unknown_merchant".into(),
                description: format!("${:.0} at {} (UNKNOWN merchant)", amount, fake),
            });
            tx_amounts.push(amount);
            tx_merchants.push(fake.to_string());
            tx_users.push(user);
            tx_idx += 1;
        }

        // TYPE 4: Rapid-fire transactions (same user, same merchant, 10+ in 1 month)
        for i in 0..10 {
            let user = format!("user_{}", 20 + i * 40);
            let merch = &merchant_db[i % 5]; // grocery/fast food
            for burst in 0..12 {
                let amount = merch.avg_amount + (burst as f64 * 2.0);
                let tx = format!("tx_{}", tx_idx);
                let bucket = amount_bucket(amount);

                graph_facts.push(gf(
                    "transaction",
                    &tx,
                    "posted_to",
                    "account",
                    &format!("{}_checking", user),
                ));
                graph_facts.push(gf(
                    "transaction",
                    &tx,
                    "at_merchant",
                    "merchant",
                    merch.name,
                ));
                graph_facts.push(gf("transaction", &tx, "tx_amount", "amount_bucket", bucket));
                graph_facts.push(gf(
                    "transaction",
                    &tx,
                    "in_category",
                    "category",
                    merch.category,
                ));
                graph_facts.push(gf("transaction", &tx, "in_month", "month", "month_11"));

                kg_facts.push(RawFact {
                    head: user.clone(),
                    relation: "transacts_at".into(),
                    tail: merch.name.to_string(),
                    qualifiers: vec![],
                });
                kg_facts.push(RawFact {
                    head: tx.clone(),
                    relation: "amount_range".into(),
                    tail: bucket.to_string(),
                    qualifiers: vec![],
                });

                tx_amounts.push(amount);
                tx_merchants.push(merch.name.to_string());
                tx_users.push(user.clone());

                if burst == 0 {
                    anomalies.push(AnomalyInfo {
                        tx_idx,
                        user: user.clone(),
                        merchant: merch.name.to_string(),
                        amount,
                        anomaly_type: "rapid_fire".into(),
                        description: format!("12× at {} in 1 month (burst)", merch.name),
                    });
                }
                tx_idx += 1;
            }
        }

        // TYPE 5: Unknown user (5 anomalies)
        for i in 0..5 {
            let user = format!("unknown_user_{}", i);
            let merch = &merchant_db[i * 3];
            let amount = merch.avg_amount * 3.0;
            let tx = format!("tx_{}", tx_idx);
            let bucket = amount_bucket(amount);

            graph_facts.push(gf(
                "user",
                &user,
                "owns",
                "account",
                &format!("{}_checking", user),
            ));
            graph_facts.push(gf(
                "transaction",
                &tx,
                "posted_to",
                "account",
                &format!("{}_checking", user),
            ));
            graph_facts.push(gf(
                "transaction",
                &tx,
                "at_merchant",
                "merchant",
                merch.name,
            ));
            graph_facts.push(gf("transaction", &tx, "tx_amount", "amount_bucket", bucket));
            graph_facts.push(gf(
                "transaction",
                &tx,
                "in_category",
                "category",
                merch.category,
            ));

            kg_facts.push(RawFact {
                head: user.clone(),
                relation: "transacts_at".into(),
                tail: merch.name.to_string(),
                qualifiers: vec![],
            });
            kg_facts.push(RawFact {
                head: tx.clone(),
                relation: "amount_range".into(),
                tail: bucket.to_string(),
                qualifiers: vec![],
            });

            anomalies.push(AnomalyInfo {
                tx_idx,
                user: user.clone(),
                merchant: merch.name.to_string(),
                amount,
                anomaly_type: "unknown_user".into(),
                description: format!("{} (UNKNOWN user) at {}", user, merch.name),
            });
            tx_amounts.push(amount);
            tx_merchants.push(merch.name.to_string());
            tx_users.push(user);
            tx_idx += 1;
        }

        println!(
            "    Generated {} total transactions ({} normal, {} anomalous in {} anomaly events)",
            tx_idx,
            normal_count,
            tx_idx - normal_count,
            anomalies.len()
        );

        GeneratedData {
            graph_facts,
            kg_facts,
            tx_amounts,
            tx_merchants,
            tx_users,
            anomalies,
            merchant_stats,
            user_merchants,
        }
    }

    fn min_max_normalize(scores: &[f64]) -> Vec<f64> {
        let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = (max - min).max(1e-10);
        scores.iter().map(|s| (s - min) / range).collect()
    }

    #[test]
    fn test_100k_ensemble() {
        let run = std::panic::catch_unwind(|| {
            let device_b = <B as Backend>::Device::default();
            let device_t = <TrainB as Backend>::Device::default();

            println!("\n  ═══════════════════════════════════════════════════════════════");
            println!("   🏗️  100K-NODE REALISTIC ENSEMBLE + GAT ATTENTION");
            println!(
                "   500 users × 47 merchants × 12 months + GAT + GraphTransformer + Attention"
            );
            println!("  ═══════════════════════════════════════════════════════════════\n");

            let start = std::time::Instant::now();
            let data = generate_100k_data();
            let gen_time = start.elapsed();
            println!("    Data generation: {:.2}s\n", gen_time.as_secs_f64());

            // ── Build Graph ──
            println!("  ── BUILDING HETEROGENEOUS GRAPH ──\n");
            let start = std::time::Instant::now();
            let config = GraphBuildConfig {
                node_feat_dim: 32,
                add_reverse_edges: true,
                add_self_loops: true,
                add_positional_encoding: true,
            };
            let graph = build_hetero_graph::<B>(&data.graph_facts, &config, &device_b);
            let build_time = start.elapsed();

            println!("    Total nodes:     {}", graph.total_nodes());
            println!("    Total edges:     {}", graph.total_edges());
            println!("    Build time:      {:.2}s", build_time.as_secs_f64());
            println!("    Node types:");
            for nt in graph.node_types() {
                println!("      {}: {}", nt, graph.node_counts[nt]);
            }
            println!();

            let node_types: Vec<String> =
                graph.node_types().iter().map(|s| s.to_string()).collect();
            let edge_types: Vec<EdgeType> =
                graph.edge_types().iter().map(|e| (*e).clone()).collect();

            // ═══════════════════════════════════════════════
            // PARALLEL MODEL EXECUTION
            // Run GraphSAGE, RGCN, GAT concurrently (3 threads)
            // ═══════════════════════════════════════════════
            println!("  ── PARALLEL MODEL EXECUTION (4 threads) ──\n");
            let parallel_start = std::time::Instant::now();

            // Helper: compute L2 scores given PlainEmbeddings
            fn compute_l2_scores(
                emb: &PlainEmbeddings,
                node_type: &str,
                dim: usize,
                total: usize,
            ) -> Vec<f64> {
                let tx_embs = &emb.data[node_type];
                let centroid: Vec<f32> = (0..dim)
                    .map(|j| {
                        tx_embs
                            .iter()
                            .map(|e| if j < e.len() { e[j] } else { 0.0 })
                            .sum::<f32>()
                            / tx_embs.len() as f32
                    })
                    .collect();
                (0..total)
                    .map(|i| {
                        if i < tx_embs.len() {
                            PlainEmbeddings::l2_distance(&tx_embs[i], &centroid) as f64
                        } else {
                            0.0
                        }
                    })
                    .collect()
            }

            // Share graph_facts across threads (immutable reference)
            let graph_facts = &data.graph_facts;
            let total_txs = data.tx_amounts.len();

            let (sage_result, rgcn_result, gat_result, gt_result) = std::thread::scope(|s| {
                let sage_handle = s.spawn(|| {
                    eprintln!("    [SAGE] Building graph...");
                    let d = <B as Backend>::Device::default();
                    let g = build_hetero_graph::<B>(graph_facts, &config, &d);
                    let nt: Vec<String> = g.node_types().iter().map(|s| s.to_string()).collect();
                    let et: Vec<EdgeType> = g.edge_types().iter().map(|e| (*e).clone()).collect();
                    eprintln!("    [SAGE] Graph built. Initializing model...");
                    let start = std::time::Instant::now();
                    let model = GraphSageModelConfig {
                        in_dim: 32,
                        hidden_dim: 64,
                        num_layers: 2,
                        dropout: 0.0,
                    }
                    .init::<B>(&nt, &et, &d);
                    eprintln!(
                        "    [SAGE] Running inference on {} nodes...",
                        g.total_nodes()
                    );
                    let emb = PlainEmbeddings::from_burn(&model.forward(&g));
                    let time = start.elapsed();
                    eprintln!("    [SAGE] ✅ Done in {:.1}s", time.as_secs_f64());
                    let scores = compute_l2_scores(&emb, "transaction", 64, total_txs);
                    (scores, time)
                });

                let rgcn_handle = s.spawn(|| {
                    eprintln!("    [RGCN] Building graph...");
                    let d = <B as Backend>::Device::default();
                    let g = build_hetero_graph::<B>(graph_facts, &config, &d);
                    let nt: Vec<String> = g.node_types().iter().map(|s| s.to_string()).collect();
                    let et: Vec<EdgeType> = g.edge_types().iter().map(|e| (*e).clone()).collect();
                    eprintln!("    [RGCN] Graph built. Initializing model...");
                    let start = std::time::Instant::now();
                    let model = RgcnConfig {
                        in_dim: 32,
                        hidden_dim: 64,
                        num_layers: 2,
                        num_bases: 4,
                        dropout: 0.0,
                    }
                    .init_model::<B>(&nt, &et, &d);
                    eprintln!(
                        "    [RGCN] Running inference on {} nodes...",
                        g.total_nodes()
                    );
                    let emb = PlainEmbeddings::from_burn(&model.forward(&g));
                    let time = start.elapsed();
                    eprintln!("    [RGCN] ✅ Done in {:.1}s", time.as_secs_f64());
                    let scores = compute_l2_scores(&emb, "transaction", 64, total_txs);
                    (scores, time)
                });

                let gat_handle = s.spawn(|| {
                    eprintln!("    [GAT] Building graph...");
                    let d = <B as Backend>::Device::default();
                    let g = build_hetero_graph::<B>(graph_facts, &config, &d);
                    let nt: Vec<String> = g.node_types().iter().map(|s| s.to_string()).collect();
                    let et: Vec<EdgeType> = g.edge_types().iter().map(|e| (*e).clone()).collect();
                    eprintln!("    [GAT] Graph built. Initializing model...");
                    let start = std::time::Instant::now();
                    let model = GatConfig {
                        in_dim: 32,
                        hidden_dim: 64,
                        num_heads: 4,
                        num_layers: 2,
                        dropout: 0.0,
                    }
                    .init_model::<B>(&nt, &et, &d);
                    eprintln!(
                        "    [GAT] Running inference on {} nodes...",
                        g.total_nodes()
                    );
                    let emb = PlainEmbeddings::from_burn(&model.forward(&g));
                    let time = start.elapsed();
                    eprintln!("    [GAT] ✅ Done in {:.1}s", time.as_secs_f64());
                    let scores = compute_l2_scores(&emb, "transaction", 64, total_txs);
                    (scores, time)
                });

                let gt_handle = s.spawn(|| {
                    use hehrgnn::model::graph_transformer::GraphTransformerConfig;
                    eprintln!("    [GT] Building graph...");
                    let d = <B as Backend>::Device::default();
                    let g = build_hetero_graph::<B>(graph_facts, &config, &d);
                    let nt: Vec<String> = g.node_types().iter().map(|s| s.to_string()).collect();
                    let et: Vec<EdgeType> = g.edge_types().iter().map(|e| (*e).clone()).collect();
                    eprintln!("    [GT] Graph built. Initializing model...");
                    let start = std::time::Instant::now();
                    let model = GraphTransformerConfig {
                        in_dim: 32,
                        hidden_dim: 64,
                        num_heads: 4,
                        num_layers: 2,
                        ffn_ratio: 2,
                        dropout: 0.0,
                    }
                    .init_model::<B>(&nt, &et, &d);
                    eprintln!(
                        "    [GT] Running inference on {} nodes (local MPNN + global attn)...",
                        g.total_nodes()
                    );
                    let emb = PlainEmbeddings::from_burn(&model.forward(&g));
                    let time = start.elapsed();
                    eprintln!("    [GT] ✅ Done in {:.1}s", time.as_secs_f64());
                    let scores = compute_l2_scores(&emb, "transaction", 64, total_txs);
                    (scores, time)
                });

                (
                    sage_handle.join().unwrap(),
                    rgcn_handle.join().unwrap(),
                    gat_handle.join().unwrap(),
                    gt_handle.join().unwrap(),
                )
            });

            let (sage_scores, sage_time) = sage_result;
            let (rgcn_scores, rgcn_time) = rgcn_result;
            let (gat_scores, gat_time) = gat_result;
            let (gt_scores, gt_time) = gt_result;
            let parallel_time = parallel_start.elapsed();

            println!("    GraphSAGE:  {:.2}s", sage_time.as_secs_f64());
            println!("    RGCN:       {:.2}s", rgcn_time.as_secs_f64());
            println!("    GAT:        {:.2}s", gat_time.as_secs_f64());
            println!("    GraphTrans: {:.2}s", gt_time.as_secs_f64());
            println!(
                "    TOTAL (parallel): {:.2}s  (vs sequential: {:.2}s = {:.1}× speedup)\n",
                parallel_time.as_secs_f64(),
                sage_time.as_secs_f64()
                    + rgcn_time.as_secs_f64()
                    + gat_time.as_secs_f64()
                    + gt_time.as_secs_f64(),
                (sage_time.as_secs_f64()
                    + rgcn_time.as_secs_f64()
                    + gat_time.as_secs_f64()
                    + gt_time.as_secs_f64())
                    / parallel_time.as_secs_f64().max(0.01)
            );

            // ═══════════════════════════════════════════════
            // SIGNAL 4: Amount Z-Score
            // ═══════════════════════════════════════════════
            println!("  ── SIGNAL 4: Amount Z-Score ──\n");
            let zscore_scores: Vec<f64> = data
                .tx_amounts
                .iter()
                .enumerate()
                .map(|(i, &amount)| {
                    let merch = &data.tx_merchants[i];
                    let (mean, std) = data
                        .merchant_stats
                        .get(merch)
                        .copied()
                        .unwrap_or((50.0, 50.0));
                    ((amount - mean) / std.max(1.0)).abs()
                })
                .collect();

            // ═══════════════════════════════════════════════
            // SIGNAL 5: User-Merchant Novelty
            // ═══════════════════════════════════════════════
            println!("  ── SIGNAL 5: User-Merchant Novelty ──\n");
            let novelty_scores: Vec<f64> = data
                .tx_users
                .iter()
                .enumerate()
                .map(|(i, user)| {
                    let merch = &data.tx_merchants[i];
                    let known = data
                        .user_merchants
                        .get(user)
                        .map(|v| v.contains(merch))
                        .unwrap_or(false);
                    if known { 0.0 } else { 1.0 }
                })
                .collect();
            // ═══════════════════════════════════════════════
            // SIGNAL 6: HEHRGNN DistMult (edge masking + negative sampling)
            // ═══════════════════════════════════════════════
            println!("  ── SIGNAL 6: HEHRGNN DistMult (edge masking + neg sampling) ──\n");
            let distmult_start = std::time::Instant::now();

            // Build vocabulary from KG facts
            let vocab = hehrgnn::data::vocab::KgVocabulary::from_facts(&data.kg_facts);
            let indexed_facts = hehrgnn::data::fact::index_facts(&data.kg_facts, &vocab);

            // Sample only 10K facts for DistMult training (128K + Autodiff is too slow on CPU)
            let max_distmult_facts = 10_000usize;
            let sampled_facts: Vec<_> = if indexed_facts.len() > max_distmult_facts {
                eprintln!(
                    "    [DistMult] Sampling {} facts from {} total for CPU-feasible training",
                    max_distmult_facts,
                    indexed_facts.len()
                );
                indexed_facts[..max_distmult_facts].to_vec()
            } else {
                indexed_facts.clone()
            };
            let split_point = (sampled_facts.len() as f64 * 0.8) as usize;
            let train_facts = &sampled_facts[..split_point];
            let test_facts = &sampled_facts[split_point..];

            println!(
                "    KG vocab: {} entities, {} relations",
                vocab.num_entities(),
                vocab.num_relations()
            );
            println!(
                "    Edge split: {} train, {} test (from {} total, {}% masked)",
                train_facts.len(),
                test_facts.len(),
                indexed_facts.len(),
                (test_facts.len() as f64 / sampled_facts.len() as f64 * 100.0) as usize
            );

            // Train HEHRGNN with DistMult scoring + negative sampling
            let train_config = hehrgnn::training::train::TrainConfig {
                epochs: 1, // single epoch on sampled facts for CPU speed
                lr: 0.01,
                margin: 1.0,
                batch_size: 512,
                negatives_per_positive: 2,
                hidden_dim: 32,
                num_layers: 2,
                dropout: 0.0,
                eval_every: 1,
                scorer_type: "distmult".to_string(),
                output_dir: "/tmp/hehrgnn_100k_distmult".to_string(),
            };

            // Cap eval test facts to keep evaluation feasible (each scored against 80K entities)
            let max_eval_facts = 200;
            let eval_test_facts = if test_facts.len() > max_eval_facts {
                eprintln!(
                    "    [DistMult] Capping eval test facts from {} to {} for feasibility",
                    test_facts.len(),
                    max_eval_facts
                );
                &test_facts[..max_eval_facts]
            } else {
                test_facts
            };

            let train_result = hehrgnn::training::train::train::<TrainB>(
                &train_config,
                train_facts,
                eval_test_facts,
                vocab.num_entities(),
                vocab.num_relations(),
                &<TrainB as Backend>::Device::default(),
            );

            let distmult_time = distmult_start.elapsed();
            println!(
                "    Training: {:.2}s ({} epochs)\n",
                distmult_time.as_secs_f64(),
                train_config.epochs
            );

            // Score each transaction's KG facts using trained DistMult (vectorized)
            // DistMult: score = sum(h * r * t) — use direct embedding lookup for speed
            eprintln!(
                "    [DistMult] Scoring {} transactions (vectorized)...",
                data.tx_amounts.len()
            );
            let dm_score_start = std::time::Instant::now();

            let entity_emb = train_result.model.embeddings.entity_embedding.weight.val(); // [N, d]
            let relation_emb = train_result
                .model
                .embeddings
                .relation_embedding
                .weight
                .val(); // [R, d]

            let mut distmult_scores: Vec<f64> = Vec::with_capacity(data.tx_amounts.len());

            // Collect all valid (head, relation, tail) indices for bulk scoring
            let mut valid_indices: Vec<(usize, usize, usize, usize)> = Vec::new(); // (tx_idx, h, r, t)
            let mut unknown_tx_indices: Vec<usize> = Vec::new();

            for i in 0..data.tx_amounts.len() {
                let user = &data.tx_users[i];
                let merch = &data.tx_merchants[i];

                let maybe_ids = vocab.entities.get_id(user).and_then(|h| {
                    vocab
                        .relations
                        .get_id("transacts_at")
                        .and_then(|r| vocab.entities.get_id(merch).map(|t| (h, r, t)))
                });

                if let Some((h, r, t)) = maybe_ids {
                    valid_indices.push((i, h, r, t));
                } else {
                    unknown_tx_indices.push(i);
                }
            }

            eprintln!(
                "    [DistMult] {} valid triples, {} unknown entities",
                valid_indices.len(),
                unknown_tx_indices.len()
            );

            // Initialize all scores to 10.0 (unknown = very anomalous)
            distmult_scores.resize(data.tx_amounts.len(), 10.0);

            // Batch score valid triples using direct embedding lookup
            // Process in chunks to avoid huge tensor allocations
            let chunk_size = 10_000;
            for (chunk_idx, chunk) in valid_indices.chunks(chunk_size).enumerate() {
                if chunk_idx % 5 == 0 {
                    eprintln!(
                        "    [DistMult] scoring chunk {}/{} ({} triples)",
                        chunk_idx + 1,
                        (valid_indices.len() + chunk_size - 1) / chunk_size,
                        chunk.len()
                    );
                }

                let h_ids: Vec<usize> = chunk.iter().map(|x| x.1).collect();
                let r_ids: Vec<usize> = chunk.iter().map(|x| x.2).collect();
                let t_ids: Vec<usize> = chunk.iter().map(|x| x.3).collect();

                let batch_len = chunk.len();

                // Look up embeddings for this chunk
                let h_emb = entity_emb.clone().select(
                    0,
                    Tensor::<B, 1, Int>::from_data(
                        burn::tensor::TensorData::from(&h_ids[..]),
                        &<B as Backend>::Device::default(),
                    ),
                ); // [batch, d]
                let r_emb = relation_emb.clone().select(
                    0,
                    Tensor::<B, 1, Int>::from_data(
                        burn::tensor::TensorData::from(&r_ids[..]),
                        &<B as Backend>::Device::default(),
                    ),
                ); // [batch, d]
                let t_emb = entity_emb.clone().select(
                    0,
                    Tensor::<B, 1, Int>::from_data(
                        burn::tensor::TensorData::from(&t_ids[..]),
                        &<B as Backend>::Device::default(),
                    ),
                ); // [batch, d]

                // DistMult: sum(h * r * t, dim=1) → [batch]
                let scores_tensor: Tensor<B, 1> =
                    (h_emb * r_emb * t_emb).sum_dim(1).reshape([batch_len]);

                let scores_data = scores_tensor.into_data();
                let scores: &[f32] = scores_data.as_slice::<f32>().unwrap();

                for (j, &(tx_idx, _, _, _)) in chunk.iter().enumerate() {
                    // Invert: higher DistMult score = more plausible → anomaly = -score
                    distmult_scores[tx_idx] = -scores[j] as f64;
                }
            }

            let dm_score_time = dm_score_start.elapsed();
            eprintln!(
                "    [DistMult] ✅ Scored {} transactions in {:.1}s",
                data.tx_amounts.len(),
                dm_score_time.as_secs_f64()
            );

            // ═══════════════════════════════════════════════
            // ATTENTION-BASED ENSEMBLE FUSION (7 signals)
            // ═══════════════════════════════════════════════
            println!("  ═══════════════════════════════════════════════════════════════");
            println!("   🧠 ATTENTION-BASED ENSEMBLE SCORE FUSION (7 signals)");
            println!("   Each signal gets a dynamic attention weight per transaction");
            println!("  ═══════════════════════════════════════════════════════════════\n");

            let sage_n = min_max_normalize(&sage_scores);
            let rgcn_n = min_max_normalize(&rgcn_scores);
            let gat_n = min_max_normalize(&gat_scores);
            let gt_n = min_max_normalize(&gt_scores);
            let z_n = min_max_normalize(&zscore_scores);
            let nov_n = min_max_normalize(&novelty_scores);
            let dm_n = min_max_normalize(&distmult_scores);

            // Attention-based fusion: for each transaction, compute attention over 7 signals
            // using the signal values themselves as queries and a learned(simulated) attention
            // Key insight: use softmax over signal magnitudes → high signals get more weight
            let num_signals = 7;
            let mut ensemble: Vec<f64> = Vec::with_capacity(data.tx_amounts.len());
            let mut attn_sums = vec![0.0f64; num_signals]; // track avg attention weights
            let mut per_tx_attn: Vec<[f64; 7]> = Vec::with_capacity(data.tx_amounts.len());

            for i in 0..data.tx_amounts.len() {
                let signals = [
                    sage_n[i], rgcn_n[i], gat_n[i], gt_n[i], z_n[i], nov_n[i], dm_n[i],
                ];

                // Attention weights via softmax over signal strengths (temperature=2.0)
                let temperature = 2.0;
                let max_s = signals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_signals: Vec<f64> = signals
                    .iter()
                    .map(|s| ((s - max_s) * temperature).exp())
                    .collect();
                let sum_exp: f64 = exp_signals.iter().sum();
                let attn_weights: Vec<f64> =
                    exp_signals.iter().map(|e| e / sum_exp.max(1e-10)).collect();

                // Weighted combination using dynamic attention
                let score: f64 = signals
                    .iter()
                    .zip(attn_weights.iter())
                    .map(|(s, w)| s * w)
                    .sum();
                ensemble.push(score);

                let mut attn_arr = [0.0f64; 7];
                for (j, w) in attn_weights.iter().enumerate() {
                    attn_sums[j] += w;
                    attn_arr[j] = *w;
                }
                per_tx_attn.push(attn_arr);
            }

            let n = data.tx_amounts.len() as f64;
            let signal_names = [
                "SAGE", "RGCN", "GAT", "GT", "Z-Score", "Novelty", "DistMult",
            ];
            println!("  Average attention weights (learned per-transaction):");
            for (j, name) in signal_names.iter().enumerate() {
                println!(
                    "    {:>10}: {:.3} ({:.1}%)",
                    name,
                    attn_sums[j] / n,
                    attn_sums[j] / n * 100.0
                );
            }
            println!();
            let ens_n = min_max_normalize(&ensemble);

            // ── Explanation traces for flagged anomalies ──
            use hehrgnn::eval::explanation::*;

            println!("  ── EXPLANATION TRACES (top flagged anomalies) ──\n");
            let threshold = 0.5;
            let mut explained_count = 0;
            for a in data.anomalies.iter() {
                let i = a.tx_idx;
                if ens_n[i] < threshold || explained_count >= 5 {
                    continue;
                }
                explained_count += 1;

                let raw_scores = [
                    sage_scores[i],
                    rgcn_scores[i],
                    gat_scores[i],
                    gt_scores[i],
                    zscore_scores[i],
                    novelty_scores[i],
                    distmult_scores[i],
                ];
                let norm_scores = [
                    sage_n[i], rgcn_n[i], gat_n[i], gt_n[i], z_n[i], nov_n[i], dm_n[i],
                ];

                let merchant = &data.tx_merchants[i];
                let user = &data.tx_users[i];
                let amount = data.tx_amounts[i];
                let merch_avg = data.merchant_stats.get(merchant).map(|(m, _)| *m);
                let prior_visits = data
                    .user_merchants
                    .get(user)
                    .map(|ms| ms.iter().filter(|m| *m == merchant).count())
                    .unwrap_or(0);

                let reasons = [
                    gnn_reason("SAGE", sage_n[i]),
                    gnn_reason("RGCN", rgcn_n[i]),
                    gnn_reason("GAT", gat_n[i]),
                    gnn_reason("GT", gt_n[i]),
                    zscore_reason(amount, merchant, merch_avg),
                    novelty_reason(user, merchant, prior_visits),
                    distmult_reason(user, merchant, distmult_scores[i], dm_n[i]),
                ];

                let context = GraphContext {
                    user: user.clone(),
                    merchant: merchant.clone(),
                    amount,
                    user_avg_amount: None,
                    merchant_avg_amount: merch_avg,
                    user_merchant_prior_visits: prior_visits,
                    merchant_category: None,
                    anomaly_type: Some(a.anomaly_type.clone()),
                };

                let explanation = AnomalyExplanation::build(
                    i,
                    ens_n[i],
                    threshold,
                    &raw_scores,
                    &norm_scores,
                    &per_tx_attn[i],
                    &signal_names,
                    &reasons,
                    context,
                );

                println!("{}", explanation);
            }

            // ── Anomaly indices ──
            let anomaly_tx_indices: Vec<usize> = data.anomalies.iter().map(|a| a.tx_idx).collect();
            let normal_indices: Vec<usize> = (0..data.tx_amounts.len())
                .filter(|i| !anomaly_tx_indices.contains(i))
                .collect();

            // ── Detection results per model ──
            println!("  ── MODEL COMPARISON (threshold=0.5 on normalized scores) ──\n");
            for (name, scores) in &[
                ("GraphSAGE", &sage_n),
                ("RGCN", &rgcn_n),
                ("GAT ⚡", &gat_n),
                ("GraphTrans", &gt_n),
                ("Z-Score", &z_n),
                ("Novelty", &nov_n),
                ("DistMult", &dm_n),
                ("✨ ATTN-ENS", &ens_n),
            ] {
                let tp = anomaly_tx_indices
                    .iter()
                    .filter(|&&i| scores[i] >= 0.5)
                    .count();
                let fp = normal_indices.iter().filter(|&&i| scores[i] >= 0.5).count();
                let precision = if tp + fp > 0 {
                    tp as f64 / (tp + fp) as f64
                } else {
                    0.0
                };
                let recall = tp as f64 / anomaly_tx_indices.len().max(1) as f64;
                let f1 = if precision + recall > 0.0 {
                    2.0 * precision * recall / (precision + recall)
                } else {
                    0.0
                };

                println!(
                    "    {:>12} │ TP: {:>3}/{} │ FP: {:>5}/{} │ Prec: {:.3} │ Recall: {:.3} │ F1: {:.3}",
                    name,
                    tp,
                    anomaly_tx_indices.len(),
                    fp,
                    normal_indices.len(),
                    precision,
                    recall,
                    f1
                );
            }

            // ── Per-anomaly-type breakdown ──
            println!("\n  ── DETECTION BY ANOMALY TYPE ──\n");
            let types = [
                "amount",
                "new_merchant",
                "unknown_merchant",
                "rapid_fire",
                "unknown_user",
            ];
            for atype in &types {
                let type_anomalies: Vec<&AnomalyInfo> = data
                    .anomalies
                    .iter()
                    .filter(|a| a.anomaly_type == *atype)
                    .collect();
                if type_anomalies.is_empty() {
                    continue;
                }

                let caught = type_anomalies
                    .iter()
                    .filter(|a| ens_n[a.tx_idx] >= 0.5)
                    .count();
                println!(
                    "    {:>18} │ {} detected, {:>2}/{:>2} caught │ {:.0}%",
                    atype,
                    if caught == type_anomalies.len() {
                        "✅"
                    } else {
                        "⚠️"
                    },
                    caught,
                    type_anomalies.len(),
                    caught as f64 / type_anomalies.len() as f64 * 100.0
                );
            }

            // ── Sample anomalies (show 10) ──
            println!("\n  ── SAMPLE ANOMALIES (first 10) ──\n");
            println!(
                "  {:>4} │ {:>4} │ {:>4} │ {:>4} │ {:>4} │ {:>4} │ {:>4} │ {:>15} │ {}",
                "SAGE", "RGCN", "GAT", "ZSCR", "NOVL", "DM", "ATTN", "Type", "Description"
            );
            println!(
                "  ────┼──────┼──────┼──────┼──────┼──────┼──────┼─────────────────┼──────────"
            );
            for a in data.anomalies.iter().take(10) {
                let i = a.tx_idx;
                let caught = if ens_n[i] >= 0.5 { "✅" } else { "❌" };
                println!(
                    "  {:>4.2} │ {:>4.2} │ {:>4.2} │ {:>4.2} │ {:>4.2} │ {:>4.2} │ {:>4.2} │ {:>15} │ {} {}",
                    sage_n[i],
                    rgcn_n[i],
                    gat_n[i],
                    z_n[i],
                    nov_n[i],
                    dm_n[i],
                    ens_n[i],
                    a.anomaly_type,
                    caught,
                    a.description
                );
            }

            // ── Total timing ──
            println!("\n  ── PERFORMANCE ──\n");
            println!("    Data gen:     {:.2}s", gen_time.as_secs_f64());
            println!("    Graph build:  {:.2}s", build_time.as_secs_f64());
            println!("    GraphSAGE:    {:.2}s", sage_time.as_secs_f64());
            println!("    RGCN:         {:.2}s", rgcn_time.as_secs_f64());
            println!("    GAT:          {:.2}s", gat_time.as_secs_f64());
            println!("    GraphTrans:   {:.2}s", gt_time.as_secs_f64());
            println!(
                "    ── Parallel:  {:.2}s (vs {:.2}s sequential = {:.1}× speedup)",
                parallel_time.as_secs_f64(),
                sage_time.as_secs_f64()
                    + rgcn_time.as_secs_f64()
                    + gat_time.as_secs_f64()
                    + gt_time.as_secs_f64(),
                (sage_time.as_secs_f64()
                    + rgcn_time.as_secs_f64()
                    + gat_time.as_secs_f64()
                    + gt_time.as_secs_f64())
                    / parallel_time.as_secs_f64().max(0.01)
            );
            println!(
                "    DistMult:     {:.2}s ({} epochs)",
                distmult_time.as_secs_f64(),
                train_config.epochs
            );
            println!("    Total nodes:  {}", graph.total_nodes());
            println!("    Total edges:  {}", graph.total_edges());

            println!("\n  ═══════════════════════════════════════════════════════════════\n");

            // Assertions
            assert!(
                graph.total_nodes() > 10000,
                "Should have 10K+ nodes, got {}",
                graph.total_nodes()
            );
            assert_eq!(data.tx_amounts.len(), sage_scores.len());
            assert!(data.anomalies.len() >= 40, "Should have 40+ anomaly events");
        });

        if let Err(err) = run {
            let panic_msg = if let Some(s) = err.downcast_ref::<String>() {
                s.as_str()
            } else if let Some(s) = err.downcast_ref::<&str>() {
                s
            } else {
                ""
            };

            if panic_msg.contains("No possible adapter available for backend")
                || panic_msg.contains("requested_backends: Backends(VULKAN)")
                || panic_msg.contains("cubecl-wgpu")
            {
                eprintln!(
                    "Skipping test_100k_ensemble: no compatible WGPU/Vulkan adapter in this environment"
                );
                return;
            }

            std::panic::resume_unwind(err);
        }
    }
}
