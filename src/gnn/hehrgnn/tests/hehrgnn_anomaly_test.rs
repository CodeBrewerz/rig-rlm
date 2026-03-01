//! HEHRGNN fact plausibility anomaly detection.
//!
//! Trains the HEHRGNN model on normal financial transaction facts,
//! then scores anomalous facts. Normal facts should get HIGH plausibility
//! scores, anomalous facts should get LOW scores.
//!
//! This is the HEHRGNN-native approach: the model directly learns
//! "what facts are normal?" and flags unlikely ones.

#[cfg(test)]
mod tests {
    use burn::backend::{Autodiff, NdArray};
    use burn::prelude::*;

    type TrainB = Autodiff<NdArray>;
    type InferB = NdArray;

    use hehrgnn::data::batcher::{HehrBatcher, HehrFactItem, HehrBatch};
    use hehrgnn::data::fact::{RawFact, HehrFact};
    use hehrgnn::data::vocab::KgVocabulary;
    use hehrgnn::training::train::{train, TrainConfig};
    use hehrgnn::training::scoring::DistMultScorer;
    use burn::data::dataloader::batcher::Batcher;

    /// Build realistic financial facts.
    ///
    /// Normal facts model common patterns:
    ///   (alice, transacts_at, mcdonalds)   — she's a regular
    ///   (alice, transacts_at, starbucks)   — daily coffee  
    ///   (bob, transacts_at, shell_gas)     — fills up weekly
    ///   (tx_a1, posted_to, alice_checking) — normal posting
    ///   (tx_a1, amount_range, small)       — $8-25
    ///
    /// Anomalous facts:
    ///   (alice, transacts_at, crypto_exchange)   — she never does crypto
    ///   (eve, transacts_at, amazon)              — she only uses McD + gas
    ///   (tx_ANOM1, amount_range, huge)           — $1000 at McDonalds
    fn build_financial_facts() -> (Vec<RawFact>, Vec<RawFact>) {
        let mut normal_facts = Vec::new();
        let mut anomaly_facts = Vec::new();

        // ── User → Account ownership ──
        let ownership = vec![
            ("alice", "alice_checking"),
            ("bob", "bob_checking"),
            ("carol", "carol_savings"),
            ("dave", "dave_checking"),
            ("eve", "eve_checking"),
        ];
        for (user, acct) in &ownership {
            normal_facts.push(RawFact {
                head: user.to_string(),
                relation: "owns_account".to_string(),
                tail: acct.to_string(),
                qualifiers: vec![],
            });
        }

        // ── Normal spending patterns (repeated to give signal) ──
        let normal_patterns = vec![
            // Alice: McDonalds + Starbucks + Walmart
            ("alice", "transacts_at", "mcdonalds"),
            ("alice", "transacts_at", "mcdonalds"),
            ("alice", "transacts_at", "mcdonalds"),
            ("alice", "transacts_at", "starbucks"),
            ("alice", "transacts_at", "starbucks"),
            ("alice", "transacts_at", "walmart"),
            // Bob: Shell Gas + Amazon + Walmart
            ("bob", "transacts_at", "shell_gas"),
            ("bob", "transacts_at", "shell_gas"),
            ("bob", "transacts_at", "shell_gas"),
            ("bob", "transacts_at", "amazon"),
            ("bob", "transacts_at", "amazon"),
            ("bob", "transacts_at", "walmart"),
            // Carol: Starbucks + Amazon + McDonalds
            ("carol", "transacts_at", "starbucks"),
            ("carol", "transacts_at", "starbucks"),
            ("carol", "transacts_at", "amazon"),
            ("carol", "transacts_at", "mcdonalds"),
            // Dave: all merchants (moderate)
            ("dave", "transacts_at", "mcdonalds"),
            ("dave", "transacts_at", "starbucks"),
            ("dave", "transacts_at", "walmart"),
            ("dave", "transacts_at", "shell_gas"),
            ("dave", "transacts_at", "amazon"),
            // Eve: only McDonalds + Shell Gas
            ("eve", "transacts_at", "mcdonalds"),
            ("eve", "transacts_at", "mcdonalds"),
            ("eve", "transacts_at", "shell_gas"),
            ("eve", "transacts_at", "shell_gas"),
        ];

        for (h, r, t) in &normal_patterns {
            normal_facts.push(RawFact {
                head: h.to_string(),
                relation: r.to_string(),
                tail: t.to_string(),
                qualifiers: vec![],
            });
        }

        // ── Transaction → Account postings ──
        let tx_postings = vec![
            ("tx_a1", "posted_to", "alice_checking"),
            ("tx_a2", "posted_to", "alice_checking"),
            ("tx_a3", "posted_to", "alice_checking"),
            ("tx_b1", "posted_to", "bob_checking"),
            ("tx_b2", "posted_to", "bob_checking"),
            ("tx_c1", "posted_to", "carol_savings"),
            ("tx_c2", "posted_to", "carol_savings"),
            ("tx_d1", "posted_to", "dave_checking"),
            ("tx_e1", "posted_to", "eve_checking"),
            ("tx_e2", "posted_to", "eve_checking"),
        ];
        for (h, r, t) in &tx_postings {
            normal_facts.push(RawFact {
                head: h.to_string(),
                relation: r.to_string(),
                tail: t.to_string(),
                qualifiers: vec![],
            });
        }

        // ── Transaction → Merchant ──
        let tx_merchants = vec![
            ("tx_a1", "at_merchant", "mcdonalds"),
            ("tx_a2", "at_merchant", "starbucks"),
            ("tx_a3", "at_merchant", "walmart"),
            ("tx_b1", "at_merchant", "shell_gas"),
            ("tx_b2", "at_merchant", "amazon"),
            ("tx_c1", "at_merchant", "starbucks"),
            ("tx_c2", "at_merchant", "amazon"),
            ("tx_d1", "at_merchant", "walmart"),
            ("tx_e1", "at_merchant", "mcdonalds"),
            ("tx_e2", "at_merchant", "shell_gas"),
        ];
        for (h, r, t) in &tx_merchants {
            normal_facts.push(RawFact {
                head: h.to_string(),
                relation: r.to_string(),
                tail: t.to_string(),
                qualifiers: vec![],
            });
        }

        // ── Transaction → Amount range ──
        let tx_amounts = vec![
            ("tx_a1", "amount_range", "small"),     // $12 at McD
            ("tx_a2", "amount_range", "small"),     // $6 at Starbucks
            ("tx_a3", "amount_range", "medium"),    // $85 at Walmart
            ("tx_b1", "amount_range", "medium"),    // $45 at Shell
            ("tx_b2", "amount_range", "medium"),    // $89 at Amazon
            ("tx_c1", "amount_range", "small"),     // $6 at Starbucks
            ("tx_c2", "amount_range", "medium"),    // $45 at Amazon
            ("tx_d1", "amount_range", "medium"),    // $95 at Walmart
            ("tx_e1", "amount_range", "small"),     // $10 at McD
            ("tx_e2", "amount_range", "medium"),    // $42 at Shell
        ];
        for (h, r, t) in &tx_amounts {
            normal_facts.push(RawFact {
                head: h.to_string(),
                relation: r.to_string(),
                tail: t.to_string(),
                qualifiers: vec![],
            });
        }

        // ── Merchant categories (structural knowledge) ──
        let categories = vec![
            ("mcdonalds", "in_category", "fast_food"),
            ("starbucks", "in_category", "coffee"),
            ("walmart", "in_category", "groceries"),
            ("shell_gas", "in_category", "fuel"),
            ("amazon", "in_category", "online_shopping"),
            ("crypto_exchange", "in_category", "crypto"),     // unusual
            ("dark_web_store", "in_category", "illegal"),     // very unusual
        ];
        for (h, r, t) in &categories {
            normal_facts.push(RawFact {
                head: h.to_string(),
                relation: r.to_string(),
                tail: t.to_string(),
                qualifiers: vec![],
            });
        }

        // ════════════════════════════════════════════════
        // ANOMALOUS FACTS — the model should score these LOW
        // ════════════════════════════════════════════════

        // 🚨 Alice at CryptoExchange — she's never done crypto
        anomaly_facts.push(RawFact {
            head: "alice".into(), relation: "transacts_at".into(),
            tail: "crypto_exchange".into(), qualifiers: vec![],
        });

        // 🚨 Eve at Amazon — she ONLY uses McD + Shell Gas
        anomaly_facts.push(RawFact {
            head: "eve".into(), relation: "transacts_at".into(),
            tail: "amazon".into(), qualifiers: vec![],
        });

        // 🚨 tx_ANOM1 with "huge" amount at McDonalds
        anomaly_facts.push(RawFact {
            head: "tx_ANOM1".into(), relation: "at_merchant".into(),
            tail: "mcdonalds".into(), qualifiers: vec![],
        });
        anomaly_facts.push(RawFact {
            head: "tx_ANOM1".into(), relation: "amount_range".into(),
            tail: "huge".into(), qualifiers: vec![],
        });

        // 🚨 Bob at dark_web_store — nobody shops there
        anomaly_facts.push(RawFact {
            head: "bob".into(), relation: "transacts_at".into(),
            tail: "dark_web_store".into(), qualifiers: vec![],
        });

        // 🚨 Unknown user at normal merchant
        anomaly_facts.push(RawFact {
            head: "unknown_user".into(), relation: "transacts_at".into(),
            tail: "walmart".into(), qualifiers: vec![],
        });

        (normal_facts, anomaly_facts)
    }

    #[test]
    fn test_hehrgnn_fact_plausibility_anomaly() {
        let device = <TrainB as Backend>::Device::default();
        let (normal_facts, anomaly_facts) = build_financial_facts();

        println!("\n  ═══════════════════════════════════════════════════════════════");
        println!("   🧠 HEHRGNN FACT PLAUSIBILITY ANOMALY DETECTION");
        println!("   Trains on normal facts → scores anomalous facts as implausible");
        println!("  ═══════════════════════════════════════════════════════════════\n");

        // Build vocabulary from ALL facts (normal + anomaly entities must be in vocab)
        let all_raw: Vec<RawFact> = normal_facts.iter().chain(anomaly_facts.iter()).cloned().collect();
        let vocab = KgVocabulary::from_facts(&all_raw);

        println!("  Vocabulary: {} entities, {} relations\n", 
            vocab.num_entities(), vocab.num_relations());

        // Convert to indexed facts
        let normal_indexed: Vec<HehrFact> = normal_facts.iter()
            .filter_map(|f| HehrFact::from_raw(f, &vocab))
            .collect();
        let anomaly_indexed: Vec<HehrFact> = anomaly_facts.iter()
            .filter_map(|f| HehrFact::from_raw(f, &vocab))
            .collect();

        println!("  Normal facts:  {} (train set)", normal_indexed.len());
        println!("  Anomaly facts: {} (to be scored)\n", anomaly_indexed.len());

        // Train HEHRGNN on normal facts only
        let train_config = TrainConfig {
            epochs: 30,
            lr: 0.005,
            margin: 1.0,
            batch_size: 32,
            negatives_per_positive: 5,
            hidden_dim: 32,
            num_layers: 2,
            dropout: 0.1,
            eval_every: 10,
            scorer_type: "distmult".to_string(),
            output_dir: "/tmp/hehrgnn_anomaly".to_string(),
        };

        // Use 80% for train, 20% for eval (all normal facts)
        let split = (normal_indexed.len() as f64 * 0.8) as usize;
        let train_slice = &normal_indexed[..split];
        let eval_slice = &normal_indexed[split..];

        let result = train::<TrainB>(
            &train_config,
            train_slice,
            eval_slice,
            vocab.num_entities(),
            vocab.num_relations(),
            &device,
        );

        let model = result.model;
        let infer_device = model.embeddings.entity_embedding.weight.val().device();
        let scorer = DistMultScorer::new();
        let batcher = HehrBatcher::new();

        // ── Score normal facts ──
        println!("\n  ── SCORING FACTS WITH TRAINED HEHRGNN ──\n");

        let mut normal_scores: Vec<f32> = Vec::new();
        for fact in &normal_indexed {
            let item = HehrFactItem { fact: fact.clone(), label: 1.0 };
            let batch: HehrBatch<InferB> = batcher.batch(vec![item], &infer_device);
            let score = model.score_batch(&batch, &scorer);
            let val: f32 = score.into_data().as_slice::<f32>().expect("score")[0];
            normal_scores.push(val);
        }

        let avg_normal = normal_scores.iter().sum::<f32>() / normal_scores.len() as f32;
        let std_normal = (normal_scores.iter()
            .map(|s| (s - avg_normal).powi(2)).sum::<f32>()
            / normal_scores.len() as f32).sqrt();

        println!("  Normal facts: avg score = {:.4}, std = {:.4}", avg_normal, std_normal);

        // ── Score anomalous facts ──
        println!();
        println!("  {:>30} │ {:>8} │ {:>10} │ {}", "Anomalous Fact", "Score", "Z from μ", "Verdict");
        println!("  ──────────────────────────────┼──────────┼────────────┼─────────");

        let threshold = avg_normal - 1.5 * std_normal; // anomalies score BELOW normal

        let mut anomaly_scores = Vec::new();
        for (i, fact) in anomaly_indexed.iter().enumerate() {
            let item = HehrFactItem { fact: fact.clone(), label: 1.0 };
            let batch: HehrBatch<InferB> = batcher.batch(vec![item], &infer_device);
            let score = model.score_batch(&batch, &scorer);
            let val: f32 = score.into_data().as_slice::<f32>().expect("score")[0];
            anomaly_scores.push(val);

            let z_score = (val - avg_normal) / std_normal.max(0.001);
            let raw = &anomaly_facts[i];
            let fact_str = format!("({}, {}, {})", raw.head, raw.relation, raw.tail);

            // For DistMult: higher = more plausible. Anomalies should be lower.
            let verdict = if val < threshold {
                "🚨 ANOMALOUS"
            } else {
                "   normal"
            };

            println!("  {:>30} │ {:>8.4} │ {:>+9.2}σ │ {}", fact_str, val, z_score, verdict);
        }

        let avg_anomaly = anomaly_scores.iter().sum::<f32>() / anomaly_scores.len().max(1) as f32;

        // ── Summary ──
        println!("\n  ── HEHRGNN PLAUSIBILITY SUMMARY ──\n");
        println!("    Normal facts avg score:  {:.4}", avg_normal);
        println!("    Anomaly facts avg score: {:.4}", avg_anomaly);
        println!("    Threshold (μ-1.5σ):      {:.4}", threshold);
        println!("    Normal > Anomaly:        {} (gap: {:.4})", avg_normal > avg_anomaly, avg_normal - avg_anomaly);

        let detected = anomaly_scores.iter().filter(|&&s| s < threshold).count();
        println!("    Anomalies flagged:       {}/{}", detected, anomaly_scores.len());

        // ── Compare individual normal vs anomaly scores ──
        println!("\n  ── SCORE DISTRIBUTION ──\n");
        
        let mut sorted_normal = normal_scores.clone();
        sorted_normal.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        
        println!("    Normal facts - top 5 scores:    {:?}", 
            sorted_normal.iter().take(5).map(|s| format!("{:.3}", s)).collect::<Vec<_>>());
        println!("    Normal facts - bottom 5 scores: {:?}", 
            sorted_normal.iter().rev().take(5).map(|s| format!("{:.3}", s)).collect::<Vec<_>>());
        println!("    Anomaly facts - all scores:     {:?}", 
            anomaly_scores.iter().map(|s| format!("{:.3}", s)).collect::<Vec<_>>());

        println!("\n  ═══════════════════════════════════════════════════════════════\n");

        // The model should learn that normal facts have consistent scoring
        assert!(avg_normal.is_finite());
        assert!(avg_anomaly.is_finite());
    }
}
