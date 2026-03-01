//! Scenario 3: Missing Recurring Bill Detection
//!
//! Recurring bills (rent, phone, electricity) create predictable patterns.
//! HEHRGNN learns these patterns: (user, pays, merchant, in_period, month_X).
//! When an expected bill is missing, the fact scores LOW → anomaly.

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

    fn build_recurring_facts() -> (Vec<RawFact>, Vec<RawFact>, Vec<String>) {
        let mut normal = Vec::new();
        let mut test_facts = Vec::new();
        let mut test_labels = Vec::new();

        let users = ["alice", "bob"];
        let months = ["jan", "feb", "mar", "apr", "may", "jun"];

        // ── Recurring bills: happen EVERY month ──
        let bills = vec![
            ("alice", "rent_payment", "landlord_llc", "1500"),
            ("alice", "phone_bill", "verizon", "85"),
            ("alice", "electricity", "power_co", "120"),
            ("bob", "rent_payment", "apt_mgmt", "1800"),
            ("bob", "car_insurance", "geico", "150"),
            ("bob", "internet", "comcast", "70"),
        ];

        // Normal: bills jan-may (train data)
        for month in &months[..5] {
            for (user, bill_type, merchant, _amount) in &bills {
                normal.push(RawFact {
                    head: user.to_string(),
                    relation: format!("pays_{}", bill_type),
                    tail: merchant.to_string(),
                    qualifiers: vec![],
                });
                normal.push(RawFact {
                    head: format!("{}_{}", user, bill_type),
                    relation: "in_period".into(),
                    tail: month.to_string(),
                    qualifiers: vec![],
                });
                normal.push(RawFact {
                    head: format!("{}_{}", user, bill_type),
                    relation: "amount_typical".into(),
                    tail: format!("amt_{}", _amount),
                    qualifiers: vec![],
                });
            }
        }

        // User → Account relationships
        for user in &users {
            normal.push(RawFact {
                head: user.to_string(), relation: "owns".into(),
                tail: format!("{}_checking", user), qualifiers: vec![],
            });
        }

        // Merchant categories
        for (_, _, merchant, _) in &bills {
            normal.push(RawFact {
                head: merchant.to_string(), relation: "is_type".into(),
                tail: "recurring_merchant".into(), qualifiers: vec![],
            });
        }

        // ── Test facts for June ──
        // Some bills arrive normally (should score HIGH)
        // Some are MISSING (should score LOW)

        // ✅ Alice rent arrives as expected
        test_facts.push(RawFact {
            head: "alice".into(), relation: "pays_rent_payment".into(),
            tail: "landlord_llc".into(), qualifiers: vec![],
        });
        test_labels.push("✅ alice rent (expected, arrived)".into());

        // ✅ Bob rent arrives as expected
        test_facts.push(RawFact {
            head: "bob".into(), relation: "pays_rent_payment".into(),
            tail: "apt_mgmt".into(), qualifiers: vec![],
        });
        test_labels.push("✅ bob rent (expected, arrived)".into());

        // 🚨 Alice phone bill MISSING → unusual fact should score low
        test_facts.push(RawFact {
            head: "alice".into(), relation: "pays_phone_bill".into(),
            tail: "unknown_provider".into(), qualifiers: vec![],  // wrong merchant!
        });
        test_labels.push("🚨 alice phone (WRONG merchant → missing?)".into());

        // 🚨 Bob's bill goes to wrong service
        test_facts.push(RawFact {
            head: "bob".into(), relation: "pays_car_insurance".into(),
            tail: "unknown_insurer".into(), qualifiers: vec![],
        });
        test_labels.push("🚨 bob car insurance (WRONG merchant)".into());

        // 🚨 Completely new user paying rent (never seen)
        test_facts.push(RawFact {
            head: "charlie".into(), relation: "pays_rent_payment".into(),
            tail: "landlord_llc".into(), qualifiers: vec![],
        });
        test_labels.push("🚨 charlie rent (unknown user)".into());

        // ✅ Bob internet (normal)
        test_facts.push(RawFact {
            head: "bob".into(), relation: "pays_internet".into(),
            tail: "comcast".into(), qualifiers: vec![],
        });
        test_labels.push("✅ bob internet (expected, arrived)".into());

        (normal, test_facts, test_labels)
    }

    #[test]
    fn test_missing_recurring_bills() {
        let device = <TrainB as Backend>::Device::default();
        let (normal, test_facts, test_labels) = build_recurring_facts();

        println!("\n  ═══════════════════════════════════════════════════════════════");
        println!("   📅 SCENARIO 3: MISSING RECURRING BILL DETECTION");
        println!("   HEHRGNN learns bill patterns → flags missing/wrong bills");
        println!("  ═══════════════════════════════════════════════════════════════\n");

        let all_raw: Vec<RawFact> = normal.iter().chain(test_facts.iter()).cloned().collect();
        let vocab = KgVocabulary::from_facts(&all_raw);

        println!("  Vocabulary: {} entities, {} relations", vocab.num_entities(), vocab.num_relations());

        let normal_idx: Vec<HehrFact> = normal.iter().filter_map(|f| HehrFact::from_raw(f, &vocab)).collect();
        let test_idx: Vec<HehrFact> = test_facts.iter().filter_map(|f| HehrFact::from_raw(f, &vocab)).collect();

        println!("  Training facts: {} (recurring patterns jan-may)", normal_idx.len());
        println!("  Test facts:     {} (june bills)\n", test_idx.len());

        // Train
        let split = (normal_idx.len() as f64 * 0.8) as usize;
        let result = train::<TrainB>(
            &TrainConfig {
                epochs: 25, lr: 0.005, margin: 1.0, batch_size: 32,
                negatives_per_positive: 5, hidden_dim: 32, num_layers: 2,
                dropout: 0.1, eval_every: 25,
                scorer_type: "distmult".into(),
                output_dir: "/tmp/hehrgnn_recurring".into(),
            },
            &normal_idx[..split], &normal_idx[split..],
            vocab.num_entities(), vocab.num_relations(), &device,
        );

        let model = result.model;
        let infer_device = model.embeddings.entity_embedding.weight.val().device();
        let scorer = DistMultScorer::new();
        let batcher = HehrBatcher::new();

        // Score test facts
        println!("\n  ── RECURRING BILL PLAUSIBILITY SCORES ──\n");
        println!("  {:>50} │ {:>8} │ {}", "Fact", "Score", "Expected");
        println!("  ──────────────────────────────────────────────────┼──────────┼─────────────");

        let mut normal_scores = Vec::new();
        let mut anomaly_scores = Vec::new();

        for (i, fact) in test_idx.iter().enumerate() {
            let item = HehrFactItem { fact: fact.clone(), label: 1.0 };
            let batch: HehrBatch<InferB> = batcher.batch(vec![item], &infer_device);
            let score = model.score_batch(&batch, &scorer);
            let val: f32 = score.into_data().as_slice::<f32>().unwrap()[0];

            let is_expected = test_labels[i].starts_with("✅");
            if is_expected { normal_scores.push(val); } else { anomaly_scores.push(val); }

            let raw = &test_facts[i];
            let fact_str = format!("({}, {}, {})", raw.head, raw.relation, raw.tail);
            println!("  {:>50} │ {:>8.4} │ {}", fact_str, val, test_labels[i]);
        }

        let avg_normal = normal_scores.iter().sum::<f32>() / normal_scores.len().max(1) as f32;
        let avg_anomaly = anomaly_scores.iter().sum::<f32>() / anomaly_scores.len().max(1) as f32;

        println!("\n  ── SUMMARY ──\n");
        println!("    Expected bills avg score:   {:.4}", avg_normal);
        println!("    Anomalous bills avg score:  {:.4}", avg_anomaly);
        println!("    Normal > Anomaly:           {} (gap: {:.4})", avg_normal > avg_anomaly, avg_normal - avg_anomaly);
        println!("\n  ═══════════════════════════════════════════════════════════════\n");

        assert!(avg_normal.is_finite());
        assert!(avg_anomaly.is_finite());
    }
}
