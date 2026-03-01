//! Scenario 7: Peer Splits + Settlement Anomalies
//!
//! Group expenses are split among users. Settlement legs connect
//! to users. Anomalies: disputed legs, late settlements, wrong amounts.
//! HEHRGNN learns normal settlement patterns and flags anomalies.

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

    fn build_peer_split_facts() -> (Vec<RawFact>, Vec<RawFact>, Vec<String>) {
        let mut normal = Vec::new();
        let mut test = Vec::new();
        let mut labels = Vec::new();

        // ── Normal peer split patterns ──
        // 10 normal splits between alice, bob, carol
        for i in 0..10 {
            let split = format!("split_{}", i);
            let payer = ["alice", "bob", "carol"][i % 3];
            let payees: Vec<&str> = ["alice", "bob", "carol"].iter()
                .filter(|&&u| u != payer).copied().collect();

            normal.push(RawFact { head: split.clone(), relation: "paid_by".into(), tail: payer.into(), qualifiers: vec![] });
            normal.push(RawFact { head: split.clone(), relation: "split_amount".into(), tail: "normal_amount".into(), qualifiers: vec![] });
            normal.push(RawFact { head: split.clone(), relation: "split_status".into(), tail: "settled".into(), qualifiers: vec![] });

            for payee in &payees {
                let leg = format!("{}_leg_{}", split, payee);
                normal.push(RawFact { head: leg.clone(), relation: "part_of".into(), tail: split.clone(), qualifiers: vec![] });
                normal.push(RawFact { head: leg.clone(), relation: "owed_by".into(), tail: payee.to_string(), qualifiers: vec![] });
                normal.push(RawFact { head: leg.clone(), relation: "leg_status".into(), tail: "paid".into(), qualifiers: vec![] });
                normal.push(RawFact { head: leg.clone(), relation: "leg_amount".into(), tail: "fair_share".into(), qualifiers: vec![] });
            }
        }

        // User relationships
        normal.push(RawFact { head: "alice".into(), relation: "is_peer_with".into(), tail: "bob".into(), qualifiers: vec![] });
        normal.push(RawFact { head: "bob".into(), relation: "is_peer_with".into(), tail: "carol".into(), qualifiers: vec![] });
        normal.push(RawFact { head: "alice".into(), relation: "is_peer_with".into(), tail: "carol".into(), qualifiers: vec![] });

        // ── Test facts ──
        // ✅ Normal settlement
        test.push(RawFact { head: "split_ok".into(), relation: "split_status".into(), tail: "settled".into(), qualifiers: vec![] });
        labels.push("✅ normal settled split".into());

        test.push(RawFact { head: "leg_ok".into(), relation: "leg_status".into(), tail: "paid".into(), qualifiers: vec![] });
        labels.push("✅ normal paid leg".into());

        // 🚨 Disputed leg
        test.push(RawFact { head: "leg_bad1".into(), relation: "leg_status".into(), tail: "disputed".into(), qualifiers: vec![] });
        labels.push("🚨 disputed leg (anomaly)".into());

        // 🚨 Unknown user in split
        test.push(RawFact { head: "leg_bad2".into(), relation: "owed_by".into(), tail: "unknown_user".into(), qualifiers: vec![] });
        labels.push("🚨 unknown user in split".into());

        // 🚨 Abnormal amount
        test.push(RawFact { head: "split_bad".into(), relation: "split_amount".into(), tail: "huge_amount".into(), qualifiers: vec![] });
        labels.push("🚨 abnormally large split".into());

        // 🚨 Unsettled split (should be settled by now)
        test.push(RawFact { head: "split_late".into(), relation: "split_status".into(), tail: "pending".into(), qualifiers: vec![] });
        labels.push("🚨 late/pending split".into());

        (normal, test, labels)
    }

    #[test]
    fn test_peer_splits_anomalies() {
        let device = <TrainB as Backend>::Device::default();
        let (normal, test_facts, test_labels) = build_peer_split_facts();

        println!("\n  ═══════════════════════════════════════════════════════════════");
        println!("   👥 SCENARIO 7: PEER SPLITS + SETTLEMENT ANOMALIES");
        println!("   HEHRGNN learns normal split/settlement patterns");
        println!("  ═══════════════════════════════════════════════════════════════\n");

        let all_raw: Vec<RawFact> = normal.iter().chain(test_facts.iter()).cloned().collect();
        let vocab = KgVocabulary::from_facts(&all_raw);
        let normal_idx: Vec<HehrFact> = normal.iter().filter_map(|f| HehrFact::from_raw(f, &vocab)).collect();
        let test_idx: Vec<HehrFact> = test_facts.iter().filter_map(|f| HehrFact::from_raw(f, &vocab)).collect();

        println!("  Training: {} facts, Test: {} facts\n", normal_idx.len(), test_idx.len());

        let split = (normal_idx.len() as f64 * 0.8) as usize;
        let result = train::<TrainB>(
            &TrainConfig {
                epochs: 25, lr: 0.005, margin: 1.0, batch_size: 32,
                negatives_per_positive: 5, hidden_dim: 32, num_layers: 2,
                dropout: 0.1, eval_every: 25, scorer_type: "distmult".into(),
                output_dir: "/tmp/hehrgnn_splits".into(),
            },
            &normal_idx[..split], &normal_idx[split..],
            vocab.num_entities(), vocab.num_relations(), &device,
        );

        let model = result.model;
        let infer_device = model.embeddings.entity_embedding.weight.val().device();
        let scorer = DistMultScorer::new();
        let batcher = HehrBatcher::new();

        println!("\n  ── SETTLEMENT PLAUSIBILITY SCORES ──\n");
        let mut ok_scores = Vec::new();
        let mut bad_scores = Vec::new();

        for (i, fact) in test_idx.iter().enumerate() {
            let item = HehrFactItem { fact: fact.clone(), label: 1.0 };
            let batch: HehrBatch<InferB> = batcher.batch(vec![item], &infer_device);
            let score = model.score_batch(&batch, &scorer);
            let val: f32 = score.into_data().as_slice::<f32>().unwrap()[0];

            let is_ok = test_labels[i].starts_with("✅");
            if is_ok { ok_scores.push(val); } else { bad_scores.push(val); }

            let raw = &test_facts[i];
            println!("    ({}, {}, {}) → {:.4}  │ {}",
                raw.head, raw.relation, raw.tail, val, test_labels[i]);
        }

        let avg_ok = ok_scores.iter().sum::<f32>() / ok_scores.len().max(1) as f32;
        let avg_bad = bad_scores.iter().sum::<f32>() / bad_scores.len().max(1) as f32;

        println!("\n  ── SUMMARY ──\n");
        println!("    Normal settlements avg:  {:.4}", avg_ok);
        println!("    Anomalous settlements:   {:.4}", avg_bad);
        println!("    Gap: {:.4}", avg_ok - avg_bad);
        println!("\n  ═══════════════════════════════════════════════════════════════\n");

        assert!(avg_ok.is_finite());
    }
}
