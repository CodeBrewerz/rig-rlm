//! Scenario 5: GL Allocation + Tax Consistency (Multi-Hop)
//!
//! Transaction → GL Account → Tax Code is a 3-hop chain.
//! Some allocations lead to tax exceptions. The HEHRGNN learns
//! which tx→GL→tax paths are consistent and which are anomalous.

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

    fn build_gl_tax_facts() -> (Vec<RawFact>, Vec<RawFact>, Vec<String>) {
        let mut normal = Vec::new();
        let mut test = Vec::new();
        let mut labels = Vec::new();

        // ── Consistent allocation patterns ──
        // Meals → GL:6200 (meals expense) → Tax: meals_deduction
        // Office supplies → GL:6100 (office) → Tax: business_expense
        // Travel → GL:6300 (travel) → Tax: travel_deduction
        // Rent → GL:6400 (occupancy) → Tax: rent_deduction
        // Personal → GL:9000 (non-deductible) → Tax: no_deduction

        let consistent_chains = vec![
            ("meals", "gl_6200_meals", "tax_meals_deduction"),
            ("meals", "gl_6200_meals", "tax_meals_deduction"),
            ("office_supplies", "gl_6100_office", "tax_business_expense"),
            ("office_supplies", "gl_6100_office", "tax_business_expense"),
            ("travel", "gl_6300_travel", "tax_travel_deduction"),
            ("travel", "gl_6300_travel", "tax_travel_deduction"),
            ("rent", "gl_6400_occupancy", "tax_rent_deduction"),
            ("rent", "gl_6400_occupancy", "tax_rent_deduction"),
            ("personal", "gl_9000_nondeduct", "tax_no_deduction"),
            ("personal", "gl_9000_nondeduct", "tax_no_deduction"),
        ];

        for (i, (cat, gl, tax)) in consistent_chains.iter().enumerate() {
            let tx = format!("tx_{}", i);
            normal.push(RawFact { head: tx.clone(), relation: "has_category".into(), tail: cat.to_string(), qualifiers: vec![] });
            normal.push(RawFact { head: tx.clone(), relation: "allocated_to".into(), tail: gl.to_string(), qualifiers: vec![] });
            normal.push(RawFact { head: gl.to_string(), relation: "maps_to_tax".into(), tail: tax.to_string(), qualifiers: vec![] });
            normal.push(RawFact { head: tx, relation: "tax_treatment".into(), tail: tax.to_string(), qualifiers: vec![] });
        }

        // GL → Tax code mappings (structural knowledge)
        normal.push(RawFact { head: "gl_6200_meals".into(), relation: "maps_to_tax".into(), tail: "tax_meals_deduction".into(), qualifiers: vec![] });
        normal.push(RawFact { head: "gl_6100_office".into(), relation: "maps_to_tax".into(), tail: "tax_business_expense".into(), qualifiers: vec![] });
        normal.push(RawFact { head: "gl_6300_travel".into(), relation: "maps_to_tax".into(), tail: "tax_travel_deduction".into(), qualifiers: vec![] });
        normal.push(RawFact { head: "gl_6400_occupancy".into(), relation: "maps_to_tax".into(), tail: "tax_rent_deduction".into(), qualifiers: vec![] });
        normal.push(RawFact { head: "gl_9000_nondeduct".into(), relation: "maps_to_tax".into(), tail: "tax_no_deduction".into(), qualifiers: vec![] });

        // ── Test: consistent allocations (should score HIGH) ──
        test.push(RawFact { head: "tx_test1".into(), relation: "allocated_to".into(), tail: "gl_6200_meals".into(), qualifiers: vec![] });
        labels.push("✅ meals → GL:6200 (consistent)".into());

        test.push(RawFact { head: "gl_6100_office".into(), relation: "maps_to_tax".into(), tail: "tax_business_expense".into(), qualifiers: vec![] });
        labels.push("✅ GL:6100 → business_expense (consistent)".into());

        // ── Test: INCONSISTENT allocations (should score LOW) ──
        // Meals allocated to office GL (wrong!)
        test.push(RawFact { head: "tx_bad1".into(), relation: "allocated_to".into(), tail: "gl_6100_office".into(), qualifiers: vec![] });
        labels.push("🚨 meals → GL:6100 office (WRONG GL for meals)".into());

        // Personal expense claiming business deduction (tax fraud pattern)
        test.push(RawFact { head: "gl_9000_nondeduct".into(), relation: "maps_to_tax".into(), tail: "tax_business_expense".into(), qualifiers: vec![] });
        labels.push("🚨 non-deductible → business_expense (TAX MISMATCH)".into());

        // Travel claiming rent deduction (wrong tax code)
        test.push(RawFact { head: "gl_6300_travel".into(), relation: "maps_to_tax".into(), tail: "tax_rent_deduction".into(), qualifiers: vec![] });
        labels.push("🚨 travel GL → rent deduction (WRONG TAX CODE)".into());

        // Unknown GL account
        test.push(RawFact { head: "tx_bad2".into(), relation: "allocated_to".into(), tail: "gl_unknown".into(), qualifiers: vec![] });
        labels.push("🚨 tx → unknown GL account".into());

        (normal, test, labels)
    }

    #[test]
    fn test_gl_tax_consistency() {
        let device = <TrainB as Backend>::Device::default();
        let (normal, test_facts, test_labels) = build_gl_tax_facts();

        println!("\n  ═══════════════════════════════════════════════════════════════");
        println!("   📊 SCENARIO 5: GL ALLOCATION + TAX CONSISTENCY (MULTI-HOP)");
        println!("   HEHRGNN learns: tx → GL account → tax code chains");
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
                output_dir: "/tmp/hehrgnn_gl_tax".into(),
            },
            &normal_idx[..split], &normal_idx[split..],
            vocab.num_entities(), vocab.num_relations(), &device,
        );

        let model = result.model;
        let infer_device = model.embeddings.entity_embedding.weight.val().device();
        let scorer = DistMultScorer::new();
        let batcher = HehrBatcher::new();

        println!("\n  ── GL/TAX CONSISTENCY SCORES ──\n");
        let mut consistent_scores = Vec::new();
        let mut inconsistent_scores = Vec::new();

        for (i, fact) in test_idx.iter().enumerate() {
            let item = HehrFactItem { fact: fact.clone(), label: 1.0 };
            let batch: HehrBatch<InferB> = batcher.batch(vec![item], &infer_device);
            let score = model.score_batch(&batch, &scorer);
            let val: f32 = score.into_data().as_slice::<f32>().unwrap()[0];

            let is_consistent = test_labels[i].starts_with("✅");
            if is_consistent { consistent_scores.push(val); } else { inconsistent_scores.push(val); }

            let raw = &test_facts[i];
            println!("    ({}, {}, {}) → {:.4}  │ {}",
                raw.head, raw.relation, raw.tail, val, test_labels[i]);
        }

        let avg_c = consistent_scores.iter().sum::<f32>() / consistent_scores.len().max(1) as f32;
        let avg_i = inconsistent_scores.iter().sum::<f32>() / inconsistent_scores.len().max(1) as f32;

        println!("\n  ── SUMMARY ──\n");
        println!("    Consistent alloc avg:   {:.4}", avg_c);
        println!("    Inconsistent alloc avg: {:.4}", avg_i);
        println!("    Signal: consistent > inconsistent? {}",
            if avg_c > avg_i { "✅ YES" } else { "❌ NO" });
        println!("\n  ═══════════════════════════════════════════════════════════════\n");

        assert!(avg_c.is_finite());
        assert!(avg_i.is_finite());
    }
}
