//! Stress-test: store 5000 facts, retrieve every one, verify correctness.
//!
//! Run: cargo test -p rig-rlm --lib nuggets::stress_5k -- --nocapture

#[cfg(test)]
mod tests {
    use crate::nuggets::memory::{Nugget, NuggetOpts};
    use std::time::Instant;

    #[test]
    fn store_and_retrieve_5000_facts() {
        let n = 5000;

        println!("\n{}", "═".repeat(60));
        println!("  STRESS TEST: Store & retrieve {n} facts");
        println!("{}\n", "═".repeat(60));

        let mut nugget = Nugget::new(NuggetOpts {
            name: "stress_5k".into(),
            d: 2048,
            banks: 4,
            auto_save: false,
            ..Default::default()
        });

        // ── Store phase ─────────────────────────────────────────
        let t0 = Instant::now();
        for i in 0..n {
            nugget.remember(&format!("fact_{i:05}"), &format!("answer_{i:05}"));
        }
        let store_ms = t0.elapsed().as_millis();
        println!("  Stored {n} facts in {store_ms} ms ({:.1} µs/fact)", store_ms as f64 * 1000.0 / n as f64);

        // ── Retrieve phase ──────────────────────────────────────
        let mut correct = 0;
        let mut wrong = Vec::new();
        let mut not_found = Vec::new();

        let t0 = Instant::now();
        for i in 0..n {
            let key = format!("fact_{i:05}");
            let expected = format!("answer_{i:05}");
            let result = nugget.recall(&key, "stress");

            if result.found {
                if result.answer.as_deref() == Some(expected.as_str()) {
                    correct += 1;
                } else {
                    wrong.push((i, result.answer.clone()));
                }
            } else {
                not_found.push(i);
            }
        }
        let retrieve_ms = t0.elapsed().as_millis();
        let accuracy = correct as f64 / n as f64 * 100.0;

        println!("  Retrieved {n} facts in {retrieve_ms} ms ({:.1} µs/fact)", retrieve_ms as f64 * 1000.0 / n as f64);
        println!();
        println!("  ┌──────────────────────────────┐");
        println!("  │  Results                      │");
        println!("  ├──────────────────────────────┤");
        println!("  │  Total:     {:>6}             │", n);
        println!("  │  Correct:   {:>6}             │", correct);
        println!("  │  Wrong:     {:>6}             │", wrong.len());
        println!("  │  Not found: {:>6}             │", not_found.len());
        println!("  │  Accuracy:  {:>5.1}%            │", accuracy);
        println!("  └──────────────────────────────┘");

        if !wrong.is_empty() {
            println!("\n  First 10 wrong answers:");
            for (i, ans) in wrong.iter().take(10) {
                println!("    fact_{i:05} → got {:?}, expected \"answer_{i:05}\"", ans);
            }
        }

        if !not_found.is_empty() {
            println!("\n  First 10 not-found:");
            for i in not_found.iter().take(10) {
                println!("    fact_{i:05} → not found");
            }
        }

        println!("\n{}", "═".repeat(60));

        // The nuggets memory backend uses fuzzy string matching, not HRR, for recall.
        // With exact key matches it should achieve very high accuracy.
        assert!(
            accuracy >= 98.0,
            "Expected >= 98% accuracy, got {accuracy:.1}%"
        );
    }
}
