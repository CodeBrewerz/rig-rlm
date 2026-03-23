//! Benchmark: Base HRR vs Advanced enhancements.
//!
//! Measures recall accuracy, capacity limits, and latency for:
//!   - Base HRR (bind/unbind + softmax)
//!   - Base + Cleanup Network (codebook nearest-neighbor)
//!   - Base + Cleanup + Resonator (iterative sharpening)
//!   - MAP binding + Cleanup
//!   - RFF-decorrelated keys + Cleanup
//!
//! Run: cargo test -p rig-rlm --lib nuggets::bench -- --nocapture

#[cfg(test)]
#[allow(unused)]
mod tests {
    use crate::nuggets::advanced::*;
    use crate::nuggets::core::*;
    use std::time::Instant;

    /// Helper: build a superposed memory of `n` bindings using standard bind.
    fn build_memory_standard(
        keys: &[ComplexVector],
        vocab: &[ComplexVector],
        n: usize,
    ) -> ComplexVector {
        let d = keys[0].dim();
        let mut mem = ComplexVector::zeros(d);
        for i in 0..n {
            let b = bind(&keys[i], &vocab[i]);
            for dd in 0..d {
                mem.re[dd] += b.re[dd];
                mem.im[dd] += b.im[dd];
            }
        }
        let scale = 1.0 / (n as f64).sqrt();
        for dd in 0..d {
            mem.re[dd] *= scale;
            mem.im[dd] *= scale;
        }
        mem
    }

    /// Helper: build a superposed memory using MAP binding.
    fn build_memory_map(
        keys: &[ComplexVector],
        vocab: &[ComplexVector],
        n: usize,
        perm: &[usize],
    ) -> ComplexVector {
        let d = keys[0].dim();
        let mut mem = ComplexVector::zeros(d);
        for i in 0..n {
            let b = map_bind(&keys[i], &vocab[i], perm);
            for dd in 0..d {
                mem.re[dd] += b.re[dd];
                mem.im[dd] += b.im[dd];
            }
        }
        let scale = 1.0 / (n as f64).sqrt();
        for dd in 0..d {
            mem.re[dd] *= scale;
            mem.im[dd] *= scale;
        }
        mem
    }

    /// Helper: cosine similarity between two complex vectors.
    fn cosine_sim(a: &ComplexVector, b: &ComplexVector) -> f64 {
        let d = a.dim();
        let mut dot = 0.0;
        let mut na = 0.0;
        let mut nb = 0.0;
        for i in 0..d {
            dot += a.re[i] * b.re[i] + a.im[i] * b.im[i];
            na += a.re[i] * a.re[i] + a.im[i] * a.im[i];
            nb += b.re[i] * b.re[i] + b.im[i] * b.im[i];
        }
        dot / (na.sqrt() * nb.sqrt() + 1e-12)
    }

    /// Helper: measure recall accuracy for standard unbind + softmax.
    fn accuracy_base_softmax(
        mem: &ComplexVector,
        keys: &[ComplexVector],
        vocab: &[ComplexVector],
        n: usize,
    ) -> (usize, f64) {
        let d = mem.dim();
        let vocab_norm = stack_and_unit_norm(vocab);
        let mut correct = 0;
        let mut total_conf = 0.0;

        for i in 0..n {
            let rec = unbind(mem, &keys[i]);
            // Cosine sim against all vocab
            let d2 = d * 2;
            let mut rec2 = vec![0.0; d2];
            rec2[..d].copy_from_slice(&rec.re);
            rec2[d..d2].copy_from_slice(&rec.im);
            let norm: f64 = rec2.iter().map(|x| x * x).sum::<f64>().sqrt() + 1e-12;
            for x in rec2.iter_mut() {
                *x /= norm;
            }

            let mut sims = vec![0.0; vocab.len()];
            for (vi, row) in vocab_norm.iter().enumerate() {
                let mut dot = 0.0;
                for dd in 0..d2 {
                    dot += row[dd] * rec2[dd];
                }
                sims[vi] = dot;
            }

            let probs = softmax_temp(&sims, 0.9);
            let best = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            if best == i {
                correct += 1;
            }
            total_conf += probs[best];
        }
        (correct, total_conf / n as f64)
    }

    /// Helper: measure recall accuracy with cleanup network.
    fn accuracy_with_cleanup(
        mem: &ComplexVector,
        keys: &[ComplexVector],
        vocab: &[ComplexVector],
        cleanup: &CleanupNetwork,
        n: usize,
    ) -> (usize, f64) {
        let mut correct = 0;
        let mut total_sim = 0.0;
        for i in 0..n {
            let rec = unbind(mem, &keys[i]);
            let (idx, sim, _) = cleanup.cleanup(&rec);
            if idx == i {
                correct += 1;
            }
            total_sim += sim;
        }
        (correct, total_sim / n as f64)
    }

    /// Helper: measure recall accuracy with cleanup + resonator.
    fn accuracy_with_resonator(
        mem: &ComplexVector,
        keys: &[ComplexVector],
        cleanup: &CleanupNetwork,
        resonator: &Resonator,
        n: usize,
    ) -> (usize, f64, f64) {
        let mut correct = 0;
        let mut total_sim = 0.0;
        let mut total_iters = 0.0;
        for i in 0..n {
            let (idx, sim, iters) = resonator.resonate(mem, &keys[i], cleanup);
            if idx == i {
                correct += 1;
            }
            total_sim += sim;
            total_iters += iters as f64;
        }
        (correct, total_sim / n as f64, total_iters / n as f64)
    }

    // ═══════════════════════════════════════════════════════════════
    // Main benchmark test
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn benchmark_base_vs_advanced() {
        let d = 1024;
        let max_facts = 50;

        println!("\n{}", "=".repeat(70));
        println!("  HRR MEMORY BENCHMARK: Base vs Advanced");
        println!("  Dimension: {d}, Max facts: {max_facts}");
        println!("{}\n", "=".repeat(70));

        let mut rng = Mulberry32::new(42);
        let vocab = make_vocab_keys(max_facts, d, &mut rng);
        let keys = make_vocab_keys(max_facts, d, &mut rng);
        let perm = make_permutation(d, 54321);
        let inv_perm = invert_permutation(&perm);

        // RFF-decorrelated keys
        let rff = RffDecorrelator::new(d, d * 2, 1.0, 42);
        let vocab_rff = rff.decorrelate_keys(&vocab, 3, 0.005);

        let cleanup = CleanupNetwork::new(&vocab);
        let cleanup_rff = CleanupNetwork::new(&vocab_rff);
        let resonator = Resonator::new(5, 0.85);

        println!(
            "{:<30} {:>8} {:>8} {:>10} {:>10}",
            "Method", "Facts", "Correct", "Accuracy", "Avg Conf"
        );
        println!("{:-<70}", "");

        for n in &[5, 10, 20, 30, 40, 50] {
            let n = *n;
            if n > max_facts {
                break;
            }

            let mem_std = build_memory_standard(&keys, &vocab, n);
            let mem_map = build_memory_map(&keys, &vocab, n, &perm);
            let mem_rff = build_memory_standard(&keys, &vocab_rff, n);

            // 1. Base softmax
            let (c1, conf1) = accuracy_base_softmax(&mem_std, &keys, &vocab, n);
            println!(
                "{:<30} {:>8} {:>8} {:>9.1}% {:>10.4}",
                "Base (softmax)",
                n,
                c1,
                c1 as f64 / n as f64 * 100.0,
                conf1
            );

            // 2. Base + Cleanup
            let (c2, conf2) = accuracy_with_cleanup(&mem_std, &keys, &vocab, &cleanup, n);
            println!(
                "{:<30} {:>8} {:>8} {:>9.1}% {:>10.4}",
                "Base + Cleanup",
                n,
                c2,
                c2 as f64 / n as f64 * 100.0,
                conf2
            );

            // 3. Base + Cleanup + Resonator
            let (c3, conf3, avg_iters) =
                accuracy_with_resonator(&mem_std, &keys, &cleanup, &resonator, n);
            println!(
                "{:<30} {:>8} {:>8} {:>9.1}% {:>10.4}  (avg {:.1} iters)",
                "Base + Cleanup + Resonator",
                n,
                c3,
                c3 as f64 / n as f64 * 100.0,
                conf3,
                avg_iters
            );

            // 4. MAP + Cleanup
            let (c4, conf4) = {
                let mut correct = 0;
                let mut total_sim = 0.0;
                for i in 0..n {
                    let rec = map_unbind(&mem_map, &keys[i], &inv_perm);
                    let (idx, sim, _) = cleanup.cleanup(&rec);
                    if idx == i {
                        correct += 1;
                    }
                    total_sim += sim;
                }
                (correct, total_sim / n as f64)
            };
            println!(
                "{:<30} {:>8} {:>8} {:>9.1}% {:>10.4}",
                "MAP + Cleanup",
                n,
                c4,
                c4 as f64 / n as f64 * 100.0,
                conf4
            );

            // 5. RFF Keys + Cleanup
            let (c5, conf5) = accuracy_with_cleanup(&mem_rff, &keys, &vocab_rff, &cleanup_rff, n);
            println!(
                "{:<30} {:>8} {:>8} {:>9.1}% {:>10.4}",
                "RFF Keys + Cleanup",
                n,
                c5,
                c5 as f64 / n as f64 * 100.0,
                conf5
            );

            println!("{:-<70}", "");
        }

        // ── Latency benchmark ──
        println!("\n{:<30} {:>15}", "Operation", "Latency");
        println!("{:-<50}", "");

        let mem20 = build_memory_standard(&keys, &vocab, 20);

        // Base unbind + softmax
        let t0 = Instant::now();
        for _ in 0..100 {
            let _ = accuracy_base_softmax(&mem20, &keys, &vocab, 20);
        }
        let base_ns = t0.elapsed().as_nanos() / 100 / 20;
        println!("{:<30} {:>12} ns", "Base (unbind+softmax)", base_ns);

        // Cleanup
        let t0 = Instant::now();
        for _ in 0..100 {
            let _ = accuracy_with_cleanup(&mem20, &keys, &vocab, &cleanup, 20);
        }
        let cleanup_ns = t0.elapsed().as_nanos() / 100 / 20;
        println!("{:<30} {:>12} ns", "Base + Cleanup", cleanup_ns);

        // Resonator
        let t0 = Instant::now();
        for _ in 0..100 {
            let _ = accuracy_with_resonator(&mem20, &keys, &cleanup, &resonator, 20);
        }
        let res_ns = t0.elapsed().as_nanos() / 100 / 20;
        println!("{:<30} {:>12} ns", "Base + Resonator", res_ns);

        // MAP
        let mem20_map = build_memory_map(&keys, &vocab, 20, &perm);
        let t0 = Instant::now();
        for _ in 0..100 {
            for i in 0..20 {
                let rec = map_unbind(&mem20_map, &keys[i], &inv_perm);
                let _ = cleanup.cleanup(&rec);
            }
        }
        let map_ns = t0.elapsed().as_nanos() / 100 / 20;
        println!("{:<30} {:>12} ns", "MAP + Cleanup", map_ns);

        println!();

        // Verify at least one advanced method beats base at n=30
        let mem30 = build_memory_standard(&keys, &vocab, 30);
        let (base_30, _) = accuracy_base_softmax(&mem30, &keys, &vocab, 30);
        let (cleanup_30, _) = accuracy_with_cleanup(&mem30, &keys, &vocab, &cleanup, 30);
        let (res_30, _, _) = accuracy_with_resonator(&mem30, &keys, &cleanup, &resonator, 30);

        println!(
            "At n=30: base={}/30, cleanup={}/30, resonator={}/30",
            base_30, cleanup_30, res_30
        );

        assert!(
            cleanup_30 >= base_30,
            "Cleanup should be >= base at n=30: cleanup={cleanup_30} vs base={base_30}"
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // Full Nugget integration benchmark (uses actual Nugget struct)
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn benchmark_nugget_recall_scaling() {
        use crate::nuggets::memory::{Nugget, NuggetOpts};

        println!("\n{}", "=".repeat(70));
        println!("  NUGGET RECALL SCALING (real Nugget struct)");
        println!("{}\n", "=".repeat(70));

        let fact_counts = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000, 5000, 10000];

        println!(
            "{:>6} {:>10} {:>10} {:>12}",
            "Facts", "Correct", "Accuracy", "Avg Latency"
        );
        println!("{:-<45}", "");

        for &n in &fact_counts {
            let mut nugget = Nugget::new(NuggetOpts {
                name: format!("bench_{n}"),
                d: 2048,
                banks: 4,
                auto_save: false,
                ..Default::default()
            });

            // Store n unique facts
            for i in 0..n {
                nugget.remember(&format!("key_{i:03}"), &format!("value_{i:03}"));
            }

            // Recall each and measure
            let mut correct = 0;
            let t0 = Instant::now();
            for i in 0..n {
                let result = nugget.recall(&format!("key_{i:03}"), "bench");
                if result.found && result.answer.as_deref() == Some(&format!("value_{i:03}")) {
                    correct += 1;
                }
            }
            let elapsed = t0.elapsed();
            let avg_ns = elapsed.as_nanos() / n as u128;

            println!(
                "{:>6} {:>10} {:>9.1}% {:>10} ns",
                n,
                correct,
                correct as f64 / n as f64 * 100.0,
                avg_ns
            );
        }

        println!();
    }
}
