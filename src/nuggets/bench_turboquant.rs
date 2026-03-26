//! Benchmark: TurboQuant 4-bit vs raw f64 + QJL Cleanup vs f64 Cleanup.
//!
//! Measures:
//!   1. Memory footprint (bytes per bank)
//!   2. Quantize / dequantize latency
//!   3. MSE distortion
//!   4. Recall accuracy: f64 Cleanup vs QJL Cleanup
//!   5. Cleanup latency: f64 dot-product vs QJL Hamming
//!   6. Disk serialization savings
//!   7. Full Nugget E2E
//!
//! Run: cargo test -p rig-rlm --lib nuggets::bench_turboquant -- --nocapture

#[cfg(test)]
mod tests {
    use crate::nuggets::core::*;
    use crate::nuggets::turboquant::*;
    use crate::nuggets::advanced::*;
    use std::time::Instant;

    // ─── Helpers ────────────────────────────────────────────────

    fn build_memory(
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

    fn mse(a: &ComplexVector, b: &ComplexVector) -> f64 {
        let d = a.dim();
        let mut sum = 0.0;
        for i in 0..d {
            let dr = a.re[i] - b.re[i];
            let di = a.im[i] - b.im[i];
            sum += dr * dr + di * di;
        }
        sum / (2.0 * d as f64)
    }

    fn snr_db(original: &ComplexVector, reconstructed: &ComplexVector) -> f64 {
        let d = original.dim();
        let mut signal_power = 0.0;
        let mut noise_power = 0.0;
        for i in 0..d {
            signal_power += original.re[i] * original.re[i] + original.im[i] * original.im[i];
            let dr = original.re[i] - reconstructed.re[i];
            let di = original.im[i] - reconstructed.im[i];
            noise_power += dr * dr + di * di;
        }
        10.0 * (signal_power / (noise_power + 1e-30)).log10()
    }

    fn recall_accuracy_f64_cleanup(
        mem: &ComplexVector,
        keys: &[ComplexVector],
        cleanup: &CleanupNetwork,
        n: usize,
    ) -> usize {
        let mut correct = 0;
        for i in 0..n {
            let rec = unbind(mem, &keys[i]);
            let (idx, _, _) = cleanup.cleanup(&rec);
            if idx == i { correct += 1; }
        }
        correct
    }

    fn recall_accuracy_qjl_cleanup(
        mem: &ComplexVector,
        keys: &[ComplexVector],
        qjl: &QjlCleanupNetwork,
        n: usize,
    ) -> usize {
        let mut correct = 0;
        for i in 0..n {
            let rec = unbind(mem, &keys[i]);
            let (idx, _, _) = qjl.cleanup(&rec);
            if idx == i { correct += 1; }
        }
        correct
    }

    // ─── Main benchmark ────────────────────────────────────────

    #[test]
    fn benchmark_turboquant_vs_f64() {
        let dims = [256, 512, 1024, 2048];
        let fact_counts = [5, 10, 20, 30, 40, 50];

        println!("\n{}", "═".repeat(80));
        println!("  TURBOQUANT FULL BENCHMARK: MSE + QJL Cleanup");
        println!("{}\n", "═".repeat(80));

        // ── §1: Memory footprint ────────────────────────────────
        println!("┌─────────────────────────────────────────────────────┐");
        println!("│  §1  MEMORY FOOTPRINT (per bank, single vector)    │");
        println!("└─────────────────────────────────────────────────────┘\n");
        println!("  {:>6}  {:>12}  {:>12}  {:>10}", "Dim", "f64 (bytes)", "4-bit (bytes)", "Ratio");
        println!("  {:-<50}", "");
        for &d in &dims {
            let f64_bytes = 2 * d * 8;
            let q4_bytes = d;
            println!("  {:>6}  {:>12}  {:>12}  {:>9.1}x", d, f64_bytes, q4_bytes, f64_bytes as f64 / q4_bytes as f64);
        }

        // ── §2: Quant/dequant latency ───────────────────────────
        println!("\n┌─────────────────────────────────────────────────────┐");
        println!("│  §2  QUANTIZE / DEQUANTIZE LATENCY                 │");
        println!("└─────────────────────────────────────────────────────┘\n");
        println!("  {:>6}  {:>14}  {:>14}  {:>14}", "Dim", "Quant (ns)", "Dequant (ns)", "Round-trip");
        println!("  {:-<60}", "");
        for &d in &dims {
            let mut rng = Mulberry32::new(42);
            let k = make_vocab_keys(20, d, &mut rng);
            let v = make_vocab_keys(20, d, &mut rng);
            let mem = build_memory(&k, &v, 20);
            for _ in 0..50 { let q = quantize_mse_4bit(&mem); let _ = dequantize_mse_4bit(&q); }
            let iters = 1000;
            let t0 = Instant::now();
            for _ in 0..iters { let _ = quantize_mse_4bit(&mem); }
            let quant_ns = t0.elapsed().as_nanos() / iters;
            let q = quantize_mse_4bit(&mem);
            let t0 = Instant::now();
            for _ in 0..iters { let _ = dequantize_mse_4bit(&q); }
            let dequant_ns = t0.elapsed().as_nanos() / iters;
            println!("  {:>6}  {:>12} ns  {:>12} ns  {:>12} ns", d, quant_ns, dequant_ns, quant_ns + dequant_ns);
        }

        // ── §3: Distortion ──────────────────────────────────────
        println!("\n┌─────────────────────────────────────────────────────┐");
        println!("│  §3  QUANTIZATION DISTORTION (MSE & SNR)           │");
        println!("└─────────────────────────────────────────────────────┘\n");
        println!("  {:>6}  {:>6}  {:>14}  {:>10}", "Dim", "Facts", "MSE", "SNR (dB)");
        println!("  {:-<45}", "");
        for &d in &[512, 1024, 2048] {
            let mut rng = Mulberry32::new(42);
            let k = make_vocab_keys(50, d, &mut rng);
            let v = make_vocab_keys(50, d, &mut rng);
            for &n in &[5, 20, 50] {
                let mem = build_memory(&k, &v, n);
                let q = quantize_mse_4bit(&mem);
                let recon = dequantize_mse_4bit(&q);
                println!("  {:>6}  {:>6}  {:>14.8}  {:>9.2} dB", d, n, mse(&mem, &recon), snr_db(&mem, &recon));
            }
        }

        // ── §4: Recall accuracy comparison ──────────────────────
        println!("\n┌──────────────────────────────────────────────────────────────────┐");
        println!("│  §4  RECALL ACCURACY: f64 Cleanup vs QJL Cleanup (d=1024)       │");
        println!("└──────────────────────────────────────────────────────────────────┘\n");

        let d = 1024;
        let mut rng = Mulberry32::new(42);
        let keys_bench = make_vocab_keys(50, d, &mut rng);
        let vocab_bench = make_vocab_keys(50, d, &mut rng);
        let cleanup_f64 = CleanupNetwork::new(&vocab_bench);
        let cleanup_qjl = QjlCleanupNetwork::new(&vocab_bench, 5);

        println!("  {:>6}  {:>14}  {:>14}  {:>10}", "Facts", "f64 Cleanup", "QJL Cleanup", "Δ");
        println!("  {:-<55}", "");

        for &n in &fact_counts {
            let mem = build_memory(&keys_bench, &vocab_bench, n);
            let c_f64 = recall_accuracy_f64_cleanup(&mem, &keys_bench, &cleanup_f64, n);
            let c_qjl = recall_accuracy_qjl_cleanup(&mem, &keys_bench, &cleanup_qjl, n);
            let acc_f64 = c_f64 as f64 / n as f64 * 100.0;
            let acc_qjl = c_qjl as f64 / n as f64 * 100.0;
            println!("  {:>6}  {:>12.1}%  {:>12.1}%  {:>+9.1}%", n, acc_f64, acc_qjl, acc_qjl - acc_f64);
        }

        // ── §5: Cleanup latency head-to-head ────────────────────
        println!("\n┌──────────────────────────────────────────────────────────────────┐");
        println!("│  §5  CLEANUP LATENCY: f64 dot-product vs QJL Hamming (per fact) │");
        println!("└──────────────────────────────────────────────────────────────────┘\n");

        println!("  {:>6}  {:>6}  {:>16}  {:>16}  {:>10}", "Dim", "Vocab", "f64 (ns/fact)", "QJL (ns/fact)", "Speedup");
        println!("  {:-<70}", "");

        for &d in &[256, 512, 1024, 2048] {
            let mut rng = Mulberry32::new(42);
            let vocab_sz = 50.min(d); // reasonable vocab size
            let k = make_vocab_keys(vocab_sz, d, &mut rng);
            let v = make_vocab_keys(vocab_sz, d, &mut rng);
            let cleanup_f = CleanupNetwork::new(&v);
            let cleanup_q = QjlCleanupNetwork::new(&v, 5);

            let n = 20.min(vocab_sz);
            let mem = build_memory(&k, &v, n);

            // Warmup
            for i in 0..n {
                let rec = unbind(&mem, &k[i]);
                let _ = cleanup_f.cleanup(&rec);
                let _ = cleanup_q.cleanup(&rec);
            }

            // f64 cleanup
            let iters = 200;
            let t0 = Instant::now();
            for _ in 0..iters {
                for i in 0..n {
                    let rec = unbind(&mem, &k[i]);
                    let _ = cleanup_f.cleanup(&rec);
                }
            }
            let f64_ns = t0.elapsed().as_nanos() / (iters * n) as u128;

            // QJL cleanup
            let t0 = Instant::now();
            for _ in 0..iters {
                for i in 0..n {
                    let rec = unbind(&mem, &k[i]);
                    let _ = cleanup_q.cleanup(&rec);
                }
            }
            let qjl_ns = t0.elapsed().as_nanos() / (iters * n) as u128;

            let speedup = if qjl_ns > 0 {
                format!("{:.1}x", f64_ns as f64 / qjl_ns as f64)
            } else {
                "∞".to_string()
            };

            println!("  {:>6}  {:>6}  {:>14} ns  {:>14} ns  {:>10}", d, vocab_sz, f64_ns, qjl_ns, speedup);
        }

        // ── §6: Disk footprint ──────────────────────────────────
        println!("\n┌─────────────────────────────────────────────────────┐");
        println!("│  §6  DISK FOOTPRINT (.hrr.bin)                     │");
        println!("└─────────────────────────────────────────────────────┘\n");
        println!("  {:>6}  {:>6}  {:>14}  {:>14}  {:>10}", "Dim", "Banks", "HRR2 (bytes)", "HRR3 (bytes)", "Savings");
        println!("  {:-<60}", "");
        for &d in &dims {
            for &banks in &[1, 4] {
                let hrr2 = 20 + banks * 2 * d * 8;
                let hrr3 = 20 + banks * d;
                println!("  {:>6}  {:>6}  {:>14}  {:>14}  {:>8.1}%", d, banks, hrr2, hrr3, (1.0 - hrr3 as f64 / hrr2 as f64) * 100.0);
            }
        }

        // ── §7: Full Nugget E2E ─────────────────────────────────
        println!("\n┌─────────────────────────────────────────────────────┐");
        println!("│  §7  FULL NUGGET E2E (TurboQuant integrated)       │");
        println!("└─────────────────────────────────────────────────────┘\n");
        {
            use crate::nuggets::memory::{Nugget, NuggetOpts};
            let fact_counts_e2e = [10, 50, 100, 500, 1000];
            println!("  {:>6}  {:>8}  {:>10}  {:>14}", "Facts", "Correct", "Accuracy", "Avg Latency");
            println!("  {:-<45}", "");
            for &n in &fact_counts_e2e {
                let mut nugget = Nugget::new(NuggetOpts {
                    name: format!("tq_bench_{n}"),
                    d: 1024,
                    banks: 4,
                    auto_save: false,
                    ..Default::default()
                });
                for i in 0..n { nugget.remember(&format!("tqk_{i:04}"), &format!("tqv_{i:04}")); }
                let mut correct = 0;
                let t0 = Instant::now();
                for i in 0..n {
                    let result = nugget.recall(&format!("tqk_{i:04}"), "bench");
                    if result.found && result.answer.as_deref() == Some(&format!("tqv_{i:04}")) { correct += 1; }
                }
                let avg_ns = t0.elapsed().as_nanos() / n as u128;
                println!("  {:>6}  {:>8}  {:>8.1}%  {:>12} ns", n, correct, correct as f64 / n as f64 * 100.0, avg_ns);
            }
        }

        println!("\n{}", "═".repeat(80));
        println!("  BENCHMARK COMPLETE");
        println!("{}\n", "═".repeat(80));
    }
}
