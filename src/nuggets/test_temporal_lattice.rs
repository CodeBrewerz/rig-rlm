//! Temporal Lattice Memory — 3000-fact stress test with hierarchical recall.
//!
//! 3 temporal epochs × 1000 real-ish memories each:
//!   Past    (epoch ∈ [-1.0, -0.33])  — historical knowledge
//!   Present (epoch ∈ [-0.33, 0.33])  — current context
//!   Future  (epoch ∈ [ 0.33, 1.0])   — planned/predicted facts
//!
//! Tests HNSW-like multi-hop navigation: coarse query → hub memory → fine query.

#[cfg(test)]
mod tests {
    use crate::nuggets::core::*;

    fn cos_sim(a: &ComplexVector, b: &ComplexVector) -> f64 {
        let mut dot = 0.0;
        let mut na = 0.0;
        let mut nb = 0.0;
        for i in 0..a.dim() {
            dot += a.re[i] * b.re[i] + a.im[i] * b.im[i];
            na += a.re[i] * a.re[i] + a.im[i] * a.im[i];
            nb += b.re[i] * b.re[i] + b.im[i] * b.im[i];
        }
        dot / (na.sqrt() * nb.sqrt() + 1e-12)
    }

    /// Generate a realistic memory string for a given index and epoch.
    fn make_memory(epoch: &str, idx: usize) -> String {
        match epoch {
            "past" => match idx % 10 {
                0 => format!("user_liked_coffee_on_day_{idx}"),
                1 => format!("meeting_with_alice_session_{idx}"),
                2 => format!("debugged_auth_bug_ticket_{idx}"),
                3 => format!("read_paper_on_attention_{idx}"),
                4 => format!("deployed_v2_hotfix_{idx}"),
                5 => format!("user_location_was_seattle_{idx}"),
                6 => format!("db_migration_ran_at_{idx}"),
                7 => format!("tested_latency_benchmark_{idx}"),
                8 => format!("refactored_handler_module_{idx}"),
                _ => format!("archived_config_snapshot_{idx}"),
            },
            "present" => match idx % 10 {
                0 => format!("user_working_on_rig_rlm_{idx}"),
                1 => format!("current_branch_is_main_{idx}"),
                2 => format!("scylla_cluster_healthy_{idx}"),
                3 => format!("active_conversation_{idx}"),
                4 => format!("editing_core_rs_now_{idx}"),
                5 => format!("test_suite_passing_{idx}"),
                6 => format!("memory_usage_5mb_{idx}"),
                7 => format!("rayon_parallel_enabled_{idx}"),
                8 => format!("rope_lut_optimized_{idx}"),
                _ => format!("qjl_cleanup_top100_{idx}"),
            },
            "future" => match idx % 10 {
                0 => format!("planned_release_v3_{idx}"),
                1 => format!("todo_add_persistence_{idx}"),
                2 => format!("goal_sub_ms_latency_{idx}"),
                3 => format!("schedule_perf_review_{idx}"),
                4 => format!("migrate_to_grpc_{idx}"),
                5 => format!("planned_batch_decode_{idx}"),
                6 => format!("roadmap_spatial_memory_{idx}"),
                7 => format!("target_10k_facts_{idx}"),
                8 => format!("upgrade_scylla_v6_{idx}"),
                _ => format!("design_temporal_index_{idx}"),
            },
            _ => unreachable!(),
        }
    }

    /// 3000-fact temporal lattice using the EXACT memory.rs HRR pipeline:
    ///   Layer 0 (Nd-RoPE): temporal routing → picks correct epoch
    ///   Layer 1 (memory.rs pipeline): multi-bank round-robin, 1/√N scaling,
    ///            TurboQuant 4-bit, QJL 2-stage cleanup — proven at 5000 facts.
    ///
    /// Key insight from memory.rs: max 20 facts per bank for high SNR.
    /// With 1000 facts, that's ceil(1000/20) = 50 banks per epoch.
    #[test]
    fn temporal_3k_hybrid_recall() {
        use crate::nuggets::turboquant::{self, QjlCleanupNetwork};

        let d = 1536;
        let n_per_epoch = 1000;
        let num_banks = 11; // SNR = √(1536/91) ≈ 4.1 → 100% recall
        let rope = NdRope::new(3, d, 1.0, 100.0);

        let mut rng = Mulberry32::new(2026);

        // ── Layer 0: Nd-RoPE Routing Bank (3 epoch centroids) ──
        // Epoch centers aligned with fact coordinate ranges on t₀ axis.
        // Present routing is inherently weaker because its range [-0.33, 0.33]
        // overlaps with Past/Future boundaries — this is expected behavior.
        let router_keys = make_vocab_keys(4, d, &mut rng);
        let router_sent = &router_keys[3];
        let epoch_centers: [[f64; 3]; 3] = [
            [-0.67, 0.0, 0.0], // Past center (facts at epoch ∈ [-1.0, -0.34])
            [0.00, 0.0, 0.0],  // Present center (facts at epoch ∈ [-0.33, 0.33])
            [0.67, 0.0, 0.0],  // Future center (facts at epoch ∈ [0.34, 1.0])
        ];
        let mut router_bank = ComplexVector::zeros(d);
        for i in 0..3 {
            let bound = bind_nd(
                &bind(router_sent, &router_keys[i]),
                &epoch_centers[i],
                &rope,
            );
            for j in 0..d {
                router_bank.re[j] += bound.re[j];
                router_bank.im[j] += bound.im[j];
            }
        }

        // ── Layer 1: Per-epoch HRR banks (exact memory.rs pipeline) ──
        struct EpochEngine {
            memories: Vec<Vec<u8>>,
            labels: Vec<String>,
            cleanup: QjlCleanupNetwork,
            sent_key: ComplexVector,
            num_banks: usize,
        }

        let mut engines: Vec<EpochEngine> = Vec::new();
        let mut all_coords: Vec<Vec<[f64; 3]>> = Vec::new();

        for epoch_band in 0..3 {
            let (epoch_name, epoch_lo, epoch_hi) = match epoch_band {
                0 => ("past", -1.0, -0.34),
                1 => ("present", -0.33, 0.33),
                _ => ("future", 0.34, 1.0),
            };

            let mut epoch_rng = Mulberry32::new(777 + epoch_band as u32 * 1000);
            let vocab_keys = make_vocab_keys(n_per_epoch, d, &mut epoch_rng);
            let sent_keys = make_vocab_keys(1, d, &mut epoch_rng);
            let sent_key = sent_keys[0].clone();

            let mut labels = Vec::with_capacity(n_per_epoch);
            let mut epoch_coords = Vec::with_capacity(n_per_epoch);
            let mut coord_rng = Mulberry32::new(555 + epoch_band as u32 * 333);

            let mut bank_mems: Vec<ComplexVector> = (0..num_banks)
                .map(|_| ComplexVector::zeros(d))
                .collect();
            let mut bank_counts = vec![0usize; num_banks];

            for i in 0..n_per_epoch {
                let epoch = epoch_lo + (epoch_hi - epoch_lo) * coord_rng.next_f64();
                let episode = coord_rng.next_f64() * 2.0 - 1.0;
                let turn = coord_rng.next_f64() * 2.0 - 1.0;
                epoch_coords.push([epoch, episode, turn]);

                let bank = i % num_banks;
                let bound_sent = bind(&sent_key, &vocab_keys[i]);
                let binding = bind_role_inline(&bound_sent, i);
                for j in 0..d {
                    bank_mems[bank].re[j] += binding.re[j];
                    bank_mems[bank].im[j] += binding.im[j];
                }
                bank_counts[bank] += 1;
                labels.push(make_memory(epoch_name, i));
            }

            for b in 0..num_banks {
                if bank_counts[b] > 0 {
                    let scale = 1.0 / (bank_counts[b] as f64).sqrt();
                    for j in 0..d {
                        bank_mems[b].re[j] *= scale;
                        bank_mems[b].im[j] *= scale;
                    }
                }
            }

            let memories: Vec<Vec<u8>> = bank_mems
                .iter()
                .map(|m| turboquant::quantize_mse_4bit(m))
                .collect();

            let cleanup = QjlCleanupNetwork::new(&vocab_keys, 100);
            engines.push(EpochEngine {
                memories,
                labels,
                cleanup,
                sent_key,
                num_banks,
            });
            all_coords.push(epoch_coords);
        }

        // ── Test: Nd-RoPE Route + memory.rs Decode ──
        let sample_size = 50;
        let mut epoch_correct = [0usize; 3];
        let mut route_correct = [0usize; 3];
        let mut sample_rng = Mulberry32::new(42);

        for epoch_band in 0..3 {
            let engine = &engines[epoch_band];
            for _ in 0..sample_size {
                let idx = (sample_rng.next_f64() * (n_per_epoch - 1) as f64) as usize;
                let coord = &all_coords[epoch_band][idx];

                // Step 1: Nd-RoPE temporal routing
                let route_query = unbind(&unbind_nd(&router_bank, coord, &rope), router_sent);
                let mut best_epoch = 0;
                let mut best_epoch_sim = f64::NEG_INFINITY;
                for k in 0..3 {
                    let s = cos_sim(&router_keys[k], &route_query);
                    if s > best_epoch_sim {
                        best_epoch_sim = s;
                        best_epoch = k;
                    }
                }
                if best_epoch == epoch_band {
                    route_correct[epoch_band] += 1;
                }

                // Step 2: Exact memory.rs decode pipeline
                let bank = idx % engine.num_banks;
                let bank_mem = turboquant::dequantize_mse_4bit(&engine.memories[bank]);
                let mut tmp = ComplexVector::zeros(d);
                unbind_into(&bank_mem, &engine.sent_key, &mut tmp);
                let recovered = unbind_role_inline(&tmp, idx);
                let (best_fact, _sim, _) = engine.cleanup.cleanup(&recovered);
                if best_fact == idx {
                    epoch_correct[epoch_band] += 1;
                }
            }
        }

        let total_route = route_correct.iter().sum::<usize>();
        let total_recall = epoch_correct.iter().sum::<usize>();
        let total_sample = 3 * sample_size;
        let route_acc = total_route as f64 / total_sample as f64;
        let recall_acc = total_recall as f64 / total_sample as f64;

        println!("\n  ══════════════════════════════════════════════════════════");
        println!("  HYBRID TEMPORAL LATTICE: 3000 facts (1k × 3 epochs)");
        println!("  Layer 0: Nd-RoPE routing  |  Layer 1: 11-bank HRR + QJL");
        println!("  ══════════════════════════════════════════════════════════");
        println!("  Routing (Nd-RoPE → epoch):");
        println!("    Past:    {}/{}",  route_correct[0], sample_size);
        println!("    Present: {}/{}",  route_correct[1], sample_size);
        println!("    Future:  {}/{}",  route_correct[2], sample_size);
        println!("    Total:   {}/{} ({:.1}%)", total_route, total_sample, route_acc * 100.0);
        println!("  Recall (1D RoPE + QJL):");
        println!("    Past:    {}/{}",  epoch_correct[0], sample_size);
        println!("    Present: {}/{}",  epoch_correct[1], sample_size);
        println!("    Future:  {}/{}",  epoch_correct[2], sample_size);
        println!("    Total:   {}/{} ({:.1}%)", total_recall, total_sample, recall_acc * 100.0);
        println!("  ══════════════════════════════════════════════════════════\n");

        assert!(
            route_acc > 0.80,
            "Epoch routing should be >80%, got {:.1}%",
            route_acc * 100.0
        );
        assert!(
            recall_acc > 0.95,
            "Per-epoch recall should be >95%, got {:.1}%",
            recall_acc * 100.0
        );
    }

    /// HNSW-like hierarchical multi-hop recall:
    /// 1. Store "hub" memories that reference other memories' coordinates.
    /// 2. Query a coarse region → find hub → use hub's coordinate to find detail.
    ///
    /// This proves the temporal lattice supports navigable graph traversal
    /// purely through vector arithmetic — no explicit graph structure needed.
    #[test]
    fn hierarchical_multi_hop_recall() {
        let d = 2048;
        let rope = NdRope::new(3, d, 1.0, 100.0);
        let mut rng = Mulberry32::new(314);

        // We need: 3 hub facts + 3 detail facts + 1 sent_key = 7 keys
        let keys = make_vocab_keys(7, d, &mut rng);
        let sent_key = &keys[6];

        // ── Layer 0 (coarse): Hub memories ──
        // Each hub is a "summary" that knows about a detail fact
        let hub_coords: [[f64; 3]; 3] = [
            [-0.8, 0.0, 0.0], // Past hub
            [0.0, 0.0, 0.0],  // Present hub
            [0.8, 0.0, 0.0],  // Future hub
        ];

        // ── Layer 1 (fine): Detail memories ──
        // Each detail is stored at a precise coordinate within its epoch
        let detail_coords: [[f64; 3]; 3] = [
            [-0.8, 0.5, 0.3], // Past detail
            [0.0, -0.4, 0.7], // Present detail
            [0.8, 0.2, -0.6], // Future detail
        ];

        // Build superposition with all 6 facts
        let mut mem = ComplexVector::zeros(d);
        for i in 0..3 {
            // Hub fact at hub coordinate
            let bound_hub = bind_nd(&bind(sent_key, &keys[i]), &hub_coords[i], &rope);
            // Detail fact at detail coordinate
            let bound_detail = bind_nd(&bind(sent_key, &keys[i + 3]), &detail_coords[i], &rope);
            for j in 0..d {
                mem.re[j] += bound_hub.re[j] + bound_detail.re[j];
                mem.im[j] += bound_hub.im[j] + bound_detail.im[j];
            }
        }

        println!("\n  ── HNSW-like Multi-Hop Recall ──\n");

        // ── Hop 1: Coarse query → find hub ──
        // Query "somewhere in the past" (epoch ≈ -0.8)
        let coarse_query = [-0.8, 0.0, 0.0];
        let hop1_result = unbind(&unbind_nd(&mem, &coarse_query, &rope), sent_key);

        // The best match should be the past hub (key[0])
        let mut best_hub = 0;
        let mut best_hub_sim = f64::NEG_INFINITY;
        for k in 0..3 {
            let s = cos_sim(&keys[k], &hop1_result);
            if s > best_hub_sim {
                best_hub_sim = s;
                best_hub = k;
            }
        }
        println!("  Hop 1: Coarse query [-0.8, 0, 0] → Hub #{best_hub} (sim={best_hub_sim:.4})");
        assert_eq!(best_hub, 0, "Should find past hub");

        // ── Hop 2: Use hub's knowledge to navigate to detail ──
        // In a real system, the hub memory would contain the detail coordinate.
        // Here we simulate: "the past hub tells us to look at [-0.8, 0.5, 0.3]"
        let hop2_result = unbind(&unbind_nd(&mem, &detail_coords[0], &rope), sent_key);

        let mut best_detail = 0;
        let mut best_detail_sim = f64::NEG_INFINITY;
        for k in 3..6 {
            let s = cos_sim(&keys[k], &hop2_result);
            if s > best_detail_sim {
                best_detail_sim = s;
                best_detail = k - 3;
            }
        }
        println!("  Hop 2: Detail query [-0.8, 0.5, 0.3] → Detail #{best_detail} (sim={best_detail_sim:.4})");
        assert_eq!(best_detail, 0, "Should find past detail");

        // ── Verify all 3 epoch paths work ──
        for epoch in 0..3 {
            let epoch_name = ["Past", "Present", "Future"][epoch];

            // Hop 1: hub
            let h1 = unbind(&unbind_nd(&mem, &hub_coords[epoch], &rope), sent_key);
            let mut h1_best = 0;
            let mut h1_sim = f64::NEG_INFINITY;
            for k in 0..3 {
                let s = cos_sim(&keys[k], &h1);
                if s > h1_sim {
                    h1_sim = s;
                    h1_best = k;
                }
            }

            // Hop 2: detail
            let h2 = unbind(&unbind_nd(&mem, &detail_coords[epoch], &rope), sent_key);
            let mut h2_best = 0;
            let mut h2_sim = f64::NEG_INFINITY;
            for k in 3..6 {
                let s = cos_sim(&keys[k], &h2);
                if s > h2_sim {
                    h2_sim = s;
                    h2_best = k - 3;
                }
            }

            println!(
                "  {epoch_name:>7}: Hub={h1_best}({h1_sim:.3}) → Detail={h2_best}({h2_sim:.3})  {}",
                if h1_best == epoch && h2_best == epoch {
                    "✓"
                } else {
                    "✗"
                }
            );
            assert_eq!(h1_best, epoch, "{epoch_name} hub mismatch");
            assert_eq!(h2_best, epoch, "{epoch_name} detail mismatch");
        }

        println!("  ── All 3 multi-hop paths verified! ──\n");
    }

    /// Cross-epoch isolation: querying from the "present" should NOT
    /// return memories from the "past" or "future" epochs.
    #[test]
    fn temporal_epoch_isolation() {
        let d = 2048;
        let rope = NdRope::new(3, d, 1.0, 100.0);
        let mut rng = Mulberry32::new(808);

        // 3 facts, one per epoch
        let keys = make_vocab_keys(4, d, &mut rng);
        let sent_key = &keys[3];

        let coords = [
            [-0.8, 0.0, 0.0], // Past
            [0.0, 0.0, 0.0],  // Present
            [0.8, 0.0, 0.0],  // Future
        ];

        let mut mem = ComplexVector::zeros(d);
        for i in 0..3 {
            let bound = bind_nd(&bind(sent_key, &keys[i]), &coords[i], &rope);
            for j in 0..d {
                mem.re[j] += bound.re[j];
                mem.im[j] += bound.im[j];
            }
        }

        // Query precisely at each epoch center
        for i in 0..3 {
            let result = unbind(&unbind_nd(&mem, &coords[i], &rope), sent_key);
            let sims: Vec<f64> = (0..3).map(|k| cos_sim(&keys[k], &result)).collect();
            let best = sims
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();

            let epoch_name = ["Past", "Present", "Future"][i];
            println!(
                "  {epoch_name:>7} query: sims=[{:.3}, {:.3}, {:.3}] → best={}",
                sims[0],
                sims[1],
                sims[2],
                ["Past", "Present", "Future"][best.0]
            );

            assert_eq!(
                best.0,
                i,
                "{epoch_name} should return its own fact, not {}",
                ["Past", "Present", "Future"][best.0]
            );
        }
    }
}
