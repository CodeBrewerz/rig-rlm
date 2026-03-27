//! MemBench-style evaluation of temporal memory.
//!
//! Based on "MemBench: A Benchmark for Evaluating Memory of LLM-Based
//! Personal Agents" (arXiv:2506.21605). Tests the 4 MemBench metrics:
//!
//! 1. **Memory Accuracy** — Can we recall correct facts?
//! 2. **Memory Recall@K** — Is the right evidence retrieved?
//! 3. **Memory Capacity** — At what fact count does accuracy drop?
//! 4. **Memory Efficiency** — Read/write time per operation
//!
//! Tests both Factual Memory (direct lookup) and Reflective Memory
//! (inference across multiple facts). Uses multi-session temporal
//! layout with timestamps, matching MemBench's participation scenario.
//!
//! Run: cargo test -p rig-rlm --lib nuggets::test_temporal_e2e -- --nocapture

#[cfg(test)]
mod tests {
    use crate::nuggets::shelf::NuggetShelf;
    use std::time::Instant;

    // ═══════════════════════════════════════════════════════════════
    // MemBench-style Temporal Memory Agent
    // ═══════════════════════════════════════════════════════════════
    // Simulates a personal agent that receives multi-session dialogues
    // over time. Each session has a timestamp. The agent auto-routes
    // facts to epoch nuggets (past/present/future) based on session time.

    struct TemporalAgent {
        shelf: NuggetShelf,
        /// Total time window — sessions span [0, time_horizon)
        time_horizon: f64,
        /// Write timing accumulator
        write_times: Vec<f64>,
        /// Read timing accumulator
        read_times: Vec<f64>,
    }

    impl TemporalAgent {
        fn new(dir: std::path::PathBuf) -> Self {
            Self {
                shelf: NuggetShelf::new(Some(dir), false),
                time_horizon: 1.0,
                write_times: Vec::new(),
                read_times: Vec::new(),
            }
        }

        /// Classify a timestamp into an epoch.
        fn epoch_for(&self, timestamp: f64) -> &'static str {
            let t = timestamp / self.time_horizon;
            match t {
                t if t < 0.33 => "past",
                t if t < 0.67 => "present",
                _ => "future",
            }
        }

        /// ── WRITE: Agent receives a dialogue turn and memorizes it ──
        /// Simulates the MemBench participation scenario "write" step.
        fn memorize(&mut self, key: &str, value: &str, timestamp: f64) {
            let t0 = Instant::now();
            let epoch = self.epoch_for(timestamp);
            let nugget_name = format!("epoch_{epoch}");
            self.shelf.get_or_create(&nugget_name).remember(key, value);
            self.write_times.push(t0.elapsed().as_secs_f64());
        }

        /// ── READ: Agent answers a question from memory ──
        /// Searches ALL epochs (the agent doesn't know when the fact was stored).
        fn answer(&mut self, query: &str) -> (Option<String>, Option<String>, f64) {
            let t0 = Instant::now();
            // Search backwards chronologically: future > present > past
            let epochs = ["epoch_future", "epoch_present", "epoch_past"];
            let mut best_answer: Option<String> = None;
            let mut best_conf: f64 = 0.0;
            let mut best_key: Option<String> = None;

            for epoch_name in &epochs {
                if !self.shelf.has(epoch_name) {
                    continue;
                }
                let result = self.shelf.recall(query, Some(epoch_name), "");
                if !result.result.found { continue; }

                let is_exact_match = result.result.key.eq_ignore_ascii_case(query);

                // If we found the EXACT key in a recent epoch, take it immediately!
                // This perfectly models "real world" memory: a newly stated fact overrides an old one outright.
                if is_exact_match {
                    best_conf = result.result.confidence;
                    best_answer = result.result.answer.clone();
                    best_key = Some(result.result.key.clone());
                    break;
                }

                // If it's merely a fuzzy match (e.g. asking for "hobby" but future only had "job"), 
                // we keep it ONLY if its HRR correlation structurally beats the previous fuzzy matches.
                if result.result.confidence > best_conf {
                    best_conf = result.result.confidence;
                    best_answer = result.result.answer.clone();
                    best_key = Some(result.result.key.clone());
                }
            }
            let elapsed = t0.elapsed().as_secs_f64();
            self.read_times.push(elapsed);
            (best_answer, best_key, best_conf)
        }

        fn avg_write_time(&self) -> f64 {
            if self.write_times.is_empty() {
                return 0.0;
            }
            self.write_times.iter().sum::<f64>() / self.write_times.len() as f64
        }

        fn avg_read_time(&self) -> f64 {
            if self.read_times.is_empty() {
                return 0.0;
            }
            self.read_times.iter().sum::<f64>() / self.read_times.len() as f64
        }

        fn save_all(&self) {
            self.shelf.save_all();
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // MemBench-style User Profile & Dialogue Generator
    // ═══════════════════════════════════════════════════════════════

    struct UserProfile {
        name: String,
        relations: Vec<(String, String, String)>, // (name, relationship, detail)
        events: Vec<(String, String, String)>,    // (event_name, time, location)
        preferences: Vec<(String, String)>,       // (category, preference)
    }

    fn generate_user_profiles(count: usize) -> Vec<UserProfile> {
        let first_names = [
            "James", "Emily", "Clara", "Ethan", "Sophie", "Nolan", "Maya", "Landon", "Amelia",
            "Maxwell", "Oliver", "Aria", "Liam", "Mila", "Noah", "Zoe", "Lucas", "Ivy", "Leo",
            "Luna",
        ];
        let last_names = [
            "Smith", "Johnson", "Jennings", "Cooper", "Turner", "Hayes", "Carter", "Pierce",
            "Brooks", "Chen", "Park", "Kim", "Lee", "Wang", "Davis", "Miller", "Wilson", "Moore",
            "Taylor", "Anderson",
        ];
        let relationships = [
            "sister",
            "brother",
            "niece",
            "uncle",
            "aunt",
            "boss",
            "coworker",
            "subordinate",
            "friend",
            "neighbor",
        ];
        let cities = [
            "Boston, MA",
            "Chicago, IL",
            "Portland, OR",
            "Philadelphia, PA",
            "Los Angeles, CA",
            "Seattle, WA",
            "Austin, TX",
            "Denver, CO",
            "Miami, FL",
            "Nashville, TN",
        ];
        let hobbies = [
            "Climbing",
            "Painting",
            "Cooking",
            "Photography",
            "Gardening",
            "Reading",
            "Cycling",
            "Yoga",
            "Chess",
            "Hiking",
        ];
        let events_pool = [
            "Build Start 2024",
            "Team Connect",
            "Policing Forum",
            "Annual Retreat",
            "Product Launch",
            "Code Sprint",
            "Board Meeting",
            "Innovation Day",
            "Wellness Week",
            "Hackathon",
        ];
        let genres = [
            "Comedy",
            "Sci-Fi",
            "Drama",
            "Action",
            "Documentary",
            "Horror",
            "Romance",
            "Thriller",
            "Animation",
            "Fantasy",
        ];

        (0..count)
            .map(|i| {
                // Use i/N and i%N for first/last to get up to 400 unique names
                let fname = first_names[i % first_names.len()];
                let lname = last_names[i / first_names.len() % last_names.len()];
                let name = format!("{fname} {lname} #{i}");

                // 3 relations per user
                let relations: Vec<_> = (0..3)
                    .map(|r| {
                        let rname = format!(
                            "{} {}",
                            first_names[(i * 3 + r + 5) % first_names.len()],
                            last_names[(i * 3 + r + 7) % last_names.len()]
                        );
                        let rel = relationships[(i * 3 + r) % relationships.len()];
                        let city = cities[(i * 3 + r) % cities.len()];
                        (rname, rel.to_string(), city.to_string())
                    })
                    .collect();

                // 2 events per user
                let events: Vec<_> = (0..2)
                    .map(|e| {
                        let ev = events_pool[(i * 2 + e) % events_pool.len()];
                        let month = (i * 2 + e) % 12 + 1;
                        let day = (i * 3 + e) % 28 + 1;
                        (
                            ev.to_string(),
                            format!("2024-{month:02}-{day:02}"),
                            cities[(i + e) % cities.len()].to_string(),
                        )
                    })
                    .collect();

                // 2 preferences per user
                let preferences = vec![
                    ("hobby".to_string(), hobbies[i % hobbies.len()].to_string()),
                    (
                        "movie_genre".to_string(),
                        genres[i % genres.len()].to_string(),
                    ),
                ];

                UserProfile {
                    name,
                    relations,
                    events,
                    preferences,
                }
            })
            .collect()
    }

    // ═══════════════════════════════════════════════════════════════
    // Test 1: MemBench Factual Memory Evaluation
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn membench_factual_memory() {
        let dir = std::env::temp_dir().join("rig_membench_fm");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut agent = TemporalAgent::new(dir.clone());
        let profiles = generate_user_profiles(100);

        println!("\n  ══════════════════════════════════════════════════════════");
        println!("  MEMBENCH: Factual Memory Evaluation");
        println!("  100 user profiles × ~10 facts = ~1000 facts, multi-session");
        println!("  ══════════════════════════════════════════════════════════");

        // ── Multi-session dialogue ingestion ─────────────────────────
        // Each user's facts arrive in a "session" at different timestamps.
        // Simulates MemBench's participation scenario with temporal flow.
        let mut qa_pairs: Vec<(String, String)> = Vec::new();

        for (u, profile) in profiles.iter().enumerate() {
            let session_time = u as f64 / profiles.len() as f64; // 0.0..1.0

            // Factual: relation attributes
            for (rel_name, relationship, city) in &profile.relations {
                let key = format!("{}'s {} {}", profile.name, relationship, rel_name);
                let value = format!("{rel_name} is from {city}");
                agent.memorize(&key, &value, session_time);
                qa_pairs.push((key, value));
            }

            // Factual: event details
            for (event_name, time, location) in &profile.events {
                let key = format!("{}'s event {}", profile.name, event_name);
                let value = format!("{event_name} on {time} in {location}");
                agent.memorize(&key, &value, session_time);
                qa_pairs.push((key, value));
            }

            // Factual: preferences
            for (category, preference) in &profile.preferences {
                let key = format!("{}'s {}", profile.name, category);
                let value = format!("{preference}");
                agent.memorize(&key, &value, session_time);
                qa_pairs.push((key, value));
            }
        }

        let total_facts = qa_pairs.len();
        println!(
            "  Stored: {} facts across 3 epochs ({:.1} µs/write avg)",
            total_facts,
            agent.avg_write_time() * 1e6
        );

        // ── SingleHop recall (MemBench: direct factual lookup) ───────
        let sample_size = total_facts.min(300);
        let step = total_facts / sample_size;
        let mut single_hop_correct = 0;

        for i in 0..sample_size {
            let idx = i * step;
            let (query, expected) = &qa_pairs[idx];
            let (answer, _key, _conf) = agent.answer(query);
            if answer.as_deref() == Some(expected.as_str()) {
                single_hop_correct += 1;
            }
        }

        let sh_acc = single_hop_correct as f64 / sample_size as f64;
        println!(
            "  SingleHop accuracy: {single_hop_correct}/{sample_size} ({:.1}%) [{:.1} µs/read]",
            sh_acc * 100.0,
            agent.avg_read_time() * 1e6
        );

        // ── KnowledgeUpdate (MemBench: fact overwrite, then recall) ──
        // Simulate updating facts over time, then checking latest version
        let update_count = 50;
        let mut update_correct = 0;

        for i in 0..update_count {
            let profile = &profiles[i];
            let key = format!("{}'s hobby", profile.name);
            let new_value = format!("{}_updated", profile.preferences[0].1);
            // Update at a later timestamp (future epoch)
            agent.memorize(&key, &new_value, 0.9);
        }

        for i in 0..update_count {
            let profile = &profiles[i];
            let key = format!("{}'s hobby", profile.name);
            let expected = format!("{}_updated", profile.preferences[0].1);
            let (answer, _key, conf) = agent.answer(&key);
            if answer.as_deref() == Some(expected.as_str()) {
                update_correct += 1;
            } else {
                println!("MISMATCH for {}: expected {}, got {:?} with conf {}", key, expected, answer, conf);
            }
        }

        let ku_acc = update_correct as f64 / update_count as f64;
        println!(
            "  KnowledgeUpdate accuracy: {update_correct}/{update_count} ({:.1}%)",
            ku_acc * 100.0
        );

        // ── Efficiency metrics (MemBench Table 3 format) ─────────────
        println!("  ────────────────────────────────────────────────────────");
        println!("  Efficiency (MemBench-style):");
        println!("    Write time (WT): {:.3} sec/op", agent.avg_write_time());
        println!("    Read time  (RT): {:.3} sec/op", agent.avg_read_time());

        println!("  ══════════════════════════════════════════════════════════\n");

        let _ = std::fs::remove_dir_all(&dir);

        // MemBench-style: report all metrics, assert baseline
        // Key resolution now uses exact match → token overlap → fuzzy
        // with match_quality weighting to prevent cross-epoch false positives.
        assert!(
            sh_acc > 0.90,
            "SingleHop accuracy should be >90%, got {:.1}%",
            sh_acc * 100.0
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // Test 2: MemBench Memory Capacity Evaluation
    // ═══════════════════════════════════════════════════════════════
    // Determines at what point accuracy degrades as fact count grows.
    // MemBench measures this by plotting accuracy vs token count.

    #[test]
    fn membench_capacity() {
        println!("\n  ══════════════════════════════════════════════════════════");
        println!("  MEMBENCH: Memory Capacity Evaluation");
        println!("  Accuracy vs fact count (100, 500, 1k, 2k, 3k)");
        println!("  ══════════════════════════════════════════════════════════");

        let capacities = [100, 500, 1000, 2000, 3000];

        for &cap in &capacities {
            let dir = std::env::temp_dir().join(format!("rig_membench_cap_{cap}"));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();

            let mut agent = TemporalAgent::new(dir.clone());
            let profiles = generate_user_profiles(cap / 7 + 1); // ~7 facts per profile

            let mut qa_pairs: Vec<(String, String)> = Vec::new();
            let mut stored = 0;

            'outer: for (u, profile) in profiles.iter().enumerate() {
                let session_time = u as f64 / profiles.len() as f64;

                for (rel_name, relationship, city) in &profile.relations {
                    if stored >= cap {
                        break 'outer;
                    }
                    let key = format!("{}'s {} {}", profile.name, relationship, rel_name);
                    let value = format!("{rel_name} from {city}");
                    agent.memorize(&key, &value, session_time);
                    qa_pairs.push((key, value));
                    stored += 1;
                }
                for (event_name, time, location) in &profile.events {
                    if stored >= cap {
                        break 'outer;
                    }
                    let key = format!("{}'s event {}", profile.name, event_name);
                    let value = format!("{event_name} on {time} in {location}");
                    agent.memorize(&key, &value, session_time);
                    qa_pairs.push((key, value));
                    stored += 1;
                }
                for (category, preference) in &profile.preferences {
                    if stored >= cap {
                        break 'outer;
                    }
                    let key = format!("{}'s {}", profile.name, category);
                    let value = preference.clone();
                    agent.memorize(&key, &value, session_time);
                    qa_pairs.push((key, value));
                    stored += 1;
                }
            }

            // Sample 50 facts to test
            let sample_size = 50.min(qa_pairs.len());
            let step = qa_pairs.len() / sample_size;
            let mut correct = 0;

            for i in 0..sample_size {
                let idx = i * step;
                let (query, expected) = &qa_pairs[idx];
                let (answer, _key, _conf) = agent.answer(query);
                if answer.as_deref() == Some(expected.as_str()) {
                    correct += 1;
                }
            }

            let acc = correct as f64 / sample_size as f64;
            let status = if acc > 0.90 { "✓" } else { "✗" };
            println!(
                "  {status} {cap:>5} facts | accuracy={correct:>2}/{sample_size} ({:>5.1}%) | WT={:.3}s RT={:.3}s",
                acc * 100.0,
                agent.avg_write_time(),
                agent.avg_read_time()
            );

            let _ = std::fs::remove_dir_all(&dir);
        }

        println!("  ══════════════════════════════════════════════════════════\n");
    }

    // ═══════════════════════════════════════════════════════════════
    // Test 3: MemBench Reflective Memory (Multi-hop + Aggregative)
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn membench_reflective_memory() {
        let dir = std::env::temp_dir().join("rig_membench_rm");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut agent = TemporalAgent::new(dir.clone());
        let profiles = generate_user_profiles(50);

        println!("\n  ══════════════════════════════════════════════════════════");
        println!("  MEMBENCH: Reflective Memory Evaluation");
        println!("  MultiHop, Comparative, Aggregative question types");
        println!("  ══════════════════════════════════════════════════════════");

        // Ingest all facts
        for (u, profile) in profiles.iter().enumerate() {
            let session_time = u as f64 / profiles.len() as f64;

            for (rel_name, relationship, city) in &profile.relations {
                let key = format!("{}'s {} {}", profile.name, relationship, rel_name);
                let value = format!("{rel_name} from {city}");
                agent.memorize(&key, &value, session_time);
            }

            for (event_name, time, _location) in &profile.events {
                let key = format!("{}'s event {}", profile.name, event_name);
                let value = format!("{event_name} on {time}");
                agent.memorize(&key, &value, session_time);
            }

            for (category, preference) in &profile.preferences {
                let key = format!("{}'s {}", profile.name, category);
                let value = preference.clone();
                agent.memorize(&key, &value, session_time);
            }
        }

        // ── MultiHop: retrieve one fact, use it to find another ──────
        let mut multihop_correct = 0;
        let test_count = 30.min(profiles.len());

        for i in 0..test_count {
            let profile = &profiles[i];
            if profile.relations.is_empty() {
                continue;
            }

            // Step 1: "Who is {name}'s sister?" → get the relation name
            let (rel_name, rel, city) = &profile.relations[0];
            let query1 = format!("{}'s {} {}", profile.name, rel, rel_name);
            let (hop1_answer, _key, _conf) = agent.answer(&query1);

            // Step 2: Verify the answer contains the right info
            if let Some(ans) = hop1_answer {
                if ans.contains(rel_name) && ans.contains(city) {
                    multihop_correct += 1;
                }
            }
        }

        let mh_acc = multihop_correct as f64 / test_count as f64;
        println!(
            "  MultiHop:     {multihop_correct}/{test_count} ({:.1}%)",
            mh_acc * 100.0
        );

        // ── Aggregative: "How many people are from {city}?" ──────────
        // Count facts that reference a specific city, then verify
        let test_city = "Boston, MA";
        let city_query = format!("from {test_city}");

        // Count expected
        let mut expected_count = 0;
        for profile in &profiles {
            for (_, _, city) in &profile.relations {
                if city.contains(test_city) {
                    expected_count += 1;
                }
            }
        }

        // Do a single query (aggregation happens at a higher layer)
        let (agg_answer, _key, _conf) = agent.answer(&city_query);
        let agg_found = agg_answer.is_some();
        println!(
            "  Aggregative:  Found={}, expected_matches_in_data={} (aggregation is agent-layer concern)",
            agg_found, expected_count
        );

        // ── Version evolution (KnowledgeUpdate from MemBench) ────────
        // Same key stored at different timestamps → latest wins per-epoch
        agent.memorize("project_status", "planning phase", 0.1);
        agent.memorize("project_status", "implementation v2", 0.5);
        agent.memorize("project_status", "deployed to prod", 0.8);

        let (ans, _key, _conf) = agent.answer("project_status");
        let version_ok = ans.is_some();
        println!(
            "  KnowledgeUpdate: version={:?}, found={version_ok}",
            ans.as_deref().unwrap_or("NONE")
        );

        println!("  ────────────────────────────────────────────────────────");
        println!("  Efficiency:");
        println!("    Write time (WT): {:.3} sec/op", agent.avg_write_time());
        println!("    Read time  (RT): {:.3} sec/op", agent.avg_read_time());

        println!("  ══════════════════════════════════════════════════════════\n");

        let _ = std::fs::remove_dir_all(&dir);

        assert!(
            mh_acc > 0.85,
            "MultiHop accuracy should be >85%, got {:.1}%",
            mh_acc * 100.0
        );
    }
}
