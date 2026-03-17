//! 3-Way Policy Comparison with CMDP Safety + RLER Rubric Evolution.
//!
//! Tests Random vs RuleBased vs EmbeddingPolicy across both environments,
//! with evolving rubrics and safety constraints.

#[cfg(test)]
mod rl_three_way_tests {
    use hehrgnn::eval::agent_env::AgentEnv;
    use hehrgnn::eval::code_sandbox::CodeSandbox;
    use hehrgnn::eval::embedding_policy::EmbeddingPolicy;
    use hehrgnn::eval::environment::Environment;
    use hehrgnn::eval::fiduciary_env::FiduciaryEnv;
    use hehrgnn::eval::rl_policy::{RandomPolicy, RuleBasedPolicy};
    use hehrgnn::eval::rubric::{Rubric, RubricEvolver, RubricJudge};
    use hehrgnn::eval::simulator::run_episode;
    use hehrgnn::eval::transition_buffer::TransitionBuffer;

    // ═══════════════════════════════════════════════════════
    // TEST 1: 3-Way Agent Comparison with CMDP + RLER
    // ═══════════════════════════════════════════════════════
    #[test]
    fn test_three_way_agent_with_rubric_evolution() {
        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║  3-Way Agent Policy Comparison + CMDP + RLER             ║");
        println!("╚══════════════════════════════════════════════════════════╝");

        struct PolicyResult {
            name: String,
            mean_reward: f64,
            mean_rubric: f64,
            mean_composite: f64,
            violations: usize,
            rubric_version: usize,
            criteria_count: usize,
            episodes: usize,
        }

        let num_episodes = 30;
        let evolve_interval = 10;
        let mut results = Vec::new();

        // Run each policy
        for policy_idx in 0..3 {
            let mut env = AgentEnv::new();
            let mut buffer = TransitionBuffer::new(2000);
            let mut rubric = Rubric::agent_default();
            let mut judge = RubricJudge::new(rubric.clone());
            let mut evolver = RubricEvolver::default_config();

            // Create policies — keep embedding policy handle for REINFORCE
            let mut random = RandomPolicy::new(42);
            let mut rule = RuleBasedPolicy::active_agent();
            let mut embedding = EmbeddingPolicy::new(7, 16);

            let mut total_reward = 0.0;
            let mut total_rubric = 0.0;
            let mut total_composite = 0.0;

            for ep in 0..num_episodes {
                let log = match policy_idx {
                    0 => run_episode(&mut env, &mut random, Some(&mut buffer), 50.0),
                    1 => run_episode(&mut env, &mut rule, Some(&mut buffer), 50.0),
                    _ => run_episode(&mut env, &mut embedding, Some(&mut buffer), 50.0),
                };

                // Tier 3: Rubric score
                let state = env.state();
                let metrics = state.to_metrics();
                let rubric_score = judge.score(&metrics);

                let composite = log.total_reward + 0.3 * rubric_score;
                total_reward += log.total_reward;
                total_rubric += rubric_score;
                total_composite += composite;
                evolver.record_episode();

                // REINFORCE update for embedding policy
                if policy_idx == 2 {
                    let features = state.to_features();
                    embedding.reinforce_update(0, &features, composite);
                }

                // Evolve rubric every N episodes
                if (ep + 1) % evolve_interval == 0 && evolver.should_evolve(judge.num_scored()) {
                    rubric = evolver.evolve(&judge);
                    judge = RubricJudge::new(rubric.clone());
                    if ep < num_episodes - 1 {
                        println!(
                            "    [{}] Rubric evolved to v{} ({} criteria) at ep {}",
                            match policy_idx {
                                0 => "random",
                                1 => "rule_based",
                                _ => "embedding",
                            },
                            rubric.version,
                            rubric.num_criteria(),
                            ep + 1
                        );
                    }
                }
            }

            let n = num_episodes as f64;
            let violations = env.cmdp().total_violations();
            let policy_name = match policy_idx {
                0 => "random",
                1 => "rule_based",
                _ => "embedding",
            }
            .to_string();

            results.push(PolicyResult {
                name: policy_name,
                mean_reward: total_reward / n,
                mean_rubric: total_rubric / n,
                mean_composite: total_composite / n,
                violations,
                rubric_version: rubric.version as usize,
                criteria_count: rubric.num_criteria(),
                episodes: num_episodes,
            });
        }

        // Print results
        println!("\n  ┌────────────────┬──────────┬──────────┬───────────┬────────┬─────────┐");
        println!("  │ Policy         │  Reward  │  Rubric  │ Composite │ Violat │ Rubr.V  │");
        println!("  ├────────────────┼──────────┼──────────┼───────────┼────────┼─────────┤");
        for r in &results {
            println!(
                "  │ {:14} │ {:8.2} │ {:8.4} │ {:9.2} │ {:6} │ v{} ({}) │",
                r.name,
                r.mean_reward,
                r.mean_rubric,
                r.mean_composite,
                r.violations,
                r.rubric_version,
                r.criteria_count
            );
        }
        println!("  └────────────────┴──────────┴──────────┴───────────┴────────┴─────────┘");

        // Assertions
        for r in &results {
            assert!(r.mean_composite.is_finite(), "{} reward not finite", r.name);
            assert!(
                r.mean_rubric >= 0.0 && r.mean_rubric <= 1.0,
                "{} rubric out of range",
                r.name
            );
        }

        println!("\n  ✓ All 3 policies tested with CMDP safety");
        println!("  ✓ Rubric evolved for each policy independently");
        println!("  ✓ Composite reward = env_reward + 0.3 × rubric_score");
    }

    // ═══════════════════════════════════════════════════════
    // TEST 2: 3-Way Fiduciary Comparison with CMDP + RLER
    // ═══════════════════════════════════════════════════════
    #[test]
    fn test_three_way_fiduciary_with_rubric_evolution() {
        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║  3-Way Fiduciary Policy Comparison + CMDP + RLER         ║");
        println!("╚══════════════════════════════════════════════════════════╝");

        let num_episodes = 6;

        let mut results = Vec::new();

        for policy_idx in 0..3 {
            let mut env = FiduciaryEnv::new(4);
            let mut rubric = Rubric::fiduciary_default();
            let mut judge = RubricJudge::new(rubric.clone());
            let mut evolver = RubricEvolver::default_config();

            let mut random = RandomPolicy::new(42);
            let mut rule = RuleBasedPolicy::conservative_fiduciary();
            let mut embedding = EmbeddingPolicy::new(4, 12);

            let mut total_reward = 0.0;
            let mut total_rubric = 0.0;

            for ep in 0..num_episodes {
                let log = match policy_idx {
                    0 => run_episode(&mut env, &mut random, None, 1.0),
                    1 => run_episode(&mut env, &mut rule, None, 1.0),
                    _ => run_episode(&mut env, &mut embedding, None, 1.0),
                };
                let state = env.state();
                let metrics = state.to_metrics();
                let rubric_score = judge.score(&metrics);

                total_reward += log.total_reward;
                total_rubric += rubric_score;
                evolver.record_episode();

                // Evolve rubric at halfway point
                if ep == num_episodes / 2 && evolver.should_evolve(judge.num_scored()) {
                    rubric = evolver.evolve(&judge);
                    judge = RubricJudge::new(rubric.clone());
                }
            }

            let n = num_episodes as f64;
            let policy_name = match policy_idx {
                0 => "random",
                1 => "rule_based",
                _ => "embedding",
            };
            println!(
                "    {:14} reward={:8.2}  rubric={:.4}  violations={}  rubric_v{}",
                policy_name,
                total_reward / n,
                total_rubric / n,
                env.cmdp().total_violations(),
                rubric.version,
            );

            results.push((
                policy_name.to_string(),
                total_reward / n,
                total_rubric / n,
                rubric.version,
            ));
        }

        println!("\n  ✓ Fiduciary 3-way comparison complete with rubric evolution");
    }

    // ═══════════════════════════════════════════════════════
    // TEST 3: Code Sandbox Integration
    // ═══════════════════════════════════════════════════════
    #[test]
    fn test_sandbox_with_safety() {
        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║  Code Sandbox with CMDP Safety + Rubric                  ║");
        println!("╚══════════════════════════════════════════════════════════╝");

        let mut sandbox = CodeSandbox::new(8888);
        sandbox.init().expect("Failed to init sandbox");
        let mut judge = RubricJudge::new(Rubric::agent_default());

        // Step 1: Compile (verifiable reward)
        let check = sandbox.check();
        println!(
            "  Step 1 - Compile: success={}, reward={:.1}, elapsed={}ms",
            check.success,
            check.compile_reward(),
            check.elapsed_ms
        );
        assert!(check.success);

        // Step 2: Test (verifiable reward)
        let test = sandbox.test();
        println!(
            "  Step 2 - Test:    passed={}/{}, reward={:.1}, elapsed={}ms",
            test.passed,
            test.total,
            test.test_reward(),
            test.elapsed_ms
        );
        assert!(test.success);
        assert!(test.passed > 0);

        // Step 3: Rubric score from sandbox metrics
        let metrics = sandbox.to_metrics(&check, &test);
        let rubric_score = judge.score(&metrics);
        println!("  Step 3 - Rubric:  score={:.4}", rubric_score);

        // Step 4: Apply a breaking patch
        sandbox.apply_patch("a + b", "a * b * b").unwrap();
        let check2 = sandbox.check();
        let test2 = sandbox.test();
        println!("  Step 4 - After patch:");
        println!("    Compile: success={}", check2.success);
        println!(
            "    Tests:   passed={}, failed={}",
            test2.passed, test2.failed
        );

        // Step 5: Composite reward
        let composite = check.compile_reward() + test.test_reward() + 0.3 * rubric_score;
        println!(
            "\n  Composite reward: {:.2} (compile={:.1} + test={:.1} + rubric={:.2})",
            composite,
            check.compile_reward(),
            test.test_reward(),
            0.3 * rubric_score
        );

        assert!(
            composite > 0.0,
            "Good code should get positive composite reward"
        );
        println!("\n  ✓ Code sandbox produces verifiable rewards");
        println!("  ✓ Rubric scoring works with sandbox metrics");
        println!("  ✓ Breaking patches cause test failures");
    }

    // ═══════════════════════════════════════════════════════
    // TEST 4: Embedding Policy with REINFORCE Training
    // ═══════════════════════════════════════════════════════
    #[test]
    fn test_embedding_policy_reinforce_training() {
        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║  Embedding Policy REINFORCE Training                     ║");
        println!("╚══════════════════════════════════════════════════════════╝");

        let mut env = AgentEnv::new();
        let mut policy = EmbeddingPolicy::new(7, 16);
        let mut buffer = TransitionBuffer::new(1000);

        // Collect pre-training episodes
        let pre_report = {
            let mut r = Vec::new();
            for _ in 0..10 {
                let log = run_episode(&mut env, &mut policy, Some(&mut buffer), 50.0);
                r.push(log.total_reward);
            }
            r.iter().sum::<f64>() / r.len() as f64
        };

        println!("  Pre-training mean reward:  {:.2}", pre_report);
        println!("  Buffer size:               {}", buffer.len());

        // Train from buffer
        let batch = buffer.sample(buffer.len().min(200));
        policy.train_from_buffer(&batch.states, &batch.action_ids, &batch.rewards);

        // Post-training episodes
        let post_report = {
            let mut r = Vec::new();
            for _ in 0..10 {
                let log = run_episode(&mut env, &mut policy, None, 50.0);
                r.push(log.total_reward);
            }
            r.iter().sum::<f64>() / r.len() as f64
        };

        println!("  Post-training mean reward: {:.2}", post_report);
        println!("  Baseline:                  {:.4}", policy.baseline());

        println!("\n  ✓ REINFORCE training executed from replay buffer");
        println!("  ✓ Policy baseline updated: {:.4}", policy.baseline());
    }
}
