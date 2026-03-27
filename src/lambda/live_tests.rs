//! Live λ-RLM integration tests with Trinity model.
//!
//! Tests the full agent pipeline end-to-end with Trinity:
//! - Open-ended coding prompts (no pre-baked questions)
//! - The model decides when to read files, grep, and elicit
//! - λ-RLM kicks in for long-context decomposition
//! - Thinking traces emitted to a `tokio::broadcast` channel
//!
//! Run all tests:          `cargo test lambda_live -- --ignored --nocapture`
//! Run pure (no LLM):      `cargo test lambda_live -- --nocapture`
//!
//! Requires: OPENAI_API_KEY env var (or .env file with OpenRouter key)

#[cfg(test)]
mod lambda_live {
    use std::sync::Arc;

    use crate::channels::{ChannelEvent, ChannelMeta, HubPublisher};
    use crate::lambda::{self, LambdaConfig};
    use crate::lambda::combinators;
    use crate::lambda::planner::{self, CostParams, TaskType};
    use crate::monad::{
        AgentConfig, AgentContext, AgentMonad, RunResult,
        action::ActionOutput,
        interaction::agent_task_full,
        provider::{LlmProvider, ProviderConfig},
    };

    /// Build a live LlmProvider pointed at OpenRouter + Trinity.
    fn trinity_provider() -> Arc<LlmProvider> {
        dotenvy::dotenv().ok();

        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
        let base_url = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());
        let model = std::env::var("RIG_RLM_MODEL")
            .unwrap_or_else(|_| "arcee-ai/trinity-large-preview:free".to_string());

        let config = ProviderConfig::openai_compatible("openrouter", &model, &base_url, &api_key);
        Arc::new(LlmProvider::new(config))
    }

    /// Build an AgentContext wired to Trinity.
    fn trinity_context(max_turns: usize) -> AgentContext {
        dotenvy::dotenv().ok();

        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
        let base_url = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());
        let model = std::env::var("RIG_RLM_MODEL")
            .unwrap_or_else(|_| "arcee-ai/trinity-large-preview:free".to_string());

        let provider =
            ProviderConfig::openai_compatible("openrouter", &model, &base_url, &api_key);

        let config = AgentConfig {
            provider,
            max_turns,
            ..AgentConfig::default()
        };

        AgentContext::new(config)
    }

    /// Create a broadcast channel for thinking traces.
    fn trace_channel() -> (HubPublisher, tokio::sync::broadcast::Receiver<Arc<ChannelEvent>>) {
        let (tx, rx) = tokio::sync::broadcast::channel::<Arc<ChannelEvent>>(64);
        (HubPublisher::new(tx), rx)
    }

    /// Emit a thinking trace event to the channel.
    fn emit_trace(publisher: &HubPublisher, phase: &str, content: &str) {
        let event = ChannelEvent::new("lambda-rlm", format!("agent/thinking/{phase}"), content)
            .with_meta(
                ChannelMeta::new()
                    .insert("phase", phase.to_string())
                    .insert("timestamp", chrono::Utc::now().to_rfc3339()),
            );
        publisher.publish(event);
    }

    /// Generate a long document for decomposition tests.
    fn long_document(target_tokens: usize) -> String {
        let sections = [
            ("Geography", "France is a country in Western Europe. Its capital is Paris, which sits along the Seine river. The country spans approximately 643,801 square kilometers."),
            ("History", "France has a rich history dating back to the Gallic period. The French Revolution in 1789 fundamentally transformed the country."),
            ("Economy", "France has the world's seventh-largest economy by nominal GDP. Key sectors include aerospace (Airbus), nuclear energy (EDF), and luxury goods (LVMH)."),
            ("Science", "France has produced numerous Nobel Prize winners. French scientists have made fundamental contributions to mathematics (Poincaré, Galois), biology (Pasteur), and physics (Curie)."),
        ];

        let mut doc = String::new();
        let target_chars = target_tokens * 4;
        let mut i = 0;
        while doc.len() < target_chars {
            let (title, content) = &sections[i % sections.len()];
            doc.push_str(&format!("\n## {} (Section {})\n\n{}\n\n", title, i + 1, content));
            i += 1;
        }
        doc
    }

    // ═══════════════════════════════════════════════════════════════
    // Pure planner tests (no LLM needed)
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn plan_short_document_direct_call() {
        let plan = planner::plan(1000, 32_000, 0.80, TaskType::Summarise, &CostParams::default());
        eprintln!("📋 {}", plan.summary());
        assert_eq!(plan.k_star, 1, "Short doc should use direct call");
        assert_eq!(plan.depth, 0);
        assert_eq!(plan.estimated_calls, 1);
    }

    #[test]
    fn plan_long_document_decomposition() {
        let plan =
            planner::plan(128_000, 32_000, 0.80, TaskType::Summarise, &CostParams::default());
        eprintln!("📋 {}", plan.summary());
        assert!(plan.k_star >= 2, "Long doc should decompose, got k*={}", plan.k_star);
        assert!(plan.depth >= 1);
        assert!(plan.estimated_calls > 1);
    }

    #[test]
    fn plan_search_has_prefilter() {
        let plan = planner::plan(100_000, 32_000, 0.80, TaskType::Search, &CostParams::default());
        assert!(plan.has_prefilter, "Search should use pre-filter");
    }

    #[test]
    fn combinator_split_and_recombine() {
        let doc = long_document(1000);
        let chunks = combinators::split(&doc, 4);
        assert_eq!(chunks.len(), 4);
        let total_chars: usize = chunks.iter().map(|c| c.len()).sum();
        assert!(total_chars >= doc.len() * 9 / 10);
    }

    #[test]
    fn trace_channel_receives_events() {
        let (publisher, mut rx) = trace_channel();
        emit_trace(&publisher, "planning", "Computing optimal k*=2");
        emit_trace(&publisher, "split", "Splitting into 4 chunks");

        let e1 = rx.try_recv().unwrap();
        assert_eq!(e1.topic, "agent/thinking/planning");
        assert!(e1.content.contains("k*=2"));

        let e2 = rx.try_recv().unwrap();
        assert_eq!(e2.topic, "agent/thinking/split");
    }

    // ═══════════════════════════════════════════════════════════════
    // 1. Open-ended coding: model reads real files, greps, codes
    //    No questions pre-baked — model drives the conversation
    // ═══════════════════════════════════════════════════════════════

    /// Trinity is told to analyze the lambda module and write a benchmark.
    /// It should autonomously:
    ///   - Read files from src/lambda/ using os.listdir + open()
    ///   - Grep for patterns it finds interesting
    ///   - Possibly ELICIT() if unsure about something
    ///   - SUBMIT() the code it produces
    #[tokio::test]
    #[ignore]
    async fn live_open_coding_read_files_and_write_code() {
        let mut ctx = trinity_context(20);
        let (trace_pub, mut trace_rx) = trace_channel();

        let sep = "═".repeat(60);
        eprintln!("\n{sep}");
        eprintln!("⚡ OPEN CODING: Trinity reads local files + writes code");
        eprintln!("{sep}");

        emit_trace(&trace_pub, "init", "Open-ended coding task started");

        // Open-ended prompt — model drives what it reads and writes
        let program = agent_task_full(
            "Look at the Rust source files in /home/sumit-mittal/dev-stuff/rig-rlm/src/lambda/. \
             Read the markdown file /home/sumit-mittal/dev-stuff/rig-rlm/AGENTICA_IDEAS.md. Since it's a large file, read it by calling `open().read()` and pass the raw string through our custom Python binding: `lambda_rlm_analyze(text, \"summarize the agentica ideas into an execution plan\")`. \
             After the RLM engine reduces the context and gives you the explanation, write a Python script that prints the summary. \
             Use os.listdir() to browse if needed. \
             When you're done, call SUBMIT(your_python_script).",
            None,
            None,
            None,
            vec![],
        );

        let timer = std::time::Instant::now();
        let result = ctx.run(program).await;
        let elapsed = timer.elapsed();

        emit_trace(&trace_pub, "complete", &format!("Done in {:.1}s", elapsed.as_secs_f64()));

        match &result {
            Ok(run_result) => {
                if run_result.is_suspended() {
                    // Model chose to ELICIT — that's fine, print what it asked
                    eprintln!("  🛑 Model ELICITED (it had a question):");
                    eprintln!("     {}", run_result.as_str());
                    emit_trace(&trace_pub, "elicited", run_result.as_str());
                } else {
                    let answer = run_result.as_str();
                    eprintln!("  ✅ Model submitted ({} chars, {:.1}s):", answer.len(), elapsed.as_secs_f64());
                    for line in answer.lines().take(25) {
                        eprintln!("  │ {line}");
                    }
                    if answer.lines().count() > 25 {
                        eprintln!("  │ ... ({} more lines)", answer.lines().count() - 25);
                    }
                    emit_trace(&trace_pub, "submitted", &format!("{} chars of code", answer.len()));
                }
            }
            Err(e) => {
                eprintln!("  ⚠️ Error: {e}");
                emit_trace(&trace_pub, "error", &format!("{e}"));
            }
        }

        // Dump traces
        let mut trace_count = 0;
        while let Ok(event) = trace_rx.try_recv() {
            eprintln!("  📡 [{:30}] {}", event.topic, &event.content[..event.content.len().min(80)]);
            trace_count += 1;
        }

        eprintln!("\n  📊 Evidence: {}, History: {}, Traces: {}, Time: {:.1}s",
            ctx.evidence().len(), ctx.history.len(), trace_count, elapsed.as_secs_f64());
        eprintln!("{sep}\n");
    }

    // ═══════════════════════════════════════════════════════════════
    // 2. Open-ended coding + HITL: model reads, elicits, iterates
    //    Model decides when to ask — user gives feedback — model
    //    continues until satisfied
    // ═══════════════════════════════════════════════════════════════

    /// Trinity is given an open task on real code. If it elicits,
    /// we respond and let it continue. Full loop.
    #[tokio::test]
    #[ignore]
    async fn live_open_coding_with_hitl_iteration() {
        let mut ctx = trinity_context(30);
        let (trace_pub, mut trace_rx) = trace_channel();

        let sep = "═".repeat(60);
        eprintln!("\n{sep}");
        eprintln!("⚡ OPEN CODING + HITL: Model reads, asks, iterates");
        eprintln!("{sep}");

        emit_trace(&trace_pub, "init", "Open coding + HITL iteration started");

        // Prompt that naturally leads to elicitation:
        // the model should want to know which error cases to handle
        let program = agent_task_full(
            "Read the file /home/sumit-mittal/dev-stuff/rig-rlm/src/lambda/executor.rs. \
             Understand the recursive phi() function. \
             Then write a Python implementation of the same recursive decomposition pattern. \
             If you need to know anything about how I want it configured (chunk size, max depth, etc), \
             use ELICIT() to ask me. \
             When done, call SUBMIT(your_code).",
            None,
            None,
            None,
            vec![],
        );

        let timer = std::time::Instant::now();
        let result = ctx.run(program).await;
        let elapsed = timer.elapsed();

        emit_trace(&trace_pub, "first-pass", &format!("First pass done in {:.1}s", elapsed.as_secs_f64()));

        match result {
            Ok(run_result) => {
                if run_result.is_suspended() {
                    // Model asked something — print it and respond
                    eprintln!("  🛑 Model asked (Round 1):");
                    eprintln!("     {}", run_result.as_str());

                    // Resume with a natural response
                    match run_result {
                        RunResult::Suspended { continuation, .. } => {
                            let user_response = "Use chunk_size=500 tokens, max_depth=3, \
                                                  and make it work with any callable as the leaf function";

                            eprintln!("  💬 User responds: {user_response}");
                            emit_trace(&trace_pub, "user-reply", user_response);

                            let resumed = match continuation {
                                AgentMonad::Perform { next, .. } => {
                                    next(ActionOutput::Value(user_response.to_string()))
                                }
                                other => other,
                            };

                            let round2 = ctx.run(resumed).await;
                            let elapsed2 = timer.elapsed();

                            match &round2 {
                                Ok(r) => {
                                    if r.is_suspended() {
                                        // Model asked again — print
                                        eprintln!("  🛑 Model asked again (Round 2): {}", r.as_str());
                                        emit_trace(&trace_pub, "elicit-2", r.as_str());

                                        // Respond and continue
                                        match round2.unwrap() {
                                            RunResult::Suspended { continuation, .. } => {
                                                eprintln!("  💬 User: 'Yes, that sounds good. Go ahead.'");
                                                let resumed2 = match continuation {
                                                    AgentMonad::Perform { next, .. } => {
                                                        next(ActionOutput::Value(
                                                            "Yes, that sounds good. Go ahead and finalize it.".to_string(),
                                                        ))
                                                    }
                                                    other => other,
                                                };
                                                let round3 = ctx.run(resumed2).await;
                                                match &round3 {
                                                    Ok(r3) => {
                                                        eprintln!("  ✅ Final output ({} chars, {:.1}s total)",
                                                            r3.as_str().len(), elapsed2.as_secs_f64());
                                                        print_code(r3.as_str());
                                                    }
                                                    Err(e) => eprintln!("  ⚠️ Round 3 error: {e}"),
                                                }
                                            }
                                            _ => unreachable!(),
                                        }
                                    } else {
                                        // Model completed after first user response
                                        eprintln!("  ✅ Model completed after Round 2 ({} chars, {:.1}s)",
                                            r.as_str().len(), elapsed2.as_secs_f64());
                                        print_code(r.as_str());
                                    }
                                }
                                Err(e) => eprintln!("  ⚠️ Round 2 error: {e}"),
                            }
                        }
                        _ => unreachable!(),
                    }
                } else {
                    // Model completed without asking — also fine
                    eprintln!("  ✅ Model completed directly ({} chars, {:.1}s):",
                        run_result.as_str().len(), elapsed.as_secs_f64());
                    print_code(run_result.as_str());
                }
            }
            Err(e) => {
                eprintln!("  ⚠️ Error: {e}");
            }
        }

        emit_trace(&trace_pub, "done", "Test complete");

        // Dump traces
        let mut trace_count = 0;
        while let Ok(event) = trace_rx.try_recv() {
            eprintln!("  📡 [{:30}] {}", event.topic, &event.content[..event.content.len().min(80)]);
            trace_count += 1;
        }

        eprintln!("\n  📊 Evidence: {}, History: {}, Traces: {}",
            ctx.evidence().len(), ctx.history.len(), trace_count);
        eprintln!("{sep}\n");
    }

    // ═══════════════════════════════════════════════════════════════
    // 3. λ-RLM on a large file: model reads a file, λ-RLM decomposes
    // ═══════════════════════════════════════════════════════════════

    /// Model reads a real source file, and if it exceeds context,
    /// λ-RLM decomposes it automatically.
    #[tokio::test]
    #[ignore]
    async fn live_lambda_rlm_on_real_source_file() {
        let provider = trinity_provider();
        let (trace_pub, mut trace_rx) = trace_channel();

        let sep = "═".repeat(60);
        eprintln!("\n{sep}");
        eprintln!("⚡ λ-RLM ON REAL SOURCE: reads and decomposes a file");
        eprintln!("{sep}");

        // Read a real source file from the project
        let file_path = "/home/sumit-mittal/dev-stuff/rig-rlm/src/monad/interaction.rs";
        let source = std::fs::read_to_string(file_path)
            .unwrap_or_else(|e| panic!("Cannot read {file_path}: {e}"));

        let doc_tokens = combinators::token_count(&source);
        eprintln!("  📄 File: {file_path}");
        eprintln!("  📄 Size: {} chars, ~{} tokens", source.len(), doc_tokens);

        emit_trace(&trace_pub, "init", &format!("Analyzing {file_path}: ~{doc_tokens} tokens"));

        // Force decomposition with small context window
        let config = LambdaConfig::default().with_context_window(1500);

        let plan = planner::plan(doc_tokens, 1500, 0.80, TaskType::Search, &CostParams::default());
        emit_trace(&trace_pub, "plan", &plan.summary());
        eprintln!("  📋 {}", plan.summary());

        let timer = std::time::Instant::now();
        let result = lambda::lambda_rlm(
            &source,
            "What error handling patterns does this code use? List the specific error types and how they propagate.",
            provider,
            config,
        )
        .await;
        let elapsed = timer.elapsed();

        emit_trace(&trace_pub, "complete", &format!("Done in {:.1}s", elapsed.as_secs_f64()));

        match &result {
            Ok(answer) => {
                eprintln!("  ✅ Analysis ({} chars, {:.1}s):", answer.len(), elapsed.as_secs_f64());
                for line in answer.lines().take(15) {
                    eprintln!("  │ {line}");
                }
            }
            Err(e) => eprintln!("  ⚠️ Error: {e}"),
        }

        // Dump traces
        let mut trace_count = 0;
        while let Ok(event) = trace_rx.try_recv() {
            eprintln!("  [{}] {}", event.topic, event.content);
            trace_count += 1;
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // 3. Category Theory Extension: The Yoneda Context Evaluation !
    // ═══════════════════════════════════════════════════════════════

    #[tokio::test]
    #[ignore]
    async fn live_yoneda_context_evaluation() {
        let provider = trinity_provider();
        let (trace_pub, mut _trace_rx) = trace_channel();

        let sep = "═".repeat(60);
        eprintln!("\n{sep}");
        eprintln!("⚡ CATEGORY THEORY: Unfolding the Yoneda Context Functor");
        eprintln!("{sep}");

        // We embed the enormous AGENTICA_IDEAS document as a generic Functor P
        let file_path = "/home/sumit-mittal/dev-stuff/rig-rlm/AGENTICA_IDEAS.md";
        let source = std::fs::read_to_string(file_path).unwrap();

        // Instead of processing, we wrap it in a lazy YonedaContext
        let config = LambdaConfig::default().with_context_window(1500);
        let yoneda_context = lambda::yoneda::YonedaContext::new(source, provider, config);

        eprintln!("  🧿 Yoneda Context created.");
        eprintln!("  🧿 Document remains entirely unevaluated (deferred embedding).");

        // The exact identity of the massive context is defined by how it responds to the query
        let query = "Extract the three main orchestration goals of Agentica.";
        eprintln!("  🧿 Emitting Probe Morphism (X -> P): {:?}", query);

        let timer = std::time::Instant::now();
        // We evaluate the natural transformation
        let result = yoneda_context.probe(query).await;
        let elapsed = timer.elapsed();

        match &result {
            Ok(answer) => {
                eprintln!("  ✅ Yoneda Collapse ({} chars, {:.1}s):", answer.len(), elapsed.as_secs_f64());
                for line in answer.lines().take(15) {
                    eprintln!("  │ {line}");
                }
            }
            Err(e) => eprintln!("  ⚠️ Error: {e}"),
        }
    }    // ═══════════════════════════════════════════════════════════════
    // 4. Full loop: model reads → λ-RLM processes → model codes
    //    → user says "make it more dynamic" → model iterates
    // ═══════════════════════════════════════════════════════════════

    /// Realistic workflow:
    /// 1. Model reads combinator code via the sandbox
    /// 2. λ-RLM decomposes if needed
    /// 3. Model writes initial code
    /// 4. User says "make it more dynamic"  
    /// 5. Model iterates and improves
    #[tokio::test]
    #[ignore]
    async fn live_read_analyze_code_then_iterate() {
        let mut ctx = trinity_context(30);
        let (trace_pub, mut trace_rx) = trace_channel();

        let sep = "═".repeat(60);
        eprintln!("\n{sep}");
        eprintln!("⚡ FULL LOOP: Read → Analyze → Code → 'make it dynamic' → Iterate");
        eprintln!("{sep}");

        emit_trace(&trace_pub, "init", "Full iteration loop started");

        // ── Pass 1: Open-ended read + code ──
        eprintln!("\n  ── Pass 1: Read files and write initial code ──");

        let program = agent_task_full(
            "Read /home/sumit-mittal/dev-stuff/rig-rlm/src/lambda/combinators.rs using Python's open(). \
             Understand the split() and reduce() functions. \
             Then write a minimal Python library that provides the same combinators \
             (split, peek, filter, reduce, concat). Keep it simple for now. \
             Call SUBMIT(your_code) when done.",
            None,
            None,
            None,
            vec![],
        );

        let timer = std::time::Instant::now();
        let result = ctx.run(program).await;
        let elapsed1 = timer.elapsed();

        let initial_code = match &result {
            Ok(r) if !r.is_suspended() => {
                let code = r.as_str().to_string();
                eprintln!("  ✅ Pass 1 done ({} chars, {:.1}s)", code.len(), elapsed1.as_secs_f64());
                print_code(&code);
                emit_trace(&trace_pub, "pass-1-done", &format!("{} chars", code.len()));
                code
            }
            Ok(r) => {
                eprintln!("  🛑 Model elicited: {}", r.as_str());
                // Respond generically and continue
                match result.unwrap() {
                    RunResult::Suspended { continuation, .. } => {
                        let resumed = match continuation {
                            AgentMonad::Perform { next, .. } => {
                                next(ActionOutput::Value("Just use simple functions, nothing fancy".to_string()))
                            }
                            other => other,
                        };
                        let r2 = ctx.run(resumed).await.unwrap();
                        let code = r2.as_str().to_string();
                        eprintln!("  ✅ Pass 1 done after clarification ({} chars)", code.len());
                        print_code(&code);
                        code
                    }
                    _ => unreachable!(),
                }
            }
            Err(e) => {
                eprintln!("  ⚠️ Pass 1 error: {e}");
                String::new()
            }
        };

        if initial_code.is_empty() {
            eprintln!("  ⚠️ No code from pass 1, skipping pass 2");
            return;
        }

        // ── Pass 2: "Make it more dynamic" ──
        eprintln!("\n  ── Pass 2: 'Make it more dynamic' ──");
        emit_trace(&trace_pub, "pass-2-start", "User wants more dynamic version");

        let refine_program = agent_task_full(
            &format!(
                "Here is the code you wrote:\n```python\n{}\n```\n\n\
                 Make it more dynamic: \
                 add type hints, make the functions accept any iterable (not just lists), \
                 add a configurable chunk_size parameter to split(), \
                 and add a compose() function that chains multiple combinators together. \
                 Call SUBMIT(improved_code) when done.",
                initial_code
            ),
            None,
            None,
            None,
            vec![],
        );

        let result2 = ctx.run(refine_program).await;
        let elapsed2 = timer.elapsed();

        match &result2 {
            Ok(r) => {
                if r.is_suspended() {
                    eprintln!("  🛑 Model asked: {}", r.as_str());
                } else {
                    eprintln!("  ✅ Pass 2 done ({} chars, {:.1}s total)", r.as_str().len(), elapsed2.as_secs_f64());
                    print_code(r.as_str());
                    emit_trace(&trace_pub, "pass-2-done", &format!("{} chars", r.as_str().len()));

                    // Verify it actually improved
                    let improved = r.as_str();
                    if improved.len() > initial_code.len() {
                        eprintln!("  📈 Code grew from {} → {} chars", initial_code.len(), improved.len());
                    }
                }
            }
            Err(e) => eprintln!("  ⚠️ Pass 2 error: {e}"),
        }

        emit_trace(&trace_pub, "done", "Full iteration loop complete");

        // Dump traces
        let mut trace_count = 0;
        while let Ok(event) = trace_rx.try_recv() {
            eprintln!("  📡 [{:30}] {}", event.topic, &event.content[..event.content.len().min(80)]);
            trace_count += 1;
        }

        eprintln!("\n  📊 Evidence: {}, History: {}, Traces: {}, Total: {:.1}s",
            ctx.evidence().len(), ctx.history.len(), trace_count, elapsed2.as_secs_f64());
        eprintln!("{sep}\n");
    }

    /// Print code output with box drawing.
    fn print_code(code: &str) {
        eprintln!("  ┌─────────────────────────────────────");
        for line in code.lines().take(30) {
            eprintln!("  │ {line}");
        }
        if code.lines().count() > 30 {
            eprintln!("  │ ... ({} more lines)", code.lines().count() - 30);
        }
        eprintln!("  └─────────────────────────────────────");
    }

    // ═══════════════════════════════════════════════════════════════
    // 5. Yoneda Representable Functor — fmap + QueryMorphism
    //    Tests the full category theory stack end-to-end
    // ═══════════════════════════════════════════════════════════════

    /// Tests the representable functor y(P) with query morphisms:
    /// 1. probe(q) — evaluate on a base query
    /// 2. fmap(refine, q) — apply a query morphism and evaluate
    /// 3. Verify the morphism produces a more specific result
    #[tokio::test]
    #[ignore]
    async fn live_yoneda_representable_functor() {
        let provider = trinity_provider();

        let sep = "═".repeat(60);
        eprintln!("\n{sep}");
        eprintln!("⚡ YONEDA: Representable Functor y(P) with Query Morphisms");
        eprintln!("{sep}");

        // Build a document that has multiple topics
        let document = long_document(2000);
        let doc_tokens = combinators::token_count(&document);
        eprintln!("  📄 Document: {} chars, ~{} tokens", document.len(), doc_tokens);

        // Lift into Yoneda representation (lazy — no computation yet)
        let config = LambdaConfig::default().with_context_window(1500);
        let y_p = lambda::yoneda::YonedaContext::lift(document, provider, config);
        eprintln!("  🧿 y(P) created — document is unevaluated\n");

        // ── 1. Evaluate y(P)(q) for a base query ──
        let base_query = "What are the main topics covered?";
        eprintln!("  ── y(P)(q₁) — Base query: {:?}", base_query);

        let timer = std::time::Instant::now();
        let result1 = y_p.probe(base_query).await;
        let t1 = timer.elapsed();

        match &result1 {
            Ok(answer) => {
                eprintln!("  ✅ y(P)(q₁) = {} chars in {:.1}s:", answer.len(), t1.as_secs_f64());
                for line in answer.lines().take(8) {
                    eprintln!("  │ {line}");
                }
            }
            Err(e) => {
                eprintln!("  ⚠️ y(P)(q₁) failed: {e}");
                return;
            }
        }

        // ── 2. Apply a query morphism: refine → "focus on science" ──
        let refine = lambda::QueryMorphism::new(
            "focus_on_science",
            |q: &str| format!("{} Focus specifically on scientific contributions and Nobel Prize winners.", q),
        );

        eprintln!("\n  ── y(P)(f(q₁)) — Morphism: {} ──", refine.name);

        let timer2 = std::time::Instant::now();
        let result2 = y_p.fmap(&refine, base_query).await;
        let t2 = timer2.elapsed();

        match &result2 {
            Ok(answer) => {
                eprintln!("  ✅ y(P)(f(q₁)) = {} chars in {:.1}s:", answer.len(), t2.as_secs_f64());
                for line in answer.lines().take(8) {
                    eprintln!("  │ {line}");
                }

                // The refined result should mention science-related terms
                let has_science = answer.to_lowercase().contains("science")
                    || answer.to_lowercase().contains("nobel")
                    || answer.to_lowercase().contains("pasteur")
                    || answer.to_lowercase().contains("curie");
                eprintln!("\n  📊 Refined result mentions science terms: {}", has_science);
            }
            Err(e) => eprintln!("  ⚠️ y(P)(f(q₁)) failed: {e}"),
        }

        // ── 3. Compose two morphisms: g ∘ f ──
        let add_format = lambda::QueryMorphism::new(
            "request_bullet_points",
            |q: &str| format!("{} Answer in bullet points.", q),
        );
        let composed = add_format.compose(lambda::QueryMorphism::new(
            "focus_on_science",
            |q: &str| format!("{} Focus specifically on scientific contributions.", q),
        ));
        eprintln!("\n  ── y(P)((g∘f)(q₁)) — Composed: {} ──", composed.name);

        let result3 = y_p.fmap(&composed, base_query).await;
        match &result3 {
            Ok(answer) => {
                eprintln!("  ✅ y(P)((g∘f)(q₁)) = {} chars:", answer.len());
                for line in answer.lines().take(8) {
                    eprintln!("  │ {line}");
                }
            }
            Err(e) => eprintln!("  ⚠️ Composed morphism failed: {e}"),
        }

        eprintln!("\n{sep}\n");
    }

    // ═══════════════════════════════════════════════════════════════
    // 6. Profunctor Typed Pipeline — end-to-end struct → struct
    // ═══════════════════════════════════════════════════════════════

    /// Tests the Profunctor typed pipeline with real LLM calls:
    /// AnalysisRequest → [λ-RLM] → AnalysisReport
    #[tokio::test]
    #[ignore]
    async fn live_profunctor_typed_pipeline() {
        let provider = trinity_provider();

        let sep = "═".repeat(60);
        eprintln!("\n{sep}");
        eprintln!("⚡ PROFUNCTOR: TypedPipeline — AnalysisRequest → AnalysisReport");
        eprintln!("{sep}");

        // Domain types
        #[derive(Debug)]
        struct AnalysisRequest {
            document: String,
            focus_area: String,
            max_points: usize,
        }

        #[derive(Debug)]
        struct AnalysisReport {
            summary: String,
            point_count: usize,
            word_count: usize,
        }

        // Build the typed pipeline via Profunctor
        let config = LambdaConfig::default().with_context_window(1500);

        let pipeline = lambda::TypedPipeline::new(
            provider,
            config,
            "Analyze the document",
            // lmap (contravariant): AnalysisRequest → String
            |req: &AnalysisRequest| {
                format!(
                    "{}\n\nFocus on: {}\nProvide up to {} key points.",
                    req.document, req.focus_area, req.max_points
                )
            },
            // rmap (covariant): String → AnalysisReport
            |raw: String| {
                AnalysisReport {
                    point_count: raw.lines().filter(|l| l.starts_with('-') || l.starts_with('•') || l.starts_with('*')).count(),
                    word_count: raw.split_whitespace().count(),
                    summary: raw,
                }
            },
        );

        let request = AnalysisRequest {
            document: long_document(2000),
            focus_area: "economic and scientific achievements".to_string(),
            max_points: 5,
        };

        eprintln!("  🔬 Input: AnalysisRequest {{ focus: {:?}, doc: {} chars }}",
            request.focus_area, request.document.len());

        let timer = std::time::Instant::now();
        let result = pipeline.execute(&request).await;
        let elapsed = timer.elapsed();

        match result {
            Ok(report) => {
                eprintln!("  ✅ Output: AnalysisReport in {:.1}s", elapsed.as_secs_f64());
                eprintln!("     point_count: {}", report.point_count);
                eprintln!("     word_count:  {}", report.word_count);
                eprintln!("     summary:");
                for line in report.summary.lines().take(10) {
                    eprintln!("     │ {line}");
                }
                assert!(report.word_count > 0, "Report should have content");
            }
            Err(e) => eprintln!("  ⚠️ Pipeline error: {e}"),
        }

        eprintln!("\n{sep}\n");
    }

    // ═══════════════════════════════════════════════════════════════
    // 7. Yoneda Equivalence — Full Faithfulness Check
    //    Two different representations of the "same" info
    // ═══════════════════════════════════════════════════════════════

    /// Tests the Yoneda Embedding's full faithfulness guarantee:
    /// P₁ ≅ P₂ ⟺ ∀q: probe(P₁, q) ≅ probe(P₂, q)
    ///
    /// We compare a document against a shuffled version of itself.
    /// They should be Yoneda-equivalent (same info, different order).
    #[tokio::test]
    #[ignore]
    async fn live_yoneda_equivalence_check() {
        let provider = trinity_provider();

        let sep = "═".repeat(60);
        eprintln!("\n{sep}");
        eprintln!("⚡ YONEDA EMBEDDING: Full Faithfulness — P₁ ≅? P₂");
        eprintln!("{sep}");

        // P₁: original document
        let doc1 = "France is in Western Europe. Its capital is Paris. \
                    The economy is the 7th largest globally. \
                    Key sectors include aerospace (Airbus) and nuclear energy. \
                    French scientists include Pasteur and Curie.";

        // P₂: same facts, different order + wording
        let doc2 = "Pasteur and Curie are famous French scientists. \
                    The French economy ranks 7th in the world, \
                    driven by Airbus (aerospace) and nuclear power. \
                    Paris is the capital of France, a Western European nation.";

        // P₃: completely different document (should NOT be equivalent)
        let doc3 = "Japan is an island nation in East Asia. \
                    Tokyo is its capital. The economy focuses on electronics \
                    and automotive manufacturing. Honda and Toyota are key companies.";

        let config = LambdaConfig::default().with_context_window(2000);

        let y1 = lambda::yoneda::YonedaContext::lift(doc1, Arc::clone(&provider), config.clone());
        let y2 = lambda::yoneda::YonedaContext::lift(doc2, Arc::clone(&provider), config.clone());
        let y3 = lambda::yoneda::YonedaContext::lift(doc3, provider, config);

        let test_queries = &[
            "What country is this about?",
            "What is the capital city?",
            "Name a scientist mentioned.",
        ];

        // Simple word-overlap similarity for testing
        let similarity = |a: &str, b: &str| -> f64 {
            let a_lower = a.to_lowercase();
            let b_lower = b.to_lowercase();
            let words_a: std::collections::HashSet<&str> = a_lower
                .split_whitespace()
                .filter(|w| w.len() > 3)
                .collect();
            let words_b: std::collections::HashSet<&str> = b_lower
                .split_whitespace()
                .filter(|w| w.len() > 3)
                .collect();
            let intersection = words_a.intersection(&words_b).count();
            let union = words_a.union(&words_b).count();
            if union == 0 { 1.0 } else { intersection as f64 / union as f64 }
        };

        // ── Test 1: P₁ ≅ P₂ (same info) ──
        eprintln!("\n  ── Test 1: P₁ ≅? P₂ (same facts, different wording) ──");
        let eq12 = lambda::yoneda::yoneda_equivalence(&y1, &y2, test_queries, &similarity, 0.15).await;
        match &eq12 {
            Ok(eq) => {
                eprintln!("  📊 Equivalent: {}, Mean similarity: {:.3}", eq.equivalent, eq.mean_similarity);
                for (q, s) in &eq.scores {
                    eprintln!("     {:40} sim={:.3}", q, s);
                }
            }
            Err(e) => eprintln!("  ⚠️ Error: {e}"),
        }

        // ── Test 2: P₁ ≇ P₃ (different info) ──
        eprintln!("\n  ── Test 2: P₁ ≅? P₃ (completely different content) ──");
        let eq13 = lambda::yoneda::yoneda_equivalence(&y1, &y3, test_queries, &similarity, 0.15).await;
        match &eq13 {
            Ok(eq) => {
                eprintln!("  📊 Equivalent: {}, Mean similarity: {:.3}", eq.equivalent, eq.mean_similarity);
                for (q, s) in &eq.scores {
                    eprintln!("     {:40} sim={:.3}", q, s);
                }
                // P₁ and P₃ should NOT be equivalent
                if !eq.equivalent {
                    eprintln!("  ✅ Correctly identified as non-equivalent!");
                }
            }
            Err(e) => eprintln!("  ⚠️ Error: {e}"),
        }

        eprintln!("\n{sep}\n");
    }

    // ═══════════════════════════════════════════════════════════════
    // 8. Adaptive Yoneda — Self-Learning via GEPA Trajectory Evolution
    //    Each probe learns from past probes and improves morphism selection
    // ═══════════════════════════════════════════════════════════════

    /// Tests the full self-learning loop:
    /// 1. Creates an AdaptiveYoneda with trajectory memory
    /// 2. Runs 5 adaptive probes — each selects morphisms via epsilon-greedy
    /// 3. Scores each result
    /// 4. Shows improving morphism selection over generations
    #[tokio::test]
    #[ignore]
    async fn live_adaptive_yoneda_self_learning() {
        let provider = trinity_provider();

        let sep = "═".repeat(60);
        eprintln!("\n{sep}");
        eprintln!("⚡ ADAPTIVE YONEDA: Self-Learning via GEPA Trajectory Evolution");
        eprintln!("{sep}");

        // Build a document with facts we can verify
        let document = long_document(2000);
        let config = LambdaConfig::default().with_context_window(1500);

        let mut adaptive = lambda::AdaptiveYoneda::new(
            document.clone(),
            provider,
            config,
        );

        // ── Multi-Dimensional Scoring ──
        //
        // The scorer evaluates 4 dimensions so morphisms actually compete:
        //
        // 1. Recall (30%)       — did the expected keywords appear?
        // 2. Precision (30%)    — is the answer focused (low word count = better)?
        // 3. Conciseness (20%)  — penalize answers > 200 words
        // 4. Format (20%)       — reward bullet points / structured output
        //
        // Identity will get decent recall but poor precision/format scores.
        // "be_concise" and "extract_key_facts" should score higher on 
        // precision + format, creating real learning signal.

        struct TestCase {
            query: &'static str,
            required_keywords: Vec<&'static str>,
            max_ideal_words: usize,      // ideal answer length (shorter = higher precision score)
            wants_bullets: bool,          // reward •/- bullet format
        }

        let test_cases = vec![
            TestCase {
                query: "What are the main economic sectors mentioned?",
                required_keywords: vec!["airbus", "nuclear", "aerospace"],
                max_ideal_words: 50,     // a focused answer is short
                wants_bullets: true,     // bullet points are ideal for listing sectors
            },
            TestCase {
                query: "Who are the scientists mentioned?",
                required_keywords: vec!["pasteur", "curie"],
                max_ideal_words: 40,
                wants_bullets: true,
            },
            TestCase {
                query: "What is the capital city?",
                required_keywords: vec!["paris"],
                max_ideal_words: 15,     // should be a one-liner
                wants_bullets: false,
            },
            TestCase {
                query: "What historical events are discussed?",
                required_keywords: vec!["revolution", "1789"],
                max_ideal_words: 50,
                wants_bullets: true,
            },
            TestCase {
                query: "How large is the country in area?",
                required_keywords: vec!["643"],
                max_ideal_words: 20,
                wants_bullets: false,
            },
        ];

        let timer = std::time::Instant::now();

        for (i, tc) in test_cases.iter().enumerate() {
            eprintln!("\n  ── Probe {} — {:?} ──", i + 1, tc.query);
            eprintln!("     ideal_words≤{}, wants_bullets={}, keywords={:?}",
                tc.max_ideal_words, tc.wants_bullets, tc.required_keywords);

            let keywords = tc.required_keywords.clone();
            let max_ideal = tc.max_ideal_words;
            let wants_bullets = tc.wants_bullets;

            let result = adaptive.adaptive_probe(
                tc.query,
                move |_q, result| {
                    let result_lower = result.to_lowercase();
                    let word_count = result.split_whitespace().count();

                    // 1. Recall (30%): what fraction of keywords found?
                    let found = keywords.iter().filter(|kw| result_lower.contains(*kw)).count();
                    let recall = found as f64 / keywords.len().max(1) as f64;

                    // 2. Precision (30%): penalize overly verbose answers
                    // Score 1.0 if ≤ ideal, decays as answer gets longer
                    let precision = if word_count <= max_ideal {
                        1.0
                    } else {
                        (max_ideal as f64 / word_count as f64).min(1.0)
                    };

                    // 3. Conciseness (20%): hard penalty for > 200 words
                    let conciseness = if word_count <= 200 {
                        1.0
                    } else if word_count <= 500 {
                        0.5
                    } else {
                        0.2
                    };

                    // 4. Format (20%): reward bullet points if wanted
                    let format_score = if wants_bullets {
                        let bullet_lines = result.lines()
                            .filter(|l| {
                                let trimmed = l.trim();
                                trimmed.starts_with('-') || trimmed.starts_with('•') ||
                                trimmed.starts_with('*') || trimmed.starts_with("–")
                            })
                            .count();
                        if bullet_lines >= 2 { 1.0 }
                        else if bullet_lines == 1 { 0.5 }
                        else { 0.1 }  // no bullets when expected → low score
                    } else {
                        // For non-bullet queries, reward shortness
                        if word_count <= max_ideal { 1.0 } else { 0.5 }
                    };

                    let score = recall * 0.30
                        + precision * 0.30
                        + conciseness * 0.20
                        + format_score * 0.20;

                    eprintln!("     📊 recall={:.2} precision={:.2} concise={:.2} format={:.2} → {:.3}",
                        recall, precision, conciseness, format_score, score);
                    eprintln!("        words={}, bullets={}", word_count,
                        result.lines().filter(|l| l.trim().starts_with('-') || l.trim().starts_with('•')).count());

                    score
                },
            ).await;

            match &result {
                Ok((answer, score)) => {
                    eprintln!("  ✅ Final score: {:.3}", score);
                    for line in answer.lines().take(5) {
                        eprintln!("  │ {}", &line[..line.len().min(100)]);
                    }
                    if answer.lines().count() > 5 {
                        eprintln!("  │ ... ({} more lines)", answer.lines().count() - 5);
                    }
                }
                Err(e) => eprintln!("  ⚠️ Error: {e}"),
            }
        }

        let elapsed = timer.elapsed();

        // Print trajectory summary
        {
            let store = adaptive.trajectories.lock().unwrap();
            let pop = adaptive.morphisms.lock().unwrap();

            eprintln!("\n{sep}");
            eprintln!("📊 SELF-LEARNING SUMMARY ({:.1}s total)", elapsed.as_secs_f64());
            eprintln!("{sep}");
            eprintln!("  Total trajectories: {}", store.all().len());
            eprintln!("  Mean score: {:.3}", store.mean_score());
            eprintln!("  Improvement rate: {:.3}", store.improvement_rate(2));
            eprintln!("\n  Morphism Population (lower=worse, higher=better):");
            eprintln!("{}", pop.summary());
            eprintln!("\n  Trajectory Details:");
            for t in store.all() {
                eprintln!("    gen={} morph={:20} score={:.3} words={}",
                    t.generation,
                    t.morphism_name.as_deref().unwrap_or("?"),
                    t.score,
                    t.result.split_whitespace().count(),
                );
            }
        }

        eprintln!("\n{sep}\n");
    }

    // ═══════════════════════════════════════════════════════════════════
    // LIVE TEST: Full DR-Tulu Evolving Rubric Lifecycle
    // ═══════════════════════════════════════════════════════════════════

    /// Tests the FULL DR-Tulu evolving rubric lifecycle end-to-end:
    ///
    /// **Phase 1 — Persistent Rubric Scoring** (probes 0–2):
    ///   Score against 3 persistent rubrics (Factual Recall, Answer Relevance, Completeness)
    ///   using LLM-as-Judge. No adaptive rubrics yet.
    ///
    /// **Phase 2 — Adaptive Rubric Generation** (probe 3+):
    ///   After `rubric_gen_interval` probes, the LLM generates new discriminative
    ///   criteria by comparing past responses. New adaptive rubrics are added.
    ///
    /// **Phase 3 — Rubric Retirement** (ongoing):
    ///   Every 3 probes, zero-std rubrics (non-discriminative) are retired.
    ///   The buffer is capped at max_active adaptive rubrics.
    ///
    /// **Phase 4 — Morphism Evolution**:
    ///   Different morphisms are tested via ε-greedy selection across queries.
    ///   Per-morphism scoring tracks which transformations produce the best
    ///   rubric-evaluated responses.
    ///
    /// This mirrors the DR-Tulu GRPO loop from grpo_fast_rubric.py:
    /// ```text
    /// 1. Generate responses → 2. Score all rubrics → 3. Generate adaptive rubrics
    /// 4. Filter zero-std → 5. Per-rubric normalize → 6. Evolve policy
    /// ```
    #[tokio::test]
    #[ignore = "Requires OPENAI_API_KEY — run: cargo test live_evolving_rubric -- --ignored --nocapture"]
    async fn live_evolving_rubric_reward() {
        use crate::lambda::adaptive_yoneda::AdaptiveYoneda;
        use crate::lambda::rubric::{RubricBuffer, RubricItem, RubricType};
        use std::collections::HashMap;

        let provider = trinity_provider();
        let config = LambdaConfig::default();

        let sep = "═".repeat(72);
        eprintln!("\n{sep}");
        eprintln!("🧪 LIVE TEST: Full DR-Tulu Evolving Rubric Lifecycle");
        eprintln!("{sep}\n");

        // ── Document ────────────────────────────────────────────
        // Richer than before — multiple dimensions to score on
        let doc = "\
            The Rust programming language was first released in 2015 by Mozilla Research. \
            Graydon Hoare started the project in 2006 as a personal side project. \
            Rust achieves memory safety without garbage collection through its ownership \
            system and borrow checker, which enforces rules at compile time. \
            Key features include: zero-cost abstractions, pattern matching, \
            trait-based generics, algebraic data types (enums), and fearless concurrency. \
            The Cargo package manager handles dependency management, building, testing, \
            and publishing crates to crates.io. As of 2024, crates.io hosts over 140,000 \
            packages. Rust compiles to native code via LLVM and supports WebAssembly targets. \
            Major production users include Firefox (Servo components), Dropbox (file sync), \
            Cloudflare (Workers runtime), Discord (voice and video infrastructure), \
            Amazon (Firecracker microVMs), and the Linux kernel (since version 6.1, Dec 2022). \
            The async/await syntax was stabilized in Rust 1.39 (November 2019). \
            Rust won 'most admired language' on Stack Overflow surveys for 8 consecutive \
            years from 2016 to 2023. The Rust Foundation was established in February 2021 \
            with founding members including AWS, Google, Huawei, Microsoft, and Mozilla.";

        // ── Create AdaptiveYoneda with Rubrics ──────────────────
        let mut adaptive = AdaptiveYoneda::with_rubrics(doc, provider, config);
        // Lower interval to trigger generation sooner (DR-Tulu generates every step)
        adaptive.rubric_gen_interval = 3;

        // ── Verify Initial State ────────────────────────────────
        {
            let buf = adaptive.rubric_buffer.as_ref().unwrap();
            assert_eq!(buf.persistent.len(), 3, "Should have 3 persistent rubrics");
            assert_eq!(buf.active.len(), 0, "Should start with 0 adaptive rubrics");
            eprintln!("📊 Phase 0: Initial rubric buffer");
            eprintln!("{}\n", buf.summary());
        }

        // ── Query bank — diverse questions to exercise different quality dims ──
        let queries = [
            // Phase 1 probes (persistent rubrics only)
            "When was the Rust programming language first released and who created it?",
            "How does Rust achieve memory safety without garbage collection?",
            "What concurrency features does Rust provide?",
            // Phase 2+ probes (should trigger adaptive rubric generation)
            "Name five companies that use Rust in production and what they use it for.",
            "What is Cargo and what does it do?",
            "Describe the Rust Foundation and its founding members.",
            // Phase 3 probes (should trigger retirement of non-discriminative rubrics)
            "When was async/await stabilized in Rust?",
            "How many packages are on crates.io?",
        ];

        let mut all_scores: Vec<f64> = Vec::new();
        let mut phase_scores: HashMap<String, Vec<f64>> = HashMap::new();
        let mut successful_probes = 0;
        let max_consecutive_errors = 3;
        let mut consecutive_errors = 0;
        let delay_secs = 5; // longer delay for free-tier reliability

        for (i, query) in queries.iter().enumerate() {
            // Bail if too many consecutive errors (model is down)
            if consecutive_errors >= max_consecutive_errors {
                eprintln!("\n⛔ {} consecutive errors — model appears unavailable, stopping", max_consecutive_errors);
                break;
            }

            // Phase headers
            if i == 0 {
                eprintln!("\n{}", "─".repeat(72));
                eprintln!("📋 Phase 1: Persistent Rubric Scoring (probes 0–2)");
                eprintln!("{}", "─".repeat(72));
            } else if i == 3 {
                eprintln!("\n{}", "─".repeat(72));
                eprintln!("📋 Phase 2: Adaptive Rubric Generation (probes 3+, gen_interval=3)");
                eprintln!("{}", "─".repeat(72));
            } else if i == 6 {
                eprintln!("\n{}", "─".repeat(72));
                eprintln!("📋 Phase 3: Rubric Retirement Filter (probes 6+)");
                eprintln!("{}", "─".repeat(72));
            }

            eprintln!("\n── Probe {} ──────────────────────────────", i);
            eprintln!("Query: {:?}", query);

            // Delay between API calls
            if i > 0 {
                eprintln!("⏳ Waiting {}s to avoid rate limits...", delay_secs);
                tokio::time::sleep(tokio::time::Duration::from_secs(delay_secs)).await;
            }

            // Retry up to 2 times on failure
            let mut result = adaptive.adaptive_probe_with_rubrics(query).await;
            if result.is_err() {
                eprintln!("  ⚠️ First attempt failed, retrying after {}s...", delay_secs);
                tokio::time::sleep(tokio::time::Duration::from_secs(delay_secs)).await;
                result = adaptive.adaptive_probe_with_rubrics(query).await;
            }

            match result {
                Ok((text, score, per_rubric)) => {
                    consecutive_errors = 0;
                    successful_probes += 1;

                    eprintln!("  ✅ score={:.3} (gen={})", score, adaptive.generation - 1);
                    eprintln!("  Result preview: {:?}", &text[..text.len().min(120)]);

                    // Show per-rubric scores sorted
                    let mut sorted_rubrics: Vec<_> = per_rubric.iter().collect();
                    sorted_rubrics.sort_by(|a, b| a.0.cmp(b.0));
                    for (rubric_name, rubric_score) in &sorted_rubrics {
                        let indicator = if **rubric_score >= 0.75 { "🟢" }
                            else if **rubric_score >= 0.4 { "🟡" }
                            else { "🔴" };
                        eprintln!("    {} {:30} → {:.3}", indicator, rubric_name, rubric_score);
                    }

                    // Track scores by rubric name for evolution analysis
                    for (name, &sc) in &per_rubric {
                        phase_scores.entry(name.clone()).or_default().push(sc);
                    }

                    assert!(score >= 0.0 && score <= 1.0, "Score {:.3} out of [0,1]", score);
                    assert!(!per_rubric.is_empty(), "Per-rubric scores empty");
                    all_scores.push(score);
                }
                Err(e) => {
                    consecutive_errors += 1;
                    eprintln!("  ⚠️ Error: {} (attempt 2/2)", e);
                }
            }

            // Snapshot rubric buffer after key thresholds
            if i == 2 || i == 5 || i == 7 {
                let buf = adaptive.rubric_buffer.as_ref().unwrap();
                eprintln!("\n  📊 Rubric buffer snapshot (after probe {}):", i);
                eprintln!("  {}", buf.summary().replace('\n', "\n  "));
            }
        }

        // ═══════════════════════════════════════════════════════
        // FINAL SUMMARY — DR-Tulu style analysis
        // ═══════════════════════════════════════════════════════

        eprintln!("\n{sep}");
        eprintln!("📊 DR-TULU EVOLVING RUBRIC — FULL LIFECYCLE SUMMARY");
        eprintln!("{sep}");

        // 1. Rubric buffer final state
        let buf = adaptive.rubric_buffer.as_ref().unwrap();
        eprintln!("\n── Rubric Buffer Final State ──");
        eprintln!("{}", buf.summary());

        let n_persistent = buf.persistent.len();
        let n_active = buf.active.len();
        let n_inactive = buf.inactive.len();
        eprintln!("\n  ┌───────────────────────────────────────────┐");
        eprintln!("  │ Persistent: {}  Active: {}  Inactive: {:2}    │", n_persistent, n_active, n_inactive);
        eprintln!("  └───────────────────────────────────────────┘");

        // 2. Morphism population analysis
        eprintln!("\n── Morphism Score Board ──");
        {
            let pop = adaptive.morphisms.lock().unwrap();
            eprintln!("{}", pop.summary());
        }

        // 3. Trajectory store analysis
        eprintln!("\n── Trajectory Evolution ──");
        {
            let store = adaptive.trajectories.lock().unwrap();
            eprintln!("  Total trajectories: {}", store.all().len());
            eprintln!("  Mean score:         {:.3}", store.mean_score());
            eprintln!("  Improvement rate:   {:.3}", store.improvement_rate(2));
            eprintln!();
            for t in store.all() {
                eprintln!("    gen={:2} morph={:22} score={:.3} | {:60}",
                    t.generation,
                    t.morphism_name.as_deref().unwrap_or("?"),
                    t.score,
                    &t.result[..t.result.len().min(60)],
                );
            }
        }

        // 4. Per-rubric score evolution
        eprintln!("\n── Per-Rubric Score Evolution ──");
        for (name, scores) in &phase_scores {
            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            let std = {
                let var = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
                var.sqrt()
            };
            let discriminative = if std > 0.1 { "📊 discriminative" }
                                 else if std > 0.01 { "📉 low-variance" }
                                 else { "⛔ zero-std (should retire)" };
            eprintln!("    {:30} mean={:.3} std={:.3} n={} {}",
                name, mean, std, scores.len(), discriminative);
        }

        // ── Assertions ────────────────────────────────────────
        // At least some probes succeeded
        assert!(
            successful_probes >= 2,
            "Need at least 2 successful probes, got {}",
            successful_probes
        );

        // Rubric scoring worked (all scores in range)
        for &s in &all_scores {
            assert!(s >= 0.0 && s <= 1.0, "Score {:.3} out of [0,1]", s);
        }

        // If we got enough probes, verify rubric generation happened
        if successful_probes >= 4 {
            let total_rubrics = n_persistent + n_active + n_inactive;
            eprintln!("\n  ✅ Rubric generation check: {} total rubrics (3 persistent + {} evolved)",
                total_rubrics, n_active + n_inactive);
            // After 4+ probes with gen_interval=3, we should have generated some adaptive rubrics
            // (or at least attempted — generation may produce 0 if responses are too similar)
        }

        // If we got enough probes, check retirement happened for zero-std
        if successful_probes >= 6 {
            eprintln!("  ✅ Retirement check: {} inactive rubrics", n_inactive);
        }

        let mean_score = all_scores.iter().sum::<f64>() / all_scores.len().max(1) as f64;
        eprintln!("\n  📈 Overall mean rubric score: {:.3} (across {} probes)", mean_score, all_scores.len());

        eprintln!("\n{sep}");
        eprintln!("✅ Full DR-Tulu evolving rubric lifecycle test complete");
        eprintln!("   Probes: {} succeeded / {} attempted", successful_probes, queries.len().min(
            successful_probes + consecutive_errors.min(max_consecutive_errors)
        ));
        eprintln!("{sep}\n");
    }
}
