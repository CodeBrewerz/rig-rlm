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
}
