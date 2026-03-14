//! Live LLM integration tests using OpenRouter + Trinity model.
//!
//! These tests make real API calls and are marked `#[ignore]` so they
//! don't run during `cargo test`. Run them explicitly:
//!
//!     cargo test live_llm -- --ignored --nocapture
//!
//! They require OPENAI_API_KEY / OPENAI_BASE_URL env vars
//! (loaded from .env via dotenvy).

#[cfg(test)]
mod live_llm {
    use crate::monad::{
        AgentConfig, AgentContext, AgentMonad,
        interaction::agent_task_full,
        recipe::{Recipe, StepKind, StepStatus},
    };
    use crate::persistence::{AgentStore, Session, Turn};
    use chrono::Utc;

    /// Build an AgentContext configured for live OpenRouter calls.
    fn live_context(max_turns: usize) -> AgentContext {
        dotenvy::dotenv().ok();

        // Initialize OTEL + LangFuse (no-op if already initialized or keys missing)
        let _ = crate::monad::otel::init_tracing();

        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
        let base_url = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());
        let model = std::env::var("RIG_RLM_MODEL")
            .unwrap_or_else(|_| "arcee-ai/trinity-large-preview:free".to_string());

        let provider = crate::monad::provider::ProviderConfig::openai_compatible(
            "openrouter",
            &model,
            &base_url,
            &api_key,
        );

        let config = AgentConfig {
            provider,
            max_turns,
            ..AgentConfig::default()
        };

        let mut ctx = AgentContext::new(config);

        // Enrich with LangFuse trace-level attributes
        ctx.trace_ctx = crate::monad::otel::TraceContext::new()
            .with_user("test-user")
            .with_session(&format!(
                "test-session-{}",
                chrono::Utc::now().format("%Y%m%d-%H%M%S")
            ))
            .with_name("rig-rlm-live-test")
            .with_environment("development")
            .with_release(env!("CARGO_PKG_VERSION"))
            .with_tags(vec!["live-test".to_string(), "trinity".to_string()])
            .with_metadata("test_framework", "cargo-test")
            .with_metadata("model_provider", "openrouter");

        ctx
    }

    // ─── 1. Single-turn LLM call ──────────────────────────────────

    #[tokio::test]
    #[ignore]
    async fn live_single_task_with_trinity() {
        let mut ctx = live_context(10);

        let program = agent_task_full(
            "What is 2 + 2? Reply with just the number.",
            None,
            None,
            None,
            vec![],
        );
        let result = ctx.run(program).await;
        match &result {
            Ok(answer) => {
                eprintln!("\n✅ Trinity answered: {answer}");
                assert!(
                    answer.contains("4"),
                    "Trinity should know 2+2=4, got: {answer}"
                );
            }
            Err(e) => eprintln!("\n⚠️ LLM error (may be rate limited): {e}"),
        }

        assert!(
            !ctx.evidence().is_empty(),
            "LLM inference should produce evidence"
        );
        eprintln!("📊 Evidence entries: {}", ctx.evidence().len());
        eprintln!("💬 History turns: {}", ctx.history.len());
    }

    // ─── 2. Think + LLM task ──────────────────────────────────────

    #[tokio::test]
    #[ignore]
    async fn live_think_then_llm_task() {
        let mut ctx = live_context(10);

        let think = AgentMonad::think("The user wants a simple answer. I'll respond concisely.");
        ctx.run(think).await.unwrap();
        assert!(!ctx.evidence().is_empty(), "think should record evidence");
        assert_eq!(ctx.history.len(), 0, "think should not add to history");

        let program = agent_task_full(
            "What is the capital of France? Reply with just the city name.",
            None,
            None,
            None,
            vec![],
        );
        let result = ctx.run(program).await;
        match &result {
            Ok(answer) => {
                eprintln!("\n✅ Trinity answered: {answer}");
                assert!(
                    answer.to_lowercase().contains("paris"),
                    "should say Paris, got: {answer}"
                );
            }
            Err(e) => eprintln!("\n⚠️ LLM error: {e}"),
        }
    }

    // ─── 3. Context + LLM task ────────────────────────────────────

    #[tokio::test]
    #[ignore]
    async fn live_context_load_then_llm_query() {
        let mut ctx = live_context(10);

        let data = "Product,Price\nWidget A,10.99\nWidget B,24.50\nWidget C,7.25\n";
        let load = AgentMonad::load_context("products", data);
        ctx.run(load).await.unwrap();

        let search = AgentMonad::search_context("products", "Widget B");
        let search_result = ctx.run(search).await.unwrap();
        eprintln!("🔍 Search result: {search_result}");
        assert!(
            search_result.contains("24.50"),
            "should find Widget B price"
        );

        let program = agent_task_full(
            "Given this product data: Widget A=$10.99, Widget B=$24.50, Widget C=$7.25. Which product is the cheapest? Reply with just the product name.",
            None,
            None,
            None,
            vec![],
        );
        let result = ctx.run(program).await;
        match &result {
            Ok(answer) => {
                eprintln!("\n✅ Trinity answered: {answer}");
                assert!(
                    answer.contains("Widget C")
                        || answer.contains("widget c")
                        || answer.contains("C"),
                    "cheapest is Widget C, got: {answer}"
                );
            }
            Err(e) => eprintln!("\n⚠️ LLM error: {e}"),
        }
    }

    // ─── 4. Multi-step Recipe with LLM ────────────────────────────

    #[tokio::test]
    #[ignore]
    async fn live_recipe_pipeline_with_trinity() {
        let mut ctx = live_context(15);

        let recipe = Recipe::new("Live Pipeline Test")
            .description("Real LLM-powered multi-step pipeline")
            .step("generate", "Generate a list of exactly 3 fruits. Reply with just the fruit names separated by commas.", StepKind::TextGen)
            .step_with_deps("count", "Count the items in this list: {{generate.output}}. Reply with just the number.", StepKind::TextGen, vec!["generate"]);

        let estimate = recipe.estimate_cost();
        eprintln!(
            "\n🍳 Recipe: {} steps, ~{} LLM calls",
            estimate.total_steps, estimate.estimated_llm_calls
        );

        let result = ctx.run_recipe(recipe).await.unwrap();
        result.print_summary();

        let gen_step = result.steps.get("generate").unwrap();
        match &gen_step.status {
            StepStatus::Completed => {
                eprintln!("✅ Generate output: {}", gen_step.output);
                assert!(!gen_step.output.is_empty(), "should have generated fruits");
            }
            StepStatus::Failed(e) => eprintln!("⚠️ Generate failed: {e}"),
            _ => {}
        }

        let count_step = result.steps.get("count").unwrap();
        match &count_step.status {
            StepStatus::Completed => {
                eprintln!("✅ Count output: {}", count_step.output);
                assert!(
                    count_step.output.contains("3"),
                    "should count 3, got: {}",
                    count_step.output
                );
            }
            StepStatus::Skipped => eprintln!("⏭️ Count skipped (generate failed)"),
            StepStatus::Failed(e) => eprintln!("⚠️ Count failed: {e}"),
        }

        eprintln!(
            "\n📊 Total turns: {}, Elapsed: {:.1}s",
            result.total_turns,
            result.elapsed.as_secs_f64()
        );

        // Flush all OTEL spans to LangFuse before test exits
        crate::monad::otel::shutdown_tracing().await;
    }

    // ─── 5. PlanRecipe action with LLM ────────────────────────────

    #[tokio::test]
    #[ignore]
    async fn live_plan_recipe_action() {
        let mut ctx = live_context(15);

        let yaml = r#"
name: Agent-Planned Analysis
steps:
  - id: brainstorm
    task: "List 3 programming languages good for data science. Reply with just the names separated by commas."
    kind: text_gen
  - id: pick
    task: "From this list: {{brainstorm.output}} — pick the most popular one. Reply with just the language name."
    kind: text_gen
    depends_on: [brainstorm]
"#;

        let m = AgentMonad::plan_recipe(yaml);
        let result = ctx.run(m).await;
        match &result {
            Ok(summary) => {
                eprintln!("\n✅ PlanRecipe summary:\n{summary}");
                assert!(
                    summary.contains("Agent-Planned Analysis"),
                    "summary should include recipe name"
                );
            }
            Err(e) => eprintln!("\n⚠️ PlanRecipe error: {e}"),
        }
        eprintln!("📊 Evidence entries: {}", ctx.evidence().len());
    }

    // ─── 6. Full E2E: LLM + Turso persistence ─────────────────────

    #[tokio::test]
    #[ignore]
    async fn live_full_e2e_with_turso() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("live_e2e.db");
        let store = AgentStore::open(db_path.to_str().unwrap()).await.unwrap();

        let mut ctx = live_context(10);

        let program = agent_task_full(
            "What programming language is Rust most similar to? Reply in one word.",
            None,
            None,
            None,
            vec![],
        );
        let result = ctx.run(program).await;
        let answer = match result {
            Ok(a) => {
                let s = a.into_completed();
                eprintln!("✅ Answer: {s}");
                s
            }
            Err(e) => {
                eprintln!("⚠️ LLM error: {e}");
                format!("error: {e}")
            }
        };

        let session_id = uuid::Uuid::new_v4().to_string();
        store
            .create_session(&Session {
                session_id: session_id.clone(),
                model: "arcee-ai/trinity-large-preview:free".to_string(),
                task: "Live E2E Test".to_string(),
                executor: "pyo3".to_string(),
                optimizer: None,
                optimized_instruction: None,
                started_at: Utc::now().to_rfc3339(),
                finished_at: None,
                final_answer: None,
                score: None,
            })
            .await
            .unwrap();

        for (i, msg) in ctx.history.messages().iter().enumerate() {
            store
                .record_turn(&Turn {
                    session_id: session_id.clone(),
                    turn_num: i as i32,
                    role: format!("{:?}", msg.role),
                    content: msg.content.to_string(),
                    code: None,
                    exec_stdout: None,
                    exec_stderr: None,
                    exec_return: None,
                    timestamp_ms: Utc::now().timestamp_millis(),
                })
                .await
                .unwrap();
        }

        store
            .finish_session(&session_id, Some(&answer), None)
            .await
            .unwrap();

        let summary = store.session_summary(&session_id).await.unwrap();
        eprintln!("📦 Turso summary: {summary}");
        assert!(summary.contains("turns"), "should have turns: {summary}");
    }

    // ─── 7. Turso Audit: Full DB dump ─────────────────────────────

    #[tokio::test]
    #[ignore]
    async fn turso_audit_full_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("audit.db");
        let store = AgentStore::open(db_path.to_str().unwrap()).await.unwrap();

        let mut ctx = live_context(15);

        let sep = "=".repeat(60);
        eprintln!("\n{sep}");
        eprintln!("🔍 TURSO AUDIT: Starting full persistence test");
        eprintln!("{sep}");

        // 1. Run a real LLM task
        let program = agent_task_full(
            "Name the 3 primary colors. Reply with just the color names separated by commas.",
            None,
            None,
            None,
            vec![],
        );
        let result = ctx.run(program).await;
        let answer = match result {
            Ok(a) => {
                let s = a.into_completed();
                eprintln!("✅ LLM answer: {s}");
                s
            }
            Err(e) => {
                eprintln!("⚠️ LLM error: {e}");
                format!("error: {e}")
            }
        };

        // 2. Run a recipe pipeline
        let recipe = Recipe::new("Audit Pipeline")
            .step(
                "list",
                "List 3 animals. Reply with just the names separated by commas.",
                StepKind::TextGen,
            )
            .step_with_deps(
                "count",
                "Count items in: {{list.output}}. Reply with just the number.",
                StepKind::TextGen,
                vec!["list"],
            );

        let recipe_result = ctx.run_recipe(recipe).await.unwrap();

        // 3. Persist to Turso
        let session_id = uuid::Uuid::new_v4().to_string();
        store
            .create_session(&Session {
                session_id: session_id.clone(),
                model: "arcee-ai/trinity-large-preview:free".to_string(),
                task: "Turso Audit Test".to_string(),
                executor: "pyo3".to_string(),
                optimizer: None,
                optimized_instruction: None,
                started_at: Utc::now().to_rfc3339(),
                finished_at: None,
                final_answer: None,
                score: None,
            })
            .await
            .unwrap();

        for (i, msg) in ctx.history.messages().iter().enumerate() {
            store
                .record_turn(&Turn {
                    session_id: session_id.clone(),
                    turn_num: i as i32,
                    role: format!("{:?}", msg.role),
                    content: msg.content.to_string(),
                    code: None,
                    exec_stdout: None,
                    exec_stderr: None,
                    exec_return: None,
                    timestamp_ms: Utc::now().timestamp_millis(),
                })
                .await
                .unwrap();
        }

        let mut turn_offset = ctx.history.len() as i32;

        for (step_id, step_result) in &recipe_result.steps {
            let status = match &step_result.status {
                StepStatus::Completed => "completed",
                StepStatus::Failed(_) => "failed",
                StepStatus::Skipped => "skipped",
            };
            store
                .record_turn(&Turn {
                    session_id: session_id.clone(),
                    turn_num: turn_offset,
                    role: format!("RecipeStep:{status}"),
                    content: format!(
                        "step={}, turns={}, elapsed={:.1}s, output={}",
                        step_id,
                        step_result.turns,
                        step_result.elapsed.as_secs_f64(),
                        &step_result.output
                    ),
                    code: None,
                    exec_stdout: None,
                    exec_stderr: None,
                    exec_return: None,
                    timestamp_ms: Utc::now().timestamp_millis(),
                })
                .await
                .unwrap();
            turn_offset += 1;
        }

        store
            .record_turn(&Turn {
                session_id: session_id.clone(),
                turn_num: turn_offset,
                role: "EvidenceSummary".to_string(),
                content: ctx.evidence_summary(),
                code: None,
                exec_stdout: None,
                exec_stderr: None,
                exec_return: None,
                timestamp_ms: Utc::now().timestamp_millis(),
            })
            .await
            .unwrap();

        store
            .finish_session(&session_id, Some(&answer), Some(1.0))
            .await
            .unwrap();

        // 4. Query raw DB and dump everything
        eprintln!("\n{sep}");
        eprintln!("📊 TURSO AUDIT: Querying raw database");
        eprintln!("{sep}");

        let db = turso::Builder::new_local(db_path.to_str().unwrap())
            .build()
            .await
            .unwrap();
        let conn = db.connect().unwrap();

        // Sessions
        let mut rows = conn.query(
            "SELECT session_id, model, task, executor, started_at, finished_at, final_answer, score FROM sessions",
            (),
        ).await.unwrap();

        let line = "─".repeat(70);
        eprintln!("\n📋 SESSIONS:");
        eprintln!("{line}");
        while let Some(row) = rows.next().await.unwrap() {
            let sid: String = row.get(0).unwrap();
            let model: String = row.get(1).unwrap();
            let task: String = row.get(2).unwrap();
            let executor: String = row.get(3).unwrap();
            let started: String = row.get(4).unwrap();
            let finished: String = row.get(5).unwrap();
            let final_ans: String = row.get(6).unwrap();
            let score: f64 = row.get(7).unwrap();
            eprintln!("  Session:  {}", &sid[..8]);
            eprintln!("  Model:    {model}");
            eprintln!("  Task:     {task}");
            eprintln!("  Executor: {executor}");
            eprintln!("  Started:  {started}");
            eprintln!("  Finished: {finished}");
            eprintln!("  Answer:   {final_ans}");
            eprintln!("  Score:    {score}");
        }

        // Turns
        let mut rows = conn
            .query(
                "SELECT turn_num, role, content FROM turns WHERE session_id = ?1 ORDER BY turn_num",
                (session_id.as_str(),),
            )
            .await
            .unwrap();

        eprintln!("\n💬 TURNS:");
        eprintln!("{line}");
        let mut turn_count = 0;
        while let Some(row) = rows.next().await.unwrap() {
            let num: i32 = row.get(0).unwrap();
            let role: String = row.get(1).unwrap();
            let content: String = row.get(2).unwrap();
            let preview = if content.len() > 120 {
                format!("{}...", &content[..120])
            } else {
                content.clone()
            };
            eprintln!("  Turn {num:2} [{role:20}] {preview}");
            turn_count += 1;
        }
        eprintln!("{line}");
        eprintln!("  Total turns in DB: {turn_count}");

        let summary = store.session_summary(&session_id).await.unwrap();
        eprintln!("\n📦 SESSION SUMMARY: {summary}");
        eprintln!("{sep}\n");

        assert!(turn_count > 0, "should have persisted turns");
        assert!(
            summary.contains("turns"),
            "summary should report turns: {summary}"
        );
    }

    // ─── 8. Subagent Recursion Test ───────────────────────────────

    #[tokio::test]
    #[ignore]
    async fn live_subagent_recursion() {
        use crate::monad::action::Action;
        use crate::monad::capabilities::Capabilities;

        let mut ctx = live_context(15);

        let sep = "=".repeat(60);
        eprintln!("\n{sep}");
        eprintln!("🧠 SUBAGENT RECURSION TEST");
        eprintln!("  Parent agent spawns child with sub-task");
        eprintln!("{sep}");

        // Step 1: parent does initial analysis
        let program = agent_task_full(
            "What are the 3 branches of the US government? Reply with just the branch names separated by commas.",
            None,
            None,
            None,
            vec![],
        );
        let parent_result = ctx.run(program).await;
        match &parent_result {
            Ok(answer) => eprintln!("✅ Parent answer: {answer}"),
            Err(e) => eprintln!("⚠️ Parent error: {e}"),
        }
        let parent_evidence_before = ctx.evidence().len();
        eprintln!("📊 Parent evidence so far: {parent_evidence_before}");

        // Step 2: Spawn a sub-agent for deeper analysis (recursive!)
        eprintln!("\n🔀 Spawning sub-agent for deeper analysis...");
        let sub_program = AgentMonad::perform(
            Action::SpawnSubAgent {
                task: "What is the role of the judicial branch in the US government? Reply in one sentence.".to_string(),
                capabilities: Capabilities::default(),
            },
            |output| AgentMonad::Pure(output.into_string()),
        );

        let sub_result = ctx.run(sub_program).await;
        match &sub_result {
            Ok(text) => {
                eprintln!("✅ Sub-agent answer: {text}");
                assert!(!text.is_empty(), "sub-agent should return non-empty result");
            }
            Err(e) => eprintln!("⚠️ Sub-agent error: {e}"),
        }

        // Verify evidence grew (sub-agent result recorded in parent)
        let parent_evidence_after = ctx.evidence().len();
        eprintln!("📊 Parent evidence after sub-agent: {parent_evidence_after}");
        assert!(
            parent_evidence_after > parent_evidence_before,
            "sub-agent should add evidence to parent: {} -> {}",
            parent_evidence_before,
            parent_evidence_after
        );

        // Step 3: Recipe pipeline for full recursive composition
        eprintln!("\n🍳 Running recipe pipeline for recursive composition...");
        let recipe = Recipe::new("Recursive Analysis")
            .step("detail", "What checks does the judicial branch have on the executive branch? Reply in one sentence.", StepKind::TextGen)
            .step_with_deps("summarize", "Summarize this in 5 words: {{detail.output}}", StepKind::TextGen, vec!["detail"]);

        let recipe_result = ctx.run_recipe(recipe).await.unwrap();
        recipe_result.print_summary();

        let final_evidence = ctx.evidence().len();
        eprintln!("\n📊 RECURSION SUMMARY:");
        eprintln!("  Parent task evidence:  {parent_evidence_before}");
        eprintln!("  After sub-agent:       {parent_evidence_after}");
        eprintln!("  After recipe pipeline: {final_evidence}");
        eprintln!("  Total evidence chain:  {final_evidence} entries");
        eprintln!("{sep}\n");
    }
}
