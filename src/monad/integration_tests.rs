//! End-to-end integration tests for all new capabilities.
//!
//! Tests exercise the full action → interpret_action → side effects chain
//! without requiring an LLM. Turso persistence is tested with temp DBs.
//!
//! Coverage:
//! - Evidence tracking (Phase 1)
//! - Context management: load, search, peek, list (Phase 2/3)
//! - Auto-compaction (Phase 5)
//! - Think & EvaluateProgress (Phase 7)
//! - Recipe DSL: validation, execution order, templates (Phase 8)
//! - PlanRecipe action (agent-defined pipelines)
//! - Turso persistence: sessions, turns, summary

#[cfg(test)]
mod integration_tests {
    use crate::monad::{ActionOutput, AgentConfig, AgentContext, AgentMonad, LogLevel, Role};
    use crate::persistence::{AgentStore, Session, Turn};
    use chrono::Utc;

    // ─── Helper ──────────────────────────────────────────────────

    fn test_context() -> AgentContext {
        AgentContext::new(AgentConfig {
            max_turns: 200,
            ..AgentConfig::default()
        })
    }

    // ─── Phase 1: Evidence Tracking ──────────────────────────────

    #[tokio::test]
    async fn evidence_recorded_on_think() {
        let m = AgentMonad::think("reasoning about the task").then(AgentMonad::pure("done"));
        let mut ctx = test_context();
        ctx.run(m).await.unwrap();

        // Think action records evidence
        assert!(
            !ctx.evidence().is_empty(),
            "evidence should be recorded for Think"
        );
    }

    #[tokio::test]
    async fn evidence_summary_is_human_readable() {
        let m = AgentMonad::insert(Role::System, "system prompt")
            .then(AgentMonad::insert(Role::User, "user query"))
            .then(AgentMonad::capture("key", "value"))
            .then(AgentMonad::log(LogLevel::Info, "test log"))
            .then(AgentMonad::pure("done"));

        let mut ctx = test_context();
        ctx.run(m).await.unwrap();

        let summary = ctx.evidence_summary();
        assert!(!summary.is_empty(), "evidence summary should not be empty");
    }

    // ─── Phase 2/3: Context Management ───────────────────────────

    #[tokio::test]
    async fn load_and_search_context() {
        let content = "line 1: hello world\nline 2: foo bar\nline 3: hello agent\n";
        let m = AgentMonad::load_context("test_ctx", content)
            .then(AgentMonad::search_context("test_ctx", "hello"))
            .bind(|result| AgentMonad::pure(result));

        let mut ctx = test_context();
        let result = ctx.run(m).await.unwrap().into_completed();

        assert!(
            result.contains("hello"),
            "search should find 'hello' matches"
        );
        assert!(result.contains("line 1"), "should include line 1");
        assert!(result.contains("line 3"), "should include line 3");
    }

    #[tokio::test]
    async fn peek_context_returns_line_range() {
        let content = "alpha\nbeta\ngamma\ndelta\nepsilon\n";
        let m = AgentMonad::load_context("peek_test", content).then(AgentMonad::peek_context(
            "peek_test",
            2,
            4,
        ));

        let mut ctx = test_context();
        let result = ctx.run(m).await.unwrap().into_completed();

        assert!(result.contains("beta"), "should contain line 2");
        assert!(result.contains("gamma"), "should contain line 3");
        assert!(result.contains("delta"), "should contain line 4");
        assert!(!result.contains("alpha"), "should NOT contain line 1");
        assert!(!result.contains("epsilon"), "should NOT contain line 5");
    }

    #[tokio::test]
    async fn list_contexts_shows_loaded() {
        let m = AgentMonad::load_context("ctx_a", "small content")
            .then(AgentMonad::load_context(
                "ctx_b",
                "another context with more data",
            ))
            .then(AgentMonad::list_contexts());

        let mut ctx = test_context();
        let result = ctx.run(m).await.unwrap().into_completed();

        assert!(result.contains("ctx_a"), "listing should include ctx_a");
        assert!(result.contains("ctx_b"), "listing should include ctx_b");
    }

    // ─── Phase 5: Auto-Compaction ────────────────────────────────

    #[tokio::test]
    async fn compaction_truncates_old_content() {
        let mut ctx = test_context();

        // Build history with 20 large messages
        let mut m = AgentMonad::insert(Role::System, "system");
        for i in 0..20 {
            let big = format!("[msg {}] {}", i, "x".repeat(500));
            m = m.then(AgentMonad::insert(Role::User, big));
        }
        m = m.then(AgentMonad::pure("done"));
        ctx.run(m).await.unwrap();

        let before = ctx.history.len();
        let before_tokens = ctx.history.estimate_tokens();
        assert!(before >= 20, "should have 20+ messages before compaction");

        // Trigger compaction with a low threshold
        ctx.maybe_compact(100); // very low threshold forces compaction

        let after_tokens = ctx.history.estimate_tokens();
        assert!(
            after_tokens < before_tokens,
            "tokens should decrease after compaction: {before_tokens} -> {after_tokens}"
        );
    }

    #[tokio::test]
    async fn compaction_preserves_recent_messages() {
        let mut ctx = test_context();

        let mut m = AgentMonad::insert(Role::System, "system");
        for i in 0..15 {
            m = m.then(AgentMonad::insert(
                Role::User,
                format!("msg {i}: {}", "y".repeat(300)),
            ));
        }
        // Add a recent distinctive message
        m = m.then(AgentMonad::insert(Role::User, "RECENT_MARKER_MESSAGE"));
        m = m.then(AgentMonad::pure("done"));
        ctx.run(m).await.unwrap();

        ctx.maybe_compact(200);

        // Recent messages should survive
        let last = ctx.history.messages().last().unwrap();
        assert_eq!(
            last.content, "RECENT_MARKER_MESSAGE",
            "most recent message should survive compaction"
        );
    }

    // ─── Phase 7: Think & EvaluateProgress ───────────────────────

    #[tokio::test]
    async fn think_records_evidence_not_history() {
        let m = AgentMonad::think("I should analyze the data structure first")
            .then(AgentMonad::pure("done"));

        let mut ctx = test_context();
        ctx.run(m).await.unwrap();

        // Think should NOT add to history
        assert_eq!(ctx.history.len(), 0, "think should not insert into history");

        // But SHOULD add evidence
        let has_think = ctx
            .evidence()
            .iter()
            .any(|e| format!("{:?}", e).contains("analyze the data"));
        assert!(has_think, "think should be recorded in evidence trail");
    }

    #[tokio::test]
    async fn evaluate_progress_records_confidence() {
        let m = AgentMonad::evaluate_progress(0.75, "need to verify edge cases")
            .bind(|summary| AgentMonad::pure(summary));

        let mut ctx = test_context();
        let result = ctx.run(m).await.unwrap().into_completed();

        assert!(
            result.contains("75"),
            "should show confidence as percentage"
        );
        assert!(
            result.contains("edge cases"),
            "should include remaining work"
        );

        // Should NOT add to history
        assert_eq!(
            ctx.history.len(),
            0,
            "evaluate should not insert into history"
        );

        // Should add evidence
        assert!(
            !ctx.evidence().is_empty(),
            "evaluate should record evidence"
        );
    }

    // ─── Phase 8: Recipe DSL ─────────────────────────────────────

    #[tokio::test]
    async fn recipe_validates_and_estimates() {
        use crate::monad::recipe::{Recipe, StepKind};

        let recipe = Recipe::new("Test Pipeline")
            .step("load", "Load data", StepKind::CodeGen)
            .step_with_deps(
                "process",
                "Process: {{load.output}}",
                StepKind::CodeGen,
                vec!["load"],
            )
            .step_with_deps(
                "report",
                "Report: {{process.output}}",
                StepKind::TextGen,
                vec!["process"],
            );

        assert!(recipe.validate().is_ok());

        let order = recipe.execution_order().unwrap();
        assert_eq!(order, vec!["load", "process", "report"]);

        let estimate = recipe.estimate_cost();
        assert_eq!(estimate.total_steps, 3);
        assert!(estimate.estimated_llm_calls > 0);
    }

    #[tokio::test]
    async fn recipe_yaml_roundtrip_and_validation() {
        use crate::monad::recipe::Recipe;

        let yaml = r#"
name: E2E Test Pipeline
description: Integration test recipe
steps:
  - id: step1
    task: "Do step 1"
    kind: code_gen
  - id: step2
    task: "Do step 2 based on: {{step1.output}}"
    kind: text_gen
    depends_on: [step1]
"#;
        let recipe = Recipe::from_yaml(yaml).unwrap();
        assert!(recipe.validate().is_ok());
        assert_eq!(recipe.name, "E2E Test Pipeline");
        assert_eq!(recipe.steps.len(), 2);

        // Roundtrip
        let yaml2 = recipe.to_yaml().unwrap();
        let recipe2 = Recipe::from_yaml(&yaml2).unwrap();
        assert_eq!(recipe2.name, recipe.name);
        assert_eq!(recipe2.steps.len(), recipe.steps.len());
    }

    // ─── Turso Persistence ───────────────────────────────────────

    #[tokio::test]
    async fn turso_session_lifecycle() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let store = AgentStore::open(db_path.to_str().unwrap()).await.unwrap();

        let session_id = uuid::Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();

        // Create session
        store
            .create_session(&Session {
                session_id: session_id.clone(),
                model: "test-model".to_string(),
                task: "E2E test".to_string(),
                executor: "pyo3".to_string(),
                optimizer: None,
                optimized_instruction: None,
                started_at: now.clone(),
                finished_at: None,
                final_answer: None,
                score: None,
            })
            .await
            .unwrap();

        // Record turns
        for i in 0..3 {
            store
                .record_turn(&Turn {
                    session_id: session_id.clone(),
                    turn_num: i,
                    role: if i == 0 {
                        "System".to_string()
                    } else {
                        "User".to_string()
                    },
                    content: format!("Turn {i} content"),
                    code: None,
                    exec_stdout: None,
                    exec_stderr: None,
                    exec_return: None,
                    timestamp_ms: Utc::now().timestamp_millis(),
                })
                .await
                .unwrap();
        }

        // Finish session
        store
            .finish_session(&session_id, Some("final answer"), Some(0.95))
            .await
            .unwrap();

        // Verify summary
        let summary = store.session_summary(&session_id).await.unwrap();
        assert!(
            summary.contains("3 turns"),
            "summary should show 3 turns, got: {summary}"
        );
    }

    #[tokio::test]
    async fn turso_persists_evidence_and_context_workflow() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("e2e.db");
        let store = AgentStore::open(db_path.to_str().unwrap()).await.unwrap();

        // Run agent workflow
        let mut ctx = test_context();
        let m = AgentMonad::insert(Role::System, "You are a helpful agent.")
            .then(AgentMonad::insert(Role::User, "Analyze the data"))
            .then(AgentMonad::load_context(
                "dataset",
                "col1,col2\n1,a\n2,b\n3,c",
            ))
            .then(AgentMonad::search_context("dataset", "col"))
            .then(AgentMonad::think("The dataset has 3 rows with 2 columns"))
            .then(AgentMonad::evaluate_progress(0.5, "need to build model"))
            .then(AgentMonad::capture("analysis", "3 rows, 2 cols"))
            .then(AgentMonad::retrieve("analysis"));

        let result = ctx.run(m).await.unwrap().into_completed();
        assert_eq!(result, "3 rows, 2 cols");

        // Verify evidence was collected
        assert!(
            ctx.evidence().len() >= 3,
            "should have evidence from multiple actions"
        );

        // Verify context manager state
        let list = ctx.context_manager.list();
        assert!(
            !list.is_empty(),
            "context manager should have loaded context"
        );

        // Persist to Turso
        let session_id = uuid::Uuid::new_v4().to_string();
        store
            .create_session(&Session {
                session_id: session_id.clone(),
                model: "test".to_string(),
                task: "E2E integration test".to_string(),
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

        // Also persist evidence summary as a meta-turn
        let ev_summary = ctx.evidence_summary();
        store
            .record_turn(&Turn {
                session_id: session_id.clone(),
                turn_num: ctx.history.len() as i32,
                role: "Evidence".to_string(),
                content: ev_summary,
                code: None,
                exec_stdout: None,
                exec_stderr: None,
                exec_return: None,
                timestamp_ms: Utc::now().timestamp_millis(),
            })
            .await
            .unwrap();

        store
            .finish_session(&session_id, Some(&result), Some(1.0))
            .await
            .unwrap();

        // Verify Turso data
        let summary = store.session_summary(&session_id).await.unwrap();
        // 2 history turns (System + User) + 1 evidence meta-turn = 3
        assert!(
            summary.contains("3 turns"),
            "should have 3 turns in DB, got: {summary}"
        );
    }

    // ─── Combined E2E: Full Pipeline ─────────────────────────────

    #[tokio::test]
    async fn full_e2e_workflow_with_turso() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("full_e2e.db");
        let store = AgentStore::open(db_path.to_str().unwrap()).await.unwrap();

        let mut ctx = test_context();

        // 1. System setup
        let m = AgentMonad::insert(Role::System, "You are a data analyst agent.");
        ctx.run(m).await.unwrap();

        // 2. User task
        let m = AgentMonad::insert(Role::User, "Analyze sales data");
        ctx.run(m).await.unwrap();

        // 3. Load context
        let data = (0..100)
            .map(|i| format!("row {i}: product_{}, ${}", i % 10, i * 10 + 5))
            .collect::<Vec<_>>()
            .join("\n");
        let m = AgentMonad::load_context("sales", &data);
        ctx.run(m).await.unwrap();

        // 4. Search context
        let m = AgentMonad::search_context("sales", "product_5");
        let search_result = ctx.run(m).await.unwrap().into_completed();
        assert!(
            search_result.contains("product_5"),
            "search should find product_5"
        );

        // 5. Peek at specific lines
        let m = AgentMonad::peek_context("sales", 1, 3);
        let peek_result = ctx.run(m).await.unwrap().into_completed();
        assert!(peek_result.contains("row 0"), "peek should show row 0");

        // 6. Think about it
        let m =
            AgentMonad::think("Sales data has 100 rows, 10 products. Product_5 appears 10 times.");
        ctx.run(m).await.unwrap();

        // 7. Evaluate progress
        let m = AgentMonad::evaluate_progress(0.6, "analyzed structure, need to compute totals");
        let eval_result = ctx.run(m).await.unwrap().into_completed();
        assert!(eval_result.contains("60"), "should show 60% confidence");

        // 8. Capture result
        let m = AgentMonad::capture("conclusion", "10 products, 100 transactions");
        ctx.run(m).await.unwrap();

        // 9. Retrieve and verify
        let m = AgentMonad::retrieve("conclusion");
        let final_result = ctx.run(m).await.unwrap().into_completed();
        assert_eq!(final_result, "10 products, 100 transactions");

        // Verify state
        assert!(ctx.history.len() >= 2, "should have system + user messages");
        // Think + EvaluateProgress record evidence (context ops don't)
        assert!(
            ctx.evidence().len() >= 2,
            "should have evidence from think + evaluate, got: {}",
            ctx.evidence().len()
        );
        assert!(
            !ctx.context_manager.list().is_empty(),
            "should have loaded context"
        );

        // 10. Persist to Turso
        let session_id = uuid::Uuid::new_v4().to_string();
        store
            .create_session(&Session {
                session_id: session_id.clone(),
                model: "integration-test".to_string(),
                task: "Full E2E Test".to_string(),
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

        // Persist evidence as final meta-turn
        store
            .record_turn(&Turn {
                session_id: session_id.clone(),
                turn_num: ctx.history.len() as i32,
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
            .finish_session(&session_id, Some(&final_result), Some(1.0))
            .await
            .unwrap();

        // 11. Verify in Turso
        let summary = store.session_summary(&session_id).await.unwrap();
        assert!(summary.contains("turns"), "should report turns: {summary}");

        // Cleanup is automatic (tempdir drops)
    }

    // ─── Compaction + Turso ──────────────────────────────────────

    #[tokio::test]
    async fn compaction_then_turso_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("compact_e2e.db");
        let store = AgentStore::open(db_path.to_str().unwrap()).await.unwrap();

        let mut ctx = test_context();

        // Build large history
        let mut m = AgentMonad::insert(Role::System, "system");
        for i in 0..25 {
            m = m.then(AgentMonad::insert(
                Role::User,
                format!("message {i}: {}", "data ".repeat(100)),
            ));
        }
        m = m.then(AgentMonad::pure("done"));
        ctx.run(m).await.unwrap();

        let pre_compact = ctx.history.len();
        let pre_tokens = ctx.history.estimate_tokens();

        // Compact
        ctx.maybe_compact(500);

        let post_compact = ctx.history.len();
        let post_tokens = ctx.history.estimate_tokens();

        assert!(
            post_tokens <= pre_tokens,
            "tokens should not increase: {pre_tokens} -> {post_tokens}"
        );

        // Persist compacted history to Turso
        let session_id = uuid::Uuid::new_v4().to_string();
        store
            .create_session(&Session {
                session_id: session_id.clone(),
                model: "compact-test".to_string(),
                task: "Compaction E2E".to_string(),
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
            .finish_session(
                &session_id,
                Some(&format!("Compacted: {pre_compact} -> {post_compact}")),
                None,
            )
            .await
            .unwrap();

        let summary = store.session_summary(&session_id).await.unwrap();
        assert!(
            summary.contains("turns"),
            "should have turns after compaction: {summary}"
        );
    }

    // ─── Phase 8: Multi-Step Pipeline Execution ───────────────────

    #[tokio::test]
    async fn run_recipe_executes_steps_in_order_and_handles_failures() {
        use crate::monad::recipe::{Recipe, StepKind, StepStatus};

        let mut ctx = test_context();

        // Build a 3-step recipe. Steps will fail at ModelInference (no LLM configured)
        // but the pipeline orchestration is fully exercised.
        let recipe = Recipe::new("Failure Handling Pipeline")
            .description("Tests step execution order and failure cascading")
            .step(
                "step_a",
                "First step — will fail at LLM call",
                StepKind::CodeGen,
            )
            .step_with_deps(
                "step_b",
                "Depends on A: {{step_a.output}}",
                StepKind::CodeGen,
                vec!["step_a"],
            )
            .step(
                "step_c",
                "Independent step — also fails at LLM",
                StepKind::TextGen,
            );

        assert!(recipe.validate().is_ok());

        let result = ctx.run_recipe(recipe).await.unwrap();

        // Verify result structure
        assert_eq!(result.recipe_name, "Failure Handling Pipeline");
        assert_eq!(result.steps.len(), 3);

        // step_a should fail (no LLM)
        let step_a = result.steps.get("step_a").unwrap();
        assert!(
            matches!(step_a.status, StepStatus::Failed(_)),
            "step_a should fail at LLM call"
        );

        // step_b depends on step_a → should be SKIPPED
        let step_b = result.steps.get("step_b").unwrap();
        assert!(
            matches!(step_b.status, StepStatus::Skipped),
            "step_b should be skipped because step_a failed"
        );
        assert_eq!(step_b.turns, 0, "skipped step should have 0 turns");

        // step_c is independent → should also fail (no LLM), NOT skipped
        let step_c = result.steps.get("step_c").unwrap();
        assert!(
            matches!(step_c.status, StepStatus::Failed(_)),
            "step_c should fail independently (not skipped): {:?}",
            step_c.status
        );

        // Total turns should be non-zero (at least step_a attempted)
        assert!(
            result.total_turns >= 1,
            "should have attempted at least 1 turn"
        );

        // Print summary (visual check in test output)
        result.print_summary();
    }

    #[tokio::test]
    async fn run_recipe_preserves_context_across_steps() {
        use crate::monad::recipe::{Recipe, StepKind};

        let mut ctx = test_context();

        // Load a context BEFORE running the recipe
        let m = AgentMonad::load_context("pre_recipe", "data loaded before recipe");
        ctx.run(m).await.unwrap();

        let recipe =
            Recipe::new("Context Test").step("only_step", "Do something", StepKind::CodeGen);

        let result = ctx.run_recipe(recipe).await.unwrap();
        assert_eq!(result.steps.len(), 1);

        // Context manager should still have the pre-loaded context
        assert!(
            !ctx.context_manager.list().is_empty(),
            "context manager should survive recipe execution"
        );
    }

    #[tokio::test]
    async fn plan_recipe_action_parses_validates_and_executes() {
        // PlanRecipe action: agent submits YAML → runtime parses, validates, runs
        let yaml = r#"
name: Agent-Planned Pipeline
steps:
  - id: analyze
    task: "Analyze the problem"
    kind: code_gen
  - id: solve
    task: "Solve based on: {{analyze.output}}"
    kind: code_gen
    depends_on: [analyze]
"#;

        let m = AgentMonad::plan_recipe(yaml);
        let mut ctx = test_context();

        // PlanRecipe will parse → validate → run_recipe → steps fail at LLM
        // But the action itself should complete (returning error summary)
        let result = ctx.run(m).await;

        // The action should succeed (recipe was valid) even though steps failed
        match &result {
            Ok(summary) => {
                assert!(
                    summary.contains("Agent-Planned Pipeline"),
                    "summary should include recipe name: {summary}"
                );
            }
            Err(e) => {
                // If max turns exceeded, that's also fine
                let err_msg = format!("{e}");
                assert!(
                    err_msg.contains("Max turns") || err_msg.contains("LLM"),
                    "error should be LLM-related: {err_msg}"
                );
            }
        }

        // Evidence should record the recipe planning intent
        assert!(
            !ctx.evidence().is_empty(),
            "PlanRecipe should record evidence"
        );
    }

    #[tokio::test]
    async fn plan_recipe_rejects_invalid_yaml() {
        let bad_yaml = "this is not valid yaml: [[[";

        let m = AgentMonad::plan_recipe(bad_yaml);
        let mut ctx = test_context();
        let result = ctx.run(m).await;

        assert!(result.is_err(), "invalid YAML should produce an error");
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("YAML") || err.contains("yaml") || err.contains("invalid"),
            "error should mention YAML: {err}"
        );
    }

    #[tokio::test]
    async fn plan_recipe_rejects_cyclic_recipe() {
        let cyclic_yaml = r#"
name: Cyclic
steps:
  - id: a
    task: "step a"
    depends_on: [b]
  - id: b
    task: "step b"
    depends_on: [a]
"#;
        let m = AgentMonad::plan_recipe(cyclic_yaml);
        let mut ctx = test_context();
        let result = ctx.run(m).await;

        assert!(result.is_err(), "cyclic recipe should be rejected");
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("cyclic") || err.contains("cycle") || err.contains("Cyclic"),
            "error should mention cycle: {err}"
        );
    }

    #[tokio::test]
    async fn recipe_result_persisted_to_turso() {
        use crate::monad::recipe::{Recipe, StepKind, StepStatus};

        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("recipe_e2e.db");
        let store = AgentStore::open(db_path.to_str().unwrap()).await.unwrap();

        let mut ctx = test_context();

        let recipe = Recipe::new("Turso Pipeline")
            .step("step_1", "First step", StepKind::CodeGen)
            .step("step_2", "Second step", StepKind::TextGen);

        let result = ctx.run_recipe(recipe).await.unwrap();

        // Persist recipe results to Turso
        let session_id = uuid::Uuid::new_v4().to_string();
        store
            .create_session(&Session {
                session_id: session_id.clone(),
                model: "recipe-test".to_string(),
                task: "Recipe Persistence E2E".to_string(),
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

        // Record each step as a turn
        for (i, (step_id, step_result)) in result.steps.iter().enumerate() {
            let status_str = match &step_result.status {
                StepStatus::Completed => "completed",
                StepStatus::Failed(e) => "failed",
                StepStatus::Skipped => "skipped",
            };
            store
                .record_turn(&Turn {
                    session_id: session_id.clone(),
                    turn_num: i as i32,
                    role: format!("RecipeStep:{status_str}"),
                    content: format!(
                        "Step '{}': {} turns, {:.1}s\nOutput: {}",
                        step_id,
                        step_result.turns,
                        step_result.elapsed.as_secs_f64(),
                        if step_result.output.is_empty() {
                            "[none]"
                        } else {
                            &step_result.output
                        }
                    ),
                    code: None,
                    exec_stdout: None,
                    exec_stderr: None,
                    exec_return: None,
                    timestamp_ms: Utc::now().timestamp_millis(),
                })
                .await
                .unwrap();
        }

        let final_output = result.final_output().unwrap_or("no output");
        store
            .finish_session(&session_id, Some(final_output), None)
            .await
            .unwrap();

        // Verify in Turso
        let summary = store.session_summary(&session_id).await.unwrap();
        assert!(
            summary.contains("2 turns"),
            "should have 2 turns (one per step): {summary}"
        );
    }
}
