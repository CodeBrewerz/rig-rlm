//! Live HITL (Human-in-the-Loop) tests with real sandbox & LLM.
//!
//! These tests exercise the actual ELICIT() Python function in the sandbox,
//! the interaction loop detection, and optionally Trinity LLM generating
//! code that calls ELICIT().
//!
//! Tests marked `#[ignore]` require OPENAI_API_KEY for LLM calls.
//! Non-ignored tests run with Pyo3 sandbox only (no LLM needed).

#[cfg(test)]
mod hitl_live {
    use crate::monad::{
        AgentConfig, AgentContext, AgentMonad, RunResult,
        action::{ActionOutput, Role},
        interaction::agent_task_full,
    };

    fn test_context() -> AgentContext {
        AgentContext::new(AgentConfig {
            max_turns: 100,
            ..AgentConfig::default()
        })
    }

    // ─── 1. Sandbox ELICIT() produces correct markers ────────────

    /// Tests that the Pyo3 sandbox correctly executes ELICIT() and
    /// produces the expected [elicit] marker in output.
    #[tokio::test]
    async fn sandbox_elicit_produces_correct_output() {
        let mut ctx = test_context();

        // The executor auto-initializes on first execute_code, which
        // injects SUBMIT() and ELICIT() functions into the Python env.
        let code = r#"result = ELICIT("What color should the background be?")"#;
        let exec_m = AgentMonad::execute_code(code);
        let result = ctx.run(exec_m).await.unwrap();

        let output = result.as_str();
        eprintln!("📦 ELICIT sandbox output: {output}");

        // ELICIT() prints __ELICIT__ markers and returns "[elicit] question"
        assert!(
            output.contains("[elicit]") || output.contains("__ELICIT__"),
            "ELICIT() output should contain marker. Got: {output}"
        );
        assert!(
            output.contains("What color should the background be?"),
            "Should contain the question text. Got: {output}"
        );
    }

    // ─── 2. ELICIT with partial result ───────────────────────────

    #[tokio::test]
    async fn sandbox_elicit_with_partial_result() {
        let mut ctx = test_context();

        let code = r#"result = ELICIT("Continue?", partial_result="Found 3 issues so far")"#;
        let exec_m = AgentMonad::execute_code(code);
        let result = ctx.run(exec_m).await.unwrap();

        let output = result.as_str();
        eprintln!("📦 ELICIT+partial output: {output}");

        assert!(
            output.contains("[elicit]") || output.contains("__ELICIT__"),
            "Should contain ELICIT marker. Got: {output}"
        );
        assert!(
            output.contains("3 issues"),
            "Should contain partial result. Got: {output}"
        );
    }

    // ─── 3. ELICIT marker extraction ─────────────────────────────

    #[test]
    fn extract_elicit_request_works() {
        use crate::session::extract_elicit_request;

        let stdout = r#"some output
__ELICIT__{"question": "Pick A or B?", "partial_result": "Analysis done"}__ELICIT__
[elicit] Pick A or B?"#;

        let result = extract_elicit_request(stdout);
        assert!(result.is_some(), "Should extract ELICIT request");

        let (question, partial) = result.unwrap();
        assert_eq!(question, "Pick A or B?");
        assert_eq!(partial.as_deref(), Some("Analysis done"));
    }

    #[test]
    fn extract_elicit_without_partial() {
        use crate::session::extract_elicit_request;

        let stdout = r#"__ELICIT__{"question": "Continue?"}__ELICIT__
[elicit] Continue?"#;

        let result = extract_elicit_request(stdout);
        assert!(result.is_some());

        let (question, partial) = result.unwrap();
        assert_eq!(question, "Continue?");
        assert!(partial.is_none());
    }

    // ─── 4. Full HITL monad: elicit_user_with_result → suspend → resume ──

    #[tokio::test]
    async fn elicit_with_result_suspend_resume_cycle() {
        let mut ctx = test_context();

        // Simulate what the interaction loop does when it detects [elicit]
        let m = AgentMonad::elicit_user_with_result(
            "Do you want CSV or JSON format?",
            "Analysis complete: 100 rows processed",
        )
        .bind(|user_response| {
            AgentMonad::pure(format!("Continuing with format: {user_response}"))
        });

        let result = ctx.run(m).await.unwrap();
        assert!(result.is_suspended(), "Should suspend for user input");

        match result {
            RunResult::Suspended {
                question,
                partial_result,
                continuation,
            } => {
                assert_eq!(question, "Do you want CSV or JSON format?");
                assert_eq!(
                    partial_result.as_deref(),
                    Some("Analysis complete: 100 rows processed")
                );

                // Resume with user's choice
                let resumed = match continuation {
                    AgentMonad::Perform { next, .. } => {
                        next(ActionOutput::Value("JSON".to_string()))
                    }
                    other => other,
                };

                let final_result = ctx.run(resumed).await.unwrap();
                assert_eq!(
                    final_result.into_completed(),
                    "Continuing with format: JSON"
                );

                eprintln!("✅ Full suspend→resume cycle completed with correct output");
            }
            _ => panic!("Expected Suspended"),
        }
    }

    // ─── 5. Sandbox ELICIT in multi-step code ────────────────────

    #[tokio::test]
    async fn sandbox_elicit_in_multi_step_code() {
        let mut ctx = test_context();

        // First execution: some setup work
        let setup_code = r#"
data = [{"name": "Widget A", "price": 10.99}, {"name": "Widget B", "price": 24.50}]
total = sum(item["price"] for item in data)
print(f"Loaded {len(data)} items, total: ${total:.2f}")
"#;
        let result = ctx.run(AgentMonad::execute_code(setup_code)).await.unwrap();
        eprintln!("Step 1: {}", result.as_str());
        assert!(result.contains("Loaded 2 items"), "Should load data");

        // Second execution: calls ELICIT
        let elicit_code = r#"
user_format = ELICIT("What format should the report be in: CSV, JSON, or YAML?", partial_result=f"Processed {len(data)} items, total=${total:.2f}")
print(f"User chose: {user_format}")
"#;
        let result = ctx.run(AgentMonad::execute_code(elicit_code)).await.unwrap();
        let output = result.as_str();
        eprintln!("Step 2: {output}");

        assert!(
            output.contains("[elicit]"),
            "Should hit ELICIT marker. Got: {output}"
        );
        assert!(
            output.contains("CSV, JSON, or YAML"),
            "Should contain the question. Got: {output}"
        );

        eprintln!("✅ Multi-step sandbox ELICIT test passed");
    }

    // ─── 6. Live Trinity HITL test ───────────────────────────────

    #[tokio::test]
    #[ignore] // Requires OPENAI_API_KEY
    async fn live_trinity_elicit_during_coding() {
        dotenvy::dotenv().ok();

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
            max_turns: 15,
            ..AgentConfig::default()
        };
        let mut ctx = AgentContext::new(config);

        let sep = "=".repeat(60);
        eprintln!("\n{sep}");
        eprintln!("🧪 LIVE TRINITY HITL TEST");
        eprintln!("  Agent should call ELICIT() during code execution");
        eprintln!("{sep}");

        // Craft a prompt that instructs the agent to use ELICIT()
        let program = agent_task_full(
            "Write Python code to generate a sales report. \
             IMPORTANT: Before generating the report, you MUST call \
             ELICIT('What format should the report be in: CSV, JSON, or YAML?') \
             to ask the user what format they prefer. \
             Then use the returned user response to generate the report. \
             The report should contain: Widget A ($10.99), Widget B ($24.50), Widget C ($7.25). \
             After generating, call SUBMIT(report_string) with the final report.",
            None,
            None,
            None,
            vec![],
        );

        let result = ctx.run(program).await;

        match &result {
            Ok(run_result) => {
                if run_result.is_suspended() {
                    eprintln!("✅ Agent SUSPENDED (HITL triggered!)");
                    eprintln!("   Question: {}", run_result.as_str());

                    // Resume with "JSON"
                    match result.unwrap() {
                        RunResult::Suspended { continuation, .. } => {
                            eprintln!("🔄 Resuming with user response: 'JSON'");

                            let resumed = match continuation {
                                AgentMonad::Perform { next, .. } => {
                                    next(ActionOutput::Value("JSON".to_string()))
                                }
                                other => other,
                            };

                            let final_result = ctx.run(resumed).await;
                            match &final_result {
                                Ok(r) => {
                                    eprintln!("✅ Agent COMPLETED after resume!");
                                    eprintln!("   Result: {r}");
                                }
                                Err(e) => {
                                    eprintln!("⚠️ Agent error after resume: {e}");
                                }
                            }
                        }
                        _ => unreachable!(),
                    }
                } else {
                    eprintln!("⚠️ Agent completed WITHOUT suspending");
                    eprintln!("   Result: {run_result}");
                    eprintln!("   (LLM may not have used ELICIT — check prompt)");
                }
            }
            Err(e) => {
                eprintln!("⚠️ Agent error: {e}");
            }
        }

        eprintln!("\n📊 Evidence entries: {}", ctx.evidence().len());
        eprintln!("💬 History turns: {}", ctx.history.len());
        eprintln!("{sep}\n");
    }
}
