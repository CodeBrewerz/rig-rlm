//! Resource-safety tests for HITL suspend/resume.
//!
//! Tests verify the CodeExecutor trait's suspend/resume API,
//! the blocking ELICIT Python function, and state preservation.
//! Uses Pyo3 executor directly (no microsandbox server needed).

#[cfg(test)]
mod hitl_resource_tests {
    use crate::sandbox::{CodeExecutor, Pyo3CodeExecutor};
    use crate::session::SessionConfig;

    // ─── 1. CodeExecutor trait default methods (Pyo3 = no-op) ────

    #[tokio::test]
    async fn pyo3_suspend_is_noop() {
        let mut exec = Pyo3CodeExecutor::new();

        let result: anyhow::Result<Option<String>> = exec.suspend_for_elicit().await;
        assert!(result.is_ok(), "suspend should succeed");
        assert!(
            result.unwrap().is_none(),
            "Pyo3 should return None (no state to snapshot)"
        );
    }

    #[tokio::test]
    async fn pyo3_resume_is_noop() {
        let mut exec = Pyo3CodeExecutor::new();

        let result: anyhow::Result<()> = exec.resume_after_elicit("JSON", None).await;
        assert!(result.is_ok(), "resume should succeed for Pyo3");
    }

    #[tokio::test]
    async fn pyo3_not_suspended_by_default() {
        let exec = Pyo3CodeExecutor::new();
        assert!(
            !exec.is_suspended(),
            "Should not be suspended initially"
        );
    }

    #[tokio::test]
    async fn pyo3_suspend_resume_roundtrip() {
        let mut exec = Pyo3CodeExecutor::new();

        // Suspend → still not really suspended (Pyo3 no-ops)
        let state: Option<String> = exec.suspend_for_elicit().await.unwrap();
        assert!(state.is_none());
        assert!(!exec.is_suspended(), "Pyo3 never really suspends");

        // Resume → also no-op
        let _: () = exec.resume_after_elicit("test response", None).await.unwrap();
        assert!(!exec.is_suspended());

        eprintln!("✅ Pyo3 suspend/resume roundtrip (no-ops) verified");
    }

    // ─── 2. Blocking ELICIT function generation ──────────────────

    #[test]
    fn blocking_elicit_code_has_sentinel_poll() {
        let code = SessionConfig::generate_elicit_code_blocking();
        assert!(
            code.contains("/tmp/.elicit_response"),
            "Blocking ELICIT should poll sentinel file"
        );
        assert!(
            code.contains("time.sleep"),
            "Blocking ELICIT should use time.sleep for polling"
        );
        assert!(
            code.contains("__ELICIT__"),
            "Blocking ELICIT should print markers"
        );
        assert!(
            code.contains("flush=True"),
            "Blocking ELICIT should flush stdout"
        );
    }

    #[test]
    fn nonblocking_elicit_code_returns_marker() {
        let code = SessionConfig::generate_elicit_code();
        assert!(
            code.contains("[elicit]"),
            "Non-blocking ELICIT should return [elicit] marker"
        );
        assert!(
            !code.contains("time.sleep"),
            "Non-blocking ELICIT should NOT use time.sleep"
        );
    }

    // ─── 3. Pyo3 state persists across ELICIT calls ──────────────

    #[tokio::test]
    async fn pyo3_state_persists_through_elicit_cycle() {
        use crate::monad::{AgentConfig, AgentContext, AgentMonad};

        let mut ctx = AgentContext::new(AgentConfig {
            max_turns: 100,
            ..AgentConfig::default()
        });

        // Step 1: Set up variables
        let setup = r#"
important_data = {"items": 42, "status": "processing"}
running_total = 99.5
print(f"Setup: {len(important_data)} fields, total={running_total}")
"#;
        let result = ctx.run(AgentMonad::execute_code(setup)).await.unwrap();
        assert!(
            result.as_str().contains("Setup"),
            "Setup should succeed. Got: {}",
            result.as_str()
        );

        // Step 2: Call ELICIT (Pyo3 generates markers in stdout)
        let elicit = r#"choice = ELICIT("Continue?")"#;
        let _r = ctx.run(AgentMonad::execute_code(elicit)).await.unwrap();

        // Step 3: Verify variables persist (Pyo3 is in-process, state is alive)
        let verify = r#"
print(f"Data: items={important_data['items']}, total={running_total}")
"#;
        let result = ctx.run(AgentMonad::execute_code(verify)).await.unwrap();
        assert!(
            result.as_str().contains("items=42"),
            "Variables should persist. Got: {}",
            result.as_str()
        );
        assert!(
            result.as_str().contains("total=99.5"),
            "Numeric state should persist. Got: {}",
            result.as_str()
        );

        eprintln!("✅ Pyo3 state preserved across ELICIT call");
    }

    // ─── 4. Full HITL suspend → resume monad cycle ───────────────

    #[tokio::test]
    async fn full_hitl_suspend_resume_with_executor() {
        use crate::monad::action::ActionOutput;
        use crate::monad::{AgentConfig, AgentContext, AgentMonad, RunResult};

        let mut ctx = AgentContext::new(AgentConfig {
            max_turns: 100,
            ..AgentConfig::default()
        });

        // Compose: ELICIT → suspend → resume with user response
        let m = AgentMonad::elicit_user_with_result(
            "Pick JSON or CSV",
            "Analysis complete: 50 rows",
        )
        .bind(|user_response| {
            AgentMonad::pure(format!("Generating {} report", user_response))
        });

        let result = ctx.run(m).await.unwrap();
        assert!(result.is_suspended(), "Should suspend");

        match result {
            RunResult::Suspended {
                question,
                partial_result,
                continuation,
            } => {
                assert_eq!(question, "Pick JSON or CSV");
                assert_eq!(
                    partial_result.as_deref(),
                    Some("Analysis complete: 50 rows")
                );

                // Simulate user responding
                let resumed = match continuation {
                    AgentMonad::Perform { next, .. } => {
                        next(ActionOutput::Value("CSV".to_string()))
                    }
                    other => other,
                };

                let final_result = ctx.run(resumed).await.unwrap();
                assert_eq!(final_result.into_completed(), "Generating CSV report");

                eprintln!("✅ Full HITL suspend→resume lifecycle verified");
            }
            _ => panic!("Expected Suspended"),
        }
    }
}
