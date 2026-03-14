//! HITL (Human-in-the-Loop) integration tests.
//!
//! Tests the full pause → suspend → resume → complete flow:
//! 1. Agent calls `elicit_user()` → `run()` returns `RunResult::Suspended`
//! 2. Caller feeds user response into the continuation
//! 3. `run()` resumes and completes with `RunResult::Completed`
//!
//! These tests don't require an LLM — they use pure monadic compositions.

#[cfg(test)]
mod hitl_tests {
    use crate::monad::{
        AgentConfig, AgentContext, AgentMonad, RunResult,
        action::ActionOutput, action::Role,
    };

    fn test_context() -> AgentContext {
        AgentContext::new(AgentConfig {
            max_turns: 100,
            ..AgentConfig::default()
        })
    }

    // ─── 1. Basic ElicitUser → Suspended ─────────────────────────

    #[tokio::test]
    async fn elicit_user_suspends_execution() {
        let m = AgentMonad::elicit_user("What is your name?");
        let mut ctx = test_context();
        let result = ctx.run(m).await.unwrap();

        // Should be Suspended, not Completed
        assert!(result.is_suspended(), "elicit_user should suspend execution");

        match result {
            RunResult::Suspended { question, partial_result, .. } => {
                assert_eq!(question, "What is your name?");
                assert!(partial_result.is_none(), "basic elicit has no partial result");
            }
            RunResult::Completed(_) => panic!("Expected Suspended, got Completed"),
        }
    }

    // ─── 2. ElicitUser with partial result ───────────────────────

    #[tokio::test]
    async fn elicit_user_with_partial_result() {
        let m = AgentMonad::elicit_user_with_result(
            "Should I continue with approach A or B?",
            "Partial analysis: found 3 patterns",
        );
        let mut ctx = test_context();
        let result = ctx.run(m).await.unwrap();

        assert!(result.is_suspended());

        match result {
            RunResult::Suspended { question, partial_result, .. } => {
                assert_eq!(question, "Should I continue with approach A or B?");
                assert_eq!(
                    partial_result.as_deref(),
                    Some("Partial analysis: found 3 patterns")
                );
            }
            _ => panic!("Expected Suspended"),
        }
    }

    // ─── 3. Full cycle: suspend → resume → complete ──────────────

    #[tokio::test]
    async fn suspend_and_resume_full_cycle() {
        // Build a program that:
        // 1. Inserts a system message
        // 2. Asks the user a question (suspends)
        // 3. Uses the user's response to produce a final answer
        let m = AgentMonad::insert(Role::System, "You are a helpful agent.")
            .then(AgentMonad::elicit_user("What color do you prefer?"))
            .bind(|user_response| {
                AgentMonad::pure(format!("User chose: {user_response}"))
            });

        let mut ctx = test_context();

        // Step 1: First run should suspend
        let result = ctx.run(m).await.unwrap();
        assert!(result.is_suspended(), "Should suspend at elicit_user");

        let continuation = match result {
            RunResult::Suspended { question, continuation, .. } => {
                assert_eq!(question, "What color do you prefer?");
                continuation
            }
            _ => panic!("Expected Suspended"),
        };

        // Step 2: Resume with user's answer
        // Feed "blue" as the user's response through the continuation
        let resumed = match continuation {
            AgentMonad::Perform { next, .. } => {
                // The continuation is wrapped as Insert(placeholder) → original next
                // Feed user response through
                next(ActionOutput::Value("blue".to_string()))
            }
            other => other,
        };

        let final_result = ctx.run(resumed).await.unwrap();

        // Step 3: Should now be completed
        assert!(!final_result.is_suspended(), "After resume, should be completed");
        assert_eq!(
            final_result.into_completed(),
            "User chose: blue",
            "Should incorporate the user's response"
        );

        // Verify history was maintained across suspend/resume
        assert!(ctx.history.len() >= 1, "System message should be in history");
    }

    // ─── 4. Elicit preserves prior state ─────────────────────────

    #[tokio::test]
    async fn elicit_preserves_captured_variables() {
        // Capture a variable, then elicit, then use both
        let m = AgentMonad::capture("project", "rig-rlm")
            .then(AgentMonad::elicit_user("Which module to work on?"))
            .bind(|user_response| {
                AgentMonad::retrieve("project").bind(move |project| {
                    AgentMonad::pure(format!("Working on {project}/{user_response}"))
                })
            });

        let mut ctx = test_context();

        // Run → should suspend
        let result = ctx.run(m).await.unwrap();
        assert!(result.is_suspended());

        let continuation = match result {
            RunResult::Suspended { continuation, .. } => continuation,
            _ => panic!("Expected Suspended"),
        };

        // Resume with "monad"
        let resumed = match continuation {
            AgentMonad::Perform { next, .. } => next(ActionOutput::Value("monad".to_string())),
            other => other,
        };

        // Variables captured before suspension should still be available
        let final_result = ctx.run(resumed).await.unwrap();
        assert_eq!(
            final_result.into_completed(),
            "Working on rig-rlm/monad",
            "Captured variable should survive suspend/resume"
        );
    }

    // ─── 5. Evidence is recorded for elicit ──────────────────────

    #[tokio::test]
    async fn elicit_records_evidence() {
        let m = AgentMonad::elicit_user("Need clarification on X");
        let mut ctx = test_context();
        let _ = ctx.run(m).await.unwrap();

        // HITL pause should be recorded in the evidence trail
        let has_hitl_evidence = ctx
            .evidence()
            .iter()
            .any(|e| format!("{:?}", e).contains("HITL"));
        assert!(
            has_hitl_evidence,
            "elicit_user should record HITL evidence, got: {:?}",
            ctx.evidence()
        );
    }

    // ─── 6. FrozenStore freeze/thaw cycle ────────────────────────

    #[tokio::test]
    async fn frozen_store_freeze_thaw_cycle() {
        use crate::monad::frozen::{FrozenStore, FrozenTask};

        let store = FrozenStore::new();

        // Create a context + continuation
        let ctx = test_context();
        let continuation = AgentMonad::pure("after resume");

        // Freeze
        let id = "test-ctx-123";
        store.freeze(id, FrozenTask {
            context: ctx,
            continuation,
            created_at: chrono::Utc::now(),
        });

        // Thaw
        let frozen = store.thaw(id);
        assert!(frozen.is_some(), "Should recover frozen task");

        let frozen = frozen.unwrap();
        // The thawed context should be usable
        assert!(frozen.context.history.len() == 0, "Fresh context has empty history");

        // Thaw again should fail (task is consumed)
        assert!(store.thaw(id).is_none(), "Task should only be thawed once");
    }

    // ─── 7. FrozenStore TTL expiry ───────────────────────────────

    #[tokio::test]
    async fn frozen_store_ttl_expiry() {
        use crate::monad::frozen::{FrozenStore, FrozenTask};

        // Very short TTL (0 seconds → immediate expiry)
        let store = FrozenStore::with_ttl(chrono::Duration::seconds(0));

        let ctx = test_context();
        store.freeze("expires-soon", FrozenTask {
            context: ctx,
            continuation: AgentMonad::pure("expired"),
            // Set created_at in the past so it's already expired
            created_at: chrono::Utc::now() - chrono::Duration::seconds(1),
        });

        // Should be expired immediately
        let result = store.thaw("expires-soon");
        assert!(result.is_none(), "Expired task should not be recoverable");
    }

    // ─── 8. Multi-step with elicit in the middle ─────────────────

    #[tokio::test]
    async fn multi_step_with_elicit_in_middle() {
        // Step 1: Think
        // Step 2: Elicit (suspend)
        // Step 3: Process response
        // Step 4: Produce final answer
        let m = AgentMonad::think("Analyzing the problem...")
            .then(AgentMonad::capture("step", "analysis"))
            .then(AgentMonad::elicit_user("Found an ambiguity. Option A or B?"))
            .bind(|user_choice| {
                AgentMonad::retrieve("step").bind(move |step| {
                    AgentMonad::think(format!("User chose {user_choice} during {step}"))
                        .then(AgentMonad::pure(format!("Completed {step} with choice: {user_choice}")))
                })
            });

        let mut ctx = test_context();

        // Should suspend at elicit
        let result = ctx.run(m).await.unwrap();
        assert!(result.is_suspended());

        // Evidence should already contain the think + HITL entries
        let evidence_count = ctx.evidence().len();
        assert!(evidence_count >= 2, "Should have think + HITL evidence: got {evidence_count}");

        // Resume with "A"
        let continuation = match result {
            RunResult::Suspended { continuation, .. } => continuation,
            _ => panic!("Expected Suspended"),
        };

        let resumed = match continuation {
            AgentMonad::Perform { next, .. } => next(ActionOutput::Value("A".to_string())),
            other => other,
        };

        let final_result = ctx.run(resumed).await.unwrap();
        assert_eq!(
            final_result.into_completed(),
            "Completed analysis with choice: A"
        );

        // After resume, evidence should have grown (second think)
        assert!(
            ctx.evidence().len() > evidence_count,
            "Resume should add more evidence"
        );
    }
}
