//! Unit tests for the monadic agent system.
//!
//! These tests verify the core monad mechanics without requiring
//! an LLM or Python runtime — they test Pure, bind, Insert,
//! Capture/Retrieve, and the context interpreter loop.

#[cfg(test)]
mod tests {
    use crate::monad::{
        ActionOutput, AgentConfig, AgentContext, AgentError, AgentMonad, LogLevel, Role,
    };

    // ─── Pure & Bind ─────────────────────────────────────────────

    #[tokio::test]
    async fn pure_returns_value() {
        let m = AgentMonad::pure("hello");
        let mut ctx = test_context();
        let result = ctx.run(m).await.unwrap();
        assert_eq!(result, "hello");
    }

    #[tokio::test]
    async fn bind_sequences_computations() {
        // Pure("hello").bind(|v| Pure(v + " world"))
        let m = AgentMonad::pure("hello").bind(|v| AgentMonad::pure(format!("{v} world")));
        let mut ctx = test_context();
        let result = ctx.run(m).await.unwrap();
        assert_eq!(result, "hello world");
    }

    #[tokio::test]
    async fn bind_chains_multiple() {
        let m = AgentMonad::pure("1")
            .bind(|v| AgentMonad::pure(format!("{v}+2")))
            .bind(|v| AgentMonad::pure(format!("{v}+3")));
        let mut ctx = test_context();
        let result = ctx.run(m).await.unwrap();
        assert_eq!(result, "1+2+3");
    }

    #[tokio::test]
    async fn then_ignores_previous_value() {
        let m = AgentMonad::pure("ignored").then(AgentMonad::pure("kept"));
        let mut ctx = test_context();
        let result = ctx.run(m).await.unwrap();
        assert_eq!(result, "kept");
    }

    // ─── Insert action ───────────────────────────────────────────

    #[tokio::test]
    async fn insert_adds_to_history() {
        let m = AgentMonad::insert(Role::User, "test message")
            .then(AgentMonad::pure("done"));
        let mut ctx = test_context();
        let result = ctx.run(m).await.unwrap();

        assert_eq!(result, "done");
        assert_eq!(ctx.history.len(), 1);
        assert_eq!(ctx.history.messages()[0].content, "test message");
    }

    #[tokio::test]
    async fn multiple_inserts_build_history() {
        let m = AgentMonad::insert(Role::System, "system prompt")
            .then(AgentMonad::insert(Role::User, "user query"))
            .then(AgentMonad::pure("done"));
        let mut ctx = test_context();
        ctx.run(m).await.unwrap();

        assert_eq!(ctx.history.len(), 2);
        assert_eq!(ctx.history.messages()[0].role, Role::System);
        assert_eq!(ctx.history.messages()[1].role, Role::User);
    }

    // ─── Capture & Retrieve ──────────────────────────────────────

    #[tokio::test]
    async fn capture_and_retrieve_roundtrip() {
        let m = AgentMonad::capture("key", "value123")
            .then(AgentMonad::retrieve("key"));
        let mut ctx = test_context();
        let result = ctx.run(m).await.unwrap();
        assert_eq!(result, "value123");
    }

    #[tokio::test]
    async fn retrieve_missing_variable_errors() {
        let m = AgentMonad::retrieve("nonexistent");
        let mut ctx = test_context();
        let err = ctx.run(m).await.unwrap_err();
        assert!(matches!(err, AgentError::VariableNotFound(_)));
    }

    #[tokio::test]
    async fn capture_overwrites_existing() {
        let m = AgentMonad::capture("x", "first")
            .then(AgentMonad::capture("x", "second"))
            .then(AgentMonad::retrieve("x"));
        let mut ctx = test_context();
        let result = ctx.run(m).await.unwrap();
        assert_eq!(result, "second");
    }

    // ─── Log action (no-op, just shouldn't panic) ────────────────

    #[tokio::test]
    async fn log_action_succeeds() {
        let m = AgentMonad::log(LogLevel::Info, "test log message")
            .then(AgentMonad::pure("ok"));
        let mut ctx = test_context();
        let result = ctx.run(m).await.unwrap();
        assert_eq!(result, "ok");
    }

    // ─── Max turns enforcement ───────────────────────────────────

    #[tokio::test]
    async fn max_turns_enforced() {
        // Build a chain of 10 inserts, but set max_turns to 5
        let mut m = AgentMonad::insert(Role::User, "msg");
        for _ in 0..9 {
            m = m.then(AgentMonad::insert(Role::User, "msg"));
        }
        m = m.then(AgentMonad::pure("done"));

        let mut ctx = AgentContext::new(AgentConfig {
            max_turns: 5,
            ..AgentConfig::default()
        });

        let err = ctx.run(m).await.unwrap_err();
        assert!(matches!(err, AgentError::MaxTurnsExceeded(5)));
    }

    // ─── Complex composition ─────────────────────────────────────

    #[tokio::test]
    async fn full_workflow_without_llm() {
        // Simulates: insert system prompt → capture a var → retrieve it → insert as user
        let m = AgentMonad::insert(Role::System, "You are a helpful agent.")
            .then(AgentMonad::capture("task", "solve ARC puzzle"))
            .then(AgentMonad::retrieve("task"))
            .bind(|task| {
                AgentMonad::insert(Role::User, format!("Please {task}"))
                    .then(AgentMonad::pure(format!("dispatched: {task}")))
            });

        let mut ctx = test_context();
        let result = ctx.run(m).await.unwrap();

        assert_eq!(result, "dispatched: solve ARC puzzle");
        assert_eq!(ctx.history.len(), 2); // system + user
        assert_eq!(ctx.history.messages()[1].content, "Please solve ARC puzzle");
    }

    // ─── ActionOutput tests ──────────────────────────────────────

    #[test]
    fn action_output_into_string() {
        assert_eq!(ActionOutput::Unit.into_string(), "");
        assert_eq!(ActionOutput::Value("hello".into()).into_string(), "hello");
    }

    #[test]
    fn action_output_has_value() {
        assert!(!ActionOutput::Unit.has_value());
        assert!(ActionOutput::Value("x".into()).has_value());
    }

    // ─── Helper ──────────────────────────────────────────────────

    fn test_context() -> AgentContext {
        AgentContext::new(AgentConfig {
            max_turns: 100,
            ..AgentConfig::default()
        })
    }
}
