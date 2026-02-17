//! Unit tests for execution results, generation parsing, and prompts.

#[cfg(test)]
mod tests {
    use crate::monad::{ExecutionResult, Generation, PromptSystem};

    // ─── Execution Result Tests ──────────────────────────────────

    #[test]
    fn execution_result_feedback_formatting() {
        let res = ExecutionResult::success("hello world");
        assert_eq!(res.to_feedback(), "[stdout]\nhello world");

        let res = ExecutionResult::with_return("calc", "42");
        assert_eq!(res.to_feedback(), "[stdout]\ncalc\n\n[return] 42");

        let res = ExecutionResult::error("ValueError", "bad input");
        assert_eq!(res.to_feedback(), "[error] ValueError: bad input");
    }

    // ─── Generation Parsing Tests ────────────────────────────────

    #[test]
    fn parse_final_answer() {
        let generation = Generation::parse("Thinking...\n\nFINAL The answer is 42");
        assert_eq!(generation.final_answer.as_deref(), Some("The answer is 42"));
        assert!(generation.code.is_none());
    }

    #[test]
    fn parse_code_block() {
        let raw = "Here is the code:\n```repl\nprint('hello')\n```";
        let generation = Generation::parse(raw);
        assert_eq!(generation.code.as_deref(), Some("print('hello')"));
        assert!(!generation.is_final());
    }

    #[test]
    fn parse_python_block() {
        let raw = "Let's use python:\n```python\nx = 1\n```";
        let generation = Generation::parse(raw);
        assert_eq!(generation.code.as_deref(), Some("x = 1"));
    }

    #[test]
    fn parse_shell_command() {
        let raw = "RUN ls -la";
        let generation = Generation::parse(raw);
        assert_eq!(generation.shell_command.as_deref(), Some("ls -la"));
    }

    // ─── Prompt System Tests ─────────────────────────────────────

    #[test]
    fn render_system_prompt() {
        let prompts = PromptSystem::default();
        let rendered = prompts.render_system(&Default::default()).unwrap();
        assert!(rendered.contains("You are an expert AI agent"));
    }

    #[test]
    fn render_user_prompt() {
        let prompts = PromptSystem::default();
        let rendered = prompts.render_user("Solve P=NP").unwrap();
        assert!(rendered.contains("Solve P=NP"));
    }
}
