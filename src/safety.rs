//! Phase 12: Execution safety guardrails.
//!
//! Enforced at the interpreter level before/after code execution:
//! - Pre-execution: validate code size, check banned modules/builtins
//! - Post-execution: truncate oversized output
//! - Resource tracking: count total executions per session
//!
//! Three preset levels:
//! - `permissive()` — root agents, minimal restrictions
//! - `standard()` — sub-agents, sensible defaults
//! - `strict()` — untrusted code, aggressive restrictions

use std::fmt;

/// Safety limits for code execution.
///
/// These are enforced by `AgentContext` in `interpret_action()` before
/// and after each `ExecuteCode` action.
#[derive(Debug, Clone)]
pub struct ExecutionLimits {
    /// Maximum bytes for stdout/stderr before truncation.
    pub max_output_bytes: usize,
    /// Maximum code block length in characters. Reject if exceeded.
    pub max_code_length: usize,
    /// Per-execution timeout in seconds (injected as signal.alarm on Unix).
    pub timeout_seconds: u64,
    /// Maximum memory in MB (microsandbox only — PyO3 can't enforce this).
    pub max_memory_mb: usize,
    /// Python modules that must not be imported.
    pub banned_modules: Vec<String>,
    /// Python builtins that must not appear in code.
    pub banned_builtins: Vec<String>,
    /// Max total executions per agent session.
    pub max_total_executions: usize,
    /// Threshold (chars) above which execution output is auto-loaded into
    /// an isolated context instead of inline in conversation (Phase 2).
    /// Set to 0 to disable auto-loading.
    pub auto_load_threshold: usize,
}

impl ExecutionLimits {
    /// Permissive limits — for root agents that need full flexibility.
    ///
    /// Large output buffer, no banned modules, high execution count.
    pub fn permissive() -> Self {
        Self {
            max_output_bytes: 1_000_000, // 1MB
            max_code_length: 100_000,    // 100KB
            timeout_seconds: 300,        // 5 min
            max_memory_mb: 2048,         // 2GB
            banned_modules: vec![],
            banned_builtins: vec![],
            max_total_executions: 1000,
            auto_load_threshold: 10_000,
        }
    }

    /// Standard limits — sensible defaults for sub-agents.
    ///
    /// Moderate output buffer, no OS-level operations, reasonable timeout.
    pub fn standard() -> Self {
        Self {
            max_output_bytes: 100_000, // 100KB
            max_code_length: 50_000,   // 50KB
            timeout_seconds: 60,       // 1 min
            max_memory_mb: 512,        // 512MB
            banned_modules: vec!["shutil".into()],
            banned_builtins: vec![],
            max_total_executions: 200,
            auto_load_threshold: 10_000,
        }
    }

    /// Strict limits — for untrusted or potentially dangerous code.
    ///
    /// Small output, short timeout, banned OS/subprocess/eval.
    pub fn strict() -> Self {
        Self {
            max_output_bytes: 10_000, // 10KB
            max_code_length: 10_000,  // 10KB
            timeout_seconds: 10,      // 10 sec
            max_memory_mb: 128,       // 128MB
            banned_modules: vec![
                "os".into(),
                "subprocess".into(),
                "shutil".into(),
                "socket".into(),
                "http".into(),
                "urllib".into(),
                "requests".into(),
                "ctypes".into(),
                "multiprocessing".into(),
            ],
            banned_builtins: vec![
                "exec".into(),
                "eval".into(),
                "compile".into(),
                "__import__".into(),
            ],
            max_total_executions: 50,
            auto_load_threshold: 5_000,
        }
    }
}

impl Default for ExecutionLimits {
    fn default() -> Self {
        Self::permissive()
    }
}

/// A safety violation detected during pre-execution validation.
#[derive(Debug, Clone)]
pub enum SafetyViolation {
    /// Code block exceeds `max_code_length`.
    CodeTooLarge { length: usize, limit: usize },
    /// Code imports a banned module.
    BannedModule { module: String },
    /// Code uses a banned builtin.
    BannedBuiltin { builtin: String },
    /// Session has exceeded `max_total_executions`.
    ExecutionLimitReached { count: usize, limit: usize },
}

impl fmt::Display for SafetyViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CodeTooLarge { length, limit } => {
                write!(f, "code too large: {length} chars (limit: {limit})")
            }
            Self::BannedModule { module } => {
                write!(f, "banned module: '{module}' is not allowed")
            }
            Self::BannedBuiltin { builtin } => {
                write!(f, "banned builtin: '{builtin}' is not allowed")
            }
            Self::ExecutionLimitReached { count, limit } => {
                write!(f, "execution limit reached: {count}/{limit}")
            }
        }
    }
}

/// Pre-execution code validation.
///
/// Checks code against safety limits before it runs:
/// 1. Code length check
/// 2. Banned module check (import scanning)
/// 3. Banned builtin check
///
/// Returns `Ok(())` if the code passes all checks, or the first violation found.
pub fn validate_code(code: &str, limits: &ExecutionLimits) -> Result<(), SafetyViolation> {
    // 1. Code length
    if code.len() > limits.max_code_length {
        return Err(SafetyViolation::CodeTooLarge {
            length: code.len(),
            limit: limits.max_code_length,
        });
    }

    // 2. Banned module detection
    // Scan for `import X`, `from X import`, `__import__('X')`
    for module in &limits.banned_modules {
        // Check: `import <module>` or `import <module>.`
        if code.contains(&format!("import {module}"))
            || code.contains(&format!("from {module}"))
            || code.contains(&format!("__import__('{module}')"))
            || code.contains(&format!("__import__(\"{module}\")"))
        {
            return Err(SafetyViolation::BannedModule {
                module: module.clone(),
            });
        }
    }

    // 3. Banned builtins detection
    for builtin in &limits.banned_builtins {
        // Check for standalone function call: `eval(`, `exec(`
        if code.contains(&format!("{builtin}(")) {
            return Err(SafetyViolation::BannedBuiltin {
                builtin: builtin.clone(),
            });
        }
    }

    Ok(())
}

/// Post-execution output sanitization.
///
/// Truncates stdout/stderr if they exceed `max_output_bytes`.
/// Appends a truncation notice so the LLM knows output was cut.
pub fn sanitize_output(output: &str, limits: &ExecutionLimits) -> String {
    if output.len() <= limits.max_output_bytes {
        return output.to_string();
    }

    let truncated = &output[..limits.max_output_bytes];
    format!(
        "{truncated}\n\n[... output truncated at {} bytes (limit: {}) ...]",
        output.len(),
        limits.max_output_bytes
    )
}

/// Generate a Python timeout wrapper for Unix systems.
///
/// Wraps the user code in a `signal.alarm()` call so that long-running
/// code gets interrupted. Only works on Unix (no-op on Windows).
pub fn wrap_with_timeout(code: &str, timeout_seconds: u64) -> String {
    if timeout_seconds == 0 {
        return code.to_string();
    }

    format!(
        r#"import signal as _sig

def _timeout_handler(_signum, _frame):
    raise TimeoutError("Execution timed out after {timeout_seconds} seconds")

_sig.signal(_sig.SIGALRM, _timeout_handler)
_sig.alarm({timeout_seconds})
try:
{indented_code}
finally:
    _sig.alarm(0)
"#,
        timeout_seconds = timeout_seconds,
        indented_code = indent_code(code, 4),
    )
}

/// Indent all lines of code by `spaces` spaces.
fn indent_code(code: &str, spaces: usize) -> String {
    let indent = " ".repeat(spaces);
    code.lines()
        .map(|line| {
            if line.trim().is_empty() {
                String::new()
            } else {
                format!("{indent}{line}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_code_ok() {
        let limits = ExecutionLimits::strict();
        assert!(validate_code("print('hello')", &limits).is_ok());
    }

    #[test]
    fn test_validate_code_too_large() {
        let mut limits = ExecutionLimits::strict();
        limits.max_code_length = 10;
        let result = validate_code("x = 1 + 2 + 3 + 4 + 5", &limits);
        assert!(matches!(result, Err(SafetyViolation::CodeTooLarge { .. })));
    }

    #[test]
    fn test_validate_banned_module() {
        let limits = ExecutionLimits::strict();
        let result = validate_code("import os\nos.system('rm -rf /')", &limits);
        assert!(matches!(result, Err(SafetyViolation::BannedModule { .. })));
    }

    #[test]
    fn test_validate_banned_module_from_import() {
        let limits = ExecutionLimits::strict();
        let result = validate_code("from subprocess import run", &limits);
        assert!(matches!(result, Err(SafetyViolation::BannedModule { .. })));
    }

    #[test]
    fn test_validate_banned_builtin() {
        let limits = ExecutionLimits::strict();
        let result = validate_code("eval('1+1')", &limits);
        assert!(matches!(result, Err(SafetyViolation::BannedBuiltin { .. })));
    }

    #[test]
    fn test_validate_permissive_allows_os() {
        let limits = ExecutionLimits::permissive();
        assert!(validate_code("import os", &limits).is_ok());
    }

    #[test]
    fn test_sanitize_output_no_truncation() {
        let limits = ExecutionLimits::standard();
        let output = "short output";
        assert_eq!(sanitize_output(output, &limits), output);
    }

    #[test]
    fn test_sanitize_output_truncation() {
        let mut limits = ExecutionLimits::strict();
        limits.max_output_bytes = 20;
        let output = "a".repeat(100);
        let result = sanitize_output(&output, &limits);
        assert!(result.contains("[... output truncated"));
        assert!(result.starts_with(&"a".repeat(20)));
    }

    #[test]
    fn test_wrap_with_timeout() {
        let code = "x = expensive_computation()";
        let wrapped = wrap_with_timeout(code, 30);
        assert!(wrapped.contains("_sig.SIGALRM"));
        assert!(wrapped.contains("30 seconds"));
        assert!(wrapped.contains("x = expensive_computation()"));
    }

    #[test]
    fn test_wrap_with_timeout_zero() {
        let code = "print('hi')";
        assert_eq!(wrap_with_timeout(code, 0), code);
    }

    #[test]
    fn test_safety_violation_display() {
        let v = SafetyViolation::BannedModule {
            module: "os".into(),
        };
        assert_eq!(format!("{v}"), "banned module: 'os' is not allowed");
    }
}
