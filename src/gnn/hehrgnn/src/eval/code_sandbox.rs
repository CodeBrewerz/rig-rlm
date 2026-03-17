//! Code execution sandbox for verifiable RL rewards.
//!
//! Creates a temporary Rust crate, writes agent code, and runs
//! `cargo check` / `cargo test` for real binary pass/fail rewards.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

// ──────────────────────────────────────────────────────
// Sandbox result
// ──────────────────────────────────────────────────────

/// Result of a sandbox execution.
#[derive(Debug, Clone)]
pub struct SandboxResult {
    /// Whether the command succeeded (exit code 0).
    pub success: bool,
    /// Exit code.
    pub exit_code: i32,
    /// Stdout output.
    pub stdout: String,
    /// Stderr output.
    pub stderr: String,
    /// Execution time in milliseconds.
    pub elapsed_ms: u64,
}

impl SandboxResult {
    /// Binary reward: +1.0 for success, -0.5 for failure.
    pub fn compile_reward(&self) -> f64 {
        if self.success {
            1.0
        } else {
            -0.5
        }
    }
}

/// Result of running tests.
#[derive(Debug, Clone)]
pub struct TestResult {
    pub passed: usize,
    pub failed: usize,
    pub total: usize,
    pub success: bool,
    pub elapsed_ms: u64,
    pub output: String,
}

impl TestResult {
    /// Reward: +2.0 per pass, -1.0 per fail.
    pub fn test_reward(&self) -> f64 {
        2.0 * self.passed as f64 - 1.0 * self.failed as f64
    }

    pub fn pass_rate(&self) -> f64 {
        if self.total == 0 {
            1.0
        } else {
            self.passed as f64 / self.total as f64
        }
    }
}

// ──────────────────────────────────────────────────────
// Code sandbox
// ──────────────────────────────────────────────────────

/// Temporary Rust crate sandbox for real code execution.
pub struct CodeSandbox {
    /// Path to the sandbox directory.
    dir: PathBuf,
    /// Whether the sandbox has been initialized.
    initialized: bool,
    /// Current source code in lib.rs.
    current_code: String,
    /// Test code.
    test_code: String,
    /// Execution timeout in seconds.
    timeout_secs: u64,
}

impl CodeSandbox {
    /// Create a new sandbox with a unique ID.
    pub fn new(episode_id: usize) -> Self {
        let dir = std::env::temp_dir().join(format!("rl_sandbox_{}", episode_id));
        Self {
            dir,
            initialized: false,
            current_code: String::new(),
            test_code: String::new(),
            timeout_secs: 30,
        }
    }

    /// Initialize the sandbox directory with Cargo.toml and default code.
    pub fn init(&mut self) -> std::io::Result<()> {
        if self.initialized {
            return Ok(());
        }

        // Create directory structure
        std::fs::create_dir_all(self.dir.join("src"))?;
        std::fs::create_dir_all(self.dir.join("tests"))?;

        // Write minimal Cargo.toml
        let cargo_toml = r#"[package]
name = "rl_sandbox"
version = "0.1.0"
edition = "2021"
"#;
        std::fs::write(self.dir.join("Cargo.toml"), cargo_toml)?;

        // Write initial lib.rs
        self.current_code = Self::default_code().to_string();
        std::fs::write(self.dir.join("src/lib.rs"), &self.current_code)?;

        // Write test file
        self.test_code = Self::default_tests().to_string();
        std::fs::write(self.dir.join("tests/test.rs"), &self.test_code)?;

        self.initialized = true;
        Ok(())
    }

    fn default_code() -> &'static str {
        r#"/// Add two numbers.
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// Multiply two numbers.
pub fn multiply(a: i32, b: i32) -> i32 {
    a * b
}

/// Check if a number is even.
pub fn is_even(n: i32) -> bool {
    n % 2 == 0
}
"#
    }

    fn default_tests() -> &'static str {
        r#"use rl_sandbox::*;

#[test]
fn test_add() {
    assert_eq!(add(2, 3), 5);
    assert_eq!(add(-1, 1), 0);
}

#[test]
fn test_multiply() {
    assert_eq!(multiply(3, 4), 12);
    assert_eq!(multiply(-2, 5), -10);
}

#[test]
fn test_is_even() {
    assert!(is_even(4));
    assert!(!is_even(3));
}
"#
    }

    /// Update the source code.
    pub fn set_code(&mut self, code: &str) -> std::io::Result<()> {
        self.current_code = code.to_string();
        if self.initialized {
            std::fs::write(self.dir.join("src/lib.rs"), &self.current_code)?;
        }
        Ok(())
    }

    /// Apply a simple text patch (find → replace).
    pub fn apply_patch(&mut self, find: &str, replace: &str) -> std::io::Result<bool> {
        if self.current_code.contains(find) {
            self.current_code = self.current_code.replace(find, replace);
            if self.initialized {
                std::fs::write(self.dir.join("src/lib.rs"), &self.current_code)?;
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Run `cargo check` — verifiable compile check.
    pub fn check(&self) -> SandboxResult {
        if !self.initialized {
            return SandboxResult {
                success: false,
                exit_code: -1,
                stdout: String::new(),
                stderr: "Sandbox not initialized".to_string(),
                elapsed_ms: 0,
            };
        }

        let start = std::time::Instant::now();
        let output = Command::new("cargo")
            .arg("check")
            .current_dir(&self.dir)
            .env("CARGO_TARGET_DIR", self.dir.join("target"))
            .output();

        let elapsed = start.elapsed().as_millis() as u64;

        match output {
            Ok(out) => SandboxResult {
                success: out.status.success(),
                exit_code: out.status.code().unwrap_or(-1),
                stdout: String::from_utf8_lossy(&out.stdout).to_string(),
                stderr: String::from_utf8_lossy(&out.stderr).to_string(),
                elapsed_ms: elapsed,
            },
            Err(e) => SandboxResult {
                success: false,
                exit_code: -1,
                stdout: String::new(),
                stderr: format!("Failed to run cargo: {}", e),
                elapsed_ms: elapsed,
            },
        }
    }

    /// Run `cargo test` — verifiable test results.
    pub fn test(&self) -> TestResult {
        if !self.initialized {
            return TestResult {
                passed: 0,
                failed: 0,
                total: 0,
                success: false,
                elapsed_ms: 0,
                output: "Sandbox not initialized".to_string(),
            };
        }

        let start = std::time::Instant::now();
        let output = Command::new("cargo")
            .arg("test")
            .current_dir(&self.dir)
            .env("CARGO_TARGET_DIR", self.dir.join("target"))
            .output();

        let elapsed = start.elapsed().as_millis() as u64;

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                let combined = format!("{}\n{}", stdout, stderr);

                let (passed, failed, total) = parse_test_output(&combined);

                TestResult {
                    passed,
                    failed,
                    total,
                    success: out.status.success(),
                    elapsed_ms: elapsed,
                    output: combined,
                }
            }
            Err(e) => TestResult {
                passed: 0,
                failed: 0,
                total: 0,
                success: false,
                elapsed_ms: elapsed,
                output: format!("Failed to run cargo test: {}", e),
            },
        }
    }

    /// Get the sandbox directory path.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Get current source code.
    pub fn current_code(&self) -> &str {
        &self.current_code
    }

    /// Convert execution results to metrics for rubric scoring.
    pub fn to_metrics(&self, check: &SandboxResult, test: &TestResult) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert(
            "compile_success".to_string(),
            if check.success { 1.0 } else { 0.0 },
        );
        m.insert("test_pass_rate".to_string(), test.pass_rate());
        m.insert("compile_time_ms".to_string(), check.elapsed_ms as f64);
        m.insert("test_time_ms".to_string(), test.elapsed_ms as f64);
        m.insert("code_length".to_string(), self.current_code.len() as f64);
        m
    }

    /// Cleanup the sandbox.
    pub fn cleanup(&self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

impl Drop for CodeSandbox {
    fn drop(&mut self) {
        self.cleanup();
    }
}

/// Parse cargo test output to extract pass/fail counts.
/// Aggregates across all test binaries (lib, integration, doc-tests).
fn parse_test_output(output: &str) -> (usize, usize, usize) {
    let mut total_passed = 0usize;
    let mut total_failed = 0usize;

    for line in output.lines() {
        if line.contains("test result:") {
            for part in line.split(';') {
                let part = part.trim();
                if part.contains("passed") {
                    if let Some(n) = part
                        .split_whitespace()
                        .find_map(|w| w.parse::<usize>().ok())
                    {
                        total_passed += n;
                    }
                }
                if part.contains("failed") && !part.contains("filtered") {
                    if let Some(n) = part
                        .split_whitespace()
                        .find_map(|w| w.parse::<usize>().ok())
                    {
                        total_failed += n;
                    }
                }
            }
        }
    }
    (total_passed, total_failed, total_passed + total_failed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sandbox_init_and_check() {
        let mut sandbox = CodeSandbox::new(9999);
        sandbox.init().expect("Failed to init sandbox");

        assert!(sandbox.dir().exists());
        assert!(sandbox.dir().join("Cargo.toml").exists());
        assert!(sandbox.dir().join("src/lib.rs").exists());

        // Compile — should succeed with default code
        let result = sandbox.check();
        println!(
            "  Sandbox check: success={}, elapsed={}ms",
            result.success, result.elapsed_ms
        );
        if !result.stderr.is_empty() {
            println!(
                "  stderr: {}",
                &result.stderr[..result.stderr.len().min(200)]
            );
        }
        assert!(result.success, "Default code should compile");
        assert!(result.compile_reward() > 0.0);
    }

    #[test]
    fn test_sandbox_broken_code() {
        let mut sandbox = CodeSandbox::new(9998);
        sandbox.init().expect("Failed to init sandbox");

        // Break the code
        sandbox
            .set_code("pub fn broken(x: i32) -> i32 { x + }")
            .unwrap();
        let result = sandbox.check();
        assert!(!result.success, "Broken code should fail to compile");
        assert!(result.compile_reward() < 0.0);
    }

    #[test]
    fn test_sandbox_test_execution() {
        let mut sandbox = CodeSandbox::new(9997);
        sandbox.init().expect("Failed to init sandbox");

        let result = sandbox.test();
        println!(
            "  Sandbox test: passed={}, failed={}, total={}, elapsed={}ms",
            result.passed, result.failed, result.total, result.elapsed_ms
        );
        assert!(result.success, "Default tests should pass");
        assert!(result.passed > 0);
        assert_eq!(result.failed, 0);
        assert!(result.test_reward() > 0.0);
    }

    #[test]
    fn test_sandbox_patch() {
        let mut sandbox = CodeSandbox::new(9996);
        sandbox.init().expect("Failed to init sandbox");

        // Apply a patch that breaks a function
        let patched = sandbox.apply_patch("a + b", "a - b").unwrap();
        assert!(patched, "Patch should apply");

        // Code should still compile but tests should fail
        let check = sandbox.check();
        assert!(check.success, "Patched code should still compile");

        let test = sandbox.test();
        println!(
            "  After patch: passed={}, failed={}",
            test.passed, test.failed
        );
        assert!(
            test.failed > 0,
            "add tests should fail after changing + to -"
        );
    }

    #[test]
    fn test_parse_test_output() {
        let output = "test result: ok. 5 passed; 2 failed; 0 ignored; 0 measured";
        let (p, f, t) = parse_test_output(output);
        assert_eq!(p, 5);
        assert_eq!(f, 2);
        assert_eq!(t, 7);
    }

    #[test]
    fn test_sandbox_cleanup() {
        let dir;
        {
            let mut sandbox = CodeSandbox::new(9995);
            sandbox.init().expect("Failed to init sandbox");
            dir = sandbox.dir().to_path_buf();
            assert!(dir.exists());
        }
        // After drop, directory should be cleaned up
        assert!(!dir.exists(), "Sandbox should clean up on drop");
    }
}
