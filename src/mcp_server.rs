//! MCP Server — expose agent tools via Model Context Protocol.
//!
//! Uses the `rmcp` crate to serve tools over stdio or HTTP so any
//! MCP client (Claude Desktop, VS Code, Cursor, etc.) can call them.

use rmcp::{
    ErrorData as McpError, ServerHandler, handler::server::tool::ToolRouter,
    handler::server::wrapper::Parameters, model::*, schemars::JsonSchema, tool, tool_handler,
    tool_router,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ── Parameter types ───────────────────────────────────────────────

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct TaskParams {
    /// The task or question for the agent to solve.
    pub task: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct CodeParams {
    /// Python code to execute in the sandbox.
    pub code: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct PatchParams {
    /// Unified diff patch content (--- a/path, +++ b/path, @@ hunks).
    pub patch: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct PolicyParams {
    /// Command or code snippet to check against the execution policy.
    pub command: String,
}

// ── MCP Server ────────────────────────────────────────────────────

/// The MCP server struct — holds the tool router and working directory.
#[derive(Clone)]
pub struct AgentMcpServer {
    /// Working directory for file operations.
    work_dir: PathBuf,
    /// Auto-generated tool router.
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl AgentMcpServer {
    /// Create a new MCP server with the given working directory.
    pub fn new(work_dir: PathBuf) -> Self {
        Self {
            work_dir,
            tool_router: Self::tool_router(),
        }
    }

    /// Run a full agent task through the Restate-backed agent workflow.
    ///
    /// Delegates to the running Restate server (localhost:8080) so the
    /// task gets full durable execution, lifecycle hooks, and persistence.
    #[tool(
        description = "Run a task through the AI agent (LLM + code execution loop via Restate). Returns the agent's final answer."
    )]
    async fn run_task(&self, params: Parameters<TaskParams>) -> Result<CallToolResult, McpError> {
        let task = &params.0.task;
        let key = format!("mcp-{}", uuid::Uuid::new_v4());
        let url = format!("http://localhost:8080/AgentWorkflow/{key}/run");

        let client = reqwest::Client::new();
        let body = serde_json::json!({ "task": task });

        match client
            .post(&url)
            .json(&body)
            .timeout(std::time::Duration::from_secs(120))
            .send()
            .await
        {
            Ok(resp) => {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                if status.is_success() {
                    // Try to extract the "output" field from JSON response
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                        if let Some(output) = json.get("output").and_then(|v| v.as_str()) {
                            return Ok(CallToolResult::success(vec![Content::text(
                                output.to_string(),
                            )]));
                        }
                    }
                    Ok(CallToolResult::success(vec![Content::text(text)]))
                } else {
                    Ok(CallToolResult::error(vec![Content::text(format!(
                        "Agent returned HTTP {status}: {text}"
                    ))]))
                }
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to reach agent server: {e}. Is `restate-server` running?"
            ))])),
        }
    }

    /// Execute Python code in the sandbox and return the output.
    #[tool(
        description = "Execute Python code in the sandbox. Returns stdout/stderr output. Code is checked against the execution policy first."
    )]
    async fn execute_python(
        &self,
        params: Parameters<CodeParams>,
    ) -> Result<CallToolResult, McpError> {
        // Use the exec policy to check the code first
        let policy = crate::exec_policy::ExecPolicy::standard();
        let eval = policy.evaluate(&params.0.code);

        if eval.is_denied() {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Execution denied by policy: {}",
                eval.reason()
            ))]));
        }

        // Execute via the REPL
        let repl = crate::repl::REPL::new();
        let command = crate::repl::Command::RunCode(params.0.code.clone());
        match repl.run_command(command) {
            Ok(output) => Ok(CallToolResult::success(vec![Content::text(output)])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Execution error: {e}"
            ))])),
        }
    }

    /// Apply a unified diff patch to files on disk.
    #[tool(
        description = "Apply a unified diff patch to files. Use standard unified diff format with --- a/path and +++ b/path headers."
    )]
    async fn apply_patch(
        &self,
        params: Parameters<PatchParams>,
    ) -> Result<CallToolResult, McpError> {
        // Check with exec policy first
        let policy = crate::exec_policy::ExecPolicy::standard();
        let eval = policy.evaluate(&params.0.patch);
        if eval.is_denied() {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Patch denied by policy: {}",
                eval.reason()
            ))]));
        }

        // Parse and apply
        match crate::apply_patch::parse_patch(&params.0.patch) {
            Ok(action) => match crate::apply_patch::apply_patch(&action, &self.work_dir) {
                Ok(results) => {
                    let summary: Vec<String> = results
                        .iter()
                        .map(|(path, desc)| format!("{}: {}", path.display(), desc))
                        .collect();
                    Ok(CallToolResult::success(vec![Content::text(
                        summary.join("\n"),
                    )]))
                }
                Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to apply patch: {e}"
                ))])),
            },
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to parse patch: {e}"
            ))])),
        }
    }

    /// Check if a command is allowed by the execution policy.
    #[tool(
        description = "Check if a shell command or code snippet is allowed by the execution policy. Returns the policy decision (allow/deny/review) and matching rules."
    )]
    async fn check_policy(
        &self,
        params: Parameters<PolicyParams>,
    ) -> Result<CallToolResult, McpError> {
        let policy = crate::exec_policy::ExecPolicy::standard();
        let eval = policy.evaluate(&params.0.command);
        let result = format!(
            "Decision: {}\nAllowed: {}\nReason: {}",
            eval.decision,
            eval.is_allowed(),
            eval.reason()
        );
        Ok(CallToolResult::success(vec![Content::text(result)]))
    }
}

// ── ServerHandler implementation ──────────────────────────────────

#[tool_handler]
impl ServerHandler for AgentMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2025_03_26,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "rig-rlm".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                title: Some("RIG-RLM Agent".into()),
                description: Some(
                    "AI agent with Python sandbox, unified diff apply-patch, \
                     and configurable execution policy."
                        .into(),
                ),
                icons: None,
                website_url: None,
            },
            instructions: Some(
                "Use run_task for full agent workflows, execute_python for \
                 direct code execution, apply_patch for file editing, and \
                 check_policy to validate commands."
                    .into(),
            ),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper — extract text from the first Content in a CallToolResult.
    fn extract_text(result: &CallToolResult) -> String {
        let json = serde_json::to_value(&result.content[0]).unwrap();
        json.get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string()
    }

    #[test]
    fn server_has_tools() {
        let server = AgentMcpServer::new(PathBuf::from("/tmp"));
        let tools = server.tool_router.list_all();
        let tool_names: Vec<String> = tools.iter().map(|t| t.name.to_string()).collect();
        assert!(
            tool_names.iter().any(|n| n == "run_task"),
            "should have run_task: {tool_names:?}"
        );
        assert!(
            tool_names.iter().any(|n| n == "execute_python"),
            "should have execute_python: {tool_names:?}"
        );
        assert!(
            tool_names.iter().any(|n| n == "apply_patch"),
            "should have apply_patch: {tool_names:?}"
        );
        assert!(
            tool_names.iter().any(|n| n == "check_policy"),
            "should have check_policy: {tool_names:?}"
        );
    }

    #[tokio::test]
    async fn check_policy_allows_safe_command() {
        let server = AgentMcpServer::new(PathBuf::from("/tmp"));
        let result = server
            .check_policy(Parameters(PolicyParams {
                command: "print('hello')".to_string(),
            }))
            .await
            .unwrap();
        let text = extract_text(&result);
        assert!(
            text.contains("Allowed: true"),
            "safe command should be allowed: {text}"
        );
    }

    #[tokio::test]
    async fn check_policy_denies_dangerous_command() {
        let server = AgentMcpServer::new(PathBuf::from("/tmp"));
        let result = server
            .check_policy(Parameters(PolicyParams {
                command: "rm -rf /".to_string(),
            }))
            .await
            .unwrap();
        let text = extract_text(&result);
        assert!(
            text.contains("Allowed: false"),
            "dangerous command should be denied: {text}"
        );
    }

    #[tokio::test]
    async fn apply_patch_creates_file() {
        let dir = std::env::temp_dir().join("rig_mcp_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let server = AgentMcpServer::new(dir.clone());
        let result = server
            .apply_patch(Parameters(PatchParams {
                patch: "--- /dev/null\n+++ b/hello.txt\n@@ -0,0 +1 @@\n+hello world\n".to_string(),
            }))
            .await
            .unwrap();
        let text = extract_text(&result);
        assert!(
            text.contains("created"),
            "should report file created: {text}"
        );
        assert!(dir.join("hello.txt").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
