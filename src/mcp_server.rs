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
use std::sync::{Arc, Mutex};

use crate::nuggets::NuggetShelf;

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

// ── Nuggets param types ──────────────────────────────────────────

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct NuggetsRememberParams {
    /// Topic nugget name (e.g. "prefs", "locations", "debug"). Auto-created if new.
    pub nugget: String,
    /// Key for the fact (e.g. "test_cmd", "auth_handler").
    pub key: String,
    /// Value to store (keep it short — one sentence max).
    pub value: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct NuggetsRecallParams {
    /// Query to search for (fuzzy-matched against stored keys).
    pub query: String,
    /// Optional: restrict search to a specific nugget.
    pub nugget: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct NuggetsForgetParams {
    /// Nugget name containing the fact to remove.
    pub nugget: String,
    /// Key to remove.
    pub key: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
pub struct NuggetsFactsParams {
    /// Nugget name to list facts from.
    pub nugget: String,
}

// ── MCP Server ────────────────────────────────────────────────────

/// The MCP server struct — holds the tool router and working directory.
#[derive(Clone)]
pub struct AgentMcpServer {
    /// Working directory for file operations.
    work_dir: PathBuf,
    /// Auto-generated tool router.
    tool_router: ToolRouter<Self>,
    /// Disk-backed memory shelf with LRU cache (thread-safe).
    shelf: Arc<Mutex<NuggetShelf>>,
}

#[tool_router]
impl AgentMcpServer {
    /// Create a new MCP server with the given working directory.
    pub fn new(work_dir: PathBuf) -> Self {
        let mut shelf = NuggetShelf::new(None, true);
        shelf.load_all();
        Self {
            work_dir,
            tool_router: Self::tool_router(),
            shelf: Arc::new(Mutex::new(shelf)),
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
        let url = format!("http://localhost:18080/AgentWorkflow/{key}/run");

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

    // ── Nuggets tools ─────────────────────────────────────────────

    /// Store a key-value fact in persistent memory.
    #[tool(
        description = "Store a key-value fact in persistent memory. Facts persist across sessions. Auto-creates the nugget if it doesn't exist. Keep values short (one sentence max). Use nugget names like 'prefs', 'locations', 'debug', 'project'."
    )]
    async fn nuggets_remember(
        &self,
        params: Parameters<NuggetsRememberParams>,
    ) -> Result<CallToolResult, McpError> {
        let mut shelf = self.shelf.lock().unwrap();
        let nugget = shelf.get_or_create(&params.0.nugget);
        nugget.remember(&params.0.key, &params.0.value);
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Remembered in '{}': {} → {}",
            params.0.nugget, params.0.key, params.0.value
        ))]))
    }

    /// Search persistent memory for a fact.
    #[tool(
        description = "Search persistent memory for a fact. Use BEFORE expensive searches (grep, file reads, codebase search). Returns the best matching answer with confidence score. Optionally restrict to a specific nugget."
    )]
    async fn nuggets_recall(
        &self,
        params: Parameters<NuggetsRecallParams>,
    ) -> Result<CallToolResult, McpError> {
        let mut shelf = self.shelf.lock().unwrap();
        let result = shelf.recall(
            &params.0.query,
            params.0.nugget.as_deref(),
            "", // no session tracking via MCP
        );

        if result.result.found {
            let json = serde_json::json!({
                "found": true,
                "answer": result.result.answer,
                "confidence": result.result.confidence,
                "margin": result.result.margin,
                "key": result.result.key,
                "nugget": result.nugget_name,
            });
            Ok(CallToolResult::success(vec![Content::text(
                serde_json::to_string_pretty(&json).unwrap_or_default(),
            )]))
        } else {
            Ok(CallToolResult::success(vec![Content::text(
                "Not found in memory.".to_string(),
            )]))
        }
    }

    /// Remove a fact from persistent memory.
    #[tool(
        description = "Remove a specific fact from a nugget in persistent memory."
    )]
    async fn nuggets_forget(
        &self,
        params: Parameters<NuggetsForgetParams>,
    ) -> Result<CallToolResult, McpError> {
        let mut shelf = self.shelf.lock().unwrap();
        if !shelf.has(&params.0.nugget) {
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "Nugget '{}' not found.",
                params.0.nugget
            ))]));
        }
        let removed = shelf.forget(&params.0.nugget, &params.0.key);
        if removed {
            Ok(CallToolResult::success(vec![Content::text(format!(
                "Removed '{}' from nugget '{}'.",
                params.0.key, params.0.nugget
            ))]))
        } else {
            Ok(CallToolResult::success(vec![Content::text(format!(
                "Key '{}' not found in nugget '{}'.",
                params.0.key, params.0.nugget
            ))]))
        }
    }

    /// List all nuggets and their status.
    #[tool(
        description = "List all nuggets in persistent memory with fact counts and capacity info."
    )]
    async fn nuggets_list(&self) -> Result<CallToolResult, McpError> {
        let mut shelf = self.shelf.lock().unwrap();
        let statuses = shelf.list();
        if statuses.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No nuggets stored yet.".to_string(),
            )]));
        }
        let json = serde_json::to_string_pretty(&statuses).unwrap_or_default();
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    /// List all facts in a specific nugget.
    #[tool(
        description = "List all facts stored in a specific nugget. Shows keys, values, and hit counts."
    )]
    async fn nuggets_facts(
        &self,
        params: Parameters<NuggetsFactsParams>,
    ) -> Result<CallToolResult, McpError> {
        let mut shelf = self.shelf.lock().unwrap();
        if !shelf.has(&params.0.nugget) {
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "Nugget '{}' not found.",
                params.0.nugget
            ))]));
        }
        let facts = shelf.get(&params.0.nugget).facts();
        if facts.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "Nugget '{}' has no facts.",
                params.0.nugget
            ))]));
        }
        let json = serde_json::to_string_pretty(&facts).unwrap_or_default();
        Ok(CallToolResult::success(vec![Content::text(json)]))
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
                     configurable execution policy, and holographic memory (nuggets)."
                        .into(),
                ),
                icons: None,
                website_url: None,
            },
            instructions: Some(
                "Use run_task for full agent workflows, execute_python for \
                 direct code execution, apply_patch for file editing, \
                 check_policy to validate commands, and nuggets_* tools for \
                 persistent holographic memory (recall before searching!)."
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
        // Nuggets tools
        for name in &[
            "nuggets_remember",
            "nuggets_recall",
            "nuggets_forget",
            "nuggets_list",
            "nuggets_facts",
        ] {
            assert!(
                tool_names.iter().any(|n| n == name),
                "should have {name}: {tool_names:?}"
            );
        }
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
