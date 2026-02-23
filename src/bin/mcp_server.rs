//! MCP Server binary — runs the agent as an MCP server over stdio.
//!
//! Usage:
//!   cargo run --bin mcp-server
//!
//! Configure in Claude Desktop's `claude_desktop_config.json`:
//! ```json
//! {
//!   "mcpServers": {
//!     "rig-rlm": {
//!       "command": "/path/to/mcp-server"
//!     }
//!   }
//! }
//! ```

use rig_rlm::mcp_server::AgentMcpServer;
use rmcp::ServiceExt;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Init tracing (stderr only — stdout is reserved for MCP JSON-RPC)
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "rig_rlm=info".into()),
        )
        .init();

    let work_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    eprintln!("🔌 MCP Server starting on stdio (work_dir: {})", work_dir.display());

    let server = AgentMcpServer::new(work_dir);
    let service = server
        .serve(rmcp::transport::stdio())
        .await
        .inspect_err(|e| {
            eprintln!("❌ Error starting MCP server: {e}");
        })?;

    eprintln!("✅ MCP Server running — tools: run_task, execute_python, apply_patch, check_policy");

    service.waiting().await?;

    Ok(())
}
