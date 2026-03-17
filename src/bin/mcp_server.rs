//! MCP Server binary — runs the agent as an MCP server over stdio or HTTP.
//!
//! ## Modes
//!
//! ### stdio (default — for MCP client subprocess spawning)
//!   cargo run --bin mcp-server
//!
//! ### HTTP (for persistent/systemd deployment)
//!   cargo run --bin mcp-server -- --http --port 3100
//!
//! ## Claude Desktop config
//! ```json
//! {
//!   "mcpServers": {
//!     "rig-rlm": {
//!       "command": "/path/to/mcp-server"
//!     }
//!   }
//! }
//! ```
//!
//! ## Streamable HTTP config (Cursor, VS Code, etc.)
//! ```json
//! {
//!   "mcpServers": {
//!     "rig-rlm": {
//!       "url": "http://localhost:3100/mcp"
//!     }
//!   }
//! }
//! ```

use clap::Parser;
use rig_rlm::mcp_server::AgentMcpServer;
use rmcp::ServiceExt;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "mcp-server", about = "RIG-RLM MCP Server")]
struct Cli {
    /// Run in HTTP mode instead of stdio.
    #[arg(long)]
    http: bool,

    /// HTTP listen port (only for --http mode).
    #[arg(long, default_value = "4100")]
    port: u16,

    /// HTTP listen host.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Disable session tracking (each POST is independent).
    #[arg(long, default_value = "true")]
    stateless: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Init tracing (stderr only — stdout is reserved for MCP JSON-RPC in stdio mode)
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "rig_rlm=info".into()),
        )
        .init();

    let work_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    if cli.http {
        run_http(cli.host, cli.port, work_dir, cli.stateless).await
    } else {
        run_stdio(work_dir).await
    }
}

async fn run_stdio(work_dir: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!(
        "🔌 MCP Server starting on stdio (work_dir: {})",
        work_dir.display()
    );

    let server = AgentMcpServer::new(work_dir);
    let service = server
        .serve(rmcp::transport::stdio())
        .await
        .inspect_err(|e| {
            eprintln!("❌ Error starting MCP server: {e}");
        })?;

    eprintln!("✅ MCP Server running on stdio — tools: run_task, execute_python, apply_patch, check_policy, nuggets_*");

    service.waiting().await?;
    Ok(())
}

async fn run_http(
    host: String,
    port: u16,
    work_dir: PathBuf,
    stateless: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use rmcp::transport::{StreamableHttpServerConfig, StreamableHttpService};
    use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
    use std::sync::Arc;

    let bind_addr = format!("{host}:{port}");
    let mode_label = if stateless { "stateless" } else { "stateful" };

    eprintln!(
        "🌐 MCP Server starting on http://{bind_addr}/mcp ({mode_label}, work_dir: {})",
        work_dir.display()
    );

    let mut config = StreamableHttpServerConfig::default();
    config.stateful_mode = !stateless;
    let cancel = config.cancellation_token.clone();

    let session_manager = Arc::new(LocalSessionManager::default());
    let work_dir_clone = work_dir.clone();
    let mcp_service = StreamableHttpService::new(
        move || Ok(AgentMcpServer::new(work_dir_clone.clone())),
        session_manager,
        config,
    );

    // Build an axum router that nests the MCP service at /mcp
    let app = axum::Router::new().nest_service("/mcp", mcp_service);

    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    eprintln!("✅ MCP Server listening on http://{bind_addr}/mcp");
    eprintln!("   Tools: run_task, execute_python, apply_patch, check_policy, nuggets_*");

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            cancel.cancelled().await;
            eprintln!("🛑 MCP Server shutting down...");
        })
        .await?;

    Ok(())
}
