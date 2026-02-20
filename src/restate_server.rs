//! Restate durable agent server.
//!
//! Separate binary that exposes the RLM agent as a Restate workflow service.
//! Restate connects to this HTTP endpoint and drives durable execution.
//!
//! ## Usage
//!
//! ```bash
//! # 1. Start this service
//! cargo run --bin restate-server
//!
//! # 2. Start Restate (separate terminal)
//! restate-server --dev
//!
//! # 3. Register this service with Restate
//! curl -X POST http://localhost:9070/deployments \
//!   -H 'Content-Type: application/json' \
//!   -d '{"uri": "http://localhost:9091"}'
//!
//! # 4. Invoke a task
//! curl -X POST http://localhost:8080/AgentWorkflow/task-1/run \
//!   -H 'Content-Type: application/json' \
//!   -d '{"task": "List 3 fruits"}'
//!
//! # 5. Check status
//! curl http://localhost:8080/AgentWorkflow/task-1/status
//! ```

use restate_sdk::prelude::*;
use tokio::signal;

use rig_rlm::monad::restate::{AgentWorkflow, AgentWorkflowImpl};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load .env
    dotenvy::dotenv().ok();

    // Initialize OTEL tracing (if configured)
    let _ = rig_rlm::monad::otel::init_tracing();

    eprintln!("🚀 Restate Agent Server starting on 0.0.0.0:9091");
    eprintln!(
        "   Register with: curl -X POST http://localhost:9070/deployments -H 'Content-Type: application/json' -d '{{\"uri\": \"http://localhost:9091\"}}'"
    );

    let endpoint = Endpoint::builder().bind(AgentWorkflowImpl.serve()).build();

    let server = HttpServer::new(endpoint).listen_and_serve("0.0.0.0:9091".parse().unwrap());

    tokio::pin!(server);

    tokio::select! {
        _ = signal::ctrl_c() => {
            eprintln!("\n⚡ Shutdown signal received");
        }
        _ = &mut server => {
            eprintln!("🔴 HTTP server stopped");
        }
    }

    // Flush OTEL spans
    rig_rlm::monad::otel::shutdown_tracing().await;

    eprintln!("👋 Restate Agent Server stopped");
    Ok(())
}
