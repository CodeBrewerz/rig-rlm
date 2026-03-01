//! A2A Server binary — runs rig-rlm as an A2A HTTP agent.
//!
//! Usage:
//!   cargo run --bin a2a-server
//!   cargo run --bin a2a-server -- --port 9999
//!
//! Then use agentgateway to proxy:
//!   agentgateway -f agentgateway-a2a.yaml

use clap::Parser;
use tracing_subscriber::{EnvFilter, fmt::Layer, layer::SubscriberExt, util::SubscriberInitExt};
use tracing::level_filters::LevelFilter;
use std::fs::OpenOptions;

use rig_rlm::a2a_server;
use rig_rlm::monad::AgentConfig;

#[derive(Parser)]
#[command(name = "a2a-server", about = "rig-rlm A2A agent server")]
struct Args {
    /// Port to listen on.
    #[arg(long, default_value_t = 9999)]
    port: u16,

    /// LLM model to use (or set RIG_RLM_MODEL env var).
    #[arg(long, default_value = "arcee-ai/trinity-large-preview:free")]
    model: String,

    /// Maximum agent turns per task.
    #[arg(long, default_value_t = 25)]
    max_turns: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load .env
    dotenvy::dotenv().ok();

    // Set up file + console tracing so we can debug agent execution
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("a2a-server.log")
        .expect("Failed to open a2a-server.log");

    tracing_subscriber::registry()
        .with(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy(),
        )
        .with(Layer::new().with_writer(std::io::stderr))
        .with(
            Layer::new()
                .with_writer(std::sync::Mutex::new(log_file))
                .with_ansi(false),
        )
        .init();

    // Initialize OTEL (no-op if keys not set)
    if let Err(e) = rig_rlm::monad::otel::init_tracing() {
        eprintln!("⚠️  OTEL init failed (continuing without tracing): {e}");
    }

    let args = Args::parse();

    // Build provider config from env
    let model = std::env::var("RIG_RLM_MODEL").unwrap_or_else(|_| args.model.clone());
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    // Auto-detect OpenRouter keys (sk-or-*) and set correct base URL
    let base_url = std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| {
        if api_key.starts_with("sk-or-") {
            "https://openrouter.ai/api/v1".to_string()
        } else {
            "https://api.openai.com/v1".to_string()
        }
    });

    let provider = if api_key.is_empty() {
        eprintln!("🏠 No OPENAI_API_KEY set — using local LLM at localhost:1234");
        rig_rlm::monad::provider::ProviderConfig::local(model)
    } else {
        eprintln!("🔑 Using API key with base URL: {base_url}");
        rig_rlm::monad::provider::ProviderConfig::openai_compatible(
            "openai", model, &base_url, &api_key,
        )
    };

    let agent_config = AgentConfig {
        max_turns: args.max_turns,
        provider,
        ..AgentConfig::default()
    };

    eprintln!("═══════════════════════════════════════════");
    eprintln!("  rig-rlm — A2A Agent Server");
    eprintln!("  Port: {}", args.port);
    eprintln!("  Max turns: {}", args.max_turns);
    eprintln!("═══════════════════════════════════════════\n");

    a2a_server::start_a2a_server(args.port, agent_config).await?;

    // Flush OTEL spans
    rig_rlm::monad::otel::shutdown_tracing().await;

    Ok(())
}
