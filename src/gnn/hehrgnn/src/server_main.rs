//! GNN Prediction Server binary.
//!
//! Starts an HTTP server that serves predictions from the GNN platform.
//!
//! Usage:
//!   cargo run -p hehrgnn --bin hehrgnn-server -- [OPTIONS]
//!
//! Options:
//!   --port <PORT>       Server port (default: 3030)
//!   --schema <PATH>     TQL schema file path
//!   --hidden-dim <DIM>  GNN hidden dimension (default: 32)
//!   --num-facts <N>     Synthetic facts to generate (default: 200)

use axum::routing::{get, post};

use axum::Router;
use tower_http::cors::CorsLayer;

use hehrgnn::server::handlers;
use hehrgnn::server::state::{AppState, ServerConfig};

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut config = ServerConfig::default();
    let mut port: u16 = 3030;

    // Simple arg parsing
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                i += 1;
                port = args[i].parse().expect("Invalid port");
            }
            "--schema" => {
                i += 1;
                config.schema_path = Some(args[i].clone());
            }
            "--hidden-dim" => {
                i += 1;
                config.hidden_dim = args[i].parse().expect("Invalid hidden-dim");
            }
            "--num-facts" => {
                i += 1;
                config.num_facts = args[i].parse().expect("Invalid num-facts");
            }
            "--instances-per-type" => {
                i += 1;
                config.instances_per_type = args[i].parse().expect("Invalid instances-per-type");
            }
            "--num-classes" => {
                i += 1;
                config.num_classes = args[i].parse().expect("Invalid num-classes");
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
            }
        }
        i += 1;
    }

    println!();
    println!("  ╔══════════════════════════════════════════════╗");
    println!("  ║       GNN Prediction Server                  ║");
    println!("  ╚══════════════════════════════════════════════╝");
    println!();
    println!("  Initializing GNN platform...");
    println!("    Hidden dim:    {}", config.hidden_dim);
    println!("    Num facts:     {}", config.num_facts);
    println!(
        "    Schema:        {}",
        config.schema_path.as_deref().unwrap_or("(built-in)")
    );
    println!();

    let state = AppState::init(&config);

    let app = Router::new()
        .route("/health", get(handlers::health))
        .route("/graph/info", get(handlers::graph_info))
        .route("/embeddings", post(handlers::get_embedding))
        .route("/match/rank", post(handlers::rank_matches))
        .route("/classify", post(handlers::classify_nodes))
        .route("/categorize", post(handlers::categorize_transaction))
        .route("/anomaly/score", post(handlers::score_anomalies))
        .route("/similarity/search", post(handlers::similarity_search))
        .route("/fiduciary/actions", post(handlers::fiduciary_next_actions))
        .route("/fiduciary/reward", post(handlers::fiduciary_reward))
        .route("/critical-path", post(handlers::critical_path))
        .route("/checkpoints", get(handlers::list_checkpoints_handler))
        .route("/retrain", post(handlers::retrain))
        .route("/predict/pql", post(handlers::predict_pql))
        .route("/explain", post(handlers::explain_link))
        .route("/graph/mutate", post(handlers::graph_mutate))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let bind = format!("0.0.0.0:{}", port);
    println!("  Server listening on http://{}", bind);
    println!();
    println!("  Endpoints:");
    println!("    GET  /health              — server health + graph stats");
    println!("    GET  /graph/info          — node/edge type details");
    println!("    POST /embeddings          — get node embedding vector");
    println!("    POST /match/rank          — rank match candidates (link prediction)");
    println!("    POST /classify            — classify nodes (category/tax code)");
    println!(
        "    POST /categorize          — transaction category prediction (TopK-NN + GNN link prediction)"
    );
    println!("    POST /anomaly/score       — anomaly scores for nodes");
    println!("    POST /similarity/search   — kNN similarity search");
    println!("    POST /fiduciary/actions   — fiduciary next-action predictions (with RL scorer)");
    println!("    POST /fiduciary/reward    — submit reward feedback for RL learning");
    println!("    POST /critical-path       — critical financial dependency analysis");
    println!("    GET  /checkpoints         — list saved model checkpoints");
    println!("    POST /retrain             — retrain all models + save checkpoints");
    println!("    POST /predict/pql         — PQL predictive query (PREDICT ... FOR ... VIA ...)");
    println!("    POST /explain             — feature importance explainability");
    println!("    POST /graph/mutate        — incremental graph update (InstantGNN)");
    println!();

    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .expect("Failed to bind");

    axum::serve(listener, app).await.expect("Server failed");
}
