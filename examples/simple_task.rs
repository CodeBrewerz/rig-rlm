//! Simple task example — run a task via AgentContext.
//!
//! ```bash
//! cargo run --example simple_task
//! ```

use rig_rlm::monad::{AgentConfig, AgentContext, agent_task};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let config = AgentConfig::openai_compatible(
        "openrouter",
        "openai/gpt-4o-mini",
        "https://openrouter.ai/api/v1",
        std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY not set"),
    );

    let mut ctx = AgentContext::new(config);
    let monad = agent_task("What is 2 + 2? Just answer with the number.");

    match ctx.run(monad).await {
        Ok(answer) => println!("✅ Agent answered: {answer}"),
        Err(e) => eprintln!("❌ Agent error: {e}"),
    }

    Ok(())
}
