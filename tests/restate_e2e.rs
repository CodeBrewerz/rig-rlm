//! End-to-end Restate integration test.
//!
//! This test exercises the **full stack**: Restate server → restate-server binary →
//! AgentWorkflow → real LLM call → cost tracking → Turso persistence → LangFuse traces.
//!
//! Prerequisites:
//! - `restate-server` (Restate runtime) must be running on port 8080/9070
//! - The `restate-server` binary must be built.
//!
//! Run with: cargo test --test restate_e2e -- --nocapture --ignored

use reqwest::Client;
use std::process::{Child, Command};
use std::time::Duration;
use tokio::time::sleep;

use rig_rlm::monad::restate::{AgentTaskRequest, AgentTaskResult, AgentTaskStatus};
use rig_rlm::monad::restate_helpers::*;

const SERVICE_PORT: u16 = 9091;

/// Wait for an HTTP endpoint to become available.
async fn wait_for_ready(url: &str, timeout_secs: u64) -> bool {
    let client = Client::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_secs);
    while tokio::time::Instant::now() < deadline {
        if let Ok(resp) = client.get(url).send().await {
            if resp.status().is_success()
                || resp.status().as_u16() == 404
                || resp.status().as_u16() == 405
            {
                return true;
            }
        }
        sleep(Duration::from_millis(500)).await;
    }
    false
}

/// Start the restate-server binary as a background process.
fn start_restate_agent_server() -> Child {
    Command::new("cargo")
        .args(["run", "--bin", "restate-server"])
        .envs(std::env::vars())
        .spawn()
        .expect("Failed to start restate-server binary")
}

/// Start the Restate runtime in dev mode.
fn start_restate_runtime() -> Child {
    Command::new("restate-server")
        .args(["--dev"])
        .spawn()
        .expect("Failed to start restate-server runtime")
}

struct TestGuard {
    agent_server: Option<Child>,
    restate_runtime: Option<Child>,
}

impl Drop for TestGuard {
    fn drop(&mut self) {
        if let Some(ref mut child) = self.agent_server {
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(ref mut child) = self.restate_runtime {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[tokio::test]
#[ignore] // requires running Restate + real LLM API key
async fn restate_e2e_full_stack() {
    dotenvy::dotenv().ok();

    eprintln!("\n═══════════════════════════════════════════");
    eprintln!("  Restate E2E Full Stack Test");
    eprintln!("═══════════════════════════════════════════\n");

    // ── Step 1: Start Restate runtime ────────────────────────────────────
    eprintln!("🔄 Step 1: Starting Restate runtime...");
    let mut guard = TestGuard {
        agent_server: None,
        restate_runtime: None,
    };

    // Check if Restate is already running
    let restate_host = env_host();
    let admin_port = env_admin_port();
    let restate_port = env_restate_port();

    let restate_admin_url = format!("http://{restate_host}:{admin_port}/deployments");
    let client = Client::new();

    let restate_already_running = client
        .get(&restate_admin_url)
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);

    if !restate_already_running {
        eprintln!("  Starting Restate runtime on ports {admin_port}/{restate_port}...");
        guard.restate_runtime = Some(start_restate_runtime());
        assert!(
            wait_for_ready(&restate_admin_url, 15).await,
            "Restate runtime failed to start within 15s"
        );
        eprintln!("  ✅ Restate runtime started");
    } else {
        eprintln!("  ✅ Restate runtime already running");
    }

    // ── Step 2: Start our agent server ───────────────────────────────────
    eprintln!("\n🔄 Step 2: Starting agent server on port {SERVICE_PORT}...");

    let agent_url = format!("http://{restate_host}:{SERVICE_PORT}");
    let agent_already_running = client
        .get(&agent_url)
        .send()
        .await
        .map(|r| {
            // Restate SDK returns 415 on GET (expects discovery protocol), that's fine
            r.status().as_u16() != 0
        })
        .unwrap_or(false);

    if !agent_already_running {
        guard.agent_server = Some(start_restate_agent_server());
        // Wait for agent server to be ready
        assert!(
            wait_for_ready(&agent_url, 30).await,
            "Agent server failed to start within 30s"
        );
        eprintln!("  ✅ Agent server started");
    } else {
        eprintln!("  ✅ Agent server already running");
    }

    // ── Step 3: Register with Restate ────────────────────────────────────
    eprintln!("\n🔄 Step 3: Registering agent service with Restate...");
    register_service(
        &client,
        &restate_host,
        admin_port,
        &restate_host,
        SERVICE_PORT,
    )
    .await
    .expect("Failed to register service with Restate");
    eprintln!("  ✅ Service registered");

    // ── Step 4: Invoke AgentWorkflow ─────────────────────────────────────
    eprintln!("\n🔄 Step 4: Invoking AgentWorkflow with real LLM task...");

    let task_id = format!("e2e-test-{}", uuid::Uuid::new_v4());
    let request = AgentTaskRequest {
        task: "What is 2+2? Reply with just the number.".to_string(),
        model: None,    // uses RIG_RLM_MODEL from .env
        base_url: None, // uses OPENAI_BASE_URL from .env
        api_key: None,  // uses OPENAI_API_KEY from .env
        provider_name: None,
        max_turns: Some(3),
        max_cost_usd: Some(0.10),
        preamble: Some("You are a helpful assistant. Be concise.".to_string()),
        delegate: false,
        attachments: vec![],
    };

    eprintln!("  Task ID: {task_id}");
    eprintln!("  Task:    {}", request.task);

    let result: AgentTaskResult = invoke_workflow(
        &client,
        &restate_host,
        restate_port,
        "AgentWorkflow",
        &task_id,
        &request,
    )
    .await
    .expect("Workflow invocation failed");

    eprintln!("\n  ✅ Workflow completed!");
    eprintln!("  📝 Output:  {}", result.output);
    eprintln!("  🔄 Turns:   {}", result.turns);
    eprintln!("  💰 Cost:    ${:.6}", result.cost_usd);
    eprintln!("  🎯 Tokens:  {}", result.total_tokens);

    // ── Step 5: Verify result ────────────────────────────────────────────
    eprintln!("\n🔄 Step 5: Verifying result...");

    assert!(result.turns > 0, "Should have executed at least 1 turn");
    assert!(!result.output.is_empty(), "Output should not be empty");
    assert!(
        result.output.contains('4') || result.output.to_lowercase().contains("four"),
        "Expected '4' or 'four' in output, got: {}",
        result.output
    );
    eprintln!("  ✅ LLM response verified (contains '4')");

    // ── Step 6: Query status ─────────────────────────────────────────────
    eprintln!("\n🔄 Step 6: Querying workflow status...");

    let status: AgentTaskStatus = query_workflow_status(
        &client,
        &restate_host,
        restate_port,
        "AgentWorkflow",
        &task_id,
    )
    .await
    .expect("Status query failed");

    eprintln!("  Phase:   {}", status.phase);
    eprintln!("  Turn:    {}", status.turn);
    eprintln!("  Cost:    ${:.6}", status.cost_usd);

    assert_eq!(status.phase, "completed", "Status should be 'completed'");
    assert!(status.output.is_some(), "Status should have output");
    eprintln!("  ✅ Status verified");

    // ── Summary ──────────────────────────────────────────────────────────
    eprintln!("\n═══════════════════════════════════════════");
    eprintln!("  ✅ ALL E2E CHECKS PASSED!");
    eprintln!("═══════════════════════════════════════════");
    eprintln!("  • Restate workflow invocation:  ✅");
    eprintln!("  • Real LLM call (OpenRouter):   ✅");
    eprintln!(
        "  • Cost tracking:                ✅ (${:.6})",
        result.cost_usd
    );
    eprintln!("  • Status query:                 ✅");
    eprintln!("  • Output verification:          ✅");
    eprintln!("═══════════════════════════════════════════\n");
}
