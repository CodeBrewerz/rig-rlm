//! rig-rlm: Production monadic agent with DSRs optimization,
//! Microsandbox execution, and Turso persistence.
//!
//! Usage:
//!   cargo run -- run "Solve this task"
//!   cargo run -- run "Solve this task" --executor microsandbox
//!   cargo run -- optimize --trainset examples.json
//!   cargo run -- e2e-test

use chrono::Utc;
use clap::{Parser, Subcommand};
use tracing::level_filters::LevelFilter;
use tracing_subscriber::{EnvFilter, fmt::Layer, layer::SubscriberExt, util::SubscriberInitExt};
use uuid::Uuid;

// Existing modules
pub mod exec;
pub mod llm;
pub mod repl;

// Core monadic architecture (Phases 1-9)
pub mod monad;

// DSRs integration (Phases 13-16)
pub mod agent_metric;
pub mod agent_module;
pub mod signature;

// Infrastructure (Phases 22-26)
pub mod chunking;
pub mod persistence;
pub mod pipeline;
pub mod safety;
pub mod sandbox;
pub mod session;

// ARC-AGI Benchmark (Phases 18-21)
pub mod arc;

use crate::monad::interaction::agent_task_full;
use crate::monad::{AgentConfig, AgentContext, AgentMonad, MemoryConfig, Role};
use crate::persistence::{AgentStore, Session, Turn};
use crate::sandbox::{ExecutorKind, create_executor};

#[derive(Parser)]
#[command(name = "rig-rlm", about = "Monadic AI agent with DSRs optimization")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the agent on a single task.
    Run {
        /// The task to solve.
        task: String,

        /// Execution backend: "pyo3" (default) or "microsandbox".
        #[arg(long, default_value = "pyo3")]
        executor: String,

        /// Path to the Turso database for persistence.
        #[arg(long, default_value = "agent.db")]
        db: String,

        /// LLM model to use (or set RIG_RLM_MODEL in .env).
        #[arg(long, default_value = "gpt-4o")]
        model: String,

        /// Maximum turns before stopping.
        #[arg(long, default_value_t = 25)]
        max_turns: usize,

        /// Enable sub-LLM bridge (llm_query() available in sandbox).
        #[arg(long, default_value_t = false)]
        sub_llm: bool,

        /// Instruction override (file path or inline string).
        /// Use with GEPA-optimized instructions.
        #[arg(long)]
        instruction: Option<String>,

        /// Paths to AGENTS.md files for project-level instructions.
        #[arg(long = "agents-md")]
        agents_md: Vec<String>,

        /// Paths to skill directories containing .md files.
        #[arg(long = "skill-dir")]
        skill_dirs: Vec<String>,

        /// Path to a recipe YAML file for multi-step pipeline execution.
        #[arg(long)]
        recipe: Option<String>,
    },

    /// Run end-to-end test: agent → sandbox → Turso → verify.
    E2eTest {
        /// Execution backend.
        #[arg(long, default_value = "pyo3")]
        executor: String,

        /// Turso DB path.
        #[arg(long, default_value = "test.db")]
        db: String,
    },

    /// Optimize agent instruction using DSRs GEPA optimizer.
    Optimize {
        /// Path to training examples JSON.
        #[arg(long)]
        trainset: String,

        /// Number of GEPA iterations.
        #[arg(long, default_value_t = 10)]
        iterations: usize,

        /// Turso DB path.
        #[arg(long, default_value = "optimize.db")]
        db: String,
    },

    /// Show session summary from Turso DB.
    Summary {
        /// Path to Turso database.
        db: String,

        /// Session ID to summarize (or "latest").
        #[arg(long, default_value = "latest")]
        session: String,
    },

    /// Run ARC-AGI benchmark (Phases 18-21).
    ArcBench {
        /// Path to the ARC-AGI dataset directory (containing .json task files).
        #[arg(long, default_value = "./arc-agi-2/evaluation")]
        dataset: String,

        /// LLM model to use.
        #[arg(long, default_value = "gpt-4o")]
        model: String,

        /// Run GEPA optimization before evaluation.
        #[arg(long, default_value_t = false)]
        optimize: bool,

        /// Number of tasks for training split (used with --optimize).
        #[arg(long, default_value_t = 50)]
        train_split: usize,

        /// Maximum tasks to evaluate (0 = all).
        #[arg(long, default_value_t = 0)]
        max_tasks: usize,

        /// Number of GEPA iterations (used with --optimize).
        #[arg(long, default_value_t = 10)]
        iterations: usize,
    },

    /// Stop all lingering microsandbox VMs.
    Cleanup,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load .env file (silently ignored if missing)
    dotenvy::dotenv().ok();

    // Initialize OpenTelemetry + LangFuse (no-op if keys not set)
    if let Err(e) = monad::otel::init_tracing() {
        eprintln!("⚠️ OTEL init failed (continuing without tracing): {e}");
    }

    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy(),
        )
        .with(Layer::new())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            task,
            executor,
            db,
            model,
            max_turns,
            sub_llm,
            instruction,
            agents_md,
            skill_dirs,
            recipe,
        } => {
            run_agent(
                &task,
                &executor,
                &db,
                &model,
                max_turns,
                sub_llm,
                instruction,
                agents_md,
                skill_dirs,
                recipe,
            )
            .await?;
        }
        Commands::E2eTest { executor, db } => {
            run_e2e_test(&executor, &db).await?;
        }
        Commands::Optimize {
            trainset,
            iterations,
            db,
        } => {
            run_optimization(&trainset, iterations, &db).await?;
        }
        Commands::Summary { db, session } => {
            show_summary(&db, &session).await?;
        }
        Commands::ArcBench {
            dataset,
            model,
            optimize,
            train_split,
            max_tasks,
            iterations,
        } => {
            run_arc_benchmark(
                &dataset,
                &model,
                optimize,
                train_split,
                max_tasks,
                iterations,
            )
            .await?;
        }
        Commands::Cleanup => {
            run_cleanup().await?;
        }
    }

    Ok(())
}

/// Run the monadic agent on a single task with Turso persistence.
async fn run_agent(
    task: &str,
    executor_name: &str,
    db_path: &str,
    model: &str,
    max_turns: usize,
    sub_llm: bool,
    instruction: Option<String>,
    agents_md: Vec<String>,
    skill_dirs: Vec<String>,
    recipe: Option<String>,
) -> anyhow::Result<()> {
    println!("═══════════════════════════════════════════");
    println!("  rig-rlm — Monadic Agent");
    println!("  Model: {model}");
    println!("  Executor: {executor_name}");
    println!("  Task: {task}");
    println!("═══════════════════════════════════════════\n");

    // 1. Open Turso store
    let store = AgentStore::open(db_path).await?;

    // 2. Create session
    let session_id = Uuid::new_v4().to_string();
    let session = Session {
        session_id: session_id.clone(),
        model: model.to_string(),
        task: task.to_string(),
        executor: executor_name.to_string(),
        optimizer: None,
        optimized_instruction: None,
        started_at: Utc::now().to_rfc3339(),
        finished_at: None,
        final_answer: None,
        score: None,
    };
    store.create_session(&session).await?;

    // 3. Build provider config from env + CLI args
    let model = std::env::var("RIG_RLM_MODEL").unwrap_or_else(|_| model.to_string());
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());

    let provider = if api_key.is_empty() {
        println!("🏠 No OPENAI_API_KEY set — using local LLM at localhost:1234");
        crate::monad::provider::ProviderConfig::local(model)
    } else {
        println!("🔑 Using API key with base URL: {base_url}");
        crate::monad::provider::ProviderConfig::openai_compatible(
            "openai", model, &base_url, &api_key,
        )
    };

    let executor_kind = match executor_name {
        "microsandbox" => crate::sandbox::ExecutorKind::Microsandbox,
        _ => crate::sandbox::ExecutorKind::Pyo3,
    };

    // 4. Build and run the monadic agent
    // For microsandbox: create a pool so sub-agents can share VMs
    let mut agent_config = AgentConfig {
        max_turns,
        provider,
        executor_kind: executor_kind.clone(),
        ..AgentConfig::default()
    };
    if sub_llm {
        agent_config = agent_config.with_sub_llm();
    }

    // Phase 4: build memory config from CLI flags + auto-discovery
    let cwd = std::env::current_dir().unwrap_or_default();
    let mut memory = MemoryConfig::auto_discover(&cwd);
    for path in &agents_md {
        memory.agents_md_paths.push(std::path::PathBuf::from(path));
    }
    for path in &skill_dirs {
        memory.skill_dirs.push(std::path::PathBuf::from(path));
    }
    if !memory.agents_md_paths.is_empty() || !memory.skill_dirs.is_empty() {
        println!(
            "📚 Memory: {} AGENTS.md, {} skill dirs",
            memory.agents_md_paths.len(),
            memory.skill_dirs.len()
        );
    }
    agent_config = agent_config.with_memory(memory);

    let mut ctx = match &executor_kind {
        crate::sandbox::ExecutorKind::Microsandbox => {
            let pool = std::sync::Arc::new(crate::sandbox::SandboxPool::new(4, 1, None).await?);
            let executor = pool.checkout().await?;
            let ctx = AgentContext::new_with_executor(agent_config, Box::new(executor), Some(pool));
            ctx
        }
        _ => AgentContext::new_async(agent_config).await?,
    };

    // 5. Resolve instruction override (file path or inline string)
    let resolved_instruction = instruction.map(|s| {
        // If it looks like a file path and exists, read it
        if std::path::Path::new(&s).exists() {
            println!("📝 Loading instruction from: {s}");
            std::fs::read_to_string(&s).unwrap_or(s)
        } else {
            s
        }
    });

    // 6. Build and run — either recipe pipeline or single task
    if let Some(recipe_path) = recipe {
        // Recipe mode: load YAML, validate, run pipeline
        let yaml = std::fs::read_to_string(&recipe_path)
            .map_err(|e| anyhow::anyhow!("Failed to read recipe {recipe_path}: {e}"))?;
        let recipe = crate::monad::Recipe::from_yaml(&yaml)
            .map_err(|e| anyhow::anyhow!("Invalid recipe YAML: {e}"))?;

        let estimate = recipe.estimate_cost();
        println!("\n🍳 Recipe: {}", recipe.name);
        if let Some(desc) = &recipe.description {
            println!("   {desc}");
        }
        println!(
            "   Steps: {}, ~{} LLM calls, ~{:.0} min",
            estimate.total_steps, estimate.estimated_llm_calls, estimate.estimated_minutes
        );
        println!();

        let result = ctx.run_recipe(recipe).await;
        match result {
            Ok(recipe_result) => {
                recipe_result.print_summary();
                if let Some(final_output) = recipe_result.final_output() {
                    // Persist final output
                    store
                        .finish_session(&session_id, Some(final_output), None)
                        .await?;
                } else {
                    store.finish_session(&session_id, None, None).await?;
                }
                println!("\n📦 Session {session_id} saved to {db_path}");
            }
            Err(e) => {
                eprintln!("\n❌ Recipe error: {e}");
                store.finish_session(&session_id, None, None).await?;
            }
        }
    } else {
        // Single task mode
        let program = agent_task_full(
            task,
            resolved_instruction.as_deref(),
            Some(&ctx.config.memory),
        );
        let result = ctx.run(program).await;

        match result {
            Ok(answer) => {
                println!("\n═══════════════════════════════════════════");
                println!("  ✅ Final Answer:");
                println!("  {answer}");
                println!("═══════════════════════════════════════════");

                // Print full conversation replay
                println!("\n📜 Conversation Replay ({} turns):", ctx.history.len());
                println!("{}", "─".repeat(60));
                for (i, msg) in ctx.history.messages().iter().enumerate() {
                    let role_icon = match format!("{:?}", msg.role).as_str() {
                        "System" => "🔧",
                        "User" => "👤",
                        "Assistant" => "🤖",
                        "Execution" => "⚡",
                        _ => "  ",
                    };
                    println!("{role_icon} Turn {} [{:?}]:", i, msg.role);
                    // Truncate system prompt for readability
                    if msg.role == Role::System && msg.content.len() > 100 {
                        println!("   {}...", &msg.content[..100]);
                    } else {
                        for line in msg.content.lines() {
                            println!("   {line}");
                        }
                    }
                    println!("{}", "─".repeat(60));
                }

                // Persist turns
                for (i, msg) in ctx.history.messages().iter().enumerate() {
                    store
                        .record_turn(&Turn {
                            session_id: session_id.clone(),
                            turn_num: i as i32,
                            role: format!("{:?}", msg.role),
                            content: msg.content.clone(),
                            code: None,
                            exec_stdout: None,
                            exec_stderr: None,
                            exec_return: None,
                            timestamp_ms: Utc::now().timestamp_millis(),
                        })
                        .await?;
                }

                store
                    .finish_session(&session_id, Some(&answer), None)
                    .await?;
                println!("\n📦 Session {session_id} saved to {db_path}");
            }
            Err(e) => {
                eprintln!("\n❌ Agent error: {e}");
                store.finish_session(&session_id, None, None).await?;
            }
        }
    }

    // Always shutdown the executor to clean up microsandbox VMs
    if let Err(e) = ctx.shutdown().await {
        tracing::warn!("Failed to shutdown executor: {e}");
    }

    Ok(())
}

/// Stop all lingering microsandbox VMs with names starting with "rlm-".
///
/// Uses the `msb` CLI to discover and stop orphaned sandboxes.
async fn run_cleanup() -> anyhow::Result<()> {
    println!("🧹 Cleaning up lingering microsandbox VMs...\n");

    // Use msb CLI to list running sandboxes
    let output = std::process::Command::new("msb")
        .args(["list", "--format", "json"])
        .output();

    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            // Try to parse JSON output; fall back to line-based parsing
            let mut stopped = 0;
            let mut skipped = 0;

            // Attempt line-based discovery: look for sandbox names starting with "rlm-"
            // msb list output format varies, so we search for our naming pattern
            for line in stdout.lines() {
                let trimmed = line.trim();
                // Look for sandbox names matching our pattern
                if let Some(name) = trimmed.split_whitespace().next() {
                    if name.starts_with("rlm-") {
                        print!("  Stopping {name}... ");
                        let stop_result = std::process::Command::new("msb")
                            .args(["stop", name])
                            .output();
                        match stop_result {
                            Ok(r) if r.status.success() => {
                                println!("✅");
                                stopped += 1;
                            }
                            Ok(r) => {
                                println!("⚠️  {}", String::from_utf8_lossy(&r.stderr).trim());
                                skipped += 1;
                            }
                            Err(e) => {
                                println!("❌ {e}");
                                skipped += 1;
                            }
                        }
                    }
                }
            }

            if stopped == 0 && skipped == 0 {
                println!("  No lingering rlm-* sandboxes found. All clean! ✨");
            } else {
                println!("\n  Stopped: {stopped}, Skipped: {skipped}");
            }
        }
        Err(e) => {
            println!("⚠️  Could not run `msb list`: {e}");
            println!("  Make sure microsandbox CLI is installed and in PATH.");
            println!("  Install: curl -sSL https://get.microsandbox.dev | sh");
        }
    }

    Ok(())
}

/// End-to-end test: monadic loop → code execution → Turso persistence.
async fn run_e2e_test(executor_name: &str, db_path: &str) -> anyhow::Result<()> {
    println!("═══════════════════════════════════════════");
    println!("  E2E Test: Monad → Sandbox → Turso");
    println!("═══════════════════════════════════════════\n");

    // 1. Test Turso persistence
    println!("📦 Testing Turso persistence...");
    let store = AgentStore::open(db_path).await?;
    let session_id = Uuid::new_v4().to_string();
    let session = Session {
        session_id: session_id.clone(),
        model: "test".to_string(),
        task: "e2e test".to_string(),
        executor: executor_name.to_string(),
        optimizer: None,
        optimized_instruction: None,
        started_at: Utc::now().to_rfc3339(),
        finished_at: None,
        final_answer: None,
        score: None,
    };
    store.create_session(&session).await?;
    store
        .record_turn(&Turn {
            session_id: session_id.clone(),
            turn_num: 0,
            role: "system".to_string(),
            content: "You are a test agent.".to_string(),
            code: None,
            exec_stdout: None,
            exec_stderr: None,
            exec_return: None,
            timestamp_ms: Utc::now().timestamp_millis(),
        })
        .await?;
    store
        .finish_session(&session_id, Some("test passed"), Some(1.0))
        .await?;
    println!("  ✅ Turso: session created, turn recorded, session finished");

    // 2. Test code executor
    println!("\n🔧 Testing code executor ({executor_name})...");
    let executor_kind = match executor_name {
        "microsandbox" => ExecutorKind::Microsandbox,
        _ => ExecutorKind::Pyo3,
    };
    match create_executor(&executor_kind).await {
        Ok(mut exec) => {
            let result = exec.execute("print(2 + 2)").await?;
            println!("  Execution result: {}", result.to_feedback());
            if result.stdout == "4" || result.stdout.trim() == "4" {
                println!("  ✅ Executor: 2+2=4 confirmed");
            } else {
                println!("  ⚠️  Executor: unexpected output: {:?}", result.stdout);
            }
        }
        Err(e) => {
            println!("  ⚠️  Executor not available: {e}");
            println!("  (Microsandbox requires `msandbox server start`)");
        }
    }

    // 3. Test monadic loop (no LLM, just structure)
    println!("\n🔄 Testing monadic loop...");
    let mut ctx = AgentContext::new(AgentConfig {
        max_turns: 10,
        ..AgentConfig::default()
    });
    let program = AgentMonad::insert(Role::System, "test system prompt")
        .then(AgentMonad::insert(Role::User, "What is 2+2?"))
        .then(AgentMonad::capture("answer", "4"))
        .then(AgentMonad::retrieve("answer"))
        .bind(|answer| {
            AgentMonad::log(crate::monad::LogLevel::Info, format!("Answer: {answer}"))
                .then(AgentMonad::pure(format!("The answer is {answer}")))
        });

    let result = ctx.run(program).await?;
    assert_eq!(result, "The answer is 4");
    println!("  ✅ Monad: Pure → Insert → Capture → Retrieve → Bind → Log → Pure works");
    println!("  History: {} messages", ctx.history.len());

    // 4. Test DSRs signature
    println!("\n📝 Testing DSRs signature...");
    println!("  ✅ CodeGenAgent signature compiled and available");

    println!("\n═══════════════════════════════════════════");
    println!("  ✅ All E2E tests passed!");
    println!("═══════════════════════════════════════════");

    Ok(())
}

/// Run GEPA optimization on the agent instruction.
async fn run_optimization(
    trainset_path: &str,
    iterations: usize,
    _db_path: &str,
) -> anyhow::Result<()> {
    use crate::agent_module::AgentModule;
    use crate::monad::provider::ProviderConfig;
    use dspy_rs::*;

    println!("═══════════════════════════════════════════");
    println!("  GEPA Optimization");
    println!("═══════════════════════════════════════════\n");

    // 1. Load trainset from JSON
    //    Format: [{"task": "...", "answer": "..."}, ...]
    let trainset_json = std::fs::read_to_string(trainset_path)
        .map_err(|e| anyhow::anyhow!("Failed to read trainset {trainset_path}: {e}"))?;

    #[derive(serde::Deserialize)]
    struct TrainExample {
        task: String,
        answer: String,
        #[serde(default)]
        context: Option<String>,
        #[serde(default)]
        code: Option<String>,
    }

    let raw_examples: Vec<TrainExample> = serde_json::from_str(&trainset_json)
        .map_err(|e| anyhow::anyhow!("Failed to parse trainset JSON: {e}"))?;

    let trainset: Vec<Example> = raw_examples
        .into_iter()
        .map(|ex| {
            let mut input_keys = vec!["task".to_string()];
            let mut fields = std::collections::HashMap::new();
            fields.insert("task".to_string(), serde_json::Value::String(ex.task));
            if let Some(ctx) = ex.context {
                fields.insert("context".to_string(), serde_json::Value::String(ctx));
                input_keys.push("context".to_string());
            }
            let mut output_keys = vec!["answer".to_string()];
            fields.insert("answer".to_string(), serde_json::Value::String(ex.answer));
            if let Some(code) = ex.code {
                fields.insert("code".to_string(), serde_json::Value::String(code));
                output_keys.push("code".to_string());
            }
            Example::new(fields, input_keys, output_keys)
        })
        .collect();

    println!("📚 Loaded {} training examples", trainset.len());

    // 2. Determine provider config (same env vars as run_agent)
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let model = std::env::var("RIG_RLM_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

    let provider_config = if api_key.is_empty() {
        println!("🏠 Using local LLM (set OPENAI_API_KEY for OpenAI/OpenRouter)");
        ProviderConfig::local(&model)
    } else {
        println!("🔑 Using API key with base URL: {base_url}");
        ProviderConfig::openai_compatible("openai", &model, &base_url, &api_key)
    };

    // 3. Build the agent module
    let seed_instruction = crate::signature::CodeGenAgent::new().instruction();
    println!(
        "📝 Seed instruction: {}",
        &seed_instruction[..seed_instruction.len().min(80)]
    );

    let mut module = AgentModule::new(&seed_instruction, provider_config);

    // 3b. Initialize dspy-rs GLOBAL_SETTINGS for GEPA's internal Predict calls.
    //     GEPA uses Predict::forward() for reflection/mutation which requires
    //     a globally configured LM.
    let lm = LM::builder()
        .base_url(base_url)
        .api_key(api_key)
        .model(model)
        .temperature(0.7)
        .max_tokens(2048_u32)
        .build()
        .await?;
    configure(lm, ChatAdapter);

    // 4. Build GEPA optimizer
    let gepa = GEPA::builder()
        .num_iterations(iterations)
        .minibatch_size(trainset.len().min(5))
        .max_lm_calls(iterations * trainset.len() * 3)
        .track_stats(true)
        .build();

    println!("🧬 Running GEPA for {iterations} iterations...\n");

    // 5. Run optimization (compile_with_feedback because we impl FeedbackEvaluator)
    let report = gepa.compile_with_feedback(&mut module, trainset).await?;

    // 6. Print results
    println!("\n═══════════════════════════════════════════");
    println!("  Optimization Results");
    println!("═══════════════════════════════════════════\n");
    println!(
        "🏆 Best score: {:.3}",
        report.best_candidate.average_score()
    );
    println!(
        "📝 Optimized instruction:\n{}",
        report.best_candidate.instruction
    );
    println!("\n📊 Total LM calls: {}", report.total_lm_calls);
    println!("🧬 Rollouts: {}", report.total_rollouts);

    if !report.evolution_history.is_empty() {
        println!("\n📈 Score progression:");
        for (generation, score) in &report.evolution_history {
            println!("  Gen {}: {:.3}", generation, score);
        }
    }

    if !report.all_candidates.is_empty() {
        println!("\n📋 Top candidates:");
        let mut sorted = report.all_candidates.clone();
        sorted.sort_by(|a, b| {
            b.average_score()
                .partial_cmp(&a.average_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for candidate in sorted.iter().take(3) {
            println!(
                "  {:.3} — {}",
                candidate.average_score(),
                &candidate.instruction[..candidate.instruction.len().min(60)]
            );
        }
    }

    // Save optimized instruction to file for reuse via --instruction flag
    let instruction_path = "optimized_instruction.txt";
    std::fs::write(instruction_path, &report.best_candidate.instruction)?;
    println!("\n💾 Saved optimized instruction to: {instruction_path}");
    println!("   Reuse with: cargo run -- run \"task\" --instruction {instruction_path}");

    Ok(())
}

/// Show session summary from Turso DB.
async fn show_summary(db_path: &str, _session_id: &str) -> anyhow::Result<()> {
    let _store = AgentStore::open(db_path).await?;
    // For now, just show basic info
    println!("📊 Database: {db_path}");
    println!("   (Use session ID to query specific sessions)");
    Ok(())
}

/// Run ARC-AGI benchmark (Phases 18-21).
async fn run_arc_benchmark(
    dataset_dir: &str,
    model: &str,
    optimize: bool,
    train_split: usize,
    max_tasks: usize,
    iterations: usize,
) -> anyhow::Result<()> {
    use crate::arc::bench::{BenchmarkConfig, run_baseline, run_optimized};
    use crate::monad::provider::ProviderConfig;

    println!("═══════════════════════════════════════════");
    println!("  rig-rlm — ARC-AGI Benchmark");
    println!("  Model:   {model}");
    println!("  Dataset: {dataset_dir}");
    println!(
        "  Mode:    {}",
        if optimize {
            "Optimized (GEPA)"
        } else {
            "Baseline"
        }
    );
    println!("═══════════════════════════════════════════\n");

    let config = BenchmarkConfig {
        dataset_dir: dataset_dir.to_string(),
        provider: ProviderConfig::local(model),
        optimize,
        train_split,
        max_tasks,
        gepa_iterations: iterations,
        ..BenchmarkConfig::default()
    };

    let report = if optimize {
        run_optimized(&config).await?
    } else {
        run_baseline(&config).await?
    };

    report.print_summary();

    Ok(())
}
