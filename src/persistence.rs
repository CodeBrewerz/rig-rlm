//! Phase 22: Turso persistence layer.
//!
//! Stores agent sessions, conversation history, execution traces,
//! optimization history, and cross-session semantic memories
//! in a local Turso (libSQL) database.
//!
//! ## Semantic Memory
//!
//! Uses Turso's native vector search (`F32_BLOB` + `vector_top_k()`) to
//! store and recall memories across sessions. At session end, the agent
//! extracts preferences, decisions, and facts; at session start, it
//! recalls the most relevant ones via cosine similarity.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use tracing::debug;

/// A Turso-backed persistence store for agent state.
pub struct AgentStore {
    db: turso::Database,
}

/// Represents an agent session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub session_id: String,
    pub model: String,
    pub task: String,
    pub executor: String,
    pub optimizer: Option<String>,
    pub optimized_instruction: Option<String>,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub final_answer: Option<String>,
    pub score: Option<f64>,
}

/// Represents a single turn in an agent conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub session_id: String,
    pub turn_num: i32,
    pub role: String,
    pub content: String,
    pub code: Option<String>,
    pub exec_stdout: Option<String>,
    pub exec_stderr: Option<String>,
    pub exec_return: Option<String>,
    pub timestamp_ms: i64,
}

/// A discrete memory extracted from a conversation.
///
/// Memories are user preferences, decisions, facts, or learnings
/// that persist across sessions and are recalled via semantic search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: Option<i64>,
    pub user_id: String,
    pub content: String,
    pub category: String, // "preference" | "learning" | "decision" | "fact"
    pub session_id: Option<String>,
    pub created_at: String,
}

impl AgentStore {
    /// Open or create a Turso database at the given path.
    pub async fn open(path: &str) -> anyhow::Result<Self> {
        let db = turso::Builder::new_local(path)
            .build()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to open Turso DB: {e}"))?;
        let conn = db
            .connect()
            .map_err(|e| anyhow::anyhow!("Failed to connect: {e}"))?;

        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS sessions (
                session_id    TEXT PRIMARY KEY,
                model         TEXT NOT NULL,
                task          TEXT NOT NULL,
                executor      TEXT NOT NULL DEFAULT 'pyo3',
                optimizer     TEXT,
                optimized_instruction TEXT,
                started_at    TEXT NOT NULL,
                finished_at   TEXT,
                final_answer  TEXT,
                score         REAL
            );

            CREATE TABLE IF NOT EXISTS turns (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT NOT NULL REFERENCES sessions(session_id),
                turn_num      INTEGER NOT NULL,
                role          TEXT NOT NULL,
                content       TEXT,
                code          TEXT,
                exec_stdout   TEXT,
                exec_stderr   TEXT,
                exec_return   TEXT,
                timestamp_ms  INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS optimization_history (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT REFERENCES sessions(session_id),
                generation    INTEGER NOT NULL,
                candidate_id  INTEGER NOT NULL,
                instruction   TEXT NOT NULL,
                avg_score     REAL,
                parent_id     INTEGER,
                feedback_summary TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
            CREATE INDEX IF NOT EXISTS idx_opt_session ON optimization_history(session_id);
        ",
        )
        .await
        .map_err(|e| anyhow::anyhow!("Schema init failed: {e}"))?;

        // ── Semantic memory table (Phase 28) ──────────────────────────
        // Stores embeddings as JSON text for universal SQLite compatibility.
        // Cosine distance is computed in application code.
        let conn3 = db.connect().map_err(|e| anyhow::anyhow!("connect: {e}"))?;
        let _ = conn3
            .execute_batch(
                "
                CREATE TABLE IF NOT EXISTS memories (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     TEXT NOT NULL DEFAULT 'default',
                    content     TEXT NOT NULL,
                    category    TEXT NOT NULL DEFAULT 'fact',
                    session_id  TEXT,
                    created_at  TEXT NOT NULL,
                    embedding   TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
                ",
            )
            .await;
        debug!("memories table ready");

        // Tier 1.2: Add cost/token columns (idempotent ALTER TABLE)
        for stmt in &[
            "ALTER TABLE sessions ADD COLUMN total_cost_usd REAL DEFAULT 0.0",
            "ALTER TABLE sessions ADD COLUMN total_input_tokens INTEGER DEFAULT 0",
            "ALTER TABLE sessions ADD COLUMN total_output_tokens INTEGER DEFAULT 0",
            "ALTER TABLE turns ADD COLUMN input_tokens INTEGER DEFAULT 0",
            "ALTER TABLE turns ADD COLUMN output_tokens INTEGER DEFAULT 0",
            "ALTER TABLE turns ADD COLUMN cost_usd REAL DEFAULT 0.0",
        ] {
            // SQLite silently ignores duplicate column additions in IF NOT EXISTS,
            // but ALTER TABLE doesn't support IF NOT EXISTS, so we catch the error.
            let conn2 = db.connect().map_err(|e| anyhow::anyhow!("connect: {e}"))?;
            let _ = conn2.execute(*stmt, ()).await; // ignore "duplicate column" errors
        }

        Ok(Self { db })
    }

    /// Start a new session.
    pub async fn create_session(&self, session: &Session) -> anyhow::Result<()> {
        let conn = self
            .db
            .connect()
            .map_err(|e| anyhow::anyhow!("connect: {e}"))?;
        conn.execute(
            "INSERT INTO sessions (session_id, model, task, executor, optimizer, started_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            (
                session.session_id.as_str(),
                session.model.as_str(),
                session.task.as_str(),
                session.executor.as_str(),
                session.optimizer.as_deref().unwrap_or(""),
                session.started_at.as_str(),
            ),
        ).await.map_err(|e| anyhow::anyhow!("insert session: {e}"))?;
        Ok(())
    }

    /// Record a conversation turn.
    pub async fn record_turn(&self, turn: &Turn) -> anyhow::Result<()> {
        let conn = self
            .db
            .connect()
            .map_err(|e| anyhow::anyhow!("connect: {e}"))?;
        conn.execute(
            "INSERT INTO turns (session_id, turn_num, role, content, code, exec_stdout, exec_stderr, exec_return, timestamp_ms) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            (
                turn.session_id.as_str(),
                turn.turn_num,
                turn.role.as_str(),
                turn.content.as_str(),
                turn.code.as_deref().unwrap_or(""),
                turn.exec_stdout.as_deref().unwrap_or(""),
                turn.exec_stderr.as_deref().unwrap_or(""),
                turn.exec_return.as_deref().unwrap_or(""),
                turn.timestamp_ms,
            ),
        ).await.map_err(|e| anyhow::anyhow!("insert turn: {e}"))?;
        Ok(())
    }

    /// Finish a session with the final answer and score.
    pub async fn finish_session(
        &self,
        session_id: &str,
        final_answer: Option<&str>,
        score: Option<f64>,
    ) -> anyhow::Result<()> {
        let conn = self
            .db
            .connect()
            .map_err(|e| anyhow::anyhow!("connect: {e}"))?;
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE sessions SET finished_at = ?1, final_answer = ?2, score = ?3 WHERE session_id = ?4",
            (now.as_str(), final_answer.unwrap_or(""), score.unwrap_or(0.0), session_id),
        ).await.map_err(|e| anyhow::anyhow!("finish session: {e}"))?;
        Ok(())
    }

    /// Record an optimization step.
    pub async fn record_optimization(
        &self,
        session_id: &str,
        generation: i32,
        candidate_id: i32,
        instruction: &str,
        avg_score: f64,
        feedback: &str,
    ) -> anyhow::Result<()> {
        let conn = self
            .db
            .connect()
            .map_err(|e| anyhow::anyhow!("connect: {e}"))?;
        conn.execute(
            "INSERT INTO optimization_history (session_id, generation, candidate_id, instruction, avg_score, feedback_summary) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            (session_id, generation, candidate_id, instruction, avg_score, feedback),
        ).await.map_err(|e| anyhow::anyhow!("insert optimization: {e}"))?;
        Ok(())
    }

    /// Get summary stats for a session.
    pub async fn session_summary(&self, session_id: &str) -> anyhow::Result<String> {
        let conn = self
            .db
            .connect()
            .map_err(|e| anyhow::anyhow!("connect: {e}"))?;
        let mut rows = conn.query(
            "SELECT COUNT(*) as num_turns, MIN(timestamp_ms) as first_ms, MAX(timestamp_ms) as last_ms FROM turns WHERE session_id = ?1",
            (session_id,),
        ).await.map_err(|e| anyhow::anyhow!("query: {e}"))?;

        if let Some(row) = rows
            .next()
            .await
            .map_err(|e| anyhow::anyhow!("next: {e}"))?
        {
            let num_turns: i64 = row.get(0).map_err(|e| anyhow::anyhow!("get: {e}"))?;
            let first_ms: i64 = row.get(1).map_err(|e| anyhow::anyhow!("get: {e}"))?;
            let last_ms: i64 = row.get(2).map_err(|e| anyhow::anyhow!("get: {e}"))?;
            let duration_s = (last_ms - first_ms) as f64 / 1000.0;
            Ok(format!("{num_turns} turns, {duration_s:.1}s"))
        } else {
            Ok("no data".to_string())
        }
    }

    // ── Semantic Memory Methods ──────────────────────────────────────

    /// Store a memory with its embedding vector.
    ///
    /// Embedding is stored as a JSON array string for universal SQLite compat.
    pub async fn store_memory(&self, memory: &Memory, embedding: &[f32]) -> anyhow::Result<()> {
        let conn = self
            .db
            .connect()
            .map_err(|e| anyhow::anyhow!("connect: {e}"))?;

        // Serialize embedding as JSON array
        let emb_json = serde_json::to_string(embedding)
            .map_err(|e| anyhow::anyhow!("serialize embedding: {e}"))?;

        conn.execute(
            "INSERT INTO memories (user_id, content, category, session_id, created_at, embedding) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            (
                memory.user_id.as_str(),
                memory.content.as_str(),
                memory.category.as_str(),
                memory.session_id.as_deref().unwrap_or(""),
                memory.created_at.as_str(),
                emb_json.as_str(),
            ),
        )
        .await
        .map_err(|e| anyhow::anyhow!("insert memory: {e}"))?;

        debug!(
            "stored memory: [{}] {}",
            memory.category,
            &memory.content[..memory.content.len().min(80)]
        );
        Ok(())
    }

    /// Recall the top-K most relevant memories using cosine similarity.
    ///
    /// Loads all memories, computes cosine distance in Rust, returns top-K.
    /// This is a brute-force approach that works on any SQLite version.
    pub async fn recall_memories(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> anyhow::Result<Vec<Memory>> {
        let conn = self
            .db
            .connect()
            .map_err(|e| anyhow::anyhow!("connect: {e}"))?;

        let mut rows = conn
            .query(
                "SELECT id, user_id, content, category, session_id, created_at, embedding \
                 FROM memories ORDER BY created_at DESC LIMIT 200",
                (),
            )
            .await
            .map_err(|e| anyhow::anyhow!("recall query: {e}"))?;

        let mut scored: Vec<(f64, Memory)> = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| anyhow::anyhow!("next: {e}"))?
        {
            let id: i64 = row.get(0).map_err(|e| anyhow::anyhow!("get: {e}"))?;
            let user_id: String = row.get(1).map_err(|e| anyhow::anyhow!("get: {e}"))?;
            let content: String = row.get(2).map_err(|e| anyhow::anyhow!("get: {e}"))?;
            let category: String = row.get(3).map_err(|e| anyhow::anyhow!("get: {e}"))?;
            let session_id: String = row.get(4).map_err(|e| anyhow::anyhow!("get: {e}"))?;
            let created_at: String = row.get(5).map_err(|e| anyhow::anyhow!("get: {e}"))?;
            let emb_json: String = row.get(6).map_err(|e| anyhow::anyhow!("get: {e}"))?;

            // Parse stored embedding
            let stored_emb: Vec<f32> = match serde_json::from_str(&emb_json) {
                Ok(v) => v,
                Err(_) => continue, // skip invalid embeddings
            };

            // Compute cosine similarity
            let sim = cosine_similarity(query_embedding, &stored_emb);

            scored.push((
                sim,
                Memory {
                    id: Some(id),
                    user_id,
                    content,
                    category,
                    session_id: if session_id.is_empty() {
                        None
                    } else {
                        Some(session_id)
                    },
                    created_at,
                },
            ));
        }

        // Sort by similarity (descending) and take top-K
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let memories: Vec<Memory> = scored.into_iter().take(top_k).map(|(_, m)| m).collect();

        debug!("recalled {} memories (top_k={top_k})", memories.len());
        Ok(memories)
    }

    /// List recent memories chronologically (no vector search).
    pub async fn list_memories(&self, user_id: &str, limit: usize) -> anyhow::Result<Vec<Memory>> {
        let conn = self
            .db
            .connect()
            .map_err(|e| anyhow::anyhow!("connect: {e}"))?;

        let mut rows = conn
            .query(
                "SELECT id, user_id, content, category, session_id, created_at \
                 FROM memories WHERE user_id = ?1 ORDER BY created_at DESC LIMIT ?2",
                (user_id, limit as i64),
            )
            .await
            .map_err(|e| anyhow::anyhow!("list query: {e}"))?;

        let mut memories = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| anyhow::anyhow!("next: {e}"))?
        {
            let id: i64 = row.get(0).map_err(|e| anyhow::anyhow!("get: {e}"))?;
            let user_id: String = row.get(1).map_err(|e| anyhow::anyhow!("get: {e}"))?;
            let content: String = row.get(2).map_err(|e| anyhow::anyhow!("get: {e}"))?;
            let category: String = row.get(3).map_err(|e| anyhow::anyhow!("get: {e}"))?;
            let session_id: String = row.get(4).map_err(|e| anyhow::anyhow!("get: {e}"))?;
            let created_at: String = row.get(5).map_err(|e| anyhow::anyhow!("get: {e}"))?;

            memories.push(Memory {
                id: Some(id),
                user_id,
                content,
                category,
                session_id: if session_id.is_empty() {
                    None
                } else {
                    Some(session_id)
                },
                created_at,
            });
        }

        Ok(memories)
    }

    /// Format recalled memories for injection into the system prompt.
    pub fn format_memories_for_prompt(memories: &[Memory]) -> String {
        if memories.is_empty() {
            return String::new();
        }

        let mut out = String::from("\n## Recalled Memories\n\n");
        out.push_str("The following are relevant memories from past sessions:\n\n");
        for m in memories {
            out.push_str(&format!("- [{}] {}\n", m.category, m.content));
        }
        out
    }
}

/// Compute cosine similarity between two vectors.
///
/// Returns a value in [-1.0, 1.0] where 1.0 means identical direction.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}
