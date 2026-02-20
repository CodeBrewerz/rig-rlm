//! Phase 22: Turso persistence layer.
//!
//! Stores agent sessions, conversation history, execution traces,
//! and optimization history in a local Turso (libSQL) database.

use chrono::Utc;
use serde::{Deserialize, Serialize};

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
}
