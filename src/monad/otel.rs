//! OpenTelemetry + LangFuse tracing for LLM observability.
//!
//! Provides:
//! - `init_tracing()` — sets up LangFuse exporter (if env vars set) or no-op
//! - `TraceContext` — session, user, tags, metadata propagated to all spans
//! - Span helpers for LLM calls, recipes, sub-agents, code execution
//! - `shutdown_tracing()` — flush all pending spans to LangFuse
//!
//! ## Environment Variables
//!
//! ```text
//! LANGFUSE_BASE_URL      https://cloud.langfuse.com (default)
//! LANGFUSE_PUBLIC_KEY    pk-lf-...  (required for LangFuse)
//! LANGFUSE_SECRET_KEY    sk-lf-...  (required for LangFuse)
//! ```

use opentelemetry::KeyValue;
use opentelemetry::global;
use opentelemetry::trace::{Span, SpanKind, Status, Tracer};
use opentelemetry_sdk::trace::SdkTracerProvider;
use std::borrow::Cow;
use std::sync::{Mutex, OnceLock};

/// Global provider stored for proper shutdown/flush.
static PROVIDER: OnceLock<Mutex<Option<SdkTracerProvider>>> = OnceLock::new();

// ─── LangFuse Attribute Constants ─────────────────────────────────

/// LangFuse trace-level attribute keys (applied to the whole trace).
pub mod langfuse {
    // Trace identity
    pub const TRACE_NAME: &str = "langfuse.trace.name";
    pub const USER_ID: &str = "langfuse.user.id";
    pub const SESSION_ID: &str = "langfuse.session.id";
    pub const RELEASE: &str = "langfuse.release";
    pub const VERSION: &str = "langfuse.version";
    pub const ENVIRONMENT: &str = "langfuse.environment";

    // Trace content
    pub const TRACE_INPUT: &str = "langfuse.trace.input";
    pub const TRACE_OUTPUT: &str = "langfuse.trace.output";
    pub const TRACE_TAGS: &str = "langfuse.trace.tags";
    pub const TRACE_PUBLIC: &str = "langfuse.trace.public";

    // Observation-level
    pub const OBS_TYPE: &str = "langfuse.observation.type";
    pub const OBS_LEVEL: &str = "langfuse.observation.level";
    pub const OBS_INPUT: &str = "langfuse.observation.input";
    pub const OBS_OUTPUT: &str = "langfuse.observation.output";
    pub const OBS_MODEL: &str = "langfuse.observation.model.name";

    // Metadata prefix for filterable keys
    pub const TRACE_META_PREFIX: &str = "langfuse.trace.metadata.";
    pub const OBS_META_PREFIX: &str = "langfuse.observation.metadata.";
}

/// OpenTelemetry GenAI semantic convention attribute keys.
pub mod genai {
    pub const SYSTEM: &str = "gen_ai.system";
    pub const REQUEST_MODEL: &str = "gen_ai.request.model";
    pub const RESPONSE_MODEL: &str = "gen_ai.response.model";
    pub const USAGE_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";
    pub const USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";
    pub const OPERATION_NAME: &str = "gen_ai.operation.name";
}

// ─── TraceContext ─────────────────────────────────────────────────

/// Context propagated to all spans in a trace for LangFuse enrichment.
///
/// Set this on `AgentContext` to enrich all spans with session/user/tags.
#[derive(Debug, Clone, Default)]
pub struct TraceContext {
    /// User ID for LangFuse user tracking.
    pub user_id: Option<String>,
    /// Session ID for grouping traces (e.g., chat thread).
    pub session_id: Option<String>,
    /// Human-readable trace name.
    pub trace_name: Option<String>,
    /// Tags for filtering (e.g., ["production", "v2"]).
    pub tags: Vec<String>,
    /// Release version (e.g., git commit hash).
    pub release: Option<String>,
    /// Environment (e.g., "production", "staging", "development").
    pub environment: Option<String>,
    /// Filterable top-level metadata key-value pairs.
    pub metadata: Vec<(String, String)>,
}

impl TraceContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.trace_name = Some(name.into());
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_release(mut self, release: impl Into<String>) -> Self {
        self.release = Some(release.into());
        self
    }

    pub fn with_environment(mut self, env: impl Into<String>) -> Self {
        self.environment = Some(env.into());
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.push((key.into(), value.into()));
        self
    }

    /// Convert to OTEL KeyValue attributes for injection into spans.
    pub fn to_attributes(&self) -> Vec<KeyValue> {
        let mut attrs = Vec::new();

        if let Some(ref uid) = self.user_id {
            attrs.push(KeyValue::new(langfuse::USER_ID, uid.clone()));
        }
        if let Some(ref sid) = self.session_id {
            attrs.push(KeyValue::new(langfuse::SESSION_ID, sid.clone()));
        }
        if let Some(ref name) = self.trace_name {
            attrs.push(KeyValue::new(langfuse::TRACE_NAME, name.clone()));
        }
        if !self.tags.is_empty() {
            // LangFuse expects tags as a JSON array string
            let tags_json = serde_json::to_string(&self.tags).unwrap_or_default();
            attrs.push(KeyValue::new(langfuse::TRACE_TAGS, tags_json));
        }
        if let Some(ref release) = self.release {
            attrs.push(KeyValue::new(langfuse::RELEASE, release.clone()));
        }
        if let Some(ref env) = self.environment {
            attrs.push(KeyValue::new(langfuse::ENVIRONMENT, env.clone()));
        }
        // Version from Cargo.toml
        attrs.push(KeyValue::new(
            langfuse::VERSION,
            env!("CARGO_PKG_VERSION").to_string(),
        ));

        // Filterable metadata (langfuse.trace.metadata.{key})
        for (k, v) in &self.metadata {
            attrs.push(KeyValue::new(
                format!("{}{k}", langfuse::TRACE_META_PREFIX),
                v.clone(),
            ));
        }

        attrs
    }
}

// ─── Token Usage ──────────────────────────────────────────────────

/// Token usage from an LLM call.
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
}

impl TokenUsage {
    pub fn total(&self) -> i64 {
        self.input_tokens.unwrap_or(0) + self.output_tokens.unwrap_or(0)
    }
}

// ─── Init / Shutdown ──────────────────────────────────────────────

/// Initialize OpenTelemetry with LangFuse exporter.
///
/// If LangFuse env vars are not set, all spans become zero-cost no-ops.
pub fn init_tracing() -> anyhow::Result<()> {
    // Only initialize once
    if PROVIDER.get().is_some() {
        return Ok(());
    }

    let has_langfuse = std::env::var("LANGFUSE_PUBLIC_KEY").is_ok()
        && std::env::var("LANGFUSE_SECRET_KEY").is_ok();

    if !has_langfuse {
        eprintln!("⚡ OTEL: LangFuse keys not set — tracing is no-op");
        let _ = PROVIDER.set(Mutex::new(None));
        return Ok(());
    }

    // Support both LANGFUSE_BASE_URL (our convention) and LANGFUSE_HOST (crate default)
    let langfuse_host = std::env::var("LANGFUSE_BASE_URL")
        .or_else(|_| std::env::var("LANGFUSE_HOST"))
        .unwrap_or_else(|_| "https://cloud.langfuse.com".to_string());

    let public_key = std::env::var("LANGFUSE_PUBLIC_KEY")
        .map_err(|_| anyhow::anyhow!("LANGFUSE_PUBLIC_KEY not set"))?;
    let secret_key = std::env::var("LANGFUSE_SECRET_KEY")
        .map_err(|_| anyhow::anyhow!("LANGFUSE_SECRET_KEY not set"))?;

    let exporter = opentelemetry_langfuse::ExporterBuilder::new()
        .with_host(&langfuse_host)
        .with_basic_auth(&public_key, &secret_key)
        .build()
        .map_err(|e| anyhow::anyhow!("LangFuse exporter build: {e}"))?;

    let processor =
        opentelemetry_sdk::trace::span_processor_with_async_runtime::BatchSpanProcessor::builder(
            exporter,
            opentelemetry_sdk::runtime::Tokio,
        )
        .build();

    let provider = SdkTracerProvider::builder()
        .with_span_processor(processor)
        .with_resource(
            opentelemetry_sdk::Resource::builder()
                .with_attributes(vec![
                    KeyValue::new("service.name", "rig-rlm"),
                    KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
                ])
                .build(),
        )
        .build();

    global::set_tracer_provider(provider.clone());
    let _ = PROVIDER.set(Mutex::new(Some(provider)));

    eprintln!("⚡ OTEL: LangFuse tracing initialized → {langfuse_host}");
    Ok(())
}

/// Shutdown the tracer provider, flushing ALL pending spans to LangFuse.
///
/// **Must be called before process exit** or spans will be lost.
/// Uses spawn_blocking to avoid deadlocking the Tokio runtime.
pub async fn shutdown_tracing() {
    if let Some(lock) = PROVIDER.get() {
        if let Ok(mut guard) = lock.lock() {
            if let Some(provider) = guard.take() {
                eprintln!("⚡ OTEL: Flushing spans to LangFuse...");
                let result = tokio::task::spawn_blocking(move || provider.shutdown()).await;
                match result {
                    Ok(Ok(())) => eprintln!("⚡ OTEL: All spans flushed ✓"),
                    Ok(Err(e)) => eprintln!("⚡ OTEL: Flush error: {e}"),
                    Err(e) => eprintln!("⚡ OTEL: Flush task failed: {e}"),
                }
            }
        }
    }
}

// ─── Span Recording Functions ─────────────────────────────────────

/// Record a completed LLM call as an OTEL span with GenAI + LangFuse attributes.
pub fn record_llm_span(
    operation: &str,
    provider_name: &str,
    model: &str,
    usage: &TokenUsage,
    duration: std::time::Duration,
    success: bool,
    trace_ctx: &TraceContext,
    input: Option<&str>,
    output: Option<&str>,
) {
    let tracer = global::tracer("rig-rlm");

    let mut attrs = vec![
        // GenAI semantic conventions
        KeyValue::new(genai::SYSTEM, provider_name.to_string()),
        KeyValue::new(genai::REQUEST_MODEL, model.to_string()),
        KeyValue::new(genai::RESPONSE_MODEL, model.to_string()),
        KeyValue::new(genai::OPERATION_NAME, operation.to_string()),
        // LangFuse observation enrichment
        KeyValue::new(langfuse::OBS_TYPE, "generation"),
        KeyValue::new(langfuse::OBS_MODEL, model.to_string()),
    ];

    // Add trace-level context (session, user, tags, etc.)
    attrs.extend(trace_ctx.to_attributes());

    // Observation I/O
    if let Some(inp) = input {
        attrs.push(KeyValue::new(langfuse::OBS_INPUT, inp.to_string()));
    }
    if let Some(out) = output {
        attrs.push(KeyValue::new(langfuse::OBS_OUTPUT, out.to_string()));
    }

    let mut span = tracer
        .span_builder(format!("{operation} {model}"))
        .with_kind(SpanKind::Client)
        .with_attributes(attrs)
        .start(&tracer);

    if let Some(input_tokens) = usage.input_tokens {
        span.set_attribute(KeyValue::new(genai::USAGE_INPUT_TOKENS, input_tokens));
    }
    if let Some(output_tokens) = usage.output_tokens {
        span.set_attribute(KeyValue::new(genai::USAGE_OUTPUT_TOKENS, output_tokens));
    }
    span.add_event(
        "llm.response",
        vec![KeyValue::new("duration_ms", duration.as_millis() as i64)],
    );

    if success {
        span.set_attribute(KeyValue::new(langfuse::OBS_LEVEL, "DEFAULT"));
    } else {
        span.set_attribute(KeyValue::new(langfuse::OBS_LEVEL, "ERROR"));
        span.set_status(Status::Error {
            description: Cow::Borrowed("LLM call failed"),
        });
    }
    span.end();
}

/// Record a recipe execution span with LangFuse enrichment.
pub fn record_recipe_span(
    recipe_name: &str,
    total_steps: usize,
    total_turns: usize,
    duration: std::time::Duration,
    success: bool,
    trace_ctx: &TraceContext,
) {
    let tracer = global::tracer("rig-rlm");

    let mut attrs = vec![
        KeyValue::new(langfuse::OBS_TYPE, "span"),
        KeyValue::new("recipe.name", recipe_name.to_string()),
        KeyValue::new("recipe.total_steps", total_steps as i64),
        KeyValue::new("recipe.total_turns", total_turns as i64),
        KeyValue::new("recipe.duration_ms", duration.as_millis() as i64),
        // Filterable metadata
        KeyValue::new(
            format!("{}recipe_name", langfuse::OBS_META_PREFIX),
            recipe_name.to_string(),
        ),
        KeyValue::new(
            format!("{}total_steps", langfuse::OBS_META_PREFIX),
            total_steps.to_string(),
        ),
    ];
    attrs.extend(trace_ctx.to_attributes());

    let mut span = tracer
        .span_builder(format!("recipe: {recipe_name}"))
        .with_kind(SpanKind::Internal)
        .with_attributes(attrs)
        .start(&tracer);

    if !success {
        span.set_attribute(KeyValue::new(langfuse::OBS_LEVEL, "ERROR"));
        span.set_status(Status::Error {
            description: Cow::Borrowed("Recipe failed"),
        });
    }
    span.end();
}

/// Record a sub-agent spawn span with LangFuse enrichment.
pub fn record_subagent_span(
    parent_id: &str,
    task: &str,
    duration: std::time::Duration,
    success: bool,
    trace_ctx: &TraceContext,
) {
    let tracer = global::tracer("rig-rlm");

    let mut attrs = vec![
        KeyValue::new(langfuse::OBS_TYPE, "span"),
        KeyValue::new("agent.parent_id", parent_id.to_string()),
        KeyValue::new("agent.duration_ms", duration.as_millis() as i64),
        KeyValue::new(langfuse::OBS_INPUT, task.to_string()),
        KeyValue::new(
            format!("{}parent_agent", langfuse::OBS_META_PREFIX),
            parent_id.to_string(),
        ),
    ];
    attrs.extend(trace_ctx.to_attributes());

    let mut span = tracer
        .span_builder("spawn_sub_agent")
        .with_kind(SpanKind::Internal)
        .with_attributes(attrs)
        .start(&tracer);

    if !success {
        span.set_attribute(KeyValue::new(langfuse::OBS_LEVEL, "ERROR"));
        span.set_status(Status::Error {
            description: Cow::Borrowed("Sub-agent failed"),
        });
    }
    span.end();
}

/// Record a code execution span with LangFuse enrichment.
pub fn record_code_exec_span(
    code_len: usize,
    duration: std::time::Duration,
    success: bool,
    trace_ctx: &TraceContext,
    code_snippet: Option<&str>,
    output: Option<&str>,
) {
    let tracer = global::tracer("rig-rlm");

    let mut attrs = vec![
        KeyValue::new(langfuse::OBS_TYPE, "span"),
        KeyValue::new("code.length", code_len as i64),
        KeyValue::new("code.duration_ms", duration.as_millis() as i64),
    ];
    attrs.extend(trace_ctx.to_attributes());

    if let Some(code) = code_snippet {
        attrs.push(KeyValue::new(langfuse::OBS_INPUT, code.to_string()));
    }
    if let Some(out) = output {
        attrs.push(KeyValue::new(langfuse::OBS_OUTPUT, out.to_string()));
    }

    let mut span = tracer
        .span_builder("execute_code")
        .with_kind(SpanKind::Internal)
        .with_attributes(attrs)
        .start(&tracer);

    if !success {
        span.set_attribute(KeyValue::new(langfuse::OBS_LEVEL, "ERROR"));
        span.set_status(Status::Error {
            description: Cow::Borrowed("Code execution failed"),
        });
    }
    span.end();
}
