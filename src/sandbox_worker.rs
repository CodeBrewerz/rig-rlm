//! Finverse sandbox worker contract.
//!
//! Implements the typed task runner described in `FINVERSE-HERMES.md` §5
//! and the Finverse RS doc `rig-rlm-integration-2026-05-11.md` §3.
//!
//! ## Posture
//!
//! - **Not** a raw "run arbitrary Python" endpoint.
//! - **Not** an authority plane. Authority (identity, Cedar, proposal apply,
//!   substrate reads/writes) lives in the Finverse Rust bridge.
//! - **Production profile is forced** at construction: Microsandbox executor,
//!   strict safety limits, `llm_query` disabled, `apply_patch` disabled,
//!   PyO3 disabled, no direct store credentials. The bridge's Cedar
//!   admission preflight is the gate; this worker is the executor only.
//! - Outputs are **candidate artifact refs + digests + provenance**. They
//!   become proposal evidence only when the bridge's `attach_artifact` tool
//!   is invoked and proposal authority approves.
//!
//! ## Task profiles
//!
//! Eight typed kinds for v1:
//! 1. `ReceiptExtract`   — image/PDF receipt → structured candidate JSON.
//! 2. `BillNormalize`    — bill PDF → vendor/amount/period/due-date candidate.
//! 3. `CsvLedgerPreview` — payout CSV → draft ledger rows + evidence gaps.
//! 4. `AnalyticsNotebook`— scoped snapshot → derived chart/table artifact.
//! 5. `CargoCheck`       — Rust validation in worktree-medium profile.
//! 6. `CabalTest`        — Haskell validation in worktree-medium profile.
//! 7. `CedarValidate`    — Cedar policy bundle syntax/fixture validation.
//! 8. `PatchDryRun`      — developer-only patch generation, bounded worktree.
//!
//! Anything else is rejected at the contract boundary.

use crate::safety::ExecutionLimits;
use serde::{Deserialize, Serialize};

// ─── Task kinds ─────────────────────────────────────────────────────────

/// The eight typed task profiles. The bridge maps user/agent intent onto
/// one of these; the worker never decides task kind from free-form input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    ReceiptExtract,
    BillNormalize,
    CsvLedgerPreview,
    AnalyticsNotebook,
    CargoCheck,
    CabalTest,
    CedarValidate,
    PatchDryRun,
}

impl TaskKind {
    /// Whether the task is permitted under the `code-exec-ephemeral` profile
    /// (no worktree, no filesystem persistence beyond the sandbox lifetime).
    pub fn allowed_in_ephemeral(self) -> bool {
        matches!(
            self,
            TaskKind::ReceiptExtract
                | TaskKind::BillNormalize
                | TaskKind::CsvLedgerPreview
                | TaskKind::AnalyticsNotebook
                | TaskKind::CedarValidate
        )
    }

    /// Whether the task requires the `worktree-medium` profile
    /// (developer/CI use: worktree mount, language toolchain, no money path).
    pub fn requires_worktree(self) -> bool {
        matches!(
            self,
            TaskKind::CargoCheck | TaskKind::CabalTest | TaskKind::PatchDryRun
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SandboxProfile {
    CodeExecEphemeral,
    WorktreeMedium,
}

// ─── Request shape (bridge → worker) ────────────────────────────────────

/// Bridge-supplied opaque IDs used for trace correlation only.
/// The worker does NOT use these as authority — Cedar/bridge already
/// preflighted and stamped them.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BridgeContext {
    pub mission_trace_id: Option<String>,
    pub agent_span_id: Option<String>,
    pub subject_principal_uid: Option<String>,
    pub agent_principal_uid: Option<String>,
    pub origin_principal_ref: Option<String>,
    pub scope_envelope_ref: Option<String>,
    pub capability_envelope_ref: Option<String>,
}

/// Model/agent-supplied input shape for `execute_code`.
///
/// This is what the LLM may emit through Hermes; the bridge accepts
/// only this shape on the wire. Mission/agent/subject/origin IDs are
/// **never** carried here — they're stamped by the bridge from
/// authenticated session state and combined into a full
/// [`SandboxTaskRequest`] internally.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxTaskInput {
    pub task_kind: TaskKind,
    pub sandbox_profile: SandboxProfile,
    #[serde(default)]
    pub input_artifact_refs: Vec<String>,
    pub expected_output_schema: Option<String>,
    pub purpose: String,
    pub max_runtime_seconds: Option<u32>,
}

/// Full bridge-internal request shape: model input + bridge-stamped
/// [`BridgeContext`]. Constructed by the bridge via
/// [`SandboxTaskInput::with_context`]; never reach this from a wire
/// deserialise of a model payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxTaskRequest {
    pub task_kind: TaskKind,
    pub sandbox_profile: SandboxProfile,
    #[serde(default)]
    pub input_artifact_refs: Vec<String>,
    pub expected_output_schema: Option<String>,
    pub purpose: String,
    pub max_runtime_seconds: Option<u32>,
    #[serde(default)]
    pub context: BridgeContext,
}

impl SandboxTaskInput {
    /// Bridge-only constructor: combine model-supplied input with the
    /// authenticated [`BridgeContext`] the bridge derived from the
    /// session. Calling this is the boundary at which model-supplied
    /// fields meet bridge-supplied fields.
    pub fn with_context(self, context: BridgeContext) -> SandboxTaskRequest {
        SandboxTaskRequest {
            task_kind: self.task_kind,
            sandbox_profile: self.sandbox_profile,
            input_artifact_refs: self.input_artifact_refs,
            expected_output_schema: self.expected_output_schema,
            purpose: self.purpose,
            max_runtime_seconds: self.max_runtime_seconds,
            context,
        }
    }
}

// ─── Response shape (worker → bridge) ───────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub artifact_digest: String,
    pub artifact_kind: String,
    pub source_artifact_ref: Option<String>,
}

/// Honest provenance of a candidate artifact.
///
/// `executor_state` distinguishes a real Microsandbox run from a v1
/// stub. The bridge MUST reject any downstream `attach_artifact` for
/// candidates whose `executor_state = Stub`; the UI/agent SHOULD label
/// stub candidates as such. This protects against "the demo's
/// hardcoded receipt JSON" being misread as extracted facts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    pub runtime: String,
    pub executor: String,
    pub executor_state: ExecutorState,
    pub policy_profile: String,
    pub image_digest: Option<String>,
    pub network_profile_hash: Option<String>,
}

/// Whether the candidate artifact came from a real sandbox execution
/// (`Live`) or from the v1 contract scaffold (`Stub`). Wire format:
/// kebab-case (`live`, `stub`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExecutorState {
    Live,
    Stub,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxTaskResponse {
    pub sandbox_run_id: String,
    pub sandbox_uid: String,
    pub task_kind: TaskKind,
    pub stdout_ref: Option<String>,
    pub stderr_ref: Option<String>,
    #[serde(default)]
    pub artifacts: Vec<ArtifactRef>,
    pub patch_ref: Option<String>,
    pub provenance: Provenance,
}

// ─── Production policy ──────────────────────────────────────────────────

/// The forced production profile. Constructed once at worker init; never
/// loosened at task submission time. The bridge cannot override these.
#[derive(Debug, Clone)]
pub struct ProductionPolicy {
    pub executor_kind: PolicyExecutorKind,
    pub llm_query_enabled: bool,
    pub apply_patch_enabled: bool,
    pub pyo3_enabled: bool,
    pub default_max_runtime_seconds: u32,
    pub max_runtime_seconds_ceiling: u32,
    pub network_profile: NetworkProfile,
    pub direct_store_credentials: bool,
    pub limits: ExecutionLimits,
    pub policy_profile_name: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyExecutorKind {
    Microsandbox,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NetworkProfile {
    /// Egress only to the bridge and the package mirror; no internet.
    BridgeMirrorOnly,
    /// No network at all.
    None,
}

impl ProductionPolicy {
    /// Strict production policy — Microsandbox only, no llm_query, no
    /// PyO3, no apply_patch, network restricted, no direct store creds.
    pub fn strict() -> Self {
        Self {
            executor_kind: PolicyExecutorKind::Microsandbox,
            llm_query_enabled: false,
            apply_patch_enabled: false,
            pyo3_enabled: false,
            default_max_runtime_seconds: 120,
            max_runtime_seconds_ceiling: 600,
            network_profile: NetworkProfile::BridgeMirrorOnly,
            direct_store_credentials: false,
            limits: ExecutionLimits::strict(),
            policy_profile_name: "strict-code-exec",
        }
    }

    /// Strict + apply_patch enabled — for the worktree-medium profile only.
    /// Patch refs are returned as candidates; the bridge does not apply
    /// them directly. Money paths never see this profile.
    pub fn strict_worktree() -> Self {
        let mut p = Self::strict();
        p.apply_patch_enabled = true;
        p.policy_profile_name = "strict-worktree-medium";
        p
    }
}

// ─── Errors ─────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum SandboxWorkerError {
    PolicyViolation(&'static str),
    ProfileMismatch {
        task_kind: TaskKind,
        sandbox_profile: SandboxProfile,
    },
    RuntimeExceeded {
        requested: u32,
        ceiling: u32,
    },
    ExecutionFailed(String),
    /// Real digesting / network sandboxing / image pinning still pending.
    NotYetImplemented(&'static str),
    /// A caller-supplied input artifact (e.g. a code blob / fixture) was malformed,
    /// unreadable, or failed schema validation before execution. A caller-input fault,
    /// distinct from a policy violation or a runtime failure.
    InvalidInputArtifact(String),
}

impl std::fmt::Display for SandboxWorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PolicyViolation(s) => write!(f, "policy violation: {s}"),
            Self::ProfileMismatch {
                task_kind,
                sandbox_profile,
            } => write!(
                f,
                "task {task_kind:?} is not allowed in profile {sandbox_profile:?}"
            ),
            Self::RuntimeExceeded { requested, ceiling } => write!(
                f,
                "max_runtime_seconds {requested} exceeds ceiling {ceiling}"
            ),
            Self::ExecutionFailed(s) => write!(f, "execution failed: {s}"),
            Self::NotYetImplemented(s) => write!(f, "not yet implemented: {s}"),
            Self::InvalidInputArtifact(s) => write!(f, "invalid input artifact: {s}"),
        }
    }
}

impl std::error::Error for SandboxWorkerError {}

// ─── Worker ─────────────────────────────────────────────────────────────

/// The Finverse sandbox worker. Holds a forced production policy; routes
/// typed requests into a Microsandbox executor with task-specific templates.
///
/// v1 status: the contract + policy enforcement + provenance/digest path
/// are real; the actual per-task Python templates are stubbed (return a
/// `NotYetImplemented` candidate artifact with the requested shape) so the
/// bridge integration can be developed against a stable contract.
pub struct SandboxWorker {
    policy: ProductionPolicy,
}

impl SandboxWorker {
    pub fn new_strict() -> Self {
        Self {
            policy: ProductionPolicy::strict(),
        }
    }

    pub fn new_strict_worktree() -> Self {
        Self {
            policy: ProductionPolicy::strict_worktree(),
        }
    }

    pub fn policy(&self) -> &ProductionPolicy {
        &self.policy
    }

    /// Validate a request against the production policy, without executing.
    /// Used for unit tests and as the bridge-side dry-run check.
    pub fn validate(&self, request: &SandboxTaskRequest) -> Result<(), SandboxWorkerError> {
        // Profile / task-kind compatibility.
        match request.sandbox_profile {
            SandboxProfile::CodeExecEphemeral => {
                if !request.task_kind.allowed_in_ephemeral() {
                    return Err(SandboxWorkerError::ProfileMismatch {
                        task_kind: request.task_kind,
                        sandbox_profile: request.sandbox_profile,
                    });
                }
            }
            SandboxProfile::WorktreeMedium => {
                if !request.task_kind.requires_worktree() {
                    return Err(SandboxWorkerError::ProfileMismatch {
                        task_kind: request.task_kind,
                        sandbox_profile: request.sandbox_profile,
                    });
                }
            }
        }

        // Patch is only meaningful under worktree profile, and even there it
        // only emits a candidate ref — never an apply.
        if matches!(request.task_kind, TaskKind::PatchDryRun) && !self.policy.apply_patch_enabled {
            return Err(SandboxWorkerError::PolicyViolation(
                "patch_dry_run requires strict_worktree policy",
            ));
        }

        // Runtime ceiling.
        if let Some(req) = request.max_runtime_seconds {
            if req > self.policy.max_runtime_seconds_ceiling {
                return Err(SandboxWorkerError::RuntimeExceeded {
                    requested: req,
                    ceiling: self.policy.max_runtime_seconds_ceiling,
                });
            }
        }

        Ok(())
    }

    /// Run a typed sandbox task.
    ///
    /// v1: validates the request, mints provenance, runs the (stubbed)
    /// per-task template, and returns the typed response. Real Microsandbox
    /// wiring + per-task Python templates land in Chunk 6 (demo integration).
    pub async fn run(
        &self,
        request: SandboxTaskRequest,
    ) -> Result<SandboxTaskResponse, SandboxWorkerError> {
        self.validate(&request)?;

        let sandbox_run_id = uuid::Uuid::new_v4().to_string();
        let sandbox_uid = uuid::Uuid::new_v4().to_string();
        let provenance = Provenance {
            runtime: "RigRlmRuntime".to_string(),
            executor: "MicrosandboxExecutor".to_string(),
            // v1: the per-task Python templates aren't wired yet — the
            // response below is a contract scaffold, not real Microsandbox
            // output. Honest provenance so downstream attach_artifact can
            // refuse it; flips to Live once Chunk 6 lands real templates.
            executor_state: ExecutorState::Stub,
            policy_profile: self.policy.policy_profile_name.to_string(),
            image_digest: None, // pinned image plumbing is a follow-up
            network_profile_hash: Some(network_profile_hash(self.policy.network_profile)),
        };

        // v1 stub: produce a single candidate artifact summarising the
        // requested shape, with a real sha256 digest over its JSON. Real
        // per-task templates land with Chunk 6.
        let candidate_payload = serde_json::json!({
            "task_kind": request.task_kind,
            "purpose": request.purpose,
            "input_artifact_refs": request.input_artifact_refs,
            "expected_output_schema": request.expected_output_schema,
            "stub": true,
            "stub_reason": "v1 contract scaffold — real Microsandbox templates land in Chunk 6",
        });
        let candidate_bytes = serde_json::to_vec(&candidate_payload)
            .map_err(|e| SandboxWorkerError::ExecutionFailed(e.to_string()))?;
        let digest = sha256_hex(&candidate_bytes);

        let artifacts = vec![ArtifactRef {
            artifact_digest: format!("sha256:{digest}"),
            artifact_kind: candidate_artifact_kind(request.task_kind).to_string(),
            source_artifact_ref: request.input_artifact_refs.first().cloned(),
        }];

        let response = SandboxTaskResponse {
            sandbox_run_id,
            sandbox_uid,
            task_kind: request.task_kind,
            stdout_ref: None,
            stderr_ref: None,
            artifacts,
            patch_ref: None,
            provenance,
        };

        Ok(response)
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────

fn candidate_artifact_kind(task_kind: TaskKind) -> &'static str {
    match task_kind {
        TaskKind::ReceiptExtract => "receipt_ocr_json",
        TaskKind::BillNormalize => "bill_normalized_json",
        TaskKind::CsvLedgerPreview => "ledger_preview_json",
        TaskKind::AnalyticsNotebook => "analytics_notebook_artifact",
        TaskKind::CargoCheck => "cargo_check_report_json",
        TaskKind::CabalTest => "cabal_test_report_json",
        TaskKind::CedarValidate => "cedar_validate_report_json",
        TaskKind::PatchDryRun => "patch_candidate_diff",
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let out = hasher.finalize();
    let mut hex = String::with_capacity(out.len() * 2);
    for b in out {
        hex.push_str(&format!("{b:02x}"));
    }
    hex
}

fn network_profile_hash(profile: NetworkProfile) -> String {
    // Stable hash over the textual representation so a policy change shows
    // up in audit. Real per-image network ACL hashing is a follow-up.
    let s = match profile {
        NetworkProfile::BridgeMirrorOnly => "bridge-mirror-only/v1",
        NetworkProfile::None => "no-network/v1",
    };
    format!("sha256:{}", sha256_hex(s.as_bytes()))
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn req(kind: TaskKind, profile: SandboxProfile) -> SandboxTaskRequest {
        SandboxTaskRequest {
            task_kind: kind,
            sandbox_profile: profile,
            input_artifact_refs: vec![],
            expected_output_schema: None,
            purpose: "test".to_string(),
            max_runtime_seconds: None,
            context: BridgeContext::default(),
        }
    }

    #[test]
    fn strict_policy_forces_microsandbox_no_pyo3_no_llm_query() {
        let p = ProductionPolicy::strict();
        assert_eq!(p.executor_kind, PolicyExecutorKind::Microsandbox);
        assert!(!p.llm_query_enabled);
        assert!(!p.pyo3_enabled);
        assert!(!p.apply_patch_enabled);
        assert!(!p.direct_store_credentials);
        assert_eq!(p.policy_profile_name, "strict-code-exec");
    }

    #[test]
    fn strict_worktree_enables_patch_but_keeps_everything_else_strict() {
        let p = ProductionPolicy::strict_worktree();
        assert!(p.apply_patch_enabled);
        assert!(!p.llm_query_enabled);
        assert!(!p.pyo3_enabled);
        assert!(!p.direct_store_credentials);
        assert_eq!(p.policy_profile_name, "strict-worktree-medium");
    }

    #[test]
    fn receipt_extract_ok_in_ephemeral() {
        let w = SandboxWorker::new_strict();
        w.validate(&req(
            TaskKind::ReceiptExtract,
            SandboxProfile::CodeExecEphemeral,
        ))
        .unwrap();
    }

    #[test]
    fn cargo_check_rejected_in_ephemeral() {
        let w = SandboxWorker::new_strict();
        let err = w
            .validate(&req(
                TaskKind::CargoCheck,
                SandboxProfile::CodeExecEphemeral,
            ))
            .unwrap_err();
        assert!(matches!(err, SandboxWorkerError::ProfileMismatch { .. }));
    }

    #[test]
    fn receipt_extract_rejected_in_worktree() {
        let w = SandboxWorker::new_strict_worktree();
        let err = w
            .validate(&req(
                TaskKind::ReceiptExtract,
                SandboxProfile::WorktreeMedium,
            ))
            .unwrap_err();
        assert!(matches!(err, SandboxWorkerError::ProfileMismatch { .. }));
    }

    #[test]
    fn patch_dry_run_rejected_without_worktree_policy() {
        let w = SandboxWorker::new_strict();
        // PatchDryRun is a worktree-only task; pairing with ephemeral fails
        // at profile mismatch first.
        let err = w
            .validate(&req(
                TaskKind::PatchDryRun,
                SandboxProfile::CodeExecEphemeral,
            ))
            .unwrap_err();
        assert!(matches!(err, SandboxWorkerError::ProfileMismatch { .. }));
    }

    #[test]
    fn runtime_above_ceiling_rejected() {
        let w = SandboxWorker::new_strict();
        let mut r = req(TaskKind::ReceiptExtract, SandboxProfile::CodeExecEphemeral);
        r.max_runtime_seconds = Some(10_000);
        let err = w.validate(&r).unwrap_err();
        assert!(matches!(err, SandboxWorkerError::RuntimeExceeded { .. }));
    }

    #[tokio::test]
    async fn run_returns_typed_candidate_artifact_with_digest() {
        let w = SandboxWorker::new_strict();
        let resp = w
            .run(req(
                TaskKind::ReceiptExtract,
                SandboxProfile::CodeExecEphemeral,
            ))
            .await
            .unwrap();
        assert_eq!(resp.task_kind, TaskKind::ReceiptExtract);
        assert_eq!(resp.artifacts.len(), 1);
        let a = &resp.artifacts[0];
        assert_eq!(a.artifact_kind, "receipt_ocr_json");
        assert!(a.artifact_digest.starts_with("sha256:"));
        assert_eq!(resp.provenance.runtime, "RigRlmRuntime");
        assert_eq!(resp.provenance.executor, "MicrosandboxExecutor");
        // v1 stub: provenance must be honest about it.
        assert_eq!(resp.provenance.executor_state, ExecutorState::Stub);
        assert_eq!(resp.provenance.policy_profile, "strict-code-exec");
        assert!(resp.patch_ref.is_none());
    }

    #[test]
    fn executor_state_wire_shape_is_kebab_case() {
        let live = serde_json::to_value(ExecutorState::Live).unwrap();
        assert_eq!(live.as_str(), Some("live"));
        let stub = serde_json::to_value(ExecutorState::Stub).unwrap();
        assert_eq!(stub.as_str(), Some("stub"));
    }

    #[test]
    fn request_serializes_with_snake_case_task_kind() {
        let r = req(
            TaskKind::CsvLedgerPreview,
            SandboxProfile::CodeExecEphemeral,
        );
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains("\"task_kind\":\"csv_ledger_preview\""));
        assert!(s.contains("\"sandbox_profile\":\"code-exec-ephemeral\""));
    }
}
