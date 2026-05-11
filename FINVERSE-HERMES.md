# Finverse Hermes Sandbox Worker Plan

Status: Finverse integration target for using `rig-rlm` as the code-sandbox
worker behind the Hermes Substrate Bridge. This document intentionally narrows
the project surface for the first Finverse buildout.

Companion RS packet:
`/home/ananya-sharma/dev-stuff/feature/lancedb-integration/crates/impl-plans/agent-gateway+kgent/rig-rlm-integration-2026-05-11.md`

## 1. Role in the Finverse architecture

`rig-rlm` is not the Hermes bridge and not the Finverse authority plane.

Target role:

```text
Hermes
  -> hermes-substrate-bridge
      -> execute_code / request_sandbox
          -> rig-rlm sandbox worker
              -> stdout/stderr/artifacts/optional patch refs
      -> attach_artifact / proposal authority / workflow authority
```

The worker receives scoped inputs from the bridge and returns candidate
artifacts with provenance. It must not directly read or write Scylla, Turso,
ClickHouse, Lance, TigerBeetle, or Finverse Git proposal state. Those stores
stay behind the bridge/substrate APIs.

## 2. Adopted surface

Keep and harden these for Finverse:

- `CodeExecutor` trait
- `MicrosandboxExecutor`
- `Pyo3CodeExecutor` as dev/test fallback only
- safety and execution policy preflights
- HITL `ELICIT()` suspend/resume
- evidence trail, OTEL, Langfuse-compatible trace events
- optional A2A server for post-demo `code-executor-agent`
- optional Restate durable workflow for long-running sandbox runs
- `apply_patch` only for developer/worktree profile, never direct product apply

Defer or disable by default:

- `gnn/hehrgnn`
- Burn/GNN dependencies
- lambda self-learning / HyperAgent modes
- nuggets HRR memory
- DSPy prompt optimization
- free-monad core as the user-channel agent loop
- `llm_query()` from inside sandbox tasks

Hermes remains the user-channel loop. The Finverse bridge remains the identity,
Cedar preflight, audit, and proposal/workflow dispatch layer.

## 3. Required feature split

Current default build compiles `hehrgnn` because it is both a workspace member
and a normal dependency. Finverse needs a default build that excludes GNN.

Target feature shape:

```toml
[features]
default = ["sandbox-worker"]
sandbox-worker = ["microsandbox", "rmcp"]
a2a-worker = ["sandbox-worker"]
restate-worker = ["restate-sdk"]
dev-pyo3 = ["pyo3"]
research-rlm = ["dspy-rs"]
gnn = ["hehrgnn"]
```

Acceptance check after the split:

```bash
cargo tree -p rig-rlm --no-default-features --features sandbox-worker
```

The output must not include `hehrgnn`, `burn`, `burn-wgpu`, or the GNN test
stack.

## 4. Production profile

Finverse production defaults:

```text
executor = MicrosandboxExecutor
capabilities = code_worker
safety = strict
sub_llm_bridge = false
pyo3 = disabled
apply_patch = disabled unless worktree-medium profile
network = bridge/package-mirror only
direct_store_credentials = never
```

PyO3 remains useful for local development and tests, but it is not an isolation
boundary.

## 5. Finverse task contract

Expose a typed task runner, not raw "execute arbitrary Python".

Input shape:

```json
{
  "task_kind": "receipt_extract | bill_normalize | csv_ledger_preview | analytics_notebook | cargo_check | cabal_test | cedar_validate | patch_dry_run",
  "sandbox_profile": "code-exec-ephemeral | worktree-medium",
  "input_artifact_refs": ["artifact:..."],
  "expected_output_schema": "schema-ref-or-inline-json-schema",
  "purpose": "short user-visible reason",
  "max_runtime_seconds": 120
}
```

Output shape:

```json
{
  "sandbox_run_id": "uuid",
  "sandbox_uid": "uuid-or-local-dev-id",
  "task_kind": "receipt_extract",
  "stdout_ref": "artifact-or-log-ref",
  "stderr_ref": "artifact-or-log-ref",
  "artifacts": [
    {
      "artifact_digest": "sha256:...",
      "artifact_kind": "receipt_ocr_json",
      "source_artifact_ref": "artifact:raw-receipt-photo"
    }
  ],
  "patch_ref": null,
  "provenance": {
    "runtime": "RigRlmRuntime",
    "executor": "MicrosandboxExecutor",
    "policy_profile": "strict-code-exec",
    "image_digest": "sha256:...",
    "network_profile_hash": "..."
  }
}
```

The Finverse bridge owns mission IDs, agent IDs, subject IDs, origin refs,
scope refs, and Cedar verdict refs. `rig-rlm` should accept them as opaque
context for trace correlation, not mint authority identities.

## 6. First task profiles

Demo-critical:

- `receipt_extract`: image/PDF receipt -> structured candidate JSON
- `bill_normalize`: bill PDF -> vendor/amount/period/due-date candidate
- `csv_ledger_preview`: payout CSV -> draft ledger rows + evidence gaps
- `analytics_notebook`: scoped snapshot -> derived chart/table artifact
- `cargo_check`: Rust validation in worktree-medium profile
- `cabal_test`: Haskell validation in worktree-medium profile
- `cedar_validate`: Cedar policy bundle syntax/fixture validation
- `patch_dry_run`: developer-only patch generation against a bounded worktree

The worker should return artifacts and refs; the bridge decides whether the
artifact can be attached to a proposal.

## 7. HITL resume

`ELICIT()` maps naturally to Finverse approval/review flow:

```text
sandbox task needs human input
  -> worker suspends and snapshots
  -> bridge returns needs_approval / needs_review
  -> Hermes asks the user or reviewer
  -> bridge records approval/review
  -> worker resumes with the supplied answer
```

The worker must treat the resumed answer as input data, not as an authority
grant. Cedar/bridge remain the source of authorization.

## 8. A2A peer-agent mode

Post-demo, deploy `rig-rlm` as `code-executor-agent`:

- own AgentCard
- own AgentPrincipal
- `AgentKind::CodeExecutorAgent`
- `OriginRuntime::RigRlmRuntime`
- behind agentgateway `:8082`
- optional Restate durability
- streaming status events

Hermes still delegates through the bridge. Direct user access to the A2A
endpoint is not the product path.

## 9. Implementation checklist

1. Feature-gate GNN/Burn out of the default build.
2. Export a slim sandbox-worker API or crate.
3. Add production config that forces Microsandbox + strict safety.
4. Disable `llm_query()` and raw `apply_patch` by default.
5. Implement typed task profiles and response shape.
6. Add path guards for worktree tasks.
7. Digest stdout/stderr/artifacts and return refs.
8. Verify microsandbox server lifecycle locally and in target k8s profile.
9. Add tests proving no default build pulls `hehrgnn` and no production profile
   uses PyO3.

