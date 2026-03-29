//! GNN-based fiduciary next-action prediction.
//!
//! Uses the heterogeneous graph GNN embeddings to score candidate financial
//! actions for a user, ranked by fiduciary benefit. Each action is a potential
//! (user, action_type, target) triple scored using:
//!
//! 1. **Embedding affinity**: cosine similarity between user and target embeddings
//! 2. **Anomaly signal**: anomaly score of the target (risk indicator)
//! 3. **Neighborhood context**: structural importance of the target
//! 4. **Fiduciary axes**: cost_reduction, risk_reduction, goal_alignment, urgency
//!
//! Action types are derived from the TQL ontology (SchemaFinverse.tql):
//! - 163 entity types, 307 relation types mapped to 18 fiduciary actions.

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════
// Fiduciary Action Types — derived from TQL ontology
// ═══════════════════════════════════════════════════════════════

/// The type of fiduciary action to recommend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum FiduciaryActionType {
    // ── Core (original 6) ──
    /// Pay down a debt or bill.
    ShouldPay,
    /// Cancel unused or wasteful subscription/service.
    ShouldCancel,
    /// Transfer money between accounts for optimization.
    ShouldTransfer,
    /// Avoid/reduce activity with this entity.
    ShouldAvoid,
    /// Investigate suspicious or anomalous activity.
    ShouldInvestigate,
    /// Consolidate redundant accounts or services.
    ShouldConsolidate,

    // ── Debt & Obligations ──
    /// Refinance an obligation to get a better interest rate.
    ShouldRefinance,
    /// Pay down a lien on an asset to increase equity.
    ShouldPayDownLien,
    /// Dispute an anomalous or incorrect obligation.
    ShouldDispute,

    // ── Goals & Budgets ──
    /// Fund an underfunded financial goal.
    ShouldFundGoal,
    /// Adjust budget when estimation diverges from actuals.
    ShouldAdjustBudget,

    // ── Tax Optimization ──
    /// Prepare for an upcoming tax due event.
    ShouldPrepareTax,
    /// Fund a tax sinking fund to cover upcoming liabilities.
    ShouldFundTaxSinking,
    /// Claim an unused tax exemption certificate.
    ShouldClaimExemption,
    /// Run a tax scenario to identify potential savings.
    ShouldRunTaxScenario,

    // ── Reconciliation ──
    /// Reconcile an instrument with unmatched items.
    ShouldReconcile,

    // ── Recurring Patterns ──
    /// Review a recurring pattern that fired an alert.
    ShouldReviewRecurring,

    // ── Assets ──
    /// Revalue an asset with stale valuation.
    ShouldRevalueAsset,
}

impl FiduciaryActionType {
    pub fn all() -> Vec<Self> {
        vec![
            // Core
            Self::ShouldPay,
            Self::ShouldCancel,
            Self::ShouldTransfer,
            Self::ShouldAvoid,
            Self::ShouldInvestigate,
            Self::ShouldConsolidate,
            // Debt & Obligations
            Self::ShouldRefinance,
            Self::ShouldPayDownLien,
            Self::ShouldDispute,
            // Goals & Budgets
            Self::ShouldFundGoal,
            Self::ShouldAdjustBudget,
            // Tax
            Self::ShouldPrepareTax,
            Self::ShouldFundTaxSinking,
            Self::ShouldClaimExemption,
            Self::ShouldRunTaxScenario,
            // Reconciliation
            Self::ShouldReconcile,
            // Recurring
            Self::ShouldReviewRecurring,
            // Assets
            Self::ShouldRevalueAsset,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::ShouldPay => "should_pay",
            Self::ShouldCancel => "should_cancel",
            Self::ShouldTransfer => "should_transfer",
            Self::ShouldAvoid => "should_avoid",
            Self::ShouldInvestigate => "should_investigate",
            Self::ShouldConsolidate => "should_consolidate",
            Self::ShouldRefinance => "should_refinance",
            Self::ShouldPayDownLien => "should_pay_down_lien",
            Self::ShouldDispute => "should_dispute",
            Self::ShouldFundGoal => "should_fund_goal",
            Self::ShouldAdjustBudget => "should_adjust_budget",
            Self::ShouldPrepareTax => "should_prepare_tax",
            Self::ShouldFundTaxSinking => "should_fund_tax_sinking",
            Self::ShouldClaimExemption => "should_claim_exemption",
            Self::ShouldRunTaxScenario => "should_run_tax_scenario",
            Self::ShouldReconcile => "should_reconcile",
            Self::ShouldReviewRecurring => "should_review_recurring",
            Self::ShouldRevalueAsset => "should_revalue_asset",
        }
    }

    /// Parse action type from string.
    ///
    /// Accepts:
    /// - canonical snake_case (`should_investigate`)
    /// - kebab/camel/Pascal variants (`should-investigate`, `ShouldInvestigate`)
    pub fn try_from_name(s: &str) -> Option<Self> {
        let normalized = normalize_action_name(s);
        match normalized.as_str() {
            "should_pay" => Some(Self::ShouldPay),
            "should_cancel" => Some(Self::ShouldCancel),
            "should_transfer" => Some(Self::ShouldTransfer),
            "should_avoid" => Some(Self::ShouldAvoid),
            "should_investigate" => Some(Self::ShouldInvestigate),
            "should_consolidate" => Some(Self::ShouldConsolidate),
            "should_refinance" => Some(Self::ShouldRefinance),
            "should_pay_down_lien" => Some(Self::ShouldPayDownLien),
            "should_dispute" => Some(Self::ShouldDispute),
            "should_fund_goal" => Some(Self::ShouldFundGoal),
            "should_adjust_budget" => Some(Self::ShouldAdjustBudget),
            "should_prepare_tax" => Some(Self::ShouldPrepareTax),
            "should_fund_tax_sinking" => Some(Self::ShouldFundTaxSinking),
            "should_claim_exemption" => Some(Self::ShouldClaimExemption),
            "should_run_tax_scenario" => Some(Self::ShouldRunTaxScenario),
            "should_reconcile" => Some(Self::ShouldReconcile),
            "should_review_recurring" => Some(Self::ShouldReviewRecurring),
            "should_revalue_asset" => Some(Self::ShouldRevalueAsset),
            _ => None,
        }
    }

    /// Parse action type from its string name.
    pub fn from_name(s: &str) -> Self {
        Self::try_from_name(s).unwrap_or(Self::ShouldInvestigate)
    }

    pub fn verb(&self) -> &'static str {
        match self {
            Self::ShouldPay => "Pay",
            Self::ShouldCancel => "Cancel",
            Self::ShouldTransfer => "Transfer to",
            Self::ShouldAvoid => "Avoid",
            Self::ShouldInvestigate => "Investigate",
            Self::ShouldConsolidate => "Consolidate",
            Self::ShouldRefinance => "Refinance",
            Self::ShouldPayDownLien => "Pay down lien on",
            Self::ShouldDispute => "Dispute",
            Self::ShouldFundGoal => "Fund goal via",
            Self::ShouldAdjustBudget => "Adjust budget for",
            Self::ShouldPrepareTax => "Prepare tax for",
            Self::ShouldFundTaxSinking => "Fund tax sinking fund for",
            Self::ShouldClaimExemption => "Claim exemption on",
            Self::ShouldRunTaxScenario => "Run tax scenario for",
            Self::ShouldReconcile => "Reconcile",
            Self::ShouldReviewRecurring => "Review recurring pattern for",
            Self::ShouldRevalueAsset => "Revalue",
        }
    }

    pub fn priority_weight(&self) -> f32 {
        match self {
            Self::ShouldInvestigate => 1.0,      // Highest: fraud/anomalies
            Self::ShouldPay => 0.95,             // Very high: avoid penalties
            Self::ShouldPrepareTax => 0.93,      // Very high: tax deadlines
            Self::ShouldDispute => 0.92,         // High: erroneous charges
            Self::ShouldAvoid => 0.90,           // High: risk reduction
            Self::ShouldPayDownLien => 0.88,     // High: equity building
            Self::ShouldFundTaxSinking => 0.85,  // High: tax preparedness
            Self::ShouldRefinance => 0.82,       // High: interest savings
            Self::ShouldClaimExemption => 0.80,  // High: tax optimization
            Self::ShouldReconcile => 0.78,       // Above-medium: accuracy
            Self::ShouldFundGoal => 0.75,        // Medium-high: goal progress
            Self::ShouldCancel => 0.70,          // Medium: cost savings
            Self::ShouldReviewRecurring => 0.68, // Medium: pattern monitoring
            Self::ShouldRunTaxScenario => 0.65,  // Medium: planning
            Self::ShouldTransfer => 0.60,        // Medium: optimization
            Self::ShouldAdjustBudget => 0.55,    // Lower: planning
            Self::ShouldConsolidate => 0.50,     // Lower: convenience
            Self::ShouldRevalueAsset => 0.75, // HNW: survive PC blend (stale values → tax/estate risk)
        }
    }

    pub fn reasoning_suffix(&self) -> &'static str {
        match self {
            Self::ShouldInvestigate => {
                "This entity shows anomalous patterns — manual review recommended."
            }
            Self::ShouldAvoid => "Reducing interaction with this entity reduces financial risk.",
            Self::ShouldPay => "Timely payment avoids penalties and builds financial health.",
            Self::ShouldCancel => "Low engagement signal suggests potential cost savings.",
            Self::ShouldTransfer => "Rebalancing funds could optimize returns.",
            Self::ShouldConsolidate => "Fewer accounts reduce fee duplication and complexity.",
            Self::ShouldRefinance => {
                "Refinancing at a lower rate reduces total interest paid over the loan term."
            }
            Self::ShouldPayDownLien => {
                "Paying down liens increases your equity position and may reduce insurance costs."
            }
            Self::ShouldDispute => {
                "Anomalous obligation amounts should be disputed promptly to limit liability."
            }
            Self::ShouldFundGoal => {
                "Goal is underfunded — regular contributions keep you on track."
            }
            Self::ShouldAdjustBudget => {
                "Actual spending diverges from budget estimate — adjustment improves forecast accuracy."
            }
            Self::ShouldPrepareTax => {
                "Tax due event approaching — early preparation avoids penalties and interest."
            }
            Self::ShouldFundTaxSinking => {
                "Tax sinking fund is below target — funding now prevents cash flow surprises."
            }
            Self::ShouldClaimExemption => {
                "Eligible exemption certificate not yet applied — claiming it reduces tax liability."
            }
            Self::ShouldRunTaxScenario => {
                "Running a what-if tax scenario could reveal savings opportunities before period closes."
            }
            Self::ShouldReconcile => {
                "Unmatched items detected — reconciliation ensures ledger accuracy."
            }
            Self::ShouldReviewRecurring => {
                "Recurring pattern triggered an alert — review to confirm it's still valid."
            }
            Self::ShouldRevalueAsset => {
                "Asset valuation may be stale — updating ensures accurate net worth reporting."
            }
        }
    }

    /// The TQL ontology domain this action belongs to.
    pub fn domain(&self) -> &'static str {
        match self {
            Self::ShouldPay
            | Self::ShouldCancel
            | Self::ShouldTransfer
            | Self::ShouldAvoid
            | Self::ShouldInvestigate
            | Self::ShouldConsolidate => "core",
            Self::ShouldRefinance | Self::ShouldPayDownLien | Self::ShouldDispute => {
                "debt_obligations"
            }
            Self::ShouldFundGoal | Self::ShouldAdjustBudget => "goals_budgets",
            Self::ShouldPrepareTax
            | Self::ShouldFundTaxSinking
            | Self::ShouldClaimExemption
            | Self::ShouldRunTaxScenario => "tax_optimization",
            Self::ShouldReconcile => "reconciliation",
            Self::ShouldReviewRecurring => "recurring_patterns",
            Self::ShouldRevalueAsset => "asset_management",
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Fiduciary Scoring Axes
// ═══════════════════════════════════════════════════════════════

/// Scores on the 6 fiduciary axes (all 0.0–1.0, higher = more beneficial).
#[derive(Debug, Clone, serde::Serialize)]
pub struct FiduciaryAxes {
    pub cost_reduction: f32,
    pub risk_reduction: f32,
    pub goal_alignment: f32,
    pub urgency: f32,
    pub conflict_freedom: f32,
    pub reversibility: f32,
}

impl FiduciaryAxes {
    /// Weighted fiduciary score using default weights.
    pub fn score(&self) -> f32 {
        self.score_with_weights(&DEFAULT_AXES_WEIGHTS)
    }

    /// Weighted fiduciary score using GEPA-optimized weights.
    /// Weight order: [cost, risk, goal, urgency, conflict, reversibility].
    pub fn score_with_weights(&self, weights: &[f32; 6]) -> f32 {
        let values = [
            self.cost_reduction,
            self.risk_reduction,
            self.goal_alignment,
            self.urgency,
            self.conflict_freedom,
            self.reversibility,
        ];
        weights.iter().zip(values.iter()).map(|(w, v)| w * v).sum()
    }
}

/// Default fiduciary axes weights (used when no GEPA-optimized weights are available).
pub const DEFAULT_AXES_WEIGHTS: [f32; 6] = [0.25, 0.25, 0.15, 0.15, 0.10, 0.10];

/// Default GNN/PC blend weights.
pub const DEFAULT_GNN_WEIGHT: f32 = 0.7;
pub const DEFAULT_PC_WEIGHT: f32 = 0.3;

/// Path where GEPA saves optimized weights.
pub const GEPA_WEIGHTS_PATH: &str = "gepa_weights.json";

/// Load GNN/PC blend weights from GEPA-optimized config, or use defaults.
pub fn load_blend_weights() -> (f32, f32) {
    use crate::optimizer::gepa::OptimizedWeights;
    let w = OptimizedWeights::load_or_default(GEPA_WEIGHTS_PATH);
    if w.total_evals > 0 {
        (w.gnn_weight, w.pc_weight)
    } else {
        (DEFAULT_GNN_WEIGHT, DEFAULT_PC_WEIGHT)
    }
}

/// Load fiduciary axes weights from GEPA-optimized config, or use defaults.
pub fn load_axes_weights() -> [f32; 6] {
    use crate::optimizer::gepa::OptimizedWeights;
    let w = OptimizedWeights::load_or_default(GEPA_WEIGHTS_PATH);
    if w.total_evals > 0 {
        w.axes_weights()
    } else {
        DEFAULT_AXES_WEIGHTS
    }
}

// ═══════════════════════════════════════════════════════════════
// Response types
// ═══════════════════════════════════════════════════════════════

/// A scored fiduciary action recommendation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FiduciaryRecommendation {
    pub rank: usize,
    pub action_type: String,
    pub domain: String,
    pub target_name: String,
    pub target_node_type: String,
    pub target_node_id: usize,
    /// Overall fiduciary score (0.0–1.0).
    pub fiduciary_score: f32,
    /// Breakdown by fiduciary axis.
    pub axes: FiduciaryAxes,
    /// GNN embedding affinity between user and target.
    pub embedding_affinity: f32,
    /// Anomaly score of target.
    pub target_anomaly_score: f32,
    /// Human-readable explanation.
    pub reasoning: String,
    /// Is this action recommended or just informational?
    pub is_recommended: bool,
    /// PC circuit analysis: calibrated risk, lift factors, counterfactuals.
    /// `None` if PC training data is insufficient.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pc_analysis: Option<crate::model::pc::fiduciary_pc::PcAnalysis>,
    /// Learned scorer logit from asymmetric RL (paper 2402.18246).
    /// Positive = model endorses recommendation, negative = model advises caution.
    /// `None` if scorer is not available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scorer_logit: Option<f32>,
}

/// Full fiduciary response.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FiduciaryResponse {
    pub user_node_type: String,
    pub user_node_id: usize,
    pub user_name: String,
    /// Recommended actions, sorted by fiduciary score (highest first).
    pub recommendations: Vec<FiduciaryRecommendation>,
    /// Coverage: how many of the 18 action types were triggered.
    pub action_types_triggered: usize,
    /// Which ontology domains have triggered actions.
    pub domains_covered: Vec<String>,
    /// Overall assessment.
    pub assessment: String,
    /// Which GNN models contributed to the scoring.
    pub models_used: Vec<String>,
    /// SAE interpretability explanation (if SAE is trained).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sae_explanation: Option<crate::eval::sae::SaeExplanation>,
    /// Whether the PC circuit was trained and used for this response.
    pub pc_trained: bool,
    /// PC EM log-likelihood (higher = better fit to data).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pc_em_ll: Option<f64>,
}

/// Context needed for fiduciary scoring.
pub struct FiduciaryContext<'a> {
    /// User embedding (from default/SAGE model).
    pub user_emb: &'a [f32],
    /// All node embeddings by type.
    pub embeddings: &'a HashMap<String, Vec<Vec<f32>>>,
    /// Anomaly scores: model → node_type → scores.
    pub anomaly_scores: &'a HashMap<String, HashMap<String, Vec<f32>>>,
    /// Graph edges.
    pub edges: &'a HashMap<(String, String, String), Vec<(usize, usize)>>,
    /// Node names.
    pub node_names: &'a HashMap<String, Vec<String>>,
    /// Node counts.
    pub node_counts: &'a HashMap<String, usize>,
    /// User's node type and id.
    pub user_type: String,
    pub user_id: usize,
    /// Hidden dim.
    pub hidden_dim: usize,
}

/// Persistent PC circuit state for self-learning across recommend() calls.
///
/// The caller owns this and passes `&mut PcState` to `recommend()` across
/// multiple calls. The circuit accumulates learning — each call runs a few
/// incremental EM epochs on the existing circuit rather than rebuilding.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PcState {
    /// The trained circuit (None = needs initial build).
    pub circuit: Option<crate::model::pc::circuit::CompiledCircuit>,
    /// Total EM epochs trained so far.
    pub total_epochs: usize,
    /// Log-likelihood history across calls.
    pub ll_history: Vec<f64>,
}

impl PcState {
    /// Create a new empty PcState (circuit will be built on first recommend() call).
    pub fn new() -> Self {
        Self {
            circuit: None,
            total_epochs: 0,
            ll_history: Vec::new(),
        }
    }

    /// Whether the circuit has been trained.
    pub fn is_trained(&self) -> bool {
        self.circuit.is_some()
    }
}

// ═══════════════════════════════════════════════════════════════
// Action Candidate Generation — driven by TQL ontology relations
// ═══════════════════════════════════════════════════════════════

/// Generate candidate actions by BFS-traversing the user's graph neighborhood.
///
/// Uses generic breadth-first search up to `MAX_DEPTH` hops from the user node.
/// This automatically discovers all reachable entities regardless of the TQL schema
/// topology — no need to manually specify hop counts per entity type.
///
/// Typical traversal paths:
/// - depth 1: user → account, goal, budget, tax-obligation, asset, recurring
/// - depth 2: account → obligation, merchant, transaction, reconciliation
/// - depth 3: obligation → rate (refinance), obligation → asset (lien)
pub fn generate_candidates(ctx: &FiduciaryContext) -> Vec<(FiduciaryActionType, String, usize)> {
    /// Maximum BFS depth from user node.
    /// 3 covers: user→account→obligation→rate (the deepest common chain).
    /// Increase if schema has deeper fiduciary-relevant chains.
    const MAX_DEPTH: usize = 3;

    let mut candidates = Vec::new();
    let mut visited: std::collections::HashSet<(String, usize)> = std::collections::HashSet::new();

    // BFS frontier: (node_type, node_id, current_depth)
    let mut frontier: std::collections::VecDeque<(String, usize, usize)> =
        std::collections::VecDeque::new();

    // Seed with user
    visited.insert((ctx.user_type.clone(), ctx.user_id));
    frontier.push_back((ctx.user_type.clone(), ctx.user_id, 0));

    while let Some((cur_type, cur_id, depth)) = frontier.pop_front() {
        if depth >= MAX_DEPTH {
            continue;
        }

        // Scan all edges where current node is src or dst
        for ((src_type, relation, dst_type), edge_list) in ctx.edges {
            // Current node as source
            if src_type == &cur_type {
                for &(src_id, dst_id) in edge_list {
                    if src_id == cur_id {
                        if dst_type == &ctx.user_type && dst_id == ctx.user_id {
                            continue; // Never recommend an action whose target is the same user node.
                        }
                        let neighbor_key = (dst_type.clone(), dst_id);
                        if is_dead_node(ctx, dst_type, dst_id) {
                            continue;
                        }

                        // Infer actions for this relation even if the node was seen via
                        // another path; relation semantics can unlock different actions.
                        let actions = infer_actions(relation, dst_type, ctx, dst_id);
                        for action in actions {
                            candidates.push((action, dst_type.clone(), dst_id));
                        }

                        if !visited.contains(&neighbor_key) {
                            visited.insert(neighbor_key);
                            frontier.push_back((dst_type.clone(), dst_id, depth + 1));
                        }
                    }
                }
            }
            // Current node as destination
            if dst_type == &cur_type {
                for &(src_id, dst_id) in edge_list {
                    if dst_id == cur_id {
                        if src_type == &ctx.user_type && src_id == ctx.user_id {
                            continue; // Never recommend an action whose target is the same user node.
                        }
                        let neighbor_key = (src_type.clone(), src_id);
                        if is_dead_node(ctx, src_type, src_id) {
                            continue;
                        }

                        // Same rationale as above: relation-specific inference should run
                        // even for already visited nodes.
                        let actions = infer_actions(relation, src_type, ctx, src_id);
                        for action in actions {
                            candidates.push((action, src_type.clone(), src_id));
                        }

                        if !visited.contains(&neighbor_key) {
                            visited.insert(neighbor_key);
                            frontier.push_back((src_type.clone(), src_id, depth + 1));
                        }
                    }
                }
            }
        }
    }

    // Deduplicate
    candidates.sort_by(|a, b| {
        a.0.name()
            .cmp(b.0.name())
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
    });
    candidates.dedup();

    candidates
}

/// Check if a node has been soft-deleted (all-zero embedding).
///
/// Entities that have been resolved (debts paid off, subscriptions cancelled,
/// fraud disputes resolved) get zeroed-out embeddings. We skip these to prevent
/// stale recommendations like "should_refinance" for a paid-off obligation.
fn is_dead_node(ctx: &FiduciaryContext, node_type: &str, node_id: usize) -> bool {
    if let Some(embs) = ctx.embeddings.get(node_type) {
        if let Some(emb) = embs.get(node_id) {
            // Dead if all components are zero (or very near zero)
            return emb.iter().all(|&v| v.abs() < 1e-8);
        }
    }
    // No embedding found = treat as dead
    true
}

/// Infer which fiduciary actions apply to a connected entity.
///
/// Maps TQL relation names and entity types to fiduciary actions.
fn infer_actions(
    relation: &str,
    target_type: &str,
    ctx: &FiduciaryContext,
    target_id: usize,
) -> Vec<FiduciaryActionType> {
    let mut actions = Vec::new();
    let rel = relation.to_lowercase();
    let ttype = target_type.to_lowercase();

    // Get anomaly score if available
    let anomaly_score = get_anomaly_score(ctx, target_type, target_id);

    // ── Core actions ──

    // High anomaly → investigate + avoid
    if anomaly_score >= 0.5 {
        actions.push(FiduciaryActionType::ShouldInvestigate);
        actions.push(FiduciaryActionType::ShouldAvoid);
    }

    // Payment-related relations
    if rel.contains("pay")
        || rel.contains("owes")
        || rel.contains("debt")
        || rel.contains("settlement")
        || rel.contains("funded-by")
    {
        actions.push(FiduciaryActionType::ShouldPay);
    }

    // Subscription/recurring
    if rel.contains("subscribe")
        || rel.contains("recurring")
        || rel.contains("pattern-owned-by")
        || rel.contains("pattern-has-counterparty")
    {
        actions.push(FiduciaryActionType::ShouldCancel);
    }

    // Account ownership → transfer & consolidate
    if rel.contains("owns")
        || rel.contains("user-has-instrument")
        || rel.contains("provider-has-instrument")
        || rel.contains("has-user-transfer-pair")
    {
        actions.push(FiduciaryActionType::ShouldTransfer);
        actions.push(FiduciaryActionType::ShouldConsolidate);
    }

    // Merchant/vendor avoidance
    if (ttype.contains("merchant") || ttype.contains("vendor") || ttype.contains("counterparty"))
        && anomaly_score >= 0.3
    {
        actions.push(FiduciaryActionType::ShouldAvoid);
    }

    // Transaction anomalies
    if (ttype.contains("transaction") || ttype.contains("tx") || ttype.contains("evidence"))
        && anomaly_score >= 0.4
    {
        actions.push(FiduciaryActionType::ShouldInvestigate);
    }

    // ── Debt & Obligations ── (TQL: obligation, interest-rate-term, lien-on-asset)

    // Refinance: obligation with interest terms
    if rel.contains("obligation-has-interest-term")
        || rel.contains("obligation-refinanced-by")
        || rel.contains("interest-applied-rate-term")
        || (ttype.contains("obligation") && rel.contains("instrument-linked-to-obligation"))
    {
        actions.push(FiduciaryActionType::ShouldRefinance);
    }

    // Pay down lien: lien-on-asset relation
    if rel.contains("lien-on-asset") || rel.contains("obligation-finances-asset") {
        actions.push(FiduciaryActionType::ShouldPayDownLien);
    }

    // Dispute: anomalous obligation
    if (ttype.contains("obligation") || ttype.contains("obligation-between-parties"))
        && anomaly_score >= 0.4
    {
        actions.push(FiduciaryActionType::ShouldDispute);
    }

    // ── Goals & Budgets ── (TQL: goal, master-budget, budget-estimation)

    // Fund goal: goal-related relations
    if rel.contains("goal")
        || rel.contains("subledger-holds-goal-funds")
        || rel.contains("general-ledger-records-goal")
        || rel.contains("job-funds-goal")
        || ttype.contains("goal")
    {
        actions.push(FiduciaryActionType::ShouldFundGoal);
    }

    // Adjust budget: budget-related relations
    if rel.contains("budget")
        || rel.contains("records-budget")
        || rel.contains("records-budget-estimation")
        || ttype.contains("budget")
        || ttype.contains("budget-estimation")
    {
        actions.push(FiduciaryActionType::ShouldAdjustBudget);
    }

    // ── Tax Optimization ── (TQL: 60+ tax entities)

    // Prepare tax: due events and obligations
    if rel.contains("tax-due-event")
        || rel.contains("tax-liability-has-due-event")
        || rel.contains("unit-has-tax-obligation")
        || rel.contains("unit-has-tax-period")
        || ttype.contains("tax-due-event")
        || ttype.contains("tax-obligation")
    {
        actions.push(FiduciaryActionType::ShouldPrepareTax);
    }

    // Fund tax sinking fund
    if rel.contains("tax-sinking-fund")
        || rel.contains("tax-sinking-fund-backed-by")
        || rel.contains("tax-sinking-fund-linked-liability")
        || ttype.contains("tax-sinking-fund")
    {
        actions.push(FiduciaryActionType::ShouldFundTaxSinking);
    }

    // Claim exemption
    if rel.contains("tax-exemption")
        || rel.contains("tax-party-has-exemption-certificate")
        || rel.contains("tax-determination-uses-exemption-certificate")
        || ttype.contains("tax-exemption-certificate")
        || ttype.contains("tax-exemption-rule")
    {
        actions.push(FiduciaryActionType::ShouldClaimExemption);
    }

    // Run tax scenario
    if rel.contains("tax-scenario")
        || rel.contains("tax-scenario-for-period")
        || rel.contains("tax-scenario-has-result")
        || rel.contains("tax-scenario-uses-assumption")
        || ttype.contains("tax-scenario")
        || ttype.contains("tax-scenario-result")
    {
        actions.push(FiduciaryActionType::ShouldRunTaxScenario);
    }

    // ── Reconciliation ── (TQL: reconciliation-case, clearing-account-check)

    if rel.contains("reconciliation")
        || rel.contains("opposite-match-group")
        || rel.contains("clearing-account-check")
        || ttype.contains("reconciliation")
        || ttype.contains("opposite-match-group")
    {
        actions.push(FiduciaryActionType::ShouldReconcile);
    }

    // ── Recurring Patterns ── (TQL: recurring-pattern, recurring-missing-alert)

    if rel.contains("recurring-alert")
        || rel.contains("pattern-has-recurring-alert")
        || rel.contains("pattern-has-case")
        || ttype.contains("recurring-pattern")
        || ttype.contains("recurring-missing-alert")
    {
        actions.push(FiduciaryActionType::ShouldReviewRecurring);
    }

    // ── Asset Management ── (TQL: asset, asset-valuation, rate-observation)

    if rel.contains("asset-has-valuation")
        || rel.contains("asset-title-transfer")
        || rel.contains("provider-reports-portfolio-position")
        || ttype.contains("asset-valuation")
        || ttype.contains("rate-observation")
    {
        actions.push(FiduciaryActionType::ShouldRevalueAsset);
    }

    // Fallback: any connection with anomaly
    if actions.is_empty() && anomaly_score >= 0.3 {
        actions.push(FiduciaryActionType::ShouldInvestigate);
    }

    actions
}

// ═══════════════════════════════════════════════════════════════
// GNN-Based Fiduciary Scoring
// ═══════════════════════════════════════════════════════════════

/// Score a candidate action using GNN embeddings and graph features.
pub fn score_action(
    action: FiduciaryActionType,
    target_type: &str,
    target_id: usize,
    ctx: &FiduciaryContext,
) -> FiduciaryAxes {
    let affinity = get_embedding_affinity(ctx, target_type, target_id);
    let target_anomaly = get_anomaly_score(ctx, target_type, target_id);
    let norm_degree = get_norm_degree(ctx, target_type, target_id);

    match action {
        // ── Core 6 ──
        FiduciaryActionType::ShouldPay => FiduciaryAxes {
            cost_reduction: 0.8 * affinity.abs().min(1.0),
            risk_reduction: target_anomaly * 0.5 + 0.3,
            goal_alignment: 0.6,
            urgency: 0.5 + target_anomaly * 0.5,
            conflict_freedom: 1.0 - target_anomaly * 0.3,
            reversibility: 0.3,
        },
        FiduciaryActionType::ShouldCancel => FiduciaryAxes {
            cost_reduction: 0.7 + (1.0 - affinity.abs()) * 0.3,
            risk_reduction: 0.3,
            goal_alignment: 0.5,
            urgency: 0.3,
            conflict_freedom: 0.9,
            reversibility: 0.9,
        },
        FiduciaryActionType::ShouldTransfer => FiduciaryAxes {
            cost_reduction: 0.3,
            risk_reduction: 0.2,
            goal_alignment: 0.7 * affinity.abs().min(1.0),
            urgency: 0.2,
            conflict_freedom: 0.9,
            reversibility: 0.8,
        },
        FiduciaryActionType::ShouldAvoid => FiduciaryAxes {
            cost_reduction: 0.4 + target_anomaly * 0.6,
            risk_reduction: target_anomaly,
            goal_alignment: 0.5,
            urgency: target_anomaly * 0.8,
            conflict_freedom: 1.0 - target_anomaly,
            reversibility: 0.7,
        },
        FiduciaryActionType::ShouldInvestigate => FiduciaryAxes {
            cost_reduction: 0.2,
            risk_reduction: target_anomaly * 0.9,
            goal_alignment: 0.4,
            urgency: target_anomaly,
            conflict_freedom: 0.5,
            reversibility: 1.0,
        },
        FiduciaryActionType::ShouldConsolidate => FiduciaryAxes {
            cost_reduction: 0.4 + norm_degree * 0.3,
            risk_reduction: 0.2,
            goal_alignment: 0.5,
            urgency: 0.1,
            conflict_freedom: 0.8,
            reversibility: 0.4,
        },

        // ── Debt & Obligations ──
        FiduciaryActionType::ShouldRefinance => FiduciaryAxes {
            cost_reduction: 0.85, // Primary benefit: lower interest
            risk_reduction: 0.4 + (1.0 - target_anomaly) * 0.3, // Less risky if entity is normal
            goal_alignment: 0.7,  // Aligns with financial health
            urgency: 0.5 + affinity.abs() * 0.3, // More urgent if deeply connected
            conflict_freedom: 0.7,
            reversibility: 0.2, // Hard to undo refinancing
        },
        FiduciaryActionType::ShouldPayDownLien => FiduciaryAxes {
            cost_reduction: 0.6 + target_anomaly * 0.2, // Reduces ongoing cost
            risk_reduction: 0.8,                        // Liens are risk
            goal_alignment: 0.75,                       // Equity building
            urgency: 0.6 + target_anomaly * 0.3,
            conflict_freedom: 0.9,
            reversibility: 0.1, // Irreversible (good)
        },
        FiduciaryActionType::ShouldDispute => FiduciaryAxes {
            cost_reduction: 0.7 * target_anomaly, // Higher savings if more anomalous
            risk_reduction: target_anomaly * 0.8,
            goal_alignment: 0.5,
            urgency: 0.8,          // Disputes are time-sensitive
            conflict_freedom: 0.3, // Disputes are inherently conflictual
            reversibility: 0.6,
        },

        // ── Goals & Budgets ──
        FiduciaryActionType::ShouldFundGoal => FiduciaryAxes {
            cost_reduction: 0.2,
            risk_reduction: 0.3,
            goal_alignment: 0.95, // Primary benefit: goal progress
            urgency: 0.5 + affinity.abs() * 0.3, // More urgent if goal is relevant (high affinity)
            conflict_freedom: 0.95,
            reversibility: 0.7, // Can un-allocate
        },
        FiduciaryActionType::ShouldAdjustBudget => FiduciaryAxes {
            cost_reduction: 0.5, // Better budgets save money
            risk_reduction: 0.3,
            goal_alignment: 0.8,
            urgency: 0.3, // Not typically urgent
            conflict_freedom: 0.95,
            reversibility: 1.0, // Just a number change
        },

        // ── Tax Optimization ──
        FiduciaryActionType::ShouldPrepareTax => FiduciaryAxes {
            cost_reduction: 0.7,
            risk_reduction: 0.85, // Avoids penalties + interest
            goal_alignment: 0.6,
            urgency: 0.9, // Tax deadlines are firm
            conflict_freedom: 0.8,
            reversibility: 0.5,
        },
        FiduciaryActionType::ShouldFundTaxSinking => FiduciaryAxes {
            cost_reduction: 0.5,
            risk_reduction: 0.75, // Prevents cash flow shock
            goal_alignment: 0.7,
            urgency: 0.6,
            conflict_freedom: 0.9,
            reversibility: 0.6,
        },
        FiduciaryActionType::ShouldClaimExemption => FiduciaryAxes {
            cost_reduction: 0.9, // Direct tax savings
            risk_reduction: 0.3,
            goal_alignment: 0.6,
            urgency: 0.5,
            conflict_freedom: 0.7, // Must verify eligibility
            reversibility: 0.8,
        },
        FiduciaryActionType::ShouldRunTaxScenario => FiduciaryAxes {
            cost_reduction: 0.6, // Potential savings from planning
            risk_reduction: 0.4,
            goal_alignment: 0.5,
            urgency: 0.3, // Planning, not urgent
            conflict_freedom: 1.0,
            reversibility: 1.0, // Just analysis
        },

        // ── Reconciliation ──
        FiduciaryActionType::ShouldReconcile => FiduciaryAxes {
            cost_reduction: 0.3,
            risk_reduction: 0.7 + target_anomaly * 0.3, // Catches errors
            goal_alignment: 0.4,
            urgency: 0.5 + target_anomaly * 0.4,
            conflict_freedom: 0.8,
            reversibility: 1.0,
        },

        // ── Recurring Patterns ──
        FiduciaryActionType::ShouldReviewRecurring => FiduciaryAxes {
            cost_reduction: 0.5 + target_anomaly * 0.3,
            risk_reduction: 0.4 + target_anomaly * 0.5,
            goal_alignment: 0.4,
            urgency: 0.4 + target_anomaly * 0.4,
            conflict_freedom: 0.8,
            reversibility: 0.9,
        },

        // ── Asset Management ──
        FiduciaryActionType::ShouldRevalueAsset => FiduciaryAxes {
            cost_reduction: 0.2,  // Stale valuations misrepresent net worth → tax/estate cost
            risk_reduction: 0.55, // Undervalued collateral affects loan terms + insurance
            goal_alignment: 0.5,
            urgency: 0.35 + norm_degree * 0.3, // More connected → more impact
            conflict_freedom: 0.95,
            reversibility: 1.0, // Just updating a number
        },
    }
}

// ═══════════════════════════════════════════════════════════════
// Recommendation Config (for GEPA weight overrides)
// ═══════════════════════════════════════════════════════════════

/// Optional weight overrides for GEPA-driven optimization.
///
/// When GEPA searches for optimal weights, it injects candidate values
/// via this struct. Production code loads persisted GEPA weights from
/// `gepa_weights.json`; tests can override per-action priorities.
#[derive(Debug, Clone, Default)]
pub struct RecommendConfig {
    /// Override fiduciary axes weights \[cost, risk, goal, urgency, conflict, reversibility\].
    pub axes_weights: Option<[f32; 6]>,
    /// Override per-action priority weights (action_name → weight).
    /// Missing entries fall back to `FiduciaryActionType::priority_weight()`.
    pub priority_overrides: HashMap<String, f32>,
}

impl RecommendConfig {
    /// Look up priority weight for an action: override first, then default.
    pub fn priority_weight(&self, action: FiduciaryActionType) -> f32 {
        self.priority_overrides
            .get(action.name())
            .copied()
            .unwrap_or_else(|| action.priority_weight())
    }
}

// ═══════════════════════════════════════════════════════════════
// Recommendation Builder
// ═══════════════════════════════════════════════════════════════

/// Build the full fiduciary recommendation set for a user.
///
/// Pass `Some(&mut pc_state)` for self-learning: the circuit will resume
/// EM training from its previous state rather than rebuilding from scratch.
/// Pass `None` for one-shot mode (fresh circuit each time).
pub fn recommend(ctx: &FiduciaryContext, pc_state: Option<&mut PcState>) -> FiduciaryResponse {
    recommend_with_config(ctx, pc_state, None)
}

/// Build the full fiduciary recommendation set with optional GEPA weight overrides.
///
/// This is the core implementation. `recommend()` delegates here with `config=None`,
/// which uses GEPA-persisted or default weights.
pub fn recommend_with_config(
    ctx: &FiduciaryContext,
    mut pc_state: Option<&mut PcState>,
    config: Option<&RecommendConfig>,
) -> FiduciaryResponse {
    let candidates = generate_candidates(ctx);
    let axes_weights = config
        .and_then(|c| c.axes_weights)
        .unwrap_or_else(load_axes_weights);

    let user_name = ctx
        .node_names
        .get(&ctx.user_type)
        .and_then(|names| names.get(ctx.user_id))
        .cloned()
        .unwrap_or_else(|| format!("{}_{}", ctx.user_type, ctx.user_id));

    let mut recommendations: Vec<FiduciaryRecommendation> = candidates
        .iter()
        .map(|(action, target_type, target_id)| {
            let axes = score_action(*action, target_type, *target_id, ctx);
            // Use GEPA-optimized fiduciary axis weights when available.
            let prio =
                config.map_or_else(|| action.priority_weight(), |c| c.priority_weight(*action));
            let fiduciary_score = sanitize_score(axes.score_with_weights(&axes_weights) * prio);

            let target_name = ctx
                .node_names
                .get(target_type)
                .and_then(|names| names.get(*target_id))
                .cloned()
                .unwrap_or_else(|| format!("{}_{}", target_type, target_id));

            let embedding_affinity = get_embedding_affinity(ctx, target_type, *target_id);
            let target_anomaly_score = get_anomaly_score(ctx, target_type, *target_id);

            let reasoning = format!(
                "{} {} (type={}). Fiduciary score: {:.3}. \
                 Cost reduction: {:.2}, Risk reduction: {:.2}, Goal alignment: {:.2}. \
                 Embedding affinity: {:.3}, Target anomaly: {:.3}. \
                 {}",
                action.verb(),
                target_name,
                target_type,
                fiduciary_score,
                axes.cost_reduction,
                axes.risk_reduction,
                axes.goal_alignment,
                embedding_affinity,
                target_anomaly_score,
                action.reasoning_suffix(),
            );

            FiduciaryRecommendation {
                rank: 0,
                action_type: action.name().to_string(),
                domain: action.domain().to_string(),
                target_name,
                target_node_type: target_type.clone(),
                target_node_id: *target_id,
                fiduciary_score,
                axes,
                embedding_affinity,
                target_anomaly_score,
                reasoning,
                is_recommended: fiduciary_score >= 0.3,
                pc_analysis: None,
                scorer_logit: None,
            }
        })
        .collect();

    // Sort by fiduciary score (highest first)
    recommendations.sort_by(|a, b| b.fiduciary_score.total_cmp(&a.fiduciary_score));

    // ── PC Circuit: train/resume from graph features and enrich recommendations ──
    use crate::model::pc::{
        bridge,
        em::{train_em, EmConfig},
        fiduciary_pc,
    };
    let training_data = bridge::generate_training_data(
        ctx.anomaly_scores,
        ctx.embeddings,
        ctx.edges,
        &ctx.node_counts,
        ctx.user_emb,
    );
    let (pc_trained, pc_em_ll) = {
        // Convert to evidence format for EM
        let evidence: Vec<Vec<Option<usize>>> = training_data
            .iter()
            .map(|obs| obs.iter().map(|&v| Some(v)).collect())
            .collect();

        // Try to extract existing circuit from PcState for resume
        let existing_circuit = pc_state.as_mut().and_then(|s| s.circuit.take());

        let (mut circuit, final_ll, epochs_this_call) = if let Some(mut prev) = existing_circuit {
            // Resume: continue EM on existing circuit (5 incremental epochs)
            let em_config = EmConfig {
                step_size: 0.05, // smaller step for fine-tuning
                pseudocount: 0.01,
                epochs: 5,
            };
            let report = train_em(&mut prev, &evidence, &em_config);
            (prev, report.final_ll, 5usize)
        } else {
            // Fresh build: full HCLT + 30 EM epochs
            let (c, report) = bridge::build_fiduciary_pc(&training_data, 30);
            (c, report.final_ll, 30usize)
        };

        // Run PC analysis per recommendation
        for rec in &mut recommendations {
            let degree =
                bridge::count_node_edges(ctx.edges, &rec.target_node_type, rec.target_node_id);
            let analysis = fiduciary_pc::analyze(
                &mut circuit,
                rec.target_anomaly_score,
                rec.embedding_affinity,
                degree,
                rec.fiduciary_score,
            );
            rec.pc_analysis = Some(analysis);
        }

        // Write back to PcState for persistence
        if let Some(state) = pc_state {
            state.total_epochs += epochs_this_call;
            state.ll_history.push(final_ll);
            state.circuit = Some(circuit);
        }

        (true, Some(final_ll))
    };

    // ── PC Risk Blending: merge calibrated PC risk into fiduciary score ──
    // Formula: final = α × gnn_score + (1-α) × pc_risk_adjusted
    // pc_risk is P(risky) ∈ [0,1]; we scale it to match GNN score range.
    // Higher PC risk → higher final score (action is more urgent).
    if pc_trained {
        let (alpha, beta) = normalized_blend(load_blend_weights());
        for rec in &mut recommendations {
            if let Some(ref analysis) = rec.pc_analysis {
                let pc_risk = sanitize_score(analysis.risk_probability as f32);
                // Scale PC risk to GNN score range: pc_risk ∈ [0,1], gnn ∈ [0,~1]
                // Blend: keep GNN base but boost/penalize based on PC calibrated risk
                let gnn_score = sanitize_score(rec.fiduciary_score);
                let blended = sanitize_score(alpha * gnn_score + beta * pc_risk);
                rec.fiduciary_score = blended.clamp(0.0, 1.0);
                rec.is_recommended = rec.fiduciary_score >= 0.3;
            }
        }
        // Re-sort after blending (PC risk may change relative order)
        recommendations.sort_by(|a, b| b.fiduciary_score.total_cmp(&a.fiduciary_score));
    }

    // ── Fiduciary conflict suppression ──
    // Principle: "Don't invest while you're on fire."
    // If high-urgency actions (debt, risk, fraud) dominate, demote
    // discretionary goals to is_recommended=false.
    let high_urgency_actions = [
        "should_investigate",
        "should_avoid",
        "should_dispute",
        "should_refinance",
        "should_pay_down_lien",
        "should_prepare_tax",
        "should_pay",
    ];
    let discretionary_actions = [
        "should_fund_goal",
        "should_adjust_budget",
        "should_transfer",
        "should_consolidate",
        "should_revalue_asset",
        "should_run_tax_scenario",
    ];
    let has_high_urgency = recommendations.iter().any(|r| {
        high_urgency_actions.contains(&r.action_type.as_str()) && r.fiduciary_score >= 0.4
    });
    if has_high_urgency {
        for rec in &mut recommendations {
            if discretionary_actions.contains(&rec.action_type.as_str())
                && rec.fiduciary_score < 0.5
            {
                rec.is_recommended = false;
            }
        }
    }

    // Track unique action types and domains before truncation
    let mut unique_actions: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut unique_domains: std::collections::HashSet<String> = std::collections::HashSet::new();
    for rec in &recommendations {
        unique_actions.insert(rec.action_type.clone());
        unique_domains.insert(rec.domain.clone());
    }
    let action_types_triggered = unique_actions.len();
    let mut domains_covered: Vec<String> = unique_domains.into_iter().collect();
    domains_covered.sort();

    // Cap at top 20 (more room now with 18 action types)
    recommendations.truncate(20);
    for (i, rec) in recommendations.iter_mut().enumerate() {
        rec.rank = i + 1;
    }

    let recommended_count = recommendations.iter().filter(|r| r.is_recommended).count();

    let assessment = format!(
        "Fiduciary analysis for {} (type={}, id={}): {} candidate actions evaluated, \
         {} recommended across {} action types and {} domains. Top action: {}. \
         Analysis used {}-dimensional GNN embeddings from 4 models \
         to score embedding affinity, anomaly risk, and structural importance.",
        user_name,
        ctx.user_type,
        ctx.user_id,
        candidates.len(),
        recommended_count,
        action_types_triggered,
        domains_covered.len(),
        recommendations
            .first()
            .map(|r| format!(
                "{} {} (score={:.3})",
                r.action_type, r.target_name, r.fiduciary_score
            ))
            .unwrap_or_else(|| "none".into()),
        ctx.hidden_dim,
    );

    FiduciaryResponse {
        user_node_type: ctx.user_type.clone(),
        user_node_id: ctx.user_id,
        user_name,
        recommendations,
        action_types_triggered,
        domains_covered,
        assessment,
        models_used: vec![
            "GraphSAGE".into(),
            "RGCN".into(),
            "GAT".into(),
            "Graph Transformer".into(),
        ],
        sae_explanation: None,
        pc_trained,
        pc_em_ll,
    }
}

// ═══════════════════════════════════════════════════════════════
// Helper functions
// ═══════════════════════════════════════════════════════════════

pub fn get_anomaly_score(ctx: &FiduciaryContext, target_type: &str, target_id: usize) -> f32 {
    aggregate_anomaly_score(ctx.anomaly_scores, target_type, target_id).unwrap_or(0.0)
}

pub fn get_embedding_affinity(ctx: &FiduciaryContext, target_type: &str, target_id: usize) -> f32 {
    ctx.embeddings
        .get(target_type)
        .and_then(|vecs| vecs.get(target_id))
        .map(|t_emb| cosine_sim(ctx.user_emb, t_emb))
        .unwrap_or(0.0)
}

fn get_norm_degree(ctx: &FiduciaryContext, target_type: &str, target_id: usize) -> f32 {
    let degree = count_edges(ctx.edges, target_type, target_id);
    (degree as f32 / 100.0).min(1.0)
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn count_edges(
    edges: &HashMap<(String, String, String), Vec<(usize, usize)>>,
    node_type: &str,
    node_id: usize,
) -> usize {
    let mut count = 0;
    for ((src_type, _, dst_type), edge_list) in edges {
        if src_type == node_type {
            count += edge_list.iter().filter(|(s, _)| *s == node_id).count();
        }
        if dst_type == node_type {
            count += edge_list.iter().filter(|(_, d)| *d == node_id).count();
        }
    }
    count
}

fn sanitize_score(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

fn normalize_action_name(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    let mut prev_is_sep = true;
    for ch in s.chars() {
        if ch == '-' || ch == ' ' {
            if !prev_is_sep {
                out.push('_');
            }
            prev_is_sep = true;
            continue;
        }

        if ch.is_ascii_uppercase() {
            if !prev_is_sep {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
            prev_is_sep = false;
        } else {
            out.push(ch.to_ascii_lowercase());
            prev_is_sep = ch == '_';
        }
    }

    // Collapse duplicate underscores.
    let mut compact = String::with_capacity(out.len());
    let mut prev_us = false;
    for ch in out.chars() {
        if ch == '_' {
            if !prev_us {
                compact.push(ch);
            }
            prev_us = true;
        } else {
            compact.push(ch);
            prev_us = false;
        }
    }
    compact.trim_matches('_').to_string()
}

fn aggregate_anomaly_score(
    anomaly_scores: &HashMap<String, HashMap<String, Vec<f32>>>,
    target_type: &str,
    target_id: usize,
) -> Option<f32> {
    let mut vals: Vec<f32> = anomaly_scores
        .values()
        .filter_map(|m| m.get(target_type))
        .filter_map(|scores| scores.get(target_id).copied())
        .filter(|v| v.is_finite())
        .collect();

    if vals.is_empty() {
        return None;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let mid = vals.len() / 2;
    let median = if vals.len() % 2 == 0 {
        (vals[mid - 1] + vals[mid]) * 0.5
    } else {
        vals[mid]
    };
    Some(median.clamp(0.0, 1.0))
}

fn normalized_blend((gnn, pc): (f32, f32)) -> (f32, f32) {
    let g = if gnn.is_finite() {
        gnn.max(0.0)
    } else {
        DEFAULT_GNN_WEIGHT
    };
    let p = if pc.is_finite() {
        pc.max(0.0)
    } else {
        DEFAULT_PC_WEIGHT
    };
    let sum = g + p;
    if sum <= 1e-8 {
        (DEFAULT_GNN_WEIGHT, DEFAULT_PC_WEIGHT)
    } else {
        (g / sum, p / sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_name_parsing_variants() {
        assert_eq!(
            FiduciaryActionType::try_from_name("should_investigate"),
            Some(FiduciaryActionType::ShouldInvestigate)
        );
        assert_eq!(
            FiduciaryActionType::try_from_name("ShouldInvestigate"),
            Some(FiduciaryActionType::ShouldInvestigate)
        );
        assert_eq!(
            FiduciaryActionType::try_from_name("should-investigate"),
            Some(FiduciaryActionType::ShouldInvestigate)
        );
        assert!(FiduciaryActionType::try_from_name("unknown_action").is_none());
    }

    #[test]
    fn test_anomaly_aggregation_is_order_independent() {
        let mut by_model_a: HashMap<String, HashMap<String, Vec<f32>>> = HashMap::new();
        let mut by_model_b: HashMap<String, HashMap<String, Vec<f32>>> = HashMap::new();

        let mut m1 = HashMap::new();
        m1.insert("account".to_string(), vec![0.9]);
        let mut m2 = HashMap::new();
        m2.insert("account".to_string(), vec![0.1]);
        let mut m3 = HashMap::new();
        m3.insert("account".to_string(), vec![0.7]);

        by_model_a.insert("SAGE".to_string(), m1.clone());
        by_model_a.insert("RGCN".to_string(), m2.clone());
        by_model_a.insert("GAT".to_string(), m3.clone());

        by_model_b.insert("GAT".to_string(), m3);
        by_model_b.insert("RGCN".to_string(), m2);
        by_model_b.insert("SAGE".to_string(), m1);

        let a = aggregate_anomaly_score(&by_model_a, "account", 0).unwrap();
        let b = aggregate_anomaly_score(&by_model_b, "account", 0).unwrap();
        assert!(
            (a - b).abs() < 1e-12,
            "aggregation should not depend on map insertion order"
        );
        assert!((a - 0.7).abs() < 1e-6, "median(0.1,0.7,0.9) should be 0.7");
    }
}

// ═══════════════════════════════════════════════════════════════
// HyperFiduciaryAxes — Metacognitive parameter co-evolution
// ═══════════════════════════════════════════════════════════════

/// Per-action-type performance tracking for metacognitive weight evolution.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ActionPerformance {
    /// Times this action was recommended.
    pub recommended: u32,
    /// Times the recommendation was considered correct by feedback.
    pub correct: u32,
    /// Times it was a false positive (recommended but harmful/wrong).
    pub false_positive: u32,
    /// Times it was a false negative (should have been recommended but wasn't).
    pub false_negative: u32,
}

impl ActionPerformance {
    /// Precision: correct / (correct + false_positive).
    pub fn precision(&self) -> f64 {
        let denom = self.correct + self.false_positive;
        if denom == 0 { 1.0 } else { self.correct as f64 / denom as f64 }
    }

    /// Recall: correct / (correct + false_negative).
    pub fn recall(&self) -> f64 {
        let denom = self.correct + self.false_negative;
        if denom == 0 { 1.0 } else { self.correct as f64 / denom as f64 }
    }

    /// F1 score.
    pub fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r <= 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
    }

    /// Total observations.
    pub fn total_observations(&self) -> u32 {
        self.correct + self.false_positive + self.false_negative
    }
}

/// Recommended weight adjustment from HyperFiduciaryAxes analysis.
#[derive(Debug, Clone)]
pub struct AxesAdjustment {
    /// Action type that triggered the adjustment.
    pub action_name: String,
    /// Current priority weight.
    pub current_weight: f32,
    /// Suggested new priority weight.
    pub suggested_weight: f32,
    /// Reason for the adjustment.
    pub reason: String,
}

/// HyperFiduciaryAxes: metacognitive parameter co-evolution for fiduciary scoring.
///
/// Tracks per-action-type precision@K and, when certain actions consistently
/// misprioritize, recommends weight adjustments that feed back into GEPA.
///
/// ## Architecture
/// ```text
/// recommend() → FiduciaryResponse
///    ↓ (feedback from ground truth or user)
/// ActionPerformance (per action type)
///    ↓ when precision < threshold
/// AxesAdjustment recommendation
///    ↓ fed into GEPA seed candidate as weight hints
/// Next GEPA cycle incorporates the hint
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HyperFiduciaryAxes {
    /// Per-action-type performance metrics.
    pub action_performance: HashMap<String, ActionPerformance>,
    /// Minimum observations before adjustment triggers.
    pub min_observations: u32,
    /// Precision threshold below which we recommend adjustment.
    pub precision_threshold: f64,
    /// Number of adjustments made so far.
    pub adjustment_count: u32,
    /// History of adjustments.
    pub adjustment_history: Vec<String>,
}

impl HyperFiduciaryAxes {
    pub fn new() -> Self {
        Self {
            action_performance: HashMap::new(),
            min_observations: 5,
            precision_threshold: 0.60,
            adjustment_count: 0,
            adjustment_history: Vec::new(),
        }
    }

    /// Record feedback for a recommendation.
    pub fn record_feedback(
        &mut self,
        action_name: &str,
        was_correct: bool,
        was_false_negative: bool,
    ) {
        let entry = self.action_performance
            .entry(action_name.to_string())
            .or_insert_with(ActionPerformance::default);

        if was_correct {
            entry.correct += 1;
            entry.recommended += 1;
        } else if was_false_negative {
            entry.false_negative += 1;
        } else {
            // False positive: recommended but wrong
            entry.false_positive += 1;
            entry.recommended += 1;
        }
    }

    /// Analyze all action types and return adjustment recommendations.
    pub fn analyze(&self) -> Vec<AxesAdjustment> {
        let mut adjustments = Vec::new();

        for (name, perf) in &self.action_performance {
            if perf.total_observations() < self.min_observations {
                continue;
            }

            let precision = perf.precision();
            if precision < self.precision_threshold {
                // Action is producing too many false positives → reduce its weight
                let current_weight = FiduciaryActionType::try_from_name(name)
                    .map(|a| a.priority_weight())
                    .unwrap_or(0.50);
                let reduction = (1.0 - precision as f32) * 0.3; // up to 30% reduction
                let suggested = (current_weight - reduction).max(0.10);

                adjustments.push(AxesAdjustment {
                    action_name: name.clone(),
                    current_weight,
                    suggested_weight: suggested,
                    reason: format!(
                        "precision={:.1}% < {:.1}% threshold ({} correct, {} false_pos)",
                        precision * 100.0,
                        self.precision_threshold * 100.0,
                        perf.correct,
                        perf.false_positive,
                    ),
                });
            }

            // Also check for high false negative rate (should boost weight)
            let recall = perf.recall();
            if recall < 0.50 && perf.false_negative >= 3 {
                let current_weight = FiduciaryActionType::try_from_name(name)
                    .map(|a| a.priority_weight())
                    .unwrap_or(0.50);
                let boost = (1.0 - recall as f32) * 0.2; // up to 20% boost
                let suggested = (current_weight + boost).min(1.0);

                adjustments.push(AxesAdjustment {
                    action_name: name.clone(),
                    current_weight,
                    suggested_weight: suggested,
                    reason: format!(
                        "recall={:.1}% — too many false negatives ({} missed)",
                        recall * 100.0,
                        perf.false_negative,
                    ),
                });
            }
        }

        adjustments
    }

    /// Apply adjustments to an OptimizedWeights' priority_overrides.
    pub fn apply_to_priority_weights(
        &mut self,
        priority_weights: &mut HashMap<String, f32>,
    ) -> usize {
        let adjustments = self.analyze();
        let count = adjustments.len();
        for adj in &adjustments {
            priority_weights.insert(adj.action_name.clone(), adj.suggested_weight);
            self.adjustment_history.push(format!(
                "{}: {:.2} → {:.2} ({})",
                adj.action_name, adj.current_weight, adj.suggested_weight, adj.reason,
            ));
        }
        self.adjustment_count += count as u32;
        count
    }

    /// Summary for diagnostics.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "HyperFiduciaryAxes: {} action types tracked, {} adjustments made\n",
            self.action_performance.len(),
            self.adjustment_count,
        );
        for (name, perf) in &self.action_performance {
            s.push_str(&format!(
                "  {}: prec={:.1}% recall={:.1}% f1={:.1}% (obs={})\n",
                name,
                perf.precision() * 100.0,
                perf.recall() * 100.0,
                perf.f1() * 100.0,
                perf.total_observations(),
            ));
        }
        s
    }
}

impl Default for HyperFiduciaryAxes {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod hyper_fiduciary_tests {
    use super::*;

    #[test]
    fn test_action_performance_precision() {
        let mut perf = ActionPerformance::default();
        perf.correct = 7;
        perf.false_positive = 3;
        assert!((perf.precision() - 0.70).abs() < 0.01);
    }

    #[test]
    fn test_action_performance_recall() {
        let mut perf = ActionPerformance::default();
        perf.correct = 4;
        perf.false_negative = 6;
        assert!((perf.recall() - 0.40).abs() < 0.01);
    }

    #[test]
    fn test_hyper_axes_detects_low_precision() {
        let mut hyper = HyperFiduciaryAxes::new();
        hyper.min_observations = 3;

        // should_investigate has 2 correct, 5 false positives
        for _ in 0..2 {
            hyper.record_feedback("should_investigate", true, false);
        }
        for _ in 0..5 {
            hyper.record_feedback("should_investigate", false, false);
        }

        let adjustments = hyper.analyze();
        assert!(!adjustments.is_empty(), "Should recommend adjustments");

        let adj = adjustments.iter()
            .find(|a| a.action_name == "should_investigate")
            .unwrap();
        assert!(adj.suggested_weight < adj.current_weight,
            "Should reduce weight for low-precision action");
    }

    #[test]
    fn test_hyper_axes_detects_high_false_negatives() {
        let mut hyper = HyperFiduciaryAxes::new();
        hyper.min_observations = 3;

        // should_pay: 1 correct, 4 false negatives
        hyper.record_feedback("should_pay", true, false);
        for _ in 0..4 {
            hyper.record_feedback("should_pay", false, true);
        }

        let adjustments = hyper.analyze();
        let adj = adjustments.iter()
            .find(|a| a.action_name == "should_pay" && a.reason.contains("recall"))
            .unwrap();
        assert!(adj.suggested_weight > adj.current_weight,
            "Should boost weight for high-miss action");
    }

    #[test]
    fn test_hyper_axes_no_trigger_with_insufficient_data() {
        let mut hyper = HyperFiduciaryAxes::new();
        hyper.record_feedback("should_cancel", false, false);
        // Only 1 observation, need 5
        let adjustments = hyper.analyze();
        assert!(adjustments.is_empty());
    }

    #[test]
    fn test_apply_to_weights() {
        let mut hyper = HyperFiduciaryAxes::new();
        hyper.min_observations = 3;

        // Create low-precision scenario
        for _ in 0..5 {
            hyper.record_feedback("should_cancel", false, false);
        }

        let mut weights = HashMap::new();
        let applied = hyper.apply_to_priority_weights(&mut weights);
        assert!(applied > 0);
        assert!(weights.contains_key("should_cancel"));
    }
}
