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
            Self::ShouldRevalueAsset => 0.45,    // Lowest: informational
        }
    }

    pub fn reasoning_suffix(&self) -> &'static str {
        match self {
            Self::ShouldInvestigate =>
                "This entity shows anomalous patterns — manual review recommended.",
            Self::ShouldAvoid =>
                "Reducing interaction with this entity reduces financial risk.",
            Self::ShouldPay =>
                "Timely payment avoids penalties and builds financial health.",
            Self::ShouldCancel =>
                "Low engagement signal suggests potential cost savings.",
            Self::ShouldTransfer =>
                "Rebalancing funds could optimize returns.",
            Self::ShouldConsolidate =>
                "Fewer accounts reduce fee duplication and complexity.",
            Self::ShouldRefinance =>
                "Refinancing at a lower rate reduces total interest paid over the loan term.",
            Self::ShouldPayDownLien =>
                "Paying down liens increases your equity position and may reduce insurance costs.",
            Self::ShouldDispute =>
                "Anomalous obligation amounts should be disputed promptly to limit liability.",
            Self::ShouldFundGoal =>
                "Goal is underfunded — regular contributions keep you on track.",
            Self::ShouldAdjustBudget =>
                "Actual spending diverges from budget estimate — adjustment improves forecast accuracy.",
            Self::ShouldPrepareTax =>
                "Tax due event approaching — early preparation avoids penalties and interest.",
            Self::ShouldFundTaxSinking =>
                "Tax sinking fund is below target — funding now prevents cash flow surprises.",
            Self::ShouldClaimExemption =>
                "Eligible exemption certificate not yet applied — claiming it reduces tax liability.",
            Self::ShouldRunTaxScenario =>
                "Running a what-if tax scenario could reveal savings opportunities before period closes.",
            Self::ShouldReconcile =>
                "Unmatched items detected — reconciliation ensures ledger accuracy.",
            Self::ShouldReviewRecurring =>
                "Recurring pattern triggered an alert — review to confirm it's still valid.",
            Self::ShouldRevalueAsset =>
                "Asset valuation may be stale — updating ensures accurate net worth reporting.",
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
    /// Weighted fiduciary score.
    pub fn score(&self) -> f32 {
        let weights = [0.25, 0.25, 0.15, 0.15, 0.10, 0.10];
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

// ═══════════════════════════════════════════════════════════════
// Action Candidate Generation — driven by TQL ontology relations
// ═══════════════════════════════════════════════════════════════

/// Generate candidate actions by scanning the user's graph neighborhood.
pub fn generate_candidates(ctx: &FiduciaryContext) -> Vec<(FiduciaryActionType, String, usize)> {
    let mut candidates = Vec::new();

    // Find all entities connected to this user
    for ((src_type, relation, dst_type), edge_list) in ctx.edges {
        // User as source
        if src_type == &ctx.user_type {
            for &(src_id, dst_id) in edge_list {
                if src_id == ctx.user_id {
                    // Skip dead nodes (zeroed embeddings = soft-deleted entities)
                    if is_dead_node(ctx, dst_type, dst_id) {
                        continue;
                    }
                    let actions = infer_actions(relation, dst_type, ctx, dst_id);
                    for action in actions {
                        candidates.push((action, dst_type.clone(), dst_id));
                    }
                }
            }
        }
        // User as destination
        if dst_type == &ctx.user_type {
            for &(src_id, dst_id) in edge_list {
                if dst_id == ctx.user_id {
                    // Skip dead nodes
                    if is_dead_node(ctx, src_type, src_id) {
                        continue;
                    }
                    let actions = infer_actions(relation, src_type, ctx, src_id);
                    for action in actions {
                        candidates.push((action, src_type.clone(), src_id));
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
            urgency: 0.4 + (1.0 - affinity.abs()) * 0.3, // More urgent if far from goal
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
            cost_reduction: 0.1,
            risk_reduction: 0.4,
            goal_alignment: 0.5,
            urgency: 0.2 + norm_degree * 0.3, // More connected → more impact
            conflict_freedom: 0.95,
            reversibility: 1.0, // Just updating a number
        },
    }
}

// ═══════════════════════════════════════════════════════════════
// Recommendation Builder
// ═══════════════════════════════════════════════════════════════

/// Build the full fiduciary recommendation set for a user.
pub fn recommend(ctx: &FiduciaryContext) -> FiduciaryResponse {
    let candidates = generate_candidates(ctx);

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
            let fiduciary_score = axes.score() * action.priority_weight();

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
            }
        })
        .collect();

    // Sort by fiduciary score (highest first)
    recommendations.sort_by(|a, b| b.fiduciary_score.partial_cmp(&a.fiduciary_score).unwrap());

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
    }
}

// ═══════════════════════════════════════════════════════════════
// Helper functions
// ═══════════════════════════════════════════════════════════════

fn get_anomaly_score(ctx: &FiduciaryContext, target_type: &str, target_id: usize) -> f32 {
    ctx.anomaly_scores
        .values()
        .next()
        .and_then(|m| m.get(target_type))
        .and_then(|scores| scores.get(target_id))
        .copied()
        .unwrap_or(0.0)
}

fn get_embedding_affinity(ctx: &FiduciaryContext, target_type: &str, target_id: usize) -> f32 {
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
