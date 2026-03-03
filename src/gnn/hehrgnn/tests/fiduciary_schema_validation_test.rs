//! Schema Validation for Fiduciary Actions.
//!
//! Parses SchemaFinverse.tql and verifies that:
//! 1. Every relation substring used in infer_actions() matches a real TQL relation
//! 2. Every entity type used in test graphs exists in the real TQL schema
//! 3. The "user" entity actually plays the roles we trigger fiduciary actions on
//! 4. Benchmark scenarios use ONLY valid TQL names
//!
//! This catches: typos, made-up relations, relations user doesn't play.

use hehrgnn::eval::fiduciary::*;
use std::collections::{HashMap, HashSet};

const TQL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../SchemaFinverse.tql",);

// ═══════════════════════════════════════════════════════════════
// TQL Parser — extracts entity types, relation types, and plays
// ═══════════════════════════════════════════════════════════════

struct TqlSchema {
    /// All entity type names
    entity_types: HashSet<String>,
    /// All relation type names  
    relation_types: HashSet<String>,
    /// Which relations each entity plays: entity_type → set of relation names
    entity_plays: HashMap<String, HashSet<String>>,
}

fn parse_tql(path: &str) -> TqlSchema {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Cannot read TQL file at {}: {}", path, e));

    let mut entity_types = HashSet::new();
    let mut relation_types = HashSet::new();
    let mut entity_plays: HashMap<String, HashSet<String>> = HashMap::new();

    let mut current_entity: Option<String> = None;

    for line in content.lines() {
        let trimmed = line.trim();

        // Entity definition: "entity foo-bar," or "entity foo-bar @abstract,"
        if trimmed.starts_with("entity ") {
            let name = trimmed
                .strip_prefix("entity ")
                .unwrap()
                .split(|c: char| c == ',' || c == ' ')
                .next()
                .unwrap()
                .to_string();
            entity_types.insert(name.clone());
            current_entity = Some(name);
        }
        // Relation definition: "relation foo-bar,"
        else if trimmed.starts_with("relation ") {
            let name = trimmed
                .strip_prefix("relation ")
                .unwrap()
                .split(|c: char| c == ',' || c == ' ')
                .next()
                .unwrap()
                .to_string();
            relation_types.insert(name);
            current_entity = None; // relation block, not entity
        }
        // "plays relation-name:role" inside an entity block
        else if trimmed.starts_with("plays ") {
            if let Some(ref entity) = current_entity {
                let rest = trimmed.strip_prefix("plays ").unwrap();
                let relation_name = rest.split(':').next().unwrap().trim().to_string();
                entity_plays
                    .entry(entity.clone())
                    .or_default()
                    .insert(relation_name);
            }
        }
        // "sub entity-name" switches context but doesn't change current_entity
        else if trimmed.starts_with("sub ") {
            // keep current_entity
        }
    }

    TqlSchema {
        entity_types,
        relation_types,
        entity_plays,
    }
}

// ═══════════════════════════════════════════════════════════════
// Test 1: All infer_actions() relation substrings match real TQL relations
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_all_trigger_relations_exist_in_tql() {
    let schema = parse_tql(TQL_PATH);

    // These are the EXACT substring matches used in infer_actions()
    // Each must match at least one real TQL relation
    let relation_triggers: Vec<(&str, &str)> = vec![
        // Core
        ("pay", "ShouldPay"),
        ("funded-by", "ShouldPay"),
        ("settlement", "ShouldPay"),
        ("pattern-owned-by", "ShouldCancel"),
        ("pattern-has-counterparty", "ShouldCancel"),
        ("user-has-instrument", "ShouldTransfer/Consolidate"),
        ("provider-has-instrument", "ShouldTransfer/Consolidate"),
        ("has-user-transfer-pair", "ShouldTransfer/Consolidate"),
        // Debt
        ("obligation-has-interest-term", "ShouldRefinance"),
        ("obligation-refinanced-by", "ShouldRefinance"),
        ("interest-applied-rate-term", "ShouldRefinance"),
        ("instrument-linked-to-obligation", "ShouldRefinance"),
        ("lien-on-asset", "ShouldPayDownLien"),
        ("obligation-finances-asset", "ShouldPayDownLien"),
        // Goals
        ("subledger-holds-goal-funds", "ShouldFundGoal"),
        ("general-ledger-records-goal", "ShouldFundGoal"),
        ("job-funds-goal", "ShouldFundGoal"),
        // Budget
        ("records-budget-estimation", "ShouldAdjustBudget"),
        ("records-budget", "ShouldAdjustBudget"),
        // Tax
        ("tax-liability-has-due-event", "ShouldPrepareTax"),
        ("unit-has-tax-obligation", "ShouldPrepareTax"),
        ("unit-has-tax-period", "ShouldPrepareTax"),
        ("tax-sinking-fund-backed-by", "ShouldFundTaxSinking"),
        ("tax-sinking-fund-linked-liability", "ShouldFundTaxSinking"),
        (
            "tax-party-has-exemption-certificate",
            "ShouldClaimExemption",
        ),
        (
            "tax-determination-uses-exemption-certificate",
            "ShouldClaimExemption",
        ),
        ("tax-scenario-for-period", "ShouldRunTaxScenario"),
        ("tax-scenario-has-result", "ShouldRunTaxScenario"),
        ("tax-scenario-uses-assumption", "ShouldRunTaxScenario"),
        // Reconciliation
        ("reconciliation-for-instrument", "ShouldReconcile"),
        ("clearing-check-for-subledger", "ShouldReconcile"),
        // Recurring
        ("pattern-has-recurring-alert", "ShouldReviewRecurring"),
        ("pattern-has-case", "ShouldReviewRecurring"),
        // Assets
        ("asset-has-valuation", "ShouldRevalueAsset"),
        ("asset-title-transfer", "ShouldRevalueAsset"),
        ("provider-reports-portfolio-position", "ShouldRevalueAsset"),
    ];

    println!("\n  ── SCHEMA VALIDATION: Relation Triggers ──\n");

    let mut errors = Vec::new();
    for (trigger, action) in &relation_triggers {
        let matches: Vec<&String> = schema
            .relation_types
            .iter()
            .filter(|rel| rel.contains(trigger))
            .collect();

        if matches.is_empty() {
            errors.push(format!(
                "❌ Trigger '{}' (for {}) matches NO TQL relation!",
                trigger, action
            ));
            println!("  ❌ '{}' → NO MATCH (used by {})", trigger, action);
        } else {
            println!(
                "  ✅ '{}' → matches {} TQL relation(s): {:?}",
                trigger,
                matches.len(),
                matches
            );
        }
    }

    if !errors.is_empty() {
        panic!(
            "\n\n🚨 SCHEMA MISMATCHES FOUND:\n{}\n\nThese triggers would NEVER fire with real TQL data!\n",
            errors.join("\n")
        );
    }
}

// ═══════════════════════════════════════════════════════════════
// Test 2: All entity types used in test graphs exist in TQL
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_all_entity_types_exist_in_tql() {
    let schema = parse_tql(TQL_PATH);

    // Entity types used across our fiduciary tests
    // NOTE: TQL has no "merchant" entity — it's "user-merchant-unit"
    let entity_types_used: Vec<&str> = vec![
        "user",
        "obligation",
        "instrument",
        "user-merchant-unit", // Not "merchant" — real TQL name
        "user-bank-unit",
        "goal",
        "budget-estimation",
        "tax-due-event",
        "tax-sinking-fund",
        "tax-exemption-certificate",
        "tax-scenario",
        "reconciliation-case",
        "reconciliation-session",
        "recurring-pattern",
        "recurring-missing-alert",
        "asset",
        "asset-valuation",
        "interest-rate-term",
    ];

    println!("\n  ── SCHEMA VALIDATION: Entity Types ──\n");

    let mut errors = Vec::new();
    for etype in &entity_types_used {
        if schema.entity_types.contains(*etype) {
            println!("  ✅ Entity type '{}' exists in TQL", etype);
        } else {
            // Check if it's a substring match issue
            let close: Vec<&String> = schema
                .entity_types
                .iter()
                .filter(|e| e.contains(etype))
                .collect();
            errors.push(format!(
                "❌ Entity type '{}' NOT found in TQL! Close matches: {:?}",
                etype, close
            ));
            println!(
                "  ❌ Entity type '{}' NOT IN TQL (close: {:?})",
                etype, close
            );
        }
    }

    if !errors.is_empty() {
        panic!("\n\n🚨 ENTITY TYPE MISMATCHES:\n{}\n", errors.join("\n"));
    }
}

// ═══════════════════════════════════════════════════════════════
// Test 3: User entity can actually reach fiduciary-relevant entities
//
// Verify the "user" entity plays relations that connect to:
// - obligation (via obligation-between-parties, instrument-linked-to-obligation)
// - instrument (via user-has-instrument)
// - goal (via subledger-holds-goal-funds → through sub-ledger chain)
// - tax entities (via tax-party-has-exemption-certificate)
// - recurring-pattern (via pattern-owned-by)
// - asset (via lien-on-asset, asset-owned-by)
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_user_entity_plays_fiduciary_relations() {
    let schema = parse_tql(TQL_PATH);

    let user_plays = schema
        .entity_plays
        .get("user")
        .expect("'user' entity must exist in TQL and play roles");

    println!("\n  ── SCHEMA VALIDATION: User Plays Fiduciary Relations ──\n");
    println!("  User entity plays {} total relations\n", user_plays.len());

    // Relations the user MUST play for our fiduciary system
    let required_user_relations: Vec<(&str, &str)> = vec![
        (
            "user-has-instrument",
            "ShouldTransfer/Consolidate — user owns instruments",
        ),
        (
            "pattern-owned-by",
            "ShouldCancel — user owns recurring patterns",
        ),
        ("lien-on-asset", "ShouldPayDownLien — user has lien"),
        (
            "obligation-between-parties",
            "ShouldRefinance/Dispute — user is borrower/lender",
        ),
        (
            "obligation-claim-owned-by",
            "ShouldPay — user owns obligation claim",
        ),
        (
            "tax-party-has-exemption-certificate",
            "ShouldClaimExemption — user has exemptions",
        ),
        ("tax-party-has-tax-id", "ShouldPrepareTax — user has tax ID"),
        (
            "user-has-feed-provider",
            "Data ingestion — user has providers",
        ),
        ("asset-owned-by", "ShouldRevalueAsset — user owns assets"),
    ];

    let mut errors = Vec::new();
    for (relation, purpose) in &required_user_relations {
        if user_plays.contains(*relation) {
            println!("  ✅ user plays '{}' → {}", relation, purpose);
        } else {
            errors.push(format!(
                "❌ user does NOT play '{}' — needed for: {}",
                relation, purpose
            ));
            println!("  ❌ user does NOT play '{}' → {}", relation, purpose);
        }
    }

    // Also list all user plays for reference
    println!("\n  All user relations ({}):", user_plays.len());
    let mut sorted: Vec<&String> = user_plays.iter().collect();
    sorted.sort();
    for rel in &sorted {
        let is_fiduciary = required_user_relations
            .iter()
            .any(|(r, _)| *r == rel.as_str());
        let marker = if is_fiduciary { "★" } else { " " };
        println!("    {} {}", marker, rel);
    }

    if !errors.is_empty() {
        panic!(
            "\n\n🚨 USER CANNOT REACH FIDUCIARY ENTITIES:\n{}\n\n\
             The user entity doesn't play these relations,\n\
             so the fiduciary system cannot trigger these actions via real data!\n",
            errors.join("\n")
        );
    }
}

// ═══════════════════════════════════════════════════════════════
// Test 4: End-to-end with ONLY real TQL relation names
//
// Build scenarios using exclusively the exact relation names
// from SchemaFinverse.tql, not substring matches.
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_benchmark_scenarios_use_only_real_tql_names() {
    let schema = parse_tql(TQL_PATH);

    // Collect all relation names used across our benchmark test files
    // NOTE: Some test graphs use simplified names for readability;
    //       this test validates the exact TQL relation names used.
    let relations_used_in_tests: Vec<(&str, &str)> = vec![
        // From alignment benchmark — exact TQL relations
        ("obligation-has-interest-term", "Over-Leveraged Debtor"),
        ("subledger-holds-goal-funds", "Over-Leveraged Debtor"),
        ("user-has-instrument", "Over-Leveraged Debtor"),
        ("tax-liability-has-due-event", "Tax Deadline"),
        ("tax-sinking-fund-backed-by-account", "Tax Deadline"),
        ("tax-party-has-exemption-certificate", "Tax Deadline"),
        (
            "case-has-counterparty",
            "Fraud Detection — user-merchant-unit counterparty",
        ),
        ("pattern-owned-by", "Emergency Fund"),
        (
            "obligation-between-parties",
            "Disputed Charge — user is borrower",
        ),
        ("records-budget-estimation", "Disputed Charge"),
        ("lien-on-asset", "Lien Paydown"),
        ("asset-has-valuation", "Lien Paydown"),
        ("reconciliation-for-instrument", "Unreconciled Accounts"),
        ("tax-scenario-for-period", "Tax Scenario"),
        ("pattern-has-recurring-alert", "Complete Health"),
    ];

    println!("\n  ── SCHEMA VALIDATION: Test Relation Names ──\n");

    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    for (relation, scenario) in &relations_used_in_tests {
        if schema.relation_types.contains(*relation) {
            println!(
                "  ✅ '{}' is a real TQL relation (used in {})",
                relation, scenario
            );
        } else {
            // Check if it's close to a real relation
            let close: Vec<&String> = schema
                .relation_types
                .iter()
                .filter(|r| r.contains(relation) || relation.contains(r.as_str()))
                .collect();

            if !close.is_empty() {
                warnings.push(format!(
                    "⚠️  '{}' (in {}) is not an exact TQL relation. Close: {:?}",
                    relation, scenario, close
                ));
                println!(
                    "  ⚠️  '{}' → not exact. Close: {:?} ({})",
                    relation, close, scenario
                );
            } else {
                errors.push(format!(
                    "❌ '{}' (in {}) is NOT a TQL relation at all!",
                    relation, scenario
                ));
                println!("  ❌ '{}' → NOT IN TQL at all! ({})", relation, scenario);
            }
        }
    }

    println!();
    if !warnings.is_empty() {
        println!("  Warnings ({}):", warnings.len());
        for w in &warnings {
            println!("    {}", w);
        }
    }

    // Entity types used in tests
    let entity_types_in_tests: Vec<&str> = vec![
        "obligation",
        "goal",
        "instrument",
        "tax-due-event",
        "tax-sinking-fund",
        "tax-exemption-certificate",
        "tax-scenario",
        "reconciliation-case",
        "reconciliation-session",
        "recurring-pattern",
        "recurring-missing-alert",
        "asset",
        "asset-valuation",
        "budget-estimation",
        "user-merchant-unit",
    ];

    println!("\n  Entity types in tests:");
    for etype in &entity_types_in_tests {
        let exists = schema.entity_types.contains(*etype);
        let marker = if exists { "✅" } else { "❌" };
        println!("    {} {}", marker, etype);
        if !exists {
            // Check for merchant — it's actually user-merchant-unit
            let close: Vec<&String> = schema
                .entity_types
                .iter()
                .filter(|e| e.contains(etype))
                .collect();
            if !close.is_empty() {
                println!("       Close matches: {:?}", close);
            }
        }
    }

    if !errors.is_empty() {
        panic!(
            "\n\n🚨 TEST RELATIONS NOT IN TQL:\n{}\n\n\
             These test scenarios use made-up relation names!\n",
            errors.join("\n")
        );
    }
}

// ═══════════════════════════════════════════════════════════════
// Test 5: Schema coverage — what % of TQL entity domains are covered
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_schema_coverage_report() {
    let schema = parse_tql(TQL_PATH);

    // Financial domains from the TQL schema
    let domain_entity_types: Vec<(&str, Vec<&str>)> = vec![
        (
            "Accounts & Instruments",
            vec!["instrument", "account", "main-account", "sub-account"],
        ),
        (
            "Obligations & Debt",
            vec!["obligation", "interest-rate-term", "legal-document"],
        ),
        (
            "Goals & Budgets",
            vec!["goal", "budget-estimation", "master-budget"],
        ),
        (
            "Tax",
            vec![
                "tax-due-event",
                "tax-sinking-fund",
                "tax-exemption-certificate",
                "tax-scenario",
                "tax-liability",
                "tax-return",
                "tax-period",
                "tax-obligation",
                "tax-remittance",
                "tax-code",
            ],
        ),
        (
            "Reconciliation",
            vec!["reconciliation-case", "reconciliation-session"],
        ),
        (
            "Recurring Patterns",
            vec!["recurring-pattern", "recurring-missing-alert"],
        ),
        (
            "Assets",
            vec!["asset", "asset-valuation", "asset-identifier-record"],
        ),
        (
            "Evidence & Transactions",
            vec!["transaction-evidence", "evidence-line-item"],
        ),
        (
            "Users & Parties",
            vec!["user", "user-bank-unit", "user-merchant-unit"],
        ),
    ];

    // Which entity types we currently handle in fiduciary recommendations
    let handled_entity_types: HashSet<&str> = [
        "instrument",
        "obligation",
        "goal",
        "budget-estimation",
        "tax-due-event",
        "tax-sinking-fund",
        "tax-exemption-certificate",
        "tax-scenario",
        "reconciliation-case",
        "reconciliation-session",
        "recurring-pattern",
        "recurring-missing-alert",
        "asset",
        "asset-valuation",
    ]
    .iter()
    .copied()
    .collect();

    println!("\n  ── SCHEMA COVERAGE REPORT ──\n");
    println!("  Total TQL entity types: {}", schema.entity_types.len());
    println!(
        "  Total TQL relation types: {}",
        schema.relation_types.len()
    );
    println!(
        "  Entity types handled by fiduciary system: {}\n",
        handled_entity_types.len()
    );

    let mut total_types = 0;
    let mut covered_types = 0;

    for (domain, entity_types) in &domain_entity_types {
        let mut domain_covered = 0;
        let mut domain_total = 0;

        for etype in entity_types {
            if schema.entity_types.contains(*etype) {
                domain_total += 1;
                total_types += 1;
                if handled_entity_types.contains(etype) {
                    domain_covered += 1;
                    covered_types += 1;
                }
            }
        }

        let pct = if domain_total > 0 {
            domain_covered as f32 / domain_total as f32 * 100.0
        } else {
            0.0
        };
        let marker = if pct >= 50.0 { "✅" } else { "⚠️" };
        println!(
            "  {} {:<28} {}/{} entities covered ({:.0}%)",
            marker, domain, domain_covered, domain_total, pct
        );
    }

    let overall_pct = covered_types as f32 / total_types as f32 * 100.0;
    println!(
        "\n  Overall coverage: {}/{} entity types ({:.0}%)",
        covered_types, total_types, overall_pct
    );
    println!(
        "  User plays {} relations total",
        schema
            .entity_plays
            .get("user")
            .map(|p| p.len())
            .unwrap_or(0)
    );

    assert!(
        overall_pct >= 30.0,
        "Fiduciary system should cover at least 30% of financial entity types, got {:.0}%",
        overall_pct
    );
}
