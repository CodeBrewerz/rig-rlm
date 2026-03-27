//! Category Theory: Profunctor Optics for LLM prompt-to-result mapping.
//! 
//! A Profunctor represents a heterogeneous relation `P(A, B)` that is:
//! - Contravariant in `A` (`lmap`: mapping new inputs backward into the LLM context)
//! - Covariant in `B` (`rmap`: mapping raw LLM outputs forward into a new datatype)
//!
//! In λ-RLM, an LLM execution is naturally a Profunctor over `(String, String)`.
//! By passing an LLM execution through `dimap`, we instantly lift it into a 
//! `(StructuredInput, StructuredOutput)` optical pipeline without breaking closure.
//!
//! # Production Usage
//!
//! ```rust,ignore
//! use rig_rlm::lambda::profunctor::TypedPipeline;
//!
//! // Define your domain types
//! struct Contract { id: u32, text: String }
//! struct RiskReport { score: f64, summary: String }
//!
//! // Build a typed pipeline that maps Contract → String → [λ-RLM] → String → RiskReport
//! let pipeline = TypedPipeline::new(
//!     executor,                             // LambdaExecutor
//!     |c: Contract| c.text,                 // lmap: extract the prompt
//!     |output: String| parse_risk(output),  // rmap: parse the result
//! );
//!
//! let report: RiskReport = pipeline.execute(contract).await?;
//! ```

use std::marker::PhantomData;
use std::sync::Arc;
use crate::lambda::executor::LambdaExecutor;
use crate::lambda::planner::{self, CostParams, TaskType};
use crate::lambda::combinators;
use crate::monad::provider::LlmProvider;

// ═════════════════════════════════════════════════════════════════════════════
// Core Synchronous Profunctor (pure functions)
// ═════════════════════════════════════════════════════════════════════════════

/// The Core Synchronous Profunctor trait.
pub trait Profunctor<A, B> {
    fn dimap<C, D, F, G>(self, lmap: F, rmap: G) -> Dimap<Self, F, G, A, B, C, D>
    where
        Self: Sized,
        F: Fn(C) -> A,
        G: Fn(B) -> D;
}

pub struct Dimap<P, F, G, A, B, C, D> {
    inner: P,
    lmap: F,
    rmap: G,
    _marker: PhantomData<(A, B, C, D)>,
}

impl<A, B, Func> Profunctor<A, B> for Func
where
    Func: Fn(A) -> B,
{
    fn dimap<C, D, F, G>(self, lmap: F, rmap: G) -> Dimap<Self, F, G, A, B, C, D>
    where
        F: Fn(C) -> A,
        G: Fn(B) -> D,
    {
        Dimap { inner: self, lmap, rmap, _marker: PhantomData }
    }
}

impl<P, F, G, A, B, C, D> Dimap<P, F, G, A, B, C, D>
where
    P: Fn(A) -> B,
    F: Fn(C) -> A,
    G: Fn(B) -> D,
{
    pub fn execute(&self, input: C) -> D {
        let a = (self.lmap)(input);
        let b = (self.inner)(a);
        (self.rmap)(b)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Async Profunctor Integration for λ-RLM Execution Engine
// ═════════════════════════════════════════════════════════════════════════════

/// Async version of Profunctor for non-blocking execution pipelines (like λ-RLM).
pub trait AsyncProfunctor<A, B> {
    fn dimap_async<C, D, F, G>(self, lmap: F, rmap: G) -> AsyncDimap<Self, F, G, A, B, C, D>
    where
        Self: Sized,
        F: Fn(C) -> A,
        G: Fn(B) -> D;
}

/// The Async equivalent of `Dimap`, which wraps an executor and preserves its `.await` nature.
pub struct AsyncDimap<P, F, G, A, B, C, D> {
    pub inner: P,
    pub lmap: F,
    pub rmap: G,
    _marker: PhantomData<(A, B, C, D)>,
}

/// Mathematically integrate AsyncProfunctor natively onto the `LambdaExecutor`!
/// This elevates it from a String->String transformer into a universal strongly-typed graph evaluator.
impl AsyncProfunctor<String, String> for LambdaExecutor {
    fn dimap_async<C, D, F, G>(self, lmap: F, rmap: G) -> AsyncDimap<Self, F, G, String, String, C, D>
    where
        F: Fn(C) -> String,
        G: Fn(String) -> D,
    {
        AsyncDimap {
            inner: self,
            lmap,
            rmap,
            _marker: PhantomData,
        }
    }
}

impl<F, G, C, D> AsyncDimap<LambdaExecutor, F, G, String, String, C, D>
where
    F: Fn(C) -> String,
    G: Fn(String) -> D,
{
    /// Evaluates the entire λ-RLM Map-Reduce execution tree dynamically bounded 
    /// by the Profunctor mappings.
    pub async fn execute_typed(&self, input: C) -> anyhow::Result<D> {
        // Contravariant mapping -> Transforms strictly typed inputs to prompt context
        let a = (self.lmap)(input);
        
        // Hylomorphism recursively computes fixed-point closure String->String
        let b = self.inner.execute(&a).await?;
        
        // Covariant mapping -> Decodes final collapsed summary into strict output structure
        Ok((self.rmap)(b))
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// TypedPipeline — Production-Ready Profunctor Integration
// ═════════════════════════════════════════════════════════════════════════════

/// A strongly-typed λ-RLM pipeline that maps `C -> D` through the recursive
/// Map-Reduce engine via Profunctor optics.
///
/// This is the **production entry point** for using profunctors with λ-RLM.
/// Instead of manually constructing prompts and parsing outputs, callers define:
///
/// - `lmap: C -> String` — serializes domain input into a prompt context
/// - `rmap: String -> D` — parses the LLM's raw output into domain output
///
/// The pipeline handles all 5 λ-RLM phases internally:
/// task detection → planning → cost estimation → Φ execution → typed output.
///
/// # Category Theory
///
/// `TypedPipeline` is the **Hom-set morphism** in the Profunctor category:
///
/// ```text
/// lmap          Φ (Hylomorphism)        rmap
/// C ────────→ String ──────────→ String ────────→ D
///   contra-                                covariant
///   variant                                
/// ```
///
/// The composition `rmap ∘ Φ ∘ lmap` forms a natural transformation
/// from `Hom(_, C)` to `Hom(_, D)` in the functor category.
pub struct TypedPipeline<C, D, F, G>
where
    F: Fn(&C) -> String,
    G: Fn(String) -> D,
{
    /// The LLM provider (used for task detection + leaf calls).
    provider: Arc<LlmProvider>,
    /// The λ-RLM configuration.
    config: super::LambdaConfig,
    /// The user query (morphism selector for the Yoneda representable).
    user_query: String,
    /// Contravariant mapping: domain input → prompt context.
    lmap: F,
    /// Covariant mapping: raw LLM output → domain output.
    rmap: G,
    _marker: PhantomData<C>,
}

impl<C, D, F, G> TypedPipeline<C, D, F, G>
where
    F: Fn(&C) -> String,
    G: Fn(String) -> D,
{
    /// Construct a new typed pipeline.
    ///
    /// # Arguments
    /// - `provider` — The LLM provider for model calls
    /// - `config` — λ-RLM configuration (context window, accuracy target, cost params)
    /// - `user_query` — The query string passed to leaf templates
    /// - `lmap` — Contravariant: serialize domain input `C` into prompt text
    /// - `rmap` — Covariant: parse raw LLM output into domain type `D`
    pub fn new(
        provider: Arc<LlmProvider>,
        config: super::LambdaConfig,
        user_query: impl Into<String>,
        lmap: F,
        rmap: G,
    ) -> Self {
        Self {
            provider,
            config,
            user_query: user_query.into(),
            lmap,
            rmap,
            _marker: PhantomData,
        }
    }

    /// Execute the full λ-RLM pipeline on a typed input.
    ///
    /// Internally performs:
    /// 1. `lmap(input)` → prompt string
    /// 2. Task detection (1 LLM call)
    /// 3. Optimal planning (0 LLM calls)
    /// 4. Φ recursive execution (bounded LLM calls)
    /// 5. `rmap(output)` → typed result
    pub async fn execute(&self, input: &C) -> crate::monad::error::Result<D> {
        // ── Contravariant: C → String ──
        let prompt = (self.lmap)(input);
        let n = combinators::token_count(&prompt);

        eprintln!(
            "🔬 [Profunctor] lmap: {} → {} tokens",
            std::any::type_name::<C>(),
            n
        );

        // ── Core λ-RLM phases (task detect → plan → execute Φ) ──
        let raw_output = super::lambda_rlm(
            &prompt,
            &self.user_query,
            Arc::clone(&self.provider),
            self.config.clone(),
        ).await?;

        // ── Covariant: String → D ──
        eprintln!(
            "🔬 [Profunctor] rmap: {} chars → {}",
            raw_output.len(),
            std::any::type_name::<D>()
        );
        let typed_output = (self.rmap)(raw_output);

        Ok(typed_output)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    struct FinancialRecord { id: u32, amount: f64 }
    struct AnalyzedResult { risk_score: f64 }

    #[test]
    fn test_llm_profunctor_dimap() {
        // Here is a mock LLM Execution. It operates solely in Category String.
        // P(String, String)
        let root_llm_evaluator = |prompt: String| -> String {
            format!("LLM Output: I read {} and calculated a score of 0.95", prompt)
        };

        // We want to evaluate it over Category FinancialRecord -> Category AnalyzedResult.
        // Instead of writing a wrapper function, we apply `dimap`.
        
        let lmap = |record: FinancialRecord| -> String {
            format!("Record #{} has ${}", record.id, record.amount)
        };
        
        let rmap = |_llm_output: String| -> AnalyzedResult {
            // Very simplistic parser for testing
            AnalyzedResult { risk_score: 0.95 }
        };

        // Mathematically lift the String->String operation to Record->Result!
        let typed_optical_pipeline = root_llm_evaluator.dimap(lmap, rmap);

        // Execute it safely
        let input_record = FinancialRecord { id: 7, amount: 1000.0 };
        let final_result = typed_optical_pipeline.execute(input_record);

        assert_eq!(final_result.risk_score, 0.95);
    }
}
