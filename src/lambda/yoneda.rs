//! Yoneda Lemma: Representable Functors for Massive Context Documents.
//!
//! # The Mathematics
//!
//! The **Yoneda Lemma** states that for a locally small category **C**,
//! an object `A` is completely determined (up to isomorphism) by the
//! representable functor `Hom(A, −)`:
//!
//! ```text
//! Nat(Hom(A, −), F) ≅ F(A)
//! ```
//!
//! The **Yoneda Embedding** `y: C → [Cᵒᵖ, Set]` given by `y(A) = Hom(−, A)`
//! is **fully faithful**: two objects are isomorphic if and only if their
//! representable functors are naturally isomorphic.
//!
//! # Application to LLM Contexts
//!
//! We define a category **Doc** where:
//! - **Objects** are massive text documents `P`
//! - **Morphisms** `f: P → Q` are document transformations (summarization,
//!   truncation, extraction — anything that maps one document to another)
//!
//! And a **query category** **Q** where:
//! - **Objects** are query types (search, summarize, classify, etc.)
//! - **Morphisms** `g: q₁ → q₂` are query refinements (e.g., "summarize" →
//!   "summarize the technical findings")
//!
//! The representable functor `y(P): Q → Set` maps each query `q` to the
//! result of applying λ-RLM on document `P` with query `q`:
//!
//! ```text
//! y(P)(q) = λ-RLM(P, q): String
//! ```
//!
//! **Key Yoneda guarantees**:
//! 1. `P` is never collapsed into a fixed embedding — it is held lazily
//! 2. `P` is completely characterized by `{probe(q) | q ∈ Q}` — the set of
//!    all possible query responses
//! 3. Two documents `P₁ ≅ P₂` iff `∀q: probe(P₁, q) ≅ probe(P₂, q)`
//!    (full faithfulness of the Yoneda embedding)
//! 4. Query morphisms (refinements) map coherently through the functor
//!    (naturality of the representation)

use std::sync::Arc;

use crate::monad::provider::LlmProvider;
use crate::monad::error::Result;
use super::{lambda_rlm, LambdaConfig};
use super::profunctor::TypedPipeline;

// ═════════════════════════════════════════════════════════════════════════════
// Query Category — Objects and Morphisms
// ═════════════════════════════════════════════════════════════════════════════

/// A morphism in the query category Q: transforms one query into another.
///
/// Query morphisms must satisfy:
/// - **Identity**: `QueryMorphism::identity()` leaves queries unchanged
/// - **Composition**: `(g ∘ f)(q) = g(f(q))`
///
/// In the Yoneda context, naturality requires that applying a morphism
/// to a query *before* probing is coherent with transforming the result
/// *after* probing.
pub struct QueryMorphism {
    /// Human-readable name for this morphism (e.g., "refine_to_technical")
    pub name: String,
    /// The transformation: takes a query string, returns a refined query string.
    transform: Box<dyn Fn(&str) -> String + Send + Sync>,
}

impl QueryMorphism {
    /// Create a new query morphism.
    pub fn new(name: impl Into<String>, f: impl Fn(&str) -> String + Send + Sync + 'static) -> Self {
        Self {
            name: name.into(),
            transform: Box::new(f),
        }
    }

    /// The identity morphism: id(q) = q.
    pub fn identity() -> Self {
        Self::new("id", |q| q.to_string())
    }

    /// Apply this morphism to a query.
    pub fn apply(&self, query: &str) -> String {
        (self.transform)(query)
    }

    /// Compose two morphisms: (self ∘ other)(q) = self(other(q)).
    pub fn compose(self, other: QueryMorphism) -> QueryMorphism {
        let name = format!("{} ∘ {}", self.name, other.name);
        let f = self.transform;
        let g = other.transform;
        QueryMorphism::new(name, move |q| {
            let intermediate = g(q);
            f(&intermediate)
        })
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Representable Functor y(P): Q → Set
// ═════════════════════════════════════════════════════════════════════════════

/// The Yoneda representation of a massive context document.
///
/// `YonedaContext` is the **representable functor** `y(P): Q → Set`.
///
/// Instead of collapsing a massive document `P` into a fixed-size embedding
/// (which is lossy), we hold `P` lazily and evaluate it only when probed.
/// The document's semantic identity is completely determined by the set of
/// all probe results: `{probe(q) | q ∈ Q}`.
///
/// This directly implements the Yoneda Lemma's key insight:
/// > An object is completely determined by the collection of all morphisms
/// > into (or out of) it.
///
/// # Functorial Properties
///
/// `YonedaContext` satisfies:
/// - **Objects**: `probe(q)` maps query objects to result sets
/// - **Morphisms**: `fmap(f, q)` maps query morphisms to result transformations
/// - **Naturality**: `probe(f(q)) ≅ fmap(f)(probe(q))` for well-formed morphisms
pub struct YonedaContext {
    /// The raw massive document (object P in category Doc).
    document: Arc<String>,
    /// The LLM provider (the "evaluation oracle").
    provider: Arc<LlmProvider>,
    /// λ-RLM configuration (bounds the Hylomorphism).
    config: LambdaConfig,
}

impl YonedaContext {
    /// **Lift** a massive document into the Yoneda representation.
    ///
    /// This is the embedding `y: Doc → [Q, Set]` applied to a specific document.
    /// The document is stored lazily — no computation happens until `probe()`.
    pub fn lift(document: impl Into<String>, provider: Arc<LlmProvider>, config: LambdaConfig) -> Self {
        Self {
            document: Arc::new(document.into()),
            provider,
            config,
        }
    }

    /// Backwards-compatible alias for `lift`.
    pub fn new(document: impl Into<String>, provider: Arc<LlmProvider>, config: LambdaConfig) -> Self {
        Self::lift(document, provider, config)
    }

    // ─── Functor: Objects (y(P) on query objects) ───────────────────────

    /// Evaluate the representable functor on a query object.
    ///
    /// `y(P)(q)` = the result of applying λ-RLM on document `P` with query `q`.
    ///
    /// This is a single point in the representable functor's image.
    /// The Yoneda Lemma guarantees that the collection of all such evaluations
    /// determines `P` up to isomorphism.
    pub async fn probe(&self, query: &str) -> Result<String> {
        eprintln!(
            "🧿 [Yoneda] y(P)(q): probing representable functor ({} chars) with query: {:?}",
            self.document.len(),
            query
        );
        let timer = std::time::Instant::now();
        
        let result = lambda_rlm(
            &self.document, 
            query, 
            Arc::clone(&self.provider), 
            self.config.clone()
        ).await;

        match &result {
            Ok(_) => eprintln!("🧿 [Yoneda] Evaluation complete in {:.2}s", timer.elapsed().as_secs_f64()),
            Err(e) => eprintln!("🧿 [Yoneda] Evaluation failed: {}", e),
        }

        result
    }

    // ─── Functor: Morphisms (y(P) on query morphisms) ──────────────────

    /// Apply the representable functor to a query **morphism**.
    ///
    /// Given a morphism `f: q₁ → q₂` in the query category, this
    /// evaluates `y(P)(q₂)` where `q₂ = f(q₁)`.
    ///
    /// For a proper functor, this should satisfy:
    /// - `fmap(id, q) = probe(q)` (identity)
    /// - `fmap(g ∘ f, q) = fmap(g, f(q))` (composition)
    pub async fn fmap(&self, morphism: &QueryMorphism, query: &str) -> Result<String> {
        let transformed_query = morphism.apply(query);
        eprintln!(
            "🧿 [Yoneda] fmap({}, {:?}) → {:?}",
            morphism.name,
            query,
            &transformed_query[..transformed_query.len().min(80)]
        );
        self.probe(&transformed_query).await
    }

    // ─── Typed Evaluation via Profunctor ────────────────────────────────

    /// Evaluate a **typed** query via Profunctor optics.
    ///
    /// Lifts the representable functor into an arbitrary typed category:
    /// ```text
    /// Q ──lmap──→ String ──y(P)──→ String ──rmap──→ R
    /// ```
    pub async fn probe_typed<Q, R, F, G>(
        &self,
        query: &Q,
        lmap: F,
        rmap: G,
    ) -> Result<R>
    where
        F: Fn(&Q) -> String,
        G: Fn(String) -> R,
    {
        let query_text = lmap(query);

        eprintln!(
            "🧿 [Yoneda/Profunctor] Typed probe: {} → y(P) → {}",
            std::any::type_name::<Q>(),
            std::any::type_name::<R>(),
        );

        let timer = std::time::Instant::now();
        let raw_result = lambda_rlm(
            &self.document,
            &query_text,
            Arc::clone(&self.provider),
            self.config.clone(),
        ).await?;

        eprintln!(
            "🧿 [Yoneda/Profunctor] Complete in {:.2}s, applying rmap",
            timer.elapsed().as_secs_f64()
        );

        Ok(rmap(raw_result))
    }

    /// Build a reusable `TypedPipeline` bound to this document's context.
    pub fn into_typed_pipeline<C, D, F, G>(
        self,
        user_query: impl Into<String>,
        lmap: F,
        rmap: G,
    ) -> TypedPipeline<C, D, F, G>
    where
        F: Fn(&C) -> String,
        G: Fn(String) -> D,
    {
        TypedPipeline::new(
            self.provider,
            self.config,
            user_query,
            lmap,
            rmap,
        )
    }

    /// Get a reference to the underlying document.
    pub fn document(&self) -> &str {
        &self.document
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Yoneda Embedding: Full Faithfulness
// ═════════════════════════════════════════════════════════════════════════════

/// Result of a Yoneda equivalence check between two documents.
#[derive(Debug)]
pub struct YonedaEquivalence {
    /// Whether the documents are semantically equivalent over the test queries.
    pub equivalent: bool,
    /// Per-query similarity scores (0.0 = different, 1.0 = identical).
    pub scores: Vec<(String, f64)>,
    /// Average similarity across all test queries.
    pub mean_similarity: f64,
}

/// Test whether two `YonedaContext`s are **Yoneda-equivalent**.
///
/// By the Yoneda Lemma's full faithfulness guarantee:
/// > `P₁ ≅ P₂` in **Doc** ⟺ `y(P₁) ≅ y(P₂)` as functors
///
/// We approximate this by probing both documents with a set of test queries
/// and measuring the similarity of their responses. If all responses are
/// semantically equivalent, the documents are Yoneda-equivalent.
///
/// # Arguments
/// - `ctx1`, `ctx2` — the two Yoneda-embedded documents
/// - `test_queries` — the set of probe morphisms to check
/// - `similarity` — a function that scores how similar two results are (0.0–1.0)
/// - `threshold` — minimum per-query similarity to consider equivalent (e.g., 0.85)
pub async fn yoneda_equivalence<S>(
    ctx1: &YonedaContext,
    ctx2: &YonedaContext,
    test_queries: &[&str],
    similarity: S,
    threshold: f64,
) -> Result<YonedaEquivalence>
where
    S: Fn(&str, &str) -> f64,
{
    eprintln!(
        "🧿 [Yoneda Embedding] Testing full faithfulness: P₁({} chars) ≅? P₂({} chars) over {} queries",
        ctx1.document.len(),
        ctx2.document.len(),
        test_queries.len(),
    );

    let mut scores = Vec::with_capacity(test_queries.len());
    let mut total_sim = 0.0;

    for &query in test_queries {
        let r1 = ctx1.probe(query).await?;
        let r2 = ctx2.probe(query).await?;
        let sim = similarity(&r1, &r2);
        total_sim += sim;
        scores.push((query.to_string(), sim));

        eprintln!(
            "🧿 [Yoneda] Query {:?}: sim = {:.3}",
            &query[..query.len().min(50)],
            sim,
        );
    }

    let mean_similarity = if test_queries.is_empty() {
        1.0
    } else {
        total_sim / test_queries.len() as f64
    };

    let equivalent = scores.iter().all(|(_, s)| *s >= threshold);

    eprintln!(
        "🧿 [Yoneda Embedding] Result: equivalent={}, mean_sim={:.3}",
        equivalent,
        mean_similarity,
    );

    Ok(YonedaEquivalence {
        equivalent,
        scores,
        mean_similarity,
    })
}

// ═════════════════════════════════════════════════════════════════════════════
// Naturality Check
// ═════════════════════════════════════════════════════════════════════════════

/// Verify the **naturality condition** of the representable functor.
///
/// For a query morphism `f: q₁ → q₂`, naturality requires:
///
/// ```text
///     y(P)(q₁) ──y(P)(f)──→ y(P)(q₂)
///        ↓                      ↓
///    probe(q₁)            probe(f(q₁))
/// ```
///
/// In other words: `probe(f(q)) ≈ transform(probe(q))` for some
/// coherent result transformation. We check this by measuring whether
/// the result of probing with the transformed query is consistent with
/// transforming after probing.
///
/// Returns the similarity score (0.0–1.0) between the two paths through
/// the naturality square.
pub async fn check_naturality<S>(
    ctx: &YonedaContext,
    base_query: &str,
    morphism: &QueryMorphism,
    result_transform: impl Fn(&str) -> String,
    similarity: S,
) -> Result<f64>
where
    S: Fn(&str, &str) -> f64,
{
    // Path 1: probe(q₁), then transform the result
    let result_base = ctx.probe(base_query).await?;
    let path1 = result_transform(&result_base);

    // Path 2: apply morphism first, then probe
    let transformed_query = morphism.apply(base_query);
    let path2 = ctx.probe(&transformed_query).await?;

    let sim = similarity(&path1, &path2);
    eprintln!(
        "🧿 [Naturality] f={}, q={:?}: path1≈path2 = {:.3}",
        morphism.name,
        &base_query[..base_query.len().min(40)],
        sim,
    );

    Ok(sim)
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_morphism_identity() {
        let id = QueryMorphism::identity();
        assert_eq!(id.apply("hello"), "hello");
    }

    #[test]
    fn test_query_morphism_composition() {
        let add_prefix = QueryMorphism::new("add_prefix", |q| format!("Summarize: {}", q));
        let add_suffix = QueryMorphism::new("add_suffix", |q| format!("{} (be concise)", q));

        // Compose: add_suffix ∘ add_prefix
        let composed = add_suffix.compose(add_prefix);
        let result = composed.apply("some text");

        assert_eq!(result, "Summarize: some text (be concise)");
    }

    #[test]
    fn test_query_morphism_identity_law() {
        // id ∘ f = f
        let f = QueryMorphism::new("f", |q| format!("refined: {}", q));
        let id = QueryMorphism::identity();
        let composed = f.compose(id);
        assert_eq!(composed.apply("test"), "refined: test");
    }

    #[test]
    fn test_query_morphism_associativity() {
        // (h ∘ g) ∘ f = h ∘ (g ∘ f)  — we test one direction
        let f = QueryMorphism::new("f", |q| format!("({} after f)", q));
        let g = QueryMorphism::new("g", |q| format!("({} after g)", q));
        let h = QueryMorphism::new("h", |q| format!("({} after h)", q));

        // (h ∘ g) ∘ f
        let hg = QueryMorphism::new("h", |q| format!("({} after h)", q))
            .compose(QueryMorphism::new("g", |q| format!("({} after g)", q)));
        let hgf = hg.compose(f);

        // h ∘ (g ∘ f)
        let gf = QueryMorphism::new("g", |q| format!("({} after g)", q))
            .compose(QueryMorphism::new("f", |q| format!("({} after f)", q)));
        let h_gf = h.compose(gf);

        let input = "x";
        assert_eq!(hgf.apply(input), h_gf.apply(input), "Associativity must hold");
    }

    /// Simple Jaccard similarity for testing (production would use embeddings).
    fn jaccard_similarity(a: &str, b: &str) -> f64 {
        let set_a: std::collections::HashSet<&str> = a.split_whitespace().collect();
        let set_b: std::collections::HashSet<&str> = b.split_whitespace().collect();
        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();
        if union == 0 { 1.0 } else { intersection as f64 / union as f64 }
    }

    #[test]
    fn test_jaccard_similarity() {
        assert!((jaccard_similarity("hello world", "hello world") - 1.0).abs() < f64::EPSILON);
        assert!(jaccard_similarity("hello world", "foo bar") < 0.01);
        assert!(jaccard_similarity("the quick brown fox", "the quick red fox") > 0.5);
    }
}
