use std::sync::Arc;

use crate::monad::provider::LlmProvider;
use crate::monad::error::Result;
use super::{lambda_rlm, LambdaConfig};

/// The Yoneda representation of a massive context document.
///
/// According to the Yoneda Lemma, an object `A` is completely determined 
/// up to isomorphism by the set of all morphisms into it (or out of it): `Hom(_, A)`.
///
/// In LLM terms, the semantic identity of a massive document `P` does not need to be 
/// crushed prematurely into a single fixed-size vector embedding (which loses information). 
/// Instead, `P` is completely defined by its natural transformation against 
/// all possible queries `X`, evaluated through the deterministic λ-RLM engine.
///
/// `YonedaContext` holds the raw text `P` lazily. It acts as an infinitely large
/// "context window" that evaluates its meaning only when explicitly probed.
pub struct YonedaContext {
    /// The raw massive document (P).
    document: Arc<String>,
    /// The provider used to execute the transformations.
    provider: Arc<LlmProvider>,
    /// The constraints for the λ-RLM framework (e.g., k*, tau* constants).
    config: LambdaConfig,
}

impl YonedaContext {
    /// Lift a massive document into the Yoneda Context.
    pub fn new(document: impl Into<String>, provider: Arc<LlmProvider>, config: LambdaConfig) -> Self {
        Self {
            document: Arc::new(document.into()),
            provider,
            config,
        }
    }

    /// Evaluates the query (morphism) against the document `P`.
    ///
    /// This triggers the Hylomorphism defined by λ-RLM:
    /// 1. Coalgebraic Unfold: `SPLIT(P, k*)`
    /// 2. Algebraic Fold: `REDUCE(⊕)` over the mapped leaf outputs.
    pub async fn probe(&self, query: &str) -> Result<String> {
        eprintln!(
            "🧿 [Yoneda] Probing massive context functor ({} chars) with query: {:?}",
            self.document.len(),
            query
        );
        let timer = std::time::Instant::now();
        
        // Feed it into the λ-RLM engine
        let result = lambda_rlm(
            &self.document, 
            query, 
            Arc::clone(&self.provider), 
            self.config.clone()
        ).await;

        match &result {
            Ok(_) => eprintln!("🧿 [Yoneda] Collapse complete in {:.2}s", timer.elapsed().as_secs_f64()),
            Err(e) => eprintln!("🧿 [Yoneda] Collapse failed: {}", e),
        }

        result
    }
}
