use std::sync::Arc;
use tokio::runtime::Handle;
use hehrgnn::optimizer::gepa::{Candidate, Evaluator, EvalResult, SideInfo};
use crate::monad::provider::{LlmProvider, ProviderConfig};
use crate::lambda::planner::{ExecutionPlan, TaskType};
use crate::lambda::executor::LambdaExecutor;

/// Build a live LlmProvider pointed at OpenRouter + Trinity for GEPA evaluation.
pub fn trinity_provider() -> Arc<LlmProvider> {
    dotenvy::dotenv().ok();
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());
    let model = std::env::var("RIG_RLM_MODEL")
        .unwrap_or_else(|_| "arcee-ai/trinity-large-preview:free".to_string());
    
    let config = ProviderConfig::openai_compatible("openrouter", &model, &base_url, &api_key);
    Arc::new(LlmProvider::new(config))
}

/// A GEPA Evaluator that runs the entire lambda-RLM Map-Reduce Engine!
pub struct LambdaExecutorEvaluator {
    pub provider: Arc<LlmProvider>,
    pub query: String,
    pub massive_context: String,
    pub expected_keyword: String,
}

impl Evaluator for LambdaExecutorEvaluator {
    fn evaluate(&self, candidate: &Candidate) -> EvalResult {
        // 1. Unpack the GEPA-mutated parameters!
        let k_star = candidate.get_f32("k_star", 2.0).round() as usize;
        let tau_star = candidate.get_f32("tau_star", 1500.0).round() as usize;
        
        // Build a synthetic execution plan based on the mutated params
        let plan = ExecutionPlan {
            task_type: TaskType::Summarise,
            k_star: k_star.max(2),      // Prevent infinite loops
            tau_star: tau_star.max(50), // Prevent token explosion
            depth: 3,
            estimated_cost: 0.0,
            estimated_calls: 0,
            has_prefilter: false,
            neural_compose: true,
        };

        // 2. GEPA Evaluator is a synchronous trait, so we spin up a local runtime
        let executor = LambdaExecutor::new(plan.clone(), self.provider.clone(), self.query.clone());
        
        println!("  [GEPA-RLM] Evaluator running Graph: k*={} τ*={}", plan.k_star, plan.tau_star);
        
        // Execute the Map-Reduce!
        let start = std::time::Instant::now();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(async {
            executor.execute(&self.massive_context).await
        });
        let elapsed = start.elapsed().as_secs_f64();

        // 3. Build the Actionable Side Information (ASI)
        let mut side_info = SideInfo::new();
        
        // Log the topological structure to GEPA!
        side_info.log("Execution Topology:");
        side_info.log(plan.to_mermaid());

        let score = match result {
            Ok(output) => {
                let mut score = 0.0;
                
                // Accuracy metric
                if output.to_lowercase().contains(&self.expected_keyword) {
                    score += 0.8;
                    side_info.log("SUCCESS: Expected keyword was found in the synthesis!");
                } else {
                    side_info.log("FAILURE: Expected keyword was missing.");
                    side_info.log(&format!("Actual Output: {}", output));
                }
                
                // Efficiency metric
                let efficiency_bonus = (100.0 / (elapsed + 1.0)).min(0.2); 
                score += efficiency_bonus;
                
                side_info.score("accuracy", score);
                side_info.score("efficiency", efficiency_bonus);
                
                score
            }
            Err(e) => {
                side_info.log(format!("CRITICAL ERROR: Execution failed: {}", e));
                0.0
            }
        };
        
        EvalResult { score, side_info }
    }
}
