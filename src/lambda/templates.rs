//! Leaf prompt templates for each task type.
//!
//! These are the task-specific formatting functions applied at the
//! leaves of the recursion tree, where the base model M is invoked
//! on bounded sub-prompts guaranteed to fit within K.

use super::planner::TaskType;

/// Format a leaf sub-prompt for the base model M.
///
/// The template adds a task-specific instruction prefix so the
/// model knows what to extract from the chunk. This is the
/// `TEMPLATE[τ_type].FMT(P)` call from Algorithm 4, line 3.
pub fn format_leaf(sub_prompt: &str, task_type: TaskType, user_query: &str) -> String {
    match task_type {
        TaskType::Search => format!(
            "You are searching for information to answer the following query:\n\
             Query: {user_query}\n\n\
             Search the following text and extract the MOST RELEVANT passage \
             that answers the query. If no relevant information is found, \
             respond with 'NOT_FOUND'.\n\n\
             Text:\n{sub_prompt}"
        ),

        TaskType::Classify => format!(
            "Classify each item in the following text according to the query.\n\
             Query: {user_query}\n\n\
             Return your classifications as a structured list.\n\n\
             Text:\n{sub_prompt}"
        ),

        TaskType::Aggregate => format!(
            "Analyze the following text and extract counts, statistics, or \
             aggregate information relevant to the query.\n\
             Query: {user_query}\n\n\
             Return your findings as key:value pairs.\n\n\
             Text:\n{sub_prompt}"
        ),

        TaskType::Pairwise => format!(
            "For each entity in the following text, extract its label and \
             relevant attributes for pairwise comparison.\n\
             Query: {user_query}\n\n\
             Return structured entity descriptions.\n\n\
             Text:\n{sub_prompt}"
        ),

        TaskType::Summarise => format!(
            "Summarize the following text concisely, preserving key facts \
             and relationships relevant to the query.\n\
             Query: {user_query}\n\n\
             Text:\n{sub_prompt}"
        ),

        TaskType::MultiHop => format!(
            "Extract evidence relevant to the query from the following text. \
             Return ONLY facts that help answer the query.\n\
             Query: {user_query}\n\n\
             Text:\n{sub_prompt}"
        ),
    }
}

/// Format the final synthesis prompt for tasks that need neural composition.
///
/// Used by Summarise and MultiHop task types where REDUCE requires
/// an M call to compose partial results (M ∘ CONCAT).
pub fn format_synthesis(partial_results: &str, task_type: TaskType, user_query: &str) -> String {
    match task_type {
        TaskType::Summarise => format!(
            "The following are summaries of different sections of a long document. \
             Synthesize them into a single coherent answer to the query.\n\
             Query: {user_query}\n\n\
             Partial summaries:\n{partial_results}\n\n\
             Provide a comprehensive final answer:"
        ),

        TaskType::MultiHop => format!(
            "The following evidence was extracted from multiple sections of a document. \
             Combine this evidence to answer the query. Reason step-by-step.\n\
             Query: {user_query}\n\n\
             Evidence:\n{partial_results}\n\n\
             Final answer:"
        ),

        _ => format!(
            "Combine these partial results to answer the query.\n\
             Query: {user_query}\n\n\
             Partial results:\n{partial_results}\n\n\
             Final answer:"
        ),
    }
}

/// Format the composition (REDUCE) step for symbolic task types.
///
/// For Search: pick the best result.
/// For Aggregate: merge count dictionaries.
/// For Classify/Pairwise: concatenate.
pub fn compose_symbolic(results: Vec<String>, task_type: TaskType) -> String {
    match task_type {
        TaskType::Search => {
            // FilterBest: discard NOT_FOUND, keep the most relevant
            let non_empty: Vec<&String> = results
                .iter()
                .filter(|r| !r.contains("NOT_FOUND") && !r.trim().is_empty())
                .collect();
            if non_empty.is_empty() {
                "No relevant information found.".to_string()
            } else if non_empty.len() == 1 {
                non_empty[0].clone()
            } else {
                // Return all non-empty results separated for upstream synthesis
                non_empty.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n---\n")
            }
        }

        TaskType::Classify | TaskType::Pairwise => {
            // Concat: join all results
            results.join("\n\n")
        }

        TaskType::Aggregate => {
            // MergeCounts: join with structured separator
            results.join("\n---\n")
        }

        // Neural compose types — should use format_synthesis instead
        TaskType::Summarise | TaskType::MultiHop => {
            results.join("\n\n")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leaf_format_search() {
        let result = format_leaf("some text here", TaskType::Search, "find the answer");
        assert!(result.contains("find the answer"));
        assert!(result.contains("some text here"));
        assert!(result.contains("NOT_FOUND"));
    }

    #[test]
    fn test_leaf_format_summarise() {
        let result = format_leaf("long text", TaskType::Summarise, "what happened?");
        assert!(result.contains("Summarize"));
        assert!(result.contains("what happened?"));
    }

    #[test]
    fn test_compose_search_filters_not_found() {
        let results = vec![
            "NOT_FOUND".to_string(),
            "The answer is 42.".to_string(),
            "NOT_FOUND".to_string(),
        ];
        let composed = compose_symbolic(results, TaskType::Search);
        assert_eq!(composed, "The answer is 42.");
    }

    #[test]
    fn test_compose_search_all_not_found() {
        let results = vec!["NOT_FOUND".to_string(), "NOT_FOUND".to_string()];
        let composed = compose_symbolic(results, TaskType::Search);
        assert!(composed.contains("No relevant information found"));
    }

    #[test]
    fn test_synthesis_prompt() {
        let result = format_synthesis("summary 1\nsummary 2", TaskType::Summarise, "explain X");
        assert!(result.contains("Synthesize"));
        assert!(result.contains("explain X"));
    }
}
