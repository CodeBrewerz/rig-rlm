use corophage::prelude::*;

#[effect(String)]
pub struct AskLLM {
    pub prompt: String,
}

#[effectful(AskLLM, send)]
pub fn evaluate_leaf(prompt: String) -> String {
    eprintln!("    [Effect] Yielding LLM evaluation for chunk (len: {})", prompt.len());
    let answer = yield_!(AskLLM { prompt });
    answer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ask_llm_effect() {
        let result = evaluate_leaf("Hello".to_string())
            .handle(async |AskLLM { prompt }| {
                eprintln!("    [Handler] Intercepted AskLLM effect for: {}", prompt);
                let answer = format!("Mocked response to -> {}", prompt);
                Control::resume(answer)
            })
            .run()
            .await;

        assert_eq!(result, Ok("Mocked response to -> Hello".to_string()));
    }
}
