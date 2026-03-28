use burn::prelude::*;
use burn::optim::{AdamWConfig, Optimizer};
use burn::backend::{Autodiff, NdArray};
use hehrgnn::model::msa::{MsaConfig, MsaModel};
use hehrgnn::model::msa::loss::aux_routing_loss;
use hehrgnn::model::msa::memory_bank::MemoryBank;
use burn::module::AutodiffModule;
use std::fs;
use serde::Deserialize;
use burn::nn;

type B = Autodiff<NdArray>;

#[derive(Deserialize, Debug)]
struct Doc {
    doc_id: usize,
    text: String,
}

#[derive(Deserialize, Debug)]
struct Query {
    query_id: usize,
    text: String,
    target_doc_id: usize,
    is_train: bool,
}

#[derive(Deserialize, Debug)]
struct Dataset {
    documents: Vec<Doc>,
    queries: Vec<Query>,
}

#[derive(Module, Debug)]
pub struct MsaQaModel<B: Backend> {
    embed: nn::Embedding<B>,
    msa: MsaModel<B>,
}

impl<B: Backend> MsaQaModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let embed = nn::EmbeddingConfig::new(256, 128).init(device);
        let msa_config = MsaConfig {
            hidden_dim: 128,
            num_heads: 4,
            router_dim: 16,
            chunk_size: 16,
            num_layers: 2,
            ffn_ratio: 2,
            topk: 1,
            ..Default::default()
        };
        let msa = MsaModel::new(&msa_config, device);
        Self { embed, msa }
    }

    /// Forward pass returning only routing scores for auxiliary loss training.
    pub fn compute_routing_scores(
        &self,
        query_tokens: Tensor<B, 2, Int>,
        doc_tokens: Vec<Tensor<B, 2, Int>>,
    ) -> Vec<Tensor<B, 1>> {
        // Embed query
        let query_len = query_tokens.dims()[1];
        let query_emb = self.embed.forward(query_tokens).reshape([query_len, 128]); // unbatched [q_len, 128]

        let layer = self.msa.get_msa_layer(0).unwrap();
        let query_routing = layer.compute_routing_query(query_emb); // [q_len, heads*head_dim]

        let mut doc_scores = Vec::new();

        // Process each document precisely to calculate similarity
        for dt in doc_tokens {
            let doc_len = dt.dims()[1];
            let doc_emb = self.embed.forward(dt).reshape([doc_len, 128]);
            
            // Encode doc using router projector
            let (_, _, kr_bar) = layer.encode_document(doc_emb); // [num_chunks, router_dim]
            
            // Score query vs document chunks (using scoring logic from MSA)
            // Simplified computation for raw tensors to keep grads
            let mut chunk_docs = vec![(0, kr_bar)];
            let scores_batch = hehrgnn::model::msa::scoring::compute_document_scores(
                query_routing.clone(),
                &chunk_docs,
                self.msa.num_heads,
            );
            
            // Recompute similarity entirely in tensor ops to preserve gradients, 
            // since `compute_document_scores` returns f32 values directly.
            // Let's implement full tensor-based score strictly for training.
            let [num_chunks, _] = chunk_docs[0].1.dims();
            let mut head_scores_sum = Tensor::<B, 2>::zeros([query_len, num_chunks], &query_routing.device());
            let head_dim = 16;
            
            for h in 0..self.msa.num_heads {
                let q_h = query_routing.clone().slice([0..query_len, h*head_dim..(h+1)*head_dim]);
                let k_h = chunk_docs[0].1.clone().slice([0..num_chunks, h*head_dim..(h+1)*head_dim]);
                let sim = hehrgnn::model::msa::scoring::cosine_similarity_matrix(q_h, k_h);
                head_scores_sum = head_scores_sum + sim;
            }
            let mean_sim = head_scores_sum / (self.msa.num_heads as f32);
            let doc_score = mean_sim.max_dim(0).max(); // scalar tensor [1]
            doc_scores.push(doc_score);
        }

        doc_scores
    }
}

fn text_to_tensor<B: Backend>(text: &str, device: &B::Device) -> Tensor<B, 2, Int> {
    let bytes: Vec<i32> = text.bytes().map(|b| b as i32).take(200).collect();
    // at least length 1 to avoid empty tensors
    let mut final_bytes = bytes;
    if final_bytes.is_empty() {
        final_bytes.push(0);
    }
    let seq_len = final_bytes.len();
    Tensor::<B, 1, Int>::from_data(final_bytes.as_slice(), device).reshape([1, seq_len])
}

fn main() {
    let dataset_path = "/home/sumit-mittal/dev-stuff/rig-rlm/src/gnn/plans/msa_qa_dataset.json";
    let contents = fs::read_to_string(dataset_path).expect("Failed to read dataset");
    let dataset: Dataset = serde_json::from_str(&contents).expect("Failed to parse JSON");

    let device = <B as Backend>::Device::default();
    let mut model = MsaQaModel::<B>::new(&device);
    let mut optim = AdamWConfig::new().init();

    println!("Starting Router Network Training...");
    
    // Simplification: train on 20 random queries to demonstrate functionality
    let train_queries: Vec<&Query> = dataset.queries.iter().filter(|q| q.is_train).take(50).collect();
    
    // Build a quick document lookup
    let docs: std::collections::HashMap<usize, &Doc> = dataset.documents.iter().map(|d| (d.doc_id, d)).collect();

    for epoch in 1..=5 {
        let mut total_loss = 0.0;
        let mut correct = 0;
        
        for num_q in 0..train_queries.len() {
            let query = train_queries[num_q];
            let q_tensor = text_to_tensor::<B>(&query.text, &device);
            
            let pos_doc = docs.get(&query.target_doc_id).unwrap();
            let pos_tensor = text_to_tensor::<B>(&pos_doc.text, &device);
            
            // Sample 3 random negative documents
            let mut neg_tensors = Vec::new();
            for _ in 0..3 {
                let random_id = dataset.documents[rand::Rng::gen_range(&mut rand::thread_rng(), 0..dataset.documents.len())].doc_id;
                let neg_doc = docs.get(&random_id).unwrap();
                neg_tensors.push(text_to_tensor::<B>(&neg_doc.text, &device));
            }
            
            // Forward pass
            let mut all_doc_tensors = vec![pos_tensor];
            all_doc_tensors.extend(neg_tensors);

            let scores = model.compute_routing_scores(q_tensor, all_doc_tensors); // [4] scalar tensors
            
            let pos_score = scores[0].clone().reshape([1]);
            let neg_scores_list: Vec<Tensor<B, 2>> = scores[1..].iter()
                .map(|s| s.clone().reshape([1, 1]))
                .collect();
            let neg_scores = Tensor::cat(neg_scores_list, 1); // [1, 3]
            
            // Calculate contrastive loss
            let loss = aux_routing_loss(pos_score.clone(), neg_scores.clone(), 0.07);
            
            // Check accuracy 
            let pos_val = pos_score.into_data().as_slice::<f32>().unwrap()[0];
            let neg_vals = neg_scores.into_data().as_slice::<f32>().unwrap().to_vec();
            if pos_val > neg_vals.into_iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() {
                correct += 1;
            }
            
            total_loss += loss.clone().into_data().as_slice::<f32>().unwrap()[0];
            
            // Backward
            let grads = loss.backward();
            let grads_params = burn::optim::GradientsParams::from_grads(grads, &model);
            model = optim.step(1e-3, model, grads_params);
        }
        
        println!("Epoch {} | Loss {:.4} | Train Acc {}/{}", epoch, total_loss / train_queries.len() as f32, correct, train_queries.len());
    }
    
    println!("\nBuilding Memory Bank and evaluating on Test set...");
    let test_queries: Vec<&Query> = dataset.queries.iter().filter(|q| !q.is_train).take(50).collect();
    
    // Evaluation using Non-Autodiff Inner Model (convert gradients off)
    let valid_model = model.valid(); 
    let mut bank = valid_model.msa.create_memory_bank();
    let layer = valid_model.msa.get_msa_layer(0).unwrap();
    
    for doc in dataset.documents.iter().take(100) {  // Test on first 100 documents corpus
        let db = <NdArray as Backend>::Device::default();
        let bytes: Vec<i32> = doc.text.bytes().map(|b| b as i32).take(200).collect();
        let dt = Tensor::<NdArray, 1, Int>::from_data(bytes.as_slice(), &db).reshape([1, bytes.len()]);
        let doc_emb = valid_model.embed.forward(dt).reshape([bytes.len(), 128]);
        bank.encode_document(doc.doc_id, doc_emb, layer);
    }
    
    let mut test_correct = 0;
    for q in &test_queries {
        if q.target_doc_id >= 100 { continue; } // only evaluated up to doc 100 for speed
        let db = <NdArray as Backend>::Device::default();
        let bytes: Vec<i32> = q.text.bytes().map(|b| b as i32).collect();
        let qt = Tensor::<NdArray, 1, Int>::from_data(bytes.as_slice(), &db).reshape([1, bytes.len()]);
        
        let query_emb = valid_model.embed.forward(qt).reshape([bytes.len(), 128]);
        let routing_query = layer.compute_routing_query(query_emb);
        
        let retrieved = bank.route(routing_query);
        let top_id = retrieved[0].0;
        
        if top_id == q.target_doc_id {
            test_correct += 1;
        }
    }
    
    // Limit to actual test queries inside the 100 doc boundary
    let valid_test: Vec<_> = test_queries.iter().filter(|q| q.target_doc_id < 100).collect();
    println!("Test Retrieval Accuracy: {}/{} ({:.1}%)", test_correct, valid_test.len(), (test_correct as f32 / valid_test.len() as f32) * 100.0);
}
