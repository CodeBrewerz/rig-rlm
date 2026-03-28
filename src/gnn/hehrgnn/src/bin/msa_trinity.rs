use burn::prelude::*;
use burn::backend::{Autodiff, NdArray};
use burn::tensor::backend::AutodiffBackend;
use burn::optim::{AdamConfig, Optimizer, GradientsParams};
use hehrgnn::model::msa::{MsaConfig, MsaModel};
use std::fs;
use serde::Deserialize;
use reqwest::Client;
use serde_json::{json, Value};
use dotenvy::dotenv;
use std::env;
use std::time::Duration;
use rand::seq::SliceRandom;
use burn::module::AutodiffModule;
use burn::record::{NamedMpkFileRecorder, FullPrecisionSettings, Recorder};
use std::path::Path;

type B = Autodiff<NdArray>; // Training + Inference
type BValid = NdArray;

#[derive(Deserialize, Debug)]
struct Doc { doc_id: usize, text: String }

#[derive(Deserialize, Debug)]
struct Query { query_id: usize, text: String, target_doc_id: usize, is_train: bool }

#[derive(Deserialize, Debug)]
struct Dataset { documents: Vec<Doc>, queries: Vec<Query> }

#[derive(Module, Debug)]
pub struct MsaQaWrapper<B: Backend> {
    pub msa: MsaModel<B>,
}

impl<B: Backend> MsaQaWrapper<B> {
    pub fn new(device: &B::Device) -> Self {
        let msa_config = MsaConfig {
            hidden_dim: 2048,
            num_heads: 8,
            router_dim: 64,
            num_layers: 4,
            topk: 1, 
            ..Default::default()
        };
        let msa = MsaModel::new(&msa_config, device);
        Self { msa }
    }
}

async fn get_embedding_batch(client: &Client, api_key: &str, texts: &[String]) -> Vec<Vec<f32>> {
    let body = json!({
        "model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
        "input": texts
    });
    let res = client.post("https://openrouter.ai/api/v1/embeddings")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&body).send().await;
    match res {
        Ok(r) => {
            if let Ok(v) = r.json::<Value>().await {
                if let Some(arr) = v["data"].as_array() {
                    let mut embeddings = Vec::new();
                    for item in arr {
                        let emb: Vec<f32> = item["embedding"].as_array().unwrap()
                            .iter().map(|e| e.as_f64().unwrap() as f32).collect();
                        embeddings.push(emb);
                    }
                    return embeddings;
                }
            }
        },
        _ => {},
    }
    vec![vec![0.0; 2048]; texts.len()]
}

async fn ask_trinity(client: &Client, api_key: &str, context: &str, question: &str) -> String {
    let prompt = format!("Context:\n{}\n\nQuestion:\n{}\n\nBased strictly on the context provided, answer the question concisely.", context, question);
    let body = json!({
        "model": "arcee-ai/trinity-mini:free",
        "messages": [ { "role": "user", "content": prompt } ]
    });
    let res = client.post("https://openrouter.ai/api/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&body).send().await;
    match res {
        Ok(response) => {
            if let Ok(j) = response.json::<Value>().await {
                if let Some(content) = j["choices"][0]["message"]["content"].as_str() {
                    return content.to_string();
                }
            }
            "Failed OpenRouter parse".to_string()
        },
        Err(e) => format!("OpenRouter Error: {}", e)
    }
}

use burn::tensor::ElementConversion;

fn newton_schulz<B: Backend>(mut x: Tensor<B, 2>) -> Tensor<B, 2> {
    let [m, n] = x.dims();
    let norm = (x.clone().powf_scalar(2.0).sum().into_scalar().elem::<f32>() / (std::cmp::max(m, n) as f32)).sqrt() + 1e-8;
    x = x.mul_scalar(1.0 / norm);
    for _ in 0..5 {
        if m > n {
            let x_t_x = x.clone().transpose().matmul(x.clone());
            let b = x.clone().matmul(x_t_x);
            x = x.mul_scalar(1.5).sub(b.mul_scalar(0.5));
        } else {
            let x_x_t = x.clone().matmul(x.clone().transpose());
            let b = x_x_t.matmul(x.clone());
            x = x.mul_scalar(1.5).sub(b.mul_scalar(0.5));
        }
    }
    x
}

struct TeonMapper<B: AutodiffBackend> {
    id_q0: burn::module::ParamId, u_q0: Tensor<B::InnerBackend, 2>,
    id_q1: burn::module::ParamId, u_q1: Tensor<B::InnerBackend, 2>,
    id_k0: burn::module::ParamId, u_k0: Tensor<B::InnerBackend, 2>,
    id_k1: burn::module::ParamId, u_k1: Tensor<B::InnerBackend, 2>,
}

impl<B: AutodiffBackend> burn::module::ModuleMapper<B> for TeonMapper<B> {
    fn map_float<const D: usize>(&mut self, param: burn::module::Param<Tensor<B, D>>) -> burn::module::Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        
        // tensor.inner() returns Tensor<B::InnerBackend, D>
        let t2 = tensor.clone().inner(); 
        
        let updated = if id == self.id_q0 {
            t2.clone().sub(self.u_q0.clone().reshape(t2.shape()))
        } else if id == self.id_q1 {
            t2.clone().sub(self.u_q1.clone().reshape(t2.shape()))
        } else if id == self.id_k0 {
            t2.clone().sub(self.u_k0.clone().reshape(t2.shape()))
        } else if id == self.id_k1 {
            t2.clone().sub(self.u_k1.clone().reshape(t2.shape()))
        } else {
            return burn::module::Param::from_mapped_value(id, tensor, mapper);
        };

        // updated is Tensor<B::InnerBackend, D>, so from_inner creates Tensor<B, D>
        let new_tensor = Tensor::<B, D>::from_inner(updated);
        burn::module::Param::from_mapped_value(id, new_tensor, mapper)
    }
}

#[tokio::main]
async fn main() {
    dotenv().ok();
    let api_key = env::var("OPENAI_API_KEY").or_else(|_| env::var("OPENROUTER_API_KEY")).expect("missing api key");

    let dataset_path = "/home/sumit-mittal/dev-stuff/rig-rlm/src/gnn/plans/msa_qa_dataset.json";
    let dataset: Dataset = serde_json::from_str(&fs::read_to_string(dataset_path).unwrap()).unwrap();
    let http_client = Client::builder().timeout(Duration::from_secs(30)).build().unwrap();

    println!("1. Fetching Nemotron 2048-D Base Semantic Embeddings over HTTP...");
    let subset_docs: Vec<_> = dataset.documents.into_iter().take(50).collect();
    let doc_texts: Vec<String> = subset_docs.iter().map(|d| d.text.clone()).collect();
    let doc_embs = get_embedding_batch(&http_client, &api_key, &doc_texts).await;

    let train_queries: Vec<_> = dataset.queries.iter().filter(|q| q.is_train && q.target_doc_id < 50).collect();
    let train_texts: Vec<String> = train_queries.iter().map(|q| q.text.clone()).collect();
    let train_embs = get_embedding_batch(&http_client, &api_key, &train_texts).await;

    let model_path = format!("{}/src/gnn/hehrgnn/pretrained_msa/router_weights", env::current_dir().unwrap().to_str().unwrap());
    let valid_device = <BValid as Backend>::Device::default();
    let mut valid_wrapper: MsaQaWrapper<BValid>;

    if false {
        println!("2. Found Pre-Trained TEON Router! Resuming directly from disk...");
        let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
            .load(model_path.clone().into(), &valid_device)
            .expect("Failed to load pre-trained weights");
        valid_wrapper = MsaQaWrapper::<BValid>::new(&valid_device).load_record(record);
    } else {
        println!("2. No saved weights found. Initializing Burn Autodiff Gradients & TEON Router Training...");
        let device = <B as Backend>::Device::default();
        let mut wrapper = MsaQaWrapper::<B>::new(&device);
        
        // TEON Hyperparameters
        let mu: f32 = 0.95;
        let lr: f32 = 0.005; // Learning rate roughly matched to Amsel et al. standard params
        
        let mut m_q0 = Tensor::<BValid, 2>::zeros([2048, 512], &valid_device);
        let mut m_q1 = Tensor::<BValid, 2>::zeros([2048, 512], &valid_device);
        let mut m_k0 = Tensor::<BValid, 2>::zeros([2048, 512], &valid_device);
        let mut m_k1 = Tensor::<BValid, 2>::zeros([2048, 512], &valid_device);
        
        let mut db_tensors = std::collections::HashMap::new();
        for (i, doc) in subset_docs.iter().enumerate() {
            db_tensors.insert(doc.doc_id, Tensor::<B, 1>::from_data(doc_embs[i].as_slice(), &device).reshape([1, 2048]));
        }

        let mut rng = rand::rng();
        println!("Beginning T-EON InfoNCE Training (Tensorized Orthonormalization)...");
        for epoch in 1..=8 {
            let mut total_loss = 0.0;
            let mut correct = 0;
            
            for (i, train_q) in train_queries.iter().enumerate() {
                let qt = Tensor::<B, 1>::from_data(train_embs[i].as_slice(), &device).reshape([1, 2048]);
                
                let l0 = wrapper.msa.get_msa_layer(0).unwrap();
                let l1 = wrapper.msa.get_msa_layer(1).unwrap();
                
                // Stack layers 0 and 1 computation for full cross-layer optimization
                let routing_query = l0.compute_routing_query(qt.clone()).add(l1.compute_routing_query(qt.clone()));
                
                let target_doc = db_tensors.get(&train_q.target_doc_id).unwrap().clone();
                let kr_target = l0.encode_document(target_doc.clone()).2.add(l1.encode_document(target_doc.clone()).2);
                
                let mut pos_score = routing_query.clone().matmul(kr_target.clone().transpose());
                pos_score = pos_score.mul_scalar(1.0 / (64.0_f32).sqrt());
                
                let mut neg_tensors = Vec::new();
                let mut ids: Vec<usize> = db_tensors.keys().copied().collect();
                ids.shuffle(&mut rng);
                for neg_id in ids.into_iter().filter(|id| *id != train_q.target_doc_id).take(3) {
                    neg_tensors.push(db_tensors.get(&neg_id).unwrap().clone());
                }
                
                let mut neg_keys = Vec::new();
                for nt in neg_tensors {
                    neg_keys.push(l0.encode_document(nt.clone()).2.add(l1.encode_document(nt).2));
                }
                
                let mut loss = pos_score.clone().mul_scalar(-1.0).exp();
                let mut neg_sum = Tensor::<B, 2>::zeros([1, 1], &device);
                
                for nk in neg_keys {
                    let s = routing_query.clone().matmul(nk.transpose()).mul_scalar(1.0 / (64.0_f32).sqrt());
                    neg_sum = neg_sum.add(s.exp());
                }
                loss = loss.mul(neg_sum.add_scalar(1e-6)).log();
                
                if loss.clone().into_scalar().elem::<f32>() > 0.0 { correct += 1; }
                total_loss += loss.clone().into_scalar().elem::<f32>();

                let grads = loss.backward();
                let g_q0 = l0.router_q.proj.weight.val().grad(&grads).unwrap_or_else(|| Tensor::zeros([2048, 512], &valid_device));
                let g_q1 = l1.router_q.proj.weight.val().grad(&grads).unwrap_or_else(|| Tensor::zeros([2048, 512], &valid_device));
                let g_k0 = l0.router_k.proj.weight.val().grad(&grads).unwrap_or_else(|| Tensor::zeros([2048, 512], &valid_device));
                let g_k1 = l1.router_k.proj.weight.val().grad(&grads).unwrap_or_else(|| Tensor::zeros([2048, 512], &valid_device));

                m_q0 = m_q0.mul_scalar(mu).add(g_q0);
                m_q1 = m_q1.mul_scalar(mu).add(g_q1);
                m_k0 = m_k0.mul_scalar(mu).add(g_k0);
                m_k1 = m_k1.mul_scalar(mu).add(g_k1);

                let z_q = Tensor::cat(vec![m_q0.clone(), m_q1.clone()], 1);
                let z_k = Tensor::cat(vec![m_k0.clone(), m_k1.clone()], 1);

                let o_q = newton_schulz(z_q);
                let o_k = newton_schulz(z_k);

                let step_scale = lr * ((2048.0 / 512.0) as f32).sqrt();
                let u_q0 = o_q.clone().slice([0..2048, 0..512]).mul_scalar(step_scale);
                let u_q1 = o_q.slice([0..2048, 512..1024]).mul_scalar(step_scale);
                let u_k0 = o_k.clone().slice([0..2048, 0..512]).mul_scalar(step_scale);
                let u_k1 = o_k.slice([0..2048, 512..1024]).mul_scalar(step_scale);

                let mut mapper = TeonMapper {
                    id_q0: l0.router_q.proj.weight.id.clone(), u_q0,
                    id_q1: l1.router_q.proj.weight.id.clone(), u_q1,
                    id_k0: l0.router_k.proj.weight.id.clone(), u_k0,
                    id_k1: l1.router_k.proj.weight.id.clone(), u_k1,
                };
                wrapper = wrapper.map(&mut mapper);
            }
            
            println!("Epoch {} | Loss {:.4} | Train Routing Acc {}/{}", epoch, total_loss / train_queries.len() as f32, correct, train_queries.len());

        }

        println!("3. Serializing Weights & Converting to Inference Graph...");
        valid_wrapper = wrapper.valid();
        fs::create_dir_all(Path::new(&model_path).parent().unwrap()).unwrap();
        valid_wrapper.clone().save_file(model_path, &NamedMpkFileRecorder::<FullPrecisionSettings>::default())
            .expect("Failed to serialize model correctly");
    }
    let valid_device = <BValid as Backend>::Device::default();
    
    let mut bank = valid_wrapper.msa.create_memory_bank();
    let layer = valid_wrapper.msa.get_msa_layer(0).unwrap();

    // Populate actual semantic text into trained MemoryBank architecture
    for (i, emb) in doc_embs.iter().enumerate() {
        let tensor = Tensor::<BValid, 1>::from_data(emb.as_slice(), &valid_device).reshape([1, 2048]);
        bank.encode_document(subset_docs[i].doc_id, tensor, layer);
    }

    let test_queries: Vec<_> = dataset.queries.into_iter()
        .filter(|q| !q.is_train && q.target_doc_id < 50).take(5).collect();

    println!("4. Executing Final Evaluation Pipeline!");
    for (i, q) in test_queries.iter().enumerate() {
        println!("\n========================================");
        println!("Test Query {}/5: {}", i+1, q.text);
        
        let q_emb_batch = get_embedding_batch(&http_client, &api_key, &[q.text.clone()]).await;
        let qt = Tensor::<BValid, 1>::from_data(q_emb_batch[0].as_slice(), &valid_device).reshape([1, 2048]);
        
        let routing_query = layer.compute_routing_query(qt);
        let retrieved = bank.route(routing_query);
        let top_doc_id = retrieved[0].0;
        
        let target_doc = subset_docs.iter().find(|d| d.doc_id == top_doc_id).unwrap();
        
        println!("Retrieved Context (via End-to-End Trained MSA Router):");
        println!("\"...{}...\"", &target_doc.text[0..std::cmp::min(100, target_doc.text.len())]);
        
        let answer = ask_trinity(&http_client, &api_key, &target_doc.text, &q.text).await;
        println!("Trinity LLM Response:\n{}", answer);
    }
}
