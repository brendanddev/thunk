// src/inference/llama_cpp.rs
//
// The llama.cpp backend — runs a .gguf model directly in-process.

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::mpsc::Sender;

use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel, Special},
    sampling::LlamaSampler,
};

use crate::error::{ParamsError, Result};
use crate::events::InferenceEvent;
use super::backend::{InferenceBackend, Message};

/// Formats a conversation into a ChatML prompt string.
/// Qwen uses ChatML format — we build it manually.
pub fn format_messages(messages: &[Message]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role, msg.content
        ));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// The llama.cpp backend. Holds the loaded model in memory permanently
/// so it doesn't need to reload between messages.
pub struct LlamaCppBackend {
    model: LlamaModel,
    // We keep the backend alive here — it must outlive the model
    _backend: LlamaBackend,
    pub model_name: String,
    pub max_tokens: i32,
    pub temperature: f32,
}

impl LlamaCppBackend {
    /// Load a model from a .gguf file path. This is the slow step — it reads
    /// the weights from disk and uploads them to GPU memory.
    pub fn load(model_path: PathBuf, max_tokens: i32, temperature: f32) -> Result<Self> {
        let mut backend = LlamaBackend::init()
            .map_err(|e| ParamsError::Inference(e.to_string()))?;
        backend.void_logs();

        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
            .map_err(|e| ParamsError::Model(e.to_string()))?;

        // Extract just the filename for display (e.g. "qwen2.5-coder-7b-q4_k_m.gguf")
        let model_name = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(Self {
            model,
            _backend: backend,
            model_name,
            max_tokens,
            temperature,
        })
    }
}

impl InferenceBackend for LlamaCppBackend {
    fn name(&self) -> String {
        format!("llama.cpp · {}", self.model_name)
    }

    fn generate(&self, messages: &[Message], tx: Sender<InferenceEvent>) -> Result<()> {
        // Create a fresh context for each generation.
        // The model weights stay loaded — only the KV cache is reset.
        // This is much cheaper than reloading the model.
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(8192));

        let mut ctx = self.model
            .new_context(&self._backend, ctx_params)
            .map_err(|e| ParamsError::Inference(e.to_string()))?;

        let prompt = format_messages(messages);

        let tokens = self.model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|e| ParamsError::Inference(e.to_string()))?;

        let mut batch = LlamaBatch::new(tokens.len().max(2048), 1);
        let last_idx = (tokens.len() - 1) as i32;
        for (i, token) in tokens.iter().enumerate() {
            batch.add(*token, i as i32, &[0], i as i32 == last_idx)
                .map_err(|e| ParamsError::Inference(e.to_string()))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| ParamsError::Inference(e.to_string()))?;

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(self.temperature),
            LlamaSampler::dist(0),
        ]);

        let mut n_generated = 0;
        let mut current_pos = tokens.len() as i32;

        loop {
            let next_token = sampler.sample(&ctx, batch.n_tokens() - 1);

            if self.model.is_eog_token(next_token) {
                break;
            }

            #[allow(deprecated)]
            let token_bytes = self.model
                .token_to_bytes(next_token, Special::Plaintext)
                .map_err(|e| ParamsError::Inference(e.to_string()))?;

            let token_str = String::from_utf8_lossy(&token_bytes).to_string();

            // If the receiver is gone (UI closed), stop generating
            if tx.send(InferenceEvent::Token(token_str)).is_err() {
                return Ok(());
            }

            n_generated += 1;
            if n_generated >= self.max_tokens {
                break;
            }

            batch.clear();
            batch.add(next_token, current_pos, &[0], true)
                .map_err(|e| ParamsError::Inference(e.to_string()))?;
            current_pos += 1;

            ctx.decode(&mut batch)
                .map_err(|e| ParamsError::Inference(e.to_string()))?;
        }

        Ok(())
    }
}