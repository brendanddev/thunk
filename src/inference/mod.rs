// src/inference/mod.rs
//
// Public API for the inference module.
//
// model_thread loads the backend once and handles prompts in a loop.
// After each generation it checks for tool calls in the response —
// if found, it runs them and continues the conversation automatically.

mod backend;
mod llama_cpp;
mod ollama;
mod openai_compat;

pub use backend::{InferenceBackend, Message, SYSTEM_PROMPT, system_prompt_with_tools};
pub use llama_cpp::LlamaCppBackend;
pub use ollama::OllamaBackend;
pub use openai_compat::OpenAICompatBackend;

use std::sync::mpsc::{Receiver, Sender};
use crate::events::InferenceEvent;
use crate::config;
use crate::tools::ToolRegistry;

/// Persistent model thread — loads the backend once, handles prompts in a loop.
/// After each response it checks for tool calls and runs a follow-up if needed.
pub fn model_thread(
    prompt_rx: Receiver<Vec<Message>>,
    token_tx: Sender<InferenceEvent>,
) {
    let cfg = match config::load() {
        Ok(c) => c,
        Err(e) => {
            let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
            return;
        }
    };

    let backend: Box<dyn InferenceBackend> = match cfg.backend.as_str() {
        "ollama" => {
            let b = OllamaBackend::new(&cfg.ollama.url, &cfg.ollama.model);
            if let Err(e) = b.health_check() {
                let _ = token_tx.send(InferenceEvent::Error(
                    format!("Ollama failed: {e} — falling back to llama.cpp")
                ));
                match load_llama_cpp_backend(&cfg) {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                        return;
                    }
                }
            } else {
                Box::new(b)
            }
        }
        "openai_compat" => {
            let provider_name = cfg.openai_compat.resolved_provider_name();
            let b = OpenAICompatBackend::new(
                &cfg.openai_compat.url,
                &cfg.openai_compat.api_key,
                &cfg.openai_compat.model,
                &provider_name,
            );
            if let Err(e) = b.health_check() {
                let _ = token_tx.send(InferenceEvent::Error(
                    format!("{provider_name} failed: {e} — check your API key")
                ));
                return;
            }
            Box::new(b)
        }
        _ => {
            match load_llama_cpp_backend(&cfg) {
                Ok(b) => b,
                Err(e) => {
                    let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                    return;
                }
            }
        }
    };

    let _ = token_tx.send(InferenceEvent::Ready);
    let _ = token_tx.send(InferenceEvent::BackendName(backend.name()));

    // Build the tool registry — same instance reused across all prompts
    let tools = ToolRegistry::default();

    while let Ok(mut messages) = prompt_rx.recv() {
        // Inject tool descriptions into the system prompt if present
        if let Some(first) = messages.first_mut() {
            if first.role == "system" {
                first.content = system_prompt_with_tools(&tools.tool_descriptions());
            }
        }

        // Run generation — collect full response for tool call detection
        let response = run_and_collect(&*backend, &messages, token_tx.clone());

        match response {
            Err(e) => {
                let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
            }
            Ok(full_response) => {
                // Check if the model used any tools
                let tool_results = tools.execute_tool_calls(&full_response);

                if let Some(result_msg) = ToolRegistry::format_results(&tool_results) {
                    // Notify the UI that tool calls are being processed
                    let _ = token_tx.send(InferenceEvent::ToolCall(
                        tool_results.iter()
                            .map(|r| format!("{}({})", r.tool_name, r.argument))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));

                    // Add the assistant's response and tool results to history
                    messages.push(Message::assistant(&full_response));
                    messages.push(Message::user(&result_msg));

                    // Run a follow-up generation with the tool results injected
                    match run_and_collect(&*backend, &messages, token_tx.clone()) {
                        Ok(_) => {}
                        Err(e) => {
                            let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                        }
                    }
                }

                let _ = token_tx.send(InferenceEvent::Done);
            }
        }
    }
}

/// Run generation and collect the full response as a String,
/// while also streaming tokens to the UI via the channel.
fn run_and_collect(
    backend: &dyn InferenceBackend,
    messages: &[Message],
    token_tx: Sender<InferenceEvent>,
) -> crate::error::Result<String> {
    use std::sync::{Arc, Mutex};

    // We need to both stream tokens to the UI and collect them locally.
    // Use a shared buffer that both the channel sender and collector write to.
    let buffer = Arc::new(Mutex::new(String::new()));
    let buffer_clone = buffer.clone();

    // Wrap the token_tx to also write to buffer
    let (intercept_tx, intercept_rx) = std::sync::mpsc::channel::<InferenceEvent>();

    // Spawn a relay thread that forwards tokens to the UI and collects them
    let relay_token_tx = token_tx.clone();
    std::thread::spawn(move || {
        while let Ok(event) = intercept_rx.recv() {
            match &event {
                InferenceEvent::Token(t) => {
                    if let Ok(mut buf) = buffer_clone.lock() {
                        buf.push_str(t);
                    }
                    let _ = relay_token_tx.send(event);
                }
                _ => {
                    // Don't forward Done/Error — model_thread handles those
                }
            }
        }
    });

    backend.generate(messages, intercept_tx)?;

    // Give the relay thread a moment to finish flushing
    std::thread::sleep(std::time::Duration::from_millis(10));

    let result = buffer.lock()
        .map(|b| b.clone())
        .unwrap_or_default();

    Ok(result)
}

fn load_llama_cpp_backend(cfg: &config::Config) -> crate::error::Result<Box<dyn InferenceBackend>> {
    let model_path = match &cfg.llama_cpp.model_path {
        Some(p) => p.clone(),
        None => config::find_model()?,
    };
    let backend = LlamaCppBackend::load(
        model_path,
        cfg.generation.max_tokens,
        cfg.generation.temperature,
    )?;
    Ok(Box::new(backend))
}