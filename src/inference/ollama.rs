// src/inference/ollama.rs
//
// The Ollama backend — talks to a running Ollama server via HTTP.
//
// Ollama is a tool that runs models as a persistent background server.
// It keeps the model loaded in memory permanently, so there's no reload
// cost between messages. It also supports running on a remote machine,
// which means you can run a 14B model on your desktop and connect from
// your M2 Air or ThinkPad over your local network.
//
// Ollama API docs: https://github.com/ollama/ollama/blob/main/docs/api.md
//
// Advantages:
//   - Model stays loaded permanently — no startup freeze per session
//   - Can point at a remote machine (e.g. desktop with better GPU)
//   - Simple HTTP API, no C++ compilation required
//   - Easy to switch models via config
//
// Disadvantages:
//   - Requires Ollama to be installed and running separately
//   - Needs network access (even if just localhost)

use std::io::{BufRead, BufReader};
use std::sync::mpsc::Sender;

use crate::error::{ParamsError, Result};
use crate::events::InferenceEvent;
use super::backend::{InferenceBackend, Message};

/// The Ollama backend. Makes HTTP requests to an Ollama server.
pub struct OllamaBackend {
    /// Base URL of the Ollama server, e.g. "http://localhost:11434"
    /// or "http://192.168.1.100:11434" for a remote machine.
    pub base_url: String,

    /// The model to use, e.g. "qwen2.5-coder:14b" or "qwen2.5-coder:7b"
    /// Must be pulled in Ollama first: `ollama pull qwen2.5-coder:14b`
    pub model: String,
}

impl OllamaBackend {
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
        }
    }

    /// Check if the Ollama server is reachable.
    /// Returns Ok(()) if healthy, Err if not running or unreachable.
    pub fn health_check(&self) -> Result<()> {
        let url = format!("{}/api/tags", self.base_url);
        let response = ureq::get(&url)
            .call()
            .map_err(|e| ParamsError::Config(format!("Ollama not reachable at {}: {}", self.base_url, e)))?;

        if response.status() == 200 {
            Ok(())
        } else {
            Err(ParamsError::Config(format!(
                "Ollama returned status {} — is it running?",
                response.status()
            )))
        }
    }
}

impl InferenceBackend for OllamaBackend {
    fn name(&self) -> String {
        format!("ollama · {}", self.model)
    }

    fn generate(&self, messages: &[Message], tx: Sender<InferenceEvent>) -> Result<()> {
        // Build the request body.
        // Ollama's /api/chat endpoint accepts a list of messages with roles,
        // exactly like OpenAI's chat format. We convert our Message type to JSON.
        let messages_json: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| serde_json::json!({
                "role": m.role,
                "content": m.content,
            }))
            .collect();

        let body = serde_json::json!({
            "model": self.model,
            "messages": messages_json,
            // stream: true means Ollama sends tokens as they're generated
            // rather than waiting for the full response
            "stream": true,
        });

        let url = format!("{}/api/chat", self.base_url);

        // Make the HTTP POST request.
        // ureq is a simple blocking HTTP client — good fit for a background thread.
        let response = ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_string(&body.to_string())
            .map_err(|e| ParamsError::Config(format!("Ollama request failed: {e}")))?;

        // Read the streaming response line by line.
        // Ollama sends one JSON object per line (newline-delimited JSON / NDJSON).
        // Each line looks like:
        //   {"model":"qwen...","message":{"role":"assistant","content":"Hello"},"done":false}
        // The last line has "done": true.
        let reader = BufReader::new(response.into_reader());

        for line in reader.lines() {
            let line = line.map_err(|e| ParamsError::Io(e))?;
            if line.is_empty() {
                continue;
            }

            // Parse the JSON line
            let parsed: serde_json::Value = serde_json::from_str(&line)
                .map_err(|e| ParamsError::Inference(format!("JSON parse error: {e}")))?;

            // Extract the token from the message content field
            if let Some(content) = parsed["message"]["content"].as_str() {
                if !content.is_empty() {
                    if tx.send(InferenceEvent::Token(content.to_string())).is_err() {
                        // UI is gone, stop generating
                        return Ok(());
                    }
                }
            }

            // Stop when Ollama signals it's done
            if parsed["done"].as_bool().unwrap_or(false) {
                break;
            }
        }

        Ok(())
    }
}