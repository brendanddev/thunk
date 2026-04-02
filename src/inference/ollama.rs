// src/inference/ollama.rs
//
// The Ollama backend — talks to a running Ollama server via HTTP.

use std::io::{BufRead, BufReader};
use std::sync::mpsc::Sender;

use crate::error::{ParamsError, Result};
use crate::events::InferenceEvent;
use super::backend::{InferenceBackend, Message};

/// The Ollama backend. Makes HTTP requests to an Ollama server.
pub struct OllamaBackend {
    pub base_url: String,
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