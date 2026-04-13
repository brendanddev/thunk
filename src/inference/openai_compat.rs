use std::io::{BufRead, BufReader};
use std::sync::mpsc::Sender;

use super::backend::{InferenceBackend, Message};
use crate::error::{ParamsError, Result};
use crate::events::InferenceEvent;
use crate::safety::{self, InspectionDecision};

/// OpenAI-compatible backend - works with any provider that implements the /v1/chat/completions endpoint
pub struct OpenAICompatBackend {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub provider_name: String,
}

impl OpenAICompatBackend {
    pub fn new(base_url: &str, api_key: &str, model: &str, provider_name: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            provider_name: provider_name.to_string(),
        }
    }

    /// Resolve the API key
    fn resolve_api_key(&self) -> String {
        if !self.api_key.is_empty() {
            return self.api_key.clone();
        }

        let env_var = if self.base_url.contains("groq.com") {
            "GROQ_API_KEY"
        } else if self.base_url.contains("openai.com") {
            "OPENAI_API_KEY"
        } else if self.base_url.contains("openrouter.ai") {
            "OPENROUTER_API_KEY"
        } else if self.base_url.contains("x.ai") {
            "XAI_API_KEY"
        } else {
            // Generic fallback
            "OPENAI_API_KEY"
        };

        std::env::var(env_var).unwrap_or_default()
    }

    /// Check if the backend is reachable and the API key is valid
    pub fn health_check(&self) -> Result<()> {
        let api_key = self.resolve_api_key();
        if api_key.is_empty() {
            return Err(ParamsError::Config(format!(
                "No API key configured for {}. Set api_key in config or the appropriate env var.",
                self.provider_name
            )));
        }

        let url = format!("{}/models", self.base_url);
        let (_, inspection) = safety::inspect_provider_request("openai_compat", &url, 0)?;
        if matches!(inspection.decision, InspectionDecision::Block) {
            return Err(ParamsError::Config(inspection.blocked_message()));
        }
        let response = ureq::get(&url)
            .set("Authorization", &format!("Bearer {api_key}"))
            .call()
            .map_err(|e| {
                ParamsError::Config(format!(
                    "{} not reachable at {}: {}",
                    self.provider_name, self.base_url, e
                ))
            })?;

        if response.status() == 200 || response.status() == 404 {
            Ok(())
        } else if response.status() == 401 {
            Err(ParamsError::Config(format!(
                "Invalid API key for {}. Check your config or env var.",
                self.provider_name
            )))
        } else {
            Err(ParamsError::Config(format!(
                "{} returned unexpected status {}",
                self.provider_name,
                response.status()
            )))
        }
    }
}

impl InferenceBackend for OpenAICompatBackend {
    fn name(&self) -> String {
        format!("{} · {}", self.provider_name, self.model)
    }

    fn generate(&self, messages: &[Message], tx: Sender<InferenceEvent>) -> Result<()> {
        let api_key = self.resolve_api_key();
        if api_key.is_empty() {
            return Err(ParamsError::Config(format!(
                "No API key for {}. Set api_key in [openai_compat] config or env var.",
                self.provider_name
            )));
        }

        // Build the OpenAI-format request body
        let messages_json: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": m.role,
                    "content": m.content,
                })
            })
            .collect();

        let body = serde_json::json!({
            "model": self.model,
            "messages": messages_json,
            "stream": true,
        });
        let body_text = body.to_string();

        let url = format!("{}/chat/completions", self.base_url);
        let (_, inspection) =
            safety::inspect_provider_request("openai_compat", &url, body_text.chars().count())?;
        if matches!(inspection.decision, InspectionDecision::Block) {
            return Err(ParamsError::Config(inspection.blocked_message()));
        }

        let mut request = ureq::post(&url)
            .set("Content-Type", "application/json")
            .set("Authorization", &format!("Bearer {api_key}"));

        // OpenRouter requires these headers to identify the app
        if self.base_url.contains("openrouter.ai") {
            request = request
                .set("HTTP-Referer", "https://github.com/brendanddev/params-cli")
                .set("X-Title", "params-cli");
        }

        let response = request.send_string(&body_text).map_err(|e| {
            ParamsError::Config(format!("{} request failed: {e}", self.provider_name))
        })?;

        // Read the Server-Sent Events (SSE) stream line by line
        let reader = BufReader::new(response.into_reader());

        for line in reader.lines() {
            let line = line.map_err(|e| ParamsError::Io(e))?;

            // Skip empty lines and event type lines (e.g. "event: message")
            if line.is_empty() || !line.starts_with("data: ") {
                continue;
            }

            let data = &line["data: ".len()..];

            // End of stream
            if data.trim() == "[DONE]" {
                break;
            }

            // Parse the JSON chunk
            let parsed: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Extract the token from choices[0].delta.content
            if let Some(content) = parsed["choices"][0]["delta"]["content"].as_str() {
                if !content.is_empty() {
                    if tx.send(InferenceEvent::Token(content.to_string())).is_err() {
                        // UI is gone — stop generating
                        return Ok(());
                    }
                }
            }

            // Check if this is the final chunk (finish_reason is set)
            if let Some(finish) = parsed["choices"][0]["finish_reason"].as_str() {
                if finish == "stop" || finish == "length" {
                    break;
                }
            }
        }

        Ok(())
    }
}
