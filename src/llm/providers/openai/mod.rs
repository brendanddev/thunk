use std::env;
use std::io::BufRead;

use serde_json::{json, Value};

use crate::app::config::OpenAiConfig;
use crate::app::{AppError, Result};
use crate::llm::backend::{
    BackendCapabilities, BackendEvent, BackendStatus, GenerateRequest, ModelBackend,
};

pub struct OpenAiBackend {
    config: OpenAiConfig,
    display_name: String,
}

impl OpenAiBackend {
    pub fn new(config: OpenAiConfig) -> Self {
        let display_name = format!("openai/{}", config.model);
        Self { config, display_name }
    }
}

impl ModelBackend for OpenAiBackend {
    fn name(&self) -> &str {
        &self.display_name
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            context_window_tokens: None,
            max_output_tokens: Some(self.config.max_tokens),
        }
    }

    fn generate(
        &mut self,
        request: GenerateRequest,
        on_event: &mut dyn FnMut(BackendEvent),
    ) -> Result<()> {
        if self.config.model.is_empty() {
            return Err(AppError::Config(
                "openai.model must not be empty".to_string(),
            ));
        }

        let api_key = env::var("OPENAI_API_KEY").map_err(|_| {
            AppError::Config("OPENAI_API_KEY environment variable is not set".to_string())
        })?;

        let messages: Vec<Value> = request
            .messages
            .iter()
            .map(|m| json!({ "role": m.role.as_str(), "content": m.content }))
            .collect();

        let body = json!({
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": true,
        });

        let url = format!("{}/chat/completions", self.config.base_url);

        let response = ureq::post(&url)
            .set("Authorization", &format!("Bearer {api_key}"))
            .set("Content-Type", "application/json")
            .send_string(&body.to_string())
            .map_err(|e| AppError::Runtime(format!("OpenAI request failed: {e}")))?;

        on_event(BackendEvent::StatusChanged(BackendStatus::Generating));

        let reader = std::io::BufReader::new(response.into_reader());
        for line in reader.lines() {
            let line = line.map_err(|e| AppError::Runtime(format!("SSE read error: {e}")))?;

            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };

            if data == "[DONE]" {
                break;
            }

            let Ok(val) = serde_json::from_str::<Value>(data) else {
                continue;
            };

            if let Some(content) = val["choices"][0]["delta"]["content"].as_str() {
                if !content.is_empty() {
                    on_event(BackendEvent::TextDelta(content.to_string()));
                }
            }
        }

        on_event(BackendEvent::Finished);
        Ok(())
    }
}
