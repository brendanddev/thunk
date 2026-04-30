mod native;
mod prompt;

use std::path::PathBuf;

use crate::app::config::LlamaCppConfig;
use crate::app::{AppError, Result};
use crate::llm::backend::{
    BackendCapabilities, BackendEvent, BackendStatus, GenerateRequest, ModelBackend,
};

use native::{load_model, run_generation, LoadedLlama};
use prompt::format_messages;

/// The llama.cpp backed implementation of the ModelBackend.
/// It lazy-loads the model on first use, formats prompts into ChatML, and runs generation
/// while streaming events back to the runtime.
pub struct LlamaCppBackend {
    config: LlamaCppConfig,
    display_name: String,
    loaded: Option<LoadedLlama>,
}

impl LlamaCppBackend {
    // Creates a new LlamaCppBackend with the given configuration. The model is not loaded until generate() is called.
    pub fn new(config: LlamaCppConfig) -> Self {
        let model_name = config
            .model_path
            .as_ref()
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
            .unwrap_or("unconfigured")
            .to_string();

        Self {
            config,
            display_name: format!("llama.cpp · {model_name}"),
            loaded: None,
        }
    }

    // Lazily loads the model once and caches it for reuse across requests.
    fn ensure_loaded(&mut self) -> Result<&mut LoadedLlama> {
        if self.loaded.is_none() {
            let model_path = self.require_model_path()?;
            let loaded = load_model(&self.config, &model_path)?;
            self.loaded = Some(loaded);
        }

        self.loaded
            .as_mut()
            .ok_or_else(|| AppError::Runtime("llama.cpp model failed to initialize.".to_string()))
    }

    // Retrieves the model path from the config or returns an error if it's not set.
    fn require_model_path(&self) -> Result<PathBuf> {
        self.config.model_path.clone().ok_or_else(|| {
            AppError::Runtime(
                "llama.cpp backend selected, but `llama_cpp.model_path` is not configured."
                    .to_string(),
            )
        })
    }
}

impl ModelBackend for LlamaCppBackend {
    // Returns the display name of the backend, which includes the model name if available.
    fn name(&self) -> &str {
        &self.display_name
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            // context_tokens == 0 means "defer to the model's trained context window at load
            // time" — report None since the true limit is not known until the model is loaded.
            context_window_tokens: if self.config.context_tokens > 0 {
                Some(self.config.context_tokens)
            } else {
                None
            },
            max_output_tokens: Some(self.config.max_tokens),
        }
    }

    // Builds the prompt, ensures the model is loaded, and streams generation events.
    fn generate(
        &mut self,
        request: GenerateRequest,
        on_event: &mut dyn FnMut(BackendEvent),
    ) -> Result<()> {
        let config = self.config.clone();
        let prompt = format_messages(&request.messages);
        let is_cold = self.loaded.is_none();
        if is_cold {
            on_event(BackendEvent::StatusChanged(BackendStatus::LoadingModel));
        }
        let t_load_start = is_cold.then(std::time::Instant::now);
        let loaded = self.ensure_loaded()?;
        if let Some(t) = t_load_start {
            on_event(BackendEvent::Timing {
                stage: "model_load",
                elapsed_ms: t.elapsed().as_millis() as u64,
            });
        }
        run_generation(loaded, &config, &prompt, on_event)
    }
}

#[cfg(test)]
mod tests {
    use super::prompt::format_messages;
    use crate::llm::backend::Message;

    #[test]
    fn appends_an_open_assistant_turn() {
        let prompt = format_messages(&[Message::system("system prompt"), Message::user("hello")]);

        assert!(prompt.contains("<|im_start|>system\nsystem prompt<|im_end|>\n"));
        assert!(prompt.contains("<|im_start|>user\nhello<|im_end|>\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }
}
