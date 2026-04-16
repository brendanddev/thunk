mod native;
mod prompt;

use std::path::PathBuf;

use crate::app::config::LlamaCppConfig;
use crate::app::{AppError, Result};
use crate::llm::backend::{BackendEvent, BackendStatus, GenerateRequest, ModelBackend};

use native::{load_model, run_generation, LoadedLlama};
use prompt::format_messages;

pub struct LlamaCppBackend {
    config: LlamaCppConfig,
    display_name: String,
    loaded: Option<LoadedLlama>,
}

impl LlamaCppBackend {
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
    fn name(&self) -> &str {
        &self.display_name
    }

    fn generate(
        &mut self,
        request: GenerateRequest,
        on_event: &mut dyn FnMut(BackendEvent),
    ) -> Result<()> {
        let config = self.config.clone();
        let prompt = format_messages(&request.messages);
        if self.loaded.is_none() {
            on_event(BackendEvent::StatusChanged(BackendStatus::LoadingModel));
        }
        let loaded = self.ensure_loaded()?;
        on_event(BackendEvent::StatusChanged(BackendStatus::Generating));
        run_generation(loaded, &config, &prompt, on_event)
    }
}

#[cfg(test)]
mod tests {
    use super::prompt::format_messages;
    use crate::llm::backend::Message;

    #[test]
    fn appends_an_open_assistant_turn() {
        let prompt = format_messages(&[
            Message::system("system prompt"),
            Message::user("hello"),
        ]);

        assert!(prompt.contains("<|im_start|>system\nsystem prompt<|im_end|>\n"));
        assert!(prompt.contains("<|im_start|>user\nhello<|im_end|>\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }
}
