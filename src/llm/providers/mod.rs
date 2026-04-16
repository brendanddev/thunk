mod llama_cpp;
mod mock;

use crate::app::config::Config;
use crate::app::{AppError, Result};
use crate::llm::backend::ModelBackend;

pub use llama_cpp::LlamaCppBackend;

use mock::MockBackend;

pub fn build_backend(config: &Config) -> Result<Box<dyn ModelBackend>> {
    match config.llm.provider.as_str() {
        "mock" => Ok(Box::new(MockBackend::new(config.app.name.clone()))),
        "llama_cpp" => Ok(Box::new(LlamaCppBackend::new(config.llama_cpp.clone()))),
        other => Err(AppError::Config(format!(
            "Unknown llm.provider `{other}`. Expected `mock` or `llama_cpp`."
        ))),
    }
}
