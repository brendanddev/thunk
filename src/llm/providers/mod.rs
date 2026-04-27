mod llama_cpp;
mod mock;

use crate::app::config::Config;
use crate::app::{AppError, Result};
use crate::llm::backend::ModelBackend;

pub use llama_cpp::LlamaCppBackend;

use mock::MockBackend;

type BackendFactory = fn(&Config) -> Result<Box<dyn ModelBackend>>;

fn make_mock(config: &Config) -> Result<Box<dyn ModelBackend>> {
    Ok(Box::new(MockBackend::new(config.app.name.clone())))
}

fn make_llama_cpp(config: &Config) -> Result<Box<dyn ModelBackend>> {
    Ok(Box::new(LlamaCppBackend::new(config.llama_cpp.clone())))
}

const BACKEND_REGISTRY: &[(&str, BackendFactory)] = &[
    ("mock",      make_mock),
    ("llama_cpp", make_llama_cpp),
];

pub fn build_backend(config: &Config) -> Result<Box<dyn ModelBackend>> {
    let name = config.llm.provider.as_str();
    BACKEND_REGISTRY
        .iter()
        .find(|(key, _)| *key == name)
        .map(|(_, factory)| factory(config))
        .unwrap_or_else(|| {
            let known = BACKEND_REGISTRY
                .iter()
                .map(|(k, _)| *k)
                .collect::<Vec<_>>()
                .join(", ");
            Err(AppError::Config(format!(
                "Unknown llm.provider `{name}`. Expected one of: {known}."
            )))
        })
}
