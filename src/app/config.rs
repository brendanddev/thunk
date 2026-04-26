use std::fs;
use std::path::Path;
use std::path::PathBuf;

use serde::Deserialize;

use super::{AppError, Result};

/// Main configuration struct for the application
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct Config {
    pub app: AppConfig,
    pub ui: UiConfig,
    pub llm: LlmConfig,
    pub llama_cpp: LlamaCppConfig,
}

/// Application configuration for the app
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub name: String,
}

/// Default app config with the name set to "thunk"
impl Default for AppConfig {
    fn default() -> Self {
        Self {
            name: "thunk".to_string(),
        }
    }
}

/// UI configuration for the application
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct UiConfig {
    pub show_activity: bool,
}

/// Default UI config with activity display enabled
impl Default for UiConfig {
    fn default() -> Self {
        Self {
            show_activity: true,
        }
    }
}

/// Model provider selection for the application
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    pub provider: String,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: "mock".to_string(),
        }
    }
}

/// llama.cpp provider configuration
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LlamaCppConfig {
    pub model_path: Option<PathBuf>,
    pub gpu_layers: u32,
    pub context_tokens: u32,
    pub batch_tokens: u32,
    pub max_tokens: usize,
    pub temperature: f32,
    pub show_native_logs: bool,
}

/// Default llama.cpp config with no model path and reasonable defaults for other parameters
impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            gpu_layers: 0,
            context_tokens: 2048,
            batch_tokens: 256,
            max_tokens: 512,
            temperature: 0.7,
            show_native_logs: false,
        }
    }
}

/// Resolves relative paths in the config to absolute paths based on the provided root directory
impl Config {
    pub fn resolve_paths(mut self, root_dir: &Path) -> Self {
        self.llama_cpp.resolve_paths(root_dir);
        self
    }
}

/// Resolves relative paths in the llama.cpp config to absolute paths based on the provided root directory
impl LlamaCppConfig {
    fn resolve_paths(&mut self, root_dir: &Path) {
        if let Some(model_path) = self.model_path.as_mut() {
            if model_path.is_relative() {
                *model_path = root_dir.join(&*model_path);
            }
        }
    }
}

/// Loads the config from a TOML file at the specified path
pub fn load(path: &Path) -> Result<Config> {
    if !path.exists() {
        return Err(AppError::Config(format!(
            "Config file not found: {}",
            path.display()
        )));
    }

    let raw = fs::read_to_string(path)?;
    if raw.trim().is_empty() {
        return Ok(Config::default());
    }

    Ok(toml::from_str(&raw)?)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{Config, LlamaCppConfig};

    #[test]
    fn resolves_relative_llama_model_paths_from_project_root() {
        let mut config = Config::default();
        config.llama_cpp = LlamaCppConfig {
            model_path: Some("data/models/model.gguf".into()),
            gpu_layers: 0,
            context_tokens: 2048,
            batch_tokens: 256,
            max_tokens: 128,
            temperature: 0.5,
            show_native_logs: false,
        };

        let resolved = config.resolve_paths(Path::new("/tmp/project"));
        assert_eq!(
            resolved.llama_cpp.model_path.as_deref(),
            Some(Path::new("/tmp/project/data/models/model.gguf"))
        );
    }
}
