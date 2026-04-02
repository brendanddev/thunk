// src/config.rs
//
// App-wide configuration loaded from file

use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use crate::error::{ParamsError, Result};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    /// Which backend to use. Options: "llama_cpp", "ollama", "openai_compat"
    #[serde(default = "default_backend")]
    pub backend: String,

    #[serde(default)]
    pub llama_cpp: LlamaCppConfig,

    #[serde(default)]
    pub ollama: OllamaConfig,

    /// OpenAI-compatible config — covers OpenAI, Groq, OpenRouter, Grok.
    /// Switch providers by changing url, api_key, and model.
    #[serde(default)]
    pub openai_compat: OpenAICompatConfig,

    #[serde(default)]
    pub generation: GenerationConfig,

    #[serde(default)]
    pub budget: BudgetConfig,

    #[serde(default)]
    pub lsp: LspConfig,

    #[serde(default)]
    pub reflection: ReflectionConfig,

    #[serde(default)]
    pub eco: EcoConfig,

    #[serde(default)]
    pub debug_logging: DebugLoggingConfig,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LlamaCppConfig {
    /// Path to .gguf file. Auto-detects from file path if unset.
    pub model_path: Option<PathBuf>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OllamaConfig {
    #[serde(default = "default_ollama_url")]
    pub url: String,

    #[serde(default = "default_ollama_model")]
    pub model: String,
}

/// Config for any OpenAI-compatible provider.
///
/// Examples:
///
///   Groq (fast, cheap, great for coding):
///     url = "https://api.groq.com/openai/v1"
///     api_key = ""   # set GROQ_API_KEY env var instead
///     model = "llama-3.3-70b-versatile"
///
///   OpenRouter (access to every model):
///     url = "https://openrouter.ai/api/v1"
///     api_key = ""   # set OPENROUTER_API_KEY env var
///     model = "anthropic/claude-sonnet-4-5"
///
///   OpenAI:
///     url = "https://api.openai.com/v1"
///     api_key = ""   # set OPENAI_API_KEY env var
///     model = "gpt-4o"
///
///   Grok (xAI):
///     url = "https://api.x.ai/v1"
///     api_key = ""   # set XAI_API_KEY env var
///     model = "grok-2-latest"
/// 
/// Note: look into how to turn this type of configuration into
/// a preset script or command line option even
#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAICompatConfig {
    /// API base URL for the provider
    #[serde(default = "default_openai_url")]
    pub url: String,

    /// API key. Leave empty and set env var or .local/keys.env instead:
    ///   GROQ_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, or XAI_API_KEY
    #[serde(default)]
    pub api_key: String,

    /// Model name as the provider knows it
    #[serde(default = "default_openai_model")]
    pub model: String,

    /// Display name shown in the sidebar. Auto-detected from URL if empty.
    #[serde(default)]
    pub provider_name: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerationConfig {
    #[serde(default = "default_max_tokens")]
    pub max_tokens: i32,

    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BudgetConfig {
    /// Optional estimated input pricing in USD per 1M tokens.
    /// Leave unset for local backends or if you only want token estimates.
    pub input_cost_per_million: Option<f64>,

    /// Optional estimated output pricing in USD per 1M tokens.
    /// If unset but input_cost_per_million is set, the input price is reused.
    pub output_cost_per_million: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LspConfig {
    /// Optional explicit path to rust-analyzer for the first LSP slice.
    pub rust_analyzer_path: Option<PathBuf>,

    /// How long to wait for diagnostics from the language server.
    #[serde(default = "default_lsp_timeout_ms")]
    pub timeout_ms: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReflectionConfig {
    /// Whether to run a hidden self-check pass before showing the final response.
    #[serde(default)]
    pub enabled: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EcoConfig {
    /// Whether to prefer lower-token prompts and context by default.
    #[serde(default)]
    pub enabled: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DebugLoggingConfig {
    /// Whether to write user prompts and final assistant replies to a separate debug log.
    #[serde(default)]
    pub content: bool,
}

fn default_backend() -> String { "llama_cpp".to_string() }
fn default_ollama_url() -> String { "http://localhost:11434".to_string() }
fn default_ollama_model() -> String { "qwen2.5-coder:7b".to_string() }
fn default_openai_url() -> String { "https://api.groq.com/openai/v1".to_string() }
fn default_openai_model() -> String { "llama-3.3-70b-versatile".to_string() }
fn default_max_tokens() -> i32 { 512 }
fn default_temperature() -> f32 { 0.8 }
fn default_lsp_timeout_ms() -> u64 { 15000 }

impl OpenAICompatConfig {
    /// Infer a human-readable provider name from the URL if not set in config.
    pub fn resolved_provider_name(&self) -> String {
        if !self.provider_name.is_empty() {
            return self.provider_name.clone();
        }
        if self.url.contains("groq.com") { "groq".to_string() }
        else if self.url.contains("openai.com") { "openai".to_string() }
        else if self.url.contains("openrouter.ai") { "openrouter".to_string() }
        else if self.url.contains("x.ai") { "grok".to_string() }
        else { "api".to_string() }
    }
}

impl Default for LlamaCppConfig {
    fn default() -> Self { Self { model_path: None } }
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self { url: default_ollama_url(), model: default_ollama_model() }
    }
}

impl Default for OpenAICompatConfig {
    fn default() -> Self {
        Self {
            url: default_openai_url(),
            api_key: String::new(),
            model: default_openai_model(),
            provider_name: String::new(),
        }
    }
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self { max_tokens: default_max_tokens(), temperature: default_temperature() }
    }
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            input_cost_per_million: None,
            output_cost_per_million: None,
        }
    }
}

impl Default for LspConfig {
    fn default() -> Self {
        Self {
            rust_analyzer_path: None,
            timeout_ms: default_lsp_timeout_ms(),
        }
    }
}

impl Default for ReflectionConfig {
    fn default() -> Self {
        Self { enabled: false }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            backend: default_backend(),
            llama_cpp: LlamaCppConfig::default(),
            ollama: OllamaConfig::default(),
            openai_compat: OpenAICompatConfig::default(),
            generation: GenerationConfig::default(),
            budget: BudgetConfig::default(),
            lsp: LspConfig::default(),
            reflection: ReflectionConfig::default(),
            eco: EcoConfig::default(),
            debug_logging: DebugLoggingConfig::default(),
        }
    }
}

impl Default for EcoConfig {
    fn default() -> Self {
        Self { enabled: false }
    }
}

impl Default for DebugLoggingConfig {
    fn default() -> Self {
        Self { content: false }
    }
}

pub fn config_path() -> Result<PathBuf> {
    Ok(local_dir()?.join("config.toml"))
}

pub fn models_dir() -> Result<PathBuf> {
    let dir = local_dir()?.join("models");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn memory_dir() -> Result<PathBuf> {
    let dir = local_dir()?.join("memory");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn log_dir() -> Result<PathBuf> {
    let dir = local_dir()?;
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn keys_env_path() -> Result<PathBuf> {
    Ok(local_dir()?.join("keys.env"))
}

pub fn debug_log_path() -> Result<PathBuf> {
    Ok(local_dir()?.join("params-debug.log"))
}

pub fn local_dir() -> Result<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".local");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn load_local_env() -> Result<()> {
    let path = keys_env_path()?;
    if !path.exists() {
        return Ok(());
    }

    let contents = std::fs::read_to_string(&path)?;
    for raw_line in contents.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };

        let key = key.trim();
        if key.is_empty() || std::env::var_os(key).is_some() {
            continue;
        }

        let value = value.trim();
        let value = value
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .or_else(|| value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')))
            .unwrap_or(value);

        std::env::set_var(key, value);
    }

    Ok(())
}

pub fn find_model() -> Result<PathBuf> {
    let dir = models_dir()?;
    if !dir.exists() {
        return Err(ParamsError::Model(format!(
            "No models directory found at {}. Run: params pull qwen2.5-coder-7b",
            dir.display()
        )));
    }
    let model_path = std::fs::read_dir(&dir)?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().and_then(|e| e.to_str()) == Some("gguf"));
    model_path.ok_or_else(|| {
        ParamsError::Model(format!(
            "No .gguf model found in {}. Run: params pull qwen2.5-coder-7b",
            dir.display()
        ))
    })
}

pub fn load() -> Result<Config> {
    let path = config_path()?;

    if !path.exists() {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let default = Config::default();
        let toml = toml::to_string_pretty(&default)
            .map_err(|e| ParamsError::Config(e.to_string()))?;

        let with_comments = format!(
            "# params-cli configuration\n\
             # Backend options: \"llama_cpp\", \"ollama\", \"openai_compat\"\n\
             # Stored locally in this repo under .local/\n\
             #\n\
             # OpenAI-compatible providers (set backend = \"openai_compat\"):\n\
             #   Groq:       url = \"https://api.groq.com/openai/v1\"\n\
             #   OpenRouter: url = \"https://openrouter.ai/api/v1\"\n\
             #   OpenAI:     url = \"https://api.openai.com/v1\"\n\
             #   Grok:       url = \"https://api.x.ai/v1\"\n\
             #\n\
             # API keys: set in config, .local/keys.env, or env vars:\n\
             #   GROQ_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, XAI_API_KEY\n\
             #\n\
             # Budget tracking uses token estimates for all backends.\n\
             # For remote APIs, you can optionally set estimated USD pricing:\n\
             #   [budget]\n\
             #   input_cost_per_million = 0.0\n\
             #   output_cost_per_million = 0.0\n\
             #\n\
             # LSP diagnostics (Rust-first initial slice):\n\
             #   [lsp]\n\
             #   rust_analyzer_path = \"/absolute/path/to/rust-analyzer\"\n\
             #   timeout_ms = 15000\n\
             #\n\
             # Reflection pass:\n\
             #   [reflection]\n\
             #   enabled = false\n\
             #\n\
             # Eco mode:\n\
             #   [eco]\n\
             #   enabled = false\n\
             #\n\
             # Separate content debug logging (prompts/final answers only):\n\
             #   [debug_logging]\n\
             #   content = false\n\n\
             {toml}"
        );

        std::fs::write(&path, with_comments)?;
        return Ok(default);
    }

    let contents = std::fs::read_to_string(&path)?;
    let config: Config = toml::from_str(&contents)
        .map_err(|e| ParamsError::Config(format!("Config parse error: {e}")))?;

    Ok(config)
}
