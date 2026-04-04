// src/config.rs
//
// App-wide configuration loaded from file.
//
// Configuration resolution order (highest precedence first):
//   1. .params.toml  — project-local profile in the current working directory
//   2. .local/config.toml — global/repo-level config
//   3. Compiled-in defaults
//
// Use load_with_profile() to get the fully-merged Config.
// Use load() only when you explicitly need the raw global config without a profile.

mod defaults_impl;
mod files;
mod profile;

use crate::safety::{ReadScope, ShellMode};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[allow(unused_imports)]
pub use files::{
    config_path, debug_log_path, find_model, keys_env_path, load, load_local_env, local_dir,
    log_dir, memory_dir, models_dir,
};
#[allow(unused_imports)]
pub use profile::{
    apply_profile, load_with_profile, ProjectBudgetProfile, ProjectCacheProfile, ProjectEcoProfile,
    ProjectGenerationProfile, ProjectLlamaCppProfile, ProjectLspProfile, ProjectMemoryProfile,
    ProjectOllamaProfile, ProjectOpenAICompatProfile, ProjectProfile, ProjectReflectionProfile,
    ProjectSafetyProfile,
};

/// File name searched in the current working directory for project-level overrides.
pub const PROJECT_PROFILE_FILE: &str = ".params.toml";

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
    pub cache: CacheConfig,

    #[serde(default)]
    pub lsp: LspConfig,

    #[serde(default)]
    pub reflection: ReflectionConfig,

    #[serde(default)]
    pub eco: EcoConfig,

    #[serde(default)]
    pub debug_logging: DebugLoggingConfig,

    #[serde(default)]
    pub memory: MemoryConfig,

    #[serde(default)]
    pub safety: SafetyConfig,

    /// Path to the active project profile, if one was found.
    /// Set by load_with_profile() — never read from or written to the TOML file.
    #[serde(skip)]
    pub active_profile: Option<PathBuf>,
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
pub struct CacheConfig {
    /// How long cache entries remain valid before they are treated as stale.
    /// Set to 0 to disable TTL-based expiration.
    #[serde(default = "default_cache_ttl_seconds")]
    pub ttl_seconds: u64,
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

#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// How many days before a fact is considered stale and pruned.
    /// Set to 0 to disable TTL-based pruning.
    #[serde(default = "default_fact_ttl_days")]
    pub fact_ttl_days: u64,

    /// Maximum number of facts to keep per project.
    /// When the cap is exceeded, the oldest (least recently seen) facts are removed.
    #[serde(default = "default_max_facts_per_project")]
    pub max_facts_per_project: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SafetyConfig {
    #[serde(default = "default_safety_enabled")]
    pub enabled: bool,

    #[serde(default)]
    pub read_scope: ReadScope,

    #[serde(default = "default_block_private_network")]
    pub block_private_network: bool,

    #[serde(default = "default_inspect_network")]
    pub inspect_network: bool,

    #[serde(default)]
    pub shell_mode: ShellMode,

    #[serde(default = "default_block_destructive_shell")]
    pub block_destructive_shell: bool,
}

fn default_fact_ttl_days() -> u64 {
    90
}

fn default_max_facts_per_project() -> usize {
    150
}

fn default_backend() -> String {
    "llama_cpp".to_string()
}

fn default_ollama_url() -> String {
    "http://localhost:11434".to_string()
}

fn default_ollama_model() -> String {
    "qwen2.5-coder:7b".to_string()
}

fn default_openai_url() -> String {
    "https://api.groq.com/openai/v1".to_string()
}

fn default_openai_model() -> String {
    "llama-3.3-70b-versatile".to_string()
}

fn default_max_tokens() -> i32 {
    512
}

fn default_temperature() -> f32 {
    0.8
}

fn default_cache_ttl_seconds() -> u64 {
    21600
}

fn default_lsp_timeout_ms() -> u64 {
    15000
}

fn default_safety_enabled() -> bool {
    true
}

fn default_block_private_network() -> bool {
    true
}

fn default_inspect_network() -> bool {
    true
}

fn default_block_destructive_shell() -> bool {
    true
}

impl OpenAICompatConfig {
    /// Infer a human-readable provider name from the URL if not set in config.
    pub fn resolved_provider_name(&self) -> String {
        if !self.provider_name.is_empty() {
            return self.provider_name.clone();
        }
        if self.url.contains("groq.com") {
            "groq".to_string()
        } else if self.url.contains("openai.com") {
            "openai".to_string()
        } else if self.url.contains("openrouter.ai") {
            "openrouter".to_string()
        } else if self.url.contains("x.ai") {
            "grok".to_string()
        } else {
            "api".to_string()
        }
    }
}
