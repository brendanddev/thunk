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

use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};
use crate::error::{ParamsError, Result};

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

fn default_fact_ttl_days() -> u64 { 90 }
fn default_max_facts_per_project() -> usize { 150 }

fn default_backend() -> String { "llama_cpp".to_string() }
fn default_ollama_url() -> String { "http://localhost:11434".to_string() }
fn default_ollama_model() -> String { "qwen2.5-coder:7b".to_string() }
fn default_openai_url() -> String { "https://api.groq.com/openai/v1".to_string() }
fn default_openai_model() -> String { "llama-3.3-70b-versatile".to_string() }
fn default_max_tokens() -> i32 { 512 }
fn default_temperature() -> f32 { 0.8 }
fn default_cache_ttl_seconds() -> u64 { 21600 }
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

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            ttl_seconds: default_cache_ttl_seconds(),
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
            cache: CacheConfig::default(),
            lsp: LspConfig::default(),
            reflection: ReflectionConfig::default(),
            eco: EcoConfig::default(),
            debug_logging: DebugLoggingConfig::default(),
            memory: MemoryConfig::default(),
            active_profile: None,
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

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            fact_ttl_days: default_fact_ttl_days(),
            max_facts_per_project: default_max_facts_per_project(),
        }
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
             # Response cache invalidation:\n\
             #   [cache]\n\
             #   ttl_seconds = 21600   # 6 hours, set 0 to disable TTL\n\
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

// ---------------------------------------------------------------------------
// Project profiles
// ---------------------------------------------------------------------------
//
// A .params.toml file in the current working directory provides project-local
// overrides on top of the global .local/config.toml. Only the fields you set
// in .params.toml are changed; everything else keeps its global value.
//
// Example .params.toml:
//
//   backend = "openai_compat"
//
//   [openai_compat]
//   model = "gpt-4o"
//
//   [reflection]
//   enabled = true
//
//   [generation]
//   max_tokens = 1024

/// Project-local config overrides loaded from .params.toml.
/// Every field is optional — only present fields are applied.
#[derive(Debug, Deserialize, Default)]
pub struct ProjectProfile {
    pub backend: Option<String>,

    #[serde(default)]
    pub llama_cpp: ProjectLlamaCppProfile,

    #[serde(default)]
    pub ollama: ProjectOllamaProfile,

    #[serde(default)]
    pub openai_compat: ProjectOpenAICompatProfile,

    #[serde(default)]
    pub generation: ProjectGenerationProfile,

    #[serde(default)]
    pub budget: ProjectBudgetProfile,

    #[serde(default)]
    pub cache: ProjectCacheProfile,

    #[serde(default)]
    pub lsp: ProjectLspProfile,

    #[serde(default)]
    pub reflection: ProjectReflectionProfile,

    #[serde(default)]
    pub eco: ProjectEcoProfile,

    #[serde(default)]
    pub memory: ProjectMemoryProfile,
}

#[derive(Debug, Deserialize, Default)]
pub struct ProjectLlamaCppProfile {
    pub model_path: Option<PathBuf>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ProjectOllamaProfile {
    pub url: Option<String>,
    pub model: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ProjectOpenAICompatProfile {
    pub url: Option<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub provider_name: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ProjectGenerationProfile {
    pub max_tokens: Option<i32>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ProjectBudgetProfile {
    pub input_cost_per_million: Option<f64>,
    pub output_cost_per_million: Option<f64>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ProjectCacheProfile {
    pub ttl_seconds: Option<u64>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ProjectLspProfile {
    pub rust_analyzer_path: Option<PathBuf>,
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ProjectReflectionProfile {
    pub enabled: Option<bool>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ProjectEcoProfile {
    pub enabled: Option<bool>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ProjectMemoryProfile {
    pub fact_ttl_days: Option<u64>,
    pub max_facts_per_project: Option<usize>,
}

/// Apply a project profile on top of a base Config.
/// Fields that are None in the profile are left unchanged.
pub fn apply_profile(mut base: Config, profile: ProjectProfile) -> Config {
    if let Some(b) = profile.backend { base.backend = b; }
    if let Some(p) = profile.llama_cpp.model_path { base.llama_cpp.model_path = Some(p); }
    if let Some(u) = profile.ollama.url { base.ollama.url = u; }
    if let Some(m) = profile.ollama.model { base.ollama.model = m; }
    if let Some(u) = profile.openai_compat.url { base.openai_compat.url = u; }
    if let Some(m) = profile.openai_compat.model { base.openai_compat.model = m; }
    if let Some(k) = profile.openai_compat.api_key { base.openai_compat.api_key = k; }
    if let Some(n) = profile.openai_compat.provider_name { base.openai_compat.provider_name = n; }
    if let Some(t) = profile.generation.max_tokens { base.generation.max_tokens = t; }
    if let Some(t) = profile.generation.temperature { base.generation.temperature = t; }
    if let Some(v) = profile.budget.input_cost_per_million { base.budget.input_cost_per_million = Some(v); }
    if let Some(v) = profile.budget.output_cost_per_million { base.budget.output_cost_per_million = Some(v); }
    if let Some(v) = profile.cache.ttl_seconds { base.cache.ttl_seconds = v; }
    if let Some(p) = profile.lsp.rust_analyzer_path { base.lsp.rust_analyzer_path = Some(p); }
    if let Some(t) = profile.lsp.timeout_ms { base.lsp.timeout_ms = t; }
    if let Some(e) = profile.reflection.enabled { base.reflection.enabled = e; }
    if let Some(e) = profile.eco.enabled { base.eco.enabled = e; }
    if let Some(v) = profile.memory.fact_ttl_days { base.memory.fact_ttl_days = v; }
    if let Some(v) = profile.memory.max_facts_per_project { base.memory.max_facts_per_project = v; }
    base
}

/// Load the global config and apply any project-local .params.toml overrides.
///
/// This is the function all callers should use. It returns the fully-merged
/// Config with `active_profile` set if a profile was found.
///
/// If .params.toml exists but cannot be parsed, a warning is logged and the
/// global config is returned unchanged so the app still starts cleanly.
pub fn load_with_profile() -> Result<Config> {
    let base = load()?;

    let profile_path = match std::env::current_dir() {
        Ok(cwd) => cwd.join(PROJECT_PROFILE_FILE),
        Err(_) => return Ok(base),
    };

    if !profile_path.exists() {
        return Ok(base);
    }

    let contents = match std::fs::read_to_string(&profile_path) {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, path = %profile_path.display(), "could not read project profile");
            return Ok(base);
        }
    };

    match toml::from_str::<ProjectProfile>(&contents) {
        Ok(profile) => {
            info!(path = %profile_path.display(), "project profile loaded");
            let mut merged = apply_profile(base, profile);
            merged.active_profile = Some(profile_path);
            Ok(merged)
        }
        Err(e) => {
            warn!(error = %e, path = %profile_path.display(), "project profile parse failed — using global config");
            Ok(base)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_profile_empty_leaves_config_unchanged() {
        let base = Config::default();
        let original_backend = base.backend.clone();
        let original_reflection = base.reflection.enabled;
        let merged = apply_profile(base, ProjectProfile::default());
        assert_eq!(merged.backend, original_backend);
        assert_eq!(merged.reflection.enabled, original_reflection);
    }

    #[test]
    fn apply_profile_overrides_backend() {
        let base = Config::default();
        let profile = ProjectProfile {
            backend: Some("ollama".to_string()),
            ..ProjectProfile::default()
        };
        let merged = apply_profile(base, profile);
        assert_eq!(merged.backend, "ollama");
    }

    #[test]
    fn apply_profile_overrides_only_set_fields() {
        let base = Config::default();
        let profile = ProjectProfile {
            reflection: ProjectReflectionProfile { enabled: Some(true) },
            ..ProjectProfile::default()
        };
        let merged = apply_profile(base, profile);
        assert!(merged.reflection.enabled);
        // eco was not set in profile — should remain at default (false)
        assert!(!merged.eco.enabled);
    }

    #[test]
    fn apply_profile_overrides_model_settings() {
        let base = Config::default();
        let profile = ProjectProfile {
            backend: Some("openai_compat".to_string()),
            openai_compat: ProjectOpenAICompatProfile {
                model: Some("gpt-4o".to_string()),
                ..ProjectOpenAICompatProfile::default()
            },
            generation: ProjectGenerationProfile {
                max_tokens: Some(2048),
                ..ProjectGenerationProfile::default()
            },
            cache: ProjectCacheProfile {
                ttl_seconds: Some(3600),
            },
            ..ProjectProfile::default()
        };
        let merged = apply_profile(base, profile);
        assert_eq!(merged.backend, "openai_compat");
        assert_eq!(merged.openai_compat.model, "gpt-4o");
        assert_eq!(merged.generation.max_tokens, 2048);
        assert_eq!(merged.cache.ttl_seconds, 3600);
        // ollama settings unchanged
        assert_eq!(merged.ollama.model, "qwen2.5-coder:7b");
    }
}
