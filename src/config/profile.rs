use std::path::PathBuf;

use serde::Deserialize;
use tracing::{info, warn};

use crate::error::Result;
use crate::safety::{ReadScope, ShellMode};

use super::{load, Config, PROJECT_PROFILE_FILE};

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

    #[serde(default)]
    pub safety: ProjectSafetyProfile,
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

#[derive(Debug, Deserialize, Default)]
pub struct ProjectSafetyProfile {
    pub enabled: Option<bool>,
    pub read_scope: Option<ReadScope>,
    pub block_private_network: Option<bool>,
    pub inspect_network: Option<bool>,
    pub shell_mode: Option<ShellMode>,
    pub block_destructive_shell: Option<bool>,
}

/// Apply a project profile on top of a base Config.
/// Fields that are None in the profile are left unchanged.
pub fn apply_profile(mut base: Config, profile: ProjectProfile) -> Config {
    if let Some(b) = profile.backend {
        base.backend = b;
    }
    if let Some(p) = profile.llama_cpp.model_path {
        base.llama_cpp.model_path = Some(p);
    }
    if let Some(u) = profile.ollama.url {
        base.ollama.url = u;
    }
    if let Some(m) = profile.ollama.model {
        base.ollama.model = m;
    }
    if let Some(u) = profile.openai_compat.url {
        base.openai_compat.url = u;
    }
    if let Some(m) = profile.openai_compat.model {
        base.openai_compat.model = m;
    }
    if let Some(k) = profile.openai_compat.api_key {
        base.openai_compat.api_key = k;
    }
    if let Some(n) = profile.openai_compat.provider_name {
        base.openai_compat.provider_name = n;
    }
    if let Some(t) = profile.generation.max_tokens {
        base.generation.max_tokens = t;
    }
    if let Some(t) = profile.generation.temperature {
        base.generation.temperature = t;
    }
    if let Some(v) = profile.budget.input_cost_per_million {
        base.budget.input_cost_per_million = Some(v);
    }
    if let Some(v) = profile.budget.output_cost_per_million {
        base.budget.output_cost_per_million = Some(v);
    }
    if let Some(v) = profile.cache.ttl_seconds {
        base.cache.ttl_seconds = v;
    }
    if let Some(p) = profile.lsp.rust_analyzer_path {
        base.lsp.rust_analyzer_path = Some(p);
    }
    if let Some(t) = profile.lsp.timeout_ms {
        base.lsp.timeout_ms = t;
    }
    if let Some(e) = profile.reflection.enabled {
        base.reflection.enabled = e;
    }
    if let Some(e) = profile.eco.enabled {
        base.eco.enabled = e;
    }
    if let Some(v) = profile.memory.fact_ttl_days {
        base.memory.fact_ttl_days = v;
    }
    if let Some(v) = profile.memory.max_facts_per_project {
        base.memory.max_facts_per_project = v;
    }
    if let Some(v) = profile.safety.enabled {
        base.safety.enabled = v;
    }
    if let Some(v) = profile.safety.read_scope {
        base.safety.read_scope = v;
    }
    if let Some(v) = profile.safety.block_private_network {
        base.safety.block_private_network = v;
    }
    if let Some(v) = profile.safety.inspect_network {
        base.safety.inspect_network = v;
    }
    if let Some(v) = profile.safety.shell_mode {
        base.safety.shell_mode = v;
    }
    if let Some(v) = profile.safety.block_destructive_shell {
        base.safety.block_destructive_shell = v;
    }
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
            reflection: ProjectReflectionProfile {
                enabled: Some(true),
            },
            ..ProjectProfile::default()
        };
        let merged = apply_profile(base, profile);
        assert!(merged.reflection.enabled);
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
        assert_eq!(merged.ollama.model, "qwen2.5-coder:7b");
    }

    #[test]
    fn apply_profile_overrides_safety_settings() {
        let base = Config::default();
        let profile = ProjectProfile {
            safety: ProjectSafetyProfile {
                enabled: Some(false),
                read_scope: Some(ReadScope::ProjectOnly),
                block_private_network: Some(false),
                inspect_network: Some(false),
                shell_mode: Some(ShellMode::ApproveInspect),
                block_destructive_shell: Some(false),
            },
            ..ProjectProfile::default()
        };
        let merged = apply_profile(base, profile);
        assert!(!merged.safety.enabled);
        assert!(!merged.safety.block_private_network);
        assert!(!merged.safety.inspect_network);
        assert!(!merged.safety.block_destructive_shell);
    }
}
