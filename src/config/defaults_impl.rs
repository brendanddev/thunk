use super::*;

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self { model_path: None }
    }
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            url: default_ollama_url(),
            model: default_ollama_model(),
        }
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
        Self {
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
        }
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

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            enabled: default_safety_enabled(),
            read_scope: ReadScope::default(),
            block_private_network: default_block_private_network(),
            inspect_network: default_inspect_network(),
            shell_mode: ShellMode::default(),
            block_destructive_shell: default_block_destructive_shell(),
            shell_allowlist: Vec::new(),
            shell_denylist: Vec::new(),
            network_allowlist: Vec::new(),
            inspect_cloud_requests: default_inspect_cloud_requests(),
        }
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
            safety: SafetyConfig::default(),
            active_profile: None,
        }
    }
}
