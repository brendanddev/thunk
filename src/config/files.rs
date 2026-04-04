use std::path::PathBuf;

use crate::error::{ParamsError, Result};

use super::Config;

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
        let toml =
            toml::to_string_pretty(&default).map_err(|e| ParamsError::Config(e.to_string()))?;

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
             #   content = false\n\
             #\n\
             # Safety policy sandbox and inspection:\n\
             #   [safety]\n\
             #   enabled = true\n\
             #   read_scope = \"project_only\"\n\
             #   block_private_network = true\n\
             #   inspect_network = true\n\
             #   shell_mode = \"approve_inspect\"\n\
             #   block_destructive_shell = true\n\n\
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
