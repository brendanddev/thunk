use std::fs;
use std::path::Path;

use serde::Deserialize;

use super::{AppError, Result};

/// Main configuration struct for the application
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct Config {
    pub app: AppConfig,
    pub ui: UiConfig,
}

/// Application configuration for the app
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub name: String,
}

/// Default app config with the name set to "params"
impl Default for AppConfig {
    fn default() -> Self {
        Self {
            name: "params".to_string(),
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
