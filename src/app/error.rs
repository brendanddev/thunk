use thiserror::Error;

/// Defines the custom error type for the app
#[derive(Debug, Error)]
pub enum AppError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Config parse error: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("Config error: {0}")]
    Config(String),

    #[error("TUI error: {0}")]
    Tui(String),

    #[error("Runtime error: {0}")]
    Runtime(String),
}

pub type Result<T> = std::result::Result<T, AppError>;
