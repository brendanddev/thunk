use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParamsError {
    #[error("Model error: {0}")]
    Model(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Config error: {0}")]
    Config(String),
}

/// A shorthand Result type
pub type Result<T> = std::result::Result<T, ParamsError>;
