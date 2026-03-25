
// error.rs
// This file defines the single error type used across the entire project

use thiserror::Error;

/// The unified error type
#[derive(Debug, Error)]
pub enum ParamsError {
    /// Fired when something goes wrong with model loading or management.
    /// The String inside carries a description of what went wrong.
    #[error("Model error: {0}")]
    Model(String),

    /// Fired when the inference engine fails mid generation
    #[error("Inference error: {0}")]
    Inference(String),

    /// Fired when a SQLite database operation fails
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    /// Fired when a file system operation fails
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Fired when an HTTP request fails
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    /// Fired when something is misconfigured
    #[error("Config error: {0}")]
    Config(String),
}

/// Assigns an alias to the Result type
pub type Result<T> = std::result::Result<T, ParamsError>;