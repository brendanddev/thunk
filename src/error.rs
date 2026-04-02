// error.rs
//
// This file defines the single error type used across the entire params-cli project.
//
// We use the `thiserror` crate which generates boilerplate for us via the #[derive(Error)]
// and #[error("...")] attributes. Without it we'd have to implement the std::error::Error
// trait manually for every variant — thiserror just handles that automatically.

use thiserror::Error;

/// The unified error type for params-cli.
///
/// Each variant represents a different category of failure.
/// The #[error("...")] attribute on each variant defines the human-readable
/// message that gets printed when that error is displayed.
#[derive(Debug, Error)]
pub enum ParamsError {
    /// Fired when something goes wrong with model loading or management.
    /// e.g. model file not found, unsupported format, corrupt weights.
    /// The String inside carries a description of what went wrong.
    #[error("Model error: {0}")]
    Model(String),

    /// Fired when the inference engine fails mid-generation.
    /// e.g. context overflow, sampling failure, tokenization error.
    #[error("Inference error: {0}")]
    Inference(String),

    /// Fired when a SQLite database operation fails.
    /// The #[from] means Rust will automatically convert a rusqlite::Error
    /// into this variant — so you can use ? on database calls without
    /// manually wrapping them.
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    /// Fired when a file system operation fails.
    /// e.g. can't read a source file, can't write to ~/.params/
    /// #[from] gives us automatic conversion from std::io::Error via ?
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Fired when something is misconfigured.
    /// e.g. ANTHROPIC_API_KEY not set, invalid config file, bad model path.
    #[error("Config error: {0}")]
    Config(String),
}

/// A shorthand Result type used throughout the project.
///
/// Instead of writing `std::result::Result<T, ParamsError>` everywhere,
/// every module can just write `Result<T>` and this alias fills in the
/// error type automatically. This is a very common Rust pattern.
///
/// Example usage in any function:
///   fn load_model(path: &str) -> Result<Model> { ... }
pub type Result<T> = std::result::Result<T, ParamsError>;
