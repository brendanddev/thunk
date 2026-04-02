// main.rs
//
// Entry point.

mod error;
mod config;
mod events;
mod inference;
mod tools;
mod tui;

use clap::{Parser, Subcommand};
use tracing::info;

// Bring our Result type into scope so every function here can use `Result<T>`
// instead of the full `std::result::Result<T, error::ParamsError>`.
use error::Result;

/// The top-level CLI struct. Clap reads this to understand what arguments
/// and subcommands the program accepts.
///
/// `#[derive(Parser)]` generates the actual argument parsing logic.
/// `#[command(...)]` attributes set metadata like the binary name and description.
#[derive(Parser)]
#[command(name = "params")]
#[command(about = "Local-first AI coding assistant")]
#[command(version)]
struct Cli {
    /// An optional one-shot prompt. If provided without a subcommand,
    /// params runs the prompt through the local model and prints the response.
    /// If nothing is provided at all, the TUI opens instead.
    ///
    /// Example: params "explain what this function does"
    prompt: Option<String>,

    /// Optional subcommand (pull, index, compare, bench, train).
    /// If a subcommand is present, the prompt field is ignored.
    #[command(subcommand)]
    command: Option<Command>,
}

/// All the subcommands params supports.
///
/// Each variant becomes a subcommand: `params pull`, `params index`, etc.
/// Fields inside each variant become that subcommand's arguments or flags.
/// Doc comments (///) on variants become the help text for that subcommand.
#[derive(Subcommand)]
enum Command {
    /// Download a model to ~/.params/models/
    ///
    /// Example: params pull qwen2.5-coder-14b
    Pull {
        /// The model name to download from HuggingFace.
        model: String,
    },

    /// Index a project so params can use it as context
    ///
    /// Walks the directory, reads source files, chunks them, and stores
    /// them in SQLite so they can be retrieved during conversations.
    /// Example: params index .
    Index {
        /// Path to the project to index. Defaults to current directory.
        #[arg(default_value = ".")]
        path: String,
    },

    /// Run a prompt through both the local model and Claude side by side
    ///
    /// Requires ANTHROPIC_API_KEY to be set in your environment.
    /// Results are logged so they contribute to your benchmark score.
    /// Example: params compare "what does the session middleware do"
    Compare {
        /// The prompt to send to both models.
        prompt: String,
    },

    /// Show a summary of your benchmark ratings
    ///
    /// Reads your stored ratings from SQLite and shows win/loss/tie stats,
    /// broken down by task type over the last N comparisons.
    /// Example: params bench --last 100
    Bench {
        /// How many recent ratings to include in the summary.
        #[arg(long, default_value = "50")]
        last: usize,
    },

    /// Fine-tune the local model on a project's codebase
    ///
    /// Generates instruction/response pairs from your indexed code,
    /// runs a LoRA fine-tune on the base model, and saves the result
    /// to ~/.params/models/ as a new .gguf file.
    /// Example: params train --project .
    Train {
        /// Path to the project to train on. Defaults to current directory.
        #[arg(long, default_value = ".")]
        project: String,
    },
}

/// Initialises file-based logging to ~/.params/params.log.
///
/// Returns a WorkerGuard that must be kept alive for the duration of the
/// program — dropping it flushes and closes the log file.  Returns None
/// if the log directory cannot be created (the app still runs, just without
/// file logging).
fn init_logging() -> Option<tracing_appender::non_blocking::WorkerGuard> {
    let home = dirs::home_dir()?;
    let log_dir = home.join(".params");
    std::fs::create_dir_all(&log_dir).ok()?;

    let level = if std::env::var("PARAMS_LOG").as_deref() == Ok("debug") {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };

    let file_appender = tracing_appender::rolling::never(&log_dir, "params.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    tracing_subscriber::fmt()
        .with_writer(non_blocking)
        .with_max_level(level)
        .with_ansi(false)
        .init();

    Some(guard)
}

/// The main function. Returns Result<()> so we can use ? to propagate
/// errors all the way up and have them printed cleanly on failure,
/// instead of panicking or unwrapping.
fn main() -> Result<()> {
    // Hold the guard for the process lifetime so the log file stays open.
    let _log_guard = init_logging();

    info!("params-cli starting");

    // Cli::parse() reads from std::env::args(), matches against our struct,
    // and either returns a populated Cli value or prints help/errors and exits.
    let cli = Cli::parse();

    // Route to the right handler based on what was typed.
    // `match` in Rust is exhaustive — the compiler forces us to handle
    // every possible variant, so nothing can silently fall through.
    match cli.command {
        Some(Command::Pull { model }) => {
            println!("Pulling model: {model}");
        }

        Some(Command::Index { path }) => {
            println!("Indexing: {path}");
        }

        Some(Command::Compare { prompt }) => {
            println!("Comparing: {prompt}");
        }

        Some(Command::Bench { last }) => {
            println!("Benchmark (last {last} ratings)");
        }

        Some(Command::Train { project }) => {
            println!("Training on: {project}");
        }

        // No subcommand was given — check if there's a one-shot prompt,
        // otherwise open the TUI.
        None => match cli.prompt {
            Some(prompt) => {
                // One-shot mode — load config, build messages, run via ollama or llama.cpp
                let cfg = config::load()?;
                let messages = vec![
                    inference::Message::system(inference::SYSTEM_PROMPT),
                    inference::Message::user(&prompt),
                ];

                // Build the backend and generate directly to stdout
                let (tx, rx) = std::sync::mpsc::channel();
                let backend: Box<dyn inference::InferenceBackend> = match cfg.backend.as_str() {
                    "ollama" => Box::new(inference::OllamaBackend::new(&cfg.ollama.url, &cfg.ollama.model)),
                    _ => {
                        let model_path = cfg.llama_cpp.model_path
                            .unwrap_or_else(|| config::find_model().unwrap());
                        Box::new(inference::LlamaCppBackend::load(model_path, cfg.generation.max_tokens, cfg.generation.temperature)?)
                    }
                };

                // Spawn generation on a thread, print tokens as they arrive
                let handle = std::thread::spawn(move || {
                    backend.generate(&messages, tx)
                });

                use events::InferenceEvent;
                for event in rx {
                    match event {
                        InferenceEvent::Token(t) => { print!("{t}"); use std::io::Write; std::io::stdout().flush().ok(); }
                        InferenceEvent::Done => break,
                        InferenceEvent::Error(e) => { eprintln!("Error: {e}"); break; }
                        _ => {}
                    }
                }
                println!();
                let _ = handle.join();
            }
            None => {
                // Launch the full TUI app
                tui::run()?;
            }
        },
    }

    // Returning Ok(()) signals success. If any TODO above returns an Err,
    // the ? operator will propagate it here and Rust will print the error message.
    Ok(())
}
