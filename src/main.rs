// main.rs
//
// Entry point.

mod cache;
mod commands;
mod config;
mod debug_log;
mod error;
mod events;
mod hooks;
mod inference;
#[allow(dead_code)]
mod memory;
mod safety;
mod session;
mod tools;
mod tui;

use clap::{Parser, Subcommand};
use tracing::info;

// Bring our Result type into scope so every function here can use `Result<T>`
// instead of the full `std::result::Result<T, error::ParamsError>`.
use error::{ParamsError, Result};

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

    /// Start the TUI with a fresh unnamed session instead of auto-resuming the most recent one.
    #[arg(long)]
    no_resume: bool,

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
    /// Download a model to .local/models/
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
    /// to .local/models/ as a new .gguf file.
    /// Example: params train --project .
    Train {
        /// Path to the project to train on. Defaults to current directory.
        #[arg(long, default_value = ".")]
        project: String,
    },

    /// Check local LSP setup for the first Rust diagnostics slice
    ///
    /// Reports how params resolves rust-analyzer and what to fix if it is not runnable.
    LspCheck,
}

/// Initialises file-based logging to .local/params.log.
///
/// Returns a WorkerGuard that must be kept alive for the duration of the
/// program — dropping it flushes and closes the log file.  Returns None
/// if the log directory cannot be created (the app still runs, just without
/// file logging).
fn init_logging() -> Option<tracing_appender::non_blocking::WorkerGuard> {
    let log_dir = config::log_dir().ok()?;

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
    config::load_local_env()?;

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
            info!(command = "pull", "cli command selected");
            return Err(ParamsError::Config(format!(
                "`params pull {model}` is not implemented yet."
            )));
        }

        Some(Command::Index { path }) => {
            info!(command = "index", "cli command selected");
            run_index_command(&path)?;
        }

        Some(Command::Compare { prompt }) => {
            info!(command = "compare", "cli command selected");
            return Err(ParamsError::Config(format!(
                "`params compare {prompt}` is not implemented yet."
            )));
        }

        Some(Command::Bench { last }) => {
            info!(command = "bench", "cli command selected");
            return Err(ParamsError::Config(format!(
                "`params bench --last {last}` is not implemented yet."
            )));
        }

        Some(Command::Train { project }) => {
            info!(command = "train", "cli command selected");
            return Err(ParamsError::Config(format!(
                "`params train --project {project}` is not implemented yet."
            )));
        }

        Some(Command::LspCheck) => {
            info!(command = "lsp-check", "cli command selected");
            println!("{}", tools::rust_lsp_health_report());
        }

        // No subcommand was given — check if there's a one-shot prompt,
        // otherwise open the TUI.
        None => match cli.prompt {
            Some(prompt) => {
                info!(mode = "one_shot", "starting one-shot generation");
                // One-shot mode — load config and run with the selected backend.
                let cfg = config::load_with_profile()?;
                if cfg.debug_logging.content {
                    debug_log::append_user_prompt(&prompt)?;
                }
                let system_prompt = if cfg.eco.enabled {
                    format!(
                        "{}\n\nEco mode is active. Prefer concise answers.",
                        inference::SYSTEM_PROMPT
                    )
                } else {
                    inference::SYSTEM_PROMPT.to_string()
                };
                let messages = vec![
                    inference::Message::system(&system_prompt),
                    inference::Message::user(&prompt),
                ];

                // Build the backend and generate directly to stdout.
                let (tx, rx) = std::sync::mpsc::channel();
                let backend = inference::load_backend_from_config(&cfg)?;
                let mut collected = String::new();

                // Spawn generation on a thread, print tokens as they arrive.
                let handle = std::thread::spawn(move || backend.generate(&messages, tx));

                for event in rx {
                    match event {
                        events::InferenceEvent::Token(t) => {
                            print!("{t}");
                            collected.push_str(&t);
                            use std::io::Write;
                            std::io::stdout().flush().ok();
                        }
                        _ => {}
                    }
                }
                println!();
                if cfg.debug_logging.content && !collected.trim().is_empty() {
                    debug_log::append_assistant_response(
                        &collected,
                        debug_log::ResponseSource::Live,
                    )?;
                }
                match handle.join() {
                    Ok(result) => result?,
                    Err(_) => {
                        return Err(ParamsError::Inference(
                            "generation thread panicked".to_string(),
                        ));
                    }
                }
            }
            None => {
                info!(mode = "tui", "starting tui");
                // Launch the full TUI app
                tui::run_with_options(tui::TuiOptions {
                    no_resume: cli.no_resume,
                })?;
            }
        },
    }

    // Returning Ok(()) signals success. If any TODO above returns an Err,
    // the ? operator will propagate it here and Rust will print the error message.
    Ok(())
}

fn run_index_command(path: &str) -> Result<()> {
    let root = std::path::PathBuf::from(path);
    if !root.exists() {
        return Err(ParamsError::Config(format!(
            "Index path does not exist: {}",
            root.display()
        )));
    }
    if !root.is_dir() {
        return Err(ParamsError::Config(format!(
            "Index path is not a directory: {}",
            root.display()
        )));
    }

    let root = std::fs::canonicalize(root)?;
    info!(command = "index", "starting project index");
    let cfg = config::load_with_profile()?;
    let backend = inference::load_backend_from_config(&cfg)?;
    let index = memory::index::ProjectIndex::open_for(&root)?;
    let delta = index.collect_delta(&root)?;

    let mut indexed = 0usize;
    let mut skipped = delta.unchanged.saturating_add(delta.skipped_large);

    for file in delta.to_index {
        let content = match std::fs::read_to_string(&file) {
            Ok(content) => content,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };

        if content.len() > 100_000 {
            skipped += 1;
            continue;
        }

        index.index_file(&file, &content, &*backend)?;
        indexed += 1;
    }

    println!(
        "Indexed {} files in {} (skipped {}, removed {}).",
        indexed,
        root.display(),
        skipped,
        delta.removed
    );
    info!(
        indexed,
        skipped,
        removed = delta.removed,
        "project index completed"
    );
    Ok(())
}
