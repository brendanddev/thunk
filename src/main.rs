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
mod skills;
mod tools;
mod tui;

use clap::{Parser, Subcommand};
use error::{ParamsError, Result};
use tracing::info;

/// The top-level CLI struct
#[derive(Parser)]
#[command(name = "params")]
#[command(about = "Local-first AI coding assistant")]
#[command(version)]
struct Cli {
    prompt: Option<String>,

    #[arg(long)]
    no_resume: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Download a model for local use
    Pull { model: String },

    /// Index a project for retrieval and repo-aware assistance
    Index {
        #[arg(default_value = ".")]
        path: String,
    },

    /// Compare local against provider responses
    Compare { prompt: String },

    /// Show summary of benchmark ratings
    Bench {
        #[arg(long, default_value = "50")]
        last: usize,
    },

    /// Fine-tune the local model on a projects codebase
    Train {
        #[arg(long, default_value = ".")]
        project: String,
    },

    /// Check local LSP setup for the first Rust diagnostics slice
    LspCheck,
}

/// Initialises file-based logging
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

fn main() -> Result<()> {
    config::load_local_env()?;

    let _log_guard = init_logging();
    info!("params-cli starting");

    let cli = Cli::parse();

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

        // No subcommand was given, check if there's a one-shot prompt
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
                let project_root =
                    std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
                let mut system_prompt = system_prompt;
                skills::append_chat_skill_guidance(&project_root, &prompt, &mut system_prompt);
                let messages = vec![
                    inference::Message::system(&system_prompt),
                    inference::Message::user(&prompt),
                ];

                // Build the backend and generate directly to stdout
                let (tx, rx) = std::sync::mpsc::channel();
                let backend = inference::load_backend_from_config(&cfg)?;
                let mut collected = String::new();

                // Spawn generation on a thread, print tokens as they arrive
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
