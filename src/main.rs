// main.rs
// Entry point for the cli. 

// Declare modules - corresponds to a file or folder in src/
mod error;
mod inference;

use clap::{Parser, Subcommand};

// Bring the Result type into scope
use error::Result;


/// Top-level CLI struct - Clap reads this to understand what arguments and subcommands 
/// are accepted
#[derive(Parser)]
#[command(name = "params")]
#[command(about = "Local-first AI coding assistant")]
#[command(version)]
struct Cli {
    
    /// Optional one-shot prompt, if provided without a subcommand, params
    /// runs the prompt through the local model and prints the response.
    /// If nothing provided at all, open the TUI
    prompt: Option<String>,
    
    /// Optional subcommand - if a subcommand is present, the prompt field 
    /// is ignored
    #[command(subcommand)]
    command: Option<Command>,
}

/// All the subcommands supported
#[derive(Subcommand)]
enum Command {
    /// Download a model to ~/.params/models/ - params pull ...
    Pull {
        /// The model name to download from HuggingFace
        model: String,
    },

    /// Index a project so params can use it as context
    ///
    /// Walks the directory, reads source files, chunks them, and stores
    /// them in SQLite so they can be retrieved during conversations.
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

/// The main function. Returns Result<()> so we can use ? to propagate
/// errors all the way up and have them printed cleanly on failure,
/// instead of panicking or unwrapping
fn main() -> Result<()> {
    // Cli::parse() reads from std::env::args(), matches against our struct,
    // and either returns a populated Cli value or prints help/errors and exits.
    let cli = Cli::parse();

    // Route to the right handler based on what was typed.
    // `match` in Rust is exhaustive, the compiler forces us to handle
    // every possible variant, so nothing can silently fall through.
    match cli.command {
        Some(Command::Pull { model }) => {
            println!("Pulling model: {model}");
            // TODO: src/commands/pull.rs — download .gguf to ~/.params/models/
        }

        Some(Command::Index { path }) => {
            println!("Indexing: {path}");
            // TODO: src/memory/mod.rs — walk repo, chunk files, store in SQLite
        }

        Some(Command::Compare { prompt }) => {
            println!("Comparing: {prompt}");
            // TODO: src/compare/mod.rs — run prompt on local + Anthropic, show side by side
        }

        Some(Command::Bench { last }) => {
            println!("Benchmark (last {last} ratings)");
            // TODO: src/bench/mod.rs — query SQLite ratings, print summary
        }

        Some(Command::Train { project }) => {
            println!("Training on: {project}");
            // TODO: src/commands/train.rs — generate data, run LoRA fine-tune
        }

        // No subcommand was given - check if there's a one-shot prompt,
        // otherwise open the TUI.
        None => match cli.prompt {
            Some(prompt) => {
                // Load the default inference config (finds model in ~/.params/models/)
                // then run the prompt, streaming tokens to stdout as they arrive.
                let config = inference::InferenceConfig::default_config()?;
                inference::run(&config, &prompt)?;
            }
            None => {
                println!("Opening TUI...");
                // TODO: src/tui/mod.rs — launch Ratatui app
            }
        },
    }

    // Returning Ok(()) signals success. If any TODO above returns an Err,
    // the ? operator will propagate it here and Rust will print the error message.
    Ok(())
}