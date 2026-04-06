mod app;
mod commands;
mod format;
mod renderer;
mod state;

use std::io::{self, IsTerminal};

use crossterm::{
    cursor::{Hide, Show},
    event::{DisableBracketedPaste, EnableBracketedPaste},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use tracing::info;

use crate::error::{ParamsError, Result};

#[derive(Clone, Copy, Default)]
pub struct TuiOptions {
    pub no_resume: bool,
}

#[allow(dead_code)]
pub fn run() -> Result<()> {
    run_with_options(TuiOptions::default())
}

pub fn run_with_options(options: TuiOptions) -> Result<()> {
    info!("tui starting");
    if !io::stdout().is_terminal() {
        return Err(ParamsError::Config(
            "The params TUI requires an interactive terminal (stdout is not a TTY).".to_string(),
        ));
    }

    if std::env::var("TERM").as_deref() == Ok("dumb") {
        return Err(ParamsError::Config(
            "The params TUI cannot run with TERM=dumb. Launch it in a real terminal, or set TERM=xterm-256color before starting params.".to_string(),
        ));
    }

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableBracketedPaste, Hide)?;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        app::run_app(&mut stdout, options)
    }));
    disable_raw_mode()?;
    execute!(stdout, LeaveAlternateScreen, DisableBracketedPaste, Show)?;
    info!("tui exiting");
    match result {
        Ok(result) => result,
        Err(_) => Err(ParamsError::Inference(
            "The TUI panicked unexpectedly after startup. Terminal state was restored.".to_string(),
        )),
    }
}
