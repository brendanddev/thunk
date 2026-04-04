mod app;
mod commands;
mod format;
mod render;
mod state;

use std::io::{self, IsTerminal};

use crossterm::{
    event::{DisableBracketedPaste, EnableBracketedPaste},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use tracing::info;

use crate::error::{ParamsError, Result};

pub fn run() -> Result<()> {
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
    execute!(stdout, EnterAlternateScreen, EnableBracketedPaste)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| app::run_app(&mut terminal)));
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableBracketedPaste
    )?;
    terminal.show_cursor()?;
    info!("tui exiting");
    match result {
        Ok(result) => result,
        Err(_) => Err(ParamsError::Inference(
            "The TUI panicked unexpectedly after startup. Terminal state was restored.".to_string(),
        )),
    }
}
