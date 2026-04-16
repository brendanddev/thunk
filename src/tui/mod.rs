mod app;
mod input;
mod render;
mod state;

use std::io::{self, IsTerminal};

use crossterm::{
    cursor::{Hide, SetCursorStyle, Show},
    event::{DisableBracketedPaste, EnableBracketedPaste},
    execute,
    terminal::{
        disable_raw_mode, enable_raw_mode, Clear, ClearType, EnterAlternateScreen,
        LeaveAlternateScreen, SetTitle,
    },
};

use crate::app::config::Config;
use crate::app::paths::AppPaths;
use crate::app::{AppError, Result};
use crate::runtime::Runtime;

/// Main entry point for the TUI, handling terminal setup and teardown
pub fn run(config: &Config, paths: &AppPaths, runtime: &mut Runtime) -> Result<()> {
    if !io::stdout().is_terminal() {
        return Err(AppError::Tui(
            "The TUI requires an interactive terminal (stdout is not a TTY).".to_string(),
        ));
    }

    // Check if the terminal is "dumb", which does not support necessary features for the TUI
    if std::env::var("TERM").as_deref() == Ok("dumb") {
        return Err(AppError::Tui(
            "The TUI cannot run with TERM=dumb.".to_string(),
        ));
    }

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(
        stdout,
        EnterAlternateScreen,
        Clear(ClearType::All),
        EnableBracketedPaste,
        Hide,
        SetCursorStyle::SteadyBar,
        SetTitle(config.app.name.as_str())
    )?;

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        app::run_app(&mut stdout, config, paths, runtime)
    }));

    disable_raw_mode()?;
    execute!(
        stdout,
        LeaveAlternateScreen,
        DisableBracketedPaste,
        Show,
        SetCursorStyle::DefaultUserShape,
        SetTitle(config.app.name.as_str())
    )?;

    match result {
        Ok(result) => result,
        Err(_) => Err(AppError::Tui(
            "The TUI panicked unexpectedly after startup. Terminal state was restored."
                .to_string(),
        )),
    }
}
