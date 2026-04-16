use std::io;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};

use crate::app::config::Config;
use crate::app::paths::AppPaths;
use crate::app::Result;
use crate::runtime::{Runtime, RuntimeEvent, RuntimeRequest};

use super::render::render;
use super::state::AppState;

/// Runs the TUI app, handling rendering and user input
pub(crate) fn run_app(
    stdout: &mut io::Stdout,
    config: &Config,
    paths: &AppPaths,
    runtime: &mut Runtime,
) -> Result<()> {
    let mut state = AppState::new(config, paths);

    loop {
        render(stdout, &state)?;

        if state.should_quit {
            return Ok(());
        }

        // Poll for events with a timeout to allow periodic rendering updates
        if event::poll(Duration::from_millis(100))? {
            match event::read()? {
                Event::Key(key) => handle_key_event(stdout, &mut state, runtime, key)?,
                Event::Paste(text) => state.insert_str(&text),
                Event::Resize(_, _) => {}
                _ => {}
            }
        }
    }
}

/// Handles a key event, updating the app state accordingly
fn handle_key_event(
    stdout: &mut io::Stdout,
    state: &mut AppState,
    runtime: &mut Runtime,
    key: KeyEvent,
) -> Result<()> {
    match (key.code, key.modifiers) {
        (KeyCode::Char('c'), KeyModifiers::CONTROL)
        | (KeyCode::Char('q'), KeyModifiers::CONTROL) => {
            state.should_quit = true;
        }
        (KeyCode::Enter, _) => {
            if let Some(prompt) = state.submit_input() {
                state.add_user_message(prompt.clone());
                let mut render_error = None;
                runtime.handle(RuntimeRequest::Submit { text: prompt }, &mut |event| {
                    if render_error.is_some() {
                        return;
                    }

                    apply_runtime_event(state, event);
                    if let Err(error) = render(stdout, state) {
                        render_error = Some(error);
                    }
                });

                if let Some(error) = render_error {
                    return Err(error);
                }
            }
        }
        (KeyCode::Backspace, _) => state.delete_char_before(),
        (KeyCode::Left, _) => state.cursor_left(),
        (KeyCode::Right, _) => state.cursor_right(),
        (KeyCode::Home, _) => state.cursor_home(),
        (KeyCode::End, _) => state.cursor_end(),
        (KeyCode::Char(c), KeyModifiers::NONE | KeyModifiers::SHIFT) => state.insert_char(c),
        _ => {}
    }

    Ok(())
}

fn apply_runtime_event(state: &mut AppState, event: RuntimeEvent) {
    match event {
        RuntimeEvent::ActivityChanged(activity) => state.set_status(activity.label()),
        RuntimeEvent::AssistantMessageStarted => state.begin_assistant_message(),
        RuntimeEvent::AssistantMessageChunk(chunk) => state.append_assistant_chunk(&chunk),
        RuntimeEvent::AssistantMessageFinished => {}
        RuntimeEvent::Failed { message } => {
            state.set_status("error");
            state.add_system_message(message);
        }
    }
}
