use std::io;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};

use crate::app::config::Config;
use crate::app::paths::AppPaths;
use crate::app::Result;
use crate::runtime::{AnswerSource, Runtime, RuntimeEvent, RuntimeRequest};

use super::commands;
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
            if let Some(input) = state.submit_input() {
                // Check for slash commands before forwarding to the runtime.
                if let Some(cmd) = commands::parse(&input) {
                    handle_command(stdout, state, runtime, cmd)?;
                } else {
                    submit_to_runtime(stdout, state, runtime, input)?;
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

fn submit_to_runtime(
    stdout: &mut io::Stdout,
    state: &mut AppState,
    runtime: &mut Runtime,
    prompt: String,
) -> Result<()> {
    state.add_user_message(prompt.clone());
    let mut render_error = None;

    runtime.handle(RuntimeRequest::Submit { text: prompt }, &mut |event| {
        if render_error.is_some() {
            return;
        }
        apply_runtime_event(state, event);
        if let Err(e) = render(stdout, state) {
            render_error = Some(e);
        }
    });

    if let Some(e) = render_error {
        return Err(e);
    }

    Ok(())
}

fn handle_command(
    _stdout: &mut io::Stdout,
    state: &mut AppState,
    runtime: &mut Runtime,
    cmd: commands::Command,
) -> Result<()> {
    match cmd {
        commands::Command::Help => {
            state.add_system_message(
                "Commands: /help — show this message  |  /clear — clear history  |  /quit — exit",
            );
        }
        commands::Command::Quit => {
            state.should_quit = true;
        }
        commands::Command::Clear => {
            state.clear_messages();
            runtime.handle(RuntimeRequest::Reset, &mut |_| {});
        }
    }
    Ok(())
}

fn apply_runtime_event(state: &mut AppState, event: RuntimeEvent) {
    match event {
        RuntimeEvent::ActivityChanged(activity) => state.set_status(activity.label()),
        RuntimeEvent::AssistantMessageStarted => state.begin_assistant_message(),
        RuntimeEvent::AssistantMessageChunk(chunk) => state.append_assistant_chunk(&chunk),
        RuntimeEvent::AssistantMessageFinished => {}
        RuntimeEvent::ToolCallStarted { name } => {
            state.add_tool_message(format!("tool: {name}"));
        }
        RuntimeEvent::ToolCallFinished { name, success } => {
            if !success {
                state.add_tool_message(format!("tool failed: {name}"));
            }
        }
        RuntimeEvent::AnswerReady(source) => {
            if let AnswerSource::ToolLimitReached = source {
                state.add_system_message("Tool limit reached. Response may be incomplete.");
            }
        }
        RuntimeEvent::Failed { message } => {
            state.set_status("error");
            state.add_system_message(message);
        }
    }
}
