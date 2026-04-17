use std::io;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};

use crate::app::config::Config;
use crate::app::AppContext;
use crate::app::paths::AppPaths;
use crate::app::Result;
use crate::runtime::{AnswerSource, RuntimeEvent, RuntimeRequest};

use super::commands;
use super::render::render;
use super::state::AppState;

pub(crate) fn run_app(
    stdout: &mut io::Stdout,
    config: &Config,
    paths: &AppPaths,
    app: &mut AppContext,
) -> Result<()> {
    let mut state = AppState::new(config, paths);

    loop {
        render(stdout, &state)?;

        if state.should_quit {
            return Ok(());
        }

        if event::poll(Duration::from_millis(100))? {
            match event::read()? {
                Event::Key(key) => handle_key_event(stdout, &mut state, app, key)?,
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
    app: &mut AppContext,
    key: KeyEvent,
) -> Result<()> {
    match (key.code, key.modifiers) {
        (KeyCode::Char('c'), KeyModifiers::CONTROL)
        | (KeyCode::Char('q'), KeyModifiers::CONTROL) => {
            state.should_quit = true;
        }
        (KeyCode::Enter, _) => {
            if let Some(input) = state.submit_input() {
                if let Some(cmd) = commands::parse(&input) {
                    handle_command(state, app, cmd)?;
                } else {
                    submit_to_app(stdout, state, app, input)?;
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

fn submit_to_app(
    stdout: &mut io::Stdout,
    state: &mut AppState,
    app: &mut AppContext,
    prompt: String,
) -> Result<()> {
    state.add_user_message(prompt.clone());
    let mut render_error = None;

    let handle_result = app.handle(RuntimeRequest::Submit { text: prompt }, &mut |event| {
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

    if let Err(e) = handle_result {
        apply_runtime_event(state, RuntimeEvent::Failed { message: e.to_string() });
    }

    Ok(())
}

fn handle_command(
    state: &mut AppState,
    app: &mut AppContext,
    cmd: commands::Command,
) -> Result<()> {
    match cmd {
        commands::Command::Help => {
            state.add_system_message(
                "Commands: /help — show this message  |  /clear — clear history  |  /quit — exit  |  /approve — confirm pending action  |  /reject — cancel pending action",
            );
        }
        commands::Command::Quit => {
            state.should_quit = true;
        }
        commands::Command::Clear => {
            state.clear_messages();
            if let Err(e) = app.reset() {
                state.add_system_message(format!("session reset failed: {e}"));
            }
        }
        commands::Command::Approve => {
            if let Err(e) = app.handle(RuntimeRequest::Approve, &mut |event| {
                apply_runtime_event(state, event);
            }) {
                apply_runtime_event(state, RuntimeEvent::Failed { message: e.to_string() });
            }
        }
        commands::Command::Reject => {
            if let Err(e) = app.handle(RuntimeRequest::Reject, &mut |event| {
                apply_runtime_event(state, event);
            }) {
                apply_runtime_event(state, RuntimeEvent::Failed { message: e.to_string() });
            }
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
        RuntimeEvent::ToolCallFinished { name, summary } => match summary {
            Some(s) => state.add_tool_message(s),
            None => state.add_tool_message(format!("tool failed: {name}")),
        },
        RuntimeEvent::AnswerReady(source) => {
            if let AnswerSource::ToolLimitReached = source {
                state.add_system_message("Tool limit reached. Response may be incomplete.");
            }
        }
        RuntimeEvent::Failed { message } => {
            state.set_status("error");
            state.add_system_message(message);
        }
        RuntimeEvent::ApprovalRequired(pending) => {
            state.add_system_message(format!(
                "[approval required] {} — type /approve to confirm or /reject to cancel",
                pending.summary
            ));
            state.set_status("awaiting approval");
        }
        // Advisory only — absorbed by the logging layer before reaching here.
        RuntimeEvent::BackendTiming { .. } => {}
    }
}
