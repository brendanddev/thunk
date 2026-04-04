use std::io;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use ratatui::{backend::CrosstermBackend, Terminal};
use tracing::{info, warn};

use crate::commands::CommandRegistry;
use crate::error::Result;
use crate::events::InferenceEvent;
use crate::inference::SessionCommand;

use super::commands::{handle_command_input, SlashJobOutcome};
use super::render::draw;
use super::state::AppState;

pub(crate) fn run_app(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    let mut state = AppState::new();
    let mut command_registry = CommandRegistry::load();

    let (token_tx, token_rx) = mpsc::channel::<InferenceEvent>();
    let (prompt_tx, prompt_rx) = mpsc::channel::<SessionCommand>();
    let (slash_tx, slash_rx) = mpsc::channel::<SlashJobOutcome>();

    let token_tx_clone = token_tx.clone();
    thread::spawn(move || {
        crate::inference::model_thread(prompt_rx, token_tx_clone);
    });

    let frame_duration = Duration::from_millis(16);

    loop {
        let frame_start = Instant::now();
        state.tick();
        terminal.draw(|frame| draw(frame, &mut state))?;

        while let Ok(event) = token_rx.try_recv() {
            match event {
                InferenceEvent::Ready => {
                    state.set_status("ready");
                }
                InferenceEvent::SessionRestored {
                    display_messages,
                    saved_at,
                } => {
                    state.restore_session(display_messages, saved_at);
                }
                InferenceEvent::BackendName(name) => {
                    state.set_backend_name(name);
                }
                InferenceEvent::GenerationStarted {
                    label,
                    show_placeholder,
                } => {
                    state.start_generation(&label, show_placeholder);
                }
                InferenceEvent::Trace(trace) => {
                    state.apply_trace(trace);
                }
                InferenceEvent::Token(token) => {
                    state.append_token(&token);
                }
                InferenceEvent::ToolCall(call) => {
                    state.clear_pending_action();
                    state.last_tool_call = Some(call);
                    state.status = "running tool...".to_string();
                }
                InferenceEvent::ContextMessage(message) => {
                    state.add_user_message(&message);
                }
                InferenceEvent::ReflectionEnabled(enabled) => {
                    state.set_reflection_enabled(enabled);
                }
                InferenceEvent::EcoEnabled(enabled) => {
                    state.set_eco_enabled(enabled);
                }
                InferenceEvent::DebugLoggingEnabled(enabled) => {
                    state.set_debug_logging_enabled(enabled);
                }
                InferenceEvent::Budget(update) => {
                    state.update_budget(
                        update.prompt_tokens,
                        update.completion_tokens,
                        update.total_tokens,
                        update.estimated_cost_usd,
                    );
                }
                InferenceEvent::Cache(update) => {
                    state.update_cache(
                        update.last_hit,
                        update.hits,
                        update.misses,
                        update.tokens_saved,
                    );
                }
                InferenceEvent::PendingAction(action) => {
                    state.set_pending_action(action);
                }
                InferenceEvent::Done => {
                    state.clear_pending_action();
                    state.finish_response();
                }
                InferenceEvent::Error(e) => {
                    state.clear_pending_action();
                    state.add_error(&e);
                }
            }
        }

        while let Ok(outcome) = slash_rx.try_recv() {
            match outcome {
                SlashJobOutcome::Trace(trace) => {
                    state.apply_trace(trace);
                }
                SlashJobOutcome::Context {
                    finished_trace,
                    context,
                } => {
                    info!(label = finished_trace.label.as_str(), "trace.finished");
                    state.apply_trace(finished_trace);
                    state.add_user_message(&context);
                    let _ = prompt_tx.send(SessionCommand::InjectUserContext(context));
                    state.finish_response();
                }
                SlashJobOutcome::ContextBatch {
                    finished_trace,
                    contexts,
                } => {
                    info!(label = finished_trace.label.as_str(), "trace.finished");
                    state.apply_trace(finished_trace);
                    for context in contexts {
                        state.add_user_message(&context);
                        let _ = prompt_tx.send(SessionCommand::InjectUserContext(context));
                    }
                    state.finish_response();
                }
                SlashJobOutcome::WorkflowPrompt {
                    finished_trace,
                    contexts,
                    prompt,
                } => {
                    info!(label = finished_trace.label.as_str(), "trace.finished");
                    state.apply_trace(finished_trace);
                    for context in contexts {
                        state.add_user_message(&context);
                        let _ = prompt_tx.send(SessionCommand::InjectUserContext(context));
                    }
                    state.add_user_message(&prompt);
                    let _ = prompt_tx.send(SessionCommand::SubmitUser(prompt));
                    state.start_generation("generating...", true);
                }
                SlashJobOutcome::WorkflowShell {
                    finished_trace,
                    contexts,
                    command,
                } => {
                    info!(label = finished_trace.label.as_str(), "trace.finished");
                    state.apply_trace(finished_trace);
                    for context in contexts {
                        state.add_user_message(&context);
                        let _ = prompt_tx.send(SessionCommand::InjectUserContext(context));
                    }
                    let _ = prompt_tx.send(SessionCommand::RequestShellCommand(command));
                }
                SlashJobOutcome::WorkflowWrite {
                    finished_trace,
                    contexts,
                    path,
                    content,
                } => {
                    info!(label = finished_trace.label.as_str(), "trace.finished");
                    state.apply_trace(finished_trace);
                    for context in contexts {
                        state.add_user_message(&context);
                        let _ = prompt_tx.send(SessionCommand::InjectUserContext(context));
                    }
                    let _ = prompt_tx.send(SessionCommand::RequestFileWrite { path, content });
                }
                SlashJobOutcome::Error {
                    failed_trace,
                    message,
                } => {
                    warn!(
                        label = failed_trace.label.as_str(),
                        message = message.as_str(),
                        "trace.failed"
                    );
                    state.apply_trace(failed_trace);
                    state.add_system_message(&message);
                    state.finish_response();
                }
            }
        }

        let elapsed = frame_start.elapsed();
        let remaining = frame_duration.saturating_sub(elapsed);

        if event::poll(remaining)? {
            match event::read()? {
                Event::Key(key) => match (key.code, key.modifiers) {
                    (KeyCode::Char('c'), KeyModifiers::CONTROL)
                    | (KeyCode::Char('q'), KeyModifiers::CONTROL) => {
                        return Ok(());
                    }
                    (KeyCode::Enter, KeyModifiers::SHIFT) => {
                        if !state.is_generating && state.is_ready() {
                            state.insert_newline();
                        }
                    }
                    (KeyCode::Enter, _) => {
                        if !state.input.is_empty() && !state.is_generating && state.is_ready() {
                            let prompt = state.submit_input();

                            if state.pending_action.is_some()
                                && !matches!(prompt.as_str(), "/approve" | "/reject")
                                && !prompt.starts_with("/approve ")
                                && !prompt.starts_with("/reject ")
                            {
                                state.add_system_message(
                                    "An action is awaiting approval. Use /approve or /reject first.",
                                );
                                continue;
                            }

                            if prompt.starts_with('/') {
                                handle_command_input(
                                    &prompt,
                                    &mut state,
                                    &prompt_tx,
                                    &slash_tx,
                                    &mut command_registry,
                                );
                            } else {
                                state.add_user_message(&prompt);
                                let _ = prompt_tx.send(SessionCommand::SubmitUser(prompt.clone()));
                                state.start_generation("generating...", true);
                            }
                        }
                    }
                    (KeyCode::Char('j'), KeyModifiers::CONTROL) => {
                        if !state.is_generating && state.is_ready() {
                            state.insert_newline();
                        }
                    }
                    (KeyCode::Backspace, KeyModifiers::NONE) => {
                        state.delete_char_before();
                    }
                    (KeyCode::Backspace, KeyModifiers::ALT) => {
                        state.delete_word_before();
                    }
                    (KeyCode::Left, _) => {
                        state.cursor_left();
                    }
                    (KeyCode::Right, _) => {
                        state.cursor_right();
                    }
                    (KeyCode::Home, _) => {
                        state.cursor_home();
                    }
                    (KeyCode::End, _) => {
                        state.cursor_end();
                    }
                    (KeyCode::Char('a'), KeyModifiers::CONTROL) => {
                        state.cursor_home();
                    }
                    (KeyCode::Char('e'), KeyModifiers::CONTROL) => {
                        state.cursor_end();
                    }
                    (KeyCode::Char('u'), KeyModifiers::CONTROL) => {
                        state.input.clear();
                        state.cursor = 0;
                    }
                    (KeyCode::Char('w'), KeyModifiers::CONTROL) => {
                        state.delete_word_before();
                    }
                    (KeyCode::Char('y'), KeyModifiers::CONTROL) => {
                        if let Some(id) = state.pending_action_id() {
                            let _ = prompt_tx.send(SessionCommand::ApproveAction(id));
                            state.mark_pending_action_submitted("processing approval");
                        }
                    }
                    (KeyCode::Char('n'), KeyModifiers::CONTROL) => {
                        if let Some(id) = state.pending_action_id() {
                            let _ = prompt_tx.send(SessionCommand::RejectAction(id));
                            state.mark_pending_action_submitted("processing rejection");
                        }
                    }
                    (KeyCode::Up, _) => {
                        state.scroll_up(1);
                    }
                    (KeyCode::Down, _) => {
                        state.scroll_down(1);
                    }
                    (KeyCode::PageUp, _) => {
                        state.scroll_up(10);
                    }
                    (KeyCode::PageDown, _) => {
                        state.scroll_down(10);
                    }
                    (KeyCode::Tab, KeyModifiers::NONE) => {
                        if state.is_ready() && !state.is_generating {
                            let autocomplete_names = command_registry.autocomplete_names();
                            state.autocomplete_command(&autocomplete_names, false);
                        }
                    }
                    (KeyCode::BackTab, _) => {
                        if state.is_ready() && !state.is_generating {
                            let autocomplete_names = command_registry.autocomplete_names();
                            state.autocomplete_command(&autocomplete_names, true);
                        }
                    }
                    (KeyCode::Char(c), KeyModifiers::NONE)
                    | (KeyCode::Char(c), KeyModifiers::SHIFT) => {
                        state.insert_char(c);
                    }
                    _ => {}
                },
                Event::Paste(text) => {
                    let clean = AppState::normalized_paste(&text);
                    state.insert_str(&clean);
                }
                _ => {}
            }
        }
    }
}
