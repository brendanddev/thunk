use std::io;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use tracing::{debug, info, warn};

use crate::commands::CommandRegistry;
use crate::error::Result;
use crate::events::InferenceEvent;
use crate::inference::{SessionCommand, SessionRuntimeOptions};

use super::commands::{handle_command_input, SlashJobOutcome};
use super::renderer::Renderer;
use super::state::{AppState, DirtySections};
use super::TuiOptions;

const ACTIVE_FRAME_INTERVAL: Duration = Duration::from_millis(33);
const SLOW_FRAME_INTERVAL: Duration = Duration::from_millis(66);
const IDLE_FRAME_INTERVAL: Duration = Duration::from_millis(180);

struct RenderScheduler {
    last_draw_at: Instant,
    heavy_frame_streak: u8,
}

impl RenderScheduler {
    fn new() -> Self {
        Self {
            last_draw_at: Instant::now() - IDLE_FRAME_INTERVAL,
            heavy_frame_streak: 0,
        }
    }

    fn desired_interval(&self, state: &AppState) -> Duration {
        if state.is_generating || state.current_trace.is_some() || !state.is_ready() {
            if self.heavy_frame_streak >= 3 {
                SLOW_FRAME_INTERVAL
            } else {
                ACTIVE_FRAME_INTERVAL
            }
        } else {
            IDLE_FRAME_INTERVAL
        }
    }

    fn should_draw(&self, state: &AppState) -> bool {
        if state.has_dirty_sections() {
            return true;
        }

        if state.is_generating || state.current_trace.is_some() || !state.is_ready() {
            return self.last_draw_at.elapsed() >= self.desired_interval(state);
        }

        false
    }

    fn poll_timeout(&self, state: &AppState) -> Duration {
        if state.has_dirty_sections() {
            Duration::ZERO
        } else {
            self.desired_interval(state)
                .saturating_sub(self.last_draw_at.elapsed())
        }
    }

    fn record_draw(&mut self, elapsed: Duration) {
        self.last_draw_at = Instant::now();
        if elapsed > Duration::from_millis(24) {
            self.heavy_frame_streak = self.heavy_frame_streak.saturating_add(1);
        } else {
            self.heavy_frame_streak = 0;
        }
    }
}

pub(crate) fn run_app(stdout: &mut io::Stdout, options: TuiOptions) -> Result<()> {
    let mut state = AppState::new();
    let mut command_registry = CommandRegistry::load();
    let (width, height) = crossterm::terminal::size()?;
    let mut renderer = Renderer::new(width, height);
    let mut render_scheduler = RenderScheduler::new();

    let (token_tx, token_rx) = mpsc::channel::<InferenceEvent>();
    let (prompt_tx, prompt_rx) = mpsc::channel::<SessionCommand>();
    let (slash_tx, slash_rx) = mpsc::channel::<SlashJobOutcome>();

    let token_tx_clone = token_tx.clone();
    thread::spawn(move || {
        crate::inference::model_thread_with_options(
            prompt_rx,
            token_tx_clone,
            SessionRuntimeOptions {
                no_resume: options.no_resume,
            },
        );
    });

    loop {
        while let Ok(event) = token_rx.try_recv() {
            match event {
                InferenceEvent::Ready => {
                    state.set_status("ready");
                }
                InferenceEvent::SessionLoaded {
                    session,
                    display_messages,
                    saved_at,
                } => {
                    state.restore_session(session, display_messages, saved_at);
                }
                InferenceEvent::SessionStatus(session) => {
                    state.set_session_info(session);
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
                    state.mark_dirty(
                        DirtySections::SIDEBAR | DirtySections::HEADER | DirtySections::APPROVAL,
                    );
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
                InferenceEvent::SystemMessage(message) => {
                    state.add_system_message(&message);
                }
                InferenceEvent::MemoryState(snapshot) => {
                    state.set_memory_snapshot(snapshot);
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
                SlashJobOutcome::WorkflowEdit {
                    finished_trace,
                    contexts,
                    path,
                    edits,
                } => {
                    info!(label = finished_trace.label.as_str(), "trace.finished");
                    state.apply_trace(finished_trace);
                    for context in contexts {
                        state.add_user_message(&context);
                        let _ = prompt_tx.send(SessionCommand::InjectUserContext(context));
                    }
                    let _ = prompt_tx.send(SessionCommand::RequestFileEdit { path, edits });
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

        if render_scheduler.should_draw(&state) {
            state.tick();
            let stats = renderer.render(stdout, &mut state)?;
            render_scheduler.record_draw(Duration::from_millis(stats.frame_time_ms));
            if state.debug_logging_enabled {
                debug!(
                    elapsed_ms = stats.frame_time_ms,
                    dirty = ?state.dirty_sections(),
                    changed_cells = stats.changed_cells,
                    changed_runs = stats.changed_runs,
                    cache_hits = stats.cache_hits,
                    cache_misses = stats.cache_misses,
                    symbol_pool_size = stats.symbol_pool_size,
                    "tui.render"
                );
            }
            state.clear_dirty_sections();
        }

        if event::poll(render_scheduler.poll_timeout(&state))? {
            match event::read()? {
                Event::Key(key) => match (key.code, key.modifiers) {
                    (KeyCode::Char('c'), KeyModifiers::CONTROL)
                    | (KeyCode::Char('q'), KeyModifiers::CONTROL) => {
                        return Ok(());
                    }
                    (KeyCode::Char('r'), KeyModifiers::CONTROL) => {
                        if state.is_ready() && !state.is_generating {
                            state.reverse_search_cycle();
                        }
                    }
                    (KeyCode::Esc, _) => {
                        if state.is_reverse_search_active() {
                            state.cancel_reverse_search();
                        }
                    }
                    (KeyCode::Enter, _) if state.is_reverse_search_active() => {
                        state.accept_reverse_search();
                    }
                    (KeyCode::Backspace, _) if state.is_reverse_search_active() => {
                        state.reverse_search_backspace();
                    }
                    (KeyCode::Char(c), KeyModifiers::NONE | KeyModifiers::SHIFT)
                        if state.is_reverse_search_active() =>
                    {
                        state.reverse_search_push_char(c);
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
                        state.clear_input();
                    }
                    (KeyCode::Char('w'), KeyModifiers::CONTROL) => {
                        state.delete_word_before();
                    }
                    (KeyCode::Char('o'), KeyModifiers::CONTROL) => {
                        if state.is_ready() && !state.is_generating {
                            state.toggle_focused_collapsible();
                        }
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
                    (KeyCode::Up, KeyModifiers::ALT) => {
                        state.recall_previous_input();
                    }
                    (KeyCode::Down, KeyModifiers::ALT) => {
                        state.recall_next_input();
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
                    (KeyCode::Char('['), _) => {
                        if state.is_ready() && !state.is_generating && state.input.is_empty() {
                            state.focus_prev_visible_collapsible();
                        } else {
                            state.insert_char('[');
                        }
                    }
                    (KeyCode::Char(']'), _) => {
                        if state.is_ready() && !state.is_generating && state.input.is_empty() {
                            state.focus_next_visible_collapsible();
                        } else {
                            state.insert_char(']');
                        }
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
                Event::Resize(width, height) => {
                    renderer.resize(width, height);
                    crossterm::execute!(
                        stdout,
                        crossterm::terminal::Clear(crossterm::terminal::ClearType::All)
                    )?;
                    renderer.invalidate();
                    state.mark_dirty(DirtySections::ALL);
                }
                Event::Paste(text) => {
                    let clean = AppState::normalized_paste(&text);
                    state.insert_str(&clean);
                }
                _ => {}
            }
        }
    }
}
