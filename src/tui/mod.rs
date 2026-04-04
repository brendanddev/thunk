// src/tui/mod.rs

mod state;

use std::io::{self, IsTerminal};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame, Terminal,
};
use tracing::{info, warn};

use crate::commands::{
    builtin_command_specs, resolve_builtin_command, BuiltinKind, CommandRegistry, CustomCommand,
    CustomCommandBody, CustomCommandStep,
};
use crate::error::{ParamsError, Result};
use crate::events::{InferenceEvent, PendingActionKind, ProgressStatus, ProgressTrace};
use crate::inference::SessionCommand;
use state::{AppState, Role};

// Spinner frames
const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const MAX_INPUT_VISIBLE_ROWS: usize = 8;

// How often to advance the spinner (every N ticks at ~60fps = ~100ms per frame)
const SPINNER_SPEED: u64 = 6;

enum SlashJobOutcome {
    Trace(ProgressTrace),
    Context {
        finished_trace: ProgressTrace,
        context: String,
    },
    ContextBatch {
        finished_trace: ProgressTrace,
        contexts: Vec<String>,
    },
    WorkflowPrompt {
        finished_trace: ProgressTrace,
        contexts: Vec<String>,
        prompt: String,
    },
    WorkflowShell {
        finished_trace: ProgressTrace,
        contexts: Vec<String>,
        command: String,
    },
    WorkflowWrite {
        finished_trace: ProgressTrace,
        contexts: Vec<String>,
        path: String,
        content: String,
    },
    Error {
        failed_trace: ProgressTrace,
        message: String,
    },
}

fn truncate_for_width(value: &str, max_chars: usize) -> String {
    let len = value.chars().count();
    if len <= max_chars {
        return value.to_string();
    }
    let keep = max_chars.saturating_sub(1);
    let truncated: String = value.chars().take(keep).collect();
    format!("{truncated}…")
}

fn format_compact_count(value: usize) -> String {
    if value >= 1_000_000 {
        format!("{:.1}m", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}k", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

fn format_cost(value: Option<f64>) -> String {
    match value {
        Some(v) if v >= 1.0 => format!("${v:.2}"),
        Some(v) => format!("${v:.4}"),
        None => "n/a".to_string(),
    }
}

fn format_hit_rate(hits: usize, misses: usize) -> String {
    let total = hits.saturating_add(misses);
    if total == 0 {
        "n/a".to_string()
    } else {
        format!("{:.0}%", (hits as f64 / total as f64) * 100.0)
    }
}

fn format_duration(duration: Duration) -> String {
    let total_ms = duration.as_millis();
    if total_ms < 10_000 {
        let secs = total_ms as f64 / 1000.0;
        format!("{secs:.1}s")
    } else {
        let total_secs = duration.as_secs();
        if total_secs < 60 {
            format!("{total_secs}s")
        } else if total_secs < 3600 {
            let mins = total_secs / 60;
            let secs = total_secs % 60;
            format!("{mins}m {secs:02}s")
        } else {
            let hours = total_secs / 3600;
            let mins = (total_secs % 3600) / 60;
            format!("{hours}h {mins:02}m")
        }
    }
}

fn wrap_plain_text(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![String::new()];
    }
    if text.is_empty() {
        return vec![String::new()];
    }

    let chars: Vec<char> = text.chars().collect();
    let mut wrapped = Vec::new();
    let mut start = 0;

    while start < chars.len() {
        let end = (start + width).min(chars.len());
        wrapped.push(chars[start..end].iter().collect());
        start = end;
    }

    wrapped
}

fn push_wrapped_styled(lines: &mut Vec<Line>, text: &str, style: Style, width: usize) {
    for part in wrap_plain_text(text, width) {
        lines.push(Line::from(Span::styled(part, style)));
    }
}

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
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_app(&mut terminal)));
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

fn run_app(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    let mut state = AppState::new();
    let mut command_registry = CommandRegistry::load();

    let (token_tx, token_rx) = mpsc::channel::<InferenceEvent>();
    let (prompt_tx, prompt_rx) = mpsc::channel::<SessionCommand>();
    let (slash_tx, slash_rx) = mpsc::channel::<SlashJobOutcome>();

    let token_tx_clone = token_tx.clone();
    thread::spawn(move || {
        crate::inference::model_thread(prompt_rx, token_tx_clone);
    });

    // Target 60fps — 16ms per frame
    let frame_duration = Duration::from_millis(16);

    loop {
        let frame_start = Instant::now();

        // Increment tick for animations
        state.tick();

        // Draw current frame
        terminal.draw(|frame| draw(frame, &mut state))?;

        // Drain all pending inference events
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

        // Calculate remaining time in frame budget
        let elapsed = frame_start.elapsed();
        let remaining = frame_duration.saturating_sub(elapsed);

        // Poll for keyboard events for the rest of the frame budget
        if event::poll(remaining)? {
            match event::read()? {
                Event::Key(key) => {
                    match (key.code, key.modifiers) {
                        // Quit
                        (KeyCode::Char('c'), KeyModifiers::CONTROL)
                        | (KeyCode::Char('q'), KeyModifiers::CONTROL) => {
                            return Ok(());
                        }

                        // Shift+Enter — newline when the terminal reports it
                        (KeyCode::Enter, KeyModifiers::SHIFT) => {
                            if !state.is_generating && state.is_ready() {
                                state.insert_newline();
                            }
                        }

                        // Submit
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

                                // Check for slash commands before sending to inference.
                                // These run tools directly and inject results into context
                                // without requiring the model to format a tool call tag.
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
                                    let _ =
                                        prompt_tx.send(SessionCommand::SubmitUser(prompt.clone()));
                                    state.start_generation("generating...", true);
                                }
                            }
                        }

                        // Ctrl+J — newline fallback for multiline input
                        (KeyCode::Char('j'), KeyModifiers::CONTROL) => {
                            if !state.is_generating && state.is_ready() {
                                state.insert_newline();
                            }
                        }

                        // Backspace
                        (KeyCode::Backspace, KeyModifiers::NONE) => {
                            state.delete_char_before();
                        }

                        // Alt+Backspace — delete whole word
                        (KeyCode::Backspace, KeyModifiers::ALT) => {
                            state.delete_word_before();
                        }

                        // Cursor movement
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

                        // Ctrl+A — go to start
                        (KeyCode::Char('a'), KeyModifiers::CONTROL) => {
                            state.cursor_home();
                        }

                        // Ctrl+E — go to end
                        (KeyCode::Char('e'), KeyModifiers::CONTROL) => {
                            state.cursor_end();
                        }

                        // Ctrl+U — clear entire input
                        (KeyCode::Char('u'), KeyModifiers::CONTROL) => {
                            state.input.clear();
                            state.cursor = 0;
                        }

                        // Ctrl+W — delete word before cursor
                        (KeyCode::Char('w'), KeyModifiers::CONTROL) => {
                            state.delete_word_before();
                        }

                        // Ctrl+Y — approve pending action
                        (KeyCode::Char('y'), KeyModifiers::CONTROL) => {
                            if let Some(id) = state.pending_action_id() {
                                let _ = prompt_tx.send(SessionCommand::ApproveAction(id));
                                state.mark_pending_action_submitted("processing approval");
                            }
                        }

                        // Ctrl+N — reject pending action
                        (KeyCode::Char('n'), KeyModifiers::CONTROL) => {
                            if let Some(id) = state.pending_action_id() {
                                let _ = prompt_tx.send(SessionCommand::RejectAction(id));
                                state.mark_pending_action_submitted("processing rejection");
                            }
                        }

                        // Scroll
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

                        // Tab / Shift+Tab — autocomplete slash commands
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

                        // Regular character input
                        (KeyCode::Char(c), KeyModifiers::NONE)
                        | (KeyCode::Char(c), KeyModifiers::SHIFT) => {
                            state.insert_char(c);
                        }

                        _ => {}
                    }
                }

                // Handle paste events
                Event::Paste(text) => {
                    let clean = AppState::normalized_paste(&text);
                    state.insert_str(&clean);
                }

                _ => {}
            }
        }
    }
}

fn draw(frame: &mut Frame, state: &mut AppState) {
    let size = frame.area();
    let horizontal = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(24), Constraint::Min(0)])
        .split(size);
    draw_sidebar(frame, state, horizontal[0]);
    draw_main(frame, state, horizontal[1]);
}

fn draw_sidebar(frame: &mut Frame, state: &AppState, area: Rect) {
    let border_color = if state.pending_action.is_some() || state.is_generating {
        Color::Yellow
    } else {
        Color::DarkGray
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color))
        .title(Span::styled(
            " params ",
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        ));

    // Spinner for loading/generating states
    let spinner_frame = SPINNER[(state.tick / SPINNER_SPEED) as usize % SPINNER.len()];

    let status_line = if !state.model_ready {
        format!("{spinner_frame} {}", state.status)
    } else if let Some(ref trace) = state.current_trace {
        format!("{spinner_frame} {}", truncate_for_width(trace, 18))
    } else if state.pending_action.is_some() {
        state.status.clone()
    } else if state.is_generating {
        format!("{spinner_frame} generating")
    } else {
        state.status.clone()
    };

    let status_color =
        if state.pending_action.is_some() || state.is_generating || !state.model_ready {
            Color::Yellow
        } else {
            Color::Green
        };

    // Truncate backend name to fit sidebar
    let backend_display = truncate_for_width(&state.backend_name, 18);
    let current_turn_duration = state.current_turn_duration();
    let last_work_duration = state.last_work_duration();
    let mut items = vec![
        ListItem::new(Line::from(vec![
            Span::styled("● ", Style::default().fg(status_color)),
            Span::styled(&status_line, Style::default().fg(status_color)),
        ])),
        ListItem::new(Line::from("")),
        ListItem::new(Line::from(vec![Span::styled(
            "backend",
            Style::default().fg(Color::DarkGray),
        )])),
        ListItem::new(Line::from(vec![Span::styled(
            format!("  {backend_display}"),
            Style::default().fg(Color::Cyan),
        )])),
        ListItem::new(Line::from("")),
        ListItem::new(Line::from(vec![
            Span::styled("msgs  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                state.message_count().to_string(),
                Style::default().fg(Color::White),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("tok   ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format_compact_count(state.total_tokens),
                Style::default().fg(Color::White),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("cost  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format_cost(state.estimated_cost_usd),
                Style::default().fg(Color::White),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("turn  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                current_turn_duration
                    .map(format_duration)
                    .unwrap_or_else(|| "n/a".to_string()),
                Style::default().fg(if current_turn_duration.is_some() {
                    Color::Yellow
                } else {
                    Color::White
                }),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("last  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                last_work_duration
                    .map(format_duration)
                    .unwrap_or_else(|| "n/a".to_string()),
                Style::default().fg(Color::White),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("refl  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                if state.reflection_enabled {
                    "on"
                } else {
                    "off"
                },
                Style::default().fg(Color::White),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("eco   ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                if state.eco_enabled { "on" } else { "off" },
                Style::default().fg(Color::White),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("dlog  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                if state.debug_logging_enabled {
                    "on"
                } else {
                    "off"
                },
                Style::default().fg(Color::White),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("cache ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                match state.last_cache_hit {
                    Some(true) => "hit",
                    Some(false) => "miss",
                    None => "n/a",
                },
                Style::default().fg(Color::White),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("rate  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format_hit_rate(state.cache_hits, state.cache_misses),
                Style::default().fg(Color::White),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("saved ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format_compact_count(state.tokens_saved),
                Style::default().fg(Color::White),
            ),
        ])),
        ListItem::new(Line::from("")),
        ListItem::new(Line::from(vec![Span::styled(
            "activity",
            Style::default().fg(Color::DarkGray),
        )])),
    ];

    if let Some(ref trace) = state.current_trace {
        items.push(ListItem::new(Line::from(vec![
            Span::styled("  now ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                truncate_for_width(trace, 14),
                Style::default().fg(Color::Yellow),
            ),
        ])));
    } else {
        items.push(ListItem::new(Line::from(vec![
            Span::styled("  now ", Style::default().fg(Color::DarkGray)),
            Span::styled("idle", Style::default().fg(Color::DarkGray)),
        ])));
    }

    if let Some(ref call) = state.last_tool_call {
        let truncated = truncate_for_width(call, 14);
        items.push(ListItem::new(Line::from(vec![
            Span::styled("  tool", Style::default().fg(Color::DarkGray)),
            Span::styled(format!(" {truncated}"), Style::default().fg(Color::Yellow)),
        ])));
    }

    if let Some(ref pending) = state.pending_action {
        let truncated = truncate_for_width(&pending.title, 14);
        items.push(ListItem::new(Line::from(vec![
            Span::styled("  wait", Style::default().fg(Color::DarkGray)),
            Span::styled(format!(" {truncated}"), Style::default().fg(Color::Yellow)),
        ])));
    }

    for trace in state.recent_traces.iter().take(3) {
        let icon = if trace.success { "  ✓ " } else { "  ✕ " };
        let color = if trace.success {
            Color::Green
        } else {
            Color::Red
        };
        items.push(ListItem::new(Line::from(vec![
            Span::styled(icon, Style::default().fg(Color::DarkGray)),
            Span::styled(
                truncate_for_width(&trace.label, 14),
                Style::default().fg(color),
            ),
        ])));
    }

    items.extend([
        ListItem::new(Line::from("")),
        ListItem::new(Line::from(vec![Span::styled(
            "─────────────────────",
            Style::default().fg(Color::DarkGray),
        )])),
        ListItem::new(Line::from("")),
        ListItem::new(Line::from(vec![
            Span::styled("enter  ", Style::default().fg(Color::DarkGray)),
            Span::styled("send", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("S-enter", Style::default().fg(Color::DarkGray)),
            Span::styled(" newline", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("^j     ", Style::default().fg(Color::DarkGray)),
            Span::styled("newline", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("↑↓ pg  ", Style::default().fg(Color::DarkGray)),
            Span::styled("scroll", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("←→     ", Style::default().fg(Color::DarkGray)),
            Span::styled("cursor", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("^u     ", Style::default().fg(Color::DarkGray)),
            Span::styled("clear", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("^y/^n  ", Style::default().fg(Color::DarkGray)),
            Span::styled("approve/reject", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("^q     ", Style::default().fg(Color::DarkGray)),
            Span::styled("quit", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from("")),
        ListItem::new(Line::from(vec![Span::styled(
            "─────────────────────",
            Style::default().fg(Color::DarkGray),
        )])),
        ListItem::new(Line::from("")),
        ListItem::new(Line::from(vec![
            Span::styled("/read  ", Style::default().fg(Color::DarkGray)),
            Span::styled("<path>", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("/ls    ", Style::default().fg(Color::DarkGray)),
            Span::styled("[path]", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("/search", Style::default().fg(Color::DarkGray)),
            Span::styled(" <q>", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("/git   ", Style::default().fg(Color::DarkGray)),
            Span::styled(" <cmd>", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("/diag  ", Style::default().fg(Color::DarkGray)),
            Span::styled(" <file>", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("/hover ", Style::default().fg(Color::DarkGray)),
            Span::styled(" <f:l:c>", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("/def   ", Style::default().fg(Color::DarkGray)),
            Span::styled(" <f:l:c>", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("/lcheck", Style::default().fg(Color::DarkGray)),
            Span::styled(" status", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("/fetch ", Style::default().fg(Color::DarkGray)),
            Span::styled(" <url>", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("/run   ", Style::default().fg(Color::DarkGray)),
            Span::styled(" <cmd>", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("/write ", Style::default().fg(Color::DarkGray)),
            Span::styled(" <p> <text>", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("/reflect", Style::default().fg(Color::DarkGray)),
            Span::styled(" on|off", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("/eco   ", Style::default().fg(Color::DarkGray)),
            Span::styled(" on|off", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("/debug-log", Style::default().fg(Color::DarkGray)),
            Span::styled(" on|off", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("/commands", Style::default().fg(Color::DarkGray)),
            Span::styled(" list", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("write_file", Style::default().fg(Color::DarkGray)),
            Span::styled(" via model", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("/approve", Style::default().fg(Color::DarkGray)),
            Span::styled(" /reject", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("/clear ", Style::default().fg(Color::DarkGray)),
            Span::styled("history", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("/clear-cache", Style::default().fg(Color::DarkGray)),
            Span::styled(" reset", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("/clear-debug-log", Style::default().fg(Color::DarkGray)),
            Span::styled(" reset", Style::default().fg(Color::DarkGray)),
        ])),
    ]);

    frame.render_widget(List::new(items).block(block), area);
}

fn draw_main(frame: &mut Frame, state: &mut AppState, area: Rect) {
    let input_height = input_area_height(state, area.width.saturating_sub(2) as usize);
    if state.has_pending_action() {
        let card_height = pending_action_height(state, area.width.saturating_sub(2) as usize);
        let vertical = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(0),
                Constraint::Length(card_height),
                Constraint::Length(input_height),
            ])
            .split(area);
        draw_chat(frame, state, vertical[0]);
        draw_pending_action(frame, state, vertical[1]);
        draw_input(frame, state, vertical[2]);
    } else {
        let vertical = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(0), Constraint::Length(input_height)])
            .split(area);
        draw_chat(frame, state, vertical[0]);
        draw_input(frame, state, vertical[1]);
    }
}

fn draw_chat(frame: &mut Frame, state: &mut AppState, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(
            " conversation ",
            Style::default().fg(Color::DarkGray),
        ));

    let inner_width = area.width.saturating_sub(2) as usize;
    let visible_height = area.height.saturating_sub(2) as usize;
    let mut lines: Vec<Line> = Vec::new();

    // Welcome message when no messages yet
    if state.messages.is_empty() {
        if state.model_ready {
            lines.push(Line::from(""));
            push_wrapped_styled(
                &mut lines,
                "  ready. type a message below.",
                Style::default().fg(Color::DarkGray),
                inner_width,
            );
        } else {
            lines.push(Line::from(""));
            push_wrapped_styled(
                &mut lines,
                "  loading model...",
                Style::default().fg(Color::DarkGray),
                inner_width,
            );
        }
    }

    for msg in &state.messages {
        match msg.role {
            Role::User => {
                // User badge
                lines.push(Line::from(vec![
                    Span::raw("  "),
                    Span::styled(
                        " you ",
                        Style::default()
                            .fg(Color::Black)
                            .bg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]));
                // User message content
                for line in msg.content.lines() {
                    push_wrapped_styled(
                        &mut lines,
                        &format!("  {line}"),
                        Style::default().fg(Color::White),
                        inner_width,
                    );
                }
                if msg.content.is_empty() {
                    lines.push(Line::from(Span::raw("  ")));
                }
                lines.push(Line::from(""));
            }
            Role::Assistant => {
                // Assistant badge
                lines.push(Line::from(vec![
                    Span::raw("  "),
                    Span::styled(
                        " params ",
                        Style::default()
                            .fg(Color::Black)
                            .bg(Color::Magenta)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]));
                // Assistant message content
                for line in msg.content.lines() {
                    push_wrapped_styled(
                        &mut lines,
                        &format!("  {line}"),
                        Style::default().fg(Color::Gray),
                        inner_width,
                    );
                }
                if msg.content.is_empty() {
                    // Show a subtle cursor while generating
                    lines.push(Line::from(Span::styled(
                        "  ▌",
                        Style::default().fg(Color::DarkGray),
                    )));
                }
                lines.push(Line::from(""));
            }
            Role::System => {
                // System messages are shown as subtle info lines
                // They are NOT sent to the model — just UI feedback
                for line in msg.content.lines() {
                    push_wrapped_styled(
                        &mut lines,
                        &format!("  ● {line}"),
                        Style::default().fg(Color::DarkGray),
                        inner_width,
                    );
                }
                lines.push(Line::from(""));
            }
        }
    }

    let total_display_lines = lines.len();
    let max_scroll = total_display_lines.saturating_sub(visible_height);

    state.max_scroll = max_scroll;
    state.scroll_offset = state.scroll_offset.min(max_scroll);

    let end = total_display_lines.saturating_sub(state.scroll_offset);
    let start = end.saturating_sub(visible_height);
    let visible_lines = lines[start..end].to_vec();

    let paragraph = Paragraph::new(Text::from(visible_lines)).block(block);
    frame.render_widget(paragraph, area);
}

fn draw_input(frame: &mut Frame, state: &AppState, area: Rect) {
    let is_multiline = state.input.contains('\n');
    let (title, border_color) = if !state.model_ready {
        (" loading... ", Color::DarkGray)
    } else if state.is_generating {
        (" generating... ", Color::Yellow)
    } else if is_multiline {
        (" message (multiline) ", Color::Cyan)
    } else {
        (" message ", Color::DarkGray)
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color))
        .title(Span::styled(title, Style::default().fg(border_color)));

    let inner_width = area.width.saturating_sub(2) as usize;
    let max_visible_rows = max_input_content_rows(area);
    let (visible_rows, cursor_row, cursor_col) =
        state.input_display_lines(inner_width, max_visible_rows);
    let dimmed = state.is_generating || !state.model_ready;
    let mut lines = Vec::new();

    for (row_idx, row_text) in visible_rows.iter().enumerate() {
        let mut row_spans = Vec::new();
        let style = if dimmed {
            Style::default().fg(Color::DarkGray)
        } else {
            Style::default().fg(Color::White)
        };

        if row_idx == cursor_row && !dimmed {
            let safe_col = cursor_col.min(row_text.chars().count());
            let before: String = row_text.chars().take(safe_col).collect();
            let after: String = row_text.chars().skip(safe_col).collect();
            if after.is_empty() {
                row_spans.push(Span::styled(before, style));
                row_spans.push(Span::styled(
                    "█",
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::SLOW_BLINK),
                ));
            } else {
                let mut after_chars = after.chars();
                let at_cursor = after_chars.next().unwrap_or(' ');
                let rest: String = after_chars.collect();
                row_spans.push(Span::styled(before, style));
                row_spans.push(Span::styled(
                    at_cursor.to_string(),
                    Style::default().fg(Color::Black).bg(Color::White),
                ));
                row_spans.push(Span::styled(rest, style));
            }
        } else if row_idx == cursor_row && dimmed && row_text.is_empty() {
            row_spans.push(Span::styled(String::new(), style));
        } else {
            row_spans.push(Span::styled(row_text.clone(), style));
        }

        lines.push(Line::from(row_spans));
    }

    if let Some(hint) = state.autocomplete_hint() {
        lines.push(Line::from(Span::styled(
            format!("  {hint}"),
            Style::default().fg(Color::DarkGray),
        )));
    }
    if !dimmed {
        let multiline_hint = if is_multiline {
            "  Enter sends • Shift+Enter / Ctrl+J add newline"
        } else {
            "  Enter sends • Shift+Enter / Ctrl+J for multiline"
        };
        lines.push(Line::from(Span::styled(
            multiline_hint,
            Style::default().fg(Color::DarkGray),
        )));
    }

    let paragraph = Paragraph::new(Text::from(lines)).block(block);
    frame.render_widget(paragraph, area);
}

fn input_area_height(state: &AppState, width: usize) -> u16 {
    let content_rows = state.input_content_rows(width).min(MAX_INPUT_VISIBLE_ROWS);
    let hint_rows = if state.autocomplete_hint().is_some() {
        2
    } else {
        1
    };
    (content_rows + hint_rows + 2).max(3) as u16
}

fn max_input_content_rows(area: Rect) -> usize {
    area.height.saturating_sub(2) as usize
}

fn pending_action_height(state: &AppState, width: usize) -> u16 {
    let Some(pending) = state.pending_action.as_ref() else {
        return 0;
    };

    let mut lines = 6usize;
    lines += wrap_plain_text(&pending.title, width).len();
    lines += 1;
    lines += wrap_plain_text(&pending.inspection.summary, width).len();
    if !pending.inspection.targets.is_empty() {
        lines += wrap_plain_text(
            &format!("Targets: {}", pending.inspection.targets.join(", ")),
            width,
        )
        .len();
    }
    if !pending.inspection.segments.is_empty() {
        lines += wrap_plain_text(
            &format!("Segments: {}", pending.inspection.segments.join(" | ")),
            width,
        )
        .len();
    }
    if !pending.inspection.network_targets.is_empty() {
        lines += wrap_plain_text(
            &format!("Network: {}", pending.inspection.network_targets.join(", ")),
            width,
        )
        .len();
    }
    for reason in &pending.inspection.reasons {
        lines += wrap_plain_text(&format!("- {reason}"), width).len();
    }

    let preview_max_lines = match pending.kind {
        PendingActionKind::ShellCommand => 2,
        PendingActionKind::FileWrite => 8,
    };
    lines += pending_preview_lines(&pending.preview, width, preview_max_lines).len();
    lines += 2;

    lines.clamp(10, 20) as u16
}

fn pending_preview_lines(preview: &str, width: usize, max_lines: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for raw_line in preview.lines() {
        lines.extend(wrap_plain_text(raw_line, width));
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    if lines.len() > max_lines {
        let mut truncated = lines[..max_lines.saturating_sub(1)].to_vec();
        truncated.push("[preview truncated]".to_string());
        truncated
    } else {
        lines
    }
}

fn draw_pending_action(frame: &mut Frame, state: &AppState, area: Rect) {
    let Some(pending) = state.pending_action.as_ref() else {
        return;
    };

    let (accent, title) = match pending.inspection.risk {
        crate::safety::RiskLevel::Low => (Color::Cyan, " pending approval "),
        crate::safety::RiskLevel::Medium => (Color::Yellow, " pending approval "),
        crate::safety::RiskLevel::High => (Color::Red, " pending approval "),
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(accent))
        .title(Span::styled(
            title,
            Style::default().fg(accent).add_modifier(Modifier::BOLD),
        ));

    let inner_width = area.width.saturating_sub(2) as usize;
    let mut lines = Vec::new();

    push_wrapped_styled(
        &mut lines,
        &pending.title,
        Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD),
        inner_width,
    );
    lines.push(Line::from(Span::styled(
        format!(
            "Policy: {} / {} risk",
            pending.inspection.decision, pending.inspection.risk
        ),
        Style::default().fg(accent),
    )));
    push_wrapped_styled(
        &mut lines,
        &format!("Summary: {}", pending.inspection.summary),
        Style::default().fg(Color::Gray),
        inner_width,
    );

    if !pending.inspection.targets.is_empty() {
        push_wrapped_styled(
            &mut lines,
            &format!("Targets: {}", pending.inspection.targets.join(", ")),
            Style::default().fg(Color::DarkGray),
            inner_width,
        );
    }
    if !pending.inspection.segments.is_empty() {
        push_wrapped_styled(
            &mut lines,
            &format!("Segments: {}", pending.inspection.segments.join(" | ")),
            Style::default().fg(Color::DarkGray),
            inner_width,
        );
    }
    if !pending.inspection.network_targets.is_empty() {
        push_wrapped_styled(
            &mut lines,
            &format!("Network: {}", pending.inspection.network_targets.join(", ")),
            Style::default().fg(Color::DarkGray),
            inner_width,
        );
    }
    if !pending.inspection.reasons.is_empty() {
        for reason in &pending.inspection.reasons {
            push_wrapped_styled(
                &mut lines,
                &format!("- {reason}"),
                Style::default().fg(Color::DarkGray),
                inner_width,
            );
        }
    }

    lines.push(Line::from(""));

    let preview_max_lines = match pending.kind {
        PendingActionKind::ShellCommand => 2,
        PendingActionKind::FileWrite => 8,
    };
    let preview_style = match pending.kind {
        PendingActionKind::ShellCommand => Style::default().fg(Color::White),
        PendingActionKind::FileWrite => Style::default().fg(Color::Gray),
    };
    for line in pending_preview_lines(&pending.preview, inner_width, preview_max_lines) {
        lines.push(Line::from(Span::styled(line, preview_style)));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "Ctrl+Y approve, Ctrl+N reject, or use /approve /reject",
        Style::default().fg(Color::DarkGray),
    )));

    let paragraph = Paragraph::new(Text::from(lines)).block(block);
    frame.render_widget(paragraph, area);
}

/// Handles slash commands typed by the user.
///
/// Slash commands run tools directly without going through the model.
/// The result is injected into the conversation as a user message so
/// the model can reason about the actual content on the next message.
///
/// Available commands:
///   /read <path>     — read a file into context
///   /ls [path]       — list directory contents
///   /search <query>  — search across source files
///   /git [command]   — git status/diff/log context
///   /help            — show available commands
///   /clear           — clear the conversation history
/// Strip ANSI escape codes and non-printable characters from a string.
/// Prevents terminal corruption when file contents are rendered in the TUI.
fn sanitize_for_display(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            // Skip ANSI escape sequences (ESC [ ... m)
            '' => {
                // Skip everything until we hit a letter (end of escape sequence)
                while let Some(&next) = chars.peek() {
                    chars.next();
                    if next.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
            // Keep newlines, tabs, and printable chars
            '\n' | '\t' => result.push(c),
            c if c.is_control() => {} // skip other control chars
            c => result.push(c),
        }
    }
    result
}

fn decode_slash_write_content(raw: &str) -> String {
    let mut output = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.peek().copied() {
                Some('n') => {
                    chars.next();
                    output.push('\n');
                }
                Some('t') => {
                    chars.next();
                    output.push('\t');
                }
                Some('\\') => {
                    chars.next();
                    output.push('\\');
                }
                Some('"') => {
                    chars.next();
                    output.push('"');
                }
                _ => output.push(ch),
            }
        } else {
            output.push(ch);
        }
    }

    output
}

fn run_tool_immediate<T: crate::tools::Tool>(
    tool: T,
    arg: &str,
) -> std::result::Result<String, String> {
    match crate::tools::Tool::run(&tool, arg) {
        Ok(crate::tools::ToolRunResult::Immediate(output)) => Ok(output),
        Ok(crate::tools::ToolRunResult::RequiresApproval(_)) => {
            Err("requested approval unexpectedly".to_string())
        }
        Err(e) => Err(e.to_string()),
    }
}

type SlashWork = Box<dyn FnOnce() -> std::result::Result<String, String> + Send>;

struct SlashContextSpec {
    running_status: String,
    started_trace: String,
    finished_trace: String,
    failed_trace: String,
    context_prefix: String,
    work: SlashWork,
}

fn make_trace(status: ProgressStatus, label: impl Into<String>, persist: bool) -> ProgressTrace {
    ProgressTrace {
        status,
        label: label.into(),
        persist,
    }
}

fn parse_command_parts(input: &str) -> (String, &str) {
    let parts: Vec<&str> = input.splitn(2, ' ').collect();
    let cmd = parts[0].to_lowercase();
    let arg = parts.get(1).map(|s| s.trim()).unwrap_or("");
    (cmd, arg)
}

fn spawn_slash_context_job(
    state: &mut AppState,
    slash_tx: &mpsc::Sender<SlashJobOutcome>,
    spec: SlashContextSpec,
    persist: bool,
) {
    state.start_generation(&spec.running_status, false);
    info!(label = spec.started_trace.as_str(), "trace.started");
    state.apply_trace(make_trace(
        ProgressStatus::Started,
        spec.started_trace.clone(),
        false,
    ));
    let tx = slash_tx.clone();
    thread::spawn(move || {
        let outcome = match (spec.work)() {
            Ok(output) => {
                let safe = sanitize_for_display(&output);
                let context = format!("{}\n\n{safe}", spec.context_prefix);
                SlashJobOutcome::Context {
                    finished_trace: make_trace(
                        ProgressStatus::Finished,
                        spec.finished_trace,
                        persist,
                    ),
                    context,
                }
            }
            Err(error) => SlashJobOutcome::Error {
                failed_trace: make_trace(ProgressStatus::Failed, spec.failed_trace, persist),
                message: error,
            },
        };
        let _ = tx.send(outcome);
    });
}

fn build_context_spec(cmd: &str, arg: &str) -> Option<SlashContextSpec> {
    let canonical = resolve_builtin_command(cmd)?.canonical;
    match canonical {
        "/read" => {
            if arg.is_empty() {
                return None;
            }
            let arg_owned = arg.to_string();
            Some(SlashContextSpec {
                running_status: "reading file...".to_string(),
                started_trace: format!("reading {arg}"),
                finished_trace: format!("loaded {arg}"),
                failed_trace: format!("read failed for {arg}"),
                context_prefix: "I've loaded this file for context:".to_string(),
                work: Box::new(move || {
                    run_tool_immediate(crate::tools::ReadFile, &arg_owned)
                        .map_err(|e| format!("error reading {arg_owned}: {e}"))
                }),
            })
        }
        "/ls" => {
            let path = if arg.is_empty() { "." } else { arg };
            let path_owned = path.to_string();
            Some(SlashContextSpec {
                running_status: "listing directory...".to_string(),
                started_trace: format!("listing {path}"),
                finished_trace: format!("listed {path}"),
                failed_trace: format!("list failed for {path}"),
                context_prefix: "Directory listing:".to_string(),
                work: Box::new(move || {
                    run_tool_immediate(crate::tools::ListDir, &path_owned)
                        .map_err(|e| format!("error listing {path_owned}: {e}"))
                }),
            })
        }
        "/search" => {
            if arg.is_empty() {
                return None;
            }
            let arg_owned = arg.to_string();
            Some(SlashContextSpec {
                running_status: "searching code...".to_string(),
                started_trace: format!("searching for {arg}"),
                finished_trace: format!("search complete for {arg}"),
                failed_trace: format!("search failed for {arg}"),
                context_prefix: "Search results:".to_string(),
                work: Box::new(move || {
                    run_tool_immediate(crate::tools::SearchCode, &arg_owned)
                        .map_err(|e| format!("error searching: {e}"))
                }),
            })
        }
        "/git" => {
            let git_arg = if arg.is_empty() { "status" } else { arg };
            let git_arg_owned = git_arg.to_string();
            Some(SlashContextSpec {
                running_status: "running git...".to_string(),
                started_trace: format!("running git {git_arg}"),
                finished_trace: format!("git: {git_arg}"),
                failed_trace: format!("git failed for {git_arg}"),
                context_prefix: format!("Git context ({git_arg}):"),
                work: Box::new(move || {
                    run_tool_immediate(crate::tools::GitTool, &git_arg_owned)
                        .map_err(|e| format!("git error: {e}"))
                }),
            })
        }
        "/diag" => {
            if arg.is_empty() {
                return None;
            }
            let arg_owned = arg.to_string();
            Some(SlashContextSpec {
                running_status: "running diagnostics...".to_string(),
                started_trace: format!("running diagnostics for {arg}"),
                finished_trace: format!("diagnostics ready for {arg}"),
                failed_trace: format!("diagnostics failed for {arg}"),
                context_prefix: "LSP diagnostics:".to_string(),
                work: Box::new(move || {
                    run_tool_immediate(crate::tools::LspDiagnosticsTool, &arg_owned)
                        .map_err(|e| format!("diagnostics error: {e}"))
                }),
            })
        }
        "/hover" => {
            if arg.is_empty() {
                return None;
            }
            let arg_owned = arg.to_string();
            Some(SlashContextSpec {
                running_status: "loading hover info...".to_string(),
                started_trace: format!("loading hover for {arg}"),
                finished_trace: format!("hover ready for {arg}"),
                failed_trace: format!("hover failed for {arg}"),
                context_prefix: "LSP hover:".to_string(),
                work: Box::new(move || {
                    run_tool_immediate(crate::tools::LspHoverTool, &arg_owned)
                        .map_err(|e| format!("hover error: {e}"))
                }),
            })
        }
        "/def" => {
            if arg.is_empty() {
                return None;
            }
            let arg_owned = arg.to_string();
            Some(SlashContextSpec {
                running_status: "resolving definition...".to_string(),
                started_trace: format!("resolving definition for {arg}"),
                finished_trace: format!("definition ready for {arg}"),
                failed_trace: format!("definition failed for {arg}"),
                context_prefix: "LSP definition:".to_string(),
                work: Box::new(move || {
                    run_tool_immediate(crate::tools::LspDefinitionTool, &arg_owned)
                        .map_err(|e| format!("definition error: {e}"))
                }),
            })
        }
        "/lcheck" => Some(SlashContextSpec {
            running_status: "checking rust lsp...".to_string(),
            started_trace: "checking rust lsp".to_string(),
            finished_trace: "rust lsp check complete".to_string(),
            failed_trace: "rust lsp check failed".to_string(),
            context_prefix: "LSP check:".to_string(),
            work: Box::new(move || Ok(crate::tools::rust_lsp_health_report())),
        }),
        "/fetch" => {
            if arg.is_empty() {
                return None;
            }
            let arg_owned = arg.to_string();
            Some(SlashContextSpec {
                running_status: "fetching webpage...".to_string(),
                started_trace: format!("fetching {arg}"),
                finished_trace: format!("fetched {arg}"),
                failed_trace: format!("fetch failed for {arg}"),
                context_prefix: "Fetched web context:".to_string(),
                work: Box::new(move || {
                    run_tool_immediate(crate::tools::FetchUrlTool, &arg_owned)
                        .map_err(|e| format!("fetch error: {e}"))
                }),
            })
        }
        _ => None,
    }
}

fn custom_help_text() -> String {
    let mut lines = vec!["built-in slash commands:".to_string()];
    for spec in builtin_command_specs() {
        let mut line = format!("  {:<18} — {}", spec.usage, spec.description);
        if !spec.aliases.is_empty() {
            line.push_str(&format!(" (aliases: {})", spec.aliases.join(", ")));
        }
        lines.push(line);
    }
    lines.push("".to_string());
    lines.push("input: Enter sends • Shift+Enter or Ctrl+J insert newlines".to_string());
    lines.push("custom commands: /commands list • /commands reload".to_string());
    lines.join("\n")
}

fn format_custom_commands_list(registry: &CommandRegistry) -> String {
    let mut lines = vec!["built-ins:".to_string()];
    for spec in builtin_command_specs() {
        lines.push(format!("  {:<12} — {}", spec.canonical, spec.description));
    }
    lines.push(String::new());
    lines.push("custom commands:".to_string());
    if registry.list().is_empty() {
        lines.push("  (none loaded)".to_string());
    } else {
        for command in registry.list() {
            let usage = command
                .usage
                .as_ref()
                .map(|value| format!(" — {value}"))
                .unwrap_or_default();
            lines.push(format!(
                "  {:<12} [{}] — {}{}",
                command.name, command.origin, command.description, usage
            ));
        }
    }
    lines.join("\n")
}

fn execute_custom_template(
    command: &CustomCommand,
    args: &[&str],
    state: &mut AppState,
    prompt_tx: &mpsc::Sender<SessionCommand>,
) {
    let CustomCommandBody::Prompt(template) = &command.body else {
        return;
    };
    let expanded = crate::commands::expand_positional_args(template, args);
    info!(
        command = command.name.as_str(),
        origin = command.origin.as_str(),
        "custom command started"
    );
    state.start_generation(&format!("running {}...", command.name), false);
    state.apply_trace(make_trace(
        ProgressStatus::Started,
        format!("running {}", command.name),
        false,
    ));
    state.apply_trace(make_trace(
        ProgressStatus::Finished,
        format!("prepared {}", command.name),
        false,
    ));
    state.add_user_message(&expanded);
    let _ = prompt_tx.send(SessionCommand::SubmitUser(expanded));
    state.start_generation("generating...", true);
}

fn execute_custom_workflow(
    command: CustomCommand,
    args: Vec<&str>,
    state: &mut AppState,
    slash_tx: &mpsc::Sender<SlashJobOutcome>,
) {
    let CustomCommandBody::Workflow(steps) = command.body.clone() else {
        return;
    };

    let expanded_steps = steps
        .into_iter()
        .map(|step| match step {
            CustomCommandStep::Slash(text) => {
                CustomCommandStep::Slash(crate::commands::expand_positional_args(&text, &args))
            }
            CustomCommandStep::Prompt(text) => {
                CustomCommandStep::Prompt(crate::commands::expand_positional_args(&text, &args))
            }
        })
        .collect::<Vec<_>>();

    let workflow_name = command.name.clone();
    let origin = command.origin;

    state.start_generation(&format!("running {}...", workflow_name), false);
    state.apply_trace(make_trace(
        ProgressStatus::Started,
        format!("running {workflow_name}"),
        false,
    ));
    info!(
        command = workflow_name.as_str(),
        origin = origin.as_str(),
        "custom command started"
    );

    let tx = slash_tx.clone();
    thread::spawn(move || {
        let mut contexts = Vec::new();

        for step in expanded_steps {
            match step {
                CustomCommandStep::Slash(slash) => {
                    let (cmd, arg) = parse_command_parts(&slash);
                    let Some(spec_meta) = resolve_builtin_command(&cmd) else {
                        let _ = tx.send(SlashJobOutcome::Error {
                            failed_trace: make_trace(
                                ProgressStatus::Failed,
                                format!("custom command failed: {workflow_name}"),
                                false,
                            ),
                            message: format!("custom workflow references unsupported step: {cmd}"),
                        });
                        return;
                    };

                    match spec_meta.kind {
                        BuiltinKind::Context => {
                            let Some(spec) = build_context_spec(&cmd, arg) else {
                                let _ = tx.send(SlashJobOutcome::Error {
                                    failed_trace: make_trace(
                                        ProgressStatus::Failed,
                                        format!("custom command failed: {workflow_name}"),
                                        false,
                                    ),
                                    message: format!("invalid usage for workflow step `{slash}`"),
                                });
                                return;
                            };

                            let _ = tx.send(SlashJobOutcome::Trace(make_trace(
                                ProgressStatus::Started,
                                spec.started_trace.clone(),
                                false,
                            )));
                            match (spec.work)() {
                                Ok(output) => {
                                    let safe = sanitize_for_display(&output);
                                    contexts.push(format!("{}\n\n{safe}", spec.context_prefix));
                                    let _ = tx.send(SlashJobOutcome::Trace(make_trace(
                                        ProgressStatus::Finished,
                                        spec.finished_trace,
                                        false,
                                    )));
                                }
                                Err(error) => {
                                    let _ = tx.send(SlashJobOutcome::Trace(make_trace(
                                        ProgressStatus::Failed,
                                        spec.failed_trace,
                                        false,
                                    )));
                                    let _ = tx.send(SlashJobOutcome::Error {
                                        failed_trace: make_trace(
                                            ProgressStatus::Failed,
                                            format!("custom command failed: {workflow_name}"),
                                            false,
                                        ),
                                        message: error,
                                    });
                                    return;
                                }
                            }
                        }
                        BuiltinKind::Mutating => {
                            let final_trace = make_trace(
                                ProgressStatus::Finished,
                                format!("completed {workflow_name}"),
                                false,
                            );
                            match spec_meta.canonical {
                                "/run" => {
                                    let _ = tx.send(SlashJobOutcome::WorkflowShell {
                                        finished_trace: final_trace,
                                        contexts,
                                        command: arg.to_string(),
                                    });
                                }
                                "/write" => {
                                    let Some((path, raw_content)) = arg.split_once(' ') else {
                                        let _ = tx.send(SlashJobOutcome::Error {
                                            failed_trace: make_trace(
                                                ProgressStatus::Failed,
                                                format!("custom command failed: {workflow_name}"),
                                                false,
                                            ),
                                            message: "workflow /write step must use `/write <path> <content>`".to_string(),
                                        });
                                        return;
                                    };
                                    let _ = tx.send(SlashJobOutcome::WorkflowWrite {
                                        finished_trace: final_trace,
                                        contexts,
                                        path: path.trim().to_string(),
                                        content: decode_slash_write_content(raw_content.trim()),
                                    });
                                }
                                _ => {}
                            }
                            return;
                        }
                        BuiltinKind::Session | BuiltinKind::Discovery => {
                            let _ = tx.send(SlashJobOutcome::Error {
                                failed_trace: make_trace(
                                    ProgressStatus::Failed,
                                    format!("custom command failed: {workflow_name}"),
                                    false,
                                ),
                                message: format!(
                                    "workflow step `{}` is not supported in custom commands",
                                    spec_meta.canonical
                                ),
                            });
                            return;
                        }
                    }
                }
                CustomCommandStep::Prompt(prompt) => {
                    let _ = tx.send(SlashJobOutcome::WorkflowPrompt {
                        finished_trace: make_trace(
                            ProgressStatus::Finished,
                            format!("completed {workflow_name}"),
                            false,
                        ),
                        contexts,
                        prompt,
                    });
                    return;
                }
            }
        }

        let _ = tx.send(SlashJobOutcome::ContextBatch {
            finished_trace: make_trace(
                ProgressStatus::Finished,
                format!("completed {workflow_name}"),
                false,
            ),
            contexts,
        });
    });
}

fn handle_command_input(
    input: &str,
    state: &mut AppState,
    prompt_tx: &mpsc::Sender<SessionCommand>,
    slash_tx: &mpsc::Sender<SlashJobOutcome>,
    command_registry: &mut CommandRegistry,
) {
    let (cmd, arg) = parse_command_parts(input);
    info!(command = cmd.as_str(), "slash command received");

    if resolve_builtin_command(&cmd).is_some() {
        handle_builtin_slash_command(&cmd, arg, state, prompt_tx, slash_tx, command_registry);
        return;
    }

    let Some(command) = command_registry.resolve(&cmd).cloned() else {
        state.add_system_message(&format!(
            "unknown command: {cmd}. Type /help for available commands."
        ));
        return;
    };

    let args = if arg.is_empty() {
        Vec::new()
    } else {
        arg.split_whitespace().collect::<Vec<_>>()
    };
    match &command.body {
        CustomCommandBody::Prompt(_) => execute_custom_template(&command, &args, state, prompt_tx),
        CustomCommandBody::Workflow(_) => execute_custom_workflow(command, args, state, slash_tx),
    }
}

fn handle_builtin_slash_command(
    cmd: &str,
    arg: &str,
    state: &mut AppState,
    prompt_tx: &mpsc::Sender<SessionCommand>,
    slash_tx: &mpsc::Sender<SlashJobOutcome>,
    command_registry: &mut CommandRegistry,
) {
    let canonical = resolve_builtin_command(cmd)
        .map(|spec| spec.canonical)
        .unwrap_or(cmd);

    match canonical {
        "/read" | "/search" | "/diag" | "/hover" | "/def" | "/fetch" => {
            let Some(spec) = build_context_spec(canonical, arg) else {
                let usage = resolve_builtin_command(canonical)
                    .map(|spec| spec.usage)
                    .unwrap_or("/help");
                state.add_system_message(&format!("Usage: {usage}"));
                return;
            };
            spawn_slash_context_job(state, slash_tx, spec, true);
        }
        "/ls" | "/git" | "/lcheck" => {
            if let Some(spec) = build_context_spec(canonical, arg) {
                spawn_slash_context_job(state, slash_tx, spec, true);
            }
        }
        "/run" => {
            if arg.is_empty() {
                state.add_system_message("Usage: /run <command>");
                return;
            }
            let _ = prompt_tx.send(SessionCommand::RequestShellCommand(arg.to_string()));
        }
        "/write" => {
            let Some((path, raw_content)) = arg.split_once(' ') else {
                state
                    .add_system_message("Usage: /write <path> <content>. Use \\n for line breaks.");
                return;
            };
            if path.trim().is_empty() || raw_content.trim().is_empty() {
                state
                    .add_system_message("Usage: /write <path> <content>. Use \\n for line breaks.");
                return;
            }
            let _ = prompt_tx.send(SessionCommand::RequestFileWrite {
                path: path.trim().to_string(),
                content: decode_slash_write_content(raw_content.trim()),
            });
        }
        "/reflect" => {
            let mode = arg.to_ascii_lowercase();
            match mode.as_str() {
                "on" => {
                    if state.eco_enabled {
                        state.add_system_message(
                            "reflection stays off while eco mode is enabled. Use /eco off first.",
                        );
                        return;
                    }
                    state.set_reflection_enabled(true);
                    state.add_system_message("reflection enabled");
                    let _ = prompt_tx.send(SessionCommand::SetReflection(true));
                }
                "off" => {
                    state.set_reflection_enabled(false);
                    state.add_system_message("reflection disabled");
                    let _ = prompt_tx.send(SessionCommand::SetReflection(false));
                }
                "" | "status" => {
                    state.add_system_message(&format!(
                        "reflection is {}",
                        if state.reflection_enabled {
                            "on"
                        } else {
                            "off"
                        }
                    ));
                }
                _ => state.add_system_message("Usage: /reflect <on|off|status>"),
            }
        }
        "/eco" => {
            let mode = arg.to_ascii_lowercase();
            match mode.as_str() {
                "on" => {
                    state.set_eco_enabled(true);
                    state.set_reflection_enabled(false);
                    state.add_system_message("eco mode enabled");
                    let _ = prompt_tx.send(SessionCommand::SetEco(true));
                }
                "off" => {
                    state.set_eco_enabled(false);
                    state.add_system_message("eco mode disabled");
                    let _ = prompt_tx.send(SessionCommand::SetEco(false));
                }
                "" | "status" => {
                    state.add_system_message(&format!(
                        "eco mode is {}",
                        if state.eco_enabled { "on" } else { "off" }
                    ));
                }
                _ => state.add_system_message("Usage: /eco <on|off|status>"),
            }
        }
        "/debug-log" => {
            let mode = arg.to_ascii_lowercase();
            match mode.as_str() {
                "on" => {
                    state.set_debug_logging_enabled(true);
                    state.add_system_message("separate debug content logging enabled");
                    let _ = prompt_tx.send(SessionCommand::SetDebugLogging(true));
                }
                "off" => {
                    state.set_debug_logging_enabled(false);
                    state.add_system_message("separate debug content logging disabled");
                    let _ = prompt_tx.send(SessionCommand::SetDebugLogging(false));
                }
                "" | "status" => {
                    state.add_system_message(&format!(
                        "debug content logging is {}",
                        if state.debug_logging_enabled {
                            "on"
                        } else {
                            "off"
                        }
                    ));
                }
                _ => state.add_system_message("Usage: /debug-log <on|off|status>"),
            }
        }
        "/approve" => match state.pending_action_id() {
            Some(id) => {
                let _ = prompt_tx.send(SessionCommand::ApproveAction(id));
                state.mark_pending_action_submitted("processing approval");
            }
            None => state.add_system_message("No pending action to approve."),
        },
        "/reject" => match state.pending_action_id() {
            Some(id) => {
                let _ = prompt_tx.send(SessionCommand::RejectAction(id));
                state.mark_pending_action_submitted("processing rejection");
            }
            None => state.add_system_message("No pending action to reject."),
        },
        "/clear" => {
            state.clear_messages();
            state.add_system_message("conversation cleared");
            let _ = prompt_tx.send(SessionCommand::ClearSession);
        }
        "/clear-cache" => {
            state.add_system_message("clearing exact cache...");
            let _ = prompt_tx.send(SessionCommand::ClearCache);
        }
        "/clear-debug-log" => {
            state.add_system_message("clearing separate debug content log...");
            let _ = prompt_tx.send(SessionCommand::ClearDebugLog);
        }
        "/help" => {
            state.add_system_message(&custom_help_text());
        }
        "/commands" => {
            let subcommand = if arg.is_empty() { "list" } else { arg };
            match subcommand {
                "list" => {
                    state.add_system_message(&format_custom_commands_list(command_registry));
                }
                "reload" => {
                    let report = CommandRegistry::load_report();
                    *command_registry = report.registry;
                    state.add_system_message(&format!(
                        "reloaded custom commands: {} loaded, {} invalid, {} source file(s)",
                        report.loaded, report.invalid, report.sources_loaded
                    ));
                }
                _ => {
                    state.add_system_message("Usage: /commands [list|reload]");
                }
            }
        }
        _ => {
            state.add_system_message(&format!(
                "unknown command: {cmd}. Type /help for available commands."
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::decode_slash_write_content;

    #[test]
    fn decode_slash_write_content_expands_common_escapes() {
        let decoded = decode_slash_write_content("hello\\nfrom\\tparams\\\\");
        assert_eq!(decoded, "hello\nfrom\tparams\\");
    }
}
