// src/tui/mod.rs

mod state;

use std::io;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use crossterm::{
    event::{
        self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
        Event, KeyCode, KeyModifiers,
    },
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

use crate::error::Result;
use crate::events::InferenceEvent;
use crate::inference::SessionCommand;
use state::{AppState, Role};

// Spinner frames
const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

// How often to advance the spinner (every N ticks at ~60fps = ~100ms per frame)
const SPINNER_SPEED: u64 = 6;

fn truncate_for_width(value: &str, max_chars: usize) -> String {
    let len = value.chars().count();
    if len <= max_chars {
        return value.to_string();
    }
    let keep = max_chars.saturating_sub(1);
    let truncated: String = value.chars().take(keep).collect();
    format!("{truncated}…")
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
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(
        stdout,
        EnterAlternateScreen,
        EnableMouseCapture,
        EnableBracketedPaste
    )?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let result = run_app(&mut terminal);
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture,
        DisableBracketedPaste
    )?;
    terminal.show_cursor()?;
    result
}

fn run_app(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    let mut state = AppState::new();

    let (token_tx, token_rx) = mpsc::channel::<InferenceEvent>();
    let (prompt_tx, prompt_rx) = mpsc::channel::<SessionCommand>();

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
                InferenceEvent::BackendName(name) => {
                    state.backend_name = name;
                }
                InferenceEvent::Token(token) => {
                    state.append_token(&token);
                }
                InferenceEvent::ToolCall(call) => {
                    state.last_tool_call = Some(call);
                    state.status = "running tool...".to_string();
                }
                InferenceEvent::Done => {
                    state.finish_response();
                }
                InferenceEvent::Error(e) => {
                    state.add_error(&e);
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

                        // Submit
                        (KeyCode::Enter, _) => {
                            if !state.input.is_empty() && !state.is_generating && state.is_ready() {
                                let prompt = state.submit_input();

                                // Check for slash commands before sending to inference.
                                // These run tools directly and inject results into context
                                // without requiring the model to format a tool call tag.
                                if prompt.starts_with('/') {
                                    handle_slash_command(&prompt, &mut state, &prompt_tx);
                                } else {
                                    state.add_user_message(&prompt);
                                    let _ = prompt_tx.send(SessionCommand::SubmitUser(prompt.clone()));
                                    state.is_generating = true;
                                    state.status = "generating...".to_string();
                                    state.start_assistant_message();
                                }
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
                    // Strip newlines from paste — treat as single line
                    let clean: String = text.chars().filter(|&c| c != '\n' && c != '\r').collect();
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
    let border_color = if state.is_generating {
        Color::Yellow
    } else if state.model_ready {
        Color::DarkGray
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
    } else if state.is_generating {
        format!("{spinner_frame} generating")
    } else {
        state.status.clone()
    };

    let status_color = if state.is_generating || !state.model_ready {
        Color::Yellow
    } else {
        Color::Green
    };

    // Truncate backend name to fit sidebar
    let backend_display = truncate_for_width(&state.backend_name, 18);

    let items = vec![
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
        ListItem::new(Line::from("")),
        {
            if let Some(ref call) = state.last_tool_call {
                let truncated = truncate_for_width(call, 18);
                ListItem::new(Line::from(vec![
                    Span::styled("tool  ", Style::default().fg(Color::DarkGray)),
                    Span::styled(truncated, Style::default().fg(Color::Yellow)),
                ]))
            } else {
                ListItem::new(Line::from(""))
            }
        },
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
            Span::styled("/clear ", Style::default().fg(Color::DarkGray)),
            Span::styled("history", Style::default().fg(Color::DarkGray)),
        ])),
    ];

    frame.render_widget(List::new(items).block(block), area);
}

fn draw_main(frame: &mut Frame, state: &mut AppState, area: Rect) {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(3)])
        .split(area);
    draw_chat(frame, state, vertical[0]);
    draw_input(frame, state, vertical[1]);
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
    let (title, border_color) = if !state.model_ready {
        (" loading... ", Color::DarkGray)
    } else if state.is_generating {
        (" generating... ", Color::Yellow)
    } else {
        (" message ", Color::DarkGray)
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color))
        .title(Span::styled(title, Style::default().fg(border_color)));

    // Show input with cursor position indicated by █
    let before_cursor = &state.input[..state.cursor];
    let after_cursor = &state.input[state.cursor..];

    let input_spans = if state.is_generating || !state.model_ready {
        // Dimmed when not accepting input
        vec![Span::styled(
            state.input.clone(),
            Style::default().fg(Color::DarkGray),
        )]
    } else if after_cursor.is_empty() {
        vec![
            Span::styled(before_cursor.to_string(), Style::default().fg(Color::White)),
            Span::styled(
                "█",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::SLOW_BLINK),
            ),
        ]
    } else {
        let mut chars = after_cursor.chars();
        let at_cursor = chars.next().unwrap_or(' ').to_string();
        let rest: String = chars.collect();
        vec![
            Span::styled(before_cursor.to_string(), Style::default().fg(Color::White)),
            Span::styled(
                at_cursor,
                Style::default().fg(Color::Black).bg(Color::White),
            ),
            Span::styled(rest, Style::default().fg(Color::White)),
        ]
    };

    let paragraph = Paragraph::new(Line::from(input_spans)).block(block);
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

fn handle_slash_command(
    input: &str,
    state: &mut AppState,
    prompt_tx: &mpsc::Sender<SessionCommand>,
) {
    // Parse command and argument
    let parts: Vec<&str> = input.splitn(2, ' ').collect();
    let cmd = parts[0].to_lowercase();
    let arg = parts.get(1).map(|s| s.trim()).unwrap_or("");

    match cmd.as_str() {
        "/read" | "/r" => {
            if arg.is_empty() {
                state.add_system_message("Usage: /read <file_path>");
                return;
            }
            let tool = crate::tools::ReadFile;
            match crate::tools::Tool::run(&tool, arg) {
                Ok(output) => {
                    state.add_system_message(&format!("loaded: {arg}"));
                    let safe = sanitize_for_display(&output);
                    let context = format!("I've loaded this file for context:\n\n{safe}");
                    state.add_user_message(&context);
                    let _ = prompt_tx.send(SessionCommand::InjectUserContext(context));
                }
                Err(e) => {
                    state.add_system_message(&format!("error reading {arg}: {e}"));
                }
            }
        }

        "/ls" | "/list" => {
            let path = if arg.is_empty() { "." } else { arg };
            let tool = crate::tools::ListDir;
            match crate::tools::Tool::run(&tool, path) {
                Ok(output) => {
                    state.add_system_message(&format!("listed: {path}"));
                    let safe = sanitize_for_display(&output);
                    let context = format!("Directory listing:\n\n{safe}");
                    state.add_user_message(&context);
                    let _ = prompt_tx.send(SessionCommand::InjectUserContext(context));
                }
                Err(e) => {
                    state.add_system_message(&format!("error listing {path}: {e}"));
                }
            }
        }

        "/search" | "/s" | "/grep" => {
            if arg.is_empty() {
                state.add_system_message("Usage: /search <query>");
                return;
            }
            let tool = crate::tools::SearchCode;
            match crate::tools::Tool::run(&tool, arg) {
                Ok(output) => {
                    state.add_system_message(&format!("searched: {arg}"));
                    let safe = sanitize_for_display(&output);
                    let context = format!("Search results:\n\n{safe}");
                    state.add_user_message(&context);
                    let _ = prompt_tx.send(SessionCommand::InjectUserContext(context));
                }
                Err(e) => {
                    state.add_system_message(&format!("error searching: {e}"));
                }
            }
        }

        "/git" => {
            let git_arg = if arg.is_empty() { "status" } else { arg };
            let tool = crate::tools::GitTool;
            match crate::tools::Tool::run(&tool, git_arg) {
                Ok(output) => {
                    state.add_system_message(&format!("git: {git_arg}"));
                    let safe = sanitize_for_display(&output);
                    let context = format!("Git context ({git_arg}):\n\n{safe}");
                    state.add_user_message(&context);
                    let _ = prompt_tx.send(SessionCommand::InjectUserContext(context));
                }
                Err(e) => {
                    state.add_system_message(&format!("git error: {e}"));
                }
            }
        }

        "/clear" | "/c" => {
            state.clear_messages();
            state.add_system_message("conversation cleared");
            let _ = prompt_tx.send(SessionCommand::ClearSession);
        }

        "/help" | "/h" | "/?" => {
            state.add_system_message(
                "/read <path>    — load a file into context\n  \
                 /ls [path]      — list directory (default: current)\n  \
                 /search <query> — search source files\n  \
                 /git [command]  — git status, diff, log\n  \
                 /clear          — clear conversation\n  \
                 /help           — show this message",
            );
        }

        _ => {
            state.add_system_message(&format!(
                "unknown command: {cmd}. Type /help for available commands."
            ));
        }
    }
}
