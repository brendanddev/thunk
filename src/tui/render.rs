use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};

use crate::events::PendingActionKind;
use crate::safety::RiskLevel;

use super::format::{
    format_compact_count, format_cost, format_duration, format_hit_rate, push_wrapped_styled,
    truncate_for_width, wrap_plain_text,
};
use super::state::{AppState, Role};
use crate::session::short_id;

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const SPINNER_SPEED: u64 = 6;
const MAX_INPUT_VISIBLE_ROWS: usize = 8;

pub(crate) fn draw(frame: &mut Frame, state: &mut AppState) {
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

    let backend_display = truncate_for_width(&state.backend_name, 18);
    let session_display = state
        .current_session
        .as_ref()
        .map(|session| {
            let label = session
                .name
                .clone()
                .unwrap_or_else(|| format!("unnamed {}", short_id(&session.id)));
            format!("{label} ({})", session.message_count)
        })
        .unwrap_or_else(|| "n/a".to_string());
    let current_turn_duration = state.current_turn_duration();
    let last_work_duration = state.last_work_duration();
    let memory_facts_count = state.memory_snapshot.loaded_facts.len();
    let memory_summaries_count = state.memory_snapshot.last_summary_paths.len();
    let memory_update = state.memory_snapshot.last_update.as_ref();
    let accepted_memory_count = memory_update
        .map(|update| update.accepted_facts.len())
        .unwrap_or(0);
    let skipped_memory_count = memory_update
        .map(|update| {
            update
                .skipped_reasons
                .iter()
                .map(|reason| reason.count)
                .sum::<usize>()
        })
        .unwrap_or(0);
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
            Span::styled("sess  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                truncate_for_width(&session_display, 18),
                Style::default().fg(Color::White),
            ),
        ])),
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
        ListItem::new(Line::from(vec![
            Span::styled("mem   ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{memory_facts_count}f/{memory_summaries_count}s"),
                Style::default().fg(Color::White),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("mupd  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("+{accepted_memory_count}/-{skipped_memory_count}"),
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
            Span::styled("/sess  ", Style::default().fg(Color::DarkGray)),
            Span::styled("manage", Style::default().fg(Color::DarkGray)),
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
            Span::styled("/edit  ", Style::default().fg(Color::DarkGray)),
            Span::styled(" <path>", Style::default().fg(Color::DarkGray)),
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
            Span::styled("/sessions", Style::default().fg(Color::DarkGray)),
            Span::styled(" list", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("/memory", Style::default().fg(Color::DarkGray)),
            Span::styled(" status", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("edit_file", Style::default().fg(Color::DarkGray)),
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
                for line in msg.content.lines() {
                    push_wrapped_styled(
                        &mut lines,
                        &format!("  {line}"),
                        Style::default().fg(Color::Gray),
                        inner_width,
                    );
                }
                if msg.content.is_empty() {
                    lines.push(Line::from(Span::styled(
                        "  ▌",
                        Style::default().fg(Color::DarkGray),
                    )));
                }
                lines.push(Line::from(""));
            }
            Role::System => {
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
        PendingActionKind::FileWrite | PendingActionKind::FileEdit => 8,
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
        RiskLevel::Low => (Color::Cyan, " pending approval "),
        RiskLevel::Medium => (Color::Yellow, " pending approval "),
        RiskLevel::High => (Color::Red, " pending approval "),
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
        PendingActionKind::FileWrite | PendingActionKind::FileEdit => 8,
    };
    let preview_style = match pending.kind {
        PendingActionKind::ShellCommand => Style::default().fg(Color::White),
        PendingActionKind::FileWrite | PendingActionKind::FileEdit => {
            Style::default().fg(Color::Gray)
        }
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
