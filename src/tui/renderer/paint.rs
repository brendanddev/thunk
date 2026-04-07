use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::events::PendingActionKind;
use crate::safety::RiskLevel;
use crate::session::short_id;
use crate::tui::state::{AppState, ChatMessage, Role};

use super::buffer::{Cell, CellBuffer};
use super::layout::{LayoutPlan, Rect, RootLayoutMode};
use super::style::{PackedStyle, Theme};
use super::symbols::SymbolPool;

#[derive(Debug, Clone)]
pub(crate) struct StyledSpan {
    pub text: String,
    pub style: PackedStyle,
}

#[derive(Debug, Clone)]
pub(crate) struct StyledLine {
    pub spans: Vec<StyledSpan>,
}

#[derive(Debug, Clone)]
pub(crate) struct RenderBlock {
    pub message_id: Option<u64>,
    pub lines: Vec<StyledLine>,
}

#[derive(Debug, Clone)]
pub(crate) struct RenderModel {
    pub top_bar: Vec<StyledLine>,
    pub transcript: Vec<RenderBlock>,
    pub composer: Vec<StyledLine>,
    pub composer_cursor: (u16, u16),
}

const SPINNER_FRAMES: &[&str] = &["·", "•", "◦", "•"];

pub(crate) fn build_render_model(
    state: &mut AppState,
    theme: Theme,
    width: u16,
    layout_mode: RootLayoutMode,
    cached_block: impl FnMut(&ChatMessage, bool, u16) -> Vec<StyledLine>,
) -> RenderModel {
    let top_bar = build_top_bar(state, theme, width);
    let _ = layout_mode;
    let transcript = build_transcript(state, theme, width, cached_block);
    let approval = build_approval(state, theme, width);
    let (composer, composer_cursor) = build_composer(state, theme, width, approval.as_deref());

    RenderModel {
        top_bar,
        transcript,
        composer,
        composer_cursor,
    }
}

fn build_top_bar(state: &AppState, theme: Theme, width: u16) -> Vec<StyledLine> {
    let session_label = state
        .current_session
        .as_ref()
        .map(|session| {
            session
                .name
                .clone()
                .unwrap_or_else(|| format!("unnamed {}", short_id(&session.id)))
        })
        .unwrap_or_else(|| "fresh".to_string());
    let runtime_label = runtime_label(state);
    let runtime_style = runtime_style(state, theme);
    let runtime_display = if is_runtime_animated(state) {
        format!(
            "{} {}",
            spinner_frame(state.tick),
            truncate(&runtime_label, 20)
        )
    } else {
        truncate(&runtime_label, 22)
    };
    let cache_label = match state.last_cache_hit {
        Some(true) => "cache hit",
        Some(false) => "cache miss",
        None => "cache n/a",
    };

    let segments = [
        TopSegment::new("params", theme.base().with_bold(), 0),
        TopSegment::new(&runtime_display, runtime_style, 1),
        TopSegment::new(&truncate(&state.backend_name, 16), theme.muted(), 2),
        TopSegment::new(&truncate(&session_label, 16), theme.muted(), 3),
        TopSegment::new(&format!("{} msgs", state.message_count()), theme.dim(), 4),
        TopSegment::new(cache_label, theme.dim(), 5),
        TopSegment::new(
            &format!(
                "mem {}/{}",
                state.memory_snapshot.loaded_facts.len(),
                state.memory_snapshot.last_summary_paths.len()
            ),
            theme.dim(),
            6,
        ),
        TopSegment::new(&format!("view +{}", state.scroll_offset), theme.dim(), 7),
    ];

    vec![build_segmented_top_line(
        segments
            .into_iter()
            .filter(|segment| segment.text != "view +0")
            .collect(),
        width,
        theme,
    )]
}

fn build_transcript(
    state: &mut AppState,
    theme: Theme,
    width: u16,
    mut cached_block: impl FnMut(&ChatMessage, bool, u16) -> Vec<StyledLine>,
) -> Vec<RenderBlock> {
    let mut blocks = Vec::new();
    if state.messages.is_empty() {
        blocks.push(RenderBlock {
            message_id: None,
            lines: vec![single_span(
                "ready. ask for code help, run a slash command, or inspect the project.",
                theme.dim(),
            )],
        });
        return blocks;
    }

    for message in &state.messages {
        if message.transcript.collapsible {
            let focused = state.is_focused_collapsible(message.id);
            blocks.push(RenderBlock {
                message_id: Some(message.id),
                lines: cached_block(message, focused, width),
            });
        } else {
            let is_active_assistant = message.role == Role::Assistant
                && state.is_generating
                && state.messages.last().map(|last| last.id) == Some(message.id);
            blocks.push(RenderBlock {
                message_id: None,
                lines: build_standard_message(
                    message,
                    theme,
                    width,
                    is_active_assistant,
                    state.tick,
                ),
            });
        }
    }
    blocks
}

pub(crate) fn build_transcript_block(
    message: &ChatMessage,
    theme: Theme,
    width: u16,
    focused: bool,
) -> Vec<StyledLine> {
    if message.transcript.collapsed {
        build_collapsed_context(message, theme, width, focused)
    } else {
        build_expanded_context(message, theme, width, focused)
    }
}

fn build_standard_message(
    message: &ChatMessage,
    theme: Theme,
    width: u16,
    is_active_assistant: bool,
    tick: u64,
) -> Vec<StyledLine> {
    match message.role {
        Role::User => build_badged_message(
            "you",
            theme.badge_user(),
            &message.content,
            theme,
            width,
            None,
        ),
        Role::Assistant => build_badged_message(
            "params",
            theme.badge_assistant(),
            if message.content.is_empty() {
                "▍"
            } else {
                &message.content
            },
            theme,
            width,
            is_active_assistant.then(|| active_cursor_span(theme, tick)),
        ),
        Role::System => wrap_plain_to_lines(&format!("• {}", message.content), theme.dim(), width),
    }
}

fn build_badged_message(
    label: &str,
    badge_style: PackedStyle,
    body: &str,
    theme: Theme,
    width: u16,
    active_cursor: Option<StyledSpan>,
) -> Vec<StyledLine> {
    let mut lines = Vec::new();
    let body_width = width
        .saturating_sub((label_width(label) + 3).min(u16::MAX as usize) as u16)
        .max(8);
    let wrapped = wrap_text(body, body_width);
    if wrapped.is_empty() {
        lines.push(StyledLine {
            spans: vec![StyledSpan {
                text: format!("{label} "),
                style: badge_style,
            }],
        });
        return lines;
    }

    lines.push(StyledLine {
        spans: vec![
            StyledSpan {
                text: label.to_string(),
                style: badge_style,
            },
            StyledSpan {
                text: "  ".to_string(),
                style: theme.base(),
            },
            StyledSpan {
                text: wrapped[0].clone(),
                style: theme.base(),
            },
        ],
    });

    for line in wrapped.into_iter().skip(1) {
        lines.push(StyledLine {
            spans: vec![StyledSpan {
                text: format!("{:width$}{}", "", line, width = label_width(label) + 2),
                style: theme.base(),
            }],
        });
    }
    if let Some(cursor_span) = active_cursor {
        if let Some(last) = lines.last_mut() {
            last.spans.push(cursor_span);
        }
    }
    lines.push(blank_line());
    lines
}

fn build_collapsed_context(
    message: &ChatMessage,
    theme: Theme,
    width: u16,
    focused: bool,
) -> Vec<StyledLine> {
    let mut lines = Vec::new();
    let edge = if focused { "▸" } else { "›" };
    let edge_style = if focused {
        theme.border_active()
    } else {
        theme.border()
    };
    let summary = message.transcript.summary.as_deref().unwrap_or("context");
    lines.push(StyledLine {
        spans: vec![
            StyledSpan {
                text: edge.to_string(),
                style: edge_style,
            },
            StyledSpan {
                text: " ".to_string(),
                style: theme.base(),
            },
            StyledSpan {
                text: summary.to_string(),
                style: if focused {
                    theme.chip_accent().with_underline()
                } else {
                    theme.muted().with_bold()
                },
            },
        ],
    });

    for preview in message
        .transcript
        .preview_lines
        .iter()
        .take(if focused { 2 } else { 1 })
        .flat_map(|line| wrap_text(line, width.saturating_sub(4)))
    {
        lines.push(StyledLine {
            spans: vec![
                StyledSpan {
                    text: edge.to_string(),
                    style: edge_style,
                },
                StyledSpan {
                    text: " ".to_string(),
                    style: theme.base(),
                },
                StyledSpan {
                    text: preview,
                    style: theme.muted(),
                },
            ],
        });
    }
    if focused {
        lines.push(StyledLine {
            spans: vec![
                StyledSpan {
                    text: edge.to_string(),
                    style: edge_style,
                },
                StyledSpan {
                    text: " open with Ctrl+O • move with [ ]".to_string(),
                    style: theme.dim(),
                },
            ],
        });
    }
    lines
}

fn build_expanded_context(
    message: &ChatMessage,
    theme: Theme,
    width: u16,
    focused: bool,
) -> Vec<StyledLine> {
    let edge = if focused { "▸" } else { "›" };
    let edge_style = if focused {
        theme.border_active()
    } else {
        theme.border()
    };
    let summary = message.transcript.summary.as_deref().unwrap_or("context");
    let mut lines = vec![StyledLine {
        spans: vec![
            StyledSpan {
                text: edge.to_string(),
                style: edge_style,
            },
            StyledSpan {
                text: " ".to_string(),
                style: theme.base(),
            },
            StyledSpan {
                text: summary.to_string(),
                style: if focused {
                    theme.chip_accent().with_underline()
                } else {
                    theme.muted().with_bold()
                },
            },
        ],
    }];
    for body_line in message
        .content
        .lines()
        .flat_map(|line| wrap_text(line, width.saturating_sub(4)))
    {
        lines.push(StyledLine {
            spans: vec![
                StyledSpan {
                    text: edge.to_string(),
                    style: edge_style,
                },
                StyledSpan {
                    text: " ".to_string(),
                    style: theme.base(),
                },
                StyledSpan {
                    text: body_line,
                    style: theme.dim(),
                },
            ],
        });
    }
    if focused {
        lines.push(StyledLine {
            spans: vec![
                StyledSpan {
                    text: edge.to_string(),
                    style: edge_style,
                },
                StyledSpan {
                    text: " collapse with Ctrl+O".to_string(),
                    style: theme.dim(),
                },
            ],
        });
    }
    lines
}

fn build_approval(state: &AppState, theme: Theme, width: u16) -> Option<Vec<StyledLine>> {
    let pending = state.pending_action.as_ref()?;
    let risk_chip = match pending.inspection.risk {
        RiskLevel::Low => theme.chip_accent(),
        RiskLevel::Medium => theme.chip_warning(),
        RiskLevel::High => theme.chip_danger(),
    };
    let approval_style = match pending.inspection.risk {
        RiskLevel::Low => theme.chip_accent(),
        RiskLevel::Medium => theme.chip_warning(),
        RiskLevel::High => theme.chip_danger(),
    };
    let mut lines = vec![
        StyledLine {
            spans: vec![
                StyledSpan {
                    text: "approval".to_string(),
                    style: approval_style,
                },
                StyledSpan {
                    text: format!("  {}", pending.title),
                    style: theme.base(),
                },
            ],
        },
        StyledLine {
            spans: vec![
                StyledSpan {
                    text: format!(
                        "{} / {}",
                        pending.inspection.decision, pending.inspection.risk
                    ),
                    style: risk_chip,
                },
                StyledSpan {
                    text: format!("  {}", pending.inspection.summary),
                    style: theme.muted(),
                },
            ],
        },
    ];
    for reason in &pending.inspection.reasons {
        lines.extend(wrap_plain_to_lines(
            &format!("• {reason}"),
            theme.dim(),
            width,
        ));
    }
    let preview_lines = match pending.kind {
        PendingActionKind::ShellCommand => 2,
        PendingActionKind::FileWrite | PendingActionKind::FileEdit => 8,
    };
    for line in preview_snippet(&pending.preview, width.saturating_sub(2), preview_lines) {
        lines.push(StyledLine {
            spans: vec![
                StyledSpan {
                    text: "│ ".to_string(),
                    style: theme.border(),
                },
                StyledSpan {
                    text: line,
                    style: theme.muted(),
                },
            ],
        });
    }
    lines.push(single_span(
        "Ctrl+Y approve  Ctrl+N reject  /approve  /reject",
        theme.dim(),
    ));
    Some(lines)
}

fn build_composer(
    state: &AppState,
    theme: Theme,
    width: u16,
    approval: Option<&[StyledLine]>,
) -> (Vec<StyledLine>, (u16, u16)) {
    let inner_width = width.saturating_sub(2).max(8) as usize;
    let max_visible_rows = 8usize;
    let (visible_rows, cursor_row, cursor_col) =
        state.input_display_lines(inner_width, max_visible_rows);
    let mut lines = Vec::new();
    let mut prompt_offset = 0u16;

    if let Some(approval_lines) = approval {
        for line in approval_lines {
            lines.push(line.clone());
        }
        lines.push(blank_line());
        prompt_offset = lines.len() as u16;
    }

    let prompt_marker = if state.is_reverse_search_active() {
        "? "
    } else if state.is_command_launcher_active() {
        ": "
    } else if state.has_pending_action() {
        "! "
    } else {
        "› "
    };
    let prompt_style = if state.has_pending_action() {
        theme.chip_warning()
    } else if state.is_generating {
        theme.badge_assistant()
    } else {
        theme.chip_accent()
    };

    for (idx, row) in visible_rows.iter().enumerate() {
        lines.push(StyledLine {
            spans: vec![
                StyledSpan {
                    text: if idx == 0 {
                        prompt_marker.to_string()
                    } else {
                        "  ".to_string()
                    },
                    style: prompt_style,
                },
                StyledSpan {
                    text: row.clone(),
                    style: if state.is_generating {
                        theme.base()
                    } else {
                        theme.base()
                    },
                },
            ],
        });
    }

    if let Some((query, entries)) = state.command_launcher_view(5) {
        let selected_entry = entries
            .iter()
            .find(|(_, selected)| *selected)
            .map(|(entry, _)| entry.clone());
        lines.push(StyledLine {
            spans: vec![
                StyledSpan {
                    text: format!(
                        "commands: {}",
                        if query.is_empty() {
                            "all commands"
                        } else {
                            &query
                        }
                    ),
                    style: theme.base().with_bold(),
                },
                StyledSpan {
                    text: "  Ctrl+K next  ↑↓ move  Enter choose  Esc close".to_string(),
                    style: theme.dim(),
                },
            ],
        });
        for (entry, selected) in entries {
            lines.push(StyledLine {
                spans: vec![
                    StyledSpan {
                        text: if selected {
                            "→ ".to_string()
                        } else {
                            "  ".to_string()
                        },
                        style: if selected {
                            theme.chip_accent()
                        } else {
                            theme.dim()
                        },
                    },
                    StyledSpan {
                        text: truncate(&entry.name, 18),
                        style: if selected {
                            theme.chip_accent()
                        } else {
                            theme.base()
                        },
                    },
                    StyledSpan {
                        text: "  ".to_string(),
                        style: theme.dim(),
                    },
                    StyledSpan {
                        text: truncate(
                            &format!("{} • {}", entry.group, entry.description),
                            width.saturating_sub(26) as usize,
                        ),
                        style: theme.muted(),
                    },
                ],
            });
        }
        if let Some(selected) = selected_entry {
            lines.push(single_span(
                &format!(
                    "usage: {}",
                    truncate(&selected.usage, width.saturating_sub(7) as usize)
                ),
                theme.dim(),
            ));
        }
    } else if let Some((query, current)) = state.reverse_search_view() {
        lines.push(StyledLine {
            spans: vec![
                StyledSpan {
                    text: format!("search: {}", query),
                    style: theme.chip_warning(),
                },
                StyledSpan {
                    text: "  ".to_string(),
                    style: theme.dim(),
                },
                StyledSpan {
                    text: truncate(&current, width.saturating_sub(12) as usize),
                    style: theme.muted(),
                },
            ],
        });
    } else if !state.autocomplete_preview_items(4).is_empty() {
        lines.push(single_span("suggestions", theme.dim()));
        for (item, selected) in state.autocomplete_preview_items(4) {
            lines.push(StyledLine {
                spans: vec![
                    StyledSpan {
                        text: if selected {
                            "→ ".to_string()
                        } else {
                            "  ".to_string()
                        },
                        style: if selected {
                            theme.chip_accent()
                        } else {
                            theme.dim()
                        },
                    },
                    StyledSpan {
                        text: truncate(&item, width.saturating_sub(4) as usize),
                        style: if selected {
                            theme.chip_accent()
                        } else {
                            theme.muted()
                        },
                    },
                ],
            });
        }
    } else {
        lines.push(single_span(
            "Enter send • Shift+Enter newline • Alt+↑ history • Ctrl+R search • Ctrl+K commands",
            theme.dim(),
        ));
    }

    let cursor_x = 2 + cursor_col as u16;
    let cursor_y = prompt_offset + cursor_row as u16;
    (lines, (cursor_x, cursor_y))
}

pub(crate) fn paint_model(
    buffer: &mut CellBuffer,
    symbols: &mut SymbolPool,
    model: &RenderModel,
    layout: LayoutPlan,
    theme: Theme,
    state: &mut AppState,
) -> (u16, u16) {
    buffer.clear();
    let background = Cell {
        symbol_id: symbols.blank_id(),
        style: PackedStyle::new(theme.text, theme.background),
    };
    buffer.fill_rect(0, 0, buffer.width(), buffer.height(), background);

    paint_lines(buffer, symbols, &layout.top_bar, &model.top_bar);
    paint_divider(
        buffer,
        symbols,
        layout.top_bar.x,
        layout
            .top_bar
            .y
            .saturating_add(layout.top_bar.height.saturating_sub(1)),
        layout.top_bar.width,
        theme,
    );

    paint_transcript(
        buffer,
        symbols,
        &layout.transcript,
        &model.transcript,
        theme,
        state,
    );

    paint_sheet(
        buffer,
        symbols,
        &layout.composer,
        &model.composer,
        theme,
        "",
    );

    (
        layout.composer.x.saturating_add(model.composer_cursor.0),
        layout
            .composer
            .y
            .saturating_add(1)
            .saturating_add(model.composer_cursor.1),
    )
}

fn paint_transcript(
    buffer: &mut CellBuffer,
    symbols: &mut SymbolPool,
    rect: &Rect,
    blocks: &[RenderBlock],
    theme: Theme,
    state: &mut AppState,
) {
    let mut flattened = Vec::new();
    let mut visible_ids = Vec::new();
    for block in blocks {
        for line in &block.lines {
            flattened.push((block.message_id, line.clone()));
        }
    }

    let max_scroll = flattened.len().saturating_sub(rect.height as usize);
    state.max_scroll = max_scroll;
    state.scroll_offset = state.scroll_offset.min(max_scroll);
    let end = flattened.len().saturating_sub(state.scroll_offset);
    let start = end.saturating_sub(rect.height as usize);
    let visible = &flattened[start..end];

    let mut y = rect.y;
    if state.scroll_offset > 0 && y < rect.y.saturating_add(rect.height) {
        let indicator = single_span(&format!("↑ {} older", state.scroll_offset), theme.dim());
        paint_line(buffer, symbols, rect.x, y, rect.width, &indicator);
        y += 1;
    }

    for (message_id, line) in visible.iter().take(rect.height as usize) {
        if y >= rect.y.saturating_add(rect.height) {
            break;
        }
        if let Some(id) = message_id {
            if !visible_ids.contains(id) {
                visible_ids.push(*id);
            }
        }
        paint_line(buffer, symbols, rect.x, y, rect.width, line);
        y += 1;
    }

    if state.scroll_offset == 0 && max_scroll > 0 && y < rect.y.saturating_add(rect.height) {
        let indicator = single_span(&format!("↓ {} earlier", max_scroll), theme.dim());
        paint_line(buffer, symbols, rect.x, y, rect.width, &indicator);
    }

    state.set_visible_collapsible_ids(visible_ids);
}

fn paint_sheet(
    buffer: &mut CellBuffer,
    symbols: &mut SymbolPool,
    rect: &Rect,
    lines: &[StyledLine],
    theme: Theme,
    title: &str,
) {
    paint_divider(buffer, symbols, rect.x, rect.y, rect.width, theme);
    if !title.is_empty() {
        paint_line(
            buffer,
            symbols,
            rect.x,
            rect.y,
            rect.width,
            &single_span(title, theme.dim().with_bold()),
        );
    }
    paint_lines(
        buffer,
        symbols,
        &Rect::new(
            rect.x,
            rect.y.saturating_add(1),
            rect.width,
            rect.height.saturating_sub(1),
        ),
        lines,
    );
}

fn paint_divider(
    buffer: &mut CellBuffer,
    symbols: &mut SymbolPool,
    x: u16,
    y: u16,
    width: u16,
    theme: Theme,
) {
    if y >= buffer.height() || width == 0 {
        return;
    }
    let divider = Cell {
        symbol_id: symbols.intern("─"),
        style: theme.border(),
    };
    for col in x..x.saturating_add(width).min(buffer.width()) {
        buffer.set(col, y, divider);
    }
}

fn paint_lines(
    buffer: &mut CellBuffer,
    symbols: &mut SymbolPool,
    rect: &Rect,
    lines: &[StyledLine],
) {
    for (idx, line) in lines.iter().enumerate() {
        let y = rect.y.saturating_add(idx as u16);
        if y >= rect.y.saturating_add(rect.height) || y >= buffer.height() {
            break;
        }
        paint_line(buffer, symbols, rect.x, y, rect.width, line);
    }
}

fn paint_line(
    buffer: &mut CellBuffer,
    symbols: &mut SymbolPool,
    x: u16,
    y: u16,
    width: u16,
    line: &StyledLine,
) {
    let mut cursor = x;
    for span in &line.spans {
        if cursor >= x.saturating_add(width) {
            break;
        }
        let remaining = x.saturating_add(width).saturating_sub(cursor);
        let written =
            buffer.write_text_clipped(cursor, y, &span.text, remaining, span.style, symbols);
        cursor = cursor.saturating_add(written);
    }
}

fn single_span(text: &str, style: PackedStyle) -> StyledLine {
    StyledLine {
        spans: vec![StyledSpan {
            text: text.to_string(),
            style,
        }],
    }
}

fn blank_line() -> StyledLine {
    StyledLine { spans: Vec::new() }
}

fn wrap_plain_to_lines(text: &str, style: PackedStyle, width: u16) -> Vec<StyledLine> {
    wrap_text(text, width)
        .into_iter()
        .map(|line| single_span(&line, style))
        .collect()
}

fn wrap_text(text: &str, width: u16) -> Vec<String> {
    let width = width.max(1) as usize;
    if text.is_empty() {
        return vec![String::new()];
    }

    let mut lines = Vec::new();
    for raw_line in text.split('\n') {
        let mut current = String::new();
        let mut current_width = 0usize;
        for ch in raw_line.chars() {
            let cell_width = UnicodeWidthChar::width(ch).unwrap_or(1).min(1);
            let printable = if cell_width == 1 { ch } else { '?' };
            if current_width + cell_width > width {
                lines.push(current);
                current = String::new();
                current_width = 0;
            }
            current.push(printable);
            current_width += cell_width;
        }
        lines.push(current);
    }
    if lines.is_empty() {
        vec![String::new()]
    } else {
        lines
    }
}

fn truncate(text: &str, max: usize) -> String {
    let mut result = String::new();
    for ch in text.chars().take(max.saturating_sub(1)) {
        result.push(ch);
    }
    if text.chars().count() > max {
        result.push('…');
        result
    } else {
        text.to_string()
    }
}

fn label_width(label: &str) -> usize {
    UnicodeWidthStr::width(label)
}

fn preview_snippet(preview: &str, width: u16, max_lines: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for raw_line in preview.lines() {
        lines.extend(wrap_text(raw_line, width));
    }
    if lines.len() > max_lines {
        let mut clipped = lines[..max_lines.saturating_sub(1)].to_vec();
        clipped.push("[preview truncated]".to_string());
        clipped
    } else {
        lines
    }
}

#[derive(Debug, Clone)]
struct TopSegment {
    text: String,
    style: PackedStyle,
    priority: u8,
}

impl TopSegment {
    fn new(text: &str, style: PackedStyle, priority: u8) -> Self {
        Self {
            text: text.to_string(),
            style,
            priority,
        }
    }
}

fn build_segmented_top_line(mut segments: Vec<TopSegment>, width: u16, theme: Theme) -> StyledLine {
    let width = width as usize;
    let separator = " · ";
    while segmented_width(&segments, separator) > width && segments.len() > 1 {
        if let Some(idx) = segments
            .iter()
            .enumerate()
            .max_by_key(|(_, segment)| segment.priority)
            .map(|(idx, _)| idx)
        {
            segments.remove(idx);
        } else {
            break;
        }
    }

    let mut spans = Vec::new();
    for (idx, segment) in segments.into_iter().enumerate() {
        if idx > 0 {
            spans.push(StyledSpan {
                text: separator.to_string(),
                style: theme.dim(),
            });
        }
        spans.push(StyledSpan {
            text: segment.text,
            style: segment.style,
        });
    }
    StyledLine { spans }
}

fn segmented_width(segments: &[TopSegment], separator: &str) -> usize {
    let text_width: usize = segments
        .iter()
        .map(|segment| label_width(&segment.text))
        .sum();
    let separators = separator
        .len()
        .saturating_mul(segments.len().saturating_sub(1));
    text_width + separators
}

fn runtime_label(state: &AppState) -> String {
    state
        .current_trace
        .as_deref()
        .map(str::to_string)
        .unwrap_or_else(|| {
            if state.pending_action.is_some() {
                "awaiting approval".to_string()
            } else if state.status.starts_with("error") {
                "error".to_string()
            } else if state.is_generating {
                "streaming".to_string()
            } else if state.is_ready() {
                "ready".to_string()
            } else {
                "loading".to_string()
            }
        })
}

fn runtime_style(state: &AppState, theme: Theme) -> PackedStyle {
    if state.pending_action.is_some() {
        theme.chip_warning()
    } else if state.status.starts_with("error") {
        theme.chip_danger()
    } else if !state.is_ready() {
        theme.chip_accent()
    } else if state.is_generating || state.current_trace.is_some() {
        theme.badge_assistant()
    } else {
        theme.dim()
    }
}

fn is_runtime_animated(state: &AppState) -> bool {
    state.pending_action.is_some()
        || !state.is_ready()
        || state.is_generating
        || state.current_trace.is_some()
}

fn spinner_frame(tick: u64) -> &'static str {
    SPINNER_FRAMES[(tick as usize / 3) % SPINNER_FRAMES.len()]
}

fn active_cursor_span(theme: Theme, tick: u64) -> StyledSpan {
    StyledSpan {
        text: if (tick / 6).is_multiple_of(2) {
            " ▍".to_string()
        } else {
            " ▌".to_string()
        },
        style: if (tick / 6).is_multiple_of(2) {
            theme.badge_assistant()
        } else {
            theme.chip_accent()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::CommandSuggestion;
    use crate::events::PendingAction;
    use crate::safety::{InspectionDecision, InspectionReport, RiskLevel};

    #[test]
    fn collapsed_context_hides_controls_until_focused() {
        let message = ChatMessage {
            id: 1,
            role: Role::User,
            content: "Tool results:\n\n--- read_file(src/main.rs) ---".to_string(),
            transcript: crate::tui::state::TranscriptPresentation {
                collapsible: true,
                collapsed: true,
                summary: Some("tool results • 1 tool".to_string()),
                preview_lines: vec!["read_file(src/main.rs)".to_string()],
            },
        };

        let unfocused = build_collapsed_context(&message, Theme::default(), 80, false);
        let focused = build_collapsed_context(&message, Theme::default(), 80, true);

        let unfocused_text = unfocused
            .iter()
            .flat_map(|line| line.spans.iter().map(|span| span.text.as_str()))
            .collect::<String>();
        let focused_text = focused
            .iter()
            .flat_map(|line| line.spans.iter().map(|span| span.text.as_str()))
            .collect::<String>();

        assert!(!unfocused_text.contains("Ctrl+O"));
        assert!(focused_text.contains("Ctrl+O"));
    }

    #[test]
    fn composer_uses_search_prompt_marker_in_reverse_search_mode() {
        let mut state = AppState::new();
        state.input = "first".to_string();
        state.cursor = state.input.len();
        assert_eq!(state.submit_input(), "first");
        state.input = "draft".to_string();
        state.cursor = state.input.len();
        assert!(state.activate_reverse_search());

        let (lines, _) = build_composer(&state, Theme::default(), 80, None);
        assert_eq!(lines[0].spans[0].text, "? ");
    }

    #[test]
    fn composer_uses_command_prompt_marker_in_launcher_mode() {
        let mut state = AppState::new();
        assert!(state.activate_command_launcher(vec![CommandSuggestion {
            name: "/read".to_string(),
            usage: "/read <path>".to_string(),
            description: "load a file".to_string(),
            source: "builtin",
            group: "context",
        }]));

        let (lines, _) = build_composer(&state, Theme::default(), 80, None);
        assert_eq!(lines[0].spans[0].text, ": ");
    }

    #[test]
    fn approval_preview_uses_inline_gutter_prefix() {
        let mut state = AppState::new();
        state.set_pending_action(PendingAction {
            id: 1,
            kind: PendingActionKind::ShellCommand,
            title: "Approve shell command".to_string(),
            preview: "cargo check".to_string(),
            inspection: InspectionReport {
                operation: "bash".to_string(),
                decision: InspectionDecision::NeedsApproval,
                risk: RiskLevel::Low,
                summary: "Shell command requires approval before execution".to_string(),
                reasons: Vec::new(),
                targets: Vec::new(),
                segments: vec!["cargo check".to_string()],
                network_targets: Vec::new(),
            },
        });

        let approval = build_approval(&state, Theme::default(), 80).expect("approval");
        let joined = approval
            .iter()
            .flat_map(|line| line.spans.iter().map(|span| span.text.as_str()))
            .collect::<String>();
        assert!(joined.contains("approval"));
        assert!(joined.contains("│ cargo check"));
    }

    #[test]
    fn top_bar_reacts_to_loading_streaming_and_approval_states() {
        let mut state = AppState::new();

        let loading = build_top_bar(&state, Theme::default(), 120);
        let loading_text = loading[0]
            .spans
            .iter()
            .map(|span| span.text.as_str())
            .collect::<String>();
        assert!(loading_text.contains("loading"));
        assert!(loading[0]
            .spans
            .iter()
            .any(|span| span.style == Theme::default().chip_accent()));

        state.model_ready = true;
        state.is_generating = true;
        state.tick = 5;
        let streaming = build_top_bar(&state, Theme::default(), 120);
        let streaming_text = streaming[0]
            .spans
            .iter()
            .map(|span| span.text.as_str())
            .collect::<String>();
        assert!(streaming_text.contains("streaming"));
        assert!(streaming[0]
            .spans
            .iter()
            .any(|span| span.style == Theme::default().badge_assistant()));

        state.is_generating = false;
        state.set_pending_action(PendingAction {
            id: 1,
            kind: PendingActionKind::ShellCommand,
            title: "Approve shell command".to_string(),
            preview: "cargo check".to_string(),
            inspection: InspectionReport {
                operation: "bash".to_string(),
                decision: InspectionDecision::NeedsApproval,
                risk: RiskLevel::Medium,
                summary: "Shell command requires approval before execution".to_string(),
                reasons: Vec::new(),
                targets: Vec::new(),
                segments: vec!["cargo check".to_string()],
                network_targets: Vec::new(),
            },
        });
        let approval = build_top_bar(&state, Theme::default(), 120);
        let approval_text = approval[0]
            .spans
            .iter()
            .map(|span| span.text.as_str())
            .collect::<String>();
        assert!(approval_text.contains("awaiting approval"));
        assert!(approval[0]
            .spans
            .iter()
            .any(|span| span.style == Theme::default().chip_warning()));
    }

    #[test]
    fn spinner_frame_is_tick_driven_and_stable() {
        assert_eq!(spinner_frame(0), "·");
        assert_eq!(spinner_frame(3), "•");
        assert_eq!(spinner_frame(6), "◦");
        assert_eq!(spinner_frame(9), "•");
    }

    #[test]
    fn narrow_top_bar_keeps_runtime_segment() {
        let mut state = AppState::new();
        state.model_ready = true;
        state.is_generating = true;

        let line = build_top_bar(&state, Theme::default(), 28);
        let text = line[0]
            .spans
            .iter()
            .map(|span| span.text.as_str())
            .collect::<String>();

        assert!(text.contains("streaming"));
        assert!(!text.contains("cache hit"));
    }

    #[test]
    fn active_assistant_message_gets_cursor_span() {
        let message = ChatMessage {
            id: 1,
            role: Role::Assistant,
            content: "working".to_string(),
            transcript: crate::tui::state::TranscriptPresentation {
                collapsible: false,
                collapsed: false,
                summary: None,
                preview_lines: Vec::new(),
            },
        };

        let lines = build_standard_message(&message, Theme::default(), 80, true, 0);
        let text = lines[0]
            .spans
            .iter()
            .map(|span| span.text.as_str())
            .collect::<String>();
        assert!(text.contains("▍") || text.contains("▌"));
    }
}
