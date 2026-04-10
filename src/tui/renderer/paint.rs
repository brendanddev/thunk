mod chrome;

use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::events::PendingActionKind;
use crate::safety::RiskLevel;
use crate::tui::state::{AppState, ChatMessage, Role};

use super::buffer::{Cell, CellBuffer};
use super::layout::{LayoutPlan, Rect};
use super::style::{PackedStyle, Theme};
use super::symbols::SymbolPool;
#[cfg(test)]
use chrome::format_duration_display;
use chrome::{active_cursor_span, build_activity_line, build_top_bar, spinner_frame};

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
const CONVERSATION_GUTTER: &str = "│ ";
const SYSTEM_GUTTER: &str = "· ";
/// Fixed display-column width for the command name in the palette.
/// Commands are padded to this width so descriptions start in the same column.
const PALETTE_NAME_COL: usize = 14;

pub(crate) fn build_render_model(
    state: &mut AppState,
    theme: Theme,
    width: u16,
    cached_block: impl FnMut(&ChatMessage, bool, u16) -> Vec<StyledLine>,
) -> RenderModel {
    let top_bar = build_top_bar(state, theme, width);
    let transcript = build_transcript(state, theme, width, cached_block);
    let approval = build_approval(state, theme, width);
    let activity = build_activity_line(state, theme, width);
    let (composer, composer_cursor) =
        build_composer(state, theme, width, approval.as_deref(), activity);

    RenderModel {
        top_bar,
        transcript,
        composer,
        composer_cursor,
    }
}

fn build_transcript(
    state: &mut AppState,
    theme: Theme,
    width: u16,
    mut cached_block: impl FnMut(&ChatMessage, bool, u16) -> Vec<StyledLine>,
) -> Vec<RenderBlock> {
    let mut blocks = Vec::new();
    let active_assistant_id = if state.is_generating {
        state
            .messages
            .iter()
            .rev()
            .find(|message| message.role == Role::Assistant)
            .map(|message| message.id)
    } else {
        None
    };
    if state.messages.is_empty() {
        blocks.push(RenderBlock {
            message_id: None,
            lines: vec![single_span(
                "type a message, or / for commands.",
                theme.dim(),
            )],
        });
        return blocks;
    }

    for (idx, message) in state.messages.iter().enumerate() {
        let next = state.messages.get(idx + 1);
        if message.transcript.collapsible {
            let focused = state.is_focused_collapsible(message.id);
            let mut lines = cached_block(message, focused, width);
            if needs_transcript_gap(message, next) {
                lines.push(blank_line());
            }
            blocks.push(RenderBlock {
                message_id: Some(message.id),
                lines,
            });
        } else {
            let is_active_assistant =
                message.role == Role::Assistant && active_assistant_id == Some(message.id);
            let mut lines =
                build_standard_message(message, theme, width, is_active_assistant, state.tick);
            if needs_transcript_gap(message, next) {
                lines.push(blank_line());
            }
            blocks.push(RenderBlock {
                message_id: None,
                lines,
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
            if message.content.is_empty() && !is_active_assistant {
                "▍"
            } else {
                &message.content
            },
            theme,
            width,
            is_active_assistant.then(|| active_cursor_span(theme, tick)),
        ),
        Role::System => build_gutter_lines(SYSTEM_GUTTER, theme.dim(), &message.content, width),
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
    let continuation_indent = " ".repeat(label_width(label) + 2);
    let body_width = width
        .saturating_sub((label_width(label) + 5).min(u16::MAX as usize) as u16)
        .max(8);
    let wrapped = wrap_text(body, body_width);
    if wrapped.is_empty() {
        lines.push(StyledLine {
            spans: vec![
                StyledSpan {
                    text: CONVERSATION_GUTTER.to_string(),
                    style: theme.border(),
                },
                StyledSpan {
                    text: label.to_string(),
                    style: badge_style,
                },
            ],
        });
        return lines;
    }

    lines.push(StyledLine {
        spans: vec![
            StyledSpan {
                text: CONVERSATION_GUTTER.to_string(),
                style: theme.border(),
            },
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
            spans: vec![
                StyledSpan {
                    text: CONVERSATION_GUTTER.to_string(),
                    style: theme.border(),
                },
                StyledSpan {
                    text: continuation_indent.clone(),
                    style: theme.base(),
                },
                StyledSpan {
                    text: line,
                    style: theme.base(),
                },
            ],
        });
    }
    if let Some(cursor_span) = active_cursor {
        if let Some(last) = lines.last_mut() {
            last.spans.push(cursor_span);
        }
    }
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
                    theme.muted()
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
                    text: " Ctrl+O".to_string(),
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
                    text: " Ctrl+O".to_string(),
                    style: theme.dim(),
                },
            ],
        });
    }
    lines
}

fn build_approval(state: &AppState, theme: Theme, width: u16) -> Option<Vec<StyledLine>> {
    let pending = state.pending_action.as_ref()?;
    let approval_style = match pending.inspection.risk {
        RiskLevel::Low => theme.chip_accent(),
        RiskLevel::Medium => theme.chip_warning(),
        RiskLevel::High => theme.chip_danger(),
    };
    let mut lines = vec![StyledLine {
        spans: vec![
            StyledSpan {
                text: "! ".to_string(),
                style: approval_style,
            },
            StyledSpan {
                text: approval_kind_label(pending.kind.clone()).to_string(),
                style: approval_style,
            },
            StyledSpan {
                text: format!(
                    "  {}",
                    truncate(&pending.title, width.saturating_sub(10) as usize)
                ),
                style: theme.base(),
            },
        ],
    }];
    lines.push(single_span_with_gutter(
        SYSTEM_GUTTER,
        theme.dim(),
        &approval_summary_line(pending, width),
        theme.muted(),
    ));

    let preview_lines = approval_preview_line_cap(pending.kind.clone(), width, None);
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
    lines.push(single_span_with_gutter(
        SYSTEM_GUTTER,
        theme.dim(),
        "^Y approve  ^N reject",
        theme.muted(),
    ));
    Some(lines)
}

fn build_composer(
    state: &AppState,
    theme: Theme,
    width: u16,
    approval: Option<&[StyledLine]>,
    activity: Option<StyledLine>,
) -> (Vec<StyledLine>, (u16, u16)) {
    let inner_width = width.saturating_sub(2).max(8) as usize;
    let max_visible_rows = 8usize;
    let (visible_rows, cursor_row, cursor_col) =
        state.input_display_lines(inner_width, max_visible_rows);
    let mut lines = vec![blank_line()];

    if let Some(activity_line) = activity {
        lines.push(activity_line);
    }
    if let Some(approval_lines) = approval {
        for line in approval_lines {
            lines.push(line.clone());
        }
    }
    let prompt_offset = lines.len() as u16;

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
                        theme.muted()
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
        if !query.is_empty() {
            lines.push(StyledLine {
                spans: vec![StyledSpan {
                    text: format!("/ {query}"),
                    style: theme.dim(),
                }],
            });
        }
        for (entry, selected) in entries {
            let name = truncate(&entry.name, PALETTE_NAME_COL);
            let pad = PALETTE_NAME_COL.saturating_sub(name.chars().count());
            let padded_name = format!("{}{}", name, " ".repeat(pad));
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
                        text: padded_name,
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
                            &entry.description,
                            width.saturating_sub(PALETTE_NAME_COL as u16 + 4) as usize,
                        ),
                        style: theme.muted(),
                    },
                ],
            });
        }
        if let Some(selected) = selected_entry {
            lines.push(single_span(
                &format!(
                    "  {}",
                    truncate(&selected.usage, width.saturating_sub(2) as usize)
                ),
                theme.dim(),
            ));
            if !selected.aliases.is_empty() {
                lines.push(single_span(
                    &format!(
                        "  aka {}",
                        truncate(
                            &selected.aliases.join(", "),
                            width.saturating_sub(6) as usize
                        )
                    ),
                    theme.dim(),
                ));
            }
        }
    } else if let Some((query, current)) = state.reverse_search_view() {
        let query_display = if query.is_empty() {
            "·".to_string()
        } else {
            truncate(&query, 20)
        };
        let match_width = width.saturating_sub(query_display.chars().count() as u16 + 4) as usize;
        lines.push(StyledLine {
            spans: vec![
                StyledSpan {
                    text: query_display,
                    style: theme.dim(),
                },
                StyledSpan {
                    text: "  ".to_string(),
                    style: theme.dim(),
                },
                StyledSpan {
                    text: truncate(&current, match_width),
                    style: theme.muted(),
                },
            ],
        });
    } else if !state.autocomplete_preview_items(4).is_empty() {
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

    paint_transcript(
        buffer,
        symbols,
        &layout.transcript,
        &model.transcript,
        theme,
        state,
    );

    let composer_offset = paint_sheet(
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
            .saturating_add(composer_offset)
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

    let indicator_rows = usize::from(state.scroll_offset > 0) * usize::from(rect.height >= 2) * 2;
    let content_capacity = rect.height as usize - indicator_rows.min(rect.height as usize);
    let max_scroll = flattened.len().saturating_sub(content_capacity.max(1));
    state.max_scroll = max_scroll;
    state.scroll_offset = state.scroll_offset.min(max_scroll);
    let end = flattened.len().saturating_sub(state.scroll_offset);
    let start = end.saturating_sub(content_capacity.max(1));
    let visible = &flattened[start..end];

    let mut y = rect.y;
    let show_jump_indicator = state.scroll_offset > 0 && rect.height >= 2;
    if show_jump_indicator && y < rect.y.saturating_add(rect.height) {
        let indicator = single_span_with_gutter(
            SYSTEM_GUTTER,
            theme.dim(),
            &format!("↑ {} above", state.scroll_offset),
            theme.dim(),
        );
        paint_line(buffer, symbols, rect.x, y, rect.width, &indicator);
        y += 1;
    }

    for (message_id, line) in visible.iter().take(content_capacity.max(1)) {
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

    if show_jump_indicator && y < rect.y.saturating_add(rect.height) {
        let indicator =
            single_span_with_gutter(SYSTEM_GUTTER, theme.dim(), "↓ jump to latest", theme.dim());
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
) -> u16 {
    if !title.is_empty() {
        paint_line(
            buffer,
            symbols,
            rect.x,
            rect.y,
            rect.width,
            &single_span(title, theme.dim().with_bold()),
        );
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
        1
    } else {
        paint_lines(buffer, symbols, rect, lines);
        0
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

fn single_span_with_gutter(
    gutter: &str,
    gutter_style: PackedStyle,
    text: &str,
    style: PackedStyle,
) -> StyledLine {
    StyledLine {
        spans: vec![
            StyledSpan {
                text: gutter.to_string(),
                style: gutter_style,
            },
            StyledSpan {
                text: text.to_string(),
                style,
            },
        ],
    }
}

fn blank_line() -> StyledLine {
    StyledLine { spans: Vec::new() }
}

fn build_gutter_lines(
    gutter: &str,
    gutter_style: PackedStyle,
    text: &str,
    width: u16,
) -> Vec<StyledLine> {
    wrap_text(text, width.saturating_sub(gutter.len() as u16))
        .into_iter()
        .map(|line| single_span_with_gutter(gutter, gutter_style, &line, gutter_style))
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

fn needs_transcript_gap(current: &ChatMessage, next: Option<&ChatMessage>) -> bool {
    let Some(next) = next else {
        return false;
    };
    transcript_block_kind(current) != transcript_block_kind(next)
}

fn transcript_block_kind(message: &ChatMessage) -> u8 {
    if message.transcript.collapsible {
        0
    } else {
        match message.role {
            Role::User => 1,
            Role::Assistant => 2,
            Role::System => 3,
        }
    }
}

fn approval_kind_label(kind: PendingActionKind) -> &'static str {
    match kind {
        PendingActionKind::ShellCommand => "shell",
        PendingActionKind::FileWrite => "write",
        PendingActionKind::FileEdit => "edit",
    }
}

fn approval_summary_line(pending: &crate::events::PendingAction, width: u16) -> String {
    let detail = pending
        .inspection
        .reasons
        .first()
        .map(String::as_str)
        .unwrap_or(&pending.inspection.summary);
    truncate(detail, width.saturating_sub(4) as usize)
}

pub(crate) fn approval_preview_line_cap(
    kind: PendingActionKind,
    width: u16,
    terminal_height: Option<u16>,
) -> usize {
    let base = match kind {
        PendingActionKind::ShellCommand => 1,
        PendingActionKind::FileWrite | PendingActionKind::FileEdit => {
            if width < 72 {
                2
            } else {
                3
            }
        }
    };
    match terminal_height {
        Some(height) if height < 22 => base,
        Some(height) if height < 30 => match kind {
            PendingActionKind::ShellCommand => base.min(2),
            PendingActionKind::FileWrite | PendingActionKind::FileEdit => (base + 1).min(3),
        },
        Some(_) | None => match kind {
            PendingActionKind::ShellCommand => 2,
            PendingActionKind::FileWrite | PendingActionKind::FileEdit => (base + 1).min(4),
        },
    }
}

pub(crate) fn estimated_approval_rows(
    kind: PendingActionKind,
    width: u16,
    terminal_height: Option<u16>,
) -> u16 {
    let preview_rows = approval_preview_line_cap(kind, width, terminal_height) as u16;
    3 + preview_rows
}

#[cfg(test)]
mod tests;
