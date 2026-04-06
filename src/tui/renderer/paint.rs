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
    pub approval: Option<Vec<StyledLine>>,
    pub composer: Vec<StyledLine>,
    pub composer_cursor: (u16, u16),
}

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
    let (composer, composer_cursor) = build_composer(state, theme, width);

    RenderModel {
        top_bar,
        transcript,
        approval,
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
    let cache_label = match state.last_cache_hit {
        Some(true) => "cache hit",
        Some(false) => "cache miss",
        None => "cache n/a",
    };
    let runtime = state.current_trace.as_deref().unwrap_or_else(|| {
        if state.pending_action.is_some() {
            "awaiting approval"
        } else if state.is_generating {
            "streaming"
        } else if state.is_ready() {
            "ready"
        } else {
            "loading"
        }
    });

    let chips = vec![
        chip("params", theme.chip_accent()),
        chip(&truncate(&state.backend_name, 18), theme.chip_neutral()),
        chip(&truncate(&session_label, 18), theme.chip_neutral()),
        chip(&truncate(runtime, 22), runtime_chip(state, theme)),
    ];

    let mut lines = wrap_chip_line(chips, width);
    lines.push(single_span(
        &format!(
            "{} msgs • {} tok • {} • mem {}f/{}s • Alt+Up history • Ctrl+R search",
            state.message_count(),
            state.total_tokens,
            cache_label,
            state.memory_snapshot.loaded_facts.len(),
            state.memory_snapshot.last_summary_paths.len()
        ),
        theme.dim(),
    ));
    lines.truncate(2);
    lines
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
            blocks.push(RenderBlock {
                message_id: None,
                lines: build_standard_message(message, theme, width),
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

fn build_standard_message(message: &ChatMessage, theme: Theme, width: u16) -> Vec<StyledLine> {
    match message.role {
        Role::User => {
            build_badged_message("you", theme.badge_user(), &message.content, theme, width)
        }
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
                style: if label.contains("params") {
                    theme.muted()
                } else {
                    theme.base()
                },
            },
        ],
    });

    for line in wrapped.into_iter().skip(1) {
        lines.push(StyledLine {
            spans: vec![StyledSpan {
                text: format!("{:width$}{}", "", line, width = label_width(label) + 2),
                style: if label.contains("params") {
                    theme.muted()
                } else {
                    theme.base()
                },
            }],
        });
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
        .take(2)
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
    lines.push(StyledLine {
        spans: vec![
            StyledSpan {
                text: edge.to_string(),
                style: edge_style,
            },
            StyledSpan {
                text: " Ctrl+O toggle • [ ] focus".to_string(),
                style: theme.dim(),
            },
        ],
    });
    lines.push(blank_line());
    lines
}

fn build_expanded_context(
    message: &ChatMessage,
    theme: Theme,
    width: u16,
    focused: bool,
) -> Vec<StyledLine> {
    let mut lines = build_collapsed_context(message, theme, width, focused);
    lines.pop();
    lines.pop();
    let edge = if focused { "▸" } else { "›" };
    let edge_style = if focused {
        theme.border_active()
    } else {
        theme.border()
    };
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
    lines.push(StyledLine {
        spans: vec![
            StyledSpan {
                text: edge.to_string(),
                style: edge_style,
            },
            StyledSpan {
                text: " Ctrl+O collapse".to_string(),
                style: theme.dim(),
            },
        ],
    });
    lines.push(blank_line());
    lines
}

fn build_approval(state: &AppState, theme: Theme, width: u16) -> Option<Vec<StyledLine>> {
    let pending = state.pending_action.as_ref()?;
    let risk_chip = match pending.inspection.risk {
        RiskLevel::Low => theme.chip_accent(),
        RiskLevel::Medium => theme.chip_warning(),
        RiskLevel::High => theme.chip_danger(),
    };
    let mut lines = vec![
        StyledLine {
            spans: vec![
                StyledSpan {
                    text: "approval".to_string(),
                    style: theme.chip_warning(),
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
    for line in preview_snippet(&pending.preview, width, preview_lines) {
        lines.push(StyledLine {
            spans: vec![StyledSpan {
                text: line,
                style: theme.muted(),
            }],
        });
    }
    lines.push(chip_line(
        &[
            (" Ctrl+Y approve ".to_string(), theme.chip_success()),
            (" Ctrl+N reject ".to_string(), theme.chip_danger()),
            (" /approve ".to_string(), theme.chip_neutral()),
            (" /reject ".to_string(), theme.chip_neutral()),
        ],
        width,
    ));
    Some(lines)
}

fn build_composer(state: &AppState, theme: Theme, width: u16) -> (Vec<StyledLine>, (u16, u16)) {
    let inner_width = width.saturating_sub(2).max(8) as usize;
    let max_visible_rows = 8usize;
    let (visible_rows, cursor_row, cursor_col) =
        state.input_display_lines(inner_width, max_visible_rows);
    let mut lines = Vec::new();
    for (idx, row) in visible_rows.iter().enumerate() {
        lines.push(StyledLine {
            spans: vec![
                StyledSpan {
                    text: if idx == 0 {
                        "› ".to_string()
                    } else {
                        "  ".to_string()
                    },
                    style: if state.is_generating {
                        theme.chip_warning()
                    } else {
                        theme.chip_accent()
                    },
                },
                StyledSpan {
                    text: row.clone(),
                    style: if state.is_generating {
                        theme.dim()
                    } else {
                        theme.base()
                    },
                },
            ],
        });
    }

    if let Some((query, current)) = state.reverse_search_view() {
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
    } else if !state.autocomplete_preview_matches(4).is_empty() {
        let chips = state
            .autocomplete_preview_matches(4)
            .into_iter()
            .map(|item| (truncate(&item, 18), theme.chip_neutral()))
            .collect::<Vec<_>>();
        lines.push(chip_line(&chips, width));
    } else {
        lines.push(single_span(
            "Enter send  Shift+Enter newline  Alt+↑ history  Ctrl+R search",
            theme.dim(),
        ));
    }

    let cursor_x = 2 + cursor_col as u16;
    let cursor_y = cursor_row as u16;
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

    if let (Some(approval_rect), Some(approval_lines)) = (layout.approval, &model.approval) {
        paint_sheet(
            buffer,
            symbols,
            &approval_rect,
            approval_lines,
            theme,
            " approval ",
        );
    }

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
        layout.composer.y.saturating_add(model.composer_cursor.1),
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
    for x in rect.x..rect.x.saturating_add(rect.width) {
        if rect.y < buffer.height() {
            buffer.set(
                x,
                rect.y,
                Cell {
                    symbol_id: symbols.intern("─"),
                    style: theme.border(),
                },
            );
        }
    }
    if !title.is_empty() {
        paint_line(
            buffer,
            symbols,
            rect.x,
            rect.y,
            rect.width,
            &single_span(title, theme.chip_neutral()),
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

fn chip(text: &str, style: PackedStyle) -> StyledLine {
    StyledLine {
        spans: vec![StyledSpan {
            text: text.to_string(),
            style,
        }],
    }
}

fn chip_line(chips: &[(String, PackedStyle)], width: u16) -> StyledLine {
    let mut spans = Vec::new();
    let mut used = 0usize;
    for (idx, (text, style)) in chips.iter().enumerate() {
        let text_width = UnicodeWidthStr::width(text.as_str());
        if idx > 0 {
            spans.push(StyledSpan {
                text: "  ".to_string(),
                style: PackedStyle::new(
                    super::style::Rgb::new(170, 180, 191),
                    super::style::Rgb::new(13, 16, 20),
                ),
            });
            used += 2;
        }
        if used + text_width > width as usize {
            break;
        }
        spans.push(StyledSpan {
            text: text.clone(),
            style: *style,
        });
        used += text_width;
    }
    StyledLine { spans }
}

fn wrap_chip_line(chips: Vec<StyledLine>, width: u16) -> Vec<StyledLine> {
    let mut lines = Vec::new();
    let mut current = Vec::<(String, PackedStyle)>::new();
    let mut used = 0usize;

    for chip in chips {
        let Some(span) = chip.spans.first() else {
            continue;
        };
        let chip_width = UnicodeWidthStr::width(span.text.as_str());
        let separator = if current.is_empty() { 0 } else { 2 };
        if !current.is_empty() && used + separator + chip_width > width as usize {
            lines.push(chip_line(&current, width));
            current.clear();
            used = 0;
        }
        if !current.is_empty() {
            used += 2;
        }
        used += chip_width;
        current.push((span.text.clone(), span.style));
    }

    if !current.is_empty() {
        lines.push(chip_line(&current, width));
    }

    if lines.is_empty() {
        lines.push(blank_line());
    }
    lines
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

fn runtime_chip(state: &AppState, theme: Theme) -> PackedStyle {
    if state.pending_action.is_some() {
        theme.chip_warning()
    } else if state.is_generating {
        theme.chip_accent()
    } else if state.is_ready() {
        theme.chip_success()
    } else {
        theme.chip_warning()
    }
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
