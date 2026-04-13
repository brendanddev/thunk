use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use super::{StyledLine, StyledSpan};
use crate::tui::renderer::style::PackedStyle;

pub(crate) const CONVERSATION_GUTTER: &str = "│ ";
pub(crate) const SYSTEM_GUTTER: &str = "· ";
/// Fixed display-column width for the command name in the palette.
/// Commands are padded to this width so descriptions start in the same column.
pub(crate) const PALETTE_NAME_COL: usize = 14;

pub(crate) fn single_span(text: &str, style: PackedStyle) -> StyledLine {
    StyledLine {
        spans: vec![StyledSpan {
            text: text.to_string(),
            style,
        }],
    }
}

pub(crate) fn single_span_with_gutter(
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

pub(crate) fn blank_line() -> StyledLine {
    StyledLine { spans: Vec::new() }
}

pub(crate) fn build_gutter_lines(
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

pub(crate) fn wrap_text(text: &str, width: u16) -> Vec<String> {
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

pub(crate) fn truncate(text: &str, max: usize) -> String {
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

pub(crate) fn label_width(label: &str) -> usize {
    UnicodeWidthStr::width(label)
}

pub(crate) fn preview_snippet(preview: &str, width: u16, max_lines: usize) -> Vec<String> {
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
