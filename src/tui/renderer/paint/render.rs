use super::util::{single_span, single_span_with_gutter, SYSTEM_GUTTER};
use super::{RenderBlock, RenderModel, StyledLine};
use crate::tui::renderer::buffer::{Cell, CellBuffer};
use crate::tui::renderer::layout::{LayoutPlan, Rect};
use crate::tui::renderer::style::{PackedStyle, Theme};
use crate::tui::renderer::symbols::SymbolPool;
use crate::tui::state::AppState;

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

pub(crate) fn paint_transcript(
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

pub(crate) fn paint_sheet(
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
