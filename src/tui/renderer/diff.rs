use std::io::{self, Write};

use crossterm::{
    cursor::MoveTo,
    queue,
    style::{Attribute, Print, SetAttribute, SetBackgroundColor, SetForegroundColor},
};

use super::buffer::CellBuffer;
use super::style::PackedStyle;
use super::symbols::SymbolPool;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PatchStats {
    pub changed_cells: usize,
    pub changed_runs: usize,
}

pub(crate) struct PatchWriter {
    last_style: Option<PackedStyle>,
}

impl PatchWriter {
    pub fn new() -> Self {
        Self { last_style: None }
    }

    pub fn reset_style(&mut self) {
        self.last_style = None;
    }

    pub fn write_diff<W: Write>(
        &mut self,
        out: &mut W,
        previous: &CellBuffer,
        current: &CellBuffer,
        symbols: &SymbolPool,
        cursor: (u16, u16),
    ) -> io::Result<PatchStats> {
        let mut stats = PatchStats::default();

        for y in 0..current.height() {
            let mut x = 0;
            while x < current.width() {
                if previous.get(x, y) == current.get(x, y) {
                    x += 1;
                    continue;
                }

                let start = x;
                let style = current.get(x, y).style;
                let mut text = String::new();

                while x < current.width() {
                    let prev_cell = previous.get(x, y);
                    let curr_cell = current.get(x, y);
                    if prev_cell == curr_cell || curr_cell.style != style {
                        break;
                    }
                    text.push_str(symbols.get(curr_cell.symbol_id));
                    x += 1;
                    stats.changed_cells += 1;
                }

                queue!(out, MoveTo(start, y))?;
                self.apply_style(out, style)?;
                queue!(out, Print(text))?;
                stats.changed_runs += 1;
            }
        }

        queue!(out, MoveTo(cursor.0, cursor.1))?;
        out.flush()?;
        Ok(stats)
    }

    fn apply_style<W: Write>(&mut self, out: &mut W, style: PackedStyle) -> io::Result<()> {
        if self.last_style == Some(style) {
            return Ok(());
        }
        queue!(
            out,
            SetAttribute(Attribute::Reset),
            SetForegroundColor(style.fg().to_crossterm()),
            SetBackgroundColor(style.bg().to_crossterm())
        )?;
        if style.is_bold() {
            queue!(out, SetAttribute(Attribute::Bold))?;
        }
        if style.is_dim() {
            queue!(out, SetAttribute(Attribute::Dim))?;
        }
        if style.is_italic() {
            queue!(out, SetAttribute(Attribute::Italic))?;
        }
        if style.is_underline() {
            queue!(out, SetAttribute(Attribute::Underlined))?;
        }
        if style.is_reverse() {
            queue!(out, SetAttribute(Attribute::Reverse))?;
        }
        self.last_style = Some(style);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::renderer::buffer::Cell;
    use crate::tui::renderer::style::{PackedStyle, Rgb};
    use crate::tui::renderer::symbols::SymbolPool;

    fn blank(pool: &mut SymbolPool) -> Cell {
        Cell {
            symbol_id: pool.blank_id(),
            style: PackedStyle::new(Rgb::new(255, 255, 255), Rgb::new(0, 0, 0)),
        }
    }

    #[test]
    fn unchanged_frames_emit_no_changes() {
        let mut pool = SymbolPool::new();
        let blank = blank(&mut pool);
        let previous = CellBuffer::new(3, 1, blank);
        let current = CellBuffer::new(3, 1, blank);
        let mut writer = PatchWriter::new();
        let mut out = Vec::new();
        let stats = writer
            .write_diff(&mut out, &previous, &current, &pool, (0, 0))
            .expect("diff");
        assert_eq!(stats.changed_cells, 0);
        assert_eq!(stats.changed_runs, 0);
    }

    #[test]
    fn contiguous_changes_coalesce_into_one_run() {
        let mut pool = SymbolPool::new();
        let blank = blank(&mut pool);
        let previous = CellBuffer::new(4, 1, blank);
        let mut current = CellBuffer::new(4, 1, blank);
        let style = blank.style;
        current.set(
            0,
            0,
            Cell {
                symbol_id: pool.intern("a"),
                style,
            },
        );
        current.set(
            1,
            0,
            Cell {
                symbol_id: pool.intern("b"),
                style,
            },
        );
        let mut writer = PatchWriter::new();
        let mut out = Vec::new();
        let stats = writer
            .write_diff(&mut out, &previous, &current, &pool, (0, 0))
            .expect("diff");
        assert_eq!(stats.changed_cells, 2);
        assert_eq!(stats.changed_runs, 1);
    }
}
