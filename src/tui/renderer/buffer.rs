use unicode_width::UnicodeWidthChar;

use super::style::PackedStyle;
use super::symbols::SymbolPool;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Cell {
    pub symbol_id: u32,
    pub style: PackedStyle,
}

#[derive(Clone)]
pub(crate) struct CellBuffer {
    width: u16,
    height: u16,
    cells: Vec<Cell>,
    blank: Cell,
}

impl CellBuffer {
    pub fn new(width: u16, height: u16, blank: Cell) -> Self {
        let len = width as usize * height as usize;
        Self {
            width,
            height,
            cells: vec![blank; len],
            blank,
        }
    }

    pub fn resize(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
        self.cells = vec![self.blank; width as usize * height as usize];
    }

    pub fn width(&self) -> u16 {
        self.width
    }

    pub fn height(&self) -> u16 {
        self.height
    }

    pub fn clear(&mut self) {
        self.cells.fill(self.blank);
    }

    pub fn fill(&mut self, cell: Cell) {
        self.cells.fill(cell);
    }

    pub fn get(&self, x: u16, y: u16) -> Cell {
        self.cells[self.index(x, y)]
    }

    pub fn set(&mut self, x: u16, y: u16, cell: Cell) {
        if x >= self.width || y >= self.height {
            return;
        }
        let idx = self.index(x, y);
        self.cells[idx] = cell;
    }

    pub fn fill_rect(&mut self, x: u16, y: u16, width: u16, height: u16, cell: Cell) {
        for row in y..y.saturating_add(height).min(self.height) {
            for col in x..x.saturating_add(width).min(self.width) {
                self.set(col, row, cell);
            }
        }
    }

    pub fn write_text_clipped(
        &mut self,
        x: u16,
        y: u16,
        text: &str,
        max_width: u16,
        style: PackedStyle,
        symbols: &mut SymbolPool,
    ) -> u16 {
        if y >= self.height || x >= self.width || max_width == 0 {
            return 0;
        }

        let mut written = 0u16;
        let mut cursor = x;
        let limit = x
            .saturating_add(max_width)
            .min(self.width)
            .saturating_sub(x);

        for ch in text.chars() {
            if written >= limit {
                break;
            }
            if ch == '\n' {
                break;
            }
            let display = match UnicodeWidthChar::width(ch) {
                Some(1) => ch,
                _ => '?',
            };
            let symbol_id = symbols.intern_char_lossy(display);
            self.set(cursor, y, Cell { symbol_id, style });
            cursor += 1;
            written += 1;
        }

        written
    }

    fn index(&self, x: u16, y: u16) -> usize {
        y as usize * self.width as usize + x as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::renderer::style::{PackedStyle, Rgb};

    fn blank_cell() -> Cell {
        Cell {
            symbol_id: 0,
            style: PackedStyle::new(Rgb::new(1, 1, 1), Rgb::new(0, 0, 0)),
        }
    }

    #[test]
    fn buffer_set_and_get_round_trip() {
        let mut buf = CellBuffer::new(4, 2, blank_cell());
        let cell = Cell {
            symbol_id: 2,
            style: blank_cell().style,
        };
        buf.set(1, 1, cell);
        assert_eq!(buf.get(1, 1), cell);
    }

    #[test]
    fn buffer_write_text_clips_to_width() {
        let mut pool = SymbolPool::new();
        let mut buf = CellBuffer::new(4, 1, blank_cell());
        let written = buf.write_text_clipped(0, 0, "hello", 3, blank_cell().style, &mut pool);
        assert_eq!(written, 3);
        assert_eq!(pool.get(buf.get(2, 0).symbol_id), "l");
    }

    #[test]
    fn buffer_fill_replaces_all_cells() {
        let mut buf = CellBuffer::new(2, 2, blank_cell());
        let filled = Cell {
            symbol_id: 9,
            style: blank_cell().style,
        };
        buf.fill(filled);
        assert_eq!(buf.get(0, 0), filled);
        assert_eq!(buf.get(1, 1), filled);
    }
}
