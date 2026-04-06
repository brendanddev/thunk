mod buffer;
mod diff;
mod layout;
mod paint;
mod style;
mod symbols;

use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::{Hash, Hasher};
use std::io::{self, Write};
use std::time::Instant;

use crate::tui::state::ChatMessage;

use self::buffer::{Cell, CellBuffer};
use self::diff::PatchWriter;
use self::layout::layout_for;
use self::paint::{build_render_model, build_transcript_block, paint_model, StyledLine};
use self::style::Theme;
use self::symbols::SymbolPool;

use super::state::AppState;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RendererStats {
    pub frame_time_ms: u64,
    pub changed_cells: usize,
    pub changed_runs: usize,
    pub symbol_pool_size: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

struct FrameState {
    previous: CellBuffer,
    current: CellBuffer,
}

pub(crate) struct Renderer {
    theme: Theme,
    symbols: SymbolPool,
    frames: FrameState,
    patch_writer: PatchWriter,
    transcript_cache: HashMap<TranscriptCacheKey, Vec<StyledLine>>,
    cache_hits: usize,
    cache_misses: usize,
}

impl Renderer {
    pub fn new(width: u16, height: u16) -> Self {
        let mut symbols = SymbolPool::new();
        let blank = Cell {
            symbol_id: symbols.blank_id(),
            style: Theme::default().base(),
        };
        Self {
            theme: Theme::default(),
            symbols,
            frames: FrameState {
                previous: CellBuffer::new(width, height, blank),
                current: CellBuffer::new(width, height, blank),
            },
            patch_writer: PatchWriter::new(),
            transcript_cache: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    pub fn resize(&mut self, width: u16, height: u16) {
        self.frames.previous.resize(width, height);
        self.frames.current.resize(width, height);
        self.invalidate();
    }

    pub fn invalidate(&mut self) {
        self.frames.previous.clear();
        self.frames.current.clear();
        self.transcript_cache.clear();
        self.symbols.reset();
        self.patch_writer.reset_style();
        self.cache_hits = 0;
        self.cache_misses = 0;
    }

    pub fn render<W: Write>(
        &mut self,
        out: &mut W,
        state: &mut AppState,
    ) -> io::Result<RendererStats> {
        let started = Instant::now();
        let width = self.frames.current.width();
        let height = self.frames.current.height();

        let approval_height = if state.has_pending_action() {
            Some(self.estimate_approval_height(width))
        } else {
            None
        };
        let composer_height = self.estimate_composer_height(state, width);
        let layout = layout_for(width, height, composer_height, approval_height);
        let transcript_width = layout.transcript.width.saturating_sub(1).max(12);
        let theme = self.theme;
        let mut cache_hits = 0usize;
        let mut cache_misses = 0usize;
        let model = build_render_model(
            state,
            theme,
            transcript_width,
            layout.mode,
            |message, focused, message_width| {
                let key = TranscriptCacheKey::from_message(message, message_width, focused);
                if let Some(lines) = self.transcript_cache.get(&key) {
                    cache_hits += 1;
                    return lines.clone();
                }

                let lines = build_transcript_block(message, theme, message_width, focused);
                self.transcript_cache.insert(key, lines.clone());
                cache_misses += 1;
                lines
            },
        );
        self.cache_hits += cache_hits;
        self.cache_misses += cache_misses;

        self.frames.current.clear();
        let cursor = paint_model(
            &mut self.frames.current,
            &mut self.symbols,
            &model,
            layout,
            self.theme,
            state,
        );

        let patch = self.patch_writer.write_diff(
            out,
            &self.frames.previous,
            &self.frames.current,
            &self.symbols,
            cursor,
        )?;
        std::mem::swap(&mut self.frames.previous, &mut self.frames.current);
        let stats = RendererStats {
            frame_time_ms: started.elapsed().as_millis() as u64,
            changed_cells: patch.changed_cells,
            changed_runs: patch.changed_runs,
            symbol_pool_size: self.symbols.len(),
            cache_hits: self.cache_hits,
            cache_misses: self.cache_misses,
        };
        if self.transcript_cache.len() > 512 {
            self.transcript_cache.clear();
        }
        Ok(stats)
    }

    fn estimate_composer_height(&self, state: &AppState, width: u16) -> u16 {
        let inner_width = width.saturating_sub(4).max(8) as usize;
        let content_rows = state.input_content_rows(inner_width).min(8) as u16;
        let hint_rows = 2u16;
        content_rows + hint_rows + 1
    }

    fn estimate_approval_height(&self, width: u16) -> u16 {
        (width / 8).clamp(7, 12)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TranscriptCacheKey {
    id: u64,
    width: u16,
    focused: bool,
    collapsed: bool,
    signature: u64,
}

impl TranscriptCacheKey {
    fn from_message(message: &ChatMessage, width: u16, focused: bool) -> Self {
        Self {
            id: message.id,
            width,
            focused,
            collapsed: message.transcript.collapsed,
            signature: message_signature(message),
        }
    }
}

fn message_signature(message: &ChatMessage) -> u64 {
    let mut hasher = DefaultHasher::new();
    message.role.hash(&mut hasher);
    message.content.hash(&mut hasher);
    message.transcript.collapsible.hash(&mut hasher);
    message.transcript.collapsed.hash(&mut hasher);
    message.transcript.summary.hash(&mut hasher);
    message.transcript.preview_lines.hash(&mut hasher);
    hasher.finish()
}
