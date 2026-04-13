mod approval;
mod chrome;
mod composer;
mod render;
mod transcript;
mod util;

use super::style::{PackedStyle, Theme};
use util::{label_width, truncate};

const SPINNER_FRAMES: &[&str] = &["·", "•", "◦", "•"];

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

pub(crate) use approval::{approval_preview_line_cap, estimated_approval_rows};
pub(crate) use render::paint_model;
pub(crate) use transcript::{build_render_model, build_transcript_block};

#[cfg(test)]
#[allow(unused_imports)]
use crate::events::PendingActionKind;
#[cfg(test)]
#[allow(unused_imports)]
use crate::tui::renderer::buffer::{Cell, CellBuffer};
#[cfg(test)]
#[allow(unused_imports)]
use crate::tui::renderer::layout::{LayoutPlan, Rect};
#[cfg(test)]
#[allow(unused_imports)]
use crate::tui::renderer::symbols::SymbolPool;
#[cfg(test)]
#[allow(unused_imports)]
use crate::tui::state::{AppState, ChatMessage, Role};
#[cfg(test)]
#[allow(unused_imports)]
use approval::build_approval;
#[cfg(test)]
#[allow(unused_imports)]
use chrome::{build_activity_line, build_top_bar, format_duration_display, spinner_frame};
#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use composer::build_composer;
#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use render::{paint_sheet, paint_transcript};
#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use transcript::{build_collapsed_context, build_standard_message, build_transcript};
#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use util::{
    blank_line, single_span, CONVERSATION_GUTTER, PALETTE_NAME_COL, SYSTEM_GUTTER,
};

#[cfg(test)]
mod tests;
