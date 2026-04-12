mod helpers;
mod input;
mod runtime;
#[cfg(test)]
mod tests;

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::commands::CommandSuggestion;
use crate::events::{MemorySnapshot, PendingAction, SessionInfo};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub id: u64,
    pub role: Role,
    pub content: String,
    pub transcript: TranscriptPresentation,
}

#[derive(Debug, Clone)]
pub struct TranscriptPresentation {
    pub collapsible: bool,
    pub collapsed: bool,
    pub summary: Option<String>,
    pub preview_lines: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct TraceEntry {
    pub label: String,
    pub success: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DirtySections(u8);

impl DirtySections {
    pub const NONE: Self = Self(0);
    pub const SIDEBAR: Self = Self(1 << 0);
    pub const HEADER: Self = Self(1 << 1);
    pub const CHAT: Self = Self(1 << 2);
    pub const APPROVAL: Self = Self(1 << 3);
    pub const INPUT: Self = Self(1 << 4);
    pub const ALL: Self =
        Self(Self::SIDEBAR.0 | Self::HEADER.0 | Self::CHAT.0 | Self::APPROVAL.0 | Self::INPUT.0);

    #[cfg(test)]
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl std::ops::BitOr for DirtySections {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for DirtySections {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

pub struct AppState {
    pub input: String,
    pub cursor: usize,
    pub messages: Vec<ChatMessage>,
    next_message_id: u64,
    pub is_generating: bool,
    pub scroll_offset: usize,
    pub max_scroll: usize,
    pub status: String,
    pub model_ready: bool,
    pub backend_name: String,
    pub tick: u64,
    pub last_tool_call: Option<String>,
    pub current_trace: Option<String>,
    pub recent_traces: VecDeque<TraceEntry>,
    pub current_session: Option<SessionInfo>,
    pub memory_snapshot: MemorySnapshot,
    pub pending_action: Option<PendingAction>,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub estimated_cost_usd: Option<f64>,
    pub show_top_bar_tokens: bool,
    pub show_top_bar_time: bool,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub tokens_saved: usize,
    pub last_cache_hit: Option<bool>,
    pub reflection_enabled: bool,
    pub eco_enabled: bool,
    pub debug_logging_enabled: bool,
    work_started_at: Option<Instant>,
    accumulated_work_duration: Duration,
    last_work_duration: Option<Duration>,
    grouped_trace_steps: Vec<String>,
    grouped_trace_failed: bool,
    autocomplete_matches: Vec<String>,
    autocomplete_index: usize,
    autocomplete_prefix: Option<String>,
    input_history: Vec<String>,
    history_cursor: Option<usize>,
    history_draft: Option<String>,
    reverse_search_active: bool,
    reverse_search_query: String,
    reverse_search_selection: usize,
    reverse_search_draft: Option<String>,
    command_launcher_active: bool,
    command_launcher_query: String,
    command_launcher_entries: Vec<CommandSuggestion>,
    command_launcher_selection: usize,
    command_launcher_draft: Option<String>,
    focused_collapsible_id: Option<u64>,
    visible_collapsible_ids: Vec<u64>,
    dirty_sections: DirtySections,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            input: String::new(),
            cursor: 0,
            messages: Vec::new(),
            next_message_id: 1,
            is_generating: false,
            scroll_offset: 0,
            max_scroll: 0,
            status: "loading...".to_string(),
            model_ready: false,
            backend_name: "...".to_string(),
            tick: 0,
            last_tool_call: None,
            current_trace: None,
            recent_traces: VecDeque::with_capacity(4),
            current_session: None,
            memory_snapshot: MemorySnapshot::default(),
            pending_action: None,
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            estimated_cost_usd: None,
            show_top_bar_tokens: true,
            show_top_bar_time: true,
            cache_hits: 0,
            cache_misses: 0,
            tokens_saved: 0,
            last_cache_hit: None,
            reflection_enabled: false,
            eco_enabled: false,
            debug_logging_enabled: false,
            work_started_at: None,
            accumulated_work_duration: Duration::ZERO,
            last_work_duration: None,
            grouped_trace_steps: Vec::new(),
            grouped_trace_failed: false,
            autocomplete_matches: Vec::new(),
            autocomplete_index: 0,
            autocomplete_prefix: None,
            input_history: Vec::new(),
            history_cursor: None,
            history_draft: None,
            reverse_search_active: false,
            reverse_search_query: String::new(),
            reverse_search_selection: 0,
            reverse_search_draft: None,
            command_launcher_active: false,
            command_launcher_query: String::new(),
            command_launcher_entries: Vec::new(),
            command_launcher_selection: 0,
            command_launcher_draft: None,
            focused_collapsible_id: None,
            visible_collapsible_ids: Vec::new(),
            dirty_sections: DirtySections::ALL,
        }
    }

    pub fn tick(&mut self) {
        self.tick = self.tick.wrapping_add(1);
    }
}
