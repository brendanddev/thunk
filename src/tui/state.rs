// src/tui/state.rs

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::commands::CommandSuggestion;
use crate::events::{MemorySnapshot, PendingAction, ProgressStatus, ProgressTrace, SessionInfo};

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
    /// Current input text
    pub input: String,

    /// Cursor position within input (byte index)
    pub cursor: usize,

    /// All chat messages
    pub messages: Vec<ChatMessage>,

    /// Next transcript item id for local UI-only metadata.
    next_message_id: u64,

    /// Whether the model is generating
    pub is_generating: bool,

    /// Lines scrolled up from bottom (0 = pinned to bottom)
    pub scroll_offset: usize,

    /// Maximum scroll value (updated each frame based on content size)
    pub max_scroll: usize,

    /// Sidebar status string
    pub status: String,

    /// Whether the model is loaded and ready
    pub model_ready: bool,

    /// Active backend name shown in sidebar
    pub backend_name: String,

    /// Frame tick counter — increments every loop iteration for animations
    pub tick: u64,

    /// Last tool call being executed, shown in sidebar
    pub last_tool_call: Option<String>,

    /// Current transient progress trace shown in the sidebar
    pub current_trace: Option<String>,

    /// Recently completed progress traces shown under the current activity
    pub recent_traces: VecDeque<TraceEntry>,

    /// Active session metadata shown in the sidebar.
    pub current_session: Option<SessionInfo>,

    /// Current mirrored memory snapshot from the model thread.
    pub memory_snapshot: MemorySnapshot,

    /// Action currently awaiting approval
    pub pending_action: Option<PendingAction>,

    /// Session budget tracking
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub estimated_cost_usd: Option<f64>,

    /// Session cache tracking
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub tokens_saved: usize,
    pub last_cache_hit: Option<bool>,

    /// Whether reflection is enabled for the current session
    pub reflection_enabled: bool,

    /// Whether eco mode is enabled for the current session
    pub eco_enabled: bool,

    /// Whether separate content debug logging is enabled for the current session
    pub debug_logging_enabled: bool,

    /// When the current active work slice started, if the assistant is actively working.
    work_started_at: Option<Instant>,

    /// Accumulated active time for the current turn, including resumed phases.
    accumulated_work_duration: Duration,

    /// Duration of the most recently completed turn or slash-command job.
    last_work_duration: Option<Duration>,

    /// Completed trace labels collected for the current turn and collapsed when it ends.
    grouped_trace_steps: Vec<String>,

    /// Whether any completed trace in the current group failed.
    grouped_trace_failed: bool,

    /// Slash-command autocomplete candidates for the current input prefix.
    autocomplete_matches: Vec<String>,

    /// Index of the currently selected autocomplete suggestion.
    autocomplete_index: usize,

    /// Input prefix used to build the current autocomplete cycle.
    autocomplete_prefix: Option<String>,

    /// Submitted prompts/commands available for recall and edit.
    input_history: Vec<String>,

    /// Current history position when recalling prior submissions.
    history_cursor: Option<usize>,

    /// Draft input preserved while navigating submitted history.
    history_draft: Option<String>,

    /// Whether reverse search is active for submitted prompts/commands.
    reverse_search_active: bool,

    /// Query text for the current reverse search.
    reverse_search_query: String,

    /// Selected match index within the current reverse-search results.
    reverse_search_selection: usize,

    /// Draft input preserved while reverse search is active.
    reverse_search_draft: Option<String>,

    /// Whether the command launcher is active.
    command_launcher_active: bool,

    /// Query text for the current command launcher.
    command_launcher_query: String,

    /// Available command suggestions for the launcher.
    command_launcher_entries: Vec<CommandSuggestion>,

    /// Selected result within the launcher.
    command_launcher_selection: usize,

    /// Draft input preserved while command launcher is active.
    command_launcher_draft: Option<String>,

    /// Focused collapsible transcript item id, if any.
    focused_collapsible_id: Option<u64>,

    /// Collapsible transcript items visible in the current chat viewport.
    visible_collapsible_ids: Vec<u64>,

    /// Dirty UI sections used by the paced render loop.
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

    /// Increment the tick counter each frame for animations
    pub fn tick(&mut self) {
        self.tick = self.tick.wrapping_add(1);
    }

    /// Submit input and clear it, returning the text
    pub fn submit_input(&mut self) -> String {
        let submitted = self.input.clone();
        if !submitted.is_empty() {
            self.input_history.push(submitted.clone());
        }
        self.history_cursor = None;
        self.history_draft = None;
        self.exit_reverse_search();
        self.exit_command_launcher();
        self.cursor = 0;
        self.scroll_offset = 0;
        self.clear_autocomplete();
        self.mark_dirty(DirtySections::INPUT | DirtySections::CHAT | DirtySections::SIDEBAR);
        std::mem::take(&mut self.input)
    }

    /// Insert a character at the cursor position
    pub fn insert_char(&mut self, c: char) {
        self.input.insert(self.cursor, c);
        self.cursor += c.len_utf8();
        self.history_cursor = None;
        self.history_draft = None;
        self.exit_reverse_search();
        self.exit_command_launcher();
        self.clear_autocomplete();
        self.mark_dirty(DirtySections::INPUT);
    }

    /// Insert a string at cursor (for paste)
    pub fn insert_str(&mut self, s: &str) {
        self.input.insert_str(self.cursor, s);
        self.cursor += s.len();
        self.history_cursor = None;
        self.history_draft = None;
        self.exit_reverse_search();
        self.exit_command_launcher();
        self.clear_autocomplete();
        self.mark_dirty(DirtySections::INPUT);
    }

    /// Insert a newline at the cursor position.
    pub fn insert_newline(&mut self) {
        self.insert_char('\n');
    }

    /// Delete character before cursor (backspace)
    pub fn delete_char_before(&mut self) {
        if self.cursor == 0 {
            return;
        }
        // Find the previous char boundary
        let mut prev = self.cursor - 1;
        while !self.input.is_char_boundary(prev) {
            prev -= 1;
        }
        self.input.remove(prev);
        self.cursor = prev;
        self.exit_command_launcher();
        self.clear_autocomplete();
        self.mark_dirty(DirtySections::INPUT);
    }

    /// Delete word before cursor (Alt+Backspace)
    pub fn delete_word_before(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let before = &self.input[..self.cursor];
        // Skip trailing spaces then find start of word
        let trim_end = before.trim_end_matches(' ').len();
        let word_start = before[..trim_end].rfind(' ').map(|i| i + 1).unwrap_or(0);
        self.input.drain(word_start..self.cursor);
        self.cursor = word_start;
        self.exit_command_launcher();
        self.clear_autocomplete();
        self.mark_dirty(DirtySections::INPUT);
    }

    /// Move cursor left one character
    pub fn cursor_left(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let mut prev = self.cursor - 1;
        while !self.input.is_char_boundary(prev) {
            prev -= 1;
        }
        self.cursor = prev;
        self.mark_dirty(DirtySections::INPUT);
    }

    /// Move cursor right one character
    pub fn cursor_right(&mut self) {
        if self.cursor >= self.input.len() {
            return;
        }
        let mut next = self.cursor + 1;
        while !self.input.is_char_boundary(next) {
            next += 1;
        }
        self.cursor = next;
        self.mark_dirty(DirtySections::INPUT);
    }

    /// Move cursor to start of line
    pub fn cursor_home(&mut self) {
        self.cursor = self.current_line_start();
        self.mark_dirty(DirtySections::INPUT);
    }

    /// Move cursor to end of line
    pub fn cursor_end(&mut self) {
        self.cursor = self.current_line_end();
        self.mark_dirty(DirtySections::INPUT);
    }

    pub fn clear_input(&mut self) {
        self.input.clear();
        self.cursor = 0;
        self.history_cursor = None;
        self.history_draft = None;
        self.exit_reverse_search();
        self.exit_command_launcher();
        self.clear_autocomplete();
        self.mark_dirty(DirtySections::INPUT);
    }

    pub fn recall_previous_input(&mut self) -> bool {
        if self.input_history.is_empty() {
            return false;
        }

        let next_index = match self.history_cursor {
            Some(current) if current > 0 => current - 1,
            Some(current) => current,
            None => {
                self.history_draft = Some(self.input.clone());
                self.input_history.len() - 1
            }
        };
        self.history_cursor = Some(next_index);
        self.set_input_text(self.input_history[next_index].clone());
        true
    }

    pub fn recall_next_input(&mut self) -> bool {
        let Some(current) = self.history_cursor else {
            return false;
        };

        if current + 1 < self.input_history.len() {
            self.history_cursor = Some(current + 1);
            self.set_input_text(self.input_history[current + 1].clone());
        } else {
            let draft = self.history_draft.take().unwrap_or_default();
            self.history_cursor = None;
            self.set_input_text(draft);
        }
        true
    }

    pub fn is_reverse_search_active(&self) -> bool {
        self.reverse_search_active
    }

    pub fn is_command_launcher_active(&self) -> bool {
        self.command_launcher_active
    }

    pub fn activate_reverse_search(&mut self) -> bool {
        if self.input_history.is_empty() {
            return false;
        }
        if !self.reverse_search_active {
            self.reverse_search_active = true;
            self.reverse_search_query.clear();
            self.reverse_search_selection = 0;
            self.reverse_search_draft = Some(self.input.clone());
        }
        self.apply_reverse_search_match();
        self.mark_dirty(DirtySections::INPUT);
        true
    }

    pub fn reverse_search_push_char(&mut self, c: char) {
        if !self.reverse_search_active {
            return;
        }
        self.reverse_search_query.push(c);
        self.reverse_search_selection = 0;
        self.apply_reverse_search_match();
        self.mark_dirty(DirtySections::INPUT);
    }

    pub fn reverse_search_backspace(&mut self) {
        if !self.reverse_search_active {
            return;
        }
        self.reverse_search_query.pop();
        self.reverse_search_selection = 0;
        self.apply_reverse_search_match();
        self.mark_dirty(DirtySections::INPUT);
    }

    pub fn reverse_search_cycle(&mut self) -> bool {
        if !self.reverse_search_active {
            return self.activate_reverse_search();
        }

        let matches = self.reverse_search_matches();
        if matches.is_empty() {
            return false;
        }

        self.reverse_search_selection = (self.reverse_search_selection + 1) % matches.len();
        self.set_input_text(matches[self.reverse_search_selection].clone());
        self.mark_dirty(DirtySections::INPUT);
        true
    }

    pub fn accept_reverse_search(&mut self) -> bool {
        if !self.reverse_search_active {
            return false;
        }
        self.reverse_search_active = false;
        self.reverse_search_query.clear();
        self.reverse_search_selection = 0;
        self.reverse_search_draft = None;
        self.mark_dirty(DirtySections::INPUT);
        true
    }

    pub fn cancel_reverse_search(&mut self) -> bool {
        if !self.reverse_search_active {
            return false;
        }
        let draft = self.reverse_search_draft.take().unwrap_or_default();
        self.reverse_search_active = false;
        self.reverse_search_query.clear();
        self.reverse_search_selection = 0;
        self.set_input_text(draft);
        self.mark_dirty(DirtySections::INPUT);
        true
    }

    pub fn reverse_search_view(&self) -> Option<(String, String)> {
        if !self.reverse_search_active {
            return None;
        }
        let current = self
            .reverse_search_matches()
            .get(self.reverse_search_selection)
            .cloned()
            .unwrap_or_default();
        Some((self.reverse_search_query.clone(), current))
    }

    pub fn activate_command_launcher(&mut self, entries: Vec<CommandSuggestion>) -> bool {
        if entries.is_empty() {
            return false;
        }
        if !self.command_launcher_active {
            self.command_launcher_active = true;
            self.command_launcher_query.clear();
            self.command_launcher_selection = 0;
            self.command_launcher_draft = Some(self.input.clone());
        }
        self.command_launcher_entries = entries;
        self.apply_command_launcher_match();
        self.mark_dirty(DirtySections::INPUT);
        true
    }

    pub fn command_launcher_push_char(&mut self, c: char) {
        if !self.command_launcher_active {
            return;
        }
        self.command_launcher_query.push(c);
        self.command_launcher_selection = 0;
        self.apply_command_launcher_match();
        self.mark_dirty(DirtySections::INPUT);
    }

    pub fn command_launcher_backspace(&mut self) {
        if !self.command_launcher_active {
            return;
        }
        self.command_launcher_query.pop();
        self.command_launcher_selection = 0;
        self.apply_command_launcher_match();
        self.mark_dirty(DirtySections::INPUT);
    }

    pub fn command_launcher_cycle(&mut self, reverse: bool) -> bool {
        if !self.command_launcher_active {
            return false;
        }
        let matches = self.command_launcher_matches();
        if matches.is_empty() {
            return false;
        }
        if reverse {
            if self.command_launcher_selection == 0 {
                self.command_launcher_selection = matches.len() - 1;
            } else {
                self.command_launcher_selection -= 1;
            }
        } else {
            self.command_launcher_selection = (self.command_launcher_selection + 1) % matches.len();
        }
        self.apply_command_launcher_match();
        self.mark_dirty(DirtySections::INPUT);
        true
    }

    pub fn accept_command_launcher(&mut self) -> Option<String> {
        if !self.command_launcher_active {
            return None;
        }
        let selected = self
            .command_launcher_matches()
            .get(self.command_launcher_selection)
            .map(|item| item.name.clone())?;
        self.command_launcher_active = false;
        self.command_launcher_query.clear();
        self.command_launcher_selection = 0;
        self.command_launcher_draft = None;
        let command = format!("{selected} ");
        self.set_input_text(command.clone());
        self.mark_dirty(DirtySections::INPUT);
        Some(command)
    }

    pub fn cancel_command_launcher(&mut self) -> bool {
        if !self.command_launcher_active {
            return false;
        }
        let draft = self.command_launcher_draft.take().unwrap_or_default();
        self.command_launcher_active = false;
        self.command_launcher_query.clear();
        self.command_launcher_selection = 0;
        self.command_launcher_entries.clear();
        self.set_input_text(draft);
        self.mark_dirty(DirtySections::INPUT);
        true
    }

    pub fn command_launcher_view(
        &self,
        max: usize,
    ) -> Option<(String, Vec<(CommandSuggestion, bool)>)> {
        if !self.command_launcher_active {
            return None;
        }
        let preview = self
            .command_launcher_matches()
            .into_iter()
            .take(max)
            .enumerate()
            .map(|(idx, item)| (item, idx == self.command_launcher_selection))
            .collect::<Vec<_>>();
        Some((self.command_launcher_query.clone(), preview))
    }

    pub fn add_user_message(&mut self, content: &str) {
        let message = self.build_chat_message(Role::User, content);
        self.messages.push(message);
        self.scroll_offset = 0;
        self.mark_dirty(DirtySections::CHAT | DirtySections::SIDEBAR | DirtySections::HEADER);
    }

    pub fn start_assistant_message(&mut self) {
        let message = self.build_chat_message(Role::Assistant, String::new());
        self.messages.push(message);
        self.scroll_offset = 0;
        self.mark_dirty(DirtySections::CHAT | DirtySections::SIDEBAR | DirtySections::HEADER);
    }

    fn ensure_open_assistant_message(&mut self) {
        let has_open_assistant = self
            .messages
            .last()
            .map(|msg| msg.role == Role::Assistant)
            .unwrap_or(false);
        if !has_open_assistant {
            self.start_assistant_message();
        }
    }

    pub fn start_generation(&mut self, label: &str, show_placeholder: bool) {
        if self.should_begin_new_trace_group() {
            self.grouped_trace_steps.clear();
            self.grouped_trace_failed = false;
        }
        self.resume_work_timer();
        self.is_generating = true;
        self.status = label.to_string();
        if show_placeholder {
            self.ensure_open_assistant_message();
        }
        self.scroll_offset = 0;
        self.mark_dirty(DirtySections::SIDEBAR | DirtySections::HEADER | DirtySections::CHAT);
    }

    pub fn append_token(&mut self, token: &str) {
        self.ensure_open_assistant_message();
        if let Some(last) = self.messages.last_mut() {
            if last.role == Role::Assistant {
                last.content.push_str(token);
            }
        }
        self.scroll_offset = 0;
        self.mark_dirty(DirtySections::CHAT);
    }

    pub fn finish_response(&mut self) {
        self.finish_work_timer();
        self.is_generating = false;
        self.status = "ready".to_string();
        self.last_tool_call = None;
        self.current_trace = None;
        self.flush_grouped_trace_summary();
        self.mark_dirty(DirtySections::ALL);
    }

    pub fn add_error(&mut self, error: &str) {
        self.finish_work_timer();
        self.is_generating = false;
        self.status = "ready".to_string();
        self.current_trace = None;
        self.flush_grouped_trace_summary();
        let message = self.build_chat_message(Role::Assistant, format!("error: {error}"));
        self.messages.push(message);
        self.scroll_offset = 0;
        self.mark_dirty(DirtySections::ALL);
    }

    pub fn message_count(&self) -> usize {
        self.messages
            .iter()
            .filter(|m| m.role != Role::System)
            .count()
    }

    pub fn scroll_up(&mut self, lines: usize) {
        self.scroll_offset = self
            .scroll_offset
            .saturating_add(lines)
            .min(self.max_scroll);
        self.mark_dirty(DirtySections::CHAT);
    }

    pub fn scroll_down(&mut self, lines: usize) {
        self.scroll_offset = self.scroll_offset.saturating_sub(lines);
        self.mark_dirty(DirtySections::CHAT);
    }

    pub fn set_status(&mut self, status: &str) {
        self.status = status.to_string();
        if status == "ready" {
            self.model_ready = true;
        }
        self.mark_dirty(DirtySections::SIDEBAR | DirtySections::HEADER | DirtySections::INPUT);
    }

    pub fn set_backend_name(&mut self, name: String) {
        self.backend_name = name.clone();
        if self.total_tokens == 0 {
            self.estimated_cost_usd = if name.starts_with("llama.cpp") || name.starts_with("ollama")
            {
                Some(0.0)
            } else {
                None
            };
        }
        self.mark_dirty(DirtySections::SIDEBAR | DirtySections::HEADER);
    }

    pub fn set_pending_action(&mut self, action: PendingAction) {
        self.pause_work_timer();
        self.pending_action = Some(action);
        self.is_generating = false;
        self.status = "awaiting approval".to_string();
        self.mark_dirty(DirtySections::SIDEBAR | DirtySections::HEADER | DirtySections::APPROVAL);
    }

    pub fn mark_pending_action_submitted(&mut self, decision: &str) {
        if self.pending_action.is_some() {
            self.status = format!("{decision}...");
            self.mark_dirty(
                DirtySections::SIDEBAR | DirtySections::HEADER | DirtySections::APPROVAL,
            );
        }
    }

    pub fn clear_pending_action(&mut self) {
        self.pending_action = None;
        if !self.is_generating {
            self.status = "ready".to_string();
        }
        self.mark_dirty(DirtySections::SIDEBAR | DirtySections::HEADER | DirtySections::APPROVAL);
    }

    pub fn pending_action_id(&self) -> Option<u64> {
        self.pending_action.as_ref().map(|action| action.id)
    }

    pub fn has_pending_action(&self) -> bool {
        self.pending_action.is_some()
    }

    pub fn normalized_paste(text: &str) -> String {
        text.replace("\r\n", "\n").replace('\r', "\n")
    }

    pub fn input_display_lines(
        &self,
        width: usize,
        max_visible_rows: usize,
    ) -> (Vec<String>, usize, usize) {
        let wrapped = wrap_input_for_display(&self.input, width);
        let cursor = cursor_visual_position(&self.input, self.cursor, width);
        let total_rows = wrapped.len().max(1);
        let start_row = if total_rows <= max_visible_rows {
            0
        } else {
            cursor
                .0
                .saturating_add(1)
                .saturating_sub(max_visible_rows)
                .min(total_rows.saturating_sub(max_visible_rows))
        };
        let end_row = (start_row + max_visible_rows).min(total_rows);
        let visible = wrapped[start_row..end_row].to_vec();
        (visible, cursor.0.saturating_sub(start_row), cursor.1)
    }

    pub fn input_content_rows(&self, width: usize) -> usize {
        wrap_input_for_display(&self.input, width).len().max(1)
    }

    pub fn update_budget(
        &mut self,
        prompt_tokens: usize,
        completion_tokens: usize,
        total_tokens: usize,
        estimated_cost_usd: Option<f64>,
    ) {
        self.prompt_tokens = prompt_tokens;
        self.completion_tokens = completion_tokens;
        self.total_tokens = total_tokens;
        self.estimated_cost_usd = estimated_cost_usd;
        self.mark_dirty(DirtySections::SIDEBAR | DirtySections::HEADER);
    }

    pub fn set_reflection_enabled(&mut self, enabled: bool) {
        self.reflection_enabled = enabled;
        self.mark_dirty(DirtySections::SIDEBAR | DirtySections::HEADER);
    }

    pub fn set_eco_enabled(&mut self, enabled: bool) {
        self.eco_enabled = enabled;
        self.mark_dirty(DirtySections::SIDEBAR | DirtySections::HEADER);
    }

    pub fn set_debug_logging_enabled(&mut self, enabled: bool) {
        self.debug_logging_enabled = enabled;
        self.mark_dirty(DirtySections::SIDEBAR | DirtySections::HEADER);
    }

    pub fn update_cache(
        &mut self,
        last_hit: bool,
        hits: usize,
        misses: usize,
        tokens_saved: usize,
    ) {
        self.cache_hits = hits;
        self.cache_misses = misses;
        self.tokens_saved = tokens_saved;
        self.last_cache_hit = Some(last_hit);
        self.mark_dirty(DirtySections::SIDEBAR | DirtySections::HEADER);
    }

    pub fn is_ready(&self) -> bool {
        self.model_ready
    }

    pub fn add_system_message(&mut self, content: &str) {
        let message = self.build_chat_message(Role::System, content);
        self.messages.push(message);
        self.scroll_offset = 0;
        self.mark_dirty(DirtySections::CHAT | DirtySections::SIDEBAR | DirtySections::HEADER);
    }

    fn push_recent_trace(&mut self, label: &str, success: bool) {
        if label.trim().is_empty() {
            return;
        }
        self.recent_traces.push_front(TraceEntry {
            label: label.to_string(),
            success,
        });
        while self.recent_traces.len() > 4 {
            self.recent_traces.pop_back();
        }
    }

    fn add_trace_chat_message(&mut self, prefix: &str, label: &str) {
        self.add_system_message(&format!("{prefix} {label}"));
    }

    pub fn apply_trace(&mut self, trace: ProgressTrace) {
        match trace.status {
            ProgressStatus::Started | ProgressStatus::Updated => {
                self.current_trace = Some(trace.label.clone());
                if trace.persist && matches!(trace.status, ProgressStatus::Started) {
                    self.add_trace_chat_message("→", &trace.label);
                }
                self.mark_dirty(DirtySections::SIDEBAR | DirtySections::HEADER);
            }
            ProgressStatus::Finished => {
                self.current_trace = None;
                if self.is_grouping_turn_traces() {
                    self.push_grouped_trace_step(&trace.label, true);
                } else {
                    self.push_recent_trace(&trace.label, true);
                }
                if trace.persist {
                    self.add_trace_chat_message("✓", &trace.label);
                }
                self.mark_dirty(DirtySections::SIDEBAR | DirtySections::HEADER);
            }
            ProgressStatus::Failed => {
                self.current_trace = None;
                if self.is_grouping_turn_traces() {
                    self.push_grouped_trace_step(&trace.label, false);
                } else {
                    self.push_recent_trace(&trace.label, false);
                }
                if trace.persist {
                    self.add_trace_chat_message("✕", &trace.label);
                }
                self.mark_dirty(DirtySections::SIDEBAR | DirtySections::HEADER);
            }
        }
    }

    /// Restore a previous session into the display. Called on startup when a saved session exists.
    /// Injected context messages (tool results, slash-command output) are shown as muted system
    /// lines rather than user messages to keep the display clean.
    pub fn restore_session(
        &mut self,
        session: SessionInfo,
        messages: Vec<(String, String)>,
        saved_at: Option<u64>,
    ) {
        self.clear_messages();
        self.set_session_info(session.clone());
        if let Some(saved_at) = saved_at {
            let age = describe_session_age(saved_at);
            let label = session
                .name
                .as_deref()
                .map(|name| format!("{name} · {age}"))
                .unwrap_or(age);
            self.add_system_message(&format!("session resumed ({label})"));
        }

        for (role, content) in messages {
            match role.as_str() {
                "assistant" => {
                    let message = self.build_chat_message(Role::Assistant, content);
                    self.messages.push(message);
                }
                "user" => {
                    let display_role = if is_injected_context(&content) {
                        Role::System
                    } else {
                        Role::User
                    };
                    let message = self.build_chat_message(display_role, content);
                    self.messages.push(message);
                }
                _ => {}
            }
        }
        self.scroll_offset = 0;
        self.mark_dirty(DirtySections::ALL);
    }

    pub fn set_session_info(&mut self, session: SessionInfo) {
        self.current_session = Some(session);
        self.mark_dirty(DirtySections::SIDEBAR | DirtySections::HEADER);
    }

    pub fn set_memory_snapshot(&mut self, snapshot: MemorySnapshot) {
        self.memory_snapshot = snapshot;
        self.mark_dirty(DirtySections::SIDEBAR | DirtySections::HEADER);
    }

    pub fn transcript_item_counts(&self) -> (usize, usize) {
        let total = self
            .messages
            .iter()
            .filter(|message| message.transcript.collapsible)
            .count();
        let collapsed = self
            .messages
            .iter()
            .filter(|message| message.transcript.collapsible && message.transcript.collapsed)
            .count();
        (total, collapsed)
    }

    pub fn collapse_all_transcript_items(&mut self) -> usize {
        let mut changed = 0usize;
        for message in &mut self.messages {
            if message.transcript.collapsible && !message.transcript.collapsed {
                message.transcript.collapsed = true;
                changed += 1;
            }
        }
        self.ensure_focused_transcript_available();
        self.mark_dirty(DirtySections::CHAT);
        changed
    }

    pub fn expand_all_transcript_items(&mut self) -> usize {
        let mut changed = 0usize;
        for message in &mut self.messages {
            if message.transcript.collapsible && message.transcript.collapsed {
                message.transcript.collapsed = false;
                changed += 1;
            }
        }
        self.ensure_focused_transcript_available();
        self.mark_dirty(DirtySections::CHAT);
        changed
    }

    pub fn toggle_all_transcript_items(&mut self) -> usize {
        if self
            .messages
            .iter()
            .any(|message| message.transcript.collapsible && message.transcript.collapsed)
        {
            self.expand_all_transcript_items()
        } else {
            self.collapse_all_transcript_items()
        }
    }

    pub fn set_visible_collapsible_ids(&mut self, ids: Vec<u64>) {
        if self.visible_collapsible_ids == ids {
            return;
        }
        let previous_focus = self.focused_collapsible_id;
        self.visible_collapsible_ids = ids;
        self.ensure_focused_transcript_available();
        if previous_focus != self.focused_collapsible_id {
            self.mark_dirty(DirtySections::CHAT);
        }
    }

    pub fn focus_next_visible_collapsible(&mut self) -> bool {
        if self.visible_collapsible_ids.is_empty() {
            return false;
        }
        let next_index = match self.focused_collapsible_id {
            Some(current) => self
                .visible_collapsible_ids
                .iter()
                .position(|id| *id == current)
                .map(|idx| (idx + 1) % self.visible_collapsible_ids.len())
                .unwrap_or(0),
            None => 0,
        };
        self.focused_collapsible_id = Some(self.visible_collapsible_ids[next_index]);
        self.mark_dirty(DirtySections::CHAT);
        true
    }

    pub fn focus_prev_visible_collapsible(&mut self) -> bool {
        if self.visible_collapsible_ids.is_empty() {
            return false;
        }
        let prev_index = match self.focused_collapsible_id {
            Some(current) => self
                .visible_collapsible_ids
                .iter()
                .position(|id| *id == current)
                .map(|idx| {
                    if idx == 0 {
                        self.visible_collapsible_ids.len() - 1
                    } else {
                        idx - 1
                    }
                })
                .unwrap_or(self.visible_collapsible_ids.len() - 1),
            None => self.visible_collapsible_ids.len() - 1,
        };
        self.focused_collapsible_id = Some(self.visible_collapsible_ids[prev_index]);
        self.mark_dirty(DirtySections::CHAT);
        true
    }

    pub fn toggle_focused_collapsible(&mut self) -> bool {
        let target_id = self
            .focused_collapsible_id
            .or_else(|| self.visible_collapsible_ids.first().copied());
        let Some(target_id) = target_id else {
            return false;
        };

        let Some(message) = self
            .messages
            .iter_mut()
            .find(|message| message.id == target_id)
        else {
            return false;
        };
        if !message.transcript.collapsible {
            return false;
        }

        message.transcript.collapsed = !message.transcript.collapsed;
        self.focused_collapsible_id = Some(target_id);
        self.mark_dirty(DirtySections::CHAT);
        true
    }

    pub fn is_focused_collapsible(&self, id: u64) -> bool {
        self.focused_collapsible_id == Some(id)
    }

    pub fn clear_messages(&mut self) {
        self.messages.clear();
        self.scroll_offset = 0;
        self.current_trace = None;
        self.recent_traces.clear();
        self.work_started_at = None;
        self.accumulated_work_duration = Duration::ZERO;
        self.last_work_duration = None;
        self.grouped_trace_steps.clear();
        self.grouped_trace_failed = false;
        self.clear_autocomplete();
        self.focused_collapsible_id = None;
        self.visible_collapsible_ids.clear();
        self.next_message_id = 1;
        self.prompt_tokens = 0;
        self.completion_tokens = 0;
        self.total_tokens = 0;
        self.estimated_cost_usd = if self.backend_name.starts_with("llama.cpp")
            || self.backend_name.starts_with("ollama")
        {
            Some(0.0)
        } else {
            None
        };
        self.cache_hits = 0;
        self.cache_misses = 0;
        self.tokens_saved = 0;
        self.last_cache_hit = None;
        self.mark_dirty(DirtySections::ALL);
    }

    pub fn current_turn_duration(&self) -> Option<Duration> {
        let mut total = self.accumulated_work_duration;
        if let Some(started_at) = self.work_started_at {
            total = total.saturating_add(started_at.elapsed());
        }

        if total.is_zero() {
            None
        } else {
            Some(total)
        }
    }

    pub fn last_work_duration(&self) -> Option<Duration> {
        self.last_work_duration
    }

    pub fn autocomplete_hint(&self) -> Option<String> {
        if self.autocomplete_matches.is_empty() {
            return None;
        }

        let preview = self
            .autocomplete_matches
            .iter()
            .take(4)
            .cloned()
            .collect::<Vec<_>>()
            .join("  ");
        let extra = self.autocomplete_matches.len().saturating_sub(4);
        if extra > 0 {
            Some(format!("{preview}  +{extra}"))
        } else {
            Some(preview)
        }
    }

    pub fn autocomplete_preview_matches(&self, max: usize) -> Vec<String> {
        self.autocomplete_matches
            .iter()
            .take(max)
            .cloned()
            .collect()
    }

    pub fn autocomplete_preview_items(&self, max: usize) -> Vec<(String, bool)> {
        self.autocomplete_matches
            .iter()
            .take(max)
            .enumerate()
            .map(|(idx, value)| (value.clone(), idx == self.autocomplete_index))
            .collect()
    }

    pub fn autocomplete_command<S: AsRef<str>>(&mut self, commands: &[S], reverse: bool) -> bool {
        let Some((start, end, typed_prefix)) = slash_prefix_range(&self.input, self.cursor) else {
            self.clear_autocomplete();
            return false;
        };

        let prefix = if !self.autocomplete_matches.is_empty()
            && self.autocomplete_index < self.autocomplete_matches.len()
            && self.autocomplete_matches[self.autocomplete_index] == self.input[..end]
        {
            self.autocomplete_prefix.clone().unwrap_or(typed_prefix)
        } else {
            typed_prefix
        };

        let matches = commands
            .iter()
            .filter(|cmd| cmd.as_ref().starts_with(prefix.as_str()))
            .map(|cmd| cmd.as_ref().to_string())
            .collect::<Vec<_>>();

        if matches.is_empty() {
            self.clear_autocomplete();
            return false;
        }

        let same_cycle = self
            .autocomplete_prefix
            .as_ref()
            .map(|existing| existing == &prefix)
            .unwrap_or(false)
            && self.autocomplete_matches == matches;

        if same_cycle {
            if reverse {
                if self.autocomplete_index == 0 {
                    self.autocomplete_index = self.autocomplete_matches.len() - 1;
                } else {
                    self.autocomplete_index -= 1;
                }
            } else {
                self.autocomplete_index =
                    (self.autocomplete_index + 1) % self.autocomplete_matches.len();
            }
        } else {
            self.autocomplete_matches = matches;
            self.autocomplete_prefix = Some(prefix);
            self.autocomplete_index = if reverse {
                self.autocomplete_matches.len() - 1
            } else {
                0
            };
        }

        let selected = &self.autocomplete_matches[self.autocomplete_index];
        self.input.replace_range(start..end, selected);
        self.cursor = start + selected.len();

        if self.autocomplete_matches.len() == 1 && self.input[self.cursor..].is_empty() {
            self.input.push(' ');
            self.cursor += 1;
        }

        self.mark_dirty(DirtySections::INPUT);
        true
    }

    fn resume_work_timer(&mut self) {
        if self.work_started_at.is_none() {
            self.work_started_at = Some(Instant::now());
        }
    }

    fn pause_work_timer(&mut self) {
        if let Some(started_at) = self.work_started_at.take() {
            self.accumulated_work_duration = self
                .accumulated_work_duration
                .saturating_add(started_at.elapsed());
        }
    }

    fn finish_work_timer(&mut self) {
        self.pause_work_timer();
        if !self.accumulated_work_duration.is_zero() {
            self.last_work_duration = Some(self.accumulated_work_duration);
        }
        self.accumulated_work_duration = Duration::ZERO;
    }

    fn should_begin_new_trace_group(&self) -> bool {
        self.work_started_at.is_none()
            && self.accumulated_work_duration.is_zero()
            && self.grouped_trace_steps.is_empty()
    }

    fn is_grouping_turn_traces(&self) -> bool {
        self.work_started_at.is_some()
            || !self.accumulated_work_duration.is_zero()
            || self.is_generating
            || self.pending_action.is_some()
    }

    fn push_grouped_trace_step(&mut self, label: &str, success: bool) {
        if label.trim().is_empty() {
            return;
        }
        if self
            .grouped_trace_steps
            .last()
            .map(|last| last == label)
            .unwrap_or(false)
        {
            if !success {
                self.grouped_trace_failed = true;
            }
            return;
        }
        self.grouped_trace_steps.push(label.to_string());
        if !success {
            self.grouped_trace_failed = true;
        }
    }

    fn flush_grouped_trace_summary(&mut self) {
        if self.grouped_trace_steps.is_empty() {
            return;
        }

        let summary = summarize_trace_steps(&self.grouped_trace_steps);
        self.push_recent_trace(&summary, !self.grouped_trace_failed);
        self.grouped_trace_steps.clear();
        self.grouped_trace_failed = false;
        self.mark_dirty(DirtySections::SIDEBAR | DirtySections::HEADER);
    }

    fn clear_autocomplete(&mut self) {
        self.autocomplete_matches.clear();
        self.autocomplete_index = 0;
        self.autocomplete_prefix = None;
        self.mark_dirty(DirtySections::INPUT);
    }

    pub fn dirty_sections(&self) -> DirtySections {
        self.dirty_sections
    }

    pub fn has_dirty_sections(&self) -> bool {
        !self.dirty_sections.is_empty()
    }

    pub fn clear_dirty_sections(&mut self) {
        self.dirty_sections = DirtySections::NONE;
    }

    pub fn mark_dirty(&mut self, sections: DirtySections) {
        self.dirty_sections |= sections;
    }

    fn next_transcript_id(&mut self) -> u64 {
        let id = self.next_message_id;
        self.next_message_id = self.next_message_id.saturating_add(1);
        id
    }

    fn build_chat_message(&mut self, role: Role, content: impl Into<String>) -> ChatMessage {
        let content = content.into();
        let transcript = transcript_presentation_for_content(&content);
        let id = self.next_transcript_id();
        if transcript.collapsible {
            self.focused_collapsible_id = Some(id);
        }

        ChatMessage {
            id,
            role,
            content,
            transcript,
        }
    }

    fn ensure_focused_transcript_available(&mut self) {
        let visible_contains_focus = self
            .focused_collapsible_id
            .map(|id| self.visible_collapsible_ids.contains(&id))
            .unwrap_or(false);
        if visible_contains_focus {
            return;
        }

        self.focused_collapsible_id = self.visible_collapsible_ids.first().copied().or_else(|| {
            self.messages
                .iter()
                .find(|message| message.transcript.collapsible)
                .map(|message| message.id)
        });
    }

    fn current_line_start(&self) -> usize {
        self.input[..self.cursor]
            .rfind('\n')
            .map(|idx| idx + 1)
            .unwrap_or(0)
    }

    fn current_line_end(&self) -> usize {
        self.input[self.cursor..]
            .find('\n')
            .map(|offset| self.cursor + offset)
            .unwrap_or(self.input.len())
    }

    fn set_input_text(&mut self, text: String) {
        self.input = text;
        self.cursor = self.input.len();
        self.clear_autocomplete();
        self.mark_dirty(DirtySections::INPUT);
    }

    fn reverse_search_matches(&self) -> Vec<String> {
        let query = self.reverse_search_query.to_lowercase();
        self.input_history
            .iter()
            .rev()
            .filter(|entry| {
                if query.is_empty() {
                    true
                } else {
                    entry.to_lowercase().contains(&query)
                }
            })
            .cloned()
            .collect()
    }

    fn apply_reverse_search_match(&mut self) {
        let matches = self.reverse_search_matches();
        if matches.is_empty() {
            self.set_input_text(self.reverse_search_query.clone());
            return;
        }
        self.reverse_search_selection = self
            .reverse_search_selection
            .min(matches.len().saturating_sub(1));
        self.set_input_text(matches[self.reverse_search_selection].clone());
    }

    fn exit_reverse_search(&mut self) {
        self.reverse_search_active = false;
        self.reverse_search_query.clear();
        self.reverse_search_selection = 0;
        self.reverse_search_draft = None;
    }

    fn command_launcher_matches(&self) -> Vec<CommandSuggestion> {
        let query = self.command_launcher_query.to_lowercase();
        let mut matches = self
            .command_launcher_entries
            .iter()
            .filter_map(|entry| {
                command_match_score(entry, &query).map(|score| (score, entry.clone()))
            })
            .collect::<Vec<_>>();
        matches.sort_by(|(left_score, left), (right_score, right)| {
            left_score
                .cmp(right_score)
                .then_with(|| group_rank(left.group).cmp(&group_rank(right.group)))
                .then_with(|| left.name.cmp(&right.name))
        });
        matches.into_iter().map(|(_, entry)| entry).collect()
    }

    fn apply_command_launcher_match(&mut self) {
        let matches = self.command_launcher_matches();
        if matches.is_empty() {
            self.set_input_text(self.command_launcher_query.clone());
            return;
        }
        self.command_launcher_selection = self
            .command_launcher_selection
            .min(matches.len().saturating_sub(1));
        self.set_input_text(format!(
            "{} ",
            matches[self.command_launcher_selection].name
        ));
    }

    fn exit_command_launcher(&mut self) {
        self.command_launcher_active = false;
        self.command_launcher_query.clear();
        self.command_launcher_selection = 0;
        self.command_launcher_entries.clear();
        self.command_launcher_draft = None;
    }
}

fn command_match_score(entry: &CommandSuggestion, query: &str) -> Option<(u8, usize)> {
    if query.is_empty() {
        return Some((0, 0));
    }

    let name = entry.name.to_lowercase();
    if name == query {
        return Some((0, 0));
    }
    if name.starts_with(query) {
        return Some((1, name.len()));
    }
    if entry
        .aliases
        .iter()
        .map(|alias| alias.to_lowercase())
        .any(|alias| alias == query)
    {
        return Some((2, 0));
    }
    if entry
        .aliases
        .iter()
        .map(|alias| alias.to_lowercase())
        .any(|alias| alias.starts_with(query))
    {
        return Some((3, 0));
    }
    if name.contains(query) {
        return Some((4, name.len()));
    }
    if entry.usage.to_lowercase().contains(query) {
        return Some((5, entry.usage.len()));
    }
    if entry.description.to_lowercase().contains(query) {
        return Some((6, entry.description.len()));
    }
    if entry.group.to_lowercase().contains(query) {
        return Some((7, entry.group.len()));
    }

    None
}

fn group_rank(group: &str) -> u8 {
    match group {
        "context" => 0,
        "action" => 1,
        "session" => 2,
        "help" => 3,
        "custom" => 4,
        _ => 5,
    }
}

fn slash_prefix_range(input: &str, cursor: usize) -> Option<(usize, usize, String)> {
    if !input.starts_with('/') {
        return None;
    }

    let safe_cursor = cursor.min(input.len());
    let active = &input[..safe_cursor];
    let command_end = active.find(' ').unwrap_or(active.len());
    if command_end == 0 || safe_cursor > command_end {
        return None;
    }

    Some((0, command_end, input[..command_end].to_string()))
}

/// Returns a human-readable description of how long ago a session was saved.
fn describe_session_age(saved_at: u64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let age = now.saturating_sub(saved_at);
    if age < 120 {
        "moments ago".to_string()
    } else if age < 3600 {
        let m = age / 60;
        format!("{m}m ago")
    } else if age < 86400 {
        let h = age / 3600;
        format!("{h}h ago")
    } else {
        let d = age / 86400;
        format!("{d}d ago")
    }
}

fn summarize_trace_steps(steps: &[String]) -> String {
    match steps {
        [] => String::new(),
        [only] => only.clone(),
        [first, second] => format!("{first} -> {second}"),
        [first, second, third] => format!("{first} -> {second} -> {third}"),
        _ => format!(
            "{} -> {} -> {} (+{})",
            steps[0],
            steps[1],
            steps[2],
            steps.len() - 3
        ),
    }
}

fn wrap_input_for_display(input: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut lines = Vec::new();

    if input.is_empty() {
        return vec![String::new()];
    }

    for raw_line in input.split('\n') {
        let wrapped = wrap_preserving_empty_line(raw_line, width);
        lines.extend(wrapped);
    }

    if input.ends_with('\n') {
        lines.push(String::new());
    }

    if lines.is_empty() {
        vec![String::new()]
    } else {
        lines
    }
}

fn wrap_preserving_empty_line(line: &str, width: usize) -> Vec<String> {
    if line.is_empty() {
        return vec![String::new()];
    }

    let chars: Vec<char> = line.chars().collect();
    let mut wrapped = Vec::new();
    let mut start = 0usize;
    while start < chars.len() {
        let end = (start + width).min(chars.len());
        wrapped.push(chars[start..end].iter().collect());
        start = end;
    }
    wrapped
}

fn cursor_visual_position(input: &str, cursor: usize, width: usize) -> (usize, usize) {
    let width = width.max(1);
    let safe_cursor = cursor.min(input.len());
    let before = &input[..safe_cursor];
    let mut row = 0usize;
    let mut col = 0usize;

    for ch in before.chars() {
        if ch == '\n' {
            row += 1;
            col = 0;
            continue;
        }
        col += 1;
        if col >= width {
            row += 1;
            col = 0;
        }
    }

    (row, col)
}

impl TranscriptPresentation {
    fn plain() -> Self {
        Self {
            collapsible: false,
            collapsed: false,
            summary: None,
            preview_lines: Vec::new(),
        }
    }
}

fn transcript_presentation_for_content(content: &str) -> TranscriptPresentation {
    classify_collapsible_context(content).unwrap_or_else(TranscriptPresentation::plain)
}

fn classify_collapsible_context(content: &str) -> Option<TranscriptPresentation> {
    if content.starts_with("Tool results:\n") {
        let tool_count = content
            .lines()
            .filter(|line| line.starts_with("--- "))
            .count()
            .max(1);
        return Some(TranscriptPresentation {
            collapsible: true,
            collapsed: true,
            summary: Some(format!(
                "tool results • {tool_count} tool{}",
                if tool_count == 1 { "" } else { "s" }
            )),
            preview_lines: extract_preview_lines(content, &["Tool results:", ""], 2),
        });
    }

    if content.starts_with("I've loaded this file for context:") {
        let file_label = extract_value_after_label(content, "File:")
            .unwrap_or_else(|| "file context".to_string());
        return Some(TranscriptPresentation {
            collapsible: true,
            collapsed: true,
            summary: Some(format!("file context • {file_label}")),
            preview_lines: extract_preview_lines(
                content,
                &["I've loaded this file for context:", ""],
                2,
            ),
        });
    }

    if content.starts_with("Directory listing:") {
        let dir = extract_value_after_label(content, "Directory:")
            .unwrap_or_else(|| "directory".to_string());
        return Some(TranscriptPresentation {
            collapsible: true,
            collapsed: true,
            summary: Some(format!("directory listing • {dir}")),
            preview_lines: extract_preview_lines(
                content,
                &["Directory listing:", "", &format!("Directory: {dir}")],
                2,
            ),
        });
    }

    if content.starts_with("Search results:\n") {
        let query = content
            .lines()
            .find_map(|line| {
                line.strip_prefix("Search results for '")
                    .and_then(|rest| rest.split_once('\'').map(|(query, _)| query.to_string()))
            })
            .unwrap_or_else(|| "query output".to_string());
        return Some(TranscriptPresentation {
            collapsible: true,
            collapsed: true,
            summary: Some(format!("search results • {query}")),
            preview_lines: extract_preview_lines(content, &["Search results:", ""], 2),
        });
    }

    if content.starts_with("Git context (") {
        let mode = content
            .strip_prefix("Git context (")
            .and_then(|rest| rest.split_once("):").map(|(mode, _)| mode.to_string()))
            .unwrap_or_else(|| "status".to_string());
        return Some(TranscriptPresentation {
            collapsible: true,
            collapsed: true,
            summary: Some(format!("git context • {mode}")),
            preview_lines: extract_preview_lines(content, &[&format!("Git context ({mode}):")], 2),
        });
    }

    if content.starts_with("LSP diagnostics:\n") {
        return Some(TranscriptPresentation {
            collapsible: true,
            collapsed: true,
            summary: Some("diagnostics".to_string()),
            preview_lines: extract_preview_lines(content, &["LSP diagnostics:", ""], 2),
        });
    }

    if content.starts_with("LSP check:\n") {
        return Some(TranscriptPresentation {
            collapsible: true,
            collapsed: true,
            summary: Some("rust lsp check".to_string()),
            preview_lines: extract_preview_lines(content, &["LSP check:", ""], 2),
        });
    }

    if content.starts_with("LSP hover:") {
        return Some(TranscriptPresentation {
            collapsible: true,
            collapsed: true,
            summary: Some("hover".to_string()),
            preview_lines: extract_preview_lines(content, &["LSP hover:", ""], 2),
        });
    }

    if content.starts_with("LSP definition:") {
        return Some(TranscriptPresentation {
            collapsible: true,
            collapsed: true,
            summary: Some("definition".to_string()),
            preview_lines: extract_preview_lines(content, &["LSP definition:", ""], 2),
        });
    }

    if content.starts_with("Fetched web context:\n") {
        let url =
            extract_value_after_label(content, "Fetched URL:").unwrap_or_else(|| "web".to_string());
        let host = url
            .split("://")
            .nth(1)
            .and_then(|rest| rest.split('/').next())
            .unwrap_or(url.as_str())
            .to_string();
        return Some(TranscriptPresentation {
            collapsible: true,
            collapsed: true,
            summary: Some(format!("web context • {host}")),
            preview_lines: extract_preview_lines(content, &["Fetched web context:", ""], 2),
        });
    }

    None
}

fn extract_preview_lines(content: &str, skip_lines: &[&str], max_lines: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || skip_lines.iter().any(|skip| trimmed == *skip) {
            continue;
        }
        lines.push(trimmed.to_string());
        if lines.len() >= max_lines {
            break;
        }
    }
    if lines.is_empty() {
        vec!["(no preview)".to_string()]
    } else {
        lines
    }
}

fn extract_value_after_label(content: &str, label: &str) -> Option<String> {
    content
        .lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix(label)
                .map(|value| value.trim().to_string())
        })
        .filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn work_timer_accumulates_and_finishes() {
        let mut state = AppState::new();
        state.start_generation("generating...", false);
        std::thread::sleep(Duration::from_millis(5));
        state.set_pending_action(PendingAction {
            id: 1,
            kind: crate::events::PendingActionKind::ShellCommand,
            title: "Approve".to_string(),
            preview: "echo hi".to_string(),
            inspection: crate::safety::InspectionReport {
                operation: "bash".to_string(),
                decision: crate::safety::InspectionDecision::NeedsApproval,
                risk: crate::safety::RiskLevel::Low,
                summary: "test".to_string(),
                reasons: Vec::new(),
                targets: Vec::new(),
                segments: vec!["echo hi".to_string()],
                network_targets: Vec::new(),
            },
        });
        let paused = state.current_turn_duration().unwrap();
        assert!(paused >= Duration::from_millis(1));

        state.start_generation("generating...", false);
        std::thread::sleep(Duration::from_millis(5));
        state.finish_response();

        assert!(state.current_turn_duration().is_none());
        assert!(state.last_work_duration().unwrap() >= paused);
    }

    #[test]
    fn grouped_traces_collapse_at_turn_end() {
        let mut state = AppState::new();
        state.start_generation("generating...", false);
        state.apply_trace(ProgressTrace {
            status: ProgressStatus::Finished,
            label: "drafting answer...".to_string(),
            persist: false,
        });
        state.apply_trace(ProgressTrace {
            status: ProgressStatus::Finished,
            label: "answer ready".to_string(),
            persist: false,
        });

        assert!(state.recent_traces.is_empty());
        state.finish_response();

        assert_eq!(state.recent_traces.len(), 1);
        assert_eq!(
            state.recent_traces.front().unwrap().label,
            "drafting answer... -> answer ready"
        );
        assert!(state.recent_traces.front().unwrap().success);
    }

    #[test]
    fn standalone_traces_still_show_individually() {
        let mut state = AppState::new();
        state.apply_trace(ProgressTrace {
            status: ProgressStatus::Finished,
            label: "profile: .params.toml".to_string(),
            persist: true,
        });

        assert_eq!(state.recent_traces.len(), 1);
        assert_eq!(
            state.recent_traces.front().unwrap().label,
            "profile: .params.toml"
        );
    }

    #[test]
    fn non_persisted_traces_do_not_inject_chat_messages() {
        let mut state = AppState::new();
        state.apply_trace(ProgressTrace {
            status: ProgressStatus::Finished,
            label: "memory: stored 1 fact".to_string(),
            persist: false,
        });

        assert!(state.messages.is_empty());
        assert_eq!(state.recent_traces.len(), 1);
        assert_eq!(
            state.recent_traces.front().unwrap().label,
            "memory: stored 1 fact"
        );
    }

    #[test]
    fn command_autocomplete_cycles_matches() {
        let mut state = AppState::new();
        state.input = "/d".to_string();
        state.cursor = 2;

        assert!(state.autocomplete_command(&["/def", "/diag", "/debug-log"], false));
        assert_eq!(state.input, "/def");

        assert!(state.autocomplete_command(&["/def", "/diag", "/debug-log"], false));
        assert_eq!(state.input, "/diag");
    }

    #[test]
    fn command_autocomplete_adds_space_for_unique_match() {
        let mut state = AppState::new();
        state.input = "/reject".to_string();
        state.cursor = state.input.len();

        assert!(state.autocomplete_command(&["/reject"], false));
        assert_eq!(state.input, "/reject ");
    }

    #[test]
    fn pending_action_sets_waiting_status_without_chat_message() {
        let mut state = AppState::new();
        assert_eq!(state.messages.len(), 0);

        state.set_pending_action(PendingAction {
            id: 1,
            kind: crate::events::PendingActionKind::ShellCommand,
            title: "Approve shell command".to_string(),
            preview: "cargo check".to_string(),
            inspection: crate::safety::InspectionReport {
                operation: "bash".to_string(),
                decision: crate::safety::InspectionDecision::NeedsApproval,
                risk: crate::safety::RiskLevel::Low,
                summary: "Shell command requires approval before execution".to_string(),
                reasons: Vec::new(),
                targets: Vec::new(),
                segments: vec!["cargo check".to_string()],
                network_targets: Vec::new(),
            },
        });

        assert!(state.has_pending_action());
        assert_eq!(state.status, "awaiting approval");
        assert!(state.messages.is_empty());
    }

    #[test]
    fn clearing_pending_action_removes_card_state() {
        let mut state = AppState::new();
        state.set_pending_action(PendingAction {
            id: 1,
            kind: crate::events::PendingActionKind::ShellCommand,
            title: "Approve shell command".to_string(),
            preview: "cargo check".to_string(),
            inspection: crate::safety::InspectionReport {
                operation: "bash".to_string(),
                decision: crate::safety::InspectionDecision::NeedsApproval,
                risk: crate::safety::RiskLevel::Low,
                summary: "Shell command requires approval before execution".to_string(),
                reasons: Vec::new(),
                targets: Vec::new(),
                segments: vec!["cargo check".to_string()],
                network_targets: Vec::new(),
            },
        });

        state.clear_pending_action();

        assert!(!state.has_pending_action());
        assert_eq!(state.status, "ready");
    }

    #[test]
    fn insert_newline_preserves_multiline_input() {
        let mut state = AppState::new();
        state.input = "hello".to_string();
        state.cursor = state.input.len();

        state.insert_newline();
        state.insert_str("world");

        assert_eq!(state.input, "hello\nworld");
    }

    #[test]
    fn submit_input_returns_multiline_content_unchanged() {
        let mut state = AppState::new();
        state.input = "one\ntwo\nthree".to_string();
        state.cursor = state.input.len();

        let submitted = state.submit_input();

        assert_eq!(submitted, "one\ntwo\nthree");
        assert!(state.input.is_empty());
    }

    #[test]
    fn normalized_paste_preserves_newlines() {
        assert_eq!(
            AppState::normalized_paste("one\r\ntwo\rthree"),
            "one\ntwo\nthree"
        );
    }

    #[test]
    fn home_and_end_operate_on_current_line() {
        let mut state = AppState::new();
        state.input = "first\nsecond\nthird".to_string();
        state.cursor = "first\nsec".len();

        state.cursor_end();
        assert_eq!(state.cursor, "first\nsecond".len());

        state.cursor_home();
        assert_eq!(state.cursor, "first\n".len());
    }

    #[test]
    fn input_display_lines_wrap_and_keep_cursor_visible() {
        let mut state = AppState::new();
        state.input = "123456789\nabc".to_string();
        state.cursor = state.input.len();

        let (lines, cursor_row, cursor_col) = state.input_display_lines(4, 3);

        assert!(!lines.is_empty());
        assert!(lines.iter().any(|line| line == "abc"));
        assert_eq!(cursor_row, 2);
        assert_eq!(cursor_col, 3);
    }

    #[test]
    fn history_recall_restores_previous_submission_and_draft() {
        let mut state = AppState::new();
        state.input = "first".to_string();
        state.cursor = state.input.len();
        assert_eq!(state.submit_input(), "first");

        state.input = "second".to_string();
        state.cursor = state.input.len();
        assert_eq!(state.submit_input(), "second");

        state.input = "draft".to_string();
        state.cursor = state.input.len();

        assert!(state.recall_previous_input());
        assert_eq!(state.input, "second");
        assert!(state.recall_previous_input());
        assert_eq!(state.input, "first");
        assert!(state.recall_next_input());
        assert_eq!(state.input, "second");
        assert!(state.recall_next_input());
        assert_eq!(state.input, "draft");
    }

    #[test]
    fn scroll_and_input_mutations_mark_dirty_sections() {
        let mut state = AppState::new();
        state.clear_dirty_sections();

        state.insert_char('a');
        assert!(state.dirty_sections().contains(DirtySections::INPUT));

        state.clear_dirty_sections();
        state.add_user_message("hello");
        assert!(state.dirty_sections().contains(DirtySections::CHAT));
        assert!(state.dirty_sections().contains(DirtySections::SIDEBAR));

        state.clear_dirty_sections();
        state.scroll_up(1);
        assert!(state.dirty_sections().contains(DirtySections::CHAT));
    }

    #[test]
    fn injected_context_messages_default_to_collapsed() {
        let mut state = AppState::new();
        state.add_user_message(
            "Tool results:\n\n--- read_file(src/main.rs) ---\nFile: src/main.rs\n\n```",
        );

        let message = state.messages.last().expect("message");
        assert!(message.transcript.collapsible);
        assert!(message.transcript.collapsed);
        assert_eq!(
            message.transcript.summary.as_deref(),
            Some("tool results • 1 tool")
        );
    }

    #[test]
    fn normal_messages_stay_non_collapsible() {
        let mut state = AppState::new();
        state.add_user_message("hello there");

        let message = state.messages.last().expect("message");
        assert!(!message.transcript.collapsible);
        assert!(!message.transcript.collapsed);
    }

    #[test]
    fn restore_session_collapses_injected_context_rows() {
        let mut state = AppState::new();
        state.restore_session(
            SessionInfo {
                id: "session".to_string(),
                name: None,
                message_count: 1,
            },
            vec![(
                "user".to_string(),
                "Search results:\n\nSearch results for 'cache' (1 matches):".to_string(),
            )],
            None,
        );

        let message = state.messages.last().expect("restored message");
        assert_eq!(message.role, Role::System);
        assert!(message.transcript.collapsible);
        assert!(message.transcript.collapsed);
    }

    #[test]
    fn transcript_focus_and_toggle_use_visible_collapsible_items() {
        let mut state = AppState::new();
        state.add_user_message("hello");
        state.add_user_message("Tool results:\n\n--- read_file(src/main.rs) ---");
        state.add_user_message("Directory listing:\n\nDirectory: src\n\nmain.rs");

        let first_id = state.messages[1].id;
        let second_id = state.messages[2].id;
        state.set_visible_collapsible_ids(vec![first_id, second_id]);

        assert!(state.is_focused_collapsible(second_id));
        assert!(state.focus_next_visible_collapsible());
        assert!(state.is_focused_collapsible(first_id));
        assert!(state.toggle_focused_collapsible());
        assert!(!state.messages[1].transcript.collapsed);
        assert!(state.focus_prev_visible_collapsible());
        assert!(state.is_focused_collapsible(second_id));
    }

    #[test]
    fn transcript_global_collapse_expand_preserves_content() {
        let mut state = AppState::new();
        state.add_user_message("Tool results:\n\n--- read_file(src/main.rs) ---");
        let original = state.messages[0].content.clone();

        assert_eq!(state.expand_all_transcript_items(), 1);
        assert!(!state.messages[0].transcript.collapsed);
        assert_eq!(state.collapse_all_transcript_items(), 1);
        assert!(state.messages[0].transcript.collapsed);
        assert_eq!(state.messages[0].content, original);
    }

    #[test]
    fn reverse_search_recalls_previous_submission_without_submitting() {
        let mut state = AppState::new();
        state.input = "first prompt".to_string();
        state.cursor = state.input.len();
        assert_eq!(state.submit_input(), "first prompt");

        state.input = "second prompt".to_string();
        state.cursor = state.input.len();
        assert_eq!(state.submit_input(), "second prompt");

        state.input = "draft".to_string();
        state.cursor = state.input.len();

        assert!(state.activate_reverse_search());
        state.reverse_search_push_char('f');
        state.reverse_search_push_char('i');

        assert!(state.is_reverse_search_active());
        assert_eq!(state.input, "first prompt");
        assert!(state.accept_reverse_search());
        assert_eq!(state.input, "first prompt");
        assert!(!state.is_reverse_search_active());
    }

    #[test]
    fn reverse_search_cancel_restores_original_draft() {
        let mut state = AppState::new();
        state.input = "alpha".to_string();
        state.cursor = state.input.len();
        assert_eq!(state.submit_input(), "alpha");

        state.input = "draft text".to_string();
        state.cursor = state.input.len();

        assert!(state.activate_reverse_search());
        state.reverse_search_push_char('a');
        assert_eq!(state.input, "alpha");

        assert!(state.cancel_reverse_search());
        assert_eq!(state.input, "draft text");
        assert!(!state.is_reverse_search_active());
    }

    #[test]
    fn reverse_search_cycle_walks_matching_history() {
        let mut state = AppState::new();
        for value in ["fix lint", "find bug", "finish docs"] {
            state.input = value.to_string();
            state.cursor = state.input.len();
            assert_eq!(state.submit_input(), value);
        }

        assert!(state.activate_reverse_search());
        state.reverse_search_push_char('f');
        state.reverse_search_push_char('i');
        assert_eq!(state.input, "finish docs");

        assert!(state.reverse_search_cycle());
        assert_eq!(state.input, "find bug");
    }

    #[test]
    fn command_launcher_selects_command_without_submitting() {
        let mut state = AppState::new();
        state.input = "draft".to_string();
        state.cursor = state.input.len();

        assert!(state.activate_command_launcher(vec![
            CommandSuggestion {
                name: "/read".to_string(),
                usage: "/read <path>".to_string(),
                description: "load a file".to_string(),
                source: "builtin",
                group: "context",
                aliases: vec!["/r".to_string()],
            },
            CommandSuggestion {
                name: "/search".to_string(),
                usage: "/search <query>".to_string(),
                description: "search project files".to_string(),
                source: "builtin",
                group: "context",
                aliases: vec!["/s".to_string()],
            },
        ]));

        state.command_launcher_push_char('s');
        let accepted = state.accept_command_launcher().expect("command");
        assert_eq!(accepted, "/search ");
        assert_eq!(state.input, "/search ");
        assert!(!state.is_command_launcher_active());
    }

    #[test]
    fn command_launcher_cancel_restores_draft() {
        let mut state = AppState::new();
        state.input = "draft".to_string();
        state.cursor = state.input.len();

        assert!(state.activate_command_launcher(vec![CommandSuggestion {
            name: "/read".to_string(),
            usage: "/read <path>".to_string(),
            description: "load a file".to_string(),
            source: "builtin",
            group: "context",
            aliases: vec!["/r".to_string()],
        }]));

        state.command_launcher_push_char('r');
        assert!(state.cancel_command_launcher());
        assert_eq!(state.input, "draft");
    }

    #[test]
    fn command_launcher_matches_aliases_and_usage() {
        let mut state = AppState::new();
        assert!(state.activate_command_launcher(vec![
            CommandSuggestion {
                name: "/read".to_string(),
                usage: "/read <path>".to_string(),
                description: "load a file".to_string(),
                source: "builtin",
                group: "context",
                aliases: vec!["/r".to_string()],
            },
            CommandSuggestion {
                name: "/sessions".to_string(),
                usage: "/sessions <list|new|rename|resume|export>".to_string(),
                description: "manage saved sessions".to_string(),
                source: "builtin",
                group: "session",
                aliases: Vec::new(),
            },
        ]));

        state.command_launcher_push_char('r');
        let view = state.command_launcher_view(5).expect("view");
        assert_eq!(view.1[0].0.name, "/read");

        state.command_launcher_backspace();
        for ch in "resume".chars() {
            state.command_launcher_push_char(ch);
        }
        let view = state.command_launcher_view(5).expect("view");
        assert_eq!(view.1[0].0.name, "/sessions");
    }
}

/// Returns true if a user-role message is an injected context payload (tool results,
/// slash-command output, etc.) rather than a genuine user turn.
fn is_injected_context(content: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "Tool results:\n",
        "I've loaded this file for context:",
        "Directory listing:",
        "Search results:\n",
        "Git context (",
        "LSP diagnostics:\n",
        "LSP check:\n",
        "LSP hover:",
        "LSP definition:",
        "Fetched web context:\n",
        "User rejected proposed action:",
    ];
    PREFIXES.iter().any(|p| content.starts_with(p))
}
