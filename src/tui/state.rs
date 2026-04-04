// src/tui/state.rs

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::events::{MemorySnapshot, PendingAction, ProgressStatus, ProgressTrace, SessionInfo};

#[derive(Debug, Clone, PartialEq)]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct TraceEntry {
    pub label: String,
    pub success: bool,
}

pub struct AppState {
    /// Current input text
    pub input: String,

    /// Cursor position within input (byte index)
    pub cursor: usize,

    /// All chat messages
    pub messages: Vec<ChatMessage>,

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
}

impl AppState {
    pub fn new() -> Self {
        Self {
            input: String::new(),
            cursor: 0,
            messages: Vec::new(),
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
        }
    }

    /// Increment the tick counter each frame for animations
    pub fn tick(&mut self) {
        self.tick = self.tick.wrapping_add(1);
    }

    /// Submit input and clear it, returning the text
    pub fn submit_input(&mut self) -> String {
        self.cursor = 0;
        self.scroll_offset = 0;
        self.clear_autocomplete();
        std::mem::take(&mut self.input)
    }

    /// Insert a character at the cursor position
    pub fn insert_char(&mut self, c: char) {
        self.input.insert(self.cursor, c);
        self.cursor += c.len_utf8();
        self.clear_autocomplete();
    }

    /// Insert a string at cursor (for paste)
    pub fn insert_str(&mut self, s: &str) {
        self.input.insert_str(self.cursor, s);
        self.cursor += s.len();
        self.clear_autocomplete();
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
        self.clear_autocomplete();
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
        self.clear_autocomplete();
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
    }

    /// Move cursor to start of line
    pub fn cursor_home(&mut self) {
        self.cursor = self.current_line_start();
    }

    /// Move cursor to end of line
    pub fn cursor_end(&mut self) {
        self.cursor = self.current_line_end();
    }

    pub fn add_user_message(&mut self, content: &str) {
        self.messages.push(ChatMessage {
            role: Role::User,
            content: content.to_string(),
        });
        self.scroll_offset = 0;
    }

    pub fn start_assistant_message(&mut self) {
        self.messages.push(ChatMessage {
            role: Role::Assistant,
            content: String::new(),
        });
        self.scroll_offset = 0;
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
    }

    pub fn append_token(&mut self, token: &str) {
        self.ensure_open_assistant_message();
        if let Some(last) = self.messages.last_mut() {
            if last.role == Role::Assistant {
                last.content.push_str(token);
            }
        }
        self.scroll_offset = 0;
    }

    pub fn finish_response(&mut self) {
        self.finish_work_timer();
        self.is_generating = false;
        self.status = "ready".to_string();
        self.last_tool_call = None;
        self.current_trace = None;
        self.flush_grouped_trace_summary();
    }

    pub fn add_error(&mut self, error: &str) {
        self.finish_work_timer();
        self.is_generating = false;
        self.status = "ready".to_string();
        self.current_trace = None;
        self.flush_grouped_trace_summary();
        self.messages.push(ChatMessage {
            role: Role::Assistant,
            content: format!("error: {error}"),
        });
        self.scroll_offset = 0;
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
    }

    pub fn scroll_down(&mut self, lines: usize) {
        self.scroll_offset = self.scroll_offset.saturating_sub(lines);
    }

    pub fn set_status(&mut self, status: &str) {
        self.status = status.to_string();
        if status == "ready" {
            self.model_ready = true;
        }
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
    }

    pub fn set_pending_action(&mut self, action: PendingAction) {
        self.pause_work_timer();
        self.pending_action = Some(action);
        self.is_generating = false;
        self.status = "awaiting approval".to_string();
    }

    pub fn mark_pending_action_submitted(&mut self, decision: &str) {
        if self.pending_action.is_some() {
            self.status = format!("{decision}...");
        }
    }

    pub fn clear_pending_action(&mut self) {
        self.pending_action = None;
        if !self.is_generating {
            self.status = "ready".to_string();
        }
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
    }

    pub fn set_reflection_enabled(&mut self, enabled: bool) {
        self.reflection_enabled = enabled;
    }

    pub fn set_eco_enabled(&mut self, enabled: bool) {
        self.eco_enabled = enabled;
    }

    pub fn set_debug_logging_enabled(&mut self, enabled: bool) {
        self.debug_logging_enabled = enabled;
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
    }

    pub fn is_ready(&self) -> bool {
        self.model_ready
    }

    pub fn add_system_message(&mut self, content: &str) {
        self.messages.push(ChatMessage {
            role: Role::System,
            content: content.to_string(),
        });
        self.scroll_offset = 0;
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
                    self.messages.push(ChatMessage {
                        role: Role::Assistant,
                        content,
                    });
                }
                "user" => {
                    let display_role = if is_injected_context(&content) {
                        Role::System
                    } else {
                        Role::User
                    };
                    self.messages.push(ChatMessage {
                        role: display_role,
                        content,
                    });
                }
                _ => {}
            }
        }
        self.scroll_offset = 0;
    }

    pub fn set_session_info(&mut self, session: SessionInfo) {
        self.current_session = Some(session);
    }

    pub fn set_memory_snapshot(&mut self, snapshot: MemorySnapshot) {
        self.memory_snapshot = snapshot;
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
    }

    fn clear_autocomplete(&mut self) {
        self.autocomplete_matches.clear();
        self.autocomplete_index = 0;
        self.autocomplete_prefix = None;
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
