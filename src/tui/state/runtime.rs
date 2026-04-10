use std::time::{Duration, Instant};

use crate::commands::CommandSuggestion;
use crate::events::{MemorySnapshot, PendingAction, ProgressStatus, ProgressTrace, SessionInfo};

use super::helpers::{
    command_match_score, describe_session_age, group_rank, is_injected_context,
    summarize_trace_steps, transcript_presentation_for_content,
};
use super::{AppState, ChatMessage, DirtySections, Role, TraceEntry};

impl AppState {
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

    pub fn set_show_top_bar_tokens(&mut self, show: bool) {
        self.show_top_bar_tokens = show;
        self.mark_dirty(DirtySections::HEADER);
    }

    pub fn set_show_top_bar_time(&mut self, show: bool) {
        self.show_top_bar_time = show;
        self.mark_dirty(DirtySections::HEADER);
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

    pub(super) fn clear_autocomplete(&mut self) {
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

    pub(super) fn current_line_start(&self) -> usize {
        self.input[..self.cursor]
            .rfind('\n')
            .map(|idx| idx + 1)
            .unwrap_or(0)
    }

    pub(super) fn current_line_end(&self) -> usize {
        self.input[self.cursor..]
            .find('\n')
            .map(|offset| self.cursor + offset)
            .unwrap_or(self.input.len())
    }

    pub(super) fn set_input_text(&mut self, text: String) {
        self.input = text;
        self.cursor = self.input.len();
        self.clear_autocomplete();
        self.mark_dirty(DirtySections::INPUT);
    }

    pub(super) fn reverse_search_matches(&self) -> Vec<String> {
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

    pub(super) fn apply_reverse_search_match(&mut self) {
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

    pub(super) fn exit_reverse_search(&mut self) {
        self.reverse_search_active = false;
        self.reverse_search_query.clear();
        self.reverse_search_selection = 0;
        self.reverse_search_draft = None;
    }

    pub(super) fn command_launcher_matches(&self) -> Vec<CommandSuggestion> {
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

    pub(super) fn apply_command_launcher_match(&mut self) {
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

    pub(super) fn exit_command_launcher(&mut self) {
        self.command_launcher_active = false;
        self.command_launcher_query.clear();
        self.command_launcher_selection = 0;
        self.command_launcher_entries.clear();
        self.command_launcher_draft = None;
    }
}
