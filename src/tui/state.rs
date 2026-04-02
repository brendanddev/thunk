// src/tui/state.rs

use crate::events::PendingAction;

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
        std::mem::take(&mut self.input)
    }

    /// Insert a character at the cursor position
    pub fn insert_char(&mut self, c: char) {
        self.input.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    /// Insert a string at cursor (for paste)
    pub fn insert_str(&mut self, s: &str) {
        self.input.insert_str(self.cursor, s);
        self.cursor += s.len();
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
        self.cursor = 0;
    }

    /// Move cursor to end of line
    pub fn cursor_end(&mut self) {
        self.cursor = self.input.len();
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

    pub fn append_token(&mut self, token: &str) {
        if let Some(last) = self.messages.last_mut() {
            if last.role == Role::Assistant {
                last.content.push_str(token);
            }
        }
        self.scroll_offset = 0;
    }

    pub fn finish_response(&mut self) {
        self.is_generating = false;
        self.status = "ready".to_string();
        self.last_tool_call = None;
    }

    pub fn add_error(&mut self, error: &str) {
        self.is_generating = false;
        self.status = "ready".to_string();
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
            self.estimated_cost_usd = if name.starts_with("llama.cpp") || name.starts_with("ollama") {
                Some(0.0)
            } else {
                None
            };
        }
    }

    pub fn set_pending_action(&mut self, action: PendingAction) {
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

    pub fn update_cache(&mut self, last_hit: bool, hits: usize, misses: usize, tokens_saved: usize) {
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

    pub fn clear_messages(&mut self) {
        self.messages.clear();
        self.scroll_offset = 0;
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
}
