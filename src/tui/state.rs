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

    pub fn set_pending_action(&mut self, action: PendingAction) {
        self.pending_action = Some(action);
        self.is_generating = false;
        self.status = "awaiting approval".to_string();
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
    }
}
