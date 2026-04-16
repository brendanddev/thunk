use crate::app::config::Config;
use crate::app::paths::AppPaths;

/// Defines the application state, including the current input, cursor position, message history, and status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

/// Represents a chat message with a role (system, user, assistant) and content
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

/// Main application state struct, holding the app name, input buffer, cursor position, message history, status, and quit flag
pub struct AppState {
    pub app_name: String,
    pub show_activity: bool,
    pub input: String,
    pub cursor: usize,
    pub messages: Vec<ChatMessage>,
    pub status: String,
    pub should_quit: bool,
}

/// Defines methods for modifying the input buffer and cursor position in the app state
impl AppState {

    /// Creates a new AppState instance, initializing the message history with a system message based on the provided config and paths
    pub fn new(config: &Config, paths: &AppPaths) -> Self {
        let mut messages = Vec::new();
        messages.push(ChatMessage {
            role: Role::System,
            content: format!(
                "{} ready. Root: {}. Press Ctrl+Q to quit.",
                config.app.name,
                paths.root_dir.display()
            ),
        });

        Self {
            app_name: config.app.name.clone(),
            show_activity: config.ui.show_activity,
            input: String::new(),
            cursor: 0,
            messages,
            status: "ready".to_string(),
            should_quit: false,
        }
    }

    /// Adds a system message to the transcript
    pub fn add_system_message(&mut self, content: impl Into<String>) {
        self.messages.push(ChatMessage {
            role: Role::System,
            content: content.into(),
        });
    }

    /// Adds a user message to the transcript
    pub fn add_user_message(&mut self, content: impl Into<String>) {
        self.messages.push(ChatMessage {
            role: Role::User,
            content: content.into(),
        });
    }

    /// Adds a complete assistant message to the transcript
    pub fn add_assistant_message(&mut self, content: impl Into<String>) {
        self.messages.push(ChatMessage {
            role: Role::Assistant,
            content: content.into(),
        });
    }

    /// Starts a new assistant message so chunks can be streamed into it
    pub fn begin_assistant_message(&mut self) {
        self.add_assistant_message(String::new());
    }

    /// Appends text to the active assistant message, creating one if needed
    pub fn append_assistant_chunk(&mut self, chunk: &str) {
        match self.messages.last_mut() {
            Some(ChatMessage {
                role: Role::Assistant,
                content,
            }) => content.push_str(chunk),
            _ => self.add_assistant_message(chunk.to_string()),
        }
    }

    /// Updates the visible status line
    pub fn set_status(&mut self, status: &str) {
        self.status = status.to_string();
    }

    /// Submits the current input, returning it as a string if it's not empty, and clears the input buffer and resets the cursor position
    pub fn submit_input(&mut self) -> Option<String> {
        if self.input.trim().is_empty() {
            self.clear_input();
            return None;
        }

        let submitted = std::mem::take(&mut self.input);
        self.cursor = 0;
        Some(submitted)
    }
}
