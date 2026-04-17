use crate::llm::backend::{Message, Role};

/// Maintains the ordered conversation history sent to the model.
///
/// The first message is always the system prompt. User and assistant
/// messages are appended as the session progresses.
#[derive(Debug, Clone)]
pub struct Conversation {
    messages: Vec<Message>,
}

impl Conversation {
    /// Starts a new conversation with a system prompt as the first message.
    pub fn new(system_prompt: String) -> Self {
        Self {
            messages: vec![Message::system(system_prompt)],
        }
    }

    /// Appends a user message to the conversation.
    pub fn push_user(&mut self, content: impl Into<String>) {
        self.messages.push(Message::user(content));
    }

    /// Starts a new assistant message so streamed text can be appended to it.
    pub fn begin_assistant_reply(&mut self) {
        self.messages.push(Message::assistant(String::new()));
    }

    /// Appends streamed assistant text to the current assistant message.
    ///
    /// If no assistant message is currently open, one is created first.
    /// This keeps streaming callers simple and ensures chunks always have
    /// a message to attach to.
    pub fn push_assistant_chunk(&mut self, chunk: &str) {
        match self.messages.last_mut() {
            Some(Message {
                role: Role::Assistant,
                content,
            }) => content.push_str(chunk),
            _ => {
                self.begin_assistant_reply();
                self.push_assistant_chunk(chunk);
            }
        }
    }

    /// Returns a clone of the full conversation history for backend requests or persistence.
    pub fn snapshot(&self) -> Vec<Message> {
        self.messages.clone()
    }

    /// Returns the content of the most recently added assistant message, if any.
    pub fn last_assistant_content(&self) -> Option<&str> {
        self.messages
            .iter()
            .rev()
            .find(|m| m.role == Role::Assistant)
            .map(|m| m.content.as_str())
    }

    /// Appends historical messages into the conversation after the current content.
    /// Used only at startup to restore a prior session. The system prompt must
    /// already be set; history messages are appended after it.
    pub fn extend_history(&mut self, messages: Vec<Message>) {
        debug_assert!(
            self.messages.len() == 1,
            "extend_history called on non-empty conversation"
        );
        self.messages.extend(messages);
    }

    /// Resets the conversation to just the system prompt.
    pub fn reset(&mut self, system_prompt: String) {
        self.messages.clear();
        self.messages.push(Message::system(system_prompt));
    }
}

#[cfg(test)]
mod tests {
    use super::Conversation;

    #[test]
    fn appends_chunks_to_the_current_assistant_message() {
        let mut conversation = Conversation::new("system".to_string());
        conversation.push_user("hello");
        conversation.begin_assistant_reply();
        conversation.push_assistant_chunk("hi");
        conversation.push_assistant_chunk(" there");

        let messages = conversation.snapshot();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[2].content, "hi there");
    }
}
