use crate::llm::backend::{Message, Role};

#[derive(Debug, Clone)]
pub struct Conversation {
    messages: Vec<Message>,
}

impl Conversation {
    pub fn new(system_prompt: String) -> Self {
        Self {
            messages: vec![Message::system(system_prompt)],
        }
    }

    pub fn push_user(&mut self, content: impl Into<String>) {
        self.messages.push(Message::user(content));
    }

    pub fn begin_assistant_reply(&mut self) {
        self.messages.push(Message::assistant(String::new()));
    }

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

    pub fn snapshot(&self) -> Vec<Message> {
        self.messages.clone()
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
