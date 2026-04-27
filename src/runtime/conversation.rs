use crate::llm::backend::{Message, Role};

/// Trigger live trimming when the conversation exceeds this many messages.
const LIVE_TRIM_THRESHOLD: usize = 40;
/// Number of trailing messages to always preserve regardless of type.
const LIVE_TRIM_KEEP_RECENT: usize = 10;

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

    /// Returns only human-visible messages: real user prompts and all assistant messages.
    /// Excludes the system prompt and any runtime-injected user messages (tool results,
    /// tool errors, and correction sentinels), as identified by `is_runtime_injected`.
    pub fn human_visible_snapshot(&self) -> Vec<Message> {
        self.messages
            .iter()
            .filter(|m| match m.role {
                Role::System => false,
                Role::User => !is_runtime_injected(&m.content),
                Role::Assistant => true,
            })
            .cloned()
            .collect()
    }

    /// Returns the content of the most recently added user message, if any.
    /// Used by the engine to inspect the last injected tool result or error.
    pub fn last_user_content(&self) -> Option<&str> {
        self.messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.as_str())
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

    /// Removes the last message if it is an assistant message.
    /// Used to discard a bad assistant response before injecting a correction.
    pub fn discard_last_if_assistant(&mut self) {
        if matches!(self.messages.last(), Some(m) if m.role == Role::Assistant) {
            self.messages.pop();
        }
    }

    /// Resets the conversation to just the system prompt.
    pub fn reset(&mut self, system_prompt: String) {
        self.messages.clear();
        self.messages.push(Message::system(system_prompt));
    }

    /// Returns the current number of messages in the conversation.
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Removes complete tool-exchange pairs (assistant tool-call + user tool-result)
    /// from the oldest part of the eligible window, until the conversation is at or
    /// below LIVE_TRIM_THRESHOLD messages.
    ///
    /// Invariants:
    /// - Index 0 (system prompt) is never touched.
    /// - The most recent LIVE_TRIM_KEEP_RECENT messages are never removed.
    /// - Only complete pairs are removed; conversational messages are never touched.
    /// - If no pairs exist in the eligible window, the method is a no-op.
    pub fn trim_tool_exchanges_if_needed(&mut self) {
        if self.messages.len() <= LIVE_TRIM_THRESHOLD {
            return;
        }

        let len = self.messages.len();
        // Eligible window: indices 1 .. (len - LIVE_TRIM_KEEP_RECENT), exclusive.
        // Index 0 = system prompt. Tail LIVE_TRIM_KEEP_RECENT messages = always kept.
        let eligible_end = len.saturating_sub(LIVE_TRIM_KEEP_RECENT);
        if eligible_end <= 1 {
            return;
        }

        // Collect indices of complete tool-exchange pairs, oldest first.
        let mut pair_starts: Vec<usize> = Vec::new();
        let mut i = 1usize;
        while i + 1 < eligible_end {
            let a = &self.messages[i];
            let b = &self.messages[i + 1];
            if a.role == Role::Assistant
                && a.content.trim_start().starts_with('[')
                && b.role == Role::User
                && is_runtime_injected(&b.content)
            {
                pair_starts.push(i);
                i += 2;
            } else {
                i += 1;
            }
        }

        if pair_starts.is_empty() {
            return;
        }

        // Mark pairs for removal oldest-first until under threshold.
        let mut remove: Vec<usize> = Vec::new();
        let mut projected = len;
        for &start in &pair_starts {
            if projected <= LIVE_TRIM_THRESHOLD {
                break;
            }
            remove.push(start);
            remove.push(start + 1);
            projected -= 2;
        }

        // Remove in reverse index order so earlier removals don't shift later indices.
        remove.sort_unstable_by(|a, b| b.cmp(a));
        for idx in remove {
            self.messages.remove(idx);
        }
    }
}

/// Returns true for user messages injected by the runtime (tool results, errors,
/// and fabrication corrections). These are the result halves of tool-exchange pairs.
fn is_runtime_injected(content: &str) -> bool {
    content.starts_with("=== tool_result:")
        || content.starts_with("=== tool_error:")
        || content.starts_with("[runtime:correction]")
}

#[cfg(test)]
mod tests {
    use super::{Conversation, LIVE_TRIM_KEEP_RECENT, LIVE_TRIM_THRESHOLD};

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

    /// Builds a conversation with alternating user/assistant messages.
    /// Tool-exchange pairs are inserted at the specified pair_count positions
    /// after the system prompt. The tail is filled with plain conversational messages.
    fn make_conversation_with_pairs(tool_pairs: usize, conversational_tail: usize) -> Conversation {
        let mut c = Conversation::new("system".to_string());
        for _ in 0..tool_pairs {
            // assistant pure tool call
            c.messages.push(crate::llm::backend::Message::assistant(
                "[read_file: foo.rs]".to_string(),
            ));
            // user tool result
            c.messages.push(crate::llm::backend::Message::user(
                "=== tool_result: read_file ===\ncontent\n=== /tool_result ===".to_string(),
            ));
        }
        for i in 0..conversational_tail {
            c.messages
                .push(crate::llm::backend::Message::user(format!("user msg {i}")));
            c.messages
                .push(crate::llm::backend::Message::assistant(format!(
                    "assistant reply {i}"
                )));
        }
        c
    }

    #[test]
    fn trim_is_noop_below_threshold() {
        // 1 system + 2 pairs (4 messages) + 2 conv (4 messages) = 9 total — well below 40
        let mut c = make_conversation_with_pairs(2, 2);
        let before = c.message_count();
        c.trim_tool_exchanges_if_needed();
        assert_eq!(
            c.message_count(),
            before,
            "must not trim when below threshold"
        );
    }

    #[test]
    fn trim_removes_oldest_pairs_first() {
        // Build a conversation over the threshold: 1 system + 20 pairs (40 messages) + 5 conv (10 messages) = 51
        // After trimming, should be at or below LIVE_TRIM_THRESHOLD (40)
        let mut c = make_conversation_with_pairs(20, 5);
        assert!(c.message_count() > LIVE_TRIM_THRESHOLD);
        c.trim_tool_exchanges_if_needed();
        assert!(
            c.message_count() <= LIVE_TRIM_THRESHOLD,
            "expected <= {LIVE_TRIM_THRESHOLD}, got {}",
            c.message_count()
        );
    }

    #[test]
    fn trim_preserves_system_prompt() {
        let mut c = make_conversation_with_pairs(20, 5);
        c.trim_tool_exchanges_if_needed();
        let messages = c.snapshot();
        assert_eq!(
            messages[0].content, "system",
            "system prompt must remain at index 0"
        );
    }

    #[test]
    fn trim_preserves_recent_tail() {
        // 1 system + 20 pairs (40) + 5 conversational pairs (10) = 51 messages
        // The 10 tail messages are the 5 conversational pairs at the end
        let mut c = make_conversation_with_pairs(20, 5);
        let messages_before = c.snapshot();
        let tail_before: Vec<_> =
            messages_before[messages_before.len() - LIVE_TRIM_KEEP_RECENT..].to_vec();

        c.trim_tool_exchanges_if_needed();

        let messages_after = c.snapshot();
        let tail_after = &messages_after[messages_after.len() - LIVE_TRIM_KEEP_RECENT..];
        assert_eq!(
            tail_before, tail_after,
            "recent tail must be unchanged after trim"
        );
    }

    #[test]
    fn trim_does_not_remove_conversational_messages() {
        // Only conversational messages (no tool pairs) — trim must be a no-op even over threshold
        let mut c = Conversation::new("system".to_string());
        // Fill past threshold with plain user/assistant pairs
        for i in 0..25 {
            c.messages
                .push(crate::llm::backend::Message::user(format!("question {i}")));
            c.messages
                .push(crate::llm::backend::Message::assistant(format!(
                    "answer {i}"
                )));
        }
        let before = c.message_count();
        assert!(before > LIVE_TRIM_THRESHOLD);
        c.trim_tool_exchanges_if_needed();
        assert_eq!(
            c.message_count(),
            before,
            "conversational messages must never be removed"
        );
    }
}
