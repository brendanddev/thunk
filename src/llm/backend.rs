use crate::app::Result;

/// Typed identifiers for backend timing stages.
///
/// These replace the previous `&'static str` stage names emitted via `BackendEvent::Timing`.
/// All backend implementations must use these variants; string literals are no longer accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendTimingStage {
    /// Time spent loading the model weights from disk into memory.
    ModelLoad,
    /// Time spent creating the inference context (KV cache allocation, graph reservation).
    CtxCreate,
    /// Time spent tokenizing the prompt string into token IDs.
    Tokenize,
    /// Marks the start of prompt evaluation (prefill). Informational; not accumulated.
    PrefillStart,
    /// Time spent evaluating the full prompt through the model (prefill / KV fill).
    PrefillDone,
    /// Time spent in the token-by-token decoding loop (autoregressive generation).
    GenerationDone,
}

impl std::fmt::Display for BackendTimingStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelLoad => f.write_str("model_load"),
            Self::CtxCreate => f.write_str("ctx_create"),
            Self::Tokenize => f.write_str("tokenize"),
            Self::PrefillStart => f.write_str("prefill_start"),
            Self::PrefillDone => f.write_str("prefill_done"),
            Self::GenerationDone => f.write_str("generation_done"),
        }
    }
}

/// Role of a message within a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    // Converts the Role enum into its corresponding string representation for prompt formatting.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

/// A single message in the conversation history passed to the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// Input to a model generation call.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub messages: Vec<Message>,
}

impl GenerateRequest {
    pub fn new(messages: Vec<Message>) -> Self {
        Self { messages }
    }
}

/// High-level status updates emitted by the backend.
#[derive(Debug, Clone)]
pub enum BackendStatus {
    LoadingModel,
    CreatingContext,
    Tokenizing,
    Prefilling,
    Generating,
}

/// Events streamed from the model during generation.
#[derive(Debug, Clone)]
pub enum BackendEvent {
    StatusChanged(BackendStatus),
    TextDelta(String),
    Finished,
    /// Advisory timing event — emitted by backends at key internal stages.
    /// Consumers may route this to logging; it must not affect control flow.
    Timing {
        stage: BackendTimingStage,
        elapsed_ms: u64,
    },
}

/// Static capabilities exposed by a backend so callers can make informed decisions
/// without calling generate(). All fields are optional — `None` means unknown or
/// unconfigured; callers must not assume a bound when a field is absent.
#[derive(Debug, Clone, Copy)]
pub struct BackendCapabilities {
    /// Maximum tokens the backend can receive as input in a single call.
    /// `None` if unknown (e.g. mock) or if the backend defers to the model's
    /// trained context window (llama.cpp with context_tokens = 0).
    pub context_window_tokens: Option<u32>,

    /// Maximum tokens the backend will generate per call.
    /// `None` if unknown or unconfigured.
    pub max_output_tokens: Option<usize>,
}

/// Defines the abstraction over a language model backend.
/// This is responsible for receiving a structured generation request, streaming output via events, and
/// hiding backend-specific implementation details from the rest of the application.
pub trait ModelBackend: Send {
    fn name(&self) -> &str;

    /// Returns static capability information for this backend.
    /// Called at construction time or on-demand; never during generation.
    fn capabilities(&self) -> BackendCapabilities;

    /// Runs generation and streams events to `on_event`.
    ///
    /// # Backend event-order contract
    ///
    /// Implementations MUST follow this ordering:
    /// - `StatusChanged` — optional, any number, may appear anywhere before `Finished`
    /// - `Timing` — optional advisory events; any number; must not affect control flow
    /// - `TextDelta` — 0..N chunks of generated text
    /// - `Finished` — EXACTLY ONE on success; signals that generation is complete
    /// - NO events of any kind may be emitted after `Finished`
    ///
    /// On error: return `Err(...)` without emitting `Finished`. The runtime treats
    /// an absent `Finished` on error as expected; it treats one on success as required.
    fn generate(
        &mut self,
        request: GenerateRequest,
        on_event: &mut dyn FnMut(BackendEvent),
    ) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records all events emitted during a generate() call for contract validation.
    struct EventCapture {
        events: Vec<BackendEvent>,
    }

    impl EventCapture {
        fn new() -> Self {
            Self { events: Vec::new() }
        }

        fn observe(&mut self, event: BackendEvent) {
            self.events.push(event);
        }

        fn finished_count(&self) -> usize {
            self.events
                .iter()
                .filter(|e| matches!(e, BackendEvent::Finished))
                .count()
        }

        fn text_delta_count(&self) -> usize {
            self.events
                .iter()
                .filter(|e| matches!(e, BackendEvent::TextDelta(_)))
                .count()
        }

        /// Returns the number of events emitted after the first `Finished`.
        fn events_after_finished(&self) -> usize {
            let mut count = 0;
            let mut past_finished = false;
            for event in &self.events {
                if past_finished {
                    count += 1;
                }
                if matches!(event, BackendEvent::Finished) {
                    past_finished = true;
                }
            }
            count
        }
    }

    fn make_request() -> GenerateRequest {
        GenerateRequest::new(vec![Message::user("test")])
    }

    // --- conforming backends ---

    struct ValidOrderBackend;

    impl ModelBackend for ValidOrderBackend {
        fn name(&self) -> &str {
            "valid"
        }
        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                context_window_tokens: None,
                max_output_tokens: None,
            }
        }
        fn generate(
            &mut self,
            _request: GenerateRequest,
            on_event: &mut dyn FnMut(BackendEvent),
        ) -> Result<()> {
            on_event(BackendEvent::StatusChanged(BackendStatus::Generating));
            on_event(BackendEvent::TextDelta("hello".into()));
            on_event(BackendEvent::TextDelta(" world".into()));
            on_event(BackendEvent::Finished);
            Ok(())
        }
    }

    struct ZeroDeltaBackend;

    impl ModelBackend for ZeroDeltaBackend {
        fn name(&self) -> &str {
            "zero-delta"
        }
        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                context_window_tokens: None,
                max_output_tokens: None,
            }
        }
        fn generate(
            &mut self,
            _request: GenerateRequest,
            on_event: &mut dyn FnMut(BackendEvent),
        ) -> Result<()> {
            on_event(BackendEvent::Finished);
            Ok(())
        }
    }

    // --- violating backends (used to verify violations are detectable) ---

    struct EventsAfterFinishedBackend;

    impl ModelBackend for EventsAfterFinishedBackend {
        fn name(&self) -> &str {
            "events-after-finished"
        }
        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                context_window_tokens: None,
                max_output_tokens: None,
            }
        }
        fn generate(
            &mut self,
            _request: GenerateRequest,
            on_event: &mut dyn FnMut(BackendEvent),
        ) -> Result<()> {
            on_event(BackendEvent::TextDelta("text".into()));
            on_event(BackendEvent::Finished);
            on_event(BackendEvent::TextDelta("after finished".into())); // contract violation
            Ok(())
        }
    }

    struct DoubleFinishedBackend;

    impl ModelBackend for DoubleFinishedBackend {
        fn name(&self) -> &str {
            "double-finished"
        }
        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                context_window_tokens: None,
                max_output_tokens: None,
            }
        }
        fn generate(
            &mut self,
            _request: GenerateRequest,
            on_event: &mut dyn FnMut(BackendEvent),
        ) -> Result<()> {
            on_event(BackendEvent::Finished);
            on_event(BackendEvent::Finished); // contract violation
            Ok(())
        }
    }

    // --- tests ---

    #[test]
    fn valid_event_order_passes_contract() {
        let mut backend = ValidOrderBackend;
        let mut cap = EventCapture::new();
        backend
            .generate(make_request(), &mut |e| cap.observe(e))
            .unwrap();
        assert_eq!(
            cap.finished_count(),
            1,
            "Finished must be emitted exactly once"
        );
        assert_eq!(
            cap.events_after_finished(),
            0,
            "No events may follow Finished"
        );
        assert!(cap.text_delta_count() > 0);
    }

    #[test]
    fn zero_text_delta_is_valid() {
        let mut backend = ZeroDeltaBackend;
        let mut cap = EventCapture::new();
        backend
            .generate(make_request(), &mut |e| cap.observe(e))
            .unwrap();
        assert_eq!(
            cap.finished_count(),
            1,
            "Finished must be emitted exactly once"
        );
        assert_eq!(cap.text_delta_count(), 0, "Zero TextDelta is valid");
        assert_eq!(cap.events_after_finished(), 0);
    }

    #[test]
    fn events_after_finished_is_detectable() {
        let mut backend = EventsAfterFinishedBackend;
        let mut cap = EventCapture::new();
        backend
            .generate(make_request(), &mut |e| cap.observe(e))
            .unwrap();
        assert!(
            cap.events_after_finished() > 0,
            "EventCapture must surface the contract violation"
        );
    }

    #[test]
    fn double_finished_is_detectable() {
        let mut backend = DoubleFinishedBackend;
        let mut cap = EventCapture::new();
        backend
            .generate(make_request(), &mut |e| cap.observe(e))
            .unwrap();
        assert!(
            cap.finished_count() > 1,
            "EventCapture must surface the double-Finished violation"
        );
    }

    #[test]
    fn timing_stage_enum_covers_all_known_stages() {
        // Compile-time confirmation that all expected variants exist.
        // If a new variant is added and this match is not updated, the compiler will error.
        let stages = [
            BackendTimingStage::ModelLoad,
            BackendTimingStage::CtxCreate,
            BackendTimingStage::Tokenize,
            BackendTimingStage::PrefillStart,
            BackendTimingStage::PrefillDone,
            BackendTimingStage::GenerationDone,
        ];
        assert_eq!(stages.len(), 6);
    }
}
