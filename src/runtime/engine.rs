use crate::app::config::Config;
use crate::app::Result;
use crate::llm::backend::{BackendEvent, BackendStatus, GenerateRequest, ModelBackend};
use crate::tools::ToolRegistry;

use super::conversation::Conversation;
use super::prompt;
use super::tool_codec;
use super::types::{Activity, AnswerSource, RuntimeEvent, RuntimeRequest};

/// Maximum tool rounds per turn. Prevents runaway loops when the model keeps
/// producing tool calls without reaching a final answer.
const MAX_TOOL_ROUNDS: usize = 10;

pub struct Runtime {
    conversation: Conversation,
    backend: Box<dyn ModelBackend>,
    registry: ToolRegistry,
    system_prompt: String,
}

impl Runtime {
    pub fn new(config: &Config, backend: Box<dyn ModelBackend>, registry: ToolRegistry) -> Self {
        let specs = registry.specs();
        let system_prompt = prompt::build_system_prompt(&config.app.name, &specs);
        Self {
            conversation: Conversation::new(system_prompt.clone()),
            backend,
            registry,
            system_prompt,
        }
    }

    pub fn handle(&mut self, request: RuntimeRequest, on_event: &mut dyn FnMut(RuntimeEvent)) {
        match request {
            RuntimeRequest::Submit { text } => self.handle_submit(text, on_event),
            RuntimeRequest::Reset => self.handle_reset(on_event),
        }
    }

    fn handle_reset(&mut self, on_event: &mut dyn FnMut(RuntimeEvent)) {
        self.conversation.reset(self.system_prompt.clone());
        on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
    }

    fn handle_submit(&mut self, text: String, on_event: &mut dyn FnMut(RuntimeEvent)) {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            on_event(RuntimeEvent::Failed {
                message: "Cannot submit an empty prompt.".to_string(),
            });
            return;
        }

        self.conversation.push_user(text);
        on_event(RuntimeEvent::ActivityChanged(Activity::Processing));

        let mut tool_rounds = 0usize;

        loop {
            let response = match run_generate_turn(
                self.backend.as_mut(),
                &mut self.conversation,
                on_event,
            ) {
                Ok(Some(r)) => r,
                Ok(None) => {
                    on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                    on_event(RuntimeEvent::Failed {
                        message: format!("{} returned no output.", self.backend.name()),
                    });
                    return;
                }
                Err(e) => {
                    on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                    on_event(RuntimeEvent::Failed { message: e.to_string() });
                    return;
                }
            };

            let calls = tool_codec::parse_tool_calls(&response);

            if calls.is_empty() {
                let source = if tool_rounds == 0 {
                    AnswerSource::Direct
                } else {
                    AnswerSource::ToolAssisted { rounds: tool_rounds }
                };
                on_event(RuntimeEvent::AnswerReady(source));
                on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                return;
            }

            tool_rounds += 1;

            if tool_rounds > MAX_TOOL_ROUNDS {
                on_event(RuntimeEvent::AnswerReady(AnswerSource::ToolLimitReached));
                on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                return;
            }

            on_event(RuntimeEvent::ActivityChanged(Activity::ExecutingTools));

            let mut results_text = String::new();
            for input in calls {
                let name = input.tool_name().to_string();
                on_event(RuntimeEvent::ToolCallStarted { name: name.clone() });
                match self.registry.dispatch(input) {
                    Ok(result) => {
                        on_event(RuntimeEvent::ToolCallFinished { name: name.clone(), success: true });
                        results_text.push_str(&tool_codec::format_tool_result(&name, &result.output));
                    }
                    Err(e) => {
                        on_event(RuntimeEvent::ToolCallFinished { name: name.clone(), success: false });
                        results_text.push_str(&tool_codec::format_tool_error(&name, &e.to_string()));
                    }
                }
            }

            self.conversation.push_user(results_text);
        }
    }
}

/// Runs a single generation turn: sends the current conversation to the backend,
/// streams tokens into the conversation and fires events, then returns the
/// complete assistant response text, or None if the backend produced no output.
fn run_generate_turn(
    backend: &mut dyn ModelBackend,
    conversation: &mut Conversation,
    on_event: &mut dyn FnMut(RuntimeEvent),
) -> Result<Option<String>> {
    let request = GenerateRequest::new(conversation.snapshot());
    let mut started = false;

    let result = backend.generate(request, &mut |event| match event {
        BackendEvent::StatusChanged(status) => {
            on_event(RuntimeEvent::ActivityChanged(map_backend_status(status)));
        }
        BackendEvent::TextDelta(chunk) => {
            if !started {
                started = true;
                conversation.begin_assistant_reply();
                on_event(RuntimeEvent::ActivityChanged(Activity::Responding));
                on_event(RuntimeEvent::AssistantMessageStarted);
            }
            conversation.push_assistant_chunk(&chunk);
            on_event(RuntimeEvent::AssistantMessageChunk(chunk));
        }
        BackendEvent::Finished => {}
    });

    result?;

    if started {
        on_event(RuntimeEvent::AssistantMessageFinished);
        Ok(conversation.last_assistant_content().map(|s| s.to_string()))
    } else {
        Ok(None)
    }
}

fn map_backend_status(status: BackendStatus) -> Activity {
    match status {
        BackendStatus::LoadingModel => Activity::LoadingModel,
        BackendStatus::Generating => Activity::Generating,
    }
}
