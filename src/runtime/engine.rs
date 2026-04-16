use crate::app::config::Config;
use crate::llm::backend::{BackendEvent, BackendStatus, GenerateRequest, ModelBackend};

use super::conversation::Conversation;
use super::prompt;
use super::types::{Activity, RuntimeEvent, RuntimeRequest};

pub struct Runtime {
    conversation: Conversation,
    backend: Box<dyn ModelBackend>,
}

impl Runtime {
    pub fn new(config: &Config, backend: Box<dyn ModelBackend>) -> Self {
        Self {
            conversation: Conversation::new(prompt::build_system_prompt(&config.app.name)),
            backend,
        }
    }

    pub fn handle(
        &mut self,
        request: RuntimeRequest,
        on_event: &mut dyn FnMut(RuntimeEvent),
    ) {
        match request {
            RuntimeRequest::Submit { text } => self.handle_submit(text, on_event),
        }
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

        let request = GenerateRequest::new(self.conversation.snapshot());
        on_event(RuntimeEvent::ActivityChanged(Activity::Processing));
        let mut started_assistant_message = false;

        let backend = &mut self.backend;
        let conversation = &mut self.conversation;
        let result = backend.generate(request, &mut |event| match event {
            BackendEvent::StatusChanged(status) => {
                on_event(RuntimeEvent::ActivityChanged(map_backend_status(status)));
            }
            BackendEvent::TextDelta(chunk) => {
                if !started_assistant_message {
                    started_assistant_message = true;
                    conversation.begin_assistant_reply();
                    on_event(RuntimeEvent::ActivityChanged(Activity::Responding));
                    on_event(RuntimeEvent::AssistantMessageStarted);
                }

                conversation.push_assistant_chunk(&chunk);
                on_event(RuntimeEvent::AssistantMessageChunk(chunk));
            }
            BackendEvent::Finished => {}
        });

        match result {
            Ok(()) => {
                if !started_assistant_message {
                    on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                    on_event(RuntimeEvent::Failed {
                        message: format!("{} returned no output.", backend.name()),
                    });
                    return;
                }

                on_event(RuntimeEvent::AssistantMessageFinished);
                on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
            }
            Err(error) => {
                if started_assistant_message {
                    on_event(RuntimeEvent::AssistantMessageFinished);
                }

                on_event(RuntimeEvent::ActivityChanged(Activity::Idle));
                on_event(RuntimeEvent::Failed {
                    message: error.to_string(),
                });
            }
        }
    }
}

fn map_backend_status(status: BackendStatus) -> Activity {
    match status {
        BackendStatus::LoadingModel => Activity::LoadingModel,
        BackendStatus::Generating => Activity::Generating,
    }
}
