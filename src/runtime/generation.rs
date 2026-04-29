use crate::app::Result;
use crate::llm::backend::{BackendEvent, BackendStatus, GenerateRequest, Message, ModelBackend};

use super::conversation::Conversation;
use super::prompt;
use super::tool_surface::ToolSurface;
use super::types::{Activity, RuntimeEvent};

/// Runs a single generation turn: sends the current conversation to the backend,
/// buffers the assistant response into conversation history, then returns the
/// complete response text, or None if the backend produced no output. Assistant
/// message events are emitted only after runtime admission.
pub(super) fn run_generate_turn(
    backend: &mut dyn ModelBackend,
    conversation: &mut Conversation,
    tool_surface: ToolSurface,
    project_snapshot_hint: Option<&str>,
    on_event: &mut dyn FnMut(RuntimeEvent),
) -> Result<Option<String>> {
    let mut messages = conversation.snapshot();
    messages.push(Message::system(prompt::render_tool_surface_hint(
        tool_surface.as_str(),
        tool_surface
            .allowed_tool_names()
            .chain(tool_surface.mutation_tool_names().iter().copied()),
    )));
    if let Some(hint) = project_snapshot_hint {
        messages.push(Message::system(hint.to_string()));
    }
    let request = GenerateRequest::new(messages);
    let mut response = String::new();

    let result = backend.generate(request, &mut |event| match event {
        BackendEvent::StatusChanged(status) => {
            on_event(RuntimeEvent::ActivityChanged(map_backend_status(status)));
        }
        BackendEvent::TextDelta(chunk) => {
            response.push_str(&chunk);
        }
        BackendEvent::Timing { stage, elapsed_ms } => {
            on_event(RuntimeEvent::BackendTiming { stage, elapsed_ms });
        }
        BackendEvent::Finished => {}
    });

    result?;

    if response.is_empty() {
        Ok(None)
    } else {
        conversation.begin_assistant_reply();
        conversation.push_assistant_chunk(&response);
        Ok(Some(response))
    }
}

pub(super) fn emit_visible_assistant_message(text: &str, on_event: &mut dyn FnMut(RuntimeEvent)) {
    on_event(RuntimeEvent::ActivityChanged(Activity::Responding));
    on_event(RuntimeEvent::AssistantMessageStarted);
    on_event(RuntimeEvent::AssistantMessageChunk(text.to_string()));
    on_event(RuntimeEvent::AssistantMessageFinished);
}

fn map_backend_status(status: BackendStatus) -> Activity {
    match status {
        BackendStatus::LoadingModel => Activity::LoadingModel,
        BackendStatus::Generating => Activity::Generating,
    }
}
