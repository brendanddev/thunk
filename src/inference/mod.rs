// src/inference/mod.rs
//
// Public API for the inference module.
//
// model_thread loads the backend once and handles prompts in a loop.
// After each generation it checks for tool calls in the response —
// if found, it runs them and continues the conversation automatically.

mod backend;
mod llama_cpp;
mod ollama;
mod openai_compat;

pub use backend::{InferenceBackend, Message, SYSTEM_PROMPT, system_prompt_with_tools};
pub use llama_cpp::LlamaCppBackend;
pub use ollama::OllamaBackend;
pub use openai_compat::OpenAICompatBackend;

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use crate::events::{InferenceEvent, PendingAction, PendingActionKind};
use crate::config;
use crate::error::{ParamsError, Result};
use crate::memory::{compression, facts::FactStore, index::ProjectIndex};
use crate::tools::{PendingToolAction, ToolRegistry};

pub enum SessionCommand {
    SubmitUser(String),
    InjectUserContext(String),
    RequestShellCommand(String),
    ApproveAction(u64),
    RejectAction(u64),
    ClearSession,
}

/// Persistent model thread — loads the backend once, handles prompts in a loop.
/// After each response it checks for tool calls and runs a follow-up if needed.
pub fn model_thread(
    prompt_rx: Receiver<SessionCommand>,
    token_tx: Sender<InferenceEvent>,
) {
    let cfg = match config::load() {
        Ok(c) => c,
        Err(e) => {
            let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
            return;
        }
    };

    let backend = match load_backend_with_fallback(&cfg, &token_tx) {
        Ok(backend) => backend,
        Err(e) => {
            let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
            return;
        }
    };

    let _ = token_tx.send(InferenceEvent::Ready);
    let _ = token_tx.send(InferenceEvent::BackendName(backend.name()));

    // Build the tool registry — same instance reused across all prompts
    let tools = ToolRegistry::default();
    let project_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let project_name = project_root.to_string_lossy().to_string();
    let fact_store = FactStore::open().ok();
    let project_index = ProjectIndex::open_for(&project_root).ok();
    let session_facts = fact_store
        .as_ref()
        .and_then(|store| store.get_relevant_facts(&project_name, "", 5).ok())
        .unwrap_or_default();
    let mut session_messages = vec![Message::system(
        &build_system_prompt(&tools, &session_facts, &[]),
    )];
    let mut next_action_id = 1u64;

    while let Ok(command) = prompt_rx.recv() {
        match command {
            SessionCommand::ClearSession => {
                session_messages.clear();
                session_messages.push(Message::system(
                    &build_system_prompt(&tools, &session_facts, &[]),
                ));
                continue;
            }
            SessionCommand::InjectUserContext(content) => {
                session_messages.push(Message::user(&content));
                continue;
            }
            SessionCommand::RequestShellCommand(command) => {
                let pending = PendingToolAction {
                    tool_name: "bash".to_string(),
                    argument: command.clone(),
                    title: "Approve shell command".to_string(),
                    preview: command,
                };
                if let Err(e) = handle_pending_action(
                    &prompt_rx,
                    &token_tx,
                    &*backend,
                    &tools,
                    &mut session_messages,
                    next_action_id,
                    pending,
                    false,
                ) {
                    let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                }
                let _ = token_tx.send(InferenceEvent::Done);
                next_action_id = next_action_id.saturating_add(1);
                continue;
            }
            SessionCommand::ApproveAction(_) | SessionCommand::RejectAction(_) => {
                let _ = token_tx.send(InferenceEvent::Error(
                    "No action is currently awaiting approval".to_string()
                ));
                continue;
            }
            SessionCommand::SubmitUser(prompt) => {
                session_messages.push(Message::user(&prompt));

                let relevant_summaries = project_index
                    .as_ref()
                    .and_then(|index| index.find_relevant(&prompt, 4).ok())
                    .unwrap_or_default();

                if let Some(first) = session_messages.first_mut() {
                    if first.role == "system" {
                        first.content = build_system_prompt(&tools, &session_facts, &relevant_summaries);
                    }
                }

                compression::compress_history(&mut session_messages, &*backend);

                // Run generation — collect full response for tool call detection
                let response = run_and_collect(&*backend, &session_messages, token_tx.clone());

                match response {
                    Err(e) => {
                        let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                    }
                    Ok(full_response) => {
                        // Check if the model used any tools
                        let tool_execution = tools.execute_tool_calls(&full_response);
                        let tool_results = tool_execution.results;

                        if let Some(pending) = tool_execution.pending {
                            session_messages.push(Message::assistant(&full_response));
                            if let Some(result_msg) = ToolRegistry::format_results(&tool_results) {
                                session_messages.push(Message::user(&result_msg));
                            }
                            if let Err(e) = handle_pending_action(
                                &prompt_rx,
                                &token_tx,
                                &*backend,
                                &tools,
                                &mut session_messages,
                                next_action_id,
                                pending,
                                true,
                            ) {
                                let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                            }
                            next_action_id = next_action_id.saturating_add(1);
                        } else if let Some(result_msg) = ToolRegistry::format_results(&tool_results) {
                            // Notify the UI that tool calls are being processed
                            let _ = token_tx.send(InferenceEvent::ToolCall(
                                tool_results.iter()
                                    .map(|r| format!("{}({})", r.tool_name, r.argument))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ));

                            session_messages.push(Message::assistant(&full_response));
                            session_messages.push(Message::user(&result_msg));

                            match run_and_collect(&*backend, &session_messages, token_tx.clone()) {
                                Ok(follow_up) => {
                                    if !follow_up.trim().is_empty() {
                                        session_messages.push(Message::assistant(&follow_up));
                                    }
                                }
                                Err(e) => {
                                    let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                                }
                            }
                        } else if !full_response.trim().is_empty() {
                            session_messages.push(Message::assistant(&full_response));
                        }

                        let _ = token_tx.send(InferenceEvent::Done);
                    }
                }
            }
        }
    }

    if let Some(store) = fact_store.as_ref() {
        store.extract_and_store(&session_messages, &project_name, &*backend);
    }
}

fn handle_pending_action(
    prompt_rx: &Receiver<SessionCommand>,
    token_tx: &Sender<InferenceEvent>,
    backend: &dyn InferenceBackend,
    tools: &ToolRegistry,
    session_messages: &mut Vec<Message>,
    action_id: u64,
    mut pending: PendingToolAction,
    run_follow_up: bool,
) -> Result<()> {
    if pending.preview.is_empty() {
        pending.preview = pending.argument.clone();
    }

    let event = PendingAction {
        id: action_id,
        kind: PendingActionKind::ShellCommand,
        title: pending.title.clone(),
        preview: pending.preview.clone(),
    };
    let _ = token_tx.send(InferenceEvent::PendingAction(event));

    loop {
        match prompt_rx.recv() {
            Ok(SessionCommand::ApproveAction(id)) if id == action_id => {
                let result = tools.execute_pending_action(&pending);
                let _ = token_tx.send(InferenceEvent::ToolCall(format!(
                    "{}({})",
                    result.tool_name, result.argument
                )));
                let result_msg = ToolRegistry::format_results(&[result]);
                if let Some(result_msg) = result_msg {
                    session_messages.push(Message::user(&result_msg));
                }
                if run_follow_up {
                    let follow_up = run_and_collect(backend, session_messages, token_tx.clone())?;
                    if !follow_up.trim().is_empty() {
                        session_messages.push(Message::assistant(&follow_up));
                    }
                }
                return Ok(());
            }
            Ok(SessionCommand::RejectAction(id)) if id == action_id => {
                let rejection = format!(
                    "User rejected proposed action: {}",
                    pending.argument
                );
                session_messages.push(Message::user(&rejection));
                if run_follow_up {
                    let follow_up = run_and_collect(backend, session_messages, token_tx.clone())?;
                    if !follow_up.trim().is_empty() {
                        session_messages.push(Message::assistant(&follow_up));
                    }
                }
                return Ok(());
            }
            Ok(SessionCommand::ClearSession) => {
                return Err(ParamsError::Config(
                    "Cannot clear session while an action is awaiting approval".to_string()
                ));
            }
            Ok(_) => {}
            Err(_) => {
                return Err(ParamsError::Inference(
                    "Approval channel closed while waiting for user decision".to_string()
                ));
            }
        }
    }
}

/// Run generation and collect the full response as a String,
/// while also streaming tokens to the UI via the channel.
fn run_and_collect(
    backend: &dyn InferenceBackend,
    messages: &[Message],
    token_tx: Sender<InferenceEvent>,
) -> Result<String> {
    use std::sync::{Arc, Mutex};

    // We need to both stream tokens to the UI and collect them locally.
    // Use a shared buffer that both the channel sender and collector write to.
    let buffer = Arc::new(Mutex::new(String::new()));
    let buffer_clone = buffer.clone();

    // Wrap the token_tx to also write to buffer
    let (intercept_tx, intercept_rx) = std::sync::mpsc::channel::<InferenceEvent>();

    // Spawn a relay thread that forwards tokens to the UI and collects them.
    let relay_token_tx = token_tx.clone();
    let relay = std::thread::spawn(move || {
        while let Ok(event) = intercept_rx.recv() {
            match &event {
                InferenceEvent::Token(t) => {
                    if let Ok(mut buf) = buffer_clone.lock() {
                        buf.push_str(t);
                    }
                    let _ = relay_token_tx.send(event);
                }
                _ => {
                    // Don't forward Done/Error — model_thread handles those
                }
            }
        }
    });

    backend.generate(messages, intercept_tx)?;
    relay.join().map_err(|_| {
        ParamsError::Inference("token relay thread panicked".to_string())
    })?;

    let result = buffer.lock()
        .map(|b| b.clone())
        .unwrap_or_default();

    Ok(result)
}

pub fn load_backend_from_config(cfg: &config::Config) -> Result<Box<dyn InferenceBackend>> {
    match cfg.backend.as_str() {
        "ollama" => {
            let backend = OllamaBackend::new(&cfg.ollama.url, &cfg.ollama.model);
            backend.health_check()?;
            Ok(Box::new(backend))
        }
        "openai_compat" => {
            let provider_name = cfg.openai_compat.resolved_provider_name();
            let backend = OpenAICompatBackend::new(
                &cfg.openai_compat.url,
                &cfg.openai_compat.api_key,
                &cfg.openai_compat.model,
                &provider_name,
            );
            backend.health_check()?;
            Ok(Box::new(backend))
        }
        "llama_cpp" => load_llama_cpp_backend(cfg),
        other => Err(ParamsError::Config(format!(
            "Unknown backend `{other}`. Expected llama_cpp, ollama, or openai_compat."
        ))),
    }
}

fn load_backend_with_fallback(
    cfg: &config::Config,
    token_tx: &Sender<InferenceEvent>,
) -> Result<Box<dyn InferenceBackend>> {
    match cfg.backend.as_str() {
        "ollama" => {
            let backend = OllamaBackend::new(&cfg.ollama.url, &cfg.ollama.model);
            if let Err(e) = backend.health_check() {
                let _ = token_tx.send(InferenceEvent::Error(
                    format!("Ollama failed: {e} — falling back to llama.cpp")
                ));
                load_llama_cpp_backend(cfg)
            } else {
                Ok(Box::new(backend))
            }
        }
        _ => load_backend_from_config(cfg),
    }
}

fn load_llama_cpp_backend(cfg: &config::Config) -> Result<Box<dyn InferenceBackend>> {
    let model_path = match &cfg.llama_cpp.model_path {
        Some(p) => p.clone(),
        None => config::find_model()?,
    };
    let backend = LlamaCppBackend::load(
        model_path,
        cfg.generation.max_tokens,
        cfg.generation.temperature,
    )?;
    Ok(Box::new(backend))
}

fn build_system_prompt(
    tools: &ToolRegistry,
    facts: &[String],
    summaries: &[(String, String)],
) -> String {
    let mut prompt = system_prompt_with_tools(&tools.tool_descriptions());

    if !facts.is_empty() {
        prompt.push_str("\n\nRelevant prior project facts:\n");
        for fact in facts {
            prompt.push_str("- ");
            prompt.push_str(fact);
            prompt.push('\n');
        }
    }

    if !summaries.is_empty() {
        prompt.push_str("\nRelevant indexed file summaries:\n");
        for (path, summary) in summaries {
            prompt.push_str("- ");
            prompt.push_str(path);
            prompt.push_str(": ");
            prompt.push_str(summary);
            prompt.push('\n');
        }
    }

    prompt
}
