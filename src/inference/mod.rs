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
use tracing::{debug, info, warn};
use crate::events::{BudgetUpdate, InferenceEvent, PendingAction, PendingActionKind};
use crate::config;
use crate::error::{ParamsError, Result};
use crate::memory::{compression, facts::FactStore, index::ProjectIndex};
use crate::tools::{PendingToolAction, ToolRegistry};

pub enum SessionCommand {
    SubmitUser(String),
    InjectUserContext(String),
    RequestShellCommand(String),
    SetReflection(bool),
    ApproveAction(u64),
    RejectAction(u64),
    ClearSession,
}

#[derive(Default)]
struct SessionBudget {
    prompt_tokens: usize,
    completion_tokens: usize,
    estimated_cost_usd: f64,
    has_cost_estimate: bool,
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
    info!(backend = backend.name(), "model thread initialized");

    let _ = token_tx.send(InferenceEvent::Ready);
    let _ = token_tx.send(InferenceEvent::BackendName(backend.name()));
    let mut reflection_enabled = cfg.reflection.enabled;
    info!(enabled = reflection_enabled, "reflection initial state");
    let _ = token_tx.send(InferenceEvent::ReflectionEnabled(reflection_enabled));

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
    let mut budget = SessionBudget {
        has_cost_estimate: cfg.backend == "llama_cpp" || cfg.backend == "ollama",
        ..SessionBudget::default()
    };
    let mut next_action_id = 1u64;

    while let Ok(command) = prompt_rx.recv() {
        match command {
            SessionCommand::ClearSession => {
                info!("session cleared");
                session_messages.clear();
                session_messages.push(Message::system(
                    &build_system_prompt(&tools, &session_facts, &[]),
                ));
                budget = SessionBudget {
                    has_cost_estimate: cfg.backend == "llama_cpp" || cfg.backend == "ollama",
                    ..SessionBudget::default()
                };
                emit_budget_update(&budget, &token_tx);
                continue;
            }
            SessionCommand::InjectUserContext(content) => {
                info!(chars = content.chars().count(), "user context injected");
                session_messages.push(Message::user(&content));
                continue;
            }
            SessionCommand::RequestShellCommand(command) => {
                info!("shell command approval requested");
                let pending = PendingToolAction {
                    kind: PendingActionKind::ShellCommand,
                    tool_name: "bash".to_string(),
                    argument: command.clone(),
                    display_argument: command.clone(),
                    title: "Approve shell command".to_string(),
                    preview: command,
                };
                if let Err(e) = handle_pending_action(
                    &prompt_rx,
                    &token_tx,
                    &*backend,
                    &tools,
                    &mut session_messages,
                    &cfg,
                    &mut budget,
                    reflection_enabled,
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
            SessionCommand::SetReflection(enabled) => {
                reflection_enabled = enabled;
                info!(enabled, "reflection state updated");
                let _ = token_tx.send(InferenceEvent::ReflectionEnabled(enabled));
                continue;
            }
            SessionCommand::ApproveAction(_) | SessionCommand::RejectAction(_) => {
                warn!("approval command received with no pending action");
                let _ = token_tx.send(InferenceEvent::Error(
                    "No action is currently awaiting approval".to_string()
                ));
                continue;
            }
            SessionCommand::SubmitUser(prompt) => {
                info!(
                    reflection_enabled,
                    existing_messages = session_messages.len(),
                    "user turn submitted"
                );
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
                let response = run_and_collect(
                    &*backend,
                    &session_messages,
                    token_tx.clone(),
                    !reflection_enabled,
                );
                debug!(
                    reflection_enabled,
                    message_count = session_messages.len(),
                    "generation started"
                );
                let prompt_tokens = estimate_message_tokens(&session_messages);

                match response {
                    Err(e) => {
                        let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                    }
                    Ok(full_response) => {
                        info!(
                            response_chars = full_response.chars().count(),
                            reflection_enabled,
                            "generation completed"
                        );
                        record_generation_budget(
                            &cfg,
                            &mut budget,
                            &token_tx,
                            prompt_tokens,
                            &full_response,
                        );

                        // Check if the model used any tools
                        let tool_execution = tools.execute_tool_calls(&full_response);
                        let tool_results = tool_execution.results;
                        info!(
                            tool_results = tool_results.len(),
                            pending = tool_execution.pending.is_some(),
                            "tool scan completed"
                        );

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
                                &cfg,
                                &mut budget,
                                reflection_enabled,
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

                            match run_and_collect(
                                &*backend,
                                &session_messages,
                                token_tx.clone(),
                                !reflection_enabled,
                            ) {
                                Ok(follow_up) => {
                                    let prompt_tokens = estimate_message_tokens(&session_messages);
                                    record_generation_budget(
                                        &cfg,
                                        &mut budget,
                                        &token_tx,
                                        prompt_tokens,
                                        &follow_up,
                                    );
                                    let final_response = if reflection_enabled {
                                        reflect_response(
                                            &*backend,
                                            &cfg,
                                            &mut budget,
                                            &token_tx,
                                            &session_messages,
                                            &follow_up,
                                        )
                                    } else {
                                        Ok(follow_up)
                                    };

                                    match final_response {
                                        Ok(final_response) => {
                                            if !final_response.trim().is_empty() {
                                                session_messages.push(Message::assistant(&final_response));
                                            }
                                        }
                                        Err(e) => {
                                            warn!(error = %e, "reflection after tool follow-up failed");
                                            let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                                        }
                                    }
                                }
                                Err(e) => {
                                    warn!(error = %e, "tool follow-up generation failed");
                                    let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                                }
                            }
                        } else {
                            let final_response = if reflection_enabled {
                                reflect_response(
                                    &*backend,
                                    &cfg,
                                    &mut budget,
                                    &token_tx,
                                    &session_messages,
                                    &full_response,
                                )
                            } else {
                                Ok(full_response)
                            };

                            match final_response {
                                Ok(final_response) => {
                                    if !final_response.trim().is_empty() {
                                        session_messages.push(Message::assistant(&final_response));
                                    }
                                }
                                Err(e) => {
                                    warn!(error = %e, "final response post-processing failed");
                                    let _ = token_tx.send(InferenceEvent::Error(e.to_string()));
                                }
                            }
                        }

                        let _ = token_tx.send(InferenceEvent::Done);
                    }
                }
            }
        }
    }

    if let Some(store) = fact_store.as_ref() {
        info!("persisting session facts");
        store.extract_and_store(&session_messages, &project_name, &*backend);
    }
}

fn handle_pending_action(
    prompt_rx: &Receiver<SessionCommand>,
    token_tx: &Sender<InferenceEvent>,
    backend: &dyn InferenceBackend,
    tools: &ToolRegistry,
    session_messages: &mut Vec<Message>,
    cfg: &config::Config,
    budget: &mut SessionBudget,
    reflection_enabled: bool,
    action_id: u64,
    mut pending: PendingToolAction,
    run_follow_up: bool,
) -> Result<()> {
    if pending.preview.is_empty() {
        pending.preview = pending.argument.clone();
    }

    let event = PendingAction {
        id: action_id,
        kind: pending.kind.clone(),
        title: pending.title.clone(),
        preview: pending.preview.clone(),
    };
    info!(
        action_id,
        tool = pending.tool_name.as_str(),
        kind = ?pending.kind,
        run_follow_up,
        "pending action proposed"
    );
    let _ = token_tx.send(InferenceEvent::PendingAction(event));

    loop {
        match prompt_rx.recv() {
            Ok(SessionCommand::ApproveAction(id)) if id == action_id => {
                info!(action_id, tool = pending.tool_name.as_str(), "pending action approved");
                let result = tools.execute_pending_action(&pending);
                let _ = token_tx.send(InferenceEvent::ToolCall(format!(
                    "{}({})",
                    result.tool_name, result.argument
                )));
                let result_msg = ToolRegistry::format_results(&[result]);
                if let Some(result_msg) = result_msg {
                    session_messages.push(Message::user(&result_msg));
                    if !run_follow_up {
                        let _ = token_tx.send(InferenceEvent::ContextMessage(result_msg));
                    }
                }
                if run_follow_up {
                    let prompt_tokens = estimate_message_tokens(session_messages);
                    let follow_up = run_and_collect(
                        backend,
                        session_messages,
                        token_tx.clone(),
                        !reflection_enabled,
                    )?;
                    record_generation_budget(
                        cfg,
                        budget,
                        token_tx,
                        prompt_tokens,
                        &follow_up,
                    );
                    let final_response = if reflection_enabled {
                        reflect_response(
                            backend,
                            cfg,
                            budget,
                            token_tx,
                            session_messages,
                            &follow_up,
                        )?
                    } else {
                        follow_up
                    };
                    if !final_response.trim().is_empty() {
                        session_messages.push(Message::assistant(&final_response));
                    }
                }
                return Ok(());
            }
            Ok(SessionCommand::RejectAction(id)) if id == action_id => {
                info!(action_id, tool = pending.tool_name.as_str(), "pending action rejected");
                let rejection = format!(
                    "User rejected proposed action: {}",
                    pending.display_argument
                );
                session_messages.push(Message::user(&rejection));
                if !run_follow_up {
                    let _ = token_tx.send(InferenceEvent::ContextMessage(rejection));
                }
                if run_follow_up {
                    let follow_up = run_and_collect(
                        backend,
                        session_messages,
                        token_tx.clone(),
                        !reflection_enabled,
                    )?;
                    let prompt_tokens = estimate_message_tokens(session_messages);
                    record_generation_budget(
                        cfg,
                        budget,
                        token_tx,
                        prompt_tokens,
                        &follow_up,
                    );
                    let final_response = if reflection_enabled {
                        reflect_response(
                            backend,
                            cfg,
                            budget,
                            token_tx,
                            session_messages,
                            &follow_up,
                        )?
                    } else {
                        follow_up
                    };
                    if !final_response.trim().is_empty() {
                        session_messages.push(Message::assistant(&final_response));
                    }
                }
                return Ok(());
            }
            Ok(SessionCommand::ClearSession) => {
                warn!(action_id, "clear requested while pending action active");
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

fn estimate_message_tokens(messages: &[Message]) -> usize {
    let mut total = 0usize;
    for message in messages {
        total = total
            .saturating_add(estimate_text_tokens(&message.role))
            .saturating_add(estimate_text_tokens(&message.content))
            .saturating_add(4);
    }
    total.saturating_add(2)
}

fn estimate_text_tokens(text: &str) -> usize {
    let chars = text.chars().count();
    if chars == 0 {
        0
    } else {
        chars.div_ceil(4)
    }
}

fn estimate_generation_cost_usd(
    cfg: &config::Config,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> Option<f64> {
    match cfg.backend.as_str() {
        "llama_cpp" | "ollama" => Some(0.0),
        "openai_compat" => {
            let input = cfg.budget.input_cost_per_million?;
            let output = cfg
                .budget
                .output_cost_per_million
                .unwrap_or(input);

            Some(
                (prompt_tokens as f64 / 1_000_000.0) * input
                    + (completion_tokens as f64 / 1_000_000.0) * output,
            )
        }
        _ => None,
    }
}

fn record_generation_budget(
    cfg: &config::Config,
    budget: &mut SessionBudget,
    token_tx: &Sender<InferenceEvent>,
    prompt_tokens: usize,
    response: &str,
) {
    let completion_tokens = estimate_text_tokens(response);
    budget.prompt_tokens = budget.prompt_tokens.saturating_add(prompt_tokens);
    budget.completion_tokens = budget.completion_tokens.saturating_add(completion_tokens);

    if let Some(cost) = estimate_generation_cost_usd(cfg, prompt_tokens, completion_tokens) {
        budget.estimated_cost_usd += cost;
        budget.has_cost_estimate = true;
    }

    emit_budget_update(budget, token_tx);
}

fn emit_budget_update(budget: &SessionBudget, token_tx: &Sender<InferenceEvent>) {
    let _ = token_tx.send(InferenceEvent::Budget(BudgetUpdate {
        prompt_tokens: budget.prompt_tokens,
        completion_tokens: budget.completion_tokens,
        total_tokens: budget.prompt_tokens.saturating_add(budget.completion_tokens),
        estimated_cost_usd: if budget.has_cost_estimate {
            Some(budget.estimated_cost_usd)
        } else {
            None
        },
    }));
}

/// Run generation and collect the full response as a String,
/// while also streaming tokens to the UI via the channel.
fn run_and_collect(
    backend: &dyn InferenceBackend,
    messages: &[Message],
    token_tx: Sender<InferenceEvent>,
    stream_tokens: bool,
) -> Result<String> {
    use std::sync::{Arc, Mutex};

    debug!(
        stream_tokens,
        message_count = messages.len(),
        "run_and_collect started"
    );

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
                    if stream_tokens {
                        let _ = relay_token_tx.send(event);
                    }
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

    debug!(chars = result.chars().count(), stream_tokens, "run_and_collect completed");

    Ok(result)
}

fn reflect_response(
    backend: &dyn InferenceBackend,
    cfg: &config::Config,
    budget: &mut SessionBudget,
    token_tx: &Sender<InferenceEvent>,
    session_messages: &[Message],
    draft: &str,
) -> Result<String> {
    if draft.trim().is_empty() {
        return Ok(String::new());
    }

    info!(draft_chars = draft.chars().count(), "reflection pass started");
    let reflection_messages = build_reflection_messages(session_messages, draft);
    let prompt_tokens = estimate_message_tokens(&reflection_messages);
    let reflected = run_and_collect(
        backend,
        &reflection_messages,
        token_tx.clone(),
        true,
    )?;
    record_generation_budget(cfg, budget, token_tx, prompt_tokens, &reflected);

    if reflected.trim().is_empty() || looks_like_reflection_meta(&reflected) {
        warn!("reflection returned empty or meta response, falling back to draft");
        Ok(draft.to_string())
    } else {
        info!(chars = reflected.chars().count(), "reflection pass completed");
        Ok(reflected)
    }
}

fn build_reflection_messages(session_messages: &[Message], draft: &str) -> Vec<Message> {
    let mut messages = vec![Message::system(
        "You are a reflection pass for params-cli. Rewrite the assistant draft into the final user-facing answer. Fix correctness, safety, unsupported claims, missed repo/tool context, and clarity issues. Return only the final answer itself. Never talk about the draft, reflection, review, edits, or whether the answer was already good. Do not call tools. Keep good answers concise.",
    )];

    for message in session_messages {
        if message.role != "system" {
            messages.push(message.clone());
        }
    }

    messages.push(Message::assistant(draft));
    messages.push(Message::user(
        "Rewrite the assistant draft above as the final answer to the user. If the draft is already good, return the same answer text with only minimal edits. Do not add reviewer commentary. Return only the final answer text.",
    ));
    messages
}

fn looks_like_reflection_meta(text: &str) -> bool {
    let trimmed = text.trim().to_ascii_lowercase();
    if trimmed.is_empty() {
        return false;
    }

    let meta_markers = [
        "the draft is already good",
        "the draft is good",
        "no further edits needed",
        "no changes needed",
        "no revisions needed",
        "the answer is already good",
        "minimal edits",
        "review the assistant draft",
        "reviewer commentary",
    ];

    meta_markers.iter().any(|marker| trimmed.contains(marker))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflection_messages_exclude_original_system_prompt() {
        let session_messages = vec![
            Message::system("tool-heavy system prompt"),
            Message::user("question"),
            Message::assistant("draft context"),
        ];

        let reflection = build_reflection_messages(&session_messages, "final draft");

        assert_eq!(reflection[0].role, "system");
        assert!(!reflection.iter().skip(1).any(|m| m.content == "tool-heavy system prompt"));
        assert_eq!(reflection[1].role, "user");
        assert_eq!(reflection[1].content, "question");
        assert_eq!(reflection[2].role, "assistant");
        assert_eq!(reflection[2].content, "draft context");
        assert_eq!(reflection[3].role, "assistant");
        assert_eq!(reflection[3].content, "final draft");
        assert_eq!(reflection[4].role, "user");
    }

    #[test]
    fn reflection_meta_text_falls_back_to_draft() {
        assert!(looks_like_reflection_meta(
            "The draft is already good. No further edits needed."
        ));
        assert!(looks_like_reflection_meta("No changes needed."));
        assert!(!looks_like_reflection_meta("A pointer stores the memory address of another value."));
    }
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
