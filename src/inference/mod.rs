// src/inference/mod.rs
//
// Public API for the inference module.
//
// model_thread loads the backend once and handles prompts in a loop.
// After each generation it checks for tool calls in the response —
// if found, it runs them and continues the conversation automatically.

mod approval;
mod backend;
mod budget;
mod cache;
mod indexing;
mod llama_cpp;
mod ollama;
mod openai_compat;
mod reflection;
mod runtime;
mod session;

pub use backend::{system_prompt_with_tools, InferenceBackend, Message, SYSTEM_PROMPT};
pub use llama_cpp::LlamaCppBackend;
pub use ollama::OllamaBackend;
pub use openai_compat::OpenAICompatBackend;
#[allow(unused_imports)]
pub use session::{model_thread, model_thread_with_options, SessionRuntimeOptions};

use std::sync::mpsc::Sender;

use crate::config;
use crate::error::{ParamsError, Result};
use crate::events::InferenceEvent;
use crate::tools::ToolRegistry;

use runtime::summary_limit;

pub enum SessionCommand {
    SubmitUser(String),
    InjectUserContext(String),
    RequestShellCommand(String),
    RequestFileWrite {
        path: String,
        content: String,
    },
    RequestFileEdit {
        path: String,
        edits: String,
    },
    ListSessions,
    NewSession(Option<String>),
    RenameSession(String),
    ResumeSession(String),
    DeleteSession(String),
    ExportSession {
        selector: String,
        format: Option<String>,
    },
    SetReflection(bool),
    SetEco(bool),
    SetDebugLogging(bool),
    ClearDebugLog,
    ClearCache,
    ApproveAction(u64),
    RejectAction(u64),
    ClearSession,
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

pub(super) fn load_backend_with_fallback(
    cfg: &config::Config,
    token_tx: &Sender<InferenceEvent>,
) -> Result<Box<dyn InferenceBackend>> {
    match cfg.backend.as_str() {
        "ollama" => {
            let backend = OllamaBackend::new(&cfg.ollama.url, &cfg.ollama.model);
            if let Err(e) = backend.health_check() {
                let _ = token_tx.send(InferenceEvent::Error(format!(
                    "Ollama failed: {e} — falling back to llama.cpp"
                )));
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

pub(super) fn build_system_prompt(
    tools: &ToolRegistry,
    facts: &[String],
    summaries: &[(String, String)],
    eco_enabled: bool,
) -> String {
    let mut prompt = if eco_enabled {
        let mut compact = system_prompt_with_tools(&tools.compact_tool_descriptions());
        compact.push_str("\n\nEco mode is active. Prefer concise answers and minimal tool use.");
        compact
    } else {
        system_prompt_with_tools(&tools.tool_descriptions())
    };

    let fact_limit = if eco_enabled { 2 } else { 5 };
    let summary_cap = summary_limit(eco_enabled);

    if !facts.is_empty() {
        prompt.push_str("\n\nRelevant prior project facts:\n");
        for fact in facts.iter().take(fact_limit) {
            prompt.push_str("- ");
            prompt.push_str(fact);
            prompt.push('\n');
        }
    }

    if !summaries.is_empty() {
        prompt.push_str("\nRelevant indexed file summaries:\n");
        for (path, summary) in summaries.iter().take(summary_cap) {
            prompt.push_str("- ");
            prompt.push_str(path);
            prompt.push_str(": ");
            prompt.push_str(summary);
            prompt.push('\n');
        }
    }

    prompt
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolRegistry;

    #[test]
    fn eco_prompt_is_compact_and_limited() {
        let tools = ToolRegistry::default();
        let facts = vec![
            "fact one".to_string(),
            "fact two".to_string(),
            "fact three".to_string(),
        ];
        let summaries = vec![
            ("a.rs".to_string(), "summary a".to_string()),
            ("b.rs".to_string(), "summary b".to_string()),
            ("c.rs".to_string(), "summary c".to_string()),
        ];

        let prompt = build_system_prompt(&tools, &facts, &summaries, true);

        assert!(prompt.contains("Eco mode is active"));
        assert!(prompt.contains("fact one"));
        assert!(prompt.contains("fact two"));
        assert!(!prompt.contains("fact three"));
        assert!(prompt.contains("a.rs: summary a"));
        assert!(prompt.contains("b.rs: summary b"));
        assert!(!prompt.contains("c.rs: summary c"));
    }
}
