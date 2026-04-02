// src/debug_log.rs
//
// Separate opt-in content debug log. This never feeds into the normal
// structural audit log and only records prompts/final answers when enabled.

use std::fs::OpenOptions;
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::config;
use crate::error::Result;

static DEBUG_LOG_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponseSource {
    Live,
    ExactCache,
    PromptCache,
    SemanticCache,
}

fn log_lock() -> &'static Mutex<()> {
    DEBUG_LOG_LOCK.get_or_init(|| Mutex::new(()))
}

pub fn append_user_prompt(text: &str) -> Result<()> {
    append_entry("user_prompt", text)
}

pub fn append_assistant_response(text: &str, source: ResponseSource) -> Result<()> {
    append_entry(
        &format!("assistant_response {}", source_label(source)),
        text,
    )
}

pub fn clear() -> Result<()> {
    let _guard = log_lock().lock().ok();
    std::fs::write(config::debug_log_path()?, "")?;
    Ok(())
}

fn append_entry(kind: &str, text: &str) -> Result<()> {
    let _guard = log_lock().lock().ok();
    let path = config::debug_log_path()?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;

    writeln!(file, "=== {} {} ===", timestamp_secs(), kind)?;
    writeln!(file, "{text}")?;
    writeln!(file)?;
    Ok(())
}

fn source_label(source: ResponseSource) -> &'static str {
    match source {
        ResponseSource::Live => "(LIVE)",
        ResponseSource::ExactCache => "(EXACT CACHE HIT)",
        ResponseSource::PromptCache => "(PROMPT CACHE HIT)",
        ResponseSource::SemanticCache => "(SEMANTIC CACHE HIT)",
    }
}

fn timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_creates_empty_debug_log() {
        clear().unwrap();
        let contents = std::fs::read_to_string(config::debug_log_path().unwrap()).unwrap();
        assert!(contents.is_empty());
    }

    #[test]
    fn source_labels_are_stable() {
        assert_eq!(source_label(ResponseSource::Live), "(LIVE)");
        assert_eq!(source_label(ResponseSource::ExactCache), "(EXACT CACHE HIT)");
        assert_eq!(source_label(ResponseSource::PromptCache), "(PROMPT CACHE HIT)");
        assert_eq!(source_label(ResponseSource::SemanticCache), "(SEMANTIC CACHE HIT)");
    }
}
