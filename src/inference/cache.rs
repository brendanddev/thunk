use std::path::Path;
use std::sync::mpsc::Sender;

use tracing::{info, warn};

use crate::cache::{build_cache_scope, ExactCache};
use crate::config;
use crate::debug_log;
use crate::error::Result;
use crate::events::InferenceEvent;

use super::budget::{
    emit_cache_update, estimate_message_tokens, estimate_text_tokens, SessionCacheStats,
};
use super::runtime::run_and_collect;
use super::{InferenceBackend, Message};

#[derive(Clone, Copy)]
pub(super) enum CacheMode {
    ExactOnly,
    PreferPromptLevel,
}

pub(super) struct CacheLookup {
    pub text: String,
    pub hit: bool,
    pub source: debug_log::ResponseSource,
    /// Wall-clock milliseconds spent on live generation.
    /// Zero for cache hits — no model work was done.
    pub elapsed_ms: u64,
}

fn prompt_level_cache_key<'a>(messages: &'a [Message]) -> Option<(&'a str, &'a str)> {
    if messages.last().map(|m| m.role.as_str()) != Some("user") {
        return None;
    }

    let non_system = messages.iter().filter(|m| m.role != "system").count();
    if non_system == 0 || non_system > 12 {
        return None;
    }

    if messages
        .iter()
        .any(|message| message.role == "user" && is_injected_context_message(&message.content))
    {
        return None;
    }

    let system_prompt = messages
        .iter()
        .find(|m| m.role == "system")
        .map(|m| m.content.as_str())?;
    let user_prompt = messages.last().map(|m| m.content.as_str())?;

    Some((system_prompt, user_prompt))
}

fn is_injected_context_message(content: &str) -> bool {
    let prefixes = [
        "Tool results:\n",
        "I've loaded this file for context:",
        "Directory listing:\n",
        "Search results:\n",
        "Git context (",
        "LSP diagnostics:\n",
        "LSP check:\n",
        "Fetched web context:\n",
        "User rejected proposed action:",
    ];

    prefixes.iter().any(|prefix| content.starts_with(prefix))
}

pub(super) fn store_prompt_level_cache(
    exact_cache: Option<&ExactCache>,
    cfg: &config::Config,
    project_root: &Path,
    backend_name: &str,
    messages: &[Message],
    response: &str,
) {
    let Some(cache) = exact_cache else {
        return;
    };
    let cache_scope = match build_cache_scope(project_root, cfg.cache.ttl_seconds) {
        Ok(scope) => scope,
        Err(e) => {
            warn!(error = %e, "cache scope build failed; skipping prompt-level cache store");
            return;
        }
    };
    let Some((system_prompt, user_prompt)) = prompt_level_cache_key(messages) else {
        return;
    };

    if let Err(e) = cache.put_prompt_level(
        backend_name,
        system_prompt,
        user_prompt,
        response,
        &cache_scope,
    ) {
        warn!(error = %e, "prompt-level cache store failed");
    }
}

pub(super) fn store_exact_cache(
    exact_cache: Option<&ExactCache>,
    cfg: &config::Config,
    project_root: &Path,
    backend_name: &str,
    messages: &[Message],
    response: &str,
) {
    let Some(cache) = exact_cache else {
        return;
    };
    let cache_scope = match build_cache_scope(project_root, cfg.cache.ttl_seconds) {
        Ok(scope) => scope,
        Err(e) => {
            warn!(error = %e, "cache scope build failed; skipping exact cache store");
            return;
        }
    };
    if let Err(e) = cache.put(backend_name, messages, response, &cache_scope) {
        warn!(error = %e, "exact cache store failed");
    }
}

pub(super) fn generate_with_cache(
    backend: &dyn InferenceBackend,
    messages: &[Message],
    cfg: &config::Config,
    project_root: &Path,
    token_tx: Sender<InferenceEvent>,
    stream_tokens: bool,
    exact_cache: Option<&ExactCache>,
    cache_stats: &mut SessionCacheStats,
    cache_mode: CacheMode,
) -> Result<CacheLookup> {
    let cache_scope = match build_cache_scope(project_root, cfg.cache.ttl_seconds) {
        Ok(scope) => Some(scope),
        Err(e) => {
            warn!(error = %e, "cache scope build failed; bypassing cache");
            None
        }
    };

    if let (Some(cache), Some(cache_scope)) = (exact_cache, cache_scope.as_ref()) {
        match cache.get(&backend.name(), messages, cache_scope) {
            Ok(Some(cached)) => {
                let saved =
                    estimate_message_tokens(messages).saturating_add(estimate_text_tokens(&cached));
                cache_stats.hits = cache_stats.hits.saturating_add(1);
                cache_stats.tokens_saved = cache_stats.tokens_saved.saturating_add(saved);
                info!(saved_tokens = saved, "exact cache hit");
                emit_cache_update(cache_stats, true, &token_tx);
                if stream_tokens {
                    let _ = token_tx.send(InferenceEvent::Token(cached.clone()));
                }
                return Ok(CacheLookup {
                    text: cached,
                    hit: true,
                    source: debug_log::ResponseSource::ExactCache,
                    elapsed_ms: 0,
                });
            }
            Ok(None) => {}
            Err(e) => {
                warn!(error = %e, "exact cache lookup failed");
            }
        }

        if matches!(cache_mode, CacheMode::PreferPromptLevel) {
            if let Some((system_prompt, user_prompt)) = prompt_level_cache_key(messages) {
                match cache.get_prompt_level(
                    &backend.name(),
                    system_prompt,
                    user_prompt,
                    cache_scope,
                ) {
                    Ok(Some(cached)) => {
                        let saved = estimate_message_tokens(messages)
                            .saturating_add(estimate_text_tokens(&cached));
                        cache_stats.hits = cache_stats.hits.saturating_add(1);
                        cache_stats.tokens_saved = cache_stats.tokens_saved.saturating_add(saved);
                        info!(saved_tokens = saved, "prompt-level cache hit");
                        emit_cache_update(cache_stats, true, &token_tx);
                        if stream_tokens {
                            let _ = token_tx.send(InferenceEvent::Token(cached.clone()));
                        }
                        return Ok(CacheLookup {
                            text: cached,
                            hit: true,
                            source: debug_log::ResponseSource::PromptCache,
                            elapsed_ms: 0,
                        });
                    }
                    Ok(None) => {}
                    Err(e) => {
                        warn!(error = %e, "prompt-level cache lookup failed");
                    }
                }

                match cache.get_semantic_prompt_level(
                    &backend.name(),
                    system_prompt,
                    user_prompt,
                    cache_scope,
                ) {
                    Ok(Some(cached)) => {
                        let saved = estimate_message_tokens(messages)
                            .saturating_add(estimate_text_tokens(&cached));
                        cache_stats.hits = cache_stats.hits.saturating_add(1);
                        cache_stats.tokens_saved = cache_stats.tokens_saved.saturating_add(saved);
                        info!(saved_tokens = saved, "semantic prompt cache hit");
                        emit_cache_update(cache_stats, true, &token_tx);
                        if stream_tokens {
                            let _ = token_tx.send(InferenceEvent::Token(cached.clone()));
                        }
                        return Ok(CacheLookup {
                            text: cached,
                            hit: true,
                            source: debug_log::ResponseSource::SemanticCache,
                            elapsed_ms: 0,
                        });
                    }
                    Ok(None) => {}
                    Err(e) => {
                        warn!(error = %e, "semantic prompt cache lookup failed");
                    }
                }
            }
        }
    }

    cache_stats.misses = cache_stats.misses.saturating_add(1);
    emit_cache_update(cache_stats, false, &token_tx);

    let gen_start = std::time::Instant::now();
    let response = run_and_collect(backend, messages, token_tx.clone(), stream_tokens)?;
    let elapsed_ms = gen_start.elapsed().as_millis() as u64;
    Ok(CacheLookup {
        text: response,
        hit: false,
        source: debug_log::ResponseSource::Live,
        elapsed_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_level_cache_key_allows_short_plain_chat() {
        let messages = vec![
            Message::system("system"),
            Message::user("What is a pointer?"),
            Message::assistant("A pointer stores an address."),
            Message::user("What is a pointer?"),
        ];

        let key = prompt_level_cache_key(&messages);

        assert_eq!(key, Some(("system", "What is a pointer?")));
    }

    #[test]
    fn prompt_level_cache_key_rejects_tool_context_sessions() {
        let messages = vec![
            Message::system("system"),
            Message::user("I've loaded this file for context:\n\nfn main() {}"),
            Message::assistant("Looks good."),
            Message::user("What does it do?"),
        ];

        assert!(prompt_level_cache_key(&messages).is_none());
    }

    #[test]
    fn prompt_level_cache_key_allows_longer_plain_chat() {
        let messages = vec![
            Message::system("system"),
            Message::user("What is a pointer?"),
            Message::assistant("A pointer stores an address."),
            Message::user("Explain ownership."),
            Message::assistant("Ownership controls cleanup."),
            Message::user("Whats a pointer?"),
        ];

        let key = prompt_level_cache_key(&messages);

        assert_eq!(key, Some(("system", "Whats a pointer?")));
    }
}
