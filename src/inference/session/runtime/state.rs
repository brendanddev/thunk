use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};

use crate::cache::ExactCache;
use crate::config;
use crate::events::{InferenceEvent, MemorySessionExcerptView};
use crate::hooks::Hooks;
use crate::memory::{facts::FactStore, index::ProjectIndex};
use crate::session::{SessionStore, SessionSummary};
use crate::skills;
use crate::tools::ToolRegistry;

use super::super::super::approval::ApprovalContext;
use super::super::super::budget::{SessionBudget, SessionCacheStats};
use super::super::super::indexing::IncrementalIndexState;
use super::super::super::{build_system_prompt, InferenceBackend, Message, SessionCommand};
use super::super::investigation::InvestigationState;
use super::super::memory::{
    clear_memory_retrieval, emit_memory_state, memory_fact_lines, RuntimeMemoryState,
};
use super::super::support::reset_session_runtime;

#[derive(Clone, Copy, Default)]
pub struct SessionRuntimeOptions {
    pub no_resume: bool,
}

pub(super) struct RuntimeContext<'a> {
    pub(super) prompt_rx: &'a Receiver<SessionCommand>,
    pub(super) token_tx: &'a Sender<InferenceEvent>,
    pub(super) cfg: &'a config::Config,
    pub(super) backend: &'a dyn InferenceBackend,
    pub(super) hooks: &'a Hooks,
    pub(super) tools: &'a ToolRegistry,
    pub(super) project_root: &'a PathBuf,
    pub(super) project_name: &'a str,
    pub(super) exact_cache: Option<&'a ExactCache>,
    pub(super) fact_store: Option<&'a FactStore>,
    pub(super) project_index: Option<&'a ProjectIndex>,
    pub(super) session_store: Option<&'a SessionStore>,
}

pub(super) struct RuntimeState {
    pub(super) eco_enabled: bool,
    pub(super) reflection_requested: bool,
    pub(super) reflection_enabled: bool,
    pub(super) debug_logging_enabled: bool,
    pub(super) index_state: Option<IncrementalIndexState>,
    pub(super) memory_state: RuntimeMemoryState,
    pub(super) investigation_state: InvestigationState,
    pub(super) session_messages: Vec<Message>,
    pub(super) budget: SessionBudget,
    pub(super) cache_stats: SessionCacheStats,
    pub(super) next_action_id: u64,
    pub(super) active_session: Option<SessionSummary>,
}

impl RuntimeState {
    pub(super) fn reset_session_scope(&mut self, ctx: &RuntimeContext<'_>) {
        reset_session_runtime(
            &mut self.session_messages,
            ctx.tools,
            self.eco_enabled,
            &mut self.budget,
            &mut self.cache_stats,
            &ctx.cfg.backend,
            ctx.token_tx,
        );
        clear_memory_retrieval(&mut self.memory_state);
        self.memory_state.last_update = None;
        self.investigation_state.clear();
        emit_memory_state(ctx.token_tx, &self.memory_state);
    }

    pub(super) fn refresh_base_system_prompt(&mut self, ctx: &RuntimeContext<'_>) {
        if let Some(first) = self.session_messages.first_mut() {
            if first.role == "system" {
                first.content = build_system_prompt(
                    ctx.tools,
                    &memory_fact_lines(&self.memory_state.last_selected_facts),
                    &[],
                    &self.memory_state.last_selected_session_excerpts,
                    self.eco_enabled,
                );
            }
        }
    }

    pub(super) fn set_chat_system_prompt(
        &mut self,
        ctx: &RuntimeContext<'_>,
        user_prompt: &str,
        facts: &[String],
        summaries: &[(String, String)],
        session_excerpts: &[MemorySessionExcerptView],
    ) {
        if let Some(first) = self.session_messages.first_mut() {
            if first.role == "system" {
                first.content = build_chat_system_prompt(
                    ctx.project_root,
                    user_prompt,
                    ctx.tools,
                    facts,
                    summaries,
                    session_excerpts,
                    self.eco_enabled,
                );
            }
        }
    }
}

pub(super) fn build_chat_system_prompt(
    project_root: &PathBuf,
    user_prompt: &str,
    tools: &ToolRegistry,
    facts: &[String],
    summaries: &[(String, String)],
    session_excerpts: &[MemorySessionExcerptView],
    eco_enabled: bool,
) -> String {
    let mut prompt = build_system_prompt(tools, facts, summaries, session_excerpts, eco_enabled);
    skills::append_chat_skill_guidance(project_root, user_prompt, &mut prompt);
    prompt
}

pub(super) fn approval_context<'a>(
    ctx: &'a RuntimeContext<'a>,
    state: &'a mut RuntimeState,
    turn_memory: Option<&'a mut crate::memory::facts::TurnMemoryEvidence>,
) -> ApprovalContext<'a> {
    ApprovalContext {
        prompt_rx: ctx.prompt_rx,
        token_tx: ctx.token_tx,
        backend: ctx.backend,
        tools: ctx.tools,
        exact_cache: ctx.exact_cache,
        session_messages: &mut state.session_messages,
        cfg: ctx.cfg,
        project_root: ctx.project_root,
        budget: &mut state.budget,
        cache_stats: &mut state.cache_stats,
        debug_logging_enabled: state.debug_logging_enabled,
        reflection_enabled: state.reflection_enabled,
        eco_enabled: state.eco_enabled,
        hooks: ctx.hooks,
        index_state: state.index_state.as_mut(),
        turn_memory,
    }
}
