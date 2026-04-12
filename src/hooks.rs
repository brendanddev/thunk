use tracing::debug;

#[derive(Debug)]
pub enum HookEvent {
    BeforeGeneration {
        backend: String,
        message_count: usize,
        eco: bool,
        reflection: bool,
    },

    AfterGeneration {
        backend: String,
        response_chars: usize,
        from_cache: bool,
        elapsed_ms: u64,
    },

    ToolExecuted {
        tool_name: String,
        argument_chars: usize,
        result_chars: usize,
    },

    ApprovalRequested {
        tool_name: String,
        kind: String,
    },

    ApprovalResolved {
        tool_name: String,
        approved: bool,
    },

    InspectionEvaluated {
        operation: String,
        decision: String,
        risk: String,
        target_count: usize,
        blocked_reason_count: usize,
    },

    SessionRestored {
        message_count: usize,
        saved_at: u64,
    },

    SessionCreated {
        session_id: String,
        named: bool,
    },

    SessionResumed {
        session_id: String,
        named: bool,
        message_count: usize,
    },

    SessionRenamed {
        session_id: String,
        named: bool,
    },

    SessionExported {
        session_id: String,
        format: String,
    },

    SessionDeleted {
        session_id: String,
        was_active: bool,
    },

    SessionCleared {
        session_id: String,
    },

    SessionEnding {
        message_count: usize,
    },

    MemoryConsolidated {
        facts_pruned: usize,
        facts_deduped: usize,
        facts_capped: usize,
    },

    MemoryFactsLoaded {
        fact_count: usize,
    },

    MemorySummariesSelected {
        summary_count: usize,
    },

    MemoryUpdateEvaluated {
        accepted_count: usize,
        skipped_count: usize,
        duplicate_count: usize,
    },
}

pub trait Hook: Send {
    fn on_event(&self, event: &HookEvent);
}

pub struct Hooks {
    hooks: Vec<Box<dyn Hook>>,
}

impl Hooks {
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    pub fn register(&mut self, hook: Box<dyn Hook>) {
        self.hooks.push(hook);
    }

    pub fn dispatch(&self, event: HookEvent) {
        for hook in &self.hooks {
            hook.on_event(&event);
        }
    }
}

impl Default for Hooks {
    fn default() -> Self {
        let mut hooks = Self::new();
        hooks.register(Box::new(StructuralLogHook));
        hooks
    }
}

struct StructuralLogHook;

impl Hook for StructuralLogHook {
    fn on_event(&self, event: &HookEvent) {
        match event {
            HookEvent::BeforeGeneration {
                backend,
                message_count,
                eco,
                reflection,
            } => {
                debug!(
                    backend,
                    message_count, eco, reflection, "hook.before_generation"
                );
            }
            HookEvent::AfterGeneration {
                backend,
                response_chars,
                from_cache,
                elapsed_ms,
            } => {
                debug!(
                    backend,
                    response_chars, from_cache, elapsed_ms, "hook.after_generation"
                );
            }
            HookEvent::ToolExecuted {
                tool_name,
                argument_chars,
                result_chars,
            } => {
                debug!(
                    tool_name,
                    argument_chars, result_chars, "hook.tool_executed"
                );
            }
            HookEvent::ApprovalRequested { tool_name, kind } => {
                debug!(tool_name, kind, "hook.approval_requested");
            }
            HookEvent::ApprovalResolved {
                tool_name,
                approved,
            } => {
                debug!(tool_name, approved, "hook.approval_resolved");
            }
            HookEvent::InspectionEvaluated {
                operation,
                decision,
                risk,
                target_count,
                blocked_reason_count,
            } => {
                debug!(
                    operation,
                    decision, risk, target_count, blocked_reason_count, "hook.inspection_evaluated"
                );
            }
            HookEvent::SessionRestored {
                message_count,
                saved_at,
            } => {
                debug!(message_count, saved_at, "hook.session_restored");
            }
            HookEvent::SessionCreated { session_id, named } => {
                debug!(session_id, named, "hook.session_created");
            }
            HookEvent::SessionResumed {
                session_id,
                named,
                message_count,
            } => {
                debug!(session_id, named, message_count, "hook.session_resumed");
            }
            HookEvent::SessionRenamed { session_id, named } => {
                debug!(session_id, named, "hook.session_renamed");
            }
            HookEvent::SessionExported { session_id, format } => {
                debug!(session_id, format, "hook.session_exported");
            }
            HookEvent::SessionDeleted {
                session_id,
                was_active,
            } => {
                debug!(session_id, was_active, "hook.session_deleted");
            }
            HookEvent::SessionCleared { session_id } => {
                debug!(session_id, "hook.session_cleared");
            }
            HookEvent::SessionEnding { message_count } => {
                debug!(message_count, "hook.session_ending");
            }
            HookEvent::MemoryConsolidated {
                facts_pruned,
                facts_deduped,
                facts_capped,
            } => {
                debug!(
                    facts_pruned,
                    facts_deduped, facts_capped, "hook.memory_consolidated"
                );
            }
            HookEvent::MemoryFactsLoaded { fact_count } => {
                debug!(fact_count, "hook.memory_facts_loaded");
            }
            HookEvent::MemorySummariesSelected { summary_count } => {
                debug!(summary_count, "hook.memory_summaries_selected");
            }
            HookEvent::MemoryUpdateEvaluated {
                accepted_count,
                skipped_count,
                duplicate_count,
            } => {
                debug!(
                    accepted_count,
                    skipped_count, duplicate_count, "hook.memory_update_evaluated"
                );
            }
        }
    }
}
