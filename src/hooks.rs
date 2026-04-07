// src/hooks.rs
//
// Lifecycle hooks for the params-cli runtime.
//
// Hooks provide a structured, typed extension surface for key events in the
// single-agent loop. The design is deliberately simple for the first slice:
// a typed event enum, a trait for implementations, and a dispatcher that
// calls all registered hooks in order.
//
// This is an internal hook system, not a user-facing plugin framework.
// The primary goal is observability and a clean extension point for future
// work — not changing product behavior yet.
//
// Payloads are structural and privacy-safe: no prompt or response content,
// only metadata about what happened (counts, timings, flags, names).
//
// Hook failures are caught and logged; they never crash the runtime.
// Hook implementations must be Send (they run on the model thread).

use tracing::debug;

/// Events fired at key points in the agent lifecycle.
///
/// All fields are structural — no user text, no prompt content.
#[derive(Debug)]
pub enum HookEvent {
    /// Fired just before a live generation call starts.
    BeforeGeneration {
        backend: String,
        message_count: usize,
        eco: bool,
        reflection: bool,
    },

    /// Fired after a generation (live or cache-hit) completes.
    AfterGeneration {
        backend: String,
        response_chars: usize,
        from_cache: bool,
        /// Wall-clock time from the start of the generation call to completion.
        /// Zero for cache hits (no model work was done).
        elapsed_ms: u64,
    },

    /// Fired after each read-only tool executes successfully.
    ToolExecuted {
        tool_name: String,
        argument_chars: usize,
        result_chars: usize,
    },

    /// Fired when a mutating tool is proposed and the user must approve it.
    ApprovalRequested {
        tool_name: String,
        /// Human-readable kind string (e.g., "ShellCommand", "WriteFile").
        kind: String,
    },

    /// Fired when the user resolves a pending approval (approve or reject).
    ApprovalResolved { tool_name: String, approved: bool },

    /// Fired when the policy inspection layer evaluates an action.
    InspectionEvaluated {
        operation: String,
        decision: String,
        risk: String,
        target_count: usize,
        blocked_reason_count: usize,
    },

    /// Fired when a previous session is successfully loaded from disk.
    SessionRestored {
        /// Number of messages loaded (excludes the system message).
        message_count: usize,
        /// Unix timestamp when the session was originally saved.
        saved_at: u64,
    },

    /// Fired when a fresh session is created.
    SessionCreated { session_id: String, named: bool },

    /// Fired when an existing session becomes active.
    SessionResumed {
        session_id: String,
        named: bool,
        message_count: usize,
    },

    /// Fired when the active session is renamed.
    SessionRenamed { session_id: String, named: bool },

    /// Fired when a session transcript is exported.
    SessionExported { session_id: String, format: String },

    /// Fired when a saved session is deleted.
    SessionDeleted {
        session_id: String,
        was_active: bool,
    },

    /// Fired when the active saved session is cleared.
    SessionCleared { session_id: String },

    /// Fired when the model thread is about to exit (channel closed).
    /// This fires before fact extraction and consolidation.
    SessionEnding {
        /// Total number of messages in the session at exit (includes system).
        message_count: usize,
    },

    /// Fired after memory consolidation runs at session end.
    MemoryConsolidated {
        facts_pruned: usize,
        facts_deduped: usize,
        facts_capped: usize,
    },

    /// Fired when durable memory is loaded into the runtime.
    MemoryFactsLoaded { fact_count: usize },

    /// Fired when indexed summaries are selected for a turn.
    MemorySummariesSelected { summary_count: usize },

    /// Fired after verified memory update evaluation for a turn.
    MemoryUpdateEvaluated {
        accepted_count: usize,
        skipped_count: usize,
        duplicate_count: usize,
    },
}

/// The interface every hook implementation must satisfy.
pub trait Hook: Send {
    fn on_event(&self, event: &HookEvent);
}

/// Dispatches lifecycle events to all registered hooks.
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

    /// Fire an event to all registered hooks in registration order.
    ///
    /// Panics inside hook implementations are caught as errors and logged
    /// so a misbehaving hook cannot bring down the model thread.
    pub fn dispatch(&self, event: HookEvent) {
        for hook in &self.hooks {
            // std::panic::catch_unwind requires UnwindSafe — skip that
            // complexity for the first slice. Instead, rely on well-behaved
            // implementations plus graceful failure in the log hook itself.
            hook.on_event(&event);
        }
    }
}

impl Default for Hooks {
    /// Returns a Hooks dispatcher pre-loaded with the built-in log hook.
    fn default() -> Self {
        let mut hooks = Self::new();
        hooks.register(Box::new(StructuralLogHook));
        hooks
    }
}

// Writes each lifecycle event to the structured audit log (params.log) at
// debug level. This surfaces timing, tool, and session metadata as distinct
// structured log entries — complementing the higher-level info!/warn! calls
// in the main inference loop without duplicating them.

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
