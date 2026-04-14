#[path = "parse.rs"]
mod parse;

mod bootstrap;
pub(super) mod guidance;
pub(super) mod observe;
mod readiness;
mod render;
mod types;

pub(super) use bootstrap::bootstrap_tool_results;
pub(super) use guidance::{format_tool_loop_results_with_limit, grounded_answer_guidance};
pub(super) use observe::observed_read_paths;
pub(super) use readiness::{
    has_relevant_file_evidence, investigation_outcome, investigation_readiness,
    targeted_investigation_followup,
};
pub(super) use render::render_structured_answer;
pub(super) use types::{
    CallSiteEvidence, ConfigEvidence, FileSummaryEvidence, FlowTraceEvidence,
    ImplementationEvidence, InvestigationOutcome, InvestigationReadiness, ObservedLine,
    ObservedStep, ObservedStepKind, RepoOverviewEvidence, StructuredEvidence, UsageEvidence,
};

#[cfg(test)]
mod tests;
