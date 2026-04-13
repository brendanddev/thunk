#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ObservedLine {
    pub(super) path: String,
    pub(super) line_number: usize,
    pub(super) line_text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ObservedStepKind {
    EntryCall,
    Definition,
    Branch,
    Return,
    Delegation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ObservedStep {
    pub(super) path: String,
    pub(super) line_number: usize,
    pub(super) line_text: String,
    pub(super) step_kind: ObservedStepKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RepoOverviewEvidence {
    pub(super) package_line: Option<ObservedLine>,
    pub(super) readme_line: Option<ObservedLine>,
    pub(super) entrypoint_line: Option<ObservedLine>,
    pub(super) subsystem_lines: Vec<ObservedLine>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FileSummaryEvidence {
    pub(super) path: String,
    pub(super) declarations: Vec<ObservedLine>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ImplementationEvidence {
    pub(super) primary: ObservedLine,
    pub(super) supporting: Vec<ObservedLine>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ConfigEvidence {
    pub(super) lines: Vec<ObservedLine>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CallSiteEvidence {
    pub(super) symbol: String,
    pub(super) sites: Vec<ObservedLine>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct UsageEvidence {
    pub(super) symbol: String,
    pub(super) usages: Vec<ObservedLine>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FlowTraceEvidence {
    pub(super) subject: String,
    pub(super) steps: Vec<ObservedStep>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum StructuredEvidence {
    RepoOverview(RepoOverviewEvidence),
    FileSummary(FileSummaryEvidence),
    Implementation(ImplementationEvidence),
    Config(ConfigEvidence),
    CallSites(CallSiteEvidence),
    Usages(UsageEvidence),
    FlowTrace(FlowTraceEvidence),
}

pub(crate) enum InvestigationOutcome {
    NeedsMore {
        required_next_step: String,
    },
    Ready {
        evidence: StructuredEvidence,
        stop_reason: &'static str,
    },
    Insufficient {
        reason: String,
    },
}

pub(crate) type InvestigationReadiness = InvestigationOutcome;
