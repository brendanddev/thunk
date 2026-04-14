#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ObservedLine {
    pub(super) path: String,
    pub(super) line_number: usize,
    pub(super) line_text: String,
}

impl ObservedLine {
    pub(crate) fn path(&self) -> &str {
        &self.path
    }
    pub(crate) fn line_number(&self) -> usize {
        self.line_number
    }
    pub(crate) fn line_text(&self) -> &str {
        &self.line_text
    }
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

impl ObservedStep {
    pub(crate) fn path(&self) -> &str {
        &self.path
    }
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

impl FileSummaryEvidence {
    pub(crate) fn path(&self) -> &str {
        &self.path
    }
    pub(crate) fn declarations(&self) -> &[ObservedLine] {
        &self.declarations
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ImplementationEvidence {
    pub(super) primary: ObservedLine,
    pub(super) supporting: Vec<ObservedLine>,
}

impl ImplementationEvidence {
    pub(crate) fn primary(&self) -> &ObservedLine {
        &self.primary
    }
    pub(crate) fn supporting(&self) -> &[ObservedLine] {
        &self.supporting
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ConfigEvidence {
    pub(super) lines: Vec<ObservedLine>,
}

impl ConfigEvidence {
    pub(crate) fn lines(&self) -> &[ObservedLine] {
        &self.lines
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CallSiteEvidence {
    pub(super) symbol: String,
    pub(super) sites: Vec<ObservedLine>,
}

impl CallSiteEvidence {
    pub(crate) fn sites(&self) -> &[ObservedLine] {
        &self.sites
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct UsageEvidence {
    pub(super) symbol: String,
    pub(super) usages: Vec<ObservedLine>,
}

impl UsageEvidence {
    pub(crate) fn usages(&self) -> &[ObservedLine] {
        &self.usages
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FlowTraceEvidence {
    pub(super) subject: String,
    pub(super) steps: Vec<ObservedStep>,
}

impl FlowTraceEvidence {
    pub(crate) fn steps(&self) -> &[ObservedStep] {
        &self.steps
    }
    pub(crate) fn subject(&self) -> &str {
        &self.subject
    }
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

impl StructuredEvidence {
    pub(crate) fn paths(&self) -> Vec<&str> {
        match self {
            StructuredEvidence::FileSummary(fe) => vec![fe.path()],
            StructuredEvidence::Implementation(ie) => {
                let mut paths = vec![ie.primary().path()];
                for line in ie.supporting() {
                    paths.push(line.path());
                }
                paths
            }
            StructuredEvidence::Config(ce) => ce.lines().iter().map(|l| l.path()).collect(),
            StructuredEvidence::CallSites(cse) => cse.sites().iter().map(|l| l.path()).collect(),
            StructuredEvidence::Usages(ue) => ue.usages().iter().map(|l| l.path()).collect(),
            StructuredEvidence::FlowTrace(fte) => fte.steps().iter().map(|s| s.path()).collect(),
            StructuredEvidence::RepoOverview(_) => vec![],
        }
    }
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
