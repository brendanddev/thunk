use serde::{Deserialize, Serialize};
use tracing::{info, warn};

#[cfg(test)]
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReadScope {
    ProjectOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ShellMode {
    ApproveInspect,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InspectionDecision {
    Allow,
    NeedsApproval,
    Block,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone)]
pub struct InspectionReport {
    pub operation: String,
    pub decision: InspectionDecision,
    pub risk: RiskLevel,
    pub summary: String,
    pub reasons: Vec<String>,
    pub targets: Vec<String>,
    pub segments: Vec<String>,
    pub network_targets: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PathInspection {
    pub resolved_path: std::path::PathBuf,
    pub display_path: String,
    pub exists: bool,
}

#[cfg(test)]
pub fn test_cwd_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectPathKind {
    File,
    Directory,
}

impl Default for ReadScope {
    fn default() -> Self {
        Self::ProjectOnly
    }
}

impl Default for ShellMode {
    fn default() -> Self {
        Self::ApproveInspect
    }
}

impl std::fmt::Display for InspectionDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            Self::Allow => "allow",
            Self::NeedsApproval => "needs approval",
            Self::Block => "block",
        };
        write!(f, "{text}")
    }
}

impl std::fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        };
        write!(f, "{text}")
    }
}

impl InspectionReport {
    pub fn blocked_message(&self) -> String {
        if self.reasons.is_empty() {
            format!("Blocked {}: {}", self.operation, self.summary)
        } else {
            format!("Blocked {}: {}", self.operation, self.reasons.join("; "))
        }
    }
}

pub fn log_inspection(report: &InspectionReport) {
    if matches!(report.decision, InspectionDecision::Block) {
        warn!(
            operation = report.operation.as_str(),
            decision = report.decision.to_string(),
            risk = report.risk.to_string(),
            target_count = report.targets.len() + report.network_targets.len(),
            blocked_reason_count = report.reasons.len(),
            "inspection.evaluated"
        );
    } else {
        info!(
            operation = report.operation.as_str(),
            decision = report.decision.to_string(),
            risk = report.risk.to_string(),
            target_count = report.targets.len() + report.network_targets.len(),
            blocked_reason_count = report.reasons.len(),
            "inspection.evaluated"
        );
    }
}
