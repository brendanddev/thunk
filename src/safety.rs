// src/safety.rs
//
// Policy-layer sandbox and request inspection.

use std::net::IpAddr;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::config;
use crate::error::{ParamsError, Result};

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
    pub resolved_path: PathBuf,
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

pub fn project_root() -> Result<PathBuf> {
    std::env::current_dir()?
        .canonicalize()
        .map_err(ParamsError::Io)
}

pub fn display_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .ok()
        .and_then(|p| p.to_str())
        .map(|p| p.to_string())
        .unwrap_or_else(|| path.display().to_string())
}

pub fn inspect_project_path(
    operation: &str,
    requested: &str,
    kind: ProjectPathKind,
    allow_missing: bool,
) -> Result<PathInspection> {
    let cfg = config::load_with_profile()?;
    let root = project_root()?;

    if !cfg.safety.enabled {
        let fallback = resolve_path_without_scope(&root, requested, allow_missing)?;
        let exists = fallback.exists();
        let report = InspectionReport {
            operation: operation.to_string(),
            decision: InspectionDecision::Allow,
            risk: RiskLevel::Low,
            summary: format!("Safety disabled for {}", display_path(&root, &fallback)),
            reasons: Vec::new(),
            targets: vec![display_path(&root, &fallback)],
            segments: Vec::new(),
            network_targets: Vec::new(),
        };
        log_inspection(&report);
        return Ok(PathInspection {
            display_path: display_path(&root, &fallback),
            resolved_path: fallback,
            exists,
        });
    }

    match cfg.safety.read_scope {
        ReadScope::ProjectOnly => {}
    }

    let resolved = resolve_project_scoped_path(&root, requested, allow_missing)?;
    let exists = resolved.exists();
    if exists || !allow_missing {
        validate_path_kind(&resolved, kind, exists)?;
    }
    let display = display_path(&root, &resolved);
    let summary = match kind {
        ProjectPathKind::File => format!("Read access limited to project file {display}"),
        ProjectPathKind::Directory => {
            format!("Directory access limited to project path {display}")
        }
    };
    let report = InspectionReport {
        operation: operation.to_string(),
        decision: InspectionDecision::Allow,
        risk: RiskLevel::Low,
        summary,
        reasons: vec!["Path is inside the current project root".to_string()],
        targets: vec![display.clone()],
        segments: Vec::new(),
        network_targets: Vec::new(),
    };
    log_inspection(&report);

    Ok(PathInspection {
        resolved_path: resolved,
        display_path: display,
        exists,
    })
}

pub fn inspect_search_scope() -> Result<InspectionReport> {
    let cfg = config::load_with_profile()?;
    let root = project_root()?;
    let report = if cfg.safety.enabled {
        InspectionReport {
            operation: "search".to_string(),
            decision: InspectionDecision::Allow,
            risk: RiskLevel::Low,
            summary: "Search scope restricted to the current project root".to_string(),
            reasons: vec!["Search walks only files under the project root".to_string()],
            targets: vec![display_path(&root, &root)],
            segments: Vec::new(),
            network_targets: Vec::new(),
        }
    } else {
        InspectionReport {
            operation: "search".to_string(),
            decision: InspectionDecision::Allow,
            risk: RiskLevel::Low,
            summary: "Safety disabled for project search".to_string(),
            reasons: Vec::new(),
            targets: vec![display_path(&root, &root)],
            segments: Vec::new(),
            network_targets: Vec::new(),
        }
    };
    log_inspection(&report);
    Ok(report)
}

pub fn inspect_git_operation(subcommand: &str) -> Result<InspectionReport> {
    let root = project_root()?;
    let report = InspectionReport {
        operation: "git".to_string(),
        decision: InspectionDecision::Allow,
        risk: RiskLevel::Low,
        summary: format!("Read-only git context via `{subcommand}`"),
        reasons: vec!["Git commands run in the current project root".to_string()],
        targets: vec![display_path(&root, &root)],
        segments: Vec::new(),
        network_targets: Vec::new(),
    };
    log_inspection(&report);
    Ok(report)
}

pub fn inspect_write_target(display_path: &str, exists: bool) -> Result<InspectionReport> {
    let cfg = config::load_with_profile()?;
    let reasons = if exists {
        vec!["Existing file will be overwritten after approval".to_string()]
    } else {
        vec!["New file will be created after approval".to_string()]
    };
    let report = InspectionReport {
        operation: "write_file".to_string(),
        decision: if cfg.safety.enabled {
            InspectionDecision::NeedsApproval
        } else {
            InspectionDecision::NeedsApproval
        },
        risk: if exists {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        },
        summary: if exists {
            format!("Approve overwrite of {display_path}")
        } else {
            format!("Approve creation of {display_path}")
        },
        reasons,
        targets: vec![display_path.to_string()],
        segments: Vec::new(),
        network_targets: Vec::new(),
    };
    log_inspection(&report);
    Ok(report)
}

pub fn inspect_edit_target(
    display_path: &str,
    replacement_count: usize,
) -> Result<InspectionReport> {
    let cfg = config::load_with_profile()?;
    let replacement_label = if replacement_count == 1 {
        "1 replacement block".to_string()
    } else {
        format!("{replacement_count} replacement blocks")
    };
    let report = InspectionReport {
        operation: "edit_file".to_string(),
        decision: if cfg.safety.enabled {
            InspectionDecision::NeedsApproval
        } else {
            InspectionDecision::NeedsApproval
        },
        risk: RiskLevel::Medium,
        summary: format!("Approve targeted edit of {display_path}"),
        reasons: vec![format!(
            "{replacement_label} will be applied after approval"
        )],
        targets: vec![display_path.to_string()],
        segments: Vec::new(),
        network_targets: Vec::new(),
    };
    log_inspection(&report);
    Ok(report)
}

pub fn inspect_shell_command(command: &str) -> Result<InspectionReport> {
    let cfg = config::load_with_profile()?;
    let trimmed = command.trim();
    if trimmed.is_empty() {
        return Err(ParamsError::Config("Shell command cannot be empty".into()));
    }
    if trimmed.contains('\n') || trimmed.contains('\r') {
        return Err(ParamsError::Config(
            "Multiline shell commands are not supported".into(),
        ));
    }

    let segments = split_shell_segments(trimmed);
    let lowered = trimmed.to_ascii_lowercase();
    let mut reasons = Vec::new();
    let decision;
    let mut risk = if segments.len() > 1 || trimmed.contains('|') {
        RiskLevel::Medium
    } else {
        RiskLevel::Low
    };

    if cfg.safety.enabled && cfg.safety.block_destructive_shell {
        if lowered.starts_with("sudo ") {
            reasons.push("sudo commands are blocked".to_string());
        }
        if contains_destructive_rm(&lowered) {
            reasons.push("destructive rm target is blocked".to_string());
        }
        if contains_token(
            &lowered,
            &["mkfs", "fdisk", "shutdown", "reboot", "halt", "poweroff"],
        ) {
            reasons.push("destructive system command is blocked".to_string());
        }
        if lowered.contains("diskutil erase") {
            reasons.push("disk erase commands are blocked".to_string());
        }
        if lowered.contains("dd ") && lowered.contains(" of=/dev/") {
            reasons.push("writing raw device output is blocked".to_string());
        }
        if looks_like_pipe_to_shell(&lowered) {
            reasons.push("pipe-to-shell execution is blocked".to_string());
        }
    }

    if cfg.safety.enabled {
        if matches_any_shell_pattern(&segments, &cfg.safety.shell_denylist) {
            reasons.push("command matched configured shell denylist".to_string());
        }
        if !cfg.safety.shell_allowlist.is_empty()
            && !matches_all_shell_segments(&segments, &cfg.safety.shell_allowlist)
        {
            reasons.push("command does not match configured shell allowlist".to_string());
        }
    }

    if !reasons.is_empty() {
        decision = InspectionDecision::Block;
        risk = RiskLevel::High;
    } else if cfg.safety.enabled {
        match cfg.safety.shell_mode {
            ShellMode::ApproveInspect => {
                decision = InspectionDecision::NeedsApproval;
            }
        }
        if segments.len() > 1 || trimmed.contains('|') {
            risk = RiskLevel::Medium;
        }
    } else {
        decision = InspectionDecision::NeedsApproval;
    }

    let summary = if matches!(decision, InspectionDecision::Block) {
        "Shell command blocked by safety policy".to_string()
    } else {
        "Shell command requires approval before execution".to_string()
    };
    let report = InspectionReport {
        operation: "bash".to_string(),
        decision,
        risk,
        summary,
        reasons,
        targets: Vec::new(),
        segments,
        network_targets: Vec::new(),
    };
    log_inspection(&report);
    Ok(report)
}

pub fn inspect_fetch_url(raw: &str) -> Result<(String, InspectionReport)> {
    let cfg = config::load_with_profile()?;
    let url = normalize_url(raw)?;
    let (scheme, host) = parse_scheme_and_host(&url)?;
    let host = normalize_host(&host);
    let mut reasons = Vec::new();
    let mut decision = InspectionDecision::Allow;
    let mut risk = RiskLevel::Low;

    if cfg.safety.enabled && cfg.safety.inspect_network {
        if cfg.safety.block_private_network && is_blocked_network_host(&host) {
            decision = InspectionDecision::Block;
            risk = RiskLevel::High;
            reasons.push(
                "private, loopback, link-local, or localhost targets are blocked".to_string(),
            );
        } else if !cfg.safety.network_allowlist.is_empty()
            && !host_matches_allowlist(&host, &cfg.safety.network_allowlist)
        {
            decision = InspectionDecision::Block;
            risk = RiskLevel::High;
            reasons.push("host is not in the configured network allowlist".to_string());
        } else {
            reasons
                .push("Outbound fetch is limited to explicit public http/https URLs".to_string());
        }
    }

    let summary = if matches!(decision, InspectionDecision::Block) {
        format!("Blocked network fetch to {host}")
    } else {
        format!("Fetching {scheme} URL from {host}")
    };
    let report = InspectionReport {
        operation: "fetch_url".to_string(),
        decision,
        risk,
        summary,
        reasons,
        targets: vec![url.clone()],
        segments: Vec::new(),
        network_targets: vec![host],
    };
    log_inspection(&report);
    Ok((url, report))
}

pub fn inspect_provider_request(
    operation: &str,
    base_url: &str,
    payload_chars: usize,
) -> Result<(String, InspectionReport)> {
    let cfg = config::load_with_profile()?;
    let url = normalize_url(base_url)?;
    let (scheme, host) = parse_scheme_and_host(&url)?;
    let host = normalize_host(&host);
    let mut reasons = Vec::new();
    let mut decision = InspectionDecision::Allow;
    let mut risk = if payload_chars > 20_000 {
        RiskLevel::Medium
    } else {
        RiskLevel::Low
    };

    if cfg.safety.enabled && cfg.safety.inspect_cloud_requests {
        if !cfg.safety.network_allowlist.is_empty()
            && !host_matches_allowlist(&host, &cfg.safety.network_allowlist)
        {
            decision = InspectionDecision::Block;
            risk = RiskLevel::High;
            reasons.push("provider host is not in the configured network allowlist".to_string());
        } else {
            reasons.push(format!(
                "Outbound provider request inspected before send ({payload_chars} chars)"
            ));
        }
    }

    let summary = if matches!(decision, InspectionDecision::Block) {
        format!("Blocked provider request to {host}")
    } else {
        format!("Sending {scheme} provider request to {host}")
    };
    let report = InspectionReport {
        operation: operation.to_string(),
        decision,
        risk,
        summary,
        reasons,
        targets: vec![url.clone()],
        segments: Vec::new(),
        network_targets: vec![host],
    };
    log_inspection(&report);
    Ok((url, report))
}

pub fn normalize_url(raw: &str) -> Result<String> {
    let url = raw.trim();
    if url.is_empty() {
        return Err(ParamsError::Config("URL cannot be empty".to_string()));
    }
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err(ParamsError::Config(
            "Only absolute http:// or https:// URLs are supported".to_string(),
        ));
    }
    if url.contains(char::is_whitespace) {
        return Err(ParamsError::Config(
            "URL cannot contain whitespace".to_string(),
        ));
    }
    Ok(url.to_string())
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

fn resolve_path_without_scope(
    root: &Path,
    requested: &str,
    allow_missing: bool,
) -> Result<PathBuf> {
    let candidate = if Path::new(requested).is_absolute() {
        PathBuf::from(requested)
    } else {
        root.join(requested)
    };
    if candidate.exists() || !allow_missing {
        candidate
            .canonicalize()
            .map_err(|_| ParamsError::Config(format!("Path not found: {}", candidate.display())))
    } else {
        Ok(normalize_path(candidate))
    }
}

fn resolve_project_scoped_path(
    root: &Path,
    requested: &str,
    allow_missing: bool,
) -> Result<PathBuf> {
    let candidate = if Path::new(requested).is_absolute() {
        PathBuf::from(requested)
    } else {
        root.join(requested)
    };

    let normalized = normalize_path(candidate.clone());
    if !normalized.starts_with(root) {
        return Err(ParamsError::Config(
            "Path must stay within the current project".to_string(),
        ));
    }

    if candidate.exists() {
        let canonical = candidate.canonicalize()?;
        if !canonical.starts_with(root) {
            return Err(ParamsError::Config(
                "Path must stay within the current project".to_string(),
            ));
        }
        return Ok(canonical);
    }

    if !allow_missing {
        return Err(ParamsError::Config(format!(
            "Path not found: {}",
            candidate.display()
        )));
    }

    if let Some(parent) = normalized.parent() {
        if parent.exists() {
            let canonical_parent = parent.canonicalize()?;
            if !canonical_parent.starts_with(root) {
                return Err(ParamsError::Config(
                    "Path must stay within the current project".to_string(),
                ));
            }
        }
    }

    Ok(normalized)
}

fn validate_path_kind(path: &Path, kind: ProjectPathKind, exists: bool) -> Result<()> {
    if !exists {
        return Err(ParamsError::Config(format!(
            "Path not found: {}",
            path.display()
        )));
    }

    match kind {
        ProjectPathKind::File if !path.is_file() => Err(ParamsError::Config(format!(
            "{} is a directory, not a file",
            path.display()
        ))),
        ProjectPathKind::Directory if !path.is_dir() => Err(ParamsError::Config(format!(
            "{} is a file, not a directory",
            path.display()
        ))),
        _ => Ok(()),
    }
}

fn normalize_path(path: PathBuf) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Normal(part) => normalized.push(part),
            Component::RootDir => normalized.push(Path::new("/")),
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
        }
    }
    normalized
}

fn split_shell_segments(command: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let bytes = command.as_bytes();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let matched = if i + 1 < bytes.len() && bytes[i] == b'&' && bytes[i + 1] == b'&' {
            Some(2)
        } else if i + 1 < bytes.len() && bytes[i] == b'|' && bytes[i + 1] == b'|' {
            Some(2)
        } else if matches!(bytes[i], b'|' | b';') {
            Some(1)
        } else {
            None
        };

        if let Some(len) = matched {
            let segment = command[start..i].trim();
            if !segment.is_empty() {
                segments.push(segment.to_string());
            }
            i += len;
            start = i;
        } else {
            i += 1;
        }
    }

    let tail = command[start..].trim();
    if !tail.is_empty() {
        segments.push(tail.to_string());
    }
    if segments.is_empty() {
        segments.push(command.trim().to_string());
    }
    segments
}

fn contains_token(haystack: &str, tokens: &[&str]) -> bool {
    tokens.iter().any(|token| haystack.contains(token))
}

fn contains_destructive_rm(command: &str) -> bool {
    let patterns = [
        "rm -rf /",
        "rm -rf ~",
        "rm -rf /*",
        "rm -fr /",
        "rm -fr ~",
        "rm -fr /*",
    ];
    patterns.iter().any(|pattern| command.contains(pattern))
}

fn normalize_host(host: &str) -> String {
    host.trim().trim_matches('.').to_ascii_lowercase()
}

fn normalize_policy_entry(entry: &str) -> Option<String> {
    let trimmed = entry.trim().trim_matches('.').to_ascii_lowercase();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn matches_shell_prefix(segment: &str, pattern: &str) -> bool {
    let segment = segment.trim().to_ascii_lowercase();
    let Some(pattern) = normalize_policy_entry(pattern) else {
        return false;
    };
    segment == pattern || segment.starts_with(&pattern)
}

fn matches_any_shell_pattern(segments: &[String], patterns: &[String]) -> bool {
    segments.iter().any(|segment| {
        patterns
            .iter()
            .any(|pattern| matches_shell_prefix(segment, pattern))
    })
}

fn matches_all_shell_segments(segments: &[String], patterns: &[String]) -> bool {
    segments.iter().all(|segment| {
        patterns
            .iter()
            .any(|pattern| matches_shell_prefix(segment, pattern))
    })
}

fn host_matches_allowlist(host: &str, allowlist: &[String]) -> bool {
    let host = normalize_host(host);
    allowlist.iter().any(|entry| {
        let Some(entry) = normalize_policy_entry(entry) else {
            return false;
        };
        host == entry || host.ends_with(&format!(".{entry}"))
    })
}

fn looks_like_pipe_to_shell(command: &str) -> bool {
    (command.contains("curl ") || command.contains("wget "))
        && command.contains("|")
        && (command.contains("| sh") || command.contains("| bash") || command.contains("| zsh"))
}

fn parse_scheme_and_host(url: &str) -> Result<(String, String)> {
    let (scheme, rest) = url
        .split_once("://")
        .ok_or_else(|| ParamsError::Config("Invalid URL".to_string()))?;
    let authority = rest
        .split(['/', '?', '#'])
        .next()
        .ok_or_else(|| ParamsError::Config("Invalid URL".to_string()))?;
    if authority.is_empty() {
        return Err(ParamsError::Config("URL host cannot be empty".to_string()));
    }

    let host = if authority.starts_with('[') {
        authority
            .split(']')
            .next()
            .map(|value| value.trim_start_matches('[').to_string())
            .ok_or_else(|| ParamsError::Config("Invalid IPv6 host".to_string()))?
    } else {
        authority.split(':').next().unwrap_or(authority).to_string()
    };

    if host.is_empty() {
        return Err(ParamsError::Config("URL host cannot be empty".to_string()));
    }

    Ok((scheme.to_string(), host))
}

fn is_blocked_network_host(host: &str) -> bool {
    let lowered = host.to_ascii_lowercase();
    if lowered == "localhost" {
        return true;
    }

    if let Ok(ip) = host.parse::<IpAddr>() {
        return match ip {
            IpAddr::V4(v4) => {
                v4.is_loopback() || v4.is_private() || v4.is_link_local() || v4.octets()[0] == 0
            }
            IpAddr::V6(v6) => v6.is_loopback() || v6.is_unicast_link_local(),
        };
    }

    false
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn temp_project_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("params-safety-test-{label}-{nonce}"));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn with_temp_project<F: FnOnce(PathBuf)>(label: &str, f: F) {
        let _guard = test_cwd_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let root = temp_project_dir(label);
        fs::create_dir_all(root.join(".local")).expect("local dir");
        let original = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&root).expect("set cwd");
        f(root.clone());
        std::env::set_current_dir(original).expect("restore cwd");
    }

    #[test]
    fn blocks_private_ipv4_hosts() {
        assert!(is_blocked_network_host("127.0.0.1"));
        assert!(is_blocked_network_host("10.0.0.5"));
        assert!(is_blocked_network_host("192.168.1.10"));
        assert!(is_blocked_network_host("172.16.0.8"));
        assert!(is_blocked_network_host("169.254.1.1"));
    }

    #[test]
    fn allows_public_hosts() {
        assert!(!is_blocked_network_host("example.com"));
        assert!(!is_blocked_network_host("8.8.8.8"));
    }

    #[test]
    fn shell_denylist_blocks_matching_segments() {
        with_temp_project("shell-deny", |_| {
            let mut cfg = crate::config::Config::default();
            cfg.safety.shell_denylist = vec!["cargo clippy".to_string()];
            fs::write(
                crate::config::config_path().unwrap(),
                toml::to_string(&cfg).unwrap(),
            )
            .unwrap();
            let report = inspect_shell_command("cargo clippy").expect("inspect");
            assert!(matches!(report.decision, InspectionDecision::Block));
        });
    }

    #[test]
    fn shell_allowlist_blocks_unmatched_commands_when_configured() {
        with_temp_project("shell-allow", |_| {
            let mut cfg = crate::config::Config::default();
            cfg.safety.shell_allowlist = vec!["cargo ".to_string()];
            fs::write(
                crate::config::config_path().unwrap(),
                toml::to_string(&cfg).unwrap(),
            )
            .unwrap();
            let report = inspect_shell_command("git status").expect("inspect");
            assert!(matches!(report.decision, InspectionDecision::Block));
        });
    }

    #[test]
    fn splits_shell_segments_on_control_operators() {
        assert_eq!(
            split_shell_segments("cargo check && cargo test | cat; git status"),
            vec!["cargo check", "cargo test", "cat", "git status"]
        );
    }

    #[test]
    fn detects_pipe_to_shell() {
        assert!(looks_like_pipe_to_shell("curl https://x | sh"));
        assert!(looks_like_pipe_to_shell("wget https://x | bash"));
        assert!(!looks_like_pipe_to_shell("curl https://x"));
    }

    #[test]
    fn shell_inspection_blocks_destructive_commands() {
        let report = inspect_shell_command("rm -rf /").expect("inspect");
        assert!(matches!(report.decision, InspectionDecision::Block));
        assert!(matches!(report.risk, RiskLevel::High));
    }

    #[test]
    fn shell_inspection_allows_benign_commands_with_approval() {
        let report = inspect_shell_command("cargo check").expect("inspect");
        assert!(matches!(report.decision, InspectionDecision::NeedsApproval));
    }

    #[test]
    fn fetch_inspection_blocks_loopback_targets() {
        let (_, report) = inspect_fetch_url("http://127.0.0.1:8080").expect("inspect");
        assert!(matches!(report.decision, InspectionDecision::Block));
    }

    #[test]
    fn fetch_allowlist_accepts_exact_and_subdomain_matches() {
        assert!(host_matches_allowlist(
            "api.openai.com",
            &[String::from("openai.com")]
        ));
        assert!(host_matches_allowlist(
            "openai.com",
            &[String::from("openai.com")]
        ));
        assert!(!host_matches_allowlist(
            "example.net",
            &[String::from("openai.com")]
        ));
    }

    #[test]
    fn provider_request_inspection_blocks_non_allowlisted_hosts() {
        with_temp_project("provider-allow", |_| {
            let mut cfg = crate::config::Config::default();
            cfg.safety.network_allowlist = vec!["api.openai.com".to_string()];
            fs::write(
                crate::config::config_path().unwrap(),
                toml::to_string(&cfg).unwrap(),
            )
            .unwrap();
            let (_, report) = inspect_provider_request(
                "openai_compat",
                "https://example.com/v1/chat/completions",
                1200,
            )
            .expect("inspect");
            assert!(matches!(report.decision, InspectionDecision::Block));
        });
    }

    #[test]
    fn project_path_rejects_parent_escape() {
        with_temp_project("escape", |_| {
            let result =
                inspect_project_path("read_file", "../outside.txt", ProjectPathKind::File, false);
            assert!(result.is_err());
        });
    }

    #[test]
    fn project_path_allows_files_inside_project() {
        with_temp_project("inside", |root| {
            let src = root.join("src");
            fs::create_dir_all(&src).expect("mkdir");
            let file = src.join("main.rs");
            fs::write(&file, "fn main() {}\n").expect("write");

            let result =
                inspect_project_path("read_file", "src/main.rs", ProjectPathKind::File, false)
                    .expect("inspect");

            assert_eq!(result.display_path, "src/main.rs");
        });
    }
}
