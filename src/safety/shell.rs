use crate::config;
use crate::error::{ParamsError, Result};

use super::policy::normalize_policy_entry;
use super::types::{log_inspection, InspectionDecision, InspectionReport, RiskLevel, ShellMode};

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

pub(crate) fn split_shell_segments(command: &str) -> Vec<String> {
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

pub(crate) fn looks_like_pipe_to_shell(command: &str) -> bool {
    (command.contains("curl ") || command.contains("wget "))
        && command.contains("|")
        && (command.contains("| sh") || command.contains("| bash") || command.contains("| zsh"))
}
