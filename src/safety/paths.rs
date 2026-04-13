use std::path::{Component, Path, PathBuf};

use crate::config;
use crate::error::{ParamsError, Result};

use super::types::{
    log_inspection, InspectionDecision, InspectionReport, PathInspection, ProjectPathKind,
    ReadScope, RiskLevel,
};

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
