use super::util::{preview_snippet, single_span_with_gutter, truncate, SYSTEM_GUTTER};
use super::{StyledLine, StyledSpan};
use crate::events::{PendingAction, PendingActionKind};
use crate::safety::RiskLevel;
use crate::tui::renderer::style::Theme;
use crate::tui::state::AppState;

pub(crate) fn build_approval(
    state: &AppState,
    theme: Theme,
    width: u16,
) -> Option<Vec<StyledLine>> {
    let pending = state.pending_action.as_ref()?;
    let approval_style = match pending.inspection.risk {
        RiskLevel::Low => theme.chip_accent(),
        RiskLevel::Medium => theme.chip_warning(),
        RiskLevel::High => theme.chip_danger(),
    };
    let mut lines = vec![StyledLine {
        spans: vec![
            StyledSpan {
                text: "! ".to_string(),
                style: approval_style,
            },
            StyledSpan {
                text: approval_kind_label(pending.kind.clone()).to_string(),
                style: approval_style,
            },
            StyledSpan {
                text: format!(
                    "  {}",
                    truncate(&pending.title, width.saturating_sub(10) as usize)
                ),
                style: theme.base(),
            },
        ],
    }];
    lines.push(single_span_with_gutter(
        SYSTEM_GUTTER,
        theme.dim(),
        &approval_summary_line(pending, width),
        theme.muted(),
    ));

    let preview_lines = approval_preview_line_cap(pending.kind.clone(), width, None);
    for line in preview_snippet(&pending.preview, width.saturating_sub(2), preview_lines) {
        lines.push(StyledLine {
            spans: vec![
                StyledSpan {
                    text: "│ ".to_string(),
                    style: theme.border(),
                },
                StyledSpan {
                    text: line,
                    style: theme.muted(),
                },
            ],
        });
    }
    lines.push(single_span_with_gutter(
        SYSTEM_GUTTER,
        theme.dim(),
        "^Y approve  ^N reject",
        theme.muted(),
    ));
    Some(lines)
}

fn approval_kind_label(kind: PendingActionKind) -> &'static str {
    match kind {
        PendingActionKind::ShellCommand => "shell",
        PendingActionKind::FileWrite => "write",
        PendingActionKind::FileEdit => "edit",
    }
}

fn approval_summary_line(pending: &PendingAction, width: u16) -> String {
    let detail = pending
        .inspection
        .reasons
        .first()
        .map(String::as_str)
        .unwrap_or(&pending.inspection.summary);
    truncate(detail, width.saturating_sub(4) as usize)
}

pub(crate) fn approval_preview_line_cap(
    kind: PendingActionKind,
    width: u16,
    terminal_height: Option<u16>,
) -> usize {
    let base = match kind {
        PendingActionKind::ShellCommand => 1,
        PendingActionKind::FileWrite | PendingActionKind::FileEdit => {
            if width < 72 {
                2
            } else {
                3
            }
        }
    };
    match terminal_height {
        Some(height) if height < 22 => base,
        Some(height) if height < 30 => match kind {
            PendingActionKind::ShellCommand => base.min(2),
            PendingActionKind::FileWrite | PendingActionKind::FileEdit => (base + 1).min(3),
        },
        Some(_) | None => match kind {
            PendingActionKind::ShellCommand => 2,
            PendingActionKind::FileWrite | PendingActionKind::FileEdit => (base + 1).min(4),
        },
    }
}

pub(crate) fn estimated_approval_rows(
    kind: PendingActionKind,
    width: u16,
    terminal_height: Option<u16>,
) -> u16 {
    let preview_rows = approval_preview_line_cap(kind, width, terminal_height) as u16;
    3 + preview_rows
}
