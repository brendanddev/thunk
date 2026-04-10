use crate::session::short_id;
use crate::tui::state::AppState;

use super::{label_width, truncate, PackedStyle, StyledLine, StyledSpan, Theme, SPINNER_FRAMES};

pub(super) fn build_top_bar(state: &AppState, theme: Theme, width: u16) -> Vec<StyledLine> {
    let session_label = session_label(state, width);
    let runtime_label = persistent_runtime_label(state);
    let runtime_style = runtime_style(state, theme);
    let runtime_display = if is_runtime_animated(state) {
        format!(
            "{} {}",
            spinner_frame(state.tick),
            truncate(&runtime_label, 20)
        )
    } else {
        truncate(&runtime_label, 22)
    };
    let mut segments = vec![
        TopSegment::new("params", theme.dim(), 4),
        TopSegment::new(&runtime_display, runtime_style, 1),
        TopSegment::new(&session_label, theme.muted(), 0),
    ];

    if state.show_top_bar_tokens {
        segments.push(TopSegment::new(
            &format_token_display(state.total_tokens),
            theme.dim(),
            2,
        ));
    }

    if state.show_top_bar_time {
        if let Some(duration) = state
            .current_turn_duration()
            .or_else(|| state.last_work_duration())
        {
            segments.push(TopSegment::new(
                &format_duration_display(duration),
                theme.dim(),
                3,
            ));
        }
    }

    vec![build_segmented_top_line(segments, width, theme)]
}

fn format_token_display(total_tokens: usize) -> String {
    if total_tokens >= 10_000 {
        format!("{:.1}k tok", total_tokens as f64 / 1000.0)
    } else {
        format!("{total_tokens} tok")
    }
}

pub(super) fn format_duration_display(duration: std::time::Duration) -> String {
    if duration.as_secs() >= 60 {
        let minutes = duration.as_secs() / 60;
        let seconds = duration.as_secs() % 60;
        format!("{minutes}m {seconds:02}s")
    } else {
        format!("{:.1}s", duration.as_secs_f32())
    }
}

#[derive(Debug, Clone)]
struct TopSegment {
    text: String,
    style: PackedStyle,
    priority: u8,
}

impl TopSegment {
    fn new(text: &str, style: PackedStyle, priority: u8) -> Self {
        Self {
            text: text.to_string(),
            style,
            priority,
        }
    }
}

fn build_segmented_top_line(mut segments: Vec<TopSegment>, width: u16, theme: Theme) -> StyledLine {
    let width = width as usize;
    let separator = " · ";
    while segmented_width(&segments, separator) > width && segments.len() > 1 {
        if let Some(idx) = segments
            .iter()
            .enumerate()
            .max_by_key(|(_, segment)| segment.priority)
            .map(|(idx, _)| idx)
        {
            segments.remove(idx);
        } else {
            break;
        }
    }

    let mut spans = Vec::new();
    for (idx, segment) in segments.into_iter().enumerate() {
        if idx > 0 {
            spans.push(StyledSpan {
                text: separator.to_string(),
                style: theme.dim(),
            });
        }
        spans.push(StyledSpan {
            text: segment.text,
            style: segment.style,
        });
    }
    StyledLine { spans }
}

fn segmented_width(segments: &[TopSegment], separator: &str) -> usize {
    let text_width: usize = segments
        .iter()
        .map(|segment| label_width(&segment.text))
        .sum();
    let separators = separator
        .len()
        .saturating_mul(segments.len().saturating_sub(1));
    text_width + separators
}

fn session_label(state: &AppState, width: u16) -> String {
    let base = state
        .current_session
        .as_ref()
        .map(|session| {
            session
                .name
                .clone()
                .unwrap_or_else(|| format!("unnamed {}", short_id(&session.id)))
        })
        .unwrap_or_else(|| "fresh".to_string());

    if width >= 72 && state.backend_name != "..." {
        format!("{base} · {}", truncate(&state.backend_name, 16))
    } else {
        truncate(&base, 22)
    }
}

fn persistent_runtime_label(state: &AppState) -> String {
    if state.pending_action.is_some() {
        "awaiting approval".to_string()
    } else if state.status.starts_with("error") {
        "error".to_string()
    } else if state.is_generating {
        "generating".to_string()
    } else if state.is_ready() {
        "ready".to_string()
    } else {
        "loading".to_string()
    }
}

pub(super) fn build_activity_line(
    state: &AppState,
    theme: Theme,
    width: u16,
) -> Option<StyledLine> {
    let label = state.current_trace.as_deref()?;
    let glyph = spinner_frame(state.tick);
    Some(StyledLine {
        spans: vec![
            StyledSpan {
                text: format!("{glyph} "),
                style: theme.dim(),
            },
            StyledSpan {
                text: truncate(label, width.saturating_sub(3) as usize),
                style: theme.dim(),
            },
        ],
    })
}

fn runtime_style(state: &AppState, theme: Theme) -> PackedStyle {
    if state.pending_action.is_some() {
        theme.chip_warning()
    } else if state.status.starts_with("error") {
        theme.chip_danger()
    } else if !state.is_ready() {
        theme.chip_accent()
    } else if state.is_generating || state.current_trace.is_some() {
        theme.badge_assistant()
    } else {
        theme.dim()
    }
}

fn is_runtime_animated(state: &AppState) -> bool {
    state.pending_action.is_some()
        || !state.is_ready()
        || state.is_generating
        || state.current_trace.is_some()
}

pub(super) fn spinner_frame(tick: u64) -> &'static str {
    SPINNER_FRAMES[(tick as usize / 3) % SPINNER_FRAMES.len()]
}

pub(super) fn active_cursor_span(theme: Theme, tick: u64) -> StyledSpan {
    StyledSpan {
        text: if (tick / 6).is_multiple_of(2) {
            " ▍".to_string()
        } else {
            " ▌".to_string()
        },
        style: if (tick / 6).is_multiple_of(2) {
            theme.badge_assistant()
        } else {
            theme.chip_accent()
        },
    }
}
