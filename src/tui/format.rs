use std::time::Duration;

use ratatui::{
    style::Style,
    text::{Line, Span},
};

pub(crate) fn truncate_for_width(value: &str, max_chars: usize) -> String {
    let len = value.chars().count();
    if len <= max_chars {
        return value.to_string();
    }
    let keep = max_chars.saturating_sub(1);
    let truncated: String = value.chars().take(keep).collect();
    format!("{truncated}…")
}

pub(crate) fn format_compact_count(value: usize) -> String {
    if value >= 1_000_000 {
        format!("{:.1}m", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}k", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

pub(crate) fn format_cost(value: Option<f64>) -> String {
    match value {
        Some(v) if v >= 1.0 => format!("${v:.2}"),
        Some(v) => format!("${v:.4}"),
        None => "n/a".to_string(),
    }
}

pub(crate) fn format_hit_rate(hits: usize, misses: usize) -> String {
    let total = hits.saturating_add(misses);
    if total == 0 {
        "n/a".to_string()
    } else {
        format!("{:.0}%", (hits as f64 / total as f64) * 100.0)
    }
}

pub(crate) fn format_duration(duration: Duration) -> String {
    let total_ms = duration.as_millis();
    if total_ms < 10_000 {
        let secs = total_ms as f64 / 1000.0;
        format!("{secs:.1}s")
    } else {
        let total_secs = duration.as_secs();
        if total_secs < 60 {
            format!("{total_secs}s")
        } else if total_secs < 3600 {
            let mins = total_secs / 60;
            let secs = total_secs % 60;
            format!("{mins}m {secs:02}s")
        } else {
            let hours = total_secs / 3600;
            let mins = (total_secs % 3600) / 60;
            format!("{hours}h {mins:02}m")
        }
    }
}

pub(crate) fn wrap_plain_text(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![String::new()];
    }
    if text.is_empty() {
        return vec![String::new()];
    }

    let chars: Vec<char> = text.chars().collect();
    let mut wrapped = Vec::new();
    let mut start = 0;

    while start < chars.len() {
        let end = (start + width).min(chars.len());
        wrapped.push(chars[start..end].iter().collect());
        start = end;
    }

    wrapped
}

pub(crate) fn push_wrapped_styled(lines: &mut Vec<Line>, text: &str, style: Style, width: usize) {
    for part in wrap_plain_text(text, width) {
        lines.push(Line::from(Span::styled(part, style)));
    }
}

pub(crate) fn sanitize_for_display(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\u{1b}' => {
                while let Some(&next) = chars.peek() {
                    chars.next();
                    if next.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
            '\n' | '\t' => result.push(c),
            c if c.is_control() => {}
            c => result.push(c),
        }
    }
    result
}
