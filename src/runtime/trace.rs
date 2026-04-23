use crate::runtime::types::RuntimeEvent;

/// Env flag to enable lightweight runtime decision tracing.
///
/// When unset, all trace emission is a no-op (zero-cost fast path).
pub(super) const RUNTIME_TRACE_ENV: &str = "PARAMS_TRACE_RUNTIME";

/// Emits a structured runtime trace line if tracing is enabled.
///
/// Used for observability of runtime decisions without coupling
/// tracing to core logic. Output is a single-line, key=value format.
pub(super) fn trace_runtime_decision(
    on_event: &mut dyn FnMut(RuntimeEvent),
    event: &str,
    fields: &[(&str, String)],
) {
    if std::env::var_os(RUNTIME_TRACE_ENV).is_none() {
        return;
    }

    let mut line = format!("[runtime:trace] event={event}");
    for (key, value) in fields {
        line.push(' ');
        line.push_str(key);
        line.push('=');
        line.push_str(&trace_field_value(value));
    }
    on_event(RuntimeEvent::RuntimeTrace(line));
}

/// Formats a field value for trace output.
///
/// Keeps simple values unquoted for readability and quotes anything
/// that contains non-safe characters.
pub(super) fn trace_field_value(value: &str) -> String {
    if value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '/' | '.' | ':' | '='))
    {
        value.to_string()
    } else {
        format!("{value:?}")
    }
}
