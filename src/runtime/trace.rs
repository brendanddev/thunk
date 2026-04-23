use crate::runtime::types::RuntimeEvent;

pub(super) const RUNTIME_TRACE_ENV: &str = "PARAMS_TRACE_RUNTIME";

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
