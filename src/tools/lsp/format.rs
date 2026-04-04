use std::path::Path;

use super::paths::display_path;
use super::types::{DefinitionLocation, LspDiagnostic, LspResponseError};

pub(super) fn format_diagnostics(display_path: &str, diagnostics: &[LspDiagnostic]) -> String {
    if diagnostics.is_empty() {
        return format!("No diagnostics for {display_path}");
    }

    let mut output = format!(
        "Diagnostics for {} ({} issues):\n\n",
        display_path,
        diagnostics.len()
    );

    for diagnostic in diagnostics {
        output.push_str(&format!(
            "{}:{}:{} [{}] {}\n",
            display_path,
            diagnostic.line,
            diagnostic.column,
            diagnostic.severity,
            diagnostic.message
        ));
        if let Some(source) = &diagnostic.source {
            output.push_str(&format!("  source: {}\n", source));
        }
    }

    output
}

pub(super) fn format_hover(
    display_path: &str,
    line: usize,
    column: usize,
    hover: Option<String>,
) -> String {
    match hover {
        Some(text) if !text.trim().is_empty() => format!(
            "Hover for {}:{}:{}:\n\n{}",
            display_path,
            line,
            column,
            text.trim()
        ),
        _ => format!("No hover info for {}:{}:{}", display_path, line, column),
    }
}

pub(super) fn format_definition(
    cwd: &Path,
    source_path: &str,
    line: usize,
    column: usize,
    definitions: &[DefinitionLocation],
) -> String {
    if definitions.is_empty() {
        return format!(
            "No definition found for {}:{}:{}",
            source_path, line, column
        );
    }

    let mut output = format!("Definition for {}:{}:{}:\n\n", source_path, line, column);
    for definition in definitions {
        let path = display_path(cwd, &definition.path);
        output.push_str(&format!(
            "{path}:{}:{}\n",
            definition.line, definition.column
        ));
    }

    output
}

pub(super) fn format_lsp_response_error(error: &LspResponseError) -> String {
    match &error.data {
        Some(data) if !data.is_empty() => {
            format!("code {}: {} ({data})", error.code, error.message)
        }
        _ => format!("code {}: {}", error.code, error.message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_empty_diagnostics() {
        let formatted = format_diagnostics("src/main.rs", &[]);
        assert_eq!(formatted, "No diagnostics for src/main.rs");
    }
}
