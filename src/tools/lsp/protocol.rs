use serde_json::Value;

use super::paths::file_uri_to_path;
use super::types::{DefinitionLocation, LspDiagnostic, LspResponseError};

pub(super) fn parse_diagnostic(value: &Value) -> Option<LspDiagnostic> {
    let line = value["range"]["start"]["line"].as_u64()? as usize + 1;
    let column = value["range"]["start"]["character"].as_u64()? as usize + 1;
    let message = value["message"].as_str()?.to_string();
    let source = value["source"].as_str().map(|s| s.to_string());
    let severity = match value["severity"].as_u64() {
        Some(1) => "error",
        Some(2) => "warning",
        Some(3) => "info",
        Some(4) => "hint",
        _ => "unknown",
    }
    .to_string();

    Some(LspDiagnostic {
        severity,
        line,
        column,
        message,
        source,
    })
}

pub(super) fn parse_lsp_response_error(message: &Value) -> Option<LspResponseError> {
    let error = message.get("error")?;
    let code = error.get("code")?.as_i64()?;
    let message = error
        .get("message")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown language server error")
        .to_string();
    let data = error.get("data").map(|value| {
        value
            .as_str()
            .map(|s| s.to_string())
            .unwrap_or_else(|| value.to_string())
    });

    Some(LspResponseError {
        code,
        message,
        data,
    })
}

pub(super) fn is_retryable_lsp_query_error(error: &LspResponseError) -> bool {
    matches!(error.code, -32803 | -32802 | -32801 | -32800 | -32002)
        || error.message.to_ascii_lowercase().contains("cancel")
        || error
            .message
            .to_ascii_lowercase()
            .contains("content modified")
}

pub(super) fn parse_hover_response(message: &Value) -> Option<String> {
    let result = message.get("result")?;
    let contents = result.get("contents")?;

    if let Some(text) = contents.as_str() {
        return Some(text.trim().to_string());
    }

    if let Some(object) = contents.as_object() {
        if let Some(value) = object.get("value").and_then(|v| v.as_str()) {
            return Some(value.trim().to_string());
        }
    }

    if let Some(items) = contents.as_array() {
        let mut parts = Vec::new();
        for item in items {
            if let Some(text) = item.as_str() {
                parts.push(text.trim().to_string());
            } else if let Some(value) = item.get("value").and_then(|v| v.as_str()) {
                parts.push(value.trim().to_string());
            }
        }
        let joined = parts
            .into_iter()
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
        if !joined.is_empty() {
            return Some(joined);
        }
    }

    None
}

pub(super) fn parse_definition_response(message: &Value) -> Vec<DefinitionLocation> {
    let Some(result) = message.get("result") else {
        return Vec::new();
    };

    if result.is_null() {
        return Vec::new();
    }

    if let Some(items) = result.as_array() {
        return items.iter().filter_map(parse_definition_location).collect();
    }

    parse_definition_location(result).into_iter().collect()
}

fn parse_definition_location(value: &Value) -> Option<DefinitionLocation> {
    let (uri, start) = if value.get("targetUri").is_some() {
        (
            value.get("targetUri")?.as_str()?,
            value
                .get("targetSelectionRange")
                .and_then(|range| range.get("start"))
                .or_else(|| {
                    value
                        .get("targetRange")
                        .and_then(|range| range.get("start"))
                })?,
        )
    } else {
        (
            value.get("uri")?.as_str()?,
            value.get("range")?.get("start")?,
        )
    };

    let path = file_uri_to_path(uri)?;
    let line = start.get("line")?.as_u64()? as usize + 1;
    let column = start.get("character")?.as_u64()? as usize + 1;

    Some(DefinitionLocation { path, line, column })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::json;

    use super::*;

    #[test]
    fn parses_diagnostic_payload() {
        let diagnostic = parse_diagnostic(&json!({
            "range": {
                "start": { "line": 4, "character": 7 }
            },
            "severity": 1,
            "message": "cannot find value `x` in this scope",
            "source": "rustc"
        }))
        .expect("parse diagnostic");

        assert_eq!(diagnostic.line, 5);
        assert_eq!(diagnostic.column, 8);
        assert_eq!(diagnostic.severity, "error");
        assert_eq!(diagnostic.source.as_deref(), Some("rustc"));
    }

    #[test]
    fn parses_string_hover_response() {
        let hover = parse_hover_response(&json!({
            "result": {
                "contents": "let x: i32"
            }
        }));

        assert_eq!(hover.as_deref(), Some("let x: i32"));
    }

    #[test]
    fn parses_markup_hover_response() {
        let hover = parse_hover_response(&json!({
            "result": {
                "contents": {
                    "kind": "markdown",
                    "value": "```rust\nfn main()\n```"
                }
            }
        }));

        assert!(hover.unwrap().contains("fn main()"));
    }

    #[test]
    fn parses_lsp_error_payload() {
        let error = parse_lsp_response_error(&json!({
            "id": 2,
            "error": {
                "code": -32801,
                "message": "Content modified",
                "data": "still indexing"
            }
        }))
        .expect("error");

        assert_eq!(error.code, -32801);
        assert_eq!(error.message, "Content modified");
        assert_eq!(error.data.as_deref(), Some("still indexing"));
        assert!(is_retryable_lsp_query_error(&error));
    }

    #[test]
    fn parses_definition_location_response() {
        let definitions = parse_definition_response(&json!({
            "result": [{
                "uri": "file:///tmp/example.rs",
                "range": {
                    "start": { "line": 9, "character": 4 }
                }
            }]
        }));

        assert_eq!(definitions.len(), 1);
        assert_eq!(definitions[0].path, PathBuf::from("/tmp/example.rs"));
        assert_eq!(definitions[0].line, 10);
        assert_eq!(definitions[0].column, 5);
    }

    #[test]
    fn parses_definition_link_response() {
        let definitions = parse_definition_response(&json!({
            "result": [{
                "targetUri": "file:///tmp/example.rs",
                "targetSelectionRange": {
                    "start": { "line": 2, "character": 7 }
                }
            }]
        }));

        assert_eq!(definitions.len(), 1);
        assert_eq!(definitions[0].line, 3);
        assert_eq!(definitions[0].column, 8);
    }
}
