use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use serde_json::Value;

use crate::error::{ParamsError, Result};

use super::format::format_lsp_response_error;
use super::protocol::{
    is_retryable_lsp_query_error, parse_definition_response, parse_diagnostic,
    parse_hover_response, parse_lsp_response_error,
};
use super::types::{DefinitionResponse, HoverResponse, LspCommandSpec, LspDiagnostic};

pub(super) fn spawn_language_server(server: &LspCommandSpec, project_root: &Path) -> Result<Child> {
    Command::new(&server.program)
        .args(&server.args)
        .current_dir(project_root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| {
            ParamsError::Inference(format!(
                "Failed to start rust-analyzer via {}: {e}",
                server.display
            ))
        })
}

pub(super) fn write_lsp_message(stdin: &mut ChildStdin, value: &Value) -> Result<()> {
    let payload = value.to_string();
    write!(
        stdin,
        "Content-Length: {}\r\n\r\n{}",
        payload.len(),
        payload
    )?;
    stdin.flush()?;
    Ok(())
}

pub(super) fn spawn_reader(stdout: ChildStdout) -> mpsc::Receiver<Value> {
    let (message_tx, message_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        while let Ok(message) = read_lsp_message(&mut reader) {
            if message_tx.send(message).is_err() {
                break;
            }
        }
    });
    message_rx
}

fn read_lsp_message(reader: &mut BufReader<ChildStdout>) -> Result<Value> {
    let mut content_length = None;

    loop {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            return Err(ParamsError::Inference(
                "Language server closed the connection".to_string(),
            ));
        }

        if line == "\r\n" || line == "\n" {
            break;
        }

        if let Some((name, value)) = line.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                let parsed = value.trim().parse::<usize>().map_err(|e| {
                    ParamsError::Inference(format!("Invalid LSP Content-Length header: {e}"))
                })?;
                content_length = Some(parsed);
            }
        }
    }

    let length = content_length
        .ok_or_else(|| ParamsError::Inference("Missing LSP Content-Length header".to_string()))?;
    let mut payload = vec![0; length];
    reader.read_exact(&mut payload)?;
    serde_json::from_slice(&payload)
        .map_err(|e| ParamsError::Inference(format!("Invalid LSP JSON payload: {e}")))
}

fn timeout_error(context: &str, timeout: Duration) -> ParamsError {
    ParamsError::Inference(format!(
        "Timed out waiting for {context} after {}ms. \
Increase [lsp].timeout_ms in .local/config.toml if rust-analyzer is slow to start.",
        timeout.as_millis()
    ))
}

pub(super) fn wait_for_response(
    rx: &mpsc::Receiver<Value>,
    id: u64,
    timeout: Duration,
) -> Result<Value> {
    loop {
        let message = rx
            .recv_timeout(timeout)
            .map_err(|_| timeout_error("language server response", timeout))?;

        if message.get("id").and_then(|v| v.as_u64()) == Some(id) {
            if let Some(error) = parse_lsp_response_error(&message) {
                return Err(ParamsError::Inference(format!(
                    "Language server error: {}",
                    format_lsp_response_error(&error)
                )));
            }
            return Ok(message);
        }
    }
}

pub(super) fn wait_for_hover_response(
    rx: &mpsc::Receiver<Value>,
    id: u64,
    timeout: Duration,
) -> Result<HoverResponse> {
    loop {
        let message = rx
            .recv_timeout(timeout)
            .map_err(|_| timeout_error("language server response", timeout))?;

        if message.get("id").and_then(|v| v.as_u64()) == Some(id) {
            if let Some(error) = parse_lsp_response_error(&message) {
                if is_retryable_lsp_query_error(&error) {
                    return Ok(HoverResponse::RetryableError(format_lsp_response_error(
                        &error,
                    )));
                }
                return Err(ParamsError::Inference(format!(
                    "Language server error: {}",
                    format_lsp_response_error(&error)
                )));
            }

            return Ok(match parse_hover_response(&message) {
                Some(text) if !text.trim().is_empty() => HoverResponse::Hover(text),
                _ => HoverResponse::NoInfo,
            });
        }
    }
}

pub(super) fn wait_for_definition_response(
    rx: &mpsc::Receiver<Value>,
    id: u64,
    timeout: Duration,
) -> Result<DefinitionResponse> {
    loop {
        let message = rx
            .recv_timeout(timeout)
            .map_err(|_| timeout_error("language server response", timeout))?;

        if message.get("id").and_then(|v| v.as_u64()) == Some(id) {
            if let Some(error) = parse_lsp_response_error(&message) {
                if is_retryable_lsp_query_error(&error) {
                    return Ok(DefinitionResponse::RetryableError(
                        format_lsp_response_error(&error),
                    ));
                }
                return Err(ParamsError::Inference(format!(
                    "Language server error: {}",
                    format_lsp_response_error(&error)
                )));
            }

            let definitions = parse_definition_response(&message);
            return Ok(if definitions.is_empty() {
                DefinitionResponse::NoInfo
            } else {
                DefinitionResponse::Definitions(definitions)
            });
        }
    }
}

pub(super) fn wait_for_diagnostics(
    rx: &mpsc::Receiver<Value>,
    target_uri: &str,
    timeout: Duration,
) -> Result<Vec<LspDiagnostic>> {
    loop {
        let message = rx
            .recv_timeout(timeout)
            .map_err(|_| timeout_error("rust-analyzer diagnostics", timeout))?;

        if message.get("method").and_then(|v| v.as_str()) == Some("textDocument/publishDiagnostics")
        {
            let params = &message["params"];
            if params["uri"].as_str() == Some(target_uri) {
                let diagnostics = params["diagnostics"]
                    .as_array()
                    .map(|items| items.iter().filter_map(parse_diagnostic).collect())
                    .unwrap_or_default();
                return Ok(diagnostics);
            }
        }
    }
}
