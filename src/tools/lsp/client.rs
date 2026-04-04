use std::path::Path;
use std::process::{Child, ChildStdin};
use std::sync::mpsc;
use std::time::Duration;

use serde_json::{json, Value};
use tracing::warn;

use crate::config;
use crate::error::{ParamsError, Result};

use super::paths::path_to_file_uri;
use super::position::{build_hover_positions, line_column_to_utf16};
use super::probe::resolve_rust_analyzer_command;
use super::transport::{
    spawn_language_server, spawn_reader, wait_for_definition_response, wait_for_diagnostics,
    wait_for_hover_response, wait_for_response, write_lsp_message,
};
use super::types::{DefinitionLocation, DefinitionResponse, HoverResponse, LspDiagnostic};

struct LspSession {
    child: Child,
    stdin: ChildStdin,
    rx: mpsc::Receiver<Value>,
    timeout: Duration,
    file_uri: String,
    next_id: u64,
}

impl LspSession {
    fn start(
        cfg: &config::Config,
        project_root: &Path,
        file_path: &Path,
        source: &str,
    ) -> Result<Self> {
        let server = resolve_rust_analyzer_command(cfg)?;
        let mut child = spawn_language_server(&server, project_root)?;
        let mut stdin = child.stdin.take().ok_or_else(|| {
            ParamsError::Inference("Failed to open rust-analyzer stdin".to_string())
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            ParamsError::Inference("Failed to open rust-analyzer stdout".to_string())
        })?;
        let rx = spawn_reader(stdout);

        let root_uri = path_to_file_uri(project_root);
        let file_uri = path_to_file_uri(file_path);
        let timeout = Duration::from_millis(cfg.lsp.timeout_ms);
        let workspace_name = project_root
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("workspace");

        write_lsp_message(
            &mut stdin,
            &json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "processId": serde_json::Value::Null,
                    "rootPath": project_root.to_str(),
                    "rootUri": root_uri,
                    "workspaceFolders": [{
                        "uri": root_uri,
                        "name": workspace_name,
                    }],
                    "capabilities": {},
                    "clientInfo": {
                        "name": "params-cli",
                        "version": env!("CARGO_PKG_VERSION"),
                    }
                }
            }),
        )?;
        wait_for_response(&rx, 1, timeout)?;

        write_lsp_message(
            &mut stdin,
            &json!({
                "jsonrpc": "2.0",
                "method": "initialized",
                "params": {}
            }),
        )?;

        write_lsp_message(
            &mut stdin,
            &json!({
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": file_uri,
                        "languageId": "rust",
                        "version": 1,
                        "text": source,
                    }
                }
            }),
        )?;

        Ok(Self {
            child,
            stdin,
            rx,
            timeout,
            file_uri,
            next_id: 2,
        })
    }

    fn next_request_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn await_diagnostics(&self) -> Result<Vec<LspDiagnostic>> {
        wait_for_diagnostics(&self.rx, &self.file_uri, self.timeout)
    }

    fn wait_until_ready(&self) {
        let _ = wait_for_diagnostics(&self.rx, &self.file_uri, self.timeout);
    }

    fn hover(&mut self, source: &str, line: usize, column: usize) -> Result<Option<String>> {
        let hover_positions = build_hover_positions(source, line, column)?;
        let mut hover = None;
        for position in hover_positions {
            for _ in 0..3 {
                let utf16_col = line_column_to_utf16(source, position.line, position.column)?;
                let id = self.next_request_id();
                write_lsp_message(
                    &mut self.stdin,
                    &json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "method": "textDocument/hover",
                        "params": {
                            "textDocument": { "uri": self.file_uri },
                            "position": {
                                "line": position.line.saturating_sub(1),
                                "character": utf16_col,
                            }
                        }
                    }),
                )?;

                match wait_for_hover_response(&self.rx, id, self.timeout)? {
                    HoverResponse::Hover(text) => {
                        hover = Some(text);
                    }
                    HoverResponse::NoInfo => {}
                    HoverResponse::RetryableError(reason) => {
                        warn!(tool = "lsp_hover", reason, "retrying hover request");
                        std::thread::sleep(Duration::from_millis(75));
                        continue;
                    }
                }

                break;
            }

            if hover.is_some() {
                break;
            }
        }

        Ok(hover)
    }

    fn definition(
        &mut self,
        source: &str,
        line: usize,
        column: usize,
    ) -> Result<Vec<DefinitionLocation>> {
        let hover_positions = build_hover_positions(source, line, column)?;
        let mut definitions = Vec::new();
        for position in hover_positions {
            for _ in 0..3 {
                let utf16_col = line_column_to_utf16(source, position.line, position.column)?;
                let id = self.next_request_id();
                write_lsp_message(
                    &mut self.stdin,
                    &json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "method": "textDocument/definition",
                        "params": {
                            "textDocument": { "uri": self.file_uri },
                            "position": {
                                "line": position.line.saturating_sub(1),
                                "character": utf16_col,
                            }
                        }
                    }),
                )?;

                match wait_for_definition_response(&self.rx, id, self.timeout)? {
                    DefinitionResponse::Definitions(items) => {
                        definitions = items;
                    }
                    DefinitionResponse::NoInfo => {}
                    DefinitionResponse::RetryableError(reason) => {
                        warn!(
                            tool = "lsp_definition",
                            reason, "retrying definition request"
                        );
                        std::thread::sleep(Duration::from_millis(75));
                        continue;
                    }
                }

                break;
            }

            if !definitions.is_empty() {
                break;
            }
        }

        Ok(definitions)
    }

    fn close(&mut self) {
        let id = self.next_request_id();
        let _ = write_lsp_message(
            &mut self.stdin,
            &json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": "shutdown",
                "params": serde_json::Value::Null,
            }),
        );
        let _ = wait_for_response(&self.rx, id, Duration::from_millis(300));
        let _ = write_lsp_message(
            &mut self.stdin,
            &json!({
                "jsonrpc": "2.0",
                "method": "exit",
                "params": serde_json::Value::Null,
            }),
        );
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for LspSession {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub(super) fn collect_rust_diagnostics(
    cfg: &config::Config,
    project_root: &Path,
    file_path: &Path,
    source: &str,
) -> Result<Vec<LspDiagnostic>> {
    let mut session = LspSession::start(cfg, project_root, file_path, source)?;
    let diagnostics = session.await_diagnostics()?;
    session.close();
    Ok(diagnostics)
}

pub(super) fn collect_rust_hover(
    cfg: &config::Config,
    project_root: &Path,
    file_path: &Path,
    source: &str,
    line: usize,
    column: usize,
) -> Result<Option<String>> {
    let mut session = LspSession::start(cfg, project_root, file_path, source)?;
    session.wait_until_ready();
    let hover = session.hover(source, line, column)?;
    session.close();
    Ok(hover)
}

pub(super) fn collect_rust_definition(
    cfg: &config::Config,
    project_root: &Path,
    file_path: &Path,
    source: &str,
    line: usize,
    column: usize,
) -> Result<Vec<DefinitionLocation>> {
    let mut session = LspSession::start(cfg, project_root, file_path, source)?;
    session.wait_until_ready();
    let definitions = session.definition(source, line, column)?;
    session.close();
    Ok(definitions)
}
