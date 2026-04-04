// src/tools/lsp.rs
//
// Rust-first LSP diagnostics tool.

use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use serde_json::{json, Value};
use tracing::{info, warn};

use super::{Tool, ToolRunResult};
use crate::config;
use crate::error::{ParamsError, Result};
use crate::safety::{self, ProjectPathKind};

pub struct LspDiagnosticsTool;
pub struct LspHoverTool;
pub struct LspDefinitionTool;

#[derive(Debug, Clone)]
struct LspDiagnostic {
    severity: String,
    line: usize,
    column: usize,
    message: String,
    source: Option<String>,
}

#[derive(Debug, Clone)]
struct LspCommandSpec {
    program: PathBuf,
    args: Vec<String>,
    display: String,
}

#[derive(Debug, Clone)]
struct LspProbe {
    spec: LspCommandSpec,
    status: LspProbeStatus,
}

#[derive(Debug, Clone)]
enum LspProbeStatus {
    Ready(String),
    Failed(String),
}

impl Tool for LspDiagnosticsTool {
    fn name(&self) -> &str {
        "lsp_diagnostics"
    }

    fn description(&self) -> &str {
        "Get Rust diagnostics for a source file via rust-analyzer. Usage: [lsp_diagnostics: src/main.rs]"
    }

    fn run(&self, arg: &str) -> Result<ToolRunResult> {
        let requested = arg.trim();
        if requested.is_empty() {
            return Err(ParamsError::Config(
                "lsp_diagnostics requires a file path".to_string(),
            ));
        }

        let cfg = config::load_with_profile()?;
        let cwd = safety::project_root()?;
        let path = resolve_input_path(requested)?;
        validate_rust_file(&path)?;
        let display_path = display_path(&cwd, &path);
        let project_root = find_rust_project_root(&path)?;
        let source = fs::read_to_string(&path)?;

        info!(tool = "lsp_diagnostics", "tool called");

        let diagnostics = collect_rust_diagnostics(&cfg, &project_root, &path, &source)?;
        let output = format_diagnostics(&display_path, &diagnostics);

        Ok(ToolRunResult::Immediate(output))
    }
}

impl Tool for LspHoverTool {
    fn name(&self) -> &str {
        "lsp_hover"
    }

    fn description(&self) -> &str {
        "Get Rust hover info for a symbol via rust-analyzer. Usage: [lsp_hover: src/main.rs:12:8]"
    }

    fn run(&self, arg: &str) -> Result<ToolRunResult> {
        let requested = arg.trim();
        if requested.is_empty() {
            return Err(ParamsError::Config(
                "lsp_hover requires <file>:<line>:<col>".to_string(),
            ));
        }

        let cfg = config::load_with_profile()?;
        let cwd = safety::project_root()?;
        let hover_input = parse_hover_input(requested)?;
        validate_rust_file(&hover_input.path)?;
        let display_path = display_path(&cwd, &hover_input.path);
        let project_root = find_rust_project_root(&hover_input.path)?;
        let source = fs::read_to_string(&hover_input.path)?;

        info!(tool = "lsp_hover", "tool called");

        let hover = collect_rust_hover(
            &cfg,
            &project_root,
            &hover_input.path,
            &source,
            hover_input.line,
            hover_input.column,
        )?;
        let output = format_hover(&display_path, hover_input.line, hover_input.column, hover);

        Ok(ToolRunResult::Immediate(output))
    }
}

impl Tool for LspDefinitionTool {
    fn name(&self) -> &str {
        "lsp_definition"
    }

    fn description(&self) -> &str {
        "Find the Rust definition for a symbol via rust-analyzer. Usage: [lsp_definition: src/main.rs:12:8]"
    }

    fn run(&self, arg: &str) -> Result<ToolRunResult> {
        let requested = arg.trim();
        if requested.is_empty() {
            return Err(ParamsError::Config(
                "lsp_definition requires <file>:<line>:<col>".to_string(),
            ));
        }

        let cfg = config::load_with_profile()?;
        let cwd = safety::project_root()?;
        let position_input = parse_hover_input(requested)?;
        validate_rust_file(&position_input.path)?;
        let display_input = display_path(&cwd, &position_input.path);
        let project_root = find_rust_project_root(&position_input.path)?;
        let source = fs::read_to_string(&position_input.path)?;

        info!(tool = "lsp_definition", "tool called");

        let definitions = collect_rust_definition(
            &cfg,
            &project_root,
            &position_input.path,
            &source,
            position_input.line,
            position_input.column,
        )?;
        let output = format_definition(
            &cwd,
            &display_input,
            position_input.line,
            position_input.column,
            &definitions,
        );

        Ok(ToolRunResult::Immediate(output))
    }
}

#[derive(Debug, Clone)]
struct HoverInput {
    path: PathBuf,
    line: usize,
    column: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HoverPosition {
    line: usize,
    column: usize,
}

#[derive(Debug, Clone)]
struct LspResponseError {
    code: i64,
    message: String,
    data: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DefinitionLocation {
    path: PathBuf,
    line: usize,
    column: usize,
}

fn resolve_input_path(requested: &str) -> Result<PathBuf> {
    Ok(
        safety::inspect_project_path("lsp_file", requested, ProjectPathKind::File, false)?
            .resolved_path,
    )
}

fn validate_rust_file(path: &Path) -> Result<()> {
    if !path.is_file() {
        return Err(ParamsError::Config(format!(
            "{} is not a file",
            path.display()
        )));
    }

    if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
        return Err(ParamsError::Config(
            "The first LSP slice currently supports Rust `.rs` files only".to_string(),
        ));
    }

    Ok(())
}

fn parse_hover_input(requested: &str) -> Result<HoverInput> {
    let Some((path_part, line_part, col_part)) = split_path_line_col(requested) else {
        return Err(ParamsError::Config(
            "Expected <file>:<line>:<col>, for example src/main.rs:12:8".to_string(),
        ));
    };

    let line = line_part
        .parse::<usize>()
        .map_err(|_| ParamsError::Config("Hover line must be a positive integer".to_string()))?;
    let column = col_part
        .parse::<usize>()
        .map_err(|_| ParamsError::Config("Hover column must be a positive integer".to_string()))?;

    if line == 0 || column == 0 {
        return Err(ParamsError::Config(
            "Hover line and column are 1-based and must be greater than 0".to_string(),
        ));
    }

    Ok(HoverInput {
        path: resolve_input_path(path_part)?,
        line,
        column,
    })
}

fn split_path_line_col(input: &str) -> Option<(&str, &str, &str)> {
    let (path_and_line, col) = input.rsplit_once(':')?;
    let (path, line) = path_and_line.rsplit_once(':')?;
    Some((path, line, col))
}

fn display_path(cwd: &Path, path: &Path) -> String {
    path.strip_prefix(cwd)
        .ok()
        .and_then(|p| p.to_str())
        .map(|p| p.to_string())
        .unwrap_or_else(|| path.display().to_string())
}

fn find_rust_project_root(path: &Path) -> Result<PathBuf> {
    for ancestor in path.ancestors() {
        if ancestor.join("Cargo.toml").exists() {
            return Ok(ancestor.to_path_buf());
        }
    }

    Err(ParamsError::Config(format!(
        "No Cargo.toml found above {}. LSP diagnostics currently expect a Rust project.",
        path.display()
    )))
}

fn collect_rust_diagnostics(
    cfg: &config::Config,
    project_root: &Path,
    file_path: &Path,
    source: &str,
) -> Result<Vec<LspDiagnostic>> {
    let server = resolve_rust_analyzer_command(cfg)?;
    let mut child = spawn_language_server(&server, project_root)?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| ParamsError::Inference("Failed to open rust-analyzer stdin".to_string()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| ParamsError::Inference("Failed to open rust-analyzer stdout".to_string()))?;

    let (message_tx, message_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        while let Ok(message) = read_lsp_message(&mut reader) {
            if message_tx.send(message).is_err() {
                break;
            }
        }
    });

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
    wait_for_response(&message_rx, 1, timeout)?;

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

    let diagnostics = wait_for_diagnostics(&message_rx, &file_uri, timeout)?;

    let _ = write_lsp_message(
        &mut stdin,
        &json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "shutdown",
            "params": serde_json::Value::Null,
        }),
    );
    let _ = wait_for_response(&message_rx, 2, Duration::from_millis(300));
    let _ = write_lsp_message(
        &mut stdin,
        &json!({
            "jsonrpc": "2.0",
            "method": "exit",
            "params": serde_json::Value::Null,
        }),
    );
    let _ = child.kill();
    let _ = child.wait();

    Ok(diagnostics)
}

fn collect_rust_hover(
    cfg: &config::Config,
    project_root: &Path,
    file_path: &Path,
    source: &str,
    line: usize,
    column: usize,
) -> Result<Option<String>> {
    let server = resolve_rust_analyzer_command(cfg)?;
    let mut child = spawn_language_server(&server, project_root)?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| ParamsError::Inference("Failed to open rust-analyzer stdin".to_string()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| ParamsError::Inference("Failed to open rust-analyzer stdout".to_string()))?;

    let (message_tx, message_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        while let Ok(message) = read_lsp_message(&mut reader) {
            if message_tx.send(message).is_err() {
                break;
            }
        }
    });

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
    wait_for_response(&message_rx, 1, timeout)?;

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

    // Let rust-analyzer finish opening/indexing this file before hover requests.
    // We already rely on publishDiagnostics as the readiness signal in `/diag`,
    // and an empty diagnostics list is still a useful "ready" response here.
    let _ = wait_for_diagnostics(&message_rx, &file_uri, timeout);

    let hover_positions = build_hover_positions(source, line, column)?;
    let mut next_id = 2;
    let mut hover = None;
    for position in hover_positions {
        for _ in 0..3 {
            let utf16_col = line_column_to_utf16(source, position.line, position.column)?;
            write_lsp_message(
                &mut stdin,
                &json!({
                    "jsonrpc": "2.0",
                    "id": next_id,
                    "method": "textDocument/hover",
                    "params": {
                        "textDocument": { "uri": file_uri },
                        "position": {
                            "line": position.line.saturating_sub(1),
                            "character": utf16_col,
                        }
                    }
                }),
            )?;

            match wait_for_hover_response(&message_rx, next_id, timeout)? {
                HoverResponse::Hover(text) => {
                    hover = Some(text);
                }
                HoverResponse::NoInfo => {}
                HoverResponse::RetryableError(reason) => {
                    warn!(tool = "lsp_hover", reason, "retrying hover request");
                    next_id += 1;
                    std::thread::sleep(Duration::from_millis(75));
                    continue;
                }
            }

            next_id += 1;
            break;
        }

        if hover.is_some() {
            break;
        }
    }

    let _ = write_lsp_message(
        &mut stdin,
        &json!({
            "jsonrpc": "2.0",
            "id": next_id,
            "method": "shutdown",
            "params": serde_json::Value::Null,
        }),
    );
    let _ = wait_for_response(&message_rx, next_id, Duration::from_millis(300));
    let _ = write_lsp_message(
        &mut stdin,
        &json!({
            "jsonrpc": "2.0",
            "method": "exit",
            "params": serde_json::Value::Null,
        }),
    );
    let _ = child.kill();
    let _ = child.wait();

    Ok(hover)
}

fn collect_rust_definition(
    cfg: &config::Config,
    project_root: &Path,
    file_path: &Path,
    source: &str,
    line: usize,
    column: usize,
) -> Result<Vec<DefinitionLocation>> {
    let server = resolve_rust_analyzer_command(cfg)?;
    let mut child = spawn_language_server(&server, project_root)?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| ParamsError::Inference("Failed to open rust-analyzer stdin".to_string()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| ParamsError::Inference("Failed to open rust-analyzer stdout".to_string()))?;

    let (message_tx, message_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        while let Ok(message) = read_lsp_message(&mut reader) {
            if message_tx.send(message).is_err() {
                break;
            }
        }
    });

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
    wait_for_response(&message_rx, 1, timeout)?;

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

    let _ = wait_for_diagnostics(&message_rx, &file_uri, timeout);

    let hover_positions = build_hover_positions(source, line, column)?;
    let mut next_id = 2;
    let mut definitions = Vec::new();
    for position in hover_positions {
        for _ in 0..3 {
            let utf16_col = line_column_to_utf16(source, position.line, position.column)?;
            write_lsp_message(
                &mut stdin,
                &json!({
                    "jsonrpc": "2.0",
                    "id": next_id,
                    "method": "textDocument/definition",
                    "params": {
                        "textDocument": { "uri": file_uri },
                        "position": {
                            "line": position.line.saturating_sub(1),
                            "character": utf16_col,
                        }
                    }
                }),
            )?;

            match wait_for_definition_response(&message_rx, next_id, timeout)? {
                DefinitionResponse::Definitions(items) => {
                    definitions = items;
                }
                DefinitionResponse::NoInfo => {}
                DefinitionResponse::RetryableError(reason) => {
                    warn!(
                        tool = "lsp_definition",
                        reason, "retrying definition request"
                    );
                    next_id += 1;
                    std::thread::sleep(Duration::from_millis(75));
                    continue;
                }
            }

            next_id += 1;
            break;
        }

        if !definitions.is_empty() {
            break;
        }
    }

    let _ = write_lsp_message(
        &mut stdin,
        &json!({
            "jsonrpc": "2.0",
            "id": next_id,
            "method": "shutdown",
            "params": serde_json::Value::Null,
        }),
    );
    let _ = wait_for_response(&message_rx, next_id, Duration::from_millis(300));
    let _ = write_lsp_message(
        &mut stdin,
        &json!({
            "jsonrpc": "2.0",
            "method": "exit",
            "params": serde_json::Value::Null,
        }),
    );
    let _ = child.kill();
    let _ = child.wait();

    Ok(definitions)
}

enum HoverResponse {
    Hover(String),
    NoInfo,
    RetryableError(String),
}

enum DefinitionResponse {
    Definitions(Vec<DefinitionLocation>),
    NoInfo,
    RetryableError(String),
}

fn build_hover_positions(source: &str, line: usize, column: usize) -> Result<Vec<HoverPosition>> {
    let lines: Vec<&str> = source.lines().collect();
    if line == 0 || line > lines.len() {
        return Err(ParamsError::Config(format!(
            "Hover line {} is out of range for this file ({} lines)",
            line,
            lines.len()
        )));
    }

    let text = lines[line - 1];
    let char_count = text.chars().count();
    let requested = column.min(char_count.saturating_add(1)).max(1);
    let mut positions = Vec::new();
    let mut seen = HashSet::new();

    push_hover_position(&mut positions, &mut seen, line, requested);

    if let Some((start, end)) = identifier_span_near(text, requested) {
        let preferred = [start + 1, start + 2, ((start + end) / 2) + 1, end];
        for candidate in preferred {
            push_hover_position(&mut positions, &mut seen, line, candidate);
        }
    }

    for candidate in [requested.saturating_sub(1), requested + 1] {
        if candidate >= 1 && candidate <= char_count.saturating_add(1) {
            push_hover_position(&mut positions, &mut seen, line, candidate);
        }
    }

    Ok(positions)
}

fn push_hover_position(
    positions: &mut Vec<HoverPosition>,
    seen: &mut HashSet<(usize, usize)>,
    line: usize,
    column: usize,
) {
    if seen.insert((line, column)) {
        positions.push(HoverPosition { line, column });
    }
}

fn identifier_span_near(text: &str, requested_column: usize) -> Option<(usize, usize)> {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return None;
    }

    let nearest = nearest_identifier_index(&chars, requested_column.saturating_sub(1))?;
    let mut start = nearest;
    while start > 0 && is_identifier_char(chars[start - 1]) {
        start -= 1;
    }

    let mut end = nearest + 1;
    while end < chars.len() && is_identifier_char(chars[end]) {
        end += 1;
    }

    Some((start, end))
}

fn nearest_identifier_index(chars: &[char], requested_index: usize) -> Option<usize> {
    if chars.is_empty() {
        return None;
    }

    let max_index = chars.len().saturating_sub(1);
    let clamped = requested_index.min(max_index);
    if is_identifier_char(chars[clamped]) {
        return Some(clamped);
    }

    for distance in 1..=chars.len() {
        let left = clamped.checked_sub(distance);
        if let Some(index) = left {
            if is_identifier_char(chars[index]) {
                return Some(index);
            }
        }

        let right = clamped + distance;
        if right < chars.len() && is_identifier_char(chars[right]) {
            return Some(right);
        }
    }

    None
}

fn is_identifier_char(ch: char) -> bool {
    ch == '_' || ch.is_alphanumeric()
}

fn line_column_to_utf16(source: &str, line: usize, column: usize) -> Result<usize> {
    let lines: Vec<&str> = source.lines().collect();
    if line == 0 || line > lines.len() {
        return Err(ParamsError::Config(format!(
            "Hover line {} is out of range for this file ({} lines)",
            line,
            lines.len()
        )));
    }

    let text = lines[line - 1];
    let char_count = text.chars().count();
    let clamped = column.min(char_count.saturating_add(1)).max(1);
    let utf16 = text
        .chars()
        .take(clamped.saturating_sub(1))
        .map(char::len_utf16)
        .sum();
    Ok(utf16)
}

pub fn rust_lsp_health_report() -> String {
    info!(tool = "lsp_health_check", "tool called");
    match config::load_with_profile() {
        Ok(cfg) => format_lsp_health_report(&cfg),
        Err(e) => format!("LSP check failed to load config: {e}"),
    }
}

fn resolve_rust_analyzer_command(cfg: &config::Config) -> Result<LspCommandSpec> {
    let probes = probe_rust_analyzer(cfg);
    for probe in &probes {
        if matches!(probe.status, LspProbeStatus::Ready(_)) {
            return Ok(probe.spec.clone());
        }
    }

    Err(ParamsError::Config(format_lsp_probe_failure(&probes)))
}

fn format_lsp_health_report(cfg: &config::Config) -> String {
    let probes = probe_rust_analyzer(cfg);
    let mut output = String::from("Rust LSP check\n\n");

    let mut found_ready = false;
    for probe in &probes {
        match &probe.status {
            LspProbeStatus::Ready(version) => {
                found_ready = true;
                output.push_str(&format!("ready: {} ({version})\n", probe.spec.display));
            }
            LspProbeStatus::Failed(reason) => {
                output.push_str(&format!("failed: {} ({reason})\n", probe.spec.display));
            }
        }
    }

    if !found_ready {
        warn!("rust lsp health check found no runnable server");
        output.push_str("\nFix:\n");
        output.push_str(
            "- Install the rust-analyzer component with `rustup component add rust-analyzer`\n",
        );
        output.push_str(
            "- Or set [lsp].rust_analyzer_path in .local/config.toml to a runnable binary\n",
        );
    }

    output
}

fn probe_rust_analyzer(cfg: &config::Config) -> Vec<LspProbe> {
    let mut probes = Vec::new();

    if let Some(path) = cfg.lsp.rust_analyzer_path.clone() {
        probes.push(run_probe(LspCommandSpec {
            display: format!("configured path {}", path.display()),
            program: path,
            args: Vec::new(),
        }));
        return probes;
    }

    for candidate in discover_rust_analyzer_candidates() {
        probes.push(run_probe(LspCommandSpec {
            display: candidate.display().to_string(),
            program: candidate,
            args: Vec::new(),
        }));
    }

    probes.push(run_probe(LspCommandSpec {
        display: "rustup run stable rust-analyzer".to_string(),
        program: PathBuf::from("rustup"),
        args: vec![
            "run".to_string(),
            "stable".to_string(),
            "rust-analyzer".to_string(),
        ],
    }));

    probes
}

fn format_lsp_probe_failure(probes: &[LspProbe]) -> String {
    let mut message = String::from(
        "rust-analyzer is not runnable. Install it or set [lsp].rust_analyzer_path in .local/config.toml.\n\nTried:\n",
    );
    for probe in probes {
        if let LspProbeStatus::Failed(reason) = &probe.status {
            message.push_str(&format!("- {}: {}\n", probe.spec.display, reason));
        }
    }

    if !rust_analyzer_component_installed() {
        message.push_str(
            "\nThe rust-analyzer rustup component is not installed for the active toolchain.\nRun: rustup component add rust-analyzer\n",
        );
    }

    message
}

fn discover_rust_analyzer_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let mut seen = HashSet::new();

    if let Some(path_var) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path_var) {
            push_candidate(&mut candidates, &mut seen, dir.join("rust-analyzer"));
        }
    }

    if let Some(home) = std::env::var_os("HOME") {
        let home = PathBuf::from(home);
        push_candidate(
            &mut candidates,
            &mut seen,
            home.join(".cargo/bin/rust-analyzer"),
        );
        push_candidate(
            &mut candidates,
            &mut seen,
            home.join(".local/bin/rust-analyzer"),
        );
    }

    push_candidate(
        &mut candidates,
        &mut seen,
        PathBuf::from("/opt/homebrew/bin/rust-analyzer"),
    );
    push_candidate(
        &mut candidates,
        &mut seen,
        PathBuf::from("/usr/local/bin/rust-analyzer"),
    );

    candidates
}

fn push_candidate(candidates: &mut Vec<PathBuf>, seen: &mut HashSet<PathBuf>, candidate: PathBuf) {
    if candidate.exists() && seen.insert(candidate.clone()) {
        candidates.push(candidate);
    }
}

fn run_probe(spec: LspCommandSpec) -> LspProbe {
    let output = Command::new(&spec.program)
        .args(&spec.args)
        .arg("--version")
        .output();

    let status = match output {
        Ok(output) => parse_probe_output(output.status, &output.stdout, &output.stderr),
        Err(e) => LspProbeStatus::Failed(e.to_string()),
    };

    LspProbe { spec, status }
}

fn parse_probe_output(status: ExitStatus, stdout: &[u8], stderr: &[u8]) -> LspProbeStatus {
    if status.success() {
        let version = String::from_utf8_lossy(stdout).trim().to_string();
        let version = if version.is_empty() {
            "version unknown".to_string()
        } else {
            version
        };
        return LspProbeStatus::Ready(version);
    }

    let stderr = String::from_utf8_lossy(stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(stdout).trim().to_string();
    let detail = if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else {
        format!("exit status {}", status.code().unwrap_or(-1))
    };

    LspProbeStatus::Failed(detail)
}

fn rust_analyzer_component_installed() -> bool {
    let output = Command::new("rustup")
        .args(["component", "list", "--installed"])
        .output();

    match output {
        Ok(output) if output.status.success() => String::from_utf8_lossy(&output.stdout)
            .lines()
            .any(|line| line.starts_with("rust-analyzer")),
        _ => false,
    }
}

fn spawn_language_server(server: &LspCommandSpec, project_root: &Path) -> Result<Child> {
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

fn write_lsp_message(stdin: &mut ChildStdin, value: &Value) -> Result<()> {
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

fn wait_for_response(rx: &mpsc::Receiver<Value>, id: u64, timeout: Duration) -> Result<Value> {
    loop {
        let message = rx.recv_timeout(timeout).map_err(|_| {
            ParamsError::Inference(format!(
                "Timed out waiting for language server response after {}ms. \
Increase [lsp].timeout_ms in .local/config.toml if rust-analyzer is slow to start.",
                timeout.as_millis()
            ))
        })?;

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

fn wait_for_hover_response(
    rx: &mpsc::Receiver<Value>,
    id: u64,
    timeout: Duration,
) -> Result<HoverResponse> {
    loop {
        let message = rx.recv_timeout(timeout).map_err(|_| {
            ParamsError::Inference(format!(
                "Timed out waiting for language server response after {}ms. \
Increase [lsp].timeout_ms in .local/config.toml if rust-analyzer is slow to start.",
                timeout.as_millis()
            ))
        })?;

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

fn wait_for_definition_response(
    rx: &mpsc::Receiver<Value>,
    id: u64,
    timeout: Duration,
) -> Result<DefinitionResponse> {
    loop {
        let message = rx.recv_timeout(timeout).map_err(|_| {
            ParamsError::Inference(format!(
                "Timed out waiting for language server response after {}ms. \
Increase [lsp].timeout_ms in .local/config.toml if rust-analyzer is slow to start.",
                timeout.as_millis()
            ))
        })?;

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

fn wait_for_diagnostics(
    rx: &mpsc::Receiver<Value>,
    target_uri: &str,
    timeout: Duration,
) -> Result<Vec<LspDiagnostic>> {
    loop {
        let message = rx.recv_timeout(timeout).map_err(|_| {
            ParamsError::Inference(format!(
                "Timed out waiting for rust-analyzer diagnostics after {}ms. \
Increase [lsp].timeout_ms in .local/config.toml if the workspace is slow to index.",
                timeout.as_millis()
            ))
        })?;

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

fn parse_diagnostic(value: &Value) -> Option<LspDiagnostic> {
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

fn parse_lsp_response_error(message: &Value) -> Option<LspResponseError> {
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

fn format_lsp_response_error(error: &LspResponseError) -> String {
    match &error.data {
        Some(data) if !data.is_empty() => {
            format!("code {}: {} ({data})", error.code, error.message)
        }
        _ => format!("code {}: {}", error.code, error.message),
    }
}

fn is_retryable_lsp_query_error(error: &LspResponseError) -> bool {
    matches!(error.code, -32803 | -32802 | -32801 | -32800 | -32002)
        || error.message.to_ascii_lowercase().contains("cancel")
        || error
            .message
            .to_ascii_lowercase()
            .contains("content modified")
}

fn parse_hover_response(message: &Value) -> Option<String> {
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

fn parse_definition_response(message: &Value) -> Vec<DefinitionLocation> {
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

fn format_diagnostics(display_path: &str, diagnostics: &[LspDiagnostic]) -> String {
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

fn format_hover(display_path: &str, line: usize, column: usize, hover: Option<String>) -> String {
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

fn format_definition(
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

fn path_to_file_uri(path: &Path) -> String {
    let path = path.to_string_lossy();
    let escaped = path
        .replace('%', "%25")
        .replace(' ', "%20")
        .replace('#', "%23")
        .replace('?', "%3F");
    format!("file://{escaped}")
}

fn file_uri_to_path(uri: &str) -> Option<PathBuf> {
    let path = uri.strip_prefix("file://")?;
    let decoded = path
        .replace("%20", " ")
        .replace("%23", "#")
        .replace("%3F", "?")
        .replace("%25", "%");
    Some(PathBuf::from(decoded))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_empty_diagnostics() {
        let formatted = format_diagnostics("src/main.rs", &[]);
        assert_eq!(formatted, "No diagnostics for src/main.rs");
    }

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
    fn builds_file_uri() {
        let uri = path_to_file_uri(Path::new("/tmp/hello world.rs"));
        assert_eq!(uri, "file:///tmp/hello%20world.rs");
    }

    #[test]
    fn probe_failure_includes_stderr() {
        let status = parse_probe_output(
            std::process::Command::new("false")
                .status()
                .expect("status"),
            b"",
            b"missing component",
        );

        match status {
            LspProbeStatus::Failed(reason) => assert!(reason.contains("missing component")),
            LspProbeStatus::Ready(_) => panic!("expected failure"),
        }
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
    fn parses_hover_input() {
        let input = parse_hover_input("src/main.rs:12:8").unwrap();

        assert_eq!(input.line, 12);
        assert_eq!(input.column, 8);
        assert!(input.path.ends_with("src/main.rs"));
    }

    #[test]
    fn finds_identifier_near_requested_hover_column() {
        let positions = build_hover_positions("fn main() {\n    let value = thing;\n}\n", 2, 1)
            .expect("positions");

        assert!(positions.contains(&HoverPosition { line: 2, column: 5 }));
        assert!(positions.contains(&HoverPosition { line: 2, column: 6 }));
    }

    #[test]
    fn converts_columns_to_utf16_offsets() {
        let source = "fn main() {\n    let cafe = \"a😀\";\n}\n";
        let utf16 = line_column_to_utf16(source, 2, 18).expect("utf16");
        assert_eq!(utf16, 17);
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
