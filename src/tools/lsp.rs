// src/tools/lsp.rs
//
// Rust-first LSP diagnostics tool.

use std::fs;
use std::collections::HashSet;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use serde_json::{json, Value};
use tracing::{info, warn};

use crate::config;
use crate::error::{ParamsError, Result};
use super::{Tool, ToolRunResult};

pub struct LspDiagnosticsTool;

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
                "lsp_diagnostics requires a file path".to_string()
            ));
        }

        let cfg = config::load()?;
        let cwd = std::env::current_dir()?;
        let path = resolve_input_path(&cwd, requested)?;
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

fn resolve_input_path(cwd: &Path, requested: &str) -> Result<PathBuf> {
    let candidate = Path::new(requested);
    let joined = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        cwd.join(candidate)
    };

    let normalized = joined
        .canonicalize()
        .map_err(|_| ParamsError::Config(format!("File not found: {}", joined.display())))?;

    Ok(normalized)
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
            "The first LSP slice currently supports Rust `.rs` files only".to_string()
        ));
    }

    Ok(())
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
    let mut stdin = child.stdin.take().ok_or_else(|| {
        ParamsError::Inference("Failed to open rust-analyzer stdin".to_string())
    })?;
    let stdout = child.stdout.take().ok_or_else(|| {
        ParamsError::Inference("Failed to open rust-analyzer stdout".to_string())
    })?;

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

    write_lsp_message(&mut stdin, &json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "processId": serde_json::Value::Null,
            "rootUri": root_uri,
            "capabilities": {},
            "clientInfo": {
                "name": "params-cli",
                "version": env!("CARGO_PKG_VERSION"),
            }
        }
    }))?;
    wait_for_response(&message_rx, 1, timeout)?;

    write_lsp_message(&mut stdin, &json!({
        "jsonrpc": "2.0",
        "method": "initialized",
        "params": {}
    }))?;

    write_lsp_message(&mut stdin, &json!({
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
    }))?;

    let diagnostics = wait_for_diagnostics(&message_rx, &file_uri, timeout)?;

    let _ = write_lsp_message(&mut stdin, &json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "shutdown",
        "params": serde_json::Value::Null,
    }));
    let _ = wait_for_response(&message_rx, 2, Duration::from_millis(300));
    let _ = write_lsp_message(&mut stdin, &json!({
        "jsonrpc": "2.0",
        "method": "exit",
        "params": serde_json::Value::Null,
    }));
    let _ = child.kill();
    let _ = child.wait();

    Ok(diagnostics)
}

pub fn rust_lsp_health_report() -> String {
    info!(tool = "lsp_health_check", "tool called");
    match config::load() {
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
        output.push_str("- Install the rust-analyzer component with `rustup component add rust-analyzer`\n");
        output.push_str("- Or set [lsp].rust_analyzer_path in .local/config.toml to a runnable binary\n");
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
        args: vec!["run".to_string(), "stable".to_string(), "rust-analyzer".to_string()],
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
        push_candidate(&mut candidates, &mut seen, home.join(".cargo/bin/rust-analyzer"));
        push_candidate(&mut candidates, &mut seen, home.join(".local/bin/rust-analyzer"));
    }

    push_candidate(&mut candidates, &mut seen, PathBuf::from("/opt/homebrew/bin/rust-analyzer"));
    push_candidate(&mut candidates, &mut seen, PathBuf::from("/usr/local/bin/rust-analyzer"));

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
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .any(|line| line.starts_with("rust-analyzer"))
        }
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
        .map_err(|e| ParamsError::Inference(format!(
            "Failed to start rust-analyzer via {}: {e}",
            server.display
        )))
}

fn write_lsp_message(stdin: &mut ChildStdin, value: &Value) -> Result<()> {
    let payload = value.to_string();
    write!(stdin, "Content-Length: {}\r\n\r\n{}", payload.len(), payload)?;
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
                "Language server closed the connection".to_string()
            ));
        }

        if line == "\r\n" || line == "\n" {
            break;
        }

        if let Some((name, value)) = line.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                let parsed = value
                    .trim()
                    .parse::<usize>()
                    .map_err(|e| ParamsError::Inference(format!(
                        "Invalid LSP Content-Length header: {e}"
                    )))?;
                content_length = Some(parsed);
            }
        }
    }

    let length = content_length.ok_or_else(|| {
        ParamsError::Inference("Missing LSP Content-Length header".to_string())
    })?;
    let mut payload = vec![0; length];
    reader.read_exact(&mut payload)?;
    serde_json::from_slice(&payload)
        .map_err(|e| ParamsError::Inference(format!("Invalid LSP JSON payload: {e}")))
}

fn wait_for_response(
    rx: &mpsc::Receiver<Value>,
    id: u64,
    timeout: Duration,
) -> Result<Value> {
    loop {
        let message = rx.recv_timeout(timeout).map_err(|_| {
            ParamsError::Inference("Timed out waiting for language server response".to_string())
        })?;

        if message.get("id").and_then(|v| v.as_u64()) == Some(id) {
            if let Some(error) = message.get("error") {
                return Err(ParamsError::Inference(format!(
                    "Language server error: {error}"
                )));
            }
            return Ok(message);
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
            ParamsError::Inference(
                "Timed out waiting for rust-analyzer diagnostics".to_string()
            )
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

fn path_to_file_uri(path: &Path) -> String {
    let path = path.to_string_lossy();
    let escaped = path
        .replace('%', "%25")
        .replace(' ', "%20")
        .replace('#', "%23")
        .replace('?', "%3F");
    format!("file://{escaped}")
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
}
