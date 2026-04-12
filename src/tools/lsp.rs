mod client;
mod format;
mod paths;
mod position;
mod probe;
mod protocol;
mod transport;
mod types;

use std::fs;

use tracing::info;

use super::{Tool, ToolRunResult};
use crate::config;
use crate::error::{ParamsError, Result};
use crate::safety;

use self::client::{collect_rust_definition, collect_rust_diagnostics, collect_rust_hover};
use self::format::{format_definition, format_diagnostics, format_hover};
use self::paths::{
    display_path, find_rust_project_root, parse_hover_input, resolve_input_path, validate_rust_file,
};
pub use self::probe::rust_lsp_health_report;

pub struct LspDiagnosticsTool;
pub struct LspHoverTool;
pub struct LspDefinitionTool;

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
