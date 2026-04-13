use std::fs;
use std::path::Path;
use std::sync::mpsc::Sender;

use crate::events::{InferenceEvent, ProgressStatus};
use crate::tools::{ToolRegistry, ToolResult};

use super::super::super::runtime::emit_trace;
use super::super::super::session::investigation::InvestigationResolution;
use super::super::super::Message;
use super::super::intent::{suggested_search_query, ToolLoopIntent};
use super::observe::{
    lookup_search_anchor, observed_read_paths, preferred_candidate_path, ranked_search_candidates,
    recent_loaded_file_context_path, search_candidates_in_output_order,
    should_answer_from_anchor_file,
};
use super::parse::{is_internal_tool_loop_path, is_legacy_auto_inspect_path};
use super::readiness::has_relevant_file_evidence;

fn auto_read_best_candidate(
    intent: ToolLoopIntent,
    prompt: &str,
    tools: &ToolRegistry,
    existing_results: &[ToolResult],
    token_tx: &Sender<InferenceEvent>,
) -> Option<ToolResult> {
    let read_paths = observed_read_paths(existing_results);
    let candidate = ranked_search_candidates(intent, prompt, existing_results)
        .into_iter()
        .find(|file| !read_paths.contains(&file.path))?;

    emit_trace(
        token_tx,
        ProgressStatus::Started,
        &format!("reading top candidate {}...", candidate.path),
        false,
    );
    let execution = tools.execute_read_only_tool_calls(&format!("[read_file: {}]", candidate.path));
    let result = execution.results.into_iter().next()?;
    emit_trace(
        token_tx,
        ProgressStatus::Finished,
        &format!("read_file {}", result.argument),
        false,
    );
    Some(result)
}

fn auto_read_best_caller_candidate(
    intent: ToolLoopIntent,
    prompt: &str,
    tools: &ToolRegistry,
    existing_results: &[ToolResult],
    token_tx: &Sender<InferenceEvent>,
) -> Option<ToolResult> {
    suggested_search_query(prompt, intent)?;
    let is_call_site = matches!(intent, ToolLoopIntent::CallSiteLookup);
    let read_paths = observed_read_paths(existing_results);

    let candidate = ranked_search_candidates(intent, prompt, existing_results)
        .into_iter()
        .find(|file| {
            !read_paths.contains(&file.path)
                && preferred_candidate_path(intent, &file.path)
                && !is_internal_tool_loop_path(&file.path)
                && !is_legacy_auto_inspect_path(&file.path)
                && lookup_search_anchor(intent, prompt, file).is_some()
        })?;

    emit_trace(
        token_tx,
        ProgressStatus::Started,
        &format!(
            "reading top {} candidate {}...",
            if is_call_site { "caller" } else { "usage" },
            candidate.path
        ),
        false,
    );
    let execution = tools.execute_read_only_tool_calls(&format!("[read_file: {}]", candidate.path));
    let result = execution.results.into_iter().next()?;
    emit_trace(
        token_tx,
        ProgressStatus::Finished,
        &format!("read_file {}", result.argument),
        false,
    );
    Some(result)
}

fn auto_read_best_flow_followup_candidate(
    prompt: &str,
    tools: &ToolRegistry,
    existing_results: &[ToolResult],
    token_tx: &Sender<InferenceEvent>,
) -> Option<ToolResult> {
    suggested_search_query(prompt, ToolLoopIntent::FlowTrace)?;
    let read_paths = observed_read_paths(existing_results);

    let candidate = search_candidates_in_output_order(existing_results)
        .into_iter()
        .find(|file| {
            !read_paths.contains(&file.path)
                && preferred_candidate_path(ToolLoopIntent::FlowTrace, &file.path)
                && !is_internal_tool_loop_path(&file.path)
                && !is_legacy_auto_inspect_path(&file.path)
        })?;

    emit_trace(
        token_tx,
        ProgressStatus::Started,
        &format!("reading related flow candidate {}...", candidate.path),
        false,
    );
    let execution = tools.execute_read_only_tool_calls(&format!("[read_file: {}]", candidate.path));
    let result = execution.results.into_iter().next()?;
    emit_trace(
        token_tx,
        ProgressStatus::Finished,
        &format!("read_file {}", result.argument),
        false,
    );
    Some(result)
}

fn read_file_result_from_project_root(project_root: &Path, path: &str) -> Option<ToolResult> {
    let resolved = if Path::new(path).is_absolute() {
        Path::new(path).to_path_buf()
    } else {
        project_root.join(path)
    };
    let content = fs::read_to_string(&resolved).ok()?;
    Some(ToolResult {
        tool_name: "read_file".to_string(),
        argument: path.to_string(),
        output: format!(
            "File: {}\nLines: {}\n\n```\n{}\n```",
            path,
            content.lines().count(),
            content
        ),
    })
}

pub(crate) fn bootstrap_tool_results(
    intent: ToolLoopIntent,
    prompt: &str,
    resolution: Option<&InvestigationResolution>,
    base_messages: &[Message],
    project_root: &Path,
    backend_name: &str,
    tools: &ToolRegistry,
    token_tx: &Sender<InferenceEvent>,
) -> Vec<ToolResult> {
    if let Some(path) = resolution
        .and_then(|resolution| resolution.anchored_file.as_deref())
        .map(str::to_string)
        .or_else(|| {
            if should_answer_from_anchor_file(prompt, resolution) {
                recent_loaded_file_context_path(base_messages)
            } else {
                None
            }
        })
    {
        emit_trace(
            token_tx,
            ProgressStatus::Started,
            &format!("bootstrapping read {}...", path),
            false,
        );
        if let Some(read_result) = read_file_result_from_project_root(project_root, &path) {
            emit_trace(
                token_tx,
                ProgressStatus::Finished,
                &format!("read_file {}", read_result.argument),
                false,
            );
            return vec![read_result];
        }
        let execution = tools.execute_read_only_tool_calls(&format!("[read_file: {path}]"));
        let results = execution.results;
        if let Some(read_result) = results.first() {
            emit_trace(
                token_tx,
                ProgressStatus::Finished,
                &format!("read_file {}", read_result.argument),
                false,
            );
        }
        return results;
    }

    if matches!(
        intent,
        ToolLoopIntent::RepoOverview | ToolLoopIntent::DirectoryOverview
    ) {
        return bootstrap_repo_overview_results(intent, project_root, tools, token_tx);
    }

    if !backend_name.contains("llama.cpp") {
        return Vec::new();
    }
    if !matches!(
        intent,
        ToolLoopIntent::CodeNavigation
            | ToolLoopIntent::ConfigLocate
            | ToolLoopIntent::CallSiteLookup
            | ToolLoopIntent::UsageLookup
            | ToolLoopIntent::FlowTrace
    ) {
        return Vec::new();
    }

    let Some(query) = suggested_search_query(prompt, intent) else {
        return Vec::new();
    };

    emit_trace(
        token_tx,
        ProgressStatus::Started,
        &format!("bootstrapping search {}...", query),
        false,
    );
    let execution = tools.execute_read_only_tool_calls(&format!("[search: {query}]"));
    let mut results = execution.results;
    if let Some(search_result) = results.first() {
        emit_trace(
            token_tx,
            ProgressStatus::Finished,
            &format!("search {}", search_result.argument),
            false,
        );
    }

    if matches!(
        intent,
        ToolLoopIntent::CodeNavigation | ToolLoopIntent::ConfigLocate | ToolLoopIntent::FlowTrace
    ) && !has_relevant_file_evidence(intent, prompt, &results)
    {
        if let Some(read_result) =
            auto_read_best_candidate(intent, prompt, tools, &results, token_tx)
        {
            results.push(read_result);
        }
    }

    if matches!(intent, ToolLoopIntent::FlowTrace)
        && !has_relevant_file_evidence(intent, prompt, &results)
    {
        if let Some(read_result) =
            auto_read_best_flow_followup_candidate(prompt, tools, &results, token_tx)
        {
            results.push(read_result);
        }
    }

    if matches!(
        intent,
        ToolLoopIntent::CallSiteLookup | ToolLoopIntent::UsageLookup
    ) && !has_relevant_file_evidence(intent, prompt, &results)
    {
        if let Some(read_result) =
            auto_read_best_caller_candidate(intent, prompt, tools, &results, token_tx)
        {
            results.push(read_result);
        }
    }

    results
}

fn bootstrap_repo_overview_results(
    intent: ToolLoopIntent,
    project_root: &Path,
    tools: &ToolRegistry,
    token_tx: &Sender<InferenceEvent>,
) -> Vec<ToolResult> {
    let mut results = tools.execute_read_only_tool_calls("[list_dir: .]").results;
    if !results.is_empty() {
        emit_trace(token_tx, ProgressStatus::Finished, "list_dir .", false);
    }

    for path in repo_bootstrap_read_targets(project_root, intent) {
        let execution = tools.execute_read_only_tool_calls(&format!("[read_file: {path}]"));
        if let Some(result) = execution.results.into_iter().next() {
            emit_trace(
                token_tx,
                ProgressStatus::Finished,
                &format!("read_file {}", result.argument),
                false,
            );
            results.push(result);
        }
    }

    results
}

pub(super) fn repo_bootstrap_read_targets(
    project_root: &Path,
    intent: ToolLoopIntent,
) -> Vec<String> {
    let candidates = match intent {
        ToolLoopIntent::RepoOverview => {
            vec!["Cargo.toml", "README.md", "src/main.rs", "src/lib.rs"]
        }
        ToolLoopIntent::DirectoryOverview => vec!["Cargo.toml", "src/main.rs", "src/lib.rs"],
        _ => Vec::new(),
    };

    candidates
        .into_iter()
        .filter(|path| project_root.join(path).is_file())
        .map(str::to_string)
        .collect()
}
