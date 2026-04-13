mod followup;
mod intent;
mod parse;
mod plan;
mod summarize;
mod types;

pub(super) use types::AutoInspectIntent;

#[cfg(test)]
#[allow(unused_imports)]
pub(super) use followup::{
    choose_followup_read_steps, config_scope_label, declaration_lines_with_numbers,
    definition_match_lines_with_numbers, file_display_name, filter_non_test_hits,
    format_numbered_hits, is_auto_inspection_read_candidate, is_code_path, is_config_path,
    is_definition_like_line, is_doc_path, is_feature_trace_anchor_line,
    is_implementation_definition_line, match_lines_with_numbers, preferred_config_paths,
    preferred_workflow_paths, primary_config_locations, primary_definition_location,
    rank_search_files, summarize_feature_trace_hits, summarize_workflow_read,
    test_module_start_line,
};
#[cfg(test)]
#[allow(unused_imports)]
pub(super) use intent::{
    detect_auto_inspect_intent, extract_auto_inspect_query, normalize_intent_text,
};
#[cfg(test)]
#[allow(unused_imports)]
pub(super) use parse::{
    clip_inline, parse_list_dir_output, parse_read_file_output, parse_search_output,
};
#[cfg(test)]
#[allow(unused_imports)]
pub(super) use plan::{auto_inspection_budget, plan_auto_inspection};
#[cfg(test)]
#[allow(unused_imports)]
pub(super) use summarize::{
    format_entry_list, summarize_cargo_manifest, summarize_entrypoint, summarize_readme,
    synthesize_auto_inspection_context, top_level_repo_type,
};
#[cfg(test)]
#[allow(unused_imports)]
pub(super) use types::{
    AutoInspectBudget, AutoInspectPlan, AutoInspectStep, SearchFileHit, SearchLineHit,
};

#[cfg(test)]
mod tests;
