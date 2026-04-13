mod prompting;
mod quality;
mod store;

use crate::events::FactProvenance;
use rusqlite::Connection;

const NO_FACTS_SENTINEL: &str = "NOTHING";
const MAX_FACT_LEN: usize = 300;
const MIN_FACT_LEN: usize = 20;
const MAX_TOOL_EVIDENCE_CHARS: usize = 600;
const MAX_REPLY_EVIDENCE_CHARS: usize = 1200;
const GENERIC_ANCHOR_TOKENS: &[&str] = &[
    "rust",
    "cargo",
    "box",
    "rc",
    "refcell",
    "unsafecell",
    "hashmap",
    "sqlite",
    "java",
    "jvm",
    "python",
    "go",
    "javascript",
    "typescript",
    "serde",
    "toml",
    "ratatui",
    "crossterm",
    "clap",
    "tracing",
    "sha2",
    "ureq",
];

pub struct FactStore {
    conn: Connection,
}

#[derive(Debug, Clone)]
pub struct StoredFact {
    pub content: String,
    pub provenance: FactProvenance,
}

#[derive(Debug, Clone)]
pub struct ToolEvidence {
    pub tool_name: String,
    pub argument: String,
    pub output: String,
    pub approved: bool,
}

#[derive(Debug, Clone, Default)]
pub struct TurnMemoryEvidence {
    pub user_prompt: String,
    pub summaries: Vec<(String, String)>,
    pub tool_results: Vec<ToolEvidence>,
    pub final_response: Option<String>,
}

pub struct ConsolidationStats {
    pub ttl_pruned: usize,
    pub dedup_removed: usize,
    pub cap_removed: usize,
}

enum StoreFactOutcome {
    Stored(StoredFact),
    Duplicate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum SkippedFactReason {
    Quality,
    Hedged,
    Unresolved,
    Unanchored,
    Duplicate,
}

impl SkippedFactReason {
    fn label(self) -> &'static str {
        match self {
            Self::Quality => "filtered quality",
            Self::Hedged => "hedged or uncertain",
            Self::Unresolved => "unresolved or intent",
            Self::Unanchored => "missing project anchor",
            Self::Duplicate => "duplicate",
        }
    }
}

impl TurnMemoryEvidence {
    pub fn new(prompt: String, summaries: Vec<(String, String)>) -> Self {
        Self {
            user_prompt: prompt,
            summaries,
            tool_results: Vec::new(),
            final_response: None,
        }
    }

    pub fn record_tool_result(
        &mut self,
        tool_name: impl Into<String>,
        argument: impl Into<String>,
        output: impl Into<String>,
        approved: bool,
    ) {
        self.tool_results.push(ToolEvidence {
            tool_name: tool_name.into(),
            argument: argument.into(),
            output: output.into(),
            approved,
        });
    }

    pub fn set_final_response(&mut self, response: impl Into<String>) {
        self.final_response = Some(response.into());
    }

    fn has_material_evidence(&self) -> bool {
        !self.user_prompt.trim().is_empty()
            || !self.summaries.is_empty()
            || !self.tool_results.is_empty()
            || self
                .final_response
                .as_ref()
                .map(|response| !response.trim().is_empty())
                .unwrap_or(false)
    }
}

#[cfg(test)]
#[allow(unused_imports)]
use prompting::{
    build_fact_extraction_prompt, merged_provenance, now_secs, parse_provenance, FactProvenanceExt,
};
#[cfg(test)]
#[allow(unused_imports)]
use quality::{
    are_near_duplicate, clip_text, collect_anchor_tokens, contains_anchor,
    contains_hedged_language, contains_project_anchor_text, contains_unresolved_language,
    evidence_anchors, is_project_anchor_token, is_quality_fact, is_retrievable_project_fact,
    looks_like_code_snippet, looks_like_summary_boilerplate, tokenize, validate_candidate_fact,
};
#[cfg(test)]
#[allow(unused_imports)]
use std::collections::HashSet;
#[cfg(test)]
#[allow(unused_imports)]
use store::StoreFactTestExt;

#[cfg(test)]
mod tests;
