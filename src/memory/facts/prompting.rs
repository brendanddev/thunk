use std::time::{SystemTime, UNIX_EPOCH};

use crate::events::FactProvenance;
use crate::inference::Message;

use super::quality::clip_text;
use super::{
    TurnMemoryEvidence, MAX_REPLY_EVIDENCE_CHARS, MAX_TOOL_EVIDENCE_CHARS, NO_FACTS_SENTINEL,
};

pub(crate) trait FactProvenanceExt {
    fn as_db_str(self) -> &'static str;
    fn as_label(self) -> &'static str;
}

impl FactProvenanceExt for FactProvenance {
    fn as_db_str(self) -> &'static str {
        match self {
            FactProvenance::Legacy => "legacy",
            FactProvenance::Verified => "verified",
        }
    }

    fn as_label(self) -> &'static str {
        self.as_db_str()
    }
}

pub(crate) fn parse_provenance(raw: &str) -> FactProvenance {
    match raw {
        "verified" => FactProvenance::Verified,
        _ => FactProvenance::Legacy,
    }
}

pub(crate) fn merged_provenance(
    existing: FactProvenance,
    incoming: FactProvenance,
) -> FactProvenance {
    match (existing, incoming) {
        (FactProvenance::Verified, _) | (_, FactProvenance::Verified) => FactProvenance::Verified,
        _ => FactProvenance::Legacy,
    }
}

pub(crate) fn build_fact_extraction_prompt(evidence: &TurnMemoryEvidence) -> Vec<Message> {
    let mut body = String::new();
    body.push_str(
        "Extract 0-4 verified project-specific technical facts from this evidence pack.\n\
         Only emit a fact when it is directly supported by the evidence.\n\
         Facts must be concrete, resolved outcomes about this repo/workspace: files, symbols, config values, URLs/hosts, commands, or approved tool results.\n\
         Do not include general programming knowledge, language explanations, hedges, TODOs, plans, user intent, or meta commentary.\n\
         If the evidence does not support a project-specific durable fact, reply with NOTHING.\n\
         If no verified facts exist, reply with NOTHING.\n\
         Write one fact per line with no bullets or numbering.\n\n",
    );

    if !evidence.user_prompt.trim().is_empty() {
        body.push_str("User prompt:\n");
        body.push_str(&evidence.user_prompt);
        body.push_str("\n\n");
    }

    if !evidence.summaries.is_empty() {
        body.push_str("Indexed summaries:\n");
        for (path, summary) in &evidence.summaries {
            body.push_str("- ");
            body.push_str(path);
            body.push_str(": ");
            body.push_str(summary);
            body.push('\n');
        }
        body.push('\n');
    }

    if !evidence.tool_results.is_empty() {
        body.push_str("Tool outcomes:\n");
        for tool in &evidence.tool_results {
            body.push_str("- ");
            body.push_str(if tool.approved {
                "approved "
            } else {
                "read-only "
            });
            body.push_str(&tool.tool_name);
            body.push('(');
            body.push_str(&tool.argument);
            body.push_str("): ");
            body.push_str(&clip_text(&tool.output, MAX_TOOL_EVIDENCE_CHARS));
            body.push('\n');
        }
        body.push('\n');
    }

    if let Some(final_response) = &evidence.final_response {
        if !final_response.trim().is_empty() {
            body.push_str("Assistant final reply:\n");
            body.push_str(&clip_text(final_response, MAX_REPLY_EVIDENCE_CHARS));
        }
    }

    vec![
        Message::system(
            "You are a strict technical verifier. Emit only durable facts that are explicitly grounded in the provided evidence.",
        ),
        Message::user(&body),
    ]
}

pub(crate) fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[allow(dead_code)]
pub(crate) fn no_facts_sentinel() -> &'static str {
    NO_FACTS_SENTINEL
}
