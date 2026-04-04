use std::sync::mpsc::Sender;

use crate::config;
use crate::events::{BudgetUpdate, CacheUpdate, InferenceEvent};

#[derive(Default)]
pub(super) struct SessionBudget {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub estimated_cost_usd: f64,
    pub has_cost_estimate: bool,
}

#[derive(Default)]
pub(super) struct SessionCacheStats {
    pub hits: usize,
    pub misses: usize,
    pub tokens_saved: usize,
}

pub(super) fn estimate_message_tokens(messages: &[super::Message]) -> usize {
    let mut total = 0usize;
    for message in messages {
        total = total
            .saturating_add(estimate_text_tokens(&message.role))
            .saturating_add(estimate_text_tokens(&message.content))
            .saturating_add(4);
    }
    total.saturating_add(2)
}

pub(super) fn estimate_text_tokens(text: &str) -> usize {
    let chars = text.chars().count();
    if chars == 0 {
        0
    } else {
        chars.div_ceil(4)
    }
}

fn estimate_generation_cost_usd(
    cfg: &config::Config,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> Option<f64> {
    match cfg.backend.as_str() {
        "llama_cpp" | "ollama" => Some(0.0),
        "openai_compat" => {
            let input = cfg.budget.input_cost_per_million?;
            let output = cfg.budget.output_cost_per_million.unwrap_or(input);

            Some(
                (prompt_tokens as f64 / 1_000_000.0) * input
                    + (completion_tokens as f64 / 1_000_000.0) * output,
            )
        }
        _ => None,
    }
}

pub(super) fn record_generation_budget(
    cfg: &config::Config,
    budget: &mut SessionBudget,
    token_tx: &Sender<InferenceEvent>,
    prompt_tokens: usize,
    response: &str,
) {
    let completion_tokens = estimate_text_tokens(response);
    budget.prompt_tokens = budget.prompt_tokens.saturating_add(prompt_tokens);
    budget.completion_tokens = budget.completion_tokens.saturating_add(completion_tokens);

    if let Some(cost) = estimate_generation_cost_usd(cfg, prompt_tokens, completion_tokens) {
        budget.estimated_cost_usd += cost;
        budget.has_cost_estimate = true;
    }

    emit_budget_update(budget, token_tx);
}

pub(super) fn emit_budget_update(budget: &SessionBudget, token_tx: &Sender<InferenceEvent>) {
    let _ = token_tx.send(InferenceEvent::Budget(BudgetUpdate {
        prompt_tokens: budget.prompt_tokens,
        completion_tokens: budget.completion_tokens,
        total_tokens: budget
            .prompt_tokens
            .saturating_add(budget.completion_tokens),
        estimated_cost_usd: if budget.has_cost_estimate {
            Some(budget.estimated_cost_usd)
        } else {
            None
        },
    }));
}

pub(super) fn emit_cache_update(
    stats: &SessionCacheStats,
    last_hit: bool,
    token_tx: &Sender<InferenceEvent>,
) {
    let _ = token_tx.send(InferenceEvent::Cache(CacheUpdate {
        last_hit,
        hits: stats.hits,
        misses: stats.misses,
        tokens_saved: stats.tokens_saved,
    }));
}
