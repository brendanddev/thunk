use super::chrome::active_cursor_span;
use super::util::{
    blank_line, build_gutter_lines, label_width, single_span, wrap_text, CONVERSATION_GUTTER,
    SYSTEM_GUTTER,
};
use super::{RenderBlock, RenderModel, StyledLine, StyledSpan};
use crate::tui::renderer::style::{PackedStyle, Theme};
use crate::tui::state::{AppState, ChatMessage, Role};

pub(crate) fn build_render_model(
    state: &mut AppState,
    theme: Theme,
    width: u16,
    cached_block: impl FnMut(&ChatMessage, bool, u16) -> Vec<StyledLine>,
) -> RenderModel {
    let top_bar = super::chrome::build_top_bar(state, theme, width);
    let transcript = build_transcript(state, theme, width, cached_block);
    let approval = super::approval::build_approval(state, theme, width);
    let activity = super::chrome::build_activity_line(state, theme, width);
    let (composer, composer_cursor) =
        super::composer::build_composer(state, theme, width, approval.as_deref(), activity);

    RenderModel {
        top_bar,
        transcript,
        composer,
        composer_cursor,
    }
}

pub(crate) fn build_transcript(
    state: &mut AppState,
    theme: Theme,
    width: u16,
    mut cached_block: impl FnMut(&ChatMessage, bool, u16) -> Vec<StyledLine>,
) -> Vec<RenderBlock> {
    let mut blocks = Vec::new();
    let active_assistant_id = if state.is_generating {
        state
            .messages
            .iter()
            .rev()
            .find(|message| message.role == Role::Assistant)
            .map(|message| message.id)
    } else {
        None
    };
    if state.messages.is_empty() {
        blocks.push(RenderBlock {
            message_id: None,
            lines: vec![single_span(
                "type a message, or / for commands.",
                theme.dim(),
            )],
        });
        return blocks;
    }

    for (idx, message) in state.messages.iter().enumerate() {
        let next = state.messages.get(idx + 1);
        if message.transcript.collapsible {
            let focused = state.is_focused_collapsible(message.id);
            let mut lines = cached_block(message, focused, width);
            if needs_transcript_gap(message, next) {
                lines.push(blank_line());
            }
            blocks.push(RenderBlock {
                message_id: Some(message.id),
                lines,
            });
        } else {
            let is_active_assistant =
                message.role == Role::Assistant && active_assistant_id == Some(message.id);
            let mut lines =
                build_standard_message(message, theme, width, is_active_assistant, state.tick);
            if needs_transcript_gap(message, next) {
                lines.push(blank_line());
            }
            blocks.push(RenderBlock {
                message_id: None,
                lines,
            });
        }
    }
    blocks
}

pub(crate) fn build_transcript_block(
    message: &ChatMessage,
    theme: Theme,
    width: u16,
    focused: bool,
) -> Vec<StyledLine> {
    if message.transcript.collapsed {
        build_collapsed_context(message, theme, width, focused)
    } else {
        build_expanded_context(message, theme, width, focused)
    }
}

pub(crate) fn build_standard_message(
    message: &ChatMessage,
    theme: Theme,
    width: u16,
    is_active_assistant: bool,
    tick: u64,
) -> Vec<StyledLine> {
    match message.role {
        Role::User => build_badged_message(
            "you",
            theme.badge_user(),
            &message.content,
            theme,
            width,
            None,
        ),
        Role::Assistant => build_badged_message(
            "params",
            theme.badge_assistant(),
            if message.content.is_empty() && !is_active_assistant {
                "▍"
            } else {
                &message.content
            },
            theme,
            width,
            is_active_assistant.then(|| active_cursor_span(theme, tick)),
        ),
        Role::System => build_gutter_lines(SYSTEM_GUTTER, theme.dim(), &message.content, width),
    }
}

fn build_badged_message(
    label: &str,
    badge_style: PackedStyle,
    body: &str,
    theme: Theme,
    width: u16,
    active_cursor: Option<StyledSpan>,
) -> Vec<StyledLine> {
    let mut lines = Vec::new();
    let continuation_indent = " ".repeat(label_width(label) + 2);
    let body_width = width
        .saturating_sub((label_width(label) + 5).min(u16::MAX as usize) as u16)
        .max(8);
    let wrapped = wrap_text(body, body_width);
    if wrapped.is_empty() {
        lines.push(StyledLine {
            spans: vec![
                StyledSpan {
                    text: CONVERSATION_GUTTER.to_string(),
                    style: theme.border(),
                },
                StyledSpan {
                    text: label.to_string(),
                    style: badge_style,
                },
            ],
        });
        return lines;
    }

    lines.push(StyledLine {
        spans: vec![
            StyledSpan {
                text: CONVERSATION_GUTTER.to_string(),
                style: theme.border(),
            },
            StyledSpan {
                text: label.to_string(),
                style: badge_style,
            },
            StyledSpan {
                text: "  ".to_string(),
                style: theme.base(),
            },
            StyledSpan {
                text: wrapped[0].clone(),
                style: theme.base(),
            },
        ],
    });

    for line in wrapped.into_iter().skip(1) {
        lines.push(StyledLine {
            spans: vec![
                StyledSpan {
                    text: CONVERSATION_GUTTER.to_string(),
                    style: theme.border(),
                },
                StyledSpan {
                    text: continuation_indent.clone(),
                    style: theme.base(),
                },
                StyledSpan {
                    text: line,
                    style: theme.base(),
                },
            ],
        });
    }
    if let Some(cursor_span) = active_cursor {
        if let Some(last) = lines.last_mut() {
            last.spans.push(cursor_span);
        }
    }
    lines
}

pub(crate) fn build_collapsed_context(
    message: &ChatMessage,
    theme: Theme,
    width: u16,
    focused: bool,
) -> Vec<StyledLine> {
    let mut lines = Vec::new();
    let edge = if focused { "▸" } else { "›" };
    let edge_style = if focused {
        theme.border_active()
    } else {
        theme.border()
    };
    let summary = message.transcript.summary.as_deref().unwrap_or("context");
    lines.push(StyledLine {
        spans: vec![
            StyledSpan {
                text: edge.to_string(),
                style: edge_style,
            },
            StyledSpan {
                text: " ".to_string(),
                style: theme.base(),
            },
            StyledSpan {
                text: summary.to_string(),
                style: if focused {
                    theme.chip_accent().with_underline()
                } else {
                    theme.muted()
                },
            },
        ],
    });

    for preview in message
        .transcript
        .preview_lines
        .iter()
        .take(if focused { 2 } else { 1 })
        .flat_map(|line| wrap_text(line, width.saturating_sub(4)))
    {
        lines.push(StyledLine {
            spans: vec![
                StyledSpan {
                    text: edge.to_string(),
                    style: edge_style,
                },
                StyledSpan {
                    text: " ".to_string(),
                    style: theme.base(),
                },
                StyledSpan {
                    text: preview,
                    style: theme.muted(),
                },
            ],
        });
    }
    if focused {
        lines.push(StyledLine {
            spans: vec![
                StyledSpan {
                    text: edge.to_string(),
                    style: edge_style,
                },
                StyledSpan {
                    text: " Ctrl+O".to_string(),
                    style: theme.dim(),
                },
            ],
        });
    }
    lines
}

fn build_expanded_context(
    message: &ChatMessage,
    theme: Theme,
    width: u16,
    focused: bool,
) -> Vec<StyledLine> {
    let edge = if focused { "▸" } else { "›" };
    let edge_style = if focused {
        theme.border_active()
    } else {
        theme.border()
    };
    let summary = message.transcript.summary.as_deref().unwrap_or("context");
    let mut lines = vec![StyledLine {
        spans: vec![
            StyledSpan {
                text: edge.to_string(),
                style: edge_style,
            },
            StyledSpan {
                text: " ".to_string(),
                style: theme.base(),
            },
            StyledSpan {
                text: summary.to_string(),
                style: if focused {
                    theme.chip_accent().with_underline()
                } else {
                    theme.muted().with_bold()
                },
            },
        ],
    }];
    for body_line in message
        .content
        .lines()
        .flat_map(|line| wrap_text(line, width.saturating_sub(4)))
    {
        lines.push(StyledLine {
            spans: vec![
                StyledSpan {
                    text: edge.to_string(),
                    style: edge_style,
                },
                StyledSpan {
                    text: " ".to_string(),
                    style: theme.base(),
                },
                StyledSpan {
                    text: body_line,
                    style: theme.dim(),
                },
            ],
        });
    }
    if focused {
        lines.push(StyledLine {
            spans: vec![
                StyledSpan {
                    text: edge.to_string(),
                    style: edge_style,
                },
                StyledSpan {
                    text: " Ctrl+O".to_string(),
                    style: theme.dim(),
                },
            ],
        });
    }
    lines
}

fn needs_transcript_gap(current: &ChatMessage, next: Option<&ChatMessage>) -> bool {
    let Some(next) = next else {
        return false;
    };
    transcript_block_kind(current) != transcript_block_kind(next)
}

fn transcript_block_kind(message: &ChatMessage) -> u8 {
    if message.transcript.collapsible {
        0
    } else {
        match message.role {
            Role::User => 1,
            Role::Assistant => 2,
            Role::System => 3,
        }
    }
}
