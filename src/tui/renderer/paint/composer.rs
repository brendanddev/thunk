use super::util::{blank_line, single_span, truncate, PALETTE_NAME_COL};
use super::{StyledLine, StyledSpan};
use crate::tui::renderer::style::Theme;
use crate::tui::state::AppState;

pub(crate) fn build_composer(
    state: &AppState,
    theme: Theme,
    width: u16,
    approval: Option<&[StyledLine]>,
    activity: Option<StyledLine>,
) -> (Vec<StyledLine>, (u16, u16)) {
    let inner_width = width.saturating_sub(2).max(8) as usize;
    let max_visible_rows = 8usize;
    let (visible_rows, cursor_row, cursor_col) =
        state.input_display_lines(inner_width, max_visible_rows);
    let mut lines = vec![blank_line()];

    if let Some(activity_line) = activity {
        lines.push(activity_line);
    }
    if let Some(approval_lines) = approval {
        for line in approval_lines {
            lines.push(line.clone());
        }
    }
    let prompt_offset = lines.len() as u16;

    let prompt_marker = if state.is_reverse_search_active() {
        "? "
    } else if state.is_command_launcher_active() {
        ": "
    } else if state.has_pending_action() {
        "! "
    } else {
        "› "
    };
    let prompt_style = if state.has_pending_action() {
        theme.chip_warning()
    } else if state.is_generating {
        theme.badge_assistant()
    } else {
        theme.chip_accent()
    };

    for (idx, row) in visible_rows.iter().enumerate() {
        lines.push(StyledLine {
            spans: vec![
                StyledSpan {
                    text: if idx == 0 {
                        prompt_marker.to_string()
                    } else {
                        "  ".to_string()
                    },
                    style: prompt_style,
                },
                StyledSpan {
                    text: row.clone(),
                    style: if state.is_generating {
                        theme.muted()
                    } else {
                        theme.base()
                    },
                },
            ],
        });
    }

    if let Some((query, entries)) = state.command_launcher_view(5) {
        let selected_entry = entries
            .iter()
            .find(|(_, selected)| *selected)
            .map(|(entry, _)| entry.clone());
        if !query.is_empty() {
            lines.push(StyledLine {
                spans: vec![StyledSpan {
                    text: format!("/ {query}"),
                    style: theme.dim(),
                }],
            });
        }
        for (entry, selected) in entries {
            let name = truncate(&entry.name, PALETTE_NAME_COL);
            let pad = PALETTE_NAME_COL.saturating_sub(name.chars().count());
            let padded_name = format!("{}{}", name, " ".repeat(pad));
            lines.push(StyledLine {
                spans: vec![
                    StyledSpan {
                        text: if selected {
                            "→ ".to_string()
                        } else {
                            "  ".to_string()
                        },
                        style: if selected {
                            theme.chip_accent()
                        } else {
                            theme.dim()
                        },
                    },
                    StyledSpan {
                        text: padded_name,
                        style: if selected {
                            theme.chip_accent()
                        } else {
                            theme.base()
                        },
                    },
                    StyledSpan {
                        text: "  ".to_string(),
                        style: theme.dim(),
                    },
                    StyledSpan {
                        text: truncate(
                            &entry.description,
                            width.saturating_sub(PALETTE_NAME_COL as u16 + 4) as usize,
                        ),
                        style: theme.muted(),
                    },
                ],
            });
        }
        if let Some(selected) = selected_entry {
            lines.push(single_span(
                &format!(
                    "  {}",
                    truncate(&selected.usage, width.saturating_sub(2) as usize)
                ),
                theme.dim(),
            ));
            if !selected.aliases.is_empty() {
                lines.push(single_span(
                    &format!(
                        "  aka {}",
                        truncate(
                            &selected.aliases.join(", "),
                            width.saturating_sub(6) as usize
                        )
                    ),
                    theme.dim(),
                ));
            }
        }
    } else if let Some((query, current)) = state.reverse_search_view() {
        let query_display = if query.is_empty() {
            "·".to_string()
        } else {
            truncate(&query, 20)
        };
        let match_width = width.saturating_sub(query_display.chars().count() as u16 + 4) as usize;
        lines.push(StyledLine {
            spans: vec![
                StyledSpan {
                    text: query_display,
                    style: theme.dim(),
                },
                StyledSpan {
                    text: "  ".to_string(),
                    style: theme.dim(),
                },
                StyledSpan {
                    text: truncate(&current, match_width),
                    style: theme.muted(),
                },
            ],
        });
    } else if !state.autocomplete_preview_items(4).is_empty() {
        for (item, selected) in state.autocomplete_preview_items(4) {
            lines.push(StyledLine {
                spans: vec![
                    StyledSpan {
                        text: if selected {
                            "→ ".to_string()
                        } else {
                            "  ".to_string()
                        },
                        style: if selected {
                            theme.chip_accent()
                        } else {
                            theme.dim()
                        },
                    },
                    StyledSpan {
                        text: truncate(&item, width.saturating_sub(4) as usize),
                        style: if selected {
                            theme.chip_accent()
                        } else {
                            theme.muted()
                        },
                    },
                ],
            });
        }
    }

    let cursor_x = 2 + cursor_col as u16;
    let cursor_y = prompt_offset + cursor_row as u16;
    (lines, (cursor_x, cursor_y))
}
