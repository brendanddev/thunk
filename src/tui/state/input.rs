use crate::commands::CommandSuggestion;

use super::{AppState, DirtySections};

impl AppState {
    pub fn submit_input(&mut self) -> String {
        let submitted = self.input.clone();
        if !submitted.is_empty() {
            self.input_history.push(submitted.clone());
        }
        self.history_cursor = None;
        self.history_draft = None;
        self.exit_reverse_search();
        self.exit_command_launcher();
        self.cursor = 0;
        self.scroll_offset = 0;
        self.clear_autocomplete();
        self.mark_dirty(DirtySections::INPUT | DirtySections::CHAT | DirtySections::SIDEBAR);
        std::mem::take(&mut self.input)
    }

    pub fn insert_char(&mut self, c: char) {
        self.input.insert(self.cursor, c);
        self.cursor += c.len_utf8();
        self.history_cursor = None;
        self.history_draft = None;
        self.exit_reverse_search();
        self.exit_command_launcher();
        self.clear_autocomplete();
        self.mark_dirty(DirtySections::INPUT);
    }

    pub fn insert_str(&mut self, s: &str) {
        self.input.insert_str(self.cursor, s);
        self.cursor += s.len();
        self.history_cursor = None;
        self.history_draft = None;
        self.exit_reverse_search();
        self.exit_command_launcher();
        self.clear_autocomplete();
        self.mark_dirty(DirtySections::INPUT);
    }

    pub fn insert_newline(&mut self) {
        self.insert_char('\n');
    }

    pub fn delete_char_before(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let mut prev = self.cursor - 1;
        while !self.input.is_char_boundary(prev) {
            prev -= 1;
        }
        self.input.remove(prev);
        self.cursor = prev;
        self.exit_command_launcher();
        self.clear_autocomplete();
        self.mark_dirty(DirtySections::INPUT);
    }

    pub fn delete_word_before(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let before = &self.input[..self.cursor];
        let trim_end = before.trim_end_matches(' ').len();
        let word_start = before[..trim_end].rfind(' ').map(|i| i + 1).unwrap_or(0);
        self.input.drain(word_start..self.cursor);
        self.cursor = word_start;
        self.exit_command_launcher();
        self.clear_autocomplete();
        self.mark_dirty(DirtySections::INPUT);
    }

    pub fn cursor_left(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let mut prev = self.cursor - 1;
        while !self.input.is_char_boundary(prev) {
            prev -= 1;
        }
        self.cursor = prev;
        self.mark_dirty(DirtySections::INPUT);
    }

    pub fn cursor_right(&mut self) {
        if self.cursor >= self.input.len() {
            return;
        }
        let mut next = self.cursor + 1;
        while !self.input.is_char_boundary(next) {
            next += 1;
        }
        self.cursor = next;
        self.mark_dirty(DirtySections::INPUT);
    }

    pub fn cursor_home(&mut self) {
        self.cursor = self.current_line_start();
        self.mark_dirty(DirtySections::INPUT);
    }

    pub fn cursor_end(&mut self) {
        self.cursor = self.current_line_end();
        self.mark_dirty(DirtySections::INPUT);
    }

    pub fn clear_input(&mut self) {
        self.input.clear();
        self.cursor = 0;
        self.history_cursor = None;
        self.history_draft = None;
        self.exit_reverse_search();
        self.exit_command_launcher();
        self.clear_autocomplete();
        self.mark_dirty(DirtySections::INPUT);
    }

    pub fn recall_previous_input(&mut self) -> bool {
        if self.input_history.is_empty() {
            return false;
        }

        let next_index = match self.history_cursor {
            Some(current) if current > 0 => current - 1,
            Some(current) => current,
            None => {
                self.history_draft = Some(self.input.clone());
                self.input_history.len() - 1
            }
        };
        self.history_cursor = Some(next_index);
        self.set_input_text(self.input_history[next_index].clone());
        true
    }

    pub fn recall_next_input(&mut self) -> bool {
        let Some(current) = self.history_cursor else {
            return false;
        };

        if current + 1 < self.input_history.len() {
            self.history_cursor = Some(current + 1);
            self.set_input_text(self.input_history[current + 1].clone());
        } else {
            let draft = self.history_draft.take().unwrap_or_default();
            self.history_cursor = None;
            self.set_input_text(draft);
        }
        true
    }

    pub fn is_reverse_search_active(&self) -> bool {
        self.reverse_search_active
    }

    pub fn is_command_launcher_active(&self) -> bool {
        self.command_launcher_active
    }

    pub fn activate_reverse_search(&mut self) -> bool {
        if self.input_history.is_empty() {
            return false;
        }
        if !self.reverse_search_active {
            self.reverse_search_active = true;
            self.reverse_search_query.clear();
            self.reverse_search_selection = 0;
            self.reverse_search_draft = Some(self.input.clone());
        }
        self.apply_reverse_search_match();
        self.mark_dirty(DirtySections::INPUT);
        true
    }

    pub fn reverse_search_push_char(&mut self, c: char) {
        if !self.reverse_search_active {
            return;
        }
        self.reverse_search_query.push(c);
        self.reverse_search_selection = 0;
        self.apply_reverse_search_match();
        self.mark_dirty(DirtySections::INPUT);
    }

    pub fn reverse_search_backspace(&mut self) {
        if !self.reverse_search_active {
            return;
        }
        self.reverse_search_query.pop();
        self.reverse_search_selection = 0;
        self.apply_reverse_search_match();
        self.mark_dirty(DirtySections::INPUT);
    }

    pub fn reverse_search_cycle(&mut self) -> bool {
        if !self.reverse_search_active {
            return self.activate_reverse_search();
        }

        let matches = self.reverse_search_matches();
        if matches.is_empty() {
            return false;
        }

        self.reverse_search_selection = (self.reverse_search_selection + 1) % matches.len();
        self.set_input_text(matches[self.reverse_search_selection].clone());
        self.mark_dirty(DirtySections::INPUT);
        true
    }

    pub fn accept_reverse_search(&mut self) -> bool {
        if !self.reverse_search_active {
            return false;
        }
        self.reverse_search_active = false;
        self.reverse_search_query.clear();
        self.reverse_search_selection = 0;
        self.reverse_search_draft = None;
        self.mark_dirty(DirtySections::INPUT);
        true
    }

    pub fn cancel_reverse_search(&mut self) -> bool {
        if !self.reverse_search_active {
            return false;
        }
        let draft = self.reverse_search_draft.take().unwrap_or_default();
        self.reverse_search_active = false;
        self.reverse_search_query.clear();
        self.reverse_search_selection = 0;
        self.set_input_text(draft);
        self.mark_dirty(DirtySections::INPUT);
        true
    }

    pub fn reverse_search_view(&self) -> Option<(String, String)> {
        if !self.reverse_search_active {
            return None;
        }
        let current = self
            .reverse_search_matches()
            .get(self.reverse_search_selection)
            .cloned()
            .unwrap_or_default();
        Some((self.reverse_search_query.clone(), current))
    }

    pub fn activate_command_launcher(&mut self, entries: Vec<CommandSuggestion>) -> bool {
        if entries.is_empty() {
            return false;
        }
        if !self.command_launcher_active {
            self.command_launcher_active = true;
            self.command_launcher_query.clear();
            self.command_launcher_selection = 0;
            self.command_launcher_draft = Some(self.input.clone());
        }
        self.command_launcher_entries = entries;
        self.apply_command_launcher_match();
        self.mark_dirty(DirtySections::INPUT);
        true
    }

    pub fn command_launcher_push_char(&mut self, c: char) {
        if !self.command_launcher_active {
            return;
        }
        self.command_launcher_query.push(c);
        self.command_launcher_selection = 0;
        self.apply_command_launcher_match();
        self.mark_dirty(DirtySections::INPUT);
    }

    pub fn command_launcher_backspace(&mut self) {
        if !self.command_launcher_active {
            return;
        }
        self.command_launcher_query.pop();
        self.command_launcher_selection = 0;
        self.apply_command_launcher_match();
        self.mark_dirty(DirtySections::INPUT);
    }

    pub fn command_launcher_cycle(&mut self, reverse: bool) -> bool {
        if !self.command_launcher_active {
            return false;
        }
        let matches = self.command_launcher_matches();
        if matches.is_empty() {
            return false;
        }
        if reverse {
            if self.command_launcher_selection == 0 {
                self.command_launcher_selection = matches.len() - 1;
            } else {
                self.command_launcher_selection -= 1;
            }
        } else {
            self.command_launcher_selection = (self.command_launcher_selection + 1) % matches.len();
        }
        self.apply_command_launcher_match();
        self.mark_dirty(DirtySections::INPUT);
        true
    }

    pub fn accept_command_launcher(&mut self) -> Option<String> {
        if !self.command_launcher_active {
            return None;
        }
        let selected = self
            .command_launcher_matches()
            .get(self.command_launcher_selection)
            .map(|item| item.name.clone())?;
        self.command_launcher_active = false;
        self.command_launcher_query.clear();
        self.command_launcher_selection = 0;
        self.command_launcher_draft = None;
        let command = format!("{selected} ");
        self.set_input_text(command.clone());
        self.mark_dirty(DirtySections::INPUT);
        Some(command)
    }

    pub fn cancel_command_launcher(&mut self) -> bool {
        if !self.command_launcher_active {
            return false;
        }
        let draft = self.command_launcher_draft.take().unwrap_or_default();
        self.command_launcher_active = false;
        self.command_launcher_query.clear();
        self.command_launcher_selection = 0;
        self.command_launcher_entries.clear();
        self.set_input_text(draft);
        self.mark_dirty(DirtySections::INPUT);
        true
    }

    pub fn command_launcher_view(
        &self,
        max: usize,
    ) -> Option<(String, Vec<(CommandSuggestion, bool)>)> {
        if !self.command_launcher_active {
            return None;
        }
        let preview = self
            .command_launcher_matches()
            .into_iter()
            .take(max)
            .enumerate()
            .map(|(idx, item)| (item, idx == self.command_launcher_selection))
            .collect::<Vec<_>>();
        Some((self.command_launcher_query.clone(), preview))
    }

    pub fn normalized_paste(text: &str) -> String {
        text.replace("\r\n", "\n").replace('\r', "\n")
    }

    pub fn input_display_lines(
        &self,
        width: usize,
        max_visible_rows: usize,
    ) -> (Vec<String>, usize, usize) {
        let wrapped = wrap_input_for_display(&self.input, width);
        let cursor = cursor_visual_position(&self.input, self.cursor, width);
        let total_rows = wrapped.len().max(1);
        let start_row = if total_rows <= max_visible_rows {
            0
        } else {
            cursor
                .0
                .saturating_add(1)
                .saturating_sub(max_visible_rows)
                .min(total_rows.saturating_sub(max_visible_rows))
        };
        let end_row = (start_row + max_visible_rows).min(total_rows);
        let visible = wrapped[start_row..end_row].to_vec();
        (visible, cursor.0.saturating_sub(start_row), cursor.1)
    }

    pub fn input_content_rows(&self, width: usize) -> usize {
        wrap_input_for_display(&self.input, width).len().max(1)
    }

    pub fn autocomplete_hint(&self) -> Option<String> {
        if self.autocomplete_matches.is_empty() {
            return None;
        }

        let preview = self
            .autocomplete_matches
            .iter()
            .take(4)
            .cloned()
            .collect::<Vec<_>>()
            .join("  ");
        let extra = self.autocomplete_matches.len().saturating_sub(4);
        if extra > 0 {
            Some(format!("{preview}  +{extra}"))
        } else {
            Some(preview)
        }
    }

    pub fn autocomplete_preview_matches(&self, max: usize) -> Vec<String> {
        self.autocomplete_matches
            .iter()
            .take(max)
            .cloned()
            .collect()
    }

    pub fn autocomplete_preview_items(&self, max: usize) -> Vec<(String, bool)> {
        self.autocomplete_matches
            .iter()
            .take(max)
            .enumerate()
            .map(|(idx, value)| (value.clone(), idx == self.autocomplete_index))
            .collect()
    }

    pub fn autocomplete_command<S: AsRef<str>>(&mut self, commands: &[S], reverse: bool) -> bool {
        let Some((start, end, typed_prefix)) = slash_prefix_range(&self.input, self.cursor) else {
            self.clear_autocomplete();
            return false;
        };

        let prefix = if !self.autocomplete_matches.is_empty()
            && self.autocomplete_index < self.autocomplete_matches.len()
            && self.autocomplete_matches[self.autocomplete_index] == self.input[..end]
        {
            self.autocomplete_prefix.clone().unwrap_or(typed_prefix)
        } else {
            typed_prefix
        };

        let matches = commands
            .iter()
            .filter(|cmd| cmd.as_ref().starts_with(prefix.as_str()))
            .map(|cmd| cmd.as_ref().to_string())
            .collect::<Vec<_>>();

        if matches.is_empty() {
            self.clear_autocomplete();
            return false;
        }

        let same_cycle = self
            .autocomplete_prefix
            .as_ref()
            .map(|existing| existing == &prefix)
            .unwrap_or(false)
            && self.autocomplete_matches == matches;

        if same_cycle {
            if reverse {
                if self.autocomplete_index == 0 {
                    self.autocomplete_index = self.autocomplete_matches.len() - 1;
                } else {
                    self.autocomplete_index -= 1;
                }
            } else {
                self.autocomplete_index =
                    (self.autocomplete_index + 1) % self.autocomplete_matches.len();
            }
        } else {
            self.autocomplete_matches = matches;
            self.autocomplete_prefix = Some(prefix);
            self.autocomplete_index = if reverse {
                self.autocomplete_matches.len() - 1
            } else {
                0
            };
        }

        let selected = &self.autocomplete_matches[self.autocomplete_index];
        self.input.replace_range(start..end, selected);
        self.cursor = start + selected.len();

        if self.autocomplete_matches.len() == 1 && self.input[self.cursor..].is_empty() {
            self.input.push(' ');
            self.cursor += 1;
        }

        self.mark_dirty(DirtySections::INPUT);
        true
    }
}

fn slash_prefix_range(input: &str, cursor: usize) -> Option<(usize, usize, String)> {
    if !input.starts_with('/') {
        return None;
    }

    let safe_cursor = cursor.min(input.len());
    let active = &input[..safe_cursor];
    let command_end = active.find(' ').unwrap_or(active.len());
    if command_end == 0 || safe_cursor > command_end {
        return None;
    }

    Some((0, command_end, input[..command_end].to_string()))
}

fn wrap_input_for_display(input: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut lines = Vec::new();

    if input.is_empty() {
        return vec![String::new()];
    }

    for raw_line in input.split('\n') {
        let wrapped = wrap_preserving_empty_line(raw_line, width);
        lines.extend(wrapped);
    }

    if input.ends_with('\n') {
        lines.push(String::new());
    }

    if lines.is_empty() {
        vec![String::new()]
    } else {
        lines
    }
}

fn wrap_preserving_empty_line(line: &str, width: usize) -> Vec<String> {
    if line.is_empty() {
        return vec![String::new()];
    }

    let chars: Vec<char> = line.chars().collect();
    let mut wrapped = Vec::new();
    let mut start = 0usize;
    while start < chars.len() {
        let end = (start + width).min(chars.len());
        wrapped.push(chars[start..end].iter().collect());
        start = end;
    }
    wrapped
}

fn cursor_visual_position(input: &str, cursor: usize, width: usize) -> (usize, usize) {
    let width = width.max(1);
    let safe_cursor = cursor.min(input.len());
    let before = &input[..safe_cursor];
    let mut row = 0usize;
    let mut col = 0usize;

    for ch in before.chars() {
        if ch == '\n' {
            row += 1;
            col = 0;
            continue;
        }
        col += 1;
        if col >= width {
            row += 1;
            col = 0;
        }
    }

    (row, col)
}
