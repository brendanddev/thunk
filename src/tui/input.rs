use super::state::AppState;

/// Defines methods for modifying the input buffer and cursor position in the app state
impl AppState {
    /// Inserts a character at the current cursor position and moves the cursor forward
    pub fn insert_char(&mut self, c: char) {
        self.input.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    /// Inserts a string at the current cursor position and moves the cursor forward
    pub fn insert_str(&mut self, s: &str) {
        self.input.insert_str(self.cursor, s);
        self.cursor += s.len();
    }

    /// Deletes the character before the current cursor position and moves the cursor back
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
    }

    /// Deletes the character before the current cursor position and moves the cursor back
    pub fn cursor_left(&mut self) {
        if self.cursor == 0 {
            return;
        }

        let mut prev = self.cursor - 1;
        while !self.input.is_char_boundary(prev) {
            prev -= 1;
        }
        self.cursor = prev;
    }

    /// Moves the cursor right, ensuring it stays on valid character boundaries
    pub fn cursor_right(&mut self) {
        if self.cursor >= self.input.len() {
            return;
        }

        let mut next = self.cursor + 1;
        while next < self.input.len() && !self.input.is_char_boundary(next) {
            next += 1;
        }
        self.cursor = next.min(self.input.len());
    }

    /// Moves the cursor to the beginning of the input
    pub fn cursor_home(&mut self) {
        self.cursor = 0;
    }

    /// Moves the cursor to the end of the input
    pub fn cursor_end(&mut self) {
        self.cursor = self.input.len();
    }

    /// Clears the input buffer and resets the cursor position
    pub fn clear_input(&mut self) {
        self.input.clear();
        self.cursor = 0;
    }
}
