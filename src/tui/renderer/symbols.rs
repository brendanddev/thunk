use std::collections::HashMap;

use unicode_width::UnicodeWidthChar;

#[derive(Default)]
pub(crate) struct SymbolPool {
    ids: HashMap<String, u32>,
    symbols: Vec<String>,
}

impl SymbolPool {
    pub fn new() -> Self {
        let mut pool = Self::default();
        pool.intern(" ");
        pool
    }

    pub fn blank_id(&mut self) -> u32 {
        self.intern(" ")
    }

    pub fn intern(&mut self, value: &str) -> u32 {
        if let Some(id) = self.ids.get(value) {
            return *id;
        }
        let id = self.symbols.len() as u32;
        let owned = value.to_string();
        self.ids.insert(owned.clone(), id);
        self.symbols.push(owned);
        id
    }

    pub fn intern_char_lossy(&mut self, value: char) -> u32 {
        let rendered = match UnicodeWidthChar::width(value) {
            Some(1) => value.to_string(),
            _ => "?".to_string(),
        };
        self.intern(&rendered)
    }

    pub fn get(&self, id: u32) -> &str {
        self.symbols
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or(" ")
    }

    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    pub fn reset(&mut self) {
        self.ids.clear();
        self.symbols.clear();
        self.intern(" ");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_reuses_symbol_ids() {
        let mut pool = SymbolPool::new();
        let a = pool.intern("x");
        let b = pool.intern("x");
        assert_eq!(a, b);
    }

    #[test]
    fn pool_degrades_wide_chars() {
        let mut pool = SymbolPool::new();
        let id = pool.intern_char_lossy('界');
        assert_eq!(pool.get(id), "?");
    }
}
