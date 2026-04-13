pub(super) fn normalize_policy_entry(entry: &str) -> Option<String> {
    let trimmed = entry.trim().trim_matches('.').to_ascii_lowercase();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}
