pub(super) fn normalize_evidence_path(path: &str) -> String {
    path.replace('\\', "/").trim_start_matches("./").to_string()
}

pub(super) fn path_has_parent_component(path: &str) -> bool {
    path.split('/').any(|component| component == "..")
}

/// Returns true when `attempted` refers to the same file as `requested`.
///
/// Accepts both exact relative equality and the case where the model emits an
/// absolute path that ends with the requested relative path. The boundary guard
/// (`"/" + requested`) prevents a partial filename suffix from matching a
/// different file.
pub(super) fn path_matches_requested(attempted: &str, requested: &str) -> bool {
    let a = normalize_evidence_path(attempted);
    let r = normalize_evidence_path(requested);
    a == r || a.ends_with(&format!("/{r}"))
}

/// Returns true when `model_path` is within (equal to or narrower than) `scope`.
///
/// Both paths are normalized before comparison. Trailing slashes are stripped so
/// "sandbox/services/" and "sandbox/services" compare identically. The boundary
/// guard (`get(s.len()) == Some(&b'/')`) prevents "sandbox/service_extra" from
/// falsely matching scope "sandbox/service".
///
/// Absolute paths (e.g. emitted by the model as "/abs/path/") are never within
/// a relative scope and will always return false, causing the caller to clamp.
/// Parent-directory components (`..`) are also rejected structurally before
/// accepting equal-or-child scope relationships.
pub(super) fn path_is_within_scope(model_path: &str, scope: &str) -> bool {
    let p = normalize_evidence_path(model_path);
    let s = normalize_evidence_path(scope);
    if path_has_parent_component(&p) || path_has_parent_component(&s) {
        return false;
    }
    let p = p.trim_end_matches('/');
    let s = s.trim_end_matches('/');
    p.starts_with(s) && (p.len() == s.len() || p.as_bytes().get(s.len()) == Some(&b'/'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_matches_requested_exact_relative() {
        assert!(path_matches_requested("sandbox/main.py", "sandbox/main.py"));
    }

    #[test]
    fn path_matches_requested_absolute_suffix_matches_relative() {
        assert!(path_matches_requested(
            "/Users/brendan/project/sandbox/main.py",
            "sandbox/main.py"
        ));
    }

    #[test]
    fn path_matches_requested_different_absolute_path_rejected() {
        assert!(!path_matches_requested(
            "/Users/brendan/project/other/foo.rs",
            "sandbox/main.py"
        ));
    }

    #[test]
    fn path_matches_requested_different_relative_path_rejected() {
        assert!(!path_matches_requested(
            "sandbox/other.py",
            "sandbox/main.py"
        ));
    }
}
