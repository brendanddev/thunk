pub(super) fn normalize_evidence_path(path: &str) -> String {
    path.replace('\\', "/").trim_start_matches("./").to_string()
}

pub(super) fn path_has_parent_component(path: &str) -> bool {
    path.split('/').any(|component| component == "..")
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
