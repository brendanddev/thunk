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

    fn enforce_scope(scope: &str, mut path: Option<String>) -> Option<String> {
        if let Some(ref p) = path.clone() {
            if !path_is_within_scope(p, scope) {
                path = Some(scope.to_string());
            }
        } else {
            path = Some(scope.to_string());
        }
        path
    }

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

    #[test]
    fn path_is_within_scope_exact_match() {
        assert!(path_is_within_scope(
            "sandbox/services/",
            "sandbox/services/"
        ));
        assert!(path_is_within_scope("sandbox/services", "sandbox/services"));
        assert!(path_is_within_scope(
            "sandbox/services/",
            "sandbox/services"
        ));
        assert!(path_is_within_scope(
            "sandbox/services",
            "sandbox/services/"
        ));
    }

    #[test]
    fn path_is_within_scope_narrower_path_accepted() {
        assert!(path_is_within_scope(
            "sandbox/services/tasks/",
            "sandbox/services/"
        ));
        assert!(path_is_within_scope(
            "sandbox/cli/handlers/",
            "sandbox/cli/"
        ));
        assert!(path_is_within_scope("sandbox/", "sandbox/"));
    }

    #[test]
    fn path_is_within_scope_broader_path_rejected() {
        assert!(!path_is_within_scope("sandbox/", "sandbox/services/"));
        assert!(!path_is_within_scope("src/", "sandbox/services/"));
        assert!(!path_is_within_scope(".", "sandbox/services/"));
    }

    #[test]
    fn path_is_within_scope_orthogonal_path_rejected() {
        assert!(!path_is_within_scope("src/runtime/", "sandbox/services/"));
        assert!(!path_is_within_scope("models/", "services/"));
    }

    #[test]
    fn path_is_within_scope_boundary_guard_prevents_prefix_collision() {
        assert!(!path_is_within_scope(
            "sandbox/service_extra/",
            "sandbox/service/"
        ));
        assert!(!path_is_within_scope(
            "sandbox/services_extended/",
            "sandbox/services/"
        ));
        assert!(path_is_within_scope(
            "sandbox/services/sub/",
            "sandbox/services/"
        ));
    }

    #[test]
    fn path_is_within_scope_absolute_path_rejected() {
        assert!(!path_is_within_scope(
            "/Users/project/sandbox/services/",
            "sandbox/services/"
        ));
        assert!(!path_is_within_scope("/abs/path/", "sandbox/"));
    }

    #[test]
    fn path_is_within_scope_parent_components_rejected() {
        assert!(!path_is_within_scope(
            "sandbox/services/../",
            "sandbox/services/"
        ));
        assert!(!path_is_within_scope(
            "sandbox/services/../../src/",
            "sandbox/services/"
        ));
        assert!(!path_is_within_scope(
            "sandbox/services/tasks/",
            "sandbox/services/../"
        ));
    }

    #[test]
    fn path_is_within_scope_dotslash_normalization() {
        assert!(path_is_within_scope(
            "./sandbox/services/",
            "sandbox/services/"
        ));
        assert!(path_is_within_scope(
            "sandbox/services/",
            "./sandbox/services/"
        ));
    }

    #[test]
    fn scope_enforcement_clamps_broader_parent_path() {
        let scope = "sandbox/services/";
        let path = enforce_scope(scope, Some("sandbox/".into()));
        assert_eq!(path.as_deref(), Some("sandbox/services/"));
    }

    #[test]
    fn scope_enforcement_clamps_parent_component_path() {
        let scope = "sandbox/services/";
        let path = enforce_scope(scope, Some("sandbox/services/../".into()));
        assert_eq!(path.as_deref(), Some("sandbox/services/"));
    }

    #[test]
    fn scope_enforcement_clamps_unrelated_path() {
        let scope = "sandbox/services/";
        let path = enforce_scope(scope, Some("src/".into()));
        assert_eq!(path.as_deref(), Some("sandbox/services/"));
    }

    #[test]
    fn scope_enforcement_preserves_exact_scope_path() {
        let scope = "sandbox/services/";
        let path = enforce_scope(scope, Some("sandbox/services/".into()));
        assert_eq!(path.as_deref(), Some("sandbox/services/"));
    }

    #[test]
    fn scope_enforcement_preserves_child_path() {
        let scope = "sandbox/services/";
        let path = enforce_scope(scope, Some("sandbox/services/tasks/".into()));
        assert_eq!(path.as_deref(), Some("sandbox/services/tasks/"));
    }

    #[test]
    fn scope_enforcement_injects_when_path_absent() {
        let scope = "sandbox/services/";
        let path = enforce_scope(scope, None);
        assert_eq!(path.as_deref(), Some("sandbox/services/"));
    }
}
