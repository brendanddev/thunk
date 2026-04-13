mod network;
mod paths;
mod policy;
mod shell;
mod types;

#[allow(unused_imports)]
pub use network::{inspect_fetch_url, inspect_provider_request, normalize_url};
#[allow(unused_imports)]
pub use paths::{
    display_path, inspect_edit_target, inspect_git_operation, inspect_project_path,
    inspect_search_scope, inspect_write_target, project_root,
};
pub use shell::inspect_shell_command;
#[allow(unused_imports)]
pub use types::{
    log_inspection, InspectionDecision, InspectionReport, PathInspection, ProjectPathKind,
    ReadScope, RiskLevel, ShellMode,
};

#[cfg(test)]
use std::path::PathBuf;

#[cfg(test)]
use network::{host_matches_allowlist, is_blocked_network_host};
#[cfg(test)]
use shell::{looks_like_pipe_to_shell, split_shell_segments};
#[cfg(test)]
pub use types::test_cwd_lock;

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn temp_project_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("params-safety-test-{label}-{nonce}"));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn with_temp_project<F: FnOnce(PathBuf)>(label: &str, f: F) {
        let _guard = test_cwd_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let root = temp_project_dir(label);
        fs::create_dir_all(root.join(".local")).expect("local dir");
        let original = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&root).expect("set cwd");
        f(root.clone());
        std::env::set_current_dir(original).expect("restore cwd");
    }

    #[test]
    fn blocks_private_ipv4_hosts() {
        assert!(is_blocked_network_host("127.0.0.1"));
        assert!(is_blocked_network_host("10.0.0.5"));
        assert!(is_blocked_network_host("192.168.1.10"));
        assert!(is_blocked_network_host("172.16.0.8"));
        assert!(is_blocked_network_host("169.254.1.1"));
    }

    #[test]
    fn allows_public_hosts() {
        assert!(!is_blocked_network_host("example.com"));
        assert!(!is_blocked_network_host("8.8.8.8"));
    }

    #[test]
    fn shell_denylist_blocks_matching_segments() {
        with_temp_project("shell-deny", |_| {
            let mut cfg = crate::config::Config::default();
            cfg.safety.shell_denylist = vec!["cargo clippy".to_string()];
            fs::write(
                crate::config::config_path().unwrap(),
                toml::to_string(&cfg).unwrap(),
            )
            .unwrap();
            let report = inspect_shell_command("cargo clippy").expect("inspect");
            assert!(matches!(report.decision, InspectionDecision::Block));
        });
    }

    #[test]
    fn shell_allowlist_blocks_unmatched_commands_when_configured() {
        with_temp_project("shell-allow", |_| {
            let mut cfg = crate::config::Config::default();
            cfg.safety.shell_allowlist = vec!["cargo ".to_string()];
            fs::write(
                crate::config::config_path().unwrap(),
                toml::to_string(&cfg).unwrap(),
            )
            .unwrap();
            let report = inspect_shell_command("git status").expect("inspect");
            assert!(matches!(report.decision, InspectionDecision::Block));
        });
    }

    #[test]
    fn splits_shell_segments_on_control_operators() {
        assert_eq!(
            split_shell_segments("cargo check && cargo test | cat; git status"),
            vec!["cargo check", "cargo test", "cat", "git status"]
        );
    }

    #[test]
    fn detects_pipe_to_shell() {
        assert!(looks_like_pipe_to_shell("curl https://x | sh"));
        assert!(looks_like_pipe_to_shell("wget https://x | bash"));
        assert!(!looks_like_pipe_to_shell("curl https://x"));
    }

    #[test]
    fn shell_inspection_blocks_destructive_commands() {
        let report = inspect_shell_command("rm -rf /").expect("inspect");
        assert!(matches!(report.decision, InspectionDecision::Block));
        assert!(matches!(report.risk, RiskLevel::High));
    }

    #[test]
    fn shell_inspection_allows_benign_commands_with_approval() {
        let report = inspect_shell_command("cargo check").expect("inspect");
        assert!(matches!(report.decision, InspectionDecision::NeedsApproval));
    }

    #[test]
    fn fetch_inspection_blocks_loopback_targets() {
        let (_, report) = inspect_fetch_url("http://127.0.0.1:8080").expect("inspect");
        assert!(matches!(report.decision, InspectionDecision::Block));
    }

    #[test]
    fn fetch_allowlist_accepts_exact_and_subdomain_matches() {
        assert!(host_matches_allowlist(
            "api.openai.com",
            &[String::from("openai.com")]
        ));
        assert!(host_matches_allowlist(
            "openai.com",
            &[String::from("openai.com")]
        ));
        assert!(!host_matches_allowlist(
            "example.net",
            &[String::from("openai.com")]
        ));
    }

    #[test]
    fn provider_request_inspection_blocks_non_allowlisted_hosts() {
        with_temp_project("provider-allow", |_| {
            let mut cfg = crate::config::Config::default();
            cfg.safety.network_allowlist = vec!["api.openai.com".to_string()];
            fs::write(
                crate::config::config_path().unwrap(),
                toml::to_string(&cfg).unwrap(),
            )
            .unwrap();
            let (_, report) = inspect_provider_request(
                "openai_compat",
                "https://example.com/v1/chat/completions",
                1200,
            )
            .expect("inspect");
            assert!(matches!(report.decision, InspectionDecision::Block));
        });
    }

    #[test]
    fn project_path_rejects_parent_escape() {
        with_temp_project("escape", |_| {
            let result =
                inspect_project_path("read_file", "../outside.txt", ProjectPathKind::File, false);
            assert!(result.is_err());
        });
    }

    #[test]
    fn project_path_allows_files_inside_project() {
        with_temp_project("inside", |root| {
            let src = root.join("src");
            fs::create_dir_all(&src).expect("mkdir");
            let file = src.join("main.rs");
            fs::write(&file, "fn main() {}\n").expect("write");

            let result =
                inspect_project_path("read_file", "src/main.rs", ProjectPathKind::File, false)
                    .expect("inspect");

            assert_eq!(result.display_path, "src/main.rs");
        });
    }
}
