// Phase 17.3: External Repo Validation Fixtures.
// Tests-only. No production behavior is changed.

use std::fs;
use tempfile::TempDir;

use super::*;
use crate::runtime::{
    project::{ProjectStructureSnapshot, MAX_SNAPSHOT_NODES},
    resolve, PathResolutionError, ProjectPath, ProjectScope, ResolvedToolInput,
};
use crate::tools::{default_registry, ToolInput, ToolOutput, ToolRunResult};

fn dir_scope(dir: &TempDir, relative: &str) -> ProjectScope {
    let canon = dir.path().canonicalize().unwrap();
    let abs = if relative == "." {
        canon
    } else {
        canon.join(relative)
    };
    ProjectScope::from_trusted_path(ProjectPath::from_trusted(abs, relative.to_string()))
}

fn build_root(dir: &TempDir) -> ProjectRoot {
    ProjectRoot::new(dir.path().to_path_buf()).unwrap()
}

// ─── project root detection ──────────────────────────────────────────────────

#[test]
fn project_root_accepts_git_repo_root() {
    let dir = TempDir::new().unwrap();
    fs::create_dir(dir.path().join(".git")).unwrap();

    let root = ProjectRoot::new(dir.path().to_path_buf());

    assert!(
        root.is_ok(),
        "ProjectRoot must accept a directory containing .git"
    );
    assert!(root.unwrap().path().is_absolute());
}

#[test]
fn project_root_accepts_nested_directory_inside_git_repo() {
    let dir = TempDir::new().unwrap();
    fs::create_dir(dir.path().join(".git")).unwrap();
    let sub = dir.path().join("src").join("app");
    fs::create_dir_all(&sub).unwrap();

    let root = ProjectRoot::new(sub);

    assert!(
        root.is_ok(),
        "ProjectRoot must accept a nested subdir regardless of .git placement"
    );
}

#[test]
fn project_root_accepts_plain_directory_without_git() {
    let dir = TempDir::new().unwrap();

    let root = ProjectRoot::new(dir.path().to_path_buf());

    assert!(
        root.is_ok(),
        "ProjectRoot must accept a directory with no .git present"
    );
}

// ─── startup behavior ────────────────────────────────────────────────────────

#[test]
fn runtime_starts_in_git_initialized_repo_without_config_toml() {
    let dir = TempDir::new().unwrap();
    init_git_repo(dir.path());
    fs::write(dir.path().join("main.rs"), "fn main() {}\n").unwrap();

    let mut rt = make_runtime_in(Vec::<&str>::new(), dir.path());
    let snapshot = rt.project_snapshot_for_test().unwrap();

    assert!(
        !snapshot.entries.is_empty(),
        "runtime started in a git repo must produce a non-empty snapshot"
    );
}

#[test]
fn runtime_starts_rooted_at_nested_subdir_of_git_repo() {
    let dir = TempDir::new().unwrap();
    init_git_repo(dir.path());
    let sub = dir.path().join("src");
    fs::create_dir_all(&sub).unwrap();
    fs::write(sub.join("lib.rs"), "pub fn f() {}\n").unwrap();

    let mut rt = make_runtime_in(Vec::<&str>::new(), &sub);
    let snapshot = rt.project_snapshot_for_test().unwrap();

    let paths: Vec<&str> = snapshot.entries.iter().map(|e| e.path.as_str()).collect();
    assert!(
        paths.contains(&"lib.rs"),
        "snapshot of nested root must contain lib.rs: {paths:?}"
    );
}

#[test]
fn runtime_starts_with_config_toml_present() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("config.toml"), "[app]\nname = \"test\"\n").unwrap();
    fs::write(dir.path().join("main.rs"), "fn main() {}\n").unwrap();

    let mut rt = make_runtime_in(Vec::<&str>::new(), dir.path());
    let snapshot = rt.project_snapshot_for_test().unwrap();

    let paths: Vec<&str> = snapshot.entries.iter().map(|e| e.path.as_str()).collect();
    assert!(
        paths.contains(&"main.rs"),
        "runtime with config.toml must produce a valid snapshot: {paths:?}"
    );
}

// ─── list_dir behavior ───────────────────────────────────────────────────────

#[test]
fn list_dir_skips_all_default_noisy_directories() {
    let dir = TempDir::new().unwrap();
    for noisy in &[".git", ".hg", "build", "dist", "node_modules", "target"] {
        fs::create_dir(dir.path().join(noisy)).unwrap();
        fs::write(dir.path().join(noisy).join("artifact.txt"), "noise").unwrap();
    }
    fs::create_dir(dir.path().join("src")).unwrap();
    fs::write(dir.path().join("Cargo.toml"), "[package]\n").unwrap();

    let result = default_registry()
        .dispatch(ResolvedToolInput::ListDir {
            path: dir_scope(&dir, "."),
        })
        .unwrap();

    let ToolRunResult::Immediate(ToolOutput::DirectoryListing(dl)) = result else {
        panic!("expected DirectoryListing")
    };
    let names: Vec<&str> = dl.entries.iter().map(|e| e.name.as_str()).collect();

    for noisy in &[".git", ".hg", "build", "dist", "node_modules", "target"] {
        assert!(
            !names.contains(noisy),
            "list_dir must skip {noisy}: {names:?}"
        );
    }
    assert!(
        names.contains(&"src"),
        "list_dir must include src: {names:?}"
    );
    assert!(
        names.contains(&"Cargo.toml"),
        "list_dir must include Cargo.toml: {names:?}"
    );
}

#[test]
fn list_dir_bounded_output_holds_with_noisy_directories_present() {
    let dir = TempDir::new().unwrap();
    // 210 source files — exceeds the 200-entry cap.
    for i in 0..210u32 {
        fs::write(dir.path().join(format!("file{i:03}.rs")), "").unwrap();
    }
    // Noisy dirs must not consume entry budget.
    fs::create_dir(dir.path().join("target")).unwrap();
    fs::create_dir(dir.path().join("node_modules")).unwrap();

    let result = default_registry()
        .dispatch(ResolvedToolInput::ListDir {
            path: dir_scope(&dir, "."),
        })
        .unwrap();

    let ToolRunResult::Immediate(ToolOutput::DirectoryListing(dl)) = result else {
        panic!("expected DirectoryListing")
    };

    assert!(
        dl.truncated,
        "output must be truncated when entries exceed cap"
    );
    assert_eq!(
        dl.entries.len(),
        200,
        "truncated listing must contain exactly 200 entries"
    );

    let names: Vec<&str> = dl.entries.iter().map(|e| e.name.as_str()).collect();
    assert!(
        !names.contains(&"target"),
        "target must not appear in output"
    );
    assert!(
        !names.contains(&"node_modules"),
        "node_modules must not appear in output"
    );
}

#[test]
fn list_dir_ordering_is_deterministic_in_mixed_repo() {
    let dir = TempDir::new().unwrap();
    fs::create_dir(dir.path().join("src")).unwrap();
    fs::create_dir(dir.path().join("docs")).unwrap();
    fs::create_dir(dir.path().join("node_modules")).unwrap();
    fs::create_dir(dir.path().join("target")).unwrap();
    fs::write(dir.path().join("Cargo.toml"), "").unwrap();
    fs::write(dir.path().join("README.md"), "").unwrap();

    let registry = default_registry();

    let r1 = registry
        .dispatch(ResolvedToolInput::ListDir {
            path: dir_scope(&dir, "."),
        })
        .unwrap();
    let ToolRunResult::Immediate(ToolOutput::DirectoryListing(dl1)) = r1 else {
        panic!("expected DirectoryListing")
    };
    let names1: Vec<String> = dl1.entries.iter().map(|e| e.name.clone()).collect();

    let r2 = registry
        .dispatch(ResolvedToolInput::ListDir {
            path: dir_scope(&dir, "."),
        })
        .unwrap();
    let ToolRunResult::Immediate(ToolOutput::DirectoryListing(dl2)) = r2 else {
        panic!("expected DirectoryListing")
    };
    let names2: Vec<String> = dl2.entries.iter().map(|e| e.name.clone()).collect();

    assert_eq!(
        names1, names2,
        "list_dir must produce identical ordering on repeated calls"
    );
}

// ─── search_code behavior ────────────────────────────────────────────────────

#[test]
fn search_code_skips_all_noisy_directories_finds_only_source() {
    let dir = TempDir::new().unwrap();

    for noisy in &[".git", ".hg", "build", "dist", "node_modules", "target"] {
        fs::create_dir(dir.path().join(noisy)).unwrap();
        // .rs extension makes these TEXT_EXTENSIONS-eligible;
        // the skip logic must exclude them before extension filtering.
        fs::write(
            dir.path().join(noisy).join("artifact.rs"),
            "fn needle() {}\n",
        )
        .unwrap();
    }
    fs::create_dir(dir.path().join("src")).unwrap();
    fs::write(dir.path().join("src").join("lib.rs"), "fn needle() {}\n").unwrap();

    let registry = default_registry().with_project_root(dir.path().canonicalize().unwrap());
    let result = registry
        .dispatch(ResolvedToolInput::SearchCode {
            query: "needle".to_string(),
            scope: None,
        })
        .unwrap();

    let ToolRunResult::Immediate(ToolOutput::SearchResults(sr)) = result else {
        panic!("expected SearchResults")
    };
    let files: Vec<&str> = sr.matches.iter().map(|m| m.file.as_str()).collect();

    for noisy in &[".git", ".hg", "build", "dist", "node_modules", "target"] {
        assert!(
            !files.iter().any(|f| f.starts_with(noisy)),
            "search_code must not return results from {noisy}: {files:?}"
        );
    }
    assert!(
        files.iter().any(|f| *f == "src/lib.rs"),
        "search_code must find src/lib.rs: {files:?}"
    );
}

// ─── project_snapshot behavior ───────────────────────────────────────────────

#[test]
fn project_snapshot_excludes_all_noisy_directories_in_realistic_fixture() {
    let dir = TempDir::new().unwrap();

    for noisy in &[".git", ".hg", "build", "dist", "node_modules", "target"] {
        fs::create_dir(dir.path().join(noisy)).unwrap();
        fs::write(dir.path().join(noisy).join("file.txt"), "x").unwrap();
    }
    fs::create_dir(dir.path().join("src")).unwrap();
    fs::write(dir.path().join("src").join("lib.rs"), "pub fn f() {}\n").unwrap();
    fs::write(dir.path().join("Cargo.toml"), "[package]\n").unwrap();

    let snapshot = ProjectStructureSnapshot::build(&build_root(&dir)).unwrap();
    let paths: Vec<&str> = snapshot.entries.iter().map(|e| e.path.as_str()).collect();

    for noisy in &[".git", ".hg", "build", "dist", "node_modules", "target"] {
        assert!(
            !paths.iter().any(|p| p.starts_with(noisy)),
            "snapshot must not contain {noisy}: {paths:?}"
        );
    }
    assert!(
        paths.contains(&"src"),
        "snapshot must include src: {paths:?}"
    );
    assert!(
        paths.contains(&"Cargo.toml"),
        "snapshot must include Cargo.toml: {paths:?}"
    );
}

#[test]
fn project_snapshot_does_not_explode_on_large_noisy_tree() {
    let dir = TempDir::new().unwrap();

    // 50 real files — exceeds MAX_SNAPSHOT_NODES (40).
    for i in 0..50u32 {
        fs::write(dir.path().join(format!("file{i:02}.rs")), "x").unwrap();
    }
    // All noisy dirs with children present — must not add to node count.
    for noisy in &[".git", ".hg", "build", "dist", "node_modules", "target"] {
        let noisy_dir = dir.path().join(noisy);
        fs::create_dir(&noisy_dir).unwrap();
        for j in 0..5u32 {
            fs::write(noisy_dir.join(format!("artifact{j}.txt")), "x").unwrap();
        }
    }

    let snapshot = ProjectStructureSnapshot::build(&build_root(&dir)).unwrap();

    assert!(
        snapshot.truncated,
        "snapshot must be truncated when entries exceed MAX_SNAPSHOT_NODES"
    );
    assert_eq!(
        snapshot.entries.len(),
        MAX_SNAPSHOT_NODES,
        "truncated snapshot must contain exactly MAX_SNAPSHOT_NODES entries"
    );
    let paths: Vec<&str> = snapshot.entries.iter().map(|e| e.path.as_str()).collect();
    for noisy in &[".git", ".hg", "build", "dist", "node_modules", "target"] {
        assert!(
            !paths.iter().any(|p| p.starts_with(noisy)),
            "snapshot must not include {noisy} at node cap: {paths:?}"
        );
    }
}

// ─── path safety ─────────────────────────────────────────────────────────────

#[test]
fn path_cannot_escape_root_via_dotdot() {
    let dir = TempDir::new().unwrap();
    fs::create_dir(dir.path().join(".git")).unwrap();
    // Create a real file one level above root so resolution would succeed if
    // the escape check were absent.
    let outside = dir.path().parent().unwrap().join("outside.txt");
    fs::write(&outside, "secret").unwrap();

    let root = build_root(&dir);
    let err = resolve(
        &root,
        &ToolInput::ReadFile {
            path: "../outside.txt".into(),
        },
    )
    .unwrap_err();

    assert!(
        matches!(err, PathResolutionError::EscapesRoot { .. }),
        ".. escape must be rejected: {err:?}"
    );
    fs::remove_file(outside).unwrap();
}

#[cfg(unix)]
#[test]
fn symlink_pointing_outside_root_is_rejected() {
    let dir = TempDir::new().unwrap();
    let outside = TempDir::new().unwrap();
    let outside_file = outside.path().join("secret.txt");
    fs::write(&outside_file, "secret").unwrap();
    std::os::unix::fs::symlink(&outside_file, dir.path().join("link.txt")).unwrap();

    let root = build_root(&dir);
    let err = resolve(
        &root,
        &ToolInput::ReadFile {
            path: "link.txt".into(),
        },
    )
    .unwrap_err();

    assert!(
        matches!(err, PathResolutionError::EscapesRoot { .. }),
        "symlink pointing outside root must be rejected: {err:?}"
    );
}
