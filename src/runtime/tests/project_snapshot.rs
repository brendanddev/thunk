use super::*;
use std::fs;
use tempfile::TempDir;

fn snapshot_paths(rt: &mut Runtime) -> Vec<String> {
    rt.project_snapshot_for_test()
        .unwrap()
        .entries
        .into_iter()
        .map(|entry| entry.path)
        .collect()
}

#[test]
fn cache_returns_same_snapshot_until_invalidated() {
    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    fs::write(tmp.path().join("src").join("lib.rs"), "pub fn demo() {}\n").unwrap();

    let mut rt = make_runtime_in(Vec::<&str>::new(), tmp.path());
    let first = rt.project_snapshot_for_test().unwrap();

    fs::write(tmp.path().join("later.txt"), "hello\n").unwrap();

    let second = rt.project_snapshot_for_test().unwrap();
    assert_eq!(
        first, second,
        "snapshot must remain cached until invalidated"
    );
    assert!(
        !second.entries.iter().any(|entry| entry.path == "later.txt"),
        "cached snapshot must not reflect external changes before invalidation"
    );
}

#[test]
fn successful_approved_write_file_invalidates_cache_and_rebuilds_snapshot() {
    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    fs::write(tmp.path().join("src").join("lib.rs"), "pub fn demo() {}\n").unwrap();

    let mut rt = make_runtime_in(Vec::<&str>::new(), tmp.path());
    let before = rt.project_snapshot_for_test().unwrap();

    fs::write(tmp.path().join("external.txt"), "external\n").unwrap();
    let cached = rt.project_snapshot_for_test().unwrap();
    assert_eq!(before, cached, "snapshot must stay cached before approval");

    let written = tmp.path().join("written.txt");
    rt.set_pending_for_test(PendingAction {
        tool_name: "write_file".into(),
        summary: "create written.txt".into(),
        risk: RiskLevel::Medium,
        payload: format!("{}\x00hello\n", written.display()),
    });

    let approve_events = collect_events(&mut rt, RuntimeRequest::Approve);
    assert!(
        !has_failed(&approve_events),
        "approve failed unexpectedly: {approve_events:?}"
    );
    assert!(written.exists(), "approved write_file must create the file");

    let rebuilt_paths = snapshot_paths(&mut rt);
    assert!(
        rebuilt_paths.iter().any(|path| path == "external.txt"),
        "rebuilt snapshot must reflect external filesystem changes after invalidation: {rebuilt_paths:?}"
    );
    assert!(
        rebuilt_paths.iter().any(|path| path == "written.txt"),
        "rebuilt snapshot must include the approved write target: {rebuilt_paths:?}"
    );
}

#[test]
fn successful_approved_edit_file_invalidates_cache() {
    let tmp = TempDir::new().unwrap();
    let editable = tmp.path().join("editable.txt");
    fs::write(&editable, "hello world\n").unwrap();

    let mut rt = make_runtime_in(Vec::<&str>::new(), tmp.path());
    let before = rt.project_snapshot_for_test().unwrap();

    fs::write(tmp.path().join("external.txt"), "external\n").unwrap();
    let cached = rt.project_snapshot_for_test().unwrap();
    assert_eq!(before, cached, "snapshot must stay cached before approval");

    rt.set_pending_for_test(PendingAction {
        tool_name: "edit_file".into(),
        summary: "edit editable.txt".into(),
        risk: RiskLevel::Medium,
        payload: format!("{}\x00hello world\x00hello runtime", editable.display()),
    });

    let approve_events = collect_events(&mut rt, RuntimeRequest::Approve);
    assert!(
        !has_failed(&approve_events),
        "approve failed unexpectedly: {approve_events:?}"
    );
    assert_eq!(fs::read_to_string(&editable).unwrap(), "hello runtime\n");

    let rebuilt_paths = snapshot_paths(&mut rt);
    assert!(
        rebuilt_paths.iter().any(|path| path == "external.txt"),
        "successful edit_file approval must invalidate the cache: {rebuilt_paths:?}"
    );
}

#[test]
fn rejected_approval_does_not_invalidate_cache() {
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("base.txt"), "base\n").unwrap();

    let mut rt = make_runtime_in(Vec::<&str>::new(), tmp.path());
    let before = rt.project_snapshot_for_test().unwrap();

    fs::write(tmp.path().join("external.txt"), "external\n").unwrap();
    let cached = rt.project_snapshot_for_test().unwrap();
    assert_eq!(before, cached, "snapshot must stay cached before rejection");

    let rejected_target = tmp.path().join("rejected.txt");
    rt.set_pending_for_test(PendingAction {
        tool_name: "write_file".into(),
        summary: "create rejected.txt".into(),
        risk: RiskLevel::Medium,
        payload: format!("{}\x00hello\n", rejected_target.display()),
    });

    let reject_events = collect_events(&mut rt, RuntimeRequest::Reject);
    assert!(
        !has_failed(&reject_events),
        "reject failed unexpectedly: {reject_events:?}"
    );
    assert!(
        !rejected_target.exists(),
        "rejected write_file must not create the file"
    );

    let after = rt.project_snapshot_for_test().unwrap();
    assert_eq!(
        cached, after,
        "rejected approval must not invalidate the cached snapshot"
    );
    assert!(
        !after
            .entries
            .iter()
            .any(|entry| entry.path == "external.txt"),
        "rejected approval must not rebuild the snapshot"
    );
}

#[test]
fn failed_approved_mutation_does_not_invalidate_cache() {
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("base.txt"), "base\n").unwrap();

    let mut rt = make_runtime_in(vec!["Recovery."], tmp.path());
    let before = rt.project_snapshot_for_test().unwrap();

    fs::write(tmp.path().join("external.txt"), "external\n").unwrap();
    let cached = rt.project_snapshot_for_test().unwrap();
    assert_eq!(before, cached, "snapshot must stay cached before failure");

    let failed_target = tmp.path().join("missing").join("out.txt");
    rt.set_pending_for_test(PendingAction {
        tool_name: "write_file".into(),
        summary: "create missing/out.txt".into(),
        risk: RiskLevel::Medium,
        payload: format!("{}\x00hello\n", failed_target.display()),
    });

    let approve_events = collect_events(&mut rt, RuntimeRequest::Approve);
    assert!(
        !has_failed(&approve_events),
        "failed mutation should recover without RuntimeEvent::Failed: {approve_events:?}"
    );
    assert!(
        !failed_target.exists(),
        "failed write_file approval must not create the target"
    );

    let after = rt.project_snapshot_for_test().unwrap();
    assert_eq!(
        cached, after,
        "failed approved mutation must not invalidate the cached snapshot"
    );
    assert!(
        !after
            .entries
            .iter()
            .any(|entry| entry.path == "external.txt"),
        "failed approved mutation must not rebuild the snapshot"
    );
}
