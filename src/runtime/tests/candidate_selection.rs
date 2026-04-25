use super::*;
use crate::app::config::Config;
use crate::llm::backend::Role;
use crate::tools::default_registry;
use std::sync::{Arc, Mutex};

// Phase 12.1.1 — First-Read Candidate Selection
//
// Validates that after a non-empty search the runtime dispatches read_file for the
// most structurally relevant candidate before giving the model control. The model
// sees search + read results in its second generation round and synthesizes directly —
// it never selects the first file to read.

#[test]
fn usage_lookup_runtime_dispatches_non_definition_candidate_first() {
    // UsageLookup: search returns two candidates — definition-only file first (alphabetically)
    // and a usage-site file second. The runtime should dispatch read_file for the usage file,
    // not the definition file. The model's second round context must contain a read_file result
    // for services/runner.py without the model having chosen it.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("models")).unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    // "models/" < "services/" alphabetically — definition file is first in search results.
    fs::write(
        tmp.path().join("models/enums.py"),
        "class TaskStatus(str, Enum):\n    PENDING = \"pending\"\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("services/runner.py"),
        "from models.enums import TaskStatus\nif task.status == TaskStatus.PENDING:\n    run()\n",
    )
    .unwrap();

    let requests = Arc::new(Mutex::new(Vec::new()));
    // Model only needs to issue a search and then produce an answer — the runtime handles
    // the first read without the model requesting it.
    let mut rt = Runtime::new(
        &Config::default(),
        tmp.path(),
        Box::new(RecordingBackend::new(
            vec![
                "[search_code: TaskStatus]",
                "TaskStatus is used in services/runner.py.",
            ],
            Arc::clone(&requests),
        )),
        default_registry(tmp.path().to_path_buf()),
    );

    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is TaskStatus used?".into(),
        },
    );

    let requests = requests.lock().unwrap();
    assert!(
        requests.len() >= 2,
        "need at least 2 generation rounds: got {}",
        requests.len()
    );

    // Round 2 must already contain the read_file result for the usage candidate, injected by
    // the runtime — not by a model tool call.
    // Identify by content: path is not included in read_file result bodies.
    let round2_has_read_result = requests[1].messages.iter().any(|m| {
        m.role == Role::User
            && m.content.contains("=== tool_result: read_file ===")
            && m.content.contains("TaskStatus.PENDING")
    });
    assert!(
        round2_has_read_result,
        "round 2 context must contain a runtime-dispatched read_file result for services/runner.py"
    );

    // The runtime must NOT have auto-read the definition-only file.
    let round2_read_definition = requests[1].messages.iter().any(|m| {
        m.role == Role::User
            && m.content.contains("=== tool_result: read_file ===")
            && m.content.contains("class TaskStatus")
    });
    assert!(
        !round2_read_definition,
        "runtime must not dispatch read_file for the definition-only candidate on a usage lookup"
    );
}

#[test]
fn initialization_lookup_runtime_dispatches_init_candidate_first() {
    // InitializationLookup: search returns a non-init file first (alphabetically) and an
    // init-site file second. Runtime must dispatch read_file for the init candidate.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    // "logger_app" < "logger_init" alphabetically — non-init file is first in search results.
    fs::write(
        tmp.path().join("services/logger_app.py"),
        "def emit_log(logger):\n    logger.info(\"logging event\")\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("services/logger_init.py"),
        "def initialize_logging():\n    logging.basicConfig(level=\"INFO\")\n",
    )
    .unwrap();

    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut rt = Runtime::new(
        &Config::default(),
        tmp.path(),
        Box::new(RecordingBackend::new(
            vec![
                "[search_code: logging]",
                "Logging is initialized in services/logger_init.py.",
            ],
            Arc::clone(&requests),
        )),
        default_registry(tmp.path().to_path_buf()),
    );

    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Find where logging is initialized in services/".into(),
        },
    );

    let requests = requests.lock().unwrap();
    assert!(requests.len() >= 2, "need at least 2 rounds");

    // Identify files by content: path is not included in read_file result bodies.
    let round2_has_init_read = requests[1].messages.iter().any(|m| {
        m.role == Role::User
            && m.content.contains("=== tool_result: read_file ===")
            && m.content.contains("initialize_logging")
    });
    assert!(
        round2_has_init_read,
        "round 2 must contain a runtime-dispatched read result for the initialization candidate"
    );

    // Non-init file must not have been auto-read.
    let round2_has_non_init_read = requests[1].messages.iter().any(|m| {
        m.role == Role::User
            && m.content.contains("=== tool_result: read_file ===")
            && m.content.contains("emit_log")
    });
    assert!(
        !round2_has_non_init_read,
        "runtime must not dispatch read_file for the non-initialization candidate on an init lookup"
    );
}

#[test]
fn single_candidate_search_auto_reads_that_candidate() {
    // With a single search candidate, the runtime still dispatches read_file for it —
    // there is exactly one candidate so selection is trivial.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("engine.rs"), "fn run_turns() {}\n").unwrap();

    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut rt = Runtime::new(
        &Config::default(),
        tmp.path(),
        Box::new(RecordingBackend::new(
            vec![
                "[search_code: run_turns]",
                "run_turns is defined in engine.rs.",
            ],
            Arc::clone(&requests),
        )),
        default_registry(tmp.path().to_path_buf()),
    );

    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is run_turns defined?".into(),
        },
    );

    let requests = requests.lock().unwrap();
    assert!(requests.len() >= 2, "need at least 2 rounds");

    // Round 2 must already contain the read_file result for the single candidate.
    let round2_has_read = requests[1].messages.iter().any(|m| {
        m.role == Role::User && m.content.contains("=== tool_result: read_file ===")
    });
    assert!(
        round2_has_read,
        "single-candidate search must still produce a runtime-dispatched read in round 2"
    );
}

#[test]
fn general_mode_search_auto_reads_first_source_candidate() {
    // General investigation mode has no structural preference — the runtime dispatches
    // the first non-lockfile source candidate.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("engine.rs"), "fn run_turns() {}\n").unwrap();
    fs::write(tmp.path().join("runner.rs"), "fn run_turns() {}\n").unwrap();

    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut rt = Runtime::new(
        &Config::default(),
        tmp.path(),
        Box::new(RecordingBackend::new(
            vec![
                "[search_code: run_turns]",
                "run_turns drives the main loop.",
            ],
            Arc::clone(&requests),
        )),
        default_registry(tmp.path().to_path_buf()),
    );

    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            // "What does" is general — no usage/definition/init terms → General mode.
            text: "What does run_turns do?".into(),
        },
    );

    let requests = requests.lock().unwrap();
    assert!(requests.len() >= 2, "need at least 2 rounds");

    // Round 2 must contain a read_file result — the runtime dispatches the first source
    // candidate even in General mode.
    let round2_has_read = requests[1].messages.iter().any(|m| {
        m.role == Role::User && m.content.contains("=== tool_result: read_file ===")
    });
    assert!(
        round2_has_read,
        "general mode must still produce a runtime-dispatched read for the first source candidate"
    );
}

#[test]
fn usage_lookup_runtime_prefers_substantive_candidate_over_import_only() {
    // When non-definition candidates are all import-only AND a substantive usage candidate
    // exists, the runtime must dispatch the substantive one, not the import-only file.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("init")).unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    // "init/" < "services/" alphabetically — import-only file is first in results.
    fs::write(
        tmp.path().join("init/header.py"),
        "from models.enums import TaskStatus\n",
    )
    .unwrap();
    // Substantive usage: has a non-import, non-definition usage line.
    fs::write(
        tmp.path().join("services/task_service.py"),
        "if task.status == TaskStatus.TODO:\n    pass\n",
    )
    .unwrap();
    fs::create_dir_all(tmp.path().join("models")).unwrap();
    fs::write(
        tmp.path().join("models/enums.py"),
        "class TaskStatus(str, Enum):\n    TODO = \"todo\"\n",
    )
    .unwrap();

    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut rt = Runtime::new(
        &Config::default(),
        tmp.path(),
        Box::new(RecordingBackend::new(
            vec![
                "[search_code: TaskStatus]",
                "TaskStatus is used in services/task_service.py.",
            ],
            Arc::clone(&requests),
        )),
        default_registry(tmp.path().to_path_buf()),
    );

    collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is TaskStatus used?".into(),
        },
    );

    let requests = requests.lock().unwrap();
    assert!(requests.len() >= 2, "need at least 2 rounds");

    // The runtime must have dispatched read_file for the substantive candidate.
    // Identify by content: path is not included in read_file result bodies.
    let round2_has_service_read = requests[1].messages.iter().any(|m| {
        m.role == Role::User
            && m.content.contains("=== tool_result: read_file ===")
            && m.content.contains("TaskStatus.TODO")
    });
    assert!(
        round2_has_service_read,
        "runtime must prefer the substantive usage candidate over the import-only file"
    );

    // Import-only file must not have been auto-read.
    // header.py contains only the import line; its read result would include it.
    let round2_has_import_read = requests[1].messages.iter().any(|m| {
        m.role == Role::User
            && m.content.contains("=== tool_result: read_file ===")
            && m.content.contains("from models.enums import TaskStatus")
    });
    assert!(
        !round2_has_import_read,
        "runtime must not dispatch read_file for the import-only candidate when substantive candidates exist"
    );
}
