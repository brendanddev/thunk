use super::*;
use crate::runtime::types::RuntimeTerminalReason;

#[test]
fn config_lookup_non_config_read_triggers_recovery_to_config_file() {
    // Config lookup: two candidates — a source file and a config file.
    // Model reads the source file first → runtime injects config recovery pointing to YAML.
    // Model follows recovery and reads the config file → evidence ready → ToolAssisted.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::create_dir_all(tmp.path().join("config")).unwrap();
    fs::write(
        tmp.path().join("services").join("database.py"),
        "DATABASE_URL = os.getenv(\"DATABASE_URL\")\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("config").join("database.yaml"),
        "database:\n  url: postgres://localhost/mydb\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: database]",
            "[read_file: services/database.py]",
            "[read_file: config/database.yaml]",
            "The database is configured in config/database.yaml.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is the database configured?".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");
    let answer_source = events.iter().find_map(|e| {
        if let RuntimeEvent::AnswerReady(src) = e {
            Some(src.clone())
        } else {
            None
        }
    });
    assert!(
        matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
        "config recovery + config read must admit synthesis: {answer_source:?}"
    );
    let snapshot = rt.messages_snapshot();
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(
        last_assistant,
        Some("The database is configured in config/database.yaml.")
    );
}

#[test]
fn config_lookup_no_config_candidates_degrades_cleanly() {
    // Config lookup triggered, but no config-file candidates exist (source files only).
    // has_non_config_candidates = true, config_file_candidates is empty.
    // Gate 2 does not fire — source file read is accepted → ToolAssisted.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::write(
        tmp.path().join("services").join("database.py"),
        "DATABASE_URL = os.getenv(\"DATABASE_URL\")\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: database]",
            "[read_file: services/database.py]",
            "The database connection is set up in services/database.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where is the database configured?".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");
    let answer_source = events.iter().find_map(|e| {
        if let RuntimeEvent::AnswerReady(src) = e {
            Some(src.clone())
        } else {
            None
        }
    });
    assert!(
        matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
        "config lookup with no config candidates must degrade to acceptance: {answer_source:?}"
    );
    let snapshot = rt.messages_snapshot();
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(
        last_assistant,
        Some("The database connection is set up in services/database.py.")
    );
}

#[test]
fn create_lookup_non_create_read_triggers_recovery_to_create_file() {
    // File A: no create-term matches → non-create candidate.
    // File B: a create-term match → create candidate.
    // Model reads A first → recovery fires pointing to B.
    // Model reads B → evidence ready → ToolAssisted.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::create_dir_all(tmp.path().join("storage")).unwrap();
    fs::write(
        tmp.path().join("services").join("task_handler.py"),
        "def handle_task(task):\n    task.run()\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("storage").join("task_store.py"),
        "def store_task(task):\n    db.create(task)\n    return task.id\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: task]",
            "[read_file: services/task_handler.py]",
            "[read_file: storage/task_store.py]",
            "Tasks are created in storage/task_store.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where are tasks created?".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");

    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("creation lookup")
                && m.content.contains("storage/task_store.py")),
        "create recovery correction must point to the create candidate"
    );

    let answer_source = events.iter().find_map(|e| {
        if let RuntimeEvent::AnswerReady(src) = e {
            Some(src.clone())
        } else {
            None
        }
    });
    assert!(
        matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
        "create lookup + recovery + create read must admit synthesis: {answer_source:?}"
    );
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(
        last_assistant,
        Some("Tasks are created in storage/task_store.py.")
    );
}

#[test]
fn create_lookup_no_create_candidates_degrades_cleanly() {
    // All candidates have no create-term matches.
    // Gate does not fire — any read is accepted (fallback behavior).
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::write(
        tmp.path().join("services").join("task_handler.py"),
        "def handle_task(task):\n    task.run()\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: task]",
            "[read_file: services/task_handler.py]",
            "Tasks are handled in services/task_handler.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where are tasks created?".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");
    let answer_source = events.iter().find_map(|e| {
        if let RuntimeEvent::AnswerReady(src) = e {
            Some(src.clone())
        } else {
            None
        }
    });
    assert!(
        matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
        "create lookup with no create candidates must degrade to acceptance: {answer_source:?}"
    );
    let snapshot = rt.messages_snapshot();
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(
        last_assistant,
        Some("Tasks are handled in services/task_handler.py.")
    );
}

#[test]
fn create_lookup_second_non_create_candidate_after_recovery_is_not_accepted() {
    // After one recovery the correction flag is set.
    // A second non-create read falls through the gate without accepting.
    // With candidate_reads_count == 2 and evidence_ready false, the runtime
    // terminates with InsufficientEvidence.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::create_dir_all(tmp.path().join("handlers")).unwrap();
    fs::create_dir_all(tmp.path().join("storage")).unwrap();
    fs::write(
        tmp.path().join("services").join("runner.py"),
        "def run_task(task):\n    task.execute()\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("handlers").join("task_handler.py"),
        "def handle_task(task):\n    pass\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("storage").join("task_store.py"),
        "def store_task(task):\n    db.create(task)\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: task]",
            "[read_file: services/runner.py]",
            "[read_file: handlers/task_handler.py]",
            "Tasks run in services/runner.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where are tasks created?".into(),
        },
    );

    assert!(!has_failed(&events), "must terminate cleanly: {events:?}");
    let answer_source = events.iter().find_map(|e| {
        if let RuntimeEvent::AnswerReady(src) = e {
            Some(src.clone())
        } else {
            None
        }
    });
    assert!(
        matches!(
            answer_source,
            Some(AnswerSource::RuntimeTerminal {
                reason: RuntimeTerminalReason::InsufficientEvidence,
                ..
            })
        ),
        "two non-create reads must terminate with InsufficientEvidence: {answer_source:?}"
    );
}

#[test]
fn create_lookup_noisy_create_term_in_comment_still_classifies_as_create() {
    // A line like "# TODO: create session handling" contains "create" as substring.
    // The classification is structural/substring — comments match the same as code.
    // This tests the known noisy behavior described in the spec.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::create_dir_all(tmp.path().join("models")).unwrap();
    fs::write(
        tmp.path().join("services").join("session_service.py"),
        "# TODO: create session handling\ndef get_session(sid):\n    return db.get(sid)\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("models").join("session.py"),
        "class Session:\n    pass\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: session]",
            "[read_file: services/session_service.py]",
            "Sessions are handled in services/session_service.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where are sessions created?".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");
    let snapshot = rt.messages_snapshot();
    assert!(
        !snapshot
            .iter()
            .any(|m| m.content.contains("creation lookup")),
        "no recovery expected when create candidate is read first"
    );
    let answer_source = events.iter().find_map(|e| {
        if let RuntimeEvent::AnswerReady(src) = e {
            Some(src.clone())
        } else {
            None
        }
    });
    assert!(
        matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
        "create candidate read must admit synthesis: {answer_source:?}"
    );
}

#[test]
fn register_lookup_non_register_read_triggers_recovery_to_register_file() {
    // File A: no register-term matches → non-register candidate.
    // File B: a register-term match → register candidate.
    // Model reads A first → recovery fires pointing to B.
    // Model reads B → evidence ready → ToolAssisted.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("cli")).unwrap();
    fs::write(
        tmp.path().join("cli").join("handlers.py"),
        "def handle_command(command):\n    return command.run()\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("cli").join("registry.py"),
        "def wire_command(command):\n    registry.register(command)\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: command]",
            "[read_file: cli/handlers.py]",
            "[read_file: cli/registry.py]",
            "Commands are registered in cli/registry.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where are commands registered?".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");

    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot
            .iter()
            .any(|m| m.content.contains("registration lookup")
                && m.content.contains("cli/registry.py")),
        "register recovery correction must point to the register candidate"
    );

    let answer_source = events.iter().find_map(|e| {
        if let RuntimeEvent::AnswerReady(src) = e {
            Some(src.clone())
        } else {
            None
        }
    });
    assert!(
        matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
        "register lookup + recovery + register read must admit synthesis: {answer_source:?}"
    );
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(
        last_assistant,
        Some("Commands are registered in cli/registry.py.")
    );
}

#[test]
fn register_lookup_no_register_candidates_degrades_cleanly() {
    // All candidates have no register-term matches.
    // Gate does not fire — any read is accepted (fallback behavior).
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("cli")).unwrap();
    fs::write(
        tmp.path().join("cli").join("handlers.py"),
        "def handle_command(command):\n    return command.run()\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: command]",
            "[read_file: cli/handlers.py]",
            "Commands are handled in cli/handlers.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where are commands registered?".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");
    let answer_source = events.iter().find_map(|e| {
        if let RuntimeEvent::AnswerReady(src) = e {
            Some(src.clone())
        } else {
            None
        }
    });
    assert!(
        matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
        "register lookup with no register candidates must degrade to acceptance: {answer_source:?}"
    );
    let snapshot = rt.messages_snapshot();
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(last_assistant, Some("Commands are handled in cli/handlers.py."));
}

#[test]
fn register_lookup_second_non_register_candidate_after_recovery_is_not_accepted() {
    // After one recovery the correction flag is set.
    // A second non-register read falls through the gate without accepting.
    // With candidate_reads_count == 2 and evidence_ready false, the runtime
    // terminates with InsufficientEvidence.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("cli")).unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::write(
        tmp.path().join("cli").join("handlers.py"),
        "def handle_command(command):\n    return command.run()\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("services").join("command_runner.py"),
        "def run_command(command):\n    command.run()\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("cli").join("registry.py"),
        "def wire_command(command):\n    registry.register(command)\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: command]",
            "[read_file: cli/handlers.py]",
            "[read_file: services/command_runner.py]",
            "Commands are registered in cli/handlers.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where are commands registered?".into(),
        },
    );

    assert!(!has_failed(&events), "must terminate cleanly: {events:?}");
    let answer_source = events.iter().find_map(|e| {
        if let RuntimeEvent::AnswerReady(src) = e {
            Some(src.clone())
        } else {
            None
        }
    });
    assert!(
        matches!(
            answer_source,
            Some(AnswerSource::RuntimeTerminal {
                reason: RuntimeTerminalReason::InsufficientEvidence,
                ..
            })
        ),
        "two non-register reads must terminate with InsufficientEvidence: {answer_source:?}"
    );
}

#[test]
fn register_lookup_noisy_register_term_in_comment_still_classifies_as_register() {
    // A line like "# TODO: register command handler" contains "register".
    // The classification is structural/substring — comments match the same as code.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("cli")).unwrap();
    fs::write(
        tmp.path().join("cli").join("commands.py"),
        "# TODO: register command handler\ndef command_handler(command):\n    return command.run()\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: command]",
            "[read_file: cli/commands.py]",
            "Commands are handled in cli/commands.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where are commands registered?".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");
    let snapshot = rt.messages_snapshot();
    assert!(
        !snapshot
            .iter()
            .any(|m| m.content.contains("registration lookup")),
        "no recovery expected when register candidate is read first"
    );
    let answer_source = events.iter().find_map(|e| {
        if let RuntimeEvent::AnswerReady(src) = e {
            Some(src.clone())
        } else {
            None
        }
    });
    assert!(
        matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
        "register candidate read must admit synthesis: {answer_source:?}"
    );
}

#[test]
fn load_lookup_non_load_read_triggers_recovery_to_load_file() {
    // File A: no load-term matches → non-load candidate.
    // File B: a load-term match → load candidate.
    // Model reads A first → recovery fires pointing to B.
    // Model reads B → evidence ready → ToolAssisted.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::write(
        tmp.path().join("services").join("session_handler.py"),
        "def handle_session(session):\n    return session.id\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("services").join("session_loader.py"),
        "def get_session(session_id):\n    return load_session(session_id)\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: session]",
            "[read_file: services/session_handler.py]",
            "[read_file: services/session_loader.py]",
            "Sessions are loaded in services/session_loader.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where are sessions loaded?".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");

    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot.iter().any(|m| m.content.contains("load lookup")
            && m.content.contains("services/session_loader.py")),
        "load recovery correction must point to the load candidate"
    );

    let answer_source = events.iter().find_map(|e| {
        if let RuntimeEvent::AnswerReady(src) = e {
            Some(src.clone())
        } else {
            None
        }
    });
    assert!(
        matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
        "load lookup + recovery + load read must admit synthesis: {answer_source:?}"
    );
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(
        last_assistant,
        Some("Sessions are loaded in services/session_loader.py.")
    );
}

#[test]
fn load_lookup_no_load_candidates_degrades_cleanly() {
    // All candidates have no load-term matches.
    // Gate does not fire — any read is accepted (fallback behavior).
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::write(
        tmp.path().join("services").join("session_handler.py"),
        "def handle_session(session):\n    return session.id\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: session]",
            "[read_file: services/session_handler.py]",
            "Sessions are handled in services/session_handler.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where are sessions loaded?".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");
    let answer_source = events.iter().find_map(|e| {
        if let RuntimeEvent::AnswerReady(src) = e {
            Some(src.clone())
        } else {
            None
        }
    });
    assert!(
        matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
        "load lookup with no load candidates must degrade to acceptance: {answer_source:?}"
    );
    let snapshot = rt.messages_snapshot();
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(
        last_assistant,
        Some("Sessions are handled in services/session_handler.py.")
    );
}

#[test]
fn load_lookup_second_non_load_candidate_after_recovery_is_not_accepted() {
    // After one recovery the correction flag is set.
    // A second non-load read falls through the gate without accepting.
    // With candidate_reads_count == 2 and evidence_ready false, the runtime
    // terminates with InsufficientEvidence.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::create_dir_all(tmp.path().join("controllers")).unwrap();
    fs::write(
        tmp.path().join("services").join("session_handler.py"),
        "def handle_session(session):\n    return session.id\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("controllers").join("session_controller.py"),
        "def show_session(session):\n    return session.id\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("services").join("session_loader.py"),
        "def get_session(session_id):\n    return load_session(session_id)\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: session]",
            "[read_file: services/session_handler.py]",
            "[read_file: controllers/session_controller.py]",
            "Sessions are loaded in services/session_handler.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where are sessions loaded?".into(),
        },
    );

    assert!(!has_failed(&events), "must terminate cleanly: {events:?}");
    let answer_source = events.iter().find_map(|e| {
        if let RuntimeEvent::AnswerReady(src) = e {
            Some(src.clone())
        } else {
            None
        }
    });
    assert!(
        matches!(
            answer_source,
            Some(AnswerSource::RuntimeTerminal {
                reason: RuntimeTerminalReason::InsufficientEvidence,
                ..
            })
        ),
        "two non-load reads must terminate with InsufficientEvidence: {answer_source:?}"
    );
}

#[test]
fn load_lookup_noisy_load_term_in_comment_still_classifies_as_load() {
    // A line like "# TODO: load session data" contains "load".
    // The classification is structural/substring — comments match the same as code.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::write(
        tmp.path().join("services").join("session_service.py"),
        "# TODO: load session data\ndef handle_session(session):\n    return session.id\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: session]",
            "[read_file: services/session_service.py]",
            "Sessions are handled in services/session_service.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where are sessions loaded?".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");
    let snapshot = rt.messages_snapshot();
    assert!(
        !snapshot.iter().any(|m| m.content.contains("load lookup")),
        "no recovery expected when load candidate is read first"
    );
    let answer_source = events.iter().find_map(|e| {
        if let RuntimeEvent::AnswerReady(src) = e {
            Some(src.clone())
        } else {
            None
        }
    });
    assert!(
        matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
        "load candidate read must admit synthesis: {answer_source:?}"
    );
}

#[test]
fn save_lookup_non_save_read_triggers_recovery_to_save_file() {
    // File A: no save-term matches → non-save candidate.
    // File B: a save-term match → save candidate.
    // Model reads A first → recovery fires pointing to B.
    // Model reads B → evidence ready → ToolAssisted.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::write(
        tmp.path().join("services").join("session_handler.py"),
        "def handle_session(session):\n    return session.id\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("services").join("session_store.py"),
        "def store_session(session):\n    save_session(session)\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: session]",
            "[read_file: services/session_handler.py]",
            "[read_file: services/session_store.py]",
            "Sessions are saved in services/session_store.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where are sessions saved?".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");

    let snapshot = rt.messages_snapshot();
    assert!(
        snapshot.iter().any(|m| m.content.contains("save lookup")
            && m.content.contains("services/session_store.py")),
        "save recovery correction must point to the save candidate"
    );

    let answer_source = events.iter().find_map(|e| {
        if let RuntimeEvent::AnswerReady(src) = e {
            Some(src.clone())
        } else {
            None
        }
    });
    assert!(
        matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
        "save lookup + recovery + save read must admit synthesis: {answer_source:?}"
    );
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(
        last_assistant,
        Some("Sessions are saved in services/session_store.py.")
    );
}

#[test]
fn save_lookup_no_save_candidates_degrades_cleanly() {
    // All candidates have no save-term matches.
    // Gate does not fire — any read is accepted (fallback behavior).
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::write(
        tmp.path().join("services").join("session_handler.py"),
        "def handle_session(session):\n    return session.id\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: session]",
            "[read_file: services/session_handler.py]",
            "Sessions are handled in services/session_handler.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where are sessions saved?".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");
    let answer_source = events.iter().find_map(|e| {
        if let RuntimeEvent::AnswerReady(src) = e {
            Some(src.clone())
        } else {
            None
        }
    });
    assert!(
        matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
        "save lookup with no save candidates must degrade to acceptance: {answer_source:?}"
    );
    let snapshot = rt.messages_snapshot();
    let last_assistant = snapshot
        .iter()
        .rev()
        .find(|m| m.role == crate::llm::backend::Role::Assistant)
        .map(|m| m.content.as_str());
    assert_eq!(
        last_assistant,
        Some("Sessions are handled in services/session_handler.py.")
    );
}

#[test]
fn save_lookup_second_non_save_candidate_after_recovery_is_not_accepted() {
    // After one recovery the correction flag is set.
    // A second non-save read falls through the gate without accepting.
    // With candidate_reads_count == 2 and evidence_ready false, the runtime
    // terminates with InsufficientEvidence.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::create_dir_all(tmp.path().join("controllers")).unwrap();
    fs::write(
        tmp.path().join("services").join("session_handler.py"),
        "def handle_session(session):\n    return session.id\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("controllers").join("session_controller.py"),
        "def show_session(session):\n    return session.id\n",
    )
    .unwrap();
    fs::write(
        tmp.path().join("services").join("session_store.py"),
        "def store_session(session):\n    save_session(session)\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: session]",
            "[read_file: services/session_handler.py]",
            "[read_file: controllers/session_controller.py]",
            "Sessions are saved in services/session_handler.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where are sessions saved?".into(),
        },
    );

    assert!(!has_failed(&events), "must terminate cleanly: {events:?}");
    let answer_source = events.iter().find_map(|e| {
        if let RuntimeEvent::AnswerReady(src) = e {
            Some(src.clone())
        } else {
            None
        }
    });
    assert!(
        matches!(
            answer_source,
            Some(AnswerSource::RuntimeTerminal {
                reason: RuntimeTerminalReason::InsufficientEvidence,
                ..
            })
        ),
        "two non-save reads must terminate with InsufficientEvidence: {answer_source:?}"
    );
}

#[test]
fn save_lookup_noisy_save_term_in_comment_still_classifies_as_save() {
    // A line like "# TODO: save session data" contains "save".
    // The classification is structural/substring — comments match the same as code.
    use std::fs;
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("services")).unwrap();
    fs::write(
        tmp.path().join("services").join("session_service.py"),
        "# TODO: save session data\ndef handle_session(session):\n    return session.id\n",
    )
    .unwrap();

    let mut rt = make_runtime_in(
        vec![
            "[search_code: session]",
            "[read_file: services/session_service.py]",
            "Sessions are handled in services/session_service.py.",
        ],
        tmp.path(),
    );

    let events = collect_events(
        &mut rt,
        RuntimeRequest::Submit {
            text: "Where are sessions saved?".into(),
        },
    );

    assert!(!has_failed(&events), "turn must not fail: {events:?}");
    let snapshot = rt.messages_snapshot();
    assert!(
        !snapshot.iter().any(|m| m.content.contains("save lookup")),
        "no recovery expected when save candidate is read first"
    );
    let answer_source = events.iter().find_map(|e| {
        if let RuntimeEvent::AnswerReady(src) = e {
            Some(src.clone())
        } else {
            None
        }
    });
    assert!(
        matches!(answer_source, Some(AnswerSource::ToolAssisted { .. })),
        "save candidate read must admit synthesis: {answer_source:?}"
    );
}
