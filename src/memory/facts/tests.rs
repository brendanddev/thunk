use super::*;
use rusqlite::Connection;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_db_path(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    std::env::temp_dir().join(format!("params-facts-test-{label}-{nonce}.db"))
}

#[test]
fn quality_fact_rejects_short() {
    assert!(!is_quality_fact("too short"));
}

#[test]
fn quality_fact_rejects_question() {
    assert!(!is_quality_fact(
        "Is this the right approach for the cache?"
    ));
}

#[test]
fn quality_fact_rejects_meta_commentary() {
    assert!(!is_quality_fact(
        "The user asked about fixing the inference loop"
    ));
}

#[test]
fn quality_fact_accepts_concrete_fact() {
    assert!(is_quality_fact(
        "src/memory/facts.rs uses Jaccard similarity for near-duplicate detection"
    ));
}

#[test]
fn quality_fact_rejects_bullet_style_lines() {
    assert!(!is_quality_fact(
        "- C/C++ uses raw pointers directly in the language"
    ));
}

#[test]
fn quality_fact_rejects_code_snippet_lines() {
    assert!(!is_quality_fact(
        "let s = String::from(\"rust pointer example\");"
    ));
    assert!(!is_quality_fact(
        "pointer = &s[3..6]; // Raw pointer to a string slice"
    ));
    assert!(!is_quality_fact("let part = &s[3..6];"));
}

#[test]
fn quality_fact_rejects_file_boilerplate_lines() {
    assert!(!is_quality_fact(
        "This file (`src/main.rs`) is the entry point of the Rust project `params-cli`."
    ));
}

#[test]
fn quality_fact_rejects_summary_boilerplate_lines() {
    assert!(!is_quality_fact(
        "/Users/brendandileo/Desktop/BDrive/params-cli/docs/context/PLANS.md: Describes project profiles, cache invalidation techniques, memory consolidation methods, and lifecycle hooks for the params-cli project."
    ));
}

#[test]
fn validation_rejects_hedged_fact() {
    let mut anchors = HashSet::new();
    anchors.insert("src/main.rs".to_string());
    assert_eq!(
        validate_candidate_fact("src/main.rs probably updates the cache logic", &anchors),
        Err(SkippedFactReason::Hedged)
    );
}

#[test]
fn validation_rejects_unanchored_fact() {
    let anchors = HashSet::new();
    assert_eq!(
        validate_candidate_fact("src/main.rs updates the cache logic", &anchors),
        Err(SkippedFactReason::Unanchored)
    );
}

#[test]
fn validation_accepts_anchored_fact() {
    let mut anchors = HashSet::new();
    anchors.insert("src/main.rs".to_string());
    assert_eq!(
        validate_candidate_fact("src/main.rs updates the cache logic", &anchors),
        Ok(())
    );
}

#[test]
fn near_duplicate_detects_high_overlap() {
    let a = "src/inference/mod.rs owns the session messages and tool call loop";
    let b = "src/inference/mod.rs owns the session messages and the tool call loop";
    assert!(are_near_duplicate(a, b));
}

#[test]
fn turn_anchors_include_paths_and_symbols() {
    let evidence = TurnMemoryEvidence {
        user_prompt: "Review src/main.rs and FactStore".to_string(),
        summaries: vec![(
            "src/memory/facts.rs".to_string(),
            "FactStore extracts verified facts".to_string(),
        )],
        tool_results: Vec::new(),
        final_response: Some("FactStore now records verified facts".to_string()),
    };
    let anchors = evidence_anchors(&evidence);
    assert!(anchors.contains("src/main.rs"));
    assert!(anchors.contains("factstore"));
    assert!(anchors.contains("src/memory/facts.rs"));
}

#[test]
fn generic_answer_content_does_not_create_project_anchors() {
    let evidence = TurnMemoryEvidence {
        user_prompt: "So no other languages have pointers?".to_string(),
        summaries: Vec::new(),
        tool_results: Vec::new(),
        final_response: Some(
            "No, other languages also have pointers, but they handle them differently.".to_string(),
        ),
    };

    let anchors = evidence_anchors(&evidence);
    assert!(anchors.is_empty());
    assert_eq!(
        validate_candidate_fact(
            "No, other languages also have pointers, but they handle them differently.",
            &anchors
        ),
        Err(SkippedFactReason::Quality)
    );
}

#[test]
fn store_promotes_legacy_duplicate_to_verified() {
    let path = temp_db_path("promote");
    let store = FactStore::open_at(&path).expect("open fact store");
    store
        .try_store_fact_deduped(
            "project",
            "src/main.rs updates cache stats in the sidebar",
            FactProvenance::Legacy,
        )
        .expect("store legacy fact");

    let outcome = store
        .try_store_fact_deduped(
            "project",
            "src/main.rs updates cache stats in the sidebar",
            FactProvenance::Verified,
        )
        .expect("store verified duplicate");
    assert!(matches!(outcome, StoreFactOutcome::Duplicate));

    let facts = store
        .get_relevant_facts("project", "", 10)
        .expect("load facts");
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].provenance, FactProvenance::Verified);

    let _ = fs::remove_file(path);
}

#[test]
fn provenance_migration_defaults_existing_rows_to_legacy() {
    let path = temp_db_path("migration");
    let conn = Connection::open(&path).expect("open sqlite");
    conn.execute_batch(
        "CREATE TABLE facts (
            id INTEGER PRIMARY KEY,
            project TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            last_seen INTEGER NOT NULL
        );
        INSERT INTO facts (project, content, created_at, last_seen)
        VALUES ('project', 'src/main.rs updates cache stats in the sidebar', 1, 1);",
    )
    .expect("seed legacy table");
    drop(conn);

    let store = FactStore::open_at(&path).expect("migrate store");
    let facts = store
        .get_relevant_facts("project", "", 10)
        .expect("load facts");
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].provenance, FactProvenance::Legacy);

    let _ = fs::remove_file(path);
}

#[test]
fn relevant_facts_are_ranked_by_prompt_match() {
    let path = temp_db_path("ranking");
    let store = FactStore::open_at(&path).expect("open fact store");
    store
        .try_store_fact_deduped(
            "project",
            "src/session/mod.rs resolves session selectors by unique id prefix",
            FactProvenance::Verified,
        )
        .expect("store first fact");
    store
        .try_store_fact_deduped(
            "project",
            "Cache stats are shown in the runtime header",
            FactProvenance::Verified,
        )
        .expect("store second fact");

    let facts = store
        .get_relevant_facts("project", "src/session/mod.rs selector", 5)
        .expect("load ranked facts");

    assert_eq!(
        facts.first().map(|fact| fact.content.as_str()),
        Some("src/session/mod.rs resolves session selectors by unique id prefix")
    );

    let _ = fs::remove_file(path);
}

#[test]
fn retrieval_filters_generic_unanchored_facts() {
    let path = temp_db_path("filter-generic");
    let store = FactStore::open_at(&path).expect("open fact store");
    store
        .try_store_fact_deduped(
            "project",
            "No, other languages also have pointers, but they handle them differently.",
            FactProvenance::Verified,
        )
        .expect("store generic fact");
    store
        .try_store_fact_deduped(
            "project",
            "src/session/mod.rs resolves session selectors by unique id prefix",
            FactProvenance::Verified,
        )
        .expect("store anchored fact");

    let facts = store
        .get_relevant_facts("project", "", 10)
        .expect("load filtered facts");

    assert_eq!(facts.len(), 1);
    assert_eq!(
        facts[0].content,
        "src/session/mod.rs resolves session selectors by unique id prefix"
    );

    let _ = fs::remove_file(path);
}

#[test]
fn prune_irrelevant_facts_removes_generic_entries() {
    let path = temp_db_path("prune-generic");
    let store = FactStore::open_at(&path).expect("open fact store");
    store
        .try_store_fact_deduped(
            "project",
            "No, other languages also have pointers, but they handle them differently.",
            FactProvenance::Verified,
        )
        .expect("store generic fact");
    store
        .try_store_fact_deduped(
            "project",
            "src/session/mod.rs resolves session selectors by unique id prefix",
            FactProvenance::Verified,
        )
        .expect("store anchored fact");

    let removed = store
        .prune_irrelevant_facts("project")
        .expect("prune irrelevant facts");
    assert_eq!(removed, 1);

    let facts = store
        .get_relevant_facts("project", "", 10)
        .expect("load pruned facts");
    assert_eq!(facts.len(), 1);
    assert_eq!(
        facts[0].content,
        "src/session/mod.rs resolves session selectors by unique id prefix"
    );

    let _ = fs::remove_file(path);
}

#[test]
fn generic_pointer_fact_is_not_retrievable_project_fact() {
    assert!(!is_retrievable_project_fact(
        "Yes, Rust does have pointers, but they are managed by the Rust compiler and runtime and are typically used through smart pointers like Box, Rc, RefCell, and UnsafeCell."
    ));
}

#[test]
fn instructional_proposal_fact_is_not_retrievable_project_fact() {
    assert!(!is_retrievable_project_fact(
        "To store cache entries in memory instead of a database, we can replace the database operations with in-memory storage using std::collections::HashMap. This will allow us to cache data in memory without the overhead of a SQLite database."
    ));
}

#[test]
fn factstore_behavior_fact_remains_retrievable() {
    assert!(is_retrievable_project_fact(
        "FactStore uses Jaccard similarity to deduplicate near-duplicate verified memory facts."
    ));
}

#[test]
fn snippet_and_boilerplate_facts_are_not_retrievable() {
    assert!(!is_retrievable_project_fact(
        "let s = String::from(\"rust pointer example\");"
    ));
    assert!(!is_retrievable_project_fact(
        "This file (`src/main.rs`) is the entry point of the Rust project `params-cli`."
    ));
    assert!(!is_retrievable_project_fact(
        "/Users/brendandileo/Desktop/BDrive/params-cli/docs/context/PLANS.md: Describes project profiles, cache invalidation techniques, memory consolidation methods, and lifecycle hooks for the params-cli project."
    ));
}
