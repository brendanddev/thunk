use std::time::{SystemTime, UNIX_EPOCH};

/// Opaque 16-char lowercase hex string. Unique enough for session IDs without a crypto dep.
pub type SessionId = String;

/// Lightweight metadata returned by list() and save(). Does not include message bodies.
#[derive(Debug, Clone)]
pub struct SessionMeta {
    pub id: SessionId,
    pub project_root: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
    pub message_count: usize,
}

/// A single message as stored on disk. Uses String for role to stay decoupled from the
/// runtime's typed Role enum — conversion is explicit at the store boundary.
#[derive(Debug, Clone)]
pub struct StoredMessage {
    pub role: String,
    pub content: String,
}

/// A fully loaded session: metadata + all messages in order.
#[derive(Debug, Clone)]
pub struct SavedSession {
    pub meta: SessionMeta,
    pub messages: Vec<StoredMessage>,
}

/// Returns current time as milliseconds since the Unix epoch.
/// Millisecond resolution prevents timestamp collisions in rapid test sequences
/// that would make ORDER BY updated_at non-deterministic.
pub(super) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

pub(super) fn generate_session_id() -> SessionId {
    // XOR nanos with pid for sufficient uniqueness without pulling in a uuid/rand crate.
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let unique = nanos ^ (std::process::id() as u128);
    format!("{:016x}", unique & 0xFFFF_FFFF_FFFF_FFFF)
}
