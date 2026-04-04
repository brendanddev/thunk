use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use tracing::{debug, info, warn};

use crate::memory::index::ProjectIndex;

use super::InferenceBackend;

pub(super) struct IncrementalIndexState {
    pending_files: VecDeque<PathBuf>,
    next_scan_at: Instant,
    scan_interval: Duration,
}

impl IncrementalIndexState {
    pub(super) fn new() -> Self {
        Self {
            pending_files: VecDeque::new(),
            next_scan_at: Instant::now(),
            scan_interval: Duration::from_secs(20),
        }
    }

    pub(super) fn request_scan_soon(&mut self) {
        self.next_scan_at = Instant::now();
    }

    fn should_scan(&self) -> bool {
        self.pending_files.is_empty() || Instant::now() >= self.next_scan_at
    }

    fn schedule_next_scan(&mut self) {
        self.next_scan_at = Instant::now() + self.scan_interval;
    }

    fn replace_queue(&mut self, files: Vec<PathBuf>) {
        self.pending_files = files.into();
    }
}

pub(super) const IDLE_INDEX_POLL_INTERVAL: Duration = Duration::from_millis(350);

fn refresh_incremental_index(
    state: &mut IncrementalIndexState,
    project_index: &ProjectIndex,
    project_root: &Path,
) {
    match project_index.collect_delta(project_root) {
        Ok(delta) => {
            let queued = delta.to_index.len();
            state.replace_queue(delta.to_index);
            state.schedule_next_scan();
            if queued > 0 || delta.removed > 0 {
                info!(
                    queued,
                    removed = delta.removed,
                    unchanged = delta.unchanged,
                    skipped_large = delta.skipped_large,
                    "incremental project index scan updated"
                );
            } else {
                debug!(
                    unchanged = delta.unchanged,
                    skipped_large = delta.skipped_large,
                    "incremental project index scan clean"
                );
            }
        }
        Err(e) => {
            warn!(error = %e, "incremental project index scan failed");
            state.schedule_next_scan();
        }
    }
}

pub(super) fn run_idle_index_step(
    index_state: &mut IncrementalIndexState,
    project_index: &ProjectIndex,
    project_root: &Path,
    backend: &dyn InferenceBackend,
) {
    if index_state.should_scan() {
        refresh_incremental_index(index_state, project_index, project_root);
    }

    let Some(path) = index_state.pending_files.pop_front() else {
        return;
    };

    let content = match std::fs::read_to_string(&path) {
        Ok(content) => content,
        Err(e) => {
            warn!(path = %path.display(), error = %e, "idle index read failed");
            return;
        }
    };

    if let Err(e) = project_index.index_file(&path, &content, backend) {
        warn!(path = %path.display(), error = %e, "idle index update failed");
    } else {
        debug!(
            path = %path.display(),
            remaining = index_state.pending_files.len(),
            "idle index updated file"
        );
    }
}
