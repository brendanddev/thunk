use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub struct SessionLog {
    writer: BufWriter<File>,
    started: Instant,
}

impl SessionLog {
    /// Opens a new log file in `logs_dir` named with a human-readable UTC timestamp.
    /// Returns None if the file cannot be created — logging is advisory.
    pub fn open(logs_dir: &Path) -> Option<Self> {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let ts = unix_secs_to_timestamp(secs);
        let path = logs_dir.join(format!("{ts}.log"));
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .ok()?;
        let mut log = Self {
            writer: BufWriter::new(file),
            started: Instant::now(),
        };
        log.log("session started");
        Some(log)
    }

    pub fn log(&mut self, msg: &str) {
        let elapsed = self.started.elapsed().as_millis();
        let _ = writeln!(self.writer, "[+{elapsed}ms] {msg}");
        let _ = self.writer.flush();
    }

    pub fn log_timed(&mut self, stage: &str, duration: Duration) {
        let elapsed = self.started.elapsed().as_millis();
        let ms = duration.as_millis();
        let _ = writeln!(self.writer, "[+{elapsed}ms] {stage} ({ms}ms)");
        let _ = self.writer.flush();
    }
}

/// Converts unix seconds to a filesystem-safe UTC timestamp string.
/// Format: `2026-04-16_19-42-10`
fn unix_secs_to_timestamp(secs: u64) -> String {
    let s = (secs % 60) as u32;
    let m = ((secs / 60) % 60) as u32;
    let h = ((secs / 3600) % 24) as u32;
    let days = (secs / 86400) as u32;
    let (y, mo, d) = days_since_epoch_to_ymd(days);
    format!("{y:04}-{mo:02}-{d:02}_{h:02}-{m:02}-{s:02}")
}

/// Hinnant civil-from-days algorithm (public domain).
/// Maps days since Unix epoch (1970-01-01) to (year, month, day).
fn days_since_epoch_to_ymd(days: u32) -> (u32, u32, u32) {
    let z = days + 719468;
    let era = z / 146097;
    let doe = z % 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    (y, mo, d)
}
