//! Test fixtures shared between `templates`, `markdown`, and `io`
//! modules.
//!
//! Centralising these here means a future schema field added to
//! `RunMetadata` or `Metrics` is updated in one place, not three
//! drifted copies. Everything is `pub(crate)` and gated behind
//! `#[cfg(test)]` (the module is only declared under that flag from
//! the binary entry).

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use bench_support::json_results::{BenchmarkConfig, HitStats, Metrics, ResultRow, RunMetadata};

pub(crate) fn metadata() -> RunMetadata {
    RunMetadata {
        timestamp: "2026-01-01T00:00:00Z".into(),
        git_commit: None,
        git_branch: None,
        git_dirty: false,
        rustc_version: "rustc test".into(),
        host_triple: "x86_64-unknown-linux-gnu".into(),
        cpu_model: None,
        config: BenchmarkConfig {
            capacity: 1024,
            universe: 10_000,
            operations: 100_000,
            seed: 42,
        },
    }
}

pub(crate) fn empty_metrics() -> Metrics {
    Metrics {
        hit_stats: None,
        throughput: None,
        latency: None,
        eviction: None,
        scan_resistance: None,
        adaptation: None,
    }
}

pub(crate) fn row(policy: &str, workload: &str, case: &str, metrics: Metrics) -> ResultRow {
    ResultRow {
        policy_id: policy.to_lowercase(),
        policy_name: policy.into(),
        workload_id: workload.to_lowercase(),
        workload_name: workload.into(),
        case_id: case.into(),
        metrics,
    }
}

pub(crate) fn hit_metrics(hit_rate: f64) -> Metrics {
    Metrics {
        hit_stats: Some(HitStats {
            hits: 0,
            misses: 0,
            inserts: 0,
            updates: 0,
            hit_rate,
            miss_rate: 1.0 - hit_rate,
        }),
        ..empty_metrics()
    }
}

/// Per-test temp directory that does not collide with concurrent
/// `cargo test` invocations or `nextest` runners. The pid scopes the
/// directory to this process; the atomic counter scopes it within
/// the process so two tests with the same `label` still don't race.
/// Caller is responsible for cleanup; on test panic the directory
/// is left behind for inspection (intentional — `tempfile`'s
/// auto-cleanup hides reproducer state).
pub(crate) fn unique_temp_dir(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let pid = std::process::id();
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("cachekit-render-docs-{label}-{pid}-{counter}"));
    if dir.exists() {
        panic!(
            "unique_temp_dir collision at {} (pid={pid}, counter={counter}); \
             environment is broken or temp dir was mutated mid-test",
            dir.display(),
        );
    }
    std::fs::create_dir_all(&dir).expect("create unique temp dir");
    dir
}
