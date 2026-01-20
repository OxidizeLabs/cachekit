//! JSON results format for benchmark artifacts.
//!
//! This module defines the stable JSON schema for benchmark results,
//! separating measurement from presentation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Version of the benchmark results schema.
pub const SCHEMA_VERSION: &str = "1.0.0";

/// Complete benchmark run artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkArtifact {
    /// Schema version for compatibility checking.
    pub schema_version: String,
    /// Metadata about the benchmark run.
    pub metadata: RunMetadata,
    /// Benchmark results organized by policy, workload, and case.
    pub results: Vec<ResultRow>,
}

/// Metadata about the benchmark run environment and configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetadata {
    /// Timestamp when the benchmark was run (ISO 8601).
    pub timestamp: String,
    /// Git commit SHA (if available).
    pub git_commit: Option<String>,
    /// Git branch name (if available).
    pub git_branch: Option<String>,
    /// Whether the working directory had uncommitted changes.
    pub git_dirty: bool,
    /// Rust compiler version.
    pub rustc_version: String,
    /// Host triple (e.g., "x86_64-apple-darwin").
    pub host_triple: String,
    /// CPU model/name.
    pub cpu_model: Option<String>,
    /// Benchmark configuration parameters.
    pub config: BenchmarkConfig,
}

/// Benchmark configuration parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Cache capacity used for benchmarks.
    pub capacity: usize,
    /// Key universe size.
    pub universe: u64,
    /// Number of operations per benchmark.
    pub operations: usize,
    /// Random seed used for reproducibility.
    pub seed: u64,
}

/// A single result row in the benchmark matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultRow {
    /// Policy identifier (e.g., "lru", "s3_fifo").
    pub policy_id: String,
    /// Policy display name (e.g., "LRU", "S3-FIFO").
    pub policy_name: String,
    /// Workload identifier (e.g., "zipfian_1.0").
    pub workload_id: String,
    /// Workload display name (e.g., "Zipfian 1.0").
    pub workload_name: String,
    /// Benchmark case type (e.g., "hit_rate", "comprehensive", "scan_resistance").
    pub case_id: String,
    /// Detailed metrics for this result.
    pub metrics: Metrics,
}

/// Metrics collected during a benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    /// Hit/miss statistics.
    pub hit_stats: Option<HitStats>,
    /// Throughput measurements.
    pub throughput: Option<ThroughputStats>,
    /// Latency distribution.
    pub latency: Option<LatencyStats>,
    /// Eviction statistics.
    pub eviction: Option<EvictionStats>,
    /// Scan resistance results.
    pub scan_resistance: Option<ScanResistanceStats>,
    /// Adaptation speed results.
    pub adaptation: Option<AdaptationStats>,
}

/// Hit/miss statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HitStats {
    pub hits: u64,
    pub misses: u64,
    pub inserts: u64,
    pub updates: u64,
    pub hit_rate: f64,
    pub miss_rate: f64,
}

/// Throughput measurements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputStats {
    /// Total duration in milliseconds.
    pub duration_ms: f64,
    /// Operations per second.
    pub ops_per_sec: f64,
    /// Gets per second.
    pub gets_per_sec: f64,
    /// Inserts per second.
    pub inserts_per_sec: f64,
}

/// Latency distribution statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    /// Number of samples collected.
    pub sample_count: usize,
    /// Minimum latency in nanoseconds.
    pub min_ns: u64,
    /// Median (p50) latency in nanoseconds.
    pub p50_ns: u64,
    /// 95th percentile latency in nanoseconds.
    pub p95_ns: u64,
    /// 99th percentile latency in nanoseconds.
    pub p99_ns: u64,
    /// Maximum latency in nanoseconds.
    pub max_ns: u64,
    /// Mean latency in nanoseconds.
    pub mean_ns: u64,
}

/// Eviction behavior statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvictionStats {
    /// Total evictions during the benchmark.
    pub total_evictions: u64,
    /// Evictions per insert (after warmup).
    pub evictions_per_insert: f64,
}

/// Scan resistance measurement results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanResistanceStats {
    /// Hit rate before the scan phase.
    pub baseline_hit_rate: f64,
    /// Hit rate during the scan phase.
    pub scan_hit_rate: f64,
    /// Hit rate after recovery from scan.
    pub recovery_hit_rate: f64,
    /// Resistance score (recovery/baseline, 1.0 = perfect).
    pub resistance_score: f64,
}

/// Adaptation speed measurement results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationStats {
    /// Final stable hit rate after workload shift.
    pub stable_hit_rate: f64,
    /// Operations needed to reach 50% of stable hit rate.
    pub ops_to_50_percent: usize,
    /// Operations needed to reach 80% of stable hit rate.
    pub ops_to_80_percent: usize,
}

impl BenchmarkArtifact {
    /// Create a new benchmark artifact with metadata.
    pub fn new(metadata: RunMetadata) -> Self {
        Self {
            schema_version: SCHEMA_VERSION.to_string(),
            metadata,
            results: Vec::new(),
        }
    }

    /// Add a result row to the artifact.
    pub fn add_result(&mut self, result: ResultRow) {
        self.results.push(result);
    }

    /// Get results grouped by case type.
    pub fn results_by_case(&self) -> HashMap<String, Vec<&ResultRow>> {
        let mut grouped: HashMap<String, Vec<&ResultRow>> = HashMap::new();
        for result in &self.results {
            grouped
                .entry(result.case_id.clone())
                .or_default()
                .push(result);
        }
        grouped
    }

    /// Get results for a specific policy.
    pub fn results_for_policy(&self, policy_id: &str) -> Vec<&ResultRow> {
        self.results
            .iter()
            .filter(|r| r.policy_id == policy_id)
            .collect()
    }

    /// Get results for a specific workload.
    pub fn results_for_workload(&self, workload_id: &str) -> Vec<&ResultRow> {
        self.results
            .iter()
            .filter(|r| r.workload_id == workload_id)
            .collect()
    }
}

/// Helper to convert Duration to nanoseconds as u64.
pub fn duration_to_nanos(d: std::time::Duration) -> u64 {
    d.as_nanos() as u64
}
