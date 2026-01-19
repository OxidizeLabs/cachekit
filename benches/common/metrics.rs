//! Standard benchmark metrics for cache policy evaluation.
//!
//! Provides consistent measurement across all cache policies for:
//! - Hit/miss rates and throughput
//! - Latency distribution (p50, p95, p99, max)
//! - Memory efficiency
//! - Eviction behavior
//! - Adaptation speed

use std::time::{Duration, Instant};

use cachekit::traits::CoreCache;
use rand::SeedableRng;

use crate::common::workload::WorkloadSpec;

// ============================================================================
// Core Metrics Structures
// ============================================================================

/// Complete benchmark results for a cache policy.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Name of the policy being tested.
    pub policy_name: String,
    /// Name of the workload used.
    pub workload_name: String,
    /// Cache capacity.
    pub capacity: usize,
    /// Key universe size.
    pub universe: u64,
    /// Total operations performed.
    pub operations: u64,
    /// Hit/miss statistics.
    pub hit_stats: HitStats,
    /// Throughput measurements.
    pub throughput: ThroughputStats,
    /// Latency distribution.
    pub latency: LatencyStats,
    /// Eviction statistics.
    pub eviction: EvictionStats,
}

impl BenchmarkResult {
    /// Format as a single-line summary.
    pub fn summary(&self) -> String {
        format!(
            "{}/{}: hit={:.2}% throughput={:.2}Mops/s p99={:.1}ns evictions={}",
            self.policy_name,
            self.workload_name,
            self.hit_stats.hit_rate() * 100.0,
            self.throughput.ops_per_sec / 1_000_000.0,
            self.latency.p99.as_nanos(),
            self.eviction.total_evictions,
        )
    }
}

/// Hit/miss statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct HitStats {
    pub hits: u64,
    pub misses: u64,
    pub inserts: u64,
    pub updates: u64,
}

impl HitStats {
    #[inline]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    #[inline]
    pub fn miss_rate(&self) -> f64 {
        1.0 - self.hit_rate()
    }

    pub fn total_ops(&self) -> u64 {
        self.hits + self.misses
    }
}

/// Throughput measurements.
#[derive(Debug, Clone, Copy, Default)]
pub struct ThroughputStats {
    /// Total wall-clock duration.
    pub total_duration: Duration,
    /// Operations per second.
    pub ops_per_sec: f64,
    /// Gets per second (hits + misses).
    pub gets_per_sec: f64,
    /// Inserts per second.
    pub inserts_per_sec: f64,
}

impl ThroughputStats {
    pub fn from_counts(hits: u64, misses: u64, inserts: u64, duration: Duration) -> Self {
        let secs = duration.as_secs_f64();
        if secs == 0.0 {
            return Self::default();
        }
        let total_ops = hits + misses + inserts;
        Self {
            total_duration: duration,
            ops_per_sec: total_ops as f64 / secs,
            gets_per_sec: (hits + misses) as f64 / secs,
            inserts_per_sec: inserts as f64 / secs,
        }
    }
}

/// Latency distribution (collected via sampling).
#[derive(Debug, Clone, Copy, Default)]
pub struct LatencyStats {
    pub min: Duration,
    pub p50: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub max: Duration,
    pub mean: Duration,
    pub sample_count: usize,
}

impl LatencyStats {
    /// Compute percentiles from a sorted slice of durations.
    pub fn from_samples(samples: &mut [Duration]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }

        samples.sort_unstable();
        let n = samples.len();
        let sum: Duration = samples.iter().sum();

        Self {
            min: samples[0],
            p50: samples[n / 2],
            p95: samples[(n * 95) / 100],
            p99: samples[(n * 99) / 100],
            max: samples[n - 1],
            mean: sum / n as u32,
            sample_count: n,
        }
    }
}

/// Eviction behavior metrics.
#[derive(Debug, Clone, Copy, Default)]
pub struct EvictionStats {
    /// Total evictions during the benchmark.
    pub total_evictions: u64,
    /// Evictions per insert (after warmup).
    pub evictions_per_insert: f64,
}

// ============================================================================
// Latency Sampler
// ============================================================================

/// Samples operation latencies without measuring every operation.
///
/// Uses reservoir sampling to collect a fixed number of latency samples
/// with minimal overhead.
#[derive(Debug)]
pub struct LatencySampler {
    samples: Vec<Duration>,
    capacity: usize,
    count: u64,
    sample_rate: u64,
}

impl LatencySampler {
    /// Create a sampler that collects up to `capacity` samples.
    /// `sample_rate` controls how often to sample (1 = every op, 100 = every 100th op).
    pub fn new(capacity: usize, sample_rate: u64) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            capacity,
            count: 0,
            sample_rate: sample_rate.max(1),
        }
    }

    /// Record a latency sample (only if selected for sampling).
    #[inline]
    pub fn record(&mut self, duration: Duration) {
        self.count += 1;
        if self.count % self.sample_rate != 0 {
            return;
        }

        if self.samples.len() < self.capacity {
            self.samples.push(duration);
        } else {
            // Reservoir sampling for uniform distribution
            let idx = (self.count / self.sample_rate) as usize;
            if idx < self.capacity {
                self.samples[idx] = duration;
            } else {
                // Simple modulo replacement for speed
                let replace_idx = (self.count as usize) % self.capacity;
                self.samples[replace_idx] = duration;
            }
        }
    }

    /// Compute latency statistics from collected samples.
    pub fn stats(&mut self) -> LatencyStats {
        LatencyStats::from_samples(&mut self.samples)
    }
}

// ============================================================================
// Benchmark Runner
// ============================================================================

/// Configuration for running a benchmark.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Name for this benchmark run.
    pub name: String,
    /// Cache capacity.
    pub capacity: usize,
    /// Number of operations to run.
    pub operations: usize,
    /// Warmup operations before measurement.
    pub warmup_ops: usize,
    /// Workload specification.
    pub workload: WorkloadSpec,
    /// Sample rate for latency collection (1 = all, 100 = 1%).
    pub latency_sample_rate: u64,
    /// Maximum latency samples to collect.
    pub max_latency_samples: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            capacity: 4096,
            operations: 100_000,
            warmup_ops: 10_000,
            workload: WorkloadSpec {
                universe: 16_384,
                workload: crate::common::workload::Workload::Zipfian { exponent: 1.0 },
                seed: 42,
            },
            latency_sample_rate: 100,
            max_latency_samples: 10_000,
        }
    }
}

/// Collected metrics during a benchmark run.
#[derive(Debug, Default)]
struct RunMetrics {
    hits: u64,
    misses: u64,
    inserts: u64,
    updates: u64,
    evictions: u64,
    post_warmup_inserts: u64,
    post_warmup_evictions: u64,
}

/// Run a complete benchmark against a cache.
///
/// Returns detailed metrics including hit rate, throughput, and latency distribution.
pub fn run_benchmark<C, V, F>(
    policy_name: &str,
    cache: &mut C,
    config: &BenchmarkConfig,
    value_for_key: F,
) -> BenchmarkResult
where
    C: CoreCache<u64, V>,
    F: Fn(u64) -> V,
{
    let mut generator = config.workload.generator();
    let mut metrics = RunMetrics::default();
    let mut sampler = LatencySampler::new(config.max_latency_samples, config.latency_sample_rate);
    let total_ops = config.warmup_ops + config.operations;
    let warmup_boundary = config.warmup_ops;

    let start = Instant::now();

    for op_idx in 0..total_ops {
        let key = generator.next_key();
        let op_start = Instant::now();

        let was_full = cache.len() >= config.capacity;

        if let Some(_value) = cache.get(&key) {
            metrics.hits += 1;
        } else {
            metrics.misses += 1;

            // Check if this is an update or insert
            let existed = cache.contains(&key);
            let value = value_for_key(key);
            let _ = cache.insert(key, value);
            generator.record_insert();

            if existed {
                metrics.updates += 1;
            } else {
                metrics.inserts += 1;
                if was_full {
                    metrics.evictions += 1;
                    if op_idx >= warmup_boundary {
                        metrics.post_warmup_evictions += 1;
                    }
                }
            }

            if op_idx >= warmup_boundary {
                metrics.post_warmup_inserts += 1;
            }
        }

        // Only sample latency during measurement phase
        if op_idx >= warmup_boundary {
            sampler.record(op_start.elapsed());
        }
    }

    let total_duration = start.elapsed();

    // Compute derived metrics
    let hit_stats = HitStats {
        hits: metrics.hits,
        misses: metrics.misses,
        inserts: metrics.inserts,
        updates: metrics.updates,
    };

    let throughput = ThroughputStats::from_counts(
        metrics.hits,
        metrics.misses,
        metrics.inserts,
        total_duration,
    );

    let latency = sampler.stats();

    let eviction = EvictionStats {
        total_evictions: metrics.evictions,
        evictions_per_insert: if metrics.post_warmup_inserts > 0 {
            metrics.post_warmup_evictions as f64 / metrics.post_warmup_inserts as f64
        } else {
            0.0
        },
    };

    BenchmarkResult {
        policy_name: policy_name.to_string(),
        workload_name: config.name.clone(),
        capacity: config.capacity,
        universe: config.workload.universe,
        operations: config.operations as u64,
        hit_stats,
        throughput,
        latency,
        eviction,
    }
}

// ============================================================================
// Specialized Benchmarks
// ============================================================================

/// Measure scan resistance by interleaving point lookups with sequential scans.
///
/// Returns (baseline_hit_rate, scan_hit_rate, recovery_hit_rate).
/// A scan-resistant policy should have recovery_hit_rate close to baseline_hit_rate.
pub fn measure_scan_resistance<C, V, F>(
    cache: &mut C,
    capacity: usize,
    universe: u64,
    value_for_key: F,
) -> ScanResistanceResult
where
    C: CoreCache<u64, V>,
    F: Fn(u64) -> V,
{
    let warmup_ops = capacity * 2;
    let baseline_ops = capacity * 2;
    let scan_ops = capacity * 2; // Scan through 2x capacity
    let recovery_ops = capacity * 2;

    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    use rand::Rng;

    // Phase 1: Warmup with Zipfian
    let zipf = rand_distr::Zipf::new(universe as f64, 1.0).unwrap();
    use rand_distr::Distribution;
    for _ in 0..warmup_ops {
        let sample: f64 = zipf.sample(&mut rng);
        let key = (sample as u64).saturating_sub(1).min(universe - 1);
        if cache.get(&key).is_none() {
            let _ = cache.insert(key, value_for_key(key));
        }
    }

    // Phase 2: Baseline measurement (Zipfian)
    let mut baseline_hits = 0u64;
    let mut baseline_total = 0u64;
    for _ in 0..baseline_ops {
        let sample: f64 = zipf.sample(&mut rng);
        let key = (sample as u64).saturating_sub(1).min(universe - 1);
        baseline_total += 1;
        if cache.get(&key).is_some() {
            baseline_hits += 1;
        } else {
            let _ = cache.insert(key, value_for_key(key));
        }
    }
    let baseline_hit_rate = baseline_hits as f64 / baseline_total as f64;

    // Phase 3: Sequential scan (should pollute non-resistant caches)
    let mut scan_hits = 0u64;
    let mut scan_total = 0u64;
    let scan_start = rng.random::<u64>() % universe;
    for i in 0..scan_ops {
        let key = (scan_start + i as u64) % universe;
        scan_total += 1;
        if cache.get(&key).is_some() {
            scan_hits += 1;
        } else {
            let _ = cache.insert(key, value_for_key(key));
        }
    }
    let scan_hit_rate = scan_hits as f64 / scan_total as f64;

    // Phase 4: Recovery measurement (back to Zipfian)
    let mut recovery_hits = 0u64;
    let mut recovery_total = 0u64;
    for _ in 0..recovery_ops {
        let sample: f64 = zipf.sample(&mut rng);
        let key = (sample as u64).saturating_sub(1).min(universe - 1);
        recovery_total += 1;
        if cache.get(&key).is_some() {
            recovery_hits += 1;
        } else {
            let _ = cache.insert(key, value_for_key(key));
        }
    }
    let recovery_hit_rate = recovery_hits as f64 / recovery_total as f64;

    ScanResistanceResult {
        baseline_hit_rate,
        scan_hit_rate,
        recovery_hit_rate,
        resistance_score: recovery_hit_rate / baseline_hit_rate.max(0.001),
    }
}

/// Results from scan resistance measurement.
#[derive(Debug, Clone, Copy)]
pub struct ScanResistanceResult {
    /// Hit rate before the scan.
    pub baseline_hit_rate: f64,
    /// Hit rate during the scan.
    pub scan_hit_rate: f64,
    /// Hit rate after recovery.
    pub recovery_hit_rate: f64,
    /// Ratio of recovery to baseline (1.0 = perfect recovery).
    pub resistance_score: f64,
}

impl ScanResistanceResult {
    pub fn summary(&self) -> String {
        format!(
            "baseline={:.2}% scan={:.2}% recovery={:.2}% score={:.2}",
            self.baseline_hit_rate * 100.0,
            self.scan_hit_rate * 100.0,
            self.recovery_hit_rate * 100.0,
            self.resistance_score,
        )
    }
}

/// Measure adaptation speed when workload shifts.
///
/// Returns metrics on how quickly the cache adapts to a new access pattern.
pub fn measure_adaptation_speed<C, V, F>(
    cache: &mut C,
    capacity: usize,
    universe: u64,
    value_for_key: F,
) -> AdaptationResult
where
    C: CoreCache<u64, V>,
    F: Fn(u64) -> V,
{
    let warmup_ops = capacity * 2;
    let stable_ops = capacity * 2;
    let adaptation_ops = capacity * 4;
    let window_size = capacity / 4;

    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    // Phase 1: Warmup and stable with region A (first half of universe)
    let region_a_max = universe / 2;
    use rand::Rng;

    for _ in 0..(warmup_ops + stable_ops) {
        let key = rng.random::<u64>() % region_a_max;
        if cache.get(&key).is_none() {
            let _ = cache.insert(key, value_for_key(key));
        }
    }

    // Phase 2: Shift to region B (second half) and measure adaptation
    let region_b_min = universe / 2;
    let mut windows: Vec<f64> = Vec::new();
    let mut window_hits = 0u64;
    let mut window_total = 0u64;

    for i in 0..adaptation_ops {
        let key = region_b_min + (rng.random::<u64>() % region_a_max);
        window_total += 1;
        if cache.get(&key).is_some() {
            window_hits += 1;
        } else {
            let _ = cache.insert(key, value_for_key(key));
        }

        if window_total >= window_size as u64 {
            windows.push(window_hits as f64 / window_total as f64);
            window_hits = 0;
            window_total = 0;
        }

        // Stop if we've reached stable state (> 80% hit rate)
        if i > capacity && windows.last().is_some_and(|&r| r > 0.8) {
            break;
        }
    }

    // Find ops to reach 50% and 80% of stable hit rate
    let stable_rate = windows.last().copied().unwrap_or(0.0);
    let threshold_50 = stable_rate * 0.5;
    let threshold_80 = stable_rate * 0.8;

    let ops_to_50 = windows
        .iter()
        .position(|&r| r >= threshold_50)
        .map(|i| (i + 1) * window_size)
        .unwrap_or(adaptation_ops);

    let ops_to_80 = windows
        .iter()
        .position(|&r| r >= threshold_80)
        .map(|i| (i + 1) * window_size)
        .unwrap_or(adaptation_ops);

    AdaptationResult {
        stable_hit_rate: stable_rate,
        ops_to_50_percent: ops_to_50,
        ops_to_80_percent: ops_to_80,
        hit_rate_curve: windows,
    }
}

/// Results from adaptation speed measurement.
#[derive(Debug, Clone)]
pub struct AdaptationResult {
    /// Final stable hit rate after adaptation.
    pub stable_hit_rate: f64,
    /// Operations needed to reach 50% of stable hit rate.
    pub ops_to_50_percent: usize,
    /// Operations needed to reach 80% of stable hit rate.
    pub ops_to_80_percent: usize,
    /// Hit rate at each measurement window.
    pub hit_rate_curve: Vec<f64>,
}

impl AdaptationResult {
    pub fn summary(&self) -> String {
        format!(
            "stable={:.2}% ops_to_50%={} ops_to_80%={}",
            self.stable_hit_rate * 100.0,
            self.ops_to_50_percent,
            self.ops_to_80_percent,
        )
    }
}

// ============================================================================
// Comparison Utilities
// ============================================================================

/// Compare hit rates across multiple workloads.
#[derive(Debug, Clone)]
pub struct PolicyComparison {
    pub policy_name: String,
    pub results: Vec<BenchmarkResult>,
}

impl PolicyComparison {
    pub fn new(policy_name: &str) -> Self {
        Self {
            policy_name: policy_name.to_string(),
            results: Vec::new(),
        }
    }

    pub fn add_result(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    /// Print a comparison table.
    pub fn print_table(&self) {
        println!("Policy: {}", self.policy_name);
        println!(
            "{:<20} {:>10} {:>12} {:>10} {:>10}",
            "Workload", "Hit Rate", "Ops/sec", "p99 (ns)", "Evictions"
        );
        println!("{}", "-".repeat(66));
        for r in &self.results {
            println!(
                "{:<20} {:>9.2}% {:>12.0} {:>10} {:>10}",
                r.workload_name,
                r.hit_stats.hit_rate() * 100.0,
                r.throughput.ops_per_sec,
                r.latency.p99.as_nanos(),
                r.eviction.total_evictions,
            );
        }
    }
}

/// Standard workload suite for comparing policies.
pub fn standard_workload_suite(universe: u64, seed: u64) -> Vec<(&'static str, WorkloadSpec)> {
    use crate::common::workload::Workload;

    vec![
        (
            "uniform",
            WorkloadSpec {
                universe,
                workload: Workload::Uniform,
                seed,
            },
        ),
        (
            "zipfian_1.0",
            WorkloadSpec {
                universe,
                workload: Workload::Zipfian { exponent: 1.0 },
                seed,
            },
        ),
        (
            "zipfian_0.8",
            WorkloadSpec {
                universe,
                workload: Workload::Zipfian { exponent: 0.8 },
                seed,
            },
        ),
        (
            "hotset_90_10",
            WorkloadSpec {
                universe,
                workload: Workload::HotSet {
                    hot_fraction: 0.1,
                    hot_prob: 0.9,
                },
                seed,
            },
        ),
        (
            "scan",
            WorkloadSpec {
                universe,
                workload: Workload::Scan,
                seed,
            },
        ),
        (
            "scan_resistance",
            WorkloadSpec {
                universe,
                workload: Workload::ScanResistance {
                    scan_fraction: 0.2,
                    scan_length: 1000,
                    point_exponent: 1.0,
                },
                seed,
            },
        ),
        (
            "loop_small",
            WorkloadSpec {
                universe,
                workload: Workload::Loop {
                    working_set_size: 512,
                },
                seed,
            },
        ),
        (
            "shifting_hotspot",
            WorkloadSpec {
                universe,
                workload: Workload::ShiftingHotspot {
                    shift_interval: 10_000,
                    hot_fraction: 0.1,
                },
                seed,
            },
        ),
        (
            "flash_crowd",
            WorkloadSpec {
                universe,
                workload: Workload::FlashCrowd {
                    base_exponent: 1.0,
                    flash_prob: 0.001,
                    flash_duration: 1000,
                    flash_keys: 10,
                    flash_intensity: 100.0,
                },
                seed,
            },
        ),
    ]
}

// ============================================================================
// Memory Measurement (basic)
// ============================================================================

/// Estimate memory overhead per entry (requires std::mem::size_of on cache).
pub fn estimate_entry_overhead<C>(cache: &C, entries: usize) -> MemoryEstimate
where
    C: Sized,
{
    let cache_size = std::mem::size_of_val(cache);
    MemoryEstimate {
        total_bytes: cache_size,
        bytes_per_entry: if entries > 0 { cache_size / entries } else { 0 },
        entry_count: entries,
    }
}

/// Memory usage estimate.
#[derive(Debug, Clone, Copy)]
pub struct MemoryEstimate {
    pub total_bytes: usize,
    pub bytes_per_entry: usize,
    pub entry_count: usize,
}

impl MemoryEstimate {
    pub fn summary(&self) -> String {
        format!(
            "total={}KB entries={} bytes/entry={}",
            self.total_bytes / 1024,
            self.entry_count,
            self.bytes_per_entry,
        )
    }
}
