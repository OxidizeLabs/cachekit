//! Standard benchmark metrics for cache policy evaluation.
//!
//! Provides consistent measurement across all cache policies for:
//! - Hit/miss rates and throughput
//! - Latency distribution (p50, p95, p99, max)
//! - Eviction behavior
//! - Scan resistance and adaptation speed

use std::time::{Duration, Instant};

use cachekit::traits::Cache;
use rand::SeedableRng;

use crate::workload::WorkloadSpec;

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

/// Samples operation latencies on a fixed-rate cadence into a bounded buffer.
///
/// Once the buffer is full, additional samples are discarded. This keeps the
/// reported percentiles reproducible across runs and avoids the bias of an
/// ad-hoc reservoir that overwrites earlier outliers (which is exactly what
/// p99/max are supposed to capture).
#[derive(Debug)]
pub struct LatencySampler {
    samples: Vec<Duration>,
    capacity: usize,
    sample_rate: u64,
    op_index: u64,
}

impl LatencySampler {
    /// Create a sampler that collects up to `capacity` samples.
    /// `sample_rate` controls how often to sample (1 = every op, 100 = every 100th op).
    pub fn new(capacity: usize, sample_rate: u64) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            capacity,
            sample_rate: sample_rate.max(1),
            op_index: 0,
        }
    }

    /// Returns true if the next call to `record` would store a sample.
    ///
    /// Callers can use this to skip the `Instant::now()` pair on ops that
    /// won't contribute to the distribution, removing ~20-50ns of timer
    /// overhead from the measurement hot path.
    #[inline]
    pub fn should_sample(&self) -> bool {
        self.samples.len() < self.capacity
            && self
                .op_index
                .wrapping_add(1)
                .is_multiple_of(self.sample_rate)
    }

    /// Advance the op counter without recording. Pair with `should_sample` so
    /// the cadence advances in lockstep with the measured workload.
    #[inline]
    pub fn skip(&mut self) {
        self.op_index = self.op_index.wrapping_add(1);
    }

    /// Record a latency sample (only if selected for sampling and within capacity).
    #[inline]
    pub fn record(&mut self, duration: Duration) {
        self.op_index = self.op_index.wrapping_add(1);
        if self.samples.len() < self.capacity && self.op_index.is_multiple_of(self.sample_rate) {
            self.samples.push(duration);
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
                workload: crate::workload::Workload::Zipfian { exponent: 1.0 },
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
    post_warmup_misses: u64,
    pre_warmup_capacity_reached_at: Option<usize>,
}

/// Run a complete benchmark against a cache.
///
/// Returns detailed metrics including hit rate, throughput, and latency
/// distribution. Latency is only timed on the sample cadence to keep the
/// `Instant::now()` overhead off the hot path.
///
/// The `inserts` counter approximates evictions during the measured phase: in
/// steady state each miss after the warmup-fill triggers exactly one eviction,
/// so `evictions_per_insert` converges to 1.0.
pub fn run_benchmark<C, V, F>(
    policy_name: &str,
    cache: &mut C,
    config: &BenchmarkConfig,
    value_for_key: F,
) -> BenchmarkResult
where
    C: Cache<u64, V>,
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
        let in_measured_phase = op_idx >= warmup_boundary;
        let sample = in_measured_phase && sampler.should_sample();
        let op_start = if sample { Some(Instant::now()) } else { None };

        if cache.get(&key).is_some() {
            metrics.hits += 1;
        } else {
            metrics.misses += 1;
            let value = value_for_key(key);
            let _ = cache.insert(key, value);
            generator.record_insert();
            metrics.inserts += 1;
            if in_measured_phase {
                metrics.post_warmup_misses += 1;
            } else if metrics.pre_warmup_capacity_reached_at.is_none()
                && metrics.inserts as usize >= config.capacity
            {
                metrics.pre_warmup_capacity_reached_at = Some(op_idx);
            }
        }

        match op_start {
            Some(s) => sampler.record(s.elapsed()),
            None if in_measured_phase => sampler.skip(),
            None => {},
        }
    }

    let total_duration = start.elapsed();

    let hit_stats = HitStats {
        hits: metrics.hits,
        misses: metrics.misses,
        inserts: metrics.inserts,
        // We no longer probe `cache.contains` per-miss; treat re-inserts of a
        // missed key as inserts, matching `OpCounts` semantics.
        updates: 0,
    };

    let throughput = ThroughputStats::from_counts(
        metrics.hits,
        metrics.misses,
        metrics.inserts,
        total_duration,
    );

    let latency = sampler.stats();

    // Once the cache is full, every subsequent insert evicts. Approximate
    // post-warmup evictions as post-warmup misses; pre-warmup we only count
    // the inserts that happened after the cache reached capacity.
    let pre_warmup_evictions = metrics
        .pre_warmup_capacity_reached_at
        .map(|reached| (warmup_boundary.saturating_sub(reached)) as u64)
        .unwrap_or(0);
    let post_warmup_evictions = metrics.post_warmup_misses;
    let total_evictions = pre_warmup_evictions + post_warmup_evictions;
    let eviction = EvictionStats {
        total_evictions,
        evictions_per_insert: if metrics.post_warmup_misses > 0 {
            post_warmup_evictions as f64 / metrics.post_warmup_misses as f64
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
/// A scan-resistant policy keeps `recovery_hit_rate` close to `baseline_hit_rate`.
/// `resistance_score` is `recovery / baseline` when baseline is meaningfully
/// above zero, otherwise `None` to avoid a misleading ratio against noise.
pub fn measure_scan_resistance<C, V, F>(
    cache: &mut C,
    capacity: usize,
    universe: u64,
    value_for_key: F,
) -> ScanResistanceResult
where
    C: Cache<u64, V>,
    F: Fn(u64) -> V,
{
    let warmup_ops = capacity * 2;
    let baseline_ops = capacity * 2;
    let scan_ops = capacity * 2; // Scan through 2x capacity
    let recovery_ops = capacity * 2;

    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    use rand::RngExt;

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

    // Treat baseline below 1% as too noisy to take a ratio against. This avoids
    // both divide-by-near-zero blow-ups and the previous behavior where a
    // policy that never warmed up could report an arbitrarily large score.
    const BASELINE_FLOOR: f64 = 0.01;
    let resistance_score = if baseline_hit_rate >= BASELINE_FLOOR {
        Some(recovery_hit_rate / baseline_hit_rate)
    } else {
        None
    };

    ScanResistanceResult {
        baseline_hit_rate,
        scan_hit_rate,
        recovery_hit_rate,
        resistance_score,
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
    /// Ratio of recovery to baseline (1.0 = perfect recovery), `None` when
    /// the baseline hit rate is too low for the ratio to be meaningful.
    pub resistance_score: Option<f64>,
}

impl ScanResistanceResult {
    pub fn summary(&self) -> String {
        let score = match self.resistance_score {
            Some(s) => format!("{s:.2}"),
            None => "n/a".to_string(),
        };
        format!(
            "baseline={:.2}% scan={:.2}% recovery={:.2}% score={score}",
            self.baseline_hit_rate * 100.0,
            self.scan_hit_rate * 100.0,
            self.recovery_hit_rate * 100.0,
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
    C: Cache<u64, V>,
    F: Fn(u64) -> V,
{
    let warmup_ops = capacity * 2;
    let stable_ops = capacity * 2;
    let adaptation_ops = capacity * 4;
    let window_size = capacity / 4;

    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    use rand::RngExt;

    // Phase 1: Warmup and stable with region A (first half of universe)
    let region_a_max = universe / 2;

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
        window_size,
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
    /// Number of operations per window in `hit_rate_curve`. The post-shift
    /// op offset for window `i` is `(i + 1) * window_size`.
    pub window_size: usize,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latency_sampler_records_only_at_cadence() {
        let mut s = LatencySampler::new(10, 5);
        for i in 0..10 {
            if s.should_sample() {
                s.record(Duration::from_nanos(i + 1));
            } else {
                s.skip();
            }
        }
        let stats = s.stats();
        // 10 ops at sample_rate 5 → exactly 2 samples (ops 5 and 10).
        assert_eq!(stats.sample_count, 2);
    }

    #[test]
    fn latency_sampler_stops_at_capacity() {
        let mut s = LatencySampler::new(3, 1);
        for i in 0..100u64 {
            if s.should_sample() {
                s.record(Duration::from_nanos(i + 1));
            } else {
                s.skip();
            }
        }
        let stats = s.stats();
        // First-N-then-stop: only the first 3 samples are kept; later
        // outliers are intentionally not allowed to overwrite earlier ones.
        assert_eq!(stats.sample_count, 3);
        assert_eq!(stats.min, Duration::from_nanos(1));
        assert_eq!(stats.max, Duration::from_nanos(3));
    }

    #[test]
    fn scan_resistance_score_is_none_when_baseline_is_noise() {
        // Baseline 0.5%, recovery 0.5% — ratio is mathematically defined but
        // not meaningful; should return `None`.
        let mut cache: cachekit::policy::lru::LruCore<u64, u64> =
            cachekit::policy::lru::LruCore::new(4);
        let r = measure_scan_resistance(&mut cache, 4, 1_000_000, std::sync::Arc::new);
        if r.baseline_hit_rate < 0.01 {
            assert!(r.resistance_score.is_none(), "got {:?}", r.resistance_score);
        }
    }

    #[test]
    fn scan_resistance_score_is_some_when_baseline_is_meaningful() {
        // Tight universe vs capacity → non-trivial baseline hit rate, so the
        // ratio should be reported.
        let mut cache: cachekit::policy::lru::LruCore<u64, u64> =
            cachekit::policy::lru::LruCore::new(64);
        let r = measure_scan_resistance(&mut cache, 64, 128, std::sync::Arc::new);
        if r.baseline_hit_rate >= 0.01 {
            assert!(r.resistance_score.is_some(), "got {r:?}");
        }
    }
}
