//! Policy comparison reports - run with: `cargo bench --bench reports`
//!
//! This is a standalone binary (not a criterion benchmark) that prints
//! human-readable comparison tables for cache policy evaluation.

mod common;

use std::sync::Arc;

use cachekit::policy::clock::ClockCache;
use cachekit::policy::heap_lfu::HeapLfuCache;
use cachekit::policy::lfu::LfuCache;
use cachekit::policy::lru::LruCore;
use cachekit::policy::lru_k::LrukCache;
use cachekit::policy::s3_fifo::S3FifoCache;
use cachekit::policy::two_q::TwoQCore;
use cachekit::traits::CoreCache;
use common::metrics::{
    BenchmarkConfig, PolicyComparison, estimate_entry_overhead, measure_adaptation_speed,
    measure_scan_resistance, run_benchmark, standard_workload_suite,
};
use common::workload::{Workload, WorkloadSpec, run_hit_rate};

const CAPACITY: usize = 4096;
const UNIVERSE: u64 = 16_384;
const OPS: usize = 200_000;
const SEED: u64 = 42;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Find first non-flag argument (skip --bench and similar flags from cargo)
    let report = args.iter().skip(1).find(|arg| !arg.starts_with('-'));

    let Some(report) = report else {
        println!("Usage: cargo bench --bench reports -- <report>");
        println!();
        println!("Available reports:");
        println!("  hit_rate      - Hit rate comparison across workloads");
        println!("  extended      - Extended hit rate with all workload patterns");
        println!("  scan          - Scan resistance comparison");
        println!("  adaptation    - Adaptation speed comparison");
        println!("  detailed      - Detailed single benchmark with all metrics");
        println!("  memory        - Memory overhead comparison");
        println!("  comprehensive - Full policy comparison tables");
        println!("  all           - Run all reports");
        return;
    };

    match report.as_str() {
        "hit_rate" => print_hit_rate_comparison(),
        "extended" => print_extended_hit_rate_comparison(),
        "scan" => print_scan_resistance_comparison(),
        "adaptation" => print_adaptation_comparison(),
        "detailed" => run_detailed_single_benchmark(),
        "memory" => print_memory_overhead_comparison(),
        "comprehensive" => run_comprehensive_comparison(),
        "all" => {
            print_hit_rate_comparison();
            print_extended_hit_rate_comparison();
            print_scan_resistance_comparison();
            print_adaptation_comparison();
            run_detailed_single_benchmark();
            print_memory_overhead_comparison();
            run_comprehensive_comparison();
        },
        other => {
            eprintln!("Unknown report: {}", other);
            eprintln!("Run without arguments to see available reports.");
        },
    }
}

// ============================================================================
// Workload definitions
// ============================================================================

fn workloads() -> Vec<(&'static str, Workload)> {
    vec![
        ("uniform", Workload::Uniform),
        (
            "hotset_90_10",
            Workload::HotSet {
                hot_fraction: 0.1,
                hot_prob: 0.9,
            },
        ),
        ("scan", Workload::Scan),
        ("zipfian_1.0", Workload::Zipfian { exponent: 1.0 }),
        (
            "scrambled_zipf",
            Workload::ScrambledZipfian { exponent: 1.0 },
        ),
        ("latest", Workload::Latest { exponent: 0.8 }),
        (
            "scan_resistance",
            Workload::ScanResistance {
                scan_fraction: 0.2,
                scan_length: 1000,
                point_exponent: 1.0,
            },
        ),
        (
            "flash_crowd",
            Workload::FlashCrowd {
                base_exponent: 1.0,
                flash_prob: 0.001,
                flash_duration: 1000,
                flash_keys: 10,
                flash_intensity: 100.0,
            },
        ),
    ]
}

fn extended_workloads() -> Vec<(&'static str, Workload)> {
    vec![
        ("uniform", Workload::Uniform),
        (
            "hotset_90_10",
            Workload::HotSet {
                hot_fraction: 0.1,
                hot_prob: 0.9,
            },
        ),
        ("scan", Workload::Scan),
        ("zipfian_1.0", Workload::Zipfian { exponent: 1.0 }),
        ("zipfian_0.8", Workload::Zipfian { exponent: 0.8 }),
        (
            "scrambled_zipf",
            Workload::ScrambledZipfian { exponent: 1.0 },
        ),
        ("latest", Workload::Latest { exponent: 0.8 }),
        (
            "shifting_hotspot",
            Workload::ShiftingHotspot {
                shift_interval: 10_000,
                hot_fraction: 0.1,
            },
        ),
        ("exponential", Workload::Exponential { lambda: 0.05 }),
        ("pareto", Workload::Pareto { shape: 1.5 }),
        (
            "scan_resistance",
            Workload::ScanResistance {
                scan_fraction: 0.2,
                scan_length: 1000,
                point_exponent: 1.0,
            },
        ),
        (
            "correlated",
            Workload::Correlated {
                stride: 1,
                burst_len: 8,
                burst_prob: 0.3,
            },
        ),
        (
            "loop_small",
            Workload::Loop {
                working_set_size: 512,
            },
        ),
        (
            "working_set_churn",
            Workload::WorkingSetChurn {
                working_set_size: 2048,
                churn_rate: 0.001,
            },
        ),
        (
            "bursty",
            Workload::Bursty {
                hurst: 0.8,
                base_exponent: 1.0,
            },
        ),
        (
            "flash_crowd",
            Workload::FlashCrowd {
                base_exponent: 1.0,
                flash_prob: 0.001,
                flash_duration: 1000,
                flash_keys: 10,
                flash_intensity: 100.0,
            },
        ),
        ("mixture", Workload::Mixture),
    ]
}

// ============================================================================
// Helper functions
// ============================================================================

fn run_lru_workload(workload: Workload) -> f64 {
    let mut cache = LruCore::new(CAPACITY);
    let mut generator = WorkloadSpec {
        universe: UNIVERSE,
        workload,
        seed: SEED,
    }
    .generator();
    let stats = run_hit_rate(&mut cache, &mut generator, OPS, Arc::new);
    stats.hit_rate()
}

fn run_direct_workload<C: CoreCache<u64, Arc<u64>>>(cache: &mut C, workload: Workload) -> f64 {
    let mut generator = WorkloadSpec {
        universe: UNIVERSE,
        workload,
        seed: SEED,
    }
    .generator();
    let stats = run_hit_rate(cache, &mut generator, OPS, Arc::new);
    stats.hit_rate()
}

// ============================================================================
// Report functions
// ============================================================================

fn print_hit_rate_comparison() {
    println!(
        "\n=== Hit Rate Comparison (capacity={}, universe={}, ops={}) ===",
        CAPACITY, UNIVERSE, OPS
    );
    let wl_list = workloads();
    print!("{:<12}", "Policy");
    for (name, _) in &wl_list {
        print!(" {:>14}", name);
    }
    println!();
    println!("{}", "-".repeat(12 + wl_list.len() * 15));

    print!("{:<12}", "LRU");
    for (_, wl) in &wl_list {
        print!(" {:>13.2}%", run_lru_workload(*wl) * 100.0);
    }
    println!();

    print!("{:<12}", "LRU-K");
    for (_, wl) in &wl_list {
        let mut cache: LrukCache<u64, Arc<u64>> = LrukCache::new(CAPACITY);
        print!(" {:>13.2}%", run_direct_workload(&mut cache, *wl) * 100.0);
    }
    println!();

    print!("{:<12}", "LFU");
    for (_, wl) in &wl_list {
        let mut cache: LfuCache<u64, u64> = LfuCache::new(CAPACITY);
        let mut generator = WorkloadSpec {
            universe: UNIVERSE,
            workload: *wl,
            seed: SEED,
        }
        .generator();
        let stats = run_hit_rate(&mut cache, &mut generator, OPS, Arc::new);
        print!(" {:>13.2}%", stats.hit_rate() * 100.0);
    }
    println!();

    print!("{:<12}", "Heap-LFU");
    for (_, wl) in &wl_list {
        let mut cache: HeapLfuCache<u64, u64> = HeapLfuCache::new(CAPACITY);
        let mut generator = WorkloadSpec {
            universe: UNIVERSE,
            workload: *wl,
            seed: SEED,
        }
        .generator();
        let stats = run_hit_rate(&mut cache, &mut generator, OPS, Arc::new);
        print!(" {:>13.2}%", stats.hit_rate() * 100.0);
    }
    println!();

    print!("{:<12}", "Clock");
    for (_, wl) in &wl_list {
        let mut cache: ClockCache<u64, Arc<u64>> = ClockCache::new(CAPACITY);
        print!(" {:>13.2}%", run_direct_workload(&mut cache, *wl) * 100.0);
    }
    println!();

    print!("{:<12}", "S3-FIFO");
    for (_, wl) in &wl_list {
        let mut cache: S3FifoCache<u64, Arc<u64>> = S3FifoCache::new(CAPACITY);
        print!(" {:>13.2}%", run_direct_workload(&mut cache, *wl) * 100.0);
    }
    println!();

    print!("{:<12}", "2Q");
    for (_, wl) in &wl_list {
        let mut cache: TwoQCore<u64, Arc<u64>> = TwoQCore::new(CAPACITY, 0.25);
        print!(" {:>13.2}%", run_direct_workload(&mut cache, *wl) * 100.0);
    }
    println!();
}

fn print_extended_hit_rate_comparison() {
    println!(
        "\n=== Extended Hit Rate Comparison (capacity={}, universe={}, ops={}) ===",
        CAPACITY, UNIVERSE, OPS
    );
    let wl_list = extended_workloads();

    for chunk in wl_list.chunks(6) {
        print!("{:<12}", "Policy");
        for (name, _) in chunk {
            print!(" {:>12}", name);
        }
        println!();
        println!("{}", "-".repeat(12 + chunk.len() * 13));

        print!("{:<12}", "LRU");
        for (_, wl) in chunk {
            print!(" {:>11.2}%", run_lru_workload(*wl) * 100.0);
        }
        println!();

        print!("{:<12}", "LRU-K");
        for (_, wl) in chunk {
            let mut cache: LrukCache<u64, Arc<u64>> = LrukCache::new(CAPACITY);
            print!(" {:>11.2}%", run_direct_workload(&mut cache, *wl) * 100.0);
        }
        println!();

        print!("{:<12}", "S3-FIFO");
        for (_, wl) in chunk {
            let mut cache: S3FifoCache<u64, Arc<u64>> = S3FifoCache::new(CAPACITY);
            print!(" {:>11.2}%", run_direct_workload(&mut cache, *wl) * 100.0);
        }
        println!();

        print!("{:<12}", "2Q");
        for (_, wl) in chunk {
            let mut cache: TwoQCore<u64, Arc<u64>> = TwoQCore::new(CAPACITY, 0.25);
            print!(" {:>11.2}%", run_direct_workload(&mut cache, *wl) * 100.0);
        }
        println!("\n");
    }
}

fn print_scan_resistance_comparison() {
    println!("\n=== Scan Resistance Comparison ===");
    println!(
        "{:<12} {:>12} {:>12} {:>12} {:>12}",
        "Policy", "Baseline", "During Scan", "Recovery", "Score"
    );
    println!("{}", "-".repeat(60));

    let mut cache = LruCore::new(CAPACITY);
    let result = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
    println!(
        "{:<12} {:>11.2}% {:>11.2}% {:>11.2}% {:>11.2}",
        "LRU",
        result.baseline_hit_rate * 100.0,
        result.scan_hit_rate * 100.0,
        result.recovery_hit_rate * 100.0,
        result.resistance_score
    );

    let mut cache: LrukCache<u64, Arc<u64>> = LrukCache::new(CAPACITY);
    let result = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
    println!(
        "{:<12} {:>11.2}% {:>11.2}% {:>11.2}% {:>11.2}",
        "LRU-K",
        result.baseline_hit_rate * 100.0,
        result.scan_hit_rate * 100.0,
        result.recovery_hit_rate * 100.0,
        result.resistance_score
    );

    let mut cache: LfuCache<u64, u64> = LfuCache::new(CAPACITY);
    let result = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
    println!(
        "{:<12} {:>11.2}% {:>11.2}% {:>11.2}% {:>11.2}",
        "LFU",
        result.baseline_hit_rate * 100.0,
        result.scan_hit_rate * 100.0,
        result.recovery_hit_rate * 100.0,
        result.resistance_score
    );

    let mut cache: HeapLfuCache<u64, u64> = HeapLfuCache::new(CAPACITY);
    let result = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
    println!(
        "{:<12} {:>11.2}% {:>11.2}% {:>11.2}% {:>11.2}",
        "Heap-LFU",
        result.baseline_hit_rate * 100.0,
        result.scan_hit_rate * 100.0,
        result.recovery_hit_rate * 100.0,
        result.resistance_score
    );

    let mut cache: S3FifoCache<u64, Arc<u64>> = S3FifoCache::new(CAPACITY);
    let result = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
    println!(
        "{:<12} {:>11.2}% {:>11.2}% {:>11.2}% {:>11.2}",
        "S3-FIFO",
        result.baseline_hit_rate * 100.0,
        result.scan_hit_rate * 100.0,
        result.recovery_hit_rate * 100.0,
        result.resistance_score
    );

    let mut cache: TwoQCore<u64, Arc<u64>> = TwoQCore::new(CAPACITY, 0.25);
    let result = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
    println!(
        "{:<12} {:>11.2}% {:>11.2}% {:>11.2}% {:>11.2}",
        "2Q",
        result.baseline_hit_rate * 100.0,
        result.scan_hit_rate * 100.0,
        result.recovery_hit_rate * 100.0,
        result.resistance_score
    );

    let mut cache: ClockCache<u64, Arc<u64>> = ClockCache::new(CAPACITY);
    let result = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
    println!(
        "{:<12} {:>11.2}% {:>11.2}% {:>11.2}% {:>11.2}",
        "Clock",
        result.baseline_hit_rate * 100.0,
        result.scan_hit_rate * 100.0,
        result.recovery_hit_rate * 100.0,
        result.resistance_score
    );

    println!("\n--- Compact Summaries ---");
    let mut cache = LruCore::new(CAPACITY);
    let result = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
    println!("LRU:      {}", result.summary());

    let mut cache: S3FifoCache<u64, Arc<u64>> = S3FifoCache::new(CAPACITY);
    let result = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
    println!("S3-FIFO:  {}", result.summary());

    let mut cache: TwoQCore<u64, Arc<u64>> = TwoQCore::new(CAPACITY, 0.25);
    let result = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
    println!("2Q:       {}", result.summary());
}

fn print_adaptation_comparison() {
    println!("\n=== Adaptation Speed Comparison ===");
    println!(
        "{:<12} {:>15} {:>15} {:>12}",
        "Policy", "Ops to 50%", "Ops to 80%", "Stable HR"
    );
    println!("{}", "-".repeat(60));

    let mut cache = LruCore::new(CAPACITY);
    let result = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
    println!(
        "{:<12} {:>15} {:>15} {:>11.2}%",
        "LRU",
        result.ops_to_50_percent,
        result.ops_to_80_percent,
        result.stable_hit_rate * 100.0
    );

    let mut cache: LrukCache<u64, Arc<u64>> = LrukCache::new(CAPACITY);
    let result = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
    println!(
        "{:<12} {:>15} {:>15} {:>11.2}%",
        "LRU-K",
        result.ops_to_50_percent,
        result.ops_to_80_percent,
        result.stable_hit_rate * 100.0
    );

    let mut cache: LfuCache<u64, u64> = LfuCache::new(CAPACITY);
    let result = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
    println!(
        "{:<12} {:>15} {:>15} {:>11.2}%",
        "LFU",
        result.ops_to_50_percent,
        result.ops_to_80_percent,
        result.stable_hit_rate * 100.0
    );

    let mut cache: HeapLfuCache<u64, u64> = HeapLfuCache::new(CAPACITY);
    let result = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
    println!(
        "{:<12} {:>15} {:>15} {:>11.2}%",
        "Heap-LFU",
        result.ops_to_50_percent,
        result.ops_to_80_percent,
        result.stable_hit_rate * 100.0
    );

    let mut cache: S3FifoCache<u64, Arc<u64>> = S3FifoCache::new(CAPACITY);
    let result = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
    println!(
        "{:<12} {:>15} {:>15} {:>11.2}%",
        "S3-FIFO",
        result.ops_to_50_percent,
        result.ops_to_80_percent,
        result.stable_hit_rate * 100.0
    );

    let mut cache: TwoQCore<u64, Arc<u64>> = TwoQCore::new(CAPACITY, 0.25);
    let result = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
    println!(
        "{:<12} {:>15} {:>15} {:>11.2}%",
        "2Q",
        result.ops_to_50_percent,
        result.ops_to_80_percent,
        result.stable_hit_rate * 100.0
    );

    let mut cache: ClockCache<u64, Arc<u64>> = ClockCache::new(CAPACITY);
    let result = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
    println!(
        "{:<12} {:>15} {:>15} {:>11.2}%",
        "Clock",
        result.ops_to_50_percent,
        result.ops_to_80_percent,
        result.stable_hit_rate * 100.0
    );

    println!("\n--- Compact Summaries ---");
    let mut cache = LruCore::new(CAPACITY);
    let result = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
    println!("LRU:      {}", result.summary());

    let mut cache: S3FifoCache<u64, Arc<u64>> = S3FifoCache::new(CAPACITY);
    let result = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
    println!("S3-FIFO:  {}", result.summary());

    println!("\n--- LRU Adaptation Curve (hit rate per window) ---");
    let mut cache = LruCore::new(CAPACITY);
    let result = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
    for (i, &rate) in result.hit_rate_curve.iter().enumerate() {
        let bar_len = (rate * 40.0) as usize;
        println!(
            "Window {:2}: {:5.1}% {}",
            i + 1,
            rate * 100.0,
            "#".repeat(bar_len)
        );
    }
}

fn run_comprehensive_comparison() {
    println!("\n=== Comprehensive Policy Comparison ===\n");

    let suite = standard_workload_suite(UNIVERSE, SEED);

    // LRU
    {
        let mut comparison = PolicyComparison::new("LRU");
        for (workload_name, spec) in &suite {
            let mut cache = LruCore::new(CAPACITY);
            let config = BenchmarkConfig {
                name: workload_name.to_string(),
                capacity: CAPACITY,
                operations: OPS,
                warmup_ops: CAPACITY,
                workload: *spec,
                latency_sample_rate: 100,
                max_latency_samples: 10_000,
            };
            let result = run_benchmark("LRU", &mut cache, &config, Arc::new);
            comparison.add_result(result);
        }
        comparison.print_table();
        println!();
    }

    // S3-FIFO
    {
        let mut comparison = PolicyComparison::new("S3-FIFO");
        for (workload_name, spec) in &suite {
            let mut cache: S3FifoCache<u64, Arc<u64>> = S3FifoCache::new(CAPACITY);
            let config = BenchmarkConfig {
                name: workload_name.to_string(),
                capacity: CAPACITY,
                operations: OPS,
                warmup_ops: CAPACITY,
                workload: *spec,
                latency_sample_rate: 100,
                max_latency_samples: 10_000,
            };
            let result = run_benchmark("S3-FIFO", &mut cache, &config, Arc::new);
            comparison.add_result(result);
        }
        comparison.print_table();
        println!();
    }

    // Clock
    {
        let mut comparison = PolicyComparison::new("Clock");
        for (workload_name, spec) in &suite {
            let mut cache: ClockCache<u64, Arc<u64>> = ClockCache::new(CAPACITY);
            let config = BenchmarkConfig {
                name: workload_name.to_string(),
                capacity: CAPACITY,
                operations: OPS,
                warmup_ops: CAPACITY,
                workload: *spec,
                latency_sample_rate: 100,
                max_latency_samples: 10_000,
            };
            let result = run_benchmark("Clock", &mut cache, &config, Arc::new);
            comparison.add_result(result);
        }
        comparison.print_table();
        println!();
    }
}

fn run_detailed_single_benchmark() {
    println!("\n=== Detailed Benchmark Results ===\n");

    let workload = Workload::Zipfian { exponent: 1.0 };
    let spec = WorkloadSpec {
        universe: UNIVERSE,
        workload,
        seed: SEED,
    };

    let config = BenchmarkConfig {
        name: "zipfian_1.0".to_string(),
        capacity: CAPACITY,
        operations: OPS,
        warmup_ops: CAPACITY,
        workload: spec,
        latency_sample_rate: 100,
        max_latency_samples: 10_000,
    };

    let mut cache = LruCore::new(CAPACITY);
    let result = run_benchmark("LRU", &mut cache, &config, Arc::new);

    println!("Summary: {}\n", result.summary());

    println!("--- Configuration ---");
    println!("  Policy:     {}", result.policy_name);
    println!("  Workload:   {}", result.workload_name);
    println!("  Capacity:   {}", result.capacity);
    println!("  Universe:   {}", result.universe);
    println!("  Operations: {}", result.operations);

    println!("\n--- Hit Statistics ---");
    println!("  Hits:       {}", result.hit_stats.hits);
    println!("  Misses:     {}", result.hit_stats.misses);
    println!("  Inserts:    {}", result.hit_stats.inserts);
    println!("  Updates:    {}", result.hit_stats.updates);
    println!("  Hit Rate:   {:.2}%", result.hit_stats.hit_rate() * 100.0);
    println!("  Miss Rate:  {:.2}%", result.hit_stats.miss_rate() * 100.0);
    println!("  Total Ops:  {}", result.hit_stats.total_ops());

    println!("\n--- Throughput ---");
    println!("  Duration:       {:?}", result.throughput.total_duration);
    println!("  Ops/sec:        {:.0}", result.throughput.ops_per_sec);
    println!("  Gets/sec:       {:.0}", result.throughput.gets_per_sec);
    println!("  Inserts/sec:    {:.0}", result.throughput.inserts_per_sec);

    println!("\n--- Latency Distribution ---");
    println!("  Samples:  {}", result.latency.sample_count);
    println!("  Min:      {:?}", result.latency.min);
    println!("  p50:      {:?}", result.latency.p50);
    println!("  p95:      {:?}", result.latency.p95);
    println!("  p99:      {:?}", result.latency.p99);
    println!("  Max:      {:?}", result.latency.max);
    println!("  Mean:     {:?}", result.latency.mean);

    println!("\n--- Eviction Statistics ---");
    println!("  Total Evictions:     {}", result.eviction.total_evictions);
    println!(
        "  Evictions per Insert: {:.3}",
        result.eviction.evictions_per_insert
    );
}

fn print_memory_overhead_comparison() {
    println!("\n=== Memory Overhead Comparison ===");
    println!(
        "{:<12} {:>12} {:>15} {:>12}",
        "Policy", "Total (B)", "Bytes/Entry", "Entries"
    );
    println!("{}", "-".repeat(55));

    // LRU
    {
        let mut cache = LruCore::new(CAPACITY);
        for i in 0..CAPACITY as u64 {
            cache.insert(i, Arc::new(i));
        }
        let estimate = estimate_entry_overhead(&cache, cache.len());
        println!(
            "{:<12} {:>12} {:>15} {:>12}",
            "LRU", estimate.total_bytes, estimate.bytes_per_entry, estimate.entry_count
        );
        println!("  -> {}", estimate.summary());
    }

    // LRU-K
    {
        let mut cache: LrukCache<u64, Arc<u64>> = LrukCache::new(CAPACITY);
        for i in 0..CAPACITY as u64 {
            cache.insert(i, Arc::new(i));
        }
        let estimate = estimate_entry_overhead(&cache, cache.len());
        println!(
            "{:<12} {:>12} {:>15} {:>12}",
            "LRU-K", estimate.total_bytes, estimate.bytes_per_entry, estimate.entry_count
        );
    }

    // Clock
    {
        let mut cache: ClockCache<u64, Arc<u64>> = ClockCache::new(CAPACITY);
        for i in 0..CAPACITY as u64 {
            cache.insert(i, Arc::new(i));
        }
        let estimate = estimate_entry_overhead(&cache, cache.len());
        println!(
            "{:<12} {:>12} {:>15} {:>12}",
            "Clock", estimate.total_bytes, estimate.bytes_per_entry, estimate.entry_count
        );
    }

    // S3-FIFO
    {
        let mut cache: S3FifoCache<u64, Arc<u64>> = S3FifoCache::new(CAPACITY);
        for i in 0..CAPACITY as u64 {
            cache.insert(i, Arc::new(i));
        }
        let estimate = estimate_entry_overhead(&cache, cache.len());
        println!(
            "{:<12} {:>12} {:>15} {:>12}",
            "S3-FIFO", estimate.total_bytes, estimate.bytes_per_entry, estimate.entry_count
        );
    }

    // 2Q
    {
        let mut cache: TwoQCore<u64, Arc<u64>> = TwoQCore::new(CAPACITY, 0.25);
        for i in 0..CAPACITY as u64 {
            cache.insert(i, Arc::new(i));
        }
        let estimate = estimate_entry_overhead(&cache, cache.len());
        println!(
            "{:<12} {:>12} {:>15} {:>12}",
            "2Q", estimate.total_bytes, estimate.bytes_per_entry, estimate.entry_count
        );
    }

    println!(
        "\nNote: These are shallow size estimates (size_of_val). Heap allocations not included."
    );
}
