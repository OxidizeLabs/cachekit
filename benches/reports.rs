//! Policy comparison reports - run with: `cargo bench --bench reports`
//!
//! This is a standalone binary (not a criterion benchmark) that prints
//! human-readable comparison tables for cache policy evaluation.

use bench_support as common;
use bench_support::for_each_policy;

use std::sync::Arc;

use cachekit::traits::CoreCache;
use common::metrics::{
    BenchmarkConfig, PolicyComparison, estimate_entry_overhead, measure_adaptation_speed,
    measure_scan_resistance, run_benchmark, standard_workload_suite,
};
use common::registry::{EXTENDED_WORKLOADS, STANDARD_WORKLOADS};
use common::workload::run_hit_rate;

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
// Helper functions
// ============================================================================

fn run_workload<C: CoreCache<u64, Arc<u64>>>(
    cache: &mut C,
    workload_case: &common::registry::WorkloadCase,
) -> f64 {
    let mut generator = workload_case.with_params(UNIVERSE, SEED).generator();
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

    // Print header
    print!("{:<12}", "Policy");
    for workload_case in STANDARD_WORKLOADS {
        print!(" {:>14}", workload_case.id);
    }
    println!();
    println!("{}", "-".repeat(12 + STANDARD_WORKLOADS.len() * 15));

    // Print each policy row
    for_each_policy! {
        with |_policy_id, display_name, make_cache| {
            print!("{:<12}", display_name);
            for workload_case in STANDARD_WORKLOADS {
                let mut cache = make_cache(CAPACITY);
                print!(" {:>13.2}%", run_workload(&mut cache, workload_case) * 100.0);
            }
            println!();
        }
    }
}

fn print_extended_hit_rate_comparison() {
    println!(
        "\n=== Extended Hit Rate Comparison (capacity={}, universe={}, ops={}) ===",
        CAPACITY, UNIVERSE, OPS
    );

    for chunk in EXTENDED_WORKLOADS.chunks(6) {
        print!("{:<12}", "Policy");
        for workload_case in chunk {
            print!(" {:>12}", workload_case.id);
        }
        println!();
        println!("{}", "-".repeat(12 + chunk.len() * 13));

        for_each_policy! {
            with |_policy_id, display_name, make_cache| {
                print!("{:<12}", display_name);
                for workload_case in chunk {
                    let mut cache = make_cache(CAPACITY);
                    print!(" {:>11.2}%", run_workload(&mut cache, workload_case) * 100.0);
                }
                println!();
            }
        }

        println!();
    }
}

fn print_scan_resistance_comparison() {
    println!("\n=== Scan Resistance Comparison ===");
    println!(
        "{:<12} {:>12} {:>12} {:>12} {:>12}",
        "Policy", "Baseline", "During Scan", "Recovery", "Score"
    );
    println!("{}", "-".repeat(60));

    for_each_policy! {
        with |_policy_id, display_name, make_cache| {
            let mut cache = make_cache(CAPACITY);
            let result = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
            println!(
                "{:<12} {:>11.2}% {:>11.2}% {:>11.2}% {:>11.2}",
                display_name,
                result.baseline_hit_rate * 100.0,
                result.scan_hit_rate * 100.0,
                result.recovery_hit_rate * 100.0,
                result.resistance_score
            );
        }
    }

    println!("\n--- Compact Summaries ---");
    for_each_policy! {
        with |_policy_id, display_name, make_cache| {
            let mut cache = make_cache(CAPACITY);
            let result = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
            println!("{:<10} {}", display_name, result.summary());
        }
    }
}

fn print_adaptation_comparison() {
    println!("\n=== Adaptation Speed Comparison ===");
    println!(
        "{:<12} {:>15} {:>15} {:>12}",
        "Policy", "Ops to 50%", "Ops to 80%", "Stable HR"
    );
    println!("{}", "-".repeat(60));

    for_each_policy! {
        with |_policy_id, display_name, make_cache| {
            let mut cache = make_cache(CAPACITY);
            let result = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
            println!(
                "{:<12} {:>15} {:>15} {:>11.2}%",
                display_name,
                result.ops_to_50_percent,
                result.ops_to_80_percent,
                result.stable_hit_rate * 100.0
            );
        }
    }

    println!("\n--- Compact Summaries ---");
    for_each_policy! {
        with |_policy_id, display_name, make_cache| {
            let mut cache = make_cache(CAPACITY);
            let result = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
            println!("{:<10} {}", display_name, result.summary());
        }
    }

    // Show detailed curve for first policy (LRU) as an example
    println!("\n--- LRU Adaptation Curve (hit rate per window) ---");
    for_each_policy! {
        with |policy_id, _display_name, make_cache| {
            if policy_id == "lru" {
                let mut cache = make_cache(CAPACITY);
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
        }
    }
}

fn run_comprehensive_comparison() {
    println!("\n=== Comprehensive Policy Comparison ===\n");

    let suite = standard_workload_suite(UNIVERSE, SEED);

    for_each_policy! {
        with |policy_id, display_name, make_cache| {
            let mut comparison = PolicyComparison::new(display_name);
            for (workload_name, spec) in &suite {
                let mut cache = make_cache(CAPACITY);
                let config = BenchmarkConfig {
                    name: workload_name.to_string(),
                    capacity: CAPACITY,
                    operations: OPS,
                    warmup_ops: CAPACITY,
                    workload: *spec,
                    latency_sample_rate: 100,
                    max_latency_samples: 10_000,
                };
                let result = run_benchmark(policy_id, &mut cache, &config, Arc::new);
                comparison.add_result(result);
            }
            comparison.print_table();
            println!();
        }
    }
}

fn run_detailed_single_benchmark() {
    println!("\n=== Detailed Benchmark Results ===\n");

    // Use first workload from standard suite (zipfian 1.0)
    let workload_case = &STANDARD_WORKLOADS[3]; // zipfian_1.0
    let spec = workload_case.with_params(UNIVERSE, SEED);

    let config = BenchmarkConfig {
        name: workload_case.id.to_string(),
        capacity: CAPACITY,
        operations: OPS,
        warmup_ops: CAPACITY,
        workload: spec,
        latency_sample_rate: 100,
        max_latency_samples: 10_000,
    };

    // Run detailed benchmark for first policy (LRU)
    for_each_policy! {
        with |policy_id, display_name, make_cache| {
            if policy_id == "lru" {
                let mut cache = make_cache(CAPACITY);
                let result = run_benchmark(display_name, &mut cache, &config, Arc::new);

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
        }
    }
}

fn print_memory_overhead_comparison() {
    println!("\n=== Memory Overhead Comparison ===");
    println!(
        "{:<12} {:>12} {:>15} {:>12}",
        "Policy", "Total (B)", "Bytes/Entry", "Entries"
    );
    println!("{}", "-".repeat(55));

    for_each_policy! {
        with |_policy_id, display_name, make_cache| {
            let mut cache = make_cache(CAPACITY);
            for i in 0..CAPACITY as u64 {
                cache.insert(i, Arc::new(i));
            }
            let estimate = estimate_entry_overhead(&cache, cache.len());
            println!(
                "{:<12} {:>12} {:>15} {:>12}",
                display_name, estimate.total_bytes, estimate.bytes_per_entry, estimate.entry_count
            );
        }
    }

    println!(
        "\nNote: These are shallow size estimates (size_of_val). Heap allocations not included."
    );
}
