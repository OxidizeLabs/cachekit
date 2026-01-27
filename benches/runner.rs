//! Benchmark runner that produces JSON artifacts.
//!
//! This runner executes the full benchmark matrix and outputs structured
//! JSON results to `target/benchmarks/<run-id>/results.json`.
//!
//! Run with: `cargo bench --bench runner`

use bench_support as common;
use bench_support::for_each_policy;

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use chrono::Utc;
use common::json_results::{
    AdaptationStats, BenchmarkArtifact, BenchmarkConfig, EvictionStats, HitStats, LatencyStats,
    Metrics, ResultRow, RunMetadata, SCHEMA_VERSION, ScanResistanceStats, ThroughputStats,
    duration_to_nanos,
};
use common::metrics::{
    BenchmarkConfig as InternalConfig, measure_adaptation_speed, measure_scan_resistance,
    run_benchmark,
};
use common::operation::{ReadThrough, run_operations};
use common::registry::STANDARD_WORKLOADS;

// Benchmark configuration constants
const CAPACITY: usize = 4096;
const UNIVERSE: u64 = 16_384;
const OPS: usize = 200_000;
const SEED: u64 = 42;

fn main() {
    println!("=== CacheKit Benchmark Runner ===");
    println!("Schema version: {}", SCHEMA_VERSION);
    println!();

    // Collect metadata
    let metadata = collect_metadata();
    println!("Run ID: {}", metadata.timestamp);
    println!(
        "Git commit: {}",
        metadata.git_commit.as_deref().unwrap_or("unknown")
    );
    println!("Rustc: {}", metadata.rustc_version);
    println!();

    // Create artifact
    let mut artifact = BenchmarkArtifact::new(metadata);

    // Run benchmarks
    println!("Running benchmark matrix...");
    println!();

    run_hit_rate_benchmarks(&mut artifact);
    run_scan_resistance_benchmarks(&mut artifact);
    run_adaptation_benchmarks(&mut artifact);
    run_comprehensive_benchmarks(&mut artifact);

    // Save results
    let output_dir = create_output_directory(&artifact.metadata.timestamp);
    let output_path = output_dir.join("results.json");

    println!();
    println!("Saving results to: {}", output_path.display());

    let json = serde_json::to_string_pretty(&artifact).expect("Failed to serialize results");
    fs::write(&output_path, json).expect("Failed to write results.json");

    println!("✓ Benchmark complete!");
    println!("  Total results: {}", artifact.results.len());
    println!("  Output: {}", output_path.display());
}

/// Collect metadata about the benchmark run environment.
fn collect_metadata() -> RunMetadata {
    RunMetadata {
        timestamp: Utc::now().to_rfc3339(),
        git_commit: get_git_commit(),
        git_branch: get_git_branch(),
        git_dirty: is_git_dirty(),
        rustc_version: get_rustc_version(),
        host_triple: get_host_triple(),
        cpu_model: get_cpu_model(),
        config: BenchmarkConfig {
            capacity: CAPACITY,
            universe: UNIVERSE,
            operations: OPS,
            seed: SEED,
        },
    }
}

/// Get the current git commit SHA.
fn get_git_commit() -> Option<String> {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
}

/// Get the current git branch name.
fn get_git_branch() -> Option<String> {
    Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
}

/// Check if the git working directory has uncommitted changes.
fn is_git_dirty() -> bool {
    Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .map(|output| !output.stdout.is_empty())
        .unwrap_or(false)
}

/// Get the rustc version string.
fn get_rustc_version() -> String {
    Command::new("rustc")
        .args(["--version"])
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Get the host triple.
fn get_host_triple() -> String {
    std::env::var("TARGET").unwrap_or_else(|_| {
        Command::new("rustc")
            .args(["-vV"])
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|s| {
                s.lines()
                    .find(|line| line.starts_with("host: "))
                    .map(|line| line.trim_start_matches("host: ").to_string())
            })
            .unwrap_or_else(|| "unknown".to_string())
    })
}

/// Get the CPU model name (platform-specific).
fn get_cpu_model() -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .map(|s| s.trim().to_string())
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|content| {
                content
                    .lines()
                    .find(|line| line.starts_with("model name"))
                    .and_then(|line| line.split(':').nth(1))
                    .map(|s| s.trim().to_string())
            })
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

/// Create output directory for benchmark results.
fn create_output_directory(run_id: &str) -> PathBuf {
    // Use a simpler filename-safe version of the timestamp
    let safe_id = run_id.replace([':', '.'], "-");
    let dir = PathBuf::from("target").join("benchmarks").join(safe_id);

    fs::create_dir_all(&dir).expect("Failed to create output directory");
    dir
}

/// Run hit rate benchmarks for all policies and workloads.
fn run_hit_rate_benchmarks(artifact: &mut BenchmarkArtifact) {
    println!("[1/4] Running hit rate benchmarks...");

    for workload_case in STANDARD_WORKLOADS {
        print!("  Workload: {:<20} ", workload_case.id);

        for_each_policy! {
            with |policy_id, policy_name, make_cache| {
                let mut cache = make_cache(CAPACITY);
                let mut generator = workload_case.with_params(UNIVERSE, SEED).generator();

                let mut op_model = ReadThrough::new(1.0, SEED);
                let stats = run_operations(&mut cache, &mut generator, OPS, &mut op_model, Arc::new);

                artifact.add_result(ResultRow {
                    policy_id: policy_id.to_string(),
                    policy_name: policy_name.to_string(),
                    workload_id: workload_case.id.to_string(),
                    workload_name: workload_case.display_name.to_string(),
                    case_id: "hit_rate".to_string(),
                    metrics: Metrics {
                        hit_stats: Some(HitStats {
                            hits: stats.hits,
                            misses: stats.misses,
                            inserts: stats.inserts,
                            updates: stats.updates,
                            hit_rate: stats.hit_rate(),
                            miss_rate: 1.0 - stats.hit_rate(),
                        }),
                        throughput: None,
                        latency: None,
                        eviction: None,
                        scan_resistance: None,
                        adaptation: None,
                    },
                });
            }
        }

        println!("✓");
    }

    println!();
}

/// Run scan resistance benchmarks for all policies.
fn run_scan_resistance_benchmarks(artifact: &mut BenchmarkArtifact) {
    println!("[2/4] Running scan resistance benchmarks...");

    for_each_policy! {
        with |policy_id, policy_name, make_cache| {
            print!("  Policy: {:<12} ", policy_name);

            let mut cache = make_cache(CAPACITY);
            let result = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);

            artifact.add_result(ResultRow {
                policy_id: policy_id.to_string(),
                policy_name: policy_name.to_string(),
                workload_id: "scan_resistance_test".to_string(),
                workload_name: "Scan Resistance Test".to_string(),
                case_id: "scan_resistance".to_string(),
                metrics: Metrics {
                    hit_stats: None,
                    throughput: None,
                    latency: None,
                    eviction: None,
                    scan_resistance: Some(ScanResistanceStats {
                        baseline_hit_rate: result.baseline_hit_rate,
                        scan_hit_rate: result.scan_hit_rate,
                        recovery_hit_rate: result.recovery_hit_rate,
                        resistance_score: result.resistance_score,
                    }),
                    adaptation: None,
                },
            });

            println!("✓");
        }
    }

    println!();
}

/// Run adaptation speed benchmarks for all policies.
fn run_adaptation_benchmarks(artifact: &mut BenchmarkArtifact) {
    println!("[3/4] Running adaptation speed benchmarks...");

    for_each_policy! {
        with |policy_id, policy_name, make_cache| {
            print!("  Policy: {:<12} ", policy_name);

            let mut cache = make_cache(CAPACITY);
            let result = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);

            artifact.add_result(ResultRow {
                policy_id: policy_id.to_string(),
                policy_name: policy_name.to_string(),
                workload_id: "adaptation_test".to_string(),
                workload_name: "Adaptation Test".to_string(),
                case_id: "adaptation".to_string(),
                metrics: Metrics {
                    hit_stats: None,
                    throughput: None,
                    latency: None,
                    eviction: None,
                    scan_resistance: None,
                    adaptation: Some(AdaptationStats {
                        stable_hit_rate: result.stable_hit_rate,
                        ops_to_50_percent: result.ops_to_50_percent,
                        ops_to_80_percent: result.ops_to_80_percent,
                    }),
                },
            });

            println!("✓");
        }
    }

    println!();
}

/// Run comprehensive benchmarks (with latency/throughput) for all policies and workloads.
fn run_comprehensive_benchmarks(artifact: &mut BenchmarkArtifact) {
    println!("[4/4] Running comprehensive benchmarks...");

    // Use a subset of workloads for comprehensive benchmarks (they're slower)
    let comprehensive_workloads = [
        &STANDARD_WORKLOADS[0], // uniform
        &STANDARD_WORKLOADS[3], // zipfian_1.0
        &STANDARD_WORKLOADS[1], // hotset_90_10
    ];

    for workload_case in &comprehensive_workloads {
        print!("  Workload: {:<20} ", workload_case.id);

        for_each_policy! {
            with |policy_id, policy_name, make_cache| {
                let mut cache = make_cache(CAPACITY);

                let config = InternalConfig {
                    name: workload_case.id.to_string(),
                    capacity: CAPACITY,
                    operations: OPS,
                    warmup_ops: CAPACITY,
                    workload: workload_case.with_params(UNIVERSE, SEED),
                    latency_sample_rate: 100,
                    max_latency_samples: 10_000,
                };

                let result = run_benchmark(policy_id, &mut cache, &config, Arc::new);

                artifact.add_result(ResultRow {
                    policy_id: policy_id.to_string(),
                    policy_name: policy_name.to_string(),
                    workload_id: workload_case.id.to_string(),
                    workload_name: workload_case.display_name.to_string(),
                    case_id: "comprehensive".to_string(),
                    metrics: Metrics {
                        hit_stats: Some(HitStats {
                            hits: result.hit_stats.hits,
                            misses: result.hit_stats.misses,
                            inserts: result.hit_stats.inserts,
                            updates: result.hit_stats.updates,
                            hit_rate: result.hit_stats.hit_rate(),
                            miss_rate: result.hit_stats.miss_rate(),
                        }),
                        throughput: Some(ThroughputStats {
                            duration_ms: result.throughput.total_duration.as_secs_f64() * 1000.0,
                            ops_per_sec: result.throughput.ops_per_sec,
                            gets_per_sec: result.throughput.gets_per_sec,
                            inserts_per_sec: result.throughput.inserts_per_sec,
                        }),
                        latency: Some(LatencyStats {
                            sample_count: result.latency.sample_count,
                            min_ns: duration_to_nanos(result.latency.min),
                            p50_ns: duration_to_nanos(result.latency.p50),
                            p95_ns: duration_to_nanos(result.latency.p95),
                            p99_ns: duration_to_nanos(result.latency.p99),
                            max_ns: duration_to_nanos(result.latency.max),
                            mean_ns: duration_to_nanos(result.latency.mean),
                        }),
                        eviction: Some(EvictionStats {
                            total_evictions: result.eviction.total_evictions,
                            evictions_per_insert: result.eviction.evictions_per_insert,
                        }),
                        scan_resistance: None,
                        adaptation: None,
                    },
                });
            }
        }

        println!("✓");
    }

    println!();
}
