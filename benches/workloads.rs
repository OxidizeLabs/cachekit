//! Workload benchmarks - the single source of truth for policy comparison.
//!
//! Run with: `cargo bench --bench workloads`
//!
//! This benchmark compares ALL cache policies under identical workloads:
//! - Hit rate under various access patterns (uniform, zipfian, scan, etc.)
//! - Scan resistance (baseline → scan → recovery)
//! - Adaptation speed (workload shift response)
//! - Comprehensive metrics (latency, throughput, eviction stats)
//!
//! For micro-ops (get/insert latency), see: `cargo bench --bench ops`
//! For policy-specific operations, see: `cargo bench --bench policy_*`
//! For external crate comparison, see: `cargo bench --bench comparison`

mod common;

use std::time::Instant;

use common::metrics::{
    BenchmarkConfig, measure_adaptation_speed, measure_scan_resistance, run_benchmark,
    standard_workload_suite,
};
use common::registry::STANDARD_WORKLOADS;
use common::workload::{WorkloadSpec, run_hit_rate};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

const CAPACITY: usize = 4096;
const UNIVERSE: u64 = 16_384;
const OPS: usize = 200_000;
const SEED: u64 = 42;

// ============================================================================
// Hit Rate Benchmarks
// ============================================================================

fn bench_hit_rates(c: &mut Criterion) {
    let mut group = c.benchmark_group("hit_rate");
    group.throughput(Throughput::Elements(OPS as u64));

    for workload_case in STANDARD_WORKLOADS {
        let workload = workload_case.workload;
        let workload_id = workload_case.id;

        for_each_policy! {
            with |policy_id, _display_name, make_cache| {
                group.bench_with_input(
                    BenchmarkId::new(policy_id, workload_id),
                    &workload,
                    |b, &wl| {
                        b.iter_custom(|iters| {
                            let mut total = std::time::Duration::default();
                            for _ in 0..iters {
                                let mut cache = make_cache(CAPACITY);
                                let mut generator = WorkloadSpec {
                                    universe: UNIVERSE,
                                    workload: wl,
                                    seed: SEED,
                                }
                                .generator();
                                let start = Instant::now();
                                let _ = run_hit_rate(&mut cache, &mut generator, OPS, Arc::new);
                                total += start.elapsed();
                            }
                            total
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

// ============================================================================
// Scan Resistance Benchmarks
// ============================================================================

fn bench_scan_resistance(c: &mut Criterion) {
    let mut group = c.benchmark_group("scan_resistance");

    for_each_policy! {
        with |policy_id, _display_name, make_cache| {
            group.bench_function(policy_id, |b| {
                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::default();
                    for _ in 0..iters {
                        let mut cache = make_cache(CAPACITY);
                        let start = Instant::now();
                        let _ = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
                        total += start.elapsed();
                    }
                    total
                });
            });
        }
    }

    group.finish();
}

// ============================================================================
// Adaptation Speed Benchmarks
// ============================================================================

fn bench_adaptation_speed(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptation_speed");

    for_each_policy! {
        with |policy_id, _display_name, make_cache| {
            group.bench_function(policy_id, |b| {
                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::default();
                    for _ in 0..iters {
                        let mut cache = make_cache(CAPACITY);
                        let start = Instant::now();
                        let _ = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
                        total += start.elapsed();
                    }
                    total
                });
            });
        }
    }

    group.finish();
}

// ============================================================================
// Comprehensive Benchmarks (latency + throughput + eviction stats)
// ============================================================================

fn bench_comprehensive(c: &mut Criterion) {
    let mut group = c.benchmark_group("comprehensive");
    let suite = standard_workload_suite(UNIVERSE, SEED);

    for (workload_name, spec) in &suite {
        let config = BenchmarkConfig {
            name: workload_name.to_string(),
            capacity: CAPACITY,
            operations: OPS,
            warmup_ops: CAPACITY,
            workload: *spec,
            latency_sample_rate: 100,
            max_latency_samples: 10_000,
        };

        for_each_policy! {
            with |policy_id, _display_name, make_cache| {
                group.bench_with_input(
                    BenchmarkId::new(policy_id, workload_name),
                    &config,
                    |b, cfg| {
                        b.iter_custom(|iters| {
                            let mut total = std::time::Duration::default();
                            for _ in 0..iters {
                                let mut cache = make_cache(CAPACITY);
                                let start = Instant::now();
                                let _ = run_benchmark(policy_id, &mut cache, cfg, Arc::new);
                                total += start.elapsed();
                            }
                            total
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_hit_rates,
    bench_scan_resistance,
    bench_adaptation_speed,
    bench_comprehensive,
);
criterion_main!(benches);

// For human-readable reports, run: cargo bench --bench reports -- <report>
// See: benches/reports.rs
