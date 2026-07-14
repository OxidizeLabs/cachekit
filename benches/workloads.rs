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
//!
//! ## Value construction convention
//!
//! Benchmark entry points take a `value_for_key: Fn(u64) -> Arc<V>` so callers
//! choose what to store. To keep `Arc::new` allocations out of the timed
//! region, we pre-build the values once per benchmark and the closure just
//! clones a handle:
//!
//! - Workloads that revisit keys (hit-rate, comprehensive) use a per-key
//!   [`Arc<u64>`] pool indexed by key, so repeated accesses share the same
//!   allocation.
//! - Scan-resistance and adaptation tests touch each key at most once and
//!   don't depend on payload contents, so they share a single [`Arc<u64>`].
//!
//! This isolates policy behavior from allocator noise on hot paths that
//! issue millions of operations.

use bench_support as common;
use bench_support::for_each_policy;

use std::sync::Arc;
use std::time::Instant;

use common::metrics::{
    BenchmarkConfig, measure_adaptation_speed, measure_scan_resistance, run_benchmark,
};
use common::operation::{ReadThrough, run_operations};
use common::registry::STANDARD_WORKLOADS;
use common::workload::{Workload, WorkloadGenerator, WorkloadSpec};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

const CAPACITY: usize = 4096;
const UNIVERSE: u64 = 16_384;
const OPS: usize = 200_000;
const SEED: u64 = 42;
/// Read-through probability used by the operation model (1.0 = always read-through).
const READ_THROUGH_RATIO: f64 = 1.0;

fn make_generator(workload: Workload) -> WorkloadGenerator {
    WorkloadSpec {
        universe: UNIVERSE,
        workload,
        seed: SEED,
    }
    .generator()
}

/// Pre-allocate one `Arc<u64>` per key in the universe.
///
/// Reused across policies and workloads so the measured loop only pays the
/// cost of an `Arc::clone` (atomic refcount bump) per insert rather than a
/// fresh heap allocation. This isolates policy hit/miss/eviction behavior
/// from allocator variance.
fn preallocate_values() -> Vec<Arc<u64>> {
    (0..UNIVERSE).map(Arc::new).collect()
}

// ============================================================================
// Hit Rate Benchmarks
// ============================================================================

fn bench_hit_rates(c: &mut Criterion) {
    let mut group = c.benchmark_group("hit_rate");
    group.throughput(Throughput::Elements(OPS as u64));
    let value_pool = preallocate_values();

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
                                let mut generator = make_generator(wl);
                                let mut op_model = ReadThrough::new(READ_THROUGH_RATIO, SEED);
                                let start = Instant::now();
                                let _ = run_operations(
                                    &mut cache,
                                    &mut generator,
                                    OPS,
                                    &mut op_model,
                                    |k| {
                                        Arc::clone(
                                            value_pool
                                                .get(k as usize)
                                                .expect("generator produced key outside configured UNIVERSE"),
                                        )
                                    },
                                );
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
    let shared_value: Arc<u64> = Arc::new(0);

    for_each_policy! {
        with |policy_id, _display_name, make_cache| {
            group.bench_function(policy_id, |b| {
                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::default();
                    for _ in 0..iters {
                        let mut cache = make_cache(CAPACITY);
                        let start = Instant::now();
                        let _ = measure_scan_resistance(
                            &mut cache,
                            CAPACITY,
                            UNIVERSE,
                            |_| Arc::clone(&shared_value),
                        );
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
    let shared_value: Arc<u64> = Arc::new(0);

    for_each_policy! {
        with |policy_id, _display_name, make_cache| {
            group.bench_function(policy_id, |b| {
                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::default();
                    for _ in 0..iters {
                        let mut cache = make_cache(CAPACITY);
                        let start = Instant::now();
                        let _ = measure_adaptation_speed(
                            &mut cache,
                            CAPACITY,
                            UNIVERSE,
                            |_| Arc::clone(&shared_value),
                        );
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
    let value_pool = preallocate_values();

    for workload_case in STANDARD_WORKLOADS {
        let config = BenchmarkConfig {
            name: workload_case.id.to_string(),
            capacity: CAPACITY,
            operations: OPS,
            warmup_ops: CAPACITY,
            workload: workload_case.with_params(UNIVERSE, SEED),
            latency_sample_rate: 100,
            max_latency_samples: 10_000,
        };

        for_each_policy! {
            with |policy_id, _display_name, make_cache| {
                group.bench_with_input(
                    BenchmarkId::new(policy_id, workload_case.id),
                    &config,
                    |b, cfg| {
                        b.iter_custom(|iters| {
                            let mut total = std::time::Duration::default();
                            for _ in 0..iters {
                                let mut cache = make_cache(CAPACITY);
                                let start = Instant::now();
                                let _ = run_benchmark(
                                    policy_id,
                                    &mut cache,
                                    cfg,
                                    |key| {
                                        let idx = key as usize;
                                        Arc::clone(value_pool.get(idx).unwrap_or_else(|| {
                                            panic!(
                                                "workload key {} out of range for value_pool (len {}, expected < UNIVERSE)",
                                                key,
                                                value_pool.len()
                                            )
                                        }))
                                    },
                                );
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
