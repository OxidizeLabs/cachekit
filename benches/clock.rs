//! Benchmarks for ClockCache.
//!
//! Run with: `cargo bench --bench clock`

mod common;

use std::sync::Arc;
use std::time::Instant;

use cachekit::policy::clock::ClockCache;
use cachekit::traits::CoreCache;
use common::workload::{Workload, WorkloadSpec, run_hit_rate};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

// ============================================================================
// Insert + Get benchmarks (mixed operations)
// ============================================================================

fn bench_clock_insert_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("clock_policy");
    let ops_per_iter = 1024u64 * 2;
    group.throughput(Throughput::Elements(ops_per_iter));

    group.bench_function("insert_get", |b| {
        b.iter_batched(
            || {
                let mut cache = ClockCache::new(1024);
                for i in 0..1024u64 {
                    cache.insert(i, Arc::new(i));
                }
                cache
            },
            |mut cache| {
                for i in 0..1024u64 {
                    cache.insert(std::hint::black_box(i + 10_000), Arc::new(i));
                    let _ = std::hint::black_box(cache.get(&std::hint::black_box(i)));
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

// ============================================================================
// Eviction churn benchmarks (continuous eviction pressure)
// ============================================================================

fn bench_clock_eviction_churn(c: &mut Criterion) {
    let mut group = c.benchmark_group("clock_policy");
    group.throughput(Throughput::Elements(4096));

    group.bench_function("eviction_churn", |b| {
        b.iter_batched(
            || {
                let mut cache = ClockCache::new(1024);
                for i in 0..1024u64 {
                    cache.insert(i, Arc::new(i));
                }
                cache
            },
            |mut cache| {
                for i in 0..4096u64 {
                    cache.insert(std::hint::black_box(10_000 + i), Arc::new(i));
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

// ============================================================================
// Get hit benchmarks (pure read performance)
// ============================================================================

fn bench_clock_get_hit_ns(c: &mut Criterion) {
    c.bench_function("clock_get_hit_ns", |b| {
        b.iter_custom(|iters| {
            let capacity = 16_384u64;
            let mut cache = ClockCache::new(capacity as usize);
            for i in 0..capacity {
                cache.insert(i, Arc::new(i));
            }
            let start = Instant::now();
            for (idx, _) in (0..iters).enumerate() {
                let key = (idx as u64) % capacity;
                let _ = std::hint::black_box(cache.get(&key));
            }
            start.elapsed()
        })
    });
}

// ============================================================================
// Insert full (eviction on every insert)
// ============================================================================

fn bench_clock_insert_full_ns(c: &mut Criterion) {
    c.bench_function("clock_insert_full_ns", |b| {
        b.iter_custom(|iters| {
            let capacity = 4096u64;
            let mut cache = ClockCache::new(capacity as usize);
            for i in 0..capacity {
                cache.insert(i, Arc::new(i));
            }
            let values: Vec<_> = (0..1024u64).map(Arc::new).collect();
            let start = Instant::now();
            for i in 0..iters {
                let key = capacity + i;
                let value = values[(i as usize) % values.len()].clone();
                cache.insert(std::hint::black_box(key), value);
            }
            start.elapsed()
        })
    });
}

// ============================================================================
// Warmup insert (filling empty cache)
// ============================================================================

fn bench_clock_warmup_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("clock_policy");
    let capacity = 4096usize;
    group.throughput(Throughput::Elements(capacity as u64));

    group.bench_function("warmup_insert", |b| {
        b.iter_batched(
            || ClockCache::<u64, u64>::new(capacity),
            |mut cache| {
                for i in 0..capacity as u64 {
                    cache.insert(std::hint::black_box(i), i);
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

// ============================================================================
// Workload-based hit rate benchmarks
// ============================================================================

fn bench_clock_workload_hit_rate(c: &mut Criterion) {
    let mut group = c.benchmark_group("clock_workload_hit_rate");
    let operations = 200_000usize;
    group.throughput(Throughput::Elements(operations as u64));

    let specs = [
        ("uniform", Workload::Uniform),
        (
            "hotset_90_10",
            Workload::Hotset {
                hot_fraction: 0.1,
                hot_prob: 0.9,
            },
        ),
        ("scan", Workload::Scan),
        ("zipfian_0.99", Workload::Zipfian { theta: 0.99 }),
    ];

    for (name, workload) in specs {
        group.bench_function(name, |b| {
            b.iter_custom(|iters| {
                let mut total = std::time::Duration::default();
                for _ in 0..iters {
                    let mut cache = ClockCache::new(4096);
                    let mut generator = WorkloadSpec {
                        universe: 16_384,
                        workload,
                        seed: 42,
                    }
                    .generator();
                    let start = Instant::now();
                    let stats = run_hit_rate(&mut cache, &mut generator, operations, Arc::new);
                    let _ = std::hint::black_box(stats.hit_rate());
                    total += start.elapsed();
                }
                total
            })
        });
    }

    group.finish();
}

criterion_group!(
    end_to_end,
    bench_clock_insert_get,
    bench_clock_eviction_churn
);
criterion_group!(
    micro_ops,
    bench_clock_get_hit_ns,
    bench_clock_insert_full_ns
);
criterion_group!(policy_level, bench_clock_warmup_insert);
criterion_group!(workloads, bench_clock_workload_hit_rate);
criterion_main!(end_to_end, micro_ops, policy_level, workloads);
