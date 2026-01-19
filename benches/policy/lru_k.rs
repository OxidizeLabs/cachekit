//! LRU-K specific benchmarks for operations unique to LRU-K policy.
//!
//! Run with: `cargo bench --bench policy_lru_k`

use std::sync::Arc;

use cachekit::policy::lru_k::LrukCache;
use cachekit::traits::{CoreCache, LrukCacheTrait};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

// ============================================================================
// LRU-K specific operations
// ============================================================================

/// Benchmark pop_lru_k - removing based on K-th access time.
fn bench_pop_lru_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("lru_k_specific");
    group.throughput(Throughput::Elements(1024));

    group.bench_function("pop_lru_k", |b| {
        b.iter_batched(
            || {
                let mut cache = LrukCache::with_k(1024, 2);
                for i in 0..1024u64 {
                    cache.insert(i, Arc::new(i));
                }
                cache
            },
            |mut cache| {
                for _ in 0..1024u64 {
                    let _ = std::hint::black_box(cache.pop_lru_k());
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

/// Benchmark touch - updating access history.
fn bench_touch(c: &mut Criterion) {
    let mut group = c.benchmark_group("lru_k_specific");
    group.throughput(Throughput::Elements(4096));

    group.bench_function("touch_hotset", |b| {
        b.iter_batched(
            || {
                let mut cache = LrukCache::with_k(4096, 2);
                for i in 0..4096u64 {
                    cache.insert(i, Arc::new(i));
                }
                cache
            },
            |mut cache| {
                for i in 0..4096u64 {
                    let _ = std::hint::black_box(cache.touch(&std::hint::black_box(i)));
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

/// Benchmark with different K values.
fn bench_k_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("lru_k_values");
    let capacity = 4096;
    let ops = 4096;
    group.throughput(Throughput::Elements(ops));

    for k in [1, 2, 3, 4] {
        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(format!("k={}", k)),
            &k,
            |b, &k| {
                b.iter_batched(
                    || {
                        let mut cache = LrukCache::with_k(capacity, k);
                        for i in 0..capacity as u64 {
                            cache.insert(i, Arc::new(i));
                        }
                        cache
                    },
                    |mut cache| {
                        for i in 0..ops {
                            let _ = std::hint::black_box(cache.get(&std::hint::black_box(i)));
                        }
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_pop_lru_k, bench_touch, bench_k_values);
criterion_main!(benches);
