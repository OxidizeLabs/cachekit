//! LFU-specific benchmarks for operations unique to LFU policy.
//!
//! Run with: `cargo bench --bench policy_lfu`

use std::sync::Arc;

use cachekit::ds::FrequencyBucketsHandle;
use cachekit::policy::lfu::LfuCache;
use cachekit::traits::{CoreCache, LfuCacheTrait};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

// ============================================================================
// LFU-specific operations
// ============================================================================

/// Benchmark pop_lfu - removing the least frequently used item.
fn bench_pop_lfu(c: &mut Criterion) {
    let mut group = c.benchmark_group("lfu_specific");
    group.throughput(Throughput::Elements(1024));

    group.bench_function("pop_lfu", |b| {
        b.iter_batched(
            || {
                let mut cache = LfuCache::new(1024);
                for i in 0..1024u64 {
                    cache.insert(i, Arc::new(i));
                }
                cache
            },
            |mut cache| {
                for _ in 0..1024u64 {
                    let _ = std::hint::black_box(cache.pop_lfu());
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

/// Benchmark increment_frequency - manually bumping frequency counters.
fn bench_frequency_updates(c: &mut Criterion) {
    let mut group = c.benchmark_group("lfu_specific");
    group.throughput(Throughput::Elements(4096));

    group.bench_function("increment_frequency", |b| {
        b.iter_batched(
            || {
                let mut cache = LfuCache::new(4096);
                for i in 0..4096u64 {
                    cache.insert(i, Arc::new(i));
                }
                cache
            },
            |mut cache| {
                for i in 0..4096u64 {
                    let _ =
                        std::hint::black_box(cache.increment_frequency(&std::hint::black_box(i)));
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

/// Benchmark raw frequency bucket operations (policy-only, no storage overhead).
fn bench_bucket_touch(c: &mut Criterion) {
    let mut group = c.benchmark_group("lfu_specific");
    group.throughput(Throughput::Elements(16_384));

    group.bench_function("bucket_touch_raw", |b| {
        b.iter_custom(|iters| {
            let capacity = 16_384u64;
            let mut buckets = FrequencyBucketsHandle::new();
            for i in 0..capacity {
                buckets.insert(i);
            }
            let start = std::time::Instant::now();
            for (idx, _) in (0..iters).enumerate() {
                let handle = (idx as u64) % capacity;
                let _ = std::hint::black_box(buckets.touch(&handle));
            }
            start.elapsed()
        })
    });

    group.finish();
}

/// Benchmark eviction churn at different cache sizes.
fn bench_eviction_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("lfu_eviction_scaling");

    for &capacity in &[256usize, 1024, 4096, 16384] {
        let inserts = capacity * 4;
        group.throughput(Throughput::Elements(inserts as u64));
        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(capacity),
            &capacity,
            |b, &capacity| {
                b.iter_batched(
                    || {
                        let mut cache = LfuCache::new(capacity);
                        for i in 0..capacity as u64 {
                            cache.insert(i, Arc::new(i));
                        }
                        let values: Vec<_> = (0..inserts as u64).map(Arc::new).collect();
                        (cache, values)
                    },
                    |(mut cache, values)| {
                        for i in 0..inserts as u64 {
                            cache.insert(
                                std::hint::black_box(10_000 + i),
                                values[i as usize].clone(),
                            );
                        }
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_pop_lfu,
    bench_frequency_updates,
    bench_bucket_touch,
    bench_eviction_scaling
);
criterion_main!(benches);
