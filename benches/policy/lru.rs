//! LRU-specific benchmarks for operations unique to LRU policy.
//!
//! Run with: `cargo bench --bench policy_lru`

use std::sync::Arc;

use cachekit::policy::lru::LruCore;
use cachekit::traits::{CoreCache, LruCacheTrait};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

// ============================================================================
// LRU-specific operations
// ============================================================================

/// Benchmark pop_lru - removing the least recently used item.
fn bench_pop_lru(c: &mut Criterion) {
    let mut group = c.benchmark_group("lru_specific");
    group.throughput(Throughput::Elements(1024));

    group.bench_function("pop_lru", |b| {
        b.iter_batched(
            || {
                let mut cache = LruCore::new(1024);
                for i in 0..1024u64 {
                    cache.insert(i, Arc::new(i));
                }
                cache
            },
            |mut cache| {
                for _ in 0..1024u64 {
                    let _ = std::hint::black_box(cache.pop_lru());
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

/// Benchmark touch - promoting an item to most recently used without get overhead.
fn bench_touch(c: &mut Criterion) {
    let mut group = c.benchmark_group("lru_specific");
    group.throughput(Throughput::Elements(4096));

    group.bench_function("touch_hotset", |b| {
        b.iter_batched(
            || {
                let mut cache = LruCore::new(4096);
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

criterion_group!(benches, bench_pop_lru, bench_touch);
criterion_main!(benches);
