use cachekit::policy::lru::LRUCore;
use cachekit::traits::{CoreCache, LRUCacheTrait};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use std::sync::Arc;

fn bench_lru_insert_get(c: &mut Criterion) {
    c.bench_function("lru_insert_get", |b| {
        b.iter_batched(
            || {
                let mut cache = LRUCore::new(1024);
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
}

fn bench_lru_pop_lru(c: &mut Criterion) {
    c.bench_function("lru_pop_lru", |b| {
        b.iter_batched(
            || {
                let mut cache = LRUCore::new(1024);
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
}

fn bench_lru_eviction_churn(c: &mut Criterion) {
    c.bench_function("lru_eviction_churn", |b| {
        b.iter_batched(
            || {
                let mut cache = LRUCore::new(1024);
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
}

fn bench_lru_touch_hotset(c: &mut Criterion) {
    c.bench_function("lru_touch_hotset", |b| {
        b.iter_batched(
            || {
                let mut cache = LRUCore::new(4096);
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
}

criterion_group!(
    benches,
    bench_lru_insert_get,
    bench_lru_pop_lru,
    bench_lru_eviction_churn,
    bench_lru_touch_hotset
);
criterion_main!(benches);
