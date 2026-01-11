use cachekit::policy::lfu::LFUCache;
use cachekit::traits::{CoreCache, LFUCacheTrait};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use std::sync::Arc;

fn bench_lfu_insert_get_end_to_end(c: &mut Criterion) {
    c.bench_function("lfu_insert_get_end_to_end", |b| {
        b.iter_batched(
            || {
                let mut cache = LFUCache::new(1024);
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

fn bench_lfu_insert_get_policy(c: &mut Criterion) {
    c.bench_function("lfu_insert_get_policy", |b| {
        b.iter_batched(
            || {
                let mut cache = LFUCache::new(1024);
                for i in 0..1024u64 {
                    cache.insert(i, Arc::new(i));
                }
                let values: Vec<_> = (0..1024u64).map(Arc::new).collect();
                (cache, values)
            },
            |(mut cache, values)| {
                for i in 0..1024u64 {
                    cache.insert(std::hint::black_box(i + 10_000), values[i as usize].clone());
                    let _ = std::hint::black_box(cache.get(&std::hint::black_box(i)));
                }
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_lfu_get_hotset_policy(c: &mut Criterion) {
    c.bench_function("lfu_get_hotset_policy", |b| {
        b.iter_batched(
            || {
                let mut cache = LFUCache::new(4096);
                for i in 0..4096u64 {
                    cache.insert(i, Arc::new(i));
                }
                cache
            },
            |mut cache| {
                for i in 0..4096u64 {
                    let _ = std::hint::black_box(cache.get(&std::hint::black_box(i)));
                }
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_lfu_eviction_churn_end_to_end(c: &mut Criterion) {
    c.bench_function("lfu_eviction_churn_end_to_end", |b| {
        b.iter_batched(
            || {
                let mut cache = LFUCache::new(1024);
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

fn bench_lfu_eviction_churn_policy(c: &mut Criterion) {
    c.bench_function("lfu_eviction_churn_policy", |b| {
        b.iter_batched(
            || {
                let mut cache = LFUCache::new(1024);
                for i in 0..1024u64 {
                    cache.insert(i, Arc::new(i));
                }
                let values: Vec<_> = (0..4096u64).map(Arc::new).collect();
                (cache, values)
            },
            |(mut cache, values)| {
                for i in 0..4096u64 {
                    cache.insert(std::hint::black_box(10_000 + i), values[i as usize].clone());
                }
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_lfu_eviction_churn_policy_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("lfu_eviction_churn_policy_sizes");
    for &capacity in &[256usize, 1024, 4096, 16384] {
        let inserts = capacity * 4;
        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(capacity),
            &capacity,
            |b, &capacity| {
                b.iter_batched(
                    || {
                        let mut cache = LFUCache::new(capacity);
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

fn bench_lfu_frequency_updates(c: &mut Criterion) {
    c.bench_function("lfu_frequency_updates", |b| {
        b.iter_batched(
            || {
                let mut cache = LFUCache::new(4096);
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
}

fn bench_lfu_pop_lfu_policy(c: &mut Criterion) {
    c.bench_function("lfu_pop_lfu_policy", |b| {
        b.iter_batched(
            || {
                let mut cache = LFUCache::new(1024);
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
}

criterion_group!(
    benches,
    bench_lfu_insert_get_end_to_end,
    bench_lfu_insert_get_policy,
    bench_lfu_get_hotset_policy,
    bench_lfu_eviction_churn_end_to_end,
    bench_lfu_eviction_churn_policy,
    bench_lfu_eviction_churn_policy_sizes,
    bench_lfu_frequency_updates,
    bench_lfu_pop_lfu_policy
);
criterion_main!(benches);
