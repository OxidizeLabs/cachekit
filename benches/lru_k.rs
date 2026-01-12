mod common;

use cachekit::policy::lru_k::LRUKCache;
use cachekit::traits::{CoreCache, LRUKCacheTrait};
use common::workload::{Workload, WorkloadSpec, run_hit_rate};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use std::sync::Arc;
use std::time::Instant;

fn bench_lru_k_insert_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("lru_k_policy");
    let ops_per_iter = 1024u64 * 2;
    group.throughput(Throughput::Elements(ops_per_iter));
    group.bench_function("insert_get", |b| {
        b.iter_batched(
            || {
                let mut cache = LRUKCache::with_k(1024, 2);
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

fn bench_lru_k_pop_lru_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("lru_k_policy");
    group.throughput(Throughput::Elements(1024));
    group.bench_function("pop_lru_k", |b| {
        b.iter_batched(
            || {
                let mut cache = LRUKCache::with_k(1024, 2);
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

fn bench_lru_k_eviction_churn(c: &mut Criterion) {
    let mut group = c.benchmark_group("lru_k_policy");
    group.throughput(Throughput::Elements(4096));
    group.bench_function("eviction_churn", |b| {
        b.iter_batched(
            || {
                let mut cache = LRUKCache::with_k(1024, 2);
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

fn bench_lru_k_touch_hotset(c: &mut Criterion) {
    let mut group = c.benchmark_group("lru_k_policy");
    group.throughput(Throughput::Elements(4096));
    group.bench_function("touch_hotset", |b| {
        b.iter_batched(
            || {
                let mut cache = LRUKCache::with_k(4096, 2);
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

fn bench_lru_k_get_hit_ns(c: &mut Criterion) {
    c.bench_function("lru_k_get_hit_ns", |b| {
        b.iter_custom(|iters| {
            let capacity = 16_384u64;
            let mut cache = LRUKCache::with_k(capacity as usize, 2);
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

fn bench_lru_k_insert_full_ns(c: &mut Criterion) {
    c.bench_function("lru_k_insert_full_ns", |b| {
        b.iter_custom(|iters| {
            let capacity = 4096u64;
            let mut cache = LRUKCache::with_k(capacity as usize, 2);
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

fn bench_lru_k_workload_hit_rate(c: &mut Criterion) {
    let mut group = c.benchmark_group("lru_k_workload_hit_rate");
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
    ];

    for (name, workload) in specs {
        group.bench_function(name, |b| {
            b.iter_custom(|iters| {
                let mut total = std::time::Duration::default();
                for _ in 0..iters {
                    let mut cache = LRUKCache::with_k(4096, 2);
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
    bench_lru_k_insert_get,
    bench_lru_k_eviction_churn
);
criterion_group!(
    policy_level,
    bench_lru_k_pop_lru_k,
    bench_lru_k_touch_hotset
);
criterion_group!(
    micro_ops,
    bench_lru_k_get_hit_ns,
    bench_lru_k_insert_full_ns
);
criterion_group!(workloads, bench_lru_k_workload_hit_rate);
criterion_main!(end_to_end, policy_level, micro_ops, workloads);
