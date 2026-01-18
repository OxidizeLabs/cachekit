mod common;

use std::sync::Arc;
use std::time::Instant;

use cachekit::ds::FrequencyBucketsHandle;
use cachekit::policy::lfu::LfuCache;
use cachekit::traits::{CoreCache, LfuCacheTrait};
use common::workload::{Workload, WorkloadSpec, run_hit_rate};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

fn bench_lfu_insert_get_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("lfu_end_to_end");
    let ops_per_iter = 1024u64 * 2;
    group.throughput(Throughput::Elements(ops_per_iter));
    group.bench_function("insert_get", |b| {
        b.iter_batched(
            || {
                let mut cache = LfuCache::new(1024);
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

fn bench_lfu_insert_get_policy(c: &mut Criterion) {
    let mut group = c.benchmark_group("lfu_policy");
    let ops_per_iter = 1024u64 * 2;
    group.throughput(Throughput::Elements(ops_per_iter));
    group.bench_function("insert_get", |b| {
        b.iter_batched(
            || {
                let mut cache = LfuCache::new(1024);
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
    group.finish();
}

fn bench_lfu_get_hotset_policy(c: &mut Criterion) {
    let mut group = c.benchmark_group("lfu_policy");
    group.throughput(Throughput::Elements(4096));
    group.bench_function("get_hotset", |b| {
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
                    let _ = std::hint::black_box(cache.get(&std::hint::black_box(i)));
                }
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn bench_lfu_eviction_churn_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("lfu_end_to_end");
    group.throughput(Throughput::Elements(4096));
    group.bench_function("eviction_churn", |b| {
        b.iter_batched(
            || {
                let mut cache = LfuCache::new(1024);
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

fn bench_lfu_eviction_churn_policy(c: &mut Criterion) {
    let mut group = c.benchmark_group("lfu_policy");
    group.throughput(Throughput::Elements(4096));
    group.bench_function("eviction_churn", |b| {
        b.iter_batched(
            || {
                let mut cache = LfuCache::new(1024);
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
    group.finish();
}

fn bench_lfu_eviction_churn_policy_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("lfu_eviction_churn_policy_sizes");
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

fn bench_lfu_frequency_updates(c: &mut Criterion) {
    c.bench_function("lfu_frequency_updates", |b| {
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
}

fn bench_lfu_pop_lfu_policy(c: &mut Criterion) {
    c.bench_function("lfu_pop_lfu_policy", |b| {
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
}

fn bench_lfu_get_hit_ns(c: &mut Criterion) {
    c.bench_function("lfu_get_hit_ns", |b| {
        b.iter_custom(|iters| {
            let capacity = 16_384u64;
            let mut cache = LfuCache::new(capacity as usize);
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

fn bench_lfu_insert_full_ns(c: &mut Criterion) {
    c.bench_function("lfu_insert_full_ns", |b| {
        b.iter_custom(|iters| {
            let capacity = 4096u64;
            let mut cache = LfuCache::new(capacity as usize);
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

fn bench_lfu_policy_only_touch_ns(c: &mut Criterion) {
    c.bench_function("lfu_policy_only_touch_ns", |b| {
        b.iter_custom(|iters| {
            let capacity = 16_384u64;
            let mut buckets = FrequencyBucketsHandle::new();
            for i in 0..capacity {
                buckets.insert(i);
            }
            let start = Instant::now();
            for (idx, _) in (0..iters).enumerate() {
                let handle = (idx as u64) % capacity;
                let _ = std::hint::black_box(buckets.touch(&handle));
            }
            start.elapsed()
        })
    });
}

fn bench_lfu_workload_hit_rate(c: &mut Criterion) {
    let mut group = c.benchmark_group("lfu_workload_hit_rate");
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
                    let mut cache = LfuCache::new(4096);
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
    bench_lfu_insert_get_end_to_end,
    bench_lfu_eviction_churn_end_to_end
);
criterion_group!(
    policy_level,
    bench_lfu_insert_get_policy,
    bench_lfu_get_hotset_policy,
    bench_lfu_eviction_churn_policy,
    bench_lfu_eviction_churn_policy_sizes,
    bench_lfu_frequency_updates,
    bench_lfu_pop_lfu_policy
);
criterion_group!(
    micro_ops,
    bench_lfu_get_hit_ns,
    bench_lfu_insert_full_ns,
    bench_lfu_policy_only_touch_ns
);
criterion_group!(workloads, bench_lfu_workload_hit_rate);
criterion_main!(end_to_end, policy_level, micro_ops, workloads);
