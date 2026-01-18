mod common;

use std::sync::Arc;
use std::time::Instant;

use cachekit::policy::s3_fifo::S3FifoCache;
use common::workload::{Workload, WorkloadSpec, run_hit_rate};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

fn bench_s3_fifo_insert_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("s3_fifo_policy");
    let ops_per_iter = 1024u64 * 2;
    group.throughput(Throughput::Elements(ops_per_iter));
    group.bench_function("insert_get", |b| {
        b.iter_batched(
            || {
                let mut cache = S3FifoCache::new(1024);
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

fn bench_s3_fifo_eviction_churn(c: &mut Criterion) {
    let mut group = c.benchmark_group("s3_fifo_policy");
    group.throughput(Throughput::Elements(4096));
    group.bench_function("eviction_churn", |b| {
        b.iter_batched(
            || {
                let mut cache = S3FifoCache::new(1024);
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

fn bench_s3_fifo_get_hot(c: &mut Criterion) {
    let mut group = c.benchmark_group("s3_fifo_policy");
    group.throughput(Throughput::Elements(4096));
    group.bench_function("get_hot", |b| {
        b.iter_batched(
            || {
                let mut cache = S3FifoCache::new(4096);
                for i in 0..4096u64 {
                    cache.insert(i, Arc::new(i));
                }
                // Access items to increase their frequency
                for i in 0..4096u64 {
                    cache.get(&i);
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

fn bench_s3_fifo_scan_resistance(c: &mut Criterion) {
    let mut group = c.benchmark_group("s3_fifo_policy");
    group.throughput(Throughput::Elements(8192));
    group.bench_function("scan_resistance", |b| {
        b.iter_batched(
            || {
                let mut cache = S3FifoCache::new(1024);
                // Create hot set with multiple accesses
                for i in 0..512u64 {
                    cache.insert(i, Arc::new(i));
                    cache.get(&i);
                    cache.get(&i);
                }
                cache
            },
            |mut cache| {
                // Scan workload (one-time accesses)
                for i in 0..8192u64 {
                    cache.insert(std::hint::black_box(10_000 + i), Arc::new(i));
                }
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn bench_s3_fifo_get_hit_ns(c: &mut Criterion) {
    c.bench_function("s3_fifo_get_hit_ns", |b| {
        b.iter_custom(|iters| {
            let capacity = 16_384u64;
            let mut cache = S3FifoCache::new(capacity as usize);
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

fn bench_s3_fifo_insert_full_ns(c: &mut Criterion) {
    c.bench_function("s3_fifo_insert_full_ns", |b| {
        b.iter_custom(|iters| {
            let capacity = 4096u64;
            let mut cache = S3FifoCache::new(capacity as usize);
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

fn bench_s3_fifo_workload_hit_rate(c: &mut Criterion) {
    let mut group = c.benchmark_group("s3_fifo_workload_hit_rate");
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
        ("zipfian", Workload::Zipfian { exponent: 1.0 }),
    ];

    for (name, workload) in specs {
        group.bench_function(name, |b| {
            b.iter_custom(|iters| {
                let mut total = std::time::Duration::default();
                for _ in 0..iters {
                    let mut cache = S3FifoCache::new(4096);
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
    bench_s3_fifo_insert_get,
    bench_s3_fifo_eviction_churn
);
criterion_group!(
    policy_level,
    bench_s3_fifo_get_hot,
    bench_s3_fifo_scan_resistance
);
criterion_group!(
    micro_ops,
    bench_s3_fifo_get_hit_ns,
    bench_s3_fifo_insert_full_ns
);
criterion_group!(workloads, bench_s3_fifo_workload_hit_rate);
criterion_main!(end_to_end, policy_level, micro_ops, workloads);
