//! S3-FIFO specific benchmarks for operations unique to S3-FIFO policy.
//!
//! Run with: `cargo bench --bench policy_s3_fifo`

use std::sync::Arc;

use cachekit::policy::s3_fifo::S3FifoCache;
#[allow(unused_imports)]
use cachekit::traits::CoreCache;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

// ============================================================================
// S3-FIFO specific operations
// ============================================================================

/// Benchmark scan resistance - S3-FIFO's key differentiator.
/// Measures how well the cache protects hot items during a sequential scan.
fn bench_scan_resistance(c: &mut Criterion) {
    let mut group = c.benchmark_group("s3_fifo_specific");
    group.throughput(Throughput::Elements(8192));

    group.bench_function("scan_resistance", |b| {
        b.iter_batched(
            || {
                let mut cache = S3FifoCache::new(1024);
                // Create hot set with multiple accesses (promotes to main queue)
                for i in 0..512u64 {
                    cache.insert(i, Arc::new(i));
                    cache.get(&i); // Second access promotes
                    cache.get(&i); // Third access increases frequency
                }
                cache
            },
            |mut cache| {
                // Scan workload - sequential one-time accesses
                // Should not evict hot items in main queue
                for i in 0..8192u64 {
                    cache.insert(std::hint::black_box(10_000 + i), Arc::new(i));
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

/// Benchmark promotion from small to main queue.
fn bench_promotion(c: &mut Criterion) {
    let mut group = c.benchmark_group("s3_fifo_specific");
    group.throughput(Throughput::Elements(4096));

    group.bench_function("small_to_main_promotion", |b| {
        b.iter_batched(
            || {
                let mut cache = S3FifoCache::new(4096);
                // Fill cache to capacity
                for i in 0..4096u64 {
                    cache.insert(i, Arc::new(i));
                }
                cache
            },
            |mut cache| {
                // Access all items - first access sets frequency bit
                // Items in small queue get promoted on eviction if freq > 0
                for i in 0..4096u64 {
                    let _ = std::hint::black_box(cache.get(&std::hint::black_box(i)));
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

/// Benchmark ghost queue behavior (tracking recently evicted).
fn bench_with_churn(c: &mut Criterion) {
    let mut group = c.benchmark_group("s3_fifo_specific");
    let capacity = 1024;
    let churn = capacity * 8;
    group.throughput(Throughput::Elements(churn as u64));

    group.bench_function("high_churn", |b| {
        b.iter_batched(
            || S3FifoCache::new(capacity),
            |mut cache| {
                // High churn with some repeated accesses
                for i in 0..churn as u64 {
                    cache.insert(std::hint::black_box(i), Arc::new(i));
                    if i % 4 == 0 && i >= capacity as u64 {
                        // Access some recent items
                        let _ = cache.get(&std::hint::black_box(i - 100));
                    }
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_scan_resistance,
    bench_promotion,
    bench_with_churn
);
criterion_main!(benches);
