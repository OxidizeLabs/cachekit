//! Micro-operation benchmarks for all cache policies.
//!
//! Run with: `cargo bench --bench ops`
//!
//! Measures per-operation latency (nanoseconds) for get and insert operations
//! across all cache policies under identical conditions.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use cachekit::policy::clock::ClockCache;
use cachekit::policy::clock_pro::ClockProCache;
use cachekit::policy::fifo::FifoCache;
use cachekit::policy::heap_lfu::HeapLfuCache;
use cachekit::policy::lfu::LfuCache;
use cachekit::policy::lru::LruCore;
use cachekit::policy::lru_k::LrukCache;
use cachekit::policy::s3_fifo::S3FifoCache;
use cachekit::policy::two_q::TwoQCore;
use cachekit::traits::CoreCache;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

const CAPACITY: usize = 16_384;
const OPS: u64 = 100_000;

// ============================================================================
// Get Hit Latency (ns/op)
// ============================================================================

fn bench_get_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_hit_ns");
    group.throughput(Throughput::Elements(OPS));

    // LRU
    group.bench_function("lru", |b| {
        b.iter_custom(|iters| {
            let mut cache = LruCore::new(CAPACITY);
            for i in 0..CAPACITY as u64 {
                cache.insert(i, Arc::new(i));
            }
            let start = Instant::now();
            for _ in 0..iters {
                for i in 0..OPS {
                    let key = i % (CAPACITY as u64);
                    black_box(cache.get(&key));
                }
            }
            start.elapsed()
        })
    });

    // LRU-K
    group.bench_function("lru_k", |b| {
        b.iter_custom(|iters| {
            let mut cache: LrukCache<u64, Arc<u64>> = LrukCache::new(CAPACITY);
            for i in 0..CAPACITY as u64 {
                cache.insert(i, Arc::new(i));
            }
            let start = Instant::now();
            for _ in 0..iters {
                for i in 0..OPS {
                    let key = i % (CAPACITY as u64);
                    black_box(cache.get(&key));
                }
            }
            start.elapsed()
        })
    });

    // LFU (wraps values in Arc)
    group.bench_function("lfu", |b| {
        b.iter_custom(|iters| {
            let mut cache: LfuCache<u64, u64> = LfuCache::new(CAPACITY);
            for i in 0..CAPACITY as u64 {
                cache.insert(i, Arc::new(i));
            }
            let start = Instant::now();
            for _ in 0..iters {
                for i in 0..OPS {
                    let key = i % (CAPACITY as u64);
                    black_box(cache.get(&key));
                }
            }
            start.elapsed()
        })
    });

    // Heap-LFU (wraps values in Arc)
    group.bench_function("heap_lfu", |b| {
        b.iter_custom(|iters| {
            let mut cache: HeapLfuCache<u64, u64> = HeapLfuCache::new(CAPACITY);
            for i in 0..CAPACITY as u64 {
                cache.insert(i, Arc::new(i));
            }
            let start = Instant::now();
            for _ in 0..iters {
                for i in 0..OPS {
                    let key = i % (CAPACITY as u64);
                    black_box(cache.get(&key));
                }
            }
            start.elapsed()
        })
    });

    // Clock
    group.bench_function("clock", |b| {
        b.iter_custom(|iters| {
            let mut cache: ClockCache<u64, u64> = ClockCache::new(CAPACITY);
            for i in 0..CAPACITY as u64 {
                cache.insert(i, i);
            }
            let start = Instant::now();
            for _ in 0..iters {
                for i in 0..OPS {
                    let key = i % (CAPACITY as u64);
                    black_box(cache.get(&key));
                }
            }
            start.elapsed()
        })
    });

    // Clock-PRO
    group.bench_function("clock_pro", |b| {
        b.iter_custom(|iters| {
            let mut cache: ClockProCache<u64, u64> = ClockProCache::new(CAPACITY);
            for i in 0..CAPACITY as u64 {
                cache.insert(i, i);
            }
            let start = Instant::now();
            for _ in 0..iters {
                for i in 0..OPS {
                    let key = i % (CAPACITY as u64);
                    black_box(cache.get(&key));
                }
            }
            start.elapsed()
        })
    });

    // S3-FIFO
    group.bench_function("s3_fifo", |b| {
        b.iter_custom(|iters| {
            let mut cache: S3FifoCache<u64, u64> = S3FifoCache::new(CAPACITY);
            for i in 0..CAPACITY as u64 {
                cache.insert(i, i);
            }
            let start = Instant::now();
            for _ in 0..iters {
                for i in 0..OPS {
                    let key = i % (CAPACITY as u64);
                    black_box(cache.get(&key));
                }
            }
            start.elapsed()
        })
    });

    // 2Q
    group.bench_function("two_q", |b| {
        b.iter_custom(|iters| {
            let mut cache: TwoQCore<u64, u64> = TwoQCore::new(CAPACITY, 0.25);
            for i in 0..CAPACITY as u64 {
                cache.insert(i, i);
            }
            let start = Instant::now();
            for _ in 0..iters {
                for i in 0..OPS {
                    let key = i % (CAPACITY as u64);
                    black_box(cache.get(&key));
                }
            }
            start.elapsed()
        })
    });

    // FIFO
    group.bench_function("fifo", |b| {
        b.iter_custom(|iters| {
            let mut cache: FifoCache<u64, u64> = FifoCache::new(CAPACITY);
            for i in 0..CAPACITY as u64 {
                cache.insert(i, i);
            }
            let start = Instant::now();
            for _ in 0..iters {
                for i in 0..OPS {
                    let key = i % (CAPACITY as u64);
                    black_box(cache.get(&key));
                }
            }
            start.elapsed()
        })
    });

    group.finish();
}

// ============================================================================
// Insert with Eviction (ns/op)
// ============================================================================

fn bench_insert_evict(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_evict_ns");
    group.throughput(Throughput::Elements(OPS));

    // LRU
    group.bench_function("lru", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut cache = LruCore::new(CAPACITY);
                for i in 0..CAPACITY as u64 {
                    cache.insert(i, Arc::new(i));
                }
                let start = Instant::now();
                for i in 0..OPS {
                    let key = CAPACITY as u64 + i;
                    cache.insert(key, Arc::new(key));
                }
                total += start.elapsed();
            }
            total
        })
    });

    // LRU-K
    group.bench_function("lru_k", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut cache: LrukCache<u64, u64> = LrukCache::new(CAPACITY);
                for i in 0..CAPACITY as u64 {
                    cache.insert(i, i);
                }
                let start = Instant::now();
                for i in 0..OPS {
                    let key = CAPACITY as u64 + i;
                    cache.insert(key, key);
                }
                total += start.elapsed();
            }
            total
        })
    });

    // LFU (wraps values in Arc)
    group.bench_function("lfu", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut cache: LfuCache<u64, u64> = LfuCache::new(CAPACITY);
                for i in 0..CAPACITY as u64 {
                    cache.insert(i, Arc::new(i));
                }
                let start = Instant::now();
                for i in 0..OPS {
                    let key = CAPACITY as u64 + i;
                    cache.insert(key, Arc::new(key));
                }
                total += start.elapsed();
            }
            total
        })
    });

    // Heap-LFU (wraps values in Arc)
    group.bench_function("heap_lfu", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut cache: HeapLfuCache<u64, u64> = HeapLfuCache::new(CAPACITY);
                for i in 0..CAPACITY as u64 {
                    cache.insert(i, Arc::new(i));
                }
                let start = Instant::now();
                for i in 0..OPS {
                    let key = CAPACITY as u64 + i;
                    cache.insert(key, Arc::new(key));
                }
                total += start.elapsed();
            }
            total
        })
    });

    // Clock
    group.bench_function("clock", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut cache: ClockCache<u64, u64> = ClockCache::new(CAPACITY);
                for i in 0..CAPACITY as u64 {
                    cache.insert(i, i);
                }
                let start = Instant::now();
                for i in 0..OPS {
                    let key = CAPACITY as u64 + i;
                    cache.insert(key, key);
                }
                total += start.elapsed();
            }
            total
        })
    });

    // Clock-PRO
    group.bench_function("clock_pro", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut cache: ClockProCache<u64, u64> = ClockProCache::new(CAPACITY);
                for i in 0..CAPACITY as u64 {
                    cache.insert(i, i);
                }
                let start = Instant::now();
                for i in 0..OPS {
                    let key = CAPACITY as u64 + i;
                    cache.insert(key, key);
                }
                total += start.elapsed();
            }
            total
        })
    });

    // S3-FIFO
    group.bench_function("s3_fifo", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut cache: S3FifoCache<u64, u64> = S3FifoCache::new(CAPACITY);
                for i in 0..CAPACITY as u64 {
                    cache.insert(i, i);
                }
                let start = Instant::now();
                for i in 0..OPS {
                    let key = CAPACITY as u64 + i;
                    cache.insert(key, key);
                }
                total += start.elapsed();
            }
            total
        })
    });

    // 2Q
    group.bench_function("two_q", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut cache: TwoQCore<u64, u64> = TwoQCore::new(CAPACITY, 0.25);
                for i in 0..CAPACITY as u64 {
                    cache.insert(i, i);
                }
                let start = Instant::now();
                for i in 0..OPS {
                    let key = CAPACITY as u64 + i;
                    cache.insert(key, key);
                }
                total += start.elapsed();
            }
            total
        })
    });

    // FIFO
    group.bench_function("fifo", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut cache: FifoCache<u64, u64> = FifoCache::new(CAPACITY);
                for i in 0..CAPACITY as u64 {
                    cache.insert(i, i);
                }
                let start = Instant::now();
                for i in 0..OPS {
                    let key = CAPACITY as u64 + i;
                    cache.insert(key, key);
                }
                total += start.elapsed();
            }
            total
        })
    });

    group.finish();
}

// ============================================================================
// Mixed Workload (get + insert)
// ============================================================================

fn bench_mixed(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_ops_ns");
    group.throughput(Throughput::Elements(OPS));

    // 80% hits, 20% misses causing inserts
    // LRU
    group.bench_function("lru", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut cache = LruCore::new(CAPACITY);
                for i in 0..CAPACITY as u64 {
                    cache.insert(i, Arc::new(i));
                }
                let start = Instant::now();
                for i in 0..OPS {
                    let key = if i % 5 == 0 {
                        CAPACITY as u64 + i
                    } else {
                        i % (CAPACITY as u64)
                    };
                    if cache.get(&key).is_none() {
                        cache.insert(key, Arc::new(key));
                    }
                }
                total += start.elapsed();
            }
            total
        })
    });

    // Clock
    group.bench_function("clock", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut cache: ClockCache<u64, u64> = ClockCache::new(CAPACITY);
                for i in 0..CAPACITY as u64 {
                    cache.insert(i, i);
                }
                let start = Instant::now();
                for i in 0..OPS {
                    let key = if i % 5 == 0 {
                        CAPACITY as u64 + i
                    } else {
                        i % (CAPACITY as u64)
                    };
                    if cache.get(&key).is_none() {
                        cache.insert(key, key);
                    }
                }
                total += start.elapsed();
            }
            total
        })
    });

    // S3-FIFO
    group.bench_function("s3_fifo", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut cache: S3FifoCache<u64, u64> = S3FifoCache::new(CAPACITY);
                for i in 0..CAPACITY as u64 {
                    cache.insert(i, i);
                }
                let start = Instant::now();
                for i in 0..OPS {
                    let key = if i % 5 == 0 {
                        CAPACITY as u64 + i
                    } else {
                        i % (CAPACITY as u64)
                    };
                    if cache.get(&key).is_none() {
                        cache.insert(key, key);
                    }
                }
                total += start.elapsed();
            }
            total
        })
    });

    // 2Q
    group.bench_function("two_q", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut cache: TwoQCore<u64, u64> = TwoQCore::new(CAPACITY, 0.25);
                for i in 0..CAPACITY as u64 {
                    cache.insert(i, i);
                }
                let start = Instant::now();
                for i in 0..OPS {
                    let key = if i % 5 == 0 {
                        CAPACITY as u64 + i
                    } else {
                        i % (CAPACITY as u64)
                    };
                    if cache.get(&key).is_none() {
                        cache.insert(key, key);
                    }
                }
                total += start.elapsed();
            }
            total
        })
    });

    group.finish();
}

criterion_group!(benches, bench_get_hit, bench_insert_evict, bench_mixed);
criterion_main!(benches);
