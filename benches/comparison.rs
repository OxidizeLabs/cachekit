//! Comparison benchmarks against other Rust cache libraries.
//!
//! Compares cachekit's LRU implementations against:
//! - `lru` crate (popular, simple LRU)
//! - `quick_cache` crate (high-performance)

use std::hint::black_box;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Instant;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use cachekit::policy::clock::ClockCache;
use cachekit::policy::clock_pro::ClockProCache;
use cachekit::policy::fast_lru::FastLru;
use cachekit::policy::lru::LruCore;
use cachekit::policy::lru_k::LrukCache;
use cachekit::policy::s3_fifo::S3FifoCache;
use cachekit::policy::two_q::TwoQCore;
use cachekit::traits::CoreCache;

const CAPACITY: usize = 4096;
const OPS: u64 = 100_000;

// =============================================================================
// Get (cache hit) benchmarks
// =============================================================================

fn bench_get_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_hit");
    group.throughput(Throughput::Elements(OPS));

    // cachekit Clock (fastest - no linked list)
    group.bench_function("cachekit_clock", |b| {
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

    // cachekit Clock-PRO (scan-resistant Clock)
    group.bench_function("cachekit_clock_pro", |b| {
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

    // cachekit FastLru (optimized)
    group.bench_function("cachekit_fast", |b| {
        b.iter_custom(|iters| {
            let mut cache = FastLru::new(CAPACITY);
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

    // cachekit LruCore (Arc-based)
    group.bench_function("cachekit_lru", |b| {
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

    // cachekit LRU-K (scan-resistant)
    group.bench_function("cachekit_lru_k", |b| {
        b.iter_custom(|iters| {
            let mut cache: LrukCache<u64, u64> = LrukCache::new(CAPACITY);
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

    // cachekit 2Q (scan-resistant)
    group.bench_function("cachekit_2q", |b| {
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

    // cachekit S3-FIFO (scan-resistant FIFO)
    group.bench_function("cachekit_s3_fifo", |b| {
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

    // lru crate
    group.bench_function("lru_crate", |b| {
        b.iter_custom(|iters| {
            let mut cache = lru::LruCache::new(NonZeroUsize::new(CAPACITY).unwrap());
            for i in 0..CAPACITY as u64 {
                cache.put(i, i);
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

    // quick_cache
    group.bench_function("quick_cache", |b| {
        b.iter_custom(|iters| {
            let mut cache = quick_cache::unsync::Cache::new(CAPACITY);
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

// =============================================================================
// Insert (into full cache, causing eviction) benchmarks
// =============================================================================

fn bench_insert_evict(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_evict");
    group.throughput(Throughput::Elements(OPS));

    // cachekit Clock (fastest - no linked list)
    group.bench_function("cachekit_clock", |b| {
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

    // cachekit Clock-PRO (scan-resistant Clock)
    group.bench_function("cachekit_clock_pro", |b| {
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

    // cachekit FastLru (optimized)
    group.bench_function("cachekit_fast", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut cache = FastLru::new(CAPACITY);
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

    // cachekit LruCore (Arc-based)
    group.bench_function("cachekit_lru", |b| {
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

    // cachekit LRU-K (scan-resistant)
    group.bench_function("cachekit_lru_k", |b| {
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

    // cachekit 2Q (scan-resistant)
    group.bench_function("cachekit_2q", |b| {
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

    // cachekit S3-FIFO (scan-resistant FIFO)
    group.bench_function("cachekit_s3_fifo", |b| {
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

    // lru crate
    group.bench_function("lru_crate", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut cache = lru::LruCache::new(NonZeroUsize::new(CAPACITY).unwrap());
                for i in 0..CAPACITY as u64 {
                    cache.put(i, i);
                }
                let start = Instant::now();
                for i in 0..OPS {
                    let key = CAPACITY as u64 + i;
                    cache.put(key, key);
                }
                total += start.elapsed();
            }
            total
        })
    });

    // quick_cache
    group.bench_function("quick_cache", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut cache = quick_cache::unsync::Cache::new(CAPACITY);
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

// =============================================================================
// Mixed workload (get + insert, simulating real usage)
// =============================================================================

fn bench_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_workload");
    group.throughput(Throughput::Elements(OPS));

    // 80% hits, 20% misses causing inserts
    // cachekit Clock (fastest - no linked list)
    group.bench_function("cachekit_clock", |b| {
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
                        // 20% miss - insert new key
                        CAPACITY as u64 + i
                    } else {
                        // 80% hit
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

    // cachekit Clock-PRO (scan-resistant Clock)
    group.bench_function("cachekit_clock_pro", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut cache: ClockProCache<u64, u64> = ClockProCache::new(CAPACITY);
                for i in 0..CAPACITY as u64 {
                    cache.insert(i, i);
                }
                let start = Instant::now();
                for i in 0..OPS {
                    let key = if i % 5 == 0 {
                        // 20% miss - insert new key
                        CAPACITY as u64 + i
                    } else {
                        // 80% hit
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

    // cachekit FastLru (optimized)
    group.bench_function("cachekit_fast", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut cache = FastLru::new(CAPACITY);
                for i in 0..CAPACITY as u64 {
                    cache.insert(i, i);
                }
                let start = Instant::now();
                for i in 0..OPS {
                    let key = if i % 5 == 0 {
                        // 20% miss - insert new key
                        CAPACITY as u64 + i
                    } else {
                        // 80% hit
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

    // cachekit LruCore (Arc-based)
    group.bench_function("cachekit_lru", |b| {
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
                        // 20% miss - insert new key
                        CAPACITY as u64 + i
                    } else {
                        // 80% hit
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

    // cachekit LRU-K (scan-resistant)
    group.bench_function("cachekit_lru_k", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut cache: LrukCache<u64, u64> = LrukCache::new(CAPACITY);
                for i in 0..CAPACITY as u64 {
                    cache.insert(i, i);
                }
                let start = Instant::now();
                for i in 0..OPS {
                    let key = if i % 5 == 0 {
                        // 20% miss - insert new key
                        CAPACITY as u64 + i
                    } else {
                        // 80% hit
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

    // cachekit 2Q (scan-resistant)
    group.bench_function("cachekit_2q", |b| {
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
                        // 20% miss - insert new key
                        CAPACITY as u64 + i
                    } else {
                        // 80% hit
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

    // cachekit S3-FIFO (scan-resistant FIFO)
    group.bench_function("cachekit_s3_fifo", |b| {
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
                        // 20% miss - insert new key
                        CAPACITY as u64 + i
                    } else {
                        // 80% hit
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

    // lru crate
    group.bench_function("lru_crate", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut cache = lru::LruCache::new(NonZeroUsize::new(CAPACITY).unwrap());
                for i in 0..CAPACITY as u64 {
                    cache.put(i, i);
                }
                let start = Instant::now();
                for i in 0..OPS {
                    let key = if i % 5 == 0 {
                        CAPACITY as u64 + i
                    } else {
                        i % (CAPACITY as u64)
                    };
                    if cache.get(&key).is_none() {
                        cache.put(key, key);
                    }
                }
                total += start.elapsed();
            }
            total
        })
    });

    // quick_cache
    group.bench_function("quick_cache", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut cache = quick_cache::unsync::Cache::new(CAPACITY);
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

// =============================================================================
// Scaling benchmark (different cache sizes)
// =============================================================================

fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_get");

    for size in [1024, 4096, 16384, 65536] {
        let ops = size as u64;

        group.throughput(Throughput::Elements(ops));

        group.bench_with_input(
            BenchmarkId::new("cachekit_clock", size),
            &size,
            |b, &size| {
                b.iter_custom(|iters| {
                    let mut cache: ClockCache<u64, u64> = ClockCache::new(size);
                    for i in 0..size as u64 {
                        cache.insert(i, i);
                    }
                    let start = Instant::now();
                    for _ in 0..iters {
                        for i in 0..ops {
                            let key = i % (size as u64);
                            black_box(cache.get(&key));
                        }
                    }
                    start.elapsed()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cachekit_clock_pro", size),
            &size,
            |b, &size| {
                b.iter_custom(|iters| {
                    let mut cache: ClockProCache<u64, u64> = ClockProCache::new(size);
                    for i in 0..size as u64 {
                        cache.insert(i, i);
                    }
                    let start = Instant::now();
                    for _ in 0..iters {
                        for i in 0..ops {
                            let key = i % (size as u64);
                            black_box(cache.get(&key));
                        }
                    }
                    start.elapsed()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cachekit_fast", size),
            &size,
            |b, &size| {
                b.iter_custom(|iters| {
                    let mut cache = FastLru::new(size);
                    for i in 0..size as u64 {
                        cache.insert(i, i);
                    }
                    let start = Instant::now();
                    for _ in 0..iters {
                        for i in 0..ops {
                            let key = i % (size as u64);
                            black_box(cache.get(&key));
                        }
                    }
                    start.elapsed()
                })
            },
        );

        group.bench_with_input(BenchmarkId::new("cachekit_lru", size), &size, |b, &size| {
            b.iter_custom(|iters| {
                let mut cache = LruCore::new(size);
                for i in 0..size as u64 {
                    cache.insert(i, Arc::new(i));
                }
                let start = Instant::now();
                for _ in 0..iters {
                    for i in 0..ops {
                        let key = i % (size as u64);
                        black_box(cache.get(&key));
                    }
                }
                start.elapsed()
            })
        });

        group.bench_with_input(
            BenchmarkId::new("cachekit_lru_k", size),
            &size,
            |b, &size| {
                b.iter_custom(|iters| {
                    let mut cache: LrukCache<u64, u64> = LrukCache::new(size);
                    for i in 0..size as u64 {
                        cache.insert(i, i);
                    }
                    let start = Instant::now();
                    for _ in 0..iters {
                        for i in 0..ops {
                            let key = i % (size as u64);
                            black_box(cache.get(&key));
                        }
                    }
                    start.elapsed()
                })
            },
        );

        group.bench_with_input(BenchmarkId::new("cachekit_2q", size), &size, |b, &size| {
            b.iter_custom(|iters| {
                let mut cache: TwoQCore<u64, u64> = TwoQCore::new(size, 0.25);
                for i in 0..size as u64 {
                    cache.insert(i, i);
                }
                let start = Instant::now();
                for _ in 0..iters {
                    for i in 0..ops {
                        let key = i % (size as u64);
                        black_box(cache.get(&key));
                    }
                }
                start.elapsed()
            })
        });

        group.bench_with_input(
            BenchmarkId::new("cachekit_s3_fifo", size),
            &size,
            |b, &size| {
                b.iter_custom(|iters| {
                    let mut cache: S3FifoCache<u64, u64> = S3FifoCache::new(size);
                    for i in 0..size as u64 {
                        cache.insert(i, i);
                    }
                    let start = Instant::now();
                    for _ in 0..iters {
                        for i in 0..ops {
                            let key = i % (size as u64);
                            black_box(cache.get(&key));
                        }
                    }
                    start.elapsed()
                })
            },
        );

        group.bench_with_input(BenchmarkId::new("lru_crate", size), &size, |b, &size| {
            b.iter_custom(|iters| {
                let mut cache = lru::LruCache::new(NonZeroUsize::new(size).unwrap());
                for i in 0..size as u64 {
                    cache.put(i, i);
                }
                let start = Instant::now();
                for _ in 0..iters {
                    for i in 0..ops {
                        let key = i % (size as u64);
                        black_box(cache.get(&key));
                    }
                }
                start.elapsed()
            })
        });

        group.bench_with_input(BenchmarkId::new("quick_cache", size), &size, |b, &size| {
            b.iter_custom(|iters| {
                let mut cache = quick_cache::unsync::Cache::new(size);
                for i in 0..size as u64 {
                    cache.insert(i, i);
                }
                let start = Instant::now();
                for _ in 0..iters {
                    for i in 0..ops {
                        let key = i % (size as u64);
                        black_box(cache.get(&key));
                    }
                }
                start.elapsed()
            })
        });
    }

    group.finish();
}

criterion_group!(
    comparison,
    bench_get_hit,
    bench_insert_evict,
    bench_mixed_workload,
    bench_scaling,
);
criterion_main!(comparison);
