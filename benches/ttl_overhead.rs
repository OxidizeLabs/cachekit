//! Overhead of the TTL decorator versus a bare LRU cache.
//!
//! Three workload groups (zipfian-read, scan + point lookup, mixed
//! read/write) compare:
//!
//! - `LruCore<u64, u64>` — the policy as it exists today.
//! - `Expiring<LruCore<u64, u64>, u64, Arc<u64>, MockClock>` with a
//!   60-second default TTL — long enough that the bench never fires an
//!   expiry but the decorator still does its bookkeeping on every op.
//!
//! Run with: `cargo bench --bench ttl_overhead --features ttl`.

#![cfg(feature = "ttl")]

use std::sync::Arc;
use std::time::Duration;

use bench_support::workload::{Workload, WorkloadSpec};
use cachekit::policy::expiring::Expiring;
use cachekit::policy::lru::LruCore;
use cachekit::time::MockClock;
use cachekit::traits::Cache;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

const CAPACITY: usize = 8_192;
const OPS: u64 = 4_096;
const UNIVERSE: u64 = 16_384;
const SEED: u64 = 0xc0fee;
const DEFAULT_TTL: Duration = Duration::from_secs(60);

fn fresh_lru() -> LruCore<u64, u64> {
    let mut cache = LruCore::new(CAPACITY);
    for i in 0..CAPACITY as u64 {
        cache.insert(i, Arc::new(i));
    }
    cache
}

fn fresh_expiring() -> Expiring<LruCore<u64, u64>, u64, Arc<u64>, MockClock> {
    let inner = fresh_lru();
    Expiring::with_default_ttl(inner, MockClock::new(), Some(DEFAULT_TTL))
}

fn make_generator(workload: Workload) -> bench_support::workload::WorkloadGenerator {
    WorkloadSpec {
        universe: UNIVERSE,
        workload,
        seed: SEED,
    }
    .generator()
}

// ---------------------------------------------------------------------------
// Group: zipfian-read (skewed get-only workload)
// ---------------------------------------------------------------------------

fn bench_zipfian_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("ttl_overhead/zipfian_read");
    group.throughput(Throughput::Elements(OPS));

    group.bench_function("plain_lru", |b| {
        b.iter_batched(
            fresh_lru,
            |mut cache| {
                let mut wl = make_generator(Workload::Zipfian { exponent: 1.1 });
                for _ in 0..OPS {
                    let _ = std::hint::black_box(cache.get(&wl.next_key()));
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("expiring_lru", |b| {
        b.iter_batched(
            fresh_expiring,
            |mut cache| {
                let mut wl = make_generator(Workload::Zipfian { exponent: 1.1 });
                for _ in 0..OPS {
                    let _ = std::hint::black_box(cache.get(&wl.next_key()));
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Group: scan + point lookup (mixed sequential scan and skewed lookups)
// ---------------------------------------------------------------------------

fn bench_scan_plus_point(c: &mut Criterion) {
    let workload = Workload::ScanResistance {
        scan_start_prob: 0.05,
        scan_length: 32,
        point_exponent: 1.0,
    };
    let mut group = c.benchmark_group("ttl_overhead/scan_plus_point");
    group.throughput(Throughput::Elements(OPS));

    group.bench_function("plain_lru", |b| {
        b.iter_batched(
            fresh_lru,
            |mut cache| {
                let mut wl = make_generator(workload);
                for _ in 0..OPS {
                    let _ = std::hint::black_box(cache.get(&wl.next_key()));
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("expiring_lru", |b| {
        b.iter_batched(
            fresh_expiring,
            |mut cache| {
                let mut wl = make_generator(workload);
                for _ in 0..OPS {
                    let _ = std::hint::black_box(cache.get(&wl.next_key()));
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Group: mixed read/write (90% read, 10% insert, both zipfian)
// ---------------------------------------------------------------------------

fn bench_mixed_rw(c: &mut Criterion) {
    let mut group = c.benchmark_group("ttl_overhead/mixed_rw");
    group.throughput(Throughput::Elements(OPS));

    group.bench_function("plain_lru", |b| {
        b.iter_batched(
            fresh_lru,
            |mut cache| {
                let mut wl = make_generator(Workload::Zipfian { exponent: 1.0 });
                for i in 0..OPS {
                    let key = wl.next_key();
                    if i % 10 == 0 {
                        cache.insert(key, Arc::new(key));
                    } else {
                        let _ = std::hint::black_box(cache.get(&key));
                    }
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("expiring_lru", |b| {
        b.iter_batched(
            fresh_expiring,
            |mut cache| {
                let mut wl = make_generator(Workload::Zipfian { exponent: 1.0 });
                for i in 0..OPS {
                    let key = wl.next_key();
                    if i % 10 == 0 {
                        cache.insert(key, Arc::new(key));
                    } else {
                        let _ = std::hint::black_box(cache.get(&key));
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
    bench_zipfian_read,
    bench_scan_plus_point,
    bench_mixed_rw
);
criterion_main!(benches);
