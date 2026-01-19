//! Workload benchmarks - the single source of truth for policy comparison.
//!
//! Run with: `cargo bench --bench workloads`
//!
//! This benchmark compares ALL cache policies under identical workloads:
//! - Hit rate under various access patterns (uniform, zipfian, scan, etc.)
//! - Scan resistance (baseline → scan → recovery)
//! - Adaptation speed (workload shift response)
//! - Comprehensive metrics (latency, throughput, eviction stats)
//!
//! For micro-ops (get/insert latency), see: `cargo bench --bench ops`
//! For policy-specific operations, see: `cargo bench --bench policy_*`
//! For external crate comparison, see: `cargo bench --bench comparison`

mod common;

use std::sync::Arc;
use std::time::Instant;

use cachekit::policy::clock::ClockCache;
use cachekit::policy::heap_lfu::HeapLfuCache;
use cachekit::policy::lfu::LfuCache;
use cachekit::policy::lru::LruCore;
use cachekit::policy::lru_k::LrukCache;
use cachekit::policy::s3_fifo::S3FifoCache;
use cachekit::policy::two_q::TwoQCore;
use common::metrics::{
    BenchmarkConfig, measure_adaptation_speed, measure_scan_resistance, run_benchmark,
    standard_workload_suite,
};
use common::workload::{Workload, WorkloadSpec, run_hit_rate};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

const CAPACITY: usize = 4096;
const UNIVERSE: u64 = 16_384;
const OPS: usize = 200_000;
const SEED: u64 = 42;

// ============================================================================
// Workload definitions
// ============================================================================

/// Workloads that differentiate policies (scan-heavy, skewed, etc.)
fn workloads() -> Vec<(&'static str, Workload)> {
    vec![
        ("uniform", Workload::Uniform),
        (
            "hotset_90_10",
            Workload::HotSet {
                hot_fraction: 0.1,
                hot_prob: 0.9,
            },
        ),
        ("scan", Workload::Scan),
        ("zipfian_1.0", Workload::Zipfian { exponent: 1.0 }),
        (
            "scrambled_zipf",
            Workload::ScrambledZipfian { exponent: 1.0 },
        ),
        ("latest", Workload::Latest { exponent: 0.8 }),
        (
            "scan_resistance",
            Workload::ScanResistance {
                scan_fraction: 0.2,
                scan_length: 1000,
                point_exponent: 1.0,
            },
        ),
        (
            "flash_crowd",
            Workload::FlashCrowd {
                base_exponent: 1.0,
                flash_prob: 0.001,
                flash_duration: 1000,
                flash_keys: 10,
                flash_intensity: 100.0,
            },
        ),
    ]
}

// ============================================================================
// Hit Rate Benchmarks
// ============================================================================

fn bench_hit_rates(c: &mut Criterion) {
    let mut group = c.benchmark_group("hit_rate");
    group.throughput(Throughput::Elements(OPS as u64));

    for (workload_name, workload) in workloads() {
        // LRU
        group.bench_with_input(
            BenchmarkId::new("lru", workload_name),
            &workload,
            |b, &wl| {
                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::default();
                    for _ in 0..iters {
                        let mut cache = LruCore::new(CAPACITY);
                        let mut generator = WorkloadSpec {
                            universe: UNIVERSE,
                            workload: wl,
                            seed: SEED,
                        }
                        .generator();
                        let start = Instant::now();
                        let _ = run_hit_rate(&mut cache, &mut generator, OPS, Arc::new);
                        total += start.elapsed();
                    }
                    total
                });
            },
        );

        // LRU-K
        group.bench_with_input(
            BenchmarkId::new("lru_k", workload_name),
            &workload,
            |b, &wl| {
                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::default();
                    for _ in 0..iters {
                        let mut cache: LrukCache<u64, Arc<u64>> = LrukCache::new(CAPACITY);
                        let mut generator = WorkloadSpec {
                            universe: UNIVERSE,
                            workload: wl,
                            seed: SEED,
                        }
                        .generator();
                        let start = Instant::now();
                        let _ = run_hit_rate(&mut cache, &mut generator, OPS, Arc::new);
                        total += start.elapsed();
                    }
                    total
                });
            },
        );

        // LFU
        group.bench_with_input(
            BenchmarkId::new("lfu", workload_name),
            &workload,
            |b, &wl| {
                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::default();
                    for _ in 0..iters {
                        let mut cache: LfuCache<u64, u64> = LfuCache::new(CAPACITY);
                        let mut generator = WorkloadSpec {
                            universe: UNIVERSE,
                            workload: wl,
                            seed: SEED,
                        }
                        .generator();
                        let start = Instant::now();
                        let _ = run_hit_rate(&mut cache, &mut generator, OPS, Arc::new);
                        total += start.elapsed();
                    }
                    total
                });
            },
        );

        // Heap-LFU
        group.bench_with_input(
            BenchmarkId::new("heap_lfu", workload_name),
            &workload,
            |b, &wl| {
                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::default();
                    for _ in 0..iters {
                        let mut cache: HeapLfuCache<u64, u64> = HeapLfuCache::new(CAPACITY);
                        let mut generator = WorkloadSpec {
                            universe: UNIVERSE,
                            workload: wl,
                            seed: SEED,
                        }
                        .generator();
                        let start = Instant::now();
                        let _ = run_hit_rate(&mut cache, &mut generator, OPS, Arc::new);
                        total += start.elapsed();
                    }
                    total
                });
            },
        );

        // Clock
        group.bench_with_input(
            BenchmarkId::new("clock", workload_name),
            &workload,
            |b, &wl| {
                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::default();
                    for _ in 0..iters {
                        let mut cache: ClockCache<u64, Arc<u64>> = ClockCache::new(CAPACITY);
                        let mut generator = WorkloadSpec {
                            universe: UNIVERSE,
                            workload: wl,
                            seed: SEED,
                        }
                        .generator();
                        let start = Instant::now();
                        let _ = run_hit_rate(&mut cache, &mut generator, OPS, Arc::new);
                        total += start.elapsed();
                    }
                    total
                });
            },
        );

        // S3-FIFO
        group.bench_with_input(
            BenchmarkId::new("s3_fifo", workload_name),
            &workload,
            |b, &wl| {
                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::default();
                    for _ in 0..iters {
                        let mut cache: S3FifoCache<u64, Arc<u64>> = S3FifoCache::new(CAPACITY);
                        let mut generator = WorkloadSpec {
                            universe: UNIVERSE,
                            workload: wl,
                            seed: SEED,
                        }
                        .generator();
                        let start = Instant::now();
                        let _ = run_hit_rate(&mut cache, &mut generator, OPS, Arc::new);
                        total += start.elapsed();
                    }
                    total
                });
            },
        );

        // 2Q
        group.bench_with_input(
            BenchmarkId::new("two_q", workload_name),
            &workload,
            |b, &wl| {
                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::default();
                    for _ in 0..iters {
                        let mut cache: TwoQCore<u64, Arc<u64>> = TwoQCore::new(CAPACITY, 0.25);
                        let mut generator = WorkloadSpec {
                            universe: UNIVERSE,
                            workload: wl,
                            seed: SEED,
                        }
                        .generator();
                        let start = Instant::now();
                        let _ = run_hit_rate(&mut cache, &mut generator, OPS, Arc::new);
                        total += start.elapsed();
                    }
                    total
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Scan Resistance Benchmarks
// ============================================================================

fn bench_scan_resistance(c: &mut Criterion) {
    let mut group = c.benchmark_group("scan_resistance");

    group.bench_function("lru", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::default();
            for _ in 0..iters {
                let mut cache = LruCore::new(CAPACITY);
                let start = Instant::now();
                let _ = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
                total += start.elapsed();
            }
            total
        });
    });

    group.bench_function("lru_k", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::default();
            for _ in 0..iters {
                let mut cache: LrukCache<u64, Arc<u64>> = LrukCache::new(CAPACITY);
                let start = Instant::now();
                let _ = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
                total += start.elapsed();
            }
            total
        });
    });

    group.bench_function("lfu", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::default();
            for _ in 0..iters {
                let mut cache: LfuCache<u64, u64> = LfuCache::new(CAPACITY);
                let start = Instant::now();
                let _ = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
                total += start.elapsed();
            }
            total
        });
    });

    group.bench_function("heap_lfu", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::default();
            for _ in 0..iters {
                let mut cache: HeapLfuCache<u64, u64> = HeapLfuCache::new(CAPACITY);
                let start = Instant::now();
                let _ = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
                total += start.elapsed();
            }
            total
        });
    });

    group.bench_function("clock", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::default();
            for _ in 0..iters {
                let mut cache: ClockCache<u64, Arc<u64>> = ClockCache::new(CAPACITY);
                let start = Instant::now();
                let _ = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
                total += start.elapsed();
            }
            total
        });
    });

    group.bench_function("s3_fifo", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::default();
            for _ in 0..iters {
                let mut cache: S3FifoCache<u64, Arc<u64>> = S3FifoCache::new(CAPACITY);
                let start = Instant::now();
                let _ = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
                total += start.elapsed();
            }
            total
        });
    });

    group.bench_function("two_q", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::default();
            for _ in 0..iters {
                let mut cache: TwoQCore<u64, Arc<u64>> = TwoQCore::new(CAPACITY, 0.25);
                let start = Instant::now();
                let _ = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
                total += start.elapsed();
            }
            total
        });
    });

    group.finish();
}

// ============================================================================
// Adaptation Speed Benchmarks
// ============================================================================

fn bench_adaptation_speed(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptation_speed");

    group.bench_function("lru", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::default();
            for _ in 0..iters {
                let mut cache = LruCore::new(CAPACITY);
                let start = Instant::now();
                let _ = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
                total += start.elapsed();
            }
            total
        });
    });

    group.bench_function("lru_k", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::default();
            for _ in 0..iters {
                let mut cache: LrukCache<u64, Arc<u64>> = LrukCache::new(CAPACITY);
                let start = Instant::now();
                let _ = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
                total += start.elapsed();
            }
            total
        });
    });

    group.bench_function("lfu", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::default();
            for _ in 0..iters {
                let mut cache: LfuCache<u64, u64> = LfuCache::new(CAPACITY);
                let start = Instant::now();
                let _ = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
                total += start.elapsed();
            }
            total
        });
    });

    group.bench_function("heap_lfu", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::default();
            for _ in 0..iters {
                let mut cache: HeapLfuCache<u64, u64> = HeapLfuCache::new(CAPACITY);
                let start = Instant::now();
                let _ = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
                total += start.elapsed();
            }
            total
        });
    });

    group.bench_function("clock", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::default();
            for _ in 0..iters {
                let mut cache: ClockCache<u64, Arc<u64>> = ClockCache::new(CAPACITY);
                let start = Instant::now();
                let _ = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
                total += start.elapsed();
            }
            total
        });
    });

    group.bench_function("s3_fifo", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::default();
            for _ in 0..iters {
                let mut cache: S3FifoCache<u64, Arc<u64>> = S3FifoCache::new(CAPACITY);
                let start = Instant::now();
                let _ = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
                total += start.elapsed();
            }
            total
        });
    });

    group.bench_function("two_q", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::default();
            for _ in 0..iters {
                let mut cache: TwoQCore<u64, Arc<u64>> = TwoQCore::new(CAPACITY, 0.25);
                let start = Instant::now();
                let _ = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
                total += start.elapsed();
            }
            total
        });
    });

    group.finish();
}

// ============================================================================
// Comprehensive Benchmarks (latency + throughput + eviction stats)
// ============================================================================

fn bench_comprehensive(c: &mut Criterion) {
    let mut group = c.benchmark_group("comprehensive");
    let suite = standard_workload_suite(UNIVERSE, SEED);

    for (workload_name, spec) in &suite {
        let config = BenchmarkConfig {
            name: workload_name.to_string(),
            capacity: CAPACITY,
            operations: OPS,
            warmup_ops: CAPACITY,
            workload: *spec,
            latency_sample_rate: 100,
            max_latency_samples: 10_000,
        };

        group.bench_with_input(BenchmarkId::new("lru", workload_name), &config, |b, cfg| {
            b.iter_custom(|iters| {
                let mut total = std::time::Duration::default();
                for _ in 0..iters {
                    let mut cache = LruCore::new(CAPACITY);
                    let start = Instant::now();
                    let _ = run_benchmark("lru", &mut cache, cfg, Arc::new);
                    total += start.elapsed();
                }
                total
            });
        });

        group.bench_with_input(
            BenchmarkId::new("lru_k", workload_name),
            &config,
            |b, cfg| {
                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::default();
                    for _ in 0..iters {
                        let mut cache: LrukCache<u64, Arc<u64>> = LrukCache::new(CAPACITY);
                        let start = Instant::now();
                        let _ = run_benchmark("lru_k", &mut cache, cfg, Arc::new);
                        total += start.elapsed();
                    }
                    total
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("lfu", workload_name), &config, |b, cfg| {
            b.iter_custom(|iters| {
                let mut total = std::time::Duration::default();
                for _ in 0..iters {
                    let mut cache: LfuCache<u64, u64> = LfuCache::new(CAPACITY);
                    let start = Instant::now();
                    let _ = run_benchmark("lfu", &mut cache, cfg, Arc::new);
                    total += start.elapsed();
                }
                total
            });
        });

        group.bench_with_input(
            BenchmarkId::new("clock", workload_name),
            &config,
            |b, cfg| {
                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::default();
                    for _ in 0..iters {
                        let mut cache: ClockCache<u64, Arc<u64>> = ClockCache::new(CAPACITY);
                        let start = Instant::now();
                        let _ = run_benchmark("clock", &mut cache, cfg, Arc::new);
                        total += start.elapsed();
                    }
                    total
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("s3_fifo", workload_name),
            &config,
            |b, cfg| {
                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::default();
                    for _ in 0..iters {
                        let mut cache: S3FifoCache<u64, Arc<u64>> = S3FifoCache::new(CAPACITY);
                        let start = Instant::now();
                        let _ = run_benchmark("s3_fifo", &mut cache, cfg, Arc::new);
                        total += start.elapsed();
                    }
                    total
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("two_q", workload_name),
            &config,
            |b, cfg| {
                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::default();
                    for _ in 0..iters {
                        let mut cache: TwoQCore<u64, Arc<u64>> = TwoQCore::new(CAPACITY, 0.25);
                        let start = Instant::now();
                        let _ = run_benchmark("two_q", &mut cache, cfg, Arc::new);
                        total += start.elapsed();
                    }
                    total
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_hit_rates,
    bench_scan_resistance,
    bench_adaptation_speed,
    bench_comprehensive,
);
criterion_main!(benches);

// For human-readable reports, run: cargo bench --bench reports -- <report>
// See: benches/reports.rs
