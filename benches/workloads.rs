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
use cachekit::traits::CoreCache;
use common::metrics::{
    BenchmarkConfig, PolicyComparison, estimate_entry_overhead, measure_adaptation_speed,
    measure_scan_resistance, run_benchmark, standard_workload_suite,
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

/// Extended workloads for comprehensive testing (all available patterns).
#[cfg(test)]
fn extended_workloads() -> Vec<(&'static str, Workload)> {
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
        ("zipfian_0.8", Workload::Zipfian { exponent: 0.8 }),
        (
            "scrambled_zipf",
            Workload::ScrambledZipfian { exponent: 1.0 },
        ),
        ("latest", Workload::Latest { exponent: 0.8 }),
        (
            "shifting_hotspot",
            Workload::ShiftingHotspot {
                shift_interval: 10_000,
                hot_fraction: 0.1,
            },
        ),
        ("exponential", Workload::Exponential { lambda: 0.05 }),
        ("pareto", Workload::Pareto { shape: 1.5 }),
        (
            "scan_resistance",
            Workload::ScanResistance {
                scan_fraction: 0.2,
                scan_length: 1000,
                point_exponent: 1.0,
            },
        ),
        (
            "correlated",
            Workload::Correlated {
                stride: 1,
                burst_len: 8,
                burst_prob: 0.3,
            },
        ),
        (
            "loop_small",
            Workload::Loop {
                working_set_size: 512,
            },
        ),
        (
            "working_set_churn",
            Workload::WorkingSetChurn {
                working_set_size: 2048,
                churn_rate: 0.001,
            },
        ),
        (
            "bursty",
            Workload::Bursty {
                hurst: 0.8,
                base_exponent: 1.0,
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
        ("mixture", Workload::Mixture),
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

// ============================================================================
// Console report functions (only compiled for tests)
// ============================================================================

#[cfg(test)]
#[allow(dead_code)]
mod reports {
    use super::*;

    /// Helper to run LRU through a workload.
    fn run_lru_workload(workload: Workload) -> f64 {
        let mut cache = LruCore::new(CAPACITY);
        let mut generator = WorkloadSpec {
            universe: UNIVERSE,
            workload,
            seed: SEED,
        }
        .generator();
        let stats = run_hit_rate(&mut cache, &mut generator, OPS, Arc::new);
        stats.hit_rate()
    }

    /// Helper to run non-Arc caches through a workload.
    fn run_direct_workload<C: CoreCache<u64, Arc<u64>>>(cache: &mut C, workload: Workload) -> f64 {
        let mut generator = WorkloadSpec {
            universe: UNIVERSE,
            workload,
            seed: SEED,
        }
        .generator();
        let stats = run_hit_rate(cache, &mut generator, OPS, Arc::new);
        stats.hit_rate()
    }

    /// Print hit rates for quick comparison.
    pub fn print_hit_rate_comparison() {
        println!(
            "\n=== Hit Rate Comparison (capacity={}, universe={}, ops={}) ===",
            CAPACITY, UNIVERSE, OPS
        );
        let wl_list = workloads();
        print!("{:<12}", "Policy");
        for (name, _) in &wl_list {
            print!(" {:>14}", name);
        }
        println!();
        println!("{}", "-".repeat(12 + wl_list.len() * 15));

        print!("{:<12}", "LRU");
        for (_, wl) in &wl_list {
            print!(" {:>13.2}%", run_lru_workload(*wl) * 100.0);
        }
        println!();

        print!("{:<12}", "LRU-K");
        for (_, wl) in &wl_list {
            let mut cache: LrukCache<u64, Arc<u64>> = LrukCache::new(CAPACITY);
            print!(" {:>13.2}%", run_direct_workload(&mut cache, *wl) * 100.0);
        }
        println!();

        print!("{:<12}", "LFU");
        for (_, wl) in &wl_list {
            let mut cache: LfuCache<u64, u64> = LfuCache::new(CAPACITY);
            let mut generator = WorkloadSpec {
                universe: UNIVERSE,
                workload: *wl,
                seed: SEED,
            }
            .generator();
            let stats = run_hit_rate(&mut cache, &mut generator, OPS, Arc::new);
            print!(" {:>13.2}%", stats.hit_rate() * 100.0);
        }
        println!();

        print!("{:<12}", "Heap-LFU");
        for (_, wl) in &wl_list {
            let mut cache: HeapLfuCache<u64, u64> = HeapLfuCache::new(CAPACITY);
            let mut generator = WorkloadSpec {
                universe: UNIVERSE,
                workload: *wl,
                seed: SEED,
            }
            .generator();
            let stats = run_hit_rate(&mut cache, &mut generator, OPS, Arc::new);
            print!(" {:>13.2}%", stats.hit_rate() * 100.0);
        }
        println!();

        print!("{:<12}", "Clock");
        for (_, wl) in &wl_list {
            let mut cache: ClockCache<u64, Arc<u64>> = ClockCache::new(CAPACITY);
            print!(" {:>13.2}%", run_direct_workload(&mut cache, *wl) * 100.0);
        }
        println!();

        print!("{:<12}", "S3-FIFO");
        for (_, wl) in &wl_list {
            let mut cache: S3FifoCache<u64, Arc<u64>> = S3FifoCache::new(CAPACITY);
            print!(" {:>13.2}%", run_direct_workload(&mut cache, *wl) * 100.0);
        }
        println!();

        print!("{:<12}", "2Q");
        for (_, wl) in &wl_list {
            let mut cache: TwoQCore<u64, Arc<u64>> = TwoQCore::new(CAPACITY, 0.25);
            print!(" {:>13.2}%", run_direct_workload(&mut cache, *wl) * 100.0);
        }
        println!();
    }

    /// Print extended hit rate comparison with all workload types.
    pub fn print_extended_hit_rate_comparison() {
        println!(
            "\n=== Extended Hit Rate Comparison (capacity={}, universe={}, ops={}) ===",
            CAPACITY, UNIVERSE, OPS
        );
        let wl_list = extended_workloads();

        // Print in groups of 6 workloads for readability
        for chunk in wl_list.chunks(6) {
            print!("{:<12}", "Policy");
            for (name, _) in chunk {
                print!(" {:>12}", name);
            }
            println!();
            println!("{}", "-".repeat(12 + chunk.len() * 13));

            print!("{:<12}", "LRU");
            for (_, wl) in chunk {
                print!(" {:>11.2}%", run_lru_workload(*wl) * 100.0);
            }
            println!();

            print!("{:<12}", "LRU-K");
            for (_, wl) in chunk {
                let mut cache: LrukCache<u64, Arc<u64>> = LrukCache::new(CAPACITY);
                print!(" {:>11.2}%", run_direct_workload(&mut cache, *wl) * 100.0);
            }
            println!();

            print!("{:<12}", "S3-FIFO");
            for (_, wl) in chunk {
                let mut cache: S3FifoCache<u64, Arc<u64>> = S3FifoCache::new(CAPACITY);
                print!(" {:>11.2}%", run_direct_workload(&mut cache, *wl) * 100.0);
            }
            println!();

            print!("{:<12}", "2Q");
            for (_, wl) in chunk {
                let mut cache: TwoQCore<u64, Arc<u64>> = TwoQCore::new(CAPACITY, 0.25);
                print!(" {:>11.2}%", run_direct_workload(&mut cache, *wl) * 100.0);
            }
            println!("\n");
        }
    }

    /// Print scan resistance comparison.
    pub fn print_scan_resistance_comparison() {
        println!("\n=== Scan Resistance Comparison ===");
        println!(
            "{:<12} {:>12} {:>12} {:>12} {:>12}",
            "Policy", "Baseline", "During Scan", "Recovery", "Score"
        );
        println!("{}", "-".repeat(60));

        let mut cache = LruCore::new(CAPACITY);
        let result = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
        println!(
            "{:<12} {:>11.2}% {:>11.2}% {:>11.2}% {:>11.2}",
            "LRU",
            result.baseline_hit_rate * 100.0,
            result.scan_hit_rate * 100.0,
            result.recovery_hit_rate * 100.0,
            result.resistance_score
        );

        let mut cache: LrukCache<u64, Arc<u64>> = LrukCache::new(CAPACITY);
        let result = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
        println!(
            "{:<12} {:>11.2}% {:>11.2}% {:>11.2}% {:>11.2}",
            "LRU-K",
            result.baseline_hit_rate * 100.0,
            result.scan_hit_rate * 100.0,
            result.recovery_hit_rate * 100.0,
            result.resistance_score
        );

        let mut cache: LfuCache<u64, u64> = LfuCache::new(CAPACITY);
        let result = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
        println!(
            "{:<12} {:>11.2}% {:>11.2}% {:>11.2}% {:>11.2}",
            "LFU",
            result.baseline_hit_rate * 100.0,
            result.scan_hit_rate * 100.0,
            result.recovery_hit_rate * 100.0,
            result.resistance_score
        );

        let mut cache: HeapLfuCache<u64, u64> = HeapLfuCache::new(CAPACITY);
        let result = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
        println!(
            "{:<12} {:>11.2}% {:>11.2}% {:>11.2}% {:>11.2}",
            "Heap-LFU",
            result.baseline_hit_rate * 100.0,
            result.scan_hit_rate * 100.0,
            result.recovery_hit_rate * 100.0,
            result.resistance_score
        );

        let mut cache: S3FifoCache<u64, Arc<u64>> = S3FifoCache::new(CAPACITY);
        let result = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
        println!(
            "{:<12} {:>11.2}% {:>11.2}% {:>11.2}% {:>11.2}",
            "S3-FIFO",
            result.baseline_hit_rate * 100.0,
            result.scan_hit_rate * 100.0,
            result.recovery_hit_rate * 100.0,
            result.resistance_score
        );

        let mut cache: TwoQCore<u64, Arc<u64>> = TwoQCore::new(CAPACITY, 0.25);
        let result = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
        println!(
            "{:<12} {:>11.2}% {:>11.2}% {:>11.2}% {:>11.2}",
            "2Q",
            result.baseline_hit_rate * 100.0,
            result.scan_hit_rate * 100.0,
            result.recovery_hit_rate * 100.0,
            result.resistance_score
        );

        let mut cache: ClockCache<u64, Arc<u64>> = ClockCache::new(CAPACITY);
        let result = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
        println!(
            "{:<12} {:>11.2}% {:>11.2}% {:>11.2}% {:>11.2}",
            "Clock",
            result.baseline_hit_rate * 100.0,
            result.scan_hit_rate * 100.0,
            result.recovery_hit_rate * 100.0,
            result.resistance_score
        );

        // Also print compact summaries
        println!("\n--- Compact Summaries ---");
        let mut cache = LruCore::new(CAPACITY);
        let result = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
        println!("LRU:      {}", result.summary());

        let mut cache: S3FifoCache<u64, Arc<u64>> = S3FifoCache::new(CAPACITY);
        let result = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
        println!("S3-FIFO:  {}", result.summary());

        let mut cache: TwoQCore<u64, Arc<u64>> = TwoQCore::new(CAPACITY, 0.25);
        let result = measure_scan_resistance(&mut cache, CAPACITY, UNIVERSE, Arc::new);
        println!("2Q:       {}", result.summary());
    }

    /// Print adaptation speed comparison.
    pub fn print_adaptation_comparison() {
        println!("\n=== Adaptation Speed Comparison ===");
        println!(
            "{:<12} {:>15} {:>15} {:>12}",
            "Policy", "Ops to 50%", "Ops to 80%", "Stable HR"
        );
        println!("{}", "-".repeat(60));

        let mut cache = LruCore::new(CAPACITY);
        let result = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
        println!(
            "{:<12} {:>15} {:>15} {:>11.2}%",
            "LRU",
            result.ops_to_50_percent,
            result.ops_to_80_percent,
            result.stable_hit_rate * 100.0
        );

        let mut cache: LrukCache<u64, Arc<u64>> = LrukCache::new(CAPACITY);
        let result = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
        println!(
            "{:<12} {:>15} {:>15} {:>11.2}%",
            "LRU-K",
            result.ops_to_50_percent,
            result.ops_to_80_percent,
            result.stable_hit_rate * 100.0
        );

        let mut cache: LfuCache<u64, u64> = LfuCache::new(CAPACITY);
        let result = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
        println!(
            "{:<12} {:>15} {:>15} {:>11.2}%",
            "LFU",
            result.ops_to_50_percent,
            result.ops_to_80_percent,
            result.stable_hit_rate * 100.0
        );

        let mut cache: HeapLfuCache<u64, u64> = HeapLfuCache::new(CAPACITY);
        let result = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
        println!(
            "{:<12} {:>15} {:>15} {:>11.2}%",
            "Heap-LFU",
            result.ops_to_50_percent,
            result.ops_to_80_percent,
            result.stable_hit_rate * 100.0
        );

        let mut cache: S3FifoCache<u64, Arc<u64>> = S3FifoCache::new(CAPACITY);
        let result = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
        println!(
            "{:<12} {:>15} {:>15} {:>11.2}%",
            "S3-FIFO",
            result.ops_to_50_percent,
            result.ops_to_80_percent,
            result.stable_hit_rate * 100.0
        );

        let mut cache: TwoQCore<u64, Arc<u64>> = TwoQCore::new(CAPACITY, 0.25);
        let result = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
        println!(
            "{:<12} {:>15} {:>15} {:>11.2}%",
            "2Q",
            result.ops_to_50_percent,
            result.ops_to_80_percent,
            result.stable_hit_rate * 100.0
        );

        let mut cache: ClockCache<u64, Arc<u64>> = ClockCache::new(CAPACITY);
        let result = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
        println!(
            "{:<12} {:>15} {:>15} {:>11.2}%",
            "Clock",
            result.ops_to_50_percent,
            result.ops_to_80_percent,
            result.stable_hit_rate * 100.0
        );

        // Print summaries and hit rate curves for a few key policies
        println!("\n--- Compact Summaries ---");
        let mut cache = LruCore::new(CAPACITY);
        let result = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
        println!("LRU:      {}", result.summary());

        let mut cache: S3FifoCache<u64, Arc<u64>> = S3FifoCache::new(CAPACITY);
        let result = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
        println!("S3-FIFO:  {}", result.summary());

        // Show hit rate curve for LRU to demonstrate adaptation over time
        println!("\n--- LRU Adaptation Curve (hit rate per window) ---");
        let mut cache = LruCore::new(CAPACITY);
        let result = measure_adaptation_speed(&mut cache, CAPACITY, UNIVERSE, Arc::new);
        for (i, &rate) in result.hit_rate_curve.iter().enumerate() {
            let bar_len = (rate * 40.0) as usize;
            println!(
                "Window {:2}: {:5.1}% {}",
                i + 1,
                rate * 100.0,
                "#".repeat(bar_len)
            );
        }
    }

    /// Run comprehensive comparison with full metrics.
    pub fn run_comprehensive_comparison() {
        println!("\n=== Comprehensive Policy Comparison ===\n");

        let suite = standard_workload_suite(UNIVERSE, SEED);

        // LRU
        {
            let mut comparison = PolicyComparison::new("LRU");
            for (workload_name, spec) in &suite {
                let mut cache = LruCore::new(CAPACITY);
                let config = BenchmarkConfig {
                    name: workload_name.to_string(),
                    capacity: CAPACITY,
                    operations: OPS,
                    warmup_ops: CAPACITY,
                    workload: *spec,
                    latency_sample_rate: 100,
                    max_latency_samples: 10_000,
                };
                let result = run_benchmark("LRU", &mut cache, &config, Arc::new);
                comparison.add_result(result);
            }
            comparison.print_table();
            println!();
        }

        // S3-FIFO
        {
            let mut comparison = PolicyComparison::new("S3-FIFO");
            for (workload_name, spec) in &suite {
                let mut cache: S3FifoCache<u64, Arc<u64>> = S3FifoCache::new(CAPACITY);
                let config = BenchmarkConfig {
                    name: workload_name.to_string(),
                    capacity: CAPACITY,
                    operations: OPS,
                    warmup_ops: CAPACITY,
                    workload: *spec,
                    latency_sample_rate: 100,
                    max_latency_samples: 10_000,
                };
                let result = run_benchmark("S3-FIFO", &mut cache, &config, Arc::new);
                comparison.add_result(result);
            }
            comparison.print_table();
            println!();
        }

        // Clock
        {
            let mut comparison = PolicyComparison::new("Clock");
            for (workload_name, spec) in &suite {
                let mut cache: ClockCache<u64, Arc<u64>> = ClockCache::new(CAPACITY);
                let config = BenchmarkConfig {
                    name: workload_name.to_string(),
                    capacity: CAPACITY,
                    operations: OPS,
                    warmup_ops: CAPACITY,
                    workload: *spec,
                    latency_sample_rate: 100,
                    max_latency_samples: 10_000,
                };
                let result = run_benchmark("Clock", &mut cache, &config, Arc::new);
                comparison.add_result(result);
            }
            comparison.print_table();
            println!();
        }
    }

    /// Run detailed benchmark showing all metric fields for a single policy/workload.
    pub fn run_detailed_single_benchmark() {
        println!("\n=== Detailed Benchmark Results ===\n");

        let workload = Workload::Zipfian { exponent: 1.0 };
        let spec = WorkloadSpec {
            universe: UNIVERSE,
            workload,
            seed: SEED,
        };

        let config = BenchmarkConfig {
            name: "zipfian_1.0".to_string(),
            capacity: CAPACITY,
            operations: OPS,
            warmup_ops: CAPACITY,
            workload: spec,
            latency_sample_rate: 100,
            max_latency_samples: 10_000,
        };

        let mut cache = LruCore::new(CAPACITY);
        let result = run_benchmark("LRU", &mut cache, &config, Arc::new);

        println!("Summary: {}\n", result.summary());

        println!("--- Configuration ---");
        println!("  Policy:     {}", result.policy_name);
        println!("  Workload:   {}", result.workload_name);
        println!("  Capacity:   {}", result.capacity);
        println!("  Universe:   {}", result.universe);
        println!("  Operations: {}", result.operations);

        println!("\n--- Hit Statistics ---");
        println!("  Hits:       {}", result.hit_stats.hits);
        println!("  Misses:     {}", result.hit_stats.misses);
        println!("  Inserts:    {}", result.hit_stats.inserts);
        println!("  Updates:    {}", result.hit_stats.updates);
        println!("  Hit Rate:   {:.2}%", result.hit_stats.hit_rate() * 100.0);
        println!("  Miss Rate:  {:.2}%", result.hit_stats.miss_rate() * 100.0);
        println!("  Total Ops:  {}", result.hit_stats.total_ops());

        println!("\n--- Throughput ---");
        println!("  Duration:       {:?}", result.throughput.total_duration);
        println!("  Ops/sec:        {:.0}", result.throughput.ops_per_sec);
        println!("  Gets/sec:       {:.0}", result.throughput.gets_per_sec);
        println!("  Inserts/sec:    {:.0}", result.throughput.inserts_per_sec);

        println!("\n--- Latency Distribution ---");
        println!("  Samples:  {}", result.latency.sample_count);
        println!("  Min:      {:?}", result.latency.min);
        println!("  p50:      {:?}", result.latency.p50);
        println!("  p95:      {:?}", result.latency.p95);
        println!("  p99:      {:?}", result.latency.p99);
        println!("  Max:      {:?}", result.latency.max);
        println!("  Mean:     {:?}", result.latency.mean);

        println!("\n--- Eviction Statistics ---");
        println!("  Total Evictions:     {}", result.eviction.total_evictions);
        println!(
            "  Evictions per Insert: {:.3}",
            result.eviction.evictions_per_insert
        );
    }

    /// Print memory overhead comparison across policies.
    pub fn print_memory_overhead_comparison() {
        println!("\n=== Memory Overhead Comparison ===");
        println!(
            "{:<12} {:>12} {:>15} {:>12}",
            "Policy", "Total (B)", "Bytes/Entry", "Entries"
        );
        println!("{}", "-".repeat(55));

        // LRU
        {
            let mut cache = LruCore::new(CAPACITY);
            // Fill cache
            for i in 0..CAPACITY as u64 {
                cache.insert(i, Arc::new(i));
            }
            let estimate = estimate_entry_overhead(&cache, cache.len());
            println!(
                "{:<12} {:>12} {:>15} {:>12}",
                "LRU", estimate.total_bytes, estimate.bytes_per_entry, estimate.entry_count
            );
            println!("  -> {}", estimate.summary());
        }

        // LRU-K
        {
            let mut cache: LrukCache<u64, Arc<u64>> = LrukCache::new(CAPACITY);
            for i in 0..CAPACITY as u64 {
                cache.insert(i, Arc::new(i));
            }
            let estimate = estimate_entry_overhead(&cache, cache.len());
            println!(
                "{:<12} {:>12} {:>15} {:>12}",
                "LRU-K", estimate.total_bytes, estimate.bytes_per_entry, estimate.entry_count
            );
        }

        // Clock
        {
            let mut cache: ClockCache<u64, Arc<u64>> = ClockCache::new(CAPACITY);
            for i in 0..CAPACITY as u64 {
                cache.insert(i, Arc::new(i));
            }
            let estimate = estimate_entry_overhead(&cache, cache.len());
            println!(
                "{:<12} {:>12} {:>15} {:>12}",
                "Clock", estimate.total_bytes, estimate.bytes_per_entry, estimate.entry_count
            );
        }

        // S3-FIFO
        {
            let mut cache: S3FifoCache<u64, Arc<u64>> = S3FifoCache::new(CAPACITY);
            for i in 0..CAPACITY as u64 {
                cache.insert(i, Arc::new(i));
            }
            let estimate = estimate_entry_overhead(&cache, cache.len());
            println!(
                "{:<12} {:>12} {:>15} {:>12}",
                "S3-FIFO", estimate.total_bytes, estimate.bytes_per_entry, estimate.entry_count
            );
        }

        // 2Q
        {
            let mut cache: TwoQCore<u64, Arc<u64>> = TwoQCore::new(CAPACITY, 0.25);
            for i in 0..CAPACITY as u64 {
                cache.insert(i, Arc::new(i));
            }
            let estimate = estimate_entry_overhead(&cache, cache.len());
            println!(
                "{:<12} {:>12} {:>15} {:>12}",
                "2Q", estimate.total_bytes, estimate.bytes_per_entry, estimate.entry_count
            );
        }

        println!(
            "\nNote: These are shallow size estimates (size_of_val). Heap allocations not included."
        );
    }

    // =========================================================================
    // Test runners
    // =========================================================================

    #[test]
    #[ignore] // Run with: cargo test --bench workloads -- --ignored --nocapture
    fn test_hit_rate_comparison() {
        print_hit_rate_comparison();
    }

    #[test]
    #[ignore]
    fn test_extended_hit_rate_comparison() {
        print_extended_hit_rate_comparison();
    }

    #[test]
    #[ignore]
    fn test_scan_resistance() {
        print_scan_resistance_comparison();
    }

    #[test]
    #[ignore]
    fn test_adaptation_speed() {
        print_adaptation_comparison();
    }

    #[test]
    #[ignore]
    fn test_comprehensive() {
        run_comprehensive_comparison();
    }

    #[test]
    #[ignore]
    fn test_detailed_single() {
        run_detailed_single_benchmark();
    }

    #[test]
    #[ignore]
    fn test_memory_overhead() {
        print_memory_overhead_comparison();
    }

    #[test]
    #[ignore]
    fn test_all_reports() {
        print_hit_rate_comparison();
        print_extended_hit_rate_comparison();
        print_scan_resistance_comparison();
        print_adaptation_comparison();
        run_detailed_single_benchmark();
        print_memory_overhead_comparison();
        run_comprehensive_comparison();
    }
}
