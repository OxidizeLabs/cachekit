//! Workload generators for hit-rate benchmarks.
//!
//! Provides deterministic key streams for cache benchmarking.

// Not all workload variants are used in every benchmark configuration.

use std::sync::Arc;

use cachekit::traits::CoreCache;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Exp, Pareto as ParetoDistr, Zipf};

use crate::operation::{ReadThrough, run_operations};

#[derive(Debug, Clone, Copy)]
pub enum Workload {
    /// Uniform random keys in `[0, universe)`.
    Uniform,
    /// Hot/cold split with a configurable hot fraction and hot access probability.
    HotSet { hot_fraction: f64, hot_prob: f64 },
    /// Sequential scan in `[0, universe)`.
    Scan,
    /// Zipfian distribution - models real-world skewed access patterns.
    /// `exponent` controls skew: 1.0 = standard Zipf, higher = more skewed.
    Zipfian { exponent: f64 },
    /// Scrambled Zipfian - Zipfian with hashed keys to avoid sequential locality.
    /// YCSB's default distribution. Prevents hardware prefetch from skewing results.
    ScrambledZipfian { exponent: f64 },
    /// Latest - recently inserted keys are more likely to be accessed.
    /// Models temporal locality (social feeds, news, logs).
    /// Keys near `insert_counter` are favored with Zipfian falloff.
    Latest { exponent: f64 },
    /// Shifting hotspot - popular keys change over time.
    /// Tests cache adaptation when access patterns shift.
    /// `shift_interval`: operations between hotspot shifts.
    /// `hot_fraction`: fraction of universe that's hot at any time.
    ShiftingHotspot {
        shift_interval: u64,
        hot_fraction: f64,
    },
    /// Exponential decay - popularity drops exponentially with key distance.
    /// Models time-series data where recent items are accessed more.
    /// `lambda`: decay rate (higher = steeper drop off, typical: 0.01-0.1).
    Exponential { lambda: f64 },
    /// Pareto rule distribution - models cases where a small percentage of items receive the
    /// vast majority of accesses.
    Pareto { shape: f64 },
    /// Key differentiator for scan-resistant policies
    ScanResistance {
        scan_fraction: f64,
        scan_length: u64,
        point_exponent: f64,
    },
    /// Access to key K makes K+1, K+2, ... more likely.
    /// Fundamental pattern in sequential data processing.
    /// Models: Array traversals, database sequential scans, file system reads, B-tree leaf scans.
    Correlated {
        /// Step between correlated accesses
        stride: u64,
        /// Number of sequential accesses in burst
        burst_len: u64,
        /// Probability of starting a burst
        burst_prob: f64,
    },
    /// Critical edge case for cache sizing
    Loop { working_set_size: u64 },
    /// Fixed-size working set that slowly drifts over time.
    /// More realistic than ShiftingHotspot for modeling gradual popularity changes.
    WorkingSetChurn {
        working_set_size: u64,
        /// Fraction of working set replaced per operation
        churn_rate: f64,
    },
    /// Traffic arrives in bursts at multiple time scales.
    /// Exhibits long-range dependence - quiet periods followed by intense bursts.
    Bursty {
        /// Hurst parameter (0.5=random, 1.0=max correlation)
        hurst: f64,
        base_exponent: f64,
    },
    /// Sudden spike in traffic to specific keys.
    /// Models viral content or breaking news scenarios where popularity explodes suddenly.
    FlashCrowd {
        base_exponent: f64,
        /// Probability of flash event starting
        flash_prob: f64,
        /// Operations during flash
        flash_duration: u64,
        /// Number of keys affected
        flash_keys: u64,
        /// Multiplier on access probability
        flash_intensity: f64,
    },
    /// Meta-workload, combines others flexibly
    Mixture,
}

#[derive(Debug, Clone, Copy)]
pub struct WorkloadSpec {
    pub universe: u64,
    pub workload: Workload,
    pub seed: u64,
}

impl WorkloadSpec {
    pub fn generator(self) -> WorkloadGenerator {
        WorkloadGenerator::new(self.universe, self.workload, self.seed)
    }
}

#[derive(Debug, Clone)]
pub struct WorkloadGenerator {
    universe: u64,
    workload: Workload,
    rng: SmallRng,
    scan_pos: u64,
    operation_count: u64,
    insert_counter: u64,
    zipfian: Option<Zipf<f64>>,
    exponential: Option<Exp<f64>>,
    pareto: Option<ParetoDistr<f64>>,
    // Correlated workload state
    burst_remaining: u64,
    burst_start_key: u64,
    // Loop workload state
    loop_pos: u64,
    // WorkingSetChurn state
    working_set_base: u64,
    // Bursty workload state
    bursty_zipfian: Option<Zipf<f64>>,
    burst_active: bool,
    // FlashCrowd state
    flash_zipfian: Option<Zipf<f64>>,
    flash_active: bool,
    flash_ops_remaining: u64,
    flash_base_key: u64,
    // ScanResistance state
    scan_resistance_zipfian: Option<Zipf<f64>>,
    in_scan: bool,
    scan_ops_remaining: u64,
    scan_start_key: u64,
}

impl WorkloadGenerator {
    pub fn new(universe: u64, workload: Workload, seed: u64) -> Self {
        let universe = universe.max(1);
        let zipfian = match workload {
            Workload::Zipfian { exponent }
            | Workload::ScrambledZipfian { exponent }
            | Workload::Latest { exponent } => Some(Zipf::new(universe as f64, exponent).unwrap()),
            _ => None,
        };
        let exponential = match workload {
            Workload::Exponential { lambda } => Some(Exp::new(lambda).unwrap()),
            _ => None,
        };
        let pareto = match workload {
            Workload::Pareto { shape } => Some(ParetoDistr::new(1.0, shape).unwrap()),
            _ => None,
        };
        let bursty_zipfian = match workload {
            Workload::Bursty { base_exponent, .. } => {
                Some(Zipf::new(universe as f64, base_exponent).unwrap())
            },
            _ => None,
        };
        let flash_zipfian = match workload {
            Workload::FlashCrowd { base_exponent, .. } => {
                Some(Zipf::new(universe as f64, base_exponent).unwrap())
            },
            _ => None,
        };
        let scan_resistance_zipfian = match workload {
            Workload::ScanResistance { point_exponent, .. } => {
                Some(Zipf::new(universe as f64, point_exponent).unwrap())
            },
            _ => None,
        };
        Self {
            universe,
            workload,
            rng: SmallRng::seed_from_u64(seed),
            scan_pos: 0,
            operation_count: 0,
            insert_counter: 0,
            zipfian,
            exponential,
            pareto,
            burst_remaining: 0,
            burst_start_key: 0,
            loop_pos: 0,
            working_set_base: 0,
            bursty_zipfian,
            burst_active: false,
            flash_zipfian,
            flash_active: false,
            flash_ops_remaining: 0,
            flash_base_key: 0,
            scan_resistance_zipfian,
            in_scan: false,
            scan_ops_remaining: 0,
            scan_start_key: 0,
        }
    }

    /// Notify the generator that a key was inserted (for Latest workload).
    pub fn record_insert(&mut self) {
        self.insert_counter = self.insert_counter.wrapping_add(1);
    }

    pub fn next_key(&mut self) -> u64 {
        self.operation_count = self.operation_count.wrapping_add(1);

        match self.workload {
            Workload::Uniform => self.rng.random::<u64>() % self.universe,

            Workload::HotSet {
                hot_fraction,
                hot_prob,
            } => {
                let hot_fraction = hot_fraction.clamp(0.0, 1.0);
                let hot_prob = hot_prob.clamp(0.0, 1.0);
                let hot_size = ((self.universe as f64) * hot_fraction).round() as u64;
                let hot_size = hot_size.max(1).min(self.universe);
                if self.rng.random::<f64>() < hot_prob {
                    self.rng.random::<u64>() % hot_size
                } else if hot_size == self.universe {
                    self.rng.random::<u64>() % self.universe
                } else {
                    hot_size + (self.rng.random::<u64>() % (self.universe - hot_size))
                }
            },

            Workload::Scan => {
                let key = self.scan_pos;
                self.scan_pos = (self.scan_pos + 1) % self.universe;
                key
            },

            Workload::Zipfian { .. } => {
                let zipf = self.zipfian.as_ref().unwrap();
                let sample: f64 = zipf.sample(&mut self.rng);
                (sample as u64).saturating_sub(1).min(self.universe - 1)
            },

            Workload::ScrambledZipfian { .. } => {
                let zipf = self.zipfian.as_ref().unwrap();
                let sample: f64 = zipf.sample(&mut self.rng);
                let key = (sample as u64).saturating_sub(1).min(self.universe - 1);
                // FNV-1a hash to scramble the key
                fnv_hash(key) % self.universe
            },

            Workload::Latest { .. } => {
                let zipf = self.zipfian.as_ref().unwrap();
                let sample: f64 = zipf.sample(&mut self.rng);
                let offset = (sample as u64).saturating_sub(1).min(self.universe - 1);
                // Access keys near the most recent insert, wrapping around
                self.insert_counter.wrapping_sub(offset) % self.universe
            },

            Workload::ShiftingHotspot {
                shift_interval,
                hot_fraction,
            } => {
                let hot_fraction = hot_fraction.clamp(0.0, 1.0);
                let hot_size = ((self.universe as f64) * hot_fraction).round() as u64;
                let hot_size = hot_size.max(1).min(self.universe);

                // Shift the hotspot base periodically
                let shift_count = self.operation_count / shift_interval.max(1);
                let hotspot_base = (shift_count * hot_size) % self.universe;

                // 80% of accesses go to the current hotspot
                if self.rng.random::<f64>() < 0.8 {
                    hotspot_base + (self.rng.random::<u64>() % hot_size)
                } else {
                    self.rng.random::<u64>() % self.universe
                }
            },

            Workload::Exponential { .. } => {
                let exp = self.exponential.as_ref().unwrap();
                let sample: f64 = exp.sample(&mut self.rng);
                // Map exponential sample to key space, favoring lower keys
                let key = (sample * (self.universe as f64 / 10.0)) as u64;
                key.min(self.universe - 1)
            },

            Workload::Pareto { .. } => {
                let pareto = self.pareto.as_ref().unwrap();
                let sample: f64 = pareto.sample(&mut self.rng);
                // Pareto samples start at scale (1.0), map to key space
                let key = ((sample - 1.0) * (self.universe as f64 / 10.0)) as u64;
                key.min(self.universe - 1)
            },

            Workload::ScanResistance {
                scan_fraction,
                scan_length,
                ..
            } => {
                // Check if we should start a scan
                if !self.in_scan && self.rng.random::<f64>() < scan_fraction {
                    self.in_scan = true;
                    self.scan_ops_remaining = scan_length;
                    self.scan_start_key = self.rng.random::<u64>() % self.universe;
                }

                if self.in_scan {
                    let key = (self.scan_start_key + (scan_length - self.scan_ops_remaining))
                        % self.universe;
                    self.scan_ops_remaining -= 1;
                    if self.scan_ops_remaining == 0 {
                        self.in_scan = false;
                    }
                    key
                } else {
                    // Point lookup with Zipfian distribution
                    let zipf = self.scan_resistance_zipfian.as_ref().unwrap();
                    let sample: f64 = zipf.sample(&mut self.rng);
                    (sample as u64).saturating_sub(1).min(self.universe - 1)
                }
            },

            Workload::Correlated {
                stride,
                burst_len,
                burst_prob,
            } => {
                // Check if we're in a burst
                if self.burst_remaining > 0 {
                    let key = (self.burst_start_key + (burst_len - self.burst_remaining) * stride)
                        % self.universe;
                    self.burst_remaining -= 1;
                    key
                } else if self.rng.random::<f64>() < burst_prob {
                    // Start a new burst
                    self.burst_remaining = burst_len.saturating_sub(1);
                    self.burst_start_key = self.rng.random::<u64>() % self.universe;
                    self.burst_start_key
                } else {
                    // Random access
                    self.rng.random::<u64>() % self.universe
                }
            },

            Workload::Loop { working_set_size } => {
                let key = self.loop_pos % working_set_size.max(1);
                self.loop_pos = self.loop_pos.wrapping_add(1);
                key
            },

            Workload::WorkingSetChurn {
                working_set_size,
                churn_rate,
            } => {
                let working_set_size = working_set_size.max(1);
                // Occasionally shift the working set base
                if self.rng.random::<f64>() < churn_rate {
                    self.working_set_base =
                        (self.working_set_base + 1) % (self.universe - working_set_size + 1).max(1);
                }
                // Access within current working set
                let offset = self.rng.random::<u64>() % working_set_size;
                (self.working_set_base + offset) % self.universe
            },

            Workload::Bursty { hurst, .. } => {
                // Simplified bursty model using Hurst parameter to control burst probability
                // Higher hurst = more likely to stay in current state (bursty or quiet)
                let state_persistence = (hurst - 0.5).max(0.0) * 2.0; // 0.0 to 1.0

                if self.burst_active {
                    if self.rng.random::<f64>() > state_persistence {
                        self.burst_active = false;
                    }
                } else if self.rng.random::<f64>() < (1.0 - state_persistence) * 0.1 {
                    self.burst_active = true;
                }

                // During bursts, concentrate on fewer keys; otherwise use full distribution
                let zipf = self.bursty_zipfian.as_ref().unwrap();
                let sample: f64 = zipf.sample(&mut self.rng);
                let key = (sample as u64).saturating_sub(1).min(self.universe - 1);

                if self.burst_active {
                    // Concentrate on a subset during bursts
                    key % (self.universe / 10).max(1)
                } else {
                    key
                }
            },

            Workload::FlashCrowd {
                flash_prob,
                flash_duration,
                flash_keys,
                flash_intensity,
                ..
            } => {
                // Check if flash event should start
                if !self.flash_active && self.rng.random::<f64>() < flash_prob {
                    self.flash_active = true;
                    self.flash_ops_remaining = flash_duration;
                    self.flash_base_key = self.rng.random::<u64>() % self.universe;
                }

                if self.flash_active {
                    self.flash_ops_remaining -= 1;
                    if self.flash_ops_remaining == 0 {
                        self.flash_active = false;
                    }

                    // During flash, heavily favor the flash keys
                    if self.rng.random::<f64>() < flash_intensity / (flash_intensity + 1.0) {
                        let flash_keys = flash_keys.max(1);
                        self.flash_base_key + (self.rng.random::<u64>() % flash_keys)
                    } else {
                        // Occasional normal access
                        let zipf = self.flash_zipfian.as_ref().unwrap();
                        let sample: f64 = zipf.sample(&mut self.rng);
                        (sample as u64).saturating_sub(1).min(self.universe - 1)
                    }
                } else {
                    // Normal operation
                    let zipf = self.flash_zipfian.as_ref().unwrap();
                    let sample: f64 = zipf.sample(&mut self.rng);
                    (sample as u64).saturating_sub(1).min(self.universe - 1)
                }
            },

            Workload::Mixture => {
                // Default mixture: 70% Zipfian, 20% Scan-like, 10% Uniform
                let r = self.rng.random::<f64>();
                if r < 0.7 {
                    // Zipfian-like with manual calculation
                    let rank =
                        (1.0 / self.rng.random::<f64>().max(0.001)).min(self.universe as f64);
                    (rank as u64).saturating_sub(1).min(self.universe - 1)
                } else if r < 0.9 {
                    // Sequential scan behavior
                    let key = self.scan_pos;
                    self.scan_pos = (self.scan_pos + 1) % self.universe;
                    key
                } else {
                    // Uniform random
                    self.rng.random::<u64>() % self.universe
                }
            },
        }
    }
}

/// FNV-1a hash for scrambling keys.
#[inline]
fn fnv_hash(key: u64) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET;
    for byte in key.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HitRate {
    pub hits: u64,
    pub misses: u64,
}

impl HitRate {
    pub fn hit_rate(self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Run a hit-rate workload against a cache.
///
/// The cache is treated like a standard lookup+insert on miss. Values are
/// provided by `value_for_key` to avoid allocation in the benchmark harness.
pub fn run_hit_rate<C, V, F>(
    cache: &mut C,
    generator: &mut WorkloadGenerator,
    operations: usize,
    value_for_key: F,
) -> HitRate
where
    C: CoreCache<u64, Arc<V>>,
    F: Fn(u64) -> Arc<V>,
{
    let mut op_model = ReadThrough::new(1.0, 0);
    let counts = run_operations(cache, generator, operations, &mut op_model, value_for_key);
    HitRate {
        hits: counts.hits,
        misses: counts.misses,
    }
}
