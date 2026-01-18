//! Workload generators for hit-rate benchmarks.
//!
//! Provides deterministic key streams for cache benchmarking.

use std::sync::Arc;

use cachekit::traits::CoreCache;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Exp, Zipf};

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum Workload {
    /// Uniform random keys in `[0, universe)`.
    Uniform,
    /// Hot/cold split with a configurable hot fraction and hot access probability.
    Hotset { hot_fraction: f64, hot_prob: f64 },
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
    /// `lambda`: decay rate (higher = steeper dropoff, typical: 0.01-0.1).
    Exponential { lambda: f64 },
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
        Self {
            universe,
            workload,
            rng: SmallRng::seed_from_u64(seed),
            scan_pos: 0,
            operation_count: 0,
            insert_counter: 0,
            zipfian,
            exponential,
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

            Workload::Hotset {
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
    let mut hits = 0u64;
    let mut misses = 0u64;

    for _ in 0..operations {
        let key = generator.next_key();
        if cache.get(&key).is_some() {
            hits += 1;
        } else {
            misses += 1;
            let value = value_for_key(key);
            let _ = cache.insert(key, value);
            generator.record_insert();
        }
    }

    HitRate { hits, misses }
}
