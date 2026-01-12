//! Workload generators for hit-rate benchmarks.
//!
//! Provides deterministic key streams without pulling in external RNG crates.

use std::sync::Arc;

use cachekit::traits::CoreCache;

#[derive(Debug, Clone, Copy)]
pub enum Workload {
    /// Uniform random keys in `[0, universe)`.
    Uniform,
    /// Hot/cold split with a configurable hot fraction and hot access probability.
    Hotset { hot_fraction: f64, hot_prob: f64 },
    /// Sequential scan in `[0, universe)`.
    Scan,
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
    rng: XorShift64,
    scan_pos: u64,
}

impl WorkloadGenerator {
    pub fn new(universe: u64, workload: Workload, seed: u64) -> Self {
        let universe = universe.max(1);
        Self {
            universe,
            workload,
            rng: XorShift64::new(seed),
            scan_pos: 0,
        }
    }

    pub fn next_key(&mut self) -> u64 {
        match self.workload {
            Workload::Uniform => self.rng.next_u64() % self.universe,
            Workload::Hotset {
                hot_fraction,
                hot_prob,
            } => {
                let hot_fraction = hot_fraction.clamp(0.0, 1.0);
                let hot_prob = hot_prob.clamp(0.0, 1.0);
                let hot_size = ((self.universe as f64) * hot_fraction).round() as u64;
                let hot_size = hot_size.max(1).min(self.universe);
                if self.rng.next_f64() < hot_prob {
                    self.rng.next_u64() % hot_size
                } else if hot_size == self.universe {
                    self.rng.next_u64() % self.universe
                } else {
                    hot_size + (self.rng.next_u64() % (self.universe - hot_size))
                }
            },
            Workload::Scan => {
                let key = self.scan_pos;
                self.scan_pos = (self.scan_pos + 1) % self.universe;
                key
            },
        }
    }
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
        }
    }

    HitRate { hits, misses }
}

#[derive(Debug, Clone, Copy)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / (u64::MAX as f64);
        (self.next_u64() as f64) * SCALE
    }
}
