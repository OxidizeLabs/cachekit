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
    /// Zipfian distribution - models real-world skewed access patterns.
    /// `theta` controls skew: 0.0 = uniform, 0.99 = highly skewed (YCSB default).
    Zipfian { theta: f64 },
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
    zipfian: Option<ZipfianState>,
}

impl WorkloadGenerator {
    pub fn new(universe: u64, workload: Workload, seed: u64) -> Self {
        let universe = universe.max(1);
        let zipfian = match workload {
            Workload::Zipfian { theta } => Some(ZipfianState::new(universe, theta)),
            _ => None,
        };
        Self {
            universe,
            workload,
            rng: XorShift64::new(seed),
            scan_pos: 0,
            zipfian,
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
            Workload::Zipfian { .. } => {
                let zipf = self.zipfian.as_ref().unwrap();
                let u = self.rng.next_f64();
                zipf.sample(u)
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

/// Zipfian distribution state for inverse CDF sampling.
///
/// Uses the algorithm from YCSB (Yahoo Cloud Serving Benchmark).
/// Pre-computes zeta values for efficient sampling.
#[derive(Debug, Clone)]
struct ZipfianState {
    n: u64,
    theta: f64,
    zeta_n: f64,
    alpha: f64,
    eta: f64,
}

impl ZipfianState {
    fn new(n: u64, theta: f64) -> Self {
        let theta = theta.clamp(0.0, 0.9999); // Avoid division issues at theta=1
        let zeta_2 = Self::zeta(2, theta);
        let zeta_n = Self::zeta(n, theta);
        let alpha = 1.0 / (1.0 - theta);
        let eta = (1.0 - (2.0 / n as f64).powf(1.0 - theta)) / (1.0 - zeta_2 / zeta_n);

        Self {
            n,
            theta,
            zeta_n,
            alpha,
            eta,
        }
    }

    /// Compute zeta(n, theta) = sum(1/i^theta for i in 1..=n)
    fn zeta(n: u64, theta: f64) -> f64 {
        let mut sum = 0.0;
        for i in 1..=n {
            sum += 1.0 / (i as f64).powf(theta);
        }
        sum
    }

    /// Sample from Zipfian distribution given uniform random u in [0, 1).
    fn sample(&self, u: f64) -> u64 {
        let uz = u * self.zeta_n;

        if uz < 1.0 {
            return 0;
        }

        if uz < 1.0 + 0.5_f64.powf(self.theta) {
            return 1;
        }

        let spread = (self.n as f64) * (self.eta * u - self.eta + 1.0).powf(self.alpha);
        (spread as u64).min(self.n - 1)
    }
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
