//! Workload generators for hit-rate benchmarks.
//!
//! ## Architecture
//!
//! [`Workload`] is a parameter-only enum that travels in [`WorkloadSpec`] and
//! describes which access pattern to run. At construction time
//! [`WorkloadGenerator::new`] consumes the [`Workload`] and builds a
//! per-variant [`GeneratorState`] which carries exactly the runtime state and
//! pre-built distributions that this access pattern needs — nothing more. The
//! per-op [`WorkloadGenerator::next_key`] then dispatches on `state` and runs
//! a tight, allocation-free arm.
//!
//! Splitting state per variant keeps [`WorkloadGenerator`] small (Zipfian
//! variants need ~80 bytes of state, not 200+) and makes each access pattern's
//! invariants local: the scan-resistance state lives next to the
//! scan-resistance code, and a new variant can't accidentally read another
//! variant's stale state.

use std::sync::Arc;

use cachekit::traits::Cache;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
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
    /// Samples are scaled by `universe / 10` and clamped to the universe so
    /// most accesses concentrate in the first ~10% of the key space.
    Exponential { lambda: f64 },
    /// Pareto distribution - heavy-tailed access; a small share of keys
    /// receives the majority of accesses. Samples are scaled by
    /// `universe / 10` and clamped to the universe (same convention as
    /// `Exponential`).
    Pareto { shape: f64 },
    /// Mixes Zipfian point lookups with sequential scans to exercise
    /// scan-resistant policies.
    ///
    /// `scan_start_prob` is the per-operation probability of *starting* a new
    /// scan when not already scanning; in steady state, the fraction of
    /// operations that are scans is roughly
    /// `scan_start_prob * scan_length / (1 + scan_start_prob * scan_length)`.
    ScanResistance {
        scan_start_prob: f64,
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

/// Per-variant runtime state. Each variant only carries the parameters and
/// mutable state its [`WorkloadGenerator::next_key`] arm reads or writes,
/// avoiding the `Option<Distribution>` swamp that the previous god-struct
/// allocated for every workload regardless of which one was active.
#[derive(Debug, Clone)]
enum GeneratorState {
    Uniform,
    HotSet {
        hot_size: u64,
        hot_prob: f64,
    },
    Scan {
        pos: u64,
    },
    Zipfian {
        zipf: Zipf<f64>,
    },
    ScrambledZipfian {
        zipf: Zipf<f64>,
    },
    Latest {
        zipf: Zipf<f64>,
        insert_counter: u64,
    },
    ShiftingHotspot {
        shift_interval: u64,
        hot_size: u64,
        op_count: u64,
    },
    Exponential {
        exp: Exp<f64>,
    },
    Pareto {
        pareto: ParetoDistr<f64>,
    },
    ScanResistance {
        zipf: Zipf<f64>,
        scan_start_prob: f64,
        scan_length: u64,
        in_scan: bool,
        ops_remaining: u64,
        start_key: u64,
    },
    Correlated {
        stride: u64,
        burst_len: u64,
        burst_prob: f64,
        burst_remaining: u64,
        burst_start_key: u64,
    },
    Loop {
        working_set_size: u64,
        pos: u64,
    },
    WorkingSetChurn {
        working_set_size: u64,
        churn_rate: f64,
        base: u64,
    },
    Bursty {
        zipf: Zipf<f64>,
        state_persistence: f64,
        active: bool,
    },
    FlashCrowd {
        zipf: Zipf<f64>,
        flash_prob: f64,
        flash_duration: u64,
        flash_keys: u64,
        flash_intensity: f64,
        active: bool,
        ops_remaining: u64,
        base_key: u64,
    },
    Mixture {
        scan_pos: u64,
    },
}

impl GeneratorState {
    fn build(universe: u64, workload: Workload) -> Self {
        let make_zipf = |exponent: f64| {
            Zipf::new(universe as f64, exponent).expect("Zipf parameters out of range")
        };
        let hot_size_of = |hot_fraction: f64| {
            let hot_fraction = hot_fraction.clamp(0.0, 1.0);
            ((universe as f64) * hot_fraction)
                .round()
                .clamp(1.0, universe as f64) as u64
        };
        match workload {
            Workload::Uniform => Self::Uniform,
            Workload::HotSet {
                hot_fraction,
                hot_prob,
            } => Self::HotSet {
                hot_size: hot_size_of(hot_fraction),
                hot_prob: hot_prob.clamp(0.0, 1.0),
            },
            Workload::Scan => Self::Scan { pos: 0 },
            Workload::Zipfian { exponent } => Self::Zipfian {
                zipf: make_zipf(exponent),
            },
            Workload::ScrambledZipfian { exponent } => Self::ScrambledZipfian {
                zipf: make_zipf(exponent),
            },
            Workload::Latest { exponent } => Self::Latest {
                zipf: make_zipf(exponent),
                insert_counter: 0,
            },
            Workload::ShiftingHotspot {
                shift_interval,
                hot_fraction,
            } => Self::ShiftingHotspot {
                shift_interval: shift_interval.max(1),
                hot_size: hot_size_of(hot_fraction),
                op_count: 0,
            },
            Workload::Exponential { lambda } => Self::Exponential {
                exp: Exp::new(lambda).expect("Exp lambda out of range"),
            },
            Workload::Pareto { shape } => Self::Pareto {
                pareto: ParetoDistr::new(1.0, shape).expect("Pareto shape out of range"),
            },
            Workload::ScanResistance {
                scan_start_prob,
                scan_length,
                point_exponent,
            } => Self::ScanResistance {
                zipf: make_zipf(point_exponent),
                scan_start_prob,
                scan_length,
                in_scan: false,
                ops_remaining: 0,
                start_key: 0,
            },
            Workload::Correlated {
                stride,
                burst_len,
                burst_prob,
            } => Self::Correlated {
                stride,
                burst_len,
                burst_prob,
                burst_remaining: 0,
                burst_start_key: 0,
            },
            Workload::Loop { working_set_size } => Self::Loop {
                working_set_size: working_set_size.max(1),
                pos: 0,
            },
            Workload::WorkingSetChurn {
                working_set_size,
                churn_rate,
            } => Self::WorkingSetChurn {
                working_set_size: working_set_size.max(1),
                churn_rate,
                base: 0,
            },
            Workload::Bursty {
                hurst,
                base_exponent,
            } => Self::Bursty {
                zipf: make_zipf(base_exponent),
                state_persistence: ((hurst - 0.5).max(0.0) * 2.0).clamp(0.0, 1.0),
                active: false,
            },
            Workload::FlashCrowd {
                base_exponent,
                flash_prob,
                flash_duration,
                flash_keys,
                flash_intensity,
            } => Self::FlashCrowd {
                zipf: make_zipf(base_exponent),
                flash_prob,
                flash_duration,
                flash_keys: flash_keys.max(1),
                flash_intensity,
                active: false,
                ops_remaining: 0,
                base_key: 0,
            },
            Workload::Mixture => Self::Mixture { scan_pos: 0 },
        }
    }
}

#[derive(Debug, Clone)]
pub struct WorkloadGenerator {
    universe: u64,
    rng: SmallRng,
    state: GeneratorState,
}

impl WorkloadGenerator {
    pub fn new(universe: u64, workload: Workload, seed: u64) -> Self {
        let universe = universe.max(1);
        Self {
            universe,
            rng: SmallRng::seed_from_u64(seed),
            state: GeneratorState::build(universe, workload),
        }
    }

    /// Notify the generator that a key was inserted (for the `Latest` workload).
    /// All other variants ignore inserts.
    pub fn record_insert(&mut self) {
        if let GeneratorState::Latest { insert_counter, .. } = &mut self.state {
            *insert_counter = insert_counter.wrapping_add(1);
        }
    }

    pub fn next_key(&mut self) -> u64 {
        let universe = self.universe;
        match &mut self.state {
            GeneratorState::Uniform => self.rng.random::<u64>() % universe,

            GeneratorState::HotSet { hot_size, hot_prob } => {
                if self.rng.random::<f64>() < *hot_prob {
                    self.rng.random::<u64>() % *hot_size
                } else if *hot_size == universe {
                    self.rng.random::<u64>() % universe
                } else {
                    *hot_size + (self.rng.random::<u64>() % (universe - *hot_size))
                }
            },

            GeneratorState::Scan { pos } => {
                let key = *pos;
                *pos = (*pos + 1) % universe;
                key
            },

            GeneratorState::Zipfian { zipf } => {
                let sample: f64 = zipf.sample(&mut self.rng);
                (sample as u64).saturating_sub(1).min(universe - 1)
            },

            GeneratorState::ScrambledZipfian { zipf } => {
                let sample: f64 = zipf.sample(&mut self.rng);
                let key = (sample as u64).saturating_sub(1).min(universe - 1);
                fnv_hash(key) % universe
            },

            GeneratorState::Latest {
                zipf,
                insert_counter,
            } => {
                let sample: f64 = zipf.sample(&mut self.rng);
                let offset = (sample as u64).saturating_sub(1).min(universe - 1);
                insert_counter.wrapping_sub(offset) % universe
            },

            GeneratorState::ShiftingHotspot {
                shift_interval,
                hot_size,
                op_count,
            } => {
                *op_count = op_count.wrapping_add(1);
                let shift_count = *op_count / *shift_interval;
                let hotspot_base = (shift_count * *hot_size) % universe;
                if self.rng.random::<f64>() < 0.8 {
                    let offset = self.rng.random::<u64>() % *hot_size;
                    // Wrap modulo universe so a hotspot that straddles the
                    // end of the key space stays in-bounds.
                    (hotspot_base + offset) % universe
                } else {
                    self.rng.random::<u64>() % universe
                }
            },

            GeneratorState::Exponential { exp } => {
                let sample: f64 = exp.sample(&mut self.rng);
                let key = (sample * (universe as f64 / 10.0)) as u64;
                key.min(universe - 1)
            },

            GeneratorState::Pareto { pareto } => {
                let sample: f64 = pareto.sample(&mut self.rng);
                let key = ((sample - 1.0) * (universe as f64 / 10.0)) as u64;
                key.min(universe - 1)
            },

            GeneratorState::ScanResistance {
                zipf,
                scan_start_prob,
                scan_length,
                in_scan,
                ops_remaining,
                start_key,
            } => {
                if !*in_scan && self.rng.random::<f64>() < *scan_start_prob {
                    *in_scan = true;
                    *ops_remaining = *scan_length;
                    *start_key = self.rng.random::<u64>() % universe;
                }
                if *in_scan {
                    let key = (*start_key + (*scan_length - *ops_remaining)) % universe;
                    *ops_remaining -= 1;
                    if *ops_remaining == 0 {
                        *in_scan = false;
                    }
                    key
                } else {
                    let sample: f64 = zipf.sample(&mut self.rng);
                    (sample as u64).saturating_sub(1).min(universe - 1)
                }
            },

            GeneratorState::Correlated {
                stride,
                burst_len,
                burst_prob,
                burst_remaining,
                burst_start_key,
            } => {
                if *burst_remaining > 0 {
                    let key =
                        (*burst_start_key + (*burst_len - *burst_remaining) * *stride) % universe;
                    *burst_remaining -= 1;
                    key
                } else if self.rng.random::<f64>() < *burst_prob {
                    *burst_remaining = burst_len.saturating_sub(1);
                    *burst_start_key = self.rng.random::<u64>() % universe;
                    *burst_start_key
                } else {
                    self.rng.random::<u64>() % universe
                }
            },

            GeneratorState::Loop {
                working_set_size,
                pos,
            } => {
                let key = *pos % *working_set_size;
                *pos = pos.wrapping_add(1);
                key
            },

            GeneratorState::WorkingSetChurn {
                working_set_size,
                churn_rate,
                base,
            } => {
                if self.rng.random::<f64>() < *churn_rate {
                    let span = (universe - *working_set_size + 1).max(1);
                    *base = (*base + 1) % span;
                }
                let offset = self.rng.random::<u64>() % *working_set_size;
                (*base + offset) % universe
            },

            GeneratorState::Bursty {
                zipf,
                state_persistence,
                active,
            } => {
                if *active {
                    if self.rng.random::<f64>() > *state_persistence {
                        *active = false;
                    }
                } else if self.rng.random::<f64>() < (1.0 - *state_persistence) * 0.1 {
                    *active = true;
                }
                let sample: f64 = zipf.sample(&mut self.rng);
                let key = (sample as u64).saturating_sub(1).min(universe - 1);
                if *active {
                    key % (universe / 10).max(1)
                } else {
                    key
                }
            },

            GeneratorState::FlashCrowd {
                zipf,
                flash_prob,
                flash_duration,
                flash_keys,
                flash_intensity,
                active,
                ops_remaining,
                base_key,
            } => {
                if !*active && self.rng.random::<f64>() < *flash_prob {
                    *active = true;
                    *ops_remaining = *flash_duration;
                    *base_key = self.rng.random::<u64>() % universe;
                }
                if *active {
                    *ops_remaining -= 1;
                    if *ops_remaining == 0 {
                        *active = false;
                    }
                    if self.rng.random::<f64>() < *flash_intensity / (*flash_intensity + 1.0) {
                        let offset = self.rng.random::<u64>() % *flash_keys;
                        (*base_key + offset) % universe
                    } else {
                        let sample: f64 = zipf.sample(&mut self.rng);
                        (sample as u64).saturating_sub(1).min(universe - 1)
                    }
                } else {
                    let sample: f64 = zipf.sample(&mut self.rng);
                    (sample as u64).saturating_sub(1).min(universe - 1)
                }
            },

            GeneratorState::Mixture { scan_pos } => {
                // Default mixture: 70% Pareto-1 (proxy for skewed traffic),
                // 20% sequential scan, 10% uniform random.
                let r = self.rng.random::<f64>();
                if r < 0.7 {
                    let rank = (1.0 / self.rng.random::<f64>().max(0.001)).min(universe as f64);
                    (rank as u64).saturating_sub(1).min(universe - 1)
                } else if r < 0.9 {
                    let key = *scan_pos;
                    *scan_pos = (*scan_pos + 1) % universe;
                    key
                } else {
                    self.rng.random::<u64>() % universe
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
    C: Cache<u64, Arc<V>>,
    F: Fn(u64) -> Arc<V>,
{
    let mut op_model = ReadThrough::new(1.0, 0);
    let counts = run_operations(cache, generator, operations, &mut op_model, value_for_key);
    HitRate {
        hits: counts.hits,
        misses: counts.misses,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gen(universe: u64, workload: Workload) -> WorkloadGenerator {
        WorkloadSpec {
            universe,
            workload,
            seed: 1,
        }
        .generator()
    }

    #[test]
    fn keys_stay_within_universe() {
        let universe = 1024;
        let workloads = [
            Workload::Uniform,
            Workload::Scan,
            Workload::HotSet {
                hot_fraction: 0.1,
                hot_prob: 0.9,
            },
            Workload::Zipfian { exponent: 1.0 },
            Workload::ScrambledZipfian { exponent: 1.0 },
            Workload::Latest { exponent: 1.0 },
            Workload::ShiftingHotspot {
                shift_interval: 100,
                hot_fraction: 0.1,
            },
            Workload::Exponential { lambda: 0.05 },
            Workload::Pareto { shape: 1.5 },
            Workload::ScanResistance {
                scan_start_prob: 0.1,
                scan_length: 16,
                point_exponent: 1.0,
            },
            Workload::Correlated {
                stride: 1,
                burst_len: 16,
                burst_prob: 0.1,
            },
            Workload::Loop {
                working_set_size: 64,
            },
            Workload::WorkingSetChurn {
                working_set_size: 64,
                churn_rate: 0.01,
            },
            Workload::Bursty {
                hurst: 0.7,
                base_exponent: 1.0,
            },
            Workload::FlashCrowd {
                base_exponent: 1.0,
                flash_prob: 0.05,
                flash_duration: 8,
                flash_keys: 4,
                flash_intensity: 10.0,
            },
            Workload::Mixture,
        ];
        for workload in workloads {
            let mut g = make_gen(universe, workload);
            for _ in 0..2_000 {
                let k = g.next_key();
                assert!(
                    k < universe,
                    "{workload:?} produced key {k} outside [0, {universe})",
                );
            }
        }
    }

    #[test]
    fn scan_walks_universe_in_order() {
        let mut g = make_gen(8, Workload::Scan);
        let seq: Vec<u64> = (0..16).map(|_| g.next_key()).collect();
        assert_eq!(seq, vec![0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn record_insert_only_affects_latest() {
        // For a non-Latest workload, record_insert should be a no-op and
        // not perturb the deterministic key stream.
        let mut g1 = make_gen(64, Workload::Zipfian { exponent: 1.0 });
        let mut g2 = make_gen(64, Workload::Zipfian { exponent: 1.0 });
        for _ in 0..50 {
            g2.record_insert();
            assert_eq!(g1.next_key(), g2.next_key());
        }
    }

    #[test]
    fn scan_resistance_emits_contiguous_run_during_scan() {
        // Pin scan_start_prob=1.0 so the very first call enters scan mode and
        // produces `scan_length` consecutive keys (mod universe).
        let mut g = make_gen(
            1024,
            Workload::ScanResistance {
                scan_start_prob: 1.0,
                scan_length: 8,
                point_exponent: 1.0,
            },
        );
        let first = g.next_key();
        for offset in 1..8u64 {
            assert_eq!(g.next_key(), (first + offset) % 1024);
        }
    }

    #[test]
    fn generator_size_is_reasonable() {
        // The previous god-struct was ~360+ bytes per generator (5 Option<Distribution>
        // fields plus per-variant scratch state for every workload). Per-variant
        // state should keep this well under that. The exact threshold is host-
        // and rustc-version-dependent, so we just guard against a major regression.
        let size = std::mem::size_of::<WorkloadGenerator>();
        assert!(
            size <= 200,
            "WorkloadGenerator grew to {size} bytes; expected <= 200",
        );
    }

    #[test]
    fn loop_cycles_through_working_set() {
        let mut g = make_gen(
            1024,
            Workload::Loop {
                working_set_size: 4,
            },
        );
        let seq: Vec<u64> = (0..10).map(|_| g.next_key()).collect();
        assert_eq!(seq, vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1]);
    }
}
