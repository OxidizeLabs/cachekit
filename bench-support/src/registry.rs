//! Central registry for benchmark policies and workloads.
//!
//! This is the single source of truth for:
//! - Policy definitions (id, display name, constructor)
//! - Workload definitions (id, display name, spec)
//!
//! To add a new policy or workload, modify this file only.
//! All benchmarks and reports automatically pick up the changes.

use crate::workload::{Workload, WorkloadSpec};

// ============================================================================
// Policy Registry
// ============================================================================

/// Macro to execute monomorphic code for each policy.
///
/// This avoids dynamic dispatch in benchmark hot paths while keeping
/// policy iteration centralized.
///
/// # Usage
///
/// ```no_run
/// # const CAPACITY: usize = 1024;
/// use bench_support::for_each_policy;
/// for_each_policy! {
///     with |policy_id, display_name, make_cache| {
///         let mut _cache = make_cache(CAPACITY);
///         println!("{policy_id} ({display_name})");
///     }
/// }
/// ```
///
/// The macro expands to separate code blocks for each policy with monomorphic types.
/// Each block defines:
/// - `policy_id`: &str - policy identifier
/// - `display_name`: &str - human-readable name
/// - `make_cache`: closure that creates a cache instance
#[macro_export]
macro_rules! for_each_policy {
    (with |$policy_id:ident, $display_name:ident, $make_cache:ident| $body:block) => {{
        use cachekit::policy::arc::ArcCore;
        use cachekit::policy::car::CarCore;
        use cachekit::policy::clock::ClockCache;
        use cachekit::policy::clock_pro::ClockProCache;
        use cachekit::policy::fast_lru::FastLru;
        use cachekit::policy::fifo::FifoCache;
        use cachekit::policy::heap_lfu::HeapLfuCache;
        use cachekit::policy::lfu::LfuCache;
        use cachekit::policy::lifo::LifoCore;
        use cachekit::policy::lru::LruCore;
        use cachekit::policy::lru_k::LrukCache;
        use cachekit::policy::mfu::MfuCore;
        use cachekit::policy::mru::MruCore;
        use cachekit::policy::nru::NruCache;
        use cachekit::policy::random::RandomCore;
        use cachekit::policy::s3_fifo::S3FifoCache;
        use cachekit::policy::slru::SlruCore;
        use cachekit::policy::two_q::TwoQCore;
        use std::sync::Arc;

        {
            let $policy_id = "lru";
            let $display_name = "LRU";
            let $make_cache = |cap: usize| LruCore::<u64, u64>::new(cap);
            $body
        }
        {
            let $policy_id = "fast_lru";
            let $display_name = "Fast-LRU";
            let $make_cache = |cap: usize| FastLru::<u64, Arc<u64>>::new(cap);
            $body
        }
        {
            let $policy_id = "lru_k";
            let $display_name = "LRU-K";
            let $make_cache = |cap: usize| LrukCache::<u64, Arc<u64>>::new(cap);
            $body
        }
        {
            let $policy_id = "lfu";
            let $display_name = "LFU";
            let $make_cache = |cap: usize| LfuCache::<u64, u64>::new(cap);
            $body
        }
        {
            let $policy_id = "heap_lfu";
            let $display_name = "Heap-LFU";
            let $make_cache = |cap: usize| HeapLfuCache::<u64, u64>::new(cap);
            $body
        }
        {
            let $policy_id = "mfu";
            let $display_name = "MFU";
            let $make_cache = |cap: usize| MfuCore::<u64, Arc<u64>>::new(cap);
            $body
        }
        {
            let $policy_id = "fifo";
            let $display_name = "FIFO";
            let $make_cache = |cap: usize| FifoCache::<u64, Arc<u64>>::new(cap);
            $body
        }
        {
            let $policy_id = "lifo";
            let $display_name = "LIFO";
            let $make_cache = |cap: usize| LifoCore::<u64, Arc<u64>>::new(cap);
            $body
        }
        {
            let $policy_id = "mru";
            let $display_name = "MRU";
            let $make_cache = |cap: usize| MruCore::<u64, Arc<u64>>::new(cap);
            $body
        }
        {
            let $policy_id = "nru";
            let $display_name = "NRU";
            let $make_cache = |cap: usize| NruCache::<u64, Arc<u64>>::new(cap);
            $body
        }
        {
            let $policy_id = "random";
            let $display_name = "Random";
            let $make_cache = |cap: usize| RandomCore::<u64, Arc<u64>>::new(cap);
            $body
        }
        {
            let $policy_id = "clock";
            let $display_name = "Clock";
            let $make_cache = |cap: usize| ClockCache::<u64, Arc<u64>>::new(cap);
            $body
        }
        {
            let $policy_id = "clock_pro";
            let $display_name = "Clock-Pro";
            let $make_cache = |cap: usize| ClockProCache::<u64, Arc<u64>>::new(cap);
            $body
        }
        {
            let $policy_id = "s3_fifo";
            let $display_name = "S3-FIFO";
            let $make_cache = |cap: usize| S3FifoCache::<u64, Arc<u64>>::new(cap);
            $body
        }
        {
            let $policy_id = "slru";
            let $display_name = "SLRU";
            let $make_cache = |cap: usize| SlruCore::<u64, Arc<u64>>::new(cap, 0.2);
            $body
        }
        {
            let $policy_id = "two_q";
            let $display_name = "2Q";
            let $make_cache = |cap: usize| TwoQCore::<u64, Arc<u64>>::new(cap, 0.25);
            $body
        }
        {
            let $policy_id = "arc";
            let $display_name = "ARC";
            let $make_cache = |cap: usize| ArcCore::<u64, Arc<u64>>::new(cap);
            $body
        }
        {
            let $policy_id = "car";
            let $display_name = "CAR";
            let $make_cache = |cap: usize| CarCore::<u64, Arc<u64>>::new(cap);
            $body
        }
    }};
}

/// Display metadata for a policy registered with [`for_each_policy!`].
///
/// `for_each_policy!` is the source of truth for *constructors*, but
/// downstream tooling (chart rendering, comparison tables, CLI flags) needs
/// presentation data that doesn't fit in a macro: stable display name, chart
/// color, etc. Keep one entry here per macro arm. The
/// `policies_metadata_matches_macro` test fails loudly if the two drift.
#[derive(Debug, Clone, Copy)]
pub struct PolicyMeta {
    /// Stable identifier (matches `policy_id` in `for_each_policy!`).
    pub id: &'static str,
    /// Human-readable name (matches `display_name` in `for_each_policy!`).
    pub display_name: &'static str,
    /// Hex color (`#rrggbb`) used when rendering this policy in charts.
    /// Pick perceptually distinct hues; family members (e.g. LRU variants)
    /// can share a hue at different lightness.
    pub color: &'static str,
}

/// All policies, in the same order as [`for_each_policy!`].
///
/// Consumers that need both presentation data and constructors should iterate
/// `POLICIES` here and look up runtime construction via `for_each_policy!`.
pub const POLICIES: &[PolicyMeta] = &[
    PolicyMeta {
        id: "lru",
        display_name: "LRU",
        color: "#3498db",
    },
    PolicyMeta {
        id: "fast_lru",
        display_name: "Fast-LRU",
        color: "#5dade2",
    },
    PolicyMeta {
        id: "lru_k",
        display_name: "LRU-K",
        color: "#2ecc71",
    },
    PolicyMeta {
        id: "lfu",
        display_name: "LFU",
        color: "#e74c3c",
    },
    PolicyMeta {
        id: "heap_lfu",
        display_name: "Heap-LFU",
        color: "#f39c12",
    },
    PolicyMeta {
        id: "mfu",
        display_name: "MFU",
        color: "#c0392b",
    },
    PolicyMeta {
        id: "fifo",
        display_name: "FIFO",
        color: "#16a085",
    },
    PolicyMeta {
        id: "lifo",
        display_name: "LIFO",
        color: "#8e44ad",
    },
    PolicyMeta {
        id: "mru",
        display_name: "MRU",
        color: "#d35400",
    },
    PolicyMeta {
        id: "nru",
        display_name: "NRU",
        color: "#2980b9",
    },
    PolicyMeta {
        id: "random",
        display_name: "Random",
        color: "#7f8c8d",
    },
    PolicyMeta {
        id: "clock",
        display_name: "Clock",
        color: "#9b59b6",
    },
    PolicyMeta {
        id: "clock_pro",
        display_name: "Clock-Pro",
        color: "#27ae60",
    },
    PolicyMeta {
        id: "s3_fifo",
        display_name: "S3-FIFO",
        color: "#1abc9c",
    },
    PolicyMeta {
        id: "slru",
        display_name: "SLRU",
        color: "#34495e",
    },
    PolicyMeta {
        id: "two_q",
        display_name: "2Q",
        color: "#e67e22",
    },
    PolicyMeta {
        id: "arc",
        display_name: "ARC",
        color: "#117a65",
    },
    PolicyMeta {
        id: "car",
        display_name: "CAR",
        color: "#b9770e",
    },
];

// ============================================================================
// Workload Registry
// ============================================================================

/// Workload case with metadata.
#[derive(Debug, Clone, Copy)]
pub struct WorkloadCase {
    /// Short identifier (e.g., "uniform", "zipfian_1.0").
    pub id: &'static str,
    /// Human-readable display name (e.g., "Uniform", "Zipfian 1.0").
    pub display_name: &'static str,
    /// Workload specification (without universe/seed).
    pub workload: Workload,
}

/// Standard workload suite - focused set that differentiates policies.
///
/// This is the primary benchmark set for policy comparison.
pub const STANDARD_WORKLOADS: &[WorkloadCase] = &[
    WorkloadCase {
        id: "uniform",
        display_name: "Uniform",
        workload: Workload::Uniform,
    },
    WorkloadCase {
        id: "hotset_90_10",
        display_name: "HotSet 90/10",
        workload: Workload::HotSet {
            hot_fraction: 0.1,
            hot_prob: 0.9,
        },
    },
    WorkloadCase {
        id: "scan",
        display_name: "Scan",
        workload: Workload::Scan,
    },
    WorkloadCase {
        id: "zipfian_1.0",
        display_name: "Zipfian 1.0",
        workload: Workload::Zipfian { exponent: 1.0 },
    },
    WorkloadCase {
        id: "scrambled_zipf",
        display_name: "Scrambled Zipfian",
        workload: Workload::ScrambledZipfian { exponent: 1.0 },
    },
    WorkloadCase {
        id: "latest",
        display_name: "Latest",
        workload: Workload::Latest { exponent: 0.8 },
    },
    WorkloadCase {
        id: "scan_resistance",
        display_name: "Scan Resistance",
        workload: Workload::ScanResistance {
            scan_start_prob: 0.2,
            scan_length: 1000,
            point_exponent: 1.0,
        },
    },
    WorkloadCase {
        id: "flash_crowd",
        display_name: "Flash Crowd",
        workload: Workload::FlashCrowd {
            base_exponent: 1.0,
            flash_prob: 0.001,
            flash_duration: 1000,
            flash_keys: 10,
            flash_intensity: 100.0,
        },
    },
];

/// Extended workload suite - comprehensive set covering all workload types.
///
/// Use this for exhaustive testing or specialized reports.
pub const EXTENDED_WORKLOADS: &[WorkloadCase] = &[
    WorkloadCase {
        id: "uniform",
        display_name: "Uniform",
        workload: Workload::Uniform,
    },
    WorkloadCase {
        id: "hotset_90_10",
        display_name: "HotSet 90/10",
        workload: Workload::HotSet {
            hot_fraction: 0.1,
            hot_prob: 0.9,
        },
    },
    WorkloadCase {
        id: "scan",
        display_name: "Scan",
        workload: Workload::Scan,
    },
    WorkloadCase {
        id: "zipfian_1.0",
        display_name: "Zipfian 1.0",
        workload: Workload::Zipfian { exponent: 1.0 },
    },
    WorkloadCase {
        id: "zipfian_0.8",
        display_name: "Zipfian 0.8",
        workload: Workload::Zipfian { exponent: 0.8 },
    },
    WorkloadCase {
        id: "scrambled_zipf",
        display_name: "Scrambled Zipfian",
        workload: Workload::ScrambledZipfian { exponent: 1.0 },
    },
    WorkloadCase {
        id: "latest",
        display_name: "Latest",
        workload: Workload::Latest { exponent: 0.8 },
    },
    WorkloadCase {
        id: "shifting_hotspot",
        display_name: "Shifting Hotspot",
        workload: Workload::ShiftingHotspot {
            shift_interval: 10_000,
            hot_fraction: 0.1,
        },
    },
    WorkloadCase {
        id: "exponential",
        display_name: "Exponential",
        workload: Workload::Exponential { lambda: 0.05 },
    },
    WorkloadCase {
        id: "pareto",
        display_name: "Pareto",
        workload: Workload::Pareto { shape: 1.5 },
    },
    WorkloadCase {
        id: "scan_resistance",
        display_name: "Scan Resistance",
        workload: Workload::ScanResistance {
            scan_start_prob: 0.2,
            scan_length: 1000,
            point_exponent: 1.0,
        },
    },
    WorkloadCase {
        id: "correlated",
        display_name: "Correlated",
        workload: Workload::Correlated {
            stride: 1,
            burst_len: 8,
            burst_prob: 0.3,
        },
    },
    WorkloadCase {
        id: "loop_small",
        display_name: "Loop (small)",
        workload: Workload::Loop {
            working_set_size: 512,
        },
    },
    WorkloadCase {
        id: "working_set_churn",
        display_name: "Working Set Churn",
        workload: Workload::WorkingSetChurn {
            working_set_size: 2048,
            churn_rate: 0.001,
        },
    },
    WorkloadCase {
        id: "bursty",
        display_name: "Bursty",
        workload: Workload::Bursty {
            hurst: 0.8,
            base_exponent: 1.0,
        },
    },
    WorkloadCase {
        id: "flash_crowd",
        display_name: "Flash Crowd",
        workload: Workload::FlashCrowd {
            base_exponent: 1.0,
            flash_prob: 0.001,
            flash_duration: 1000,
            flash_keys: 10,
            flash_intensity: 100.0,
        },
    },
    WorkloadCase {
        id: "mixture",
        display_name: "Mixture",
        workload: Workload::Mixture,
    },
];

/// Build a `WorkloadSpec` from a workload case and runtime parameters.
impl WorkloadCase {
    pub fn with_params(self, universe: u64, seed: u64) -> WorkloadSpec {
        WorkloadSpec {
            universe,
            workload: self.workload,
            seed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Walk every `for_each_policy!` arm and assert that `POLICIES` matches
    /// it position-by-position. If you add a policy to the macro, add a
    /// `PolicyMeta` here too (or vice versa) — this test exists so a single
    /// untracked addition can't silently regress chart colors.
    #[test]
    fn policies_metadata_matches_macro() {
        let mut from_macro: Vec<(&'static str, &'static str)> = Vec::new();
        for_each_policy! {
            with |policy_id, display_name, _make_cache| {
                from_macro.push((policy_id, display_name));
            }
        }
        let from_const: Vec<(&'static str, &'static str)> =
            POLICIES.iter().map(|p| (p.id, p.display_name)).collect();
        assert_eq!(
            from_macro, from_const,
            "for_each_policy! and POLICIES drifted; \
             update bench-support/src/registry.rs",
        );
    }

    #[test]
    fn policy_colors_are_well_formed_hex() {
        for meta in POLICIES {
            let bytes = meta.color.as_bytes();
            assert!(
                bytes.first() == Some(&b'#') && bytes.len() == 7,
                "{}: color {:?} must be #rrggbb",
                meta.id,
                meta.color,
            );
            assert!(
                bytes[1..].iter().all(|b| b.is_ascii_hexdigit()),
                "{}: color {:?} has non-hex digits",
                meta.id,
                meta.color,
            );
        }
    }

    #[test]
    fn policy_ids_and_names_are_unique() {
        let mut ids: Vec<&str> = POLICIES.iter().map(|p| p.id).collect();
        let mut names: Vec<&str> = POLICIES.iter().map(|p| p.display_name).collect();
        ids.sort_unstable();
        names.sort_unstable();
        let id_count = ids.len();
        let name_count = names.len();
        ids.dedup();
        names.dedup();
        assert_eq!(ids.len(), id_count, "duplicate policy id in POLICIES");
        assert_eq!(
            names.len(),
            name_count,
            "duplicate display_name in POLICIES",
        );
    }
}
