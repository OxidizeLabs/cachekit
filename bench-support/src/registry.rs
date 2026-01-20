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
/// ```ignore
/// for_each_policy! {
///     with |policy_id, display_name, make_cache| {
///         // Your code here using the bindings
///         let mut cache = make_cache(CAPACITY);
///         // ... benchmark code ...
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
        use cachekit::policy::clock::ClockCache;
        use cachekit::policy::heap_lfu::HeapLfuCache;
        use cachekit::policy::lfu::LfuCache;
        use cachekit::policy::lru::LruCore;
        use cachekit::policy::lru_k::LrukCache;
        use cachekit::policy::s3_fifo::S3FifoCache;
        use cachekit::policy::two_q::TwoQCore;
        use std::sync::Arc;

        {
            let $policy_id = "lru";
            let $display_name = "LRU";
            let $make_cache = |cap: usize| LruCore::<u64, u64>::new(cap);
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
            let $policy_id = "clock";
            let $display_name = "Clock";
            let $make_cache = |cap: usize| ClockCache::<u64, Arc<u64>>::new(cap);
            $body
        }
        {
            let $policy_id = "s3_fifo";
            let $display_name = "S3-FIFO";
            let $make_cache = |cap: usize| S3FifoCache::<u64, Arc<u64>>::new(cap);
            $body
        }
        {
            let $policy_id = "two_q";
            let $display_name = "2Q";
            let $make_cache = |cap: usize| TwoQCore::<u64, Arc<u64>>::new(cap, 0.25);
            $body
        }
    }};
}

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
            scan_fraction: 0.2,
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
            scan_fraction: 0.2,
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
