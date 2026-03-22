//! Point-in-time snapshots of cache metrics counters.
//!
//! Each eviction policy has a dedicated snapshot struct that captures both the
//! core counters (gets, inserts, evictions) and policy-specific signals at the
//! moment [`MetricsSnapshotProvider::snapshot`] is called.
//!
//! Snapshots are cheap `Copy` value types intended for assertion in tests,
//! export via [`PrometheusTextExporter`], or ad-hoc inspection with `Debug`.
//!
//! ## Example
//!
//! ```
//! use cachekit::metrics::snapshot::CoreOnlyMetricsSnapshot;
//!
//! let snap = CoreOnlyMetricsSnapshot::default();
//! assert_eq!(snap.get_calls, snap.get_hits + snap.get_misses);
//! ```
//!
//! [`MetricsSnapshotProvider::snapshot`]: crate::metrics::traits::MetricsSnapshotProvider::snapshot
//! [`PrometheusTextExporter`]: crate::metrics::exporter::PrometheusTextExporter

/// FIFO / insertion-order cache metrics.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CacheMetricsSnapshot {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,

    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,

    pub evict_calls: u64,
    pub evicted_entries: u64,
    /// Queue entries popped that were already removed from the map.
    pub stale_skips: u64,
    /// How many `pop_front` iterations inside a single eviction call.
    pub evict_scan_steps: u64,

    pub pop_oldest_calls: u64,
    pub pop_oldest_found: u64,
    pub pop_oldest_empty_or_stale: u64,

    pub peek_oldest_calls: u64,
    pub peek_oldest_found: u64,

    pub age_rank_calls: u64,
    pub age_rank_found: u64,
    pub age_rank_scan_steps: u64,

    /// Current number of entries in the cache (gauge).
    pub cache_len: usize,
    /// Current length of the insertion-order queue (gauge).
    pub insertion_order_len: usize,
    /// Configured maximum capacity (gauge).
    pub capacity: usize,
}

/// LRU (Least Recently Used) cache metrics.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LruMetricsSnapshot {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,

    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,

    pub evict_calls: u64,
    pub evicted_entries: u64,

    pub pop_lru_calls: u64,
    pub pop_lru_found: u64,
    pub peek_lru_calls: u64,
    pub peek_lru_found: u64,
    pub touch_calls: u64,
    pub touch_found: u64,
    pub recency_rank_calls: u64,
    pub recency_rank_found: u64,
    pub recency_rank_scan_steps: u64,

    pub cache_len: usize,
    pub capacity: usize,
}

/// LFU (Least Frequently Used) cache metrics.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LfuMetricsSnapshot {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,

    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,

    pub evict_calls: u64,
    pub evicted_entries: u64,

    pub pop_lfu_calls: u64,
    pub pop_lfu_found: u64,
    pub peek_lfu_calls: u64,
    pub peek_lfu_found: u64,
    pub frequency_calls: u64,
    pub frequency_found: u64,
    pub reset_frequency_calls: u64,
    pub reset_frequency_found: u64,
    pub increment_frequency_calls: u64,
    pub increment_frequency_found: u64,

    pub cache_len: usize,
    pub capacity: usize,
}

/// LRU-K cache metrics, combining standard LRU counters with K-distance tracking.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LruKMetricsSnapshot {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,

    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,

    pub evict_calls: u64,
    pub evicted_entries: u64,

    pub pop_lru_calls: u64,
    pub pop_lru_found: u64,
    pub peek_lru_calls: u64,
    pub peek_lru_found: u64,
    pub touch_calls: u64,
    pub touch_found: u64,
    pub recency_rank_calls: u64,
    pub recency_rank_found: u64,
    pub recency_rank_scan_steps: u64,

    pub pop_lru_k_calls: u64,
    pub pop_lru_k_found: u64,
    pub peek_lru_k_calls: u64,
    pub peek_lru_k_found: u64,
    pub k_distance_calls: u64,
    pub k_distance_found: u64,
    pub k_distance_rank_calls: u64,
    pub k_distance_rank_found: u64,
    pub k_distance_rank_scan_steps: u64,

    pub cache_len: usize,
    pub capacity: usize,
}

/// Core-only metrics for policies that add no policy-specific counters.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CoreOnlyMetricsSnapshot {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,

    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,

    pub evict_calls: u64,
    pub evicted_entries: u64,

    pub cache_len: usize,
    pub capacity: usize,
}

/// ARC (Adaptive Replacement Cache) metrics.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ArcMetricsSnapshot {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,

    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,

    pub evict_calls: u64,
    pub evicted_entries: u64,

    pub t1_to_t2_promotions: u64,
    pub b1_ghost_hits: u64,
    pub b2_ghost_hits: u64,
    pub p_increases: u64,
    pub p_decreases: u64,
    pub t1_evictions: u64,
    pub t2_evictions: u64,

    pub cache_len: usize,
    pub capacity: usize,
}

/// CAR (Clock with Adaptive Replacement) cache metrics.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CarMetricsSnapshot {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,

    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,

    pub evict_calls: u64,
    pub evicted_entries: u64,

    pub recent_to_frequent_promotions: u64,
    pub ghost_recent_hits: u64,
    pub ghost_frequent_hits: u64,
    pub target_increases: u64,
    pub target_decreases: u64,
    pub hand_sweeps: u64,

    pub cache_len: usize,
    pub capacity: usize,
}

/// CLOCK cache metrics.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ClockMetricsSnapshot {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,

    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,

    pub evict_calls: u64,
    pub evicted_entries: u64,

    pub hand_advances: u64,
    pub ref_bit_resets: u64,

    pub cache_len: usize,
    pub capacity: usize,
}

/// CLOCK-Pro cache metrics.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ClockProMetricsSnapshot {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,

    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,

    pub evict_calls: u64,
    pub evicted_entries: u64,

    pub cold_to_hot_promotions: u64,
    pub hot_to_cold_demotions: u64,
    pub test_insertions: u64,
    pub test_hits: u64,

    pub cache_len: usize,
    pub capacity: usize,
}

/// MFU (Most Frequently Used) cache metrics.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MfuMetricsSnapshot {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,

    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,

    pub evict_calls: u64,
    pub evicted_entries: u64,

    pub pop_mfu_calls: u64,
    pub pop_mfu_found: u64,
    pub peek_mfu_calls: u64,
    pub peek_mfu_found: u64,
    pub frequency_calls: u64,
    pub frequency_found: u64,

    pub cache_len: usize,
    pub capacity: usize,
}

/// NRU (Not Recently Used) cache metrics.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NruMetricsSnapshot {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,

    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,

    pub evict_calls: u64,
    pub evicted_entries: u64,

    pub sweep_steps: u64,
    pub ref_bit_resets: u64,

    pub cache_len: usize,
    pub capacity: usize,
}

/// SLRU (Segmented LRU) cache metrics.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SlruMetricsSnapshot {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,

    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,

    pub evict_calls: u64,
    pub evicted_entries: u64,

    pub probationary_to_protected: u64,
    pub protected_evictions: u64,

    pub cache_len: usize,
    pub capacity: usize,
}

/// 2Q (Two-Queue) cache metrics.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TwoQMetricsSnapshot {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,

    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,

    pub evict_calls: u64,
    pub evicted_entries: u64,

    pub a1in_to_am_promotions: u64,
    pub a1out_ghost_hits: u64,

    pub cache_len: usize,
    pub capacity: usize,
}

/// S3-FIFO cache metrics.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct S3FifoMetricsSnapshot {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,

    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,

    pub evict_calls: u64,
    pub evicted_entries: u64,

    pub promotions: u64,
    pub main_reinserts: u64,
    pub small_evictions: u64,
    pub main_evictions: u64,
    pub ghost_hits: u64,

    pub cache_len: usize,
    pub capacity: usize,
}
