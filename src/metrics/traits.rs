//! # Metrics Trait Hierarchy
//!
//! This module mirrors the cache trait design by separating *recording*,
//! *snapshotting*, and *export* responsibilities into small, composable traits.
//! It enables production monitoring and bench/testing without coupling those
//! concerns to cache policy logic.
//!
//! ## Architecture
//!
//! ```text
//!                                     ┌─────────────────────────────┐
//!                                     │     CoreMetricsRecorder     │
//!                                     │  get_hit/get_miss/insert    │
//!                                     │  evict/clear                │
//!                                     └──────────────┬──────────────┘
//!                                                    │
//!     ┌──────────────┬───────────────┬───────────────┼───────────────┬───────────────┐
//!     │              │               │               │               │               │
//!     ▼              ▼               ▼               ▼               ▼               ▼
//!  ┌────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐
//!  │  Fifo  │  │   Lru    │  │   Lfu    │  │   Arc    │  │  Clock   │  │  S3Fifo/     │
//!  │Recorder│  │ Recorder │  │ Recorder │  │ Recorder │  │ Recorder │  │  Car/Slru/   │
//!  └────────┘  └────┬─────┘  └──────────┘  └──────────┘  └──────────┘  │  TwoQ/Mfu/   │
//!                    │                                                   │  NRU/ClkPro  │
//!                    ▼                                                   └──────────────┘
//!              ┌──────────┐
//!              │  LruK    │
//!              │ Recorder │
//!              └──────────┘
//!
//!   Consumption (decoupled from recording):
//!   ┌──────────────────────────────┐    ┌──────────────────────────────┐
//!   │ MetricsSnapshotProvider<S>   │    │ MetricsExporter<S>           │
//!   │ (bench/test)                 │    │ (production monitoring)      │
//!   └──────────────────────────────┘    └──────────────────────────────┘
//! ```
//!
//! ## Design Goals
//! - **Single responsibility**: recorders only write counters; providers only
//!   read/snapshot; exporters only publish to monitoring systems.
//! - **Shared hierarchy**: policy metrics extend the core recorder to reuse
//!   shared counters while adding policy-specific signals.
//! - **Environment split**:
//!   - Production: use lightweight recorders + exporters.
//!   - Bench/Test: use snapshot providers + resettable metrics.

/// Common counters for any cache policy.
pub trait CoreMetricsRecorder {
    fn record_get_hit(&mut self);
    fn record_get_miss(&mut self);
    fn record_insert_call(&mut self);
    fn record_insert_new(&mut self);
    fn record_insert_update(&mut self);
    fn record_evict_call(&mut self);
    fn record_evicted_entry(&mut self);
    fn record_clear(&mut self);
}

/// Metrics for FIFO behavior (insertion order).
pub trait FifoMetricsRecorder: CoreMetricsRecorder {
    fn record_evict_scan_step(&mut self);
    fn record_stale_skip(&mut self);
    fn record_pop_oldest_call(&mut self);
    fn record_pop_oldest_found(&mut self);
    fn record_pop_oldest_empty_or_stale(&mut self);
}

/// Read-only FIFO metrics for &self methods (uses interior mutability).
///
/// Use this for cache operations that only take `&self` (e.g., `peek_oldest`,
/// `age_rank`) where a mutable recorder is not available.
pub trait FifoMetricsReadRecorder {
    fn record_peek_oldest_call(&self);
    fn record_peek_oldest_found(&self);
    fn record_age_rank_call(&self);
    fn record_age_rank_found(&self);
    fn record_age_rank_scan_step(&self);
}

/// Metrics for LRU behavior (recency order).
pub trait LruMetricsRecorder: CoreMetricsRecorder {
    fn record_pop_lru_call(&mut self);
    fn record_pop_lru_found(&mut self);
    fn record_peek_lru_call(&mut self);
    fn record_peek_lru_found(&mut self);
    fn record_touch_call(&mut self);
    fn record_touch_found(&mut self);
    fn record_recency_rank_call(&mut self);
    fn record_recency_rank_found(&mut self);
    fn record_recency_rank_scan_step(&mut self);
}

/// Read-only LRU metrics for &self methods (uses interior mutability).
pub trait LruMetricsReadRecorder {
    fn record_peek_lru_call(&self);
    fn record_peek_lru_found(&self);
    fn record_recency_rank_call(&self);
    fn record_recency_rank_found(&self);
    fn record_recency_rank_scan_step(&self);
}

/// Metrics for LFU behavior (frequency order).
pub trait LfuMetricsRecorder: CoreMetricsRecorder {
    fn record_pop_lfu_call(&mut self);
    fn record_pop_lfu_found(&mut self);
    fn record_peek_lfu_call(&mut self);
    fn record_peek_lfu_found(&mut self);
    fn record_frequency_call(&mut self);
    fn record_frequency_found(&mut self);
    fn record_reset_frequency_call(&mut self);
    fn record_reset_frequency_found(&mut self);
    fn record_increment_frequency_call(&mut self);
    fn record_increment_frequency_found(&mut self);
}

/// Read-only LFU metrics for &self methods (uses interior mutability).
pub trait LfuMetricsReadRecorder {
    fn record_peek_lfu_call(&self);
    fn record_peek_lfu_found(&self);
    fn record_frequency_call(&self);
    fn record_frequency_found(&self);
}

/// Metrics for LRU-K behavior (K-distance order).
pub trait LruKMetricsRecorder: LruMetricsRecorder {
    fn record_pop_lru_k_call(&mut self);
    fn record_pop_lru_k_found(&mut self);
    fn record_peek_lru_k_call(&mut self);
    fn record_peek_lru_k_found(&mut self);
    fn record_k_distance_call(&mut self);
    fn record_k_distance_found(&mut self);
    fn record_k_distance_rank_call(&mut self);
    fn record_k_distance_rank_found(&mut self);
    fn record_k_distance_rank_scan_step(&mut self);
}

/// Read-only LRU-K metrics for &self methods (uses interior mutability).
pub trait LruKMetricsReadRecorder {
    fn record_peek_lru_k_call(&self);
    fn record_peek_lru_k_found(&self);
    fn record_k_distance_call(&self);
    fn record_k_distance_found(&self);
    fn record_k_distance_rank_call(&self);
    fn record_k_distance_rank_found(&self);
    fn record_k_distance_rank_scan_step(&self);
}

/// Metrics for ARC behavior (adaptive replacement with ghost lists).
pub trait ArcMetricsRecorder: CoreMetricsRecorder {
    fn record_t1_to_t2_promotion(&mut self);
    fn record_b1_ghost_hit(&mut self);
    fn record_b2_ghost_hit(&mut self);
    fn record_p_increase(&mut self);
    fn record_p_decrease(&mut self);
    fn record_t1_eviction(&mut self);
    fn record_t2_eviction(&mut self);
}

/// Metrics for CAR behavior (clock with adaptive replacement).
pub trait CarMetricsRecorder: CoreMetricsRecorder {
    fn record_recent_to_frequent_promotion(&mut self);
    fn record_ghost_recent_hit(&mut self);
    fn record_ghost_frequent_hit(&mut self);
    fn record_target_increase(&mut self);
    fn record_target_decrease(&mut self);
    fn record_hand_sweep(&mut self);
}

/// Metrics for Clock behavior (clock hand sweep).
pub trait ClockMetricsRecorder: CoreMetricsRecorder {
    fn record_hand_advance(&mut self);
    fn record_ref_bit_reset(&mut self);
}

/// Metrics for Clock-PRO behavior (hot/cold/test states).
pub trait ClockProMetricsRecorder: CoreMetricsRecorder {
    fn record_cold_to_hot_promotion(&mut self);
    fn record_hot_to_cold_demotion(&mut self);
    fn record_test_insertion(&mut self);
    fn record_test_hit(&mut self);
}

/// Metrics for MFU behavior (most frequently used eviction).
pub trait MfuMetricsRecorder: CoreMetricsRecorder {
    fn record_pop_mfu_call(&mut self);
    fn record_pop_mfu_found(&mut self);
    fn record_peek_mfu_call(&mut self);
    fn record_peek_mfu_found(&mut self);
    fn record_frequency_call(&mut self);
    fn record_frequency_found(&mut self);
}

/// Read-only MFU metrics for &self methods (uses interior mutability).
pub trait MfuMetricsReadRecorder {
    fn record_peek_mfu_call(&self);
    fn record_peek_mfu_found(&self);
    fn record_frequency_call(&self);
    fn record_frequency_found(&self);
}

/// Metrics for NRU behavior (not recently used, clock sweep).
pub trait NruMetricsRecorder: CoreMetricsRecorder {
    fn record_sweep_step(&mut self);
    fn record_ref_bit_reset(&mut self);
}

/// Metrics for SLRU behavior (segmented LRU).
pub trait SlruMetricsRecorder: CoreMetricsRecorder {
    fn record_probationary_to_protected(&mut self);
    fn record_protected_eviction(&mut self);
}

/// Metrics for Two-Q behavior (A1in/A1out/Am queues).
pub trait TwoQMetricsRecorder: CoreMetricsRecorder {
    fn record_a1in_to_am_promotion(&mut self);
    fn record_a1out_ghost_hit(&mut self);
}

/// Metrics for S3-FIFO behavior (small/main/ghost queues).
pub trait S3FifoMetricsRecorder: CoreMetricsRecorder {
    fn record_promotion(&mut self);
    fn record_main_reinsert(&mut self);
    fn record_small_eviction(&mut self);
    fn record_main_eviction(&mut self);
    fn record_ghost_hit(&mut self);
}

/// Snapshot provider for bench/testing.
pub trait MetricsSnapshotProvider<S> {
    fn snapshot(&self) -> S;
}

/// Reset metrics between tests or benchmark iterations.
pub trait MetricsReset {
    fn reset_metrics(&self);
}

/// Export/publish metrics to production monitoring backends.
pub trait MetricsExporter<S> {
    fn export(&self, snapshot: &S);
}

/// Shared counters for any policy.
#[derive(Debug, Default, Clone, Copy)]
pub struct CoreMetricsSnapshot {
    pub get_calls: u64,
    pub get_hits: u64,
    pub get_misses: u64,
    pub insert_calls: u64,
    pub insert_updates: u64,
    pub insert_new: u64,
    pub evict_calls: u64,
    pub evicted_entries: u64,
}

/// FIFO-specific snapshot composed from the core snapshot.
#[derive(Debug, Default, Clone, Copy)]
pub struct FifoMetricsSnapshot {
    pub core: CoreMetricsSnapshot,
    pub pop_oldest_calls: u64,
    pub pop_oldest_found: u64,
    pub pop_oldest_empty_or_stale: u64,
    pub peek_oldest_calls: u64,
    pub peek_oldest_found: u64,
    pub age_rank_calls: u64,
    pub age_rank_found: u64,
    pub age_rank_scan_steps: u64,
}
