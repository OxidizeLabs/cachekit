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
//!                                ┌─────────────────────────────┐
//!                                │     CoreMetricsRecorder     │
//!                                │  get_hit/get_miss/insert    │
//!                                │  evict/clear                │
//!                                └──────────────┬──────────────┘
//!                                               │
//!                     ┌─────────────────────────┼─────────────────────────┐
//!                     │                         │                         │
//!                     ▼                         ▼                         ▼
//!       ┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐
//!       │  FifoMetricsRecorder    │  │  LruMetricsRecorder     │  │  LfuMetricsRecorder     │
//!       │  pop_oldest/peek/age    │  │  pop_lru/peek/touch     │  │  pop_lfu/peek/frequency │
//!       └─────────────────────────┘  └─────────────────────────┘  └─────────────────────────┘
//!                                               │
//!                                               ▼
//!                                   ┌─────────────────────────┐
//!                                   │  LruKMetricsRecorder    │
//!                                   │  pop_lru_k/k_distance   │
//!                                   └─────────────────────────┘
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
