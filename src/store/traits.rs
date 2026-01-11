//! Storage backends for cache policies.
//!
//! Stores focus on key/value ownership and lookup semantics, while policies
//! manage eviction order and metadata. This keeps policy logic independent
//! of how values are stored (e.g., HashMap, concurrent map, arena).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Snapshot of store-level metrics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StoreMetrics {
    pub hits: u64,
    pub misses: u64,
    pub inserts: u64,
    pub updates: u64,
    pub removes: u64,
    pub evictions: u64,
}

/// Eviction hook for policy code to update store-level eviction metrics.
#[derive(Debug, Default)]
pub struct StoreEvictionHook {
    evictions: AtomicU64,
}

impl StoreEvictionHook {
    pub fn record_eviction(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }

    pub fn metrics(&self) -> StoreMetrics {
        StoreMetrics {
            evictions: self.evictions.load(Ordering::Relaxed),
            ..StoreMetrics::default()
        }
    }
}

/// Error returned when a store is at capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoreFull;

/// Core store operations common to all backends.
pub trait StoreCore<K, V> {
    /// Fetch a value by key.
    fn get(&self, key: &K) -> Option<Arc<V>>;

    /// Check if a key exists.
    fn contains(&self, key: &K) -> bool;

    /// Current number of entries.
    fn len(&self) -> usize;

    /// Check if the store is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Maximum entries allowed by the policy.
    fn capacity(&self) -> usize;

    /// Snapshot the store's current metrics.
    fn metrics(&self) -> StoreMetrics {
        StoreMetrics::default()
    }

    /// Record that the policy evicted an entry.
    fn record_eviction(&self) {}
}

/// Mutable store operations for single-threaded backends.
pub trait StoreMut<K, V>: StoreCore<K, V> {
    /// Insert or update a value. Returns the previous value if present.
    /// Returns `StoreFull` if at capacity and inserting a new key.
    fn try_insert(&mut self, key: K, value: Arc<V>) -> Result<Option<Arc<V>>, StoreFull>;

    /// Remove a value by key.
    fn remove(&mut self, key: &K) -> Option<Arc<V>>;

    /// Remove all entries.
    fn clear(&mut self);
}

/// Mutable store operations for concurrent backends (interior mutability).
pub trait ConcurrentStore<K, V>: StoreCore<K, V> + Send + Sync {
    /// Insert or update a value. Returns the previous value if present.
    /// Returns `StoreFull` if at capacity and inserting a new key.
    fn try_insert(&self, key: K, value: Arc<V>) -> Result<Option<Arc<V>>, StoreFull>;

    /// Remove a value by key.
    fn remove(&self, key: &K) -> Option<Arc<V>>;

    /// Remove all entries.
    fn clear(&self);
}

/// Factory trait for creating store instances.
pub trait StoreFactory<K, V> {
    type Store: StoreCore<K, V>;

    /// Create a new store with the specified capacity.
    fn create(capacity: usize) -> Self::Store;
}
