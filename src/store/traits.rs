//! Storage backends for cache policies.
//!
//! Stores focus on key/value ownership and lookup semantics, while policies
//! manage eviction order and metadata. This keeps policy logic independent
//! of how values are stored (e.g., HashMap, concurrent map, arena).
//!
//! ## Ownership Model
//!
//! Single-threaded stores (`StoreCore` + `StoreMut`) use direct ownership:
//! - Store owns `K` and `V` after insertion
//! - `get` returns `&V` (zero overhead)
//! - `remove` returns owned `V`
//!
//! Concurrent stores (`ConcurrentStore`) use shared ownership:
//! - Store holds `Arc<V>` internally
//! - `get` returns `Arc<V>` (can outlive the lock)
//! - Required because references can't outlive lock guards

use std::sync::Arc;

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

/// Error returned when a store is at capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoreFull;

// =============================================================================
// Single-threaded store traits
// =============================================================================

/// Read-only store operations for single-threaded backends.
///
/// Returns borrowed references to avoid allocation overhead.
pub trait StoreCore<K, V> {
    /// Fetch a reference to a value by key.
    fn get(&self, key: &K) -> Option<&V>;

    /// Check if a key exists.
    fn contains(&self, key: &K) -> bool;

    /// Current number of entries.
    fn len(&self) -> usize;

    /// Check if the store is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Maximum entries allowed.
    fn capacity(&self) -> usize;

    /// Snapshot the store's current metrics.
    fn metrics(&self) -> StoreMetrics {
        StoreMetrics::default()
    }
}

/// Mutable store operations for single-threaded backends.
///
/// Takes and returns owned values directly—no `Arc` overhead.
pub trait StoreMut<K, V>: StoreCore<K, V> {
    /// Insert or update a value. Returns the previous value if present.
    ///
    /// Returns `StoreFull` if at capacity and inserting a new key.
    fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>, StoreFull>;

    /// Remove a value by key, returning the owned value.
    fn remove(&mut self, key: &K) -> Option<V>;

    /// Remove all entries.
    fn clear(&mut self);
}

// =============================================================================
// Concurrent store traits
// =============================================================================

/// Read-only store operations for concurrent backends.
///
/// Returns `Arc<V>` because references can't outlive lock guards.
pub trait ConcurrentStoreRead<K, V>: Send + Sync {
    /// Fetch a shared reference to a value by key.
    fn get(&self, key: &K) -> Option<Arc<V>>;

    /// Check if a key exists.
    fn contains(&self, key: &K) -> bool;

    /// Current number of entries.
    fn len(&self) -> usize;

    /// Check if the store is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Maximum entries allowed.
    fn capacity(&self) -> usize;

    /// Snapshot the store's current metrics.
    fn metrics(&self) -> StoreMetrics {
        StoreMetrics::default()
    }
}

/// Mutable store operations for concurrent backends (interior mutability).
///
/// Uses `Arc<V>` for values since multiple threads may hold references.
pub trait ConcurrentStore<K, V>: ConcurrentStoreRead<K, V> {
    /// Insert or update a value. Returns the previous value if present.
    ///
    /// Returns `StoreFull` if at capacity and inserting a new key.
    fn try_insert(&self, key: K, value: Arc<V>) -> Result<Option<Arc<V>>, StoreFull>;

    /// Remove a value by key.
    fn remove(&self, key: &K) -> Option<Arc<V>>;

    /// Remove all entries.
    fn clear(&self);
}

// =============================================================================
// Factory traits
// =============================================================================

/// Factory trait for creating single-threaded store instances.
pub trait StoreFactory<K, V> {
    type Store: StoreCore<K, V>;

    /// Create a new store with the specified capacity.
    fn create(capacity: usize) -> Self::Store;
}

/// Factory trait for creating concurrent store instances.
pub trait ConcurrentStoreFactory<K, V> {
    type Store: ConcurrentStore<K, V>;

    /// Create a new store with the specified capacity.
    fn create(capacity: usize) -> Self::Store;
}
