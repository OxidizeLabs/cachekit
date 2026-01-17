//! Handle-based store for zero-copy policy metadata.
//!
//! ## Architecture
//! - Stores values keyed by compact handles (e.g., interner IDs).
//! - A `HashMap<Handle, Arc<V>>` provides O(1) access.
//! - Policies operate on handles; the interner maps handles back to keys.
//!
//! ```text
//! key -> KeyInterner -> handle -> policy/DS -> eviction -> handle -> KeyInterner -> key
//! ```
//!
//! ## Key Components
//! - `HandleStore`: single-threaded handle-backed store.
//! - `ConcurrentHandleStore`: thread-safe wrapper using `RwLock`.
//! - Metrics counters for hits/misses/updates/evictions.
//!
//! ## Core Operations
//! - `try_insert`: insert or update by handle.
//! - `get`: fetch by handle (updates hit/miss metrics).
//! - `remove`: delete by handle.
//! - `clear`: drop all entries.
//!
//! ## Performance Trade-offs
//! - Avoids cloning large keys by storing handles only.
//! - Extra indirection requires a separate interner for key lookup.
//! - Uses `Arc<V>` values; cloning is cheap but still allocates on insert.
//!
//! ## When to Use
//! - You already use a `KeyInterner` or stable handle IDs.
//! - You want to minimize key cloning for large keys.
//! - Policy metadata is keyed by handle rather than full keys.
//!
//! ## Example Usage
//! ```rust
//! use std::sync::Arc;
//!
//! use cachekit::ds::KeyInterner;
//! use cachekit::store::handle::HandleStore;
//!
//! let mut interner = KeyInterner::new();
//! let handle = interner.intern(&"alpha".to_string());
//! let mut store: HandleStore<u64, String> = HandleStore::new(2);
//! store.try_insert(handle, Arc::new("value".to_string())).unwrap();
//! assert!(store.contains(&handle));
//! assert_eq!(store.get(&handle), Some(Arc::new("value".to_string())));
//! assert_eq!(store.remove(&handle), Some(Arc::new("value".to_string())));
//! assert!(!store.contains(&handle));
//! ```
//!
//! ## Type Constraints
//! - `H: Copy + Eq + Hash` for handle lookup.
//! - Values are stored as `Arc<V>`.
//!
//! ## Thread Safety
//! - `HandleStore` is single-threaded.
//! - `ConcurrentHandleStore` is `Send + Sync` via `RwLock`.
//!
//! ## Implementation Notes
//! - Handles must remain stable for the lifetime of stored entries.
//! - Metrics are stored separately to keep the hot path simple.
//! - Does NOT implement `StoreCore`/`StoreMut` traits (uses `Arc<V>` API).

use std::cell::Cell;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use crate::store::traits::{
    ConcurrentStore, ConcurrentStoreFactory, ConcurrentStoreRead, StoreFull, StoreMetrics,
};

/// Store metrics counters for single-threaded handle stores.
#[derive(Debug, Default)]
struct StoreCounters {
    hits: Cell<u64>,
    misses: Cell<u64>,
    inserts: Cell<u64>,
    updates: Cell<u64>,
    removes: Cell<u64>,
    evictions: Cell<u64>,
}

impl StoreCounters {
    fn snapshot(&self) -> StoreMetrics {
        StoreMetrics {
            hits: self.hits.get(),
            misses: self.misses.get(),
            inserts: self.inserts.get(),
            updates: self.updates.get(),
            removes: self.removes.get(),
            evictions: self.evictions.get(),
        }
    }

    fn inc_hit(&self) {
        self.hits.set(self.hits.get() + 1);
    }

    fn inc_miss(&self) {
        self.misses.set(self.misses.get() + 1);
    }

    fn inc_insert(&self) {
        self.inserts.set(self.inserts.get() + 1);
    }

    fn inc_update(&self) {
        self.updates.set(self.updates.get() + 1);
    }

    fn inc_remove(&self) {
        self.removes.set(self.removes.get() + 1);
    }

    fn inc_eviction(&self) {
        self.evictions.set(self.evictions.get() + 1);
    }
}

/// Store metrics counters for concurrent handle stores.
#[derive(Debug, Default)]
struct ConcurrentStoreCounters {
    hits: AtomicU64,
    misses: AtomicU64,
    inserts: AtomicU64,
    updates: AtomicU64,
    removes: AtomicU64,
    evictions: AtomicU64,
}

impl ConcurrentStoreCounters {
    fn snapshot(&self) -> StoreMetrics {
        StoreMetrics {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            inserts: self.inserts.load(Ordering::Relaxed),
            updates: self.updates.load(Ordering::Relaxed),
            removes: self.removes.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
        }
    }

    fn inc_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_insert(&self) {
        self.inserts.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_update(&self) {
        self.updates.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_remove(&self) {
        self.removes.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_eviction(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }
}

// =============================================================================
// Single-threaded HandleStore
// =============================================================================

/// Store keyed by compact handles (IDs) instead of full keys.
///
/// Uses `Arc<V>` for value sharing with interner patterns.
#[derive(Debug)]
pub struct HandleStore<H, V> {
    map: HashMap<H, Arc<V>>,
    capacity: usize,
    metrics: StoreCounters,
}

impl<H, V> HandleStore<H, V>
where
    H: Copy + Eq + Hash,
{
    /// Create a handle store with a fixed capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::new(),
            capacity,
            metrics: StoreCounters::default(),
        }
    }

    /// Fetch a value by handle.
    pub fn get(&self, handle: &H) -> Option<Arc<V>> {
        match self.map.get(handle).cloned() {
            Some(value) => {
                self.metrics.inc_hit();
                Some(value)
            },
            None => {
                self.metrics.inc_miss();
                None
            },
        }
    }

    /// Fetch a value by handle without updating metrics.
    pub fn peek(&self, handle: &H) -> Option<&Arc<V>> {
        self.map.get(handle)
    }

    /// Check whether a handle exists.
    pub fn contains(&self, handle: &H) -> bool {
        self.map.contains_key(handle)
    }

    /// Insert or update an entry.
    pub fn try_insert(&mut self, handle: H, value: Arc<V>) -> Result<Option<Arc<V>>, StoreFull> {
        if !self.map.contains_key(&handle) && self.map.len() >= self.capacity {
            return Err(StoreFull);
        }
        let previous = self.map.insert(handle, value);
        if previous.is_some() {
            self.metrics.inc_update();
        } else {
            self.metrics.inc_insert();
        }
        Ok(previous)
    }

    /// Remove a value by handle.
    pub fn remove(&mut self, handle: &H) -> Option<Arc<V>> {
        let removed = self.map.remove(handle);
        if removed.is_some() {
            self.metrics.inc_remove();
        }
        removed
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Return the number of entries.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Check if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Return the maximum capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Snapshot store metrics.
    pub fn metrics(&self) -> StoreMetrics {
        self.metrics.snapshot()
    }

    /// Record an eviction.
    pub fn record_eviction(&self) {
        self.metrics.inc_eviction();
    }
}

// =============================================================================
// Concurrent HandleStore
// =============================================================================

/// Concurrent handle store using a `parking_lot::RwLock`.
#[derive(Debug)]
pub struct ConcurrentHandleStore<H, V> {
    map: RwLock<HashMap<H, Arc<V>>>,
    capacity: usize,
    metrics: ConcurrentStoreCounters,
}

impl<H, V> ConcurrentHandleStore<H, V>
where
    H: Copy + Eq + Hash + Send + Sync,
{
    /// Create a concurrent handle store with a fixed capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            map: RwLock::new(HashMap::new()),
            capacity,
            metrics: ConcurrentStoreCounters::default(),
        }
    }

    /// Record an eviction.
    pub fn record_eviction(&self) {
        self.metrics.inc_eviction();
    }
}

impl<H, V> ConcurrentStoreRead<H, V> for ConcurrentHandleStore<H, V>
where
    H: Copy + Eq + Hash + Send + Sync,
    V: Send + Sync,
{
    fn get(&self, key: &H) -> Option<Arc<V>> {
        match self.map.read().get(key).cloned() {
            Some(value) => {
                self.metrics.inc_hit();
                Some(value)
            },
            None => {
                self.metrics.inc_miss();
                None
            },
        }
    }

    fn contains(&self, key: &H) -> bool {
        self.map.read().contains_key(key)
    }

    fn len(&self) -> usize {
        self.map.read().len()
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn metrics(&self) -> StoreMetrics {
        self.metrics.snapshot()
    }
}

impl<H, V> ConcurrentStore<H, V> for ConcurrentHandleStore<H, V>
where
    H: Copy + Eq + Hash + Send + Sync,
    V: Send + Sync,
{
    fn try_insert(&self, key: H, value: Arc<V>) -> Result<Option<Arc<V>>, StoreFull> {
        let mut map = self.map.write();
        if !map.contains_key(&key) && map.len() >= self.capacity {
            return Err(StoreFull);
        }
        let previous = map.insert(key, value);
        if previous.is_some() {
            self.metrics.inc_update();
        } else {
            self.metrics.inc_insert();
        }
        Ok(previous)
    }

    fn remove(&self, key: &H) -> Option<Arc<V>> {
        let removed = self.map.write().remove(key);
        if removed.is_some() {
            self.metrics.inc_remove();
        }
        removed
    }

    fn clear(&self) {
        self.map.write().clear();
    }
}

impl<H, V> ConcurrentStoreFactory<H, V> for ConcurrentHandleStore<H, V>
where
    H: Copy + Eq + Hash + Send + Sync,
    V: Send + Sync,
{
    type Store = ConcurrentHandleStore<H, V>;

    fn create(capacity: usize) -> Self::Store {
        Self::new(capacity)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_store_basic_ops() {
        let mut store = HandleStore::new(2);
        let value = Arc::new("v1".to_string());
        assert_eq!(store.try_insert(1u64, value.clone()), Ok(None));
        assert_eq!(store.get(&1u64), Some(value.clone()));
        assert!(store.contains(&1u64));
        assert_eq!(store.len(), 1);
        assert_eq!(store.capacity(), 2);
        assert_eq!(store.remove(&1u64), Some(value));
        assert!(!store.contains(&1u64));
    }

    #[test]
    fn handle_store_capacity_enforced() {
        let mut store = HandleStore::new(1);
        assert_eq!(store.try_insert(1u64, Arc::new("v1".to_string())), Ok(None));
        assert_eq!(
            store.try_insert(2u64, Arc::new("v2".to_string())),
            Err(StoreFull)
        );
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn concurrent_handle_store_basic_ops() {
        let store = ConcurrentHandleStore::new(2);
        let value = Arc::new("v1".to_string());
        assert_eq!(store.try_insert(1u64, value.clone()), Ok(None));
        assert_eq!(store.get(&1u64), Some(value.clone()));
        assert!(store.contains(&1u64));
        assert_eq!(store.len(), 1);
        assert_eq!(store.capacity(), 2);
        assert_eq!(store.remove(&1u64), Some(value));
        assert!(!store.contains(&1u64));
    }

    #[test]
    fn handle_store_metrics_counts() {
        let mut store = HandleStore::new(2);
        let value = Arc::new("v1".to_string());

        assert_eq!(store.metrics(), StoreMetrics::default());
        assert_eq!(store.get(&1u64), None);
        assert_eq!(store.try_insert(1u64, value.clone()), Ok(None));
        assert_eq!(
            store.try_insert(1u64, value.clone()),
            Ok(Some(value.clone()))
        );
        assert_eq!(store.get(&1u64), Some(value.clone()));
        assert_eq!(store.remove(&1u64), Some(value));
        store.record_eviction();

        let metrics = store.metrics();
        assert_eq!(metrics.hits, 1);
        assert_eq!(metrics.misses, 1);
        assert_eq!(metrics.inserts, 1);
        assert_eq!(metrics.updates, 1);
        assert_eq!(metrics.removes, 1);
        assert_eq!(metrics.evictions, 1);
    }
}
