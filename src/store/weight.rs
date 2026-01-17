//! Weight-aware store with entry and byte-based limits.
//!
//! ## Architecture
//! - Stores `Arc<V>` values in a `HashMap<K, WeightEntry<V>>`.
//! - Tracks total weight to enforce a weight capacity.
//! - Uses a caller-provided weight function `F: Fn(&V) -> usize`.
//!
//! ## Key Components
//! - `WeightStore`: single-threaded weight-aware store.
//! - `ConcurrentWeightStore`: thread-safe wrapper using `RwLock`.
//! - `WeightEntry`: stores value plus its computed weight.
//!
//! ## Core Operations
//! - `try_insert`: insert/update while enforcing entry + weight limits.
//! - `get`: fetch by key (updates hit/miss metrics).
//! - `remove`: delete by key and adjust total weight.
//! - `clear`: drop all entries and reset weight to zero.
//!
//! ## Performance Trade-offs
//! - Weight checks add a small constant cost on insert/update.
//! - Weight function can be as cheap or expensive as you choose.
//! - Keeps eviction policy logic separate from size accounting.
//!
//! ## When to Use
//! - You want size-based capacity instead of entry-count limits.
//! - Values vary widely in size and you want fair eviction pressure.
//! - You need to report total bytes/weight for observability.
//!
//! ## Example Usage
//! ```rust
//! use std::sync::Arc;
//!
//! use cachekit::store::weight::WeightStore;
//!
//! let mut store = WeightStore::with_capacity(10, 64, |v: &String| v.len());
//! store.try_insert("k1", Arc::new("value".to_string())).unwrap();
//! assert_eq!(store.total_weight(), 5);
//! ```
//!
//! ## Type Constraints
//! - `K: Eq + Hash` for key lookup.
//! - `F: Fn(&V) -> usize` to compute weight.
//!
//! ## Thread Safety
//! - `WeightStore` is single-threaded.
//! - `ConcurrentWeightStore` is `Send + Sync` via `RwLock`.
//!
//! ## Implementation Notes
//! - Weight is stored per entry to avoid recomputation on reads.
//! - Updates recompute weight and adjust `total_weight`.
//! - Does NOT implement `StoreCore`/`StoreMut` traits (uses `Arc<V>` API for weight computation).

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use crate::store::traits::{ConcurrentStore, ConcurrentStoreRead, StoreFull, StoreMetrics};

/// Entry with precomputed weight for fast accounting.
#[derive(Debug)]
struct WeightEntry<V> {
    value: Arc<V>,
    weight: usize,
}

/// Store metrics counters for weight stores.
#[derive(Debug, Default)]
struct StoreCounters {
    hits: AtomicU64,
    misses: AtomicU64,
    inserts: AtomicU64,
    updates: AtomicU64,
    removes: AtomicU64,
    evictions: AtomicU64,
}

impl StoreCounters {
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
// Single-threaded WeightStore
// =============================================================================

/// HashMap-backed store that tracks total weight (bytes) for values.
///
/// Uses `Arc<V>` for values to enable weight computation on shared data.
#[derive(Debug)]
pub struct WeightStore<K, V, F>
where
    F: Fn(&V) -> usize,
{
    map: HashMap<K, WeightEntry<V>>,
    capacity_entries: usize,
    capacity_weight: usize,
    total_weight: usize,
    weight_fn: F,
    metrics: StoreCounters,
}

impl<K, V, F> WeightStore<K, V, F>
where
    K: Eq + Hash,
    F: Fn(&V) -> usize,
{
    /// Create a store with entry and weight limits plus a weight function.
    pub fn with_capacity(capacity_entries: usize, capacity_weight: usize, weight_fn: F) -> Self {
        Self {
            map: HashMap::with_capacity(capacity_entries),
            capacity_entries,
            capacity_weight,
            total_weight: 0,
            weight_fn,
            metrics: StoreCounters::default(),
        }
    }

    /// Returns the current total weight.
    pub fn total_weight(&self) -> usize {
        self.total_weight
    }

    /// Returns the configured weight capacity.
    pub fn capacity_weight(&self) -> usize {
        self.capacity_weight
    }

    /// Compute the weight for a value.
    fn compute_weight(&self, value: &V) -> usize {
        (self.weight_fn)(value)
    }

    /// Fetch a value by key.
    pub fn get(&self, key: &K) -> Option<Arc<V>> {
        match self.map.get(key).map(|entry| Arc::clone(&entry.value)) {
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

    /// Fetch a value by key without updating metrics.
    pub fn peek(&self, key: &K) -> Option<&Arc<V>> {
        self.map.get(key).map(|entry| &entry.value)
    }

    /// Check whether a key exists.
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Return the number of entries.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Check if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Return the maximum entry capacity.
    pub fn capacity(&self) -> usize {
        self.capacity_entries
    }

    /// Snapshot store metrics.
    pub fn metrics(&self) -> StoreMetrics {
        self.metrics.snapshot()
    }

    /// Record an eviction.
    pub fn record_eviction(&self) {
        self.metrics.inc_eviction();
    }

    /// Insert or update an entry while enforcing weight limits.
    pub fn try_insert(&mut self, key: K, value: Arc<V>) -> Result<Option<Arc<V>>, StoreFull> {
        let new_weight = self.compute_weight(value.as_ref());
        if let Some(entry) = self.map.get_mut(&key) {
            let next_total = self.total_weight - entry.weight + new_weight;
            if next_total > self.capacity_weight {
                return Err(StoreFull);
            }
            let previous = Arc::clone(&entry.value);
            entry.value = value;
            entry.weight = new_weight;
            self.total_weight = next_total;
            self.metrics.inc_update();
            return Ok(Some(previous));
        }

        if self.map.len() >= self.capacity_entries {
            return Err(StoreFull);
        }
        if self.total_weight + new_weight > self.capacity_weight {
            return Err(StoreFull);
        }

        self.map.insert(
            key,
            WeightEntry {
                value,
                weight: new_weight,
            },
        );
        self.total_weight += new_weight;
        self.metrics.inc_insert();
        Ok(None)
    }

    /// Remove a value by key and update total weight.
    pub fn remove(&mut self, key: &K) -> Option<Arc<V>> {
        let entry = self.map.remove(key)?;
        self.total_weight = self.total_weight.saturating_sub(entry.weight);
        self.metrics.inc_remove();
        Some(entry.value)
    }

    /// Clear all entries and reset weight to zero.
    pub fn clear(&mut self) {
        self.map.clear();
        self.total_weight = 0;
    }
}

// =============================================================================
// Concurrent WeightStore
// =============================================================================

/// Concurrent weight store using a `parking_lot::RwLock`.
#[derive(Debug)]
pub struct ConcurrentWeightStore<K, V, F>
where
    F: Fn(&V) -> usize,
{
    inner: RwLock<WeightStore<K, V, F>>,
    metrics: StoreCounters,
}

impl<K, V, F> ConcurrentWeightStore<K, V, F>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
    F: Fn(&V) -> usize,
{
    /// Create a concurrent store with entry and weight limits.
    pub fn with_capacity(capacity_entries: usize, capacity_weight: usize, weight_fn: F) -> Self {
        Self {
            inner: RwLock::new(WeightStore::with_capacity(
                capacity_entries,
                capacity_weight,
                weight_fn,
            )),
            metrics: StoreCounters::default(),
        }
    }

    /// Returns the current total weight.
    pub fn total_weight(&self) -> usize {
        self.inner.read().total_weight()
    }

    /// Returns the configured weight capacity.
    pub fn capacity_weight(&self) -> usize {
        self.inner.read().capacity_weight()
    }

    /// Record an eviction.
    pub fn record_eviction(&self) {
        self.metrics.inc_eviction();
    }
}

impl<K, V, F> ConcurrentStoreRead<K, V> for ConcurrentWeightStore<K, V, F>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
    F: Fn(&V) -> usize + Send + Sync,
{
    fn get(&self, key: &K) -> Option<Arc<V>> {
        let store = self.inner.read();
        match store.map.get(key).map(|entry| Arc::clone(&entry.value)) {
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

    fn contains(&self, key: &K) -> bool {
        self.inner.read().contains(key)
    }

    fn len(&self) -> usize {
        self.inner.read().len()
    }

    fn capacity(&self) -> usize {
        self.inner.read().capacity()
    }

    fn metrics(&self) -> StoreMetrics {
        self.metrics.snapshot()
    }
}

impl<K, V, F> ConcurrentStore<K, V> for ConcurrentWeightStore<K, V, F>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
    F: Fn(&V) -> usize + Send + Sync,
{
    fn try_insert(&self, key: K, value: Arc<V>) -> Result<Option<Arc<V>>, StoreFull> {
        let mut store = self.inner.write();
        store.try_insert(key, value)
    }

    fn remove(&self, key: &K) -> Option<Arc<V>> {
        let mut store = self.inner.write();
        store.remove(key)
    }

    fn clear(&self) {
        let mut store = self.inner.write();
        store.clear();
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::ptr_arg)]
    fn weight_by_len(value: &String) -> usize {
        value.len()
    }

    #[test]
    fn weight_store_tracks_weight() {
        let mut store = WeightStore::with_capacity(3, 10, weight_by_len);
        assert_eq!(store.total_weight(), 0);
        assert_eq!(store.try_insert("k1", Arc::new("aa".to_string())), Ok(None));
        assert_eq!(store.total_weight(), 2);
        assert_eq!(
            store.try_insert("k2", Arc::new("bbbb".to_string())),
            Ok(None)
        );
        assert_eq!(store.total_weight(), 6);
        assert_eq!(store.remove(&"k1"), Some(Arc::new("aa".to_string())));
        assert_eq!(store.total_weight(), 4);
    }

    #[test]
    fn weight_store_enforces_capacity() {
        let mut store = WeightStore::with_capacity(10, 5, weight_by_len);
        assert_eq!(
            store.try_insert("k1", Arc::new("aaaaa".to_string())),
            Ok(None)
        );
        assert_eq!(
            store.try_insert("k2", Arc::new("bb".to_string())),
            Err(StoreFull)
        );
    }

    #[test]
    fn weight_store_update_adjusts_weight() {
        let mut store = WeightStore::with_capacity(10, 10, weight_by_len);
        assert_eq!(store.try_insert("k1", Arc::new("aa".to_string())), Ok(None));
        assert_eq!(store.total_weight(), 2);
        assert_eq!(
            store.try_insert("k1", Arc::new("aaaa".to_string())),
            Ok(Some(Arc::new("aa".to_string())))
        );
        assert_eq!(store.total_weight(), 4);
    }

    #[test]
    fn concurrent_weight_store_basic_ops() {
        let store = ConcurrentWeightStore::with_capacity(2, 10, weight_by_len);
        let value = Arc::new("aa".to_string());
        assert_eq!(store.try_insert("k1", value.clone()), Ok(None));
        assert_eq!(store.get(&"k1"), Some(value.clone()));
        assert_eq!(store.total_weight(), 2);
        assert_eq!(store.remove(&"k1"), Some(value));
        assert_eq!(store.total_weight(), 0);
    }
}
