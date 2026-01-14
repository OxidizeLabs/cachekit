//! HashMap-backed store implementations.
//!
//! ## Architecture
//! - Keys are stored in a `HashMap<K, Arc<V>>` for O(1) lookup.
//! - Capacity is enforced by entry count, not byte size.
//! - Concurrent variants use `RwLock` (single) or sharded locks.
//!
//! ## Key Components
//! - `HashMapStore`: single-threaded store.
//! - `ConcurrentHashMapStore`: thread-safe store with a global `RwLock`.
//! - `ShardedHashMapStore`: thread-safe store with per-shard locks.
//!
//! ## Core Operations
//! - `try_insert`: insert or update by key.
//! - `get`: fetch by key (updates hit/miss metrics).
//! - `remove`: delete by key.
//! - `clear`: drop all entries.
//!
//! ## Performance Trade-offs
//! - Fast lookup and update with predictable O(1) average cost.
//! - Uses `Arc<V>`; cloning is cheap but insert still allocates.
//! - Sharding reduces contention at the cost of extra hashing.
//!
//! ## When to Use
//! - General-purpose storage where keys are owned by the store.
//! - You need straightforward capacity enforcement by entry count.
//! - You want a concurrent store with optional sharding.
//!
//! ## Example Usage
//! ```rust
//! use std::sync::Arc;
//!
//! use cachekit::store::hashmap::HashMapStore;
//! use cachekit::store::traits::StoreMut;
//! use crate::cachekit::store::traits::StoreCore;
//!
//! let mut store: HashMapStore<u64, String> = HashMapStore::new(2);
//! store.try_insert(1, Arc::new("a".to_string())).unwrap();
//! assert!(store.contains(&1));
//! ```
//!
//! ## Type Constraints
//! - `K: Eq + Hash` for key lookup.
//! - `S: BuildHasher` for custom hashers (defaults to `RandomState`).
//!
//! ## Thread Safety
//! - `HashMapStore` is single-threaded.
//! - `ConcurrentHashMapStore` and `ShardedHashMapStore` are `Send + Sync`.
//!
//! ## Implementation Notes
//! - Sharded store uses the configured hasher to pick shards.
//! - Metrics are tracked with atomics for concurrent access.
use std::collections::HashMap;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use parking_lot::RwLock;

use crate::store::traits::{
    ConcurrentStore, StoreCore, StoreFactory, StoreFull, StoreMetrics, StoreMut,
};

/// Store metrics counters for concurrent hash map stores.
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
    /// Snapshot current store metrics.
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

    /// Increment hit counter.
    fn inc_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment miss counter.
    fn inc_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment insert counter.
    fn inc_insert(&self) {
        self.inserts.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment update counter.
    fn inc_update(&self) {
        self.updates.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment remove counter.
    fn inc_remove(&self) {
        self.removes.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment eviction counter.
    fn inc_eviction(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }
}

/// Single-threaded HashMap-backed store.
#[derive(Debug)]
pub struct HashMapStore<K, V, S = RandomState> {
    map: HashMap<K, Arc<V>, S>,
    capacity: usize,
    metrics: ConcurrentStoreCounters,
}

impl<K, V> HashMapStore<K, V, RandomState>
where
    K: Eq + Hash,
{
    /// Create a store with a fixed capacity and default hasher.
    pub fn new(capacity: usize) -> Self {
        Self::with_hasher(capacity, RandomState::new())
    }
}

impl<K, V, S> HashMapStore<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    /// Create a store with a fixed capacity and custom hasher.
    pub fn with_hasher(capacity: usize, hasher: S) -> Self {
        Self {
            map: HashMap::with_capacity_and_hasher(capacity, hasher),
            capacity,
            metrics: ConcurrentStoreCounters::default(),
        }
    }

    /// Fetch a value by key without updating metrics.
    pub fn get_ref(&self, key: &K) -> Option<&Arc<V>>
    where
        K: Eq + Hash,
    {
        match self.map.get(key) {
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

    /// Fetch a value by key without touching access counters.
    pub fn peek_ref(&self, key: &K) -> Option<&Arc<V>>
    where
        K: Eq + Hash,
    {
        self.map.get(key)
    }

    /// Return the backing hash map capacity.
    pub fn map_capacity(&self) -> usize {
        self.map.capacity()
    }
}

impl<K, V, S> StoreCore<K, V> for HashMapStore<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    /// Fetch a value by key.
    fn get(&self, key: &K) -> Option<Arc<V>> {
        match self.map.get(key).cloned() {
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

    /// Check whether a key exists.
    fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Return the number of entries.
    fn len(&self) -> usize {
        self.map.len()
    }

    /// Return the maximum capacity.
    fn capacity(&self) -> usize {
        self.capacity
    }

    /// Snapshot store metrics.
    fn metrics(&self) -> StoreMetrics {
        self.metrics.snapshot()
    }

    /// Record an eviction.
    fn record_eviction(&self) {
        self.metrics.inc_eviction();
    }
}

impl<K, V, S> StoreMut<K, V> for HashMapStore<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    /// Insert or update an entry.
    fn try_insert(&mut self, key: K, value: Arc<V>) -> Result<Option<Arc<V>>, StoreFull> {
        if !self.map.contains_key(&key) && self.map.len() >= self.capacity {
            return Err(StoreFull);
        }
        let previous = self.map.insert(key, value);
        if previous.is_some() {
            self.metrics.inc_update();
        } else {
            self.metrics.inc_insert();
        }
        Ok(previous)
    }

    /// Remove a value by key.
    fn remove(&mut self, key: &K) -> Option<Arc<V>> {
        let removed = self.map.remove(key);
        if removed.is_some() {
            self.metrics.inc_remove();
        }
        removed
    }

    /// Clear all entries.
    fn clear(&mut self) {
        self.map.clear();
    }
}

impl<K, V> StoreFactory<K, V> for HashMapStore<K, V, RandomState>
where
    K: Eq + Hash + Send,
{
    type Store = HashMapStore<K, V, RandomState>;

    /// Create a new store with the given capacity.
    fn create(capacity: usize) -> Self::Store {
        Self::new(capacity)
    }
}

/// Concurrent HashMap-backed store using interior mutability.
#[derive(Debug)]
pub struct ConcurrentHashMapStore<K, V, S = RandomState> {
    map: RwLock<HashMap<K, Arc<V>, S>>,
    capacity: usize,
    metrics: ConcurrentStoreCounters,
}

impl<K, V> ConcurrentHashMapStore<K, V, RandomState>
where
    K: Eq + Hash + Send,
{
    /// Create a concurrent store with default hasher.
    pub fn new(capacity: usize) -> Self {
        Self::with_hasher(capacity, RandomState::new())
    }
}

impl<K, V, S> ConcurrentHashMapStore<K, V, S>
where
    K: Eq + Hash + Send,
    S: BuildHasher,
{
    /// Create a concurrent store with a custom hasher.
    pub fn with_hasher(capacity: usize, hasher: S) -> Self {
        Self {
            map: RwLock::new(HashMap::with_capacity_and_hasher(capacity, hasher)),
            capacity,
            metrics: ConcurrentStoreCounters::default(),
        }
    }
}

impl<K, V, S> StoreCore<K, V> for ConcurrentHashMapStore<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher + Send + Sync,
{
    /// Fetch a value by key.
    fn get(&self, key: &K) -> Option<Arc<V>> {
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

    /// Check whether a key exists.
    fn contains(&self, key: &K) -> bool {
        self.map.read().contains_key(key)
    }

    /// Return the number of entries.
    fn len(&self) -> usize {
        self.map.read().len()
    }

    /// Return the maximum capacity.
    fn capacity(&self) -> usize {
        self.capacity
    }

    /// Snapshot store metrics.
    fn metrics(&self) -> StoreMetrics {
        self.metrics.snapshot()
    }

    /// Record an eviction.
    fn record_eviction(&self) {
        self.metrics.inc_eviction();
    }
}

impl<K, V, S> ConcurrentStore<K, V> for ConcurrentHashMapStore<K, V, S>
where
    K: Eq + Hash + Send + Sync,
    V: Sync + Send,
    S: BuildHasher + Send + Sync,
{
    /// Insert or update an entry.
    fn try_insert(&self, key: K, value: Arc<V>) -> Result<Option<Arc<V>>, StoreFull> {
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

    /// Remove a value by key.
    fn remove(&self, key: &K) -> Option<Arc<V>> {
        let removed = self.map.write().remove(key);
        if removed.is_some() {
            self.metrics.inc_remove();
        }
        removed
    }

    /// Clear all entries.
    fn clear(&self) {
        self.map.write().clear()
    }
}

impl<K, V> StoreFactory<K, V> for ConcurrentHashMapStore<K, V, RandomState>
where
    K: Eq + Hash + Send,
{
    type Store = ConcurrentHashMapStore<K, V, RandomState>;

    /// Create a new store with the given capacity.
    fn create(capacity: usize) -> Self::Store {
        Self::new(capacity)
    }
}

/// Concurrent HashMap-backed store with sharded locking.
#[derive(Debug)]
pub struct ShardedHashMapStore<K, V, S = RandomState> {
    shards: Vec<RwLock<HashMap<K, Arc<V>, S>>>,
    capacity: usize,
    size: AtomicUsize,
    metrics: ConcurrentStoreCounters,
    hasher: S,
}

impl<K, V> ShardedHashMapStore<K, V, RandomState>
where
    K: Eq + Hash + Send + Sync,
{
    /// Create a sharded store with the default hasher.
    pub fn new(capacity: usize, shards: usize) -> Self {
        Self::with_hasher(capacity, shards, RandomState::new())
    }
}

impl<K, V, S> ShardedHashMapStore<K, V, S>
where
    K: Eq + Hash + Send + Sync,
    S: BuildHasher + Clone,
{
    /// Create a sharded store with a custom hasher.
    pub fn with_hasher(capacity: usize, shards: usize, hasher: S) -> Self {
        let shard_count = shards.max(1);
        let mut shard_vec = Vec::with_capacity(shard_count);
        for _ in 0..shard_count {
            shard_vec.push(RwLock::new(HashMap::with_hasher(hasher.clone())));
        }
        Self {
            shards: shard_vec,
            capacity,
            size: AtomicUsize::new(0),
            metrics: ConcurrentStoreCounters::default(),
            hasher,
        }
    }

    /// Return the number of shards.
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Compute the shard index for a key.
    fn shard_index(&self, key: &K) -> usize {
        (self.hasher.hash_one(key) as usize) % self.shards.len()
    }
}

impl<K, V, S> StoreCore<K, V> for ShardedHashMapStore<K, V, S>
where
    K: Eq + Hash + Send + Sync,
    S: BuildHasher + Clone,
{
    /// Fetch a value by key.
    fn get(&self, key: &K) -> Option<Arc<V>> {
        let idx = self.shard_index(key);
        match self.shards[idx].read().get(key).cloned() {
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

    /// Check whether a key exists.
    fn contains(&self, key: &K) -> bool {
        let idx = self.shard_index(key);
        self.shards[idx].read().contains_key(key)
    }

    /// Return the number of entries.
    fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    /// Return the maximum capacity.
    fn capacity(&self) -> usize {
        self.capacity
    }

    /// Snapshot store metrics.
    fn metrics(&self) -> StoreMetrics {
        self.metrics.snapshot()
    }

    /// Record an eviction.
    fn record_eviction(&self) {
        self.metrics.inc_eviction();
    }
}

impl<K, V, S> ConcurrentStore<K, V> for ShardedHashMapStore<K, V, S>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
    S: BuildHasher + Clone + Send + Sync,
{
    /// Insert or update an entry.
    fn try_insert(&self, key: K, value: Arc<V>) -> Result<Option<Arc<V>>, StoreFull> {
        let idx = self.shard_index(&key);
        let mut map = self.shards[idx].write();
        match map.entry(key) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                let previous = Some(entry.insert(value));
                self.metrics.inc_update();
                Ok(previous)
            },
            std::collections::hash_map::Entry::Vacant(entry) => {
                if self.capacity == 0 {
                    return Err(StoreFull);
                }

                loop {
                    let current = self.size.load(Ordering::Relaxed);
                    if current >= self.capacity {
                        return Err(StoreFull);
                    }
                    if self
                        .size
                        .compare_exchange(current, current + 1, Ordering::AcqRel, Ordering::Relaxed)
                        .is_ok()
                    {
                        break;
                    }
                }

                entry.insert(value);
                self.metrics.inc_insert();
                Ok(None)
            },
        }
    }

    /// Remove a value by key.
    fn remove(&self, key: &K) -> Option<Arc<V>> {
        let idx = self.shard_index(key);
        let removed = self.shards[idx].write().remove(key);
        if removed.is_some() {
            self.size.fetch_sub(1, Ordering::Relaxed);
            self.metrics.inc_remove();
        }
        removed
    }

    /// Clear all entries.
    fn clear(&self) {
        let mut guards = Vec::with_capacity(self.shards.len());
        for shard in &self.shards {
            guards.push(shard.write());
        }
        for guard in guards.iter_mut() {
            guard.clear();
        }
        self.size.store(0, Ordering::Relaxed);
    }
}

impl<K, V> StoreFactory<K, V> for ShardedHashMapStore<K, V, RandomState>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
{
    type Store = ShardedHashMapStore<K, V, RandomState>;

    /// Create a new store with capacity and a default shard count.
    fn create(capacity: usize) -> Self::Store {
        let shards = std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1);
        Self::new(capacity, shards)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashmap_store_basic_ops() {
        let mut store = HashMapStore::new(2);
        let value = Arc::new("v1".to_string());
        assert_eq!(store.try_insert("k1", value.clone()), Ok(None));
        assert_eq!(store.get(&"k1"), Some(value.clone()));
        assert!(store.contains(&"k1"));
        assert_eq!(store.len(), 1);
        assert_eq!(store.capacity(), 2);
        assert_eq!(store.remove(&"k1"), Some(value));
        assert!(!store.contains(&"k1"));
    }

    #[test]
    fn concurrent_store_basic_ops() {
        let store = ConcurrentHashMapStore::new(2);
        let value = Arc::new("v1".to_string());
        assert_eq!(store.try_insert("k1", value.clone()), Ok(None));
        assert_eq!(store.get(&"k1"), Some(value.clone()));
        assert!(store.contains(&"k1"));
        assert_eq!(store.len(), 1);
        assert_eq!(store.capacity(), 2);
        assert_eq!(store.remove(&"k1"), Some(value));
        assert!(!store.contains(&"k1"));
    }

    #[test]
    fn hashmap_store_capacity_enforced() {
        let mut store = HashMapStore::new(1);
        assert_eq!(store.try_insert("k1", Arc::new("v1".to_string())), Ok(None));
        assert_eq!(
            store.try_insert("k2", Arc::new("v2".to_string())),
            Err(StoreFull)
        );
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn sharded_store_basic_ops() {
        let store = ShardedHashMapStore::new(2, 2);
        let value = Arc::new("v1".to_string());
        assert_eq!(store.try_insert("k1", value.clone()), Ok(None));
        assert_eq!(store.get(&"k1"), Some(value.clone()));
        assert!(store.contains(&"k1"));
        assert_eq!(store.len(), 1);
        assert_eq!(store.capacity(), 2);
        assert_eq!(store.remove(&"k1"), Some(value));
        assert!(!store.contains(&"k1"));
    }

    #[test]
    fn sharded_store_capacity_enforced() {
        let store = ShardedHashMapStore::new(1, 2);
        assert_eq!(store.try_insert("k1", Arc::new("v1".to_string())), Ok(None));
        assert_eq!(
            store.try_insert("k2", Arc::new("v2".to_string())),
            Err(StoreFull)
        );
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn hashmap_store_metrics_counts() {
        let mut store = HashMapStore::new(2);
        let value = Arc::new("v1".to_string());

        assert_eq!(store.metrics(), StoreMetrics::default());
        assert_eq!(store.get(&"missing"), None);
        assert_eq!(store.try_insert("k1", value.clone()), Ok(None));
        assert_eq!(
            store.try_insert("k1", value.clone()),
            Ok(Some(value.clone()))
        );
        assert_eq!(store.get(&"k1"), Some(value.clone()));
        assert_eq!(store.remove(&"k1"), Some(value));
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
