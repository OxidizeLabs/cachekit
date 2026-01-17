//! HashMap-backed store implementations.
//!
//! ## Architecture
//! - Single-threaded `HashMapStore` stores `V` directly (no `Arc` overhead).
//! - Concurrent variants use `Arc<V>` for safe shared access across threads.
//! - Capacity is enforced by entry count, not byte size.
//!
//! ## Key Components
//! - `HashMapStore`: single-threaded store with zero-overhead value access.
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
//! - Single-threaded: returns `&V` references, no allocation on access.
//! - Concurrent: returns `Arc<V>`, requires atomic ref-count increment.
//! - Sharding reduces contention at the cost of extra hashing.
//!
//! ## When to Use
//! - `HashMapStore`: single-threaded hot paths where allocation matters.
//! - `ConcurrentHashMapStore`: simple thread-safe caching.
//! - `ShardedHashMapStore`: high-contention concurrent workloads.
//!
//! ## Example Usage
//! ```rust
//! use cachekit::store::hashmap::HashMapStore;
//! use cachekit::store::traits::StoreMut;
//! use cachekit::store::traits::StoreCore;
//!
//! let mut store: HashMapStore<u64, String> = HashMapStore::new(2);
//! store.try_insert(1, "hello".to_string()).unwrap();
//! assert_eq!(store.get(&1), Some(&"hello".to_string()));
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
    ConcurrentStore, ConcurrentStoreFactory, ConcurrentStoreRead, StoreCore, StoreFactory,
    StoreFull, StoreMetrics, StoreMut,
};

// =============================================================================
// Metrics counters
// =============================================================================

/// Store metrics counters using atomics for thread-safe updates.
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
// Single-threaded HashMapStore
// =============================================================================

/// Single-threaded HashMap-backed store.
///
/// Stores values directly without `Arc` wrapper for zero-overhead access.
#[derive(Debug)]
pub struct HashMapStore<K, V, S = RandomState> {
    map: HashMap<K, V, S>,
    capacity: usize,
    metrics: StoreCounters,
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
            metrics: StoreCounters::default(),
        }
    }

    /// Fetch a value by key without touching metrics.
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.map.get(key)
    }

    /// Fetch a mutable reference to a value without touching metrics.
    pub fn peek_mut(&mut self, key: &K) -> Option<&mut V> {
        self.map.get_mut(key)
    }

    /// Return the backing hash map capacity.
    pub fn map_capacity(&self) -> usize {
        self.map.capacity()
    }

    /// Record that the policy evicted an entry.
    pub fn record_eviction(&self) {
        self.metrics.inc_eviction();
    }
}

impl<K, V, S> StoreCore<K, V> for HashMapStore<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    fn get(&self, key: &K) -> Option<&V> {
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

    fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn metrics(&self) -> StoreMetrics {
        self.metrics.snapshot()
    }
}

impl<K, V, S> StoreMut<K, V> for HashMapStore<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>, StoreFull> {
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

    fn remove(&mut self, key: &K) -> Option<V> {
        let removed = self.map.remove(key);
        if removed.is_some() {
            self.metrics.inc_remove();
        }
        removed
    }

    fn clear(&mut self) {
        self.map.clear();
    }
}

impl<K, V> StoreFactory<K, V> for HashMapStore<K, V, RandomState>
where
    K: Eq + Hash,
{
    type Store = HashMapStore<K, V, RandomState>;

    fn create(capacity: usize) -> Self::Store {
        Self::new(capacity)
    }
}

// =============================================================================
// Concurrent HashMap store (global lock)
// =============================================================================

/// Concurrent HashMap-backed store using interior mutability.
///
/// Uses `Arc<V>` for values since references can't outlive lock guards.
#[derive(Debug)]
pub struct ConcurrentHashMapStore<K, V, S = RandomState> {
    map: RwLock<HashMap<K, Arc<V>, S>>,
    capacity: usize,
    metrics: StoreCounters,
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
            metrics: StoreCounters::default(),
        }
    }

    /// Record that the policy evicted an entry.
    pub fn record_eviction(&self) {
        self.metrics.inc_eviction();
    }
}

impl<K, V, S> ConcurrentStoreRead<K, V> for ConcurrentHashMapStore<K, V, S>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
    S: BuildHasher + Send + Sync,
{
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

    fn contains(&self, key: &K) -> bool {
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

impl<K, V, S> ConcurrentStore<K, V> for ConcurrentHashMapStore<K, V, S>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
    S: BuildHasher + Send + Sync,
{
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

    fn remove(&self, key: &K) -> Option<Arc<V>> {
        let removed = self.map.write().remove(key);
        if removed.is_some() {
            self.metrics.inc_remove();
        }
        removed
    }

    fn clear(&self) {
        self.map.write().clear()
    }
}

impl<K, V> ConcurrentStoreFactory<K, V> for ConcurrentHashMapStore<K, V, RandomState>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
{
    type Store = ConcurrentHashMapStore<K, V, RandomState>;

    fn create(capacity: usize) -> Self::Store {
        Self::new(capacity)
    }
}

// =============================================================================
// Sharded HashMap store
// =============================================================================

/// Concurrent HashMap-backed store with sharded locking.
///
/// Reduces contention by distributing keys across multiple independent shards.
#[derive(Debug)]
pub struct ShardedHashMapStore<K, V, S = RandomState> {
    shards: Vec<RwLock<HashMap<K, Arc<V>, S>>>,
    capacity: usize,
    size: AtomicUsize,
    metrics: StoreCounters,
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
            metrics: StoreCounters::default(),
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

    /// Record that the policy evicted an entry.
    pub fn record_eviction(&self) {
        self.metrics.inc_eviction();
    }
}

impl<K, V, S> ConcurrentStoreRead<K, V> for ShardedHashMapStore<K, V, S>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
    S: BuildHasher + Clone + Send + Sync,
{
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

    fn contains(&self, key: &K) -> bool {
        let idx = self.shard_index(key);
        self.shards[idx].read().contains_key(key)
    }

    fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn metrics(&self) -> StoreMetrics {
        self.metrics.snapshot()
    }
}

impl<K, V, S> ConcurrentStore<K, V> for ShardedHashMapStore<K, V, S>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
    S: BuildHasher + Clone + Send + Sync,
{
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

    fn remove(&self, key: &K) -> Option<Arc<V>> {
        let idx = self.shard_index(key);
        let removed = self.shards[idx].write().remove(key);
        if removed.is_some() {
            self.size.fetch_sub(1, Ordering::Relaxed);
            self.metrics.inc_remove();
        }
        removed
    }

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

impl<K, V> ConcurrentStoreFactory<K, V> for ShardedHashMapStore<K, V, RandomState>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
{
    type Store = ShardedHashMapStore<K, V, RandomState>;

    fn create(capacity: usize) -> Self::Store {
        let shards = std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1);
        Self::new(capacity, shards)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashmap_store_basic_ops() {
        let mut store = HashMapStore::new(2);
        assert_eq!(store.try_insert("k1", "v1".to_string()), Ok(None));
        assert_eq!(store.get(&"k1"), Some(&"v1".to_string()));
        assert!(store.contains(&"k1"));
        assert_eq!(store.len(), 1);
        assert_eq!(store.capacity(), 2);
        assert_eq!(store.remove(&"k1"), Some("v1".to_string()));
        assert!(!store.contains(&"k1"));
    }

    #[test]
    fn hashmap_store_returns_reference() {
        let mut store = HashMapStore::new(2);
        store.try_insert("k1", "hello".to_string()).unwrap();

        // get() returns &V, no allocation
        let value: &String = store.get(&"k1").unwrap();
        assert_eq!(value, "hello");

        // peek() also returns &V without metrics
        let peeked: &String = store.peek(&"k1").unwrap();
        assert_eq!(peeked, "hello");
    }

    #[test]
    fn hashmap_store_capacity_enforced() {
        let mut store = HashMapStore::new(1);
        assert_eq!(store.try_insert("k1", "v1".to_string()), Ok(None));
        assert_eq!(store.try_insert("k2", "v2".to_string()), Err(StoreFull));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn hashmap_store_update_returns_previous() {
        let mut store = HashMapStore::new(2);
        assert_eq!(store.try_insert("k1", "v1".to_string()), Ok(None));
        assert_eq!(
            store.try_insert("k1", "v2".to_string()),
            Ok(Some("v1".to_string()))
        );
        assert_eq!(store.get(&"k1"), Some(&"v2".to_string()));
    }

    #[test]
    fn hashmap_store_metrics_counts() {
        let mut store = HashMapStore::new(2);

        assert_eq!(store.metrics(), StoreMetrics::default());
        assert_eq!(store.get(&"missing"), None);
        assert_eq!(store.try_insert("k1", "v1".to_string()), Ok(None));
        assert_eq!(
            store.try_insert("k1", "v2".to_string()),
            Ok(Some("v1".to_string()))
        );
        assert_eq!(store.get(&"k1"), Some(&"v2".to_string()));
        assert_eq!(store.remove(&"k1"), Some("v2".to_string()));
        store.record_eviction();

        let metrics = store.metrics();
        assert_eq!(metrics.hits, 1);
        assert_eq!(metrics.misses, 1);
        assert_eq!(metrics.inserts, 1);
        assert_eq!(metrics.updates, 1);
        assert_eq!(metrics.removes, 1);
        assert_eq!(metrics.evictions, 1);
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
}
