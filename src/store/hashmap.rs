use crate::store::traits::{
    ConcurrentStore, StoreCore, StoreFactory, StoreFull, StoreMetrics, StoreMut,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

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

/// Single-threaded HashMap-backed store.
#[derive(Debug)]
pub struct HashMapStore<K, V> {
    map: HashMap<K, Arc<V>>,
    capacity: usize,
    metrics: ConcurrentStoreCounters,
}

impl<K, V> HashMapStore<K, V>
where
    K: Eq + Hash,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::new(),
            capacity,
            metrics: ConcurrentStoreCounters::default(),
        }
    }

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

    pub fn peek_ref(&self, key: &K) -> Option<&Arc<V>>
    where
        K: Eq + Hash,
    {
        self.map.get(key)
    }

    pub fn map_capacity(&self) -> usize {
        self.map.capacity()
    }
}

impl<K, V> StoreCore<K, V> for HashMapStore<K, V>
where
    K: Eq + Hash,
{
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

    fn record_eviction(&self) {
        self.metrics.inc_eviction();
    }
}

impl<K, V> StoreMut<K, V> for HashMapStore<K, V>
where
    K: Eq + Hash,
{
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

    fn remove(&mut self, key: &K) -> Option<Arc<V>> {
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

impl<K, V> StoreFactory<K, V> for HashMapStore<K, V>
where
    K: Eq + Hash + Send,
{
    type Store = HashMapStore<K, V>;

    fn create(capacity: usize) -> Self::Store {
        Self::new(capacity)
    }
}

/// Concurrent HashMap-backed store using interior mutability.
#[derive(Debug)]
pub struct ConcurrentHashMapStore<K, V> {
    map: RwLock<HashMap<K, Arc<V>>>,
    capacity: usize,
    metrics: ConcurrentStoreCounters,
}

impl<K, V> ConcurrentHashMapStore<K, V>
where
    K: Eq + Hash + Send,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            map: RwLock::new(HashMap::new()),
            capacity,
            metrics: ConcurrentStoreCounters::default(),
        }
    }
}

impl<K, V> StoreCore<K, V> for ConcurrentHashMapStore<K, V>
where
    K: Eq + Hash,
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

    fn record_eviction(&self) {
        self.metrics.inc_eviction();
    }
}

impl<K, V> ConcurrentStore<K, V> for ConcurrentHashMapStore<K, V>
where
    K: Eq + Hash + Send + Sync,
    V: Sync + Send,
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

impl<K, V> StoreFactory<K, V> for ConcurrentHashMapStore<K, V>
where
    K: Eq + Hash + Send,
{
    type Store = ConcurrentHashMapStore<K, V>;

    fn create(capacity: usize) -> Self::Store {
        Self::new(capacity)
    }
}

/// Concurrent HashMap-backed store with sharded locking.
#[derive(Debug)]
pub struct ShardedHashMapStore<K, V> {
    shards: Vec<RwLock<HashMap<K, Arc<V>>>>,
    capacity: usize,
    size: AtomicUsize,
    metrics: ConcurrentStoreCounters,
}

impl<K, V> ShardedHashMapStore<K, V>
where
    K: Eq + Hash + Send + Sync,
{
    pub fn new(capacity: usize, shards: usize) -> Self {
        let shard_count = shards.max(1);
        let mut shard_vec = Vec::with_capacity(shard_count);
        for _ in 0..shard_count {
            shard_vec.push(RwLock::new(HashMap::new()));
        }
        Self {
            shards: shard_vec,
            capacity,
            size: AtomicUsize::new(0),
            metrics: ConcurrentStoreCounters::default(),
        }
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    fn shard_index(&self, key: &K) -> usize {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        (hasher.finish() as usize) % self.shards.len()
    }
}

impl<K, V> StoreCore<K, V> for ShardedHashMapStore<K, V>
where
    K: Eq + Hash + Send + Sync,
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

    fn record_eviction(&self) {
        self.metrics.inc_eviction();
    }
}

impl<K, V> ConcurrentStore<K, V> for ShardedHashMapStore<K, V>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
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

impl<K, V> StoreFactory<K, V> for ShardedHashMapStore<K, V>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
{
    type Store = ShardedHashMapStore<K, V>;

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
