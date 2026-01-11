//! Handle-based store for zero-copy policy metadata.
//!
//! Flow overview:
//! ```text
//! key -> KeyInterner -> handle -> policy/DS -> eviction -> handle -> KeyInterner -> key
//! ```
use crate::store::traits::{
    ConcurrentStore, StoreCore, StoreFactory, StoreFull, StoreMetrics, StoreMut,
};
use parking_lot::RwLock;
use std::cell::Cell;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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

/// Store keyed by compact handles (IDs) instead of full keys.
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
    pub fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::new(),
            capacity,
            metrics: StoreCounters::default(),
        }
    }

    /// Fetch a value by handle without updating metrics.
    pub fn peek_by_handle(&self, handle: &H) -> Option<&Arc<V>> {
        self.map.get(handle)
    }
}

impl<H, V> StoreCore<H, V> for HandleStore<H, V>
where
    H: Copy + Eq + Hash,
{
    fn get(&self, key: &H) -> Option<Arc<V>> {
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

    fn contains(&self, key: &H) -> bool {
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

impl<H, V> StoreMut<H, V> for HandleStore<H, V>
where
    H: Copy + Eq + Hash,
{
    fn try_insert(&mut self, key: H, value: Arc<V>) -> Result<Option<Arc<V>>, StoreFull> {
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

    fn remove(&mut self, key: &H) -> Option<Arc<V>> {
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

impl<H, V> StoreFactory<H, V> for HandleStore<H, V>
where
    H: Copy + Eq + Hash + Send,
{
    type Store = HandleStore<H, V>;

    fn create(capacity: usize) -> Self::Store {
        Self::new(capacity)
    }
}

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
    pub fn new(capacity: usize) -> Self {
        Self {
            map: RwLock::new(HashMap::new()),
            capacity,
            metrics: ConcurrentStoreCounters::default(),
        }
    }
}

impl<H, V> StoreCore<H, V> for ConcurrentHandleStore<H, V>
where
    H: Copy + Eq + Hash + Send + Sync,
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

    fn record_eviction(&self) {
        self.metrics.inc_eviction();
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

impl<H, V> StoreFactory<H, V> for ConcurrentHandleStore<H, V>
where
    H: Copy + Eq + Hash + Send + Sync,
{
    type Store = ConcurrentHandleStore<H, V>;

    fn create(capacity: usize) -> Self::Store {
        Self::new(capacity)
    }
}

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
