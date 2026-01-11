use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use crate::store::traits::{StoreCore, StoreFactory, StoreFull, StoreMetrics, StoreMut};

/// Opaque entry handle for slab-based storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntryId(usize);

#[derive(Debug)]
struct Entry<K, V> {
    key: K,
    value: V,
}

pub trait ValueModel<V> {
    type Stored;
    type Output<'a>
    where
        V: 'a;
    type Removed;

    fn store(value: V) -> Self::Stored;
    fn output(stored: &Self::Stored) -> Self::Output<'_>;
    fn replace(stored: &mut Self::Stored, value: Self::Stored) -> Self::Removed;
    fn remove(stored: Self::Stored) -> Self::Removed;
}

#[derive(Debug, Default)]
pub struct SharedValue;

impl<V> ValueModel<V> for SharedValue {
    type Stored = Arc<V>;
    type Output<'a>
        = Arc<V>
    where
        V: 'a;
    type Removed = Arc<V>;

    fn store(value: V) -> Self::Stored {
        Arc::new(value)
    }

    fn output(stored: &Self::Stored) -> Self::Output<'_> {
        Arc::clone(stored)
    }

    fn replace(stored: &mut Self::Stored, value: Self::Stored) -> Self::Removed {
        std::mem::replace(stored, value)
    }

    fn remove(stored: Self::Stored) -> Self::Removed {
        stored
    }
}

#[derive(Debug, Default)]
pub struct OwnedValue;

impl<V> ValueModel<V> for OwnedValue {
    type Stored = V;
    type Output<'a>
        = &'a V
    where
        V: 'a;
    type Removed = V;

    fn store(value: V) -> Self::Stored {
        value
    }

    fn output(stored: &Self::Stored) -> Self::Output<'_> {
        stored
    }

    fn replace(stored: &mut Self::Stored, value: Self::Stored) -> Self::Removed {
        std::mem::replace(stored, value)
    }

    fn remove(stored: Self::Stored) -> Self::Removed {
        stored
    }
}

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

pub type SharedSlabStore<K, V> = SlabStore<K, V, SharedValue>;
pub type OwnedSlabStore<K, V> = SlabStore<K, V, OwnedValue>;

/// Concurrent slab-backed store using a `parking_lot::RwLock`.
#[derive(Debug)]
pub struct ConcurrentSlabStore<K, V> {
    inner: RwLock<SlabStore<K, V, SharedValue>>,
}

impl<K, V> ConcurrentSlabStore<K, V>
where
    K: Eq + Hash,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: RwLock::new(SlabStore::new(capacity)),
        }
    }

    /// Return the EntryId for a key if it exists.
    pub fn entry_id(&self, key: &K) -> Option<EntryId> {
        let store = self.inner.read();
        store.entry_id(key)
    }

    /// Fetch a value by EntryId.
    pub fn get_by_id(&self, id: EntryId) -> Option<Arc<V>> {
        let store = self.inner.read();
        store.get_by_id(id)
    }

    /// Fetch a key by EntryId.
    pub fn key_by_id(&self, id: EntryId) -> Option<K>
    where
        K: Clone,
    {
        let store = self.inner.read();
        store.key_by_id(id).cloned()
    }
}

/// Slab-backed store with EntryId indirection.
#[derive(Debug)]
pub struct SlabStore<K, V, M = SharedValue>
where
    M: ValueModel<V>,
{
    entries: Vec<Option<Entry<K, M::Stored>>>,
    free_list: Vec<usize>,
    index: HashMap<K, EntryId>,
    capacity: usize,
    metrics: StoreCounters,
    _marker: PhantomData<M>,
}

impl<K, V, M> SlabStore<K, V, M>
where
    K: Eq + Hash,
    M: ValueModel<V>,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            free_list: Vec::new(),
            index: HashMap::with_capacity(capacity),
            capacity,
            metrics: StoreCounters::default(),
            _marker: PhantomData,
        }
    }

    /// Return the EntryId for a key if it exists.
    pub fn entry_id(&self, key: &K) -> Option<EntryId> {
        self.index.get(key).copied()
    }

    /// Fetch a value by EntryId.
    pub fn get_by_id(&self, id: EntryId) -> Option<M::Output<'_>> {
        self.entries
            .get(id.0)
            .and_then(|slot| slot.as_ref().map(|entry| M::output(&entry.value)))
    }

    /// Fetch a key by EntryId.
    pub fn key_by_id(&self, id: EntryId) -> Option<&K> {
        self.entries
            .get(id.0)
            .and_then(|slot| slot.as_ref().map(|entry| &entry.key))
    }

    fn allocate_slot(&mut self) -> usize {
        if let Some(idx) = self.free_list.pop() {
            idx
        } else {
            self.entries.push(None);
            self.entries.len() - 1
        }
    }

    pub fn try_insert_value(&mut self, key: K, value: V) -> Result<Option<M::Removed>, StoreFull>
    where
        K: Clone,
    {
        let stored = M::store(value);
        self.try_insert_stored(key, stored)
    }

    pub fn remove_value(&mut self, key: &K) -> Option<M::Removed> {
        let id = self.index.remove(key)?;
        let entry = self.entries[id.0].take()?;
        self.free_list.push(id.0);
        self.metrics.inc_remove();
        Some(M::remove(entry.value))
    }

    fn try_insert_stored(
        &mut self,
        key: K,
        value: M::Stored,
    ) -> Result<Option<M::Removed>, StoreFull>
    where
        K: Clone,
    {
        if let Some(id) = self.index.get(&key).copied() {
            let entry = self.entries[id.0].as_mut().expect("slab entry missing");
            let previous = M::replace(&mut entry.value, value);
            self.metrics.inc_update();
            return Ok(Some(previous));
        }

        if self.index.len() >= self.capacity {
            return Err(StoreFull);
        }

        let idx = self.allocate_slot();
        self.entries[idx] = Some(Entry {
            key: key.clone(),
            value,
        });
        self.index.insert(key, EntryId(idx));
        self.metrics.inc_insert();
        Ok(None)
    }
}

impl<K, V> StoreCore<K, V> for SlabStore<K, V, SharedValue>
where
    K: Eq + Hash,
{
    fn get(&self, key: &K) -> Option<Arc<V>> {
        match self.index.get(key).and_then(|id| self.get_by_id(*id)) {
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
        self.index.contains_key(key)
    }

    fn len(&self) -> usize {
        self.index.len()
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

impl<K, V> StoreMut<K, V> for SlabStore<K, V, SharedValue>
where
    K: Eq + Hash + Clone,
{
    fn try_insert(&mut self, key: K, value: Arc<V>) -> Result<Option<Arc<V>>, StoreFull> {
        self.try_insert_stored(key, value)
    }

    fn remove(&mut self, key: &K) -> Option<Arc<V>> {
        self.remove_value(key)
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.free_list.clear();
        self.index.clear();
    }
}

impl<K, V> StoreFactory<K, V> for SlabStore<K, V, SharedValue>
where
    K: Eq + Hash + Send,
{
    type Store = SlabStore<K, V, SharedValue>;

    fn create(capacity: usize) -> Self::Store {
        Self::new(capacity)
    }
}

impl<K, V> StoreCore<K, V> for ConcurrentSlabStore<K, V>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
{
    fn get(&self, key: &K) -> Option<Arc<V>> {
        let store = self.inner.read();
        store.get(key)
    }

    fn contains(&self, key: &K) -> bool {
        let store = self.inner.read();
        store.contains(key)
    }

    fn len(&self) -> usize {
        let store = self.inner.read();
        store.len()
    }

    fn capacity(&self) -> usize {
        let store = self.inner.read();
        store.capacity()
    }

    fn metrics(&self) -> StoreMetrics {
        let store = self.inner.read();
        store.metrics()
    }

    fn record_eviction(&self) {
        let store = self.inner.read();
        store.record_eviction();
    }
}

impl<K, V> crate::store::traits::ConcurrentStore<K, V> for ConcurrentSlabStore<K, V>
where
    K: Eq + Hash + Send + Sync + Clone,
    V: Send + Sync,
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

impl<K, V> StoreFactory<K, V> for ConcurrentSlabStore<K, V>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
{
    type Store = ConcurrentSlabStore<K, V>;

    fn create(capacity: usize) -> Self::Store {
        Self::new(capacity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::traits::ConcurrentStore;

    #[test]
    fn slab_store_basic_ops() {
        let mut store = SlabStore::new(2);
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
    fn slab_store_capacity_enforced() {
        let mut store = SlabStore::new(1);
        assert_eq!(store.try_insert("k1", Arc::new("v1".to_string())), Ok(None));
        assert_eq!(
            store.try_insert("k2", Arc::new("v2".to_string())),
            Err(StoreFull)
        );
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn slab_store_entry_id_indirection() {
        let mut store = SlabStore::new(2);
        let value = Arc::new("v1".to_string());
        assert_eq!(store.try_insert("k1", value.clone()), Ok(None));
        let id = store.entry_id(&"k1").expect("missing entry id");
        assert_eq!(store.get_by_id(id), Some(value));
    }

    #[test]
    fn slab_store_key_by_id() {
        let mut store = SlabStore::new(1);
        assert_eq!(store.try_insert("k1", Arc::new("v1".to_string())), Ok(None));
        let id = store.entry_id(&"k1").expect("missing entry id");
        assert_eq!(store.key_by_id(id), Some(&"k1"));
    }

    #[test]
    fn slab_store_owned_value_mode() {
        let mut store: OwnedSlabStore<&'static str, String> = SlabStore::new(2);
        assert_eq!(store.try_insert_value("k1", "v1".to_string()), Ok(None));
        let id = store.entry_id(&"k1").expect("missing entry id");
        assert_eq!(store.get_by_id(id).map(|value| value.as_str()), Some("v1"));
        assert_eq!(store.remove_value(&"k1"), Some("v1".to_string()));
    }

    #[test]
    fn concurrent_slab_store_basic_ops() {
        let store = ConcurrentSlabStore::new(2);
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
    fn concurrent_slab_store_entry_id_roundtrip() {
        let store = ConcurrentSlabStore::new(2);
        assert_eq!(store.try_insert("k1", Arc::new("v1".to_string())), Ok(None));
        let id = store.entry_id(&"k1").expect("missing entry id");
        assert_eq!(store.get_by_id(id), Some(Arc::new("v1".to_string())));
        assert_eq!(store.key_by_id(id), Some("k1"));
    }
}
