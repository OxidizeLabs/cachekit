use std::cell::Cell;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

use crate::store::traits::{StoreCore, StoreFactory, StoreFull, StoreMetrics, StoreMut};

/// Opaque entry handle for slab-based storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntryId(usize);

#[derive(Debug)]
struct Entry<K, V> {
    key: K,
    value: Arc<V>,
}

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

/// Slab-backed store with EntryId indirection.
#[derive(Debug)]
pub struct SlabStore<K, V> {
    entries: Vec<Option<Entry<K, V>>>,
    free_list: Vec<usize>,
    index: HashMap<K, EntryId>,
    capacity: usize,
    metrics: StoreCounters,
}

impl<K, V> SlabStore<K, V>
where
    K: Eq + Hash,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            free_list: Vec::new(),
            index: HashMap::with_capacity(capacity),
            capacity,
            metrics: StoreCounters::default(),
        }
    }

    /// Return the EntryId for a key if it exists.
    pub fn entry_id(&self, key: &K) -> Option<EntryId> {
        self.index.get(key).copied()
    }

    /// Fetch a value by EntryId.
    pub fn get_by_id(&self, id: EntryId) -> Option<Arc<V>> {
        self.entries
            .get(id.0)
            .and_then(|slot| slot.as_ref().map(|entry| entry.value.clone()))
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
}

impl<K, V> StoreCore<K, V> for SlabStore<K, V>
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

impl<K, V> StoreMut<K, V> for SlabStore<K, V>
where
    K: Eq + Hash + Clone,
{
    fn try_insert(&mut self, key: K, value: Arc<V>) -> Result<Option<Arc<V>>, StoreFull> {
        if let Some(id) = self.index.get(&key).copied() {
            let entry = self.entries[id.0].as_mut().expect("slab entry missing");
            let previous = std::mem::replace(&mut entry.value, value);
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

    fn remove(&mut self, key: &K) -> Option<Arc<V>> {
        let id = self.index.remove(key)?;
        let entry = self.entries[id.0].take()?;
        self.free_list.push(id.0);
        self.metrics.inc_remove();
        Some(entry.value)
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.free_list.clear();
        self.index.clear();
    }
}

impl<K, V> StoreFactory<K, V> for SlabStore<K, V>
where
    K: Eq + Hash + Send,
{
    type Store = SlabStore<K, V>;

    fn create(capacity: usize) -> Self::Store {
        Self::new(capacity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
