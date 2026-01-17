//! Slab-backed store with EntryId indirection.
//!
//! ## Architecture
//! - Values are stored in a `Vec<Option<Entry<...>>>` with a free-list for reuse.
//! - A `HashMap<K, EntryId>` provides key-to-slot lookup.
//! - `EntryId` is a stable handle that survives internal reallocations.
//!
//! ```text
//! entries: Vec<Option<Entry<K,V>>>
//! index:   HashMap<K, EntryId>
//! free:    Vec<usize>
//!
//! EntryId -> entries[idx] -> { key, value }
//! key -> index[key] -> EntryId
//! ```
//!
//! ## Key Components
//! - `EntryId`: opaque handle for stable slot access.
//! - `SlabStore<K, V>`: single-threaded store with direct value ownership.
//! - `ConcurrentSlabStore`: thread-safe wrapper using `Arc<V>`.
//!
//! ## Core Operations
//! - `try_insert`: inserts or updates; reuses free slots when possible.
//! - `remove`: removes by key and returns the stored value.
//! - `entry_id` + `get_by_id` / `key_by_id`: stable handle lookup.
//! - `clear`: clears all entries and free list.
//!
//! ## Performance Trade-offs
//! - Stable `EntryId` handles avoid pointer chasing and allow O(1) access.
//! - Extra indirection vs direct hash map lookup.
//! - Memory reuse reduces allocation churn in eviction-heavy workloads.
//!
//! ## When to Use
//! - You want stable IDs for policy metadata (e.g., LRU/LFU lists).
//! - You need to store values once and reference them by handle.
//! - You want predictable memory reuse under churn.
//!
//! ## Example Usage
//! ```rust
//! use cachekit::store::slab::SlabStore;
//! use cachekit::store::traits::StoreMut;
//!
//! let mut store: SlabStore<u64, String> = SlabStore::new(4);
//! store.try_insert(1, "hello".to_string()).unwrap();
//! let id = store.entry_id(&1).unwrap();
//! assert_eq!(store.get_by_id(id), Some(&"hello".to_string()));
//! ```
//!
//! ## Type Constraints
//! - `K: Eq + Hash` for key lookup.
//!
//! ## Thread Safety
//! - `SlabStore` is single-threaded (no interior mutability).
//! - `ConcurrentSlabStore` wraps with `RwLock` and is `Send + Sync`.
//!
//! ## Implementation Notes
//! - `EntryId` indices are reused; stale IDs are invalid after removal.
//! - Metrics are collected via atomic counters for compatibility with concurrent use.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use crate::store::traits::{
    ConcurrentStore, ConcurrentStoreFactory, ConcurrentStoreRead, StoreCore, StoreFactory,
    StoreFull, StoreMetrics, StoreMut,
};

/// Opaque entry handle for slab-based storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntryId(usize);

impl EntryId {
    /// Get the raw index value.
    pub fn index(&self) -> usize {
        self.0
    }
}

#[derive(Debug)]
struct Entry<K, V> {
    key: K,
    value: V,
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

// =============================================================================
// Single-threaded SlabStore
// =============================================================================

/// Slab-backed store with EntryId indirection.
///
/// Stores values directly without `Arc` wrapper for zero-overhead access.
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
    /// Create a slab store with a fixed capacity.
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

    /// Fetch a value by EntryId without touching metrics.
    pub fn get_by_id(&self, id: EntryId) -> Option<&V> {
        self.entries
            .get(id.0)
            .and_then(|slot| slot.as_ref().map(|entry| &entry.value))
    }

    /// Fetch a mutable reference to a value by EntryId.
    pub fn get_by_id_mut(&mut self, id: EntryId) -> Option<&mut V> {
        self.entries
            .get_mut(id.0)
            .and_then(|slot| slot.as_mut().map(|entry| &mut entry.value))
    }

    /// Fetch a key by EntryId.
    pub fn key_by_id(&self, id: EntryId) -> Option<&K> {
        self.entries
            .get(id.0)
            .and_then(|slot| slot.as_ref().map(|entry| &entry.key))
    }

    /// Fetch a value by key without touching metrics.
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.index.get(key).and_then(|id| self.get_by_id(*id))
    }

    /// Reserve a slot, reusing the free list when possible.
    fn allocate_slot(&mut self) -> usize {
        if let Some(idx) = self.free_list.pop() {
            idx
        } else {
            self.entries.push(None);
            self.entries.len() - 1
        }
    }

    /// Record that the policy evicted an entry.
    pub fn record_eviction(&self) {
        self.metrics.inc_eviction();
    }

    /// Remove by EntryId directly (used by policies for O(1) eviction).
    pub fn remove_by_id(&mut self, id: EntryId) -> Option<(K, V)> {
        let entry = self.entries.get_mut(id.0)?.take()?;
        self.index.remove(&entry.key);
        self.free_list.push(id.0);
        self.metrics.inc_remove();
        Some((entry.key, entry.value))
    }
}

impl<K, V> StoreCore<K, V> for SlabStore<K, V>
where
    K: Eq + Hash,
{
    fn get(&self, key: &K) -> Option<&V> {
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
}

impl<K, V> StoreMut<K, V> for SlabStore<K, V>
where
    K: Eq + Hash + Clone,
{
    fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>, StoreFull> {
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

    fn remove(&mut self, key: &K) -> Option<V> {
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
    K: Eq + Hash + Clone,
{
    type Store = SlabStore<K, V>;

    fn create(capacity: usize) -> Self::Store {
        Self::new(capacity)
    }
}

// =============================================================================
// Concurrent SlabStore
// =============================================================================

/// Concurrent slab-backed store using a `parking_lot::RwLock`.
///
/// Uses `Arc<V>` for values since references can't outlive lock guards.
#[derive(Debug)]
#[allow(clippy::type_complexity)]
pub struct ConcurrentSlabStore<K, V> {
    entries: RwLock<Vec<Option<Entry<K, Arc<V>>>>>,
    free_list: RwLock<Vec<usize>>,
    index: RwLock<HashMap<K, EntryId>>,
    capacity: usize,
    metrics: StoreCounters,
}

impl<K, V> ConcurrentSlabStore<K, V>
where
    K: Eq + Hash,
{
    /// Create a concurrent slab store with a fixed capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: RwLock::new(Vec::with_capacity(capacity)),
            free_list: RwLock::new(Vec::new()),
            index: RwLock::new(HashMap::with_capacity(capacity)),
            capacity,
            metrics: StoreCounters::default(),
        }
    }

    /// Return the EntryId for a key if it exists.
    pub fn entry_id(&self, key: &K) -> Option<EntryId> {
        self.index.read().get(key).copied()
    }

    /// Fetch a value by EntryId.
    pub fn get_by_id(&self, id: EntryId) -> Option<Arc<V>> {
        self.entries
            .read()
            .get(id.0)
            .and_then(|slot| slot.as_ref().map(|entry| Arc::clone(&entry.value)))
    }

    /// Fetch a key by EntryId.
    pub fn key_by_id(&self, id: EntryId) -> Option<K>
    where
        K: Clone,
    {
        self.entries
            .read()
            .get(id.0)
            .and_then(|slot| slot.as_ref().map(|entry| entry.key.clone()))
    }

    /// Record that the policy evicted an entry.
    pub fn record_eviction(&self) {
        self.metrics.inc_eviction();
    }

    fn allocate_slot(&self) -> usize {
        let mut free_list = self.free_list.write();
        if let Some(idx) = free_list.pop() {
            idx
        } else {
            let mut entries = self.entries.write();
            entries.push(None);
            entries.len() - 1
        }
    }
}

impl<K, V> ConcurrentStoreRead<K, V> for ConcurrentSlabStore<K, V>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
{
    fn get(&self, key: &K) -> Option<Arc<V>> {
        let index = self.index.read();
        let id = index.get(key)?;
        let entries = self.entries.read();
        match entries.get(id.0).and_then(|slot| slot.as_ref()) {
            Some(entry) => {
                self.metrics.inc_hit();
                Some(Arc::clone(&entry.value))
            },
            None => {
                self.metrics.inc_miss();
                None
            },
        }
    }

    fn contains(&self, key: &K) -> bool {
        self.index.read().contains_key(key)
    }

    fn len(&self) -> usize {
        self.index.read().len()
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn metrics(&self) -> StoreMetrics {
        self.metrics.snapshot()
    }
}

impl<K, V> ConcurrentStore<K, V> for ConcurrentSlabStore<K, V>
where
    K: Eq + Hash + Send + Sync + Clone,
    V: Send + Sync,
{
    fn try_insert(&self, key: K, value: Arc<V>) -> Result<Option<Arc<V>>, StoreFull> {
        // Check for update case first
        {
            let index = self.index.read();
            if let Some(id) = index.get(&key).copied() {
                drop(index);
                let mut entries = self.entries.write();
                if let Some(slot) = entries.get_mut(id.0) {
                    if let Some(entry) = slot.as_mut() {
                        let previous = std::mem::replace(&mut entry.value, value);
                        self.metrics.inc_update();
                        return Ok(Some(previous));
                    }
                }
            }
        }

        // Insert case - check capacity
        {
            let index = self.index.read();
            if index.len() >= self.capacity {
                return Err(StoreFull);
            }
        }

        let idx = self.allocate_slot();
        {
            let mut entries = self.entries.write();
            entries[idx] = Some(Entry {
                key: key.clone(),
                value,
            });
        }
        {
            let mut index = self.index.write();
            index.insert(key, EntryId(idx));
        }
        self.metrics.inc_insert();
        Ok(None)
    }

    fn remove(&self, key: &K) -> Option<Arc<V>> {
        let id = {
            let mut index = self.index.write();
            index.remove(key)?
        };
        let entry = {
            let mut entries = self.entries.write();
            entries.get_mut(id.0)?.take()?
        };
        {
            let mut free_list = self.free_list.write();
            free_list.push(id.0);
        }
        self.metrics.inc_remove();
        Some(entry.value)
    }

    fn clear(&self) {
        self.entries.write().clear();
        self.free_list.write().clear();
        self.index.write().clear();
    }
}

impl<K, V> ConcurrentStoreFactory<K, V> for ConcurrentSlabStore<K, V>
where
    K: Eq + Hash + Send + Sync + Clone,
    V: Send + Sync,
{
    type Store = ConcurrentSlabStore<K, V>;

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
    fn slab_store_basic_ops() {
        let mut store = SlabStore::new(2);
        assert_eq!(store.try_insert("k1", "v1".to_string()), Ok(None));
        assert_eq!(store.get(&"k1"), Some(&"v1".to_string()));
        assert!(store.contains(&"k1"));
        assert_eq!(store.len(), 1);
        assert_eq!(store.capacity(), 2);
        assert_eq!(store.remove(&"k1"), Some("v1".to_string()));
        assert!(!store.contains(&"k1"));
    }

    #[test]
    fn slab_store_returns_reference() {
        let mut store = SlabStore::new(2);
        store.try_insert("k1", "hello".to_string()).unwrap();

        // get() returns &V
        let value: &String = store.get(&"k1").unwrap();
        assert_eq!(value, "hello");

        // get_by_id() also returns &V
        let id = store.entry_id(&"k1").unwrap();
        let by_id: &String = store.get_by_id(id).unwrap();
        assert_eq!(by_id, "hello");
    }

    #[test]
    fn slab_store_capacity_enforced() {
        let mut store = SlabStore::new(1);
        assert_eq!(store.try_insert("k1", "v1".to_string()), Ok(None));
        assert_eq!(store.try_insert("k2", "v2".to_string()), Err(StoreFull));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn slab_store_entry_id_indirection() {
        let mut store = SlabStore::new(2);
        assert_eq!(store.try_insert("k1", "v1".to_string()), Ok(None));
        let id = store.entry_id(&"k1").expect("missing entry id");
        assert_eq!(store.get_by_id(id), Some(&"v1".to_string()));
    }

    #[test]
    fn slab_store_key_by_id() {
        let mut store = SlabStore::new(1);
        assert_eq!(store.try_insert("k1", "v1".to_string()), Ok(None));
        let id = store.entry_id(&"k1").expect("missing entry id");
        assert_eq!(store.key_by_id(id), Some(&"k1"));
    }

    #[test]
    fn slab_store_remove_by_id() {
        let mut store = SlabStore::new(2);
        store.try_insert("k1", "v1".to_string()).unwrap();
        let id = store.entry_id(&"k1").unwrap();
        let (key, value) = store.remove_by_id(id).unwrap();
        assert_eq!(key, "k1");
        assert_eq!(value, "v1".to_string());
        assert!(!store.contains(&"k1"));
    }

    #[test]
    fn slab_store_slot_reuse() {
        let mut store = SlabStore::new(2);
        store.try_insert("k1", "v1".to_string()).unwrap();
        let id1 = store.entry_id(&"k1").unwrap();

        store.remove(&"k1");

        // New insert should reuse the freed slot
        store.try_insert("k2", "v2".to_string()).unwrap();
        let id2 = store.entry_id(&"k2").unwrap();

        assert_eq!(id1.index(), id2.index());
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
