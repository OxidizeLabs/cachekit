//! Handle-based store for zero-copy policy metadata.
//!
//! Stores values keyed by compact handles (e.g., interner IDs) rather than
//! full keys. This avoids cloning large keys while providing O(1) access
//! via `HashMap<Handle, Arc<V>>`.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                        Handle-Based Storage Flow                            │
//! │                                                                             │
//! │   User Key                                                                  │
//! │      │                                                                      │
//! │      ▼                                                                      │
//! │  ┌────────────────┐         ┌─────────────────┐         ┌──────────────┐   │
//! │  │  KeyInterner   │────────►│     Handle      │────────►│ HandleStore  │   │
//! │  │  (K -> Handle) │  intern │  (u64 / usize)  │   key   │ HashMap<H,V> │   │
//! │  └────────────────┘         └─────────────────┘         └──────────────┘   │
//! │         ▲                          │                           │           │
//! │         │                          │                           │           │
//! │         │                          ▼                           ▼           │
//! │         │                   ┌─────────────────┐         ┌──────────────┐   │
//! │         │                   │  Policy Layer   │         │   Arc<V>     │   │
//! │         │                   │  (LRU, LFU...)  │         │   (value)    │   │
//! │         │                   └─────────────────┘         └──────────────┘   │
//! │         │                          │                                       │
//! │         │  resolve                 │ evict handle                          │
//! │         └──────────────────────────┘                                       │
//! └─────────────────────────────────────────────────────────────────────────────┘
//!
//! Data Flow
//! ─────────
//!
//!   INSERT:  key ──► interner.intern() ──► handle ──► store.try_insert(handle, Arc<V>)
//!
//!   LOOKUP:  key ──► interner.get_handle() ──► handle ──► store.get(handle) ──► Arc<V>
//!
//!   EVICT:   policy.evict() ──► handle ──► store.remove(handle)
//!                                  │
//!                                  └──► interner.resolve(handle) ──► key (for callbacks)
//!
//! Component Relationships
//! ───────────────────────
//!
//!   ┌──────────────────┐
//!   │   HandleStore    │  Single-threaded, Cell-based metrics
//!   │   (not Sync)     │  Direct HashMap access
//!   └──────────────────┘
//!
//!   ┌──────────────────────────┐
//!   │  ConcurrentHandleStore   │  Thread-safe via parking_lot::RwLock
//!   │  (Send + Sync)           │  Atomic metrics counters
//!   │  impl ConcurrentStore    │  Implements trait from store::traits
//!   └──────────────────────────┘
//! ```
//!
//! ## Key Components
//!
//! - [`HandleStore`]: Single-threaded handle-backed store with `Cell`-based metrics.
//! - [`ConcurrentHandleStore`]: Thread-safe wrapper using `parking_lot::RwLock`.
//! - Internal counters track hits, misses, inserts, updates, removes, evictions.
//!
//! ## Core Operations
//!
//! | Operation     | Description                          | Complexity |
//! |---------------|--------------------------------------|------------|
//! | `try_insert`  | Insert or update by handle           | O(1) avg   |
//! | `get`         | Fetch by handle (updates metrics)    | O(1) avg   |
//! | `peek`        | Fetch without updating metrics       | O(1) avg   |
//! | `remove`      | Delete by handle                     | O(1) avg   |
//! | `clear`       | Drop all entries                     | O(n)       |
//!
//! ## Performance Trade-offs
//!
//! **Advantages:**
//! - Avoids cloning large keys—stores only compact handles
//! - `Arc<V>` enables cheap value sharing across policy and user code
//! - Metrics tracking with minimal overhead (Cell/Atomic)
//!
//! **Costs:**
//! - Requires separate `KeyInterner` for key ↔ handle mapping
//! - `Arc<V>` allocation on every insert
//! - Extra indirection compared to direct key storage
//!
//! ## When to Use
//!
//! - You already use a [`KeyInterner`](crate::ds::KeyInterner) or stable handle IDs
//! - Keys are large (strings, compound types) and cloning is expensive
//! - Policy metadata is keyed by handle rather than full keys
//! - You need `Arc<V>` semantics for value sharing
//!
//! ## Example Usage
//!
//! ```rust
//! use std::sync::Arc;
//! use cachekit::ds::KeyInterner;
//! use cachekit::store::handle::HandleStore;
//!
//! // Create interner and store
//! let mut interner = KeyInterner::new();
//! let mut store: HandleStore<u64, String> = HandleStore::new(100);
//!
//! // Intern a key to get a handle
//! let handle = interner.intern(&"user:12345".to_string());
//!
//! // Store value by handle
//! store.try_insert(handle, Arc::new("cached_data".to_string())).unwrap();
//!
//! // Retrieve by handle
//! assert_eq!(store.get(&handle), Some(Arc::new("cached_data".to_string())));
//!
//! // Check metrics
//! let metrics = store.metrics();
//! assert_eq!(metrics.hits, 1);
//! assert_eq!(metrics.inserts, 1);
//! ```
//!
//! ## Type Constraints
//!
//! - `H: Copy + Eq + Hash` — handles must be cheap to copy and hashable
//! - Values stored as `Arc<V>` — enables shared ownership
//! - For concurrent: `H: Send + Sync`, `V: Send + Sync`
//!
//! ## Thread Safety
//!
//! - [`HandleStore`] is **not** thread-safe (uses `Cell` for metrics)
//! - [`ConcurrentHandleStore`] is `Send + Sync` via `parking_lot::RwLock`
//!
//! ## Implementation Notes
//!
//! - Handles must remain stable for the lifetime of stored entries
//! - Metrics are stored separately from the map to keep the hot path simple
//! - Does **not** implement `StoreCore`/`StoreMut` (uses `Arc<V>` API instead)
//! - `record_eviction()` is separate from `remove()` for policy-driven eviction tracking

use std::cell::Cell;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use crate::store::traits::{
    ConcurrentStore, ConcurrentStoreFactory, ConcurrentStoreRead, StoreFull, StoreMetrics,
};

/// Metrics counters for single-threaded handle stores.
///
/// Uses `Cell<u64>` for interior mutability without synchronization overhead.
/// Not thread-safe—intended for use with [`HandleStore`] only.
#[derive(Debug, Default)]
struct StoreCounters {
    /// Successful lookups via `get()`.
    hits: Cell<u64>,
    /// Failed lookups via `get()`.
    misses: Cell<u64>,
    /// New key insertions.
    inserts: Cell<u64>,
    /// Value updates for existing keys.
    updates: Cell<u64>,
    /// Explicit removals via `remove()`.
    removes: Cell<u64>,
    /// Policy-driven evictions via `record_eviction()`.
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

/// Metrics counters for concurrent handle stores.
///
/// Uses `AtomicU64` with relaxed ordering for lock-free counter updates.
/// Counters may be slightly stale in concurrent reads but are eventually consistent.
#[derive(Debug, Default)]
struct ConcurrentStoreCounters {
    /// Successful lookups via `get()`.
    hits: AtomicU64,
    /// Failed lookups via `get()`.
    misses: AtomicU64,
    /// New key insertions.
    inserts: AtomicU64,
    /// Value updates for existing keys.
    updates: AtomicU64,
    /// Explicit removals via `remove()`.
    removes: AtomicU64,
    /// Policy-driven evictions via `record_eviction()`.
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

/// Single-threaded store keyed by compact handles instead of full keys.
///
/// Designed for use with [`KeyInterner`](crate::ds::KeyInterner) where handles
/// (typically `u64`) replace expensive-to-clone keys. Values are stored as
/// `Arc<V>` to enable cheap sharing between the store and policy layers.
///
/// # Type Parameters
///
/// - `H`: Handle type, must be `Copy + Eq + Hash` (typically `u64` or `usize`)
/// - `V`: Value type, wrapped in `Arc<V>` internally
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use cachekit::store::handle::HandleStore;
///
/// let mut store: HandleStore<u64, String> = HandleStore::new(3);
///
/// // Insert entries using handles (e.g., from an interner)
/// let handle_a = 1u64;
/// let handle_b = 2u64;
///
/// store.try_insert(handle_a, Arc::new("alice".into())).unwrap();
/// store.try_insert(handle_b, Arc::new("bob".into())).unwrap();
///
/// // Lookup returns Arc<V>
/// assert_eq!(store.get(&handle_a), Some(Arc::new("alice".into())));
///
/// // Peek without affecting metrics
/// assert!(store.peek(&handle_b).is_some());
///
/// // Update existing entry
/// let old = store.try_insert(handle_a, Arc::new("alice_v2".into())).unwrap();
/// assert_eq!(old, Some(Arc::new("alice".into())));
///
/// // Check metrics
/// let m = store.metrics();
/// assert_eq!(m.inserts, 2);  // handle_a, handle_b
/// assert_eq!(m.updates, 1);  // handle_a updated
/// assert_eq!(m.hits, 1);     // one get() call
/// ```
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
    /// Creates a new handle store with the specified maximum capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::store::handle::HandleStore;
    ///
    /// let store: HandleStore<u64, i32> = HandleStore::new(1000);
    /// assert_eq!(store.capacity(), 1000);
    /// assert!(store.is_empty());
    /// ```
    pub fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::with_capacity(capacity),
            capacity,
            metrics: StoreCounters::default(),
        }
    }

    /// Returns a clone of the value for the given handle.
    ///
    /// Updates hit/miss metrics. Use [`peek`](Self::peek) to avoid metric updates.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cachekit::store::handle::HandleStore;
    ///
    /// let mut store: HandleStore<u64, &str> = HandleStore::new(10);
    /// store.try_insert(1, Arc::new("value")).unwrap();
    ///
    /// assert_eq!(store.get(&1), Some(Arc::new("value")));
    /// assert_eq!(store.get(&999), None);  // miss
    ///
    /// let m = store.metrics();
    /// assert_eq!(m.hits, 1);
    /// assert_eq!(m.misses, 1);
    /// ```
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

    /// Returns a reference to the value without updating metrics.
    ///
    /// Useful for internal operations that shouldn't affect hit/miss counts.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cachekit::store::handle::HandleStore;
    ///
    /// let mut store: HandleStore<u64, i32> = HandleStore::new(10);
    /// store.try_insert(1, Arc::new(42)).unwrap();
    ///
    /// // Peek doesn't update metrics
    /// assert_eq!(store.peek(&1), Some(&Arc::new(42)));
    /// assert_eq!(store.metrics().hits, 0);
    /// ```
    pub fn peek(&self, handle: &H) -> Option<&Arc<V>> {
        self.map.get(handle)
    }

    /// Returns `true` if the handle exists in the store.
    ///
    /// Does not update metrics.
    pub fn contains(&self, handle: &H) -> bool {
        self.map.contains_key(handle)
    }

    /// Inserts or updates a value by handle.
    ///
    /// Returns `Ok(Some(old_value))` if the handle existed, `Ok(None)` for new
    /// insertions. Updates the `inserts` or `updates` metric accordingly.
    ///
    /// # Errors
    ///
    /// Returns [`StoreFull`] if the store is at capacity and the handle is new.
    /// Updates to existing handles always succeed.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cachekit::store::handle::HandleStore;
    /// use cachekit::store::traits::StoreFull;
    ///
    /// let mut store: HandleStore<u64, &str> = HandleStore::new(1);
    ///
    /// // First insert succeeds
    /// assert_eq!(store.try_insert(1, Arc::new("a")), Ok(None));
    ///
    /// // Update existing handle succeeds even at capacity
    /// assert_eq!(store.try_insert(1, Arc::new("b")), Ok(Some(Arc::new("a"))));
    ///
    /// // New handle fails when at capacity
    /// assert_eq!(store.try_insert(2, Arc::new("c")), Err(StoreFull));
    /// ```
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

    /// Removes and returns the value for the given handle.
    ///
    /// Updates the `removes` metric if an entry was removed.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cachekit::store::handle::HandleStore;
    ///
    /// let mut store: HandleStore<u64, &str> = HandleStore::new(10);
    /// store.try_insert(1, Arc::new("value")).unwrap();
    ///
    /// assert_eq!(store.remove(&1), Some(Arc::new("value")));
    /// assert_eq!(store.remove(&1), None);  // already removed
    /// assert_eq!(store.metrics().removes, 1);
    /// ```
    pub fn remove(&mut self, handle: &H) -> Option<Arc<V>> {
        let removed = self.map.remove(handle);
        if removed.is_some() {
            self.metrics.inc_remove();
        }
        removed
    }

    /// Removes all entries from the store.
    ///
    /// Does not update metrics. Capacity remains unchanged.
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Returns the current number of entries.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the store contains no entries.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns the maximum number of entries this store can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns a snapshot of the store's metrics.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cachekit::store::handle::HandleStore;
    ///
    /// let mut store: HandleStore<u64, i32> = HandleStore::new(10);
    /// store.try_insert(1, Arc::new(100)).unwrap();
    /// store.get(&1);
    /// store.get(&999);  // miss
    ///
    /// let m = store.metrics();
    /// assert_eq!(m.inserts, 1);
    /// assert_eq!(m.hits, 1);
    /// assert_eq!(m.misses, 1);
    /// ```
    pub fn metrics(&self) -> StoreMetrics {
        self.metrics.snapshot()
    }

    /// Records an eviction in the metrics.
    ///
    /// Call this when the policy evicts an entry. Separate from [`remove`](Self::remove)
    /// to distinguish user-initiated removals from policy-driven evictions.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cachekit::store::handle::HandleStore;
    ///
    /// let mut store: HandleStore<u64, i32> = HandleStore::new(10);
    /// store.try_insert(1, Arc::new(100)).unwrap();
    ///
    /// // Policy decides to evict
    /// store.remove(&1);
    /// store.record_eviction();
    ///
    /// let m = store.metrics();
    /// assert_eq!(m.removes, 1);
    /// assert_eq!(m.evictions, 1);
    /// ```
    pub fn record_eviction(&self) {
        self.metrics.inc_eviction();
    }
}

// =============================================================================
// Concurrent HandleStore
// =============================================================================

/// Thread-safe handle store using `parking_lot::RwLock`.
///
/// Provides the same functionality as [`HandleStore`] but safe for concurrent
/// access from multiple threads. Implements [`ConcurrentStore`] and
/// [`ConcurrentStoreFactory`] traits for integration with concurrent policies.
///
/// # Type Parameters
///
/// - `H`: Handle type, must be `Copy + Eq + Hash + Send + Sync`
/// - `V`: Value type, must be `Send + Sync` (wrapped in `Arc<V>`)
///
/// # Synchronization
///
/// - Read operations (`get`, `contains`, `len`) acquire a read lock
/// - Write operations (`try_insert`, `remove`, `clear`) acquire a write lock
/// - Metrics use `AtomicU64` with relaxed ordering (lock-free)
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use std::thread;
/// use cachekit::store::handle::ConcurrentHandleStore;
/// use cachekit::store::traits::ConcurrentStoreRead;
///
/// let store = Arc::new(ConcurrentHandleStore::<u64, String>::new(100));
///
/// // Spawn writers
/// let store_w = Arc::clone(&store);
/// let writer = thread::spawn(move || {
///     use cachekit::store::traits::ConcurrentStore;
///     for i in 0..10 {
///         store_w.try_insert(i, Arc::new(format!("value_{}", i))).unwrap();
///     }
/// });
///
/// // Spawn readers
/// let store_r = Arc::clone(&store);
/// let reader = thread::spawn(move || {
///     // Readers may see partial writes
///     let _ = store_r.get(&5);
///     store_r.len()
/// });
///
/// writer.join().unwrap();
/// reader.join().unwrap();
///
/// assert_eq!(store.len(), 10);
/// ```
///
/// # Trait Implementations
///
/// - [`ConcurrentStoreRead<H, V>`] — read operations
/// - [`ConcurrentStore<H, V>`] — write operations
/// - [`ConcurrentStoreFactory<H, V>`] — factory for creating instances
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
    /// Creates a new concurrent handle store with the specified capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::store::handle::ConcurrentHandleStore;
    /// use cachekit::store::traits::ConcurrentStoreRead;
    ///
    /// let store: ConcurrentHandleStore<u64, String> = ConcurrentHandleStore::new(1000);
    /// assert_eq!(store.capacity(), 1000);
    /// assert!(store.is_empty());
    /// ```
    pub fn new(capacity: usize) -> Self {
        Self {
            map: RwLock::new(HashMap::with_capacity(capacity)),
            capacity,
            metrics: ConcurrentStoreCounters::default(),
        }
    }

    /// Records an eviction in the metrics.
    ///
    /// Thread-safe via atomic increment. Call when the policy evicts an entry.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cachekit::store::handle::ConcurrentHandleStore;
    /// use cachekit::store::traits::{ConcurrentStore, ConcurrentStoreRead};
    ///
    /// let store: ConcurrentHandleStore<u64, i32> = ConcurrentHandleStore::new(10);
    /// store.try_insert(1, Arc::new(100)).unwrap();
    ///
    /// // Policy evicts the entry
    /// store.remove(&1);
    /// store.record_eviction();
    ///
    /// assert_eq!(store.metrics().evictions, 1);
    /// ```
    pub fn record_eviction(&self) {
        self.metrics.inc_eviction();
    }
}

/// Read operations for [`ConcurrentHandleStore`].
///
/// All methods acquire a read lock on the internal map. Multiple readers
/// can access the store concurrently.
impl<H, V> ConcurrentStoreRead<H, V> for ConcurrentHandleStore<H, V>
where
    H: Copy + Eq + Hash + Send + Sync,
    V: Send + Sync,
{
    /// Returns a clone of the value for the given handle.
    ///
    /// Acquires a read lock. Updates hit/miss metrics atomically.
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

    /// Returns `true` if the handle exists.
    ///
    /// Acquires a read lock. Does not update metrics.
    fn contains(&self, key: &H) -> bool {
        self.map.read().contains_key(key)
    }

    /// Returns the current number of entries.
    ///
    /// Acquires a read lock. Value may be stale by the time it's used.
    fn len(&self) -> usize {
        self.map.read().len()
    }

    /// Returns the maximum capacity.
    fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns a snapshot of the store's metrics.
    ///
    /// Metrics are read atomically but may reflect a slightly inconsistent
    /// view across different counters under high concurrency.
    fn metrics(&self) -> StoreMetrics {
        self.metrics.snapshot()
    }
}

/// Write operations for [`ConcurrentHandleStore`].
///
/// All methods acquire a write lock on the internal map. Writers have
/// exclusive access—no concurrent readers or writers during the operation.
impl<H, V> ConcurrentStore<H, V> for ConcurrentHandleStore<H, V>
where
    H: Copy + Eq + Hash + Send + Sync,
    V: Send + Sync,
{
    /// Inserts or updates a value by handle.
    ///
    /// Acquires a write lock. Returns `Ok(Some(old))` for updates,
    /// `Ok(None)` for new insertions, `Err(StoreFull)` if at capacity.
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

    /// Removes and returns the value for the given handle.
    ///
    /// Acquires a write lock. Updates `removes` metric if entry existed.
    fn remove(&self, key: &H) -> Option<Arc<V>> {
        let removed = self.map.write().remove(key);
        if removed.is_some() {
            self.metrics.inc_remove();
        }
        removed
    }

    /// Removes all entries.
    ///
    /// Acquires a write lock. Does not update metrics.
    fn clear(&self) {
        self.map.write().clear();
    }
}

/// Factory implementation for [`ConcurrentHandleStore`].
///
/// Enables generic construction in concurrent cache policies.
///
/// # Example
///
/// ```
/// use cachekit::store::handle::ConcurrentHandleStore;
/// use cachekit::store::traits::{ConcurrentStoreFactory, ConcurrentStoreRead};
///
/// fn create_store<F, H, V>(capacity: usize) -> F::Store
/// where
///     F: ConcurrentStoreFactory<H, V>,
/// {
///     F::create(capacity)
/// }
///
/// let store = create_store::<ConcurrentHandleStore<u64, String>, _, _>(100);
/// assert_eq!(store.capacity(), 100);
/// ```
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
