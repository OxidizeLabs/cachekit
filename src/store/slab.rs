//! Slab-backed store with stable `EntryId` indirection.
//!
//! Provides arena-style storage where values are accessed via stable handles
//! ([`EntryId`]) rather than direct pointers. Handles remain valid across
//! internal reallocations, making this ideal for policy metadata that needs
//! to reference entries without pointer chasing.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                          Slab Store Layout                                  │
//! │                                                                             │
//! │   ┌─────────────────────────────────────────────────────────────────────┐  │
//! │   │  Key Lookup                              Slot Access                │  │
//! │   │                                                                     │  │
//! │   │   "user:1"  ──┐                    ┌─► ┌─────────────────────┐     │  │
//! │   │               │    ┌───────────┐   │   │ entries[0]          │     │  │
//! │   │   "user:2"  ──┼──► │  index    │ ──┤   │   key: "user:1"     │     │  │
//! │   │               │    │ HashMap   │   │   │   value: {...}      │     │  │
//! │   │   "user:3"  ──┘    │ <K,EntryId>   │   ├─────────────────────┤     │  │
//! │   │                    └───────────┘   ├─► │ entries[1]          │     │  │
//! │   │                          │         │   │   key: "user:2"     │     │  │
//! │   │                          ▼         │   │   value: {...}      │     │  │
//! │   │                    ┌───────────┐   │   ├─────────────────────┤     │  │
//! │   │                    │  EntryId  │ ──┘   │ entries[2]: None    │◄──┐ │  │
//! │   │                    │  (usize)  │       │   (free slot)       │   │ │  │
//! │   │                    └───────────┘       ├─────────────────────┤   │ │  │
//! │   │                                        │ entries[3]          │   │ │  │
//! │   │                                        │   key: "user:3"     │   │ │  │
//! │   │                                        │   value: {...}      │   │ │  │
//! │   │                                        └─────────────────────┘   │ │  │
//! │   │                                                                  │ │  │
//! │   │   ┌────────────────┐                                             │ │  │
//! │   │   │   free_list    │  [2] ───────────────────────────────────────┘ │  │
//! │   │   │   Vec<usize>   │  Tracks empty slots for reuse                 │  │
//! │   │   └────────────────┘                                               │  │
//! │   └─────────────────────────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────────────────┘
//!
//! Data Flow
//! ─────────
//!
//!   INSERT (new key):
//!     1. Check capacity
//!     2. Allocate slot: pop from free_list OR push new entry
//!     3. Store Entry { key, value } at slot
//!     4. index.insert(key, EntryId(slot))
//!
//!   LOOKUP by key:
//!     key ──► index.get(key) ──► EntryId ──► entries[id.0] ──► value
//!
//!   LOOKUP by EntryId:
//!     EntryId ──► entries[id.0] ──► value   (O(1), no hash)
//!
//!   REMOVE:
//!     1. index.remove(key) ──► EntryId
//!     2. entries[id.0].take() ──► Entry
//!     3. free_list.push(id.0)
//!
//! Slot Lifecycle
//! ──────────────
//!
//!   [Allocated] ──remove()──► [Free] ──insert()──► [Reused]
//!        │                       │                     │
//!        │                       └── free_list ────────┘
//!        │
//!        └── entries[idx] = Some(Entry { key, value })
//!
//! Component Comparison
//! ────────────────────
//!
//!   │ Store              │ Value Type │ EntryId Access │ Thread Safety │
//!   │────────────────────│────────────│────────────────│───────────────│
//!   │ SlabStore          │ V (owned)  │ O(1) &V        │ Single-thread │
//!   │ ConcurrentSlabStore│ Arc<V>     │ O(1) Arc<V>    │ Send + Sync   │
//! ```
//!
//! ## Key Components
//!
//! - [`EntryId`]: Opaque handle for stable O(1) slot access
//! - [`SlabStore`]: Single-threaded store with direct `V` ownership
//! - [`ConcurrentSlabStore`]: Thread-safe store with `Arc<V>` values
//!
//! ## Core Operations
//!
//! | Operation       | Description                           | Complexity |
//! |-----------------|---------------------------------------|------------|
//! | `try_insert`    | Insert/update, reuses free slots      | O(1) avg   |
//! | `get`           | Lookup by key (updates metrics)       | O(1) avg   |
//! | `get_by_id`     | Lookup by EntryId (no hash)           | O(1)       |
//! | `entry_id`      | Get stable handle for a key           | O(1) avg   |
//! | `remove`        | Remove by key, slot goes to free list | O(1) avg   |
//! | `remove_by_id`  | Remove by handle (for eviction)       | O(1)       |
//!
//! ## Performance Trade-offs
//!
//! **Advantages:**
//! - Stable handles survive internal Vec reallocations
//! - O(1) access by `EntryId` without hashing
//! - Slot reuse reduces allocation churn in eviction-heavy workloads
//! - Ideal for policy metadata (LRU lists, frequency counters)
//!
//! **Costs:**
//! - Extra indirection: key → index → EntryId → entries
//! - Keys must be `Clone` for insertion (stored in entry)
//! - Stale `EntryId` after removal may access wrong/empty slot
//!
//! ## When to Use
//!
//! - Policy metadata needs stable references to entries
//! - Eviction patterns benefit from slot reuse
//! - You need O(1) lookup by handle (e.g., for LRU list nodes)
//! - Memory locality matters (entries in contiguous Vec)
//!
//! ## Example Usage
//!
//! ```rust
//! use cachekit::store::slab::{SlabStore, EntryId};
//! use cachekit::store::traits::{StoreCore, StoreMut};
//!
//! let mut store: SlabStore<String, Vec<u8>> = SlabStore::new(100);
//!
//! // Insert and get stable handle
//! store.try_insert("image.png".into(), vec![0x89, 0x50]).unwrap();
//! let id: EntryId = store.entry_id(&"image.png".into()).unwrap();
//!
//! // Access by handle (O(1), no hash lookup)
//! assert_eq!(store.get_by_id(id), Some(&vec![0x89, 0x50]));
//!
//! // Access key from handle
//! assert_eq!(store.key_by_id(id), Some(&"image.png".into()));
//!
//! // Remove and verify slot reuse
//! store.remove(&"image.png".into());
//! store.try_insert("icon.png".into(), vec![0x00]).unwrap();
//! let new_id = store.entry_id(&"icon.png".into()).unwrap();
//! assert_eq!(id.index(), new_id.index());  // Same slot reused
//! ```
//!
//! ## Type Constraints
//!
//! - `K: Eq + Hash` — keys must be hashable for index lookup
//! - `K: Clone` — keys are stored in entries (required for `StoreMut`)
//! - Concurrent: `K: Send + Sync`, `V: Send + Sync`
//!
//! ## Thread Safety
//!
//! - [`SlabStore`] is **not** thread-safe (single-threaded only)
//! - [`ConcurrentSlabStore`] is `Send + Sync` via `parking_lot::RwLock`
//!
//! ## Implementation Notes
//!
//! - `EntryId` indices are reused; stale IDs after removal are undefined
//! - Metrics use atomic counters for concurrent compatibility
//! - `remove_by_id` enables O(1) eviction without key lookup

use std::hash::Hash;
#[cfg(feature = "concurrency")]
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "concurrency")]
use parking_lot::RwLock;
use rustc_hash::FxHashMap;

#[cfg(feature = "concurrency")]
use crate::store::traits::{ConcurrentStore, ConcurrentStoreFactory, ConcurrentStoreRead};
use crate::store::traits::{StoreCore, StoreFactory, StoreFull, StoreMetrics, StoreMut};

/// Opaque handle for stable O(1) access to slab entries.
///
/// An `EntryId` is a lightweight identifier (wrapping a `usize` index) that
/// provides direct access to a slab slot without hashing. Handles remain
/// valid as long as the entry exists—removal invalidates the handle.
///
/// # Stability
///
/// Unlike pointers, `EntryId` values survive internal `Vec` reallocations.
/// The index points to a slot, not a memory address.
///
/// # Invalidation
///
/// After `remove()` or `remove_by_id()`, the slot is returned to the free
/// list and may be reused. Accessing a stale `EntryId` may return `None`
/// or a different entry's data.
///
/// # Example
///
/// ```
/// use cachekit::store::slab::{SlabStore, EntryId};
/// use cachekit::store::traits::StoreMut;
///
/// let mut store: SlabStore<&str, i32> = SlabStore::new(10);
/// store.try_insert("key", 42).unwrap();
///
/// // Get stable handle
/// let id: EntryId = store.entry_id(&"key").unwrap();
///
/// // Access by handle (O(1))
/// assert_eq!(store.get_by_id(id), Some(&42));
///
/// // Inspect raw index (for debugging/logging)
/// println!("Entry at slot {}", id.index());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntryId(usize);

impl EntryId {
    /// Returns the raw slot index.
    ///
    /// Useful for debugging, logging, or custom data structures that need
    /// to track slot positions.
    pub fn index(&self) -> usize {
        self.0
    }
}

/// Internal entry holding key-value pair in a slab slot.
#[derive(Debug)]
struct Entry<K, V> {
    key: K,
    value: V,
}

/// Metrics counters using atomics for thread-safe updates.
///
/// All counters use `Ordering::Relaxed` for low-overhead increments.
#[derive(Debug, Default)]
struct StoreCounters {
    /// Successful lookups via `get()`.
    hits: AtomicU64,
    /// Failed lookups via `get()`.
    misses: AtomicU64,
    /// New key insertions.
    inserts: AtomicU64,
    /// Value updates for existing keys.
    updates: AtomicU64,
    /// Explicit removals via `remove()` or `remove_by_id()`.
    removes: AtomicU64,
    /// Policy-driven evictions via `record_eviction()`.
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

/// Single-threaded slab-backed store with stable `EntryId` handles.
///
/// Stores values directly (no `Arc`) in a contiguous `Vec` with a free-list
/// for slot reuse. Provides O(1) access by [`EntryId`] without hashing,
/// making it ideal for policy metadata that needs stable references.
///
/// # Type Parameters
///
/// - `K`: Key type, must be `Eq + Hash + Clone`
/// - `V`: Value type, stored directly (owned)
///
/// # Example
///
/// ```
/// use cachekit::store::slab::{SlabStore, EntryId};
/// use cachekit::store::traits::{StoreCore, StoreMut};
///
/// let mut store: SlabStore<String, Vec<u8>> = SlabStore::new(100);
///
/// // Insert entries
/// store.try_insert("file1.txt".into(), vec![1, 2, 3]).unwrap();
/// store.try_insert("file2.txt".into(), vec![4, 5, 6]).unwrap();
///
/// // Get stable handle for policy metadata
/// let id1 = store.entry_id(&"file1.txt".into()).unwrap();
///
/// // O(1) access by handle (no hash lookup)
/// assert_eq!(store.get_by_id(id1), Some(&vec![1, 2, 3]));
///
/// // Mutable access by handle
/// if let Some(data) = store.get_by_id_mut(id1) {
///     data.push(99);
/// }
///
/// // Remove and observe slot reuse
/// store.remove(&"file1.txt".into());
/// store.try_insert("file3.txt".into(), vec![7, 8, 9]).unwrap();
/// let id3 = store.entry_id(&"file3.txt".into()).unwrap();
/// assert_eq!(id1.index(), id3.index());  // Same slot reused
/// ```
///
/// # Policy Integration
///
/// ```
/// use cachekit::store::slab::{SlabStore, EntryId};
/// use cachekit::store::traits::StoreMut;
///
/// let mut store: SlabStore<u64, String> = SlabStore::new(10);
/// store.try_insert(1, "value".into()).unwrap();
///
/// // Policy tracks EntryId for O(1) eviction
/// let victim_id = store.entry_id(&1).unwrap();
///
/// // Evict by handle (no key lookup needed)
/// let (key, value) = store.remove_by_id(victim_id).unwrap();
/// store.record_eviction();
///
/// assert_eq!(key, 1);
/// assert_eq!(value, "value");
/// ```
#[derive(Debug)]
pub struct SlabStore<K, V> {
    entries: Vec<Option<Entry<K, V>>>,
    free_list: Vec<usize>,
    index: FxHashMap<K, EntryId>,
    capacity: usize,
    metrics: StoreCounters,
}

impl<K, V> SlabStore<K, V>
where
    K: Eq + Hash,
{
    /// Creates a slab store with the specified maximum capacity.
    ///
    /// Pre-allocates internal structures for the given capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::store::slab::SlabStore;
    /// use cachekit::store::traits::StoreCore;
    ///
    /// let store: SlabStore<String, i32> = SlabStore::new(1000);
    /// assert_eq!(store.capacity(), 1000);
    /// assert!(store.is_empty());
    /// ```
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            free_list: Vec::new(),
            index: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            capacity,
            metrics: StoreCounters::default(),
        }
    }

    /// Returns the `EntryId` handle for a key.
    ///
    /// The handle provides O(1) access to the entry without further hashing.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::store::slab::SlabStore;
    /// use cachekit::store::traits::StoreMut;
    ///
    /// let mut store: SlabStore<&str, i32> = SlabStore::new(10);
    /// store.try_insert("key", 42).unwrap();
    ///
    /// let id = store.entry_id(&"key").unwrap();
    /// assert_eq!(store.get_by_id(id), Some(&42));
    ///
    /// assert!(store.entry_id(&"missing").is_none());
    /// ```
    pub fn entry_id(&self, key: &K) -> Option<EntryId> {
        self.index.get(key).copied()
    }

    /// Returns a reference to the value at the given `EntryId`.
    ///
    /// O(1) direct slot access without hashing. Does not update metrics.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::store::slab::SlabStore;
    /// use cachekit::store::traits::StoreMut;
    ///
    /// let mut store: SlabStore<&str, i32> = SlabStore::new(10);
    /// store.try_insert("key", 100).unwrap();
    /// let id = store.entry_id(&"key").unwrap();
    ///
    /// assert_eq!(store.get_by_id(id), Some(&100));
    /// ```
    pub fn get_by_id(&self, id: EntryId) -> Option<&V> {
        self.entries
            .get(id.0)
            .and_then(|slot| slot.as_ref().map(|entry| &entry.value))
    }

    /// Returns a mutable reference to the value at the given `EntryId`.
    ///
    /// Allows in-place modification without remove/insert cycle.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::store::slab::SlabStore;
    /// use cachekit::store::traits::StoreMut;
    ///
    /// let mut store: SlabStore<&str, Vec<i32>> = SlabStore::new(10);
    /// store.try_insert("nums", vec![1, 2, 3]).unwrap();
    /// let id = store.entry_id(&"nums").unwrap();
    ///
    /// // Modify in place
    /// if let Some(nums) = store.get_by_id_mut(id) {
    ///     nums.push(4);
    /// }
    ///
    /// assert_eq!(store.get_by_id(id), Some(&vec![1, 2, 3, 4]));
    /// ```
    pub fn get_by_id_mut(&mut self, id: EntryId) -> Option<&mut V> {
        self.entries
            .get_mut(id.0)
            .and_then(|slot| slot.as_mut().map(|entry| &mut entry.value))
    }

    /// Returns a reference to the key at the given `EntryId`.
    ///
    /// Useful for logging or callbacks that need the original key.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::store::slab::SlabStore;
    /// use cachekit::store::traits::StoreMut;
    ///
    /// let mut store: SlabStore<String, i32> = SlabStore::new(10);
    /// store.try_insert("my_key".into(), 42).unwrap();
    /// let id = store.entry_id(&"my_key".into()).unwrap();
    ///
    /// assert_eq!(store.key_by_id(id), Some(&"my_key".into()));
    /// ```
    pub fn key_by_id(&self, id: EntryId) -> Option<&K> {
        self.entries
            .get(id.0)
            .and_then(|slot| slot.as_ref().map(|entry| &entry.key))
    }

    /// Returns a reference to the value without updating metrics.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::store::slab::SlabStore;
    /// use cachekit::store::traits::{StoreCore, StoreMut};
    ///
    /// let mut store: SlabStore<&str, i32> = SlabStore::new(10);
    /// store.try_insert("key", 42).unwrap();
    ///
    /// // Peek doesn't update metrics
    /// assert_eq!(store.peek(&"key"), Some(&42));
    /// assert_eq!(store.metrics().hits, 0);
    /// ```
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.index.get(key).and_then(|id| self.get_by_id(*id))
    }

    /// Allocates a slot, reusing from free list when possible.
    fn allocate_slot(&mut self) -> usize {
        if let Some(idx) = self.free_list.pop() {
            idx
        } else {
            self.entries.push(None);
            self.entries.len() - 1
        }
    }

    /// Records an eviction in the metrics.
    ///
    /// Call when the policy evicts an entry. Separate from `remove()` to
    /// distinguish user-initiated removals from policy-driven evictions.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::store::slab::SlabStore;
    /// use cachekit::store::traits::{StoreCore, StoreMut};
    ///
    /// let mut store: SlabStore<&str, i32> = SlabStore::new(10);
    /// store.try_insert("victim", 1).unwrap();
    /// let id = store.entry_id(&"victim").unwrap();
    ///
    /// // Policy evicts
    /// store.remove_by_id(id);
    /// store.record_eviction();
    ///
    /// let m = store.metrics();
    /// assert_eq!(m.removes, 1);
    /// assert_eq!(m.evictions, 1);
    /// ```
    pub fn record_eviction(&self) {
        self.metrics.inc_eviction();
    }

    /// Removes an entry by `EntryId`, returning the key and value.
    ///
    /// Enables O(1) eviction when the policy tracks handles. The slot is
    /// returned to the free list for reuse.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::store::slab::SlabStore;
    /// use cachekit::store::traits::{StoreCore, StoreMut};
    ///
    /// let mut store: SlabStore<&str, i32> = SlabStore::new(10);
    /// store.try_insert("key", 42).unwrap();
    /// let id = store.entry_id(&"key").unwrap();
    ///
    /// let (key, value) = store.remove_by_id(id).unwrap();
    /// assert_eq!(key, "key");
    /// assert_eq!(value, 42);
    ///
    /// // Slot is now free for reuse
    /// assert!(!store.contains(&"key"));
    /// ```
    pub fn remove_by_id(&mut self, id: EntryId) -> Option<(K, V)> {
        let entry = self.entries.get_mut(id.0)?.take()?;
        self.index.remove(&entry.key);
        self.free_list.push(id.0);
        self.metrics.inc_remove();
        Some((entry.key, entry.value))
    }
}

/// Read operations for [`SlabStore`].
///
/// Returns borrowed `&V` references for zero-copy access.
impl<K, V> StoreCore<K, V> for SlabStore<K, V>
where
    K: Eq + Hash,
{
    /// Returns a reference to the value for the given key.
    ///
    /// Updates hit/miss metrics. Use [`peek`](SlabStore::peek) or
    /// [`get_by_id`](SlabStore::get_by_id) to avoid metric updates.
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

    /// Returns `true` if the key exists. Does not update metrics.
    fn contains(&self, key: &K) -> bool {
        self.index.contains_key(key)
    }

    /// Returns the current number of entries.
    fn len(&self) -> usize {
        self.index.len()
    }

    /// Returns the logical capacity limit.
    fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns a snapshot of the store's metrics.
    fn metrics(&self) -> StoreMetrics {
        self.metrics.snapshot()
    }
}

/// Write operations for [`SlabStore`].
///
/// Takes and returns owned `V` values. Keys must be `Clone` since they're
/// stored alongside values in the slab.
impl<K, V> StoreMut<K, V> for SlabStore<K, V>
where
    K: Eq + Hash + Clone,
{
    /// Inserts or updates a value.
    ///
    /// For new keys, allocates a slot (reusing from free list if available).
    /// For existing keys, updates in place and returns the old value.
    ///
    /// # Errors
    ///
    /// Returns [`StoreFull`] if at capacity and the key is new.
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

    /// Removes and returns the value for the given key.
    ///
    /// The slot is returned to the free list for reuse.
    fn remove(&mut self, key: &K) -> Option<V> {
        let id = self.index.remove(key)?;
        let entry = self.entries[id.0].take()?;
        self.free_list.push(id.0);
        self.metrics.inc_remove();
        Some(entry.value)
    }

    /// Removes all entries and clears the free list.
    fn clear(&mut self) {
        self.entries.clear();
        self.free_list.clear();
        self.index.clear();
    }
}

/// Factory for creating [`SlabStore`] instances.
///
/// # Example
///
/// ```
/// use cachekit::store::slab::SlabStore;
/// use cachekit::store::traits::{StoreFactory, StoreCore};
///
/// fn create_store<F: StoreFactory<String, i32>>(cap: usize) -> F::Store {
///     F::create(cap)
/// }
///
/// let store = create_store::<SlabStore<String, i32>>(100);
/// assert_eq!(store.capacity(), 100);
/// ```
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

/// Thread-safe slab-backed store with stable `EntryId` handles.
///
/// Provides the same functionality as [`SlabStore`] but safe for concurrent
/// access. Uses `Arc<V>` for values since references cannot outlive lock
/// guards. Each internal structure (`entries`, `index`, `free_list`) has
/// its own `RwLock` for fine-grained locking.
///
/// # Type Parameters
///
/// - `K`: Key type, must be `Eq + Hash + Clone + Send + Sync`
/// - `V`: Value type, must be `Send + Sync` (wrapped in `Arc<V>`)
///
/// # Synchronization
///
/// - Read operations acquire read locks on `index` and `entries`
/// - Write operations acquire write locks as needed
/// - Metrics use atomic counters (lock-free)
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use std::thread;
/// use cachekit::store::slab::ConcurrentSlabStore;
/// use cachekit::store::traits::{ConcurrentStore, ConcurrentStoreRead};
///
/// let store = Arc::new(ConcurrentSlabStore::<u64, String>::new(100));
///
/// // Spawn writers
/// let handles: Vec<_> = (0..4).map(|t| {
///     let store = Arc::clone(&store);
///     thread::spawn(move || {
///         for i in 0..25 {
///             let key = (t * 25 + i) as u64;
///             store.try_insert(key, Arc::new(format!("v{}", key))).unwrap();
///         }
///     })
/// }).collect();
///
/// for h in handles {
///     h.join().unwrap();
/// }
///
/// assert_eq!(store.len(), 100);
///
/// // Access by EntryId (key 0 is guaranteed to exist)
/// let id = store.entry_id(&0).unwrap();
/// assert!(store.get_by_id(id).is_some());
/// ```
#[cfg(feature = "concurrency")]
#[derive(Debug)]
#[allow(clippy::type_complexity)]
pub struct ConcurrentSlabStore<K, V> {
    entries: RwLock<Vec<Option<Entry<K, Arc<V>>>>>,
    free_list: RwLock<Vec<usize>>,
    index: RwLock<FxHashMap<K, EntryId>>,
    capacity: usize,
    metrics: StoreCounters,
}

#[cfg(feature = "concurrency")]
impl<K, V> ConcurrentSlabStore<K, V>
where
    K: Eq + Hash,
{
    /// Creates a concurrent slab store with the specified capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::store::slab::ConcurrentSlabStore;
    /// use cachekit::store::traits::ConcurrentStoreRead;
    ///
    /// let store: ConcurrentSlabStore<String, Vec<u8>> =
    ///     ConcurrentSlabStore::new(1000);
    /// assert_eq!(store.capacity(), 1000);
    /// ```
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: RwLock::new(Vec::with_capacity(capacity)),
            free_list: RwLock::new(Vec::new()),
            index: RwLock::new(FxHashMap::with_capacity_and_hasher(
                capacity,
                Default::default(),
            )),
            capacity,
            metrics: StoreCounters::default(),
        }
    }

    /// Returns the `EntryId` handle for a key.
    ///
    /// Acquires read lock on index.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cachekit::store::slab::ConcurrentSlabStore;
    /// use cachekit::store::traits::ConcurrentStore;
    ///
    /// let store: ConcurrentSlabStore<&str, i32> = ConcurrentSlabStore::new(10);
    /// store.try_insert("key", Arc::new(42)).unwrap();
    ///
    /// let id = store.entry_id(&"key").unwrap();
    /// assert!(store.get_by_id(id).is_some());
    /// ```
    pub fn entry_id(&self, key: &K) -> Option<EntryId> {
        self.index.read().get(key).copied()
    }

    /// Returns a clone of the value at the given `EntryId`.
    ///
    /// Acquires read lock on entries. Returns `Arc<V>` that can be held
    /// after the lock is released.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cachekit::store::slab::ConcurrentSlabStore;
    /// use cachekit::store::traits::ConcurrentStore;
    ///
    /// let store: ConcurrentSlabStore<&str, i32> = ConcurrentSlabStore::new(10);
    /// store.try_insert("key", Arc::new(42)).unwrap();
    /// let id = store.entry_id(&"key").unwrap();
    ///
    /// let value: Arc<i32> = store.get_by_id(id).unwrap();
    /// assert_eq!(*value, 42);
    /// ```
    pub fn get_by_id(&self, id: EntryId) -> Option<Arc<V>> {
        self.entries
            .read()
            .get(id.0)
            .and_then(|slot| slot.as_ref().map(|entry| Arc::clone(&entry.value)))
    }

    /// Returns a clone of the key at the given `EntryId`.
    ///
    /// Acquires read lock on entries.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cachekit::store::slab::ConcurrentSlabStore;
    /// use cachekit::store::traits::ConcurrentStore;
    ///
    /// let store: ConcurrentSlabStore<String, i32> = ConcurrentSlabStore::new(10);
    /// store.try_insert("my_key".into(), Arc::new(42)).unwrap();
    /// let id = store.entry_id(&"my_key".into()).unwrap();
    ///
    /// assert_eq!(store.key_by_id(id), Some("my_key".into()));
    /// ```
    pub fn key_by_id(&self, id: EntryId) -> Option<K>
    where
        K: Clone,
    {
        self.entries
            .read()
            .get(id.0)
            .and_then(|slot| slot.as_ref().map(|entry| entry.key.clone()))
    }

    /// Records an eviction in the metrics.
    ///
    /// Thread-safe via atomic increment.
    pub fn record_eviction(&self) {
        self.metrics.inc_eviction();
    }

    /// Allocates a slot, reusing from free list when possible.
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

/// Read operations for [`ConcurrentSlabStore`].
///
/// Acquires read locks on internal structures as needed.
#[cfg(feature = "concurrency")]
impl<K, V> ConcurrentStoreRead<K, V> for ConcurrentSlabStore<K, V>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
{
    /// Returns a clone of the value for the given key.
    ///
    /// Acquires read locks on index and entries sequentially.
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

    /// Returns `true` if the key exists. Acquires read lock on index.
    fn contains(&self, key: &K) -> bool {
        self.index.read().contains_key(key)
    }

    /// Returns the current number of entries.
    ///
    /// Acquires read lock on index. Value may be stale under concurrency.
    fn len(&self) -> usize {
        self.index.read().len()
    }

    /// Returns the logical capacity limit.
    fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns a snapshot of the store's metrics.
    fn metrics(&self) -> StoreMetrics {
        self.metrics.snapshot()
    }
}

/// Write operations for [`ConcurrentSlabStore`].
///
/// Uses fine-grained locking—different internal structures are locked
/// independently to minimize contention.
#[cfg(feature = "concurrency")]
impl<K, V> ConcurrentStore<K, V> for ConcurrentSlabStore<K, V>
where
    K: Eq + Hash + Send + Sync + Clone,
    V: Send + Sync,
{
    /// Inserts or updates a value.
    ///
    /// Acquires locks in stages to minimize hold time:
    /// 1. Read lock on index to check for existing key
    /// 2. Write lock on entries for update (if key exists)
    /// 3. Write locks on free_list, entries, index for new insert
    ///
    /// # Errors
    ///
    /// Returns [`StoreFull`] if at capacity and the key is new.
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

    /// Removes and returns the value for the given key.
    ///
    /// Acquires write locks on index, entries, and free_list in sequence.
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

    /// Removes all entries.
    ///
    /// Acquires write locks on all internal structures.
    fn clear(&self) {
        self.entries.write().clear();
        self.free_list.write().clear();
        self.index.write().clear();
    }
}

/// Factory for creating [`ConcurrentSlabStore`] instances.
///
/// # Example
///
/// ```
/// use cachekit::store::slab::ConcurrentSlabStore;
/// use cachekit::store::traits::{ConcurrentStoreFactory, ConcurrentStoreRead};
///
/// fn create<F: ConcurrentStoreFactory<String, i32>>(cap: usize) -> F::Store {
///     F::create(cap)
/// }
///
/// let store = create::<ConcurrentSlabStore<String, i32>>(100);
/// assert_eq!(store.capacity(), 100);
/// ```
#[cfg(feature = "concurrency")]
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

    #[cfg(feature = "concurrency")]
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

    #[cfg(feature = "concurrency")]
    #[test]
    fn concurrent_slab_store_entry_id_roundtrip() {
        let store = ConcurrentSlabStore::new(2);
        assert_eq!(store.try_insert("k1", Arc::new("v1".to_string())), Ok(None));
        let id = store.entry_id(&"k1").expect("missing entry id");
        assert_eq!(store.get_by_id(id), Some(Arc::new("v1".to_string())));
        assert_eq!(store.key_by_id(id), Some("k1"));
    }
}
