//! HashMap-backed store implementations.
//!
//! Provides three store variants optimized for different concurrency needs:
//! single-threaded with zero-overhead access, global-lock concurrent, and
//! sharded concurrent for high-contention workloads.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                        HashMap Store Variants                               │
//! │                                                                             │
//! │  ┌─────────────────────────────────────────────────────────────────────┐   │
//! │  │                     HashMapStore (single-threaded)                   │   │
//! │  │                                                                      │   │
//! │  │   ┌──────────────┐      get(&K) -> &V                               │   │
//! │  │   │ HashMap<K,V> │      (zero-copy, no Arc)                         │   │
//! │  │   └──────────────┘                                                   │   │
//! │  │         │                                                            │   │
//! │  │         └── Direct ownership, best for single-threaded hot paths    │   │
//! │  └─────────────────────────────────────────────────────────────────────┘   │
//! │                                                                             │
//! │  ┌─────────────────────────────────────────────────────────────────────┐   │
//! │  │                 ConcurrentHashMapStore (global lock)                 │   │
//! │  │                                                                      │   │
//! │  │   ┌─────────────────────────┐     get(&K) -> Arc<V>                 │   │
//! │  │   │ RwLock<HashMap<K,Arc<V>>│     (clone Arc, releases lock)        │   │
//! │  │   └─────────────────────────┘                                        │   │
//! │  │         │                                                            │   │
//! │  │         └── Simple concurrency, moderate contention OK              │   │
//! │  └─────────────────────────────────────────────────────────────────────┘   │
//! │                                                                             │
//! │  ┌─────────────────────────────────────────────────────────────────────┐   │
//! │  │                  ShardedHashMapStore (per-shard locks)               │   │
//! │  │                                                                      │   │
//! │  │   key ──► hash(key) % N ──► shard[i]                                │   │
//! │  │                                                                      │   │
//! │  │   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                       │   │
//! │  │   │Shard 0 │ │Shard 1 │ │Shard 2 │ │Shard N │                       │   │
//! │  │   │RwLock  │ │RwLock  │ │RwLock  │ │RwLock  │                       │   │
//! │  │   │HashMap │ │HashMap │ │HashMap │ │HashMap │                       │   │
//! │  │   └────────┘ └────────┘ └────────┘ └────────┘                       │   │
//! │  │         │                                                            │   │
//! │  │         └── High concurrency, independent shard access              │   │
//! │  └─────────────────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────────────┘
//!
//! Trait Implementations
//! ─────────────────────
//!
//!   HashMapStore<K, V, S>
//!     ├── StoreCore<K, V>      read ops, returns &V
//!     ├── StoreMut<K, V>       write ops, owned V
//!     └── StoreFactory<K, V>   factory (default hasher only)
//!
//!   ConcurrentHashMapStore<K, V, S>
//!     ├── ConcurrentStoreRead<K, V>     read ops, returns Arc<V>
//!     ├── ConcurrentStore<K, V>         write ops, Arc<V>
//!     └── ConcurrentStoreFactory<K, V>  factory (default hasher only)
//!
//!   ShardedHashMapStore<K, V, S>
//!     ├── ConcurrentStoreRead<K, V>     read ops, returns Arc<V>
//!     ├── ConcurrentStore<K, V>         write ops, Arc<V>
//!     └── ConcurrentStoreFactory<K, V>  factory (auto-detects shard count)
//!
//! Concurrency Comparison
//! ──────────────────────
//!
//!   │ Store                   │ Lock Scope │ Contention │ Use Case            │
//!   │─────────────────────────│────────────│────────────│─────────────────────│
//!   │ HashMapStore            │ None       │ N/A        │ Single-threaded     │
//!   │ ConcurrentHashMapStore  │ Global     │ Moderate   │ Simple concurrency  │
//!   │ ShardedHashMapStore     │ Per-shard  │ Low        │ High parallelism    │
//! ```
//!
//! ## Key Components
//!
//! - [`HashMapStore`]: Single-threaded store with zero-overhead `&V` access
//! - [`ConcurrentHashMapStore`]: Thread-safe store with global `RwLock`
//! - [`ShardedHashMapStore`]: Thread-safe store with per-shard locks
//!
//! ## Core Operations
//!
//! | Operation     | Single-threaded      | Concurrent           |
//! |---------------|----------------------|----------------------|
//! | `get`         | `&V` (zero-copy)     | `Arc<V>` (cloned)    |
//! | `try_insert`  | `V` owned            | `Arc<V>` owned       |
//! | `remove`      | `Option<V>`          | `Option<Arc<V>>`     |
//! | `peek`        | `&V` (no metrics)    | N/A                  |
//!
//! ## Performance Trade-offs
//!
//! **Single-threaded (`HashMapStore`):**
//! - Returns `&V` references—no allocation on access
//! - Fastest option when concurrency isn't needed
//! - Custom hasher support for specialized workloads
//!
//! **Global lock (`ConcurrentHashMapStore`):**
//! - Simple implementation, predictable behavior
//! - All operations serialize on the same lock
//! - Good for low-to-moderate concurrency
//!
//! **Sharded (`ShardedHashMapStore`):**
//! - Keys distributed across N independent shards
//! - Operations on different shards run in parallel
//! - Extra hashing cost to select shard
//! - Atomic size counter for global `len()`
//!
//! ## When to Use
//!
//! - [`HashMapStore`]: Single-threaded hot paths, no allocation on access
//! - [`ConcurrentHashMapStore`]: Simple thread-safe caching, moderate load
//! - [`ShardedHashMapStore`]: High-contention concurrent workloads
//!
//! ## Example Usage
//!
//! ```rust
//! use cachekit::store::hashmap::HashMapStore;
//! use cachekit::store::traits::{StoreCore, StoreMut};
//!
//! let mut store: HashMapStore<&str, i32> = HashMapStore::new(100);
//!
//! // Insert and access
//! store.try_insert("key", 42).unwrap();
//! assert_eq!(store.get(&"key"), Some(&42));  // Returns &V, zero-copy
//!
//! // Peek without metrics
//! assert_eq!(store.peek(&"key"), Some(&42));
//!
//! // Check metrics
//! let m = store.metrics();
//! assert_eq!(m.hits, 1);  // get() counted
//! assert_eq!(m.inserts, 1);
//! ```
//!
//! ## Type Constraints
//!
//! - `K: Eq + Hash` — keys must be hashable and comparable
//! - `S: BuildHasher` — custom hasher support (defaults to `RandomState`)
//! - Concurrent variants require `K: Send + Sync`, `V: Send + Sync`
//!
//! ## Thread Safety
//!
//! - [`HashMapStore`] is **not** thread-safe (single-threaded only)
//! - [`ConcurrentHashMapStore`] and [`ShardedHashMapStore`] are `Send + Sync`
//!
//! ## Implementation Notes
//!
//! - Sharded store uses `hash(key) % shard_count` for shard selection
//! - Metrics use `AtomicU64` with relaxed ordering (lock-free)
//! - Factory implementations use default hasher; use `with_hasher` for custom

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

/// Metrics counters using atomics for thread-safe updates.
///
/// All counters use `Ordering::Relaxed` for low-overhead increments.
/// Snapshot reads may reflect a slightly inconsistent view across counters
/// under high concurrency, but individual counters are always accurate.
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
    /// Explicit removals via `remove()`.
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
// Single-threaded HashMapStore
// =============================================================================

/// Single-threaded HashMap-backed store with zero-overhead value access.
///
/// Stores values directly (not wrapped in `Arc`) so `get()` returns `&V`
/// without allocation. Implements [`StoreCore`] and [`StoreMut`] traits
/// for integration with single-threaded cache policies.
///
/// # Type Parameters
///
/// - `K`: Key type, must be `Eq + Hash`
/// - `V`: Value type, stored directly (owned)
/// - `S`: Hasher type, defaults to `RandomState`
///
/// # Example
///
/// ```
/// use cachekit::store::hashmap::HashMapStore;
/// use cachekit::store::traits::{StoreCore, StoreMut};
///
/// let mut store: HashMapStore<String, Vec<u8>> = HashMapStore::new(1000);
///
/// // Insert data
/// store.try_insert("image.png".into(), vec![0x89, 0x50, 0x4E, 0x47]).unwrap();
///
/// // Access returns &V (zero-copy)
/// let data: &Vec<u8> = store.get(&"image.png".into()).unwrap();
/// assert_eq!(data[0], 0x89);
///
/// // Mutable access for in-place updates
/// if let Some(data) = store.peek_mut(&"image.png".into()) {
///     data.push(0x0D);
/// }
///
/// // Metrics tracking
/// let m = store.metrics();
/// assert_eq!(m.hits, 1);
/// assert_eq!(m.inserts, 1);
/// ```
///
/// # Custom Hasher
///
/// ```ignore
/// use std::hash::BuildHasherDefault;
/// use rustc_hash::FxHasher;  // Add rustc-hash to Cargo.toml
/// use cachekit::store::hashmap::HashMapStore;
///
/// type FxBuildHasher = BuildHasherDefault<FxHasher>;
///
/// let store: HashMapStore<u64, String, FxBuildHasher> =
///     HashMapStore::with_hasher(100, FxBuildHasher::default());
/// ```
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
    /// Creates a store with the specified capacity using the default hasher.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::store::hashmap::HashMapStore;
    /// use cachekit::store::traits::StoreCore;
    ///
    /// let store: HashMapStore<i32, String> = HashMapStore::new(100);
    /// assert_eq!(store.capacity(), 100);
    /// assert!(store.is_empty());
    /// ```
    pub fn new(capacity: usize) -> Self {
        Self::with_hasher(capacity, RandomState::new())
    }
}

impl<K, V, S> HashMapStore<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    /// Creates a store with the specified capacity and custom hasher.
    ///
    /// Use this when you need a deterministic or faster hasher than the
    /// default `RandomState`.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::hash_map::RandomState;
    /// use cachekit::store::hashmap::HashMapStore;
    /// use cachekit::store::traits::StoreCore;
    ///
    /// // Use explicit RandomState (or any BuildHasher impl)
    /// let store: HashMapStore<&str, i32, RandomState> =
    ///     HashMapStore::with_hasher(50, RandomState::new());
    /// assert_eq!(store.capacity(), 50);
    /// ```
    pub fn with_hasher(capacity: usize, hasher: S) -> Self {
        Self {
            map: HashMap::with_capacity_and_hasher(capacity, hasher),
            capacity,
            metrics: StoreCounters::default(),
        }
    }

    /// Returns a reference to the value without updating metrics.
    ///
    /// Useful for internal operations that shouldn't affect hit/miss counts.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::store::hashmap::HashMapStore;
    /// use cachekit::store::traits::{StoreCore, StoreMut};
    ///
    /// let mut store: HashMapStore<&str, i32> = HashMapStore::new(10);
    /// store.try_insert("key", 42).unwrap();
    ///
    /// // Peek doesn't update metrics
    /// assert_eq!(store.peek(&"key"), Some(&42));
    /// assert_eq!(store.metrics().hits, 0);
    /// ```
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.map.get(key)
    }

    /// Returns a mutable reference to the value without updating metrics.
    ///
    /// Allows in-place modification of stored values.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::store::hashmap::HashMapStore;
    /// use cachekit::store::traits::StoreMut;
    ///
    /// let mut store: HashMapStore<&str, Vec<i32>> = HashMapStore::new(10);
    /// store.try_insert("nums", vec![1, 2, 3]).unwrap();
    ///
    /// // Modify in place
    /// if let Some(nums) = store.peek_mut(&"nums") {
    ///     nums.push(4);
    /// }
    ///
    /// assert_eq!(store.peek(&"nums"), Some(&vec![1, 2, 3, 4]));
    /// ```
    pub fn peek_mut(&mut self, key: &K) -> Option<&mut V> {
        self.map.get_mut(key)
    }

    /// Returns the underlying HashMap's allocated capacity.
    ///
    /// This may be larger than the logical capacity limit set at creation.
    pub fn map_capacity(&self) -> usize {
        self.map.capacity()
    }

    /// Records an eviction in the metrics.
    ///
    /// Call when the policy evicts an entry. Separate from `remove()` to
    /// distinguish user-initiated removals from policy-driven evictions.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::store::hashmap::HashMapStore;
    /// use cachekit::store::traits::{StoreCore, StoreMut};
    ///
    /// let mut store: HashMapStore<&str, i32> = HashMapStore::new(10);
    /// store.try_insert("key", 1).unwrap();
    ///
    /// // Policy evicts the entry
    /// store.remove(&"key");
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

/// Read operations for [`HashMapStore`].
///
/// Returns borrowed `&V` references for zero-copy access.
impl<K, V, S> StoreCore<K, V> for HashMapStore<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    /// Returns a reference to the value for the given key.
    ///
    /// Updates hit/miss metrics. Use [`peek`](HashMapStore::peek) to avoid
    /// metric updates.
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

    /// Returns `true` if the key exists. Does not update metrics.
    fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Returns the current number of entries.
    fn len(&self) -> usize {
        self.map.len()
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

/// Write operations for [`HashMapStore`].
///
/// Takes and returns owned `V` values directly.
impl<K, V, S> StoreMut<K, V> for HashMapStore<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    /// Inserts or updates a value.
    ///
    /// Returns `Ok(Some(old))` for updates, `Ok(None)` for new insertions.
    ///
    /// # Errors
    ///
    /// Returns [`StoreFull`] if at capacity and the key is new.
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

    /// Removes and returns the value for the given key.
    ///
    /// Updates `removes` metric if an entry was removed.
    fn remove(&mut self, key: &K) -> Option<V> {
        let removed = self.map.remove(key);
        if removed.is_some() {
            self.metrics.inc_remove();
        }
        removed
    }

    /// Removes all entries. Does not update metrics.
    fn clear(&mut self) {
        self.map.clear();
    }
}

/// Factory for creating [`HashMapStore`] instances with default hasher.
///
/// # Example
///
/// ```
/// use cachekit::store::hashmap::HashMapStore;
/// use cachekit::store::traits::{StoreFactory, StoreCore};
///
/// fn create_store<F: StoreFactory<String, i32>>(cap: usize) -> F::Store {
///     F::create(cap)
/// }
///
/// let store = create_store::<HashMapStore<String, i32>>(100);
/// assert_eq!(store.capacity(), 100);
/// ```
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

/// Thread-safe HashMap store with a single global `RwLock`.
///
/// Uses `Arc<V>` for values since borrowed references cannot outlive lock
/// guards. Simple concurrency model—all operations serialize on the same
/// lock. For high-contention workloads, consider [`ShardedHashMapStore`].
///
/// # Type Parameters
///
/// - `K`: Key type, must be `Eq + Hash + Send + Sync`
/// - `V`: Value type, must be `Send + Sync` (wrapped in `Arc<V>`)
/// - `S`: Hasher type, must be `Send + Sync`, defaults to `RandomState`
///
/// # Synchronization
///
/// - Read operations (`get`, `contains`, `len`) acquire a read lock
/// - Write operations (`try_insert`, `remove`, `clear`) acquire a write lock
/// - Metrics use atomic counters (lock-free)
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use std::thread;
/// use cachekit::store::hashmap::ConcurrentHashMapStore;
/// use cachekit::store::traits::{ConcurrentStore, ConcurrentStoreRead};
///
/// let store = Arc::new(ConcurrentHashMapStore::<String, i32>::new(100));
///
/// // Spawn multiple writers
/// let handles: Vec<_> = (0..4).map(|t| {
///     let store = Arc::clone(&store);
///     thread::spawn(move || {
///         for i in 0..25 {
///             let key = format!("key_{}_{}", t, i);
///             store.try_insert(key, Arc::new(i)).unwrap();
///         }
///     })
/// }).collect();
///
/// for h in handles {
///     h.join().unwrap();
/// }
///
/// assert_eq!(store.len(), 100);
/// ```
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
    /// Creates a concurrent store with the specified capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::store::hashmap::ConcurrentHashMapStore;
    /// use cachekit::store::traits::ConcurrentStoreRead;
    ///
    /// let store: ConcurrentHashMapStore<String, Vec<u8>> =
    ///     ConcurrentHashMapStore::new(1000);
    /// assert_eq!(store.capacity(), 1000);
    /// ```
    pub fn new(capacity: usize) -> Self {
        Self::with_hasher(capacity, RandomState::new())
    }
}

impl<K, V, S> ConcurrentHashMapStore<K, V, S>
where
    K: Eq + Hash + Send,
    S: BuildHasher,
{
    /// Creates a concurrent store with the specified capacity and custom hasher.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::hash_map::RandomState;
    /// use cachekit::store::hashmap::ConcurrentHashMapStore;
    /// use cachekit::store::traits::ConcurrentStoreRead;
    ///
    /// // Use explicit RandomState (or any BuildHasher impl)
    /// let store: ConcurrentHashMapStore<u64, String, RandomState> =
    ///     ConcurrentHashMapStore::with_hasher(100, RandomState::new());
    /// assert_eq!(store.capacity(), 100);
    /// ```
    pub fn with_hasher(capacity: usize, hasher: S) -> Self {
        Self {
            map: RwLock::new(HashMap::with_capacity_and_hasher(capacity, hasher)),
            capacity,
            metrics: StoreCounters::default(),
        }
    }

    /// Records an eviction in the metrics.
    ///
    /// Thread-safe via atomic increment.
    pub fn record_eviction(&self) {
        self.metrics.inc_eviction();
    }
}

/// Read operations for [`ConcurrentHashMapStore`].
///
/// All methods acquire a read lock. Multiple readers can access concurrently.
impl<K, V, S> ConcurrentStoreRead<K, V> for ConcurrentHashMapStore<K, V, S>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
    S: BuildHasher + Send + Sync,
{
    /// Returns a clone of the value for the given key.
    ///
    /// Acquires read lock, clones `Arc<V>`, releases lock. The returned
    /// `Arc` can be held indefinitely without blocking other operations.
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

    /// Returns `true` if the key exists. Acquires read lock.
    fn contains(&self, key: &K) -> bool {
        self.map.read().contains_key(key)
    }

    /// Returns the current number of entries.
    ///
    /// Acquires read lock. Value may be stale by the time it's used.
    fn len(&self) -> usize {
        self.map.read().len()
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

/// Write operations for [`ConcurrentHashMapStore`].
///
/// All methods acquire a write lock. Writers have exclusive access.
impl<K, V, S> ConcurrentStore<K, V> for ConcurrentHashMapStore<K, V, S>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
    S: BuildHasher + Send + Sync,
{
    /// Inserts or updates a value.
    ///
    /// Acquires write lock for the duration of the operation.
    ///
    /// # Errors
    ///
    /// Returns [`StoreFull`] if at capacity and the key is new.
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

    /// Removes and returns the value for the given key.
    ///
    /// Acquires write lock.
    fn remove(&self, key: &K) -> Option<Arc<V>> {
        let removed = self.map.write().remove(key);
        if removed.is_some() {
            self.metrics.inc_remove();
        }
        removed
    }

    /// Removes all entries. Acquires write lock.
    fn clear(&self) {
        self.map.write().clear()
    }
}

/// Factory for creating [`ConcurrentHashMapStore`] instances.
///
/// # Example
///
/// ```
/// use cachekit::store::hashmap::ConcurrentHashMapStore;
/// use cachekit::store::traits::{ConcurrentStoreFactory, ConcurrentStoreRead};
///
/// fn create<F: ConcurrentStoreFactory<String, i32>>(cap: usize) -> F::Store {
///     F::create(cap)
/// }
///
/// let store = create::<ConcurrentHashMapStore<String, i32>>(50);
/// assert_eq!(store.capacity(), 50);
/// ```
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

/// Thread-safe HashMap store with per-shard locking for high concurrency.
///
/// Distributes keys across N independent shards, each with its own `RwLock`.
/// Operations on different shards can proceed in parallel, reducing contention
/// compared to [`ConcurrentHashMapStore`].
///
/// # Type Parameters
///
/// - `K`: Key type, must be `Eq + Hash + Send + Sync`
/// - `V`: Value type, must be `Send + Sync` (wrapped in `Arc<V>`)
/// - `S`: Hasher type, must be `Clone + Send + Sync`, defaults to `RandomState`
///
/// # Shard Selection
///
/// Keys are assigned to shards via `hash(key) % shard_count`. The same hasher
/// is used for both shard selection and within-shard HashMap operations.
///
/// # Capacity Enforcement
///
/// Global capacity is enforced via an `AtomicUsize` counter. The counter is
/// updated with compare-and-swap on insert to prevent over-capacity races.
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use std::thread;
/// use cachekit::store::hashmap::ShardedHashMapStore;
/// use cachekit::store::traits::{ConcurrentStore, ConcurrentStoreRead};
///
/// // Create store with 8 shards for 8-core machine
/// let store = Arc::new(ShardedHashMapStore::<u64, String>::new(10000, 8));
///
/// // Spawn workers that operate on different key ranges (likely different shards)
/// let handles: Vec<_> = (0..8).map(|t| {
///     let store = Arc::clone(&store);
///     thread::spawn(move || {
///         let base = t * 1000;
///         for i in 0..1000 {
///             store.try_insert(base + i, Arc::new(format!("v{}", base + i))).unwrap();
///         }
///     })
/// }).collect();
///
/// for h in handles {
///     h.join().unwrap();
/// }
///
/// assert_eq!(store.len(), 8000);
/// assert_eq!(store.shard_count(), 8);
/// ```
///
/// # Performance Considerations
///
/// - More shards = less contention but more memory overhead
/// - Optimal shard count is typically 2-4x the number of CPU cores
/// - Keys that hash to the same shard still contend
/// - Use [`ConcurrentStoreFactory`] to auto-detect shard count
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
    /// Creates a sharded store with the specified capacity and shard count.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of entries across all shards
    /// * `shards` - Number of independent shards (minimum 1)
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::store::hashmap::ShardedHashMapStore;
    /// use cachekit::store::traits::ConcurrentStoreRead;
    ///
    /// let store: ShardedHashMapStore<String, Vec<u8>> =
    ///     ShardedHashMapStore::new(10000, 16);
    /// assert_eq!(store.capacity(), 10000);
    /// assert_eq!(store.shard_count(), 16);
    /// ```
    pub fn new(capacity: usize, shards: usize) -> Self {
        Self::with_hasher(capacity, shards, RandomState::new())
    }
}

impl<K, V, S> ShardedHashMapStore<K, V, S>
where
    K: Eq + Hash + Send + Sync,
    S: BuildHasher + Clone,
{
    /// Creates a sharded store with custom hasher.
    ///
    /// The hasher is cloned for each shard's internal HashMap.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::hash_map::RandomState;
    /// use cachekit::store::hashmap::ShardedHashMapStore;
    ///
    /// // Use explicit RandomState (or any Clone + BuildHasher impl)
    /// let store: ShardedHashMapStore<u64, String, RandomState> =
    ///     ShardedHashMapStore::with_hasher(1000, 4, RandomState::new());
    /// assert_eq!(store.shard_count(), 4);
    /// ```
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

    /// Returns the number of shards.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::store::hashmap::ShardedHashMapStore;
    ///
    /// let store: ShardedHashMapStore<u64, i32> = ShardedHashMapStore::new(100, 8);
    /// assert_eq!(store.shard_count(), 8);
    /// ```
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Computes the shard index for a key.
    fn shard_index(&self, key: &K) -> usize {
        (self.hasher.hash_one(key) as usize) % self.shards.len()
    }

    /// Records an eviction in the metrics.
    ///
    /// Thread-safe via atomic increment.
    pub fn record_eviction(&self) {
        self.metrics.inc_eviction();
    }
}

/// Read operations for [`ShardedHashMapStore`].
///
/// Each operation only locks the shard containing the target key.
impl<K, V, S> ConcurrentStoreRead<K, V> for ShardedHashMapStore<K, V, S>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
    S: BuildHasher + Clone + Send + Sync,
{
    /// Returns a clone of the value for the given key.
    ///
    /// Only locks the shard containing the key. Other shards remain accessible.
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

    /// Returns `true` if the key exists. Only locks the target shard.
    fn contains(&self, key: &K) -> bool {
        let idx = self.shard_index(key);
        self.shards[idx].read().contains_key(key)
    }

    /// Returns the current number of entries across all shards.
    ///
    /// Uses atomic load—no locking required. Value may be stale under
    /// concurrent modifications.
    fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    /// Returns the global capacity limit.
    fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns a snapshot of the store's metrics.
    fn metrics(&self) -> StoreMetrics {
        self.metrics.snapshot()
    }
}

/// Write operations for [`ShardedHashMapStore`].
///
/// Each operation only locks the shard containing the target key, except
/// `clear()` which locks all shards.
impl<K, V, S> ConcurrentStore<K, V> for ShardedHashMapStore<K, V, S>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
    S: BuildHasher + Clone + Send + Sync,
{
    /// Inserts or updates a value.
    ///
    /// Only locks the target shard. Uses compare-and-swap on the global
    /// size counter to enforce capacity without locking all shards.
    ///
    /// # Errors
    ///
    /// Returns [`StoreFull`] if at capacity and the key is new.
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

                // CAS loop to atomically reserve a slot
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

    /// Removes and returns the value for the given key.
    ///
    /// Only locks the target shard. Atomically decrements size counter.
    fn remove(&self, key: &K) -> Option<Arc<V>> {
        let idx = self.shard_index(key);
        let removed = self.shards[idx].write().remove(key);
        if removed.is_some() {
            self.size.fetch_sub(1, Ordering::Relaxed);
            self.metrics.inc_remove();
        }
        removed
    }

    /// Removes all entries from all shards.
    ///
    /// Acquires write locks on all shards simultaneously to ensure
    /// consistency. This is the only operation that locks multiple shards.
    fn clear(&self) {
        // Lock all shards first to prevent concurrent modifications
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

/// Factory for creating [`ShardedHashMapStore`] instances.
///
/// Automatically detects the number of available CPU cores and creates
/// that many shards for optimal parallelism.
///
/// # Example
///
/// ```
/// use cachekit::store::hashmap::ShardedHashMapStore;
/// use cachekit::store::traits::{ConcurrentStoreFactory, ConcurrentStoreRead};
///
/// // Factory auto-detects optimal shard count
/// let store = <ShardedHashMapStore<String, i32>>::create(10000);
/// assert_eq!(store.capacity(), 10000);
/// // shard_count() == number of CPU cores (or 1 if detection fails)
/// ```
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
