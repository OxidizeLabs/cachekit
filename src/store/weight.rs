//! Weight-aware store with dual entry-count and byte-based capacity limits.
//!
//! Enforces both a maximum number of entries and a maximum total weight
//! (e.g., bytes). Ideal for caches where values vary significantly in size
//! and you want fair eviction pressure based on actual memory usage.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                         Weight Store Layout                                 │
//! │                                                                             │
//! │   ┌──────────────────────────────────────────────────────────────────────┐ │
//! │   │  Dual Capacity Limits                                                │ │
//! │   │                                                                      │ │
//! │   │   Entry Limit: max 100 entries    Weight Limit: max 10MB total      │ │
//! │   │        │                                │                            │ │
//! │   │        ▼                                ▼                            │ │
//! │   │   ┌─────────┐                    ┌─────────────┐                     │ │
//! │   │   │ len()   │                    │total_weight │                     │ │
//! │   │   │  = 3    │                    │  = 2.5MB    │                     │ │
//! │   │   └─────────┘                    └─────────────┘                     │ │
//! │   └──────────────────────────────────────────────────────────────────────┘ │
//! │                                                                             │
//! │   ┌──────────────────────────────────────────────────────────────────────┐ │
//! │   │  Storage: HashMap<K, WeightEntry<V>>                                 │ │
//! │   │                                                                      │ │
//! │   │   key        WeightEntry                                             │ │
//! │   │   ────       ───────────────────────────                             │ │
//! │   │   "img1" ──► { value: Arc<[u8; 1MB]>, weight: 1_000_000 }           │ │
//! │   │   "img2" ──► { value: Arc<[u8; 1MB]>, weight: 1_000_000 }           │ │
//! │   │   "icon" ──► { value: Arc<[u8; 500KB]>, weight: 500_000 }           │ │
//! │   │                                                                      │ │
//! │   │   Weight is precomputed on insert to avoid repeated computation.     │ │
//! │   └──────────────────────────────────────────────────────────────────────┘ │
//! │                                                                             │
//! │   ┌──────────────────────────────────────────────────────────────────────┐ │
//! │   │  Weight Function: F: Fn(&V) -> usize                                 │ │
//! │   │                                                                      │ │
//! │   │   Examples:                                                          │ │
//! │   │   • |s: &String| s.len()           // String byte length            │ │
//! │   │   • |v: &Vec<u8>| v.len()          // Vec capacity                  │ │
//! │   │   • |img: &Image| img.width * img.height * 4  // RGBA pixels        │ │
//! │   │   • |_: &T| 1                      // Treat as entry-count only     │ │
//! │   └──────────────────────────────────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────────────────────┘
//!
//! Capacity Enforcement
//! ────────────────────
//!
//!   try_insert(key, value):
//!     │
//!     ├─► Existing key (UPDATE)
//!     │     │
//!     │     ├── new_weight = weight_fn(&value)
//!     │     ├── next_total = total_weight - old_weight + new_weight
//!     │     │
//!     │     └── next_total > capacity_weight? ──► Err(StoreFull)
//!     │                                     └──► Ok(Some(old_value))
//!     │
//!     └─► New key (INSERT)
//!           │
//!           ├── len() >= capacity_entries? ──► Err(StoreFull)
//!           │
//!           ├── new_weight = weight_fn(&value)
//!           ├── total_weight + new_weight > capacity_weight? ──► Err(StoreFull)
//!           │
//!           └── Ok(None)
//!
//! Component Comparison
//! ────────────────────
//!
//!   │ Store                  │ Entry Limit │ Weight Limit │ Thread Safety │
//!   │────────────────────────│─────────────│──────────────│───────────────│
//!   │ WeightStore            │ ✓           │ ✓            │ Single-thread │
//!   │ ConcurrentWeightStore  │ ✓           │ ✓            │ Send + Sync   │
//! ```
//!
//! ## Key Components
//!
//! - [`WeightStore`]: Single-threaded store with entry + weight limits
//! - [`ConcurrentWeightStore`]: Thread-safe wrapper using `RwLock`
//! - `WeightEntry`: Internal struct storing value + precomputed weight
//!
//! ## Core Operations
//!
//! | Operation       | Description                             | Complexity |
//! |-----------------|-----------------------------------------|------------|
//! | `try_insert`    | Insert/update with dual limit checks    | O(1) avg   |
//! | `get`           | Lookup by key (updates metrics)         | O(1) avg   |
//! | `peek`          | Lookup without metrics                  | O(1) avg   |
//! | `remove`        | Remove and adjust total weight          | O(1) avg   |
//! | `total_weight`  | Current sum of all entry weights        | O(1)       |
//!
//! ## Performance Trade-offs
//!
//! **Advantages:**
//! - Fair eviction pressure based on actual size, not entry count
//! - Weight precomputed on insert—reads don't recompute
//! - Dual limits prevent both entry-count and memory exhaustion
//! - Weight function is caller-defined for maximum flexibility
//!
//! **Costs:**
//! - Weight function called on every insert/update
//! - Uses `Arc<V>` even for single-threaded store (needed for weight_fn)
//! - Slightly more state to track than simple HashMap stores
//!
//! ## When to Use
//!
//! - Values vary significantly in size (images, documents, serialized data)
//! - Memory budget matters more than entry count
//! - You need observability into total cached bytes
//! - Eviction policy should consider size, not just access patterns
//!
//! ## Example Usage
//!
//! ```rust
//! use std::sync::Arc;
//! use cachekit::store::weight::WeightStore;
//!
//! // Cache with max 100 entries OR 1MB total, whichever is hit first
//! let mut store = WeightStore::with_capacity(100, 1_000_000, |v: &Vec<u8>| v.len());
//!
//! // Insert entries
//! store.try_insert("small", Arc::new(vec![0u8; 100])).unwrap();
//! store.try_insert("large", Arc::new(vec![0u8; 10_000])).unwrap();
//!
//! assert_eq!(store.len(), 2);
//! assert_eq!(store.total_weight(), 10_100);
//!
//! // Remove adjusts weight
//! store.remove(&"large");
//! assert_eq!(store.total_weight(), 100);
//! ```
//!
//! ## Type Constraints
//!
//! - `K: Eq + Hash` — keys must be hashable
//! - `F: Fn(&V) -> usize` — weight function computes size from value
//! - Concurrent: `K: Send + Sync`, `V: Send + Sync`, `F: Send + Sync`
//!
//! ## Thread Safety
//!
//! - [`WeightStore`] is **not** thread-safe (single-threaded only)
//! - [`ConcurrentWeightStore`] is `Send + Sync` via `parking_lot::RwLock`
//!
//! ## Implementation Notes
//!
//! - Weight is stored per entry in `WeightEntry` to avoid recomputation
//! - Updates recompute weight and adjust `total_weight` atomically
//! - Does **not** implement `StoreCore`/`StoreMut` (uses `Arc<V>` API)
//! - Metrics use atomic counters for concurrent compatibility

use std::hash::Hash;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "concurrency")]
use parking_lot::RwLock;
use rustc_hash::FxHashMap;

#[cfg(feature = "concurrency")]
use crate::store::traits::{ConcurrentStore, ConcurrentStoreRead};
use crate::store::traits::{StoreFull, StoreMetrics};

/// Internal entry storing value and its precomputed weight.
///
/// Weight is computed once on insert/update to avoid repeated weight
/// function calls during reads or accounting operations.
#[derive(Debug)]
struct WeightEntry<V> {
    /// The cached value, wrapped in Arc for shared access.
    value: Arc<V>,
    /// Precomputed weight from the weight function.
    weight: usize,
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
// Single-threaded WeightStore
// =============================================================================

/// Single-threaded store with dual entry-count and weight-based capacity limits.
///
/// Enforces both a maximum number of entries and a maximum total weight.
/// Weight is computed via a caller-provided function and cached per entry.
/// Uses `Arc<V>` for values to enable weight function access.
///
/// # Type Parameters
///
/// - `K`: Key type, must be `Eq + Hash`
/// - `V`: Value type, wrapped in `Arc<V>`
/// - `F`: Weight function, `Fn(&V) -> usize`
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use cachekit::store::weight::WeightStore;
///
/// // Image cache: max 50 images OR 100MB, whichever is hit first
/// let mut store = WeightStore::with_capacity(
///     50,           // max entries
///     100_000_000,  // max weight (100MB)
///     |img: &Vec<u8>| img.len(),  // weight = byte length
/// );
///
/// // Insert images
/// let small_img = Arc::new(vec![0u8; 1_000]);      // 1KB
/// let large_img = Arc::new(vec![0u8; 10_000_000]); // 10MB
///
/// store.try_insert("thumbnail", small_img).unwrap();
/// store.try_insert("fullsize", large_img).unwrap();
///
/// assert_eq!(store.len(), 2);
/// assert_eq!(store.total_weight(), 10_001_000);
/// assert_eq!(store.capacity_weight(), 100_000_000);
///
/// // Check metrics
/// let _ = store.get(&"thumbnail");
/// let _ = store.get(&"missing");
/// assert_eq!(store.metrics().hits, 1);
/// assert_eq!(store.metrics().misses, 1);
/// ```
///
/// # Weight Function Examples
///
/// ```
/// use std::sync::Arc;
/// use cachekit::store::weight::WeightStore;
///
/// // String length
/// let mut s1: WeightStore<&str, String, _> =
///     WeightStore::with_capacity(100, 10_000, |s: &String| s.len());
///
/// // Fixed weight (degrades to entry-count only)
/// let mut s2: WeightStore<&str, i32, _> =
///     WeightStore::with_capacity(100, 100, |_: &i32| 1);
///
/// // Struct with custom size calculation
/// struct Document { content: String, metadata: Vec<u8> }
/// let mut s3: WeightStore<&str, Document, _> = WeightStore::with_capacity(
///     100, 1_000_000,
///     |doc: &Document| doc.content.len() + doc.metadata.len()
/// );
/// ```
#[derive(Debug)]
pub struct WeightStore<K, V, F>
where
    F: Fn(&V) -> usize,
{
    map: FxHashMap<K, WeightEntry<V>>,
    capacity_entries: usize,
    capacity_weight: usize,
    total_weight: usize,
    weight_fn: F,
    metrics: StoreCounters,
}

impl<K, V, F> WeightStore<K, V, F>
where
    K: Eq + Hash,
    F: Fn(&V) -> usize,
{
    /// Creates a store with entry and weight limits plus a weight function.
    ///
    /// # Arguments
    ///
    /// * `capacity_entries` - Maximum number of entries
    /// * `capacity_weight` - Maximum total weight across all entries
    /// * `weight_fn` - Function to compute weight from a value
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::store::weight::WeightStore;
    ///
    /// // Max 1000 entries or 50MB total
    /// let store: WeightStore<String, Vec<u8>, _> = WeightStore::with_capacity(
    ///     1000,
    ///     50_000_000,
    ///     |data: &Vec<u8>| data.len(),
    /// );
    ///
    /// assert_eq!(store.capacity(), 1000);
    /// assert_eq!(store.capacity_weight(), 50_000_000);
    /// assert_eq!(store.total_weight(), 0);
    /// ```
    pub fn with_capacity(capacity_entries: usize, capacity_weight: usize, weight_fn: F) -> Self {
        Self {
            map: FxHashMap::with_capacity_and_hasher(capacity_entries, Default::default()),
            capacity_entries,
            capacity_weight,
            total_weight: 0,
            weight_fn,
            metrics: StoreCounters::default(),
        }
    }

    /// Returns the current total weight of all entries.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cachekit::store::weight::WeightStore;
    ///
    /// let mut store = WeightStore::with_capacity(10, 1000, |s: &String| s.len());
    ///
    /// store.try_insert("a", Arc::new("hello".into())).unwrap();  // weight 5
    /// store.try_insert("b", Arc::new("world!".into())).unwrap(); // weight 6
    ///
    /// assert_eq!(store.total_weight(), 11);
    /// ```
    pub fn total_weight(&self) -> usize {
        self.total_weight
    }

    /// Returns the configured maximum weight capacity.
    pub fn capacity_weight(&self) -> usize {
        self.capacity_weight
    }

    /// Computes the weight for a value using the configured weight function.
    fn compute_weight(&self, value: &V) -> usize {
        (self.weight_fn)(value)
    }

    /// Returns a clone of the value for the given key.
    ///
    /// Updates hit/miss metrics. Use [`peek`](Self::peek) to avoid metric updates.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cachekit::store::weight::WeightStore;
    ///
    /// let mut store = WeightStore::with_capacity(10, 1000, |s: &String| s.len());
    /// store.try_insert("key", Arc::new("value".into())).unwrap();
    ///
    /// assert_eq!(store.get(&"key"), Some(Arc::new("value".into())));
    /// assert_eq!(store.get(&"missing"), None);
    ///
    /// assert_eq!(store.metrics().hits, 1);
    /// assert_eq!(store.metrics().misses, 1);
    /// ```
    pub fn get(&self, key: &K) -> Option<Arc<V>> {
        match self.map.get(key).map(|entry| Arc::clone(&entry.value)) {
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
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cachekit::store::weight::WeightStore;
    ///
    /// let mut store = WeightStore::with_capacity(10, 1000, |s: &String| s.len());
    /// store.try_insert("key", Arc::new("value".into())).unwrap();
    ///
    /// // Peek doesn't affect metrics
    /// assert!(store.peek(&"key").is_some());
    /// assert_eq!(store.metrics().hits, 0);
    /// ```
    pub fn peek(&self, key: &K) -> Option<&Arc<V>> {
        self.map.get(key).map(|entry| &entry.value)
    }

    /// Returns `true` if the key exists in the store.
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Returns the current number of entries.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the store contains no entries.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns the maximum entry capacity.
    pub fn capacity(&self) -> usize {
        self.capacity_entries
    }

    /// Returns a snapshot of the store's metrics.
    pub fn metrics(&self) -> StoreMetrics {
        self.metrics.snapshot()
    }

    /// Records an eviction in the metrics.
    ///
    /// Call when the policy evicts an entry. Separate from `remove()` to
    /// distinguish user-initiated removals from policy-driven evictions.
    pub fn record_eviction(&self) {
        self.metrics.inc_eviction();
    }

    /// Inserts or updates a value while enforcing dual capacity limits.
    ///
    /// For updates, the weight is recomputed and the total adjusted. The
    /// update fails if the new total would exceed `capacity_weight`.
    ///
    /// For new insertions, both entry count and weight limits are checked.
    ///
    /// # Errors
    ///
    /// Returns [`StoreFull`] if:
    /// - Entry count would exceed `capacity_entries` (new key only)
    /// - Total weight would exceed `capacity_weight`
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cachekit::store::weight::WeightStore;
    /// use cachekit::store::traits::StoreFull;
    ///
    /// let mut store = WeightStore::with_capacity(10, 100, |s: &String| s.len());
    ///
    /// // Insert succeeds
    /// assert!(store.try_insert("a", Arc::new("hello".into())).is_ok());
    ///
    /// // Update returns old value
    /// let old = store.try_insert("a", Arc::new("hi".into())).unwrap();
    /// assert_eq!(old, Some(Arc::new("hello".into())));
    ///
    /// // Weight limit exceeded
    /// let huge = Arc::new("x".repeat(200));
    /// assert_eq!(store.try_insert("b", huge), Err(StoreFull));
    /// ```
    pub fn try_insert(&mut self, key: K, value: Arc<V>) -> Result<Option<Arc<V>>, StoreFull> {
        let new_weight = self.compute_weight(value.as_ref());
        if let Some(entry) = self.map.get_mut(&key) {
            let next_total = self.total_weight - entry.weight + new_weight;
            if next_total > self.capacity_weight {
                return Err(StoreFull);
            }
            let previous = Arc::clone(&entry.value);
            entry.value = value;
            entry.weight = new_weight;
            self.total_weight = next_total;
            self.metrics.inc_update();
            return Ok(Some(previous));
        }

        if self.map.len() >= self.capacity_entries {
            return Err(StoreFull);
        }
        if self.total_weight + new_weight > self.capacity_weight {
            return Err(StoreFull);
        }

        self.map.insert(
            key,
            WeightEntry {
                value,
                weight: new_weight,
            },
        );
        self.total_weight += new_weight;
        self.metrics.inc_insert();
        Ok(None)
    }

    /// Removes and returns the value for the given key.
    ///
    /// Adjusts `total_weight` by subtracting the entry's weight.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cachekit::store::weight::WeightStore;
    ///
    /// let mut store = WeightStore::with_capacity(10, 1000, |s: &String| s.len());
    /// store.try_insert("key", Arc::new("value".into())).unwrap();
    ///
    /// assert_eq!(store.total_weight(), 5);
    /// store.remove(&"key");
    /// assert_eq!(store.total_weight(), 0);
    /// ```
    pub fn remove(&mut self, key: &K) -> Option<Arc<V>> {
        let entry = self.map.remove(key)?;
        self.total_weight = self.total_weight.saturating_sub(entry.weight);
        self.metrics.inc_remove();
        Some(entry.value)
    }

    /// Removes all entries and resets total weight to zero.
    pub fn clear(&mut self) {
        self.map.clear();
        self.total_weight = 0;
    }
}

// =============================================================================
// Concurrent WeightStore
// =============================================================================

/// Thread-safe store with dual entry-count and weight-based capacity limits.
///
/// Wraps [`WeightStore`] with a `parking_lot::RwLock` for thread-safe access.
/// Implements [`ConcurrentStore`] and [`ConcurrentStoreRead`] traits.
///
/// # Type Parameters
///
/// - `K`: Key type, must be `Eq + Hash + Send + Sync`
/// - `V`: Value type, must be `Send + Sync`
/// - `F`: Weight function, must be `Fn(&V) -> usize + Send + Sync`
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use std::thread;
/// use cachekit::store::weight::ConcurrentWeightStore;
/// use cachekit::store::traits::{ConcurrentStore, ConcurrentStoreRead};
///
/// let store = Arc::new(ConcurrentWeightStore::with_capacity(
///     100,        // max entries
///     1_000_000,  // max weight (1MB)
///     |data: &Vec<u8>| data.len(),
/// ));
///
/// // Spawn writers
/// let handles: Vec<_> = (0..4).map(|t| {
///     let store = Arc::clone(&store);
///     thread::spawn(move || {
///         for i in 0..25 {
///             let key = format!("key_{}_{}", t, i);
///             let data = Arc::new(vec![0u8; 100]);  // 100 bytes each
///             let _ = store.try_insert(key, data);
///         }
///     })
/// }).collect();
///
/// for h in handles {
///     h.join().unwrap();
/// }
///
/// assert_eq!(store.len(), 100);
/// assert_eq!(store.total_weight(), 10_000);  // 100 entries × 100 bytes
/// ```
#[derive(Debug)]
#[cfg(feature = "concurrency")]
pub struct ConcurrentWeightStore<K, V, F>
where
    F: Fn(&V) -> usize,
{
    inner: RwLock<WeightStore<K, V, F>>,
    metrics: StoreCounters,
}

#[cfg(feature = "concurrency")]
impl<K, V, F> ConcurrentWeightStore<K, V, F>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
    F: Fn(&V) -> usize,
{
    /// Creates a concurrent store with entry and weight limits.
    ///
    /// # Arguments
    ///
    /// * `capacity_entries` - Maximum number of entries
    /// * `capacity_weight` - Maximum total weight across all entries
    /// * `weight_fn` - Function to compute weight from a value
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::store::weight::ConcurrentWeightStore;
    /// use cachekit::store::traits::ConcurrentStoreRead;
    ///
    /// let store: ConcurrentWeightStore<String, String, _> = ConcurrentWeightStore::with_capacity(
    ///     1000,
    ///     50_000_000,
    ///     |s: &String| s.len(),
    /// );
    ///
    /// assert_eq!(store.capacity(), 1000);
    /// assert_eq!(store.capacity_weight(), 50_000_000);
    /// ```
    pub fn with_capacity(capacity_entries: usize, capacity_weight: usize, weight_fn: F) -> Self {
        Self {
            inner: RwLock::new(WeightStore::with_capacity(
                capacity_entries,
                capacity_weight,
                weight_fn,
            )),
            metrics: StoreCounters::default(),
        }
    }

    /// Returns the current total weight of all entries.
    ///
    /// Acquires read lock. Value may be stale under concurrent modifications.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cachekit::store::weight::ConcurrentWeightStore;
    /// use cachekit::store::traits::ConcurrentStore;
    ///
    /// let store = ConcurrentWeightStore::with_capacity(10, 1000, |s: &String| s.len());
    /// store.try_insert("key", Arc::new("hello".into())).unwrap();
    ///
    /// assert_eq!(store.total_weight(), 5);
    /// ```
    pub fn total_weight(&self) -> usize {
        self.inner.read().total_weight()
    }

    /// Returns the configured maximum weight capacity.
    ///
    /// Acquires read lock.
    pub fn capacity_weight(&self) -> usize {
        self.inner.read().capacity_weight()
    }

    /// Records an eviction in the metrics.
    ///
    /// Thread-safe via atomic increment.
    pub fn record_eviction(&self) {
        self.metrics.inc_eviction();
    }
}

/// Read operations for [`ConcurrentWeightStore`].
///
/// All methods acquire a read lock on the inner store.
#[cfg(feature = "concurrency")]
impl<K, V, F> ConcurrentStoreRead<K, V> for ConcurrentWeightStore<K, V, F>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
    F: Fn(&V) -> usize + Send + Sync,
{
    /// Returns a clone of the value for the given key.
    ///
    /// Acquires read lock. Updates hit/miss metrics atomically.
    fn get(&self, key: &K) -> Option<Arc<V>> {
        let store = self.inner.read();
        match store.map.get(key).map(|entry| Arc::clone(&entry.value)) {
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
        self.inner.read().contains(key)
    }

    /// Returns the current number of entries.
    ///
    /// Acquires read lock. Value may be stale under concurrency.
    fn len(&self) -> usize {
        self.inner.read().len()
    }

    /// Returns the maximum entry capacity.
    fn capacity(&self) -> usize {
        self.inner.read().capacity()
    }

    /// Returns a snapshot of the store's metrics.
    fn metrics(&self) -> StoreMetrics {
        self.metrics.snapshot()
    }
}

/// Write operations for [`ConcurrentWeightStore`].
///
/// All methods acquire a write lock on the inner store.
#[cfg(feature = "concurrency")]
impl<K, V, F> ConcurrentStore<K, V> for ConcurrentWeightStore<K, V, F>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
    F: Fn(&V) -> usize + Send + Sync,
{
    /// Inserts or updates a value while enforcing dual capacity limits.
    ///
    /// Acquires write lock. Weight is computed during the operation.
    ///
    /// # Errors
    ///
    /// Returns [`StoreFull`] if entry count or weight limit would be exceeded.
    fn try_insert(&self, key: K, value: Arc<V>) -> Result<Option<Arc<V>>, StoreFull> {
        let mut store = self.inner.write();
        store.try_insert(key, value)
    }

    /// Removes and returns the value for the given key.
    ///
    /// Acquires write lock. Adjusts total weight.
    fn remove(&self, key: &K) -> Option<Arc<V>> {
        let mut store = self.inner.write();
        store.remove(key)
    }

    /// Removes all entries and resets total weight.
    ///
    /// Acquires write lock.
    fn clear(&self) {
        let mut store = self.inner.write();
        store.clear();
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::ptr_arg)]
    fn weight_by_len(value: &String) -> usize {
        value.len()
    }

    #[test]
    fn weight_store_tracks_weight() {
        let mut store = WeightStore::with_capacity(3, 10, weight_by_len);
        assert_eq!(store.total_weight(), 0);
        assert_eq!(store.try_insert("k1", Arc::new("aa".to_string())), Ok(None));
        assert_eq!(store.total_weight(), 2);
        assert_eq!(
            store.try_insert("k2", Arc::new("bbbb".to_string())),
            Ok(None)
        );
        assert_eq!(store.total_weight(), 6);
        assert_eq!(store.remove(&"k1"), Some(Arc::new("aa".to_string())));
        assert_eq!(store.total_weight(), 4);
    }

    #[test]
    fn weight_store_enforces_capacity() {
        let mut store = WeightStore::with_capacity(10, 5, weight_by_len);
        assert_eq!(
            store.try_insert("k1", Arc::new("aaaaa".to_string())),
            Ok(None)
        );
        assert_eq!(
            store.try_insert("k2", Arc::new("bb".to_string())),
            Err(StoreFull)
        );
    }

    #[test]
    fn weight_store_update_adjusts_weight() {
        let mut store = WeightStore::with_capacity(10, 10, weight_by_len);
        assert_eq!(store.try_insert("k1", Arc::new("aa".to_string())), Ok(None));
        assert_eq!(store.total_weight(), 2);
        assert_eq!(
            store.try_insert("k1", Arc::new("aaaa".to_string())),
            Ok(Some(Arc::new("aa".to_string())))
        );
        assert_eq!(store.total_weight(), 4);
    }

    #[cfg(feature = "concurrency")]
    #[test]
    fn concurrent_weight_store_basic_ops() {
        let store = ConcurrentWeightStore::with_capacity(2, 10, weight_by_len);
        let value = Arc::new("aa".to_string());
        assert_eq!(store.try_insert("k1", value.clone()), Ok(None));
        assert_eq!(store.get(&"k1"), Some(value.clone()));
        assert_eq!(store.total_weight(), 2);
        assert_eq!(store.remove(&"k1"), Some(value));
        assert_eq!(store.total_weight(), 0);
    }
}
