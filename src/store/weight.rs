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
//! │   ┌──────────────────────────────────────────────────────────────────────┐  │
//! │   │  Dual Capacity Limits                                                │  │
//! │   │                                                                      │  │
//! │   │   Entry Limit: max 100 entries    Weight Limit: max 10MB total       │  │
//! │   │        │                                │                            │  │
//! │   │        ▼                                ▼                            │  │
//! │   │   ┌─────────┐                    ┌─────────────┐                     │  │
//! │   │   │ len()   │                    │total_weight │                     │  │
//! │   │   │  = 3    │                    │  = 2.5MB    │                     │  │
//! │   │   └─────────┘                    └─────────────┘                     │  │
//! │   └──────────────────────────────────────────────────────────────────────┘  │
//! │                                                                             │
//! │   ┌──────────────────────────────────────────────────────────────────────┐  │
//! │   │  Storage: HashMap<K, WeightEntry<V>>                                 │  │
//! │   │                                                                      │  │
//! │   │   key        WeightEntry                                             │  │
//! │   │   ────       ───────────────────────────                             │  │
//! │   │   "img1" ──► { value: Arc<[u8; 1MB]>, weight: 1_000_000 }            │  │
//! │   │   "img2" ──► { value: Arc<[u8; 1MB]>, weight: 1_000_000 }            │  │
//! │   │   "icon" ──► { value: Arc<[u8; 500KB]>, weight: 500_000 }            │  │
//! │   │                                                                      │  │
//! │   │   Weight is precomputed on insert to avoid repeated computation.     │  │
//! │   └──────────────────────────────────────────────────────────────────────┘  │
//! │                                                                             │
//! │   ┌──────────────────────────────────────────────────────────────────────┐  │
//! │   │  Weight Function: F: Fn(&V) -> usize                                 │  │
//! │   │                                                                      │  │
//! │   │   Examples:                                                          │  │
//! │   │   • |s: &String| s.len()           // String byte length             │  │
//! │   │   • |v: &Vec<u8>| v.len()          // Vec capacity                   │  │
//! │   │   • |img: &Image| img.width * img.height * 4  // RGBA pixels         │  │
//! │   │   • |_: &T| 1                      // Treat as entry-count only      │  │
//! │   └──────────────────────────────────────────────────────────────────────┘  │
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
            assert!(
                self.total_weight >= entry.weight,
                "WeightStore invariant violated: total_weight ({}) is less than entry.weight ({})",
                self.total_weight,
                entry.weight
            );
            let base_total = self
                .total_weight
                .checked_sub(entry.weight)
                .expect("WeightStore invariant violated: checked_sub failed after invariant check");
            let next_total = base_total.checked_add(new_weight).ok_or(StoreFull)?;
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
        let next_total = self.total_weight.checked_add(new_weight).ok_or(StoreFull)?;
        if next_total > self.capacity_weight {
            return Err(StoreFull);
        }

        self.map.insert(
            key,
            WeightEntry {
                value,
                weight: new_weight,
            },
        );
        self.total_weight = next_total;
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
        assert!(
            self.total_weight >= entry.weight,
            "total_weight underflow in remove"
        );
        self.total_weight = self
            .total_weight
            .checked_sub(entry.weight)
            .expect("WeightStore invariant violated: checked_sub failed after invariant check");
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
        let result = store.try_insert(key, value);
        match &result {
            Ok(Some(_)) => self.metrics.inc_update(),
            Ok(None) => self.metrics.inc_insert(),
            Err(_) => {},
        }
        result
    }

    /// Removes and returns the value for the given key.
    ///
    /// Acquires write lock. Adjusts total weight.
    fn remove(&self, key: &K) -> Option<Arc<V>> {
        let mut store = self.inner.write();
        let result = store.remove(key);
        if result.is_some() {
            self.metrics.inc_remove();
        }
        result
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
    use std::collections::HashMap;
    #[cfg(feature = "concurrency")]
    use std::thread;

    use proptest::prelude::*;

    #[allow(clippy::ptr_arg)]
    fn weight_by_len(value: &String) -> usize {
        value.len()
    }

    fn sum_entry_weights<K, V, F>(store: &WeightStore<K, V, F>) -> usize
    where
        K: Eq + Hash,
        F: Fn(&V) -> usize,
    {
        let mut total = 0usize;
        for entry in store.map.values() {
            total += entry.weight;
        }
        total
    }

    #[derive(Clone, Debug)]
    enum Op {
        Insert(u8, Vec<u8>),
        Remove(u8),
        Clear,
    }

    fn op_strategy() -> impl Strategy<Value = Op> {
        prop_oneof![
            (any::<u8>(), proptest::collection::vec(any::<u8>(), 0..64))
                .prop_map(|(k, v)| Op::Insert(k, v)),
            any::<u8>().prop_map(Op::Remove),
            Just(Op::Clear),
        ]
    }

    #[derive(Clone, Debug)]
    enum MetricsOp {
        Insert(u8, Vec<u8>),
        Remove(u8),
        Get(u8),
        Peek(u8),
        Clear,
        RecordEviction,
    }

    fn metrics_op_strategy() -> impl Strategy<Value = MetricsOp> {
        prop_oneof![
            (any::<u8>(), proptest::collection::vec(any::<u8>(), 0..32))
                .prop_map(|(k, v)| MetricsOp::Insert(k, v)),
            any::<u8>().prop_map(MetricsOp::Remove),
            any::<u8>().prop_map(MetricsOp::Get),
            any::<u8>().prop_map(MetricsOp::Peek),
            Just(MetricsOp::Clear),
            Just(MetricsOp::RecordEviction),
        ]
    }

    fn near_max_weight(seed: u16) -> usize {
        match seed % 4 {
            0 => (seed as usize) % 128,
            _ => usize::MAX - (seed as usize),
        }
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

    #[test]
    fn weight_store_update_rejected_when_new_weight_exceeds_capacity() {
        let mut store = WeightStore::with_capacity(4, 5, weight_by_len);
        assert_eq!(store.try_insert("k1", Arc::new("aa".to_string())), Ok(None));
        assert_eq!(store.try_insert("k2", Arc::new("bb".to_string())), Ok(None));
        assert_eq!(store.total_weight(), 4);

        // Updating existing key from 2 -> 4 would make total 6, exceeding capacity 5.
        assert_eq!(
            store.try_insert("k1", Arc::new("zzzz".to_string())),
            Err(StoreFull)
        );
        assert_eq!(store.total_weight(), 4);
        assert_eq!(store.get(&"k1"), Some(Arc::new("aa".to_string())));
    }

    #[test]
    fn weight_store_zero_weight_with_zero_weight_capacity() {
        let mut store = WeightStore::with_capacity(2, 0, |_v: &String| 0);
        assert_eq!(store.try_insert("k1", Arc::new("a".to_string())), Ok(None));
        assert_eq!(store.try_insert("k2", Arc::new("b".to_string())), Ok(None));
        assert_eq!(store.total_weight(), 0);
        assert_eq!(
            store.try_insert("k3", Arc::new("c".to_string())),
            Err(StoreFull)
        );
    }

    #[test]
    fn weight_store_remove_missing_does_not_mutate_weight_or_metrics() {
        let mut store = WeightStore::with_capacity(4, 20, weight_by_len);
        assert_eq!(store.try_insert("k1", Arc::new("aa".to_string())), Ok(None));
        let before_weight = store.total_weight();
        let before_metrics = store.metrics();

        assert_eq!(store.remove(&"missing"), None);
        assert_eq!(store.total_weight(), before_weight);
        assert_eq!(store.metrics().removes, before_metrics.removes);
    }

    #[test]
    fn weight_store_clear_resets_contents_but_preserves_counters() {
        let mut store = WeightStore::with_capacity(4, 20, weight_by_len);
        assert_eq!(store.try_insert("k1", Arc::new("aa".to_string())), Ok(None));
        assert_eq!(store.try_insert("k2", Arc::new("bbb".to_string())), Ok(None));
        let _ = store.get(&"k1");
        let _ = store.get(&"missing");
        store.record_eviction();
        let before = store.metrics();

        store.clear();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
        assert_eq!(store.total_weight(), 0);

        let after = store.metrics();
        assert_eq!(after.inserts, before.inserts);
        assert_eq!(after.hits, before.hits);
        assert_eq!(after.misses, before.misses);
        assert_eq!(after.evictions, before.evictions);
    }

    #[test]
    fn weight_store_api_surface_and_peek_semantics() {
        let mut store = WeightStore::with_capacity(3, 9, weight_by_len);
        assert_eq!(store.capacity(), 3);
        assert_eq!(store.capacity_weight(), 9);
        assert!(store.is_empty());
        assert!(!store.contains(&"k1"));

        assert_eq!(store.try_insert("k1", Arc::new("abc".to_string())), Ok(None));
        assert!(!store.is_empty());
        assert!(store.contains(&"k1"));
        assert_eq!(store.peek(&"k1"), Some(&Arc::new("abc".to_string())));

        let before = store.metrics();
        let _ = store.peek(&"k1");
        let _ = store.peek(&"missing");
        let after = store.metrics();
        assert_eq!(after.hits, before.hits, "peek must not update hits");
        assert_eq!(after.misses, before.misses, "peek must not update misses");

        store.record_eviction();
        assert_eq!(store.metrics().evictions, after.evictions + 1);
    }

    #[test]
    fn weight_store_rejects_insert_weight_overflow() {
        let mut store = WeightStore::with_capacity(2, usize::MAX, |_v: &u8| usize::MAX);
        assert_eq!(store.try_insert(1u8, Arc::new(1u8)), Ok(None));
        assert_eq!(store.try_insert(2u8, Arc::new(2u8)), Err(StoreFull));
    }

    #[test]
    fn weight_store_rejects_update_weight_overflow_without_mutation() {
        fn weight_fn(v: &u8) -> usize {
            match *v {
                0 => usize::MAX - 1,
                1 => 1,
                2 => 3,
                _ => 1,
            }
        }

        let mut store = WeightStore::with_capacity(2, usize::MAX, weight_fn);
        assert_eq!(store.try_insert(1u8, Arc::new(1u8)), Ok(None));
        assert_eq!(store.try_insert(2u8, Arc::new(0u8)), Ok(None));
        assert_eq!(store.total_weight(), usize::MAX);

        assert_eq!(store.try_insert(1u8, Arc::new(2u8)), Err(StoreFull));
        assert_eq!(store.get(&1u8), Some(Arc::new(1u8)));
        assert_eq!(store.total_weight(), usize::MAX);
    }

    proptest! {
        #[test]
        fn weight_store_state_invariants_hold_under_random_ops(
            capacity_entries in 0usize..24,
            capacity_weight in 0usize..1024,
            ops in proptest::collection::vec(op_strategy(), 1..300),
        ) {
            let mut store =
                WeightStore::with_capacity(capacity_entries, capacity_weight, |v: &Vec<u8>| v.len());
            let mut model: HashMap<u8, Vec<u8>> = HashMap::new();

            for op in ops {
                match op {
                    Op::Insert(key, value) => {
                        let model_total_before: usize = model.values().map(|v| v.len()).sum();
                        let store_result = store.try_insert(key, Arc::new(value.clone()));

                        if let Some(old) = model.get(&key).cloned() {
                            let expected_total = model_total_before - old.len() + value.len();
                            if expected_total > capacity_weight {
                                prop_assert_eq!(store_result, Err(StoreFull));
                            } else {
                                prop_assert_eq!(store_result, Ok(Some(Arc::new(old))));
                                model.insert(key, value);
                            }
                        } else if model.len() >= capacity_entries {
                            prop_assert_eq!(store_result, Err(StoreFull));
                        } else {
                            let expected_total = model_total_before + value.len();
                            if expected_total > capacity_weight {
                                prop_assert_eq!(store_result, Err(StoreFull));
                            } else {
                                prop_assert_eq!(store_result, Ok(None));
                                model.insert(key, value);
                            }
                        }
                    },
                    Op::Remove(key) => {
                        let expected = model.remove(&key).map(Arc::new);
                        let actual = store.remove(&key);
                        prop_assert_eq!(actual, expected);
                    },
                    Op::Clear => {
                        model.clear();
                        store.clear();
                    },
                }

                let model_total_after: usize = model.values().map(|v| v.len()).sum();
                prop_assert_eq!(store.total_weight(), model_total_after);
                prop_assert_eq!(store.total_weight(), sum_entry_weights(&store));
                prop_assert_eq!(store.len(), model.len());
                prop_assert!(store.total_weight() <= capacity_weight);
                prop_assert!(store.len() <= capacity_entries);
            }
        }

        #[test]
        fn weight_store_hot_key_update_accounting(
            capacity_weight in 1usize..512,
            updates in proptest::collection::vec(proptest::collection::vec(any::<u8>(), 0..128), 1..200),
        ) {
            let mut store = WeightStore::with_capacity(1, capacity_weight, |v: &Vec<u8>| v.len());
            let mut model_value: Option<Vec<u8>> = None;

            for next_value in updates {
                let result = store.try_insert(1u8, Arc::new(next_value.clone()));
                let expected_weight = next_value.len();
                if expected_weight > capacity_weight {
                    prop_assert_eq!(result, Err(StoreFull));
                } else if let Some(previous) = model_value.replace(next_value.clone()) {
                    prop_assert_eq!(result, Ok(Some(Arc::new(previous))));
                } else {
                    prop_assert_eq!(result, Ok(None));
                }

                let model_total = model_value.as_ref().map(|v| v.len()).unwrap_or(0);
                prop_assert_eq!(store.total_weight(), model_total);
                prop_assert_eq!(store.total_weight(), sum_entry_weights(&store));
                prop_assert!(store.total_weight() <= capacity_weight);
                prop_assert!(store.len() <= 1);
            }
        }

        #[test]
        fn weight_store_near_usize_max_preserves_invariants(
            ops in proptest::collection::vec((any::<u8>(), any::<u16>()), 1..120),
        ) {
            let mut store = WeightStore::with_capacity(8, usize::MAX, |v: &usize| *v);
            let mut model: HashMap<u8, usize> = HashMap::new();

            for (key, raw) in ops {
                let candidate_weight = near_max_weight(raw);
                let model_total_before = model.values().try_fold(0usize, |acc, w| acc.checked_add(*w));
                let model_total_before = model_total_before.unwrap_or(usize::MAX);
                let result = store.try_insert(key, Arc::new(candidate_weight));

                if let Some(old_weight) = model.get(&key).copied() {
                    let base_total = model_total_before
                        .checked_sub(old_weight)
                        .expect("model subtraction should not underflow");
                    let expected_total = base_total.checked_add(candidate_weight);
                    if expected_total.is_none() {
                        prop_assert_eq!(result, Err(StoreFull));
                    } else {
                        prop_assert!(result.is_ok());
                        model.insert(key, candidate_weight);
                    }
                } else if model.len() >= 8 {
                    prop_assert_eq!(result, Err(StoreFull));
                } else {
                    let expected_total = model_total_before.checked_add(candidate_weight);
                    if expected_total.is_none() {
                        prop_assert_eq!(result, Err(StoreFull));
                    } else {
                        prop_assert_eq!(result, Ok(None));
                        model.insert(key, candidate_weight);
                    }
                }

                let recomputed = model.values().try_fold(0usize, |acc, w| acc.checked_add(*w));
                if let Some(model_total_after) = recomputed {
                    prop_assert_eq!(store.total_weight(), model_total_after);
                    prop_assert_eq!(store.total_weight(), sum_entry_weights(&store));
                }
                prop_assert!(store.len() <= 8);
            }
        }

        #[test]
        fn weight_store_metrics_match_reference_model(
            capacity_entries in 0usize..16,
            capacity_weight in 0usize..256,
            ops in proptest::collection::vec(metrics_op_strategy(), 1..250),
        ) {
            let mut store =
                WeightStore::with_capacity(capacity_entries, capacity_weight, |v: &Vec<u8>| v.len());
            let mut model: HashMap<u8, Vec<u8>> = HashMap::new();

            let mut exp_hits = 0u64;
            let mut exp_misses = 0u64;
            let mut exp_inserts = 0u64;
            let mut exp_updates = 0u64;
            let mut exp_removes = 0u64;
            let mut exp_evictions = 0u64;

            for op in ops {
                match op {
                    MetricsOp::Insert(key, value) => {
                        let model_total_before: usize = model.values().map(|v| v.len()).sum();
                        let result = store.try_insert(key, Arc::new(value.clone()));
                        if let Some(old) = model.get(&key).cloned() {
                            let expected_total = model_total_before - old.len() + value.len();
                            if expected_total <= capacity_weight {
                                prop_assert!(matches!(result, Ok(Some(_))));
                                model.insert(key, value);
                                exp_updates += 1;
                            } else {
                                prop_assert_eq!(result, Err(StoreFull));
                            }
                        } else if model.len() < capacity_entries {
                            let expected_total = model_total_before + value.len();
                            if expected_total <= capacity_weight {
                                prop_assert_eq!(result, Ok(None));
                                model.insert(key, value);
                                exp_inserts += 1;
                            } else {
                                prop_assert_eq!(result, Err(StoreFull));
                            }
                        } else {
                            prop_assert_eq!(result, Err(StoreFull));
                        }
                    },
                    MetricsOp::Remove(key) => {
                        let expected = model.remove(&key).map(Arc::new);
                        let actual = store.remove(&key);
                        if expected.is_some() {
                            exp_removes += 1;
                        }
                        prop_assert_eq!(actual, expected);
                    },
                    MetricsOp::Get(key) => {
                        let expected = model.get(&key).cloned().map(Arc::new);
                        let actual = store.get(&key);
                        if expected.is_some() {
                            exp_hits += 1;
                        } else {
                            exp_misses += 1;
                        }
                        prop_assert_eq!(actual, expected);
                    },
                    MetricsOp::Peek(key) => {
                        let expected = model.get(&key).cloned().map(Arc::new);
                        let actual = store.peek(&key).cloned();
                        prop_assert_eq!(actual, expected);
                    },
                    MetricsOp::Clear => {
                        model.clear();
                        store.clear();
                    },
                    MetricsOp::RecordEviction => {
                        store.record_eviction();
                        exp_evictions += 1;
                    },
                }

                let model_total: usize = model.values().map(|v| v.len()).sum();
                prop_assert_eq!(store.total_weight(), model_total);
            }

            let metrics = store.metrics();
            prop_assert_eq!(metrics.hits, exp_hits);
            prop_assert_eq!(metrics.misses, exp_misses);
            prop_assert_eq!(metrics.inserts, exp_inserts);
            prop_assert_eq!(metrics.updates, exp_updates);
            prop_assert_eq!(metrics.removes, exp_removes);
            prop_assert_eq!(metrics.evictions, exp_evictions);
        }
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

    #[cfg(feature = "concurrency")]
    #[test]
    fn concurrent_weight_store_parallel_read_write_preserves_capacity_invariants() {
        let store = Arc::new(ConcurrentWeightStore::with_capacity(
            128,
            4_096,
            |v: &Vec<u8>| v.len(),
        ));

        let mut handles = Vec::new();
        for thread_id in 0..8u8 {
            let shared = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                for i in 0..1_500u16 {
                    let key = ((thread_id as u16) * 64 + (i % 64)) as u32;
                    let len = ((i as usize) % 96) + 1;
                    let payload = Arc::new(vec![thread_id; len]);
                    let _ = shared.try_insert(key, payload);
                    if i % 3 == 0 {
                        let _ = shared.get(&key);
                    }
                    if i % 11 == 0 {
                        let _ = shared.remove(&key);
                    }
                }
            }));
        }

        for handle in handles {
            handle.join().expect("worker thread should not panic");
        }

        assert!(store.len() <= store.capacity());
        assert!(store.total_weight() <= store.capacity_weight());
    }

    #[cfg(feature = "concurrency")]
    #[test]
    fn concurrent_weight_store_failed_ops_do_not_increment_success_counters() {
        let store = Arc::new(ConcurrentWeightStore::with_capacity(1, 1, weight_by_len));
        assert_eq!(
            store.try_insert("seed".to_string(), Arc::new("x".to_string())),
            Ok(None)
        );
        let before = store.metrics();

        let mut handles = Vec::new();
        for i in 0..8usize {
            let shared = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                let missing = "absent_key".to_string();
                for j in 0..500usize {
                    let key = format!("k_{}_{}", i, j);
                    let _ = shared.try_insert(key, Arc::new("yy".to_string()));
                    let _ = shared.remove(&missing);
                }
            }));
        }

        for handle in handles {
            handle.join().expect("worker thread should not panic");
        }

        let after = store.metrics();
        assert_eq!(after.inserts, before.inserts);
        assert_eq!(after.updates, before.updates);
        assert_eq!(after.removes, before.removes);
    }

    #[cfg(feature = "concurrency")]
    #[test]
    fn concurrent_weight_store_clear_race_preserves_weight_invariants() {
        let store = Arc::new(ConcurrentWeightStore::with_capacity(
            64,
            2_048,
            |v: &Vec<u8>| v.len(),
        ));

        let writer = {
            let shared = Arc::clone(&store);
            thread::spawn(move || {
                for i in 0..4_000u32 {
                    let key = i % 128;
                    let len = ((i as usize) % 48) + 1;
                    let _ = shared.try_insert(key, Arc::new(vec![7u8; len]));
                    if i % 5 == 0 {
                        let _ = shared.remove(&key);
                    }
                }
            })
        };

        let clearer = {
            let shared = Arc::clone(&store);
            thread::spawn(move || {
                for _ in 0..300 {
                    shared.clear();
                }
            })
        };

        writer.join().expect("writer thread should not panic");
        clearer.join().expect("clearer thread should not panic");

        assert!(store.len() <= store.capacity());
        assert!(store.total_weight() <= store.capacity_weight());
    }

    // ==============================================
    // Regression Tests
    // ==============================================

    #[cfg(feature = "concurrency")]
    #[test]
    fn concurrent_weight_store_metrics_track_inserts() {
        let store: ConcurrentWeightStore<&str, String, _> =
            ConcurrentWeightStore::with_capacity(100, 100_000, weight_by_len);

        store.try_insert("k1", Arc::new("v1".into())).unwrap();
        store.try_insert("k2", Arc::new("v2".into())).unwrap();
        store.try_insert("k3", Arc::new("v3".into())).unwrap();

        let metrics = store.metrics();
        assert_eq!(metrics.inserts, 3, "metrics should track inserts");
    }

    #[cfg(feature = "concurrency")]
    #[test]
    fn concurrent_weight_store_metrics_track_updates() {
        let store: ConcurrentWeightStore<&str, String, _> =
            ConcurrentWeightStore::with_capacity(100, 100_000, weight_by_len);

        store.try_insert("k1", Arc::new("v1".into())).unwrap();
        store.try_insert("k1", Arc::new("updated".into())).unwrap();

        let metrics = store.metrics();
        assert_eq!(metrics.updates, 1, "metrics should track updates");
    }

    #[cfg(feature = "concurrency")]
    #[test]
    fn concurrent_weight_store_metrics_track_removes() {
        let store: ConcurrentWeightStore<&str, String, _> =
            ConcurrentWeightStore::with_capacity(100, 100_000, weight_by_len);

        store.try_insert("k1", Arc::new("v1".into())).unwrap();
        store.remove(&"k1");

        let metrics = store.metrics();
        assert_eq!(metrics.removes, 1, "metrics should track removes");
    }

    #[cfg(feature = "concurrency")]
    #[test]
    fn concurrent_weight_store_metrics_track_hits_misses() {
        let store: ConcurrentWeightStore<&str, String, _> =
            ConcurrentWeightStore::with_capacity(100, 100_000, weight_by_len);

        store.try_insert("k1", Arc::new("v1".into())).unwrap();
        let _ = store.get(&"k1");
        let _ = store.get(&"missing");

        let metrics = store.metrics();
        assert_eq!(metrics.hits, 1);
        assert_eq!(metrics.misses, 1);
    }
}
