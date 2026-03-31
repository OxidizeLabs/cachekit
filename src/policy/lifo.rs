//! LIFO (Last In, First Out) cache replacement policy.
//!
//! Implements a stack-based eviction algorithm where the most recently inserted
//! entry is evicted first when capacity is reached. This is the opposite of FIFO
//! and is useful for specific workload patterns.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                        LifoCore<K, V> Layout                                │
//! │                                                                             │
//! │   ┌─────────────────────────────────────────────────────────────────────┐   │
//! │   │  map: HashMap<K, V>          stack: Vec<K>                          │   │
//! │   │       key → value                   insertion stack                 │   │
//! │   │                                                                     │   │
//! │   │  ┌──────────┬──────┐          ┌─────────────────────────┐           │   │
//! │   │  │   Key    │Value │          │ Bottom        Top       │           │   │
//! │   │  ├──────────┼──────┤          ├─────────────────────────┤           │   │
//! │   │  │  "page1" │  v1  │          │ [p1] [p2] [p3] [p4]     │           │   │
//! │   │  │  "page2" │  v2  │          │  ↑    ↑    ↑    ↑       │           │   │
//! │   │  │  "page3" │  v3  │          │ old           newest    │           │   │
//! │   │  │  "page4" │  v4  │          │ keep          EVICT     │           │   │
//! │   │  └──────────┴──────┘          └─────────────────────────┘           │   │
//! │   └─────────────────────────────────────────────────────────────────────┘   │
//! │                                                                             │
//! │   ┌─────────────────────────────────────────────────────────────────────┐   │
//! │   │                    LIFO Eviction (Stack)                            │   │
//! │   │                                                                     │   │
//! │   │   • New items pushed to top of stack                                │   │
//! │   │   • Eviction pops from top (most recent)                            │   │
//! │   │   • Opposite of FIFO (which evicts oldest)                          │   │
//! │   │                                                                     │   │
//! │   │   Example: Insert A, B, C, D                                        │   │
//! │   │     Stack: [A, B, C, D]                                             │   │
//! │   │            bottom  ^top                                             │   │
//! │   │                                                                     │   │
//! │   │     Evict → D removed first (newest)                                │   │
//! │   │     Stack: [A, B, C]                                                │   │
//! │   └─────────────────────────────────────────────────────────────────────┘   │
//! │                                                                             │
//! └─────────────────────────────────────────────────────────────────────────────┘
//!
//! Insert Flow (new key)
//! ──────────────────────
//!
//!   insert("new_key", value):
//!     1. Check map - not found
//!     2. Evict if at capacity (pop from top/newest)
//!     3. Push key to top of stack
//!     4. Insert (key, value) into map
//!
//! Access Flow (existing key)
//! ──────────────────────────
//!
//!   get("existing_key"):
//!     1. Lookup value in map
//!     2. Return &value (no reordering!)
//!
//! Eviction Flow
//! ─────────────
//!
//!   evict_if_needed():
//!     while len >= capacity:
//!       pop from stack top (most recent)
//!       remove from map
//! ```
//!
//! ## Key Components
//!
//! - [`LifoCore`]: Main LIFO cache implementation
//!
//! ## Operations
//!
//! | Operation      | Time   | Notes                                      |
//! |----------------|--------|--------------------------------------------|
//! | `get`          | O(1)   | HashMap lookup, records metrics             |
//! | `peek`         | O(1)   | Immutable lookup, no side effects           |
//! | `insert`       | O(1)*  | *Amortized, may trigger eviction            |
//! | `contains`     | O(1)   | HashMap lookup only                         |
//! | `len`          | O(1)   | Returns total entries                       |
//! | `pop_newest`   | O(1)   | Remove most recently inserted               |
//! | `peek_newest`  | O(1)   | Inspect most recently inserted              |
//! | `clear`        | O(n)   | Clears all structures                       |
//!
//! ## Algorithm Properties
//!
//! - **Stack Order**: Most recent insertion at top
//! - **No Access Tracking**: Zero overhead for access patterns
//! - **Opposite of FIFO**: FIFO evicts oldest, LIFO evicts newest
//! - **Niche Use Case**: Only useful for specific workload patterns
//!
//! ## Use Cases
//!
//! - Undo/redo buffers where recent operations are temporary
//! - Temporary scratch space where newest items are least needed
//! - Specific batch processing patterns
//!
//! ## Example Usage
//!
//! ```
//! use cachekit::policy::lifo::LifoCore;
//!
//! // Create LIFO cache with capacity 10
//! let mut cache = LifoCore::new(10);
//!
//! // Insert items (pushed to stack); returns None for new keys
//! assert_eq!(cache.insert(1, 100), None);
//! assert_eq!(cache.insert(2, 200), None);
//! assert_eq!(cache.insert(3, 300), None);
//!
//! // peek provides immutable access without side effects
//! assert_eq!(cache.peek(&1), Some(&100));
//!
//! // pop_newest removes the most recent entry
//! assert_eq!(cache.pop_newest(), Some((3, 300)));
//! assert_eq!(cache.len(), 2);
//!
//! // When cache is full, most recent insertion will be evicted!
//! for i in 3..=15 {
//!     cache.insert(i, i * 10);
//! }
//!
//! assert_eq!(cache.len(), 10);
//! ```
//!
//! ## Thread Safety
//!
//! - [`LifoCore`]: Not thread-safe, designed for single-threaded use
//! - For concurrent access, wrap in external synchronization
//!
//! ## Implementation Notes
//!
//! - Uses `Vec<K>` as stack for insertion order
//! - Uses `HashMap<K, V>` for O(1) lookup
//! - No stale entry tracking (always pops valid entries)
//! - New items pushed to top, eviction from top
//!
//! ## When to Use
//!
//! **Use LIFO when:**
//! - Newest insertions are least likely to be reused
//! - Building temporary scratch spaces
//! - Undo/redo buffer management
//! - Specific batch processing patterns
//!
//! **Avoid LIFO when:**
//! - Temporal locality exists (use LRU instead)
//! - Frequency matters (use LFU instead)
//! - General-purpose caching (use LRU, SLRU, S3-FIFO)
//! - Predictable behavior needed (FIFO is more intuitive)
//!
//! ## References
//!
//! - Wikipedia: Cache replacement policies

#[cfg(feature = "metrics")]
use crate::metrics::metrics_impl::CoreOnlyMetrics;
#[cfg(feature = "metrics")]
use crate::metrics::snapshot::CoreOnlyMetricsSnapshot;
#[cfg(feature = "metrics")]
use crate::metrics::traits::{CoreMetricsRecorder, MetricsSnapshotProvider};
use crate::traits::{Cache, EvictingCache, VictimInspectable};
use rustc_hash::FxHashMap;
use std::hash::Hash;

/// Core LIFO (Last In, First Out) cache implementation.
///
/// Implements stack-based eviction where the most recently inserted
/// entry is evicted first when capacity is reached.
///
/// # Type Parameters
///
/// - `K`: Key type, must be `Clone + Eq + Hash` on most operations
/// - `V`: Value type
///
/// # Example
///
/// ```
/// use cachekit::policy::lifo::LifoCore;
///
/// let mut cache = LifoCore::new(100);
///
/// // New insert returns None
/// assert_eq!(cache.insert("key1", "value1"), None);
/// assert!(cache.contains(&"key1"));
///
/// // peek provides immutable access without metrics recording
/// assert_eq!(cache.peek(&"key1"), Some(&"value1"));
///
/// // Update returns the previous value
/// assert_eq!(cache.insert("key1", "new_value"), Some("value1"));
/// assert_eq!(cache.peek(&"key1"), Some(&"new_value"));
/// ```
///
/// # Eviction Behavior
///
/// When capacity is exceeded, evicts the most recently inserted entry (top of stack).
///
/// # Implementation
///
/// Uses Vec as stack + HashMap for O(1) operations.
#[must_use]
pub struct LifoCore<K, V> {
    map: FxHashMap<K, V>,
    stack: Vec<K>,
    capacity: usize,
    #[cfg(feature = "metrics")]
    metrics: CoreOnlyMetrics,
}

impl<K, V> LifoCore<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Creates a new LIFO cache with the specified capacity.
    ///
    /// A capacity of `0` creates a cache that accepts no entries;
    /// all [`insert`](Self::insert) calls return `None` and are no-ops.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::lifo::LifoCore;
    ///
    /// let cache: LifoCore<String, i32> = LifoCore::new(100);
    /// assert_eq!(cache.capacity(), 100);
    /// assert!(cache.is_empty());
    /// ```
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Self {
            map: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            stack: Vec::with_capacity(capacity),
            capacity,
            #[cfg(feature = "metrics")]
            metrics: CoreOnlyMetrics::default(),
        }
    }

    /// Retrieves a value by key without affecting eviction order.
    ///
    /// Unlike LRU, accessing an item in a LIFO cache doesn't change
    /// its position in the stack. Records metrics when the `metrics`
    /// feature is enabled; use [`peek`](Self::peek) for a side-effect-free
    /// alternative.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::lifo::LifoCore;
    ///
    /// let mut cache = LifoCore::new(100);
    /// cache.insert("key", 42);
    ///
    /// assert_eq!(cache.get(&"key"), Some(&42));
    /// assert_eq!(cache.get(&"missing"), None);
    /// ```
    #[inline]
    pub fn get(&mut self, key: &K) -> Option<&V> {
        #[cfg(feature = "metrics")]
        if self.map.contains_key(key) {
            self.metrics.record_get_hit();
        } else {
            self.metrics.record_get_miss();
        }
        self.map.get(key)
    }

    /// Looks up a value by key without requiring mutable access.
    ///
    /// Unlike [`get`](Self::get), this takes `&self` and never records
    /// metrics. Useful when you only need to read without side effects.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::lifo::LifoCore;
    ///
    /// let mut cache = LifoCore::new(100);
    /// cache.insert("key", 42);
    ///
    /// assert_eq!(cache.peek(&"key"), Some(&42));
    /// assert_eq!(cache.peek(&"missing"), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.map.get(key)
    }

    /// Inserts or updates a key-value pair, returning the previous value.
    ///
    /// - If the key exists, updates the value in place (no stack change)
    ///   and returns `Some(old_value)`
    /// - If the key is new, pushes to top of stack and returns `None`
    /// - May trigger eviction from top of stack (most recent) if at capacity
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::lifo::LifoCore;
    ///
    /// let mut cache = LifoCore::new(100);
    ///
    /// // New insert returns None
    /// assert_eq!(cache.insert("key", "initial"), None);
    /// assert_eq!(cache.len(), 1);
    ///
    /// // Update returns the previous value
    /// assert_eq!(cache.insert("key", "updated"), Some("initial"));
    /// assert_eq!(cache.peek(&"key"), Some(&"updated"));
    /// assert_eq!(cache.len(), 1);  // Still 1 entry
    /// ```
    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        #[cfg(feature = "metrics")]
        self.metrics.record_insert_call();

        if self.capacity == 0 {
            return None;
        }

        if let Some(v) = self.map.get_mut(&key) {
            #[cfg(feature = "metrics")]
            self.metrics.record_insert_update();
            return Some(std::mem::replace(v, value));
        }

        #[cfg(feature = "metrics")]
        self.metrics.record_insert_new();

        self.evict_if_needed();

        self.stack.push(key.clone());
        self.map.insert(key, value);
        None
    }

    /// Evicts entries from top of stack until there is room.
    ///
    /// LIFO evicts from the top (most recently inserted).
    #[inline]
    fn evict_if_needed(&mut self) {
        #[cfg(feature = "metrics")]
        if self.len() >= self.capacity && !self.stack.is_empty() {
            self.metrics.record_evict_call();
        }

        while self.len() >= self.capacity && !self.stack.is_empty() {
            if let Some(key) = self.stack.pop() {
                self.map.remove(&key);
                #[cfg(feature = "metrics")]
                self.metrics.record_evicted_entry();
            } else {
                break;
            }
        }

        #[cfg(debug_assertions)]
        self.validate_invariants();
    }

    /// Returns the number of entries in the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::lifo::LifoCore;
    ///
    /// let mut cache = LifoCore::new(100);
    /// assert_eq!(cache.len(), 0);
    ///
    /// cache.insert("a", 1);
    /// cache.insert("b", 2);
    /// assert_eq!(cache.len(), 2);
    /// ```
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the cache is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::lifo::LifoCore;
    ///
    /// let mut cache: LifoCore<&str, i32> = LifoCore::new(100);
    /// assert!(cache.is_empty());
    ///
    /// cache.insert("key", 42);
    /// assert!(!cache.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns the total cache capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::lifo::LifoCore;
    ///
    /// let cache: LifoCore<String, i32> = LifoCore::new(500);
    /// assert_eq!(cache.capacity(), 500);
    /// ```
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns `true` if the key exists in the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::lifo::LifoCore;
    ///
    /// let mut cache = LifoCore::new(100);
    /// cache.insert("key", 42);
    ///
    /// assert!(cache.contains(&"key"));
    /// assert!(!cache.contains(&"missing"));
    /// ```
    #[inline]
    #[must_use]
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Clears all entries from the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::lifo::LifoCore;
    ///
    /// let mut cache = LifoCore::new(100);
    /// cache.insert("a", 1);
    /// cache.insert("b", 2);
    ///
    /// cache.clear();
    /// assert!(cache.is_empty());
    /// assert!(!cache.contains(&"a"));
    /// ```
    pub fn clear(&mut self) {
        #[cfg(feature = "metrics")]
        self.metrics.record_clear();

        self.map.clear();
        self.stack.clear();

        #[cfg(debug_assertions)]
        self.validate_invariants();
    }

    /// Removes and returns the most recently inserted entry (top of stack).
    ///
    /// Returns `None` if the cache is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::lifo::LifoCore;
    ///
    /// let mut cache = LifoCore::new(10);
    /// cache.insert(1, "first");
    /// cache.insert(2, "second");
    /// cache.insert(3, "third");
    ///
    /// assert_eq!(cache.pop_newest(), Some((3, "third")));
    /// assert_eq!(cache.pop_newest(), Some((2, "second")));
    /// assert_eq!(cache.len(), 1);
    /// ```
    #[must_use]
    pub fn pop_newest(&mut self) -> Option<(K, V)> {
        let key = self.stack.pop()?;
        let value = self.map.remove(&key)?;
        Some((key, value))
    }

    /// Peeks at the most recently inserted entry without removing it.
    ///
    /// Returns `None` if the cache is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::lifo::LifoCore;
    ///
    /// let mut cache = LifoCore::new(10);
    /// cache.insert(1, "first");
    /// cache.insert(2, "second");
    ///
    /// assert_eq!(cache.peek_newest(), Some((&2, &"second")));
    /// assert_eq!(cache.len(), 2); // Not removed
    /// ```
    #[must_use]
    pub fn peek_newest(&self) -> Option<(&K, &V)> {
        let key = self.stack.last()?;
        let value = self.map.get(key)?;
        Some((key, value))
    }

    /// Returns an iterator over all key-value pairs in arbitrary order.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::lifo::LifoCore;
    ///
    /// let mut cache = LifoCore::new(10);
    /// cache.insert("a", 1);
    /// cache.insert("b", 2);
    ///
    /// let pairs: Vec<_> = cache.iter().collect();
    /// assert_eq!(pairs.len(), 2);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.map.iter()
    }

    /// Returns an iterator over all keys in arbitrary order.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.map.keys()
    }

    /// Returns an iterator over all values in arbitrary order.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.map.values()
    }

    /// Validates internal data structure invariants.
    ///
    /// This method checks that:
    /// - Map size matches stack size
    /// - All keys in map exist in stack
    /// - All keys in stack exist in map
    /// - No duplicate keys in stack
    ///
    /// Only runs when debug assertions are enabled.
    #[cfg(debug_assertions)]
    fn validate_invariants(&self) {
        // Map and stack should have same size
        debug_assert_eq!(
            self.map.len(),
            self.stack.len(),
            "Map and stack have different sizes"
        );

        // All keys in map should exist in stack
        for key in self.map.keys() {
            debug_assert!(self.stack.contains(key), "Key in map not found in stack");
        }

        // All keys in stack should exist in map
        for key in &self.stack {
            debug_assert!(self.map.contains_key(key), "Key in stack not found in map");
        }

        // No duplicates in stack
        let unique_count = self
            .stack
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        debug_assert_eq!(unique_count, self.stack.len(), "Duplicate keys in stack");
    }
}

impl<K, V> std::fmt::Debug for LifoCore<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LifoCore")
            .field("capacity", &self.capacity)
            .field("len", &self.map.len())
            .field("stack_len", &self.stack.len())
            .finish_non_exhaustive()
    }
}

impl<K, V> Cache<K, V> for LifoCore<K, V>
where
    K: Clone + Eq + Hash,
{
    #[inline]
    fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    #[inline]
    fn len(&self) -> usize {
        self.map.len()
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    fn peek(&self, key: &K) -> Option<&V> {
        self.map.get(key)
    }

    #[inline]
    fn get(&mut self, key: &K) -> Option<&V> {
        LifoCore::get(self, key)
    }

    #[inline]
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        LifoCore::insert(self, key, value)
    }

    fn remove(&mut self, key: &K) -> Option<V> {
        let value = self.map.remove(key)?;
        if let Some(pos) = self.stack.iter().position(|k| k == key) {
            self.stack.remove(pos);
        }
        #[cfg(debug_assertions)]
        self.validate_invariants();
        Some(value)
    }

    fn clear(&mut self) {
        LifoCore::clear(self);
    }
}

impl<K, V> EvictingCache<K, V> for LifoCore<K, V>
where
    K: Clone + Eq + Hash,
{
    fn evict_one(&mut self) -> Option<(K, V)> {
        self.pop_newest()
    }
}

impl<K, V> VictimInspectable<K, V> for LifoCore<K, V>
where
    K: Clone + Eq + Hash,
{
    fn peek_victim(&self) -> Option<(&K, &V)> {
        self.peek_newest()
    }
}

#[cfg(feature = "metrics")]
impl<K, V> LifoCore<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Returns a snapshot of cache metrics.
    pub fn metrics_snapshot(&self) -> CoreOnlyMetricsSnapshot {
        CoreOnlyMetricsSnapshot {
            get_calls: self.metrics.get_calls,
            get_hits: self.metrics.get_hits,
            get_misses: self.metrics.get_misses,
            insert_calls: self.metrics.insert_calls,
            insert_updates: self.metrics.insert_updates,
            insert_new: self.metrics.insert_new,
            evict_calls: self.metrics.evict_calls,
            evicted_entries: self.metrics.evicted_entries,
            cache_len: self.len(),
            capacity: self.capacity,
        }
    }
}

#[cfg(feature = "metrics")]
impl<K, V> MetricsSnapshotProvider<CoreOnlyMetricsSnapshot> for LifoCore<K, V>
where
    K: Clone + Eq + Hash,
{
    fn snapshot(&self) -> CoreOnlyMetricsSnapshot {
        self.metrics_snapshot()
    }
}

impl<K, V> Default for LifoCore<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Returns a zero-capacity cache that rejects all insertions.
    fn default() -> Self {
        Self::new(0)
    }
}

impl<K, V> Clone for LifoCore<K, V>
where
    K: Clone + Eq + Hash,
    V: Clone,
{
    fn clone(&self) -> Self {
        Self {
            map: self.map.clone(),
            stack: self.stack.clone(),
            capacity: self.capacity,
            #[cfg(feature = "metrics")]
            metrics: CoreOnlyMetrics::default(),
        }
    }
}

impl<K, V> Extend<(K, V)> for LifoCore<K, V>
where
    K: Clone + Eq + Hash,
{
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<K, V> FromIterator<(K, V)> for LifoCore<K, V>
where
    K: Clone + Eq + Hash,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut cache = Self::new(lower.max(16));
        for (k, v) in iter {
            cache.insert(k, v);
        }
        cache
    }
}

impl<K, V> IntoIterator for LifoCore<K, V>
where
    K: Clone + Eq + Hash,
{
    type Item = (K, V);
    type IntoIter = std::collections::hash_map::IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.map.into_iter()
    }
}

impl<'a, K, V> IntoIterator for &'a LifoCore<K, V>
where
    K: Clone + Eq + Hash,
{
    type Item = (&'a K, &'a V);
    type IntoIter = std::collections::hash_map::Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.map.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==============================================
    // LifoCore Basic Operations
    // ==============================================

    mod basic_operations {
        use super::*;

        #[test]
        fn new_cache_is_empty() {
            let cache: LifoCore<&str, i32> = LifoCore::new(100);
            assert!(cache.is_empty());
            assert_eq!(cache.len(), 0);
            assert_eq!(cache.capacity(), 100);
        }

        #[test]
        fn insert_and_get() {
            let mut cache = LifoCore::new(100);
            cache.insert("key1", "value1");

            assert_eq!(cache.len(), 1);
            assert_eq!(cache.get(&"key1"), Some(&"value1"));
        }

        #[test]
        fn insert_multiple_items() {
            let mut cache = LifoCore::new(100);
            cache.insert("a", 1);
            cache.insert("b", 2);
            cache.insert("c", 3);

            assert_eq!(cache.len(), 3);
            assert_eq!(cache.get(&"a"), Some(&1));
            assert_eq!(cache.get(&"b"), Some(&2));
            assert_eq!(cache.get(&"c"), Some(&3));
        }

        #[test]
        fn get_missing_key_returns_none() {
            let mut cache: LifoCore<&str, i32> = LifoCore::new(100);

            assert_eq!(cache.get(&"missing"), None);
        }

        #[test]
        fn update_existing_key() {
            let mut cache = LifoCore::new(100);
            cache.insert("key", "initial");
            cache.insert("key", "updated");

            assert_eq!(cache.len(), 1);
            assert_eq!(cache.get(&"key"), Some(&"updated"));
        }

        #[test]
        fn contains_returns_correct_result() {
            let mut cache = LifoCore::new(100);
            cache.insert("exists", 1);

            assert!(cache.contains(&"exists"));
            assert!(!cache.contains(&"missing"));
        }

        #[test]
        fn clear_removes_all_entries() {
            let mut cache = LifoCore::new(100);
            cache.insert("a", 1);
            cache.insert("b", 2);

            cache.clear();

            assert!(cache.is_empty());
            assert_eq!(cache.len(), 0);
            assert!(!cache.contains(&"a"));
            assert!(!cache.contains(&"b"));
        }

        #[test]
        fn capacity_returns_correct_value() {
            let cache: LifoCore<i32, i32> = LifoCore::new(500);
            assert_eq!(cache.capacity(), 500);
        }
    }

    // ==============================================
    // LIFO-Specific Behavior (Evict Most Recent)
    // ==============================================

    mod lifo_behavior {
        use super::*;

        #[test]
        fn evicts_most_recently_inserted() {
            let mut cache = LifoCore::new(3);

            cache.insert("first", 1);
            cache.insert("second", 2);
            cache.insert("third", 3);

            // All 3 should be present
            assert_eq!(cache.len(), 3);

            // Insert "fourth" - should evict "third" (most recent)
            cache.insert("fourth", 4);

            assert_eq!(cache.len(), 3);
            assert!(cache.contains(&"first"));
            assert!(cache.contains(&"second"));
            assert!(
                !cache.contains(&"third"),
                "Most recent 'third' should be evicted"
            );
            assert!(cache.contains(&"fourth"));
        }

        #[test]
        fn stack_order_maintained() {
            let mut cache = LifoCore::new(3);

            cache.insert(1, 10);
            cache.insert(2, 20);
            cache.insert(3, 30);

            // Insert 4 - evicts 3 (most recent)
            cache.insert(4, 40);
            assert!(!cache.contains(&3));
            assert!(cache.contains(&1));
            assert!(cache.contains(&2));
            assert!(cache.contains(&4));

            // Insert 5 - evicts 4 (most recent)
            cache.insert(5, 50);
            assert!(!cache.contains(&4));
            assert!(cache.contains(&1));
            assert!(cache.contains(&2));
            assert!(cache.contains(&5));
        }

        #[test]
        fn opposite_of_fifo_behavior() {
            let mut cache = LifoCore::new(3);

            cache.insert("oldest", 1);
            cache.insert("middle", 2);
            cache.insert("newest", 3);

            // In FIFO, "oldest" would be evicted
            // In LIFO, "newest" is evicted
            cache.insert("new", 4);

            assert!(cache.contains(&"oldest"), "Oldest should stay in LIFO");
            assert!(cache.contains(&"middle"));
            assert!(
                !cache.contains(&"newest"),
                "Newest should be evicted in LIFO"
            );
            assert!(cache.contains(&"new"));
        }
    }

    // ==============================================
    // Eviction Behavior
    // ==============================================

    mod eviction_behavior {
        use super::*;

        #[test]
        fn eviction_occurs_when_over_capacity() {
            let mut cache = LifoCore::new(5);

            for i in 0..10 {
                cache.insert(i, i * 10);
            }

            assert_eq!(cache.len(), 5);
        }

        #[test]
        fn eviction_removes_from_map() {
            let mut cache = LifoCore::new(3);

            cache.insert("a", 1);
            cache.insert("b", 2);
            cache.insert("c", 3);

            assert!(cache.contains(&"c"));

            cache.insert("d", 4);

            // "c" was most recent, so it should be evicted
            assert!(!cache.contains(&"c"));
            assert_eq!(cache.get(&"c"), None);
        }

        #[test]
        fn continuous_insertions_evict_correctly() {
            let mut cache = LifoCore::new(3);

            cache.insert(1, 10);
            cache.insert(2, 20);
            cache.insert(3, 30);
            assert_eq!(cache.len(), 3);

            cache.insert(4, 40);
            assert_eq!(cache.len(), 3);
            assert!(!cache.contains(&3)); // 3 was most recent

            cache.insert(5, 50);
            assert_eq!(cache.len(), 3);
            assert!(!cache.contains(&4)); // 4 was most recent
        }

        #[test]
        fn oldest_items_survive() {
            let mut cache = LifoCore::new(3);

            cache.insert(1, 10);
            cache.insert(2, 20);
            cache.insert(3, 30);

            // Insert many more items - oldest should survive
            for i in 4..=10 {
                cache.insert(i, i * 10);
            }

            // Item 1 should still exist (oldest)
            assert!(cache.contains(&1), "Oldest item should survive in LIFO");
            assert_eq!(cache.len(), 3);
        }
    }

    // ==============================================
    // Get Does Not Affect Eviction
    // ==============================================

    mod get_behavior {
        use super::*;

        #[test]
        fn get_does_not_change_eviction_order() {
            let mut cache = LifoCore::new(3);

            cache.insert(1, 10);
            cache.insert(2, 20);
            cache.insert(3, 30);

            // Access item 1 many times
            for _ in 0..100 {
                cache.get(&1);
            }

            // Insert item 4 - should still evict 3 (most recent insertion)
            // even though 1 was accessed more
            cache.insert(4, 40);

            assert!(cache.contains(&1));
            assert!(cache.contains(&2));
            assert!(
                !cache.contains(&3),
                "Most recent insert evicted despite 1 being accessed"
            );
            assert!(cache.contains(&4));
        }
    }

    // ==============================================
    // Edge Cases
    // ==============================================

    mod edge_cases {
        use super::*;

        #[test]
        fn single_capacity_cache() {
            let mut cache = LifoCore::new(1);

            cache.insert("a", 1);
            assert_eq!(cache.get(&"a"), Some(&1));

            cache.insert("b", 2);
            assert!(!cache.contains(&"a"));
            assert_eq!(cache.get(&"b"), Some(&2));
        }

        #[test]
        fn zero_capacity_cache() {
            let mut cache = LifoCore::new(0);

            cache.insert("a", 1);
            assert_eq!(cache.len(), 0);
            assert!(!cache.contains(&"a"));
        }

        #[test]
        fn get_after_update() {
            let mut cache = LifoCore::new(100);

            cache.insert("key", "v1");
            assert_eq!(cache.get(&"key"), Some(&"v1"));

            cache.insert("key", "v2");
            assert_eq!(cache.get(&"key"), Some(&"v2"));

            cache.insert("key", "v3");
            cache.insert("key", "v4");
            assert_eq!(cache.get(&"key"), Some(&"v4"));
        }

        #[test]
        fn large_capacity() {
            let mut cache = LifoCore::new(10000);

            for i in 0..10000 {
                cache.insert(i, i * 2);
            }

            assert_eq!(cache.len(), 10000);

            assert_eq!(cache.get(&5000), Some(&10000));
            assert_eq!(cache.get(&9999), Some(&19998));
        }

        #[test]
        fn empty_cache_operations() {
            let mut cache: LifoCore<i32, i32> = LifoCore::new(100);

            assert!(cache.is_empty());
            assert_eq!(cache.get(&1), None);
            assert!(!cache.contains(&1));
        }

        #[test]
        fn string_keys_and_values() {
            let mut cache = LifoCore::new(100);

            cache.insert(String::from("hello"), String::from("world"));
            cache.insert(String::from("foo"), String::from("bar"));

            assert_eq!(
                cache.get(&String::from("hello")),
                Some(&String::from("world"))
            );
            assert_eq!(cache.get(&String::from("foo")), Some(&String::from("bar")));
        }

        #[test]
        fn update_preserves_stack_position() {
            let mut cache = LifoCore::new(3);

            cache.insert(1, 10);
            cache.insert(2, 20);
            cache.insert(3, 30);

            // Update item 1 (oldest) - should not change stack position
            cache.insert(1, 100);

            // Insert item 4 - should still evict 3 (most recent insert)
            cache.insert(4, 40);

            assert!(cache.contains(&1), "Updated item should preserve position");
            assert!(cache.contains(&2));
            assert!(!cache.contains(&3), "Most recent insert still evicted");
            assert!(cache.contains(&4));
        }
    }

    // ==============================================
    // New API: peek, pop_newest, peek_newest
    // ==============================================

    mod lifo_api {
        use super::*;

        #[test]
        fn peek_returns_value_immutably() {
            let mut cache = LifoCore::new(100);
            cache.insert("key", 42);

            assert_eq!(cache.peek(&"key"), Some(&42));
            assert_eq!(cache.peek(&"missing"), None);
        }

        #[test]
        fn peek_allows_multiple_borrows() {
            let mut cache = LifoCore::new(100);
            cache.insert("a", 1);
            cache.insert("b", 2);

            let a = cache.peek(&"a");
            let b = cache.peek(&"b");
            assert_eq!(a, Some(&1));
            assert_eq!(b, Some(&2));
        }

        #[test]
        fn pop_newest_returns_most_recent() {
            let mut cache = LifoCore::new(10);
            cache.insert(1, "first");
            cache.insert(2, "second");
            cache.insert(3, "third");

            assert_eq!(cache.pop_newest(), Some((3, "third")));
            assert_eq!(cache.pop_newest(), Some((2, "second")));
            assert_eq!(cache.pop_newest(), Some((1, "first")));
            assert_eq!(cache.pop_newest(), None);
        }

        #[test]
        fn peek_newest_shows_top_of_stack() {
            let mut cache = LifoCore::new(10);
            assert_eq!(cache.peek_newest(), None);

            cache.insert(1, "first");
            assert_eq!(cache.peek_newest(), Some((&1, &"first")));

            cache.insert(2, "second");
            assert_eq!(cache.peek_newest(), Some((&2, &"second")));
            assert_eq!(cache.len(), 2);
        }

        #[test]
        fn insert_returns_previous_value() {
            let mut cache = LifoCore::new(100);

            assert_eq!(cache.insert("key", "v1"), None);
            assert_eq!(cache.insert("key", "v2"), Some("v1"));
            assert_eq!(cache.insert("key", "v3"), Some("v2"));
            assert_eq!(cache.insert("new", "val"), None);
        }
    }

    // ==============================================
    // Iterator and Collection Traits
    // ==============================================

    mod collection_traits {
        use super::*;

        #[test]
        fn iter_yields_all_entries() {
            let mut cache = LifoCore::new(10);
            cache.insert("a", 1);
            cache.insert("b", 2);
            cache.insert("c", 3);

            let mut pairs: Vec<_> = cache.iter().collect();
            pairs.sort_by_key(|(k, _)| *k);
            assert_eq!(pairs, vec![(&"a", &1), (&"b", &2), (&"c", &3)]);
        }

        #[test]
        fn keys_and_values() {
            let mut cache = LifoCore::new(10);
            cache.insert(1, "a");
            cache.insert(2, "b");

            let mut keys: Vec<_> = cache.keys().collect();
            keys.sort();
            assert_eq!(keys, vec![&1, &2]);

            let mut values: Vec<_> = cache.values().collect();
            values.sort();
            assert_eq!(values, vec![&"a", &"b"]);
        }

        #[test]
        fn extend_inserts_all() {
            let mut cache = LifoCore::new(10);
            cache.insert(1, "one");

            cache.extend(vec![(2, "two"), (3, "three")]);

            assert_eq!(cache.len(), 3);
            assert_eq!(cache.peek(&2), Some(&"two"));
            assert_eq!(cache.peek(&3), Some(&"three"));
        }

        #[test]
        fn from_iterator() {
            let cache: LifoCore<i32, &str> = vec![(1, "one"), (2, "two"), (3, "three")]
                .into_iter()
                .collect();

            assert_eq!(cache.len(), 3);
            assert!(cache.capacity() >= 3);
            assert!(cache.contains(&1));
            assert!(cache.contains(&2));
            assert!(cache.contains(&3));
        }

        #[test]
        fn into_iterator_owned() {
            let mut cache = LifoCore::new(10);
            cache.insert(1, "a");
            cache.insert(2, "b");

            let mut pairs: Vec<_> = cache.into_iter().collect();
            pairs.sort_by_key(|(k, _)| *k);
            assert_eq!(pairs, vec![(1, "a"), (2, "b")]);
        }

        #[test]
        fn into_iterator_ref() {
            let mut cache = LifoCore::new(10);
            cache.insert(1, "a");
            cache.insert(2, "b");

            let mut pairs: Vec<_> = (&cache).into_iter().collect();
            pairs.sort_by_key(|(k, _)| *k);
            assert_eq!(pairs, vec![(&1, &"a"), (&2, &"b")]);
        }
    }

    // ==============================================
    // Default and Clone
    // ==============================================

    mod default_and_clone {
        use super::*;

        #[test]
        fn default_creates_zero_capacity() {
            let cache: LifoCore<String, i32> = LifoCore::default();
            assert_eq!(cache.capacity(), 0);
            assert!(cache.is_empty());
        }

        #[test]
        fn clone_preserves_state() {
            let mut original = LifoCore::new(10);
            original.insert(1, "a");
            original.insert(2, "b");

            let cloned = original.clone();
            assert_eq!(cloned.len(), 2);
            assert_eq!(cloned.capacity(), 10);
            assert_eq!(cloned.peek(&1), Some(&"a"));
            assert_eq!(cloned.peek(&2), Some(&"b"));
        }

        #[test]
        fn clone_is_independent() {
            let mut original = LifoCore::new(10);
            original.insert(1, "a");

            let mut cloned = original.clone();
            cloned.insert(2, "b");

            assert_eq!(original.len(), 1);
            assert_eq!(cloned.len(), 2);
        }
    }

    // ==============================================
    // Validation Tests
    // ==============================================

    #[test]
    #[cfg(debug_assertions)]
    fn validate_invariants_after_operations() {
        let mut cache = LifoCore::new(10);

        for i in 1..=10 {
            cache.insert(i, i * 100);
        }
        cache.validate_invariants();

        for _ in 0..5 {
            cache.get(&5);
        }
        cache.validate_invariants();

        cache.insert(11, 1100);
        cache.validate_invariants();

        cache.insert(12, 1200);
        cache.validate_invariants();

        cache.clear();
        cache.validate_invariants();

        assert_eq!(cache.len(), 0);
        assert_eq!(cache.stack.len(), 0);
    }

    #[test]
    #[cfg(debug_assertions)]
    fn validate_invariants_with_stack_consistency() {
        let mut cache = LifoCore::new(5);
        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.insert(3, 300);
        cache.validate_invariants();

        for i in 4..=10 {
            cache.insert(i, i * 100);
            cache.validate_invariants();
        }

        assert_eq!(cache.len(), 5);
        assert_eq!(cache.stack.len(), 5);

        for key in &cache.stack {
            assert!(cache.map.contains_key(key));
        }
    }
}
