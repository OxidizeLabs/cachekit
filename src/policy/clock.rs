//! Clock cache replacement policy.
//!
//! Implements the Clock algorithm (also known as Second-Chance), which approximates
//! LRU with O(1) access operations by avoiding linked list manipulation.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                         ClockCache<K, V> Layout                             │
//! │                                                                             │
//! │   ┌─────────────────────────────────────────────────────────────────────┐   │
//! │   │  index: FxHashMap<K, usize>     (key -> slot index)                 │   │
//! │   └─────────────────────────────────────────────────────────────────────┘   │
//! │                                                                             │
//! │   ┌─────────────────────────────────────────────────────────────────────┐   │
//! │   │  slots: Vec<Option<Entry<K,V>>>   (circular buffer)                 │   │
//! │   │                                                                     │   │
//! │   │    [0]     [1]     [2]     [3]     [4]     [5]     [6]     [7]      │   │
//! │   │   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐     │   │
//! │   │   │ A │   │ B │   │ C │   │ D │   │ E │   │   │   │   │   │   │     │   │
//! │   │   │ref│   │ref│   │   │   │ref│   │   │   │   │   │   │   │   │     │   │
//! │   │   └───┘   └───┘   └───┘   └───┘   └───┘   └───┘   └───┘   └───┘     │   │
//! │   │                     ▲                                               │   │
//! │   │                     │                                               │   │
//! │   │                   hand (clock pointer)                              │   │
//! │   └─────────────────────────────────────────────────────────────────────┘   │
//! │                                                                             │
//! │   On access: Set referenced bit (no list operations!)                       │
//! │   On eviction: Sweep from hand, clear ref bits, evict first unreferenced    │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Algorithm
//!
//! ```text
//! GET(key):
//!   1. Look up slot index in hash map
//!   2. Set referenced = true
//!   3. Return value
//!   Cost: O(1) - just a hash lookup and bit set!
//!
//! INSERT(key, value):
//!   1. If key exists: update value, set referenced = true
//!   2. If at capacity: run eviction
//!   3. Find empty slot, insert entry
//!
//! EVICT():
//!   while true:
//!     entry = slots[hand]
//!     if entry.referenced:
//!       entry.referenced = false  // second chance
//!       hand = (hand + 1) % capacity
//!     else:
//!       remove entry
//!       return slot
//! ```
//!
//! ## Performance Characteristics
//!
//! | Operation | Time    | Notes                             |
//! |-----------|---------|-----------------------------------|
//! | `get`     | O(1)    | Hash lookup + bit set             |
//! | `insert`  | O(1)*   | *Amortized, eviction may sweep    |
//! | `contains`| O(1)    | Hash lookup only                  |
//! | `remove`  | O(1)    | Hash lookup + clear slot          |
//!
//! ## Why Clock is Fast
//!
//! - **No linked list**: Unlike LRU, no pointer manipulation on access
//! - **Cache-friendly**: Contiguous Vec storage, good memory locality
//! - **Bit operations**: Just setting a boolean on access
//! - **Amortized eviction**: Sweeping is O(n) worst case but O(1) amortized
//!
//! ## Trade-offs
//!
//! | Aspect        | Clock                    | True LRU                |
//! |---------------|--------------------------|-------------------------|
//! | Access cost   | O(1) bit set             | O(1) list move          |
//! | Memory layout | Contiguous (cache-friendly) | Scattered nodes      |
//! | Eviction      | Approximate LRU          | Exact LRU               |
//! | Overhead/entry| ~1 byte (ref bit)        | ~16 bytes (2 pointers)  |
//!
//! ## Example Usage
//!
//! ```
//! use cachekit::policy::clock::ClockCache;
//! use cachekit::traits::{CoreCache, ReadOnlyCache};
//!
//! let mut cache = ClockCache::new(100);
//!
//! // Insert items
//! cache.insert("page1", "content1");
//! cache.insert("page2", "content2");
//!
//! // Access sets reference bit (no list operations!)
//! assert_eq!(cache.get(&"page1"), Some(&"content1"));
//!
//! // Referenced items get a "second chance" before eviction
//! ```
//!
//! ## Implementation
//!
//! This implementation uses the [`ClockRing`] data structure,
//! which provides:
//! - Cache-line optimized entry layout (`#[repr(C)]` with `referenced` first)
//! - Hand-based empty slot finding (O(1) amortized, no linear scan)
//! - O(1) amortized operations

use std::hash::Hash;

use crate::ds::ClockRing;
use crate::traits::{CoreCache, MutableCache, ReadOnlyCache};

#[cfg(feature = "metrics")]
use crate::metrics::metrics_impl::ClockMetrics;
#[cfg(feature = "metrics")]
use crate::metrics::snapshot::ClockMetricsSnapshot;
#[cfg(feature = "metrics")]
use crate::metrics::traits::MetricsSnapshotProvider;

/// High-performance Clock cache with O(1) access operations.
///
/// Uses the [`ClockRing`] data structure with a sweeping clock hand for eviction.
/// Approximates LRU without the linked list overhead.
///
/// # Type Parameters
///
/// - `K`: Key type, must be `Clone + Eq + Hash`
/// - `V`: Value type
///
/// # Example
///
/// ```
/// use cachekit::policy::clock::ClockCache;
/// use cachekit::traits::{CoreCache, ReadOnlyCache};
///
/// let mut cache = ClockCache::new(100);
///
/// cache.insert("key1", "value1");
/// cache.insert("key2", "value2");
///
/// assert_eq!(cache.get(&"key1"), Some(&"value1"));
/// assert_eq!(cache.len(), 2);
/// ```
pub struct ClockCache<K, V>
where
    K: Clone + Eq + Hash,
{
    ring: ClockRing<K, V>,
    #[cfg(feature = "metrics")]
    metrics: ClockMetrics,
}

impl<K, V> ClockCache<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Creates a new Clock cache with the specified capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::clock::ClockCache;
    /// use cachekit::traits::{CoreCache, ReadOnlyCache};
    ///
    /// let cache: ClockCache<String, i32> = ClockCache::new(100);
    /// assert_eq!(cache.capacity(), 100);
    /// assert!(cache.is_empty());
    /// ```
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Self {
            ring: ClockRing::new(capacity),
            #[cfg(feature = "metrics")]
            metrics: ClockMetrics::default(),
        }
    }

    /// Returns `true` if the cache is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    /// Returns the underlying [`ClockRing`] for advanced operations.
    ///
    /// This provides access to additional methods like `peek()`, `touch()`,
    /// `peek_victim()`, and `pop_victim()`.
    #[inline]
    pub fn as_ring(&self) -> &ClockRing<K, V> {
        &self.ring
    }

    /// Returns a mutable reference to the underlying [`ClockRing`].
    #[inline]
    pub fn as_ring_mut(&mut self) -> &mut ClockRing<K, V> {
        &mut self.ring
    }
}

impl<K, V> ReadOnlyCache<K, V> for ClockCache<K, V>
where
    K: Clone + Eq + Hash,
{
    fn contains(&self, key: &K) -> bool {
        self.ring.contains(key)
    }

    fn len(&self) -> usize {
        self.ring.len()
    }

    fn capacity(&self) -> usize {
        self.ring.capacity()
    }
}

impl<K, V> CoreCache<K, V> for ClockCache<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Inserts a key-value pair into the cache.
    ///
    /// If the key exists, updates the value and sets the reference bit.
    /// If at capacity, evicts using the clock algorithm.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::clock::ClockCache;
    /// use cachekit::traits::{CoreCache, ReadOnlyCache};
    ///
    /// let mut cache = ClockCache::new(2);
    /// cache.insert("a", 1);
    /// cache.insert("b", 2);
    ///
    /// // Update existing
    /// let old = cache.insert("a", 10);
    /// assert_eq!(old, Some(1));
    /// ```
    #[inline]
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        #[cfg(feature = "metrics")]
        {
            self.metrics.insert_calls += 1;
        }

        if self.ring.capacity() == 0 {
            return None;
        }
        if self.ring.contains(&key) {
            #[cfg(feature = "metrics")]
            {
                self.metrics.insert_updates += 1;
            }
            return self.ring.update(&key, value);
        }

        #[cfg(feature = "metrics")]
        {
            self.metrics.insert_new += 1;
        }

        #[cfg(feature = "metrics")]
        let need_evict = self.ring.len() >= self.ring.capacity();
        #[cfg(feature = "metrics")]
        let ha_before = self.ring.sweep_hand_advances();
        #[cfg(feature = "metrics")]
        let rb_before = self.ring.sweep_ref_bit_resets();

        let evicted = self.ring.insert(key, value);

        #[cfg(feature = "metrics")]
        if need_evict {
            self.metrics.evict_calls += 1;
            if evicted.is_some() {
                self.metrics.evicted_entries += 1;
            }
            self.metrics.hand_advances += self.ring.sweep_hand_advances() - ha_before;
            self.metrics.ref_bit_resets += self.ring.sweep_ref_bit_resets() - rb_before;
        }

        #[allow(unused_variables)]
        let _ = evicted;
        None
    }

    /// Gets a reference to the value for a key.
    ///
    /// Sets the reference bit on access (O(1) - no list operations!).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::clock::ClockCache;
    /// use cachekit::traits::{CoreCache, ReadOnlyCache};
    ///
    /// let mut cache = ClockCache::new(10);
    /// cache.insert("key", 42);
    ///
    /// // Access sets reference bit - this entry gets "second chance"
    /// assert_eq!(cache.get(&"key"), Some(&42));
    /// ```
    #[inline]
    fn get(&mut self, key: &K) -> Option<&V> {
        let result = self.ring.get(key);
        #[cfg(feature = "metrics")]
        if result.is_some() {
            self.metrics.get_calls += 1;
            self.metrics.get_hits += 1;
        } else {
            self.metrics.get_calls += 1;
            self.metrics.get_misses += 1;
        }
        result
    }

    /// Clears all entries from the cache.
    fn clear(&mut self) {
        self.ring.clear();
        #[cfg(feature = "metrics")]
        {
            use crate::metrics::traits::CoreMetricsRecorder;
            self.metrics.record_clear();
        }
    }
}

impl<K, V> MutableCache<K, V> for ClockCache<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Removes a key from the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::clock::ClockCache;
    /// use cachekit::traits::{CoreCache, MutableCache, ReadOnlyCache};
    ///
    /// let mut cache = ClockCache::new(10);
    /// cache.insert("key", 42);
    ///
    /// let removed = cache.remove(&"key");
    /// assert_eq!(removed, Some(42));
    /// assert!(!cache.contains(&"key"));
    /// ```
    #[inline]
    fn remove(&mut self, key: &K) -> Option<V> {
        self.ring.remove(key)
    }
}

#[cfg(feature = "metrics")]
impl<K, V> ClockCache<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Returns a snapshot of cache metrics.
    pub fn metrics_snapshot(&self) -> ClockMetricsSnapshot {
        ClockMetricsSnapshot {
            get_calls: self.metrics.get_calls,
            get_hits: self.metrics.get_hits,
            get_misses: self.metrics.get_misses,
            insert_calls: self.metrics.insert_calls,
            insert_updates: self.metrics.insert_updates,
            insert_new: self.metrics.insert_new,
            evict_calls: self.metrics.evict_calls,
            evicted_entries: self.metrics.evicted_entries,
            hand_advances: self.metrics.hand_advances,
            ref_bit_resets: self.metrics.ref_bit_resets,
            cache_len: self.ring.len(),
            capacity: self.ring.capacity(),
        }
    }
}

#[cfg(feature = "metrics")]
impl<K, V> MetricsSnapshotProvider<ClockMetricsSnapshot> for ClockCache<K, V>
where
    K: Clone + Eq + Hash,
{
    fn snapshot(&self) -> ClockMetricsSnapshot {
        self.metrics_snapshot()
    }
}

impl<K, V> std::fmt::Debug for ClockCache<K, V>
where
    K: Clone + Eq + Hash + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClockCache")
            .field("capacity", &self.ring.capacity())
            .field("len", &self.ring.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::MutableCache;

    #[allow(dead_code)]
    const _: () = {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        fn check() {
            assert_send::<ClockCache<String, i32>>();
            assert_sync::<ClockCache<String, i32>>();
        }
    };

    mod basic_operations {
        use super::*;

        #[test]
        fn test_new_cache() {
            let cache: ClockCache<i32, i32> = ClockCache::new(10);
            assert_eq!(cache.capacity(), 10);
            assert_eq!(cache.len(), 0);
            assert!(cache.is_empty());
        }

        #[test]
        fn test_insert_and_get() {
            let mut cache = ClockCache::new(10);
            cache.insert("a", 1);
            cache.insert("b", 2);

            assert_eq!(cache.get(&"a"), Some(&1));
            assert_eq!(cache.get(&"b"), Some(&2));
            assert_eq!(cache.get(&"c"), None);
        }

        #[test]
        fn test_insert_returns_old_value() {
            let mut cache = ClockCache::new(10);
            assert_eq!(cache.insert("a", 1), None);
            assert_eq!(cache.insert("a", 2), Some(1));
            assert_eq!(cache.get(&"a"), Some(&2));
        }

        #[test]
        fn test_contains() {
            let mut cache = ClockCache::new(10);
            cache.insert("a", 1);

            assert!(cache.contains(&"a"));
            assert!(!cache.contains(&"b"));
        }

        #[test]
        fn test_remove() {
            let mut cache = ClockCache::new(10);
            cache.insert("a", 1);
            cache.insert("b", 2);

            assert_eq!(cache.remove(&"a"), Some(1));
            assert!(!cache.contains(&"a"));
            assert_eq!(cache.len(), 1);

            assert_eq!(cache.remove(&"c"), None);
        }

        #[test]
        fn test_clear() {
            let mut cache = ClockCache::new(10);
            cache.insert("a", 1);
            cache.insert("b", 2);

            cache.clear();
            assert!(cache.is_empty());
            assert!(!cache.contains(&"a"));
        }
    }

    mod eviction {
        use super::*;

        #[test]
        fn test_eviction_at_capacity() {
            let mut cache = ClockCache::new(3);
            cache.insert("a", 1);
            cache.insert("b", 2);
            cache.insert("c", 3);

            assert_eq!(cache.len(), 3);

            // Insert fourth item - should evict one
            cache.insert("d", 4);
            assert_eq!(cache.len(), 3);

            // One of a, b, c should be evicted
            let count = [
                cache.contains(&"a"),
                cache.contains(&"b"),
                cache.contains(&"c"),
            ]
            .iter()
            .filter(|&&x| x)
            .count();
            assert_eq!(count, 2);
            assert!(cache.contains(&"d"));
        }

        #[test]
        fn test_second_chance() {
            let mut cache = ClockCache::new(3);
            cache.insert("a", 1);
            cache.insert("b", 2);
            cache.insert("c", 3);

            // Access "a" to set its reference bit
            cache.get(&"a");

            // Insert "d" - "a" should survive due to second chance
            cache.insert("d", 4);

            assert!(cache.contains(&"a"));
            assert!(cache.contains(&"d"));
            assert_eq!(cache.len(), 3);
        }

        #[test]
        fn test_all_referenced_eviction() {
            let mut cache = ClockCache::new(3);
            cache.insert("a", 1);
            cache.insert("b", 2);
            cache.insert("c", 3);

            // Access all to set reference bits
            cache.get(&"a");
            cache.get(&"b");
            cache.get(&"c");

            // Insert "d" - clock must sweep, clear all refs, then evict
            cache.insert("d", 4);
            assert_eq!(cache.len(), 3);
        }

        #[test]
        fn test_repeated_eviction() {
            let mut cache = ClockCache::new(2);

            for i in 0..100 {
                cache.insert(i, i * 10);
            }

            assert_eq!(cache.len(), 2);
        }
    }

    mod edge_cases {
        use super::*;

        #[test]
        fn test_capacity_one() {
            let mut cache = ClockCache::new(1);
            cache.insert("a", 1);
            assert_eq!(cache.get(&"a"), Some(&1));

            cache.insert("b", 2);
            assert!(!cache.contains(&"a"));
            assert!(cache.contains(&"b"));
        }

        #[test]
        fn test_zero_capacity_clamped() {
            let cache: ClockCache<i32, i32> = ClockCache::new(0);
            assert_eq!(cache.capacity(), 0);
        }

        #[test]
        fn test_string_keys() {
            let mut cache = ClockCache::new(10);
            cache.insert("hello".to_string(), 1);
            cache.insert("world".to_string(), 2);

            assert_eq!(cache.get(&"hello".to_string()), Some(&1));
        }

        #[test]
        fn test_large_capacity() {
            let mut cache = ClockCache::new(10000);
            for i in 0..5000 {
                cache.insert(i, i * 2);
            }
            assert_eq!(cache.len(), 5000);

            for i in 0..5000 {
                assert_eq!(cache.get(&i), Some(&(i * 2)));
            }
        }
    }

    mod ring_access {
        use super::*;

        #[test]
        fn test_as_ring() {
            let mut cache = ClockCache::new(10);
            cache.insert("a", 1);
            cache.insert("b", 2);

            // Access underlying ring for peek (no ref bit set)
            assert_eq!(cache.as_ring().peek(&"a"), Some(&1));

            // Use ring's touch method
            assert!(cache.as_ring_mut().touch(&"b"));
        }

        #[test]
        fn test_peek_vs_get() {
            let mut cache = ClockCache::new(2);
            cache.insert("a", 1);
            cache.insert("b", 2);

            // peek doesn't set reference bit
            let _ = cache.as_ring().peek(&"a");

            // Insert to trigger eviction - "a" should be evicted (no ref bit from peek)
            cache.insert("c", 3);

            // "a" was evicted because peek didn't set ref bit
            assert!(!cache.contains(&"a"));
            assert!(cache.contains(&"b"));
            assert!(cache.contains(&"c"));
        }
    }
}
