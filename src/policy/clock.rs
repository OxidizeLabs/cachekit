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
//! │   │  entries: Vec<Option<Entry<K,V>>>   (circular buffer)               │   │
//! │   │                                                                     │   │
//! │   │    [0]     [1]     [2]     [3]     [4]     [5]     [6]     [7]      │   │
//! │   │   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐    │   │
//! │   │   │ A │   │ B │   │ C │   │ D │   │ E │   │   │   │   │   │   │    │   │
//! │   │   │ref│   │ref│   │   │   │ref│   │   │   │   │   │   │   │   │    │   │
//! │   │   └───┘   └───┘   └───┘   └───┘   └───┘   └───┘   └───┘   └───┘    │   │
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
//!     entry = entries[hand]
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
//! | Operation | Time    | Notes                              |
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
//! use cachekit::traits::CoreCache;
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

use rustc_hash::FxHashMap;
use std::hash::Hash;

use crate::traits::CoreCache;

/// Entry in the clock buffer.
#[derive(Debug)]
struct Entry<K, V> {
    key: K,
    value: V,
    referenced: bool,
}

/// High-performance Clock cache with O(1) access operations.
///
/// Uses a circular buffer with a sweeping clock hand for eviction.
/// Approximates LRU without the linked list overhead.
pub struct ClockCache<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Maps keys to their slot index in the entries buffer.
    index: FxHashMap<K, usize>,
    /// Circular buffer of entries.
    entries: Vec<Option<Entry<K, V>>>,
    /// Current position of the clock hand.
    hand: usize,
    /// Number of occupied slots.
    len: usize,
    /// Maximum capacity.
    capacity: usize,
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
    ///
    /// let cache: ClockCache<String, i32> = ClockCache::new(100);
    /// assert_eq!(cache.capacity(), 100);
    /// assert!(cache.is_empty());
    /// ```
    #[inline]
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        let mut entries = Vec::with_capacity(capacity);
        entries.resize_with(capacity, || None);

        Self {
            index: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            entries,
            hand: 0,
            len: 0,
            capacity,
        }
    }

    /// Returns `true` if the cache is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Finds an empty slot, evicting if necessary.
    ///
    /// Returns the index of the available slot.
    #[inline]
    fn find_slot(&mut self) -> usize {
        // If not at capacity, find first empty slot
        if self.len < self.capacity {
            for i in 0..self.capacity {
                if self.entries[i].is_none() {
                    return i;
                }
            }
        }

        // At capacity - need to evict using clock algorithm
        self.evict()
    }

    /// Runs the clock eviction algorithm.
    ///
    /// Sweeps from current hand position, giving referenced entries
    /// a "second chance" by clearing their reference bit.
    /// Returns the index of the evicted slot.
    #[inline]
    fn evict(&mut self) -> usize {
        loop {
            if let Some(entry) = &mut self.entries[self.hand] {
                if entry.referenced {
                    // Give second chance - clear reference bit
                    entry.referenced = false;
                } else {
                    // Not referenced - evict this entry
                    let slot = self.hand;
                    self.index.remove(&entry.key);
                    self.entries[slot] = None;
                    self.len -= 1;
                    self.hand = (self.hand + 1) % self.capacity;
                    return slot;
                }
            }
            self.hand = (self.hand + 1) % self.capacity;
        }
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
    /// use cachekit::traits::CoreCache;
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
        if self.capacity == 0 {
            return None;
        }

        // Check if key exists - update in place
        if let Some(&slot) = self.index.get(&key) {
            let entry = self.entries[slot].as_mut().unwrap();
            let old = std::mem::replace(&mut entry.value, value);
            entry.referenced = true;
            return Some(old);
        }

        // Find slot (may evict)
        let slot = self.find_slot();

        // Insert new entry
        self.entries[slot] = Some(Entry {
            key: key.clone(),
            value,
            referenced: false, // New entries start unreferenced
        });
        self.index.insert(key, slot);
        self.len += 1;

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
    /// use cachekit::traits::CoreCache;
    ///
    /// let mut cache = ClockCache::new(10);
    /// cache.insert("key", 42);
    ///
    /// // Access sets reference bit - this entry gets "second chance"
    /// assert_eq!(cache.get(&"key"), Some(&42));
    /// ```
    #[inline]
    fn get(&mut self, key: &K) -> Option<&V> {
        let slot = *self.index.get(key)?;
        let entry = self.entries[slot].as_mut()?;
        entry.referenced = true; // This is the entire cost of "LRU" tracking!
        Some(&entry.value)
    }

    /// Returns `true` if the cache contains the key.
    ///
    /// Does not affect the reference bit.
    #[inline]
    fn contains(&self, key: &K) -> bool {
        self.index.contains_key(key)
    }

    /// Returns the number of entries in the cache.
    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    /// Returns the maximum capacity of the cache.
    #[inline]
    fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clears all entries from the cache.
    fn clear(&mut self) {
        self.index.clear();
        for entry in &mut self.entries {
            *entry = None;
        }
        self.len = 0;
        self.hand = 0;
    }
}

impl<K, V> crate::traits::MutableCache<K, V> for ClockCache<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Removes a key from the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::clock::ClockCache;
    /// use cachekit::traits::{CoreCache, MutableCache};
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
        let slot = self.index.remove(key)?;
        let entry = self.entries[slot].take()?;
        self.len -= 1;
        Some(entry.value)
    }
}

// Debug implementation
impl<K, V> std::fmt::Debug for ClockCache<K, V>
where
    K: Clone + Eq + Hash + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClockCache")
            .field("capacity", &self.capacity)
            .field("len", &self.len)
            .field("hand", &self.hand)
            .finish_non_exhaustive()
    }
}

// SAFETY: ClockCache can be sent between threads if K and V are Send.
unsafe impl<K, V> Send for ClockCache<K, V>
where
    K: Clone + Eq + Hash + Send,
    V: Send,
{
}

// SAFETY: ClockCache can be shared between threads if K and V are Sync.
unsafe impl<K, V> Sync for ClockCache<K, V>
where
    K: Clone + Eq + Hash + Sync,
    V: Sync,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::MutableCache;

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

            // "a" should be evicted (first unreferenced)
            assert!(!cache.contains(&"a"));
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

            // Insert "d" - should evict "b" (first unreferenced after "a")
            cache.insert("d", 4);

            // "a" got second chance, "b" was evicted
            assert!(cache.contains(&"a"));
            assert!(!cache.contains(&"b"));
            assert!(cache.contains(&"c"));
            assert!(cache.contains(&"d"));
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
            assert_eq!(cache.capacity(), 1); // Clamped to 1
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

    mod clock_behavior {
        use super::*;

        #[test]
        fn test_hand_advances() {
            let mut cache = ClockCache::new(5);

            // Fill cache
            for i in 0..5 {
                cache.insert(i, i);
            }

            // Hand should be at 0
            let initial_hand = cache.hand;

            // Evict one - hand should advance
            cache.insert(100, 100);

            // Hand moved past the evicted entry
            assert_ne!(cache.hand, initial_hand);
        }

        #[test]
        fn test_reference_bit_cleared_on_sweep() {
            let mut cache = ClockCache::new(3);
            cache.insert(1, 1);
            cache.insert(2, 2);
            cache.insert(3, 3);

            // Reference item 1
            cache.get(&1);

            // Evict - should clear ref bit of 1 and evict 2
            cache.insert(4, 4);

            // Now reference 3
            cache.get(&3);

            // Evict again - 1's ref was cleared, so 1 should be evicted
            cache.insert(5, 5);

            assert!(!cache.contains(&1)); // Was evicted this time
            assert!(cache.contains(&3)); // Still has ref bit
        }
    }
}
