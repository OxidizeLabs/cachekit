//! Random cache replacement policy.
//!
//! Implements a random eviction algorithm where victims are selected uniformly
//! at random when capacity is reached. This provides a baseline policy with
//! minimal overhead and no access pattern tracking.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                        RandomCore<K, V> Layout                              │
//! │                                                                             │
//! │   ┌─────────────────────────────────────────────────────────────────────┐   │
//! │   │  map: HashMap<K, (usize, V)>   keys: Vec<K>                         │   │
//! │   │       key → (index, value)          dense array of keys             │   │
//! │   │                                                                     │   │
//! │   │  ┌──────────┬────────────┐          ┌─────┬─────┬─────┬─────┐       │   │
//! │   │  │   Key    │(idx, val)  │          │  0  │  1  │  2  │  3  │       │   │
//! │   │  ├──────────┼────────────┤          ├─────┼─────┼─────┼─────┤       │   │
//! │   │  │  "page1" │(0, v1)     │────┐     │ p1  │ p2  │ p3  │ p4  │       │   │
//! │   │  │  "page2" │(1, v2)     │────┼────►└─────┴─────┴─────┴─────┘       │   │
//! │   │  │  "page3" │(2, v3)     │────┘                                     │   │
//! │   │  │  "page4" │(3, v4)     │                                          │   │
//! │   │  └──────────┴────────────┘                                          │   │
//! │   └─────────────────────────────────────────────────────────────────────┘   │
//! │                                                                             │
//! │   ┌─────────────────────────────────────────────────────────────────────┐   │
//! │   │                    Random Eviction (O(1))                           │   │
//! │   │                                                                     │   │
//! │   │   1. Generate random index: i = rand() % len                        │   │
//! │   │   2. Get victim key: victim = keys[i]                               │   │
//! │   │   3. Swap with last: keys.swap(i, len-1)                            │   │
//! │   │   4. Update swapped key's index in map                              │   │
//! │   │   5. Pop last key: keys.pop()                                       │   │
//! │   │   6. Remove victim from map                                         │   │
//! │   │                                                                     │   │
//! │   │   Example: evict random from [A, B, C, D]                           │   │
//! │   │     - Random picks index 1 (B)                                      │   │
//! │   │     - Swap B with D: [A, D, C, B]                                   │   │
//! │   │     - Update D's index: 1                                           │   │
//! │   │     - Pop: [A, D, C]                                                │   │
//! │   │     - Remove B from map                                             │   │
//! │   └─────────────────────────────────────────────────────────────────────┘   │
//! │                                                                             │
//! └─────────────────────────────────────────────────────────────────────────────┘
//!
//! Insert Flow (new key)
//! ──────────────────────
//!
//!   insert("new_key", value):
//!     1. Check map - not found
//!     2. Evict random if at capacity
//!     3. Get next index: idx = keys.len()
//!     4. Push key to keys vec
//!     5. Insert (idx, value) into map
//!
//! Access Flow (existing key)
//! ──────────────────────────
//!
//!   get("existing_key"):
//!     1. Lookup (idx, value) in map
//!     2. Return &value (no reordering!)
//!
//! Eviction Flow
//! ─────────────
//!
//!   evict_random():
//!     1. Generate random index in range [0, len)
//!     2. Swap keys[random_idx] with keys[last]
//!     3. Update swapped key's index in map
//!     4. Pop last key
//!     5. Remove victim from map
//! ```
//!
//! ## Key Components
//!
//! - [`RandomCore`]: Main random eviction cache implementation
//!
//! ## Operations
//!
//! | Operation   | Time   | Notes                                      |
//! |-------------|--------|--------------------------------------------|
//! | `get`       | O(1)   | HashMap lookup, no reordering              |
//! | `insert`    | O(1)*  | *Amortized, may trigger random eviction    |
//! | `contains`  | O(1)   | HashMap lookup only                        |
//! | `len`       | O(1)   | Returns total entries                      |
//! | `clear`     | O(n)   | Clears all structures                      |
//!
//! ## Algorithm Properties
//!
//! - **No Access Pattern Tracking**: Zero overhead for access tracking
//! - **Uniform Random Selection**: All entries equally likely to be evicted
//! - **Unpredictable**: Can evict hot or cold entries
//! - **Baseline Policy**: Often used for comparison with smarter policies
//!
//! ## Use Cases
//!
//! - Performance baselines for benchmarks
//! - When access patterns are truly random
//! - When minimal overhead is critical
//! - Testing and debugging cache infrastructure
//!
//! ## Example Usage
//!
//! ```
//! use cachekit::policy::random::RandomCore;
//!
//! // Create random eviction cache with capacity 10
//! let mut cache = RandomCore::new(10);
//!
//! // Insert items
//! cache.insert(1, 100);
//! cache.insert(2, 200);
//! cache.insert(3, 300);
//!
//! // Get doesn't affect eviction probability (unlike LRU)
//! assert_eq!(cache.get(&1), Some(&100));
//!
//! // When cache is full, random entry will be evicted
//! for i in 4..=15 {
//!     cache.insert(i, i * 10);
//! }
//!
//! assert_eq!(cache.len(), 10);
//! ```
//!
//! ## Thread Safety
//!
//! - [`RandomCore`]: Not thread-safe, designed for single-threaded use
//! - For concurrent access, wrap in external synchronization
//!
//! ## Implementation Notes
//!
//! - Uses `Vec<K>` for O(1) random access and swap-remove
//! - Uses `HashMap<K, (usize, V)>` to store index and value
//! - RNG state managed internally using XorShift64 (Miri-compatible)
//! - No access pattern tracking = zero metadata overhead
//!
//! ## When to Use
//!
//! **Use Random when:**
//! - You need a baseline for benchmarking other policies
//! - Access patterns are truly random (no locality)
//! - Minimal overhead is more important than hit rate
//! - Testing cache infrastructure
//!
//! **Avoid Random when:**
//! - Access patterns have locality (use LRU, SLRU)
//! - Access patterns have frequency skew (use LFU)
//! - Scan resistance is needed (use S3-FIFO, LRU-K)
//! - Predictable performance is required
//!
//! ## References
//!
//! - Wikipedia: Cache replacement policies

use crate::prelude::ReadOnlyCache;
use crate::traits::CoreCache;
use rustc_hash::FxHashMap;
use std::hash::Hash;

/// Core Random eviction cache implementation.
///
/// Implements random victim selection using O(1) swap-remove technique.
/// Maintains a dense `Vec<K>` for random access and `HashMap<K, (usize, V)>`
/// for O(1) lookup and index tracking.
///
/// # Type Parameters
///
/// - `K`: Key type, must be `Clone + Eq + Hash`
/// - `V`: Value type
///
/// # Example
///
/// ```
/// use cachekit::policy::random::RandomCore;
///
/// let mut cache = RandomCore::new(100);
///
/// // Insert items
/// cache.insert("key1", "value1");
/// assert!(cache.contains(&"key1"));
///
/// // Get doesn't affect eviction (unlike LRU)
/// cache.get(&"key1");
///
/// // Update existing key
/// cache.insert("key1", "new_value");
/// assert_eq!(cache.get(&"key1"), Some(&"new_value"));
/// ```
///
/// # Eviction Behavior
///
/// When capacity is exceeded, evicts a uniformly random entry.
///
/// # Implementation
///
/// Uses Vec + HashMap for O(1) random eviction via swap-remove.
/// RNG state uses XorShift64 for deterministic testing and Miri compatibility.
pub struct RandomCore<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Maps key to (index in keys vec, value)
    map: FxHashMap<K, (usize, V)>,
    /// Dense array of keys for O(1) random access
    keys: Vec<K>,
    /// Maximum cache capacity
    capacity: usize,
    /// Internal PRNG state for random eviction (XorShift)
    rng_state: u64,
}

impl<K, V> RandomCore<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Creates a new random eviction cache with the specified capacity.
    ///
    /// # Arguments
    ///
    /// - `capacity`: Maximum number of entries the cache can hold
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::random::RandomCore;
    ///
    /// let cache: RandomCore<String, i32> = RandomCore::new(100);
    /// assert_eq!(cache.capacity(), 100);
    /// assert!(cache.is_empty());
    /// ```
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Self {
            map: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            keys: Vec::with_capacity(capacity),
            capacity,
            // Initialize with non-zero seed for XorShift (capacity + 1 ensures non-zero)
            rng_state: capacity as u64 + 0x9e3779b97f4a7c15,
        }
    }

    /// Retrieves a value by key without affecting eviction probability.
    ///
    /// Unlike LRU or LFU, accessing an item in a random cache doesn't change
    /// its eviction probability.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::random::RandomCore;
    ///
    /// let mut cache = RandomCore::new(100);
    /// cache.insert("key", 42);
    ///
    /// assert_eq!(cache.get(&"key"), Some(&42));
    /// assert_eq!(cache.get(&"missing"), None);
    /// ```
    #[inline]
    pub fn get(&self, key: &K) -> Option<&V> {
        self.map.get(key).map(|(_, v)| v)
    }

    /// Inserts or updates a key-value pair.
    ///
    /// - If the key exists, updates the value in place (no index change)
    /// - If the key is new, inserts at end of keys vec
    /// - May trigger random eviction if capacity is exceeded
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::random::RandomCore;
    ///
    /// let mut cache = RandomCore::new(100);
    ///
    /// // New insert
    /// cache.insert("key", "initial");
    /// assert_eq!(cache.len(), 1);
    ///
    /// // Update existing key
    /// cache.insert("key", "updated");
    /// assert_eq!(cache.get(&"key"), Some(&"updated"));
    /// assert_eq!(cache.len(), 1);  // Still 1 entry
    /// ```
    #[inline]
    pub fn insert(&mut self, key: K, value: V) {
        // Handle zero capacity - reject all insertions
        if self.capacity == 0 {
            return;
        }

        // Check for existing key - update in place
        if let Some((_, v)) = self.map.get_mut(&key) {
            *v = value;
            return;
        }

        // Evict random entry if at capacity
        self.evict_if_needed();

        // Insert new entry
        let idx = self.keys.len();
        self.keys.push(key.clone());
        self.map.insert(key, (idx, value));
    }

    /// Evicts a random entry if cache is at capacity.
    ///
    /// Uses simple random for O(1) selection with swap-remove technique.
    #[inline]
    fn evict_if_needed(&mut self) {
        while self.len() >= self.capacity && self.capacity > 0 && !self.keys.is_empty() {
            self.evict_random();
        }

        #[cfg(debug_assertions)]
        self.validate_invariants();
    }

    /// Evicts a uniformly random entry.
    ///
    /// Implementation:
    /// 1. Pick random index (using XorShift64)
    /// 2. Swap with last element
    /// 3. Update swapped element's index
    /// 4. Pop last element
    /// 5. Remove victim from map
    #[inline]
    fn evict_random(&mut self) {
        if self.keys.is_empty() {
            return;
        }

        let len = self.keys.len();
        // XorShift64 PRNG (fast and doesn't require system time)
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        let random_idx = (x as usize) % len;

        // Get the victim key
        let victim_key = self.keys[random_idx].clone();

        // Swap with last element if not already last
        let last_idx = len - 1;
        if random_idx != last_idx {
            self.keys.swap(random_idx, last_idx);

            // Update the swapped key's index in the map
            let swapped_key = &self.keys[random_idx];
            if let Some((idx, _)) = self.map.get_mut(swapped_key) {
                *idx = random_idx;
            }
        }

        // Pop the victim (now at the end)
        self.keys.pop();

        // Remove from map
        self.map.remove(&victim_key);
    }

    /// Returns the number of entries in the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::random::RandomCore;
    ///
    /// let mut cache = RandomCore::new(100);
    /// assert_eq!(cache.len(), 0);
    ///
    /// cache.insert("a", 1);
    /// cache.insert("b", 2);
    /// assert_eq!(cache.len(), 2);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the cache is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::random::RandomCore;
    ///
    /// let mut cache: RandomCore<&str, i32> = RandomCore::new(100);
    /// assert!(cache.is_empty());
    ///
    /// cache.insert("key", 42);
    /// assert!(!cache.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns the total cache capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::random::RandomCore;
    ///
    /// let cache: RandomCore<String, i32> = RandomCore::new(500);
    /// assert_eq!(cache.capacity(), 500);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns `true` if the key exists in the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::random::RandomCore;
    ///
    /// let mut cache = RandomCore::new(100);
    /// cache.insert("key", 42);
    ///
    /// assert!(cache.contains(&"key"));
    /// assert!(!cache.contains(&"missing"));
    /// ```
    #[inline]
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Clears all entries from the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::random::RandomCore;
    ///
    /// let mut cache = RandomCore::new(100);
    /// cache.insert("a", 1);
    /// cache.insert("b", 2);
    ///
    /// cache.clear();
    /// assert!(cache.is_empty());
    /// assert!(!cache.contains(&"a"));
    /// ```
    pub fn clear(&mut self) {
        self.map.clear();
        self.keys.clear();

        #[cfg(debug_assertions)]
        self.validate_invariants();
    }

    /// Validates internal data structure invariants.
    ///
    /// This method checks that:
    /// - Map size matches keys vector size
    /// - All keys in map have correct indices
    /// - All keys in vector exist in map
    /// - No duplicate keys in vector
    ///
    /// Only runs when debug assertions are enabled.
    #[cfg(debug_assertions)]
    fn validate_invariants(&self) {
        // Map and keys should have same size
        debug_assert_eq!(
            self.map.len(),
            self.keys.len(),
            "Map and keys vector have different sizes"
        );

        // All keys in map should have valid indices
        for (key, &(idx, _)) in &self.map {
            debug_assert!(idx < self.keys.len(), "Index out of bounds");
            debug_assert!(
                &self.keys[idx] == key,
                "Index mismatch: map points to wrong position"
            );
        }

        // All keys in vector should exist in map
        for (i, key) in self.keys.iter().enumerate() {
            debug_assert!(self.map.contains_key(key), "Key in vector not found in map");
            if let Some(&(idx, _)) = self.map.get(key) {
                debug_assert!(idx == i, "Vector position doesn't match map index");
            }
        }

        // No duplicates in keys vector
        let unique_count = self
            .keys
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        debug_assert!(unique_count == self.keys.len(), "Duplicate keys in vector");
    }
}

// Debug implementation
impl<K, V> std::fmt::Debug for RandomCore<K, V>
where
    K: Clone + Eq + Hash + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RandomCore")
            .field("capacity", &self.capacity)
            .field("len", &self.map.len())
            .finish_non_exhaustive()
    }
}

impl<K, V> ReadOnlyCache<K, V> for RandomCore<K, V>
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
}

/// Implementation of the [`CoreCache`](crate::traits::CoreCache) trait for Random.
///
/// Allows `RandomCore` to be used through the unified cache interface.
///
/// # Example
///
/// ```
/// use cachekit::traits::{CoreCache, ReadOnlyCache};
/// use cachekit::policy::random::RandomCore;
///
/// let mut cache: RandomCore<&str, i32> = RandomCore::new(100);
///
/// // Use via CoreCache trait
/// cache.insert("key", 42);
/// assert_eq!(cache.get(&"key"), Some(&42));
/// assert!(cache.contains(&"key"));
/// ```
impl<K, V> CoreCache<K, V> for RandomCore<K, V>
where
    K: Clone + Eq + Hash,
{
    #[inline]
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        // Check if key exists - update in place
        if let Some((_, v)) = self.map.get_mut(&key) {
            return Some(std::mem::replace(v, value));
        }

        // New insert
        RandomCore::insert(self, key, value);
        None
    }

    #[inline]
    fn get(&mut self, key: &K) -> Option<&V> {
        RandomCore::get(self, key)
    }

    fn clear(&mut self) {
        RandomCore::clear(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==============================================
    // RandomCore Basic Operations
    // ==============================================

    mod basic_operations {
        use super::*;

        #[test]
        fn new_cache_is_empty() {
            let cache: RandomCore<&str, i32> = RandomCore::new(100);
            assert!(cache.is_empty());
            assert_eq!(cache.len(), 0);
            assert_eq!(cache.capacity(), 100);
        }

        #[test]
        fn insert_and_get() {
            let mut cache = RandomCore::new(100);
            cache.insert("key1", "value1");

            assert_eq!(cache.len(), 1);
            assert_eq!(cache.get(&"key1"), Some(&"value1"));
        }

        #[test]
        fn insert_multiple_items() {
            let mut cache = RandomCore::new(100);
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
            let cache: RandomCore<&str, i32> = RandomCore::new(100);

            assert_eq!(cache.get(&"missing"), None);
        }

        #[test]
        fn update_existing_key() {
            let mut cache = RandomCore::new(100);
            cache.insert("key", "initial");
            cache.insert("key", "updated");

            assert_eq!(cache.len(), 1);
            assert_eq!(cache.get(&"key"), Some(&"updated"));
        }

        #[test]
        fn contains_returns_correct_result() {
            let mut cache = RandomCore::new(100);
            cache.insert("exists", 1);

            assert!(cache.contains(&"exists"));
            assert!(!cache.contains(&"missing"));
        }

        #[test]
        fn clear_removes_all_entries() {
            let mut cache = RandomCore::new(100);
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
            let cache: RandomCore<i32, i32> = RandomCore::new(500);
            assert_eq!(cache.capacity(), 500);
        }
    }

    // ==============================================
    // Random Eviction Behavior
    // ==============================================

    mod eviction_behavior {
        use super::*;

        #[test]
        fn eviction_occurs_when_over_capacity() {
            let mut cache = RandomCore::new(5);

            for i in 0..10 {
                cache.insert(i, i * 10);
            }

            assert_eq!(cache.len(), 5);
        }

        #[test]
        fn eviction_maintains_capacity() {
            let mut cache = RandomCore::new(3);

            cache.insert(1, 10);
            cache.insert(2, 20);
            cache.insert(3, 30);
            assert_eq!(cache.len(), 3);

            cache.insert(4, 40);
            assert_eq!(cache.len(), 3);

            cache.insert(5, 50);
            assert_eq!(cache.len(), 3);
        }

        #[test]
        fn eviction_removes_from_index() {
            let mut cache = RandomCore::new(3);

            cache.insert(1, 10);
            cache.insert(2, 20);
            cache.insert(3, 30);

            assert_eq!(cache.len(), 3);

            // Insert more items - some will be evicted
            for i in 4..15 {
                cache.insert(i, i * 10);
            }

            assert_eq!(cache.len(), 3);
        }

        #[test]
        fn random_eviction_is_unpredictable() {
            // This test verifies that eviction happens but doesn't assert
            // on which specific keys are evicted (since it's random)
            let mut cache = RandomCore::new(5);

            for i in 0..5 {
                cache.insert(i, i * 10);
            }

            // All 5 should be present
            assert_eq!(cache.len(), 5);

            // Insert 10 more - 10 random items will be evicted
            for i in 5..15 {
                cache.insert(i, i * 10);
            }

            // Should still be at capacity
            assert_eq!(cache.len(), 5);

            // Count how many of the original items remain
            let original_remaining = (0..5).filter(|i| cache.contains(i)).count();

            // Some should remain, some should be evicted (probabilistically)
            // With 5 capacity and 15 inserts, statistically ~1-2 originals remain
            // But we can't assert exact count due to randomness
            assert!(original_remaining <= 5);
        }
    }

    // ==============================================
    // Get Does Not Affect Eviction
    // ==============================================

    mod get_behavior {
        use super::*;

        #[test]
        fn get_does_not_change_eviction_probability() {
            let mut cache = RandomCore::new(5);

            // Insert 5 items
            for i in 0..5 {
                cache.insert(i, i * 10);
            }

            // Access item 0 many times (would move to MRU in LRU cache)
            for _ in 0..100 {
                cache.get(&0);
            }

            // In LRU, item 0 would be protected
            // In Random, item 0 has same eviction probability as others
            // We can't test this deterministically, but we verify behavior is consistent
            assert!(cache.contains(&0));
            assert_eq!(cache.len(), 5);
        }
    }

    // ==============================================
    // Edge Cases
    // ==============================================

    mod edge_cases {
        use super::*;

        #[test]
        fn single_capacity_cache() {
            let mut cache = RandomCore::new(1);

            cache.insert("a", 1);
            assert_eq!(cache.get(&"a"), Some(&1));

            cache.insert("b", 2);
            // One of them should be evicted
            assert_eq!(cache.len(), 1);
        }

        #[test]
        fn zero_capacity_cache() {
            let mut cache = RandomCore::new(0);

            cache.insert("a", 1);
            assert_eq!(cache.len(), 0);
            assert!(!cache.contains(&"a"));
        }

        #[test]
        fn get_after_update() {
            let mut cache = RandomCore::new(100);

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
            let mut cache = RandomCore::new(10000);

            for i in 0..10000 {
                cache.insert(i, i * 2);
            }

            assert_eq!(cache.len(), 10000);

            assert_eq!(cache.get(&5000), Some(&10000));
            assert_eq!(cache.get(&9999), Some(&19998));
        }

        #[test]
        fn empty_cache_operations() {
            let cache: RandomCore<i32, i32> = RandomCore::new(100);

            assert!(cache.is_empty());
            assert_eq!(cache.get(&1), None);
            assert!(!cache.contains(&1));
        }

        #[test]
        fn string_keys_and_values() {
            let mut cache = RandomCore::new(100);

            cache.insert(String::from("hello"), String::from("world"));
            cache.insert(String::from("foo"), String::from("bar"));

            assert_eq!(
                cache.get(&String::from("hello")),
                Some(&String::from("world"))
            );
            assert_eq!(cache.get(&String::from("foo")), Some(&String::from("bar")));
        }

        #[test]
        fn internal_consistency_after_evictions() {
            let mut cache = RandomCore::new(10);

            // Insert many items to trigger multiple evictions
            for i in 0..100 {
                cache.insert(i, i * 10);
            }

            assert_eq!(cache.len(), 10);
            assert_eq!(cache.keys.len(), 10);
            assert_eq!(cache.map.len(), 10);

            // Verify all keys in vec are in map and indices match
            for (idx, key) in cache.keys.iter().enumerate() {
                assert!(cache.map.contains_key(key));
                let (stored_idx, _) = cache.map.get(key).unwrap();
                assert_eq!(*stored_idx, idx);
            }
        }
    }

    // ==============================================
    // Baseline Comparison Properties
    // ==============================================

    mod baseline_properties {
        use super::*;

        #[test]
        fn no_access_pattern_tracking() {
            let mut cache = RandomCore::new(10);

            // Insert items
            for i in 0..10 {
                cache.insert(i, i * 10);
            }

            // Access some items multiple times
            for _ in 0..100 {
                cache.get(&0);
                cache.get(&1);
            }

            // Random policy doesn't track this, so all items have equal probability
            // Just verify cache state is consistent
            assert_eq!(cache.len(), 10);
            assert!(cache.contains(&0));
            assert!(cache.contains(&1));
        }

        #[test]
        fn works_as_baseline_policy() {
            // Random provides a baseline: any smarter policy should beat it
            // on workloads with temporal or frequency locality
            let mut cache = RandomCore::new(5);

            for i in 0..10 {
                cache.insert(i, i);
            }

            // Verify it maintains capacity
            assert_eq!(cache.len(), 5);

            // Verify it can store and retrieve
            let count = (0..10).filter(|i| cache.contains(i)).count();
            assert_eq!(count, 5);
        }
    }

    // ==============================================
    // Validation Tests
    // ==============================================

    #[test]
    #[cfg(debug_assertions)]
    fn validate_invariants_after_operations() {
        let mut cache = RandomCore::new(10);

        // Insert items
        for i in 1..=10 {
            cache.insert(i, i * 100);
        }
        cache.validate_invariants();

        // Access items (doesn't affect eviction)
        for _ in 0..5 {
            cache.get(&5);
        }
        cache.validate_invariants();

        // Trigger evictions
        cache.insert(11, 1100);
        cache.validate_invariants();

        cache.insert(12, 1200);
        cache.validate_invariants();

        // Clear
        cache.clear();
        cache.validate_invariants();

        // Verify empty state
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.keys.len(), 0);
    }

    #[test]
    #[cfg(debug_assertions)]
    fn validate_invariants_with_index_consistency() {
        let mut cache = RandomCore::new(5);
        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.insert(3, 300);
        cache.validate_invariants();

        // Multiple inserts to trigger evictions and index updates
        for i in 4..=10 {
            cache.insert(i, i * 100);
            cache.validate_invariants();
        }

        assert_eq!(cache.len(), 5);
        assert_eq!(cache.keys.len(), 5);

        // Verify all indices are correct
        for (key, &(idx, _)) in &cache.map {
            assert_eq!(&cache.keys[idx], key);
        }
    }
}
