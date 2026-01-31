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
//! | Operation   | Time   | Notes                                      |
//! |-------------|--------|--------------------------------------------|
//! | `get`       | O(1)   | HashMap lookup, no reordering              |
//! | `insert`    | O(1)*  | *Amortized, may trigger eviction           |
//! | `contains`  | O(1)   | HashMap lookup only                        |
//! | `len`       | O(1)   | Returns total entries                      |
//! | `clear`     | O(n)   | Clears all structures                      |
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
//! // Insert items (pushed to stack)
//! cache.insert(1, 100);
//! cache.insert(2, 200);
//! cache.insert(3, 300);
//!
//! // Get doesn't affect eviction order (unlike LRU)
//! assert_eq!(cache.get(&1), Some(&100));
//!
//! // When cache is full, most recent insertion will be evicted!
//! for i in 4..=15 {
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

use crate::prelude::ReadOnlyCache;
use crate::traits::CoreCache;
use rustc_hash::FxHashMap;
use std::hash::Hash;

/// Core LIFO (Last In, First Out) cache implementation.
///
/// Implements stack-based eviction where the most recently inserted
/// entry is evicted first when capacity is reached.
///
/// # Type Parameters
///
/// - `K`: Key type, must be `Clone + Eq + Hash`
/// - `V`: Value type
///
/// # Example
///
/// ```
/// use cachekit::policy::lifo::LifoCore;
///
/// let mut cache = LifoCore::new(100);
///
/// // Insert items (pushed to stack)
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
/// When capacity is exceeded, evicts the most recently inserted entry (top of stack).
///
/// # Implementation
///
/// Uses Vec as stack + HashMap for O(1) operations.
pub struct LifoCore<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Maps key to value
    map: FxHashMap<K, V>,
    /// Stack of keys in insertion order (top = most recent)
    stack: Vec<K>,
    /// Maximum cache capacity
    capacity: usize,
}

impl<K, V> LifoCore<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Creates a new LIFO cache with the specified capacity.
    ///
    /// # Arguments
    ///
    /// - `capacity`: Maximum number of entries the cache can hold
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
        }
    }

    /// Retrieves a value by key without affecting eviction order.
    ///
    /// Unlike LRU, accessing an item in a LIFO cache doesn't change
    /// its position in the stack.
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
    pub fn get(&self, key: &K) -> Option<&V> {
        self.map.get(key)
    }

    /// Inserts or updates a key-value pair.
    ///
    /// - If the key exists, updates the value in place (no stack change)
    /// - If the key is new, pushes to top of stack
    /// - May trigger eviction from top of stack (most recent) if capacity is exceeded
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::lifo::LifoCore;
    ///
    /// let mut cache = LifoCore::new(100);
    ///
    /// // New insert pushes to stack
    /// cache.insert("key", "initial");
    /// assert_eq!(cache.len(), 1);
    ///
    /// // Update existing key (no stack change)
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

        // Check for existing key - update in place (no stack change)
        if let Some(v) = self.map.get_mut(&key) {
            *v = value;
            return;
        }

        // Evict from top of stack if at capacity
        self.evict_if_needed();

        // Push new entry to top of stack
        self.stack.push(key.clone());
        self.map.insert(key, value);
    }

    /// Evicts entries from top of stack until there is room.
    ///
    /// LIFO evicts from the top (most recently inserted).
    #[inline]
    fn evict_if_needed(&mut self) {
        while self.len() >= self.capacity && !self.stack.is_empty() {
            // Pop from top of stack (most recent)
            if let Some(key) = self.stack.pop() {
                self.map.remove(&key);
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
        self.map.clear();
        self.stack.clear();

        #[cfg(debug_assertions)]
        self.validate_invariants();
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

// Debug implementation
impl<K, V> std::fmt::Debug for LifoCore<K, V>
where
    K: Clone + Eq + Hash + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LifoCore")
            .field("capacity", &self.capacity)
            .field("len", &self.map.len())
            .field("stack_len", &self.stack.len())
            .finish_non_exhaustive()
    }
}

impl<K, V> ReadOnlyCache<K, V> for LifoCore<K, V>
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

/// Implementation of the [`CoreCache`] trait for LIFO.
///
/// Allows `LifoCore` to be used through the unified cache interface.
///
/// # Example
///
/// ```
/// use cachekit::traits::{CoreCache, ReadOnlyCache};
/// use cachekit::policy::lifo::LifoCore;
///
/// let mut cache: LifoCore<&str, i32> = LifoCore::new(100);
///
/// // Use via CoreCache trait
/// cache.insert("key", 42);
/// assert_eq!(cache.get(&"key"), Some(&42));
/// assert!(cache.contains(&"key"));
/// ```
impl<K, V> CoreCache<K, V> for LifoCore<K, V>
where
    K: Clone + Eq + Hash,
{
    #[inline]
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        // Check if key exists - update in place
        if let Some(v) = self.map.get_mut(&key) {
            return Some(std::mem::replace(v, value));
        }

        // New insert
        LifoCore::insert(self, key, value);
        None
    }

    #[inline]
    fn get(&mut self, key: &K) -> Option<&V> {
        LifoCore::get(self, key)
    }

    fn clear(&mut self) {
        LifoCore::clear(self);
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
            let cache: LifoCore<&str, i32> = LifoCore::new(100);

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
            let cache: LifoCore<i32, i32> = LifoCore::new(100);

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
    // Validation Tests
    // ==============================================

    #[test]
    #[cfg(debug_assertions)]
    fn validate_invariants_after_operations() {
        let mut cache = LifoCore::new(10);

        // Insert items
        for i in 1..=10 {
            cache.insert(i, i * 100);
        }
        cache.validate_invariants();

        // Access items (doesn't affect eviction in LIFO)
        for _ in 0..5 {
            cache.get(&5);
        }
        cache.validate_invariants();

        // Trigger evictions (evicts most recent)
        cache.insert(11, 1100);
        cache.validate_invariants();

        cache.insert(12, 1200);
        cache.validate_invariants();

        // Clear
        cache.clear();
        cache.validate_invariants();

        // Verify empty state
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

        // Multiple inserts to trigger LIFO evictions
        for i in 4..=10 {
            cache.insert(i, i * 100);
            cache.validate_invariants();
        }

        assert_eq!(cache.len(), 5);
        assert_eq!(cache.stack.len(), 5);

        // Verify all stack keys exist in map
        for key in &cache.stack {
            assert!(cache.map.contains_key(key));
        }
    }
}
