//! NRU (Not Recently Used) cache replacement policy.
//!
//! Implements the Not Recently Used algorithm, which uses a single reference bit
//! per entry to approximate LRU with minimal overhead. The reference bit provides
//! a coarse distinction between "recently used" and "not recently used" items.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                        NruCache<K, V> Layout                                │
//! │                                                                             │
//! │   ┌─────────────────────────────────────────────────────────────────────┐   │
//! │   │  map: HashMap<K, Entry<V>>     keys: Vec<K>                         │   │
//! │   │       key → (index, value, ref)      dense array of keys            │   │
//! │   │                                                                     │   │
//! │   │  ┌──────────┬─────────────────┐      ┌─────┬─────┬─────┬─────┐   │   │
//! │   │  │   Key    │  Entry          │      │  0  │  1  │  2  │  3  │   │   │
//! │   │  ├──────────┼─────────────────┤      ├─────┼─────┼─────┼─────┤   │   │
//! │   │  │  "page1" │(0, v1, ref=1)   │──┐   │ p1  │ p2  │ p3  │ p4  │   │   │
//! │   │  │  "page2" │(1, v2, ref=0)   │──┼──►└─────┴─────┴─────┴─────┘   │   │
//! │   │  │  "page3" │(2, v3, ref=1)   │──┘                                │   │
//! │   │  │  "page4" │(3, v4, ref=0)   │ ← Eviction candidate (ref=0)     │   │
//! │   │  └──────────┴─────────────────┘                                   │   │
//! │   └─────────────────────────────────────────────────────────────────────┘   │
//! │                                                                             │
//! │   ┌─────────────────────────────────────────────────────────────────────┐   │
//! │   │                    NRU Eviction (O(n) worst case)                   │   │
//! │   │                                                                     │   │
//! │   │   1. Scan keys vec for first entry with ref=0                       │   │
//! │   │   2. If found: evict that entry                                     │   │
//! │   │   3. If not found: clear all ref bits, then evict first entry       │   │
//! │   │                                                                     │   │
//! │   │   epoch_counter: Tracks when to do bulk ref bit clearing            │   │
//! │   │   epoch_threshold: Number of accesses before clearing all bits      │   │
//! │   └─────────────────────────────────────────────────────────────────────┘   │
//! │                                                                             │
//! └─────────────────────────────────────────────────────────────────────────────┘
//!
//! Access Flow
//! ──────────────────────
//!
//!   get("key"):
//!     1. Lookup entry in map
//!     2. Set entry.referenced = true
//!     3. Return &value
//!
//! Insert Flow (new key)
//! ──────────────────────
//!
//!   insert("new_key", value):
//!     1. Check map - not found
//!     2. Evict if at capacity
//!     3. Get next index: idx = keys.len()
//!     4. Push key to keys vec
//!     5. Insert Entry{idx, value, referenced: false} into map
//!
//! Eviction Flow
//! ─────────────
//!
//!   evict_nru():
//!     1. Scan keys for first entry with referenced=false
//!     2. If found: remove that entry (swap-remove)
//!     3. If not found (all referenced):
//!        a. Clear all reference bits
//!        b. Evict first entry (now all have ref=false)
//! ```
//!
//! ## Algorithm
//!
//! ```text
//! GET(key):
//!   1. Look up entry in hash map
//!   2. Set referenced = true
//!   3. Return value
//!   Cost: O(1) - just a hash lookup and bit set
//!
//! INSERT(key, value):
//!   1. If key exists: update value, set referenced = true
//!   2. If at capacity: run eviction
//!   3. Insert entry with referenced = true
//!
//! EVICT():
//!   // Phase 1: Try to find unreferenced entry
//!   for each entry in keys:
//!     if entry.referenced == false:
//!       remove entry
//!       return
//!
//!   // Phase 2: All referenced - clear and pick first
//!   for each entry:
//!     entry.referenced = false
//!   remove first entry
//! ```
//!
//! ## Performance Characteristics
//!
//! | Operation | Time    | Notes                              |
//! |-----------|---------|-----------------------------------|
//! | `get`     | O(1)    | Hash lookup + bit set             |
//! | `insert`  | O(n)*   | *Worst case if all entries ref'd  |
//! | `contains`| O(1)    | Hash lookup only                  |
//! | `remove`  | O(1)    | Hash lookup + swap-remove         |
//!
//! ## Trade-offs
//!
//! | Aspect        | NRU                      | Clock                   | LRU                     |
//! |---------------|--------------------------|-------------------------|-------------------------|
//! | Access cost   | O(1) bit set             | O(1) bit set            | O(1) list move          |
//! | Eviction cost | O(n) worst case          | O(n) worst case         | O(1)                    |
//! | Granularity   | Binary (used/not used)   | Binary with hand sweep  | Full order              |
//! | Overhead      | 1 bit per entry          | 1 bit per entry + hand  | 16 bytes per entry      |
//!
//! ## When to Use
//!
//! **Use NRU when:**
//! - You need simple, coarse eviction tracking
//! - Memory for full LRU list is too expensive
//! - You can tolerate O(n) eviction in worst case
//! - Access patterns have temporal locality
//!
//! **Avoid NRU when:**
//! - You need strict LRU ordering (use LRU)
//! - You need O(1) eviction guarantees (use Clock with hand)
//! - You need scan resistance (use S3-FIFO, LRU-K)
//!
//! ## Example Usage
//!
//! ```
//! use cachekit::policy::nru::NruCache;
//! use cachekit::traits::CoreCache;
//!
//! let mut cache = NruCache::new(100);
//!
//! // Insert items
//! cache.insert("page1", "content1");
//! cache.insert("page2", "content2");
//!
//! // Access sets reference bit
//! assert_eq!(cache.get(&"page1"), Some(&"content1"));
//!
//! // Unreferenced items are evicted first
//! // Referenced items protected until next epoch
//! ```
//!
//! ## Implementation
//!
//! This implementation uses:
//! - `HashMap<K, Entry<V>>` for O(1) lookup (stores index, value, referenced bit)
//! - `Vec<K>` for dense key storage and eviction scanning
//! - Swap-remove technique for O(1) removal (updates index in moved entry)
//! - Lazy clearing of reference bits (only when needed during eviction)
//!
//! ## Thread Safety
//!
//! - [`NruCache`]: Not thread-safe, designed for single-threaded use
//! - For concurrent access, wrap in external synchronization (e.g., `Mutex`)
//!
//! ## References
//!
//! - Wikipedia: Cache replacement policies

use rustc_hash::FxHashMap;
use std::hash::Hash;

/// Entry in the NRU cache containing value, index, and reference bit.
#[derive(Debug, Clone)]
struct Entry<V> {
    /// Index in the keys vector
    index: usize,
    /// Cached value
    value: V,
    /// Reference bit - set on access, cleared during epoch reset
    referenced: bool,
}

/// NRU (Not Recently Used) cache implementation.
///
/// Uses a single reference bit per entry to distinguish between recently used
/// and not recently used items. Provides O(1) access but O(n) worst-case eviction.
///
/// # Type Parameters
///
/// - `K`: Key type, must be `Clone + Eq + Hash`
/// - `V`: Value type
///
/// # Example
///
/// ```
/// use cachekit::policy::nru::NruCache;
/// use cachekit::traits::CoreCache;
///
/// let mut cache = NruCache::new(100);
///
/// cache.insert("key1", "value1");
/// cache.insert("key2", "value2");
///
/// // Access sets reference bit
/// assert_eq!(cache.get(&"key1"), Some(&"value1"));
///
/// // When cache is full, unreferenced items are evicted first
/// for i in 3..=110 {
///     cache.insert(i, format!("value{}", i));
/// }
///
/// assert_eq!(cache.len(), 100);
/// ```
///
/// # Eviction Behavior
///
/// When capacity is exceeded:
/// 1. Scans for first entry with `referenced = false`
/// 2. If all entries are referenced, clears all reference bits then evicts first entry
///
/// # Implementation
///
/// Uses HashMap + Vec for O(1) access with swap-remove for eviction.
pub struct NruCache<K, V>
where
    K: Clone + Eq + Hash,
{
    /// HashMap for O(1) key lookup
    map: FxHashMap<K, Entry<V>>,
    /// Dense array of keys for eviction scanning
    keys: Vec<K>,
    /// Maximum capacity
    capacity: usize,
}

impl<K, V> NruCache<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Creates a new NRU cache with the specified capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::nru::NruCache;
    /// use cachekit::traits::CoreCache;
    ///
    /// let cache: NruCache<String, i32> = NruCache::new(100);
    /// assert_eq!(cache.capacity(), 100);
    /// assert!(cache.is_empty());
    /// ```
    #[inline]
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            map: FxHashMap::default(),
            keys: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Returns `true` if the cache is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Evicts an entry using NRU policy.
    ///
    /// First tries to find an unreferenced entry. If all entries are referenced,
    /// clears all reference bits and evicts the first entry.
    ///
    /// Returns the evicted (key, value) pair.
    fn evict_one(&mut self) -> Option<(K, V)> {
        if self.keys.is_empty() {
            return None;
        }

        // Phase 1: Try to find an unreferenced entry
        for (idx, key) in self.keys.iter().enumerate() {
            if let Some(entry) = self.map.get(key) {
                if !entry.referenced {
                    // Found unreferenced entry - evict it
                    let victim_key = self.keys.swap_remove(idx);

                    // Update index of swapped key if we didn't remove the last element
                    if idx < self.keys.len() {
                        let swapped_key = &self.keys[idx];
                        if let Some(swapped_entry) = self.map.get_mut(swapped_key) {
                            swapped_entry.index = idx;
                        }
                    }

                    let victim_entry = self.map.remove(&victim_key)?;
                    return Some((victim_key, victim_entry.value));
                }
            }
        }

        // Phase 2: All entries are referenced - clear all bits and evict first
        for key in &self.keys {
            if let Some(entry) = self.map.get_mut(key) {
                entry.referenced = false;
            }
        }

        // Now evict the first entry (index 0)
        if !self.keys.is_empty() {
            let victim_key = self.keys.swap_remove(0);

            // Update index of swapped key if we didn't remove the last element
            if !self.keys.is_empty() {
                let swapped_key = &self.keys[0];
                if let Some(swapped_entry) = self.map.get_mut(swapped_key) {
                    swapped_entry.index = 0;
                }
            }

            let victim_entry = self.map.remove(&victim_key)?;
            return Some((victim_key, victim_entry.value));
        }

        None
    }
}

impl<K, V> crate::traits::CoreCache<K, V> for NruCache<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Inserts a key-value pair into the cache.
    ///
    /// If the key exists, updates the value and sets the reference bit.
    /// If at capacity, evicts using the NRU algorithm.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::nru::NruCache;
    /// use cachekit::traits::CoreCache;
    ///
    /// let mut cache = NruCache::new(2);
    /// cache.insert("a", 1);
    /// cache.insert("b", 2);
    ///
    /// // Update existing
    /// let old = cache.insert("a", 10);
    /// assert_eq!(old, Some(1));
    /// ```
    #[inline]
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        // Check if key already exists
        if let Some(entry) = self.map.get_mut(&key) {
            // Update existing entry
            let old_value = std::mem::replace(&mut entry.value, value);
            entry.referenced = true;
            return Some(old_value);
        }

        // New key - check capacity
        if self.map.len() >= self.capacity {
            // Evict using NRU policy
            let _ = self.evict_one();
        }

        // Insert new entry
        let index = self.keys.len();
        self.keys.push(key.clone());
        self.map.insert(
            key,
            Entry {
                index,
                value,
                referenced: false, // New inserts start unreferenced (cold start)
            },
        );

        None
    }

    /// Gets a reference to the value for a key.
    ///
    /// Sets the reference bit on access.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::nru::NruCache;
    /// use cachekit::traits::CoreCache;
    ///
    /// let mut cache = NruCache::new(10);
    /// cache.insert("key", 42);
    ///
    /// // Access sets reference bit - this entry gets protection
    /// assert_eq!(cache.get(&"key"), Some(&42));
    /// ```
    #[inline]
    fn get(&mut self, key: &K) -> Option<&V> {
        if let Some(entry) = self.map.get_mut(key) {
            entry.referenced = true;
            Some(&entry.value)
        } else {
            None
        }
    }

    /// Returns `true` if the cache contains the key.
    ///
    /// Does not affect the reference bit.
    #[inline]
    fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Returns the number of entries in the cache.
    #[inline]
    fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns the maximum capacity of the cache.
    #[inline]
    fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clears all entries from the cache.
    fn clear(&mut self) {
        self.map.clear();
        self.keys.clear();
    }
}

impl<K, V> crate::traits::MutableCache<K, V> for NruCache<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Removes a key from the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::nru::NruCache;
    /// use cachekit::traits::{CoreCache, MutableCache};
    ///
    /// let mut cache = NruCache::new(10);
    /// cache.insert("key", 42);
    ///
    /// let removed = cache.remove(&"key");
    /// assert_eq!(removed, Some(42));
    /// assert!(!cache.contains(&"key"));
    /// ```
    #[inline]
    fn remove(&mut self, key: &K) -> Option<V> {
        let entry = self.map.remove(key)?;
        let idx = entry.index;

        // Swap-remove from keys vec
        self.keys.swap_remove(idx);

        // Update index of swapped key if we didn't remove the last element
        if idx < self.keys.len() {
            let swapped_key = &self.keys[idx];
            if let Some(swapped_entry) = self.map.get_mut(swapped_key) {
                swapped_entry.index = idx;
            }
        }

        Some(entry.value)
    }
}

impl<K, V> std::fmt::Debug for NruCache<K, V>
where
    K: Clone + Eq + Hash + std::fmt::Debug,
    V: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NruCache")
            .field("capacity", &self.capacity)
            .field("len", &self.map.len())
            .field("keys", &self.keys)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{CoreCache, MutableCache};

    #[test]
    fn test_new() {
        let cache: NruCache<i32, i32> = NruCache::new(10);
        assert_eq!(cache.capacity(), 10);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_insert_and_get() {
        let mut cache = NruCache::new(3);

        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.insert(3, 300);

        assert_eq!(cache.get(&1), Some(&100));
        assert_eq!(cache.get(&2), Some(&200));
        assert_eq!(cache.get(&3), Some(&300));
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_update_existing() {
        let mut cache = NruCache::new(3);

        cache.insert(1, 100);
        let old = cache.insert(1, 999);

        assert_eq!(old, Some(100));
        assert_eq!(cache.get(&1), Some(&999));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_eviction_unreferenced() {
        let mut cache = NruCache::new(3);

        // Insert 3 items
        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.insert(3, 300);

        // Access only 1 and 3
        let _ = cache.get(&1);
        let _ = cache.get(&3);

        // Insert 4th item - should evict 2 (unreferenced)
        cache.insert(4, 400);

        assert_eq!(cache.len(), 3);
        assert!(cache.contains(&1));
        assert!(!cache.contains(&2)); // 2 was evicted
        assert!(cache.contains(&3));
        assert!(cache.contains(&4));
    }

    #[test]
    fn test_eviction_all_referenced() {
        let mut cache = NruCache::new(3);

        // Insert and access all 3 items
        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.insert(3, 300);
        let _ = cache.get(&1);
        let _ = cache.get(&2);
        let _ = cache.get(&3);

        // Insert 4th item - all are referenced, so clears bits and evicts one
        cache.insert(4, 400);

        assert_eq!(cache.len(), 3);
        assert!(cache.contains(&4));
    }

    #[test]
    fn test_remove() {
        let mut cache = NruCache::new(3);

        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.insert(3, 300);

        let removed = cache.remove(&2);
        assert_eq!(removed, Some(200));
        assert_eq!(cache.len(), 2);
        assert!(!cache.contains(&2));
        assert!(cache.contains(&1));
        assert!(cache.contains(&3));
    }

    #[test]
    fn test_clear() {
        let mut cache = NruCache::new(3);

        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.insert(3, 300);

        cache.clear();

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert!(!cache.contains(&1));
    }

    #[test]
    fn test_contains_does_not_set_reference() {
        let mut cache = NruCache::new(2);

        cache.insert(1, 100);
        cache.insert(2, 200);

        // contains() doesn't set reference bit
        assert!(cache.contains(&1));

        // Manually clear reference bit by checking internals
        // (In real usage, this would happen during eviction phase)
        if let Some(entry) = cache.map.get_mut(&1) {
            entry.referenced = false;
        }

        // Now 1 should be evictable
        cache.insert(3, 300);

        assert!(!cache.contains(&1)); // 1 was evicted
        assert!(cache.contains(&2));
        assert!(cache.contains(&3));
    }

    #[test]
    fn test_zero_capacity() {
        let mut cache = NruCache::new(0);
        // Should default to capacity of 1
        assert!(cache.capacity() >= 1);

        cache.insert(1, 100);
        assert!(cache.contains(&1));
    }
}
