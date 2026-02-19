//! Clock-PRO cache replacement policy.
//!
//! An improvement over basic Clock that provides scan resistance by tracking
//! hot/cold page classification and maintaining ghost entries for recently
//! evicted cold pages.
//!
//! ## Architecture
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────────────────┐
//! │                      ClockProCache<K, V> Layout                           │
//! │                                                                           │
//! │   ┌─────────────────────────────────────────────────────────────────────┐ │
//! │   │  index: FxHashMap<K, usize>     (key -> slot in resident buffer)    │ │
//! │   └─────────────────────────────────────────────────────────────────────┘ │
//! │                                                                           │
//! │   ┌─────────────────────────────────────────────────────────────────────┐ │
//! │   │  entries: Vec<Option<Entry<K,V>>>   (resident pages)                │ │
//! │   │                                                                     │ │
//! │   │    [0]     [1]     [2]     [3]     [4]     [5]     [6]     [7]      │ │
//! │   │   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐     │ │
//! │   │   │HOT│   │cld│   │HOT│   │cld│   │cld│   │   │   │   │   │   │     │ │
//! │   │   │ref│   │   │   │ref│   │ref│   │   │   │   │   │   │   │   │     │ │
//! │   │   └───┘   └───┘   └───┘   └───┘   └───┘   └───┘   └───┘   └───┘     │ │
//! │   │             ▲               ▲                                       │ │
//! │   │             │               │                                       │ │
//! │   │         hand_cold       hand_hot                                    │ │
//! │   └─────────────────────────────────────────────────────────────────────┘ │
//! │                                                                           │
//! │   ┌─────────────────────────────────────────────────────────────────────┐ │
//! │   │  ghost: GhostRing<K>   (keys only, recently evicted cold pages)     │ │
//! │   │                                                                     │ │
//! │   │    [k1]   [k2]   [k3]   [k4]   [ ]   [ ]   [ ]   [ ]                │ │
//! │   │                   ▲                                                 │ │
//! │   │               ghost_hand                                            │ │
//! │   └─────────────────────────────────────────────────────────────────────┘ │
//! │                                                                           │
//! │   Cold entries: candidates for eviction                                   │
//! │   Hot entries: protected, must be demoted to cold first                   │
//! │   Ghost entries: detect re-access → promote immediately to hot            │
//! └───────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Algorithm
//!
//! ```text
//! GET(key):
//!   if key in resident:
//!     set referenced = true
//!     if cold and referenced: mark for promotion to hot
//!     return value
//!   if key in ghost:
//!     (ghost hit indicates we should have kept this page)
//!     remove from ghost
//!     return miss (but next insert of this key → hot)
//!   return miss
//!
//! INSERT(key, value):
//!   if key exists: update value, set referenced
//!   if key was recently in ghost: insert as HOT
//!   else: insert as COLD
//!   if at capacity: run eviction
//!
//! EVICT():
//!   while true:
//!     // First, try to find unreferenced cold page
//!     entry = entries[hand_cold]
//!     if entry is cold:
//!       if referenced:
//!         promote to hot, clear referenced
//!       else:
//!         evict, add key to ghost ring
//!         return slot
//!     // Demote hot pages inline if over limit
//!     hand_cold = (hand_cold + 1) % capacity
//! ```
//!
//! ## Scan Resistance
//!
//! Clock-PRO resists scan pollution because:
//! 1. Sequential scans only touch cold pages (new inserts are cold)
//! 2. Cold pages need a second access to become hot
//! 3. Hot pages are protected from eviction
//! 4. Ghost hits boost re-accessed keys directly to hot status
//!
//! ## Performance Characteristics
//!
//! | Operation | Time    | Notes                               |
//! |-----------|---------|-------------------------------------|
//! | `get`     | O(1)    | Hash lookup + bit operations        |
//! | `insert`  | O(1)*   | *Amortized, eviction may sweep      |
//! | `contains`| O(1)    | Hash lookup only                    |
//! | `remove`  | O(1)    | Hash lookup + clear slot            |
//!
//! ## Example Usage
//!
//! ```
//! use cachekit::policy::clock_pro::ClockProCache;
//! use cachekit::traits::{CoreCache, ReadOnlyCache};
//!
//! let mut cache: ClockProCache<String, String> = ClockProCache::new(100);
//!
//! // New inserts start as "cold"
//! cache.insert("page1".to_string(), "content1".to_string());
//! cache.insert("page2".to_string(), "content2".to_string());
//!
//! // Access promotes cold → hot (scan resistant!)
//! cache.get(&"page1".to_string());  // page1 now marked for hot promotion
//!
//! // Hot pages are protected from eviction by scans
//! for i in 0..200 {
//!     cache.insert(format!("scan_{i}"), format!("data_{i}"));  // Scans churn through cold pages
//! }
//!
//! // Hot page1 likely survives the scan (scan-resistant)
//! // Note: With small cache and 200 inserts, hot pages may still be evicted
//! let _ = cache.contains(&"page1".to_string());
//! ```

use crate::prelude::ReadOnlyCache;
use crate::traits::{CoreCache, MutableCache};
use rustc_hash::FxHashMap;
use std::hash::Hash;

/// Status of a resident page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PageStatus {
    /// Cold page: candidate for eviction.
    Cold,
    /// Hot page: protected from eviction.
    Hot,
}

/// Entry in the resident buffer.
#[derive(Debug)]
struct Entry<K, V> {
    key: K,
    value: V,
    status: PageStatus,
    referenced: bool,
}

/// Ghost ring entry (key only, no value).
#[derive(Debug)]
struct GhostEntry<K> {
    key: K,
}

/// High-performance Clock-PRO cache with scan resistance.
///
/// Improves on Clock by distinguishing hot (frequently accessed) and cold
/// (candidates for eviction) pages, plus tracking ghost entries for recently
/// evicted cold pages.
pub struct ClockProCache<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Maps keys to their slot index in the entries buffer.
    index: FxHashMap<K, usize>,
    /// Circular buffer of resident entries.
    entries: Vec<Option<Entry<K, V>>>,
    /// Ghost ring: recently evicted cold page keys.
    ghost: Vec<Option<GhostEntry<K>>>,
    /// Ghost index for O(1) lookup.
    ghost_index: FxHashMap<K, usize>,
    /// Clock hand for cold page eviction.
    hand_cold: usize,
    /// Clock hand for hot page demotion.
    hand_hot: usize,
    /// Clock hand for ghost ring.
    ghost_hand: usize,
    /// Number of resident entries.
    len: usize,
    /// Number of hot pages.
    hot_count: usize,
    /// Number of ghost entries.
    ghost_len: usize,
    /// Maximum resident capacity.
    capacity: usize,
    /// Maximum ghost capacity (typically same as resident capacity).
    ghost_capacity: usize,
    /// Target ratio of hot pages (adaptive).
    target_hot_ratio: f64,
}

// Safety: ClockProCache uses no interior mutability or non-Send types
unsafe impl<K: Send, V: Send> Send for ClockProCache<K, V> where K: Clone + Eq + Hash {}
unsafe impl<K: Sync, V: Sync> Sync for ClockProCache<K, V> where K: Clone + Eq + Hash {}

impl<K, V> ClockProCache<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Creates a new Clock-PRO cache with the specified capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::clock_pro::ClockProCache;
    /// use cachekit::traits::{CoreCache, ReadOnlyCache};
    ///
    /// let cache: ClockProCache<String, i32> = ClockProCache::new(100);
    /// assert_eq!(cache.capacity(), 100);
    /// assert!(cache.is_empty());
    /// ```
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Self::with_ghost_capacity(capacity, capacity)
    }

    /// Creates a new Clock-PRO cache with custom ghost capacity.
    ///
    /// A larger ghost capacity can improve hit rates on workloads with
    /// periodic re-access patterns.
    #[inline]
    pub fn with_ghost_capacity(capacity: usize, ghost_capacity: usize) -> Self {
        let mut entries = Vec::with_capacity(capacity);
        entries.resize_with(capacity, || None);

        let mut ghost = Vec::with_capacity(ghost_capacity);
        ghost.resize_with(ghost_capacity, || None);

        Self {
            index: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            entries,
            ghost,
            ghost_index: FxHashMap::with_capacity_and_hasher(ghost_capacity, Default::default()),
            hand_cold: 0,
            hand_hot: 0,
            ghost_hand: 0,
            len: 0,
            hot_count: 0,
            ghost_len: 0,
            capacity,
            ghost_capacity,
            target_hot_ratio: 0.5, // Start with 50% hot target
        }
    }

    /// Returns `true` if the cache is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of hot pages.
    #[inline]
    pub fn hot_count(&self) -> usize {
        self.hot_count
    }

    /// Returns the number of cold pages.
    #[inline]
    pub fn cold_count(&self) -> usize {
        self.len - self.hot_count
    }

    /// Returns the number of ghost entries.
    #[inline]
    pub fn ghost_count(&self) -> usize {
        self.ghost_len
    }

    /// Checks if a key is in the ghost ring.
    #[inline]
    fn is_ghost(&self, key: &K) -> bool {
        self.ghost_index.contains_key(key)
    }

    /// Removes a key from the ghost ring.
    #[inline]
    fn remove_ghost(&mut self, key: &K) {
        if let Some(slot) = self.ghost_index.remove(key) {
            self.ghost[slot] = None;
            self.ghost_len -= 1;
        }
    }

    /// Adds a key to the ghost ring (evicting old ghost if full).
    #[inline]
    fn add_ghost(&mut self, key: K) {
        // Don't add if already in ghost
        if self.ghost_index.contains_key(&key) {
            return;
        }

        // Always use ghost_hand position - evict existing if present
        let slot = self.ghost_hand;
        if let Some(old) = self.ghost[slot].take() {
            self.ghost_index.remove(&old.key);
            self.ghost_len -= 1;
        }

        self.ghost[slot] = Some(GhostEntry { key: key.clone() });
        self.ghost_index.insert(key, slot);
        self.ghost_len += 1;
        self.ghost_hand = (self.ghost_hand + 1) % self.ghost_capacity;
    }

    /// Finds an empty slot, evicting if necessary.
    #[inline]
    fn find_slot(&mut self) -> usize {
        if self.len < self.capacity {
            // Use hand_cold to find empty slot - it sweeps anyway
            for _ in 0..self.capacity {
                if self.entries[self.hand_cold].is_none() {
                    let slot = self.hand_cold;
                    self.hand_cold = (self.hand_cold + 1) % self.capacity;
                    return slot;
                }
                self.hand_cold = (self.hand_cold + 1) % self.capacity;
            }
        }
        // At capacity - need to evict
        self.evict()
    }

    /// Runs the Clock-PRO eviction algorithm.
    ///
    /// Sweeps with a strict limit of 2×capacity iterations.
    /// Returns the index of the evicted slot.
    #[inline]
    fn evict(&mut self) -> usize {
        let max_iterations = self.capacity * 2;
        let max_hot = ((self.capacity as f64) * self.target_hot_ratio).ceil() as usize;
        let max_hot = max_hot.max(1).min(self.capacity.saturating_sub(1));

        for _ in 0..max_iterations {
            if let Some(entry) = &mut self.entries[self.hand_cold] {
                match entry.status {
                    PageStatus::Cold => {
                        if entry.referenced {
                            // Cold but referenced → promote to hot
                            entry.status = PageStatus::Hot;
                            entry.referenced = false;
                            self.hot_count += 1;
                        } else {
                            // Cold and unreferenced → evict immediately
                            let slot = self.hand_cold;
                            let key = entry.key.clone();
                            self.index.remove(&key);
                            self.entries[slot] = None;
                            self.len -= 1;
                            self.add_ghost(key);
                            self.hand_cold = (self.hand_cold + 1) % self.capacity;
                            return slot;
                        }
                    },
                    PageStatus::Hot => {
                        // Demote hot pages inline if we have too many
                        if self.hot_count > max_hot {
                            if entry.referenced {
                                entry.referenced = false;
                            } else {
                                entry.status = PageStatus::Cold;
                                self.hot_count -= 1;
                            }
                        } else if entry.referenced {
                            // Just clear the reference bit
                            entry.referenced = false;
                        }
                    },
                }
            }
            self.hand_cold = (self.hand_cold + 1) % self.capacity;
        }

        // Fallback: force evict at current hand position
        // This should rarely happen - only when all pages are hot and referenced
        let slot = self.hand_cold;
        if let Some(entry) = &self.entries[slot] {
            let key = entry.key.clone();
            self.index.remove(&key);
            if entry.status == PageStatus::Hot {
                self.hot_count -= 1;
            }
            self.add_ghost(key);
        }
        self.entries[slot] = None;
        self.len -= 1;
        self.hand_cold = (self.hand_cold + 1) % self.capacity;
        slot
    }
}

impl<K, V> ReadOnlyCache<K, V> for ClockProCache<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Returns `true` if the cache contains the key.
    ///
    /// Does not affect the reference bit or page status.
    #[inline]
    fn contains(&self, key: &K) -> bool {
        self.index.contains_key(key)
    }

    /// Returns the number of resident entries in the cache.
    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    /// Returns the maximum capacity of the cache.
    #[inline]
    fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<K, V> CoreCache<K, V> for ClockProCache<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Inserts a key-value pair into the cache.
    ///
    /// New entries start as cold unless the key was recently evicted (ghost hit),
    /// in which case they start as hot.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::clock_pro::ClockProCache;
    /// use cachekit::traits::CoreCache;
    ///
    /// let mut cache = ClockProCache::new(2);
    /// cache.insert("a", 1);  // Inserted as cold
    /// cache.insert("b", 2);  // Inserted as cold
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

        // Check if this was a ghost hit
        let was_ghost = self.is_ghost(&key);
        if was_ghost {
            self.remove_ghost(&key);
            // Increase hot ratio since we're seeing re-accesses
            self.target_hot_ratio = (self.target_hot_ratio + 0.05).min(0.9);
        }

        // Find slot (may evict)
        let slot = self.find_slot();

        // Determine initial status
        let status = if was_ghost {
            self.hot_count += 1;
            PageStatus::Hot // Ghost hit → insert as hot
        } else {
            PageStatus::Cold // Normal insert → cold
        };

        // Insert new entry
        self.entries[slot] = Some(Entry {
            key: key.clone(),
            value,
            status,
            referenced: false,
        });
        self.index.insert(key, slot);
        self.len += 1;

        // Hot balancing happens during eviction sweeps, not here
        None
    }

    /// Gets a reference to the value for a key.
    ///
    /// Sets the reference bit on access. Cold pages with their reference
    /// bit set will be promoted to hot during eviction sweeps.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::clock_pro::ClockProCache;
    /// use cachekit::traits::CoreCache;
    ///
    /// let mut cache = ClockProCache::new(10);
    /// cache.insert("key", 42);
    ///
    /// // Access marks for potential hot promotion
    /// assert_eq!(cache.get(&"key"), Some(&42));
    /// ```
    #[inline]
    fn get(&mut self, key: &K) -> Option<&V> {
        let slot = *self.index.get(key)?;
        let entry = self.entries[slot].as_mut()?;
        entry.referenced = true;
        Some(&entry.value)
    }

    /// Clears all entries from the cache.
    fn clear(&mut self) {
        self.index.clear();
        self.ghost_index.clear();
        for entry in &mut self.entries {
            *entry = None;
        }
        for ghost in &mut self.ghost {
            *ghost = None;
        }
        self.len = 0;
        self.hot_count = 0;
        self.ghost_len = 0;
        self.hand_cold = 0;
        self.hand_hot = 0;
        self.ghost_hand = 0;
        self.target_hot_ratio = 0.5;
    }
}

impl<K, V> MutableCache<K, V> for ClockProCache<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Removes a key from the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::clock_pro::ClockProCache;
    /// use cachekit::traits::{CoreCache, MutableCache, ReadOnlyCache};
    ///
    /// let mut cache = ClockProCache::new(10);
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
        if entry.status == PageStatus::Hot {
            self.hot_count -= 1;
        }
        Some(entry.value)
    }
}

impl<K, V> std::fmt::Debug for ClockProCache<K, V>
where
    K: Clone + Eq + Hash + std::fmt::Debug,
    V: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClockProCache")
            .field("len", &self.len)
            .field("capacity", &self.capacity)
            .field("hot_count", &self.hot_count)
            .field("ghost_len", &self.ghost_len)
            .field("target_hot_ratio", &self.target_hot_ratio)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::MutableCache;

    #[test]
    fn test_basic_operations() {
        let mut cache = ClockProCache::new(3);

        // Insert
        assert!(cache.insert("a", 1).is_none());
        assert!(cache.insert("b", 2).is_none());
        assert!(cache.insert("c", 3).is_none());

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get(&"a"), Some(&1));
        assert_eq!(cache.get(&"b"), Some(&2));
        assert_eq!(cache.get(&"c"), Some(&3));
    }

    #[test]
    fn test_update_existing() {
        let mut cache = ClockProCache::new(3);

        cache.insert("a", 1);
        let old = cache.insert("a", 10);

        assert_eq!(old, Some(1));
        assert_eq!(cache.get(&"a"), Some(&10));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_eviction() {
        let mut cache = ClockProCache::new(3);

        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3);

        // This should trigger eviction
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
    fn test_hot_cold_promotion() {
        let mut cache = ClockProCache::new(4);

        // Insert cold
        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3);
        cache.insert("d", 4);

        // All should start cold
        assert_eq!(cache.cold_count(), 4);
        assert_eq!(cache.hot_count(), 0);

        // Access "a" multiple times to mark it
        cache.get(&"a");
        cache.get(&"a");

        // Trigger eviction to promote "a" to hot
        cache.insert("e", 5);
        cache.insert("f", 6);

        // "a" should have been promoted
        assert!(cache.contains(&"a"));
    }

    #[test]
    fn test_ghost_hit() {
        let mut cache = ClockProCache::new(2);

        cache.insert("a", 1);
        cache.insert("b", 2);

        // Evict "a"
        cache.insert("c", 3);

        // "a" should be in ghost
        assert!(!cache.contains(&"a"));
        assert!(cache.is_ghost(&"a"));

        // Re-insert "a" - should come in as hot (ghost hit)
        cache.insert("a", 10);
        assert!(cache.contains(&"a"));
        assert!(!cache.is_ghost(&"a"));
        assert!(cache.hot_count() >= 1); // "a" should be hot
    }

    #[test]
    fn test_scan_resistance() {
        let mut cache = ClockProCache::new(100);

        // Insert and access working set
        for i in 0..50 {
            cache.insert(i, i);
            cache.get(&i); // Mark as accessed
        }

        // Scan through many items
        for i in 1000..2000 {
            cache.insert(i, i);
        }

        // Check how many of original working set survived
        let survived: usize = (0..50).filter(|i| cache.contains(i)).count();

        // With scan resistance, some working set should survive
        // Basic Clock would likely lose most of it
        assert!(
            survived > 10,
            "Expected scan resistance: {} of 50 survived",
            survived
        );
    }

    #[test]
    fn test_remove() {
        let mut cache = ClockProCache::new(3);

        cache.insert("a", 1);
        cache.insert("b", 2);

        let removed = cache.remove(&"a");
        assert_eq!(removed, Some(1));
        assert!(!cache.contains(&"a"));
        assert_eq!(cache.len(), 1);

        // Remove non-existent
        assert!(cache.remove(&"z").is_none());
    }

    #[test]
    fn test_clear() {
        let mut cache = ClockProCache::new(3);

        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.get(&"a"); // Access to mark

        cache.clear();

        assert_eq!(cache.len(), 0);
        assert_eq!(cache.hot_count(), 0);
        assert_eq!(cache.ghost_count(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_capacity_one() {
        let mut cache = ClockProCache::new(1);

        cache.insert("a", 1);
        assert_eq!(cache.get(&"a"), Some(&1));

        cache.insert("b", 2);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains(&"b"));
    }

    #[test]
    fn test_contains_no_side_effect() {
        let mut cache = ClockProCache::new(3);

        cache.insert("a", 1);

        // contains should not affect reference bit
        let _ = cache.contains(&"a");
        let _ = cache.contains(&"a");

        // Entry should still be cold with no reference
        assert_eq!(cache.cold_count(), 1);
    }

    #[test]
    fn test_ghost_capacity() {
        let mut cache: ClockProCache<u64, u64> = ClockProCache::with_ghost_capacity(2, 5);

        // Fill cache
        cache.insert(0, 1);
        cache.insert(1, 2);

        // Evict items into ghost
        for i in 10..15 {
            cache.insert(i, i);
        }

        // Ghost should have entries
        assert!(cache.ghost_count() > 0);
        assert!(cache.ghost_count() <= 5);
    }

    #[test]
    fn test_debug_impl() {
        let mut cache = ClockProCache::new(10);
        cache.insert("a", 1);
        cache.insert("b", 2);

        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("ClockProCache"));
        assert!(debug_str.contains("len"));
        assert!(debug_str.contains("capacity"));
    }

    #[test]
    fn test_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<ClockProCache<String, i32>>();
        assert_sync::<ClockProCache<String, i32>>();
    }
}
