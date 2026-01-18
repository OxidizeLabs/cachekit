//! Bounded recency list for ghost entries.
//!
//! Used by adaptive policies (ARC/2Q-style) to track recently evicted keys
//! without storing values. Implemented as an [`IntrusiveList`]
//! plus a [`HashMap`] index for O(1) lookups.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                           GhostList Layout                                  │
//! │                                                                             │
//! │   ┌─────────────────────────────┐   ┌─────────────────────────────────┐   │
//! │   │  index: HashMap<K, SlotId>  │   │  list: IntrusiveList<K>         │   │
//! │   │                             │   │                                 │   │
//! │   │  ┌───────────┬──────────┐  │   │  head ──► [A] ◄──► [B] ◄──► [C] │   │
//! │   │  │    Key    │  SlotId  │  │   │            MRU             LRU  │   │
//! │   │  ├───────────┼──────────┤  │   │                          ▲      │   │
//! │   │  │  "key_a"  │   id_0   │──┼───┼─────────► [A]            │      │   │
//! │   │  │  "key_b"  │   id_1   │──┼───┼─────────► [B]            │      │   │
//! │   │  │  "key_c"  │   id_2   │──┼───┼─────────► [C] ◄──────────┘      │   │
//! │   │  └───────────┴──────────┘  │   │                    tail         │   │
//! │   └─────────────────────────────┘   └─────────────────────────────────┘   │
//! │                                                                             │
//! │   Record Flow (capacity = 3)                                               │
//! │   ──────────────────────────────                                           │
//! │                                                                             │
//! │   record("key_d") when full:                                               │
//! │     1. Check index: "key_d" not found                                      │
//! │     2. At capacity: evict LRU ("key_c")                                    │
//! │        - pop_back() from list                                              │
//! │        - remove("key_c") from index                                        │
//! │     3. Insert "key_d" at front (MRU)                                       │
//! │        - push_front("key_d") in list                                       │
//! │        - insert("key_d", id) in index                                      │
//! │                                                                             │
//! │   record("key_a") when present:                                            │
//! │     1. Check index: "key_a" found with id_0                                │
//! │     2. move_to_front(id_0) in list                                         │
//! │     3. Done (no eviction needed)                                           │
//! │                                                                             │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Components
//!
//! - [`GhostList`]: Bounded recency tracker for evicted keys
//!
//! ## Operations
//!
//! | Operation      | Description                           | Complexity |
//! |----------------|---------------------------------------|------------|
//! | `record`       | Add/promote key to MRU, evict if full | O(1) avg   |
//! | `remove`       | Remove key from ghost list            | O(1) avg   |
//! | `contains`     | Check if key is tracked               | O(1) avg   |
//! | `record_batch` | Record multiple keys                  | O(n)       |
//! | `remove_batch` | Remove multiple keys                  | O(n)       |
//!
//! ## Use Cases
//!
//! - **ARC policy**: Track B1 (recently evicted once) and B2 (recently evicted twice)
//! - **2Q policy**: Track ghost entries from A1out queue
//! - **Adaptive tuning**: Detect re-references to recently evicted keys
//!
//! ## Example Usage
//!
//! ```
//! use cachekit::ds::GhostList;
//!
//! // Track last 100 evicted keys
//! let mut ghost = GhostList::new(100);
//!
//! // Record evicted keys
//! ghost.record("page_1");
//! ghost.record("page_2");
//! ghost.record("page_3");
//!
//! // Check if a key was recently evicted (ghost hit)
//! if ghost.contains(&"page_1") {
//!     // Key was recently evicted - consider increasing cache size
//!     println!("Ghost hit! Should have kept page_1");
//! }
//!
//! // Re-recording moves to MRU position
//! ghost.record("page_1");  // Now page_1 is most recent
//! ```
//!
//! ## Use Case: ARC-Style Adaptation
//!
//! ```
//! use cachekit::ds::GhostList;
//!
//! struct AdaptiveCache {
//!     ghost_recent: GhostList<String>,   // B1: recently evicted from recency list
//!     ghost_frequent: GhostList<String>, // B2: recently evicted from frequency list
//!     p: usize,  // Adaptation parameter
//! }
//!
//! impl AdaptiveCache {
//!     fn new(capacity: usize) -> Self {
//!         Self {
//!             ghost_recent: GhostList::new(capacity),
//!             ghost_frequent: GhostList::new(capacity),
//!             p: capacity / 2,
//!         }
//!     }
//!
//!     fn on_miss(&mut self, key: &str) {
//!         if self.ghost_recent.contains(&key.to_string()) {
//!             // Hit in B1: increase recency preference
//!             self.p = self.p.saturating_add(1);
//!             self.ghost_recent.remove(&key.to_string());
//!         } else if self.ghost_frequent.contains(&key.to_string()) {
//!             // Hit in B2: increase frequency preference
//!             self.p = self.p.saturating_sub(1);
//!             self.ghost_frequent.remove(&key.to_string());
//!         }
//!     }
//! }
//!
//! let mut cache = AdaptiveCache::new(100);
//! cache.ghost_recent.record("evicted_key".to_string());
//! cache.on_miss("evicted_key");  // Adapts based on ghost hit
//! ```
//!
//! ## Thread Safety
//!
//! `GhostList` is not thread-safe. For concurrent use, wrap in
//! `parking_lot::RwLock` or similar synchronization primitive.
//!
//! ## Implementation Notes
//!
//! - Backed by [`IntrusiveList`] for O(1) reordering
//! - Keys are stored in both the list and index (requires `Clone`)
//! - Zero-capacity ghost lists are no-ops (record does nothing)
//! - `debug_validate_invariants()` available in debug/test builds

use rustc_hash::FxHashMap;
use std::hash::Hash;

use crate::ds::intrusive_list::IntrusiveList;
use crate::ds::slot_arena::SlotId;

/// Bounded recency list of keys (no values), typically for ARC-style ghost tracking.
///
/// Tracks recently evicted keys to detect when items should have been kept in cache.
/// When a "ghost hit" occurs (accessing a key in the ghost list), adaptive policies
/// can adjust their behavior.
///
/// # Type Parameters
///
/// - `K`: Key type, must be `Eq + Hash + Clone`
///
/// # Example
///
/// ```
/// use cachekit::ds::GhostList;
///
/// let mut ghost = GhostList::new(3);
///
/// // Record evicted keys
/// ghost.record("a");
/// ghost.record("b");
/// ghost.record("c");
/// assert_eq!(ghost.len(), 3);
///
/// // At capacity, oldest is evicted
/// ghost.record("d");
/// assert!(!ghost.contains(&"a"));  // "a" was evicted
/// assert!(ghost.contains(&"d"));   // "d" is now tracked
///
/// // Re-recording promotes to MRU
/// ghost.record("b");
/// ghost.record("e");
/// assert!(ghost.contains(&"b"));   // "b" survives (was promoted)
/// assert!(!ghost.contains(&"c"));  // "c" was LRU, evicted
/// ```
///
/// # Use Case: Detecting Thrashing
///
/// ```
/// use cachekit::ds::GhostList;
///
/// let mut ghost = GhostList::new(50);
/// let mut ghost_hits = 0;
///
/// // Simulate evictions and re-accesses
/// let evicted_keys = vec!["key_1", "key_2", "key_3"];
/// for key in &evicted_keys {
///     ghost.record(*key);
/// }
///
/// // Later, check if we're re-accessing evicted keys
/// let accessed = vec!["key_1", "key_5", "key_2"];
/// for key in &accessed {
///     if ghost.contains(key) {
///         ghost_hits += 1;
///         // Could signal need for larger cache
///     }
/// }
///
/// assert_eq!(ghost_hits, 2);  // key_1 and key_2 were ghost hits
/// ```
#[derive(Debug)]
pub struct GhostList<K> {
    list: IntrusiveList<K>,
    index: FxHashMap<K, SlotId>,
    capacity: usize,
}

impl<K> GhostList<K>
where
    K: Eq + Hash + Clone,
{
    /// Creates a new ghost list with a maximum of `capacity` keys.
    ///
    /// A capacity of 0 creates a no-op ghost list that ignores all records.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::GhostList;
    ///
    /// let ghost: GhostList<String> = GhostList::new(100);
    /// assert_eq!(ghost.capacity(), 100);
    /// assert!(ghost.is_empty());
    /// ```
    pub fn new(capacity: usize) -> Self {
        Self {
            list: IntrusiveList::with_capacity(capacity),
            index: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            capacity,
        }
    }

    /// Returns the configured capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::GhostList;
    ///
    /// let ghost: GhostList<&str> = GhostList::new(50);
    /// assert_eq!(ghost.capacity(), 50);
    /// ```
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of keys currently tracked.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::GhostList;
    ///
    /// let mut ghost = GhostList::new(10);
    /// assert_eq!(ghost.len(), 0);
    ///
    /// ghost.record("a");
    /// ghost.record("b");
    /// assert_eq!(ghost.len(), 2);
    ///
    /// // Re-recording same key doesn't increase length
    /// ghost.record("a");
    /// assert_eq!(ghost.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.list.len()
    }

    /// Returns `true` if there are no keys tracked.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::GhostList;
    ///
    /// let mut ghost: GhostList<&str> = GhostList::new(10);
    /// assert!(ghost.is_empty());
    ///
    /// ghost.record("key");
    /// assert!(!ghost.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    /// Returns `true` if `key` is present in the ghost list.
    ///
    /// This is the "ghost hit" check used by adaptive policies.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::GhostList;
    ///
    /// let mut ghost = GhostList::new(10);
    /// ghost.record("evicted_page");
    ///
    /// // Check for ghost hit
    /// if ghost.contains(&"evicted_page") {
    ///     println!("Ghost hit! Key was recently evicted.");
    /// }
    ///
    /// assert!(ghost.contains(&"evicted_page"));
    /// assert!(!ghost.contains(&"never_seen"));
    /// ```
    pub fn contains(&self, key: &K) -> bool {
        self.index.contains_key(key)
    }

    /// Records `key` as most-recently-seen, evicting the least recent if needed.
    ///
    /// If the key is already present, it is promoted to MRU position.
    /// If at capacity, the LRU key is evicted before inserting.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::GhostList;
    ///
    /// let mut ghost = GhostList::new(2);
    ///
    /// ghost.record("a");
    /// ghost.record("b");
    /// assert!(ghost.contains(&"a"));
    /// assert!(ghost.contains(&"b"));
    ///
    /// // At capacity: "a" is LRU, will be evicted
    /// ghost.record("c");
    /// assert!(!ghost.contains(&"a"));  // Evicted
    /// assert!(ghost.contains(&"b"));
    /// assert!(ghost.contains(&"c"));
    ///
    /// // Re-recording "b" promotes it to MRU
    /// ghost.record("b");
    /// ghost.record("d");
    /// assert!(ghost.contains(&"b"));   // Survived (was MRU)
    /// assert!(!ghost.contains(&"c"));  // Evicted (was LRU)
    /// ```
    pub fn record(&mut self, key: K) {
        if self.capacity == 0 {
            return;
        }

        if let Some(&id) = self.index.get(&key) {
            self.list.move_to_front(id);
            return;
        }

        if self.list.len() >= self.capacity {
            if let Some(old_key) = self.list.pop_back() {
                self.index.remove(&old_key);
            }
        }

        let id = self.list.push_front(key.clone());
        self.index.insert(key, id);
    }

    /// Records a batch of keys; returns number of keys processed.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::GhostList;
    ///
    /// let mut ghost = GhostList::new(10);
    ///
    /// let evicted = vec!["page_1", "page_2", "page_3"];
    /// let count = ghost.record_batch(evicted);
    ///
    /// assert_eq!(count, 3);
    /// assert_eq!(ghost.len(), 3);
    /// ```
    pub fn record_batch<I>(&mut self, keys: I) -> usize
    where
        I: IntoIterator<Item = K>,
    {
        let mut count = 0;
        for key in keys {
            self.record(key);
            count += 1;
        }
        count
    }

    /// Removes `key` from the ghost list; returns `true` if it was present.
    ///
    /// Typically called after a ghost hit to prevent double-counting.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::GhostList;
    ///
    /// let mut ghost = GhostList::new(10);
    /// ghost.record("key");
    ///
    /// // Ghost hit: remove from ghost list
    /// assert!(ghost.remove(&"key"));
    /// assert!(!ghost.contains(&"key"));
    ///
    /// // Removing missing key returns false
    /// assert!(!ghost.remove(&"missing"));
    /// ```
    pub fn remove(&mut self, key: &K) -> bool {
        let id = match self.index.remove(key) {
            Some(id) => id,
            None => return false,
        };
        self.list.remove(id);
        true
    }

    /// Removes a batch of keys; returns number of keys actually removed.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::GhostList;
    ///
    /// let mut ghost = GhostList::new(10);
    /// ghost.record_batch(["a", "b", "c"]);
    ///
    /// // Remove some keys (including one that doesn't exist)
    /// let removed = ghost.remove_batch(["a", "c", "missing"]);
    ///
    /// assert_eq!(removed, 2);  // Only "a" and "c" were removed
    /// assert!(ghost.contains(&"b"));
    /// ```
    pub fn remove_batch<I>(&mut self, keys: I) -> usize
    where
        I: IntoIterator<Item = K>,
    {
        let mut removed = 0;
        for key in keys {
            if self.remove(&key) {
                removed += 1;
            }
        }
        removed
    }

    /// Clears all tracked keys.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::GhostList;
    ///
    /// let mut ghost = GhostList::new(10);
    /// ghost.record("a");
    /// ghost.record("b");
    ///
    /// ghost.clear();
    /// assert!(ghost.is_empty());
    /// assert!(!ghost.contains(&"a"));
    /// ```
    pub fn clear(&mut self) {
        self.list.clear();
        self.index.clear();
    }

    /// Clears all tracked keys and shrinks internal storage.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::GhostList;
    ///
    /// let mut ghost = GhostList::new(100);
    /// ghost.record_batch((0..50).map(|i| format!("key_{}", i)));
    ///
    /// ghost.clear_shrink();
    /// assert!(ghost.is_empty());
    /// ```
    pub fn clear_shrink(&mut self) {
        self.clear();
        self.list.clear_shrink();
        self.index.shrink_to_fit();
    }

    /// Returns an approximate memory footprint in bytes.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::GhostList;
    ///
    /// let ghost: GhostList<u64> = GhostList::new(100);
    /// let bytes = ghost.approx_bytes();
    /// assert!(bytes > 0);
    /// ```
    pub fn approx_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.list.approx_bytes()
            + self.index.capacity() * std::mem::size_of::<(K, SlotId)>()
    }

    #[cfg(any(test, debug_assertions))]
    /// Returns keys in MRU -> LRU order (requires `K: Clone`).
    pub fn debug_snapshot_keys(&self) -> Vec<K>
    where
        K: Clone,
    {
        self.list
            .iter_entries()
            .map(|(_, key)| key.clone())
            .collect()
    }

    #[cfg(any(test, debug_assertions))]
    /// Returns SlotIds in MRU -> LRU order.
    pub fn debug_snapshot_ids(&self) -> Vec<SlotId> {
        self.list.iter_ids().collect()
    }

    #[cfg(any(test, debug_assertions))]
    /// Returns SlotIds sorted by index for deterministic snapshots.
    pub fn debug_snapshot_ids_sorted(&self) -> Vec<SlotId> {
        let mut ids: Vec<_> = self.list.iter_ids().collect();
        ids.sort_by_key(|id| id.index());
        ids
    }

    #[cfg(any(test, debug_assertions))]
    /// Returns keys ordered by SlotId index for deterministic snapshots.
    pub fn debug_snapshot_keys_sorted(&self) -> Vec<K>
    where
        K: Clone,
    {
        let mut ids: Vec<_> = self.list.iter_ids().collect();
        ids.sort_by_key(|id| id.index());
        ids.into_iter()
            .filter_map(|id| self.list.get(id).cloned())
            .collect()
    }

    #[cfg(any(test, debug_assertions))]
    pub fn debug_validate_invariants(&self) {
        assert_eq!(self.list.len(), self.index.len());
        assert!(self.list.len() <= self.capacity);
        if self.capacity == 0 {
            assert!(self.list.is_empty());
        }
        for &id in self.index.values() {
            assert!(self.list.contains(id));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ghost_list_records_and_evictions() {
        let mut ghost = GhostList::new(2);
        ghost.record("a");
        ghost.record("b");
        assert!(ghost.contains(&"a"));
        assert!(ghost.contains(&"b"));

        ghost.record("a");
        ghost.record("c");

        assert!(ghost.contains(&"a"));
        assert!(ghost.contains(&"c"));
        assert!(!ghost.contains(&"b"));
    }

    #[test]
    fn ghost_list_zero_capacity_is_noop() {
        let mut ghost = GhostList::new(0);
        ghost.record("a");
        ghost.record("b");
        assert!(ghost.is_empty());
        assert_eq!(ghost.len(), 0);
        assert!(!ghost.contains(&"a"));
        assert!(!ghost.contains(&"b"));
    }

    #[test]
    fn ghost_list_record_existing_moves_to_front() {
        let mut ghost = GhostList::new(3);
        ghost.record("a");
        ghost.record("b");
        ghost.record("c");
        assert!(ghost.contains(&"a"));
        assert!(ghost.contains(&"b"));
        assert!(ghost.contains(&"c"));

        ghost.record("a");
        ghost.record("d");

        assert!(ghost.contains(&"a"));
        assert!(!ghost.contains(&"b"));
        assert!(ghost.contains(&"c"));
        assert!(ghost.contains(&"d"));
    }

    #[test]
    fn ghost_list_remove_existing_and_missing() {
        let mut ghost = GhostList::new(2);
        ghost.record("a");
        ghost.record("b");
        assert!(ghost.remove(&"a"));
        assert!(!ghost.contains(&"a"));
        assert_eq!(ghost.len(), 1);

        assert!(!ghost.remove(&"missing"));
        assert_eq!(ghost.len(), 1);
    }

    #[test]
    fn ghost_list_clear_resets_state() {
        let mut ghost = GhostList::new(2);
        ghost.record("a");
        ghost.record("b");
        ghost.clear();

        assert!(ghost.is_empty());
        assert_eq!(ghost.len(), 0);
        assert!(!ghost.contains(&"a"));
        assert!(!ghost.contains(&"b"));
    }

    #[test]
    fn ghost_list_debug_invariants_hold() {
        let mut ghost = GhostList::new(2);
        ghost.record("a");
        ghost.record("b");
        ghost.record("a");
        ghost.debug_validate_invariants();
    }

    #[test]
    fn ghost_list_debug_snapshots() {
        let mut ghost = GhostList::new(2);
        ghost.record("a");
        ghost.record("b");
        let keys = ghost.debug_snapshot_keys();
        let ids = ghost.debug_snapshot_ids();
        assert_eq!(keys.len(), 2);
        assert_eq!(ids.len(), 2);
        assert_eq!(ghost.debug_snapshot_ids_sorted().len(), 2);
        assert_eq!(ghost.debug_snapshot_keys_sorted().len(), 2);
    }

    #[test]
    fn ghost_list_batch_ops() {
        let mut ghost = GhostList::new(3);
        assert_eq!(ghost.record_batch(["a", "b", "c"]), 3);
        assert_eq!(ghost.remove_batch(["b", "d"]), 1);
        assert!(ghost.contains(&"a"));
        assert!(ghost.contains(&"c"));
        assert!(!ghost.contains(&"b"));
    }
}
