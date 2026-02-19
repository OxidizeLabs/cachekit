//! Bounded recency list for ghost entries.
//!
//! Used by adaptive policies (ARC/2Q-style) to track recently evicted keys
//! without storing values. Implemented as an [`IntrusiveList`]
//! plus a `HashMap` index for O(1) lookups.
//!
//! ## Architecture
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────────────────┐
//! │                           GhostList Layout                                │
//! │                                                                           │
//! │   ┌─────────────────────────────┐   ┌─────────────────────────────────┐   │
//! │   │  index: HashMap<K, SlotId>  │   │  list: IntrusiveList<K>         │   │
//! │   │                             │   │                                 │   │
//! │   │  ┌───────────┬──────────┐   │   │  head ──► [A] ◄──► [B] ◄──► [C] │   │
//! │   │  │    Key    │  SlotId  │   │   │            MRU             LRU  │   │
//! │   │  ├───────────┼──────────┤   │   │                             ▲   │   │
//! │   │  │  "key_a"  │   id_0   │───┼───┼─────────► [A]               │   │   │
//! │   │  │  "key_b"  │   id_1   │───┼───┼─────────► [B]               │   │   │
//! │   │  │  "key_c"  │   id_2   │───┼───┼─────────► [C] ◄─────────────┘   │   │
//! │   │  └───────────┴──────────┘   │   │                    tail         │   │
//! │   └─────────────────────────────┘   └─────────────────────────────────┘   │
//! │                                                                           │
//! │   Record Flow (capacity = 3)                                              │
//! │   ──────────────────────────────                                          │
//! │                                                                           │
//! │   record("key_d") when full:                                              │
//! │     1. Check index: "key_d" not found                                     │
//! │     2. At capacity: evict LRU ("key_c")                                   │
//! │        - pop_back() from list                                             │
//! │        - remove("key_c") from index                                       │
//! │     3. Insert "key_d" at front (MRU)                                      │
//! │        - push_front("key_d") in list                                      │
//! │        - insert("key_d", id) in index                                     │
//! │                                                                           │
//! │   record("key_a") when present:                                           │
//! │     1. Check index: "key_a" found with id_0                               │
//! │     2. move_to_front(id_0) in list                                        │
//! │     3. Done (no eviction needed)                                          │
//! │                                                                           │
//! └───────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Components
//!
//! - [`GhostList`]: Bounded recency tracker for evicted keys
//! - [`Iter`]: Iterator over keys in MRU to LRU order; created by [`GhostList::iter`]
//! - [`IntoIter`]: Consuming iterator over keys; created by `.into_iter()`
//!
//! ## Operations
//!
//! | Operation      | Description                           | Complexity |
//! |----------------|---------------------------------------|------------|
//! | [`record`](GhostList::record) | Add/promote key to MRU, evict if full | O(1) avg |
//! | [`remove`](GhostList::remove) | Remove key from ghost list | O(1) avg |
//! | [`contains`](GhostList::contains) | Check if key is tracked | O(1) avg |
//! | [`evict_lru`](GhostList::evict_lru) | Pop the least recently used key | O(1) avg |
//! | [`record_batch`](GhostList::record_batch) | Record multiple keys | O(n) |
//! | [`remove_batch`](GhostList::remove_batch) | Remove multiple keys | O(n) |
//! | [`iter`](GhostList::iter) | Iterate keys in MRU to LRU order | O(n) |
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
//!     fn on_miss(&mut self, key: String) {
//!         if self.ghost_recent.contains(&key) {
//!             // Hit in B1: increase recency preference
//!             self.p = self.p.saturating_add(1);
//!             self.ghost_recent.remove(&key);
//!         } else if self.ghost_frequent.contains(&key) {
//!             // Hit in B2: increase frequency preference
//!             self.p = self.p.saturating_sub(1);
//!             self.ghost_frequent.remove(&key);
//!         }
//!     }
//! }
//!
//! let mut cache = AdaptiveCache::new(100);
//! cache.ghost_recent.record("evicted_key".to_string());
//! cache.on_miss("evicted_key".to_string());  // Adapts based on ghost hit
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
//!

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
///
/// # Traits
///
/// Implements [`Clone`], [`PartialEq`], [`Eq`], [`Default`], [`Extend<K>`](Extend),
/// [`IntoIterator`] (consuming and borrowed).
#[derive(Debug)]
pub struct GhostList<K> {
    list: IntrusiveList<K>,
    index: FxHashMap<K, SlotId>,
    capacity: usize,
}

impl<K> Clone for GhostList<K>
where
    K: Eq + Hash + Clone,
{
    fn clone(&self) -> Self {
        let mut new_list = IntrusiveList::with_capacity(self.capacity);
        let mut new_index = FxHashMap::with_capacity_and_hasher(self.capacity, Default::default());

        // Rebuild list and index from current state
        for (_, key) in self.list.iter_entries() {
            let id = new_list.push_back(key.clone());
            new_index.insert(key.clone(), id);
        }

        Self {
            list: new_list,
            index: new_index,
            capacity: self.capacity,
        }
    }
}

/// Iterator over keys in a [`GhostList`], yielding references in MRU to LRU order.
///
/// Created by [`GhostList::iter`].
pub struct Iter<'a, K> {
    inner: MapToKey<'a, K>,
}

type MapToKey<'a, K> = std::iter::Map<
    crate::ds::intrusive_list::IntrusiveListEntryIter<'a, K>,
    fn((SlotId, &'a K)) -> &'a K,
>;

impl<'a, K: std::fmt::Debug> std::fmt::Debug for Iter<'a, K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Iter").finish_non_exhaustive()
    }
}

impl<'a, K> Iterator for Iter<'a, K> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, K> ExactSizeIterator for Iter<'a, K> {}

impl<'a, K> std::iter::FusedIterator for Iter<'a, K> {}

/// Consuming iterator over keys in a [`GhostList`], yielding owned keys in MRU to LRU order.
///
/// Created by calling `.into_iter()` on a `GhostList`.
#[derive(Debug)]
pub struct IntoIter<K> {
    inner: std::vec::IntoIter<K>,
}

impl<K> Iterator for IntoIter<K> {
    type Item = K;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K> ExactSizeIterator for IntoIter<K> {}

impl<K> std::iter::FusedIterator for IntoIter<K> {}

impl<K> IntoIterator for GhostList<K>
where
    K: Eq + Hash + Clone,
{
    type Item = K;
    type IntoIter = IntoIter<K>;

    /// Consumes the ghost list and yields keys in MRU to LRU order.
    fn into_iter(self) -> Self::IntoIter {
        let keys: Vec<K> = self.list.iter_entries().map(|(_, k)| k.clone()).collect();
        IntoIter {
            inner: keys.into_iter(),
        }
    }
}

impl<'a, K> IntoIterator for &'a GhostList<K>
where
    K: Eq + Hash + Clone,
{
    type Item = &'a K;
    type IntoIter = Iter<'a, K>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K> PartialEq for GhostList<K>
where
    K: Eq + Hash + Clone,
{
    fn eq(&self, other: &Self) -> bool {
        self.capacity == other.capacity
            && self.len() == other.len()
            && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<K> Eq for GhostList<K> where K: Eq + Hash + Clone {}

impl<K> Extend<K> for GhostList<K>
where
    K: Eq + Hash + Clone,
{
    fn extend<I: IntoIterator<Item = K>>(&mut self, iter: I) {
        for key in iter {
            self.record(key);
        }
    }
}

impl<K> Default for GhostList<K>
where
    K: Eq + Hash + Clone,
{
    /// Creates an empty ghost list with zero capacity (no-op mode).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::GhostList;
    ///
    /// let ghost: GhostList<String> = GhostList::default();
    /// assert_eq!(ghost.capacity(), 0);
    /// assert!(ghost.is_empty());
    /// ```
    fn default() -> Self {
        Self::new(0)
    }
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
    /// # Complexity
    ///
    /// O(1) average case, O(n) worst case due to hash collisions.
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
    /// If the key is already present, it is promoted to MRU position and returns `None`.
    /// If at capacity, the LRU key is evicted before inserting and returned.
    ///
    /// # Returns
    ///
    /// - `Some(evicted_key)` if a key was evicted to make room
    /// - `None` if the key was already present (promoted) or capacity not reached
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::GhostList;
    ///
    /// let mut ghost = GhostList::new(2);
    ///
    /// assert_eq!(ghost.record("a"), None);  // No eviction
    /// assert_eq!(ghost.record("b"), None);  // No eviction
    /// assert!(ghost.contains(&"a"));
    /// assert!(ghost.contains(&"b"));
    ///
    /// // At capacity: "a" is LRU, will be evicted
    /// assert_eq!(ghost.record("c"), Some("a"));  // Returns evicted key
    /// assert!(!ghost.contains(&"a"));  // Evicted
    /// assert!(ghost.contains(&"b"));
    /// assert!(ghost.contains(&"c"));
    ///
    /// // Re-recording "b" promotes it to MRU (no eviction)
    /// assert_eq!(ghost.record("b"), None);  // Already present
    /// assert_eq!(ghost.record("d"), Some("c"));  // Evicts "c"
    /// assert!(ghost.contains(&"b"));   // Survived (was MRU)
    /// assert!(!ghost.contains(&"c"));  // Evicted (was LRU)
    /// ```
    pub fn record(&mut self, key: K) -> Option<K> {
        if self.capacity == 0 {
            return None;
        }

        if let Some(&id) = self.index.get(&key) {
            self.list.move_to_front(id);
            return None;
        }

        let evicted = if self.list.len() >= self.capacity {
            let old_key = self.list.pop_back()?;
            self.index.remove(&old_key);
            Some(old_key)
        } else {
            None
        };

        let id = self.list.push_front(key.clone());
        self.index.insert(key, id);

        evicted
    }

    /// Records a batch of keys; returns number of keys processed.
    ///
    /// Note: All keys are processed (count equals input length), since `record` is infallible.
    /// Duplicates are automatically promoted to MRU rather than inserted twice.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::GhostList;
    ///
    /// let mut ghost = GhostList::new(10);
    ///
    /// let evicted = vec!["page_1", "page_2", "page_3"];
    /// let count = ghost.record_batch(&evicted);
    ///
    /// assert_eq!(count, 3);
    /// assert_eq!(ghost.len(), 3);
    /// ```
    pub fn record_batch<'a, I>(&mut self, keys: I) -> usize
    where
        I: IntoIterator<Item = &'a K>,
        K: 'a,
    {
        let mut count = 0;
        for key in keys {
            self.record(key.clone());
            count += 1;
        }
        count
    }

    /// Removes `key` from the ghost list; returns `true` if it was present.
    ///
    /// Typically called after a ghost hit to prevent double-counting.
    ///
    /// # Complexity
    ///
    /// O(1) average case, O(n) worst case due to hash collisions.
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

    /// Removes and returns the LRU (least recently used) key.
    ///
    /// Returns `None` if the ghost list is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::GhostList;
    ///
    /// let mut ghost = GhostList::new(3);
    /// ghost.record("a");
    /// ghost.record("b");
    /// ghost.record("c");
    ///
    /// // "a" is LRU (inserted first, never promoted)
    /// assert_eq!(ghost.evict_lru(), Some("a"));
    /// assert!(!ghost.contains(&"a"));
    /// assert_eq!(ghost.len(), 2);
    ///
    /// assert_eq!(ghost.evict_lru(), Some("b"));
    /// assert_eq!(ghost.evict_lru(), Some("c"));
    /// assert_eq!(ghost.evict_lru(), None);  // Empty
    /// ```
    pub fn evict_lru(&mut self) -> Option<K> {
        let key = self.list.pop_back()?;
        self.index.remove(&key);
        Some(key)
    }

    /// Removes a batch of keys; returns number of keys actually removed.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::GhostList;
    ///
    /// let mut ghost = GhostList::new(10);
    /// ghost.record_batch(&["a", "b", "c"]);
    ///
    /// // Remove some keys (including one that doesn't exist)
    /// let removed = ghost.remove_batch(&["a", "c", "missing"]);
    ///
    /// assert_eq!(removed, 2);  // Only "a" and "c" were removed
    /// assert!(ghost.contains(&"b"));
    /// ```
    pub fn remove_batch<'a, I>(&mut self, keys: I) -> usize
    where
        I: IntoIterator<Item = &'a K>,
        K: 'a,
    {
        let mut removed = 0;
        for key in keys {
            if self.remove(key) {
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
    /// let keys: Vec<_> = (0..50).map(|i| format!("key_{}", i)).collect();
    /// ghost.record_batch(&keys);
    ///
    /// ghost.clear_shrink();
    /// assert!(ghost.is_empty());
    /// ```
    pub fn clear_shrink(&mut self) {
        self.clear();
        self.list.clear_shrink();
        self.index.shrink_to_fit();
    }

    /// Returns a conservative lower-bound memory footprint in bytes.
    ///
    /// This estimate includes the struct size, intrusive list overhead,
    /// and HashMap entry storage. It underestimates actual memory usage by
    /// approximately 20-30% for small capacities due to:
    /// - HashMap control bytes and alignment padding
    /// - Key storage overhead in both structures
    /// - Internal capacity buffering
    ///
    /// Use this for rough capacity planning, not precise memory accounting.
    /// For accurate measurements, use a memory profiler.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::GhostList;
    ///
    /// let ghost: GhostList<u64> = GhostList::new(100);
    /// let bytes = ghost.approx_bytes();
    /// assert!(bytes > 0);
    ///
    /// // Approximate memory per entry
    /// let per_entry = bytes / 100.max(1);
    /// println!("~{} bytes per entry (lower bound)", per_entry);
    /// ```
    pub fn approx_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.list.approx_bytes()
            + self.index.capacity() * std::mem::size_of::<(K, SlotId)>()
    }

    /// Returns an iterator over keys in MRU -> LRU order.
    ///
    /// Useful for observability and advanced use cases. However, prefer point lookups
    /// via `contains()` for performance-critical paths.
    ///
    /// # Complexity
    ///
    /// Iteration is O(n) where n is the number of keys tracked.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::GhostList;
    ///
    /// let mut ghost = GhostList::new(3);
    /// ghost.record("a");
    /// ghost.record("b");
    /// ghost.record("c");
    ///
    /// let keys: Vec<_> = ghost.iter().collect();
    /// assert_eq!(keys, vec![&"c", &"b", &"a"]);  // MRU to LRU
    /// ```
    pub fn iter(&self) -> Iter<'_, K> {
        fn extract_key<K>((_, key): (SlotId, &K)) -> &K {
            key
        }
        Iter {
            inner: self.list.iter_entries().map(extract_key),
        }
    }

    /// Returns an iterator over keys in MRU -> LRU order.
    ///
    /// This is a semantic alias for [`iter()`](Self::iter) that makes the intent clearer,
    /// matching the convention of `HashMap::keys()`.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::GhostList;
    ///
    /// let mut ghost = GhostList::new(3);
    /// ghost.record("a");
    /// ghost.record("b");
    ///
    /// // More explicit than iter()
    /// for key in ghost.keys() {
    ///     println!("Ghost entry: {}", key);
    /// }
    /// ```
    pub fn keys(&self) -> Iter<'_, K> {
        self.iter()
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
    /// Returns a reference to the LRU (least recently used) key, if any.
    ///
    /// Useful for debugging and observability.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::GhostList;
    ///
    /// let mut ghost = GhostList::new(3);
    /// ghost.record("a");
    /// ghost.record("b");
    /// ghost.record("c");
    ///
    /// assert_eq!(ghost.debug_peek_lru(), Some(&"a"));
    /// ```
    pub fn debug_peek_lru(&self) -> Option<&K> {
        self.list.iter_entries().last().map(|(_, k)| k)
    }

    #[cfg(any(test, debug_assertions))]
    /// Returns a reference to the MRU (most recently used) key, if any.
    ///
    /// Useful for debugging and observability.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::GhostList;
    ///
    /// let mut ghost = GhostList::new(3);
    /// ghost.record("a");
    /// ghost.record("b");
    /// ghost.record("c");
    ///
    /// assert_eq!(ghost.debug_peek_mru(), Some(&"c"));
    /// ```
    pub fn debug_peek_mru(&self) -> Option<&K> {
        self.list.iter_entries().next().map(|(_, k)| k)
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
        assert_eq!(ghost.record_batch(&["a", "b", "c"]), 3);
        assert_eq!(ghost.remove_batch(&["b", "d"]), 1);
        assert!(ghost.contains(&"a"));
        assert!(ghost.contains(&"c"));
        assert!(!ghost.contains(&"b"));
    }

    #[test]
    fn ghost_list_iter() {
        let mut ghost = GhostList::new(5);
        ghost.record("a");
        ghost.record("b");
        ghost.record("c");

        let keys: Vec<_> = ghost.iter().cloned().collect();
        assert_eq!(keys, vec!["c", "b", "a"]); // MRU to LRU

        // Verify iterator works on empty list
        ghost.clear();
        assert_eq!(ghost.iter().count(), 0);
    }

    #[test]
    fn ghost_list_debug_peek_lru_mru() {
        let mut ghost = GhostList::new(3);
        assert_eq!(ghost.debug_peek_lru(), None);
        assert_eq!(ghost.debug_peek_mru(), None);

        ghost.record("a");
        assert_eq!(ghost.debug_peek_lru(), Some(&"a"));
        assert_eq!(ghost.debug_peek_mru(), Some(&"a"));

        ghost.record("b");
        ghost.record("c");
        assert_eq!(ghost.debug_peek_lru(), Some(&"a")); // Oldest
        assert_eq!(ghost.debug_peek_mru(), Some(&"c")); // Newest

        // Promote "a" to MRU
        assert_eq!(ghost.record("a"), None); // No eviction (already present)
        assert_eq!(ghost.debug_peek_lru(), Some(&"b"));
        assert_eq!(ghost.debug_peek_mru(), Some(&"a"));
    }

    #[test]
    fn ghost_list_record_returns_evicted() {
        let mut ghost = GhostList::new(2);

        // No eviction when capacity not reached
        assert_eq!(ghost.record("a"), None);
        assert_eq!(ghost.record("b"), None);
        assert_eq!(ghost.len(), 2);

        // Eviction when at capacity
        assert_eq!(ghost.record("c"), Some("a")); // "a" was LRU
        assert!(!ghost.contains(&"a"));
        assert!(ghost.contains(&"b"));
        assert!(ghost.contains(&"c"));

        // Re-recording existing key returns None (no eviction)
        assert_eq!(ghost.record("b"), None);
        assert_eq!(ghost.record("d"), Some("c")); // "c" was LRU

        // Zero-capacity ghost list never evicts
        let mut zero_ghost = GhostList::new(0);
        assert_eq!(zero_ghost.record("x"), None);
        assert!(zero_ghost.is_empty());
    }

    #[test]
    fn ghost_list_keys_alias() {
        let mut ghost = GhostList::new(3);
        ghost.record("a");
        ghost.record("b");
        ghost.record("c");

        // keys() should work identically to iter()
        let keys_via_keys: Vec<_> = ghost.keys().cloned().collect();
        let keys_via_iter: Vec<_> = ghost.iter().cloned().collect();
        assert_eq!(keys_via_keys, keys_via_iter);
        assert_eq!(keys_via_keys, vec!["c", "b", "a"]);
    }

    #[test]
    fn ghost_list_default() {
        let ghost: GhostList<String> = GhostList::default();
        assert_eq!(ghost.capacity(), 0);
        assert!(ghost.is_empty());
        assert_eq!(ghost.len(), 0);
    }

    #[test]
    fn ghost_list_clone() {
        let mut ghost = GhostList::new(3);
        ghost.record("a");
        ghost.record("b");
        ghost.record("c");

        let cloned = ghost.clone();
        assert_eq!(cloned.len(), ghost.len());
        assert_eq!(cloned.capacity(), ghost.capacity());
        assert!(cloned.contains(&"a"));
        assert!(cloned.contains(&"b"));
        assert!(cloned.contains(&"c"));

        // Verify they are independent
        ghost.record("d");
        assert!(ghost.contains(&"d"));
        assert!(!cloned.contains(&"d"));
    }

    // -------------------------------------------------------------------------
    // IntoIterator (consuming)
    // -------------------------------------------------------------------------

    #[test]
    fn ghost_list_into_iter_yields_mru_to_lru() {
        let mut ghost = GhostList::new(3);
        ghost.record("a");
        ghost.record("b");
        ghost.record("c");

        let keys: Vec<_> = ghost.into_iter().collect();
        assert_eq!(keys, vec!["c", "b", "a"]);
    }

    #[test]
    fn ghost_list_into_iter_empty() {
        let ghost: GhostList<&str> = GhostList::new(5);
        let keys: Vec<_> = ghost.into_iter().collect();
        assert!(keys.is_empty());
    }

    #[test]
    fn ghost_list_into_iter_size_hint() {
        let mut ghost = GhostList::new(4);
        ghost.record("a");
        ghost.record("b");
        ghost.record("c");

        let iter = ghost.into_iter();
        assert_eq!(iter.size_hint(), (3, Some(3)));
    }

    #[test]
    fn ghost_list_into_iter_exact_size() {
        let mut ghost = GhostList::new(3);
        ghost.record("x");
        ghost.record("y");

        let mut iter = ghost.into_iter();
        assert_eq!(iter.len(), 2);
        iter.next();
        assert_eq!(iter.len(), 1);
        iter.next();
        assert_eq!(iter.len(), 0);
    }

    // -------------------------------------------------------------------------
    // IntoIterator (borrowed, &GhostList)
    // -------------------------------------------------------------------------

    #[test]
    fn ghost_list_ref_into_iter_yields_mru_to_lru() {
        let mut ghost = GhostList::new(3);
        ghost.record("a");
        ghost.record("b");
        ghost.record("c");

        let keys: Vec<_> = (&ghost).into_iter().cloned().collect();
        assert_eq!(keys, vec!["c", "b", "a"]);
        // Ghost list is still usable after borrowed iteration
        assert_eq!(ghost.len(), 3);
    }

    #[test]
    fn ghost_list_ref_into_iter_via_for_loop() {
        let mut ghost = GhostList::new(3);
        ghost.record(1u32);
        ghost.record(2u32);
        ghost.record(3u32);

        let mut seen = Vec::new();
        for key in &ghost {
            seen.push(*key);
        }
        assert_eq!(seen, vec![3u32, 2, 1]); // MRU to LRU
    }

    // -------------------------------------------------------------------------
    // PartialEq / Eq
    // -------------------------------------------------------------------------

    #[test]
    fn ghost_list_equal_same_content_and_capacity() {
        let mut a = GhostList::new(3);
        a.record("x");
        a.record("y");

        let mut b = GhostList::new(3);
        b.record("x");
        b.record("y");

        assert_eq!(a, b);
    }

    #[test]
    fn ghost_list_not_equal_different_order() {
        let mut a = GhostList::new(3);
        a.record("x");
        a.record("y");

        let mut b = GhostList::new(3);
        b.record("y");
        b.record("x");

        // Both contain the same keys but in different MRU order
        assert_ne!(a, b);
    }

    #[test]
    fn ghost_list_not_equal_different_capacity() {
        let mut a = GhostList::new(3);
        a.record("x");

        let mut b = GhostList::new(5);
        b.record("x");

        assert_ne!(a, b);
    }

    #[test]
    fn ghost_list_not_equal_different_content() {
        let mut a = GhostList::new(3);
        a.record("x");
        a.record("y");

        let mut b = GhostList::new(3);
        b.record("x");
        b.record("z");

        assert_ne!(a, b);
    }

    #[test]
    fn ghost_list_equal_empty_lists_same_capacity() {
        let a: GhostList<&str> = GhostList::new(10);
        let b: GhostList<&str> = GhostList::new(10);
        assert_eq!(a, b);
    }

    #[test]
    fn ghost_list_partial_eq_is_reflexive() {
        let mut ghost = GhostList::new(3);
        ghost.record("a");
        ghost.record("b");
        assert_eq!(ghost, ghost.clone());
    }

    // -------------------------------------------------------------------------
    // Extend
    // -------------------------------------------------------------------------

    #[test]
    fn ghost_list_extend_records_all_keys() {
        let mut ghost = GhostList::new(5);
        ghost.extend(["a", "b", "c"]);

        assert_eq!(ghost.len(), 3);
        assert!(ghost.contains(&"a"));
        assert!(ghost.contains(&"b"));
        assert!(ghost.contains(&"c"));
    }

    #[test]
    fn ghost_list_extend_respects_capacity() {
        let mut ghost = GhostList::new(2);
        ghost.extend(["a", "b", "c", "d"]);

        // Capacity is 2; oldest keys evicted
        assert_eq!(ghost.len(), 2);
        assert!(ghost.contains(&"c"));
        assert!(ghost.contains(&"d"));
        assert!(!ghost.contains(&"a"));
        assert!(!ghost.contains(&"b"));
    }

    #[test]
    fn ghost_list_extend_promotes_existing_keys() {
        let mut ghost = GhostList::new(3);
        ghost.record("a");
        ghost.record("b");
        ghost.record("c");

        // Extending with existing key "a" promotes it to MRU
        ghost.extend(["a"]);
        let keys: Vec<_> = ghost.iter().cloned().collect();
        assert_eq!(keys[0], "a"); // "a" is now MRU
        assert_eq!(ghost.len(), 3);
    }

    #[test]
    fn ghost_list_extend_from_iterator() {
        let mut ghost = GhostList::new(10);
        let keys = vec!["p", "q", "r", "s"];
        ghost.extend(keys);

        assert_eq!(ghost.len(), 4);
        assert!(ghost.contains(&"p"));
        assert!(ghost.contains(&"s"));
    }

    // -------------------------------------------------------------------------
    // evict_lru
    // -------------------------------------------------------------------------

    #[test]
    fn ghost_list_evict_lru_removes_oldest() {
        let mut ghost = GhostList::new(3);
        ghost.record("a");
        ghost.record("b");
        ghost.record("c");

        // LRU is "a" (inserted first, never promoted)
        assert_eq!(ghost.evict_lru(), Some("a"));
        assert!(!ghost.contains(&"a"));
        assert_eq!(ghost.len(), 2);

        assert_eq!(ghost.evict_lru(), Some("b"));
        assert_eq!(ghost.evict_lru(), Some("c"));
        assert_eq!(ghost.evict_lru(), None); // Empty
        assert!(ghost.is_empty());
    }

    #[test]
    fn ghost_list_evict_lru_empty_returns_none() {
        let mut ghost: GhostList<&str> = GhostList::new(5);
        assert_eq!(ghost.evict_lru(), None);
    }

    #[test]
    fn ghost_list_evict_lru_after_promotion() {
        let mut ghost = GhostList::new(3);
        ghost.record("a");
        ghost.record("b");
        ghost.record("c");

        // Promote "a" to MRU — now "b" is LRU
        ghost.record("a");
        assert_eq!(ghost.evict_lru(), Some("b"));
        assert!(ghost.contains(&"a"));
        assert!(ghost.contains(&"c"));
    }

    // -------------------------------------------------------------------------
    // Debug impls
    // -------------------------------------------------------------------------

    #[test]
    fn ghost_list_iter_debug() {
        let mut ghost = GhostList::new(3);
        ghost.record("a");
        ghost.record("b");

        let iter = ghost.iter();
        let debug_str = format!("{:?}", iter);
        assert!(debug_str.contains("Iter"));
    }

    #[test]
    fn ghost_list_into_iter_debug() {
        let mut ghost = GhostList::new(3);
        ghost.record("a");

        let iter = ghost.into_iter();
        let debug_str = format!("{:?}", iter);
        assert!(debug_str.contains("IntoIter"));
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // =============================================================================
    // Property Tests - Core Invariants
    // =============================================================================

    proptest! {
        /// Property: Invariants hold after any sequence of operations
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_invariants_always_hold(
            capacity in 1usize..20,
            ops in prop::collection::vec((0u8..3, any::<u32>()), 0..50)
        ) {
            let mut ghost: GhostList<u32> = GhostList::new(capacity);

            for (op, key) in ops {
                match op % 3 {
                    0 => { ghost.record(key); }
                    1 => { ghost.remove(&key); }
                    2 => { let _ = ghost.contains(&key); }
                    _ => unreachable!(),
                }

                ghost.debug_validate_invariants();
            }
        }

        /// Property: len() never exceeds capacity
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_len_never_exceeds_capacity(
            capacity in 1usize..30,
            keys in prop::collection::vec(any::<u32>(), 0..100)
        ) {
            let mut ghost: GhostList<u32> = GhostList::new(capacity);

            for key in keys {
                ghost.record(key);
                prop_assert!(ghost.len() <= capacity);
            }
        }

        /// Property: Empty state is consistent
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_empty_state_consistent(
            capacity in 1usize..20,
            keys in prop::collection::vec(any::<u32>(), 1..20)
        ) {
            let mut ghost: GhostList<u32> = GhostList::new(capacity);

            for key in keys {
                ghost.record(key);
            }

            ghost.clear();

            prop_assert!(ghost.is_empty());
            prop_assert_eq!(ghost.len(), 0);
            prop_assert_eq!(ghost.capacity(), capacity);
        }
    }

    // =============================================================================
    // Property Tests - LRU Eviction Order
    // =============================================================================

    proptest! {
        /// Property: LRU eviction - oldest keys evicted first
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_lru_eviction_order(
            capacity in 2usize..10,
            keys in prop::collection::vec(0u32..50, 1..30)
        ) {
            let mut ghost: GhostList<u32> = GhostList::new(capacity);
            let mut reference: std::collections::VecDeque<u32> = std::collections::VecDeque::new();

            for key in keys {
                ghost.record(key);

                if let Some(pos) = reference.iter().position(|&k| k == key) {
                    reference.remove(pos);
                } else if reference.len() >= capacity {
                    reference.pop_back();
                }
                reference.push_front(key);
            }

            prop_assert_eq!(ghost.len(), reference.len());
            for &ref_key in &reference {
                prop_assert!(ghost.contains(&ref_key));
            }

            let snapshot = ghost.debug_snapshot_keys();
            prop_assert_eq!(snapshot, reference.iter().copied().collect::<Vec<_>>());

            ghost.debug_validate_invariants();
        }

        /// Property: Recording existing key doesn't increase length
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_rerecord_no_length_increase(
            capacity in 2usize..10,
            key in any::<u32>()
        ) {
            let mut ghost: GhostList<u32> = GhostList::new(capacity);

            ghost.record(key);
            let len_after_first = ghost.len();

            ghost.record(key);
            let len_after_second = ghost.len();

            prop_assert_eq!(len_after_first, len_after_second);
            prop_assert_eq!(len_after_first, 1);
        }
    }

    // =============================================================================
    // Property Tests - Promotion to MRU
    // =============================================================================

    proptest! {
        /// Property: Re-recording promotes to MRU position
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_rerecord_promotes_to_mru(
            capacity in 2usize..10,
            keys in prop::collection::vec(0u32..20, 2..10)
        ) {
            prop_assume!(keys.len() >= 2);
            let mut ghost: GhostList<u32> = GhostList::new(capacity);

            // Record keys
            for &key in &keys {
                ghost.record(key);
            }

            if ghost.is_empty() {
                return Ok(());
            }

            // Get snapshot to find a key that's in the list
            let snapshot = ghost.debug_snapshot_keys();
            if snapshot.is_empty() {
                return Ok(());
            }

            let promoted_key = snapshot[snapshot.len() - 1]; // Pick LRU key

            // Re-record to promote
            ghost.record(promoted_key);

            // Should now be at MRU position (index 0 in snapshot)
            let new_snapshot = ghost.debug_snapshot_keys();
            if !new_snapshot.is_empty() {
                prop_assert_eq!(new_snapshot[0], promoted_key);
            }
        }

        /// Property: Promoted keys survive longer than unpromoted
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_promoted_keys_survive_longer(
            capacity in 3usize..8,
            fill_keys in prop::collection::vec(0u32..10, 3..8)
        ) {
            prop_assume!(fill_keys.len() >= capacity);
            let mut ghost: GhostList<u32> = GhostList::new(capacity);

            // Fill to capacity with unique keys
            let mut unique = Vec::new();
            for &key in &fill_keys {
                if !unique.contains(&key) && unique.len() < capacity {
                    ghost.record(key);
                    unique.push(key);
                }
            }

            if unique.len() < capacity {
                return Ok(());
            }

            // Pick LRU key and promote it
            let snapshot = ghost.debug_snapshot_keys();
            let promoted_key = snapshot[snapshot.len() - 1];
            ghost.record(promoted_key);

            // Record enough new keys to cause evictions
            for i in 100..(100 + capacity - 1) {
                ghost.record(i as u32);
            }

            // Promoted key should still be present
            prop_assert!(ghost.contains(&promoted_key));
        }
    }

    // =============================================================================
    // Property Tests - Contains and Remove
    // =============================================================================

    proptest! {
        /// Property: recorded keys are present, removed keys are not
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_contains_after_record(
            capacity in 1usize..20,
            key in any::<u32>()
        ) {
            let mut ghost: GhostList<u32> = GhostList::new(capacity);

            ghost.record(key);
            prop_assert!(ghost.contains(&key));

            ghost.remove(&key);
            prop_assert!(!ghost.contains(&key));
        }

        /// Property: remove returns true only for present keys
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_remove_returns_correct_bool(
            capacity in 1usize..10,
            keys in prop::collection::vec(any::<u32>(), 1..20)
        ) {
            let mut ghost: GhostList<u32> = GhostList::new(capacity);

            for &key in &keys {
                ghost.record(key);
            }

            for &key in &keys {
                let was_present = ghost.contains(&key);
                let remove_result = ghost.remove(&key);

                if was_present {
                    prop_assert!(remove_result);
                    prop_assert!(!ghost.contains(&key));
                }

                // Removing again should return false
                prop_assert!(!ghost.remove(&key));
            }
        }
    }

    // =============================================================================
    // Property Tests - Clear Operations
    // =============================================================================

    proptest! {
        /// Property: clear resets to empty state
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_clear_resets_state(
            capacity in 1usize..20,
            keys in prop::collection::vec(any::<u32>(), 1..30)
        ) {
            let mut ghost: GhostList<u32> = GhostList::new(capacity);

            for key in keys {
                ghost.record(key);
            }

            ghost.clear();

            prop_assert!(ghost.is_empty());
            prop_assert_eq!(ghost.len(), 0);
            prop_assert_eq!(ghost.capacity(), capacity);
            ghost.debug_validate_invariants();
        }

        /// Property: clear_shrink behaves like clear
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_clear_shrink_same_as_clear(
            capacity in 1usize..20,
            keys in prop::collection::vec(any::<u32>(), 1..30)
        ) {
            let mut ghost1: GhostList<u32> = GhostList::new(capacity);
            let mut ghost2: GhostList<u32> = GhostList::new(capacity);

            for &key in &keys {
                ghost1.record(key);
                ghost2.record(key);
            }

            ghost1.clear();
            ghost2.clear_shrink();

            prop_assert_eq!(ghost1.len(), ghost2.len());
            prop_assert_eq!(ghost1.is_empty(), ghost2.is_empty());
            prop_assert_eq!(ghost1.capacity(), ghost2.capacity());
        }

        /// Property: usable after clear
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_usable_after_clear(
            capacity in 1usize..10,
            keys1 in prop::collection::vec(any::<u32>(), 1..20),
            keys2 in prop::collection::vec(any::<u32>(), 1..20)
        ) {
            let mut ghost: GhostList<u32> = GhostList::new(capacity);

            for key in keys1 {
                ghost.record(key);
            }

            ghost.clear();

            for &key in &keys2 {
                ghost.record(key);
            }

            let unique_keys2: std::collections::HashSet<_> = keys2.into_iter().collect();
            let expected_len = unique_keys2.len().min(capacity);
            prop_assert_eq!(ghost.len(), expected_len);
        }
    }

    // =============================================================================
    // Property Tests - Zero Capacity
    // =============================================================================

    proptest! {
        /// Property: zero capacity ghost list is always empty (no-op)
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_zero_capacity_always_empty(
            keys in prop::collection::vec(any::<u32>(), 0..30)
        ) {
            let mut ghost: GhostList<u32> = GhostList::new(0);

            for key in keys {
                ghost.record(key);
                prop_assert!(ghost.is_empty());
                prop_assert_eq!(ghost.len(), 0);
                prop_assert_eq!(ghost.capacity(), 0);
                prop_assert!(!ghost.contains(&key));
            }
        }
    }

    // =============================================================================
    // Property Tests - Batch Operations
    // =============================================================================

    proptest! {
        /// Property: record_batch records all keys
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_record_batch_records_all(
            capacity in 5usize..20,
            keys in prop::collection::vec(0u32..20, 1..10)
        ) {
            let mut ghost: GhostList<u32> = GhostList::new(capacity);

            let count = ghost.record_batch(&keys);

            prop_assert_eq!(count, keys.len());

            let unique_keys: std::collections::HashSet<_> = keys.into_iter().collect();
            let expected_len = unique_keys.len().min(capacity);
            prop_assert_eq!(ghost.len(), expected_len);
        }

        /// Property: remove_batch removes only present keys
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_remove_batch_only_present(
            capacity in 5usize..20,
            record_keys in prop::collection::vec(0u32..10, 1..10),
            remove_keys in prop::collection::vec(0u32..20, 1..10)
        ) {
            let mut ghost: GhostList<u32> = GhostList::new(capacity);

            ghost.record_batch(&record_keys);

            let removed = ghost.remove_batch(&remove_keys);

            let record_set: std::collections::HashSet<_> = record_keys.into_iter().collect();
            let remove_set: std::collections::HashSet<_> = remove_keys.into_iter().collect();

            // Count how many keys to remove were actually present
            let mut expected_removed = 0;
            for key in remove_set {
                if record_set.contains(&key) && !ghost.is_empty() {
                    expected_removed += 1;
                }
            }

            // removed count should match or be less (due to LRU evictions before removal)
            prop_assert!(removed <= expected_removed || removed <= record_set.len());
        }
    }

    // =============================================================================
    // Property Tests - Reference Implementation Equivalence
    // =============================================================================

    proptest! {
        /// Property: Behavior matches reference VecDeque implementation
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_matches_reference_implementation(
            capacity in 2usize..10,
            keys in prop::collection::vec(0u32..20, 0..30)
        ) {
            let mut ghost: GhostList<u32> = GhostList::new(capacity);
            let mut reference: std::collections::VecDeque<u32> = std::collections::VecDeque::new();

            for key in keys {
                // Update ghost list
                ghost.record(key);

                // Update reference implementation
                if let Some(pos) = reference.iter().position(|&k| k == key) {
                    reference.remove(pos);
                } else if reference.len() >= capacity {
                    reference.pop_back(); // Remove LRU
                }
                reference.push_front(key); // Add at MRU

                // Verify length matches
                prop_assert_eq!(ghost.len(), reference.len());

                // Verify all keys in reference are in ghost list
                for &ref_key in &reference {
                    prop_assert!(ghost.contains(&ref_key));
                }

                // Verify snapshot matches reference
                let snapshot = ghost.debug_snapshot_keys();
                prop_assert_eq!(snapshot.len(), reference.len());
                for (snap_key, ref_key) in snapshot.iter().zip(reference.iter()) {
                    prop_assert_eq!(snap_key, ref_key);
                }
            }
        }
    }
}
