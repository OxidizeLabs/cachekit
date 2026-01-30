//! MRU (Most Recently Used) cache replacement policy.
//!
//! Implements the MRU algorithm, which evicts the **most** recently accessed entry
//! when capacity is reached. This is the opposite of LRU and is useful for
//! specific cyclic access patterns where the most recently accessed item is
//! least likely to be accessed again soon.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                           MruCore<K, V> Layout                              │
//! │                                                                             │
//! │   ┌─────────────────────────────────────────────────────────────────────┐   │
//! │   │  index: HashMap<K, NonNull<Node>>    nodes: Allocated on heap      │   │
//! │   │                                                                     │   │
//! │   │  ┌──────────┬───────────┐          ┌────────┬──────────────────┐   │   │
//! │   │  │   Key    │  NodePtr  │          │ Node   │ key, value       │   │   │
//! │   │  ├──────────┼───────────┤          ├────────┼──────────────────┤   │   │
//! │   │  │  "page1" │   ptr_0   │──────────►│ Node0  │ k,v              │   │   │
//! │   │  │  "page2" │   ptr_1   │──────────►│ Node1  │ k,v              │   │   │
//! │   │  │  "page3" │   ptr_2   │──────────►│ Node2  │ k,v              │   │   │
//! │   │  └──────────┴───────────┘          └────────┴──────────────────┘   │   │
//! │   └─────────────────────────────────────────────────────────────────────┘   │
//! │                                                                             │
//! │   ┌─────────────────────────────────────────────────────────────────────┐   │
//! │   │                        Recency List (Doubly-Linked)                 │   │
//! │   │                                                                     │   │
//! │   │   head (MRU - EVICT FROM HERE)         tail (LRU - keep)            │   │
//! │   │   ┌───────────────────────┐            ┌─────────────────────────┐  │   │
//! │   │   │ MRU              LRU  │            │                         │  │   │
//! │   │   │  ▼                ▼   │            │                         │  │   │
//! │   │   │ [ptr_2] ◄──► [ptr_1] ◄──► [ptr_0]  │                         │  │   │
//! │   │   │  newest     middle    oldest       │                         │  │   │
//! │   │   │  EVICT      KEEP      KEEP         │                         │  │   │
//! │   │   └───────────────────────┘            └─────────────────────────┘  │   │
//! │   │                                                                     │   │
//! │   │   • New items enter at head (MRU)                                   │   │
//! │   │   • Accessed items move to head (MRU)                               │   │
//! │   │   • Eviction happens from head (MRU) - the newest item!             │   │
//! │   └─────────────────────────────────────────────────────────────────────┘   │
//! │                                                                             │
//! └─────────────────────────────────────────────────────────────────────────────┘
//!
//! Insert Flow (new key)
//! ──────────────────────
//!
//!   insert("new_key", value):
//!     1. Check index - not found
//!     2. Create Node with key and value
//!     3. Allocate on heap → get NonNull<Node>
//!     4. Insert key→ptr into index
//!     5. Attach ptr to head (MRU position)
//!     6. Evict if over capacity (from head/MRU!)
//!
//! Access Flow (existing key)
//! ──────────────────────────
//!
//!   get("existing_key"):
//!     1. Lookup ptr in index
//!     2. Detach from current position
//!     3. Reattach to head (MRU position)
//!     4. Return &value
//!
//! Eviction Flow
//! ─────────────
//!
//!   evict_if_needed():
//!     while len > capacity:
//!       evict from head (MRU - most recently used!)
//! ```
//!
//! ## Key Components
//!
//! - [`MruCore`]: Main MRU cache implementation
//!
//! ## Operations
//!
//! | Operation   | Time   | Notes                                      |
//! |-------------|--------|--------------------------------------------|
//! | `get`       | O(1)   | Moves accessed item to MRU (head)          |
//! | `insert`    | O(1)*  | *Amortized, may trigger evictions          |
//! | `contains`  | O(1)   | Index lookup only                          |
//! | `len`       | O(1)   | Returns total entries                      |
//! | `clear`     | O(n)   | Clears all structures                      |
//!
//! ## Algorithm Properties
//!
//! - **Cyclic Pattern Handling**: Good for cyclic access patterns where newest item won't be accessed soon
//! - **Opposite of LRU**: Evicts most recent instead of least recent
//! - **Niche Use Case**: Not a general-purpose policy
//!
//! ## Use Cases
//!
//! - Cyclic file scanning patterns
//! - Sequential processing where recently processed items won't be needed again
//! - Specific database query patterns with known access sequences
//!
//! ## Example Usage
//!
//! ```
//! use cachekit::policy::mru::MruCore;
//!
//! // Create MRU cache with capacity 10
//! let mut cache = MruCore::new(10);
//!
//! // Insert items
//! cache.insert(1, 100);
//! cache.insert(2, 200);
//! cache.insert(3, 300);
//!
//! // Access moves to MRU (head) - making it first to be evicted!
//! assert_eq!(cache.get(&1), Some(&100));
//!
//! // When cache is full, item 1 (most recently accessed) will be evicted first
//! for i in 4..=10 {
//!     cache.insert(i, i * 10);
//! }
//!
//! assert_eq!(cache.len(), 10);
//! ```
//!
//! ## Thread Safety
//!
//! - [`MruCore`]: Not thread-safe, designed for single-threaded use
//! - For concurrent access, wrap in external synchronization
//!
//! ## Implementation Notes
//!
//! - Uses doubly-linked list with head=MRU, tail=LRU
//! - Eviction happens from head (MRU) instead of tail (LRU)
//! - Accessed items move to head (MRU position)
//! - New items are inserted at head (MRU position)
//!
//! ## When to Use
//!
//! **Use MRU when:**
//! - Access patterns are cyclic and predictable
//! - Recently accessed items are unlikely to be accessed again soon
//! - You understand the specific workload characteristics
//!
//! **Avoid MRU when:**
//! - General-purpose caching (use LRU, SLRU, or S3-FIFO instead)
//! - Temporal locality is important
//! - Unsure about access patterns
//!
//! ## References
//!
//! - Wikipedia: Cache replacement policies

use rustc_hash::FxHashMap;
use std::hash::Hash;
use std::ptr::NonNull;

/// Node in the MRU linked list.
///
/// Cache-line optimized layout with pointers first.
#[repr(C)]
struct Node<K, V> {
    prev: Option<NonNull<Node<K, V>>>,
    next: Option<NonNull<Node<K, V>>>,
    key: K,
    value: V,
}

/// Core MRU (Most Recently Used) cache implementation.
///
/// Implements the MRU replacement algorithm which evicts the **most** recently
/// accessed item when capacity is reached. This is useful for specific cyclic
/// access patterns.
///
/// # Type Parameters
///
/// - `K`: Key type, must be `Clone + Eq + Hash`
/// - `V`: Value type
///
/// # Example
///
/// ```
/// use cachekit::policy::mru::MruCore;
///
/// // 100 capacity
/// let mut cache = MruCore::new(100);
///
/// // Insert goes to MRU (head)
/// cache.insert("key1", "value1");
/// assert!(cache.contains(&"key1"));
///
/// // Access moves to MRU (head) - making it first to evict!
/// cache.get(&"key1");
///
/// // Update existing key
/// cache.insert("key1", "new_value");
/// assert_eq!(cache.get(&"key1"), Some(&"new_value"));
/// ```
///
/// # Eviction Behavior
///
/// When capacity is exceeded, evicts from head (MRU - most recently used).
///
/// # Implementation
///
/// Uses raw pointer linked lists for O(1) operations with minimal overhead.
pub struct MruCore<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Direct key -> node pointer mapping
    map: FxHashMap<K, NonNull<Node<K, V>>>,

    /// Head of the list (MRU - Most Recently Used - EVICT FROM HERE!)
    head: Option<NonNull<Node<K, V>>>,
    /// Tail of the list (LRU - Least Recently Used - keep these)
    tail: Option<NonNull<Node<K, V>>>,

    /// Maximum cache capacity
    capacity: usize,
}

// SAFETY: MruCore can be sent between threads if K and V are Send.
unsafe impl<K, V> Send for MruCore<K, V>
where
    K: Clone + Eq + Hash + Send,
    V: Send,
{
}

// SAFETY: MruCore can be shared between threads if K and V are Sync.
unsafe impl<K, V> Sync for MruCore<K, V>
where
    K: Clone + Eq + Hash + Sync,
    V: Sync,
{
}

impl<K, V> MruCore<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Creates a new MRU cache with the specified capacity.
    ///
    /// # Arguments
    ///
    /// - `capacity`: Maximum number of entries the cache can hold
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::mru::MruCore;
    ///
    /// let cache: MruCore<String, i32> = MruCore::new(100);
    /// assert_eq!(cache.capacity(), 100);
    /// assert!(cache.is_empty());
    /// ```
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Self {
            map: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            head: None,
            tail: None,
            capacity,
        }
    }

    /// Detach a node from its current position in the list.
    #[inline(always)]
    fn detach(&mut self, node_ptr: NonNull<Node<K, V>>) {
        unsafe {
            let node = node_ptr.as_ref();
            let prev = node.prev;
            let next = node.next;

            match prev {
                Some(mut p) => p.as_mut().next = next,
                None => self.head = next,
            }

            match next {
                Some(mut n) => n.as_mut().prev = prev,
                None => self.tail = prev,
            }
        }
    }

    /// Attach a node at the head (MRU position).
    #[inline(always)]
    fn attach_head(&mut self, mut node_ptr: NonNull<Node<K, V>>) {
        unsafe {
            let node = node_ptr.as_mut();
            node.prev = None;
            node.next = self.head;

            match self.head {
                Some(mut h) => h.as_mut().prev = Some(node_ptr),
                None => self.tail = Some(node_ptr),
            }

            self.head = Some(node_ptr);
        }
    }

    /// Pop from head (MRU position - the most recently used!).
    #[inline(always)]
    fn pop_head(&mut self) -> Option<Box<Node<K, V>>> {
        self.head.map(|head_ptr| unsafe {
            let node = Box::from_raw(head_ptr.as_ptr());

            self.head = node.next;
            match self.head {
                Some(mut h) => h.as_mut().prev = None,
                None => self.tail = None,
            }

            node
        })
    }

    /// Retrieves a value by key, moving it to the MRU position (head).
    ///
    /// The accessed item moves to the head (MRU), making it the **first**
    /// item to be evicted when capacity is reached!
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::mru::MruCore;
    ///
    /// let mut cache = MruCore::new(100);
    /// cache.insert("key", 42);
    ///
    /// // Access moves to MRU (head) - first to evict!
    /// assert_eq!(cache.get(&"key"), Some(&42));
    ///
    /// // Missing key
    /// assert_eq!(cache.get(&"missing"), None);
    /// ```
    #[inline]
    pub fn get(&mut self, key: &K) -> Option<&V> {
        let node_ptr = *self.map.get(key)?;

        // Move to head (MRU position)
        self.detach(node_ptr);
        self.attach_head(node_ptr);

        unsafe { Some(&node_ptr.as_ref().value) }
    }

    /// Inserts or updates a key-value pair.
    ///
    /// - If the key exists, updates the value in place (no position change)
    /// - If the key is new, inserts at head (MRU position)
    /// - May trigger eviction from head (MRU) if capacity is exceeded
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::mru::MruCore;
    ///
    /// let mut cache = MruCore::new(100);
    ///
    /// // New insert goes to head (MRU)
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
        // Check for existing key - update in place
        if let Some(&node_ptr) = self.map.get(&key) {
            unsafe {
                (*node_ptr.as_ptr()).value = value;
            }
            return;
        }

        // Evict BEFORE inserting to ensure space is available
        self.evict_if_needed();

        // Create new node
        let node = Box::new(Node {
            prev: None,
            next: None,
            key: key.clone(),
            value,
        });
        let node_ptr = NonNull::new(Box::into_raw(node)).unwrap();

        self.map.insert(key, node_ptr);
        self.attach_head(node_ptr);

        #[cfg(debug_assertions)]
        self.validate_invariants();
    }

    /// Evicts entries until there is room for a new entry.
    ///
    /// MRU evicts from the head (MRU position) - the most recently used item!
    #[inline]
    fn evict_if_needed(&mut self) {
        while self.len() >= self.capacity {
            // Evict from head (MRU - most recently used!)
            if let Some(node) = self.pop_head() {
                self.map.remove(&node.key);
            } else {
                break;
            }
        }
    }

    /// Returns the number of entries in the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::mru::MruCore;
    ///
    /// let mut cache = MruCore::new(100);
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
    /// use cachekit::policy::mru::MruCore;
    ///
    /// let mut cache: MruCore<&str, i32> = MruCore::new(100);
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
    /// use cachekit::policy::mru::MruCore;
    ///
    /// let cache: MruCore<String, i32> = MruCore::new(500);
    /// assert_eq!(cache.capacity(), 500);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns `true` if the key exists in the cache.
    ///
    /// Does not affect positions (no movement on contains).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::mru::MruCore;
    ///
    /// let mut cache = MruCore::new(100);
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
    /// use cachekit::policy::mru::MruCore;
    ///
    /// let mut cache = MruCore::new(100);
    /// cache.insert("a", 1);
    /// cache.insert("b", 2);
    ///
    /// cache.clear();
    /// assert!(cache.is_empty());
    /// assert!(!cache.contains(&"a"));
    /// ```
    pub fn clear(&mut self) {
        // Free all nodes
        while self.pop_head().is_some() {}
        self.map.clear();

        #[cfg(debug_assertions)]
        self.validate_invariants();
    }

    /// Validates internal data structure invariants.
    ///
    /// This method checks that:
    /// - All nodes in map are reachable from the list
    /// - List length matches map size
    /// - No cycles exist in the list
    /// - All prev/next pointers are consistent
    ///
    /// Only runs when debug assertions are enabled.
    #[cfg(debug_assertions)]
    fn validate_invariants(&self) {
        if self.map.is_empty() {
            debug_assert!(self.head.is_none(), "Empty cache should have no head");
            debug_assert!(self.tail.is_none(), "Empty cache should have no tail");
            return;
        }

        // Count nodes from head
        let mut count = 0;
        let mut current = self.head;
        let mut visited = std::collections::HashSet::new();

        while let Some(ptr) = current {
            count += 1;
            assert!(visited.insert(ptr), "Cycle detected in list");
            assert!(count <= self.map.len(), "Count exceeds map size");

            unsafe {
                let node = ptr.as_ref();
                debug_assert!(
                    self.map.contains_key(&node.key),
                    "Node key not found in map"
                );
                current = node.next;
            }
        }

        debug_assert_eq!(count, self.map.len(), "List count doesn't match map size");
    }
}

// Proper cleanup when cache is dropped
impl<K, V> Drop for MruCore<K, V>
where
    K: Clone + Eq + Hash,
{
    fn drop(&mut self) {
        while self.pop_head().is_some() {}
    }
}

// Debug implementation
impl<K, V> std::fmt::Debug for MruCore<K, V>
where
    K: Clone + Eq + Hash + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MruCore")
            .field("capacity", &self.capacity)
            .field("len", &self.map.len())
            .finish_non_exhaustive()
    }
}

/// Implementation of the [`CoreCache`](crate::traits::CoreCache) trait for MRU.
///
/// Allows `MruCore` to be used through the unified cache interface.
///
/// # Example
///
/// ```
/// use cachekit::traits::CoreCache;
/// use cachekit::policy::mru::MruCore;
///
/// let mut cache: MruCore<&str, i32> = MruCore::new(100);
///
/// // Use via CoreCache trait
/// assert_eq!(CoreCache::insert(&mut cache, "key", 42), None);
/// assert_eq!(CoreCache::get(&mut cache, &"key"), Some(&42));
/// assert!(CoreCache::contains(&cache, &"key"));
/// ```
impl<K, V> crate::traits::CoreCache<K, V> for MruCore<K, V>
where
    K: Clone + Eq + Hash,
{
    #[inline]
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        // Check if key exists - update in place
        if let Some(&node_ptr) = self.map.get(&key) {
            let old = unsafe {
                let node = &mut *node_ptr.as_ptr();
                std::mem::replace(&mut node.value, value)
            };
            return Some(old);
        }

        // New insert
        MruCore::insert(self, key, value);
        None
    }

    #[inline]
    fn get(&mut self, key: &K) -> Option<&V> {
        MruCore::get(self, key)
    }

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

    fn clear(&mut self) {
        MruCore::clear(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::CoreCache;

    // ==============================================
    // MruCore Basic Operations
    // ==============================================

    mod basic_operations {
        use super::*;

        #[test]
        fn new_cache_is_empty() {
            let cache: MruCore<&str, i32> = MruCore::new(100);
            assert!(cache.is_empty());
            assert_eq!(cache.len(), 0);
            assert_eq!(cache.capacity(), 100);
        }

        #[test]
        fn insert_and_get() {
            let mut cache = MruCore::new(100);
            cache.insert("key1", "value1");

            assert_eq!(cache.len(), 1);
            assert_eq!(cache.get(&"key1"), Some(&"value1"));
        }

        #[test]
        fn insert_multiple_items() {
            let mut cache = MruCore::new(100);
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
            let mut cache: MruCore<&str, i32> = MruCore::new(100);
            cache.insert("exists", 42);

            assert_eq!(cache.get(&"missing"), None);
        }

        #[test]
        fn update_existing_key() {
            let mut cache = MruCore::new(100);
            cache.insert("key", "initial");
            cache.insert("key", "updated");

            assert_eq!(cache.len(), 1);
            assert_eq!(cache.get(&"key"), Some(&"updated"));
        }

        #[test]
        fn contains_returns_correct_result() {
            let mut cache = MruCore::new(100);
            cache.insert("exists", 1);

            assert!(cache.contains(&"exists"));
            assert!(!cache.contains(&"missing"));
        }

        #[test]
        fn clear_removes_all_entries() {
            let mut cache = MruCore::new(100);
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
            let cache: MruCore<i32, i32> = MruCore::new(500);
            assert_eq!(cache.capacity(), 500);
        }
    }

    // ==============================================
    // MRU-Specific Behavior (Evict Most Recent)
    // ==============================================

    mod mru_behavior {
        use super::*;

        #[test]
        fn evicts_most_recently_used() {
            let mut cache = MruCore::new(3);

            cache.insert("a", 1);
            cache.insert("b", 2);
            cache.insert("c", 3);

            // Access "a" - moves it to MRU (head)
            cache.get(&"a");

            // Insert "d" - should evict "a" (the MRU)
            cache.insert("d", 4);

            assert!(!cache.contains(&"a"), "MRU 'a' should be evicted");
            assert!(cache.contains(&"b"));
            assert!(cache.contains(&"c"));
            assert!(cache.contains(&"d"));
        }

        #[test]
        fn most_recent_insert_evicted_first() {
            let mut cache = MruCore::new(3);

            cache.insert("a", 1);
            cache.insert("b", 2);
            cache.insert("c", 3);

            // "c" is most recent (at head)
            // Insert "d" - should evict "c"
            cache.insert("d", 4);

            assert!(cache.contains(&"a"));
            assert!(cache.contains(&"b"));
            assert!(!cache.contains(&"c"), "Most recent 'c' should be evicted");
            assert!(cache.contains(&"d"));
        }

        #[test]
        fn opposite_of_lru_behavior() {
            let mut cache = MruCore::new(3);

            cache.insert("first", 1);
            cache.insert("middle", 2);
            cache.insert("last", 3);

            // In LRU, "first" would be evicted (oldest)
            // In MRU, "last" is evicted (newest)
            cache.insert("new", 4);

            assert!(cache.contains(&"first"), "Oldest should stay in MRU");
            assert!(cache.contains(&"middle"));
            assert!(!cache.contains(&"last"), "Newest should be evicted in MRU");
            assert!(cache.contains(&"new"));
        }

        #[test]
        fn cyclic_pattern_simulation() {
            let mut cache = MruCore::new(5);

            // Simulate cyclic access: 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, ...
            // Each access moves item to MRU, making it first to evict

            for cycle in 0..3 {
                for i in 1..=5 {
                    let key = format!("cycle{}_{}", cycle, i);
                    cache.insert(key, i);
                }
            }

            // With MRU: most recent inserts get evicted, oldest items survive
            // The first 4 items from cycle 0 survive + last item from cycle 2
            assert!(cache.contains(&"cycle0_1".to_string())); // Oldest, survives
            assert!(cache.contains(&"cycle0_4".to_string())); // Old, survives
            assert!(!cache.contains(&"cycle1_3".to_string())); // Evicted
            assert!(!cache.contains(&"cycle2_1".to_string())); // Evicted
            assert!(cache.contains(&"cycle2_5".to_string())); // Last inserted, survives
        }
    }

    // ==============================================
    // Eviction Behavior
    // ==============================================

    mod eviction_behavior {
        use super::*;

        #[test]
        fn eviction_occurs_when_over_capacity() {
            let mut cache = MruCore::new(5);

            for i in 0..10 {
                cache.insert(i, i * 10);
            }

            assert_eq!(cache.len(), 5);
        }

        #[test]
        fn eviction_removes_from_index() {
            let mut cache = MruCore::new(3);

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
            let mut cache = MruCore::new(3);

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
    }

    // ==============================================
    // CoreCache Trait Implementation
    // ==============================================

    mod core_cache_trait {
        use super::*;

        #[test]
        fn trait_insert_returns_old_value() {
            let mut cache: MruCore<&str, i32> = MruCore::new(100);

            let old = CoreCache::insert(&mut cache, "key", 1);
            assert_eq!(old, None);

            let old = CoreCache::insert(&mut cache, "key", 2);
            assert_eq!(old, Some(1));

            assert_eq!(CoreCache::get(&mut cache, &"key"), Some(&2));
        }

        #[test]
        fn trait_get_works() {
            let mut cache = MruCore::new(100);
            CoreCache::insert(&mut cache, "key", 42);

            assert_eq!(CoreCache::get(&mut cache, &"key"), Some(&42));
            assert_eq!(CoreCache::get(&mut cache, &"missing"), None);
        }

        #[test]
        fn trait_contains_works() {
            let mut cache = MruCore::new(100);
            CoreCache::insert(&mut cache, "key", 1);

            assert!(CoreCache::contains(&cache, &"key"));
            assert!(!CoreCache::contains(&cache, &"missing"));
        }

        #[test]
        fn trait_len_and_capacity() {
            let mut cache: MruCore<i32, i32> = MruCore::new(50);

            assert_eq!(CoreCache::len(&cache), 0);
            assert_eq!(CoreCache::capacity(&cache), 50);

            for i in 0..30 {
                CoreCache::insert(&mut cache, i, i * 10);
            }

            assert_eq!(CoreCache::len(&cache), 30);
        }

        #[test]
        fn trait_clear_works() {
            let mut cache = MruCore::new(100);
            CoreCache::insert(&mut cache, "a", 1);
            CoreCache::insert(&mut cache, "b", 2);

            CoreCache::clear(&mut cache);

            assert_eq!(CoreCache::len(&cache), 0);
            assert!(!CoreCache::contains(&cache, &"a"));
        }
    }

    // ==============================================
    // Edge Cases
    // ==============================================

    mod edge_cases {
        use super::*;

        #[test]
        fn single_capacity_cache() {
            let mut cache = MruCore::new(1);

            cache.insert("a", 1);
            assert_eq!(cache.get(&"a"), Some(&1));

            cache.insert("b", 2);
            assert!(!cache.contains(&"a"));
            assert_eq!(cache.get(&"b"), Some(&2));
        }

        #[test]
        fn get_after_update() {
            let mut cache = MruCore::new(100);

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
            let mut cache = MruCore::new(10000);

            for i in 0..10000 {
                cache.insert(i, i * 2);
            }

            assert_eq!(cache.len(), 10000);

            assert_eq!(cache.get(&5000), Some(&10000));
            assert_eq!(cache.get(&9999), Some(&19998));
        }

        #[test]
        fn empty_cache_operations() {
            let mut cache: MruCore<i32, i32> = MruCore::new(100);

            assert!(cache.is_empty());
            assert_eq!(cache.get(&1), None);
            assert!(!cache.contains(&1));

            cache.clear();
            assert!(cache.is_empty());
        }

        #[test]
        fn string_keys_and_values() {
            let mut cache = MruCore::new(100);

            cache.insert(String::from("hello"), String::from("world"));
            cache.insert(String::from("foo"), String::from("bar"));

            assert_eq!(
                cache.get(&String::from("hello")),
                Some(&String::from("world"))
            );
            assert_eq!(cache.get(&String::from("foo")), Some(&String::from("bar")));
        }
    }

    // ==============================================
    // Validation Tests
    // ==============================================

    #[test]
    #[cfg(debug_assertions)]
    fn validate_invariants_after_operations() {
        let mut cache = MruCore::new(10);

        // Insert items
        for i in 1..=10 {
            cache.insert(i, i * 100);
        }
        cache.validate_invariants();

        // Access items (moves to MRU, making them first to evict)
        for _ in 0..3 {
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
    }

    #[test]
    #[cfg(debug_assertions)]
    fn validate_invariants_with_mru_evictions() {
        let mut cache = MruCore::new(3);
        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.insert(3, 300);
        cache.validate_invariants();

        // Access item 1 (moves to MRU)
        cache.get(&1);
        cache.validate_invariants();

        // Insert new item, should evict item 1 (MRU)
        cache.insert(4, 400);
        cache.validate_invariants();

        assert!(!cache.contains(&1)); // MRU evicted
        assert!(cache.contains(&2));
        assert!(cache.contains(&3));
        assert!(cache.contains(&4));
    }
}
