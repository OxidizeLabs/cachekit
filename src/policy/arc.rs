//! Adaptive Replacement Cache (ARC) replacement policy.
//!
//! Implements the ARC algorithm, which automatically adapts between recency and
//! frequency preferences by maintaining four lists and adjusting a dynamic target
//! parameter based on access patterns.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                           ARCCore<K, V> Layout                              │
//! │                                                                             │
//! │   ┌─────────────────────────────────────────────────────────────────────┐   │
//! │   │  index: HashMap<K, NonNull<Node>>    nodes: Allocated on heap       │   │
//! │   │                                                                     │   │
//! │   │  ┌──────────┬───────────┐           ┌────────┬──────────────────┐   │   │
//! │   │  │   Key    │  NodePtr  │           │ Node   │ key,val,list     │   │   │
//! │   │  ├──────────┼───────────┤           ├────────┼──────────────────┤   │   │
//! │   │  │  "page1" │   ptr_0   │──────────►│ Node0  │ k,v,T1           │   │   │
//! │   │  │  "page2" │   ptr_1   │──────────►│ Node1  │ k,v,T2           │   │   │
//! │   │  │  "page3" │   ptr_2   │──────────►│ Node2  │ k,v,T1           │   │   │
//! │   │  └──────────┴───────────┘           └────────┴──────────────────┘   │   │
//! │   └─────────────────────────────────────────────────────────────────────┘   │
//! │                                                                             │
//! │   ┌─────────────────────────────────────────────────────────────────────┐   │
//! │   │                        List Organization                            │   │
//! │   │                                                                     │   │
//! │   │   T1 (Recency - Recent Once)          T2 (Frequency - Repeated)     │   │
//! │   │   ┌─────────────────────────┐          ┌─────────────────────────┐  │   │
//! │   │   │ MRU               LRU   │          │ MRU               LRU   │  │   │
//! │   │   │  ▼                  ▼   │          │  ▼                  ▼   │  │   │
//! │   │   │ [ptr_2] ◄──► [ptr_0] ◄┤ │          │ [ptr_1] ◄──► [...] ◄┤   │  │   │
//! │   │   │  new      older   evict │          │ hot          cold evict │  │   │
//! │   │   └─────────────────────────┘          └─────────────────────────┘  │   │
//! │   │                                                                     │   │
//! │   │   B1 (Ghost - evicted from T1)       B2 (Ghost - evicted from T2)   │   │
//! │   │   ┌─────────────────────────┐          ┌─────────────────────────┐  │   │
//! │   │   │ Keys only (no values)   │          │ Keys only (no values)   │  │   │
//! │   │   └─────────────────────────┘          └─────────────────────────┘  │   │
//! │   │                                                                     │   │
//! │   │   Adaptation Parameter: p (target size for T1)                      │   │
//! │   │   • Hit in B1 → increase p (favor recency)                          │   │
//! │   │   • Hit in B2 → decrease p (favor frequency)                        │   │
//! │   └─────────────────────────────────────────────────────────────────────┘   │
//! │                                                                             │
//! └─────────────────────────────────────────────────────────────────────────────┘
//!
//! Insert Flow (new key, not in any list)
//! ────────────────────────────────────────
//!
//!   insert("new_key", value):
//!     1. Check index - not found
//!     2. Check ghost lists (B1/B2) for adaptation
//!     3. Create Node with ListKind::T1
//!     4. Allocate on heap → get NonNull<Node>
//!     5. Insert key→ptr into index
//!     6. Attach ptr to T1 MRU
//!     7. Evict if over capacity (replace algorithm)
//!
//! Access Flow (existing key in T1/T2)
//! ────────────────────────────────────
//!
//!   get("existing_key"):
//!     1. Lookup ptr in index
//!     2. Check node's list:
//!        - If T1: promote to T2 (move to T2 MRU)
//!        - If T2: move to MRU position within T2
//!     3. Return &value
//!
//! Ghost Hit Flow (key in B1/B2)
//! ──────────────────────────────
//!
//!   get("ghost_key"):
//!     1. Found in B1: increase p (favor recency)
//!     2. Found in B2: decrease p (favor frequency)
//!     3. Perform replacement to make space
//!     4. Insert into T2 (proven reuse)
//!     5. Remove from ghost list
//!
//! Eviction Flow (Replace Algorithm)
//! ──────────────────────────────────
//!
//!   replace():
//!     if |T1| >= max(1, p):
//!       evict from T1 LRU → move key to B1
//!     else:
//!       evict from T2 LRU → move key to B2
//! ```
//!
//! ## Key Components
//!
//! - [`ARCCore`]: Main ARC cache implementation
//! - Four lists: T1 (recent once), T2 (frequent), B1 (ghost for T1), B2 (ghost for T2)
//! - Adaptation parameter `p`: target size for T1 vs T2
//!
//! ## Operations
//!
//! | Operation   | Time   | Notes                                      |
//! |-------------|--------|--------------------------------------------|
//! | `get`       | O(1)   | May promote T1→T2 or adapt via ghost hit   |
//! | `insert`    | O(1)*  | *Amortized, may trigger evictions          |
//! | `contains`  | O(1)   | Index lookup only                          |
//! | `len`       | O(1)   | Returns T1 + T2 entries                    |
//! | `clear`     | O(n)   | Clears all structures                      |
//!
//! ## Algorithm Properties
//!
//! - **Adaptive**: Automatically balances recency vs frequency based on workload
//! - **Scan Resistant**: Ghost lists prevent one-time scans from polluting cache
//! - **Self-Tuning**: No manual parameter tuning required
//! - **Competitive**: O(1) operations, proven optimal in certain workload classes
//!
//! ## Use Cases
//!
//! - Database buffer pools with mixed access patterns
//! - File system caches
//! - Web caches with varying temporal/frequency characteristics
//! - Workloads where optimal recency/frequency balance is unknown
//!
//! ## Example Usage
//!
//! ```
//! use cachekit::policy::arc::ARCCore;
//! use cachekit::traits::{CoreCache, ReadOnlyCache};
//!
//! // Create ARC cache with 100 entry capacity
//! let mut cache = ARCCore::new(100);
//!
//! // Insert items (go to T1 - recent list)
//! cache.insert("page1", "content1");
//! cache.insert("page2", "content2");
//!
//! // First access promotes to T2 (frequent list)
//! assert_eq!(cache.get(&"page1"), Some(&"content1"));
//!
//! // Second access keeps in T2 (MRU position)
//! assert_eq!(cache.get(&"page1"), Some(&"content1"));
//!
//! assert_eq!(cache.len(), 2);
//! ```
//!
//! ## Thread Safety
//!
//! - [`ARCCore`]: Not thread-safe, designed for single-threaded use
//! - For concurrent access, wrap in external synchronization
//!
//! ## Implementation Notes
//!
//! - T1 uses LRU ordering (recent once entries)
//! - T2 uses LRU ordering (frequent entries)
//! - B1/B2 are ghost lists (keys only, no values)
//! - Default initial `p` is `capacity / 2`
//! - Ghost list sizes are each up to `capacity`
//! - Promotion from T1 to T2 happens on re-access
//!
//! ## References
//!
//! - Megiddo & Modha, "ARC: A Self-Tuning, Low Overhead Replacement Cache",
//!   FAST 2003
//! - Wikipedia: Cache replacement policies (ARC section)
//!
//! ## Performance Trade-offs
//!
//! - **When to Use**: Unknown or shifting workload patterns; need adaptive behavior
//! - **Memory Overhead**: 4 lists + ghost entries (up to 2× capacity in keys)
//! - **vs LRU**: Better on mixed workloads, slightly higher metadata overhead
//! - **vs 2Q/SLRU**: More adaptive, no manual tuning needed

use crate::ds::GhostList;
use crate::prelude::ReadOnlyCache;
use crate::traits::{CoreCache, MutableCache};
use rustc_hash::FxHashMap;
use std::hash::Hash;
use std::ptr::NonNull;

/// Indicates which list an entry resides in.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum ListKind {
    /// Entry is in T1 (recent once).
    T1,
    /// Entry is in T2 (frequent).
    T2,
}

/// Node in the ARC linked list.
///
/// Cache-line optimized layout with pointers first.
#[repr(C)]
struct Node<K, V> {
    prev: Option<NonNull<Node<K, V>>>,
    next: Option<NonNull<Node<K, V>>>,
    list: ListKind,
    key: K,
    value: V,
}

/// Core Adaptive Replacement Cache (ARC) implementation.
///
/// Implements the ARC replacement algorithm with automatic adaptation between
/// recency and frequency preferences:
/// - **T1**: Recently accessed once (recency list)
/// - **T2**: Accessed multiple times (frequency list)
/// - **B1**: Ghost list for items evicted from T1
/// - **B2**: Ghost list for items evicted from T2
///
/// The cache maintains an adaptation parameter `p` that controls the target
/// size of T1 vs T2. Ghost hits in B1/B2 adjust `p` to favor recency or frequency.
///
/// # Type Parameters
///
/// - `K`: Key type, must be `Clone + Eq + Hash`
/// - `V`: Value type
///
/// # Example
///
/// ```
/// use cachekit::policy::arc::ARCCore;
/// use cachekit::traits::{CoreCache, ReadOnlyCache};
///
/// // 100 capacity ARC cache
/// let mut cache = ARCCore::new(100);
///
/// // Insert goes to T1 (recent list)
/// cache.insert("key1", "value1");
/// assert!(cache.contains(&"key1"));
///
/// // First get promotes to T2 (frequent list)
/// cache.get(&"key1");
///
/// // Update existing key
/// cache.insert("key1", "new_value");
/// assert_eq!(cache.get(&"key1"), Some(&"new_value"));
/// ```
///
/// # Eviction Behavior
///
/// The replacement algorithm selects victims based on the adaptation parameter `p`:
/// - If `|T1| >= max(1, p)`: evict from T1 LRU, move key to B1
/// - Otherwise: evict from T2 LRU, move key to B2
///
/// # Implementation
///
/// Uses raw pointer linked lists for O(1) operations with minimal overhead.
/// Ghost lists track recently evicted keys to enable adaptation.
pub struct ARCCore<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Direct key -> node pointer mapping
    map: FxHashMap<K, NonNull<Node<K, V>>>,

    /// T1 list (recent once): head=MRU, tail=LRU
    t1_head: Option<NonNull<Node<K, V>>>,
    t1_tail: Option<NonNull<Node<K, V>>>,
    t1_len: usize,

    /// T2 list (frequent): head=MRU, tail=LRU
    t2_head: Option<NonNull<Node<K, V>>>,
    t2_tail: Option<NonNull<Node<K, V>>>,
    t2_len: usize,

    /// B1 ghost list (evicted from T1)
    b1: GhostList<K>,

    /// B2 ghost list (evicted from T2)
    b2: GhostList<K>,

    /// Adaptation parameter: target size for T1
    p: usize,

    /// Maximum total cache capacity
    capacity: usize,
}

// SAFETY: ARCCore can be sent between threads if K and V are Send.
unsafe impl<K, V> Send for ARCCore<K, V>
where
    K: Clone + Eq + Hash + Send,
    V: Send,
{
}

// SAFETY: ARCCore can be shared between threads if K and V are Sync.
unsafe impl<K, V> Sync for ARCCore<K, V>
where
    K: Clone + Eq + Hash + Sync,
    V: Sync,
{
}

impl<K, V> ARCCore<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Creates a new ARC cache with the specified capacity.
    ///
    /// # Arguments
    ///
    /// - `capacity`: Maximum number of entries in T1 + T2
    ///
    /// Ghost lists (B1/B2) can each hold up to `capacity` keys.
    /// Initial adaptation parameter `p` is set to `capacity / 2`.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::arc::ARCCore;
    /// use cachekit::traits::ReadOnlyCache;
    ///
    /// // 100 capacity ARC cache
    /// let cache: ARCCore<String, i32> = ARCCore::new(100);
    /// assert_eq!(cache.capacity(), 100);
    /// assert!(cache.is_empty());
    /// ```
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Self {
            map: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            t1_head: None,
            t1_tail: None,
            t1_len: 0,
            t2_head: None,
            t2_tail: None,
            t2_len: 0,
            b1: GhostList::new(capacity),
            b2: GhostList::new(capacity),
            p: capacity / 2,
            capacity,
        }
    }

    /// Detach a node from its current list (T1 or T2).
    #[inline(always)]
    fn detach(&mut self, node_ptr: NonNull<Node<K, V>>) {
        unsafe {
            let node = node_ptr.as_ref();
            let prev = node.prev;
            let next = node.next;
            let list = node.list;

            let (head, tail, len) = match list {
                ListKind::T1 => (&mut self.t1_head, &mut self.t1_tail, &mut self.t1_len),
                ListKind::T2 => (&mut self.t2_head, &mut self.t2_tail, &mut self.t2_len),
            };

            match prev {
                Some(mut p) => p.as_mut().next = next,
                None => *head = next,
            }

            match next {
                Some(mut n) => n.as_mut().prev = prev,
                None => *tail = prev,
            }

            *len -= 1;
        }
    }

    /// Attach a node at the head of T1 list (MRU position).
    #[inline(always)]
    fn attach_t1_head(&mut self, mut node_ptr: NonNull<Node<K, V>>) {
        unsafe {
            let node = node_ptr.as_mut();
            node.prev = None;
            node.next = self.t1_head;
            node.list = ListKind::T1;

            match self.t1_head {
                Some(mut h) => h.as_mut().prev = Some(node_ptr),
                None => self.t1_tail = Some(node_ptr),
            }

            self.t1_head = Some(node_ptr);
            self.t1_len += 1;
        }
    }

    /// Attach a node at the head of T2 list (MRU position).
    #[inline(always)]
    fn attach_t2_head(&mut self, mut node_ptr: NonNull<Node<K, V>>) {
        unsafe {
            let node = node_ptr.as_mut();
            node.prev = None;
            node.next = self.t2_head;
            node.list = ListKind::T2;

            match self.t2_head {
                Some(mut h) => h.as_mut().prev = Some(node_ptr),
                None => self.t2_tail = Some(node_ptr),
            }

            self.t2_head = Some(node_ptr);
            self.t2_len += 1;
        }
    }

    /// Replace: select a victim from T1 or T2 based on adaptation parameter p.
    ///
    /// This is the core ARC replacement algorithm.
    fn replace(&mut self, in_b2: bool) {
        // Decide whether to evict from T1 or T2
        let evict_from_t1 =
            if self.t1_len > 0 && (self.t1_len > self.p || (self.t1_len == self.p && in_b2)) {
                true
            } else if self.t2_len > 0 {
                false
            } else {
                // Both conditions failed, default to T1 if it has entries
                self.t1_len > 0
            };

        if evict_from_t1 {
            // Evict from T1 LRU
            if let Some(victim_ptr) = self.t1_tail {
                unsafe {
                    let victim = victim_ptr.as_ref();
                    let key = victim.key.clone();

                    self.detach(victim_ptr);
                    self.map.remove(&key);

                    // Move key to B1 ghost list
                    self.b1.record(key.clone());

                    // Deallocate the node
                    let _ = Box::from_raw(victim_ptr.as_ptr());
                }
            }
        } else {
            // Evict from T2 LRU
            if let Some(victim_ptr) = self.t2_tail {
                unsafe {
                    let victim = victim_ptr.as_ref();
                    let key = victim.key.clone();

                    self.detach(victim_ptr);
                    self.map.remove(&key);

                    // Move key to B2 ghost list
                    self.b2.record(key.clone());

                    // Deallocate the node
                    let _ = Box::from_raw(victim_ptr.as_ptr());
                }
            }
        }
    }

    /// Adapt parameter p based on ghost hit location.
    fn adapt(&mut self, in_b1: bool, in_b2: bool) {
        if in_b1 {
            // Hit in B1: increase p (favor recency/T1)
            let delta = if self.b2.len() >= self.b1.len() {
                1
            } else if !self.b1.is_empty() {
                ((self.b2.len() as f64 / self.b1.len() as f64).ceil() as usize).max(1)
            } else {
                1
            };
            self.p = (self.p + delta).min(self.capacity);
        } else if in_b2 {
            // Hit in B2: decrease p (favor frequency/T2)
            let delta = if self.b1.len() >= self.b2.len() {
                1
            } else if !self.b2.is_empty() {
                ((self.b1.len() as f64 / self.b2.len() as f64).ceil() as usize).max(1)
            } else {
                1
            };
            self.p = self.p.saturating_sub(delta);
        }
    }

    /// Returns the current value of the adaptation parameter p.
    ///
    /// This represents the target size for T1. Higher values favor recency,
    /// lower values favor frequency.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::arc::ARCCore;
    ///
    /// let cache: ARCCore<String, i32> = ARCCore::new(100);
    /// // Initial p is capacity / 2
    /// assert_eq!(cache.p_value(), 50);
    /// ```
    pub fn p_value(&self) -> usize {
        self.p
    }

    /// Returns the number of entries in T1 (recent once list).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::arc::ARCCore;
    /// use cachekit::traits::CoreCache;
    ///
    /// let mut cache = ARCCore::new(100);
    /// cache.insert("key", "value");
    /// assert_eq!(cache.t1_len(), 1);  // New entries go to T1
    /// ```
    pub fn t1_len(&self) -> usize {
        self.t1_len
    }

    /// Returns the number of entries in T2 (frequent list).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::arc::ARCCore;
    /// use cachekit::traits::CoreCache;
    ///
    /// let mut cache = ARCCore::new(100);
    /// cache.insert("key", "value");
    /// cache.get(&"key");  // Promotes to T2
    /// assert_eq!(cache.t2_len(), 1);
    /// ```
    pub fn t2_len(&self) -> usize {
        self.t2_len
    }

    /// Returns the number of keys in B1 ghost list.
    pub fn b1_len(&self) -> usize {
        self.b1.len()
    }

    /// Returns the number of keys in B2 ghost list.
    pub fn b2_len(&self) -> usize {
        self.b2.len()
    }

    #[cfg(any(test, debug_assertions))]
    /// Validates internal invariants of the ARC cache.
    ///
    /// Panics if any invariant is violated.
    pub fn debug_validate_invariants(&self)
    where
        K: std::fmt::Debug,
    {
        // 1. Total length matches sum of T1 and T2
        assert_eq!(
            self.len(),
            self.t1_len + self.t2_len,
            "len() should equal t1_len + t2_len"
        );

        // 2. Map size equals total entries
        assert_eq!(
            self.map.len(),
            self.t1_len + self.t2_len,
            "map.len() should equal total entries"
        );

        // 3. Total entries don't exceed capacity
        assert!(
            self.t1_len + self.t2_len <= self.capacity,
            "total entries ({}) exceed capacity ({})",
            self.t1_len + self.t2_len,
            self.capacity
        );

        // 4. p is within valid range
        assert!(
            self.p <= self.capacity,
            "p ({}) exceeds capacity ({})",
            self.p,
            self.capacity
        );

        // 5. Ghost lists don't exceed capacity
        assert!(
            self.b1.len() <= self.capacity,
            "B1 length ({}) exceeds capacity ({})",
            self.b1.len(),
            self.capacity
        );
        assert!(
            self.b2.len() <= self.capacity,
            "B2 length ({}) exceeds capacity ({})",
            self.b2.len(),
            self.capacity
        );

        // 6. Count actual T1 entries
        let mut t1_count = 0;
        let mut current = self.t1_head;
        let mut visited_t1 = std::collections::HashSet::new();
        while let Some(node_ptr) = current {
            unsafe {
                let node = node_ptr.as_ref();

                // Check for cycles
                assert!(visited_t1.insert(node_ptr), "Cycle detected in T1 list");

                // Verify list kind
                assert_eq!(node.list, ListKind::T1, "Node in T1 has wrong list kind");

                // Verify node is in map
                assert!(self.map.contains_key(&node.key), "T1 node key not in map");

                t1_count += 1;
                current = node.next;
            }
        }
        assert_eq!(
            t1_count, self.t1_len,
            "T1 actual count doesn't match t1_len"
        );

        // 7. Count actual T2 entries
        let mut t2_count = 0;
        let mut current = self.t2_head;
        let mut visited_t2 = std::collections::HashSet::new();
        while let Some(node_ptr) = current {
            unsafe {
                let node = node_ptr.as_ref();

                // Check for cycles
                assert!(visited_t2.insert(node_ptr), "Cycle detected in T2 list");

                // Verify list kind
                assert_eq!(node.list, ListKind::T2, "Node in T2 has wrong list kind");

                // Verify node is in map
                assert!(self.map.contains_key(&node.key), "T2 node key not in map");

                t2_count += 1;
                current = node.next;
            }
        }
        assert_eq!(
            t2_count, self.t2_len,
            "T2 actual count doesn't match t2_len"
        );

        // 8. Verify no overlap between T1 and T2
        for t1_ptr in &visited_t1 {
            assert!(
                !visited_t2.contains(t1_ptr),
                "Node appears in both T1 and T2"
            );
        }

        // 9. Verify all map entries are accounted for in T1 or T2
        assert_eq!(
            visited_t1.len() + visited_t2.len(),
            self.map.len(),
            "Map contains entries not in T1 or T2"
        );

        // 10. Ghost lists don't contain keys that are in the cache
        for key in self.map.keys() {
            assert!(
                !self.b1.contains(key),
                "Key {:?} is in both cache and B1",
                key
            );
            assert!(
                !self.b2.contains(key),
                "Key {:?} is in both cache and B2",
                key
            );
        }
    }
}

impl<K, V> std::fmt::Debug for ARCCore<K, V>
where
    K: Clone + Eq + Hash + std::fmt::Debug,
    V: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ARCCore")
            .field("capacity", &self.capacity)
            .field("t1_len", &self.t1_len)
            .field("t2_len", &self.t2_len)
            .field("b1_len", &self.b1.len())
            .field("b2_len", &self.b2.len())
            .field("p", &self.p)
            .field("total_len", &(self.t1_len + self.t2_len))
            .finish()
    }
}

impl<K, V> ReadOnlyCache<K, V> for ARCCore<K, V>
where
    K: Clone + Eq + Hash,
{
    fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    fn len(&self) -> usize {
        self.t1_len + self.t2_len
    }

    fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<K, V> CoreCache<K, V> for ARCCore<K, V>
where
    K: Clone + Eq + Hash,
{
    fn get(&mut self, key: &K) -> Option<&V> {
        let node_ptr = *self.map.get(key)?;

        unsafe {
            let node = node_ptr.as_ref();

            match node.list {
                ListKind::T1 => {
                    // Promote from T1 to T2
                    self.detach(node_ptr);
                    self.attach_t2_head(node_ptr);
                },
                ListKind::T2 => {
                    // Move to T2 MRU
                    self.detach(node_ptr);
                    self.attach_t2_head(node_ptr);
                },
            }

            Some(&node_ptr.as_ref().value)
        }
    }

    fn insert(&mut self, key: K, value: V) -> Option<V> {
        // Handle zero capacity edge case
        if self.capacity == 0 {
            return None;
        }

        // Case 1: Key already in cache (T1 or T2)
        if let Some(&node_ptr) = self.map.get(&key) {
            unsafe {
                let mut node_ptr = node_ptr; // Make mutable copy
                let node = node_ptr.as_mut();
                let old_value = std::mem::replace(&mut node.value, value);

                // Update position based on current list
                match node.list {
                    ListKind::T1 => {
                        // Promote to T2
                        self.detach(node_ptr);
                        self.attach_t2_head(node_ptr);
                    },
                    ListKind::T2 => {
                        // Move to T2 MRU
                        self.detach(node_ptr);
                        self.attach_t2_head(node_ptr);
                    },
                }

                return Some(old_value);
            }
        }

        // Check for ghost hits
        let in_b1 = self.b1.contains(&key);
        let in_b2 = self.b2.contains(&key);

        // Case 2: Ghost hit in B1
        if in_b1 {
            self.adapt(true, false);
            self.b1.remove(&key);

            // Make space if needed
            if self.t1_len + self.t2_len >= self.capacity {
                self.replace(false);
            }

            // Insert into T2 (proven reuse)
            let node = Box::new(Node {
                prev: None,
                next: None,
                list: ListKind::T2,
                key: key.clone(),
                value,
            });
            let node_ptr = NonNull::new(Box::into_raw(node)).unwrap();
            self.map.insert(key, node_ptr);
            self.attach_t2_head(node_ptr);

            return None;
        }

        // Case 3: Ghost hit in B2
        if in_b2 {
            self.adapt(false, true);
            self.b2.remove(&key);

            // Make space if needed
            if self.t1_len + self.t2_len >= self.capacity {
                self.replace(true);
            }

            // Insert into T2 (proven reuse)
            let node = Box::new(Node {
                prev: None,
                next: None,
                list: ListKind::T2,
                key: key.clone(),
                value,
            });
            let node_ptr = NonNull::new(Box::into_raw(node)).unwrap();
            self.map.insert(key, node_ptr);
            self.attach_t2_head(node_ptr);

            return None;
        }

        // Case 4: Complete miss (not in cache or ghost lists)
        // Handle ghost list capacity management
        if self.t1_len + self.t2_len >= self.capacity {
            self.replace(false);
        } else {
            // L1 is not full, but we may need to prune ghost lists
            // This is for when T1 + T2 + B1 + B2 exceeds 2 * capacity
            let total_size = self.t1_len + self.t2_len + self.b1.len() + self.b2.len();
            if total_size >= 2 * self.capacity {
                // Remove oldest from B1 if it's larger
                if !self.b1.is_empty() {
                    // Ghost list automatically handles LRU eviction
                } else if !self.b2.is_empty() {
                    // Same for B2
                }
            }
        }

        // Insert into T1
        let node = Box::new(Node {
            prev: None,
            next: None,
            list: ListKind::T1,
            key: key.clone(),
            value,
        });
        let node_ptr = NonNull::new(Box::into_raw(node)).unwrap();
        self.map.insert(key, node_ptr);
        self.attach_t1_head(node_ptr);

        None
    }

    fn clear(&mut self) {
        // Deallocate all nodes in T1
        let mut current = self.t1_head;
        while let Some(node_ptr) = current {
            unsafe {
                let node = node_ptr.as_ref();
                current = node.next;
                let _ = Box::from_raw(node_ptr.as_ptr());
            }
        }

        // Deallocate all nodes in T2
        let mut current = self.t2_head;
        while let Some(node_ptr) = current {
            unsafe {
                let node = node_ptr.as_ref();
                current = node.next;
                let _ = Box::from_raw(node_ptr.as_ptr());
            }
        }

        self.map.clear();
        self.t1_head = None;
        self.t1_tail = None;
        self.t1_len = 0;
        self.t2_head = None;
        self.t2_tail = None;
        self.t2_len = 0;
        self.b1.clear();
        self.b2.clear();
        self.p = self.capacity / 2;
    }
}

impl<K, V> MutableCache<K, V> for ARCCore<K, V>
where
    K: Clone + Eq + Hash,
{
    fn remove(&mut self, key: &K) -> Option<V> {
        let node_ptr = self.map.remove(key)?;

        self.detach(node_ptr);

        unsafe {
            let node = Box::from_raw(node_ptr.as_ptr());
            Some(node.value)
        }
    }
}

impl<K, V> Drop for ARCCore<K, V>
where
    K: Clone + Eq + Hash,
{
    fn drop(&mut self) {
        self.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arc_new_cache() {
        let cache: ARCCore<String, i32> = ARCCore::new(100);
        assert_eq!(cache.capacity(), 100);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.t1_len(), 0);
        assert_eq!(cache.t2_len(), 0);
        assert_eq!(cache.b1_len(), 0);
        assert_eq!(cache.b2_len(), 0);
        assert_eq!(cache.p_value(), 50); // Initial p = capacity / 2
    }

    #[test]
    fn arc_insert_and_get() {
        let mut cache = ARCCore::new(10);

        // First insert goes to T1
        cache.insert("key1", "value1");
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.t1_len(), 1);
        assert_eq!(cache.t2_len(), 0);

        // Get promotes to T2
        assert_eq!(cache.get(&"key1"), Some(&"value1"));
        assert_eq!(cache.t1_len(), 0);
        assert_eq!(cache.t2_len(), 1);

        // Second get keeps in T2
        assert_eq!(cache.get(&"key1"), Some(&"value1"));
        assert_eq!(cache.t1_len(), 0);
        assert_eq!(cache.t2_len(), 1);
    }

    #[test]
    fn arc_update_existing() {
        let mut cache = ARCCore::new(10);

        cache.insert("key1", "value1");
        assert_eq!(cache.t1_len(), 1);

        // Update in T1 promotes to T2
        let old = cache.insert("key1", "new_value");
        assert_eq!(old, Some("value1"));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.t1_len(), 0);
        assert_eq!(cache.t2_len(), 1);

        assert_eq!(cache.get(&"key1"), Some(&"new_value"));
    }

    #[test]
    fn arc_eviction_fills_ghost_lists() {
        let mut cache = ARCCore::new(2);

        // Fill cache
        cache.insert("a", 1);
        cache.insert("b", 2);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.t1_len(), 2);

        // Insert third item triggers eviction
        cache.insert("c", 3);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.t1_len(), 2); // c and b (a evicted)
        assert_eq!(cache.b1_len(), 1); // a moved to B1
        assert!(!cache.contains(&"a"));
    }

    #[test]
    fn arc_ghost_hit_promotes_to_t2() {
        let mut cache = ARCCore::new(2);

        // Fill and evict
        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3); // Evicts "a" to B1

        cache.debug_validate_invariants();

        assert!(!cache.contains(&"a"));
        assert_eq!(cache.b1_len(), 1);
        assert!(cache.b1.contains(&"a"));

        // Ghost hit on "a"
        // This should: remove "a" from B1, make space (evicting something else), insert "a" to T2
        cache.insert("a", 10);
        cache.debug_validate_invariants();

        println!(
            "After ghost hit: len={}, t1={}, t2={}, b1={}, b2={}",
            cache.len(),
            cache.t1_len(),
            cache.t2_len(),
            cache.b1_len(),
            cache.b2_len()
        );

        assert_eq!(
            cache.len(),
            2,
            "Cache length should be 2, got {}",
            cache.len()
        );
        assert_eq!(cache.t2_len(), 1); // "a" goes to T2 (ghost hit)
        assert!(!cache.b1.contains(&"a")); // "a" removed from B1
        // Note: B1 may still have entries from the eviction that happened to make space for "a"
    }

    #[test]
    fn arc_adaptation_increases_p() {
        let mut cache = ARCCore::new(4);
        let initial_p = cache.p_value();

        // Create scenario with B1 ghost hit
        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3);
        cache.insert("d", 4);
        cache.insert("e", 5); // Evicts "a" to B1

        println!(
            "Before ghost hit: p={}, t1={}, t2={}, b1={}, b2={}",
            cache.p_value(),
            cache.t1_len(),
            cache.t2_len(),
            cache.b1_len(),
            cache.b2_len()
        );
        println!("B1 contains a={}", cache.b1.contains(&"a"));

        // Ghost hit in B1 should increase p
        cache.insert("a", 10);

        println!(
            "After ghost hit: p={}, t1={}, t2={}, b1={}, b2={}",
            cache.p_value(),
            cache.t1_len(),
            cache.t2_len(),
            cache.b1_len(),
            cache.b2_len()
        );

        // Note: If b1.len() == b2.len() initially, delta would be 1
        // p should increase by at least 1
        assert!(
            cache.p_value() > initial_p,
            "Expected p to increase from {} but got {}",
            initial_p,
            cache.p_value()
        );
    }

    #[test]
    fn arc_remove() {
        let mut cache = ARCCore::new(10);

        cache.insert("key1", "value1");
        cache.insert("key2", "value2");
        assert_eq!(cache.len(), 2);

        let removed = cache.remove(&"key1");
        assert_eq!(removed, Some("value1"));
        assert_eq!(cache.len(), 1);
        assert!(!cache.contains(&"key1"));
        assert!(cache.contains(&"key2"));
    }

    #[test]
    fn arc_clear() {
        let mut cache = ARCCore::new(10);

        cache.insert("key1", "value1");
        cache.insert("key2", "value2");
        cache.get(&"key1"); // Promote to T2

        cache.clear();

        assert_eq!(cache.len(), 0);
        assert_eq!(cache.t1_len(), 0);
        assert_eq!(cache.t2_len(), 0);
        assert_eq!(cache.b1_len(), 0);
        assert_eq!(cache.b2_len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn arc_contains() {
        let mut cache = ARCCore::new(10);

        assert!(!cache.contains(&"key1"));

        cache.insert("key1", "value1");
        assert!(cache.contains(&"key1"));

        cache.remove(&"key1");
        assert!(!cache.contains(&"key1"));
    }

    #[test]
    fn arc_promotion_t1_to_t2() {
        let mut cache = ARCCore::new(10);

        // Insert into T1
        cache.insert("key", "value");
        assert_eq!(cache.t1_len(), 1);
        assert_eq!(cache.t2_len(), 0);

        // First access promotes to T2
        cache.get(&"key");
        assert_eq!(cache.t1_len(), 0);
        assert_eq!(cache.t2_len(), 1);

        // Second access stays in T2
        cache.get(&"key");
        assert_eq!(cache.t1_len(), 0);
        assert_eq!(cache.t2_len(), 1);
    }

    #[test]
    fn arc_multiple_entries() {
        let mut cache = ARCCore::new(5);

        for i in 0..5 {
            cache.insert(i, i * 10);
        }

        assert_eq!(cache.len(), 5);

        for i in 0..5 {
            assert_eq!(cache.get(&i), Some(&(i * 10)));
        }

        // All should be promoted to T2 after access
        assert_eq!(cache.t2_len(), 5);
        assert_eq!(cache.t1_len(), 0);
    }

    #[test]
    fn arc_eviction_and_ghost_tracking() {
        let mut cache = ARCCore::new(3);

        // Fill cache
        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.insert(3, 300);

        // Access 1 and 2 to promote them to T2
        cache.get(&1);
        cache.get(&2);

        assert_eq!(cache.t1_len(), 1); // 3 remains in T1
        assert_eq!(cache.t2_len(), 2); // 1, 2 promoted to T2

        // Insert 4. With p=1, t1_len=1, t2_len=2:
        // Condition: t1_len > p → 1 > 1 is false
        // So we evict from T2 (not T1)
        // T2 has [2 (MRU), 1 (LRU)], so 1 gets evicted to B2
        cache.insert(4, 400);

        assert_eq!(cache.len(), 3);
        assert!(!cache.contains(&1)); // 1 was evicted (LRU of T2)
        assert!(cache.contains(&2)); // 2 remains (MRU of T2)
        assert!(cache.contains(&3)); // 3 remains (in T1)
        assert!(cache.contains(&4)); // 4 just inserted
        assert_eq!(cache.b2_len(), 1); // 1 moved to B2 (evicted from T2)
    }

    #[test]
    fn arc_zero_capacity() {
        let mut cache = ARCCore::new(0);
        assert_eq!(cache.capacity(), 0);
        assert_eq!(cache.len(), 0);

        cache.insert("key", "value");
        assert_eq!(cache.len(), 0);
        assert!(!cache.contains(&"key"));
    }
}
