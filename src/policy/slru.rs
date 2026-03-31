//! Segmented LRU (SLRU) cache replacement policy.
//!
//! Implements the SLRU algorithm, which separates recently inserted items from
//! frequently accessed items using two LRU segments. This provides scan resistance
//! by preventing one-time accesses from polluting the protected (frequently used) segment.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                           SlruCore<K, V> Layout                             │
//! │                                                                             │
//! │   ┌─────────────────────────────────────────────────────────────────────┐   │
//! │   │  index: HashMap<K, NonNull<Node>>    nodes: Allocated on heap       │   │
//! │   │                                                                     │   │
//! │   │  ┌──────────┬───────────┐           ┌────────┬──────────────────┐   │   │
//! │   │  │   Key    │  NodePtr  │           │ Node   │ key,val,segment  │   │   │
//! │   │  ├──────────┼───────────┤           ├────────┼──────────────────┤   │   │
//! │   │  │  "page1" │   ptr_0   │──────────►│ Node0  │ k,v,Probationary │   │   │
//! │   │  │  "page2" │   ptr_1   │──────────►│ Node1  │ k,v,Protected    │   │   │
//! │   │  │  "page3" │   ptr_2   │──────────►│ Node2  │ k,v,Probationary │   │   │
//! │   │  └──────────┴───────────┘           └────────┴──────────────────┘   │   │
//! │   └─────────────────────────────────────────────────────────────────────┘   │
//! │                                                                             │
//! │   ┌─────────────────────────────────────────────────────────────────────┐   │
//! │   │                        Segment Organization                         │   │
//! │   │                                                                     │   │
//! │   │   PROBATIONARY (LRU)                     PROTECTED (LRU)            │   │
//! │   │   ┌─────────────────────────┐            ┌─────────────────────────┐│   │
//! │   │   │ MRU               LRU   │            │ MRU               LRU   ││   │
//! │   │   │  ▼                  ▼   │            │  ▼                  ▼   ││   │
//! │   │   │ [ptr_2] ◄──► [ptr_0] ◄┤ │            │ [ptr_1] ◄──► [...] ◄┤   ││   │
//! │   │   │  new        older evict │            │ hot          cold  evict││   │
//! │   │   └─────────────────────────┘            └─────────────────────────┘│   │
//! │   │                                                                     │   │
//! │   │   • New items enter probationary LRU                                │   │
//! │   │   • Re-access promotes to protected LRU                             │   │
//! │   │   • Eviction: probationary LRU first, then protected LRU            │   │
//! │   └─────────────────────────────────────────────────────────────────────┘   │
//! │                                                                             │
//! └─────────────────────────────────────────────────────────────────────────────┘
//!
//! Insert Flow (new key)
//! ──────────────────────
//!
//!   insert("new_key", value):
//!     1. Check index - not found
//!     2. Create Node with Segment::Probationary
//!     3. Allocate on heap → get NonNull<Node>
//!     4. Insert key→ptr into index
//!     5. Attach ptr to probationary MRU
//!     6. Evict if over capacity
//!
//! Access Flow (existing key)
//! ──────────────────────────
//!
//!   get("existing_key"):
//!     1. Lookup ptr in index
//!     2. Check node's segment:
//!        - If Probationary: promote to Protected (move to protected MRU)
//!        - If Protected: move to MRU position
//!     3. Return &value
//!
//! Eviction Flow
//! ─────────────
//!
//!   evict_if_needed():
//!     while len > capacity:
//!       if probationary.len > probationary_cap:
//!         evict from probationary LRU
//!       else:
//!         evict from protected LRU
//! ```
//!
//! ## Key Components
//!
//! - [`SlruCore`]: Main SLRU cache implementation
//!
//! ## Operations
//!
//! | Operation   | Time   | Notes                                      |
//! |-------------|--------|--------------------------------------------|
//! | `get`       | O(1)   | May promote from probationary to protected |
//! | `peek`      | O(1)   | Read without promotion                     |
//! | `insert`    | O(1)*  | *Amortized, may trigger evictions          |
//! | `contains`  | O(1)   | Index lookup only                          |
//! | `len`       | O(1)   | Returns total entries                      |
//! | `clear`     | O(n)   | Clears all structures                      |
//!
//! ## Algorithm Properties
//!
//! - **Scan Resistance**: One-time accesses stay in probationary, don't pollute protected
//! - **Frequency Awareness**: Repeated access promotes to protected LRU
//! - **Tunable**: `probationary_frac` controls probationary/protected size ratio
//!
//! ## Use Cases
//!
//! - Database buffer pools (scan-heavy workloads)
//! - File system caches
//! - Web caches with mixed access patterns
//!
//! ## Example Usage
//!
//! ```
//! use cachekit::policy::slru::SlruCore;
//!
//! // Create SLRU cache: 100 total capacity, 25% for probationary
//! let mut cache = SlruCore::new(100, 0.25);
//!
//! // Insert items (go to probationary)
//! cache.insert("page1", "content1");
//! cache.insert("page2", "content2");
//!
//! // First access promotes to protected
//! assert_eq!(cache.get(&"page1"), Some(&"content1"));
//!
//! // Second access keeps in protected (MRU position)
//! assert_eq!(cache.get(&"page1"), Some(&"content1"));
//!
//! assert_eq!(cache.len(), 2);
//! ```
//!
//! ## Removal Policy
//!
//! `SlruCore` supports key removal via [`Cache::remove`].
//! Removing an entry detaches it from its segment and adjusts the segment counters.
//!
//! ## Thread Safety
//!
//! - [`SlruCore`]: Not thread-safe, designed for single-threaded use
//! - For concurrent access, wrap in external synchronization
//!
//! ## Implementation Notes
//!
//! - Probationary uses LRU ordering (MRU at front, evict from back)
//! - Protected uses LRU ordering (MRU at front, evict from back)
//! - Promotion from probationary to protected happens on re-access
//! - Default `probationary_frac` of 0.25 means 25% of capacity for probationary
//!
//! ## References
//!
//! - Karedla et al., "Caching Strategies to Improve Disk System Performance", 1994
//! - Wikipedia: Cache replacement policies

#[cfg(feature = "metrics")]
use crate::metrics::metrics_impl::SlruMetrics;
#[cfg(feature = "metrics")]
use crate::metrics::snapshot::SlruMetricsSnapshot;
#[cfg(feature = "metrics")]
use crate::metrics::traits::{CoreMetricsRecorder, MetricsSnapshotProvider, SlruMetricsRecorder};
use crate::traits::Cache;
use rustc_hash::FxHashMap;
use std::hash::Hash;
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::ptr::NonNull;

/// Indicates which segment an entry resides in.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Segment {
    /// Entry is in the probationary segment (new/unproven entries).
    Probationary,
    /// Entry is in the protected segment (frequently accessed entries).
    Protected,
}

/// Node in the SLRU linked list.
///
/// Cache-line optimized layout with pointers first.
#[repr(C)]
struct Node<K, V> {
    prev: Option<NonNull<Node<K, V>>>,
    next: Option<NonNull<Node<K, V>>>,
    segment: Segment,
    key: K,
    value: V,
}

/// Core Segmented LRU (SLRU) cache implementation.
///
/// Implements the SLRU replacement algorithm with two segments:
/// - **Probationary**: LRU queue for newly inserted items
/// - **Protected**: LRU queue for frequently accessed items
///
/// New items enter probationary. Re-accessing an item in probationary promotes it
/// to protected. This provides scan resistance by keeping one-time accesses
/// from polluting the main cache.
///
/// `SlruCore` supports key removal via [`Cache::remove`].
/// Removing an entry detaches it from its segment and adjusts the segment counters.
///
/// # Type Parameters
///
/// - `K`: Key type, must be `Clone + Eq + Hash`
/// - `V`: Value type
///
/// # Example
///
/// ```
/// use cachekit::policy::slru::SlruCore;
///
/// // 100 capacity, 25% probationary
/// let mut cache = SlruCore::new(100, 0.25);
///
/// // Insert goes to probationary
/// cache.insert("key1", "value1");
/// assert!(cache.contains(&"key1"));
///
/// // First get promotes to protected
/// cache.get(&"key1");
///
/// // Update existing key
/// cache.insert("key1", "new_value");
/// assert_eq!(cache.get(&"key1"), Some(&"new_value"));
/// ```
///
/// # Eviction Behavior
///
/// When capacity is exceeded:
/// 1. If probationary exceeds its cap, evict from probationary LRU
/// 2. Otherwise, evict from protected LRU
///
/// # Implementation
///
/// Uses raw pointer linked lists for O(1) operations with minimal overhead.
pub struct SlruCore<K, V> {
    map: FxHashMap<K, NonNull<Node<K, V>>>,

    /// Probationary segment (LRU): head=MRU, tail=LRU
    probationary_head: Option<NonNull<Node<K, V>>>,
    probationary_tail: Option<NonNull<Node<K, V>>>,
    probationary_len: usize,

    /// Protected segment (LRU): head=MRU, tail=LRU
    protected_head: Option<NonNull<Node<K, V>>>,
    protected_tail: Option<NonNull<Node<K, V>>>,
    protected_len: usize,

    probationary_cap: usize,
    capacity: usize,

    #[cfg(feature = "metrics")]
    metrics: SlruMetrics,
}

// SAFETY: SlruCore exclusively owns all heap-allocated Node<K, V> via NonNull
// pointers. No aliasing occurs — each node is reachable only through the map
// or the linked lists, both owned by this struct. Sending the entire structure
// is safe when K and V are Send because ownership transfers atomically.
unsafe impl<K: Send, V: Send> Send for SlruCore<K, V> {}

// SAFETY: A shared &SlruCore only permits read-only operations (len, capacity,
// contains, peek, Debug). These access only the FxHashMap (Sync when K: Sync)
// and primitive counters. The &mut self requirement on get/insert/clear prevents
// data races on the linked-list pointers.
unsafe impl<K: Sync, V: Sync> Sync for SlruCore<K, V> {}

impl<K, V> SlruCore<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Creates a new SLRU cache with the specified capacity and probationary fraction.
    ///
    /// A typical value for `probationary_frac` is 0.25 (25% for probationary).
    ///
    /// # Panics
    ///
    /// Panics if `probationary_frac` is not in `0.0..=1.0` or is NaN.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::slru::SlruCore;
    ///
    /// // 100 capacity, 25% probationary (25 items max in probationary)
    /// let cache: SlruCore<String, i32> = SlruCore::new(100, 0.25);
    /// assert_eq!(cache.capacity(), 100);
    /// assert!(cache.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn new(capacity: usize, probationary_frac: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&probationary_frac),
            "probationary_frac must be in 0.0..=1.0, got {probationary_frac}"
        );

        let probationary_cap = (capacity as f64 * probationary_frac) as usize;
        let total_cap = capacity + probationary_cap;

        Self {
            map: FxHashMap::with_capacity_and_hasher(total_cap, Default::default()),
            probationary_head: None,
            probationary_tail: None,
            probationary_len: 0,
            protected_head: None,
            protected_tail: None,
            protected_len: 0,
            probationary_cap,
            capacity,
            #[cfg(feature = "metrics")]
            metrics: SlruMetrics::default(),
        }
    }

    /// Detach a node from its current segment.
    #[inline(always)]
    fn detach(&mut self, node_ptr: NonNull<Node<K, V>>) {
        unsafe {
            let node = node_ptr.as_ref();
            let prev = node.prev;
            let next = node.next;
            let segment = node.segment;

            let (head, tail, len) = match segment {
                Segment::Probationary => (
                    &mut self.probationary_head,
                    &mut self.probationary_tail,
                    &mut self.probationary_len,
                ),
                Segment::Protected => (
                    &mut self.protected_head,
                    &mut self.protected_tail,
                    &mut self.protected_len,
                ),
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

    /// Attach a node at the head of probationary segment (LRU: MRU at head).
    #[inline(always)]
    fn attach_probationary_head(&mut self, mut node_ptr: NonNull<Node<K, V>>) {
        unsafe {
            let node = node_ptr.as_mut();
            node.prev = None;
            node.next = self.probationary_head;
            node.segment = Segment::Probationary;

            match self.probationary_head {
                Some(mut h) => h.as_mut().prev = Some(node_ptr),
                None => self.probationary_tail = Some(node_ptr),
            }

            self.probationary_head = Some(node_ptr);
            self.probationary_len += 1;
        }
    }

    /// Attach a node at the head of protected segment (LRU: MRU at head).
    #[inline(always)]
    fn attach_protected_head(&mut self, mut node_ptr: NonNull<Node<K, V>>) {
        unsafe {
            let node = node_ptr.as_mut();
            node.prev = None;
            node.next = self.protected_head;
            node.segment = Segment::Protected;

            match self.protected_head {
                Some(mut h) => h.as_mut().prev = Some(node_ptr),
                None => self.protected_tail = Some(node_ptr),
            }

            self.protected_head = Some(node_ptr);
            self.protected_len += 1;
        }
    }

    /// Pop from probationary tail (LRU: LRU at tail).
    #[inline(always)]
    fn pop_probationary_tail(&mut self) -> Option<Box<Node<K, V>>> {
        self.probationary_tail.map(|tail_ptr| unsafe {
            let node = Box::from_raw(tail_ptr.as_ptr());

            self.probationary_tail = node.prev;
            match self.probationary_tail {
                Some(mut t) => t.as_mut().next = None,
                None => self.probationary_head = None,
            }
            self.probationary_len -= 1;

            node
        })
    }

    /// Pop from protected tail (LRU: LRU at tail).
    #[inline(always)]
    fn pop_protected_tail(&mut self) -> Option<Box<Node<K, V>>> {
        self.protected_tail.map(|tail_ptr| unsafe {
            let node = Box::from_raw(tail_ptr.as_ptr());

            self.protected_tail = node.prev;
            match self.protected_tail {
                Some(mut t) => t.as_mut().next = None,
                None => self.protected_head = None,
            }
            self.protected_len -= 1;

            node
        })
    }

    /// Returns a reference to the value without affecting segment position.
    ///
    /// Unlike [`get`](Self::get), this does not promote entries from probationary
    /// to protected and does not update MRU position.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::slru::SlruCore;
    ///
    /// let mut cache = SlruCore::new(100, 0.25);
    /// cache.insert("key", 42);
    ///
    /// // Peek does not trigger promotion
    /// assert_eq!(cache.peek(&"key"), Some(&42));
    /// assert_eq!(cache.peek(&"missing"), None);
    /// ```
    #[inline]
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.map.get(key).map(|&ptr| unsafe { &ptr.as_ref().value })
    }

    /// Retrieves a value by key, promoting from probationary to protected if needed.
    ///
    /// If the key is in probationary, accessing it promotes the entry to the
    /// protected segment (demonstrating it's not a one-time access).
    /// If already in protected, moves it to the MRU position.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::slru::SlruCore;
    ///
    /// let mut cache = SlruCore::new(100, 0.25);
    /// cache.insert("key", 42);
    ///
    /// // First access: in probationary, now promotes to protected
    /// assert_eq!(cache.get(&"key"), Some(&42));
    ///
    /// // Second access: already in protected, moves to MRU
    /// assert_eq!(cache.get(&"key"), Some(&42));
    ///
    /// // Missing key
    /// assert_eq!(cache.get(&"missing"), None);
    /// ```
    #[inline]
    pub fn get(&mut self, key: &K) -> Option<&V> {
        let node_ptr = match self.map.get(key) {
            Some(&ptr) => ptr,
            None => {
                #[cfg(feature = "metrics")]
                self.metrics.record_get_miss();
                return None;
            },
        };

        #[cfg(feature = "metrics")]
        self.metrics.record_get_hit();

        let segment = unsafe { node_ptr.as_ref().segment };

        match segment {
            Segment::Probationary => {
                self.detach(node_ptr);
                self.attach_protected_head(node_ptr);

                #[cfg(feature = "metrics")]
                self.metrics.record_probationary_to_protected();
            },
            Segment::Protected => {
                self.detach(node_ptr);
                self.attach_protected_head(node_ptr);
            },
        }

        unsafe { Some(&node_ptr.as_ref().value) }
    }

    /// Inserts or updates a key-value pair.
    ///
    /// - If the key exists, updates the value in place (no segment change)
    /// - If the key is new, inserts into the probationary segment
    /// - May trigger eviction if capacity is exceeded
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::slru::SlruCore;
    ///
    /// let mut cache = SlruCore::new(100, 0.25);
    ///
    /// // New insert goes to probationary
    /// cache.insert("key", "initial");
    /// assert_eq!(cache.len(), 1);
    ///
    /// // Update existing key
    /// cache.insert("key", "updated");
    /// assert_eq!(cache.get(&"key"), Some(&"updated"));
    /// assert_eq!(cache.len(), 1);  // Still 1 entry
    /// ```
    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        #[cfg(feature = "metrics")]
        self.metrics.record_insert_call();

        if self.capacity == 0 {
            return None;
        }

        if let Some(&node_ptr) = self.map.get(&key) {
            #[cfg(feature = "metrics")]
            self.metrics.record_insert_update();

            let old = unsafe { std::mem::replace(&mut (*node_ptr.as_ptr()).value, value) };
            return Some(old);
        }

        #[cfg(feature = "metrics")]
        self.metrics.record_insert_new();

        self.evict_if_needed();

        let node = Box::new(Node {
            prev: None,
            next: None,
            segment: Segment::Probationary,
            key: key.clone(),
            value,
        });
        let node_ptr = NonNull::new(Box::into_raw(node)).unwrap();

        self.map.insert(key, node_ptr);
        self.attach_probationary_head(node_ptr);

        #[cfg(debug_assertions)]
        self.validate_invariants();
        None
    }

    /// Evicts entries until there is room for a new entry.
    #[inline]
    fn evict_if_needed(&mut self) {
        while self.len() >= self.capacity {
            #[cfg(feature = "metrics")]
            self.metrics.record_evict_call();

            if self.probationary_len > self.probationary_cap {
                if let Some(node) = self.pop_probationary_tail() {
                    self.map.remove(&node.key);
                    #[cfg(feature = "metrics")]
                    self.metrics.record_evicted_entry();
                    continue;
                }
            }
            if let Some(node) = self.pop_protected_tail() {
                self.map.remove(&node.key);
                #[cfg(feature = "metrics")]
                {
                    self.metrics.record_evicted_entry();
                    self.metrics.record_protected_eviction();
                }
                continue;
            }
            if let Some(node) = self.pop_probationary_tail() {
                self.map.remove(&node.key);
                #[cfg(feature = "metrics")]
                self.metrics.record_evicted_entry();
                continue;
            }
            break;
        }
    }

    /// Returns the number of entries in the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::slru::SlruCore;
    ///
    /// let mut cache = SlruCore::new(100, 0.25);
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
    /// use cachekit::policy::slru::SlruCore;
    ///
    /// let mut cache: SlruCore<&str, i32> = SlruCore::new(100, 0.25);
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
    /// use cachekit::policy::slru::SlruCore;
    ///
    /// let cache: SlruCore<String, i32> = SlruCore::new(500, 0.25);
    /// assert_eq!(cache.capacity(), 500);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns `true` if the key exists in the cache.
    ///
    /// Does not affect segment positions (no promotion on contains).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::slru::SlruCore;
    ///
    /// let mut cache = SlruCore::new(100, 0.25);
    /// cache.insert("key", 42);
    ///
    /// assert!(cache.contains(&"key"));
    /// assert!(!cache.contains(&"missing"));
    /// ```
    #[inline]
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Returns the number of entries in the probationary segment.
    #[inline]
    pub fn probationary_len(&self) -> usize {
        self.probationary_len
    }

    /// Returns the number of entries in the protected segment.
    #[inline]
    pub fn protected_len(&self) -> usize {
        self.protected_len
    }

    /// Clears all entries from the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::slru::SlruCore;
    ///
    /// let mut cache = SlruCore::new(100, 0.25);
    /// cache.insert("a", 1);
    /// cache.insert("b", 2);
    ///
    /// cache.clear();
    /// assert!(cache.is_empty());
    /// assert!(!cache.contains(&"a"));
    /// ```
    pub fn clear(&mut self) {
        #[cfg(feature = "metrics")]
        self.metrics.record_clear();

        while self.pop_probationary_tail().is_some() {}
        while self.pop_protected_tail().is_some() {}
        self.map.clear();

        #[cfg(debug_assertions)]
        self.validate_invariants();
    }

    /// Removes a specific key-value pair, returning the value if it existed.
    ///
    /// Detaches the entry from its segment (probationary or protected) and
    /// adjusts the segment counters.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::slru::SlruCore;
    ///
    /// let mut cache = SlruCore::new(100, 0.25);
    /// cache.insert("key", 42);
    /// assert_eq!(cache.remove(&"key"), Some(42));
    /// assert!(!cache.contains(&"key"));
    /// ```
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let node_ptr = self.map.remove(key)?;
        self.detach(node_ptr);
        unsafe {
            let node = Box::from_raw(node_ptr.as_ptr());
            Some(node.value)
        }
    }

    /// Returns an iterator over shared references to cached key-value pairs.
    ///
    /// Visits probationary entries (MRU to LRU) then protected entries.
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            current: self.probationary_head,
            protected_head: self.protected_head,
            in_protected: false,
            remaining: self.probationary_len + self.protected_len,
            _marker: PhantomData,
        }
    }

    /// Returns a mutable iterator over cached key-value pairs.
    ///
    /// Keys are yielded as shared references; values as mutable references.
    /// Modifying values does not affect eviction ordering.
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut {
            current: self.probationary_head,
            protected_head: self.protected_head,
            in_protected: false,
            remaining: self.probationary_len + self.protected_len,
            _marker: PhantomData,
        }
    }

    /// Validates internal data structure invariants.
    ///
    /// This method checks that:
    /// - All nodes in map are reachable from either probationary or protected lists
    /// - List lengths match tracked counts
    /// - No cycles exist in the lists
    /// - All nodes have valid prev/next pointers
    ///
    /// Only runs when debug assertions are enabled.
    #[cfg(debug_assertions)]
    fn validate_invariants(&self) {
        let mut prob_count = 0;
        let mut current = self.probationary_head;
        let mut visited = std::collections::HashSet::new();

        while let Some(ptr) = current {
            prob_count += 1;
            assert!(visited.insert(ptr), "Cycle detected in probationary list");
            assert!(
                prob_count <= self.map.len(),
                "Probationary count exceeds total entries"
            );

            unsafe {
                let node = ptr.as_ref();
                assert!(
                    matches!(node.segment, Segment::Probationary),
                    "Non-probationary node in probationary list"
                );
                current = node.next;
            }
        }

        debug_assert_eq!(
            prob_count, self.probationary_len,
            "Probationary count mismatch"
        );

        let mut prot_count = 0;
        let mut current = self.protected_head;
        visited.clear();

        while let Some(ptr) = current {
            prot_count += 1;
            assert!(visited.insert(ptr), "Cycle detected in protected list");
            assert!(
                prot_count <= self.map.len(),
                "Protected count exceeds total entries"
            );

            unsafe {
                let node = ptr.as_ref();
                assert!(
                    matches!(node.segment, Segment::Protected),
                    "Non-protected node in protected list"
                );
                current = node.next;
            }
        }

        debug_assert_eq!(prot_count, self.protected_len, "Protected count mismatch");

        debug_assert_eq!(
            prob_count + prot_count,
            self.map.len(),
            "List counts don't match map size"
        );

        for &node_ptr in self.map.values() {
            unsafe {
                let node = node_ptr.as_ref();
                match node.segment {
                    Segment::Probationary => {
                        debug_assert!(prob_count > 0, "Node marked probationary but list empty");
                    },
                    Segment::Protected => {
                        debug_assert!(prot_count > 0, "Node marked protected but list empty");
                    },
                }
            }
        }
    }
}

#[cfg(feature = "metrics")]
impl<K, V> SlruCore<K, V>
where
    K: Clone + Eq + Hash,
{
    pub fn metrics_snapshot(&self) -> SlruMetricsSnapshot {
        SlruMetricsSnapshot {
            get_calls: self.metrics.get_calls,
            get_hits: self.metrics.get_hits,
            get_misses: self.metrics.get_misses,
            insert_calls: self.metrics.insert_calls,
            insert_updates: self.metrics.insert_updates,
            insert_new: self.metrics.insert_new,
            evict_calls: self.metrics.evict_calls,
            evicted_entries: self.metrics.evicted_entries,
            probationary_to_protected: self.metrics.probationary_to_protected,
            protected_evictions: self.metrics.protected_evictions,
            cache_len: self.len(),
            capacity: self.capacity(),
        }
    }
}

#[cfg(feature = "metrics")]
impl<K, V> MetricsSnapshotProvider<SlruMetricsSnapshot> for SlruCore<K, V>
where
    K: Clone + Eq + Hash,
{
    fn snapshot(&self) -> SlruMetricsSnapshot {
        self.metrics_snapshot()
    }
}

impl<K, V> Drop for SlruCore<K, V> {
    fn drop(&mut self) {
        unsafe {
            let mut current = self.probationary_head.take();
            while let Some(ptr) = current {
                current = ptr.as_ref().next;
                drop(Box::from_raw(ptr.as_ptr()));
            }
            let mut current = self.protected_head.take();
            while let Some(ptr) = current {
                current = ptr.as_ref().next;
                drop(Box::from_raw(ptr.as_ptr()));
            }
        }
    }
}

impl<K, V> std::fmt::Debug for SlruCore<K, V>
where
    K: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SlruCore")
            .field("capacity", &self.capacity)
            .field("probationary_cap", &self.probationary_cap)
            .field("len", &self.map.len())
            .field("probationary_len", &self.probationary_len)
            .field("protected_len", &self.protected_len)
            .finish_non_exhaustive()
    }
}

impl<K, V> Default for SlruCore<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Returns an empty `SlruCore` with capacity 0.
    ///
    /// Use [`SlruCore::new`] to specify a capacity.
    fn default() -> Self {
        Self::new(0, 0.25)
    }
}

impl<K, V> Clone for SlruCore<K, V>
where
    K: Clone + Eq + Hash,
    V: Clone,
{
    fn clone(&self) -> Self {
        let mut new_cache = SlruCore::new(
            self.capacity,
            if self.capacity > 0 {
                self.probationary_cap as f64 / self.capacity as f64
            } else {
                0.25
            },
        );

        // Rebuild probationary from tail (LRU) to head (MRU) so attach_head
        // reproduces the original ordering.
        let mut prob_entries = Vec::with_capacity(self.probationary_len);
        let mut current = self.probationary_tail;
        while let Some(ptr) = current {
            unsafe {
                let node = ptr.as_ref();
                prob_entries.push((node.key.clone(), node.value.clone()));
                current = node.prev;
            }
        }
        for (key, value) in prob_entries {
            let node = Box::new(Node {
                prev: None,
                next: None,
                segment: Segment::Probationary,
                key: key.clone(),
                value,
            });
            let ptr = NonNull::new(Box::into_raw(node)).unwrap();
            new_cache.map.insert(key, ptr);
            new_cache.attach_probationary_head(ptr);
        }

        let mut prot_entries = Vec::with_capacity(self.protected_len);
        let mut current = self.protected_tail;
        while let Some(ptr) = current {
            unsafe {
                let node = ptr.as_ref();
                prot_entries.push((node.key.clone(), node.value.clone()));
                current = node.prev;
            }
        }
        for (key, value) in prot_entries {
            let node = Box::new(Node {
                prev: None,
                next: None,
                segment: Segment::Protected,
                key: key.clone(),
                value,
            });
            let ptr = NonNull::new(Box::into_raw(node)).unwrap();
            new_cache.map.insert(key, ptr);
            new_cache.attach_protected_head(ptr);
        }

        #[cfg(feature = "metrics")]
        {
            new_cache.metrics = self.metrics;
        }

        new_cache
    }
}

/// Implementation of the [`Cache`] trait for SLRU.
///
/// Allows `SlruCore` to be used through the unified cache interface.
///
/// # Example
///
/// ```
/// use cachekit::traits::Cache;
/// use cachekit::policy::slru::SlruCore;
///
/// let mut cache: SlruCore<&str, i32> = SlruCore::new(100, 0.25);
///
/// // Use via Cache trait
/// cache.insert("key", 42);
/// assert_eq!(cache.get(&"key"), Some(&42));
/// assert!(cache.contains(&"key"));
/// ```
impl<K, V> Cache<K, V> for SlruCore<K, V>
where
    K: Clone + Eq + Hash,
{
    #[inline]
    fn contains(&self, key: &K) -> bool {
        SlruCore::contains(self, key)
    }

    #[inline]
    fn len(&self) -> usize {
        SlruCore::len(self)
    }

    #[inline]
    fn capacity(&self) -> usize {
        SlruCore::capacity(self)
    }

    #[inline]
    fn peek(&self, key: &K) -> Option<&V> {
        SlruCore::peek(self, key)
    }

    #[inline]
    fn get(&mut self, key: &K) -> Option<&V> {
        SlruCore::get(self, key)
    }

    #[inline]
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        SlruCore::insert(self, key, value)
    }

    #[inline]
    fn remove(&mut self, key: &K) -> Option<V> {
        SlruCore::remove(self, key)
    }

    fn clear(&mut self) {
        SlruCore::clear(self);
    }
}

// ---------------------------------------------------------------------------
// Iterators
// ---------------------------------------------------------------------------

/// Iterator over shared references to cached key-value pairs.
///
/// Created by [`SlruCore::iter`]. Visits probationary entries (MRU to LRU)
/// then protected entries.
pub struct Iter<'a, K, V> {
    current: Option<NonNull<Node<K, V>>>,
    protected_head: Option<NonNull<Node<K, V>>>,
    in_protected: bool,
    remaining: usize,
    _marker: PhantomData<&'a (K, V)>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(node_ptr) = self.current {
                unsafe {
                    let node = node_ptr.as_ref();
                    self.current = node.next;
                    self.remaining -= 1;
                    return Some((&node.key, &node.value));
                }
            } else if !self.in_protected {
                self.in_protected = true;
                self.current = self.protected_head;
            } else {
                return None;
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K, V> ExactSizeIterator for Iter<'_, K, V> {}
impl<K, V> FusedIterator for Iter<'_, K, V> {}

/// Iterator over mutable references to cached values.
///
/// Created by [`SlruCore::iter_mut`]. Keys are shared; values are mutable.
pub struct IterMut<'a, K, V> {
    current: Option<NonNull<Node<K, V>>>,
    protected_head: Option<NonNull<Node<K, V>>>,
    in_protected: bool,
    remaining: usize,
    _marker: PhantomData<&'a mut (K, V)>,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(mut node_ptr) = self.current {
                unsafe {
                    let node = node_ptr.as_mut();
                    self.current = node.next;
                    self.remaining -= 1;
                    return Some((&node.key, &mut node.value));
                }
            } else if !self.in_protected {
                self.in_protected = true;
                self.current = self.protected_head;
            } else {
                return None;
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K, V> ExactSizeIterator for IterMut<'_, K, V> {}
impl<K, V> FusedIterator for IterMut<'_, K, V> {}

/// Owning iterator over cached key-value pairs.
///
/// Created by calling [`IntoIterator`] on an `SlruCore`.
pub struct IntoIter<K, V> {
    current: Option<NonNull<Node<K, V>>>,
    protected_head: Option<NonNull<Node<K, V>>>,
    in_protected: bool,
    remaining: usize,
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(node_ptr) = self.current {
                unsafe {
                    let node = Box::from_raw(node_ptr.as_ptr());
                    self.current = node.next;
                    self.remaining -= 1;
                    return Some((node.key, node.value));
                }
            } else if !self.in_protected {
                self.in_protected = true;
                self.current = self.protected_head;
            } else {
                return None;
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K, V> ExactSizeIterator for IntoIter<K, V> {}
impl<K, V> FusedIterator for IntoIter<K, V> {}

impl<K, V> Drop for IntoIter<K, V> {
    fn drop(&mut self) {
        while self.next().is_some() {}
    }
}

// SAFETY: IntoIter exclusively owns its nodes (moved out of SlruCore).
unsafe impl<K: Send, V: Send> Send for IntoIter<K, V> {}
unsafe impl<K: Sync, V: Sync> Sync for IntoIter<K, V> {}

impl<K, V> IntoIterator for SlruCore<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(mut self) -> IntoIter<K, V> {
        let iter = IntoIter {
            current: self.probationary_head,
            protected_head: self.protected_head,
            in_protected: false,
            remaining: self.probationary_len + self.protected_len,
        };
        // Prevent Drop from double-freeing nodes the iterator now owns.
        self.probationary_head = None;
        self.probationary_tail = None;
        self.probationary_len = 0;
        self.protected_head = None;
        self.protected_tail = None;
        self.protected_len = 0;
        iter
    }
}

impl<'a, K, V> IntoIterator for &'a SlruCore<K, V>
where
    K: Clone + Eq + Hash,
{
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Iter<'a, K, V> {
        self.iter()
    }
}

impl<'a, K, V> IntoIterator for &'a mut SlruCore<K, V>
where
    K: Clone + Eq + Hash,
{
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    fn into_iter(self) -> IterMut<'a, K, V> {
        self.iter_mut()
    }
}

impl<K, V> FromIterator<(K, V)> for SlruCore<K, V>
where
    K: Clone + Eq + Hash,
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut cache = SlruCore::new(lower.max(16), 0.25);
        cache.extend(iter);
        cache
    }
}

impl<K, V> Extend<(K, V)> for SlruCore<K, V>
where
    K: Clone + Eq + Hash,
{
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==============================================
    // SlruCore Basic Operations
    // ==============================================

    mod basic_operations {
        use super::*;

        #[test]
        fn new_cache_is_empty() {
            let cache: SlruCore<&str, i32> = SlruCore::new(100, 0.25);
            assert!(cache.is_empty());
            assert_eq!(cache.len(), 0);
            assert_eq!(cache.capacity(), 100);
        }

        #[test]
        fn insert_and_get() {
            let mut cache = SlruCore::new(100, 0.25);
            cache.insert("key1", "value1");

            assert_eq!(cache.len(), 1);
            assert_eq!(cache.get(&"key1"), Some(&"value1"));
        }

        #[test]
        fn insert_multiple_items() {
            let mut cache = SlruCore::new(100, 0.25);
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
            let mut cache: SlruCore<&str, i32> = SlruCore::new(100, 0.25);
            cache.insert("exists", 42);

            assert_eq!(cache.get(&"missing"), None);
        }

        #[test]
        fn update_existing_key() {
            let mut cache = SlruCore::new(100, 0.25);
            cache.insert("key", "initial");
            cache.insert("key", "updated");

            assert_eq!(cache.len(), 1);
            assert_eq!(cache.get(&"key"), Some(&"updated"));
        }

        #[test]
        fn contains_returns_correct_result() {
            let mut cache = SlruCore::new(100, 0.25);
            cache.insert("exists", 1);

            assert!(cache.contains(&"exists"));
            assert!(!cache.contains(&"missing"));
        }

        #[test]
        fn contains_does_not_promote() {
            let mut cache: SlruCore<String, i32> = SlruCore::new(10, 0.3);
            cache.insert("a".to_string(), 1);
            cache.insert("b".to_string(), 2);
            cache.insert("c".to_string(), 3);

            assert!(cache.contains(&"a".to_string()));
            assert!(cache.contains(&"b".to_string()));
            assert!(cache.contains(&"c".to_string()));

            for i in 0..10 {
                cache.insert(format!("new{}", i), i);
            }

            assert!(!cache.contains(&"a".to_string()));
            assert!(!cache.contains(&"b".to_string()));
            assert!(!cache.contains(&"c".to_string()));
        }

        #[test]
        fn clear_removes_all_entries() {
            let mut cache = SlruCore::new(100, 0.25);
            cache.insert("a", 1);
            cache.insert("b", 2);
            cache.get(&"a");

            cache.clear();

            assert!(cache.is_empty());
            assert_eq!(cache.len(), 0);
            assert!(!cache.contains(&"a"));
            assert!(!cache.contains(&"b"));
        }

        #[test]
        fn capacity_returns_correct_value() {
            let cache: SlruCore<i32, i32> = SlruCore::new(500, 0.25);
            assert_eq!(cache.capacity(), 500);
        }
    }

    // ==============================================
    // Peek
    // ==============================================

    mod peek_tests {
        use super::*;

        #[test]
        fn peek_returns_value_without_promotion() {
            let mut cache = SlruCore::new(100, 0.25);
            cache.insert("key", 42);

            assert_eq!(cache.peek(&"key"), Some(&42));
            assert_eq!(cache.peek(&"missing"), None);
        }

        #[test]
        fn peek_does_not_promote_to_protected() {
            let mut cache: SlruCore<String, i32> = SlruCore::new(10, 0.3);
            cache.insert("key".to_string(), 1);

            // Peek many times — should NOT promote
            for _ in 0..10 {
                assert_eq!(cache.peek(&"key".to_string()), Some(&1));
            }

            // Fill up to trigger eviction — "key" should be evicted from probationary
            for i in 0..12 {
                cache.insert(format!("filler{}", i), i);
            }

            assert!(!cache.contains(&"key".to_string()));
        }
    }

    // ==============================================
    // Segment Behavior (Probationary vs Protected)
    // ==============================================

    mod segment_behavior {
        use super::*;

        #[test]
        fn new_insert_goes_to_probationary() {
            let mut cache = SlruCore::new(10, 0.3);
            cache.insert("key", "value");

            assert!(cache.contains(&"key"));
            assert_eq!(cache.len(), 1);
            assert_eq!(cache.probationary_len(), 1);
            assert_eq!(cache.protected_len(), 0);
        }

        #[test]
        fn get_promotes_from_probationary_to_protected() {
            let mut cache: SlruCore<String, i32> = SlruCore::new(10, 0.3);
            cache.insert("key".to_string(), 0);

            let _ = cache.get(&"key".to_string());

            for i in 0..12 {
                cache.insert(format!("new{}", i), i);
            }

            assert!(cache.contains(&"key".to_string()));
        }

        #[test]
        fn item_in_protected_stays_in_protected() {
            let mut cache = SlruCore::new(10, 0.3);
            cache.insert("key", "value");

            cache.get(&"key");
            cache.get(&"key");
            cache.get(&"key");

            assert_eq!(cache.get(&"key"), Some(&"value"));
        }

        #[test]
        fn multiple_accesses_keep_item_alive() {
            let mut cache: SlruCore<String, i32> = SlruCore::new(10, 0.3);

            cache.insert("hot".to_string(), 0);
            cache.get(&"hot".to_string());

            for i in 0..15 {
                cache.insert(format!("cold{}", i), i);
                cache.get(&"hot".to_string());
            }

            assert!(cache.contains(&"hot".to_string()));
        }

        #[test]
        fn segment_len_accessors() {
            let mut cache = SlruCore::new(100, 0.25);
            cache.insert("a", 1);
            cache.insert("b", 2);
            assert_eq!(cache.probationary_len(), 2);
            assert_eq!(cache.protected_len(), 0);

            cache.get(&"a");
            assert_eq!(cache.probationary_len(), 1);
            assert_eq!(cache.protected_len(), 1);
        }
    }

    // ==============================================
    // Eviction Behavior
    // ==============================================

    mod eviction_behavior {
        use super::*;

        #[test]
        fn eviction_occurs_when_over_capacity() {
            let mut cache = SlruCore::new(5, 0.2);

            for i in 0..10 {
                cache.insert(i, i * 10);
            }

            assert_eq!(cache.len(), 5);
        }

        #[test]
        fn probationary_evicts_lru_order() {
            let mut cache = SlruCore::new(5, 0.4);

            cache.insert("first", 1);
            cache.insert("second", 2);
            cache.insert("third", 3);
            cache.insert("fourth", 4);
            cache.insert("fifth", 5);
            cache.insert("sixth", 6);

            assert!(!cache.contains(&"first"));
            assert_eq!(cache.len(), 5);
        }

        #[test]
        fn protected_evicts_lru_when_probationary_under_cap() {
            let mut cache = SlruCore::new(5, 0.4);

            cache.insert("p1", 1);
            cache.get(&"p1");
            cache.insert("p2", 2);
            cache.get(&"p2");
            cache.insert("p3", 3);
            cache.get(&"p3");

            cache.insert("new1", 10);
            cache.insert("new2", 20);
            cache.insert("new3", 30);

            assert!(!cache.contains(&"p1"));
            assert_eq!(cache.len(), 5);
        }

        #[test]
        fn scan_items_evicted_before_hot_items() {
            let mut cache: SlruCore<String, i32> = SlruCore::new(10, 0.3);

            cache.insert("hot1".to_string(), 1);
            cache.get(&"hot1".to_string());
            cache.insert("hot2".to_string(), 2);
            cache.get(&"hot2".to_string());

            for i in 0..20 {
                cache.insert(format!("scan{}", i), i);
            }

            assert!(cache.contains(&"hot1".to_string()));
            assert!(cache.contains(&"hot2".to_string()));
            assert_eq!(cache.len(), 10);
        }

        #[test]
        fn eviction_removes_from_index() {
            let mut cache = SlruCore::new(3, 0.33);

            cache.insert("a", 1);
            cache.insert("b", 2);
            cache.insert("c", 3);

            assert!(cache.contains(&"a"));

            cache.insert("d", 4);

            assert!(!cache.contains(&"a"));
            assert_eq!(cache.get(&"a"), None);
        }
    }

    // ==============================================
    // Scan Resistance
    // ==============================================

    mod scan_resistance {
        use super::*;

        #[test]
        fn scan_does_not_pollute_protected() {
            let mut cache = SlruCore::new(100, 0.25);

            for i in 0..50 {
                let key = format!("working{}", i);
                cache.insert(key.clone(), i);
                cache.get(&key);
            }

            for i in 0..200 {
                cache.insert(format!("scan{}", i), i);
            }

            let mut working_set_hits = 0;
            for i in 0..50 {
                if cache.contains(&format!("working{}", i)) {
                    working_set_hits += 1;
                }
            }

            assert!(
                working_set_hits >= 40,
                "Working set should survive scan, but only {} items remained",
                working_set_hits
            );
        }

        #[test]
        fn one_time_access_stays_in_probationary() {
            let mut cache: SlruCore<String, i32> = SlruCore::new(10, 0.3);

            cache.insert("once".to_string(), 1);

            for i in 0..5 {
                cache.insert(format!("other{}", i), i);
            }

            cache.get(&"once".to_string());

            for i in 0..10 {
                cache.insert(format!("new{}", i), i);
            }

            assert!(cache.contains(&"once".to_string()));
        }

        #[test]
        fn repeated_scans_dont_evict_hot_items() {
            let mut cache = SlruCore::new(20, 0.25);

            for i in 0..10 {
                let key = format!("hot{}", i);
                cache.insert(key.clone(), i);
                cache.get(&key);
                cache.get(&key);
                cache.get(&key);
            }

            for scan in 0..3 {
                for i in 0..30 {
                    cache.insert(format!("scan{}_{}", scan, i), i);
                }
            }

            let mut hot_survivors = 0;
            for i in 0..10 {
                if cache.contains(&format!("hot{}", i)) {
                    hot_survivors += 1;
                }
            }

            assert!(
                hot_survivors >= 8,
                "Hot items should survive scans, but only {} survived",
                hot_survivors
            );
        }
    }

    // ==============================================
    // Edge Cases
    // ==============================================

    mod edge_cases {
        use super::*;

        #[test]
        fn single_capacity_cache() {
            let mut cache = SlruCore::new(1, 0.5);

            cache.insert("a", 1);
            assert_eq!(cache.get(&"a"), Some(&1));

            cache.insert("b", 2);
            assert!(!cache.contains(&"a"));
            assert_eq!(cache.get(&"b"), Some(&2));
        }

        #[test]
        fn zero_probationary_fraction() {
            let mut cache = SlruCore::new(10, 0.0);

            for i in 0..10 {
                cache.insert(i, i * 10);
            }

            assert_eq!(cache.len(), 10);

            cache.insert(100, 1000);
            assert_eq!(cache.len(), 10);
        }

        #[test]
        fn one_hundred_percent_probationary() {
            let mut cache = SlruCore::new(10, 1.0);

            for i in 0..10 {
                cache.insert(i, i * 10);
            }

            for i in 0..10 {
                cache.get(&i);
            }

            assert_eq!(cache.len(), 10);
        }

        #[test]
        fn get_after_update() {
            let mut cache = SlruCore::new(100, 0.25);

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
            let mut cache = SlruCore::new(10000, 0.25);

            for i in 0..10000 {
                cache.insert(i, i * 2);
            }

            assert_eq!(cache.len(), 10000);

            assert_eq!(cache.get(&5000), Some(&10000));
            assert_eq!(cache.get(&9999), Some(&19998));
        }

        #[test]
        fn empty_cache_operations() {
            let mut cache: SlruCore<i32, i32> = SlruCore::new(100, 0.25);

            assert!(cache.is_empty());
            assert_eq!(cache.get(&1), None);
            assert!(!cache.contains(&1));

            cache.clear();
            assert!(cache.is_empty());
        }

        #[test]
        fn small_fractions() {
            let mut cache = SlruCore::new(100, 0.01);

            for i in 0..10 {
                cache.insert(i, i);
            }

            assert_eq!(cache.len(), 10);
        }

        #[test]
        fn string_keys_and_values() {
            let mut cache = SlruCore::new(100, 0.25);

            cache.insert(String::from("hello"), String::from("world"));
            cache.insert(String::from("foo"), String::from("bar"));

            assert_eq!(
                cache.get(&String::from("hello")),
                Some(&String::from("world"))
            );
            assert_eq!(cache.get(&String::from("foo")), Some(&String::from("bar")));
        }

        #[test]
        fn integer_keys() {
            let mut cache = SlruCore::new(100, 0.25);

            for i in 0..50 {
                cache.insert(i, format!("value_{}", i));
            }

            assert_eq!(cache.get(&25), Some(&String::from("value_25")));
            assert_eq!(cache.get(&49), Some(&String::from("value_49")));
        }

        #[test]
        #[should_panic(expected = "probationary_frac must be in 0.0..=1.0")]
        fn negative_fraction_panics() {
            let _cache: SlruCore<i32, i32> = SlruCore::new(100, -0.5);
        }

        #[test]
        #[should_panic(expected = "probationary_frac must be in 0.0..=1.0")]
        fn fraction_above_one_panics() {
            let _cache: SlruCore<i32, i32> = SlruCore::new(100, 1.5);
        }

        #[test]
        #[should_panic(expected = "probationary_frac must be in 0.0..=1.0")]
        fn nan_fraction_panics() {
            let _cache: SlruCore<i32, i32> = SlruCore::new(100, f64::NAN);
        }
    }

    // ==============================================
    // Boundary Tests
    // ==============================================

    mod boundary_tests {
        use super::*;

        #[test]
        fn exact_capacity_no_eviction() {
            let mut cache = SlruCore::new(10, 0.3);

            for i in 0..10 {
                cache.insert(i, i);
            }

            assert_eq!(cache.len(), 10);
            for i in 0..10 {
                assert!(cache.contains(&i));
            }
        }

        #[test]
        fn one_over_capacity_triggers_eviction() {
            let mut cache = SlruCore::new(10, 0.3);

            for i in 0..10 {
                cache.insert(i, i);
            }

            cache.insert(10, 10);

            assert_eq!(cache.len(), 10);
            assert!(!cache.contains(&0));
            assert!(cache.contains(&10));
        }

        #[test]
        fn probationary_cap_boundary() {
            let mut cache = SlruCore::new(10, 0.3);

            cache.insert("a", 1);
            cache.insert("b", 2);
            cache.insert("c", 3);

            assert_eq!(cache.len(), 3);

            cache.insert("d", 4);
            assert_eq!(cache.len(), 4);

            for key in &["a", "b", "c", "d"] {
                assert!(cache.contains(key));
            }
        }

        #[test]
        fn promotion_fills_protected() {
            let mut cache = SlruCore::new(10, 0.3);

            for i in 0..5 {
                cache.insert(i, i);
            }

            for i in 0..5 {
                cache.get(&i);
            }

            for i in 5..10 {
                cache.insert(i, i);
            }

            assert_eq!(cache.len(), 10);

            cache.insert(10, 10);
            assert_eq!(cache.len(), 10);
        }
    }

    // ==============================================
    // Regression Tests
    // ==============================================

    mod regression_tests {
        use super::*;

        #[test]
        fn promotion_actually_moves_to_protected_segment() {
            let mut cache: SlruCore<String, i32> = SlruCore::new(5, 0.4);

            cache.insert("key".to_string(), 0);
            cache.get(&"key".to_string());

            cache.insert("p1".to_string(), 1);
            cache.insert("p2".to_string(), 2);
            cache.insert("p3".to_string(), 3);
            cache.insert("p4".to_string(), 4);

            assert!(
                cache.contains(&"key".to_string()),
                "Promoted item should be in protected segment and survive probationary eviction"
            );
        }

        #[test]
        fn update_preserves_segment_position() {
            let mut cache: SlruCore<String, i32> = SlruCore::new(10, 0.3);

            cache.insert("key".to_string(), 1);
            cache.get(&"key".to_string());

            cache.insert("key".to_string(), 2);

            assert_eq!(cache.get(&"key".to_string()), Some(&2));

            for i in 0..15 {
                cache.insert(format!("other{}", i), i);
            }

            assert!(cache.contains(&"key".to_string()));
        }

        #[test]
        fn eviction_order_consistency() {
            for _ in 0..10 {
                let mut cache = SlruCore::new(5, 0.4);

                cache.insert("a", 1);
                cache.insert("b", 2);
                cache.insert("c", 3);
                cache.insert("d", 4);
                cache.insert("e", 5);
                cache.insert("f", 6);

                assert!(!cache.contains(&"a"), "First item should be evicted");
                assert!(cache.contains(&"f"), "New item should exist");
            }
        }
    }

    // ==============================================
    // Workload Simulation
    // ==============================================

    mod workload_simulation {
        use super::*;

        #[test]
        fn database_buffer_pool_workload() {
            let mut cache = SlruCore::new(100, 0.25);

            for i in 0..10 {
                let key = format!("index_page_{}", i);
                cache.insert(key.clone(), format!("index_data_{}", i));
                cache.get(&key);
                cache.get(&key);
            }

            for i in 0..200 {
                cache.insert(format!("table_page_{}", i), format!("row_data_{}", i));
            }

            let mut index_hits = 0;
            for i in 0..10 {
                if cache.contains(&format!("index_page_{}", i)) {
                    index_hits += 1;
                }
            }

            assert!(
                index_hits >= 8,
                "Index pages should survive table scan, got {} hits",
                index_hits
            );
        }

        #[test]
        fn web_cache_simulation() {
            let mut cache: SlruCore<String, String> = SlruCore::new(50, 0.3);

            let popular = vec!["home", "about", "products", "contact"];
            for page in &popular {
                cache.insert(page.to_string(), format!("{}_content", page));
                cache.get(&page.to_string());
                cache.get(&page.to_string());
            }

            for i in 0..100 {
                cache.insert(format!("blog_post_{}", i), format!("content_{}", i));
            }

            for page in &popular {
                assert!(
                    cache.contains(&page.to_string()),
                    "Popular page '{}' should survive",
                    page
                );
            }
        }

        #[test]
        fn mixed_workload() {
            let mut cache = SlruCore::new(100, 0.25);

            for i in 0..30 {
                let key = format!("working_{}", i);
                cache.insert(key.clone(), i);
                cache.get(&key);
            }

            for round in 0..5 {
                for i in (0..30).step_by(3) {
                    cache.get(&format!("working_{}", i));
                }

                for i in 0..20 {
                    cache.insert(format!("round_{}_{}", round, i), i);
                }
            }

            let mut working_set_hits = 0;
            for i in (0..30).step_by(3) {
                if cache.contains(&format!("working_{}", i)) {
                    working_set_hits += 1;
                }
            }

            assert!(
                working_set_hits >= 8,
                "Frequently accessed working set should survive, got {} hits",
                working_set_hits
            );
        }
    }

    // ==============================================
    // Validation Tests
    // ==============================================

    #[test]
    #[cfg(debug_assertions)]
    fn validate_invariants_after_operations() {
        let mut cache = SlruCore::new(10, 0.3);

        for i in 1..=10 {
            cache.insert(i, i * 100);
        }
        cache.validate_invariants();

        for _ in 0..3 {
            cache.get(&1);
            cache.get(&2);
        }
        cache.validate_invariants();

        cache.insert(11, 1100);
        cache.validate_invariants();

        cache.insert(12, 1200);
        cache.validate_invariants();

        cache.clear();
        cache.validate_invariants();

        assert_eq!(cache.len(), 0);
    }

    #[test]
    #[cfg(debug_assertions)]
    fn validate_invariants_with_segment_transitions() {
        let mut cache = SlruCore::new(5, 0.4);
        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.insert(3, 300);

        cache.get(&1);
        cache.validate_invariants();

        cache.get(&2);
        cache.validate_invariants();

        cache.insert(4, 400);
        cache.insert(5, 500);
        cache.validate_invariants();

        cache.insert(6, 600);
        cache.validate_invariants();

        assert_eq!(cache.len(), 5);
    }

    // ==============================================
    // Standard Trait Tests
    // ==============================================

    #[test]
    fn zero_capacity_rejects_inserts() {
        let mut cache: SlruCore<&str, i32> = SlruCore::new(0, 0.25);
        assert_eq!(cache.capacity(), 0);

        cache.insert("key", 42);

        assert_eq!(
            cache.len(),
            0,
            "SlruCore with capacity=0 should reject inserts"
        );
    }

    #[test]
    fn trait_insert_returns_old_value() {
        let mut cache: SlruCore<&str, i32> = SlruCore::new(10, 0.25);

        let first = Cache::insert(&mut cache, "key", 1);
        assert_eq!(first, None);

        let second = Cache::insert(&mut cache, "key", 2);
        assert_eq!(
            second,
            Some(1),
            "Second insert via trait should return old value"
        );
    }

    #[test]
    fn inherent_insert_updates_value() {
        let mut cache: SlruCore<&str, i32> = SlruCore::new(10, 0.25);

        cache.insert("key", 1);
        cache.insert("key", 2);

        assert_eq!(cache.get(&"key"), Some(&2));
    }

    #[test]
    fn default_creates_zero_capacity() {
        let cache: SlruCore<String, i32> = SlruCore::default();
        assert_eq!(cache.capacity(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn clone_preserves_entries() {
        let mut cache = SlruCore::new(100, 0.25);
        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.get(&"a");

        let cloned = cache.clone();
        assert_eq!(cloned.len(), 2);
        assert_eq!(cloned.peek(&"a"), Some(&1));
        assert_eq!(cloned.peek(&"b"), Some(&2));
        assert_eq!(cloned.capacity(), 100);
    }

    #[test]
    fn clone_is_independent() {
        let mut cache = SlruCore::new(100, 0.25);
        cache.insert("a", 1);

        let mut cloned = cache.clone();
        cloned.insert("a", 999);
        cloned.insert("b", 2);

        assert_eq!(cache.peek(&"a"), Some(&1));
        assert!(!cache.contains(&"b"));
    }

    #[test]
    fn from_iterator() {
        let data = vec![("a", 1), ("b", 2), ("c", 3)];
        let cache: SlruCore<&str, i32> = data.into_iter().collect();

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.peek(&"a"), Some(&1));
        assert_eq!(cache.peek(&"b"), Some(&2));
        assert_eq!(cache.peek(&"c"), Some(&3));
    }

    #[test]
    fn into_iterator_owned() {
        let mut cache = SlruCore::new(100, 0.25);
        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3);

        let mut items: Vec<_> = cache.into_iter().collect();
        items.sort_by_key(|(k, _)| *k);
        assert_eq!(items, vec![("a", 1), ("b", 2), ("c", 3)]);
    }

    #[test]
    fn into_iterator_ref() {
        let mut cache = SlruCore::new(100, 0.25);
        cache.insert("a", 1);
        cache.insert("b", 2);

        let mut items: Vec<_> = (&cache).into_iter().collect();
        items.sort_by_key(|(k, _)| **k);
        assert_eq!(items, vec![(&"a", &1), (&"b", &2)]);
    }

    #[test]
    fn iter_exact_size() {
        let mut cache = SlruCore::new(100, 0.25);
        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.get(&"a");

        let iter = cache.iter();
        assert_eq!(iter.len(), 2);
    }

    #[test]
    fn extend_adds_entries() {
        let mut cache = SlruCore::new(100, 0.25);
        cache.insert("a", 1);

        cache.extend(vec![("b", 2), ("c", 3)]);

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.peek(&"b"), Some(&2));
    }

    #[test]
    fn debug_impl_works() {
        let mut cache = SlruCore::new(100, 0.25);
        cache.insert("a", 1);
        let debug = format!("{:?}", cache);
        assert!(debug.contains("SlruCore"));
        assert!(debug.contains("capacity"));
    }
}
