//! Two-Queue (2Q) cache replacement policy.
//!
//! Implements the 2Q algorithm, which separates recently inserted items from
//! frequently accessed items using two queues. This provides scan resistance
//! by preventing one-time accesses from polluting the main cache.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                           TwoQCore<K, V> Layout                             │
//! │                                                                             │
//! │   ┌─────────────────────────────────────────────────────────────────────┐   │
//! │   │  index: HashMapStore<K, SlotId>    store: SlotArena<Entry<K,V>>    │   │
//! │   │                                                                     │   │
//! │   │  ┌──────────┬──────────┐          ┌────────┬──────────────────┐   │   │
//! │   │  │   Key    │  SlotId  │          │ SlotId │ Entry            │   │   │
//! │   │  ├──────────┼──────────┤          ├────────┼──────────────────┤   │   │
//! │   │  │  "page1" │   id_0   │──────────►│  id_0  │ key,val,Probation│   │   │
//! │   │  │  "page2" │   id_1   │──────────►│  id_1  │ key,val,Protected│   │   │
//! │   │  │  "page3" │   id_2   │──────────►│  id_2  │ key,val,Probation│   │   │
//! │   │  └──────────┴──────────┘          └────────┴──────────────────┘   │   │
//! │   └─────────────────────────────────────────────────────────────────────┘   │
//! │                                                                             │
//! │   ┌─────────────────────────────────────────────────────────────────────┐   │
//! │   │                        Queue Organization                           │   │
//! │   │                                                                     │   │
//! │   │   PROBATION QUEUE (A1in - FIFO)          PROTECTED QUEUE (Am - LRU) │   │
//! │   │   ┌─────────────────────────┐            ┌─────────────────────────┐│   │
//! │   │   │ front               back│            │ MRU               LRU  ││   │
//! │   │   │  ▼                    ▼ │            │  ▼                  ▼  ││   │
//! │   │   │ [id_2] ◄──► [id_0] ◄──┤ │            │ [id_1] ◄──► [...] ◄──┤ ││   │
//! │   │   │  new        older  evict│            │ hot          cold  evict││   │
//! │   │   └─────────────────────────┘            └─────────────────────────┘│   │
//! │   │                                                                     │   │
//! │   │   • New items enter probation FIFO                                  │   │
//! │   │   • Re-access promotes to protected LRU                             │   │
//! │   │   • Eviction: probation first (if over cap), then protected LRU     │   │
//! │   └─────────────────────────────────────────────────────────────────────┘   │
//! │                                                                             │
//! └─────────────────────────────────────────────────────────────────────────────┘
//!
//! Insert Flow (new key)
//! ──────────────────────
//!
//!   insert("new_key", value):
//!     1. Check index - not found
//!     2. Create Entry with QueueKind::Probation
//!     3. Insert into store → get SlotId
//!     4. Insert key→SlotId into index
//!     5. Push SlotId to back of probation queue
//!     6. Evict if over capacity
//!
//! Access Flow (existing key)
//! ──────────────────────────
//!
//!   get("existing_key"):
//!     1. Lookup SlotId in index
//!     2. Check entry's queue kind:
//!        - If Probation: promote to Protected (move to protected MRU)
//!        - If Protected: move to MRU position
//!     3. Return &value
//!
//! Eviction Flow
//! ─────────────
//!
//!   evict_if_needed():
//!     while len > protected_cap:
//!       if probation.len > probation_cap:
//!         evict from probation front (oldest)
//!       else:
//!         evict from protected back (LRU)
//! ```
//!
//! ## Key Components
//!
//! - [`TwoQCore`]: Main 2Q cache implementation
//! - [`TwoQWithGhost`]: 2Q with ghost list for tracking evicted keys
//! - [`LruQueue`]: LRU queue wrapper around [`IntrusiveList`]
//!
//! ## Operations
//!
//! | Operation   | Time   | Notes                                      |
//! |-------------|--------|--------------------------------------------|
//! | `get`       | O(1)   | May promote from probation to protected    |
//! | `insert`    | O(1)*  | *Amortized, may trigger evictions          |
//! | `contains`  | O(1)   | Index lookup only                          |
//! | `len`       | O(1)   | Returns total entries                      |
//! | `clear`     | O(n)   | Clears all structures                      |
//!
//! ## Algorithm Properties
//!
//! - **Scan Resistance**: One-time accesses stay in probation, don't pollute protected
//! - **Frequency Awareness**: Repeated access promotes to protected LRU
//! - **Tunable**: `a1_frac` controls probation/protected size ratio
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
//! use cachekit::policy::two_q::TwoQCore;
//!
//! // Create 2Q cache: 100 total capacity, 25% for probation
//! let mut cache = TwoQCore::new(100, 0.25);
//!
//! // Insert items (go to probation)
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
//! ## Thread Safety
//!
//! - [`TwoQCore`]: Not thread-safe, designed for single-threaded use
//! - For concurrent access, wrap in external synchronization
//!
//! ## Implementation Notes
//!
//! - Probation uses FIFO ordering (push back, evict front)
//! - Protected uses LRU ordering (MRU at front, evict from back)
//! - Promotion from probation to protected happens on re-access
//! - Default `a1_frac` of 0.25 means 25% of capacity for probation
//!
//! ## References
//!
//! - Johnson & Shasha, "2Q: A Low Overhead High Performance Buffer Management
//!   Replacement Algorithm", VLDB 1994

use crate::ds::{IntrusiveList, SlotId};
use rustc_hash::FxHashMap;
use std::collections::VecDeque;
use std::hash::Hash;
use std::ptr::NonNull;

/// Indicates which queue an entry resides in.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum QueueKind {
    /// Entry is in the probation (A1in) FIFO queue.
    Probation,
    /// Entry is in the protected (Am) LRU queue.
    Protected,
}

/// Node in the optimized 2Q linked list.
///
/// Cache-line optimized layout with pointers first.
#[repr(C)]
struct Node<K, V> {
    prev: Option<NonNull<Node<K, V>>>,
    next: Option<NonNull<Node<K, V>>>,
    queue: QueueKind,
    key: K,
    value: V,
}

/// LRU queue backed by an intrusive doubly-linked list.
///
/// Provides O(1) insert, touch (move to front), and evict (pop back) operations.
/// Used for the protected queue in 2Q where access frequency matters.
///
/// # Type Parameters
///
/// - `T`: Element type stored in the queue
///
/// # Example
///
/// ```
/// use cachekit::policy::two_q::LruQueue;
///
/// let mut lru = LruQueue::new();
///
/// // Insert items (most recent at front)
/// let id1 = lru.insert("page1");
/// let id2 = lru.insert("page2");
///
/// assert_eq!(lru.len(), 2);
///
/// // Touch moves to MRU position
/// lru.touch(id1);
///
/// // Evict LRU (page2, since page1 was touched)
/// let evicted = lru.evict();
/// assert_eq!(evicted, Some("page2"));
/// ```
#[derive(Debug)]
pub struct LruQueue<T> {
    list: IntrusiveList<T>,
}

/// Two-Queue cache with ghost list for tracking evicted keys.
///
/// Extends [`TwoQCore`] with a ghost list that remembers recently evicted keys.
/// This allows detecting when a previously evicted key is re-accessed, which
/// can be used for adaptive tuning or admission decisions.
///
/// # Type Parameters
///
/// - `K`: Key type, must be `Clone + Eq + Hash`
/// - `V`: Value type
#[allow(dead_code)]
pub struct TwoQWithGhost<K, V>
where
    K: Clone + Eq + Hash,
{
    core: TwoQCore<K, V>,
    ghost_list: VecDeque<K>,
    ghost_list_cap: usize,
}

impl<K, V> std::fmt::Debug for TwoQWithGhost<K, V>
where
    K: Clone + Eq + Hash + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TwoQWithGhost")
            .field("core", &self.core)
            .field("ghost_list_len", &self.ghost_list.len())
            .field("ghost_list_cap", &self.ghost_list_cap)
            .finish()
    }
}

/// Core Two-Queue (2Q) cache implementation.
///
/// Implements the 2Q replacement algorithm with two queues:
/// - **Probation (A1in)**: FIFO queue for newly inserted items
/// - **Protected (Am)**: LRU queue for frequently accessed items
///
/// New items enter probation. Re-accessing an item in probation promotes it
/// to protected. This provides scan resistance by keeping one-time accesses
/// from polluting the main cache.
///
/// # Type Parameters
///
/// - `K`: Key type, must be `Clone + Eq + Hash`
/// - `V`: Value type
///
/// # Example
///
/// ```
/// use cachekit::policy::two_q::TwoQCore;
///
/// // 100 capacity, 25% probation
/// let mut cache = TwoQCore::new(100, 0.25);
///
/// // Insert goes to probation
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
/// 1. If probation exceeds its cap, evict from probation front (oldest)
/// 2. Otherwise, evict from protected back (LRU)
///
/// # Implementation
///
/// Uses raw pointer linked lists for O(1) operations with minimal overhead.
pub struct TwoQCore<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Direct key -> node pointer mapping
    map: FxHashMap<K, NonNull<Node<K, V>>>,

    /// Probation queue (FIFO): head=newest, tail=oldest
    probation_head: Option<NonNull<Node<K, V>>>,
    probation_tail: Option<NonNull<Node<K, V>>>,
    probation_len: usize,

    /// Protected queue (LRU): head=MRU, tail=LRU
    protected_head: Option<NonNull<Node<K, V>>>,
    protected_tail: Option<NonNull<Node<K, V>>>,
    protected_len: usize,

    /// Maximum size of the probation queue.
    probation_cap: usize,
    /// Maximum total cache capacity.
    protected_cap: usize,
}

// SAFETY: TwoQCore can be sent between threads if K and V are Send.
unsafe impl<K, V> Send for TwoQCore<K, V>
where
    K: Clone + Eq + Hash + Send,
    V: Send,
{
}

// SAFETY: TwoQCore can be shared between threads if K and V are Sync.
unsafe impl<K, V> Sync for TwoQCore<K, V>
where
    K: Clone + Eq + Hash + Sync,
    V: Sync,
{
}

impl<T> Default for LruQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> LruQueue<T> {
    /// Creates an empty LRU queue.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::two_q::LruQueue;
    ///
    /// let lru: LruQueue<i32> = LruQueue::new();
    /// assert!(lru.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            list: IntrusiveList::new(),
        }
    }

    /// Creates an LRU queue with pre-allocated capacity.
    ///
    /// Pre-allocates space to avoid reallocation during growth.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::two_q::LruQueue;
    ///
    /// let lru: LruQueue<i32> = LruQueue::with_capacity(1000);
    /// assert!(lru.is_empty());
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            list: IntrusiveList::with_capacity(capacity),
        }
    }

    /// Returns `true` if the queue is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::two_q::LruQueue;
    ///
    /// let mut lru = LruQueue::new();
    /// assert!(lru.is_empty());
    ///
    /// lru.insert(42);
    /// assert!(!lru.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.list.len() == 0
    }

    /// Inserts an item at the MRU position (front).
    ///
    /// Returns the `SlotId` that can be used for future `touch` or `remove` calls.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::two_q::LruQueue;
    ///
    /// let mut lru = LruQueue::new();
    /// let id = lru.insert("page");
    /// assert_eq!(lru.len(), 1);
    /// ```
    pub fn insert(&mut self, id: T) -> SlotId {
        // new item is most-recently-used
        self.list.push_front(id)
    }

    /// Moves an item to the MRU position (front).
    ///
    /// Returns `true` if the item was found and moved, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::two_q::LruQueue;
    ///
    /// let mut lru = LruQueue::new();
    /// let id1 = lru.insert("old");
    /// let id2 = lru.insert("new");
    ///
    /// // Touch makes "old" the MRU
    /// assert!(lru.touch(id1));
    ///
    /// // Now "new" is LRU and will be evicted first
    /// assert_eq!(lru.evict(), Some("new"));
    /// ```
    pub fn touch(&mut self, id: SlotId) -> bool {
        // move accessed item to MRU position
        self.list.move_to_front(id)
    }

    /// Removes and returns the LRU item (from back).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::two_q::LruQueue;
    ///
    /// let mut lru = LruQueue::new();
    /// lru.insert("first");
    /// lru.insert("second");
    ///
    /// // "first" is LRU (inserted first, never touched)
    /// assert_eq!(lru.evict(), Some("first"));
    /// assert_eq!(lru.evict(), Some("second"));
    /// assert_eq!(lru.evict(), None);
    /// ```
    pub fn evict(&mut self) -> Option<T> {
        // remove least-recently-used
        self.list.pop_back()
    }

    /// Removes an item by its `SlotId`.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::two_q::LruQueue;
    ///
    /// let mut lru = LruQueue::new();
    /// let id = lru.insert("page");
    ///
    /// assert_eq!(lru.remove(id), Some("page"));
    /// assert!(lru.is_empty());
    /// ```
    pub fn remove(&mut self, id: SlotId) -> Option<T> {
        self.list.remove(id)
    }

    /// Returns the number of items in the queue.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::two_q::LruQueue;
    ///
    /// let mut lru = LruQueue::new();
    /// assert_eq!(lru.len(), 0);
    ///
    /// lru.insert(1);
    /// lru.insert(2);
    /// assert_eq!(lru.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.list.len()
    }

    /// Pushes an item to the front (MRU position).
    ///
    /// Alias for [`insert`](Self::insert).
    pub fn push_front(&mut self, id: T) -> SlotId {
        self.list.push_front(id)
    }

    /// Moves an item to the front (MRU position).
    ///
    /// Alias for [`touch`](Self::touch).
    pub fn move_to_front(&mut self, id: SlotId) -> bool {
        self.list.move_to_front(id)
    }

    /// Removes and returns the item from the back (LRU position).
    ///
    /// Alias for [`evict`](Self::evict).
    pub fn pop_back(&mut self) -> Option<T> {
        self.list.pop_back()
    }
}

impl<K, V> TwoQCore<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Creates a new 2Q cache with the specified capacity and probation fraction.
    ///
    /// # Arguments
    ///
    /// - `protected_cap`: Total cache capacity (maximum number of entries)
    /// - `a1_frac`: Fraction of capacity allocated to probation queue (0.0 to 1.0)
    ///
    /// A typical value for `a1_frac` is 0.25 (25% for probation).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::two_q::TwoQCore;
    ///
    /// // 100 capacity, 25% probation (25 items max in probation)
    /// let cache: TwoQCore<String, i32> = TwoQCore::new(100, 0.25);
    /// assert_eq!(cache.capacity(), 100);
    /// assert!(cache.is_empty());
    /// ```
    #[inline]
    pub fn new(protected_cap: usize, a1_frac: f64) -> Self {
        let probation_cap = (protected_cap as f64 * a1_frac) as usize;
        let total_cap = protected_cap + probation_cap;

        Self {
            map: FxHashMap::with_capacity_and_hasher(total_cap, Default::default()),
            probation_head: None,
            probation_tail: None,
            probation_len: 0,
            protected_head: None,
            protected_tail: None,
            protected_len: 0,
            probation_cap,
            protected_cap,
        }
    }

    /// Detach a node from its current queue.
    #[inline(always)]
    fn detach(&mut self, node_ptr: NonNull<Node<K, V>>) {
        unsafe {
            let node = node_ptr.as_ref();
            let prev = node.prev;
            let next = node.next;
            let queue = node.queue;

            let (head, tail, len) = match queue {
                QueueKind::Probation => (
                    &mut self.probation_head,
                    &mut self.probation_tail,
                    &mut self.probation_len,
                ),
                QueueKind::Protected => (
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

    /// Attach a node at the head of probation queue (FIFO: new items at head).
    #[inline(always)]
    fn attach_probation_head(&mut self, mut node_ptr: NonNull<Node<K, V>>) {
        unsafe {
            let node = node_ptr.as_mut();
            node.prev = None;
            node.next = self.probation_head;
            node.queue = QueueKind::Probation;

            match self.probation_head {
                Some(mut h) => h.as_mut().prev = Some(node_ptr),
                None => self.probation_tail = Some(node_ptr),
            }

            self.probation_head = Some(node_ptr);
            self.probation_len += 1;
        }
    }

    /// Attach a node at the head of protected queue (LRU: MRU at head).
    #[inline(always)]
    fn attach_protected_head(&mut self, mut node_ptr: NonNull<Node<K, V>>) {
        unsafe {
            let node = node_ptr.as_mut();
            node.prev = None;
            node.next = self.protected_head;
            node.queue = QueueKind::Protected;

            match self.protected_head {
                Some(mut h) => h.as_mut().prev = Some(node_ptr),
                None => self.protected_tail = Some(node_ptr),
            }

            self.protected_head = Some(node_ptr);
            self.protected_len += 1;
        }
    }

    /// Pop from probation tail (FIFO: oldest at tail).
    #[inline(always)]
    fn pop_probation_tail(&mut self) -> Option<Box<Node<K, V>>> {
        self.probation_tail.map(|tail_ptr| unsafe {
            let node = Box::from_raw(tail_ptr.as_ptr());

            self.probation_tail = node.prev;
            match self.probation_tail {
                Some(mut t) => t.as_mut().next = None,
                None => self.probation_head = None,
            }
            self.probation_len -= 1;

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

    /// Retrieves a value by key, promoting from probation to protected if needed.
    ///
    /// If the key is in probation, accessing it promotes the entry to the
    /// protected queue (demonstrating it's not a one-time access).
    /// If already in protected, moves it to the MRU position.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::two_q::TwoQCore;
    ///
    /// let mut cache = TwoQCore::new(100, 0.25);
    /// cache.insert("key", 42);
    ///
    /// // First access: in probation, now promotes to protected
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
        let node_ptr = *self.map.get(key)?;

        let queue = unsafe { node_ptr.as_ref().queue };

        match queue {
            QueueKind::Probation => {
                // Promote from probation to protected
                self.detach(node_ptr);
                self.attach_protected_head(node_ptr);
            },
            QueueKind::Protected => {
                // Move to MRU position
                self.detach(node_ptr);
                self.attach_protected_head(node_ptr);
            },
        }

        unsafe { Some(&node_ptr.as_ref().value) }
    }

    /// Inserts or updates a key-value pair.
    ///
    /// - If the key exists, updates the value in place (no queue change)
    /// - If the key is new, inserts into the probation queue
    /// - May trigger eviction if capacity is exceeded
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::two_q::TwoQCore;
    ///
    /// let mut cache = TwoQCore::new(100, 0.25);
    ///
    /// // New insert goes to probation
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

        // Create new node in probation
        let node = Box::new(Node {
            prev: None,
            next: None,
            queue: QueueKind::Probation,
            key: key.clone(),
            value,
        });
        let node_ptr = NonNull::new(Box::into_raw(node)).unwrap();

        self.map.insert(key, node_ptr);
        self.attach_probation_head(node_ptr);
    }

    /// Evicts entries until there is room for a new entry.
    #[inline]
    fn evict_if_needed(&mut self) {
        while self.len() >= self.protected_cap {
            if self.probation_len > self.probation_cap {
                // Evict from probation tail (oldest)
                if let Some(node) = self.pop_probation_tail() {
                    self.map.remove(&node.key);
                    continue;
                }
            }
            // Evict from protected tail (LRU)
            if let Some(node) = self.pop_protected_tail() {
                self.map.remove(&node.key);
                continue;
            }
            // Fallback: evict from probation even if under cap
            if let Some(node) = self.pop_probation_tail() {
                self.map.remove(&node.key);
                continue;
            }
            // No entries to evict
            break;
        }
    }

    /// Returns the number of entries in the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::two_q::TwoQCore;
    ///
    /// let mut cache = TwoQCore::new(100, 0.25);
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
    /// use cachekit::policy::two_q::TwoQCore;
    ///
    /// let mut cache: TwoQCore<&str, i32> = TwoQCore::new(100, 0.25);
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
    /// use cachekit::policy::two_q::TwoQCore;
    ///
    /// let cache: TwoQCore<String, i32> = TwoQCore::new(500, 0.25);
    /// assert_eq!(cache.capacity(), 500);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        self.protected_cap
    }

    /// Returns `true` if the key exists in the cache.
    ///
    /// Does not affect queue positions (no promotion on contains).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::two_q::TwoQCore;
    ///
    /// let mut cache = TwoQCore::new(100, 0.25);
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
    /// use cachekit::policy::two_q::TwoQCore;
    ///
    /// let mut cache = TwoQCore::new(100, 0.25);
    /// cache.insert("a", 1);
    /// cache.insert("b", 2);
    ///
    /// cache.clear();
    /// assert!(cache.is_empty());
    /// assert!(!cache.contains(&"a"));
    /// ```
    pub fn clear(&mut self) {
        // Free all nodes
        while self.pop_probation_tail().is_some() {}
        while self.pop_protected_tail().is_some() {}
        self.map.clear();
    }
}

// Proper cleanup when cache is dropped
impl<K, V> Drop for TwoQCore<K, V>
where
    K: Clone + Eq + Hash,
{
    fn drop(&mut self) {
        while self.pop_probation_tail().is_some() {}
        while self.pop_protected_tail().is_some() {}
    }
}

// Debug implementation
impl<K, V> std::fmt::Debug for TwoQCore<K, V>
where
    K: Clone + Eq + Hash + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TwoQCore")
            .field("capacity", &self.protected_cap)
            .field("probation_cap", &self.probation_cap)
            .field("len", &self.map.len())
            .field("probation_len", &self.probation_len)
            .field("protected_len", &self.protected_len)
            .finish_non_exhaustive()
    }
}

/// Implementation of the [`CoreCache`](crate::traits::CoreCache) trait for 2Q.
///
/// Allows `TwoQCore` to be used through the unified cache interface.
///
/// # Example
///
/// ```
/// use cachekit::traits::CoreCache;
/// use cachekit::policy::two_q::TwoQCore;
///
/// let mut cache: TwoQCore<&str, i32> = TwoQCore::new(100, 0.25);
///
/// // Use via CoreCache trait
/// assert_eq!(CoreCache::insert(&mut cache, "key", 42), None);
/// assert_eq!(CoreCache::get(&mut cache, &"key"), Some(&42));
/// assert!(CoreCache::contains(&cache, &"key"));
/// ```
impl<K, V> crate::traits::CoreCache<K, V> for TwoQCore<K, V>
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
        TwoQCore::insert(self, key, value);
        None
    }

    #[inline]
    fn get(&mut self, key: &K) -> Option<&V> {
        TwoQCore::get(self, key)
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
        self.protected_cap
    }

    fn clear(&mut self) {
        TwoQCore::clear(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::CoreCache;

    // ==============================================
    // LruQueue Tests
    // ==============================================

    mod lru_queue_tests {
        use super::*;

        #[test]
        fn new_queue_is_empty() {
            let lru: LruQueue<i32> = LruQueue::new();
            assert!(lru.is_empty());
            assert_eq!(lru.len(), 0);
        }

        #[test]
        fn default_creates_empty_queue() {
            let lru: LruQueue<&str> = LruQueue::default();
            assert!(lru.is_empty());
        }

        #[test]
        fn insert_increases_length() {
            let mut lru = LruQueue::new();
            lru.insert(1);
            assert_eq!(lru.len(), 1);
            assert!(!lru.is_empty());

            lru.insert(2);
            lru.insert(3);
            assert_eq!(lru.len(), 3);
        }

        #[test]
        fn evict_returns_lru_item() {
            let mut lru = LruQueue::new();
            lru.insert("first");
            lru.insert("second");
            lru.insert("third");

            // "first" is the LRU (inserted first, never touched)
            assert_eq!(lru.evict(), Some("first"));
            assert_eq!(lru.evict(), Some("second"));
            assert_eq!(lru.evict(), Some("third"));
            assert_eq!(lru.evict(), None);
        }

        #[test]
        fn touch_moves_to_mru() {
            let mut lru = LruQueue::new();
            let first = lru.insert("first");
            lru.insert("second");
            lru.insert("third");

            // Touch "first" - moves it to MRU
            assert!(lru.touch(first));

            // Now "second" is LRU
            assert_eq!(lru.evict(), Some("second"));
            assert_eq!(lru.evict(), Some("third"));
            assert_eq!(lru.evict(), Some("first")); // "first" is now MRU, evicted last
        }

        #[test]
        fn remove_returns_item() {
            let mut lru = LruQueue::new();
            let id = lru.insert("item");
            assert_eq!(lru.len(), 1);

            assert_eq!(lru.remove(id), Some("item"));
            assert!(lru.is_empty());
        }

        #[test]
        fn remove_from_middle() {
            let mut lru = LruQueue::new();
            lru.insert("first");
            let middle = lru.insert("middle");
            lru.insert("last");

            assert_eq!(lru.remove(middle), Some("middle"));
            assert_eq!(lru.len(), 2);

            // Remaining items evict in LRU order
            assert_eq!(lru.evict(), Some("first"));
            assert_eq!(lru.evict(), Some("last"));
        }

        #[test]
        fn evict_from_empty_returns_none() {
            let mut lru: LruQueue<i32> = LruQueue::new();
            assert_eq!(lru.evict(), None);
        }

        #[test]
        fn push_front_alias_works() {
            let mut lru = LruQueue::new();
            lru.push_front("a");
            lru.push_front("b");
            assert_eq!(lru.len(), 2);
            // "a" is LRU since "b" was pushed front after
            assert_eq!(lru.pop_back(), Some("a"));
        }

        #[test]
        fn move_to_front_alias_works() {
            let mut lru = LruQueue::new();
            let a = lru.insert("a");
            lru.insert("b");

            assert!(lru.move_to_front(a));
            // Now "b" is LRU
            assert_eq!(lru.pop_back(), Some("b"));
        }
    }

    // ==============================================
    // TwoQCore Basic Operations
    // ==============================================

    mod basic_operations {
        use super::*;

        #[test]
        fn new_cache_is_empty() {
            let cache: TwoQCore<&str, i32> = TwoQCore::new(100, 0.25);
            assert!(cache.is_empty());
            assert_eq!(cache.len(), 0);
            assert_eq!(cache.capacity(), 100);
        }

        #[test]
        fn insert_and_get() {
            let mut cache = TwoQCore::new(100, 0.25);
            cache.insert("key1", "value1");

            assert_eq!(cache.len(), 1);
            assert_eq!(cache.get(&"key1"), Some(&"value1"));
        }

        #[test]
        fn insert_multiple_items() {
            let mut cache = TwoQCore::new(100, 0.25);
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
            let mut cache: TwoQCore<&str, i32> = TwoQCore::new(100, 0.25);
            cache.insert("exists", 42);

            assert_eq!(cache.get(&"missing"), None);
        }

        #[test]
        fn update_existing_key() {
            let mut cache = TwoQCore::new(100, 0.25);
            cache.insert("key", "initial");
            cache.insert("key", "updated");

            assert_eq!(cache.len(), 1);
            assert_eq!(cache.get(&"key"), Some(&"updated"));
        }

        #[test]
        fn contains_returns_correct_result() {
            let mut cache = TwoQCore::new(100, 0.25);
            cache.insert("exists", 1);

            assert!(cache.contains(&"exists"));
            assert!(!cache.contains(&"missing"));
        }

        #[test]
        fn contains_does_not_promote() {
            let mut cache: TwoQCore<String, i32> = TwoQCore::new(10, 0.3);
            cache.insert("a".to_string(), 1);
            cache.insert("b".to_string(), 2);
            cache.insert("c".to_string(), 3);

            // Contains check should not promote
            assert!(cache.contains(&"a".to_string()));
            assert!(cache.contains(&"b".to_string()));
            assert!(cache.contains(&"c".to_string()));

            // Fill up to trigger eviction
            for i in 0..10 {
                cache.insert(format!("new{}", i), i);
            }

            // Original items should be evicted (they were only in probation)
            assert!(!cache.contains(&"a".to_string()));
            assert!(!cache.contains(&"b".to_string()));
            assert!(!cache.contains(&"c".to_string()));
        }

        #[test]
        fn clear_removes_all_entries() {
            let mut cache = TwoQCore::new(100, 0.25);
            cache.insert("a", 1);
            cache.insert("b", 2);
            cache.get(&"a"); // Promote "a" to protected

            cache.clear();

            assert!(cache.is_empty());
            assert_eq!(cache.len(), 0);
            assert!(!cache.contains(&"a"));
            assert!(!cache.contains(&"b"));
        }

        #[test]
        fn capacity_returns_correct_value() {
            let cache: TwoQCore<i32, i32> = TwoQCore::new(500, 0.25);
            assert_eq!(cache.capacity(), 500);
        }
    }

    // ==============================================
    // Queue Behavior (Probation vs Protected)
    // ==============================================

    mod queue_behavior {
        use super::*;

        #[test]
        fn new_insert_goes_to_probation() {
            let mut cache = TwoQCore::new(10, 0.3);
            cache.insert("key", "value");

            assert!(cache.contains(&"key"));
            assert_eq!(cache.len(), 1);
        }

        #[test]
        fn get_promotes_from_probation_to_protected() {
            let mut cache: TwoQCore<String, i32> = TwoQCore::new(10, 0.3);
            cache.insert("key".to_string(), 0);

            // First get promotes to protected
            let _ = cache.get(&"key".to_string());

            // Insert enough items to fill probation and exceed capacity
            for i in 0..12 {
                cache.insert(format!("new{}", i), i);
            }

            // "key" should still exist because it was promoted to protected
            assert!(cache.contains(&"key".to_string()));
        }

        #[test]
        fn item_in_protected_stays_in_protected() {
            let mut cache = TwoQCore::new(10, 0.3);
            cache.insert("key", "value");

            // Promote to protected
            cache.get(&"key");

            // Access again - should stay in protected, move to MRU
            cache.get(&"key");
            cache.get(&"key");

            assert_eq!(cache.get(&"key"), Some(&"value"));
        }

        #[test]
        fn multiple_accesses_keep_item_alive() {
            let mut cache: TwoQCore<String, i32> = TwoQCore::new(10, 0.3);

            cache.insert("hot".to_string(), 0);
            cache.get(&"hot".to_string());

            for i in 0..15 {
                cache.insert(format!("cold{}", i), i);
                cache.get(&"hot".to_string());
            }

            assert!(cache.contains(&"hot".to_string()));
        }
    }

    // ==============================================
    // Eviction Behavior
    // ==============================================

    mod eviction_behavior {
        use super::*;

        #[test]
        fn eviction_occurs_when_over_capacity() {
            let mut cache = TwoQCore::new(5, 0.2);

            for i in 0..10 {
                cache.insert(i, i * 10);
            }

            assert_eq!(cache.len(), 5);
        }

        #[test]
        fn probation_evicts_fifo_order() {
            let mut cache = TwoQCore::new(5, 0.4);

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
        fn protected_evicts_lru_when_probation_under_cap() {
            let mut cache = TwoQCore::new(5, 0.4);

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
            let mut cache: TwoQCore<String, i32> = TwoQCore::new(10, 0.3);

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
            let mut cache = TwoQCore::new(3, 0.33);

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
            let mut cache = TwoQCore::new(100, 0.25);

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
        fn one_time_access_stays_in_probation() {
            let mut cache: TwoQCore<String, i32> = TwoQCore::new(10, 0.3);

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
            let mut cache = TwoQCore::new(20, 0.25);

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
    // CoreCache Trait Implementation
    // ==============================================

    mod core_cache_trait {
        use super::*;

        #[test]
        fn trait_insert_returns_old_value() {
            let mut cache: TwoQCore<&str, i32> = TwoQCore::new(100, 0.25);

            let old = CoreCache::insert(&mut cache, "key", 1);
            assert_eq!(old, None);

            let old = CoreCache::insert(&mut cache, "key", 2);
            assert_eq!(old, Some(1));

            assert_eq!(CoreCache::get(&mut cache, &"key"), Some(&2));
        }

        #[test]
        fn trait_get_works() {
            let mut cache = TwoQCore::new(100, 0.25);
            CoreCache::insert(&mut cache, "key", 42);

            assert_eq!(CoreCache::get(&mut cache, &"key"), Some(&42));
            assert_eq!(CoreCache::get(&mut cache, &"missing"), None);
        }

        #[test]
        fn trait_contains_works() {
            let mut cache = TwoQCore::new(100, 0.25);
            CoreCache::insert(&mut cache, "key", 1);

            assert!(CoreCache::contains(&cache, &"key"));
            assert!(!CoreCache::contains(&cache, &"missing"));
        }

        #[test]
        fn trait_len_and_capacity() {
            let mut cache: TwoQCore<i32, i32> = TwoQCore::new(50, 0.25);

            assert_eq!(CoreCache::len(&cache), 0);
            assert_eq!(CoreCache::capacity(&cache), 50);

            for i in 0..30 {
                CoreCache::insert(&mut cache, i, i * 10);
            }

            assert_eq!(CoreCache::len(&cache), 30);
        }

        #[test]
        fn trait_clear_works() {
            let mut cache = TwoQCore::new(100, 0.25);
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
            let mut cache = TwoQCore::new(1, 0.5);

            cache.insert("a", 1);
            assert_eq!(cache.get(&"a"), Some(&1));

            cache.insert("b", 2);
            assert!(!cache.contains(&"a"));
            assert_eq!(cache.get(&"b"), Some(&2));
        }

        #[test]
        fn zero_probation_fraction() {
            let mut cache = TwoQCore::new(10, 0.0);

            for i in 0..10 {
                cache.insert(i, i * 10);
            }

            assert_eq!(cache.len(), 10);

            cache.insert(100, 1000);
            assert_eq!(cache.len(), 10);
        }

        #[test]
        fn one_hundred_percent_probation() {
            let mut cache = TwoQCore::new(10, 1.0);

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
            let mut cache = TwoQCore::new(100, 0.25);

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
            let mut cache = TwoQCore::new(10000, 0.25);

            for i in 0..10000 {
                cache.insert(i, i * 2);
            }

            assert_eq!(cache.len(), 10000);

            assert_eq!(cache.get(&5000), Some(&10000));
            assert_eq!(cache.get(&9999), Some(&19998));
        }

        #[test]
        fn empty_cache_operations() {
            let mut cache: TwoQCore<i32, i32> = TwoQCore::new(100, 0.25);

            assert!(cache.is_empty());
            assert_eq!(cache.get(&1), None);
            assert!(!cache.contains(&1));

            cache.clear();
            assert!(cache.is_empty());
        }

        #[test]
        fn small_fractions() {
            let mut cache = TwoQCore::new(100, 0.01);

            for i in 0..10 {
                cache.insert(i, i);
            }

            assert_eq!(cache.len(), 10);
        }

        #[test]
        fn string_keys_and_values() {
            let mut cache = TwoQCore::new(100, 0.25);

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
            let mut cache = TwoQCore::new(100, 0.25);

            for i in 0..50 {
                cache.insert(i, format!("value_{}", i));
            }

            assert_eq!(cache.get(&25), Some(&String::from("value_25")));
            assert_eq!(cache.get(&49), Some(&String::from("value_49")));
        }
    }

    // ==============================================
    // Capacity and Eviction Boundary Tests
    // ==============================================

    mod boundary_tests {
        use super::*;

        #[test]
        fn exact_capacity_no_eviction() {
            let mut cache = TwoQCore::new(10, 0.3);

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
            let mut cache = TwoQCore::new(10, 0.3);

            for i in 0..10 {
                cache.insert(i, i);
            }

            cache.insert(10, 10);

            assert_eq!(cache.len(), 10);
            assert!(!cache.contains(&0));
            assert!(cache.contains(&10));
        }

        #[test]
        fn probation_cap_boundary() {
            let mut cache = TwoQCore::new(10, 0.3);

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
            let mut cache = TwoQCore::new(10, 0.3);

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
        fn promotion_actually_moves_to_protected_queue() {
            let mut cache: TwoQCore<String, i32> = TwoQCore::new(5, 0.4);

            cache.insert("key".to_string(), 0);
            cache.get(&"key".to_string());

            cache.insert("p1".to_string(), 1);
            cache.insert("p2".to_string(), 2);
            cache.insert("p3".to_string(), 3);
            cache.insert("p4".to_string(), 4);

            assert!(
                cache.contains(&"key".to_string()),
                "Promoted item should be in protected queue and survive probation eviction"
            );
        }

        #[test]
        fn update_preserves_queue_position() {
            let mut cache: TwoQCore<String, i32> = TwoQCore::new(10, 0.3);

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
                let mut cache = TwoQCore::new(5, 0.4);

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
            let mut cache = TwoQCore::new(100, 0.25);

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
            let mut cache: TwoQCore<String, String> = TwoQCore::new(50, 0.3);

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
            let mut cache = TwoQCore::new(100, 0.25);

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
}
