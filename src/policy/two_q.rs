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

use crate::ds::{IntrusiveList, SlotArena, SlotId};
use crate::store::hashmap::HashMapStore;
use crate::store::traits::{StoreCore, StoreMut};
use std::collections::VecDeque;
use std::hash::Hash;

/// Indicates which queue an entry resides in.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum QueueKind {
    /// Entry is in the probation (A1in) FIFO queue.
    Probation,
    /// Entry is in the protected (Am) LRU queue.
    Protected,
}

/// Internal entry storing key, value, and queue membership.
#[derive(Debug)]
struct Entry<K, V> {
    key: K,
    value: V,
    queue: QueueKind,
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
#[derive(Debug)]
pub struct TwoQWithGhost<K, V> {
    core: TwoQCore<K, V>,
    ghost_list: VecDeque<K>,
    ghost_list_cap: usize,
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
#[derive(Debug)]
pub struct TwoQCore<K, V> {
    /// Maps keys to their SlotId in the store.
    index: HashMapStore<K, SlotId>,
    /// Arena storing all entries.
    store: SlotArena<Entry<K, V>>,

    /// FIFO queue for newly inserted items.
    probation: IntrusiveList<SlotId>,
    /// LRU queue for frequently accessed items.
    protected: LruQueue<SlotId>,

    /// Maximum size of the probation queue.
    probation_cap: usize,
    /// Maximum total cache capacity.
    protected_cap: usize,
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
    pub fn new(protected_cap: usize, a1_frac: f64) -> Self {
        let probation_cap = (protected_cap as f64 * a1_frac) as usize;

        Self {
            index: HashMapStore::new(protected_cap),
            store: SlotArena::new(),
            probation: IntrusiveList::new(),
            protected: LruQueue::new(),
            probation_cap,
            protected_cap,
        }
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
    pub fn get(&mut self, key: &K) -> Option<&V> {
        let id = self.index.get(key)?;

        let queue = self.store.get(*id)?.queue;
        match queue {
            QueueKind::Probation => {
                self.probation.remove(*id);
                self.probation.push_front(*id);
                if let Some(e) = self.store.get_mut(*id) {
                    e.queue = QueueKind::Protected;
                }
            },
            QueueKind::Protected => {
                self.protected.move_to_front(*id);
            },
        }

        self.store.get(*id).map(|e| &e.value)
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
    pub fn insert(&mut self, key: K, value: V) {
        if let Some(id) = self.index.get(&key) {
            if let Some(e) = self.store.get_mut(*id) {
                e.value = value;
            }
            return;
        }

        let entry = Entry {
            key: key.clone(),
            value,
            queue: QueueKind::Probation,
        };
        let id = self.store.insert(entry);

        self.index
            .try_insert(key, id)
            .expect("Failed to insert entry");
        self.probation.push_back(id);

        self.evict_if_needed();
    }

    /// Evicts entries until within capacity.
    fn evict_if_needed(&mut self) {
        while self.len() > self.protected_cap {
            if self.probation.len() > self.probation_cap {
                if let Some(id) = self.probation.pop_front() {
                    self.remove_id(id);
                }
            } else if let Some(id) = self.protected.pop_back() {
                self.remove_id(id);
            }
        }
    }

    /// Removes an entry by its SlotId.
    fn remove_id(&mut self, id: SlotId) {
        if let Some(entry) = self.store.remove(id) {
            self.index.remove(&entry.key);
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
    pub fn len(&self) -> usize {
        self.index.len()
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
    pub fn is_empty(&self) -> bool {
        self.index.len() == 0
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
    pub fn contains(&self, key: &K) -> bool {
        self.index.contains(key)
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
        self.index.clear();
        self.store.clear();
        self.probation.clear();
        self.protected.list.clear();
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
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        // Check if key exists - update in place
        if let Some(id) = self.index.get(&key) {
            if let Some(e) = self.store.get_mut(*id) {
                let old = std::mem::replace(&mut e.value, value);
                return Some(old);
            }
        }

        // New insert
        TwoQCore::insert(self, key, value);
        None
    }

    fn get(&mut self, key: &K) -> Option<&V> {
        TwoQCore::get(self, key)
    }

    fn contains(&self, key: &K) -> bool {
        self.index.contains(key)
    }

    fn len(&self) -> usize {
        self.index.len()
    }

    fn capacity(&self) -> usize {
        self.protected_cap
    }

    fn clear(&mut self) {
        self.index.clear();
        self.store.clear();
        self.probation.clear();
        self.protected.list.clear();
    }
}
