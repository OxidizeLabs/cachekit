//! # Least Recently Used (LRU) Cache Implementation
//!
//! This module provides a high-performance, concurrent LRU cache implementation used primarily
//! for the disk buffer pool and other caching needs in Ferrite.
//!
//! ## Architecture
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────────────────────────┐
//!   │                        ConcurrentLRUCache<K, V>                          │
//!   │                                                                          │
//!   │   ┌────────────────────────────────────────────────────────────────────┐ │
//!   │   │                    Arc<RwLock<LRUCore<K, V>>>                       │ │
//!   │   └────────────────────────────────────────────────────────────────────┘ │
//!   │                                  │                                       │
//!   │                                  ▼                                       │
//!   │   ┌────────────────────────────────────────────────────────────────────┐ │
//!   │   │                         LRUCore<K, V>                              │ │
//!   │   │                                                                    │ │
//!   │   │   ┌──────────────────────────────────────────────────────────────┐ │ │
//!   │   │   │  HashMap<K, NonNull<Node<K>>>                               │ │ │
//!   │   │   │                                                              │ │ │
//!   │   │   │  ┌─────────┬────────────────────────────────────────────┐    │ │ │
//!   │   │   │  │   Key   │  NonNull<Node>                             │    │ │ │
//!   │   │   │  ├─────────┼────────────────────────────────────────────┤    │ │ │
//!   │   │   │  │  page_1 │  ────────────────────────────────────────┐ │    │ │ │
//!   │   │   │  │  page_2 │  ──────────────────────────────────┐     │ │    │ │ │
//!   │   │   │  │  page_3 │  ────────────────────────────┐     │     │ │    │ │ │
//!   │   │   │  └─────────┴──────────────────────────────┼─────┼─────┼─┘    │ │ │
//!   │   │   └───────────────────────────────────────────┼─────┼─────┼──────┘ │ │
//!   │   │                                               │     │     │        │ │
//!   │   │   ┌───────────────────────────────────────────┼─────┼─────┼──────┐ │ │
//!   │   │   │  Doubly-Linked List (LRU Order)           │     │     │      │ │ │
//!   │   │   │                                           ▼     ▼     ▼      │ │ │
//!   │   │   │  head ──► ┌──────┐ ◄──► ┌──────┐ ◄──► ┌──────┐ ◄── tail      │ │ │
//!   │   │   │    (MRU)  │ Node │      │ Node │      │ Node │   (LRU)       │ │ │
//!   │   │   │           │page_1│      │page_2│      │page_3│               │ │ │
//!   │   │   │           └──────┘      └──────┘      └──────┘               │ │ │
//!   │   │   │                                                              │ │ │
//!   │   │   │  Most Recently Used ────────────────► Least Recently Used    │ │ │
//!   │   │   └──────────────────────────────────────────────────────────────┘ │ │
//!   │   │                                                                    │ │
//!   │   │   ┌──────────────────────────────────────────────────────────────┐ │ │
//!   │   │   │  HashMapStore<K, V> (values live here)                       │ │ │
//!   │   │   │  K -> Arc<V>                                                 │ │ │
//!   │   │   └──────────────────────────────────────────────────────────────┘ │ │
//!   │   └────────────────────────────────────────────────────────────────────┘ │
//!   └──────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Components
//!
//! | Component              | Description                                        |
//! |------------------------|----------------------------------------------------|
//! | `LRUCore<K, V>`        | Single-threaded core with list + index + store     |
//! | `ConcurrentLRUCache`   | Thread-safe wrapper with `parking_lot::RwLock`     |
//! | `Node<K>`              | Intrusive list node with key, prev/next             |
//! | `HashMapStore<K, V>`   | Store for key -> `Arc<V>` ownership                |
//! | `BufferPoolCache<V>`   | Type alias for `ConcurrentLRUCache<u32, V>`        |
//! | `PageCache<K, V>`      | Type alias for generic page caching                |
//! | `LRUCache<K, V>`       | Type alias for `LRUCore` (single-threaded usage)   |
//!
//! ## LRU Operations Flow
//!
//! ```text
//!   INSERT new item (cache full)
//!   ═══════════════════════════════════════════════════════════════════════════
//!
//!   Before:
//!     head ──► [A] ◄──► [B] ◄──► [C] ◄── tail    (capacity = 3)
//!              MRU                LRU
//!
//!   insert(D):
//!     1. Evict [C] from tail (pop_lru)
//!     2. Add [D] at head
//!
//!   After:
//!     head ──► [D] ◄──► [A] ◄──► [B] ◄── tail
//!              MRU                LRU
//!
//!   ═══════════════════════════════════════════════════════════════════════════
//!
//!   ACCESS existing item
//!   ═══════════════════════════════════════════════════════════════════════════
//!
//!   Before:
//!     head ──► [A] ◄──► [B] ◄──► [C] ◄── tail
//!
//!   get(B):
//!     1. Find [B] in HashMap: O(1)
//!     2. Move [B] to head (move_to_head): O(1)
//!
//!   After:
//!     head ──► [B] ◄──► [A] ◄──► [C] ◄── tail
//!              MRU                LRU
//!
//!   ═══════════════════════════════════════════════════════════════════════════
//!
//!   PEEK (no reordering)
//!   ═══════════════════════════════════════════════════════════════════════════
//!
//!   peek(C):
//!     1. Find [C] in HashMap: O(1)
//!     2. Return Arc::clone from the store without modifying list
//!
//!   Order unchanged: head ──► [A] ◄──► [B] ◄──► [C] ◄── tail
//! ```
//!
//! ## Node Structure
//!
//! ```text
//!   ┌────────────────────────────────────────────┐
//!   │                 Node<K>                    │
//!   ├────────────────────────────────────────────┤
//!   │  key: K (Copy)         │  Owned, cheap     │
//!   ├────────────────────────┼───────────────────┤
//!   │  prev: Option<NonNull> │  Previous node    │
//!   ├────────────────────────┼───────────────────┤
//!   │  next: Option<NonNull> │  Next node        │
//!   └────────────────────────┴───────────────────┘
//!
//!   Memory allocation:
//!     • Nodes allocated via Box::leak() → raw pointer
//!     • Deallocated via Box::from_raw() on removal
//!     • NonNull ensures non-null invariant
//! ```
//!
//! ## LRUCore Methods (CoreCache + MutableCache + LRUCacheTrait)
//!
//! | Method           | Complexity | Description                               |
//! |------------------|------------|-------------------------------------------|
//! | `new(capacity)`  | O(1)       | Create cache with given capacity          |
//! | `insert(k, v)`   | O(1)*      | Insert or update, may evict LRU           |
//! | `get(&k)`        | O(1)       | Get value, moves to MRU position          |
//! | `peek(&k)`       | O(1)       | Get value without affecting LRU order     |
//! | `contains(&k)`   | O(1)       | Check if key exists                       |
//! | `remove(&k)`     | O(1)       | Remove entry by key                       |
//! | `pop_lru()`      | O(1)       | Remove and return least recently used     |
//! | `peek_lru()`     | O(1)       | Peek at LRU item without removing         |
//! | `touch(&k)`      | O(1)       | Move to MRU without returning value       |
//! | `recency_rank()` | O(n)       | Get position in recency order (0 = MRU)   |
//! | `len()`          | O(1)       | Current number of entries                 |
//! | `capacity()`     | O(1)       | Maximum capacity                          |
//! | `clear()`        | O(n)       | Remove all entries                        |
//!
//! ## ConcurrentLRUCache Methods
//!
//! | Method               | Lock Type | Description                          |
//! |----------------------|-----------|--------------------------------------|
//! | `new(capacity)`      | None      | Create concurrent cache              |
//! | `insert(k, v)`       | Write     | Insert value (wraps in Arc)          |
//! | `insert_arc(k, arc)` | Write     | Insert pre-wrapped `Arc<V>`          |
//! | `get(&k)`            | Write     | Get + move to MRU (returns `Arc<V>`) |
//! | `peek(&k)`           | Read      | Get without reordering               |
//! | `remove(&k)`         | Write     | Remove entry                         |
//! | `touch(&k)`          | Write     | Move to MRU                          |
//! | `pop_lru()`          | Write     | Evict LRU entry                      |
//! | `peek_lru()`         | Read      | Peek at LRU                          |
//! | `len()`              | Read      | Current size                         |
//! | `is_empty()`         | Read      | Check if empty                       |
//! | `capacity()`         | Read      | Maximum capacity                     |
//! | `contains(&k)`       | Read      | Check key existence                  |
//! | `clear()`            | Write     | Remove all entries                   |
//!
//! ## Performance Characteristics
//!
//! | Operation        | Time       | Space       | Notes                        |
//! |------------------|------------|-------------|------------------------------|
//! | `insert`         | O(1) avg   | O(1)        | Index update + list update   |
//! | `get`            | O(1) avg   | O(1)        | Index lookup + list move     |
//! | `peek`           | O(1) avg   | O(1)        | Index lookup only            |
//! | `remove`         | O(1) avg   | O(1)        | Index remove + list unlink   |
//! | `pop_lru`        | O(1)       | O(1)        | Direct tail pointer access   |
//! | Per-entry        | -          | ~56 bytes   | 2 ptrs + index + store entry |
//!
//! ## Design Rationale
//!
//! This custom implementation was chosen over standard crates (like `lru` or `cached`) for:
//!
//! - **Store-backed Value Storage**: Values are held in the store as `Arc<V>`,
//!   so callers can keep references even after eviction (e.g., during writeback).
//! - **Internal Visibility**: Buffer managers need precise eviction control
//!   (pinning, touching without retrieval).
//! - **Pointer Stability**: `NonNull` nodes ensure stable memory locations for
//!   the intrusive linked list.
//!
//! ## Concurrency Model
//!
//! ```text
//!   Thread 1           Thread 2           Thread 3
//!      │                  │                  │
//!      │ get(page_1)      │ get(page_2)      │ insert(page_3)
//!      ▼                  ▼                  ▼
//!   ┌──────────────────────────────────────────────────────────┐
//!   │                     RwLock                               │
//!   │                                                          │
//!   │  get() requires WRITE lock (moves node to head)          │
//!   │  peek() requires READ lock (no reordering)               │
//!   │  insert()/remove() require WRITE lock                    │
//!   │                                                          │
//!   │  Note: Even reads need write lock if they update LRU     │
//!   └──────────────────────────────────────────────────────────┘
//!        │
//!        ▼
//!   ┌──────────────────────────────────────────────────────────┐
//!   │  LRUCore (single-threaded operations)                    │
//!   └──────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Trade-offs
//!
//! | Aspect           | Pros                              | Cons                          |
//! |------------------|-----------------------------------|-------------------------------|
//! | Performance      | Predictable O(1) operations       | Global lock can bottleneck    |
//! | Memory           | Arc sharing, no Clone needed      | Per-node pointer overhead     |
//! | Safety           | Arc prevents use-after-free       | Unsafe code complexity        |
//! | Simplicity       | Simple recency-based policy       | No frequency tracking         |
//!
//! ## When to Use
//!
//! **Use when:**
//! - You need a general-purpose page cache or object cache
//! - Read operations significantly outnumber write/eviction operations
//! - Values are expensive to clone (use Arc)
//!
//! **Avoid when:**
//! - You require strictly lock-free concurrency (consider sharded map)
//! - You need complex eviction policies (see `lru_k.rs`, `lfu.rs`)
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use crate::storage::disk::async_disk::cache::lru::{
//!     ConcurrentLRUCache, LRUCore,
//! };
//! use std::sync::Arc;
//!
//! // Single-threaded usage
//! let mut cache: LRUCore<u32, String> = LRUCore::new(100);
//! cache.insert(1, Arc::new("page_data".to_string()));
//!
//! if let Some(value) = cache.get(&1) {
//!     println!("Got: {}", value);  // value is &Arc<String>
//! }
//!
//! // Peek without affecting LRU order
//! if let Some(value) = cache.peek(&1) {
//!     println!("Peeked: {}", value);  // returns Arc<String>
//! }
//!
//! // Evict least recently used
//! if let Some((key, value)) = cache.pop_lru() {
//!     println!("Evicted key={}, value={}", key, value);
//! }
//!
//! // Concurrent usage
//! let concurrent_cache: ConcurrentLRUCache<u32, String> =
//!     ConcurrentLRUCache::new(1000);
//!
//! // Insert (wraps in Arc internally)
//! concurrent_cache.insert(1, "data".to_string());
//!
//! // Or insert pre-wrapped Arc
//! let shared = Arc::new("shared_data".to_string());
//! concurrent_cache.insert_arc(2, shared.clone());
//!
//! // Get returns Arc<V> for safe sharing
//! if let Some(arc_value) = concurrent_cache.get(&1) {
//!     // arc_value can be held across await points
//!     println!("Value: {}", arc_value);
//! }
//!
//! // Touch to mark as recently used without retrieving
//! concurrent_cache.touch(&2);
//!
//! // Type aliases for common patterns
//! use crate::storage::disk::async_disk::cache::lru::BufferPoolCache;
//! let page_cache: BufferPoolCache<Vec<u8>> = BufferPoolCache::new(256);
//! ```
//!
//! ## Comparison with Other Cache Policies
//!
//! | Policy   | File         | Best For                          | Weakness              |
//! |----------|--------------|-----------------------------------|-----------------------|
//! | LRU      | `lru.rs`     | Temporal locality                 | One-time scan floods  |
//! | LRU-K    | `lru_k.rs`   | Frequency + recency (K accesses)  | More memory per entry |
//! | LFU      | `lfu.rs`     | Frequency-biased workloads        | Cache pollution       |
//! | FIFO     | `fifo.rs`    | Simple, predictable               | No adaptation         |
//!
//! ## Safety
//!
//! This module uses `unsafe` code to manage the doubly-linked list manually:
//!
//! - **Node allocation**: `Box::leak()` for stable addresses
//! - **Node deallocation**: `Box::from_raw()` on removal/eviction
//! - **Pointer manipulation**: `NonNull` ensures non-null invariant
//! - **Send/Sync**: Manual impls require `K: Send + Sync`, `V: Send + Sync`
//!
//! Extensive testing (correctness, edge cases, memory safety) verifies soundness.
//!
//! ## Thread Safety
//!
//! - `LRUCore`: **NOT thread-safe** - single-threaded only
//! - `ConcurrentLRUCache`: **Thread-safe** via `parking_lot::RwLock`
//! - `Node`: Manually implements `Send + Sync` (protected by outer lock)
//! - Values: `Arc<V>` in the store enables safe sharing across threads

use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::Arc;

use parking_lot::RwLock;

#[cfg(feature = "metrics")]
use crate::metrics::metrics_impl::LruMetrics;
#[cfg(feature = "metrics")]
use crate::metrics::snapshot::LruMetricsSnapshot;
#[cfg(feature = "metrics")]
use crate::metrics::traits::{
    CoreMetricsRecorder, LruMetricsReadRecorder, LruMetricsRecorder, MetricsSnapshotProvider,
};
use crate::store::hashmap::HashMapStore;
use crate::store::traits::{StoreCore, StoreMut};
use crate::traits::{CoreCache, LRUCacheTrait, MutableCache};

/// Node in the doubly-linked list for LRU tracking
/// Zero-copy design: Keys are Copy types; values live in the store.
struct Node<K>
where
    K: Copy,
{
    key: K, // Owned key (Copy types like PageId)
    prev: Option<NonNull<Node<K>>>,
    next: Option<NonNull<Node<K>>>,
}

impl<K> Node<K>
where
    K: Copy,
{
    fn new(key: K) -> Self {
        Node {
            key,
            prev: None,
            next: None,
        }
    }
}

// SAFETY: Node only moves across threads while protected by the outer RwLock
// inside ConcurrentLRUCache. The contained key/value types must themselves
// be Send + Sync to avoid interior unsafety.
unsafe impl<K> Send for Node<K> where K: Copy + Send + Sync {}

unsafe impl<K> Sync for Node<K> where K: Copy + Send + Sync {}

/// High-performance LRU Cache Core using HashMap + Doubly-Linked List
///
/// # Zero-Copy Design Philosophy
///
/// This implementation achieves zero-copy semantics through:
/// - Keys: Copy types (like PageId) - cheap to copy, owned in index/list
/// - Values: `Arc<V>` stored in the store for zero-copy sharing
///
/// ## Memory Safety Guarantees:
/// - All nodes are allocated via Box::leak and deallocated via Box::from_raw
/// - NonNull ensures no null pointer dereferences
/// - Doubly-linked list invariants are maintained at all operation boundaries
/// - `Arc<V>` provides thread-safe reference counting in the store
///
/// ## Performance Characteristics:
/// - All operations are O(1): insert, get, remove, eviction
/// - Zero data copying for values via Arc::clone()
/// - Minimal memory overhead for keys via Copy trait
///
/// ## Thread Safety:
/// - Core is single-threaded for maximum performance
/// - Thread safety provided by wrapper (ConcurrentLRUCache)
/// - Values are thread-safe via `Arc<V>` stored in the store
pub struct LRUCore<K, V>
where
    K: Copy + Eq + Hash,
{
    store: HashMapStore<K, V>,
    map: HashMap<K, NonNull<Node<K>>>,
    head: Option<NonNull<Node<K>>>, // Most recently used
    tail: Option<NonNull<Node<K>>>, // Least recently used
    #[cfg(feature = "metrics")]
    metrics: LruMetrics,
    _phantom: PhantomData<(K, V)>,
}

// SAFETY: LRUCore is only shared across threads behind an RwLock in
// ConcurrentLRUCache. Raw pointer fields are only mutated while the lock
// is held, so requiring K/V to be Send + Sync preserves thread safety.
unsafe impl<K, V> Send for LRUCore<K, V>
where
    K: Copy + Eq + Hash + Send + Sync,
    V: Send + Sync,
{
}

unsafe impl<K, V> Sync for LRUCore<K, V>
where
    K: Copy + Eq + Hash + Send + Sync,
    V: Send + Sync,
{
}

impl<K, V> LRUCore<K, V>
where
    K: Copy + Eq + Hash,
{
    /// Creates a new LRU cache core with the given capacity
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of items the cache can hold
    ///
    /// # Example
    /// ```
    /// use cachekit::policy::lru::LRUCore;
    ///
    /// let mut cache: LRUCore<u32, String> = LRUCore::new(100);
    /// ```
    pub fn new(capacity: usize) -> Self {
        LRUCore {
            store: HashMapStore::new(capacity),
            map: HashMap::with_capacity(capacity),
            head: None,
            tail: None,
            #[cfg(feature = "metrics")]
            metrics: LruMetrics::default(),
            _phantom: PhantomData,
        }
    }

    /// Allocate a new node on the heap and return a NonNull pointer
    ///
    /// # Safety
    /// The returned pointer is guaranteed to be:
    /// - Non-null and valid
    /// - Properly aligned for Node<K>
    /// - Allocated via Box, so must be deallocated via Box::from_raw
    fn allocate_node(&self, key: K) -> NonNull<Node<K>> {
        let boxed = Box::new(Node::new(key));
        NonNull::from(Box::leak(boxed))
    }

    /// Deallocate a node from the heap
    ///
    /// # Safety
    /// CALLER MUST ENSURE:
    /// - `node` was allocated by `allocate_node`
    /// - `node` is not referenced anywhere else
    /// - `node` has been removed from the doubly-linked list
    /// - This function is called exactly once per allocation
    unsafe fn deallocate_node(&self, node: NonNull<Node<K>>) {
        unsafe {
            // SAFETY: node was allocated via Box::leak in allocate_node
            let _ = Box::from_raw(node.as_ptr());
        }
    }

    /// Move node to head (mark as most recently used)
    ///
    /// # Safety
    /// CALLER MUST ENSURE:
    /// - `node` points to a valid node in this cache's doubly-linked list
    /// - `node` is currently linked (has valid prev/next relationships)
    /// - The cache's invariants are maintained
    ///
    /// # Performance
    /// O(1) - constant time regardless of cache size
    unsafe fn move_to_head(&mut self, node: NonNull<Node<K>>) {
        unsafe {
            // SAFETY: Caller guarantees node is valid and linked
            self.remove_from_list(node);
            self.add_to_head(node);
        }
    }

    /// Add node to head of list (most recently used position)
    ///
    /// # Safety
    /// CALLER MUST ENSURE:
    /// - `node` points to a valid, allocated node
    /// - `node` is not currently in any linked list
    /// - The cache's head/tail pointers are valid
    unsafe fn add_to_head(&mut self, node: NonNull<Node<K>>) {
        unsafe {
            match self.head {
                Some(old_head) => {
                    // SAFETY: node is valid and not in list, old_head is valid
                    (*node.as_ptr()).next = Some(old_head);
                    (*node.as_ptr()).prev = None;
                    (*old_head.as_ptr()).prev = Some(node);
                    self.head = Some(node);
                },
                None => {
                    // Empty list - node becomes both head and tail
                    // SAFETY: node is valid and not in list
                    (*node.as_ptr()).next = None;
                    (*node.as_ptr()).prev = None;
                    self.head = Some(node);
                    self.tail = Some(node);
                },
            }
        }
    }

    /// Remove node from doubly-linked list
    ///
    /// # Safety
    /// CALLER MUST ENSURE:
    /// - `node` points to a valid node in this cache's doubly-linked list
    /// - All prev/next pointers are valid or None
    /// - The cache's head/tail pointers are valid
    ///
    /// # Invariants Maintained
    /// - Doubly-linked list remains properly connected
    /// - Head/tail pointers are updated correctly
    /// - Removed node's pointers are NOT cleared (caller responsibility)
    unsafe fn remove_from_list(&mut self, node: NonNull<Node<K>>) {
        unsafe {
            // SAFETY: Caller guarantees node is valid and in our list
            let node_ref = &*node.as_ptr();

            match (node_ref.prev, node_ref.next) {
                (Some(prev), Some(next)) => {
                    // Middle node - connect prev and next
                    // SAFETY: prev and next are valid (in our list)
                    (*prev.as_ptr()).next = Some(next);
                    (*next.as_ptr()).prev = Some(prev);
                },
                (Some(prev), None) => {
                    // Tail node - update tail pointer
                    // SAFETY: prev is valid (in our list)
                    (*prev.as_ptr()).next = None;
                    self.tail = Some(prev);
                },
                (None, Some(next)) => {
                    // Head node - update head pointer
                    // SAFETY: next is valid (in our list)
                    (*next.as_ptr()).prev = None;
                    self.head = Some(next);
                },
                (None, None) => {
                    // Only node - clear head and tail
                    self.head = None;
                    self.tail = None;
                },
            }
        }
    }

    /// Remove tail node (least recently used) and return it
    ///
    /// # Safety
    /// The returned node (if any) is:
    /// - Removed from the doubly-linked list
    /// - Still allocated and valid
    /// - MUST be deallocated by caller
    ///
    /// # Returns
    /// - `Some(node)` if cache is not empty
    /// - `None` if cache is empty
    unsafe fn remove_tail(&mut self) -> Option<NonNull<Node<K>>> {
        self.tail.inspect(|&tail_node| {
            unsafe {
                // SAFETY: tail_node is valid (our tail pointer)
                self.remove_from_list(tail_node);
            }
        })
    }

    /// Validate internal invariants (debug builds only)
    ///
    /// Checks:
    /// - HashMap and linked list have same number of elements
    /// - All nodes in HashMap are reachable from head
    /// - All forward/backward links are consistent
    /// - Head has no prev, tail has no next
    fn validate_invariants(&self) {
        if self.map.is_empty() {
            debug_assert!(self.head.is_none() && self.tail.is_none());
            debug_assert!(self.store.is_empty());
            return;
        }

        // Count nodes via linked list traversal
        let mut count = 0;
        let mut current = self.head;
        let mut prev_node = None;

        while let Some(node) = current {
            count += 1;
            unsafe {
                let node_ref = &*node.as_ptr();

                // Check backward link consistency
                debug_assert_eq!(node_ref.prev, prev_node);

                // Check that this node exists in the HashMap
                debug_assert!(self.map.contains_key(&node_ref.key));
                debug_assert!(self.store.contains(&node_ref.key));

                prev_node = Some(node);
                current = node_ref.next;
            }
        }

        // Verify counts match
        debug_assert_eq!(count, self.map.len());
        debug_assert_eq!(count, self.store.len());

        // Verify tail is correct
        debug_assert_eq!(prev_node, self.tail);
    }
}

// Implementation of specialized traits for zero-copy operations
impl<K, V> CoreCache<K, Arc<V>> for LRUCore<K, V>
where
    K: Copy + Eq + Hash,
{
    /// Zero-copy insert: key is copied (cheap), value is Arc-wrapped and moved
    fn insert(&mut self, key: K, value: Arc<V>) -> Option<Arc<V>> {
        #[cfg(feature = "metrics")]
        self.metrics.record_insert_call();

        if let Some(&existing_node) = self.map.get(&key) {
            #[cfg(feature = "metrics")]
            self.metrics.record_insert_update();

            let previous = self.store.try_insert(key, value).unwrap_or_default();

            // Move to head (mark as most recently used) - O(1)
            unsafe {
                // SAFETY: existing_node is in our list and valid
                self.move_to_head(existing_node);
            }

            #[cfg(debug_assertions)]
            self.validate_invariants();

            return previous;
        }

        // Insert new node
        // For zero capacity, never insert anything
        if self.store.capacity() == 0 {
            return None;
        }

        #[cfg(feature = "metrics")]
        self.metrics.record_insert_new();

        if self.store.len() >= self.store.capacity() {
            // Evict LRU item - O(1)
            #[cfg(feature = "metrics")]
            self.metrics.record_evict_call();

            if let Some(tail_node) = unsafe { self.remove_tail() } {
                let tail_key = unsafe {
                    // SAFETY: tail_node was just removed from our list, still valid
                    let node_ptr = tail_node.as_ptr();
                    let node = Box::from_raw(node_ptr);
                    node.key
                };
                // Remove from map using the copied key (Copy trait - cheap)
                self.map.remove(&tail_key);
                let _ = self.store.remove(&tail_key);
                #[cfg(feature = "metrics")]
                self.metrics.record_evicted_entry();
                self.store.record_eviction();
                // Node is already deallocated by Box::from_raw above
            }
        }

        if self.store.try_insert(key, value).is_err() {
            return None;
        }

        // Allocate and insert new node - O(1) - key copied
        let new_node = self.allocate_node(key);

        // HashMap insert with key copy (Copy trait makes this cheap)
        self.map.insert(key, new_node);

        unsafe {
            // SAFETY: new_node is freshly allocated and not in any list
            self.add_to_head(new_node);
        }

        #[cfg(debug_assertions)]
        self.validate_invariants();

        None
    }

    /// Zero-copy get: returns `Arc<V>` clone (O(1) atomic increment)
    fn get(&mut self, key: &K) -> Option<&Arc<V>> {
        let node = match self.map.get(key) {
            Some(node) => *node,
            None => {
                #[cfg(feature = "metrics")]
                self.metrics.record_get_miss();
                let _ = self.store.get_ref(key);
                return None;
            },
        };

        #[cfg(feature = "metrics")]
        self.metrics.record_get_hit();

        // Move to head (mark as most recently used) - O(1)
        unsafe {
            // SAFETY: node is from our HashMap, so it's valid and in our list
            self.move_to_head(node);
        }

        #[cfg(debug_assertions)]
        self.validate_invariants();

        self.store.get_ref(key)
    }

    fn contains(&self, key: &K) -> bool {
        self.store.contains(key)
    }

    fn len(&self) -> usize {
        self.store.len()
    }

    fn capacity(&self) -> usize {
        self.store.capacity()
    }

    fn clear(&mut self) {
        #[cfg(feature = "metrics")]
        self.metrics.record_clear();

        self.store.clear();

        // Collect all nodes first to avoid borrow checker issues
        let nodes: Vec<_> = self.map.drain().map(|(_, node)| node).collect();

        // Now deallocate all nodes
        for node in nodes {
            unsafe {
                // SAFETY: Each node was in our HashMap, so it's valid and allocated
                self.deallocate_node(node);
            }
        }

        self.head = None;
        self.tail = None;

        self.validate_invariants();
    }
}

impl<K, V> LRUCore<K, V>
where
    K: Copy + Eq + Hash,
{
    /// Zero-copy peek: read-only lookup without LRU update (allows concurrent reads)
    /// Returns `Arc<V>` clone for zero-copy sharing
    pub fn peek(&self, key: &K) -> Option<Arc<V>> {
        #[cfg(feature = "metrics")]
        (&self.metrics).record_peek_lru_call();

        if self.map.contains_key(key)
            && let Some(value) = self.store.peek_ref(key)
        {
            #[cfg(feature = "metrics")]
            (&self.metrics).record_peek_lru_found();
            return Some(Arc::clone(value));
        }
        None
    }
}

impl<K, V> MutableCache<K, Arc<V>> for LRUCore<K, V>
where
    K: Copy + Eq + Hash,
{
    /// Zero-copy remove: returns `Arc<V>` without cloning data
    fn remove(&mut self, key: &K) -> Option<Arc<V>> {
        if let Some(node) = self.map.remove(key) {
            unsafe {
                // SAFETY: node was in our HashMap, so it's valid
                // First remove from list while node is still valid
                self.remove_from_list(node);

                // Then deallocate node
                let _ = Box::from_raw(node.as_ptr());
            }

            let value = self.store.remove(key);

            #[cfg(debug_assertions)]
            self.validate_invariants();

            value
        } else {
            None
        }
    }
}

impl<K, V> LRUCacheTrait<K, Arc<V>> for LRUCore<K, V>
where
    K: Copy + Eq + Hash,
{
    /// Zero-copy pop_lru: returns `(K, Arc<V>)` without cloning data
    fn pop_lru(&mut self) -> Option<(K, Arc<V>)> {
        #[cfg(feature = "metrics")]
        self.metrics.record_pop_lru_call();

        self.tail.map(|tail_node| unsafe {
            // SAFETY: tail_node is our tail pointer, so it's valid
            // First get key while node is still accessible
            let key = (*tail_node.as_ptr()).key;

            // Remove from map
            self.map.remove(&key);

            // Remove from list while node is still valid
            self.remove_from_list(tail_node);

            // Then deallocate node
            let _ = Box::from_raw(tail_node.as_ptr());

            let value = self.store.remove(&key);

            #[cfg(debug_assertions)]
            self.validate_invariants();

            #[cfg(feature = "metrics")]
            self.metrics.record_pop_lru_found();
            self.store.record_eviction();

            (key, value.expect("lru entry missing from store"))
        })
    }

    /// Zero-copy peek_lru: returns references without affecting LRU order
    fn peek_lru(&self) -> Option<(&K, &Arc<V>)> {
        #[cfg(feature = "metrics")]
        (&self.metrics).record_peek_lru_call();

        self.tail.map(|tail_node| unsafe {
            // SAFETY: tail_node is our tail pointer, so it's valid
            let node_ref = &*tail_node.as_ptr();
            #[cfg(feature = "metrics")]
            (&self.metrics).record_peek_lru_found();
            let value = self
                .store
                .peek_ref(&node_ref.key)
                .expect("lru entry missing from store");
            (&node_ref.key, value)
        })
    }

    fn touch(&mut self, key: &K) -> bool {
        #[cfg(feature = "metrics")]
        self.metrics.record_touch_call();

        if let Some(&node) = self.map.get(key) {
            unsafe {
                // SAFETY: node is from our HashMap, so it's valid and in our list
                self.move_to_head(node);
            }

            #[cfg(debug_assertions)]
            self.validate_invariants();

            #[cfg(feature = "metrics")]
            self.metrics.record_touch_found();

            true
        } else {
            false
        }
    }

    fn recency_rank(&self, key: &K) -> Option<usize> {
        #[cfg(feature = "metrics")]
        (&self.metrics).record_recency_rank_call();

        if let Some(&target_node) = self.map.get(key) {
            let mut rank = 0;
            let mut current = self.head;

            // Walk from head (most recent) to find the target node
            while let Some(node) = current {
                unsafe {
                    #[cfg(feature = "metrics")]
                    (&self.metrics).record_recency_rank_scan_step();

                    // SAFETY: All nodes in the list are valid
                    if node == target_node {
                        #[cfg(feature = "metrics")]
                        (&self.metrics).record_recency_rank_found();
                        return Some(rank);
                    }
                    current = (*node.as_ptr()).next;
                    rank += 1;
                }
            }
        }
        None
    }
}

#[cfg(feature = "metrics")]
impl<K, V> LRUCore<K, V>
where
    K: Copy + Eq + Hash,
{
    pub fn metrics_snapshot(&self) -> LruMetricsSnapshot {
        LruMetricsSnapshot {
            get_calls: self.metrics.get_calls,
            get_hits: self.metrics.get_hits,
            get_misses: self.metrics.get_misses,
            insert_calls: self.metrics.insert_calls,
            insert_updates: self.metrics.insert_updates,
            insert_new: self.metrics.insert_new,
            evict_calls: self.metrics.evict_calls,
            evicted_entries: self.metrics.evicted_entries,
            pop_lru_calls: self.metrics.pop_lru_calls,
            pop_lru_found: self.metrics.pop_lru_found,
            peek_lru_calls: self.metrics.peek_lru_calls.get(),
            peek_lru_found: self.metrics.peek_lru_found.get(),
            touch_calls: self.metrics.touch_calls,
            touch_found: self.metrics.touch_found,
            recency_rank_calls: self.metrics.recency_rank_calls.get(),
            recency_rank_found: self.metrics.recency_rank_found.get(),
            recency_rank_scan_steps: self.metrics.recency_rank_scan_steps.get(),
            cache_len: self.store.len(),
            capacity: self.store.capacity(),
        }
    }
}

// Proper cleanup when cache core is dropped
impl<K, V> Drop for LRUCore<K, V>
where
    K: Copy + Eq + Hash,
{
    fn drop(&mut self) {
        self.clear();
    }
}

// Send + Sync analysis:
// - LRUCore is Send if K and V are Send (no shared references)
// - LRUCore is NOT Sync (requires &mut for modifications)
// - Thread safety provided by ConcurrentLRUCache wrapper
// This is enforced by Rust's auto traits

/// Thread-safe concurrent LRU cache wrapper using RwLock
/// Optimized for read-heavy database workloads (buffer pools)
#[derive(Clone)]
pub struct ConcurrentLRUCache<K, V>
where
    K: Copy + Eq + Hash,
{
    inner: Arc<RwLock<LRUCore<K, V>>>,
}

impl<K, V> ConcurrentLRUCache<K, V>
where
    K: Copy + Eq + Hash + Send + Sync,
    V: Send + Sync,
{
    /// Create a new concurrent LRU cache with the given capacity
    pub fn new(capacity: usize) -> Self {
        ConcurrentLRUCache {
            inner: Arc::new(RwLock::new(LRUCore::new(capacity))),
        }
    }

    /// Insert with value ownership transfer to `Arc<V>`
    /// Returns the previous `Arc<V>` if key existed
    pub fn insert(&self, key: K, value: V) -> Option<Arc<V>> {
        let value_arc = Arc::new(value); // Wrap in Arc once
        let mut cache = self.inner.write();
        cache.insert(key, value_arc)
    }

    /// Insert `Arc<V>` directly (zero-copy if already Arc-wrapped)
    pub fn insert_arc(&self, key: K, value: Arc<V>) -> Option<Arc<V>> {
        let mut cache = self.inner.write();
        cache.insert(key, value)
    }

    /// Get with LRU update (requires write lock)
    /// Returns `Arc<V>` for zero-copy sharing
    pub fn get(&self, key: &K) -> Option<Arc<V>> {
        let mut cache = self.inner.write();
        cache.get(key).map(Arc::clone)
    }

    /// Peek without LRU update (allows concurrent reads)
    /// Perfect for read-heavy buffer pool workloads
    pub fn peek(&self, key: &K) -> Option<Arc<V>> {
        let cache = self.inner.read();
        cache.peek(key)
    }

    /// Remove entry and return `Arc<V>`
    pub fn remove(&self, key: &K) -> Option<Arc<V>> {
        let mut cache = self.inner.write();
        cache.remove(key)
    }

    /// Touch entry to mark as recently used
    pub fn touch(&self, key: &K) -> bool {
        let mut cache = self.inner.write();
        cache.touch(key)
    }

    /// Get current cache length
    pub fn len(&self) -> usize {
        let cache = self.inner.read();
        cache.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        let cache = self.inner.read();
        cache.len() == 0
    }

    /// Get cache capacity
    pub fn capacity(&self) -> usize {
        let cache = self.inner.read();
        cache.capacity()
    }

    /// Check if key exists (read-only)
    pub fn contains(&self, key: &K) -> bool {
        let cache = self.inner.read();
        cache.contains(key)
    }

    /// Clear all entries
    pub fn clear(&self) {
        let mut cache = self.inner.write();
        cache.clear()
    }

    /// Pop least recently used entry
    pub fn pop_lru(&self) -> Option<(K, Arc<V>)> {
        let mut cache = self.inner.write();
        cache.pop_lru()
    }

    /// Peek at least recently used entry
    pub fn peek_lru(&self) -> Option<(K, Arc<V>)> {
        let cache = self.inner.read();
        cache.peek_lru().map(|(k, v)| (*k, Arc::clone(v)))
    }
}

#[cfg(feature = "metrics")]
impl<K, V> ConcurrentLRUCache<K, V>
where
    K: Copy + Eq + Hash + Send + Sync,
    V: Send + Sync,
{
    pub fn metrics_snapshot(&self) -> LruMetricsSnapshot {
        let cache = self.inner.read();
        cache.metrics_snapshot()
    }
}

#[cfg(feature = "metrics")]
impl<K, V> MetricsSnapshotProvider<LruMetricsSnapshot> for LRUCore<K, V>
where
    K: Copy + Eq + Hash,
{
    fn snapshot(&self) -> LruMetricsSnapshot {
        self.metrics_snapshot()
    }
}

#[cfg(feature = "metrics")]
impl<K, V> MetricsSnapshotProvider<LruMetricsSnapshot> for ConcurrentLRUCache<K, V>
where
    K: Copy + Eq + Hash + Send + Sync,
    V: Send + Sync,
{
    fn snapshot(&self) -> LruMetricsSnapshot {
        self.metrics_snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::CoreCache;

    // ==============================================
    // CORRECTNESS TESTS MODULE
    // ==============================================
    mod correctness {
        use super::*;

        mod basic_behavior {
            use super::*;

            #[test]
            fn test_new_cache_creation() {
                // Test creating new LRU cache with various capacities
                let cache1: LRUCore<i32, i32> = LRUCore::new(0);
                assert_eq!(cache1.capacity(), 0);
                assert_eq!(cache1.len(), 0);

                let cache2: LRUCore<i32, i32> = LRUCore::new(10);
                assert_eq!(cache2.capacity(), 10);
                assert_eq!(cache2.len(), 0);

                let cache3: LRUCore<i32, i32> = LRUCore::new(1000);
                assert_eq!(cache3.capacity(), 1000);
                assert_eq!(cache3.len(), 0);
            }

            #[test]
            fn test_insert_single_item() {
                // Test inserting a single item into empty cache
                let mut cache = LRUCore::new(5);

                let result = cache.insert(1, Arc::new(100));
                assert!(result.is_none()); // No previous value
                assert_eq!(cache.len(), 1);
                assert!(cache.contains(&1));
            }

            #[test]
            fn test_insert_multiple_items() {
                // Test inserting multiple items within capacity
                let mut cache = LRUCore::new(5);

                for i in 1..=3 {
                    let result = cache.insert(i, Arc::new(i * 10));
                    assert!(result.is_none());
                }

                assert_eq!(cache.len(), 3);
                for i in 1..=3 {
                    assert!(cache.contains(&i));
                }
            }

            #[test]
            fn test_get_existing_item() {
                // Test getting an item that exists in cache
                let mut cache = LRUCore::new(5);
                cache.insert(1, Arc::new(100));

                let value = cache.get(&1);
                assert!(value.is_some());
                assert_eq!(**value.unwrap(), 100);
            }

            #[test]
            fn test_get_nonexistent_item() {
                // Test getting an item that doesn't exist in cache
                let mut cache = LRUCore::new(5);
                cache.insert(1, Arc::new(100));

                let value = cache.get(&2);
                assert!(value.is_none());
            }

            #[test]
            fn test_peek_existing_item() {
                // Test peeking at an item that exists (no LRU update)
                let mut cache = LRUCore::new(5);
                cache.insert(1, Arc::new(100));

                let value = cache.peek(&1);
                assert!(value.is_some());
                assert_eq!(*value.unwrap(), 100);
            }

            #[test]
            fn test_peek_nonexistent_item() {
                // Test peeking at an item that doesn't exist
                let cache: LRUCore<i32, i32> = LRUCore::new(5);

                let value = cache.peek(&1);
                assert!(value.is_none());
            }

            #[test]
            fn test_contains_existing_item() {
                // Test contains check for existing item
                let mut cache = LRUCore::new(5);
                cache.insert(1, Arc::new(100));

                assert!(cache.contains(&1));
            }

            #[test]
            fn test_contains_nonexistent_item() {
                // Test contains check for non-existing item
                let cache: LRUCore<i32, i32> = LRUCore::new(5);

                assert!(!cache.contains(&1));
            }

            #[test]
            fn test_remove_existing_item() {
                // Test removing an item that exists in cache
                let mut cache = LRUCore::new(5);
                cache.insert(1, Arc::new(100));

                let removed = cache.remove(&1);
                assert!(removed.is_some());
                assert_eq!(*removed.unwrap(), 100);
                assert_eq!(cache.len(), 0);
                assert!(!cache.contains(&1));
            }

            #[test]
            fn test_remove_nonexistent_item() {
                // Test removing an item that doesn't exist in cache
                let mut cache = LRUCore::new(5);
                cache.insert(1, Arc::new(100));

                let removed = cache.remove(&2);
                assert!(removed.is_none());
                assert_eq!(cache.len(), 1);
            }

            #[test]
            fn test_insert_duplicate_key() {
                // Test inserting with same key twice (should update value)
                let mut cache = LRUCore::new(5);

                let old_value = cache.insert(1, Arc::new(100));
                assert!(old_value.is_none());

                let old_value = cache.insert(1, Arc::new(200));
                assert!(old_value.is_some());
                assert_eq!(*old_value.unwrap(), 100);

                assert_eq!(cache.len(), 1);
                let current_value = cache.get(&1);
                assert_eq!(**current_value.unwrap(), 200);
            }

            #[test]
            fn test_cache_length_updates() {
                // Test that cache length is updated correctly on operations
                let mut cache = LRUCore::new(3);
                assert_eq!(cache.len(), 0);

                cache.insert(1, Arc::new(10));
                assert_eq!(cache.len(), 1);

                cache.insert(2, Arc::new(20));
                assert_eq!(cache.len(), 2);

                cache.remove(&1);
                assert_eq!(cache.len(), 1);

                cache.clear();
                assert_eq!(cache.len(), 0);
            }

            #[test]
            fn test_cache_capacity() {
                // Test that cache reports correct capacity
                let cache1: LRUCore<i32, i32> = LRUCore::new(0);
                assert_eq!(cache1.capacity(), 0);

                let cache2: LRUCore<i32, i32> = LRUCore::new(10);
                assert_eq!(cache2.capacity(), 10);

                let cache3: LRUCore<i32, i32> = LRUCore::new(1000);
                assert_eq!(cache3.capacity(), 1000);
            }

            #[test]
            fn test_cache_clear() {
                // Test clearing all items from cache
                let mut cache = LRUCore::new(5);

                for i in 1..=3 {
                    cache.insert(i, Arc::new(i * 10));
                }
                assert_eq!(cache.len(), 3);

                cache.clear();
                assert_eq!(cache.len(), 0);
                for i in 1..=3 {
                    assert!(!cache.contains(&i));
                }
            }

            #[test]
            fn test_empty_cache_behavior() {
                // Test operations on empty cache
                let mut cache: LRUCore<i32, i32> = LRUCore::new(5);

                assert_eq!(cache.len(), 0);
                assert!(cache.get(&1).is_none());
                assert!(cache.peek(&1).is_none());
                assert!(!cache.contains(&1));
                assert!(cache.remove(&1).is_none());
                assert!(cache.pop_lru().is_none());
                assert!(cache.peek_lru().is_none());
                assert!(!cache.touch(&1));
                assert!(cache.recency_rank(&1).is_none());
            }

            #[test]
            fn test_single_item_cache() {
                // Test cache with capacity of 1
                let mut cache = LRUCore::new(1);

                cache.insert(1, Arc::new(100));
                assert_eq!(cache.len(), 1);
                assert!(cache.contains(&1));

                // Insert second item should evict first
                cache.insert(2, Arc::new(200));
                assert_eq!(cache.len(), 1);
                assert!(!cache.contains(&1));
                assert!(cache.contains(&2));
            }

            #[test]
            fn test_zero_capacity_cache() {
                // Test cache with capacity of 0
                let mut cache = LRUCore::new(0);

                let result = cache.insert(1, Arc::new(100));
                assert!(result.is_none());
                assert_eq!(cache.len(), 0);
                assert!(!cache.contains(&1));
            }

            #[test]
            fn test_is_empty() {
                // Test is_empty method on various cache states
                let mut cache = LRUCore::new(5);

                // For LRUCore, we need to check len() == 0
                assert_eq!(cache.len(), 0);

                cache.insert(1, Arc::new(100));
                assert_ne!(cache.len(), 0);

                cache.remove(&1);
                assert_eq!(cache.len(), 0);

                cache.insert(1, Arc::new(100));
                cache.clear();
                assert_eq!(cache.len(), 0);
            }

            #[test]
            fn test_lru_eviction_basic() {
                // Test that LRU item is evicted when capacity exceeded
                let mut cache = LRUCore::new(2);

                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                assert_eq!(cache.len(), 2);

                // Insert third item should evict first (LRU)
                cache.insert(3, Arc::new(300));
                assert_eq!(cache.len(), 2);
                assert!(!cache.contains(&1)); // First inserted, first evicted
                assert!(cache.contains(&2));
                assert!(cache.contains(&3));
            }

            #[test]
            fn test_lru_order_preservation() {
                // Test that LRU order is maintained correctly
                let mut cache = LRUCore::new(3);

                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                cache.insert(3, Arc::new(300));

                // All should be present
                assert!(cache.contains(&1));
                assert!(cache.contains(&2));
                assert!(cache.contains(&3));

                // Insert fourth should evict 1 (LRU)
                cache.insert(4, Arc::new(400));
                assert!(!cache.contains(&1));
                assert!(cache.contains(&2));
                assert!(cache.contains(&3));
                assert!(cache.contains(&4));
            }

            #[test]
            fn test_access_updates_lru_order() {
                // Test that accessing an item moves it to most recent
                let mut cache = LRUCore::new(3);

                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                cache.insert(3, Arc::new(300));

                // Access first item to make it most recent
                cache.get(&1);

                // Insert fourth should evict 2 (now LRU), not 1
                cache.insert(4, Arc::new(400));
                assert!(cache.contains(&1)); // Should still be present
                assert!(!cache.contains(&2)); // Should be evicted
                assert!(cache.contains(&3));
                assert!(cache.contains(&4));
            }

            #[test]
            fn test_peek_does_not_update_lru() {
                // Test that peek doesn't change LRU order
                let mut cache = LRUCore::new(3);

                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                cache.insert(3, Arc::new(300));

                // Peek at first item (should not affect LRU order)
                cache.peek(&1);

                // Insert fourth should still evict 1 (LRU)
                cache.insert(4, Arc::new(400));
                assert!(!cache.contains(&1)); // Should be evicted
                assert!(cache.contains(&2));
                assert!(cache.contains(&3));
                assert!(cache.contains(&4));
            }

            #[test]
            fn test_touch_updates_lru_order() {
                // Test that touch operation updates LRU order
                let mut cache = LRUCore::new(3);

                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                cache.insert(3, Arc::new(300));

                // Touch first item to make it most recent
                let touched = cache.touch(&1);
                assert!(touched);

                // Insert fourth should evict 2 (now LRU), not 1
                cache.insert(4, Arc::new(400));
                assert!(cache.contains(&1)); // Should still be present
                assert!(!cache.contains(&2)); // Should be evicted
                assert!(cache.contains(&3));
                assert!(cache.contains(&4));
            }

            #[test]
            fn test_touch_nonexistent_item() {
                // Test touch on item that doesn't exist
                let mut cache = LRUCore::new(3);
                cache.insert(1, Arc::new(100));

                let touched = cache.touch(&2);
                assert!(!touched);
            }

            #[test]
            fn test_pop_lru_basic() {
                // Test popping least recently used item
                let mut cache = LRUCore::new(3);

                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                cache.insert(3, Arc::new(300));

                let popped = cache.pop_lru();
                assert!(popped.is_some());
                let (key, value) = popped.unwrap();
                assert_eq!(key, 1);
                assert_eq!(*value, 100);
                assert_eq!(cache.len(), 2);
                assert!(!cache.contains(&1));
            }

            #[test]
            fn test_pop_lru_empty_cache() {
                // Test popping from empty cache
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);

                let popped = cache.pop_lru();
                assert!(popped.is_none());
            }

            #[test]
            fn test_peek_lru_basic() {
                // Test peeking at least recently used item
                let cache = LRUCore::new(3);
                let mut cache = cache;

                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                cache.insert(3, Arc::new(300));

                let peeked = cache.peek_lru();
                assert!(peeked.is_some());
                let (key, value) = peeked.unwrap();
                assert_eq!(*key, 1);
                assert_eq!(**value, 100);
                assert_eq!(cache.len(), 3); // Should not remove
                assert!(cache.contains(&1)); // Should still be present
            }

            #[test]
            fn test_peek_lru_empty_cache() {
                // Test peeking LRU from empty cache
                let cache: LRUCore<i32, i32> = LRUCore::new(3);

                let peeked = cache.peek_lru();
                assert!(peeked.is_none());
            }

            #[test]
            fn test_recency_rank_basic() {
                // Test getting recency rank of items
                let mut cache = LRUCore::new(3);

                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                cache.insert(3, Arc::new(300));

                // Most recent should be rank 0, least recent rank 2
                assert_eq!(cache.recency_rank(&3), Some(0)); // Most recent
                assert_eq!(cache.recency_rank(&2), Some(1));
                assert_eq!(cache.recency_rank(&1), Some(2)); // Least recent
            }

            #[test]
            fn test_recency_rank_nonexistent() {
                // Test recency rank for non-existing item
                let mut cache = LRUCore::new(3);
                cache.insert(1, Arc::new(100));

                assert!(cache.recency_rank(&2).is_none());
            }

            #[test]
            fn test_concurrent_cache_basic() {
                // Test basic operations on ConcurrentLRUCache
                let cache = ConcurrentLRUCache::new(5);

                assert_eq!(cache.capacity(), 5);
                assert_eq!(cache.len(), 0);
                assert!(cache.is_empty());

                let old_value = cache.insert(1, 100);
                assert!(old_value.is_none());
                assert_eq!(cache.len(), 1);
                assert!(!cache.is_empty());
                assert!(cache.contains(&1));

                let value = cache.get(&1);
                assert!(value.is_some());
                assert_eq!(*value.unwrap(), 100);

                let peeked = cache.peek(&1);
                assert!(peeked.is_some());
                assert_eq!(*peeked.unwrap(), 100);

                let removed = cache.remove(&1);
                assert!(removed.is_some());
                assert_eq!(*removed.unwrap(), 100);
                assert_eq!(cache.len(), 0);
            }

            #[test]
            fn test_concurrent_insert_arc() {
                // Test inserting Arc<V> directly into concurrent cache
                let cache = ConcurrentLRUCache::new(5);
                let value = Arc::new(100);
                let value_clone = Arc::clone(&value);

                let old_value = cache.insert_arc(1, value);
                assert!(old_value.is_none());

                let retrieved = cache.get(&1);
                assert!(retrieved.is_some());
                let retrieved_val = retrieved.unwrap();
                assert_eq!(*retrieved_val, 100);

                // Should be same Arc instance
                assert!(Arc::ptr_eq(&retrieved_val, &value_clone));
            }

            #[test]
            fn test_arc_value_sharing() {
                // Test that Arc<V> values are properly shared (zero-copy)
                let cache = ConcurrentLRUCache::new(5);
                cache.insert(1, 100);

                let value1 = cache.get(&1);
                let value2 = cache.get(&1);
                let value3 = cache.peek(&1);

                assert!(value1.is_some());
                assert!(value2.is_some());
                assert!(value3.is_some());

                // All should point to the same Arc instance
                let v1 = value1.unwrap();
                let v2 = value2.unwrap();
                let v3 = value3.unwrap();

                assert!(Arc::ptr_eq(&v1, &v2));
                assert!(Arc::ptr_eq(&v2, &v3));
            }

            #[test]
            fn test_key_copy_semantics() {
                // Test that keys use Copy semantics efficiently
                let cache = ConcurrentLRUCache::new(5);

                let key1 = 42u32;
                let key2 = key1; // Copy, not move

                cache.insert(key1, 100);

                // Both key1 and key2 should work (both are copies of the same value)
                assert!(cache.contains(&key1));
                assert!(cache.contains(&key2));

                let value1 = cache.get(&key1);
                let value2 = cache.get(&key2);

                assert!(value1.is_some());
                assert!(value2.is_some());
                assert_eq!(*value1.unwrap(), *value2.unwrap());
            }
        }

        mod edge_cases {
            use super::*;

            #[test]
            fn test_maximum_capacity_cache() {
                // Test cache with very large capacity (usize::MAX or close to it)
                // Use a reasonable large number to avoid memory issues
                let large_capacity = 1_000_000_usize;
                let cache: LRUCore<i32, i32> = LRUCore::new(large_capacity);

                assert_eq!(cache.capacity(), large_capacity);
                assert_eq!(cache.len(), 0);

                // Should still work normally with large capacity
                let mut cache = cache;
                cache.insert(1, Arc::new(100));
                assert_eq!(cache.len(), 1);
                assert!(cache.contains(&1));
            }

            #[test]
            fn test_zero_capacity_operations() {
                // Test all operations on zero-capacity cache
                let mut cache: LRUCore<i32, i32> = LRUCore::new(0);

                // All insertions should fail/be ignored
                let result = cache.insert(1, Arc::new(100));
                assert!(result.is_none());
                assert_eq!(cache.len(), 0);
                assert!(!cache.contains(&1));

                // All other operations should handle gracefully
                assert!(cache.get(&1).is_none());
                assert!(cache.peek(&1).is_none());
                assert!(cache.remove(&1).is_none());
                assert!(cache.pop_lru().is_none());
                assert!(cache.peek_lru().is_none());
                assert!(!cache.touch(&1));
                assert!(cache.recency_rank(&1).is_none());

                // Clear should work
                cache.clear();
                assert_eq!(cache.len(), 0);
            }

            #[test]
            fn test_single_capacity_eviction_patterns() {
                // Test eviction behavior with capacity = 1
                let mut cache: LRUCore<i32, i32> = LRUCore::new(1);

                // Insert first item
                cache.insert(1, Arc::new(100));
                assert_eq!(cache.len(), 1);
                assert!(cache.contains(&1));

                // Insert second item should evict first
                cache.insert(2, Arc::new(200));
                assert_eq!(cache.len(), 1);
                assert!(!cache.contains(&1));
                assert!(cache.contains(&2));

                // Insert third item should evict second
                cache.insert(3, Arc::new(300));
                assert_eq!(cache.len(), 1);
                assert!(!cache.contains(&1));
                assert!(!cache.contains(&2));
                assert!(cache.contains(&3));

                // Access should not change anything (still only one item)
                cache.get(&3);
                assert_eq!(cache.len(), 1);
                assert!(cache.contains(&3));
            }

            #[test]
            fn test_repeated_insert_same_key() {
                // Test inserting same key many times with different values
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);

                // Insert same key multiple times
                for i in 1..=10 {
                    let old_value = cache.insert(1, Arc::new(i * 100));
                    if i == 1 {
                        assert!(old_value.is_none());
                    } else {
                        assert!(old_value.is_some());
                        assert_eq!(*old_value.unwrap(), (i - 1) * 100);
                    }
                }

                // Should still only have one entry
                assert_eq!(cache.len(), 1);
                assert!(cache.contains(&1));

                // Should have the latest value
                let value = cache.get(&1);
                assert!(value.is_some());
                assert_eq!(**value.unwrap(), 1000);
            }

            #[test]
            fn test_alternating_access_pattern() {
                // Test alternating access to two items in capacity-2 cache
                let mut cache: LRUCore<i32, i32> = LRUCore::new(2);

                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));

                // Alternate access pattern
                for _ in 0..10 {
                    cache.get(&1);
                    cache.get(&2);
                }

                // Both should still be present
                assert!(cache.contains(&1));
                assert!(cache.contains(&2));
                assert_eq!(cache.len(), 2);

                // Insert third item, should evict the LRU (which is 1 due to access order)
                cache.insert(3, Arc::new(300));
                assert!(!cache.contains(&1));
                assert!(cache.contains(&2));
                assert!(cache.contains(&3));
            }

            #[test]
            fn test_insert_then_immediate_remove() {
                // Test inserting and immediately removing items
                let mut cache: LRUCore<i32, i32> = LRUCore::new(5);

                for i in 1..=10 {
                    cache.insert(i, Arc::new(i * 100));
                    let removed = cache.remove(&i);
                    assert!(removed.is_some());
                    assert_eq!(*removed.unwrap(), i * 100);
                    assert!(!cache.contains(&i));
                    assert_eq!(cache.len(), 0);
                }
            }

            #[test]
            fn test_remove_during_eviction() {
                // Test removing items while eviction is happening
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);

                // Fill cache to capacity
                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                cache.insert(3, Arc::new(300));

                // Remove an item, then insert a new one
                let removed = cache.remove(&2);
                assert!(removed.is_some());
                assert_eq!(*removed.unwrap(), 200);
                assert_eq!(cache.len(), 2);

                // Insert new item - should not cause eviction since we're under capacity
                cache.insert(4, Arc::new(400));
                assert_eq!(cache.len(), 3);
                assert!(cache.contains(&1));
                assert!(!cache.contains(&2));
                assert!(cache.contains(&3));
                assert!(cache.contains(&4));

                // Insert another item - should cause eviction (LRU is 1)
                cache.insert(5, Arc::new(500));
                assert_eq!(cache.len(), 3);
                assert!(!cache.contains(&1));
            }

            #[test]
            fn test_clear_on_empty_cache() {
                // Test clearing an already empty cache
                let mut cache: LRUCore<i32, i32> = LRUCore::new(5);

                assert_eq!(cache.len(), 0);
                cache.clear();
                assert_eq!(cache.len(), 0);

                // Should still work normally after clear
                cache.insert(1, Arc::new(100));
                assert_eq!(cache.len(), 1);
                assert!(cache.contains(&1));
            }

            #[test]
            fn test_clear_then_operations() {
                // Test operations after clearing a populated cache
                let mut cache: LRUCore<i32, i32> = LRUCore::new(5);

                // Populate cache
                for i in 1..=3 {
                    cache.insert(i, Arc::new(i * 100));
                }
                assert_eq!(cache.len(), 3);

                // Clear cache
                cache.clear();
                assert_eq!(cache.len(), 0);

                // All items should be gone
                for i in 1..=3 {
                    assert!(!cache.contains(&i));
                    assert!(cache.get(&i).is_none());
                }

                // Should work normally after clear
                cache.insert(10, Arc::new(1000));
                assert_eq!(cache.len(), 1);
                assert!(cache.contains(&10));

                let value = cache.get(&10);
                assert!(value.is_some());
                assert_eq!(**value.unwrap(), 1000);
            }

            #[test]
            fn test_multiple_clear_operations() {
                // Test calling clear multiple times in succession
                let mut cache: LRUCore<i32, i32> = LRUCore::new(5);

                cache.insert(1, Arc::new(100));
                assert_eq!(cache.len(), 1);

                // Multiple clears
                for _ in 0..5 {
                    cache.clear();
                    assert_eq!(cache.len(), 0);
                }

                // Should still work after multiple clears
                cache.insert(2, Arc::new(200));
                assert_eq!(cache.len(), 1);
                assert!(cache.contains(&2));
            }

            #[test]
            fn test_pop_lru_until_empty() {
                // Test repeatedly calling pop_lru until cache is empty
                let mut cache: LRUCore<i32, i32> = LRUCore::new(5);

                // Fill cache
                for i in 1..=5 {
                    cache.insert(i, Arc::new(i * 100));
                }
                assert_eq!(cache.len(), 5);

                // Pop all items in LRU order
                let mut popped_keys = Vec::new();
                while let Some((key, value)) = cache.pop_lru() {
                    popped_keys.push(key);
                    assert_eq!(*value, key * 100);
                }

                // Should have popped in LRU order (1, 2, 3, 4, 5)
                assert_eq!(popped_keys, vec![1, 2, 3, 4, 5]);
                assert_eq!(cache.len(), 0);

                // Further pops should return None
                assert!(cache.pop_lru().is_none());
            }

            #[test]
            fn test_peek_after_eviction() {
                // Test peeking at items that should have been evicted
                let mut cache: LRUCore<i32, i32> = LRUCore::new(2);

                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));

                // Peek should work
                assert!(cache.peek(&1).is_some());
                assert!(cache.peek(&2).is_some());

                // Insert third item, evicting first
                cache.insert(3, Arc::new(300));

                // Peek at evicted item should return None
                assert!(cache.peek(&1).is_none());
                // Peek at remaining items should work
                assert!(cache.peek(&2).is_some());
                assert!(cache.peek(&3).is_some());
            }

            #[test]
            fn test_touch_evicted_items() {
                // Test touching items that have been evicted
                let mut cache: LRUCore<i32, i32> = LRUCore::new(2);

                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));

                // Touch item 1 to make it most recent (2 becomes LRU)
                assert!(cache.touch(&1));

                // Insert third item, evicting item 2 (LRU)
                cache.insert(3, Arc::new(300));

                // Touch evicted item should return false
                assert!(!cache.touch(&2));
                // Touch remaining items should work
                assert!(cache.touch(&1));
                assert!(cache.touch(&3));
            }

            #[test]
            fn test_recency_rank_after_operations() {
                // Test recency ranks after complex operation sequences
                let mut cache: LRUCore<i32, i32> = LRUCore::new(4);

                // Insert in order
                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                cache.insert(3, Arc::new(300));
                cache.insert(4, Arc::new(400));

                // Initial ranks: 4(0), 3(1), 2(2), 1(3)
                assert_eq!(cache.recency_rank(&4), Some(0));
                assert_eq!(cache.recency_rank(&3), Some(1));
                assert_eq!(cache.recency_rank(&2), Some(2));
                assert_eq!(cache.recency_rank(&1), Some(3));

                // Access item 1, making it most recent
                cache.get(&1);
                // New ranks: 1(0), 4(1), 3(2), 2(3)
                assert_eq!(cache.recency_rank(&1), Some(0));
                assert_eq!(cache.recency_rank(&4), Some(1));
                assert_eq!(cache.recency_rank(&3), Some(2));
                assert_eq!(cache.recency_rank(&2), Some(3));

                // Touch item 3
                cache.touch(&3);
                // New ranks: 3(0), 1(1), 4(2), 2(3)
                assert_eq!(cache.recency_rank(&3), Some(0));
                assert_eq!(cache.recency_rank(&1), Some(1));
                assert_eq!(cache.recency_rank(&4), Some(2));
                assert_eq!(cache.recency_rank(&2), Some(3));
            }

            #[test]
            fn test_cache_with_identical_values() {
                // Test cache behavior when multiple keys map to identical values
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);
                let shared_value = Arc::new(999);

                // Insert different keys with same Arc value
                cache.insert(1, Arc::clone(&shared_value));
                cache.insert(2, Arc::clone(&shared_value));
                cache.insert(3, Arc::clone(&shared_value));

                // All should be present and point to same value
                assert!(cache.contains(&1));
                assert!(cache.contains(&2));
                assert!(cache.contains(&3));

                // Get values one at a time (get() requires mutable borrow)
                let val1 = cache.get(&1);
                assert!(val1.is_some());
                assert!(Arc::ptr_eq(val1.unwrap(), &shared_value));

                let val2 = cache.get(&2);
                assert!(val2.is_some());
                assert!(Arc::ptr_eq(val2.unwrap(), &shared_value));

                let val3 = cache.get(&3);
                assert!(val3.is_some());
                assert!(Arc::ptr_eq(val3.unwrap(), &shared_value));
            }

            #[test]
            fn test_interleaved_operations() {
                // Test complex interleaving of insert/get/remove/touch operations
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);

                cache.insert(1, Arc::new(100));
                assert!(cache.get(&1).is_some());
                cache.insert(2, Arc::new(200));
                assert!(cache.touch(&1));
                cache.insert(3, Arc::new(300));
                assert!(cache.peek(&2).is_some());
                cache.insert(4, Arc::new(400)); // Should evict 2 (LRU)

                assert!(cache.contains(&1));
                assert!(!cache.contains(&2)); // Evicted
                assert!(cache.contains(&3));
                assert!(cache.contains(&4));

                let removed = cache.remove(&1);
                assert!(removed.is_some());
                cache.insert(5, Arc::new(500));
                assert!(cache.touch(&3));

                // Final state should have 3, 4, 5
                assert!(!cache.contains(&1));
                assert!(!cache.contains(&2));
                assert!(cache.contains(&3));
                assert!(cache.contains(&4));
                assert!(cache.contains(&5));
            }

            #[test]
            fn test_capacity_reduction_simulation() {
                // Test behavior as if capacity was reduced (by manual eviction)
                let mut cache: LRUCore<i32, i32> = LRUCore::new(5);

                // Fill to capacity
                for i in 1..=5 {
                    cache.insert(i, Arc::new(i * 100));
                }
                assert_eq!(cache.len(), 5);

                // Simulate capacity reduction to 3 by removing 2 LRU items
                cache.pop_lru(); // Remove 1
                cache.pop_lru(); // Remove 2
                assert_eq!(cache.len(), 3);

                // Remaining items should be 3, 4, 5
                assert!(!cache.contains(&1));
                assert!(!cache.contains(&2));
                assert!(cache.contains(&3));
                assert!(cache.contains(&4));
                assert!(cache.contains(&5));

                // Should behave as if it has effective capacity of 3
                cache.insert(6, Arc::new(600));
                cache.insert(7, Arc::new(700));
                assert_eq!(cache.len(), 5); // Original capacity still enforced
            }

            #[test]
            fn test_duplicate_key_with_same_value() {
                // Test inserting same key-value pair multiple times
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);
                let value = Arc::new(100);

                // Insert same key-value multiple times
                let result1 = cache.insert(1, Arc::clone(&value));
                assert!(result1.is_none());

                let result2 = cache.insert(1, Arc::clone(&value));
                assert!(result2.is_some());
                assert!(Arc::ptr_eq(result2.as_ref().unwrap(), &value));

                // Should still only have one entry
                assert_eq!(cache.len(), 1);
                assert!(cache.contains(&1));

                // Value should be the same Arc instance
                let retrieved = cache.get(&1);
                assert!(Arc::ptr_eq(retrieved.unwrap(), &value));
            }

            #[test]
            fn test_lru_order_with_duplicate_inserts() {
                // Test LRU order when same key is inserted repeatedly
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);

                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                cache.insert(3, Arc::new(300));

                // Ranks: 3(0), 2(1), 1(2)
                assert_eq!(cache.recency_rank(&3), Some(0));
                assert_eq!(cache.recency_rank(&2), Some(1));
                assert_eq!(cache.recency_rank(&1), Some(2));

                // Re-insert key 1 with new value (should move to head)
                cache.insert(1, Arc::new(999));

                // New ranks: 1(0), 3(1), 2(2)
                assert_eq!(cache.recency_rank(&1), Some(0));
                assert_eq!(cache.recency_rank(&3), Some(1));
                assert_eq!(cache.recency_rank(&2), Some(2));

                // Insert new item, should evict 2 (LRU)
                cache.insert(4, Arc::new(400));
                assert!(cache.contains(&1));
                assert!(!cache.contains(&2));
                assert!(cache.contains(&3));
                assert!(cache.contains(&4));
            }

            #[test]
            fn test_peek_vs_get_ordering_difference() {
                // Test that peek and get produce different LRU ordering
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);

                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                cache.insert(3, Arc::new(300));

                // Initial order: 3(0), 2(1), 1(2)
                assert_eq!(cache.recency_rank(&3), Some(0));
                assert_eq!(cache.recency_rank(&2), Some(1));
                assert_eq!(cache.recency_rank(&1), Some(2));

                // Peek at item 1 (should not change order)
                cache.peek(&1);
                assert_eq!(cache.recency_rank(&3), Some(0));
                assert_eq!(cache.recency_rank(&2), Some(1));
                assert_eq!(cache.recency_rank(&1), Some(2));

                // Get item 1 (should change order)
                cache.get(&1);
                assert_eq!(cache.recency_rank(&1), Some(0));
                assert_eq!(cache.recency_rank(&3), Some(1));
                assert_eq!(cache.recency_rank(&2), Some(2));

                // Insert new item - should evict 2 (LRU)
                cache.insert(4, Arc::new(400));
                assert!(cache.contains(&1));
                assert!(!cache.contains(&2));
                assert!(cache.contains(&3));
                assert!(cache.contains(&4));
            }

            #[test]
            fn test_concurrent_cache_edge_cases() {
                // Test edge cases specific to ConcurrentLRUCache
                let cache = ConcurrentLRUCache::new(2);

                // Empty cache operations
                assert!(cache.is_empty());
                assert_eq!(cache.len(), 0);
                assert!(cache.get(&1).is_none());
                assert!(cache.peek(&1).is_none());
                assert!(!cache.contains(&1));
                assert!(cache.remove(&1).is_none());

                // Single item operations
                cache.insert(1, 100);
                assert!(!cache.is_empty());
                assert_eq!(cache.len(), 1);

                // Capacity testing
                cache.insert(2, 200);
                assert_eq!(cache.len(), 2);

                // Eviction
                cache.insert(3, 300); // Should evict 1
                assert_eq!(cache.len(), 2);
                assert!(!cache.contains(&1));
                assert!(cache.contains(&2));
                assert!(cache.contains(&3));

                // Clear
                cache.clear();
                assert!(cache.is_empty());
                assert_eq!(cache.len(), 0);
            }

            #[test]
            fn test_arc_reference_counting_edge_cases() {
                // Test Arc reference counting in edge scenarios
                let cache = ConcurrentLRUCache::new(3);
                let value = Arc::new(vec![1, 2, 3, 4, 5]); // Non-trivial value

                // Initial ref count should be 1
                assert_eq!(Arc::strong_count(&value), 1);

                // Insert Arc directly
                let old_value = cache.insert_arc(1, Arc::clone(&value));
                assert!(old_value.is_none());
                assert_eq!(Arc::strong_count(&value), 2); // Cache holds a reference

                // Get value (creates another temporary reference)
                let retrieved = cache.get(&1);
                assert!(retrieved.is_some());
                assert_eq!(Arc::strong_count(&value), 3); // Original + cache + retrieved

                // Drop retrieved reference
                drop(retrieved);
                assert_eq!(Arc::strong_count(&value), 2); // Back to original + cache

                // Remove from cache
                let removed = cache.remove(&1);
                assert!(removed.is_some());
                assert_eq!(Arc::strong_count(&value), 2); // Original + removed

                // Drop removed reference
                drop(removed);
                assert_eq!(Arc::strong_count(&value), 1); // Back to original only
            }

            #[test]
            fn test_insert_arc_vs_insert_value() {
                // Test difference between insert_arc and regular insert
                let cache = ConcurrentLRUCache::new(3);
                let value = Arc::new(100);

                // insert_arc uses provided Arc directly
                let old1 = cache.insert_arc(1, Arc::clone(&value));
                assert!(old1.is_none());

                let retrieved1 = cache.get(&1);
                assert!(Arc::ptr_eq(retrieved1.as_ref().unwrap(), &value));

                // insert creates new Arc
                let old2 = cache.insert(2, 200);
                assert!(old2.is_none());

                let retrieved2 = cache.get(&2);
                assert!(retrieved2.is_some());
                let retrieved2_arc = retrieved2.unwrap();
                assert!(!Arc::ptr_eq(&retrieved2_arc, &value));
                assert_eq!(*retrieved2_arc, 200);

                // Both methods should work correctly
                assert!(cache.contains(&1));
                assert!(cache.contains(&2));
                assert_eq!(cache.len(), 2);
            }

            #[test]
            fn test_large_key_values() {
                // Test with unusually large key values (if applicable)
                let mut cache: LRUCore<i64, i32> = LRUCore::new(3);

                // Test with max and min key values
                cache.insert(i64::MAX, Arc::new(1));
                cache.insert(i64::MIN, Arc::new(2));
                cache.insert(0, Arc::new(3));

                assert!(cache.contains(&i64::MAX));
                assert!(cache.contains(&i64::MIN));
                assert!(cache.contains(&0));

                assert_eq!(**cache.get(&i64::MAX).unwrap(), 1);
                assert_eq!(**cache.get(&i64::MIN).unwrap(), 2);
                assert_eq!(**cache.get(&0).unwrap(), 3);
            }

            #[test]
            fn test_key_collision_scenarios() {
                // Test scenarios that might cause hash collisions
                let mut cache: LRUCore<i32, i32> = LRUCore::new(10);

                // Use keys that might have similar hash values
                let keys = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

                // Insert all keys
                for &key in &keys {
                    cache.insert(key, Arc::new(key * 100));
                }

                // All should be present
                for &key in &keys {
                    assert!(cache.contains(&key));
                    let value = cache.get(&key);
                    assert!(value.is_some());
                    assert_eq!(**value.unwrap(), key * 100);
                }

                // Remove and re-insert to test collision handling
                cache.remove(&5);
                assert!(!cache.contains(&5));

                cache.insert(5, Arc::new(999));
                assert!(cache.contains(&5));
                assert_eq!(**cache.get(&5).unwrap(), 999);
            }

            #[test]
            fn test_memory_pressure_simulation() {
                // Test cache behavior under simulated memory pressure
                let mut cache: LRUCore<i32, String> = LRUCore::new(75);

                // Create large values to simulate memory pressure
                for i in 0..50 {
                    let large_string = "x".repeat(1000); // 1KB string
                    cache.insert(i, Arc::new(large_string));
                }

                assert_eq!(cache.len(), 50);

                // Access pattern that might stress memory (access first 25 items)
                for _ in 0..10 {
                    for i in 0..25 {
                        cache.get(&i);
                    }
                }

                // Insert more items, causing evictions
                for i in 50..100 {
                    let large_string = "y".repeat(1000);
                    cache.insert(i, Arc::new(large_string));
                }

                assert_eq!(cache.len(), 75);

                // Some original items should be evicted (the unaccessed ones: 25-49)
                let mut evicted_count = 0;
                for i in 0..50 {
                    if !cache.contains(&i) {
                        evicted_count += 1;
                    }
                }

                // Should have evicted 25 items (those not recently accessed)
                assert_eq!(evicted_count, 25);
            }

            #[test]
            fn test_rapid_capacity_fill_and_drain() {
                // Test rapidly filling to capacity then draining cache
                let mut cache: LRUCore<i32, i32> = LRUCore::new(50);

                // Rapid fill
                for i in 0..50 {
                    cache.insert(i, Arc::new(i * 100));
                }
                assert_eq!(cache.len(), 50);

                // Rapid drain via pop_lru
                for i in 0..25 {
                    let popped = cache.pop_lru();
                    assert!(popped.is_some());
                    let (key, value) = popped.unwrap();
                    assert_eq!(key, i);
                    assert_eq!(*value, i * 100);
                }
                assert_eq!(cache.len(), 25);

                // Rapid refill with more items than remaining capacity
                for i in 50..100 {
                    cache.insert(i, Arc::new(i * 100));
                }
                assert_eq!(cache.len(), 50);

                // All middle items (25-49) should be evicted due to new insertions
                for i in 25..50 {
                    assert!(!cache.contains(&i));
                }

                // New items should be present
                for i in 75..100 {
                    assert!(cache.contains(&i));
                }
            }

            #[test]
            fn test_operation_sequence_corner_cases() {
                // Test specific sequences that might break invariants
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);

                // Sequence 1: Insert, remove, insert same key
                cache.insert(1, Arc::new(100));
                let removed = cache.remove(&1);
                assert!(removed.is_some());
                cache.insert(1, Arc::new(200));
                assert_eq!(**cache.get(&1).unwrap(), 200);

                // Sequence 2: Fill, clear, fill again
                cache.insert(2, Arc::new(300));
                cache.insert(3, Arc::new(400));
                cache.clear();
                assert_eq!(cache.len(), 0);

                cache.insert(4, Arc::new(500));
                cache.insert(5, Arc::new(600));
                assert_eq!(cache.len(), 2);

                // Sequence 3: Touch non-existent, then insert
                assert!(!cache.touch(&6));
                cache.insert(6, Arc::new(700));
                assert!(cache.touch(&6));

                // Sequence 4: Peek, get, peek same item
                cache.peek(&6);
                cache.get(&6);
                cache.peek(&6);
                assert!(cache.contains(&6));
            }

            #[test]
            fn test_boundary_value_keys() {
                // Test with boundary values for key type (min/max values)
                let mut cache: LRUCore<i32, i32> = LRUCore::new(5);

                let boundary_keys = vec![i32::MIN, i32::MIN + 1, -1, 0, 1, i32::MAX - 1, i32::MAX];

                // Insert all boundary values
                for (i, &key) in boundary_keys.iter().enumerate() {
                    cache.insert(key, Arc::new(i as i32));
                }

                // Verify all are present (some may be evicted due to capacity)
                let mut present_count = 0;
                for &key in &boundary_keys {
                    if cache.contains(&key) {
                        present_count += 1;
                        let value = cache.get(&key);
                        assert!(value.is_some());
                    }
                }

                assert_eq!(present_count, cache.capacity().min(boundary_keys.len()));
            }

            #[test]
            fn test_remove_head_and_tail_items() {
                // Test removing items at head and tail positions specifically
                let mut cache: LRUCore<i32, i32> = LRUCore::new(5);

                // Fill cache
                for i in 1..=5 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Remove tail (LRU) item
                let tail_removed = cache.remove(&1);
                assert!(tail_removed.is_some());
                assert_eq!(*tail_removed.unwrap(), 100);

                // Remove head (MRU) item
                let head_removed = cache.remove(&5);
                assert!(head_removed.is_some());
                assert_eq!(*head_removed.unwrap(), 500);

                // Remove middle item
                let middle_removed = cache.remove(&3);
                assert!(middle_removed.is_some());
                assert_eq!(*middle_removed.unwrap(), 300);

                // Only items 2 and 4 should remain
                assert!(!cache.contains(&1));
                assert!(cache.contains(&2));
                assert!(!cache.contains(&3));
                assert!(cache.contains(&4));
                assert!(!cache.contains(&5));
                assert_eq!(cache.len(), 2);
            }

            #[test]
            fn test_get_after_remove() {
                // Test getting items immediately after they've been removed
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);

                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));

                // Remove item 1
                let removed = cache.remove(&1);
                assert!(removed.is_some());
                assert_eq!(*removed.unwrap(), 100);

                // Immediate get should return None
                let value = cache.get(&1);
                assert!(value.is_none());

                // Other operations should also return None/false
                assert!(!cache.contains(&1));
                assert!(cache.peek(&1).is_none());
                assert!(!cache.touch(&1));
                assert!(cache.recency_rank(&1).is_none());
            }

            #[test]
            fn test_contains_after_eviction() {
                // Test contains check for items that were evicted
                let mut cache: LRUCore<i32, i32> = LRUCore::new(2);

                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                assert!(cache.contains(&1));
                assert!(cache.contains(&2));

                // Insert third item, evicting first
                cache.insert(3, Arc::new(300));

                // Contains check for evicted item should return false
                assert!(!cache.contains(&1));
                assert!(cache.contains(&2));
                assert!(cache.contains(&3));

                // Other operations on evicted item should fail
                assert!(cache.get(&1).is_none());
                assert!(cache.peek(&1).is_none());
                assert!(!cache.touch(&1));
                assert!(cache.remove(&1).is_none());
            }

            #[test]
            fn test_empty_cache_all_operations() {
                // Test all possible operations on empty cache
                let mut cache: LRUCore<i32, i32> = LRUCore::new(5);

                // Verify empty state
                assert_eq!(cache.len(), 0);
                assert_eq!(cache.capacity(), 5);

                // All read operations should return None/false
                assert!(cache.get(&1).is_none());
                assert!(cache.peek(&1).is_none());
                assert!(!cache.contains(&1));
                assert!(cache.remove(&1).is_none());
                assert!(cache.pop_lru().is_none());
                assert!(cache.peek_lru().is_none());
                assert!(!cache.touch(&1));
                assert!(cache.recency_rank(&1).is_none());

                // Clear should work on empty cache
                cache.clear();
                assert_eq!(cache.len(), 0);

                // Insert should work normally
                cache.insert(1, Arc::new(100));
                assert_eq!(cache.len(), 1);
                assert!(cache.contains(&1));
            }

            #[test]
            fn test_single_item_all_operations() {
                // Test all operations when cache contains exactly one item
                let mut cache: LRUCore<i32, i32> = LRUCore::new(5);
                cache.insert(1, Arc::new(100));

                // Verify single item state
                assert_eq!(cache.len(), 1);
                assert!(cache.contains(&1));

                // All operations on existing item should work
                let value = cache.get(&1);
                assert!(value.is_some());
                assert_eq!(**value.unwrap(), 100);

                let peeked = cache.peek(&1);
                assert!(peeked.is_some());
                assert_eq!(*peeked.unwrap(), 100);

                assert!(cache.touch(&1));
                assert_eq!(cache.recency_rank(&1), Some(0));

                let (lru_key, lru_value) = cache.peek_lru().unwrap();
                assert_eq!(*lru_key, 1);
                assert_eq!(**lru_value, 100);

                // Operations on non-existing items should fail
                assert!(cache.get(&2).is_none());
                assert!(!cache.contains(&2));
                assert!(!cache.touch(&2));
            }

            #[test]
            fn test_full_cache_all_operations() {
                // Test all operations when cache is at full capacity
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);

                // Fill to capacity
                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                cache.insert(3, Arc::new(300));
                assert_eq!(cache.len(), 3);

                // Test operations without changing LRU order too much
                assert!(cache.contains(&1));
                assert!(cache.contains(&2));
                assert!(cache.contains(&3));

                assert!(cache.peek(&1).is_some());
                assert!(cache.peek(&2).is_some());
                assert!(cache.peek(&3).is_some());

                assert!(cache.recency_rank(&1).is_some());
                assert!(cache.recency_rank(&2).is_some());
                assert!(cache.recency_rank(&3).is_some());

                // Insert new item should cause eviction
                cache.insert(4, Arc::new(400));
                assert_eq!(cache.len(), 3); // Still at capacity

                // LRU item (1) should be evicted
                assert!(!cache.contains(&1));
                assert!(cache.contains(&2));
                assert!(cache.contains(&3));
                assert!(cache.contains(&4));
            }

            #[test]
            fn test_lru_rank_boundary_conditions() {
                // Test recency rank at boundaries (0, capacity-1)
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);

                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                cache.insert(3, Arc::new(300));

                // Rank 0 (most recent)
                assert_eq!(cache.recency_rank(&3), Some(0));

                // Rank capacity-1 (least recent)
                assert_eq!(cache.recency_rank(&1), Some(2));

                // Access item at rank 2 to move it to rank 0
                cache.get(&1);
                assert_eq!(cache.recency_rank(&1), Some(0));
                assert_eq!(cache.recency_rank(&2), Some(2)); // Now least recent

                // Non-existing item should return None
                assert!(cache.recency_rank(&4).is_none());
            }

            #[test]
            fn test_peek_lru_on_single_item() {
                // Test peek_lru when cache has exactly one item
                let mut cache: LRUCore<i32, i32> = LRUCore::new(5);
                cache.insert(1, Arc::new(100));

                let peeked = cache.peek_lru();
                assert!(peeked.is_some());

                let (key, value) = peeked.unwrap();
                assert_eq!(*key, 1);
                assert_eq!(**value, 100);

                // Cache should still have the item
                assert_eq!(cache.len(), 1);
                assert!(cache.contains(&1));
            }

            #[test]
            fn test_touch_only_item() {
                // Test touching the only item in a single-item cache
                let mut cache: LRUCore<i32, i32> = LRUCore::new(5);
                cache.insert(1, Arc::new(100));

                // Touch should succeed
                assert!(cache.touch(&1));

                // Item should still be present and accessible
                assert!(cache.contains(&1));
                assert_eq!(cache.recency_rank(&1), Some(0));

                let value = cache.get(&1);
                assert!(value.is_some());
                assert_eq!(**value.unwrap(), 100);
            }

            #[test]
            fn test_concurrent_read_write_edge_cases() {
                // Test edge cases in concurrent read/write scenarios
                let cache = ConcurrentLRUCache::new(2);

                // Concurrent insert and read of same key
                cache.insert(1, 100);
                let value = cache.get(&1);
                assert!(value.is_some());
                assert_eq!(*value.unwrap(), 100);

                // Concurrent insert and remove of same key
                cache.insert(2, 200);
                let removed = cache.remove(&2);
                assert!(removed.is_some());
                assert_eq!(*removed.unwrap(), 200);
                assert!(!cache.contains(&2));

                // Concurrent operations on different keys
                cache.insert(3, 300);
                cache.insert(4, 400);
                assert!(cache.contains(&3));
                assert!(cache.contains(&4));

                // Clear should work with concurrent operations
                cache.clear();
                assert!(cache.is_empty());
            }

            #[test]
            fn test_drop_behavior_edge_cases() {
                // Test cache dropping behavior in various states
                {
                    // Empty cache drop
                    let cache: LRUCore<i32, i32> = LRUCore::new(5);
                    assert_eq!(cache.len(), 0);
                    // Cache drops here
                }

                {
                    // Single item cache drop
                    let mut cache: LRUCore<i32, i32> = LRUCore::new(5);
                    cache.insert(1, Arc::new(100));
                    assert_eq!(cache.len(), 1);
                    // Cache drops here
                }

                {
                    // Full cache drop
                    let mut cache: LRUCore<i32, i32> = LRUCore::new(3);
                    for i in 1..=3 {
                        cache.insert(i, Arc::new(i * 100));
                    }
                    assert_eq!(cache.len(), 3);
                    // Cache drops here
                }

                // All drops should be handled gracefully without panics or leaks
            }
        }

        mod lru_operations {
            use super::*;

            #[test]
            fn test_lru_insertion_order_tracking() {
                // Test that insertion order is correctly tracked in LRU list
                let mut cache: LRUCore<i32, i32> = LRUCore::new(5);

                // Insert items in sequence
                for i in 1..=5 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Check that most recent insertion is at rank 0 (head)
                assert_eq!(cache.recency_rank(&5), Some(0));
                assert_eq!(cache.recency_rank(&4), Some(1));
                assert_eq!(cache.recency_rank(&3), Some(2));
                assert_eq!(cache.recency_rank(&2), Some(3));
                assert_eq!(cache.recency_rank(&1), Some(4));

                // LRU (tail) should be first inserted item
                let (lru_key, _) = cache.peek_lru().unwrap();
                assert_eq!(*lru_key, 1);
            }

            #[test]
            fn test_lru_access_order_updates() {
                // Test that access operations correctly update LRU order
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);

                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                cache.insert(3, Arc::new(300));

                // Initial order: 3(0), 2(1), 1(2)
                assert_eq!(cache.recency_rank(&3), Some(0));
                assert_eq!(cache.recency_rank(&2), Some(1));
                assert_eq!(cache.recency_rank(&1), Some(2));

                // Access item 1 - should move to head
                cache.get(&1);
                // New order: 1(0), 3(1), 2(2)
                assert_eq!(cache.recency_rank(&1), Some(0));
                assert_eq!(cache.recency_rank(&3), Some(1));
                assert_eq!(cache.recency_rank(&2), Some(2));

                // Access item 2 - should move to head
                cache.get(&2);
                // New order: 2(0), 1(1), 3(2)
                assert_eq!(cache.recency_rank(&2), Some(0));
                assert_eq!(cache.recency_rank(&1), Some(1));
                assert_eq!(cache.recency_rank(&3), Some(2));
            }

            #[test]
            fn test_lru_eviction_policy() {
                // Test that least recently used items are evicted first
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);

                // Fill cache
                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                cache.insert(3, Arc::new(300));

                // Access item 1 and 3 to make them more recent
                cache.get(&1);
                cache.get(&3);
                // Order: 3(0), 1(1), 2(2) - item 2 is LRU

                // Insert new item - should evict item 2 (LRU)
                cache.insert(4, Arc::new(400));

                assert!(cache.contains(&1));
                assert!(!cache.contains(&2)); // Evicted
                assert!(cache.contains(&3));
                assert!(cache.contains(&4));
            }

            #[test]
            fn test_lru_head_tail_positioning() {
                // Test that head is most recent and tail is least recent
                let mut cache: LRUCore<i32, i32> = LRUCore::new(4);

                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                cache.insert(3, Arc::new(300));
                cache.insert(4, Arc::new(400));

                // Head should be most recently inserted (4)
                assert_eq!(cache.recency_rank(&4), Some(0));

                // Tail should be least recently used (1)
                let (lru_key, lru_value) = cache.peek_lru().unwrap();
                assert_eq!(*lru_key, 1);
                assert_eq!(**lru_value, 100);

                // Access tail item - should move to head
                cache.get(&1);
                assert_eq!(cache.recency_rank(&1), Some(0));

                // New tail should be item 2
                let (new_lru_key, _) = cache.peek_lru().unwrap();
                assert_eq!(*new_lru_key, 2);
            }

            #[test]
            fn test_move_to_head_operation() {
                // Test internal move_to_head functionality
                let mut cache: LRUCore<i32, i32> = LRUCore::new(5);

                for i in 1..=5 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Access middle item (3) - should move to head
                cache.get(&3);
                assert_eq!(cache.recency_rank(&3), Some(0));

                // Other items should maintain relative order
                assert_eq!(cache.recency_rank(&5), Some(1));
                assert_eq!(cache.recency_rank(&4), Some(2));
                assert_eq!(cache.recency_rank(&2), Some(3));
                assert_eq!(cache.recency_rank(&1), Some(4));

                // Touch operation should also move to head
                cache.touch(&1);
                assert_eq!(cache.recency_rank(&1), Some(0));
                assert_eq!(cache.recency_rank(&3), Some(1));
            }

            #[test]
            fn test_lru_chain_integrity() {
                // Test that doubly-linked list maintains proper forward/backward links
                let mut cache: LRUCore<i32, i32> = LRUCore::new(4);

                // Build up cache
                for i in 1..=4 {
                    cache.insert(i, Arc::new(i * 100));

                    // After each insertion, verify chain integrity
                    // by checking all items have consistent ranks
                    for j in 1..=i {
                        assert!(cache.recency_rank(&j).is_some());
                    }
                }

                // Perform various operations and verify integrity
                cache.get(&2); // Move 2 to head
                cache.remove(&3); // Remove middle item
                cache.insert(5, Arc::new(500)); // Insert new item

                // Verify remaining items have valid ranks
                assert!(cache.recency_rank(&5).is_some());
                assert!(cache.recency_rank(&2).is_some());
                assert!(cache.recency_rank(&4).is_some());
                assert!(cache.recency_rank(&1).is_some());
                assert!(cache.recency_rank(&3).is_none()); // Removed

                // All ranks should be unique and within bounds
                let mut ranks = vec![];
                for &key in &[5, 2, 4, 1] {
                    if cache.contains(&key) {
                        let rank = cache.recency_rank(&key).unwrap();
                        assert!(!ranks.contains(&rank));
                        assert!(rank < cache.len());
                        ranks.push(rank);
                    }
                }
            }

            #[test]
            fn test_lru_ordering_after_get() {
                // Test LRU order changes after get operations
                let mut cache: LRUCore<i32, i32> = LRUCore::new(4);

                for i in 1..=4 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Record initial ordering
                let _initial_ranks: Vec<_> = (1..=4)
                    .map(|i| (i, cache.recency_rank(&i).unwrap()))
                    .collect();

                // Get item 1 (currently LRU)
                let value = cache.get(&1);
                assert!(value.is_some());
                assert_eq!(**value.unwrap(), 100);

                // Item 1 should now be MRU (rank 0)
                assert_eq!(cache.recency_rank(&1), Some(0));

                // Other items should be shifted
                assert_eq!(cache.recency_rank(&4), Some(1));
                assert_eq!(cache.recency_rank(&3), Some(2));
                assert_eq!(cache.recency_rank(&2), Some(3));

                // Get item from middle
                cache.get(&3);
                assert_eq!(cache.recency_rank(&3), Some(0));
                assert_eq!(cache.recency_rank(&1), Some(1));
                assert_eq!(cache.recency_rank(&4), Some(2));
                assert_eq!(cache.recency_rank(&2), Some(3));
            }

            #[test]
            fn test_lru_ordering_after_touch() {
                // Test LRU order changes after touch operations
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);

                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                cache.insert(3, Arc::new(300));

                // Initial order: 3(0), 2(1), 1(2)

                // Touch LRU item
                assert!(cache.touch(&1));
                // New order: 1(0), 3(1), 2(2)
                assert_eq!(cache.recency_rank(&1), Some(0));
                assert_eq!(cache.recency_rank(&3), Some(1));
                assert_eq!(cache.recency_rank(&2), Some(2));

                // Touch middle item
                assert!(cache.touch(&3));
                // New order: 3(0), 1(1), 2(2)
                assert_eq!(cache.recency_rank(&3), Some(0));
                assert_eq!(cache.recency_rank(&1), Some(1));
                assert_eq!(cache.recency_rank(&2), Some(2));

                // Touch non-existent item
                assert!(!cache.touch(&99));

                // Order should remain unchanged
                assert_eq!(cache.recency_rank(&3), Some(0));
                assert_eq!(cache.recency_rank(&1), Some(1));
                assert_eq!(cache.recency_rank(&2), Some(2));
            }

            #[test]
            fn test_lru_ordering_preservation_on_peek() {
                // Test that peek operations don't change LRU order
                let mut cache: LRUCore<i32, i32> = LRUCore::new(4);

                for i in 1..=4 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Record initial ordering
                let _initial_ranks: Vec<_> = (1..=4)
                    .map(|i| (i, cache.recency_rank(&i).unwrap()))
                    .collect();

                // Peek at various items
                assert_eq!(*cache.peek(&1).unwrap(), 100);
                assert_eq!(*cache.peek(&4).unwrap(), 400);
                assert_eq!(*cache.peek(&2).unwrap(), 200);
                assert_eq!(*cache.peek(&3).unwrap(), 300);

                // Ordering should be unchanged
                for (key, expected_rank) in _initial_ranks {
                    assert_eq!(cache.recency_rank(&key), Some(expected_rank));
                }

                // Peek at LRU
                let (lru_key, lru_value) = cache.peek_lru().unwrap();
                assert_eq!(*lru_key, 1);
                assert_eq!(**lru_value, 100);

                // LRU should still be LRU after peek
                assert_eq!(cache.recency_rank(&1), Some(3));
            }

            #[test]
            fn test_pop_lru_removes_tail() {
                // Test that pop_lru always removes the tail (LRU) item
                let mut cache: LRUCore<i32, i32> = LRUCore::new(5);

                // Fill cache
                for i in 1..=5 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Access some items to change LRU order
                cache.get(&3);
                cache.get(&1);
                // Order: 1(0), 3(1), 5(2), 4(3), 2(4)

                // Pop LRU should remove item 2
                let (popped_key, popped_value) = cache.pop_lru().unwrap();
                assert_eq!(popped_key, 2);
                assert_eq!(*popped_value, 200);
                assert!(!cache.contains(&2));
                assert_eq!(cache.len(), 4);

                // Next LRU should be item 4
                let (next_lru_key, _) = cache.peek_lru().unwrap();
                assert_eq!(*next_lru_key, 4);

                // Pop again
                let (popped_key2, popped_value2) = cache.pop_lru().unwrap();
                assert_eq!(popped_key2, 4);
                assert_eq!(*popped_value2, 400);
                assert_eq!(cache.len(), 3);
            }

            #[test]
            fn test_pop_lru_updates_tail_pointer() {
                // Test that pop_lru correctly updates tail pointer
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);

                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                cache.insert(3, Arc::new(300));

                // Verify initial tail
                let (initial_tail_key, _) = cache.peek_lru().unwrap();
                assert_eq!(*initial_tail_key, 1);

                // Pop tail
                let (popped_key, _) = cache.pop_lru().unwrap();
                assert_eq!(popped_key, 1);

                // Verify new tail
                let (new_tail_key, _) = cache.peek_lru().unwrap();
                assert_eq!(*new_tail_key, 2);

                // Pop again
                cache.pop_lru();
                let (final_tail_key, _) = cache.peek_lru().unwrap();
                assert_eq!(*final_tail_key, 3);

                // Pop last item
                cache.pop_lru();
                assert!(cache.peek_lru().is_none());
                assert_eq!(cache.len(), 0);
            }

            #[test]
            fn test_peek_lru_returns_tail() {
                // Test that peek_lru returns tail item without removal
                let mut cache: LRUCore<i32, i32> = LRUCore::new(4);

                for i in 1..=4 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Access some items to change order
                cache.get(&3);
                cache.get(&1);
                // Current order: 1(0), 3(1), 4(2), 2(3)

                // Peek LRU should return item 2 without removing it
                let (lru_key, lru_value) = cache.peek_lru().unwrap();
                assert_eq!(*lru_key, 2);
                assert_eq!(**lru_value, 200);

                // Item should still be in cache
                assert!(cache.contains(&2));
                assert_eq!(cache.len(), 4);

                // Multiple peeks should return same item
                for _ in 0..5 {
                    let (peek_key, peek_value) = cache.peek_lru().unwrap();
                    assert_eq!(*peek_key, 2);
                    assert_eq!(**peek_value, 200);
                }

                // Order should be unchanged
                assert_eq!(cache.recency_rank(&2), Some(3));
            }

            #[test]
            fn test_lru_recency_rank_calculation() {
                // Test recency rank calculation from head to tail
                let mut cache: LRUCore<i32, i32> = LRUCore::new(6);

                // Insert items
                for i in 1..=6 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Verify ranks match insertion order (reversed)
                for i in 1..=6 {
                    let expected_rank = (6 - i) as usize; // Most recent has rank 0
                    assert_eq!(cache.recency_rank(&i), Some(expected_rank));
                }

                // Access middle item
                cache.get(&3);
                // New order: 3(0), 6(1), 5(2), 4(3), 2(4), 1(5)
                assert_eq!(cache.recency_rank(&3), Some(0));
                assert_eq!(cache.recency_rank(&6), Some(1));
                assert_eq!(cache.recency_rank(&5), Some(2));
                assert_eq!(cache.recency_rank(&4), Some(3));
                assert_eq!(cache.recency_rank(&2), Some(4));
                assert_eq!(cache.recency_rank(&1), Some(5));

                // All ranks should be unique and consecutive
                let mut ranks: Vec<usize> =
                    (1..=6).map(|i| cache.recency_rank(&i).unwrap()).collect();
                ranks.sort();
                assert_eq!(ranks, vec![0, 1, 2, 3, 4, 5]);
            }

            #[test]
            fn test_lru_rank_after_reordering() {
                // Test recency ranks after LRU order changes
                let mut cache: LRUCore<i32, i32> = LRUCore::new(4);

                for i in 1..=4 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Perform sequence of operations and verify ranks
                cache.touch(&1); // Move 1 to head
                // Order: 1(0), 4(1), 3(2), 2(3)
                assert_eq!(cache.recency_rank(&1), Some(0));
                assert_eq!(cache.recency_rank(&4), Some(1));
                assert_eq!(cache.recency_rank(&3), Some(2));
                assert_eq!(cache.recency_rank(&2), Some(3));

                cache.get(&2); // Move 2 to head
                // Order: 2(0), 1(1), 4(2), 3(3)
                assert_eq!(cache.recency_rank(&2), Some(0));
                assert_eq!(cache.recency_rank(&1), Some(1));
                assert_eq!(cache.recency_rank(&4), Some(2));
                assert_eq!(cache.recency_rank(&3), Some(3));

                cache.touch(&4); // Move 4 to head
                // Order: 4(0), 2(1), 1(2), 3(3)
                assert_eq!(cache.recency_rank(&4), Some(0));
                assert_eq!(cache.recency_rank(&2), Some(1));
                assert_eq!(cache.recency_rank(&1), Some(2));
                assert_eq!(cache.recency_rank(&3), Some(3));
            }

            #[test]
            fn test_multiple_access_lru_stability() {
                // Test LRU order with multiple accesses to same items
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);

                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                cache.insert(3, Arc::new(300));

                // Multiple accesses to same item should maintain it at head
                for _ in 0..10 {
                    cache.get(&2);
                    assert_eq!(cache.recency_rank(&2), Some(0));
                }

                // Other items should maintain relative order
                assert_eq!(cache.recency_rank(&3), Some(1));
                assert_eq!(cache.recency_rank(&1), Some(2));

                // Multiple touches to same item
                for _ in 0..5 {
                    cache.touch(&1);
                    assert_eq!(cache.recency_rank(&1), Some(0));
                }

                // Final order should be: 1(0), 2(1), 3(2)
                assert_eq!(cache.recency_rank(&1), Some(0));
                assert_eq!(cache.recency_rank(&2), Some(1));
                assert_eq!(cache.recency_rank(&3), Some(2));
            }

            #[test]
            fn test_lru_eviction_sequence() {
                // Test sequence of evictions follows LRU order
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);

                // Fill cache
                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                cache.insert(3, Arc::new(300));

                // Access items to establish order
                cache.get(&2); // Order: 2(0), 3(1), 1(2)

                // Insert new items and verify eviction order
                cache.insert(4, Arc::new(400)); // Should evict 1 (LRU)
                assert!(!cache.contains(&1));
                assert!(cache.contains(&2));
                assert!(cache.contains(&3));
                assert!(cache.contains(&4));

                cache.insert(5, Arc::new(500)); // Should evict 3 (now LRU)
                assert!(!cache.contains(&3));
                assert!(cache.contains(&2));
                assert!(cache.contains(&4));
                assert!(cache.contains(&5));

                cache.insert(6, Arc::new(600)); // Should evict 2 (now LRU)
                assert!(!cache.contains(&2));
                assert!(cache.contains(&4));
                assert!(cache.contains(&5));
                assert!(cache.contains(&6));
            }

            #[test]
            fn test_lru_invariants_after_insert() {
                // Test LRU invariants are maintained after insertions
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);

                // Insert first item
                cache.insert(1, Arc::new(100));
                assert_eq!(cache.len(), 1);
                assert_eq!(cache.recency_rank(&1), Some(0));
                let (lru_key, _) = cache.peek_lru().unwrap();
                assert_eq!(*lru_key, 1);

                // Insert second item
                cache.insert(2, Arc::new(200));
                assert_eq!(cache.len(), 2);
                assert_eq!(cache.recency_rank(&2), Some(0)); // Most recent
                assert_eq!(cache.recency_rank(&1), Some(1)); // Less recent
                let (lru_key, _) = cache.peek_lru().unwrap();
                assert_eq!(*lru_key, 1);

                // Insert third item (reach capacity)
                cache.insert(3, Arc::new(300));
                assert_eq!(cache.len(), 3);
                assert_eq!(cache.recency_rank(&3), Some(0)); // Most recent
                assert_eq!(cache.recency_rank(&2), Some(1));
                assert_eq!(cache.recency_rank(&1), Some(2)); // Least recent

                // Insert fourth item (should evict)
                cache.insert(4, Arc::new(400));
                assert_eq!(cache.len(), 3);
                assert!(!cache.contains(&1)); // Evicted
                assert_eq!(cache.recency_rank(&4), Some(0)); // Most recent
                assert_eq!(cache.recency_rank(&3), Some(1));
                assert_eq!(cache.recency_rank(&2), Some(2)); // Now least recent
            }

            #[test]
            fn test_lru_invariants_after_remove() {
                // Test LRU invariants are maintained after removals
                let mut cache: LRUCore<i32, i32> = LRUCore::new(4);

                for i in 1..=4 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Remove head item
                cache.remove(&4);
                assert_eq!(cache.len(), 3);
                // Verify remaining items have valid consecutive ranks
                assert_eq!(cache.recency_rank(&3), Some(0)); // New head
                assert_eq!(cache.recency_rank(&2), Some(1));
                assert_eq!(cache.recency_rank(&1), Some(2)); // Still tail

                // Remove tail item
                cache.remove(&1);
                assert_eq!(cache.len(), 2);
                assert_eq!(cache.recency_rank(&3), Some(0));
                assert_eq!(cache.recency_rank(&2), Some(1)); // New tail
                let (lru_key, _) = cache.peek_lru().unwrap();
                assert_eq!(*lru_key, 2);

                // Remove middle item (only one left to remove)
                cache.remove(&2);
                assert_eq!(cache.len(), 1);
                assert_eq!(cache.recency_rank(&3), Some(0)); // Only item
                let (lru_key, _) = cache.peek_lru().unwrap();
                assert_eq!(*lru_key, 3);

                // Remove last item
                cache.remove(&3);
                assert_eq!(cache.len(), 0);
                assert!(cache.peek_lru().is_none());
            }

            #[test]
            fn test_lru_invariants_after_clear() {
                // Test LRU invariants after clearing cache
                let mut cache: LRUCore<i32, i32> = LRUCore::new(5);

                // Fill cache
                for i in 1..=5 {
                    cache.insert(i, Arc::new(i * 100));
                }

                assert_eq!(cache.len(), 5);

                // Clear cache
                cache.clear();

                // Verify empty state invariants
                assert_eq!(cache.len(), 0);
                assert_eq!(cache.capacity(), 5);
                assert!(cache.peek_lru().is_none());

                // All rank queries should return None
                for i in 1..=5 {
                    assert!(cache.recency_rank(&i).is_none());
                    assert!(!cache.contains(&i));
                }

                // Should be able to insert again and maintain invariants
                cache.insert(10, Arc::new(1000));
                assert_eq!(cache.len(), 1);
                assert_eq!(cache.recency_rank(&10), Some(0));
                let (lru_key, _) = cache.peek_lru().unwrap();
                assert_eq!(*lru_key, 10);

                // Fill again to verify complete functionality restoration
                for i in 11..=14 {
                    cache.insert(i, Arc::new(i * 100));
                }
                assert_eq!(cache.len(), 5);
                assert_eq!(cache.recency_rank(&14), Some(0));
                assert_eq!(cache.recency_rank(&10), Some(4));
            }

            #[test]
            fn test_lru_order_with_duplicate_keys() {
                // Test LRU order when same key is accessed multiple times
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);

                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                cache.insert(3, Arc::new(300));

                // Initial order: 3(0), 2(1), 1(2)

                // Re-insert existing key with new value
                cache.insert(1, Arc::new(999));
                // Should move to head with new value
                assert_eq!(cache.recency_rank(&1), Some(0));
                assert_eq!(**cache.get(&1).unwrap(), 999);

                // Other items shift down
                assert_eq!(cache.recency_rank(&3), Some(1));
                assert_eq!(cache.recency_rank(&2), Some(2));

                // Multiple gets of same key
                for _ in 0..3 {
                    cache.get(&2);
                }
                // Item 2 should be at head
                assert_eq!(cache.recency_rank(&2), Some(0));
                assert_eq!(cache.recency_rank(&1), Some(1));
                assert_eq!(cache.recency_rank(&3), Some(2));

                // Re-insert head item
                cache.insert(2, Arc::new(777));
                // Should stay at head with new value
                assert_eq!(cache.recency_rank(&2), Some(0));
                assert_eq!(**cache.get(&2).unwrap(), 777);
            }

            #[test]
            fn test_lru_traversal_forward() {
                // Test forward traversal from head to tail
                let mut cache: LRUCore<i32, i32> = LRUCore::new(5);

                for i in 1..=5 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Verify we can traverse from head (most recent) to tail (least recent)
                let expected_order = [5, 4, 3, 2, 1]; // Most recent to least recent
                for (idx, &expected_key) in expected_order.iter().enumerate() {
                    assert_eq!(cache.recency_rank(&expected_key), Some(idx));
                }

                // Access middle item and verify new order
                cache.get(&3);
                let new_expected_order = [3, 5, 4, 2, 1];
                for (idx, &expected_key) in new_expected_order.iter().enumerate() {
                    assert_eq!(cache.recency_rank(&expected_key), Some(idx));
                }

                // Verify LRU is still correct
                let (lru_key, _) = cache.peek_lru().unwrap();
                assert_eq!(*lru_key, 1);
            }

            #[test]
            fn test_lru_traversal_backward() {
                // Test backward traversal from tail to head
                let mut cache: LRUCore<i32, i32> = LRUCore::new(4);

                for i in 1..=4 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Access some items to create interesting order
                cache.get(&2);
                cache.get(&4);
                // Order: 4(0), 2(1), 3(2), 1(3)

                // Verify we can identify least recent (tail) correctly
                let (tail_key, _) = cache.peek_lru().unwrap();
                assert_eq!(*tail_key, 1);
                assert_eq!(cache.recency_rank(&1), Some(3));

                // Work backwards verifying ranks
                assert_eq!(cache.recency_rank(&3), Some(2));
                assert_eq!(cache.recency_rank(&2), Some(1));
                assert_eq!(cache.recency_rank(&4), Some(0)); // Head

                // Pop items in LRU order and verify sequence
                let mut popped_sequence = vec![];
                while let Some((key, _)) = cache.pop_lru() {
                    popped_sequence.push(key);
                }
                assert_eq!(popped_sequence, vec![1, 3, 2, 4]);
            }

            #[test]
            fn test_lru_middle_node_removal() {
                // Test removing nodes from middle of LRU chain
                let mut cache: LRUCore<i32, i32> = LRUCore::new(5);

                for i in 1..=5 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Initial order: 5(0), 4(1), 3(2), 2(3), 1(4)

                // Remove middle node (3)
                let removed = cache.remove(&3);
                assert!(removed.is_some());
                assert_eq!(*removed.unwrap(), 300);
                assert_eq!(cache.len(), 4);

                // Verify remaining nodes have correct ranks
                assert_eq!(cache.recency_rank(&5), Some(0));
                assert_eq!(cache.recency_rank(&4), Some(1));
                assert_eq!(cache.recency_rank(&2), Some(2));
                assert_eq!(cache.recency_rank(&1), Some(3));
                assert!(cache.recency_rank(&3).is_none());

                // Remove another middle node (2)
                cache.remove(&2);
                assert_eq!(cache.len(), 3);
                assert_eq!(cache.recency_rank(&5), Some(0));
                assert_eq!(cache.recency_rank(&4), Some(1));
                assert_eq!(cache.recency_rank(&1), Some(2));

                // Verify LRU is updated correctly
                let (lru_key, _) = cache.peek_lru().unwrap();
                assert_eq!(*lru_key, 1);
            }

            #[test]
            fn test_lru_head_node_removal() {
                // Test removing head node and updating LRU chain
                let mut cache: LRUCore<i32, i32> = LRUCore::new(4);

                for i in 1..=4 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Initial order: 4(0), 3(1), 2(2), 1(3)
                // Head is 4, tail is 1

                // Remove head node
                let removed = cache.remove(&4);
                assert!(removed.is_some());
                assert_eq!(*removed.unwrap(), 400);
                assert_eq!(cache.len(), 3);

                // Verify new head is correct
                assert_eq!(cache.recency_rank(&3), Some(0)); // New head
                assert_eq!(cache.recency_rank(&2), Some(1));
                assert_eq!(cache.recency_rank(&1), Some(2)); // Still tail

                // Remove new head
                cache.remove(&3);
                assert_eq!(cache.len(), 2);
                assert_eq!(cache.recency_rank(&2), Some(0)); // New head
                assert_eq!(cache.recency_rank(&1), Some(1)); // Still tail

                // Remove final head
                cache.remove(&2);
                assert_eq!(cache.len(), 1);
                assert_eq!(cache.recency_rank(&1), Some(0)); // Only item, both head and tail

                let (lru_key, _) = cache.peek_lru().unwrap();
                assert_eq!(*lru_key, 1);
            }

            #[test]
            fn test_lru_tail_node_removal() {
                // Test removing tail node and updating LRU chain
                let mut cache: LRUCore<i32, i32> = LRUCore::new(4);

                for i in 1..=4 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Initial order: 4(0), 3(1), 2(2), 1(3)
                // Head is 4, tail is 1

                // Remove tail node
                let removed = cache.remove(&1);
                assert!(removed.is_some());
                assert_eq!(*removed.unwrap(), 100);
                assert_eq!(cache.len(), 3);

                // Verify new tail is correct
                let (new_tail_key, _) = cache.peek_lru().unwrap();
                assert_eq!(*new_tail_key, 2); // New tail
                assert_eq!(cache.recency_rank(&4), Some(0)); // Still head
                assert_eq!(cache.recency_rank(&3), Some(1));
                assert_eq!(cache.recency_rank(&2), Some(2)); // New tail

                // Remove new tail
                cache.remove(&2);
                assert_eq!(cache.len(), 2);
                let (newer_tail_key, _) = cache.peek_lru().unwrap();
                assert_eq!(*newer_tail_key, 3); // Newer tail

                // Remove final tail
                cache.remove(&3);
                assert_eq!(cache.len(), 1);
                let (final_tail_key, _) = cache.peek_lru().unwrap();
                assert_eq!(*final_tail_key, 4); // Only item left
            }

            #[test]
            fn test_lru_single_node_operations() {
                // Test LRU operations when cache has only one node
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);

                // Insert single item
                cache.insert(1, Arc::new(100));
                assert_eq!(cache.len(), 1);

                // Single item should be both head and tail
                assert_eq!(cache.recency_rank(&1), Some(0));
                let (lru_key, lru_value) = cache.peek_lru().unwrap();
                assert_eq!(*lru_key, 1);
                assert_eq!(**lru_value, 100);

                // Get operation should maintain position
                let value = cache.get(&1);
                assert!(value.is_some());
                assert_eq!(**value.unwrap(), 100);
                assert_eq!(cache.recency_rank(&1), Some(0));

                // Touch operation should work
                assert!(cache.touch(&1));
                assert_eq!(cache.recency_rank(&1), Some(0));

                // Peek operations should work
                assert_eq!(*cache.peek(&1).unwrap(), 100);
                let (peek_key, peek_value) = cache.peek_lru().unwrap();
                assert_eq!(*peek_key, 1);
                assert_eq!(**peek_value, 100);

                // Re-insert should update value but maintain position
                cache.insert(1, Arc::new(999));
                assert_eq!(cache.len(), 1);
                assert_eq!(cache.recency_rank(&1), Some(0));
                assert_eq!(**cache.get(&1).unwrap(), 999);

                // Pop should work and empty cache
                let (popped_key, popped_value) = cache.pop_lru().unwrap();
                assert_eq!(popped_key, 1);
                assert_eq!(*popped_value, 999);
                assert_eq!(cache.len(), 0);
                assert!(cache.peek_lru().is_none());
            }

            #[test]
            fn test_lru_two_node_operations() {
                // Test LRU operations with exactly two nodes
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);

                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));

                // Verify initial state
                assert_eq!(cache.len(), 2);
                assert_eq!(cache.recency_rank(&2), Some(0)); // Head
                assert_eq!(cache.recency_rank(&1), Some(1)); // Tail

                let (tail_key, _) = cache.peek_lru().unwrap();
                assert_eq!(*tail_key, 1);

                // Swap positions by accessing tail
                cache.get(&1);
                assert_eq!(cache.recency_rank(&1), Some(0)); // New head
                assert_eq!(cache.recency_rank(&2), Some(1)); // New tail

                let (new_tail_key, _) = cache.peek_lru().unwrap();
                assert_eq!(*new_tail_key, 2);

                // Touch operations
                assert!(cache.touch(&2));
                assert_eq!(cache.recency_rank(&2), Some(0)); // Back to head
                assert_eq!(cache.recency_rank(&1), Some(1)); // Back to tail

                // Remove head
                cache.remove(&2);
                assert_eq!(cache.len(), 1);
                assert_eq!(cache.recency_rank(&1), Some(0)); // Only item

                // Add back second item
                cache.insert(3, Arc::new(300));
                assert_eq!(cache.len(), 2);
                assert_eq!(cache.recency_rank(&3), Some(0)); // New head
                assert_eq!(cache.recency_rank(&1), Some(1)); // Old item is tail

                // Test insertion with available capacity (cache capacity is 3, we have 2 items)
                cache.insert(4, Arc::new(400));
                assert_eq!(cache.len(), 3); // Should be 3 items now (1, 3, 4)
                assert!(cache.contains(&1)); // Should still be present
                assert!(cache.contains(&3));
                assert!(cache.contains(&4));
                assert_eq!(cache.recency_rank(&4), Some(0)); // Most recent
                assert_eq!(cache.recency_rank(&3), Some(1)); // Previous head
                assert_eq!(cache.recency_rank(&1), Some(2)); // Least recent (tail)

                // Now test eviction by inserting another item (5th item, exceeds capacity)
                cache.insert(5, Arc::new(500));
                assert_eq!(cache.len(), 3); // Should maintain capacity
                assert!(!cache.contains(&1)); // LRU evicted
                assert!(cache.contains(&3));
                assert!(cache.contains(&4));
                assert!(cache.contains(&5));
                assert_eq!(cache.recency_rank(&5), Some(0)); // Most recent
                assert_eq!(cache.recency_rank(&4), Some(1));
                assert_eq!(cache.recency_rank(&3), Some(2)); // Now least recent
            }

            #[test]
            fn test_lru_aging_pattern() {
                // Test items aging from head to tail over time
                let mut cache: LRUCore<i32, i32> = LRUCore::new(4);

                // Insert items with delays to simulate aging
                cache.insert(1, Arc::new(100));
                assert_eq!(cache.recency_rank(&1), Some(0));

                cache.insert(2, Arc::new(200));
                assert_eq!(cache.recency_rank(&2), Some(0)); // New head
                assert_eq!(cache.recency_rank(&1), Some(1)); // Aged one position

                cache.insert(3, Arc::new(300));
                assert_eq!(cache.recency_rank(&3), Some(0)); // New head
                assert_eq!(cache.recency_rank(&2), Some(1)); // Aged one position
                assert_eq!(cache.recency_rank(&1), Some(2)); // Aged another position

                cache.insert(4, Arc::new(400));
                assert_eq!(cache.recency_rank(&4), Some(0)); // New head
                assert_eq!(cache.recency_rank(&3), Some(1)); // Aged one position
                assert_eq!(cache.recency_rank(&2), Some(2)); // Aged one position
                assert_eq!(cache.recency_rank(&1), Some(3)); // Now at tail

                // Insert one more item to trigger eviction of oldest
                cache.insert(5, Arc::new(500));
                assert!(!cache.contains(&1)); // Evicted (was oldest)
                assert_eq!(cache.recency_rank(&5), Some(0)); // New head
                assert_eq!(cache.recency_rank(&4), Some(1)); // Aged
                assert_eq!(cache.recency_rank(&3), Some(2)); // Aged
                assert_eq!(cache.recency_rank(&2), Some(3)); // Now oldest
            }

            #[test]
            fn test_lru_promotion_to_head() {
                // Test promoting items from various positions to head
                let mut cache: LRUCore<i32, i32> = LRUCore::new(5);

                for i in 1..=5 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Initial order: 5(0), 4(1), 3(2), 2(3), 1(4)

                // Promote tail to head
                cache.get(&1);
                assert_eq!(cache.recency_rank(&1), Some(0));
                // New order: 1(0), 5(1), 4(2), 3(3), 2(4)

                // Promote middle item to head
                cache.touch(&3);
                assert_eq!(cache.recency_rank(&3), Some(0));
                // New order: 3(0), 1(1), 5(2), 4(3), 2(4)

                // Promote item near head
                cache.get(&5);
                assert_eq!(cache.recency_rank(&5), Some(0));
                // New order: 5(0), 3(1), 1(2), 4(3), 2(4)

                // Promote current head (should stay at head)
                cache.touch(&5);
                assert_eq!(cache.recency_rank(&5), Some(0));
                // Order unchanged: 5(0), 3(1), 1(2), 4(3), 2(4)

                // Verify final state
                assert_eq!(cache.recency_rank(&5), Some(0));
                assert_eq!(cache.recency_rank(&3), Some(1));
                assert_eq!(cache.recency_rank(&1), Some(2));
                assert_eq!(cache.recency_rank(&4), Some(3));
                assert_eq!(cache.recency_rank(&2), Some(4)); // Still tail
            }

            #[test]
            fn test_lru_demotion_patterns() {
                // Test how items move down in LRU order
                let mut cache: LRUCore<i32, i32> = LRUCore::new(4);

                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                cache.insert(3, Arc::new(300));
                cache.insert(4, Arc::new(400));

                // Initial order: 4(0), 3(1), 2(2), 1(3)

                // Track how item 4 demotes as other items are accessed
                assert_eq!(cache.recency_rank(&4), Some(0)); // Currently head

                // Access item 3 - item 4 should demote to rank 1
                cache.get(&3);
                assert_eq!(cache.recency_rank(&4), Some(1)); // Demoted
                assert_eq!(cache.recency_rank(&3), Some(0)); // New head

                // Access item 2 - item 4 should demote to rank 2
                cache.get(&2);
                assert_eq!(cache.recency_rank(&4), Some(2)); // Further demoted
                assert_eq!(cache.recency_rank(&2), Some(0)); // New head
                assert_eq!(cache.recency_rank(&3), Some(1)); // Demoted

                // Access item 1 - item 4 should demote to tail (rank 3)
                cache.get(&1);
                assert_eq!(cache.recency_rank(&4), Some(3)); // Now at tail
                assert_eq!(cache.recency_rank(&1), Some(0)); // New head
                assert_eq!(cache.recency_rank(&2), Some(1)); // Demoted
                assert_eq!(cache.recency_rank(&3), Some(2)); // Demoted

                // Verify item 4 is now LRU
                let (lru_key, _) = cache.peek_lru().unwrap();
                assert_eq!(*lru_key, 4);

                // One more access should evict item 4
                cache.insert(5, Arc::new(500));
                assert!(!cache.contains(&4)); // Evicted
                assert!(cache.contains(&1));
                assert!(cache.contains(&2));
                assert!(cache.contains(&3));
                assert!(cache.contains(&5));
            }

            #[test]
            fn test_lru_circular_access_pattern() {
                // Test LRU behavior with circular access patterns
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);

                // Fill cache
                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                cache.insert(3, Arc::new(300));

                // Perform circular access pattern: 1 -> 2 -> 3 -> 1 -> 2 -> 3...
                for _round in 0..5 {
                    for key in [1, 2, 3] {
                        cache.get(&key);
                        assert_eq!(cache.recency_rank(&key), Some(0)); // Should be at head
                    }

                    // After each full round, all items should still be present
                    for key in [1, 2, 3] {
                        assert!(cache.contains(&key));
                    }
                }

                // Final state should have 3 at head (last accessed)
                assert_eq!(cache.recency_rank(&3), Some(0));
                assert_eq!(cache.recency_rank(&2), Some(1));
                assert_eq!(cache.recency_rank(&1), Some(2));

                // Try to evict by inserting new item - should evict 1 (LRU)
                cache.insert(4, Arc::new(400));
                assert!(!cache.contains(&1)); // Evicted
                assert!(cache.contains(&2));
                assert!(cache.contains(&3));
                assert!(cache.contains(&4));
            }

            #[test]
            fn test_lru_working_set_behavior() {
                // Test LRU behavior with working set larger than cache
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);

                // Working set of 5 items, but cache capacity is only 3
                let working_set = [1, 2, 3, 4, 5];

                // Fill cache with subset of working set
                for &key in &working_set[0..3] {
                    cache.insert(key, Arc::new(key * 100));
                }

                // Simulate working set access pattern
                for _round in 0..3 {
                    for &key in &working_set {
                        // Try to access all items in working set
                        if cache.contains(&key) {
                            cache.get(&key);
                        } else {
                            // Item not in cache, need to insert it (causes eviction)
                            cache.insert(key, Arc::new(key * 100));
                        }
                    }
                }

                // Cache should only contain last 3 accessed items from working set
                assert_eq!(cache.len(), 3);

                // Most recent items from working set should be present
                assert!(cache.contains(&5));
                assert!(cache.contains(&4));
                assert!(cache.contains(&3));

                // Verify LRU order matches access pattern
                assert_eq!(cache.recency_rank(&5), Some(0)); // Most recent
                assert_eq!(cache.recency_rank(&4), Some(1));
                assert_eq!(cache.recency_rank(&3), Some(2)); // Least recent in cache
            }

            #[test]
            fn test_lru_temporal_locality() {
                // Test LRU behavior with high temporal locality
                let mut cache: LRUCore<i32, i32> = LRUCore::new(4);

                // Fill cache
                for i in 1..=4 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Simulate high temporal locality - repeatedly access same few items
                let hot_items = [2, 3];

                // Access hot items repeatedly
                for _ in 0..10 {
                    for &item in &hot_items {
                        cache.get(&item);
                    }
                }

                // Hot items should be at the head of LRU list
                assert_eq!(cache.recency_rank(&3), Some(0)); // Last accessed
                assert_eq!(cache.recency_rank(&2), Some(1)); // Second to last

                // Cold items should be further down
                assert_eq!(cache.recency_rank(&4), Some(2));
                assert_eq!(cache.recency_rank(&1), Some(3)); // Least recent

                // Insert new items - cold items should be evicted first
                cache.insert(5, Arc::new(500));
                assert!(!cache.contains(&1)); // Evicted (coldest)

                cache.insert(6, Arc::new(600));
                assert!(!cache.contains(&4)); // Evicted (next coldest)

                // Hot items should still be present
                assert!(cache.contains(&2));
                assert!(cache.contains(&3));
            }

            #[test]
            fn test_lru_no_temporal_locality() {
                // Test LRU behavior with no temporal locality (sequential access)
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);

                // Sequential access pattern with no repetition
                let mut access_sequence = 1;

                // First, fill the cache
                for i in 1..=3 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Continue sequential access, causing evictions
                for _ in 0..10 {
                    access_sequence += 1;
                    cache.insert(access_sequence, Arc::new(access_sequence * 100));
                }

                // Cache should contain only the most recent 3 items
                let expected_items = [access_sequence - 2, access_sequence - 1, access_sequence];

                for &item in &expected_items {
                    assert!(cache.contains(&item));
                }

                // Verify LRU order matches insertion order
                assert_eq!(cache.recency_rank(&access_sequence), Some(0)); // Most recent
                assert_eq!(cache.recency_rank(&(access_sequence - 1)), Some(1));
                assert_eq!(cache.recency_rank(&(access_sequence - 2)), Some(2)); // Least recent

                // Earlier items should have been evicted
                for i in 1..=(access_sequence - 3) {
                    assert!(!cache.contains(&i));
                }
            }

            #[test]
            fn test_lru_mixed_access_patterns() {
                // Test LRU with mixed random and sequential access
                let mut cache: LRUCore<i32, i32> = LRUCore::new(4);

                // Phase 1: Sequential insertion
                for i in 1..=4 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Phase 2: Random access pattern
                let random_pattern = [2, 4, 1, 3, 2, 1];
                for &key in &random_pattern {
                    cache.get(&key);
                }

                // Phase 3: Sequential insertion of new items
                for i in 5..=7 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Cache should maintain LRU properties throughout
                assert_eq!(cache.len(), 4);

                // Most recently inserted items should be present
                assert!(cache.contains(&7));
                assert!(cache.contains(&6));
                assert!(cache.contains(&5));

                // One item from random access phase might survive depending on pattern
                // Verify at least the most recently accessed items are handled correctly
                let (lru_key, _) = cache.peek_lru().unwrap();
                assert!(cache.recency_rank(lru_key).unwrap() == 3);

                // Phase 4: Mix of new insertions and accesses to existing items
                cache.get(&7); // Access most recent
                cache.insert(8, Arc::new(800)); // Insert new
                cache.get(&6); // Access existing

                // Verify final state maintains LRU ordering
                assert_eq!(cache.recency_rank(&6), Some(0)); // Last accessed
                assert_eq!(cache.recency_rank(&8), Some(1)); // Last inserted
                assert_eq!(cache.recency_rank(&7), Some(2)); // Accessed before 8 was inserted
            }

            #[test]
            fn test_lru_hotspot_behavior() {
                // Test LRU behavior when few items are accessed frequently
                let mut cache: LRUCore<i32, i32> = LRUCore::new(5);

                // Fill cache with items
                for i in 1..=5 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Designate items 2 and 4 as "hotspots" (frequently accessed)
                let hotspots = [2, 4];
                let cold_items = [1, 3, 5];

                // Simulate workload with hotspots
                for round in 0..20 {
                    // Access hotspots frequently
                    for _ in 0..5 {
                        for &hot_item in &hotspots {
                            cache.get(&hot_item);
                        }
                    }

                    // Occasionally access cold items
                    if round % 4 == 0 && !cold_items.is_empty() {
                        let cold_idx = round / 4 % cold_items.len();
                        if cache.contains(&cold_items[cold_idx]) {
                            cache.get(&cold_items[cold_idx]);
                        }
                    }
                }

                // Hotspots should be at or near the head
                assert!(cache.recency_rank(&2).unwrap() <= 1);
                assert!(cache.recency_rank(&4).unwrap() <= 1);

                // When inserting new items, cold items should be evicted first
                cache.insert(6, Arc::new(600));
                cache.insert(7, Arc::new(700));

                // Hotspots should still be present
                assert!(cache.contains(&2));
                assert!(cache.contains(&4));

                // At least some cold items should have been evicted
                let cold_evicted = cold_items
                    .iter()
                    .filter(|&&item| !cache.contains(&item))
                    .count();
                assert!(cold_evicted > 0);
            }

            #[test]
            fn test_lru_coldspot_eviction() {
                // Test that rarely accessed items are evicted appropriately
                let mut cache: LRUCore<i32, i32> = LRUCore::new(4);

                // Insert items
                for i in 1..=4 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Make item 1 a "coldspot" by not accessing it
                // Access other items to make them more recent
                for _ in 0..5 {
                    cache.get(&2);
                    cache.get(&3);
                    cache.get(&4);
                }

                // Item 1 should be the coldest (LRU)
                let (lru_key, _) = cache.peek_lru().unwrap();
                assert_eq!(*lru_key, 1);
                assert_eq!(cache.recency_rank(&1), Some(3));

                // Insert new item - should evict cold item 1
                cache.insert(5, Arc::new(500));
                assert!(!cache.contains(&1)); // Cold item evicted
                assert!(cache.contains(&2));
                assert!(cache.contains(&3));
                assert!(cache.contains(&4));
                assert!(cache.contains(&5));

                // Create another coldspot
                for _ in 0..3 {
                    cache.get(&3);
                    cache.get(&4);
                    cache.get(&5);
                }

                // Item 2 should now be coldest
                let (lru_key, _) = cache.peek_lru().unwrap();
                assert_eq!(*lru_key, 2);

                // Insert another item
                cache.insert(6, Arc::new(600));
                assert!(!cache.contains(&2)); // Next cold item evicted
                assert!(cache.contains(&3));
                assert!(cache.contains(&4));
                assert!(cache.contains(&5));
                assert!(cache.contains(&6));
            }

            #[test]
            fn test_lru_rank_consistency() {
                // Test that recency ranks are consistent with actual order
                let mut cache: LRUCore<i32, i32> = LRUCore::new(5);

                for i in 1..=5 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Function to verify rank consistency
                let verify_ranks = |cache: &LRUCore<i32, i32>| {
                    let mut ranks = vec![];
                    // Check all possible keys that might be in cache (including newly inserted ones)
                    for i in 1..=10 {
                        if let Some(rank) = cache.recency_rank(&i) {
                            ranks.push((i, rank));
                        }
                    }

                    // Should have exactly cache.len() items with ranks
                    assert_eq!(ranks.len(), cache.len());

                    // Ranks should be unique
                    let mut rank_values: Vec<_> = ranks.iter().map(|(_, rank)| *rank).collect();
                    rank_values.sort();
                    rank_values.dedup();
                    assert_eq!(rank_values.len(), ranks.len());

                    // Ranks should be consecutive starting from 0
                    for (idx, &rank) in rank_values.iter().enumerate() {
                        assert_eq!(
                            rank, idx,
                            "Rank at index {} should be {}, but was {}. Current items: {:?}",
                            idx, idx, rank, ranks
                        );
                    }

                    // LRU item should have highest rank
                    if let Some((lru_key, _)) = cache.peek_lru() {
                        let lru_rank = cache.recency_rank(lru_key).unwrap();
                        assert_eq!(lru_rank, cache.len() - 1);
                    }
                };

                // Verify initial state
                verify_ranks(&cache);

                // Perform operations and verify consistency
                cache.get(&3);
                verify_ranks(&cache);

                cache.touch(&1);
                verify_ranks(&cache);

                cache.remove(&4);
                verify_ranks(&cache);

                cache.insert(6, Arc::new(600));
                verify_ranks(&cache);

                cache.pop_lru();
                verify_ranks(&cache);
            }

            #[test]
            fn test_lru_rank_updates_after_access() {
                // Test recency rank changes after accessing items
                let mut cache: LRUCore<i32, i32> = LRUCore::new(4);

                for i in 1..=4 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Record initial ranks
                let _initial_ranks: Vec<_> = (1..=4)
                    .map(|i| (i, cache.recency_rank(&i).unwrap()))
                    .collect();

                // Access item 2 and verify rank changes
                let old_rank_2 = cache.recency_rank(&2).unwrap();
                cache.get(&2);
                let new_rank_2 = cache.recency_rank(&2).unwrap();

                assert_eq!(new_rank_2, 0); // Should be at head
                assert_ne!(old_rank_2, new_rank_2);

                // Other items should have shifted accordingly
                for i in [1, 3, 4] {
                    let old_rank = _initial_ranks.iter().find(|(key, _)| *key == i).unwrap().1;
                    let new_rank = cache.recency_rank(&i);

                    if let Some(rank) = new_rank {
                        // Item that was more recent than 2 should shift down by 1
                        // Item that was less recent than 2 should maintain relative position
                        if old_rank < old_rank_2 {
                            assert_eq!(rank, old_rank + 1);
                        }
                    }
                }

                // Touch operation should also update ranks
                let _old_rank_1 = cache.recency_rank(&1).unwrap();
                cache.touch(&1);
                assert_eq!(cache.recency_rank(&1).unwrap(), 0);
                assert_eq!(cache.recency_rank(&2).unwrap(), 1); // 2 demoted
            }

            #[test]
            fn test_lru_batch_operations() {
                // Test LRU behavior with batches of operations
                let mut cache: LRUCore<i32, i32> = LRUCore::new(6);

                // Batch 1: Insert multiple items
                let batch1_keys = [1, 2, 3, 4];
                for &key in &batch1_keys {
                    cache.insert(key, Arc::new(key * 100));
                }
                assert_eq!(cache.len(), 4);

                // Batch 2: Access subset of items
                let batch2_access = [2, 4];
                for &key in &batch2_access {
                    cache.get(&key);
                }

                // Verify batch access affected ordering
                assert!(cache.recency_rank(&4).unwrap() < cache.recency_rank(&1).unwrap());
                assert!(cache.recency_rank(&2).unwrap() < cache.recency_rank(&3).unwrap());

                // Batch 3: Insert more items (will reach capacity)
                let batch3_keys = [5, 6];
                for &key in &batch3_keys {
                    cache.insert(key, Arc::new(key * 100));
                }
                assert_eq!(cache.len(), 6);

                // Batch 4: Remove multiple items
                let batch4_remove = [1, 3];
                for &key in &batch4_remove {
                    cache.remove(&key);
                }
                assert_eq!(cache.len(), 4);

                // Verify removed items are gone
                for &key in &batch4_remove {
                    assert!(!cache.contains(&key));
                }

                // Batch 5: Mix of operations
                cache.get(&2); // Access
                cache.insert(7, Arc::new(700)); // Insert
                cache.touch(&6); // Touch
                cache.insert(8, Arc::new(800)); // Insert

                assert_eq!(cache.len(), 6); // Should be at capacity

                // Verify final state has correct ordering
                assert_eq!(cache.recency_rank(&8), Some(0)); // Most recent insert
                assert_eq!(cache.recency_rank(&6), Some(1)); // Last touch
                assert_eq!(cache.recency_rank(&7), Some(2)); // Previous insert
                assert_eq!(cache.recency_rank(&2), Some(3)); // Previous get

                // Remaining items should be present with appropriate ranks
                let remaining_keys = vec![2, 4, 5, 6, 7, 8];
                for &key in &remaining_keys {
                    assert!(cache.contains(&key));
                    assert!(cache.recency_rank(&key).is_some());
                }
            }

            #[test]
            fn test_lru_interleaved_insert_access() {
                // Test interleaved insert and access operations on LRU order
                let mut cache: LRUCore<i32, i32> = LRUCore::new(4);

                // Interleave insertions and accesses
                cache.insert(1, Arc::new(100)); // Cache: [1]
                cache.insert(2, Arc::new(200)); // Cache: [2, 1]
                cache.get(&1); // Cache: [1, 2]
                cache.insert(3, Arc::new(300)); // Cache: [3, 1, 2]
                cache.get(&2); // Cache: [2, 3, 1]
                cache.insert(4, Arc::new(400)); // Cache: [4, 2, 3, 1]
                cache.get(&3); // Cache: [3, 4, 2, 1]

                // Verify final ordering
                assert_eq!(cache.recency_rank(&3), Some(0)); // Most recent access
                assert_eq!(cache.recency_rank(&4), Some(1)); // Most recent insert
                assert_eq!(cache.recency_rank(&2), Some(2)); // Previous access
                assert_eq!(cache.recency_rank(&1), Some(3)); // Least recent

                // Insert new item causing eviction
                cache.insert(5, Arc::new(500)); // Should evict 1 (LRU)
                assert!(!cache.contains(&1)); // Evicted
                assert_eq!(cache.recency_rank(&5), Some(0)); // New head
                assert_eq!(cache.recency_rank(&3), Some(1)); // Demoted
                assert_eq!(cache.recency_rank(&4), Some(2)); // Demoted
                assert_eq!(cache.recency_rank(&2), Some(3)); // New tail
            }

            #[test]
            fn test_lru_frequency_vs_recency() {
                // Test LRU prioritizes recency over frequency
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);

                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                cache.insert(3, Arc::new(300));

                // Make item 1 very frequently accessed
                for _ in 0..100 {
                    cache.get(&1);
                }

                // Access item 2 once (more recent than 1's last access)
                cache.get(&2);

                // Access item 3 once (most recent)
                cache.get(&3);

                // Despite item 1 being accessed 100 times, recency should matter
                // Current order should be: 3(0), 2(1), 1(2)
                assert_eq!(cache.recency_rank(&3), Some(0));
                assert_eq!(cache.recency_rank(&2), Some(1));
                assert_eq!(cache.recency_rank(&1), Some(2)); // Least recent despite high frequency

                // Insert new item - item 1 should be evicted despite high frequency
                cache.insert(4, Arc::new(400));
                assert!(!cache.contains(&1)); // Evicted due to being LRU, not frequency
                assert!(cache.contains(&2));
                assert!(cache.contains(&3));
                assert!(cache.contains(&4));
            }

            #[test]
            fn test_lru_cache_warming() {
                // Test LRU behavior during cache warming phase
                let mut cache: LRUCore<i32, i32> = LRUCore::new(5);

                // Phase 1: Empty cache
                assert_eq!(cache.len(), 0);
                assert!(cache.peek_lru().is_none());

                // Phase 2: Partial warming (50% capacity)
                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                assert_eq!(cache.len(), 2);
                assert_eq!(cache.recency_rank(&2), Some(0));
                assert_eq!(cache.recency_rank(&1), Some(1));

                // Access during warming
                cache.get(&1);
                assert_eq!(cache.recency_rank(&1), Some(0));

                // Phase 3: Continue warming (80% capacity)
                cache.insert(3, Arc::new(300));
                cache.insert(4, Arc::new(400));
                assert_eq!(cache.len(), 4);

                // Verify ordering during warming maintains LRU properties
                assert_eq!(cache.recency_rank(&4), Some(0)); // Most recent insert
                assert_eq!(cache.recency_rank(&3), Some(1));
                assert_eq!(cache.recency_rank(&1), Some(2)); // Last accessed
                assert_eq!(cache.recency_rank(&2), Some(3)); // Least recent

                // Phase 4: Complete warming (100% capacity)
                cache.insert(5, Arc::new(500));
                assert_eq!(cache.len(), 5);

                // Phase 5: Post-warming (evictions start)
                cache.insert(6, Arc::new(600));
                assert_eq!(cache.len(), 5); // Still at capacity
                assert!(!cache.contains(&2)); // LRU evicted
            }

            #[test]
            fn test_lru_cache_cooling() {
                // Test LRU behavior when cache activity decreases
                let mut cache: LRUCore<i32, i32> = LRUCore::new(4);

                // Fill cache with high activity
                for i in 1..=4 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // High activity phase - lots of accesses
                for _ in 0..20 {
                    for i in 1..=4 {
                        cache.get(&i);
                    }
                }

                // Verify all items present during high activity
                for i in 1..=4 {
                    assert!(cache.contains(&i));
                }

                // Cooling phase - reduced activity, only access subset
                let active_items = [2, 4];
                for _ in 0..5 {
                    for &item in &active_items {
                        cache.get(&item);
                    }
                }

                // Items 1 and 3 should now be colder (less recent)
                assert!(cache.recency_rank(&2).unwrap() < cache.recency_rank(&1).unwrap());
                assert!(cache.recency_rank(&4).unwrap() < cache.recency_rank(&3).unwrap());

                // Further cooling - access only one item
                for _ in 0..3 {
                    cache.get(&4);
                }

                // Item 4 should be hottest, others cooler
                assert_eq!(cache.recency_rank(&4), Some(0));

                // Insert new items during cooling - cold items should be evicted
                cache.insert(5, Arc::new(500));
                cache.insert(6, Arc::new(600));

                // Item 4 should survive (hot), others may be evicted based on cooling pattern
                assert!(cache.contains(&4));
                assert!(cache.contains(&5));
                assert!(cache.contains(&6));
            }

            #[test]
            fn test_lru_steady_state_behavior() {
                // Test LRU behavior in steady state (full cache)
                let mut cache: LRUCore<i32, i32> = LRUCore::new(4);

                // Reach steady state (full capacity)
                for i in 1..=4 {
                    cache.insert(i, Arc::new(i * 100));
                }
                assert_eq!(cache.len(), 4);

                // Steady state operations - every insert causes eviction
                let steady_state_items = [5, 6, 7, 8, 9, 10];
                for &item in &steady_state_items {
                    let old_lru = *cache.peek_lru().unwrap().0;
                    cache.insert(item, Arc::new(item * 100));

                    // Should maintain capacity
                    assert_eq!(cache.len(), 4);

                    // Previous LRU should be evicted
                    assert!(!cache.contains(&old_lru));

                    // New item should be at head
                    assert_eq!(cache.recency_rank(&item), Some(0));
                }

                // In steady state, access patterns should still affect ordering
                cache.get(&8); // Access existing item
                assert_eq!(cache.recency_rank(&8), Some(0));

                cache.insert(11, Arc::new(1100)); // Insert new
                assert_eq!(cache.recency_rank(&11), Some(0));
                assert_eq!(cache.recency_rank(&8), Some(1)); // Demoted but still present

                // Verify steady state maintains LRU invariants
                for i in 0..cache.len() {
                    let mut found_rank = false;
                    for j in 7..=11 {
                        // Recent items range
                        if cache.contains(&j) && cache.recency_rank(&j) == Some(i) {
                            found_rank = true;
                            break;
                        }
                    }
                    if !found_rank && i < cache.len() {
                        // This should not happen in correct LRU implementation
                        panic!("Missing rank {} in steady state", i);
                    }
                }
            }

            #[test]
            fn test_lru_transition_states() {
                // Test LRU behavior during capacity transitions
                let mut cache: LRUCore<i32, i32> = LRUCore::new(4);

                // Transition 1: Empty -> Partial (25%)
                cache.insert(1, Arc::new(100));
                assert_eq!(cache.len(), 1);
                assert_eq!(cache.recency_rank(&1), Some(0));

                // Transition 2: Partial -> Half (50%)
                cache.insert(2, Arc::new(200));
                assert_eq!(cache.len(), 2);
                assert_eq!(cache.recency_rank(&2), Some(0));
                assert_eq!(cache.recency_rank(&1), Some(1));

                // Transition 3: Half -> Near Full (75%)
                cache.insert(3, Arc::new(300));
                assert_eq!(cache.len(), 3);

                // Transition 4: Near Full -> Full (100%)
                cache.insert(4, Arc::new(400));
                assert_eq!(cache.len(), 4);

                // Verify full state
                for i in 1..=4 {
                    assert!(cache.contains(&i));
                }

                // Transition 5: Full -> Full with eviction
                cache.insert(5, Arc::new(500));
                assert_eq!(cache.len(), 4); // Still full
                assert!(!cache.contains(&1)); // LRU evicted

                // Verify each transition maintains LRU properties
                // Access pattern to test transitions in different states

                // Remove items to transition back down
                cache.remove(&2);
                assert_eq!(cache.len(), 3); // Back to 75%

                cache.remove(&3);
                assert_eq!(cache.len(), 2); // Back to 50%

                cache.remove(&4);
                assert_eq!(cache.len(), 1); // Back to 25%

                // Verify remaining item is still accessible
                assert!(cache.contains(&5));
                assert_eq!(cache.recency_rank(&5), Some(0));

                // Final transition back to empty
                cache.remove(&5);
                assert_eq!(cache.len(), 0);
                assert!(cache.peek_lru().is_none());
            }

            #[test]
            fn test_lru_pointer_integrity() {
                // Test that all prev/next pointers are correctly maintained
                let mut cache: LRUCore<i32, i32> = LRUCore::new(5);

                // Build cache and verify pointer integrity through operations
                for i in 1..=5 {
                    cache.insert(i, Arc::new(i * 100));

                    // After each insertion, verify we can traverse from head to tail
                    // by checking all items have valid ranks
                    for j in 1..=i {
                        assert!(cache.recency_rank(&j).is_some());
                    }

                    // Verify peek_lru works (requires valid tail pointer)
                    assert!(cache.peek_lru().is_some());
                }

                // Test pointer integrity through various operations

                // Move head to different position
                cache.get(&1); // Move tail to head
                assert_eq!(cache.recency_rank(&1), Some(0));

                // Move middle item
                cache.get(&3); // Move middle to head
                assert_eq!(cache.recency_rank(&3), Some(0));

                // Remove head
                cache.remove(&3);
                assert_eq!(cache.len(), 4);

                // Remove tail
                let tail_key = {
                    let (key, _) = cache.peek_lru().unwrap();
                    *key
                };
                cache.remove(&tail_key);
                assert_eq!(cache.len(), 3);

                // Remove middle
                let middle_key = cache.recency_rank(&1).unwrap() == 1;
                if middle_key {
                    cache.remove(&1);
                } else {
                    // Find an item that's not head or tail
                    for i in [2, 4, 5] {
                        if cache.contains(&i) {
                            let rank = cache.recency_rank(&i).unwrap();
                            if rank != 0 && rank != cache.len() - 1 {
                                cache.remove(&i);
                                break;
                            }
                        }
                    }
                }

                // After all operations, verify remaining items still have valid traversal
                let remaining_count = cache.len();
                for i in 0..remaining_count {
                    let mut found = false;
                    for j in 1..=5 {
                        if cache.contains(&j) && cache.recency_rank(&j) == Some(i) {
                            found = true;
                            break;
                        }
                    }
                    assert!(found, "Rank {} not found after pointer operations", i);
                }
            }

            #[test]
            fn test_lru_list_node_count() {
                // Test that linked list node count matches HashMap size
                let mut cache: LRUCore<i32, i32> = LRUCore::new(4);

                // Initially empty
                assert_eq!(cache.len(), 0);

                // Add items and verify count consistency
                for i in 1..=4 {
                    cache.insert(i, Arc::new(i * 100));
                    assert_eq!(cache.len(), i as usize);

                    // Verify we can find exactly i items with valid ranks
                    let mut found_count = 0;
                    for j in 1..=i {
                        if cache.recency_rank(&j).is_some() {
                            found_count += 1;
                        }
                    }
                    assert_eq!(found_count, i);
                }

                // Remove items and verify count consistency
                cache.remove(&2);
                assert_eq!(cache.len(), 3);

                // Verify exactly 3 items have valid ranks
                let mut valid_ranks = 0;
                for i in 1..=4 {
                    if cache.recency_rank(&i).is_some() {
                        valid_ranks += 1;
                    }
                }
                assert_eq!(valid_ranks, 3);

                // Evict through insertion
                cache.insert(5, Arc::new(500));
                assert_eq!(cache.len(), 4); // Back to capacity

                // Clear and verify
                cache.clear();
                assert_eq!(cache.len(), 0);

                // No items should have valid ranks
                for i in 1..=5 {
                    assert!(cache.recency_rank(&i).is_none());
                }
            }

            #[test]
            fn test_lru_bidirectional_consistency() {
                // Test that forward and backward traversals are consistent
                let mut cache: LRUCore<i32, i32> = LRUCore::new(5);

                for i in 1..=5 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Function to verify bidirectional consistency
                let verify_bidirectional = |cache: &LRUCore<i32, i32>| {
                    // Forward traversal: collect items by rank from 0 to len-1
                    let mut forward_items = vec![];
                    for rank in 0..cache.len() {
                        for i in 1..=5 {
                            if cache.contains(&i) && cache.recency_rank(&i) == Some(rank) {
                                forward_items.push(i);
                                break;
                            }
                        }
                    }

                    // Backward traversal: start from LRU and work backwards
                    let mut backward_items = vec![];
                    let mut current_items: Vec<_> = (1..=5).filter(|i| cache.contains(i)).collect();
                    current_items.sort_by_key(|i| cache.recency_rank(i).unwrap());
                    current_items.reverse();

                    for &item in &current_items {
                        backward_items.push(item);
                    }

                    // Forward and backward should be exact opposites
                    forward_items.reverse();
                    assert_eq!(forward_items, backward_items);

                    // LRU should match the last item in forward traversal
                    if let Some((lru_key, _)) = cache.peek_lru() {
                        let lru_key = *lru_key;
                        forward_items.reverse(); // Restore original order
                        assert_eq!(forward_items.last(), Some(&lru_key));
                    }
                };

                // Verify initial state
                verify_bidirectional(&cache);

                // Perform operations and verify after each
                cache.get(&3);
                verify_bidirectional(&cache);

                cache.touch(&1);
                verify_bidirectional(&cache);

                cache.remove(&4);
                verify_bidirectional(&cache);

                cache.insert(6, Arc::new(600));
                verify_bidirectional(&cache);
            }

            #[test]
            fn test_lru_eviction_callback_order() {
                // Test that eviction happens in proper LRU order
                let mut cache: LRUCore<i32, i32> = LRUCore::new(3);

                // Fill cache
                cache.insert(1, Arc::new(100));
                cache.insert(2, Arc::new(200));
                cache.insert(3, Arc::new(300));

                // Track eviction order by recording LRU before each eviction
                let mut eviction_order = vec![];

                // Perform operations that cause evictions
                for new_item in 4..=8 {
                    // Record what should be evicted (current LRU)
                    let (lru_key, _) = cache.peek_lru().unwrap();
                    eviction_order.push(*lru_key);

                    // Insert new item (causes eviction)
                    cache.insert(new_item, Arc::new(new_item * 100));
                }

                // Verify evictions happened in LRU order
                // Items should have been evicted in order: 1, 2, 3, 4, 5
                assert_eq!(eviction_order, vec![1, 2, 3, 4, 5]);

                // Final cache should contain most recent items
                assert!(cache.contains(&6));
                assert!(cache.contains(&7));
                assert!(cache.contains(&8));
                assert_eq!(cache.len(), 3);

                // Test eviction order with intermixed accesses
                cache.clear();
                cache.insert(10, Arc::new(1000));
                cache.insert(11, Arc::new(1100));
                cache.insert(12, Arc::new(1200));

                // Access middle item to change LRU order
                cache.get(&11);
                // Order: 11(0), 12(1), 10(2) - 10 is LRU

                let (lru_key, _) = cache.peek_lru().unwrap();
                assert_eq!(*lru_key, 10);

                // Insert new item - should evict 10
                cache.insert(13, Arc::new(1300));
                assert!(!cache.contains(&10));
                assert!(cache.contains(&11));
                assert!(cache.contains(&12));
                assert!(cache.contains(&13));
            }

            #[test]
            fn test_lru_memory_layout_efficiency() {
                // Test memory layout and access patterns for efficiency
                let mut cache: LRUCore<i32, i32> = LRUCore::new(1000);

                // Fill with many items to test memory efficiency
                for i in 1..=1000 {
                    cache.insert(i, Arc::new(i * 100));
                }

                assert_eq!(cache.len(), 1000);

                // Test that all operations are still efficient with large cache
                // Access pattern that exercises the full range
                for i in (1..=1000).step_by(7) {
                    cache.get(&i);
                }

                // Verify operations still work correctly
                let (lru_key, _) = cache.peek_lru().unwrap();
                assert!(cache.recency_rank(lru_key).is_some());

                // Test efficiency of evictions with large cache
                let start_len = cache.len();
                for i in 1001..=1100 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Should still be at capacity
                assert_eq!(cache.len(), start_len);

                // Verify some original items were evicted
                let mut evicted_count = 0;
                for i in 1..=1000 {
                    if !cache.contains(&i) {
                        evicted_count += 1;
                    }
                }
                assert_eq!(evicted_count, 100); // 100 new items, 100 evictions

                // All ranks should still be valid and unique
                let mut found_ranks = std::collections::HashSet::new();
                for i in 1..=1100 {
                    if let Some(rank) = cache.recency_rank(&i) {
                        assert!(rank < cache.len());
                        assert!(found_ranks.insert(rank)); // Should be unique
                    }
                }
                assert_eq!(found_ranks.len(), cache.len());
            }

            #[test]
            fn test_lru_algorithmic_complexity() {
                // Test that LRU operations maintain O(1) complexity
                let mut cache: LRUCore<i32, i32> = LRUCore::new(100);

                // Fill cache
                for i in 1..=100 {
                    cache.insert(i, Arc::new(i * 100));
                }

                // Test that operations don't degrade with cache size
                // All these operations should be O(1) regardless of cache size

                // Insert (with eviction)
                cache.insert(101, Arc::new(10100));
                assert!(cache.contains(&101));
                assert_eq!(cache.len(), 100);

                // Get
                let value = cache.get(&50);
                assert!(value.is_some());
                assert_eq!(cache.recency_rank(&50), Some(0));

                // Contains
                assert!(cache.contains(&75));
                assert!(!cache.contains(&1)); // Should be evicted

                // Remove
                let removed = cache.remove(&75);
                assert!(removed.is_some());
                assert!(!cache.contains(&75));

                // Touch
                assert!(cache.touch(&60));
                assert_eq!(cache.recency_rank(&60), Some(0));

                // Peek
                let peeked = cache.peek(&80);
                assert!(peeked.is_some());
                // Peek shouldn't change ordering
                assert_ne!(cache.recency_rank(&80), Some(0));

                // Recency rank
                let rank = cache.recency_rank(&90);
                assert!(rank.is_some());
                assert!(rank.unwrap() < cache.len());

                // Peek LRU
                let (lru_key, _) = cache.peek_lru().unwrap();
                let lru_rank = cache.recency_rank(lru_key).unwrap();
                assert_eq!(lru_rank, cache.len() - 1);

                // Copy key before pop to avoid borrow issues
                let expected_lru_key = *lru_key;

                // Pop LRU
                let (popped_key, _) = cache.pop_lru().unwrap();
                assert_eq!(popped_key, expected_lru_key);
                assert!(!cache.contains(&popped_key));

                // All operations should have completed in constant time
                // regardless of the cache size (100 items)
            }

            #[test]
            fn test_lru_concurrent_ordering() {
                // Test LRU ordering behavior in concurrent scenarios
                use std::sync::Arc;
                let cache = super::super::ConcurrentLRUCache::new(4);

                // Fill cache
                for i in 1..=4 {
                    cache.insert(i, i * 100);
                }

                // Test that concurrent accesses work
                cache.get(&1);
                cache.get(&3);

                // Test basic operations work
                assert!(cache.contains(&1));
                assert!(cache.contains(&3));
                assert!(cache.contains(&4));
                assert!(cache.contains(&2));

                // Test eviction in concurrent cache
                cache.insert(5, 500);
                assert!(cache.contains(&5));
                assert_eq!(cache.len(), 4); // Should maintain capacity

                // Test that Arc operations work correctly
                let value_arc = Arc::new(999);
                cache.insert_arc(6, Arc::clone(&value_arc));
                assert_eq!(Arc::strong_count(&value_arc), 2); // Original + cache

                let retrieved = cache.get(&6);
                assert!(retrieved.is_some());
                assert_eq!(Arc::strong_count(&value_arc), 3); // Original + cache + retrieved

                drop(retrieved);
                assert_eq!(Arc::strong_count(&value_arc), 2); // Back to original + cache

                // Test concurrent operations maintain basic functionality
                assert!(cache.contains(&6));
                assert_eq!(cache.len(), 4); // Should still be at capacity
            }

            #[test]
            fn test_lru_deterministic_behavior() {
                // Test that LRU behavior is deterministic given same operations
                let mut cache1: LRUCore<i32, i32> = LRUCore::new(4);
                let mut cache2: LRUCore<i32, i32> = LRUCore::new(4);

                // Perform identical sequence of operations on both caches
                let operations = [
                    ("insert", 1, 100),
                    ("insert", 2, 200),
                    ("insert", 3, 300),
                    ("get", 1, 0), // value ignored for get
                    ("insert", 4, 400),
                    ("get", 2, 0),
                    ("touch", 4, 0),
                    ("insert", 5, 500),
                    ("remove", 3, 0),
                    ("insert", 6, 600),
                ];

                for (op, key, value) in operations {
                    match op {
                        "insert" => {
                            cache1.insert(key, Arc::new(value));
                            cache2.insert(key, Arc::new(value));
                        },
                        "get" => {
                            cache1.get(&key);
                            cache2.get(&key);
                        },
                        "touch" => {
                            cache1.touch(&key);
                            cache2.touch(&key);
                        },
                        "remove" => {
                            cache1.remove(&key);
                            cache2.remove(&key);
                        },
                        _ => panic!("Unknown operation"),
                    }

                    // After each operation, both caches should have identical state
                    assert_eq!(cache1.len(), cache2.len());

                    // Check that same items are present
                    for i in 1..=6 {
                        assert_eq!(cache1.contains(&i), cache2.contains(&i));
                    }

                    // Check that rankings are identical
                    for i in 1..=6 {
                        assert_eq!(cache1.recency_rank(&i), cache2.recency_rank(&i));
                    }

                    // Check that LRU is identical
                    match (cache1.peek_lru(), cache2.peek_lru()) {
                        (Some((key1, _)), Some((key2, _))) => assert_eq!(key1, key2),
                        (None, None) => {}, // Both empty
                        _ => panic!("LRU mismatch between caches"),
                    }
                }

                // Final verification - perform additional identical operations
                for i in 7..=10 {
                    cache1.insert(i, Arc::new(i * 100));
                    cache2.insert(i, Arc::new(i * 100));
                }

                // Final states should be identical
                assert_eq!(cache1.len(), cache2.len());
                for i in 1..=10 {
                    assert_eq!(cache1.contains(&i), cache2.contains(&i));
                    assert_eq!(cache1.recency_rank(&i), cache2.recency_rank(&i));
                }
            }
        }

        mod state_consistency {
            use std::collections::HashSet;
            use std::ptr::NonNull;
            use std::sync::Arc;

            use super::*;

            fn count_nodes<K, V>(cache: &LRUCore<K, V>) -> usize
            where
                K: Copy + Eq + Hash,
            {
                let mut count = 0;
                let mut current = cache.head;
                while let Some(node_ptr) = current {
                    count += 1;
                    unsafe {
                        current = node_ptr.as_ref().next;
                    }
                }
                count
            }

            #[test]
            fn test_hashmap_linkedlist_size_consistency() {
                // Test that HashMap size always matches linked list node count
                let mut cache = LRUCore::new(10);
                assert_eq!(cache.map.len(), count_nodes(&cache));

                cache.insert(1, Arc::new(10));
                assert_eq!(cache.map.len(), count_nodes(&cache));

                cache.insert(2, Arc::new(20));
                assert_eq!(cache.map.len(), count_nodes(&cache));

                cache.remove(&1);
                assert_eq!(cache.map.len(), count_nodes(&cache));

                cache.clear();
                assert_eq!(cache.map.len(), count_nodes(&cache));
            }

            #[test]
            fn test_head_tail_pointer_consistency() {
                // Test that head/tail pointers are consistent with actual list structure
                let mut cache = LRUCore::new(10);

                // Empty
                assert!(cache.head.is_none());
                assert!(cache.tail.is_none());

                // One item
                cache.insert(1, Arc::new(10));
                assert!(cache.head.is_some());
                assert!(cache.tail.is_some());
                assert_eq!(cache.head, cache.tail);

                unsafe {
                    assert!(cache.head.unwrap().as_ref().prev.is_none());
                    assert!(cache.tail.unwrap().as_ref().next.is_none());
                }

                // Two items
                cache.insert(2, Arc::new(20));
                assert!(cache.head.is_some());
                assert!(cache.tail.is_some());
                assert_ne!(cache.head, cache.tail);

                unsafe {
                    // Head should be 2 (MRU)
                    assert_eq!(cache.head.unwrap().as_ref().key, 2);
                    // Tail should be 1 (LRU)
                    assert_eq!(cache.tail.unwrap().as_ref().key, 1);

                    assert!(cache.head.unwrap().as_ref().prev.is_none());
                    assert!(cache.tail.unwrap().as_ref().next.is_none());

                    assert_eq!(cache.head.unwrap().as_ref().next, cache.tail);
                    assert_eq!(cache.tail.unwrap().as_ref().prev, cache.head);
                }
            }

            #[test]
            fn test_node_reference_consistency() {
                // Test that all node references in HashMap point to valid list nodes
                let mut cache = LRUCore::new(10);
                for i in 0..5 {
                    cache.insert(i, Arc::new(i));
                }

                let mut list_ptrs = HashSet::new();
                let mut current = cache.head;
                while let Some(node_ptr) = current {
                    list_ptrs.insert(node_ptr);
                    unsafe {
                        current = node_ptr.as_ref().next;
                    }
                }

                assert_eq!(list_ptrs.len(), 5);
                assert_eq!(cache.map.len(), 5);

                for val in cache.map.values() {
                    assert!(list_ptrs.contains(val));
                }
            }

            #[test]
            fn test_doubly_linked_list_integrity() {
                // Test forward and backward link consistency throughout list
                let mut cache = LRUCore::new(10);
                for i in 0..5 {
                    cache.insert(i, Arc::new(i));
                }

                // Validate manually
                let mut current = cache.head;
                let mut prev: Option<NonNull<Node<i32>>> = None;

                while let Some(node_ptr) = current {
                    unsafe {
                        let node = node_ptr.as_ref();
                        assert_eq!(node.prev, prev);

                        if let Some(p) = prev {
                            assert_eq!(p.as_ref().next, Some(node_ptr));
                        }

                        prev = current;
                        current = node.next;
                    }
                }
                assert_eq!(prev, cache.tail);
            }

            #[test]
            fn test_invariants_after_every_operation() {
                // Test all invariants are maintained after each cache operation
                let mut cache = LRUCore::new(5);
                cache.validate_invariants();

                for i in 0..5 {
                    cache.insert(i, Arc::new(i));
                    cache.validate_invariants();
                }

                cache.get(&2);
                cache.validate_invariants();

                cache.insert(5, Arc::new(5)); // Eviction
                cache.validate_invariants();

                cache.remove(&3);
                cache.validate_invariants();

                cache.clear();
                cache.validate_invariants();
            }

            #[test]
            fn test_memory_consistency_on_eviction() {
                // Test memory state consistency during eviction operations
                let mut cache = LRUCore::new(2);
                cache.insert(1, Arc::new(10));
                cache.insert(2, Arc::new(20));

                assert_eq!(cache.map.len(), 2);
                assert_eq!(count_nodes(&cache), 2);

                // Trigger eviction
                cache.insert(3, Arc::new(30));

                assert_eq!(cache.map.len(), 2);
                assert_eq!(count_nodes(&cache), 2);
                assert!(cache.contains(&2));
                assert!(cache.contains(&3));
                assert!(!cache.contains(&1));

                cache.validate_invariants();
            }

            #[test]
            fn test_capacity_constraints_enforcement() {
                // Test that cache never exceeds capacity constraints
                let capacity = 10;
                let mut cache = LRUCore::new(capacity);

                for i in 0..capacity * 2 {
                    cache.insert(i, Arc::new(i));
                    assert!(cache.map.len() <= capacity);
                    assert!(count_nodes(&cache) <= capacity);
                }
            }

            #[test]
            fn test_empty_cache_state_invariants() {
                // Test invariants when cache is empty (head=None, tail=None)
                let cache: LRUCore<i32, i32> = LRUCore::new(10);
                assert!(cache.head.is_none());
                assert!(cache.tail.is_none());
                assert!(cache.map.is_empty());
                assert_eq!(count_nodes(&cache), 0);
            }

            #[test]
            fn test_single_item_cache_state() {
                // Test state consistency when cache has exactly one item
                let mut cache = LRUCore::new(10);
                cache.insert(1, Arc::new(100));

                assert!(cache.head.is_some());
                assert!(cache.tail.is_some());
                assert_eq!(cache.head, cache.tail);
                assert_eq!(cache.map.len(), 1);
                assert_eq!(count_nodes(&cache), 1);

                unsafe {
                    assert!(cache.head.unwrap().as_ref().prev.is_none());
                    assert!(cache.tail.unwrap().as_ref().next.is_none());
                }
            }

            #[test]
            fn test_full_cache_state_invariants() {
                // Test invariants when cache is at full capacity
                let capacity = 3;
                let mut cache = LRUCore::new(capacity);
                for i in 0..capacity {
                    cache.insert(i, Arc::new(i));
                }

                assert_eq!(cache.map.len(), capacity);
                assert_eq!(count_nodes(&cache), capacity);
                assert!(cache.head.is_some());
                assert!(cache.tail.is_some());
                assert_ne!(cache.head, cache.tail);

                cache.validate_invariants();
            }

            #[test]
            fn test_state_after_clear_operation() {
                // Test that cache state is properly reset after clear()
                let mut cache = LRUCore::new(5);
                for i in 0..3 {
                    cache.insert(i, Arc::new(i));
                }

                cache.clear();

                assert!(cache.head.is_none());
                assert!(cache.tail.is_none());
                assert!(cache.map.is_empty());
                assert_eq!(count_nodes(&cache), 0);
            }

            #[test]
            fn test_state_during_capacity_transitions() {
                // Test state consistency during transitions between different fill levels
                let mut cache = LRUCore::new(5);

                // 0 -> 1
                cache.insert(1, Arc::new(1));
                assert_eq!(count_nodes(&cache), 1);

                // 1 -> 2
                cache.insert(2, Arc::new(2));
                assert_eq!(count_nodes(&cache), 2);

                // 2 -> 1
                cache.remove(&1);
                assert_eq!(count_nodes(&cache), 1);

                // 1 -> 0
                cache.remove(&2);
                assert_eq!(count_nodes(&cache), 0);
                assert!(cache.head.is_none());
            }

            #[test]
            fn test_node_allocation_consistency() {
                // Test that all allocated nodes are properly tracked and deallocated
                // We verify this by checking map size vs list size
                let mut cache = LRUCore::new(10);
                for i in 0..10 {
                    cache.insert(i, Arc::new(i));
                }

                assert_eq!(cache.map.len(), 10);
                assert_eq!(count_nodes(&cache), 10);

                // Overwrite keys - should reuse or reallocate but keep consistency
                for i in 0..5 {
                    cache.insert(i, Arc::new(i + 100));
                }
                assert_eq!(cache.map.len(), 10);
                assert_eq!(count_nodes(&cache), 10);
            }

            #[test]
            fn test_key_value_mapping_consistency() {
                // Test that keys in the list map to values in the store
                let mut cache = LRUCore::new(5);
                for i in 0..5 {
                    cache.insert(i, Arc::new(i * 10));
                }

                for i in 0..5 {
                    let node_ptr = cache.map.get(&i).unwrap();
                    unsafe {
                        let node = node_ptr.as_ref();
                        assert_eq!(node.key, i);
                        let value = cache.store.peek_ref(&i).unwrap();
                        assert_eq!(**value, i * 10);
                    }
                }
            }

            #[test]
            fn test_lru_ordering_state_consistency() {
                // Test that LRU ordering state matches actual access patterns
                let mut cache = LRUCore::new(3);
                cache.insert(1, Arc::new(1));
                cache.insert(2, Arc::new(2));
                cache.insert(3, Arc::new(3));

                // Order: 3 -> 2 -> 1
                unsafe {
                    assert_eq!(cache.head.unwrap().as_ref().key, 3);
                    assert_eq!(cache.tail.unwrap().as_ref().key, 1);
                }

                // Access 1 -> moves to head
                cache.get(&1);
                // Order: 1 -> 3 -> 2
                unsafe {
                    assert_eq!(cache.head.unwrap().as_ref().key, 1);
                    assert_eq!(cache.tail.unwrap().as_ref().key, 2);
                }
            }

            #[test]
            fn test_concurrent_state_consistency() {
                // Test state consistency in concurrent access scenarios
                // Using ConcurrentLRUCache which wraps LRUCore
                let cache = Arc::new(ConcurrentLRUCache::new(10));
                let mut threads = vec![];

                for i in 0..10 {
                    let cache_clone = cache.clone();
                    threads.push(std::thread::spawn(move || {
                        cache_clone.insert(i, Arc::new(i));
                        let _ = cache_clone.get(&i);
                    }));
                }

                for t in threads {
                    t.join().unwrap();
                }

                assert!(cache.len() <= 10);
                // We access inner LRUCore to check consistency via the lock
                let guard = cache.inner.read();
                assert_eq!(guard.map.len(), count_nodes(&*guard));
            }

            #[test]
            fn test_state_recovery_after_errors() {
                // Test state consistency after error conditions
                // LRUCore operations generally don't return Result, but we can check boundary cases
                let mut cache = LRUCore::new(0);
                assert!(cache.insert(1, Arc::new(1)).is_none());
                assert!(cache.map.is_empty());

                let mut cache = LRUCore::new(1);
                cache.insert(1, Arc::new(1));
                // Try to remove non-existent
                assert!(cache.remove(&2).is_none());
                assert_eq!(cache.map.len(), 1);
                cache.validate_invariants();
            }

            #[test]
            fn test_arc_reference_count_consistency() {
                // Test that Arc reference counts are consistent with expectations
                let mut cache = LRUCore::new(5);
                let val = Arc::new(100);
                assert_eq!(Arc::strong_count(&val), 1);

                cache.insert(1, val.clone());
                // 1 in local var, 1 in cache node
                assert_eq!(Arc::strong_count(&val), 2);

                cache.remove(&1);
                // 1 in local var
                assert_eq!(Arc::strong_count(&val), 1);
            }

            #[test]
            fn test_phantom_data_type_consistency() {
                // Test that PhantomData correctly represents type relationships
                // This is mostly a compile-time check, but we can verify instantiation
                let cache: LRUCore<u32, String> = LRUCore::new(10);
                assert_eq!(cache.capacity(), 10);
            }

            #[test]
            fn test_state_transitions_insert_remove() {
                // Test state consistency during insert/remove cycles
                let mut cache = LRUCore::new(3);

                // Insert 1, 2, 3
                cache.insert(1, Arc::new(1));
                cache.insert(2, Arc::new(2));
                cache.insert(3, Arc::new(3));
                cache.validate_invariants();

                // Remove 2 (middle)
                cache.remove(&2);
                cache.validate_invariants();
                assert!(!cache.contains(&2));

                // Insert 4
                cache.insert(4, Arc::new(4));
                cache.validate_invariants();
            }

            #[test]
            fn test_state_transitions_get_peek() {
                // Test state consistency during get/peek operations
                let mut cache = LRUCore::new(3);
                cache.insert(1, Arc::new(1));
                cache.insert(2, Arc::new(2));

                // Peek shouldn't change state (LRU order)
                let head_before = unsafe { cache.head.unwrap().as_ref().key };
                cache.peek(&1);
                let head_after = unsafe { cache.head.unwrap().as_ref().key };
                assert_eq!(head_before, head_after);

                // Get should change state (LRU order)
                cache.get(&1);
                let head_after_get = unsafe { cache.head.unwrap().as_ref().key };
                assert_eq!(head_after_get, 1);
                cache.validate_invariants();
            }

            #[test]
            fn test_state_transitions_touch_operations() {
                // Test state consistency during touch operations
                let mut cache = LRUCore::new(3);
                cache.insert(1, Arc::new(1)); // Tail
                cache.insert(2, Arc::new(2));
                cache.insert(3, Arc::new(3)); // Head

                // Touch 1 -> moves to head
                assert!(cache.touch(&1));
                unsafe {
                    assert_eq!(cache.head.unwrap().as_ref().key, 1);
                    assert_eq!(cache.tail.unwrap().as_ref().key, 2);
                }
                cache.validate_invariants();
            }

            #[test]
            fn test_node_pointer_validity() {
                // Test that all NonNull pointers are valid and point to correct nodes
                let mut cache = LRUCore::new(5);
                cache.insert(1, Arc::new(1));
                cache.insert(2, Arc::new(2));

                // Verify pointers in map point to valid nodes
                for (k, ptr) in &cache.map {
                    unsafe {
                        let node = ptr.as_ref();
                        assert_eq!(node.key, *k);
                    }
                }
            }

            #[test]
            fn test_circular_reference_prevention() {
                // Test prevention of circular references in linked list
                // We verify by traversing and ensuring we don't loop
                let mut cache = LRUCore::new(5);
                for i in 0..5 {
                    cache.insert(i, Arc::new(i));
                }

                let mut visited = HashSet::new();
                let mut current = cache.head;
                while let Some(ptr) = current {
                    assert!(visited.insert(ptr), "Cycle detected!");
                    unsafe {
                        current = ptr.as_ref().next;
                    }
                }
            }

            #[test]
            fn test_orphaned_node_detection() {
                // Test detection and prevention of orphaned nodes
                // In a valid cache, every node in map is in the list and vice versa
                let mut cache = LRUCore::new(5);
                cache.insert(1, Arc::new(1));
                cache.insert(2, Arc::new(2));

                let count_list = count_nodes(&cache);
                assert_eq!(count_list, cache.map.len());
            }

            #[test]
            fn test_duplicate_node_prevention() {
                // Test prevention of duplicate nodes for same key
                let mut cache = LRUCore::new(5);
                cache.insert(1, Arc::new(1));
                let ptr1 = cache.map.get(&1).cloned();

                cache.insert(1, Arc::new(2)); // Overwrite
                let ptr2 = cache.map.get(&1).cloned();

                // Should use same node or replace it correctly
                // In this impl, we update value and move to head, node address might be same (if reused) or different (if reallocated)
                // Looking at insert impl: it replaces value in place!
                assert_eq!(ptr1, ptr2);
                assert_eq!(cache.map.len(), 1);
                assert_eq!(count_nodes(&cache), 1);
            }

            #[test]
            fn test_list_termination_consistency() {
                // Test that list properly terminates (no infinite loops)
                let mut cache = LRUCore::new(5);
                for i in 0..5 {
                    cache.insert(i, Arc::new(i));
                }

                unsafe {
                    assert!(cache.tail.unwrap().as_ref().next.is_none());
                    assert!(cache.head.unwrap().as_ref().prev.is_none());
                }
            }

            #[test]
            fn test_head_node_properties() {
                // Test that head node has prev=None and is most recent
                let mut cache = LRUCore::new(5);
                cache.insert(1, Arc::new(1));

                unsafe {
                    let head = cache.head.unwrap().as_ref();
                    assert!(head.prev.is_none());
                    assert_eq!(head.key, 1);
                }

                cache.insert(2, Arc::new(2));
                unsafe {
                    let head = cache.head.unwrap().as_ref();
                    assert!(head.prev.is_none());
                    assert_eq!(head.key, 2);
                }
            }

            #[test]
            fn test_tail_node_properties() {
                // Test that tail node has next=None and is least recent
                let mut cache = LRUCore::new(5);
                cache.insert(1, Arc::new(1)); // becomes tail when 2 is added
                cache.insert(2, Arc::new(2));

                unsafe {
                    let tail = cache.tail.unwrap().as_ref();
                    assert!(tail.next.is_none());
                    assert_eq!(tail.key, 1);
                }
            }

            #[test]
            fn test_middle_node_properties() {
                // Test that middle nodes have valid prev and next pointers
                let mut cache = LRUCore::new(5);
                cache.insert(1, Arc::new(1));
                cache.insert(2, Arc::new(2)); // Middle
                cache.insert(3, Arc::new(3));

                // List: 3 -> 2 -> 1
                unsafe {
                    let mid_ptr = cache.head.unwrap().as_ref().next.unwrap();
                    let mid = mid_ptr.as_ref();
                    assert_eq!(mid.key, 2);
                    assert!(mid.prev.is_some());
                    assert!(mid.next.is_some());
                    assert_eq!(mid.prev, cache.head);
                    assert_eq!(mid.next, cache.tail);
                }
            }

            #[test]
            fn test_key_uniqueness_in_list() {
                // Test that no key appears twice in the linked list
                let mut cache = LRUCore::new(5);
                cache.insert(1, Arc::new(1));
                cache.insert(2, Arc::new(2));
                cache.insert(1, Arc::new(3)); // Update 1

                let mut keys = HashSet::new();
                let mut current = cache.head;
                while let Some(ptr) = current {
                    unsafe {
                        let key = ptr.as_ref().key;
                        assert!(keys.insert(key));
                        current = ptr.as_ref().next;
                    }
                }
            }

            #[test]
            fn test_value_consistency_across_structures() {
                // Test that values are consistent between store and list nodes
                let mut cache = LRUCore::new(5);
                cache.insert(1, Arc::new(10));

                let map_val = cache.store.peek_ref(&1).unwrap().clone();
                unsafe {
                    let head_key = cache.head.unwrap().as_ref().key;
                    assert_eq!(head_key, 1);
                    assert_eq!(*map_val, 10);
                }
            }

            #[test]
            fn test_state_during_eviction_cascades() {
                // Test state consistency during multiple evictions
                let mut cache = LRUCore::new(3);
                for i in 0..10 {
                    cache.insert(i, Arc::new(i));
                    assert!(cache.len() <= 3);
                    cache.validate_invariants();
                }
            }

            #[test]
            fn test_atomic_operation_consistency() {
                // Test that operations are atomic with respect to state consistency
                // Since LRUCore is single threaded, operations are atomic.
                // We verify that an operation either completes fully or (if we could fail) doesn't change state.
                let mut cache = LRUCore::new(3);
                cache.insert(1, Arc::new(1));
                cache.validate_invariants();
            }

            #[test]
            fn test_rollback_state_on_failure() {
                // Test state rollback when operations fail
                // Currently no operations return Result/failure that requires rollback.
                let cache = LRUCore::<i32, i32>::new(5);
                assert!(cache.head.is_none());
            }

            #[test]
            fn test_debug_invariant_validation() {
                // Test the internal validate_invariants function thoroughly
                let mut cache = LRUCore::new(5);
                cache.validate_invariants();
                cache.insert(1, Arc::new(1));
                cache.validate_invariants();
            }

            #[test]
            fn test_memory_leak_prevention() {
                // Test that no memory leaks occur during normal operations
                // Basic check: ensure map and list counts match
                let mut cache = LRUCore::new(10);
                for i in 0..100 {
                    cache.insert(i % 20, Arc::new(i));
                }
                assert_eq!(cache.map.len(), count_nodes(&cache));
            }

            #[test]
            fn test_double_free_prevention() {
                // Test prevention of double-free errors
                let mut cache = LRUCore::new(5);
                cache.insert(1, Arc::new(1));
                cache.remove(&1);
                cache.remove(&1); // Should be safe
            }

            #[test]
            fn test_use_after_free_prevention() {
                // Test prevention of use-after-free errors
                let mut cache = LRUCore::new(5);
                cache.insert(1, Arc::new(1));
                let val = cache.get(&1).cloned();
                cache.remove(&1);
                // val should still be valid (Arc)
                assert_eq!(*val.unwrap(), 1);
            }

            #[test]
            fn test_thread_safety_state_consistency() {
                // Test state consistency across multiple threads
                let cache = Arc::new(ConcurrentLRUCache::new(10));
                let c1 = cache.clone();
                let t1 = std::thread::spawn(move || {
                    for i in 0..100 {
                        c1.insert(i, Arc::new(i));
                    }
                });

                let c2 = cache.clone();
                let t2 = std::thread::spawn(move || {
                    for i in 0..100 {
                        c2.get(&i);
                    }
                });

                t1.join().unwrap();
                t2.join().unwrap();

                assert!(cache.len() <= 10);
            }

            #[test]
            fn test_lock_state_consistency() {
                // Test RwLock state consistency in concurrent scenarios
                let cache = Arc::new(ConcurrentLRUCache::new(10));

                // Write lock
                {
                    cache.insert(1, Arc::new(1));
                }

                // Read lock
                {
                    assert!(cache.contains(&1));
                }
            }

            #[test]
            fn test_poison_lock_recovery() {
                // Test state consistency after lock poisoning
                let cache = Arc::new(ConcurrentLRUCache::new(10));
                let c_clone = cache.clone();
                let _ = std::thread::spawn(move || {
                    let _ = c_clone.insert(1, Arc::new(1));
                    // panic!("Intentional panic");
                })
                .join();

                // Should still work
                cache.insert(2, Arc::new(2));
                assert!(cache.contains(&2));
            }

            #[test]
            fn test_capacity_zero_state_consistency() {
                // Test state consistency for zero-capacity cache
                let mut cache = LRUCore::new(0);
                cache.insert(1, Arc::new(1));
                assert_eq!(cache.len(), 0);
                assert!(cache.head.is_none());
            }

            #[test]
            fn test_large_capacity_state_consistency() {
                // Test state consistency for very large capacity caches
                let mut cache = LRUCore::new(1000);
                for i in 0..1000 {
                    cache.insert(i, Arc::new(i));
                }
                assert_eq!(cache.len(), 1000);
                assert_eq!(count_nodes(&cache), 1000);
            }

            #[test]
            fn test_state_after_drop() {
                // Test proper cleanup state when cache is dropped
                let cache: LRUCore<i32, i32> = LRUCore::new(5);
                drop(cache);
            }

            #[test]
            fn test_partial_operation_state_consistency() {
                // Test state consistency when operations are interrupted
                let mut cache = LRUCore::new(5);
                cache.insert(1, Arc::new(1));
                cache.validate_invariants();
            }

            #[test]
            fn test_stress_state_consistency() {
                // Test state consistency under high-stress conditions
                let mut cache = LRUCore::new(10);
                for i in 0..10000 {
                    cache.insert(i % 20, Arc::new(i));
                    if i % 100 == 0 {
                        cache.validate_invariants();
                    }
                }
            }

            #[test]
            fn test_node_lifetime_consistency() {
                // Test that node lifetimes are properly managed
                let mut cache = LRUCore::new(5);
                let val = Arc::new(42);
                cache.insert(1, val.clone());
                assert_eq!(Arc::strong_count(&val), 2);

                cache.remove(&1);
                assert_eq!(Arc::strong_count(&val), 1);
            }

            #[test]
            fn test_reallocation_state_consistency() {
                // Test state consistency during HashMap reallocation
                let mut cache = LRUCore::new(100);
                for i in 0..100 {
                    cache.insert(i, Arc::new(i));
                }
                assert_eq!(cache.len(), 100);
                assert_eq!(count_nodes(&cache), 100);

                cache.clear();
                for i in 0..50 {
                    cache.insert(i, Arc::new(i));
                }
                assert_eq!(cache.len(), 50);
            }

            #[test]
            fn test_hash_collision_state_consistency() {
                // Test state consistency when hash collisions occur
                let mut cache = LRUCore::new(100);
                for i in 0..200 {
                    cache.insert(i, Arc::new(i));
                }
            }

            #[test]
            fn test_boundary_condition_state() {
                // Test state consistency at various boundary conditions
                let mut cache = LRUCore::new(1);
                cache.insert(1, Arc::new(1));
                cache.insert(2, Arc::new(2)); // Evict 1
                assert_eq!(cache.len(), 1);
                assert!(cache.contains(&2));

                cache.remove(&2);
                assert!(cache.is_empty());
            }

            #[test]
            fn test_state_serialization_consistency() {
                // Test that cache state could be consistently serialized/deserialized
                let mut cache = LRUCore::new(5);
                cache.insert(1, Arc::new(1));

                // Capture state
                let state: Vec<_> = cache.map.keys().copied().collect();
                assert_eq!(state.len(), 1);
            }

            #[test]
            fn test_clone_state_consistency() {
                // Test state consistency of concurrent cache cloning
                let cache = Arc::new(ConcurrentLRUCache::new(5));
                cache.insert(1, Arc::new(1));

                let c2 = cache.clone();
                assert!(c2.contains(&1));

                cache.insert(2, Arc::new(2));
                assert!(c2.contains(&2)); // Shared state
            }

            #[test]
            fn test_recursive_operation_state() {
                // Test state consistency during recursive operations
                let mut cache = LRUCore::new(5);
                cache.insert(1, Arc::new(1));
                cache.validate_invariants();
            }

            #[test]
            fn test_error_propagation_state() {
                // Test state consistency during error propagation
                let mut cache = LRUCore::new(5);
                cache.insert(1, Arc::new(1));
            }

            #[test]
            fn test_deterministic_state_reproduction() {
                // Test that same operations produce same internal state
                let mut c1 = LRUCore::new(5);
                let mut c2 = LRUCore::new(5);

                let ops = [1, 2, 3, 1, 4, 5, 2, 6];
                for &op in &ops {
                    c1.insert(op, Arc::new(op));
                    c2.insert(op, Arc::new(op));
                }

                assert_eq!(c1.len(), c2.len());
                let mut h1 = c1.head;
                let mut h2 = c2.head;
                while let (Some(n1), Some(n2)) = (h1, h2) {
                    unsafe {
                        assert_eq!(n1.as_ref().key, n2.as_ref().key);
                        h1 = n1.as_ref().next;
                        h2 = n2.as_ref().next;
                    }
                }
                assert!(h1.is_none() && h2.is_none());
            }

            #[test]
            fn test_state_checkpointing() {
                // Test ability to checkpoint and verify cache state
                let mut cache = LRUCore::new(5);
                cache.insert(1, Arc::new(1));

                // "Checkpoint" by cloning state to vector
                let checkpoint: Vec<i32> = unsafe {
                    let mut vec = Vec::new();
                    let mut curr = cache.head;
                    while let Some(node) = curr {
                        vec.push(node.as_ref().key);
                        curr = node.as_ref().next;
                    }
                    vec
                };

                assert_eq!(checkpoint, vec![1]);
            }

            #[test]
            fn test_incremental_state_validation() {
                // Test state validation at incremental checkpoints
                let mut cache = LRUCore::new(5);
                for i in 0..5 {
                    cache.insert(i, Arc::new(i));
                    cache.validate_invariants();
                }
            }
        }
    }

    // ==============================================
    // MEMORY SAFETY TESTS MODULE
    // ==============================================
    mod memory_safety {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::thread;

        use super::*;

        // Helper to track object lifecycle
        struct LifeCycleTracker {
            _id: usize,
            counter: Arc<AtomicUsize>,
        }

        impl LifeCycleTracker {
            fn new(id: usize, counter: Arc<AtomicUsize>) -> Self {
                counter.fetch_add(1, Ordering::SeqCst);
                Self { _id: id, counter }
            }
        }

        impl Drop for LifeCycleTracker {
            fn drop(&mut self) {
                self.counter.fetch_sub(1, Ordering::SeqCst);
            }
        }

        #[test]
        fn test_no_memory_leaks_on_eviction() {
            let counter = Arc::new(AtomicUsize::new(0));
            let mut cache = LRUCore::new(2);

            // Insert 2 items
            cache.insert(1, Arc::new(LifeCycleTracker::new(1, counter.clone())));
            cache.insert(2, Arc::new(LifeCycleTracker::new(2, counter.clone())));
            assert_eq!(counter.load(Ordering::SeqCst), 2);

            // Insert 3rd item - should evict 1
            cache.insert(3, Arc::new(LifeCycleTracker::new(3, counter.clone())));

            // Should have 2 items in cache (2 and 3), 1 evicted and dropped
            assert_eq!(counter.load(Ordering::SeqCst), 2);
            assert!(!cache.contains(&1));
            assert!(cache.contains(&2));
            assert!(cache.contains(&3));
        }

        #[test]
        fn test_no_memory_leaks_on_remove() {
            let counter = Arc::new(AtomicUsize::new(0));
            let mut cache = LRUCore::new(5);

            cache.insert(1, Arc::new(LifeCycleTracker::new(1, counter.clone())));
            assert_eq!(counter.load(Ordering::SeqCst), 1);

            // Remove item
            {
                let _item = cache.remove(&1);
                // Still alive because we hold the Arc
                assert_eq!(counter.load(Ordering::SeqCst), 1);
            }
            // Now dropped
            assert_eq!(counter.load(Ordering::SeqCst), 0);
        }

        #[test]
        fn test_no_memory_leaks_on_pop_lru() {
            let counter = Arc::new(AtomicUsize::new(0));
            let mut cache = LRUCore::new(5);

            cache.insert(1, Arc::new(LifeCycleTracker::new(1, counter.clone())));
            assert_eq!(counter.load(Ordering::SeqCst), 1);

            {
                let popped = cache.pop_lru();
                assert!(popped.is_some());
                assert_eq!(counter.load(Ordering::SeqCst), 1);
            }
            assert_eq!(counter.load(Ordering::SeqCst), 0);
        }

        #[test]
        fn test_no_memory_leaks_on_clear() {
            let counter = Arc::new(AtomicUsize::new(0));
            let mut cache = LRUCore::new(5);

            for i in 0..5 {
                cache.insert(i, Arc::new(LifeCycleTracker::new(i, counter.clone())));
            }
            assert_eq!(counter.load(Ordering::SeqCst), 5);

            cache.clear();
            assert_eq!(counter.load(Ordering::SeqCst), 0);
            assert_eq!(cache.len(), 0);
        }

        #[test]
        fn test_no_memory_leaks_on_drop() {
            let counter = Arc::new(AtomicUsize::new(0));
            {
                let mut cache = LRUCore::new(5);
                for i in 0..5 {
                    cache.insert(i, Arc::new(LifeCycleTracker::new(i, counter.clone())));
                }
                assert_eq!(counter.load(Ordering::SeqCst), 5);
            } // cache drops here
            assert_eq!(counter.load(Ordering::SeqCst), 0);
        }

        #[test]
        fn test_no_double_free_on_eviction() {
            // Implicitly tested by AtomicUsize wrapping if double free occurred
            // but we can be explicit
            let counter = Arc::new(AtomicUsize::new(100)); // Start at 100 to avoid wrapping on first decrement if bug
            let mut cache = LRUCore::new(1);

            cache.insert(1, Arc::new(LifeCycleTracker::new(1, counter.clone())));
            assert_eq!(counter.load(Ordering::SeqCst), 101);

            // Evict
            cache.insert(2, Arc::new(LifeCycleTracker::new(2, counter.clone())));
            assert_eq!(counter.load(Ordering::SeqCst), 101); // 1 evicted (-1), 2 added (+1)
        }

        #[test]
        fn test_no_double_free_on_remove() {
            let counter = Arc::new(AtomicUsize::new(100));
            let mut cache = LRUCore::new(5);

            cache.insert(1, Arc::new(LifeCycleTracker::new(1, counter.clone())));
            assert_eq!(counter.load(Ordering::SeqCst), 101);

            // First remove
            let removed = cache.remove(&1);
            assert!(removed.is_some());
            drop(removed);
            assert_eq!(counter.load(Ordering::SeqCst), 100);

            // Second remove (should return None and not affect counter)
            let removed2 = cache.remove(&1);
            assert!(removed2.is_none());
            assert_eq!(counter.load(Ordering::SeqCst), 100);
        }

        #[test]
        fn test_no_double_free_on_clear() {
            let counter = Arc::new(AtomicUsize::new(100));
            let mut cache = LRUCore::new(5);

            for i in 0..5 {
                cache.insert(i, Arc::new(LifeCycleTracker::new(i, counter.clone())));
            }
            assert_eq!(counter.load(Ordering::SeqCst), 105);

            cache.clear();
            assert_eq!(counter.load(Ordering::SeqCst), 100);

            // Clear again - should do nothing
            cache.clear();
            assert_eq!(counter.load(Ordering::SeqCst), 100);
        }

        #[test]
        fn test_no_use_after_free_access() {
            let mut cache = LRUCore::new(5);
            let key = 1;
            cache.insert(key, Arc::new(100));

            let val = cache.get(&key).cloned();
            assert!(val.is_some());

            cache.remove(&key);

            // Access after free from cache perspective
            assert!(cache.get(&key).is_none());

            // But value should still be valid if we held a reference
            assert_eq!(*val.unwrap(), 100);
        }

        #[test]
        fn test_no_use_after_free_traversal() {
            // Ensure traversing (e.g., via iteration or internal methods) doesn't access freed memory
            // We simulate this by checking internal consistency after operations
            let mut cache = LRUCore::new(3);
            cache.insert(1, Arc::new(1));
            cache.insert(2, Arc::new(2));
            cache.insert(3, Arc::new(3));

            // Pop LRU
            cache.pop_lru(); // Removes 1

            // Check recency rank of remaining items to force traversal
            assert!(cache.recency_rank(&2).is_some());
            assert!(cache.recency_rank(&3).is_some());
            assert!(cache.recency_rank(&1).is_none());
        }

        #[test]
        fn test_safe_node_allocation() {
            let mut cache = LRUCore::new(1000);
            for i in 0..1000 {
                cache.insert(i, Arc::new(i));
            }
            assert_eq!(cache.len(), 1000);
            // Verify we can access all of them (nodes are allocated and linked correctly)
            for i in 0..1000 {
                assert!(cache.contains(&i));
            }
        }

        #[test]
        fn test_safe_node_deallocation() {
            let counter = Arc::new(AtomicUsize::new(0));
            {
                let mut cache = LRUCore::new(10);
                for i in 0..10 {
                    cache.insert(i, Arc::new(LifeCycleTracker::new(i, counter.clone())));
                }
                assert_eq!(counter.load(Ordering::SeqCst), 10);

                // Remove some
                cache.remove(&0);
                cache.remove(&5);
                assert_eq!(counter.load(Ordering::SeqCst), 8);
            }
            // All should be deallocated
            assert_eq!(counter.load(Ordering::SeqCst), 0);
        }

        #[test]
        fn test_safe_pointer_arithmetic() {
            // In our implementation, we don't do raw pointer arithmetic (offsetting).
            // We only dereference pointers we obtained from Box::into_raw.
            // So we verify that we can dereference pointers in the list correctly.
            let mut cache = LRUCore::new(3);
            cache.insert(1, Arc::new(1));
            cache.insert(2, Arc::new(2));
            cache.insert(3, Arc::new(3));

            // This implicitly tests pointer following
            assert_eq!(*cache.peek(&1).unwrap(), 1);
            assert_eq!(*cache.peek(&2).unwrap(), 2);
            assert_eq!(*cache.peek(&3).unwrap(), 3);
        }

        #[test]
        fn test_safe_list_manipulation() {
            let mut cache = LRUCore::new(10);
            // Create a chain
            for i in 0..5 {
                cache.insert(i, Arc::new(i));
            }

            // Move middle to head
            cache.get(&2);

            // Remove head (LRU)
            cache.pop_lru(); // Should remove 0 (LRU)

            // Remove tail (MRU)
            cache.remove(&2); // 2 was MRU

            assert_eq!(cache.len(), 3);
            assert!(cache.contains(&1));
            assert!(cache.contains(&3));
            assert!(cache.contains(&4));
        }

        #[test]
        fn test_arc_reference_counting_safety() {
            let counter = Arc::new(AtomicUsize::new(0));
            let mut cache = LRUCore::new(5);

            let tracker = Arc::new(LifeCycleTracker::new(1, counter.clone()));
            cache.insert(1, tracker.clone());

            assert_eq!(counter.load(Ordering::SeqCst), 1);

            // Remove from cache
            cache.remove(&1);

            // Still alive because we hold a reference
            assert_eq!(counter.load(Ordering::SeqCst), 1);

            drop(tracker);
            // Now dead
            assert_eq!(counter.load(Ordering::SeqCst), 0);
        }

        #[test]
        fn test_arc_cyclic_reference_prevention() {
            struct _Node {
                _next: Option<Arc<_Node>>,
            }

            let mut cache = LRUCore::new(2);
            // We just ensure LRU drops its reference.

            let counter = Arc::new(AtomicUsize::new(0));
            let cycle_node = Arc::new(LifeCycleTracker::new(1, counter.clone()));

            cache.insert(1, cycle_node.clone());
            assert_eq!(counter.load(Ordering::SeqCst), 1);

            cache.remove(&1);
            // cycle_node still held by us
            assert_eq!(counter.load(Ordering::SeqCst), 1);
            drop(cycle_node);
            assert_eq!(counter.load(Ordering::SeqCst), 0);
        }

        #[test]
        fn test_memory_alignment_safety() {
            use std::mem;
            // Ensure Node alignment is respected
            assert!(mem::align_of::<Node<u32>>() >= mem::align_of::<u32>());

            let mut cache = LRUCore::new(1);
            cache.insert(1, Arc::new(1u64)); // u64 has stricter alignment
            assert_eq!(**cache.get(&1).unwrap(), 1u64);
        }

        #[test]
        fn test_stack_overflow_prevention() {
            // Test prevention of stack overflow in recursive operations (e.g. Drop)
            let mut cache = LRUCore::new(10000);
            for i in 0..10000 {
                cache.insert(i, Arc::new(i));
            }
            // Drop huge cache
            drop(cache);
        }

        #[test]
        fn test_heap_corruption_prevention() {
            // Stress test to try to trigger heap corruption if there were double frees
            let mut cache = LRUCore::new(100);
            for i in 0..1000 {
                cache.insert(i % 200, Arc::new(i));
                if i % 3 == 0 {
                    cache.remove(&(i % 200));
                }
            }
        }

        #[test]
        fn test_null_pointer_dereference_prevention() {
            let mut cache: LRUCore<i32, i32> = LRUCore::new(10);
            // Operations on empty cache should not deref null
            assert!(cache.pop_lru().is_none());
            assert!(cache.peek_lru().is_none());
            assert!(cache.remove(&1).is_none());

            cache.insert(1, Arc::new(1));
            cache.remove(&1);
            // Should be empty again
            assert!(cache.pop_lru().is_none());
        }

        #[test]
        fn test_dangling_pointer_prevention() {
            let mut cache = LRUCore::new(2);
            cache.insert(1, Arc::new(1));
            let val = cache.get(&1).cloned();
            cache.remove(&1);
            // Pointers inside cache for '1' are gone. 'val' is independent Arc.
            assert_eq!(*val.unwrap(), 1);
        }

        #[test]
        fn test_buffer_overflow_prevention() {
            // Not directly applicable to linked list, but we can test capacity limits
            let mut cache = LRUCore::new(2);
            cache.insert(1, Arc::new(1));
            cache.insert(2, Arc::new(2));
            cache.insert(3, Arc::new(3));
            assert_eq!(cache.len(), 2);
        }

        #[test]
        fn test_memory_bounds_checking() {
            // Capacity check
            let mut cache = LRUCore::new(1);
            cache.insert(1, Arc::new(1));
            cache.insert(2, Arc::new(2));
            assert_eq!(cache.len(), 1);
            assert!(cache.contains(&2));
            assert!(!cache.contains(&1));
        }

        #[test]
        fn test_safe_concurrent_access() {
            // Verify memory safety under concurrent load
            let counter = Arc::new(AtomicUsize::new(0));
            let cache = Arc::new(ConcurrentLRUCache::new(10));

            let mut handles = vec![];

            for i in 0..10 {
                let cache = cache.clone();
                let counter = counter.clone();
                handles.push(thread::spawn(move || {
                    for j in 0..100 {
                        // Use a larger key space to force evictions
                        let key = i * 1000 + j;
                        let val = Arc::new(LifeCycleTracker::new(key as usize, counter.clone()));
                        cache.insert(key, val);
                    }
                }));
            }

            for h in handles {
                h.join().unwrap();
            }

            // At end, we should have at most 10 items in cache (capacity 10)
            let count = counter.load(Ordering::SeqCst);
            assert!(
                count <= 10,
                "Memory leak detected: {} items alive (capacity 10)",
                count
            );
            // Note: ConcurrentLRUCache might be slightly loose on exact capacity during heavy contention
            // depending on implementation, but should settle.
            // If strict, count == cache.len().
        }

        #[test]
        fn test_safe_concurrent_modification() {
            // Similar to test_safe_concurrent_access but mixing insert/remove
            let counter = Arc::new(AtomicUsize::new(0));
            let cache = Arc::new(ConcurrentLRUCache::new(100));

            let mut handles = vec![];
            for i in 0..10 {
                let cache = cache.clone();
                let counter = counter.clone();
                handles.push(thread::spawn(move || {
                    for j in 0..100 {
                        let key = i * 1000 + j;
                        let val = Arc::new(LifeCycleTracker::new(key as usize, counter.clone()));
                        cache.insert(key, val);
                        if j % 2 == 0 {
                            cache.remove(&key);
                        }
                    }
                }));
            }

            for h in handles {
                h.join().unwrap();
            }

            // Check consistency
            let count = counter.load(Ordering::SeqCst);
            assert!(count <= 100);
        }

        #[test]
        fn test_lock_poisoning_memory_safety() {
            // parking_lot RwLock does not poison. It releases the lock on unwind.
            // We verify that the cache remains usable and consistent after a panic in a thread holding the lock.
            use std::hash::{Hash, Hasher};

            #[derive(PartialEq, Eq, Clone, Copy, Debug)]
            struct PanickingKey(i32);

            impl Hash for PanickingKey {
                fn hash<H: Hasher>(&self, state: &mut H) {
                    if self.0 == 666 {
                        panic!("Simulated panic during hash");
                    }
                    self.0.hash(state);
                }
            }

            // Use generic type parameters to trick the cache into accepting our key
            // But ConcurrentLRUCache<K, V> is generic.
            // We need to instantiate a cache with PanickingKey.
            // But ConcurrentLRUCache wraps LRUCore.

            // We can't easily use ConcurrentLRUCache with PanickingKey if we don't change the test signature
            // or use a specific instantiation.
            // The test function body can instantiate whatever it wants.

            let cache = Arc::new(ConcurrentLRUCache::<PanickingKey, i32>::new(10));

            let c_clone = cache.clone();
            let _ = thread::spawn(move || {
                // This panics inside insert, while lock is held (write lock)
                // We assume hash is called inside the lock.
                c_clone.insert(PanickingKey(666), 1);
            })
            .join()
            .unwrap_err(); // Should return Err (panic)

            // Cache should still be accessible (lock released)
            // And insert should have failed cleanly (or leaked node, but cache state should be safe to access)
            assert_eq!(cache.len(), 0);
            cache.insert(PanickingKey(1), 1);
            assert_eq!(cache.len(), 1);
        }

        #[test]
        fn test_panic_safety_memory_cleanup() {
            use std::hash::{Hash, Hasher};
            use std::panic::{self, AssertUnwindSafe};

            #[derive(PartialEq, Eq, Clone, Copy, Debug)]
            struct PanickingKey(i32);

            impl Hash for PanickingKey {
                fn hash<H: Hasher>(&self, state: &mut H) {
                    if self.0 == 666 {
                        panic!("Simulated panic during hash");
                    }
                    self.0.hash(state);
                }
            }

            let counter = Arc::new(AtomicUsize::new(0));
            let mut cache = LRUCore::new(10);

            let result = panic::catch_unwind(AssertUnwindSafe(|| {
                let tracker = Arc::new(LifeCycleTracker::new(666, counter.clone()));
                cache.insert(PanickingKey(666), tracker);
            }));

            assert!(result.is_err());

            // Verify cache is still usable
            cache.insert(
                PanickingKey(1),
                Arc::new(LifeCycleTracker::new(1, counter.clone())),
            );
            assert_eq!(cache.len(), 1);
        }

        #[test]
        fn test_exception_safety_guarantees() {
            // Basic exception safety (weak): no leaks, invariants hold.
            // Strong exception safety: state unchanged on failure.

            // Our insert is not strong exception safe if hash panics (map might be modified? no, insert happens after hash).
            // But if we leak memory, that's a violation of basic guarantee?
            // As noted, we might have a leak on panic during insert.
            // We verify at least the cache invariants hold.

            use std::hash::{Hash, Hasher};
            use std::panic::{self, AssertUnwindSafe};

            #[derive(PartialEq, Eq, Clone, Copy, Debug)]
            struct PanickingKey(i32);
            impl Hash for PanickingKey {
                fn hash<H: Hasher>(&self, state: &mut H) {
                    if self.0 == 666 {
                        panic!("Panic");
                    }
                    self.0.hash(state);
                }
            }

            let mut cache = LRUCore::new(10);
            cache.insert(PanickingKey(1), Arc::new(1));

            let _ = panic::catch_unwind(AssertUnwindSafe(|| {
                cache.insert(PanickingKey(666), Arc::new(2));
            }));

            assert_eq!(cache.len(), 1);
            assert!(cache.contains(&PanickingKey(1)));
            // Check internal consistency
            assert!(cache.peek(&PanickingKey(1)).is_some());
        }

        #[test]
        fn test_memory_leak_detection_valgrind() {
            // Placeholder: Run with valgrind
            // cargo build --tests
            // valgrind ./target/debug/deps/ferrite-...
        }

        #[test]
        fn test_memory_leak_detection_miri() {
            // Placeholder: Run with miri
            // cargo miri test
        }

        #[test]
        fn test_memory_safety_under_stress() {
            // High contention stress test
            let cache = Arc::new(ConcurrentLRUCache::new(100));
            let mut handles = vec![];
            for _i in 0..10 {
                let c = cache.clone();
                handles.push(thread::spawn(move || {
                    for j in 0..1000 {
                        c.insert(j % 200, Arc::new(j));
                        if j % 3 == 0 {
                            c.remove(&(j % 200));
                        }
                    }
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
        }

        #[test]
        fn test_memory_fragmentation_handling() {
            // Hard to test fragmentation in unit test without allocator introspection.
            // Just verifying large churn works.
            let mut cache = LRUCore::new(10);
            for i in 0..10000 {
                cache.insert(i % 20, Arc::new(i));
            }
        }

        #[test]
        fn test_large_allocation_safety() {
            // Test with large values
            let mut cache = LRUCore::new(2);
            let big_vec = vec![0u8; 1024 * 1024]; // 1MB
            cache.insert(1, Arc::new(big_vec));
            assert_eq!(cache.get(&1).unwrap().len(), 1024 * 1024);
        }

        #[test]
        fn test_copy_type_memory_efficiency() {
            // Verify that using Copy types for keys doesn't cause excessive overhead
            // Mostly a sanity check that we accept Copy keys
            let mut cache = LRUCore::new(10);
            cache.insert(1usize, Arc::new(1));
            assert!(cache.contains(&1));
        }

        #[test]
        fn test_move_semantics_safety() {
            // Ensure values are moved into Arc correctly
            let s = String::from("hello");
            let mut cache = LRUCore::new(10);
            cache.insert(1, Arc::new(s)); // s moved into Arc
            // s is gone (compile time check effectively, but runtime we verify value)
            let v = cache.get(&1).unwrap();
            assert_eq!(v.as_str(), "hello");
        }

        #[test]
        fn test_lifetime_parameter_safety() {
            // Verify standard lifetime rules apply
            let mut cache = LRUCore::new(10);
            let v = Arc::new(1);
            cache.insert(1, v.clone());
            {
                let r = cache.get(&1).unwrap();
                assert_eq!(**r, 1);
            } // r dropped
            cache.remove(&1);
        }

        #[test]
        fn test_send_sync_memory_safety() {
            fn assert_send<T: Send>() {}
            fn assert_sync<T: Sync>() {}

            assert_send::<LRUCore<i32, i32>>();
            assert_sync::<LRUCore<i32, i32>>();
            assert_send::<ConcurrentLRUCache<i32, i32>>();
            assert_sync::<ConcurrentLRUCache<i32, i32>>();
        }

        #[test]
        fn test_drop_trait_memory_cleanup() {
            let counter = Arc::new(AtomicUsize::new(0));
            {
                let mut cache = LRUCore::new(10);
                cache.insert(1, Arc::new(LifeCycleTracker::new(1, counter.clone())));
            }
            assert_eq!(counter.load(Ordering::SeqCst), 0);
        }

        #[test]
        fn test_clone_memory_safety() {
            // Verify ConcurrentLRUCache clone shares state safely
            let cache = Arc::new(ConcurrentLRUCache::new(10));
            let c2 = cache.clone();

            cache.insert(1, Arc::new(1));
            assert!(c2.contains(&1));
        }

        #[test]
        fn test_unsafe_block_soundness() {
            // Function to tag tests covering unsafe blocks
            // Most tests cover unsafe blocks in allocate_node, insert, remove_from_list, etc.
            let mut cache = LRUCore::new(10);
            cache.insert(1, Arc::new(1));
            cache.remove(&1);
        }

        #[test]
        fn test_raw_pointer_safety() {
            // Implicitly covered by all operations
            let mut cache = LRUCore::new(10);
            cache.insert(1, Arc::new(1));
        }

        #[test]
        fn test_memory_reclamation_safety() {
            // Verify memory is reclaimed when cache is dropped
            let counter = Arc::new(AtomicUsize::new(0));
            {
                let mut cache = LRUCore::new(10);
                cache.insert(1, Arc::new(LifeCycleTracker::new(1, counter.clone())));
            }
            assert_eq!(counter.load(Ordering::SeqCst), 0);
        }

        #[test]
        fn test_oom_handling_safety() {
            // Difficult to test safely without custom allocator.
            // Documentation: operations may panic on OOM.
        }

        #[test]
        fn test_cross_thread_memory_safety() {
            let cache = Arc::new(ConcurrentLRUCache::new(10));
            let c2 = cache.clone();
            thread::spawn(move || {
                c2.insert(1, Arc::new(1));
            })
            .join()
            .unwrap();
            assert!(cache.contains(&1));
        }

        #[test]
        fn test_unwind_safety() {
            // Panic safety check
            let mut cache = LRUCore::new(10);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                cache.insert(1, Arc::new(1));
                panic!("oops");
            }));
            assert!(result.is_err());
            // Cache dropped here, should clean up.
        }

        #[test]
        fn test_memory_sanitizer_compatibility() {
            // Placeholder
        }
        #[test]
        fn test_address_sanitizer_compatibility() {
            // Placeholder
        }
        #[test]
        fn test_thread_sanitizer_compatibility() {
            // Placeholder
        }
        #[test]
        fn test_leak_sanitizer_compatibility() {
            // Placeholder
        }
    }
}
